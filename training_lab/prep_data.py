#!/usr/bin/env python
"""training_lab/prep_data.py — 为训练冒烟准备 TinyStories token 数据（SPEC §5.1）。

做什么
------
1. 只读官方 fixture ``tinystories_sample_5M.txt`` 的前 ``--bytes`` 字节（默认 1MB）；
2. 用 ``bpe_lab.toy_bpe.train_bpe_from_file`` 在**前 512KB** 上训练 byte-level BPE
   （``vocab_size=2048``，special token ``<|endoftext|>``），保存到 ``data/tokenizer/``；
3. 用该 tokenizer 把整段文本编码成 ``uint16`` token 序列，按 95/5 切成
   ``data/train.bin`` / ``data/val.bin``（裸 uint16，无 header）；
4. 写 ``data/meta.json``（字节数、token 数、压缩比、耗时……），并用 ``np.memmap``
   读回自检：长度对、所有 id < vocab_size。

为什么这么设计
--------------
* **训练 BPE 与编码用不同的切片**：BPE 只在 512KB 上训练（SPEC 要求），但编码覆盖
  整个 ``--bytes`` 段；这样数据准备很快，也避免 BPE 训练吃满热预算。
* **裸 uint16 + meta.json**：训练脚本只关心「一条 int token 流」，不需要每步解析
  JSON；meta 负责记录来源与口径，方便实验结果复现。
* **缓存命中直接退出**：重复跑 ``train_demo.py`` 前不会每次都重新训练 BPE。
  ``--force`` 强制重跑。
* **UTF-8 边界**：``--bytes`` 截断可能落在多字节字符中间，这里统一退到合法边界再
  ``decode``，保证 BPE 语料与编码文本都是合法 UTF-8。

热预算
------
``torch.set_num_threads(2)``；单命令 < 120 秒（默认配置实测约 0.3–0.5 秒，
缓存命中约 0.5 秒；见 LAB_NOTES §4）。

用法::

    .venv/bin/python training_lab/prep_data.py
    .venv/bin/python training_lab/prep_data.py --force
    .venv/bin/python training_lab/prep_data.py --bytes 500000 --vocab-size 1024
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch

# SPEC §1：所有可能超过 1 秒的脚本必须 2 线程
torch.set_num_threads(2)

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT / "bpe_lab"))

from toy_bpe import BPETokenizer, train_bpe_from_file  # noqa: E402

DEFAULT_FIXTURE = Path(
    "/Users/yunni/Projects/assignment1-basics/tests/fixtures/tinystories_sample_5M.txt"
)
DEFAULT_DATA_DIR = HERE / "data"
SPECIAL = "<|endoftext|>"
SPECIAL_TOKENS = [SPECIAL]
MIB = 1024 * 1024


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------
def peak_rss_mib() -> float:
    """进程峰值 RSS（MiB）。macOS 的 ru_maxrss 单位是字节，Linux 是 KiB。"""
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru / MIB if sys.platform == "darwin" else ru / 1024.0


def truncate_utf8(raw: bytes) -> bytes:
    """把可能截断在多字节字符中间的 bytes 退到最近的合法 UTF-8 边界。

    最多回退 3 个字节（UTF-8 单字符最长 4 字节），然后用 ``decode`` 验证；
    极端情况下（数据本身不是合法 UTF-8）用 ``errors="ignore"`` 兜底并重新编码。
    """
    for back in range(4):
        candidate = raw[: len(raw) - back] if back else raw
        try:
            return candidate.decode("utf-8").encode("utf-8")
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore").encode("utf-8")


def _cache_ok(meta_path: Path, data_dir: Path, args: argparse.Namespace) -> dict | None:
    """缓存是否可直接复用：meta.json 存在、参数一致、两个 bin 都在且长度对得上。"""
    if not meta_path.is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    # 只比较「会改变产物」的参数；elapsed_s 之类的统计字段不参与
    expected = {
        "bytes": int(meta.get("bytes", -1)),
        "vocab_size": int(meta.get("vocab_size", -1)),
        "special_tokens": meta.get("special_tokens"),
        "train_fraction": meta.get("train_fraction"),
        "bpe_train_bytes": meta.get("bpe_train_bytes"),
    }
    want = {
        "bytes": args.bytes,
        "vocab_size": args.vocab_size,
        "special_tokens": SPECIAL_TOKENS,
        "train_fraction": args.train_fraction,
        "bpe_train_bytes": args.bpe_train_bytes,
    }
    if expected != want:
        return None

    train_path = data_dir / "train.bin"
    val_path = data_dir / "val.bin"
    tokenizer_dir = data_dir / "tokenizer"
    if not train_path.is_file() or not val_path.is_file():
        return None
    if not (tokenizer_dir / "vocab.json").is_file() or not (tokenizer_dir / "merges.txt").is_file():
        return None
    if train_path.stat().st_size != 2 * int(meta["train_tokens"]):
        return None
    if val_path.stat().st_size != 2 * int(meta["val_tokens"]):
        return None
    return meta


def _memmap_check(path: Path, expected_tokens: int, vocab_size: int) -> dict:
    """用 np.memmap 读回 bin 文件做自检：长度、dtype、id 上界。"""
    arr = np.memmap(path, dtype=np.uint16, mode="r")
    if arr.shape[0] != expected_tokens:
        raise ValueError(
            f"{path.name} 长度 {arr.shape[0]} != meta 记录的 {expected_tokens}"
        )
    max_id = int(arr.max()) if arr.shape[0] else -1
    if max_id >= vocab_size:
        raise ValueError(f"{path.name} 出现 id={max_id} >= vocab_size={vocab_size}")
    return {
        "tokens": int(arr.shape[0]),
        "dtype": str(arr.dtype),
        "max_id": max_id,
        "min_id": int(arr.min()) if arr.shape[0] else -1,
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def prepare(args: argparse.Namespace) -> dict:
    wall_start = time.perf_counter()
    data_dir: Path = args.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_dir = data_dir / "tokenizer"
    meta_path = data_dir / "meta.json"

    # ---- 0. 缓存命中：打印上次结果后直接退出
    cached = None if args.force else _cache_ok(meta_path, data_dir, args)
    if cached is not None:
        if not args.quiet:
            print("=" * 84)
            print("prep_data：缓存命中，跳过重新生成（--force 可强制重跑）")
            print("=" * 84)
            print(
                f"bytes={cached['bytes']:,}  vocab={cached['vocab_size']}  "
                f"train_tokens={cached['train_tokens']:,}  val_tokens={cached['val_tokens']:,}  "
                f"ratio={cached['compression_ratio']:.3f}  last_elapsed={cached['elapsed_s']:.2f}s"
            )
            print(f"[data] {meta_path}")
        return cached

    # ---- 1. 读取 fixture 前缀（只读）
    fixture: Path = args.fixture
    if not fixture.is_file():
        raise FileNotFoundError(f"fixture 不存在：{fixture}")
    with open(fixture, "rb") as f:
        raw = f.read(args.bytes)  # 只读，不修改官方仓库
    if not raw:
        raise ValueError(f"fixture 为空：{fixture}")
    raw = truncate_utf8(raw)
    text = raw.decode("utf-8")
    n_bytes = len(raw)

    # ---- 2. 训练 / 加载 tokenizer
    t_tok = time.perf_counter()
    bpe_corpus_path = data_dir / "bpe_train_corpus.txt"
    if args.force or not (tokenizer_dir / "vocab.json").is_file():
        bpe_raw = truncate_utf8(raw[: args.bpe_train_bytes])
        bpe_corpus_path.write_bytes(bpe_raw)
        vocab, merges = train_bpe_from_file(
            bpe_corpus_path,
            args.vocab_size,
            SPECIAL_TOKENS,
            verbose=not args.quiet,
        )
        tokenizer = BPETokenizer(vocab, merges, SPECIAL_TOKENS)
        tokenizer.save(tokenizer_dir)
    else:
        tokenizer = BPETokenizer.load(tokenizer_dir, special_tokens=SPECIAL_TOKENS)
    tokenizer_elapsed = time.perf_counter() - t_tok
    vocab_size = tokenizer.vocab_size

    # ---- 3. 编码整段文本（流式，避免一次性构造大 list 之外的额外拷贝）
    t_enc = time.perf_counter()
    text_tokens: list[int] = []
    chunk_chars = 64 * 1024
    text_tokens.extend(
        tokenizer.encode_iterable(
            text[i : i + chunk_chars] for i in range(0, len(text), chunk_chars)
        )
    )
    encode_elapsed = time.perf_counter() - t_enc

    # ---- 4. 95/5 切分并写裸 uint16
    t_write = time.perf_counter()
    tokens = np.asarray(text_tokens, dtype=np.uint16)
    # 按 token 数严格切 95/5（训练段在前，验证段是尾部 5%）
    split = int(len(tokens) * args.train_fraction)
    split = min(max(split, 1), max(len(tokens) - 1, 1))
    train_tokens = tokens[:split]
    val_tokens = tokens[split:]
    train_path = data_dir / "train.bin"
    val_path = data_dir / "val.bin"
    train_tokens.tofile(train_path)
    val_tokens.tofile(val_path)
    write_elapsed = time.perf_counter() - t_write

    compression_ratio = n_bytes / len(tokens)

    # ---- 5. 自检：memmap 读回
    train_check = _memmap_check(train_path, len(train_tokens), vocab_size)
    val_check = _memmap_check(val_path, len(val_tokens), vocab_size)

    elapsed = time.perf_counter() - wall_start
    peak_rss = peak_rss_mib()

    meta = {
        "bytes": n_bytes,
        "vocab_size": vocab_size,
        "train_tokens": int(train_tokens.shape[0]),
        "val_tokens": int(val_tokens.shape[0]),
        "total_tokens": int(tokens.shape[0]),
        "compression_ratio": round(compression_ratio, 4),
        "elapsed_s": round(elapsed, 4),
        "tokenizer_dir": str(tokenizer_dir),
        "fixture": str(fixture),
        "bpe_train_bytes": args.bpe_train_bytes,
        "train_fraction": args.train_fraction,
        "special_tokens": SPECIAL_TOKENS,
        "dtype": "uint16",
        "train_bin": str(train_path),
        "val_bin": str(val_path),
        "train_bytes": int(train_tokens.nbytes),
        "val_bytes": int(val_tokens.nbytes),
        "split_token_index": int(split),
        "check": {
            "train": train_check,
            "val": val_check,
            "ids_below_vocab_size": bool(
                train_check["max_id"] < vocab_size and val_check["max_id"] < vocab_size
            ),
        },
        "timing": {
            "tokenizer_s": round(tokenizer_elapsed, 4),
            "encode_s": round(encode_elapsed, 4),
            "write_s": round(write_elapsed, 4),
        },
        "peak_rss_mib": round(peak_rss, 1),
        "threads": torch.get_num_threads(),
        "meta": {
            "spec": "training_lab/SPEC.md §5.1",
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if not args.quiet:
        print("=" * 84)
        print("prep_data：TinyStories 前缀 -> uint16 token 切片")
        print("=" * 84)
        print(f"fixture={fixture}")
        print(f"bytes={n_bytes:,}  vocab={vocab_size}  bpe_train_bytes={args.bpe_train_bytes:,}")
        print(
            f"tokens={len(tokens):,}  train={len(train_tokens):,}  val={len(val_tokens):,}  "
            f"compression_ratio={compression_ratio:.3f} bytes/token"
        )
        print(
            f"elapsed={elapsed:.2f}s (tokenizer {tokenizer_elapsed:.2f}s / "
            f"encode {encode_elapsed:.2f}s / write {write_elapsed:.2f}s)  "
            f"peak_rss={peak_rss:.1f} MiB  threads={torch.get_num_threads()}"
        )
        print(
            f"check: train max_id={train_check['max_id']}  val max_id={val_check['max_id']}  "
            f"all < vocab_size={meta['check']['ids_below_vocab_size']}"
        )
        print(f"[data] {data_dir}  (train.bin / val.bin / meta.json / tokenizer/)")

    if not meta["check"]["ids_below_vocab_size"]:
        raise AssertionError("token id 上界检查失败")
    return meta


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--fixture", type=Path, default=DEFAULT_FIXTURE, help="TinyStories fixture（只读）"
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--bytes", type=int, default=1_000_000, help="读取的前缀字节数")
    parser.add_argument("--vocab-size", type=int, default=2048)
    parser.add_argument(
        "--bpe-train-bytes", type=int, default=512 * 1024, help="训练 BPE 用的前缀字节数"
    )
    parser.add_argument("--train-fraction", type=float, default=0.95)
    parser.add_argument("--force", action="store_true", help="忽略缓存，重新生成")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 0.0 < args.train_fraction < 1.0:
        raise SystemExit("--train-fraction 必须在 (0, 1) 之间")
    if args.bytes <= 0 or args.bpe_train_bytes <= 0:
        raise SystemExit("--bytes / --bpe-train-bytes 必须为正")
    meta = prepare(args)
    elapsed = float(meta["elapsed_s"])
    if elapsed > 120.0:
        print(f"[WARN] 本次耗时 {elapsed:.1f}s 超过 120 秒热预算", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
