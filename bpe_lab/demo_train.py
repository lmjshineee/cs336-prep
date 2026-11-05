"""demo_train.py —— 在小语料上训练一个 toy BPE，并把「教学关心的数字」全打出来。

默认配置（约 3 秒跑完，远低于 3 分钟上限）：
  语料 = TinyStories fixture 的前 320KB（前 ~300KB 训练，后 ~20KB 留出）
  vocab_size = 1000，special_tokens = ["<|endoftext|>"]

跑法：
    export UV_CACHE_DIR=$PWD/.tmp/uvcache HF_HOME=$PWD/.tmp/hf MPLCONFIGDIR=$PWD/.tmp/mpl
    .venv/bin/python bpe_lab/demo_train.py
    # 想换规模：
    .venv/bin/python bpe_lab/demo_train.py --bytes 1000000 --vocab-size 2000
    # 想同时看一眼 naive 实现有多慢（只在 ~30KB 小片上调）：
    .venv/bin/python bpe_lab/demo_train.py --compare-naive

输出：
  * 终端打印中文汇总表；
  * `bpe_lab/results/toy_demo.json`（机器可读，教程正文引用的数字都来自这里）。

注意：`peak_rss_mb` 用的是 `resource.getrusage` 的高水位值；macOS 单位是字节，
Linux 是 KB，脚本里做了区分。
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import sys
import time
from collections import Counter
from pathlib import Path

# 让 `python bpe_lab/demo_train.py` 能找到同目录的 toy_bpe
sys.path.insert(0, str(Path(__file__).resolve().parent))

from toy_bpe import BPETokenizer, train_bpe  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
DEFAULT_CORPUS = Path(
    "/Users/yunni/Projects/assignment1-basics/tests/fixtures/tinystories_sample_5M.txt"
)
SPECIAL = "<|endoftext|>"


def peak_rss_mb() -> float:
    """进程内存高水位（MB）。macOS 上 ru_maxrss 单位是字节，Linux 上是 KB。"""
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return value / (1024 * 1024)
    return value / 1024


def _split_train_holdout(
    text: str, total_bytes: int, holdout_bytes: int, special: str
) -> tuple[str, str]:
    """按「文档边界」（special token）切开训练段和留出段。

    先把语料截到 total_bytes，再在 total_bytes - holdout_bytes 之前找最后一个
    `<|endoftext|>` 作为切点，这样留出段从文档开头开始，不会从半个词中间开始。
    """
    head = text[:total_bytes]
    target = max(total_bytes - holdout_bytes, 1)
    cut = head.rfind(special, 0, target)
    if cut <= 0:
        cut = target
    return head[:cut], head[cut:]


def _bytes_repr(token: bytes) -> str:
    """把 bytes 写成短小可读的 repr（太长就截断）。"""
    text = repr(token)
    return text if len(text) <= 60 else text[:57] + "...'"


def _fmt_token(token: bytes) -> str:
    return _bytes_repr(token)


def _top_learned_tokens(
    tokenizer: BPETokenizer, text: str, vocab_size: int, top_n: int = 15
) -> list[dict]:
    """统计训练文本里出现次数最多的「学出来的 token」（id >= 257，排除 special）。"""
    ids = tokenizer.encode(text)
    counts = Counter(ids)
    rows = []
    for token_id, count in counts.most_common():
        if token_id < 257:  # 0..255 是单字节，256 是 special
            continue
        rows.append(
            {
                "id": token_id,
                "token": _fmt_token(tokenizer.vocab[token_id]),
                "length_bytes": len(tokenizer.vocab[token_id]),
                "count": count,
            }
        )
        if len(rows) >= top_n:
            break
    return rows


def _compare_naive(text: str, vocab_size: int) -> dict:
    """在同一个（小）语料上跑一次 naive 训练，作为「增量更新到底省了多少」的对照。"""
    slice_text = text[:30_000]
    t0 = time.perf_counter()
    vocab_fast, merges_fast = train_bpe(slice_text, vocab_size, [SPECIAL])
    fast_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    vocab_naive, merges_naive = train_bpe(slice_text, vocab_size, [SPECIAL], naive=True)
    naive_s = time.perf_counter() - t0
    return {
        "corpus_bytes": len(slice_text.encode("utf-8")),
        "vocab_size": vocab_size,
        "fast_seconds": round(fast_s, 4),
        "naive_seconds": round(naive_s, 4),
        "speedup": round(naive_s / fast_s, 2) if fast_s > 0 else None,
        "same_merges": merges_fast == merges_naive,
        "same_vocab": vocab_fast == vocab_naive,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="在 TinyStories 子集上训练 toy BPE 并汇报指标")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS, help="训练语料路径（UTF-8 文本）")
    parser.add_argument("--bytes", type=int, default=320_000, help="从语料开头取多少字节（默认 320KB）")
    parser.add_argument("--vocab-size", type=int, default=1000, help="最终词表大小（含 specials）")
    parser.add_argument("--holdout-bytes", type=int, default=20_000, help="留出多少字节做压缩比/吞吐")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "toy_demo.json")
    parser.add_argument("--compare-naive", action="store_true", help="额外跑一次 naive 训练做速度对照")
    parser.add_argument("--quiet", action="store_true", help="不打印训练过程日志")
    args = parser.parse_args()

    if not args.corpus.exists():
        print(f"找不到语料：{args.corpus}", file=sys.stderr)
        return 2

    script_t0 = time.perf_counter()
    full_text = args.corpus.read_text(encoding="utf-8")
    train_text, holdout_text = _split_train_holdout(
        full_text, args.bytes, args.holdout_bytes, SPECIAL
    )
    train_bytes = len(train_text.encode("utf-8"))
    holdout_bytes = len(holdout_text.encode("utf-8"))

    print("=" * 74)
    print("toy BPE 训练演示")
    print("=" * 74)
    print(f"语料文件      : {args.corpus}")
    print(f"训练段        : {len(train_text):,} 字符 / {train_bytes:,} 字节")
    print(f"留出段        : {len(holdout_text):,} 字符 / {holdout_bytes:,} 字节")
    print(f"vocab_size    : {args.vocab_size}（含 256 单字节 + {SPECIAL}）")
    print("-" * 74)

    # ---- 训练 ----------------------------------------------------------
    mem_before = peak_rss_mb()
    t0 = time.perf_counter()
    vocab, merges = train_bpe(
        train_text, args.vocab_size, [SPECIAL], verbose=not args.quiet
    )
    train_seconds = time.perf_counter() - t0
    mem_after_train = peak_rss_mb()

    tokenizer = BPETokenizer(vocab, merges, [SPECIAL])

    # ---- 词表画像 ------------------------------------------------------
    longest = max((tok for tok in vocab.values() if tok != SPECIAL.encode()), key=len)
    longest_tokens = sorted(
        ({tok for tok in vocab.values() if tok != SPECIAL.encode()}),
        key=len,
        reverse=True,
    )[:5]
    first_15 = [
        {
            "pair": [_bytes_repr(a), _bytes_repr(b)],
            "merged_token": _bytes_repr(a + b),
            "merged_len": len(a + b),
        }
        for a, b in merges[:15]
    ]

    # ---- 压缩比 / 往返 -------------------------------------------------
    t0 = time.perf_counter()
    holdout_ids = tokenizer.encode(holdout_text)
    holdout_encode_seconds = time.perf_counter() - t0
    holdout_tokens = len(holdout_ids)
    compression = holdout_bytes / holdout_tokens if holdout_tokens else float("nan")
    roundtrip_ok = tokenizer.decode(holdout_ids) == holdout_text

    # ---- 吞吐：encode vs encode_iterable ------------------------------
    # pre-token 缓存会让「同一段文本第二次编码」非常快，所以这里先清空缓存，
    # 测的是「冷启动」吞吐；另外再记一个 warm（热缓存）作为对照。
    tokenizer._pretoken_cache.clear()
    t0 = time.perf_counter()
    train_ids = tokenizer.encode(train_text)
    encode_seconds = time.perf_counter() - t0

    tokenizer._pretoken_cache.clear()
    t0 = time.perf_counter()
    stream_ids = list(
        tokenizer.encode_iterable(
            train_text[i : i + 65_536] for i in range(0, len(train_text), 65_536)
        )
    )
    iterable_seconds = time.perf_counter() - t0
    iterable_matches = stream_ids == train_ids

    t0 = time.perf_counter()
    tokenizer.encode(train_text)  # 这次缓存已经热了
    warm_encode_seconds = time.perf_counter() - t0

    mem_final = peak_rss_mb()

    # ---- 学出来的高频 token --------------------------------------------
    top_tokens = _top_learned_tokens(tokenizer, holdout_text, args.vocab_size, top_n=15)

    naive_compare = _compare_naive(train_text, args.vocab_size) if args.compare_naive else None

    total_seconds = time.perf_counter() - script_t0

    # ---- 汇总 JSON -----------------------------------------------------
    result = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "corpus": str(args.corpus),
        "config": {
            "bytes_requested": args.bytes,
            "train_chars": len(train_text),
            "train_bytes": train_bytes,
            "holdout_chars": len(holdout_text),
            "holdout_bytes": holdout_bytes,
            "vocab_size": args.vocab_size,
            "special_tokens": [SPECIAL],
            "pretokenizer": "gpt2_regex",
            "impl": "fast（反向索引 + 懒删除堆增量更新）",
        },
        "train": {
            "seconds": round(train_seconds, 4),
            "peak_rss_mb_after_train": round(mem_after_train, 1),
            "peak_rss_mb_before": round(mem_before, 1),
            "num_merges": len(merges),
        },
        "vocab": {
            "size": len(vocab),
            "num_special": 1,
            "num_merges": len(merges),
            "longest_token": _fmt_token(longest),
            "longest_token_len_bytes": len(longest),
            "top5_longest_tokens": [_fmt_token(t) for t in longest_tokens],
            "first_15_merges": first_15,
        },
        "holdout": {
            "num_tokens": holdout_tokens,
            "tokens_per_byte": round(holdout_tokens / holdout_bytes, 5)
            if holdout_bytes
            else None,
            "compression_bytes_per_token": round(compression, 4)
            if holdout_tokens
            else None,
            "encode_seconds": round(holdout_encode_seconds, 4),
            "roundtrip_ok": roundtrip_ok,
            "top15_learned_tokens_by_frequency": top_tokens,
        },
        "throughput": {
            "encode_cold": {
                "bytes": train_bytes,
                "seconds": round(encode_seconds, 4),
                "bytes_per_second": round(train_bytes / encode_seconds)
                if encode_seconds
                else None,
            },
            "encode_warm": {
                "seconds": round(warm_encode_seconds, 4),
                "bytes_per_second": round(train_bytes / warm_encode_seconds)
                if warm_encode_seconds
                else None,
            },
            "encode_iterable_cold": {
                "bytes": train_bytes,
                "seconds": round(iterable_seconds, 4),
                "bytes_per_second": round(train_bytes / iterable_seconds)
                if iterable_seconds
                else None,
                "chunk_bytes": 65_536,
                "matches_encode": iterable_matches,
            },
        },
        "naive_compare": naive_compare,
        "peak_rss_mb_final": round(mem_final, 1),
        "total_seconds": round(total_seconds, 3),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 终端中文汇总 --------------------------------------------------
    print("-" * 74)
    print(f"训练耗时        : {train_seconds:.3f} s（{len(merges)} 次 merge）")
    print(f"峰值内存        : {mem_final:.1f} MB（训练后 {mem_after_train:.1f} MB）")
    print(f"词表大小        : {len(vocab)}（256 字节 + 1 special + {len(merges)} merges）")
    print(f"最长 token      : {_fmt_token(longest)}（{len(longest)} 字节）")
    print(f"top5 最长 token : {', '.join(_fmt_token(t) for t in longest_tokens)}")
    print(f"前 5 条 merges  : ", end="")
    print(", ".join(f"{a!r}+{b!r}" for a, b in merges[:5]))
    print(f"留出段压缩比    : {compression:.3f} bytes/token（{holdout_bytes:,} 字节 -> {holdout_tokens:,} tokens）")
    print(f"留出段 tokens/B : {holdout_tokens / holdout_bytes:.4f}")
    print(f"roundtrip       : {'OK' if roundtrip_ok else 'FAILED'}")
    print(f"encode 吞吐     : {train_bytes / encode_seconds:,.0f} bytes/s（冷缓存 {encode_seconds:.3f}s；热缓存 {train_bytes / warm_encode_seconds:,.0f} bytes/s）")
    print(f"iterable 吞吐   : {train_bytes / iterable_seconds:,.0f} bytes/s（冷缓存 {iterable_seconds:.3f}s，与整段一致={iterable_matches}）")
    if naive_compare:
        print(
            f"naive 对照      : fast {naive_compare['fast_seconds']:.3f}s vs "
            f"naive {naive_compare['naive_seconds']:.3f}s "
            f"（{naive_compare['speedup']}x，merges 一致={naive_compare['same_merges']}）"
        )
    print("-" * 74)
    print("出现最多的学出来的 token（留出段统计）：")
    for row in top_tokens[:10]:
        print(f"  id={row['id']:<5} {row['token']:<24} len={row['length_bytes']:<3} count={row['count']}")
    print("-" * 74)
    print(f"总耗时          : {total_seconds:.2f} s")
    print(f"结果已写入      : {args.out}")
    print("=" * 74)
    return 0 if roundtrip_ok and iterable_matches else 1


if __name__ == "__main__":
    sys.exit(main())
