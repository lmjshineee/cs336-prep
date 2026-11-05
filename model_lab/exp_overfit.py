#!/usr/bin/env python
"""model_lab/exp_overfit.py — 单 batch 过拟合实验（SPEC §7.2）。

目的：证明 ``model_lab/toy_transformer.py`` 的实现**真的可训练**（不是只形状对）。
在固定的 B=8、T=64 小 batch 上反复优化，观察 loss 是否能被压下去；训练结束时用
greedy 解码生成 64 字节，直观看看模型是否记住了这段字节串。

口径（SPEC §7.2）
-----------------
- 数据：``{A1_REPO}/tests/fixtures/tinystories_sample.txt``（约 3.8KB，只读）；
  缺失或过短时回退到内置合成文本。
- tokenizer：纯字节级（``bytes`` <-> id，vocab=256），不依赖 ``bpe_lab``。
- 模型：``d_model=64, num_layers=2, num_heads=4, d_ff=128, ctx=64``，untied。
- 训练：**固定一个 batch**，AdamW ``lr=3e-3``、``weight_decay=0.0``，
  最多 300 步或 ``--max-seconds``（默认 90）硬停；每 10 步记录一次固定 batch 上的
  eval loss；结束后 greedy 生成 64 字节。
- 输出：``model_lab/results/overfit.json``，字段见 SPEC §7.2。
- ``final_loss < 0.5`` 是**目标**而不是判定标准：达不到就如实记录，不调参包装。

热预算：2 线程、单命令 < 120 秒。

用法::

    .venv/bin/python model_lab/exp_overfit.py
    .venv/bin/python model_lab/exp_overfit.py --max-seconds 90 --max-steps 300
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import resource
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

# SPEC §1：所有可能超过 1 秒的脚本必须 2 线程
torch.set_num_threads(2)

MIB = 1024 * 1024
HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "results" / "overfit.json"
DEFAULT_A1_REPO = Path("/Users/yunni/Projects/assignment1-basics")
DATA_REL = Path("tests") / "fixtures" / "tinystories_sample.txt"

# 内置兜底文本（> 1024 字节，足够凑出 B*T+1 = 513 个 token）
SYNTHETIC_TEXT = (
    "Once upon a time, there was a little girl named Lily. She lived in a small house "
    "near the edge of the forest with her mother and her dog Max. Every morning Lily "
    "would wake up early and run to the garden to see the flowers. The sun was warm and "
    "the birds were singing. One day, Lily found a tiny blue bird under a big tree. The "
    "bird could not fly, so Lily picked it up gently and put it in a soft box. She gave "
    "it water and a little bread. The next morning, the bird was strong again. It flew "
    "up into the sky and sang a happy song. Lily smiled and waved goodbye. Max barked "
    "and ran around the garden. That night, Lily told her mother the story of the blue "
    "bird. Her mother hugged her and said that kind hearts always find friends. Lily "
    "fell asleep with a smile on her face, dreaming of the bird flying over the green "
    "hills. The end. "
)


# ---------------------------------------------------------------- tokenizer


class ByteTokenizer:
    """最朴素的字节级 tokenizer：一个字节就是一个 token，vocab=256。"""

    vocab_size = 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids) -> str:
        raw = bytes(int(i) & 0xFF for i in ids)
        return raw.decode("utf-8", errors="replace")

    def decode_hex(self, ids) -> str:
        return bytes(int(i) & 0xFF for i in ids).hex()


# ---------------------------------------------------------------- data


def load_data(a1_repo: Path, seed: int) -> dict:
    """读取 TinyStories 片段（只读）；失败时用内置合成文本兜底。"""
    path = a1_repo / DATA_REL
    info = {"path": str(path), "source": "synthetic_fallback", "bytes": 0, "fallback_reason": None}
    if path.is_file():
        try:
            raw = path.read_bytes()  # 只读，不修改官方仓库
            if len(raw) >= 1024:
                info.update(source="tinystories_sample.txt", bytes=len(raw))
                return {"raw": raw, "info": info}
            info["fallback_reason"] = f"file too short ({len(raw)} bytes)"
        except OSError as exc:  # pragma: no cover - 防御性
            info["fallback_reason"] = f"{type(exc).__name__}: {exc}"
    else:
        info["fallback_reason"] = "file not found"

    raw = SYNTHETIC_TEXT.encode("utf-8")
    info.update(bytes=len(raw))
    return {"raw": raw, "info": info}


def make_batch(raw: bytes, batch_size: int, seq_len: int, offset: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """从字节串切出固定 batch：x 与 y 相差一个位置（next-token）。"""
    data = torch.tensor(list(raw), dtype=torch.long)
    need = batch_size * seq_len + 1
    if data.numel() < need:
        # 数据太短：重复拼接到达标（兜底文本足够长，正常不会走到这里）
        reps = math.ceil(need / max(data.numel(), 1))
        data = data.repeat(reps)
    start = offset % max(data.numel() - need + 1, 1)
    chunk = data[start : start + batch_size * seq_len + 1]
    x = chunk[:-1].reshape(batch_size, seq_len)
    y = chunk[1:].reshape(batch_size, seq_len)
    return x, y


# ---------------------------------------------------------------- model


def build_model(cfg: dict):
    """从 toy_transformer 构建模型（延迟导入以便先跑数据检查）。"""
    sys.path.insert(0, str(HERE))
    import toy_transformer

    return toy_transformer.TransformerLM(
        vocab_size=cfg["vocab_size"],
        context_length=cfg["context_length"],
        d_model=cfg["d_model"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        d_ff=cfg["d_ff"],
        tie_weights=cfg["tie_weights"],
    )


# ---------------------------------------------------------------- runtime helpers


def peak_rss_mib() -> float:
    """进程峰值 RSS（MiB）。macOS 的 ru_maxrss 单位是字节，Linux 是 KiB。"""
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru / MIB if sys.platform == "darwin" else ru / 1024.0


@torch.no_grad()
def eval_loss(model, x: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    logits = model(x)
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
    model.train()
    return float(loss.item())


@torch.no_grad()
def greedy_generate(model, prompt: torch.Tensor, n_new: int, context_length: int) -> torch.Tensor:
    """贪心解码：每次取 argmax，滑动窗口不超过 context_length。"""
    model.eval()
    ids = prompt.reshape(1, -1)
    for _ in range(n_new):
        window = ids[:, -context_length:]
        logits = model(window)
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        ids = torch.cat([ids, next_id], dim=1)
    model.train()
    return ids[0, prompt.numel():]


# ---------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    wall_start = time.perf_counter()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--a1-repo", type=Path, default=Path(os.environ.get("A1_REPO", DEFAULT_A1_REPO)))
    parser.add_argument("--max-seconds", type=float, default=90.0, help="硬停时间（秒）")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample-bytes", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    cfg = {
        "vocab_size": 256,
        "context_length": 64,
        "d_model": 64,
        "num_layers": 2,
        "num_heads": 4,
        "d_ff": 128,
        "tie_weights": False,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "optimizer": "AdamW",
        "lr": args.lr,
        "weight_decay": 0.0,
        "max_steps": args.max_steps,
        "max_seconds": args.max_seconds,
        "seed": args.seed,
        "loss": "F.cross_entropy",
    }

    # ---- 数据
    data = load_data(args.a1_repo, args.seed)
    tokenizer = ByteTokenizer()
    x, y = make_batch(data["raw"], cfg["batch_size"], cfg["seq_len"], offset=0)
    data_info = dict(data["info"])
    data_info.update(
        batch_shape=list(x.shape),
        tokens_per_step=int(x.numel()),
        offset=0,
    )

    # ---- 模型
    torch.manual_seed(cfg["seed"])
    model = build_model(cfg)
    num_params = int(sum(p.numel() for p in model.parameters()))
    n_params_method = int(model.num_parameters()) if hasattr(model, "num_parameters") else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    # ---- 初始 loss（第 0 步，尚未更新）
    train_start = time.perf_counter()
    initial_loss = eval_loss(model, x, y)
    curve: list[list[float]] = [[0, initial_loss]]

    # ---- 训练：固定 batch，硬停
    step = 0
    timed_out = False
    all_finite = math.isfinite(initial_loss)
    while step < cfg["max_steps"]:
        if time.perf_counter() - train_start >= cfg["max_seconds"]:
            timed_out = True
            break
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        loss.backward()
        optimizer.step()
        step += 1
        all_finite = all_finite and math.isfinite(float(loss.item()))

        if step % 10 == 0:
            value = eval_loss(model, x, y)
            all_finite = all_finite and math.isfinite(value)
            curve.append([step, value])

    train_elapsed = time.perf_counter() - train_start

    # ---- 最终 loss（训练结束后的固定 batch eval）
    final_loss = eval_loss(model, x, y)
    all_finite = all_finite and math.isfinite(final_loss)
    if curve and curve[-1][0] == step:
        curve[-1][1] = final_loss
    else:
        curve.append([step, final_loss])

    # ---- greedy 生成 64 字节（prompt 取固定 batch 的首 8 个字节）
    prompt = x.reshape(-1)[:8].clone()
    gen_start = time.perf_counter()
    generated = greedy_generate(model, prompt, args.sample_bytes, cfg["context_length"])
    gen_elapsed = time.perf_counter() - gen_start
    sample_prompt = tokenizer.decode(prompt.tolist())
    sample = tokenizer.decode(generated.tolist())
    sample_hex = tokenizer.decode_hex(generated.tolist())

    elapsed = time.perf_counter() - wall_start
    threads = torch.get_num_threads()
    peak_rss = peak_rss_mib()
    tokens_per_s = (step * cfg["batch_size"] * cfg["seq_len"] / train_elapsed) if train_elapsed > 0 else 0.0

    checks = {
        "all_losses_finite": bool(all_finite),
        "final_loss_below_initial": bool(final_loss < initial_loss),
        "elapsed_within_120s": bool(elapsed <= 120.0),
        "two_threads": bool(torch.get_num_threads() == 2),
        "steps_at_least_one": bool(step >= 1),
        "sample_nonempty": bool(len(sample) > 0),
        # 目标，不是判定标准：如实记录是否达到
        "final_loss_below_0_5_target": bool(final_loss < 0.5),
        "timed_out": bool(timed_out),
        "param_count_consistent": bool(num_params == n_params_method),
    }

    payload = {
        "config": cfg,
        "steps_run": step,
        "max_seconds": cfg["max_seconds"],
        "curve": curve,
        "final_loss": final_loss,
        "initial_loss": initial_loss,
        "elapsed_s": round(elapsed, 4),
        "train_elapsed_s": round(train_elapsed, 4),
        "tokens_per_s": round(tokens_per_s, 1),
        "threads": threads,
        "peak_rss_mib": round(peak_rss, 1),
        "sample": sample,
        "sample_prompt": sample_prompt,
        "sample_hex": sample_hex,
        "sample_bytes": len(generated),
        "num_parameters": num_params,
        "data": data_info,
        "checks": checks,
        "meta": {
            "spec": "model_lab/SPEC.md §7.2",
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "generation_elapsed_s": round(gen_elapsed, 4),
            "curve_kind": "eval_loss_on_fixed_batch",
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

    if not args.quiet:
        print("=" * 84)
        print("单 batch 过拟合实验（固定 batch，AdamW）")
        print("=" * 84)
        print(
            f"data={data_info['source']} ({data_info['bytes']} bytes)  "
            f"batch={cfg['batch_size']}x{cfg['seq_len']}  params={num_params:,}"
        )
        print(f"steps={step} (max {cfg['max_steps']})  timed_out={timed_out}")
        print(f"loss: initial={initial_loss:.6f}  final={final_loss:.6f}")
        print(f"curve ({len(curve)} 点): {[(int(s), round(v, 4)) for s, v in curve[:4]]} ... "
              f"{[(int(s), round(v, 4)) for s, v in curve[-3:]]}")
        print(f"elapsed={elapsed:.3f}s  train_elapsed={train_elapsed:.3f}s  tokens/s={tokens_per_s:.1f}")
        print(f"threads={threads}  peak_rss_mib={peak_rss:.1f}")
        print(f"greedy sample ({len(generated)} bytes): {sample[:120]!r}")
        print(f"checks: {checks}")
        print(f"[out] {args.out}")

    failed = [
        name
        for name in ("all_losses_finite", "final_loss_below_initial", "elapsed_within_120s", "two_threads", "param_count_consistent")
        if not checks[name]
    ]
    if failed:
        print(f"[FAIL] checks failed: {failed}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
