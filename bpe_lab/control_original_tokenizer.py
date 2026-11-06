"""control_original_tokenizer.py —— 补上“原 tokenizer、同预算继续训练”的 control。

背景：`transplant_gpt2.py` 把 GPT-2 的 tokenizer 换成一个自训 4096 BPE 后，继续训练
600 步，BPB 从 1.019 降到 0.782，低于原基线 0.956。但 0.782 也可能是“在 TinyStories
上继续训练”本身的域适应收益，而不是换 tokenizer 的功劳。要判断“换 tokenizer 的额外
代价”，必须有一个 control：**同一个 GPT-2、原 tokenizer、同样的 600 步 schedule、
同一份数据/seed/eval**。

这个脚本只做这一件事，输出 `bpe_lab/results/control_original_tokenizer.json`，供
教程 §7.3.5 与 LAB_NOTES 引用。

预算（1 线程 + 主动降温、无风扇笔记本）：约 11 分钟（本机实测 636s）；`--max-seconds`
默认 900，超时会保存部分结果并标 `completed=false`，不会把机器烤到底。

复现（仓库根目录）：
    export UV_CACHE_DIR=$PWD/.tmp/uvcache HF_HOME=$PWD/.tmp/hf MPLCONFIGDIR=$PWD/.tmp/mpl \
           HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    .venv/bin/python bpe_lab/control_original_tokenizer.py --threads 2
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from transplant_gpt2 import (  # noqa: E402
    DEFAULT_CORPUS,
    DEFAULT_WEIGHTS,
    RESULTS_DIR,
    SPECIAL,
    _load_gpt2,
    _spans_with_bytes,
    eval_bpb,
    peak_rss_mb,
)
from toy_bpe import unicode_to_bytes  # noqa: E402


def _split_docs(corpus: Path, train_fraction: float, eval_fraction: float):
    docs = corpus.read_text(encoding="utf-8").split(SPECIAL)
    n = len(docs)
    n_train = int(n * train_fraction)
    n_eval_end = int(n * (train_fraction + eval_fraction))
    return docs[:n_train], docs[n_train:n_eval_end]


def _take_docs_by_bytes(docs: list[str], max_bytes: int) -> list[str]:
    """按文档边界取前若干篇，累计不超过 max_bytes（至少取 1 篇）。"""
    out, total = [], 0
    for doc in docs:
        if out and total >= max_bytes:
            break
        out.append(doc)
        total += len(doc.encode("utf-8")) + len(SPECIAL)
    return out


def _make_batch(ids: torch.Tensor, batch: int, seq: int, gen: torch.Generator):
    starts = torch.randint(0, len(ids) - seq - 1, (batch,), generator=gen)
    return torch.stack([ids[s : s + seq] for s in starts])


def main() -> int:
    parser = argparse.ArgumentParser(description="原 tokenizer 同预算继续训练（control）")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--eval-fraction", type=float, default=0.05)
    parser.add_argument(
        "--steps-phase1", type=int, default=0, help="默认 0；warmup 只对新初始化的 embedding 行有意义"
    )
    parser.add_argument("--steps-phase2", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--lr-phase1", type=float, default=1e-4)
    parser.add_argument("--lr-phase2", type=float, default=1e-4)
    parser.add_argument("--lm-train-bytes", type=int, default=400_000)
    parser.add_argument("--eval-window", type=int, default=256)
    parser.add_argument("--eval-batch", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--threads", type=int, default=1, help="默认 1 线程，避免无风扇机器过热"
    )
    parser.add_argument(
        "--sleep-every", type=int, default=0, help="每 N 步睡 --sleep-seconds 秒（0=关闭）"
    )
    parser.add_argument("--sleep-seconds", type=float, default=0.5)
    parser.add_argument(
        "--cooldown-every", type=int, default=200, help="每 N 步主动降温一次（0=关闭）"
    )
    parser.add_argument("--cooldown-seconds", type=float, default=60.0)
    parser.add_argument("--max-seconds", type=float, default=900.0, help="硬超时：到点保存部分结果")
    parser.add_argument(
        "--out", type=Path, default=RESULTS_DIR / "control_original_tokenizer.json"
    )
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    t_start = time.perf_counter()

    if not args.corpus.exists():
        print(f"找不到语料：{args.corpus}", file=sys.stderr)
        return 2

    train_docs, eval_docs = _split_docs(args.corpus, args.train_fraction, args.eval_fraction)
    train_docs = _take_docs_by_bytes(train_docs, args.lm_train_bytes)
    print(
        f"文档：训练 {len(train_docs)} 篇（≈{sum(len(d.encode()) for d in train_docs):,} B）；"
        f"eval {len(eval_docs)} 篇"
    )

    old_tok, model = _load_gpt2(args.weights)
    model.train()

    u2b = unicode_to_bytes()
    old_byte_of_id = {
        tid: bytes(u2b[ch] for ch in token_str)
        for token_str, tid in old_tok.get_vocab().items()
    }
    eos_id = old_tok.eos_token_id
    eos_bytes = old_byte_of_id[eos_id]

    encode = lambda s: old_tok.encode(s, add_special_tokens=False)  # noqa: E731
    train_ids, train_tb = _spans_with_bytes(encode, old_byte_of_id.get, train_docs, eos_id, eos_bytes)
    eval_ids, eval_tb = _spans_with_bytes(encode, old_byte_of_id.get, eval_docs, eos_id, eos_bytes)
    print(f"训练 {len(train_ids):,} tokens；eval {len(eval_ids):,} tokens")

    def run_eval() -> dict:
        return eval_bpb(
            model,
            eval_ids,
            eval_tb,
            model.config.vocab_size,
            window=args.eval_window,
            batch_size=args.eval_batch,
        )

    t0 = time.perf_counter()
    baseline = run_eval()
    print(f"基线 BPB（原 tokenizer，未继续训练）= {baseline['bpb']:.5f}（{baseline['seconds']}s）")

    id_tensor = torch.tensor(train_ids, dtype=torch.long)
    gen = torch.Generator().manual_seed(args.seed)
    recent: list[float] = []
    curve: list[dict] = [{"step": 0, "phase": "baseline", "eval_full_bpb": round(baseline["bpb"], 5)}]
    phase_times: dict[str, float] = {}
    timed_out = False
    steps_done = 0

    def step_once(optimizer) -> float:
        x = _make_batch(id_tensor, args.batch_size, args.seq_len, gen)
        logits = model(x).logits[:, :-1, :]          # 位置 t 预测 token t+1
        loss = F.cross_entropy(
            logits.reshape(-1, model.config.vocab_size), x[:, 1:].reshape(-1)
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        recent.append(float(loss.detach()))
        return float(loss.detach())

    def pace(step: int) -> None:
        """热管理：定期打印进度 + 主动降温，避免无风扇机器持续满载。"""
        if step % 50 == 0:
            elapsed = time.perf_counter() - t_start
            print(
                f"  step {step:4d}  loss(近20)={sum(recent[-20:])/max(len(recent[-20:]),1):.4f}"
                f"  elapsed={elapsed:6.1f}s",
                flush=True,
            )
        if args.sleep_every and step % args.sleep_every == 0:
            time.sleep(args.sleep_seconds)
        if (
            args.cooldown_every
            and step % args.cooldown_every == 0
            and step < args.steps_phase1 + args.steps_phase2
        ):
            print(f"  cooling down {args.cooldown_seconds:.0f}s ...", flush=True)
            time.sleep(args.cooldown_seconds)

    # 相位 1：只训 embedding（GPT-2 tied，lm_head 共享同一参数）
    for p in model.parameters():
        p.requires_grad_(False)
    emb = model.get_input_embeddings().weight
    emb.requires_grad_(True)
    opt1 = torch.optim.AdamW([emb], lr=args.lr_phase1)
    t1 = time.perf_counter()
    for _ in range(args.steps_phase1):
        if time.perf_counter() - t_start > args.max_seconds:
            timed_out = True
            break
        step_once(opt1)
        steps_done += 1
        pace(steps_done)
    phase_times["phase1_seconds"] = round(time.perf_counter() - t1, 2)
    model.eval()
    after_phase1 = run_eval() if (not timed_out and args.steps_phase1 > 0) else None
    model.train()
    if after_phase1:
        print(f"相位 1 后 BPB = {after_phase1['bpb']:.5f}")
        curve.append(
            {"step": steps_done, "phase": "phase1", "eval_full_bpb": round(after_phase1["bpb"], 5)}
        )

    # 相位 2：全参
    if not timed_out:
        for p in model.parameters():
            p.requires_grad_(True)
        opt2 = torch.optim.AdamW(model.parameters(), lr=args.lr_phase2)
        t2 = time.perf_counter()
        for _ in range(args.steps_phase2):
            if time.perf_counter() - t_start > args.max_seconds:
                timed_out = True
                break
            step_once(opt2)
            steps_done += 1
            pace(steps_done)
        phase_times["phase2_seconds"] = round(time.perf_counter() - t2, 2)

    model.eval()
    after_training = None
    if not timed_out:
        after_training = run_eval()
        print(f"继续训练后 BPB = {after_training['bpb']:.5f}")
        curve.append(
            {"step": steps_done, "phase": "phase2", "eval_full_bpb": round(after_training["bpb"], 5)}
        )

    result = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "experiment": "control: original tokenizer, same continued-training budget",
        "config": {
            "corpus": str(args.corpus),
            "weights": str(args.weights),
            "train_fraction": args.train_fraction,
            "eval_fraction": args.eval_fraction,
            "steps_phase1": args.steps_phase1,
            "steps_phase2": args.steps_phase2,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "lr_phase1": args.lr_phase1,
            "lr_phase2": args.lr_phase2,
            "lm_train_bytes": args.lm_train_bytes,
            "eval_window": args.eval_window,
            "seed": args.seed,
            "threads": args.threads,
            "max_seconds": args.max_seconds,
            "sleep_every": args.sleep_every,
            "sleep_seconds": args.sleep_seconds,
            "cooldown_every": args.cooldown_every,
            "cooldown_seconds": args.cooldown_seconds,
        },
        "corpus": {
            "train_docs": len(train_docs),
            "eval_docs": len(eval_docs),
            "train_tokens": len(train_ids),
            "eval_tokens": len(eval_ids),
            "eval_bytes": sum(len(b) for b in eval_tb),
        },
        "bpb": {
            "baseline": {k: (round(v, 5) if isinstance(v, float) else v) for k, v in baseline.items()},
            "after_phase1": (
                {k: (round(v, 5) if isinstance(v, float) else v) for k, v in after_phase1.items()}
                if after_phase1
                else None
            ),
            "after_training": (
                {k: (round(v, 5) if isinstance(v, float) else v) for k, v in after_training.items()}
                if after_training
                else None
            ),
        },
        "training": {
            "curve": curve,
            "steps_done": steps_done,
            "timed_out": timed_out,
            "phase_times_seconds": phase_times,
            "train_loss_mean_last20": round(sum(recent[-20:]) / max(len(recent[-20:]), 1), 4)
            if recent
            else None,
        },
        "timings_seconds": {
            "total": round(time.perf_counter() - t_start, 1),
            "baseline_eval": baseline["seconds"],
        },
        "peak_rss_mb": round(peak_rss_mb(), 1),
        "completed": not timed_out,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "torch": torch.__version__,
            "threads": torch.get_num_threads(),
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已写入 {args.out}（completed={result['completed']}，{result['timings_seconds']['total']}s）")

    # 与 transplant 结果并排打印（如果存在）
    transplant_path = RESULTS_DIR / "transplant.json"
    if transplant_path.exists() and after_training:
        t = json.loads(transplant_path.read_text(encoding="utf-8"))
        b = t["bpb"]
        print("\n=== transplant vs control（同一 eval、同一预算）===")
        print(f"基线（原 tokenizer）      : {b['baseline_old_tokenizer']['bpb']:.4f}")
        print(f"换 tokenizer 后训练 600 步: {b['after_training']['bpb']:.4f}")
        print(f"原 tokenizer 训练 600 步  : {after_training['bpb']:.4f}")
        print(
            f"换 tokenizer 的额外代价   : "
            f"{b['after_training']['bpb'] - after_training['bpb']:+.4f} BPB"
            f"（负值 = 换后更好）"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
