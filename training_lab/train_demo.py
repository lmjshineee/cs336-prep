#!/usr/bin/env python
"""training_lab/train_demo.py — 端到端低资源冒烟训练（SPEC §5.2）。

把训练工具链（``toy_training.py``）接到真实的小 Transformer + 真实 token 数据上：

    data/*.bin (memmap) -> TransformerLM -> AdamW + cosine warmup
      -> get_batch / forward / cross_entropy / backward / clip_gradients / step
      -> 周期 eval / checkpoint / resume / 采样 -> results/train_demo.json

它证明的不是「能训出好模型」，而是**整条训练链路真的可跑、可复现、可恢复**：
固定 val batch 上的 loss 有记录，中断恢复后参数逐位一致，采样不崩。

口径
----
- 模型：``model_lab.toy_transformer.TransformerLM``（untied，RoPE）。
- 工具：``training_lab.toy_training``（CE / AdamW / cosine_lr / clip / get_batch /
  save_checkpoint / load_checkpoint / generate）。
- 数据：``prep_data.py`` 产出的 ``data/meta.json`` + train.bin / val.bin。
- 默认配置：d_model=128、2 层、4 头、ctx=64、batch=16、600 步，目标 < 90 秒。
- ``--max-seconds`` 硬停，超时先存已跑结果。
- ``--verify-resume``：6 步 vs 3 步→存→新实例读→再 3 步，比较最终参数 maxdiff 与
  val loss；因为每步 batch 由 ``(seed, step)`` 确定性生成，两次运行数据流完全一致，
  正常应得到 **0 差异**。

热预算：``torch.set_num_threads(2)``；默认配置实测约 10–20 秒（远低于 90 秒）。

用法::

    .venv/bin/python training_lab/prep_data.py
    .venv/bin/python training_lab/train_demo.py
    .venv/bin/python training_lab/train_demo.py --steps 100 --max-seconds 30
    .venv/bin/python training_lab/train_demo.py --verify-resume
    .venv/bin/python training_lab/train_demo.py --resume training_lab/checkpoints/latest.pt
"""

from __future__ import annotations

import argparse
import json
import math
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
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT / "model_lab"))
sys.path.insert(0, str(REPO_ROOT / "bpe_lab"))

from toy_bpe import BPETokenizer  # noqa: E402

DEFAULT_DATA_DIR = HERE / "data"
DEFAULT_RESULTS_DIR = HERE / "results"
DEFAULT_CHECKPOINT_DIR = HERE / "checkpoints"
DEFAULT_OUT = DEFAULT_RESULTS_DIR / "train_demo.json"
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


def import_toy_training():
    """延迟导入工具箱：这样 toy_training.py 还没写好时也能看 --help / 报错清楚。"""
    try:
        import toy_training  # noqa: F401
    except ImportError as exc:  # pragma: no cover - 联调前会走到
        raise SystemExit(
            "找不到 training_lab/toy_training.py（或导入失败）。\n"
            f"原始错误：{type(exc).__name__}: {exc}\n"
            "toolbox 由 impl 按 SPEC §3 实现，完成后重跑本脚本。"
        ) from exc
    return toy_training


def build_model(TransformerLM, cfg: dict):
    """按配置构建模型；d_ff 默认 4*d_model。"""
    return TransformerLM(
        vocab_size=cfg["vocab_size"],
        context_length=cfg["context_length"],
        d_model=cfg["d_model"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        d_ff=cfg["d_ff"],
        tie_weights=False,
    )


@torch.no_grad()
def eval_loss(model, x, y, toy_training) -> float:
    """固定 batch 上的 loss（eval 模式，不建图）。"""
    model.eval()
    logits = model(x)
    loss = toy_training.cross_entropy(logits, y)
    model.train()
    return float(loss.item())


def batch_rng(seed: int, step: int) -> np.random.Generator:
    """第 ``step`` 步的确定性随机源：与训练历史无关，便于 resume 等价复现。"""
    return np.random.default_rng([int(seed), int(step)])


# ---------------------------------------------------------------------------
# 中断恢复等价性
# ---------------------------------------------------------------------------
def run_verify_resume(toy_training, TransformerLM, cfg: dict, args, val_x, val_y) -> dict:
    """6 步 vs 3 步 + 存档 + 新实例读档 + 3 步，比较参数与 val loss。"""
    t0 = time.perf_counter()
    ckpt_path = args.checkpoint_dir / "resume_verify.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    def one_step(model, optimizer, step: int) -> float:
        x, y = toy_training.get_batch(
            cfg["_train_data"],
            args.batch_size,
            args.context_length,
            device="cpu",
            rng=batch_rng(args.seed, step),
        )
        lr = toy_training.cosine_lr(step, args.lr, args.min_lr, args.warmup, args.steps)
        toy_training.set_lr(optimizer, lr)
        optimizer.zero_grad(set_to_none=True)
        loss = toy_training.cross_entropy(model(x), y)
        loss.backward()
        toy_training.clip_gradients(model.parameters(), args.grad_clip)
        optimizer.step()
        return float(loss.item())

    # ---- A：连续 6 步
    torch.manual_seed(args.seed)
    model_a = build_model(TransformerLM, cfg)
    opt_a = toy_training.AdamW(
        model_a.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    for step in range(6):
        one_step(model_a, opt_a, step)
    val_a = eval_loss(model_a, val_x, val_y, toy_training)
    params_a = [p.detach().clone() for p in model_a.parameters()]

    # ---- B：3 步 -> 存档 -> 用「不同初始化」的新实例读档 -> 再 3 步
    torch.manual_seed(args.seed)
    model_b = build_model(TransformerLM, cfg)
    opt_b = toy_training.AdamW(
        model_b.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    for step in range(3):
        one_step(model_b, opt_b, step)
    toy_training.save_checkpoint(model_b, opt_b, 3, ckpt_path)

    torch.manual_seed(args.seed + 12_345)  # 故意换初始化，证明 load 覆盖了全部参数
    model_c = build_model(TransformerLM, cfg)
    opt_c = toy_training.AdamW(
        model_c.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    loaded_iter = int(toy_training.load_checkpoint(ckpt_path, model_c, opt_c))
    for step in range(3, 6):
        one_step(model_c, opt_c, step)
    val_b = eval_loss(model_c, val_x, val_y, toy_training)
    params_b = [p.detach().clone() for p in model_c.parameters()]

    max_param_diff = max(
        (float((a - b).abs().max()) if a.numel() else 0.0)
        for a, b in zip(params_a, params_b)
    )
    val_loss_diff = abs(val_a - val_b)
    passed = max_param_diff < 1e-6 and val_loss_diff < 1e-6 and loaded_iter == 3
    return {
        "enabled": True,
        "plan": "A=6 steps; B=3 steps -> save -> fresh model loads -> 3 steps",
        "steps_total": 6,
        "steps_before_checkpoint": 3,
        "loaded_iteration": loaded_iter,
        "max_param_diff": max_param_diff,
        "val_loss_a": val_a,
        "val_loss_b": val_b,
        "val_loss_diff": val_loss_diff,
        "passed": bool(passed),
        "checkpoint": str(ckpt_path),
        "elapsed_s": round(time.perf_counter() - t0, 4),
    }


# ---------------------------------------------------------------------------
# 采样
# ---------------------------------------------------------------------------
def sample_texts(toy_training, model, tokenizer, val_data, cfg: dict, args) -> list[dict]:
    """用温度 + top-p 从两段 val 前缀续写，记录 prompt / 输出 / 参数。"""
    samples: list[dict] = []
    prompts = [(0, 16), (256, 16)]
    for idx, (offset, length) in enumerate(prompts):
        prompt_ids = [int(t) for t in val_data[offset : offset + length]]
        generator = torch.Generator().manual_seed(args.seed + 100 * idx)
        model.eval()
        out_ids = toy_training.generate(
            model,
            prompt_ids,
            max_new_tokens=args.sample_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            context_length=cfg["context_length"],
            generator=generator,
        )
        model.train()
        samples.append(
            {
                "prompt": tokenizer.decode(prompt_ids),
                "prompt_ids": prompt_ids,
                "output": tokenizer.decode(out_ids),
                "generated_text": tokenizer.decode(out_ids[len(prompt_ids) :]),
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_new_tokens": args.sample_tokens,
                "num_generated": len(out_ids) - len(prompt_ids),
            }
        )
    return samples


# ---------------------------------------------------------------------------
# 主训练
# ---------------------------------------------------------------------------
def train(toy_training, TransformerLM, meta: dict, cfg: dict, args) -> dict:
    wall_start = time.perf_counter()
    train_data = toy_training.load_tokens(meta["train_bin"])
    val_data = toy_training.load_tokens(meta["val_bin"])
    cfg["_train_data"] = train_data  # 给 verify_resume 用（不写进 JSON）

    device = "cpu"
    torch.manual_seed(args.seed)  # 模型初始化可复现
    model = build_model(TransformerLM, cfg)
    num_params = int(model.num_parameters()) if hasattr(model, "num_parameters") else int(
        sum(p.numel() for p in model.parameters())
    )
    optimizer = toy_training.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # 固定 val batch：整个训练过程都用它，曲线可比
    val_x, val_y = toy_training.get_batch(
        val_data,
        args.eval_batch_size,
        args.context_length,
        device=device,
        rng=np.random.default_rng([args.seed, 10_000_019]),
    )

    # ---- 起始状态：支持 --resume
    start_iter = 0
    if args.resume is not None:
        resume_path = args.resume
        if resume_path.is_dir():
            resume_path = resume_path / "latest.pt"
        if not resume_path.is_file():
            raise SystemExit(f"--resume 指定的 checkpoint 不存在：{resume_path}")
        start_iter = int(toy_training.load_checkpoint(resume_path, model, optimizer))
        if not args.quiet:
            print(f"[resume] 从 {resume_path} 恢复，iteration={start_iter}")

    curve: list[dict] = []
    checkpoints: list[dict] = []
    all_finite = True
    timed_out = False
    step = start_iter

    # 初始 loss（step 0：未做任何更新，用 step-0 的确定性 batch）
    t_train = time.perf_counter()
    # 「初始」loss：本段训练起点（step=start_iter）batch 上、任何更新之前的 loss。
    # 对 --resume 也成立，避免 NaN 污染 JSON。
    x0, y0 = toy_training.get_batch(
        train_data,
        args.batch_size,
        args.context_length,
        device=device,
        rng=batch_rng(args.seed, start_iter),
    )
    initial_train_loss = eval_loss(model, x0, y0, toy_training)
    initial_val_loss = eval_loss(model, val_x, val_y, toy_training)
    curve.append(
        {
            "step": start_iter,
            "train_loss": initial_train_loss,
            "val_loss": initial_val_loss,
            "lr": toy_training.cosine_lr(start_iter, args.lr, args.min_lr, args.warmup, args.steps),
            "grad_norm": 0.0,
            "tokens_per_s": 0.0,
        }
    )

    tokens_processed = 0
    last_train_loss = initial_train_loss
    last_grad_norm = 0.0
    last_lr = float(curve[0]["lr"])

    ckpt_dir = args.checkpoint_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    def save_ckpt(iteration: int) -> None:
        path = ckpt_dir / f"checkpoint_{iteration:06d}.pt"
        toy_training.save_checkpoint(model, optimizer, iteration, path)
        latest = ckpt_dir / "latest.pt"
        toy_training.save_checkpoint(model, optimizer, iteration, latest)
        checkpoints.append({"iteration": int(iteration), "path": str(path)})

    while step < args.steps:
        if time.perf_counter() - t_train >= args.max_seconds:
            timed_out = True
            break

        x, y = toy_training.get_batch(
            train_data,
            args.batch_size,
            args.context_length,
            device=device,
            rng=batch_rng(args.seed, step),
        )
        last_lr = float(
            toy_training.cosine_lr(step, args.lr, args.min_lr, args.warmup, args.steps)
        )
        toy_training.set_lr(optimizer, last_lr)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = toy_training.cross_entropy(logits, y)
        loss.backward()
        last_grad_norm = float(toy_training.clip_gradients(model.parameters(), args.grad_clip))
        optimizer.step()

        step += 1
        tokens_processed += int(x.numel())
        last_train_loss = float(loss.item())
        all_finite = all_finite and math.isfinite(last_train_loss)
        all_finite = all_finite and math.isfinite(last_grad_norm)

        is_last = step >= args.steps
        do_eval = (step % args.eval_every == 0) or is_last
        do_log = (step % args.log_every == 0) or do_eval
        if do_log:
            val_loss = eval_loss(model, val_x, val_y, toy_training) if do_eval else None
            if val_loss is not None:
                all_finite = all_finite and math.isfinite(val_loss)
            elapsed_train = max(time.perf_counter() - t_train, 1e-9)
            curve.append(
                {
                    "step": step,
                    "train_loss": last_train_loss,
                    "val_loss": val_loss,
                    "lr": last_lr,
                    "grad_norm": last_grad_norm,
                    "tokens_per_s": round(tokens_processed / elapsed_train, 1),
                }
            )
        if step % args.checkpoint_every == 0 and not is_last:
            save_ckpt(step)

    train_elapsed = time.perf_counter() - t_train

    # 最终 eval（可能和上面最后一条 curve 重复，避免重复点）
    final_val_loss = eval_loss(model, val_x, val_y, toy_training)
    all_finite = all_finite and math.isfinite(final_val_loss)
    if not curve or curve[-1]["step"] != step or curve[-1]["val_loss"] is None:
        curve.append(
            {
                "step": step,
                "train_loss": last_train_loss,
                "val_loss": final_val_loss,
                "lr": last_lr,
                "grad_norm": last_grad_norm,
                "tokens_per_s": round(tokens_processed / max(train_elapsed, 1e-9), 1),
            }
        )
    else:
        curve[-1]["val_loss"] = final_val_loss

    # 结束必存最终 checkpoint
    save_ckpt(step)

    # ---- 采样
    tokenizer = BPETokenizer.load(meta["tokenizer_dir"], special_tokens=SPECIAL_TOKENS)
    samples = sample_texts(toy_training, model, tokenizer, val_data, cfg, args)

    # ---- 中断恢复等价性（独立的一小组模型，不影响主训练结果）
    if args.verify_resume:
        resume_check = run_verify_resume(toy_training, TransformerLM, cfg, args, val_x, val_y)
    else:
        resume_check = {"enabled": False, "passed": None}

    # 初始/最终 train loss 用「起点更新前」与「最后一次训练 batch」比较。
    # --resume 时「下降 20%」不适用（起点 loss 已经很低），记为 None。
    if start_iter == 0 and math.isfinite(initial_train_loss) and initial_train_loss > 0:
        loss_reduction_pct = (initial_train_loss - last_train_loss) / initial_train_loss * 100.0
        drop_applicable = True
    else:
        loss_reduction_pct = None
        drop_applicable = False

    elapsed = time.perf_counter() - wall_start
    tokens_per_s = tokens_processed / max(train_elapsed, 1e-9)
    threads = torch.get_num_threads()

    checks = {
        "all_losses_finite": bool(all_finite),
        "loss_drop_applicable": bool(drop_applicable),
        "train_loss_dropped_20pct": bool(
            drop_applicable and loss_reduction_pct is not None and loss_reduction_pct >= 20.0
        ),
        "val_loss_recorded": bool(
            any(entry.get("val_loss") is not None for entry in curve)
        ),
        "elapsed_within_120s": bool(elapsed <= 120.0),
        "two_threads": bool(threads == 2),
        "steps_at_least_one": bool(step > start_iter),
        "samples_nonempty": bool(all(s["num_generated"] > 0 for s in samples)),
        "timed_out": bool(timed_out),
        "resume_check_passed": bool(resume_check.get("passed", True)),
        "num_parameters_positive": bool(num_params > 0),
    }

    return {
        "config": {
            "steps": args.steps,
            "start_iteration": start_iter,
            "batch_size": args.batch_size,
            "context_length": args.context_length,
            "d_model": args.d_model,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "d_ff": args.d_ff,
            "lr": args.lr,
            "min_lr": args.min_lr,
            "warmup": args.warmup,
            "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip,
            "max_seconds": args.max_seconds,
            "eval_every": args.eval_every,
            "log_every": args.log_every,
            "checkpoint_every": args.checkpoint_every,
            "checkpoint_dir": str(args.checkpoint_dir),
            "resume": str(args.resume) if args.resume else None,
            "sample_tokens": args.sample_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
            "vocab_size": cfg["vocab_size"],
            "tie_weights": False,
        },
        "curve": curve,
        "summary": {
            "elapsed_s": round(elapsed, 4),
            "train_elapsed_s": round(train_elapsed, 4),
            "steps_run": int(step),
            "steps_this_run": int(step - start_iter),
            "tokens_processed": int(tokens_processed),
            "tokens_per_s": round(tokens_per_s, 1),
            "initial_train_loss": initial_train_loss,
            "initial_train_loss_step": int(start_iter),
            "final_train_loss": last_train_loss,
            "initial_val_loss": initial_val_loss,
            "final_val_loss": final_val_loss,
            "loss_reduction_pct": round(loss_reduction_pct, 2) if loss_reduction_pct is not None else None,
            "num_parameters": num_params,
            "peak_rss_mib": round(peak_rss_mib(), 1),
            "threads": threads,
            "timed_out": bool(timed_out),
        },
        "checkpoints": checkpoints,
        "resume_check": resume_check,
        "samples": samples,
        "data": {
            "meta_path": str(meta.get("meta_path", "")),
            "train_bin": meta["train_bin"],
            "val_bin": meta["val_bin"],
            "train_tokens": meta["train_tokens"],
            "val_tokens": meta["val_tokens"],
            "compression_ratio": meta["compression_ratio"],
            "tokenizer_dir": meta["tokenizer_dir"],
        },
        "checks": checks,
        "meta": {
            "spec": "training_lab/SPEC.md §5.2",
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=None, help="默认 4*d_model")
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--min-lr", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-seconds", type=float, default=90.0, help="硬停时间（秒）")
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=200)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument(
        "--resume",
        nargs="?",
        const="__LATEST__",
        default=None,
        help="从 checkpoint 继续（省略路径时用 checkpoint-dir/latest.pt）",
    )
    parser.add_argument("--sample-tokens", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verify-resume", action="store_true")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--quiet", action="store_true")
    return parser


def normalize_resume(args) -> None:
    """``--resume`` 不带路径时解析成 checkpoint-dir/latest.pt。"""
    if args.resume == "__LATEST__":
        args.resume = args.checkpoint_dir / "latest.pt"
    elif args.resume is not None:
        args.resume = Path(args.resume)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    normalize_resume(args)
    if args.d_ff is None:
        args.d_ff = 4 * args.d_model
    if args.steps < 0 or args.batch_size <= 0 or args.context_length <= 1:
        raise SystemExit("参数非法：--steps >= 0、--batch-size > 0、--context-length > 1")
    if args.d_model % args.num_heads != 0:
        raise SystemExit("--d-model 必须能被 --num-heads 整除")

    toy_training = import_toy_training()
    from toy_transformer import TransformerLM

    meta_path = args.data_dir / "meta.json"
    if not meta_path.is_file():
        raise SystemExit(
            f"缺少 {meta_path}。先运行：.venv/bin/python training_lab/prep_data.py"
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["meta_path"] = str(meta_path)
    for key in ("train_bin", "val_bin", "tokenizer_dir", "vocab_size", "train_tokens", "val_tokens"):
        if key not in meta:
            raise SystemExit(f"meta.json 缺少字段 {key!r}，请重跑 prep_data.py")
    if args.context_length + 1 > int(meta["train_tokens"]):
        raise SystemExit("--context-length 超过训练 token 总数")

    cfg = {
        "vocab_size": int(meta["vocab_size"]),
        "context_length": args.context_length,
        "d_model": args.d_model,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "d_ff": args.d_ff,
    }

    result = train(toy_training, TransformerLM, meta, cfg, args)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    if not args.quiet:
        s = result["summary"]
        print("=" * 84)
        print("train_demo：端到端冒烟训练（TransformerLM + toy_training 工具箱）")
        print("=" * 84)
        print(
            f"data: train={meta['train_tokens']:,} tokens  val={meta['val_tokens']:,} tokens  "
            f"vocab={cfg['vocab_size']}"
        )
        print(
            f"model: d_model={cfg['d_model']} layers={cfg['num_layers']} heads={cfg['num_heads']} "
            f"d_ff={cfg['d_ff']} ctx={cfg['context_length']} params={s['num_parameters']:,}"
        )
        print(
            f"steps={s['steps_run']} (this run {s['steps_this_run']})  batch={args.batch_size}  "
            f"elapsed={s['elapsed_s']:.2f}s  train_elapsed={s['train_elapsed_s']:.2f}s  "
            f"tokens/s={s['tokens_per_s']:.1f}"
        )
        drop_txt = (
            f"{s['loss_reduction_pct']:.2f}% down"
            if s["loss_reduction_pct"] is not None
            else "n/a (resumed run)"
        )
        print(
            f"train loss: initial={s['initial_train_loss']:.4f} -> final={s['final_train_loss']:.4f} "
            f"({drop_txt})"
        )
        print(
            f"val loss: initial={s['initial_val_loss']:.4f} -> final={s['final_val_loss']:.4f}  "
            f"timed_out={s['timed_out']}  threads={s['threads']}  peak_rss={s['peak_rss_mib']} MiB"
        )
        if result["resume_check"]["enabled"]:
            rc = result["resume_check"]
            print(
                f"verify-resume: max_param_diff={rc['max_param_diff']:.3e}  "
                f"val_loss_diff={rc['val_loss_diff']:.3e}  passed={rc['passed']}  "
                f"loaded_iteration={rc['loaded_iteration']}"
            )
        for sample in result["samples"]:
            print(
                f"sample (T={sample['temperature']}, top_p={sample['top_p']}): "
                f"{sample['output'][:110]!r}"
            )
        print(f"checkpoints: {[c['iteration'] for c in result['checkpoints']]}")
        print(f"checks: {result['checks']}")
        print(f"[out] {args.out}")

    checks = result["checks"]
    required = [
        "all_losses_finite",
        "elapsed_within_120s",
        "two_threads",
        "steps_at_least_one",
        "samples_nonempty",
    ]
    if args.steps >= 100 and result["config"]["start_iteration"] == 0 and not checks["timed_out"]:
        required.append("train_loss_dropped_20pct")
    if args.verify_resume:
        required.append("resume_check_passed")
    failed = [name for name in required if not checks[name]]
    if failed:
        print(f"[FAIL] checks failed: {failed}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
