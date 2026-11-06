"""transplant_gpt2.py —— 把「自己训的 BPE tokenizer」换到真实 GPT-2 上。

实验问题：给定一个已经在旧 tokenizer（GPT-2，50,257 词表）上训练好的 LM，如果把
tokenizer 换成一个从零训练的新 BPE（词表 4,096），模型会掉多少？继续训练能不能补回来？

步骤（全部离线、CPU）：
  1. 读 TinyStories fixture（5MB，`<|endoftext|>` 分文档），按文档切：
     前 80% 文档训练新 tokenizer，80%–85% 的文档做 eval（与训练完全不相交）。
  2. 用 `toy_bpe.train_bpe` 训练 4096 词表（special = `<|endoftext|>`）。
  3. 基线：原始 GPT-2 + 原 tokenizer 在 eval 上的 bits-per-byte（滑窗 256、batch 4）。
  4. 换 tokenizer 的关键工程：`resize_token_embeddings(4096)` 之后如何初始化新 embedding。
     - 单字节 token 0..255：复制旧词表里「字节序列恰好是该单字节」的 token 的 embedding；
     - 多字节 token：bytes 能 strict 解成 UTF-8 → 用旧 tokenizer 编码成旧 id，取均值；
       不能（非法 UTF-8）→ 取该 token 各字节对应单字节 embedding 的均值；
     - special token → 复制旧 `<|endoftext|>`（50256）那一行；
     - 因为 GPT-2 是 tied embedding，还要 `tie_weights()` 并断言 lm_head/wte 共享存储。
  5. 评估三点：换之前（基线）/ 换之后未训练 / 继续训练之后。
  6. 继续训练：相位 1（200 步）只训 embedding+lm_head，lr 1e-3；
     相位 2（400 步）全参，lr 1e-4；AdamW、grad clip 1.0、batch 4×128。
     每 100 步在固定 eval 切片上记录 BPB，最后在完整 eval 上重测。

产出：
  * `bpe_lab/results/transplant.json`：配置 / tokens-per-byte / 三点 BPB / 训练曲线 / 耗时 / 机器信息；
  * `bpe_lab/results/transplant_bpb.png`：只用「完整 eval 的三个点」+ baseline 水平线画的图
    （不混入小切片 BPB；逐步曲线只保留在 JSON 的 `training.curve` 里）。

复现（在仓库根目录）：
    export UV_CACHE_DIR=$PWD/.tmp/uvcache HF_HOME=$PWD/.tmp/hf MPLCONFIGDIR=$PWD/.tmp/mpl \
           HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    .venv/bin/python bpe_lab/transplant_gpt2.py
    # 本仓库结果 JSON 对应的完整命令（含初始化消融）：
    .venv/bin/python -u bpe_lab/transplant_gpt2.py --ablation-steps 50
    # 冒烟测试（几十秒）：
    .venv/bin/python bpe_lab/transplant_gpt2.py --quick

注意：这是教学实验，不是「生产级 tokenizer 替换」；局限见 LAB_NOTES.md。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import resource
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from toy_bpe import BPETokenizer, train_bpe, unicode_to_bytes  # noqa: E402

SPECIAL = "<|endoftext|>"
DEFAULT_CORPUS = Path(
    "/Users/yunni/Projects/assignment1-basics/tests/fixtures/tinystories_sample_5M.txt"
)
DEFAULT_WEIGHTS = Path("/Users/yunni/Projects/cs336-prep-code/real_llm_lab/weights/gpt2")
RESULTS_DIR = HERE / "results"


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------
def peak_rss_mb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1024 * 1024) if sys.platform == "darwin" else value / 1024


def _load_gpt2(weights: Path):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(weights), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(str(weights), local_files_only=True)
    return tokenizer, model


# ---------------------------------------------------------------------------
# 编码 + 每个 token 的原始字节（用于精确统计 bytes）
# ---------------------------------------------------------------------------
def _spans_with_bytes(encode_fn, bytes_of_id, docs, special_id, special_bytes):
    """把文档列表编码成一条 id 流，并同步记录每个 id 对应的 UTF-8 原始字节。

    为什么需要 bytes_of_id：BPB 的分母是「原始 UTF-8 字节数」。直接 decode 一段
    token 窗口可能因为窗口边界切断多字节字符而失真；按 token 的原始字节累加则永远精确。
    """
    ids: list[int] = []
    token_bytes: list[bytes] = []
    for index, doc in enumerate(docs):
        if index:
            ids.append(special_id)
            token_bytes.append(special_bytes)
        part = encode_fn(doc)
        ids.extend(part)
        token_bytes.extend(bytes_of_id(i) for i in part)
    return ids, token_bytes


def eval_bpb(
    model: torch.nn.Module,
    ids: list[int],
    token_bytes: list[bytes],
    vocab_size: int,
    *,
    window: int = 256,
    batch_size: int = 4,
) -> dict:
    """在一条 id 流上算 bits-per-byte（滑窗、无梯度、batch 4×256）。

    BPB = (总交叉熵 nats) / ln2 / (被计分 token 覆盖的原始字节数)。
    窗口内第 0 个 token 没有上下文，不参与计分（handout 的 ids[:, :-1] 写法），
    因此分母用「被计分 token 的原始字节和」而不是整段文本字节数；
    `coverage` 字段给出计分覆盖的字节比例（通常 >99%）。
    """
    total_loss = 0.0
    total_bytes = 0
    total_scored = 0
    started = time.perf_counter()
    model.eval()

    full_windows: list[tuple[int, torch.Tensor]] = []
    tail_windows: list[tuple[int, torch.Tensor]] = []
    for start in range(0, len(ids), window):
        chunk = ids[start : start + window]
        tensor = torch.tensor(chunk, dtype=torch.long)
        (full_windows if len(chunk) == window else tail_windows).append((start, tensor))

    def count_bytes(start: int, length: int) -> int:
        return sum(len(b) for b in token_bytes[start + 1 : start + length])

    with torch.no_grad():
        for i in range(0, len(full_windows), batch_size):
            batch = full_windows[i : i + batch_size]
            x = torch.stack([tensor for _, tensor in batch])
            logits = model(x).logits[:, :-1]
            loss = F.cross_entropy(
                logits.reshape(-1, vocab_size),
                x[:, 1:].reshape(-1),
                reduction="sum",
            )
            total_loss += float(loss.item())
            total_scored += int(x[:, 1:].numel())
            for start, tensor in batch:
                total_bytes += count_bytes(start, len(tensor))

        for start, tensor in tail_windows:
            if len(tensor) < 2:
                continue
            logits = model(tensor.unsqueeze(0)).logits[:, :-1]
            loss = F.cross_entropy(
                logits.reshape(-1, vocab_size),
                tensor[1:].unsqueeze(0).reshape(-1),
                reduction="sum",
            )
            total_loss += float(loss.item())
            total_scored += int(tensor[1:].numel())
            total_bytes += count_bytes(start, len(tensor))

    all_bytes = sum(len(b) for b in token_bytes)
    bpb = total_loss / math.log(2) / total_bytes if total_bytes else float("nan")
    return {
        "bpb": bpb,
        "loss_sum_nats": total_loss,
        "scored_bytes": total_bytes,
        "scored_tokens": total_scored,
        "eval_total_bytes": all_bytes,
        "coverage": round(total_bytes / all_bytes, 5) if all_bytes else None,
        "windows": len(full_windows) + len(tail_windows),
        "seconds": round(time.perf_counter() - started, 2),
    }


# ---------------------------------------------------------------------------
# 新 embedding 初始化（本实验最关键的一步）
# ---------------------------------------------------------------------------
def build_new_embedding_matrix(
    old_model: torch.nn.Module,
    old_tokenizer,
    new_vocab: dict[int, bytes],
    special_tokens: list[str],
) -> tuple[torch.Tensor, dict]:
    """按 handout/任务要求初始化 4096 行新 embedding，返回 (matrix, 统计信息)。"""
    old_w = old_model.transformer.wte.weight.detach()
    old_dim = old_w.shape[1]

    # 旧词表：bytes -> old id（GPT-2 的 token 字符串是 bytes 的可打印双射）
    u2b = unicode_to_bytes()
    old_bytes_to_id: dict[bytes, int] = {}
    for token_str, token_id in old_tokenizer.get_vocab().items():
        try:
            raw = bytes(u2b[ch] for ch in token_str)
        except KeyError:  # 理论上不会发生
            continue
        old_bytes_to_id.setdefault(raw, token_id)
    single_byte_to_id = {
        raw: tid for raw, tid in old_bytes_to_id.items() if len(raw) == 1
    }
    if len(single_byte_to_id) != 256:
        raise RuntimeError(f"旧词表里单字节 token 不完整：{len(single_byte_to_id)}/256")

    # special token 的旧 id：优先按「字节恰好相等」找，找不到就退回 eos
    special_old_id: dict[str, int] = {}
    for token in special_tokens:
        raw = token.encode("utf-8")
        special_old_id[token] = old_bytes_to_id.get(
            raw, old_tokenizer.eos_token_id
        )

    new_vocab_size = len(new_vocab)
    matrix = torch.empty(new_vocab_size, old_dim, dtype=old_w.dtype)
    stats = {
        "special": 0,
        "single_byte": 0,
        "multi_byte_utf8_mean": 0,
        "multi_byte_invalid_utf8_byte_mean": 0,
        "overlap_exact_old_token": 0,
    }

    # 判断哪些新 id 是 special
    special_new_bytes = {token.encode("utf-8") for token in special_tokens}

    for new_id in range(new_vocab_size):
        raw = new_vocab[new_id]
        if raw in special_new_bytes:
            token_str = raw.decode("utf-8")
            matrix[new_id] = old_w[special_old_id[token_str]]
            stats["special"] += 1
            continue

        if len(raw) == 1:
            # 单字节 token：直接找旧词表里同一个字节的 token
            matrix[new_id] = old_w[single_byte_to_id[raw]]
            stats["single_byte"] += 1
            continue

        if raw in old_bytes_to_id:
            stats["overlap_exact_old_token"] += 1

        try:
            text = raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            text = None

        if text is not None:
            old_ids = old_tokenizer.encode(text, add_special_tokens=False)
            if old_ids:
                matrix[new_id] = old_w[old_ids].mean(dim=0)
                stats["multi_byte_utf8_mean"] += 1
                continue
        # 非法 UTF-8（或旧 tokenizer 编出空序列）：对每个字节取单字节 embedding 的均值
        byte_rows = [single_byte_to_id[bytes([b])] for b in raw]
        matrix[new_id] = old_w[byte_rows].mean(dim=0)
        stats["multi_byte_invalid_utf8_byte_mean"] += 1

    stats["multi_byte_tokens"] = (
        stats["multi_byte_utf8_mean"] + stats["multi_byte_invalid_utf8_byte_mean"]
    )
    stats["non_special_tokens"] = new_vocab_size - len(special_tokens)
    stats["overlap_count_incl_single_byte"] = (
        stats["overlap_exact_old_token"] + stats["single_byte"]
    )
    stats["overlap_rate"] = (
        stats["overlap_count_incl_single_byte"] / stats["non_special_tokens"]
    )
    stats["multi_byte_overlap_rate"] = (
        stats["overlap_exact_old_token"] / stats["multi_byte_tokens"]
        if stats["multi_byte_tokens"]
        else 0.0
    )
    return matrix, stats


# ---------------------------------------------------------------------------
# 画图：只画「完整 eval」的三个点，绝不把 8KB 小切片的 BPB 混进来
# ---------------------------------------------------------------------------
def _plot_bpb(result: dict, png_path: Path) -> None:
    """从结果 JSON 画 BPB 图（不依赖模型，不跑 eval；可从已有 JSON 重画）。

    左图：换之前（baseline 水平线）+ 换后三个完整 eval 点（step 0 / phase1 后 / phase2 后）。
    右图（若做过初始化消融）：mean-init vs random-init 在相同 50 步后的完整 eval BPB。

    为什么不画逐步的 slice 曲线：8KB slice 只有约 10 篇文档，方差可达 0.08 BPB，
    会和完整 eval 的数字打架；逐步曲线只保留在 JSON 的 `training.curve` 里。
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bpb = result["bpb"]
    baseline = bpb["baseline_old_tokenizer"]["bpb"]
    steps_phase1 = result["config"]["steps_phase1"]
    steps_phase2 = result["config"]["steps_phase2"]
    points = [
        (0, bpb["after_swap_untrained"]["bpb"], "swap, untrained (step 0)", "#d62728", "o"),
        (steps_phase1, bpb["after_phase1"]["bpb"],
         f"after phase 1 (embedding only, {steps_phase1} steps)", "#9467bd", "s"),
        (steps_phase1 + steps_phase2, bpb["after_training"]["bpb"],
         f"after phase 2 (all params, {steps_phase1 + steps_phase2} steps)", "#ff7f0e", "D"),
    ]

    fig, ax = plt.subplots(figsize=(9.2, 5.0), dpi=140)
    # 注意：折线只是「视觉引导」，不是测过的中间点，所以用半透明虚线
    ax.plot(
        [p[0] for p in points], [p[1] for p in points],
        linestyle="--", color="#1f77b4", linewidth=1.1, alpha=0.5, zorder=2,
        label="linear guide only (no intermediate evaluations)",
    )
    for x, y, label, color, marker in points:
        ax.scatter([x], [y], marker=marker, s=130, color=color, zorder=3)
        ax.annotate(
            f"{y:.3f}", (x, y), textcoords="offset points", xytext=(8, 8), fontsize=9,
        )
    ax.axhline(
        baseline, linestyle="--", color="#2ca02c", linewidth=1.6,
        label=f"GPT-2 baseline with the OLD tokenizer ({baseline:.3f} BPB)",
    )

    # 如果存在 control（原 tokenizer、同预算继续训练），画一条点划线
    control_bpb = None
    control_path = RESULTS_DIR / "control_original_tokenizer.json"
    if control_path.exists():
        try:
            control = json.loads(control_path.read_text(encoding="utf-8"))
            control_bpb = (control.get("bpb") or {}).get("after_training", {}).get("bpb")
        except Exception:  # noqa: BLE001 - 画图不该因为可选文件失败
            control_bpb = None
    if control_bpb is not None:
        steps = steps_phase1 + steps_phase2
        ax.axhline(
            control_bpb, linestyle="-.", color="#8c564b", linewidth=1.4,
            label=f"control: OLD tokenizer, same {steps} steps ({control_bpb:.3f} BPB)",
        )
    ax.axvline(
        steps_phase1, linestyle=":", color="gray",
        label="phase 1 (embedding only) -> phase 2 (all params)",
    )
    ax.set_xlabel("continued-training step on the NEW tokenizer's token ids")
    ax.set_ylabel("bits per byte, full held-out split")
    ax.set_title(
        "GPT-2 after replacing its tokenizer with a trained 4096-token byte-level BPE\n"
        "only 3 full-eval checkpoints measured; dashed line is a linear guide",
        fontsize=10.5,
    )
    y_values = [p[1] for p in points] + [baseline] + ([control_bpb] if control_bpb is not None else [])
    ax.set_ylim(min(y_values) - 0.06, max(y_values) + 0.06)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")

    init_ablation = result.get("init_ablation")
    if init_ablation:
        ax2 = ax.inset_axes([0.08, 0.10, 0.36, 0.36])
        positions = [0, 1]
        values = [
            init_ablation["mean_init"]["bpb_after_ablation"],
            init_ablation["random_init"]["bpb_after_ablation"],
        ]
        ax2.bar(positions, values, color=["#1f77b4", "#d62728"], width=0.55)
        ax2.set_xticks(positions, ["mean", "random"])
        for position, value in zip(positions, values):
            ax2.text(position, value + 0.04, f"{value:.2f}", ha="center", fontsize=8)
        ax2.axhline(baseline, linestyle="--", color="#2ca02c", linewidth=1.0)
        ax2.set_ylim(0, max(values) * 1.25)
        ax2.set_title(
            f"init ablation: {init_ablation['steps']} steps\n(embedding only)",
            fontsize=8,
        )
        ax2.tick_params(labelsize=7)
        ax2.grid(alpha=0.25, axis="y")

    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="把自训 BPE tokenizer 换到 GPT-2 上并测 BPB")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--eval-fraction", type=float, default=0.05)
    parser.add_argument("--steps-phase1", type=int, default=200)
    parser.add_argument("--steps-phase2", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--lr-phase1", type=float, default=1e-3)
    parser.add_argument("--lr-phase2", type=float, default=1e-4)
    parser.add_argument("--eval-window", type=int, default=256)
    parser.add_argument("--eval-batch", type=int, default=4)
    parser.add_argument("--curve-bytes", type=int, default=8192, help="训练日志用多大的 eval 切片（仅写入 JSON，不画主图）")
    parser.add_argument("--lm-train-bytes", type=int, default=400_000, help="继续训练从训练文档里取多少字节的 token")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "transplant.json")
    parser.add_argument("--png", type=Path, default=RESULTS_DIR / "transplant_bpb.png")
    parser.add_argument("--quick", action="store_true", help="冒烟测试：极小步数/切片，几十秒跑完")
    parser.add_argument(
        "--ablation-steps",
        type=int,
        default=0,
        help="初始化消融的步数（>0 时额外跑 mean-init vs random-init 各 N 步，用于证明均值初始化有用）",
    )
    args = parser.parse_args()

    if args.quick:
        args.steps_phase1 = min(args.steps_phase1, 4)
        args.steps_phase2 = min(args.steps_phase2, 8)
        args.curve_bytes = 2048
        args.lm_train_bytes = 40_000
        args.vocab_size = min(args.vocab_size, 512)

    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(False)

    script_t0 = time.perf_counter()
    print("=" * 78)
    print("把自训 BPE tokenizer 换到 GPT-2 上（bits-per-byte 实验）")
    print("=" * 78)

    # ---- 1. 读语料、按文档切分 --------------------------------------
    if not args.corpus.exists():
        print(f"找不到语料：{args.corpus}", file=sys.stderr)
        return 2
    full_text = args.corpus.read_text(encoding="utf-8")
    docs = full_text.split(SPECIAL)
    n_docs = len(docs)
    n_train = int(n_docs * args.train_fraction)
    n_eval_end = int(n_docs * (args.train_fraction + args.eval_fraction))
    train_docs = docs[:n_train]
    eval_docs = docs[n_train:n_eval_end]
    if args.quick:  # 冒烟测试：只用极少量文档，保证几十秒跑完
        train_docs = train_docs[:200]
        eval_docs = eval_docs[:40]
    train_text = SPECIAL.join(train_docs)
    eval_text = SPECIAL.join(eval_docs)
    eval_bytes = len(eval_text.encode("utf-8"))
    print(
        f"文档数 {n_docs}；训练 {len(train_docs)} 篇（{len(train_text.encode()):,} 字节）；"
        f"eval {len(eval_docs)} 篇（{eval_bytes:,} 字节）"
    )

    # ---- 2. 训练新 tokenizer ----------------------------------------
    if args.quick:
        corpus_for_tok = train_text[:200_000]
    else:
        corpus_for_tok = train_text
    t0 = time.perf_counter()
    new_vocab, new_merges = train_bpe(
        corpus_for_tok, args.vocab_size, [SPECIAL], verbose=True
    )
    tokenizer_train_seconds = time.perf_counter() - t0
    new_tok = BPETokenizer(new_vocab, new_merges, [SPECIAL])
    print(
        f"新 tokenizer 训练完成：{len(new_merges)} merges，{tokenizer_train_seconds:.2f}s"
    )

    # ---- 3. 加载 GPT-2，构造基线 -------------------------------------
    print("加载 GPT-2 权重 ...")
    old_tok, model = _load_gpt2(args.weights)
    model.eval()
    old_vocab_size = model.config.vocab_size

    # 旧 id -> 原始字节
    u2b = unicode_to_bytes()
    old_byte_of_id: dict[int, bytes] = {}
    for token_str, token_id in old_tok.get_vocab().items():
        old_byte_of_id[token_id] = bytes(u2b[ch] for ch in token_str)
    old_eos_id = old_tok.eos_token_id
    special_bytes = SPECIAL.encode("utf-8")

    def old_encode(text: str) -> list[int]:
        return old_tok.encode(text, add_special_tokens=False)

    old_ids, old_token_bytes = _spans_with_bytes(
        old_encode, old_byte_of_id.__getitem__, eval_docs, old_eos_id, special_bytes
    )
    # 新 tokenizer 的 id 流
    new_ids, new_token_bytes = _spans_with_bytes(
        new_tok.encode,
        lambda i: new_vocab[i],
        eval_docs,
        new_tok._special_to_id[SPECIAL],
        special_bytes,
    )
    # 合理性检查：两条 id 流 decode 回来都必须是原文
    assert old_tok.decode(old_ids, clean_up_tokenization_spaces=False) == eval_text
    assert new_tok.decode(new_ids) == eval_text

    old_tokens_per_byte = len(old_ids) / eval_bytes
    new_tokens_per_byte = len(new_ids) / eval_bytes
    print(
        f"eval tokens/byte：旧 {old_tokens_per_byte:.4f}（{len(old_ids):,} tokens），"
        f"新 {new_tokens_per_byte:.4f}（{len(new_ids):,} tokens）"
    )

    print("评估基线（GPT-2 + 原 tokenizer）...")
    baseline = eval_bpb(
        model, old_ids, old_token_bytes, old_vocab_size,
        window=args.eval_window, batch_size=args.eval_batch,
    )
    baseline["tokens_per_byte"] = old_tokens_per_byte
    baseline["num_tokens"] = len(old_ids)
    print(f"  基线 BPB = {baseline['bpb']:.4f}（{baseline['seconds']}s）")

    # ---- 4. 换 tokenizer：resize + 初始化新 embedding -----------------
    print("初始化新 embedding 并 resize 模型 ...")
    t0 = time.perf_counter()
    new_matrix, init_stats = build_new_embedding_matrix(
        model, old_tok, new_vocab, [SPECIAL]
    )
    model.resize_token_embeddings(len(new_vocab), mean_resizing=False)
    with torch.no_grad():
        model.transformer.wte.weight.copy_(new_matrix)
    model.tie_weights()
    assert (
        model.lm_head.weight.data_ptr() == model.transformer.wte.weight.data_ptr()
    ), "tie_weights 之后 lm_head 与 wte 应该共享同一块存储"
    assert not torch.isnan(model.transformer.wte.weight).any(), "新 embedding 出现 NaN"

    new_eos_id = new_tok._special_to_id[SPECIAL]
    model.config.vocab_size = len(new_vocab)
    model.config.eos_token_id = new_eos_id
    model.config.bos_token_id = new_eos_id
    swap_seconds = time.perf_counter() - t0
    print(
        f"  resize 完成：vocab {old_vocab_size} -> {len(new_vocab)}；"
        f"初始化统计 {init_stats}；耗时 {swap_seconds:.2f}s"
    )

    print("评估「换完之后、未训练」（预期明显变差）...")
    after_swap = eval_bpb(
        model, new_ids, new_token_bytes, len(new_vocab),
        window=args.eval_window, batch_size=args.eval_batch,
    )
    after_swap["tokens_per_byte"] = new_tokens_per_byte
    after_swap["num_tokens"] = len(new_ids)
    print(f"  换后未训练 BPB = {after_swap['bpb']:.4f}（{after_swap['seconds']}s）")

    # ---- 5. 继续训练 -------------------------------------------------
    # 训练数据：新 tokenizer 的 id 流（从训练文档里取一段）
    lm_train_text = train_text[: args.lm_train_bytes]
    lm_train_ids = new_tok.encode(lm_train_text)
    id_tensor = torch.tensor(lm_train_ids, dtype=torch.long)
    print(
        f"继续训练数据：{len(lm_train_text):,} 字节 -> {len(lm_train_ids):,} tokens"
    )

    # 曲线用的固定 eval 切片（和完整 eval 用同一套 BPB 公式）
    curve_text = eval_text[: args.curve_bytes]
    curve_ids = new_tok.encode(curve_text)
    curve_token_bytes = [new_vocab[i] for i in curve_ids]
    curve_eval = lambda: eval_bpb(
        model, curve_ids, curve_token_bytes, len(new_vocab),
        window=args.eval_window, batch_size=args.eval_batch,
    )

    generator = torch.Generator().manual_seed(args.seed)

    def sample_batch() -> torch.Tensor:
        max_start = max(len(id_tensor) - args.seq_len - 1, 1)
        starts = torch.randint(0, max_start, (args.batch_size,), generator=generator)
        return torch.stack([id_tensor[s : s + args.seq_len] for s in starts])

    def grad_params():
        return [p for p in model.parameters() if p.requires_grad]

    curve: list[dict] = []
    phase_times: dict[str, float] = {}
    recent_losses: list[float] = []

    # 曲线的第 0 个点 = 换完 tokenizer、还没训练（必须在训练前测！）
    swap_slice = curve_eval()
    curve.append(
        {
            "step": 0,
            "phase": "after_swap",
            "train_loss_mean_last20": None,
            "eval_slice_bpb": round(swap_slice["bpb"], 5),
            "elapsed_seconds": round(time.perf_counter() - script_t0, 1),
        }
    )
    print(f"  [step    0 | after_swap] eval切片 BPB={swap_slice['bpb']:.4f}")

    def record(step: int, phase: str):
        point = {
            "step": step,
            "phase": phase,
            "train_loss_mean_last20": round(
                sum(recent_losses[-20:]) / max(len(recent_losses[-20:]), 1), 4
            )
            if recent_losses
            else None,
        }
        ev = curve_eval()
        point.update(
            {
                "eval_slice_bpb": round(ev["bpb"], 5),
                "eval_slice_seconds": ev["seconds"],
                "elapsed_seconds": round(time.perf_counter() - script_t0, 1),
            }
        )
        curve.append(point)
        print(
            f"  [step {step:4d} | {phase}] train loss(近20步均值)="
            f"{point['train_loss_mean_last20']}  eval切片 BPB={point['eval_slice_bpb']:.4f}"
        )
        model.train()  # eval_bpb 会切到 eval 模式，这里切回来

    print("=" * 78)
    print(f"相位 1：只训 embedding+lm_head（tied），{args.steps_phase1} 步，lr={args.lr_phase1}")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.transformer.wte.weight.requires_grad_(True)
    trainable = grad_params()
    assert len(trainable) == 1, f"tied 之后应该只有一个可训矩阵，实际 {len(trainable)}"
    optimizer = torch.optim.AdamW(trainable, lr=args.lr_phase1, weight_decay=0.0)

    phase1_t0 = time.perf_counter()
    model.train()
    for step in range(1, args.steps_phase1 + 1):
        x = sample_batch()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x).logits[:, :-1]
        loss = F.cross_entropy(logits.reshape(-1, len(new_vocab)), x[:, 1:].reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        recent_losses.append(float(loss.item()))
        if step % 100 == 0 or step == args.steps_phase1:
            record(step, "phase1")
    phase_times["phase1_seconds"] = round(time.perf_counter() - phase1_t0, 1)

    print("相位 1 结束后完整 eval ...")
    after_phase1 = eval_bpb(
        model, new_ids, new_token_bytes, len(new_vocab),
        window=args.eval_window, batch_size=args.eval_batch,
    )
    print(f"  相位1后 BPB = {after_phase1['bpb']:.4f}（{after_phase1['seconds']}s）")

    print("=" * 78)
    print(f"相位 2：全参训练，{args.steps_phase2} 步，lr={args.lr_phase2}")
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        grad_params(), lr=args.lr_phase2, weight_decay=0.01
    )
    phase2_t0 = time.perf_counter()
    model.train()
    for step in range(1, args.steps_phase2 + 1):
        x = sample_batch()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x).logits[:, :-1]
        loss = F.cross_entropy(logits.reshape(-1, len(new_vocab)), x[:, 1:].reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(grad_params(), 1.0)
        optimizer.step()
        recent_losses.append(float(loss.item()))
        if step % 100 == 0 or step == args.steps_phase2:
            record(args.steps_phase1 + step, "phase2")
    phase_times["phase2_seconds"] = round(time.perf_counter() - phase2_t0, 1)

    print("训练结束，完整 eval ...")
    after_training = eval_bpb(
        model, new_ids, new_token_bytes, len(new_vocab),
        window=args.eval_window, batch_size=args.eval_batch,
    )
    after_training["tokens_per_byte"] = new_tokens_per_byte
    after_training["num_tokens"] = len(new_ids)
    print(f"  训练后 BPB = {after_training['bpb']:.4f}（{after_training['seconds']}s）")

    # ---- 5b. 初始化消融（可选）----------------------------------------
    # 目的：证明「用旧子词 embedding 均值初始化」比「resize 默认随机初始化」好。
    # 两个 case 用完全相同的：新词表、训练数据、batch/seq、lr、步数、随机种子。
    init_ablation = None
    if args.ablation_steps > 0:
        print("=" * 78)
        print(
            f"初始化消融：mean-init vs random-init，各 {args.ablation_steps} 步"
            "（只训 embedding+lm_head，其余冻结）"
        )

        def run_ablation_case(use_mean_init: bool, label: str) -> dict:
            _, fresh_model = _load_gpt2(args.weights)
            fresh_model.eval()
            fresh_model.resize_token_embeddings(len(new_vocab), mean_resizing=False)
            if use_mean_init:
                with torch.no_grad():
                    fresh_model.transformer.wte.weight.copy_(new_matrix)
            fresh_model.tie_weights()
            for parameter in fresh_model.parameters():
                parameter.requires_grad_(False)
            fresh_model.transformer.wte.weight.requires_grad_(True)
            trainable_params = [
                p for p in fresh_model.parameters() if p.requires_grad
            ]
            opt = torch.optim.AdamW(
                trainable_params, lr=args.lr_phase1, weight_decay=0.0
            )
            gen = torch.Generator().manual_seed(args.seed)  # 两个 case 用同一串 batch
            losses: list[float] = []
            fresh_model.train()
            case_started = time.perf_counter()
            for _ in range(args.ablation_steps):
                starts = torch.randint(
                    0, max(len(id_tensor) - args.seq_len - 1, 1),
                    (args.batch_size,), generator=gen,
                )
                x = torch.stack(
                    [id_tensor[int(s) : int(s) + args.seq_len] for s in starts]
                )
                opt.zero_grad(set_to_none=True)
                logits = fresh_model(x).logits[:, :-1]
                loss = F.cross_entropy(
                    logits.reshape(-1, len(new_vocab)), x[:, 1:].reshape(-1)
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                opt.step()
                losses.append(float(loss.item()))
            ev = eval_bpb(
                fresh_model, new_ids, new_token_bytes, len(new_vocab),
                window=args.eval_window, batch_size=args.eval_batch,
            )
            case = {
                "init": label,
                "steps": args.ablation_steps,
                "bpb_after_ablation": round(ev["bpb"], 5),
                "initial_train_loss": round(losses[0], 4),
                "final_train_loss_mean_last10": round(
                    sum(losses[-10:]) / min(len(losses), 10), 4
                ),
                "seconds": round(time.perf_counter() - case_started, 1),
            }
            print(
                f"  {label:<12} {args.ablation_steps} 步后 BPB={case['bpb_after_ablation']:.4f}"
                f"（初始 loss {case['initial_train_loss']:.3f}，末 10 步均值 "
                f"{case['final_train_loss_mean_last10']:.4f}，{case['seconds']}s）"
            )
            return case

        mean_case = run_ablation_case(True, "mean_init")
        random_case = run_ablation_case(False, "random_init")
        init_ablation = {
            "steps": args.ablation_steps,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "lr": args.lr_phase1,
            "note": "random_init = transformers resize 的默认随机初始化（normal, std=initializer_range=0.02）",
            "mean_init": mean_case,
            "random_init": random_case,
            "bpb_gap_mean_minus_random": round(
                random_case["bpb_after_ablation"] - mean_case["bpb_after_ablation"], 5
            ),
        }
        print(
            f"  => 同 {args.ablation_steps} 步下，mean-init 比 random-init 好 "
            f"{init_ablation['bpb_gap_mean_minus_random']:.4f} BPB"
        )

    # ---- 6. 合理性自检 ----------------------------------------------
    nan_in_curve = any(
        point["eval_slice_bpb"] is None
        or (
            isinstance(point["eval_slice_bpb"], float)
            and math.isnan(point["eval_slice_bpb"])
        )
        for point in curve
    )
    sanity = {
        "baseline_bpb_in_0.5_3": 0.5 <= baseline["bpb"] <= 3.0,
        "swap_makes_it_worse": after_swap["bpb"] > baseline["bpb"],
        "training_improves": after_training["bpb"] < after_swap["bpb"],
        "no_nan_in_curve": not nan_in_curve,
        "decode_encode_roundtrip_eval": new_tok.decode(new_ids) == eval_text,
        "old_decode_roundtrip_eval": old_tok.decode(
            old_ids, clean_up_tokenization_spaces=False
        )
        == eval_text,
        "tied_weights_shared": model.lm_head.weight.data_ptr()
        == model.transformer.wte.weight.data_ptr(),
    }

    # ---- 7. 保存 JSON ------------------------------------------------
    result = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "experiment": "replace GPT-2 tokenizer with a freshly trained 4096-token byte-level BPE",
        "config": {
            "corpus": str(args.corpus),
            "weights": str(args.weights),
            "vocab_size": args.vocab_size,
            "train_fraction": args.train_fraction,
            "eval_fraction": args.eval_fraction,
            "steps_phase1": args.steps_phase1,
            "steps_phase2": args.steps_phase2,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "lr_phase1": args.lr_phase1,
            "lr_phase2": args.lr_phase2,
            "eval_window": args.eval_window,
            "eval_batch": args.eval_batch,
            "curve_bytes": args.curve_bytes,
            "lm_train_bytes": args.lm_train_bytes,
            "seed": args.seed,
            "threads": args.threads,
            "quick_mode": args.quick,
        },
        "corpus": {
            "total_docs": n_docs,
            "train_docs": len(train_docs),
            "eval_docs": len(eval_docs),
            "train_bytes": len(train_text.encode("utf-8")),
            "eval_bytes": eval_bytes,
            "tokenizer_train_bytes": len(corpus_for_tok.encode("utf-8")),
        },
        "new_tokenizer": {
            "vocab_size": len(new_vocab),
            "num_merges": len(new_merges),
            "special_tokens": [SPECIAL],
            "special_ids": {SPECIAL: new_eos_id},
            "train_seconds": round(tokenizer_train_seconds, 3),
            "longest_token": repr(max(
                (tok for tok in new_vocab.values() if tok != special_bytes), key=len
            )),
            "embedding_init": init_stats,
            "eval_tokens": len(new_ids),
            "eval_tokens_per_byte": round(new_tokens_per_byte, 5),
            "eval_bytes_per_token": round(1.0 / new_tokens_per_byte, 4),
        },
        "old_tokenizer": {
            "vocab_size": old_vocab_size,
            "eval_tokens": len(old_ids),
            "eval_tokens_per_byte": round(old_tokens_per_byte, 5),
            "eval_bytes_per_token": round(1.0 / old_tokens_per_byte, 4),
        },
        "bpb": {
            "baseline_old_tokenizer": {k: (round(v, 5) if isinstance(v, float) else v) for k, v in baseline.items()},
            "after_swap_untrained": {k: (round(v, 5) if isinstance(v, float) else v) for k, v in after_swap.items()},
            "after_phase1": {k: (round(v, 5) if isinstance(v, float) else v) for k, v in after_phase1.items()},
            "after_training": {k: (round(v, 5) if isinstance(v, float) else v) for k, v in after_training.items()},
        },
        "training": {
            "curve": curve,
            "phase_times_seconds": phase_times,
            "nan_in_curve": nan_in_curve,
        },
        "bpb_gap_closure": {
            "formula": "(after_swap - after_training) / (after_swap - baseline)",
            "after_swap_to_baseline_gap": round(
                after_swap["bpb"] - baseline["bpb"], 5
            ),
            "phase1_closes": round(
                (after_swap["bpb"] - after_phase1["bpb"])
                / (after_swap["bpb"] - baseline["bpb"]),
                4,
            ),
            "phase1_plus_phase2_closes": round(
                (after_swap["bpb"] - after_training["bpb"])
                / (after_swap["bpb"] - baseline["bpb"]),
                4,
            ),
            "remaining_gap_after_training": round(
                after_training["bpb"] - baseline["bpb"], 5
            ),
        },
        "init_ablation": init_ablation,
        "sanity": sanity,
        "timings_seconds": {
            "tokenizer_train": round(tokenizer_train_seconds, 2),
            "swap": round(swap_seconds, 2),
            "baseline_eval": baseline["seconds"],
            "after_swap_eval": after_swap["seconds"],
            "phase1_train": phase_times["phase1_seconds"],
            "after_phase1_eval": after_phase1["seconds"],
            "phase2_train": phase_times["phase2_seconds"],
            "after_training_eval": after_training["seconds"],
            "total": round(time.perf_counter() - script_t0, 1),
        },
        "peak_rss_mb": round(peak_rss_mb(), 1),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "threads": torch.get_num_threads(),
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 8. 画图 ------------------------------------------------------
    try:
        _plot_bpb(result, args.png)
        print(f"BPB 图已写入：{args.png}")
    except Exception as exc:  # noqa: BLE001 - 画图失败不该让实验白跑
        print(f"[warn] 画图失败（结果 JSON 已保存）：{exc}", file=sys.stderr)

    # ---- 9. 终端中文汇总 --------------------------------------------
    print()
    print("=" * 78)
    print("实验结果汇总（bits per byte，越低越好）")
    print("=" * 78)
    print(f"{'阶段':<28}{'BPB':>10}{'tokens/byte':>14}{'eval tokens':>14}")
    print("-" * 78)
    rows = [
        ("① 换之前：GPT-2 + 原 tokenizer", baseline, old_tokens_per_byte, len(old_ids)),
        ("② 换之后：未训练", after_swap, new_tokens_per_byte, len(new_ids)),
        ("   相位1后（embedding 只训）", after_phase1, new_tokens_per_byte, len(new_ids)),
        ("③ 继续训练后（phase1+2）", after_training, new_tokens_per_byte, len(new_ids)),
    ]
    for name, data, tpb, ntok in rows:
        print(f"{name:<28}{data['bpb']:>10.4f}{tpb:>14.4f}{ntok:>14,}")
    print("-" * 78)
    print(
        f"词表重叠：新词表 {init_stats['overlap_count_incl_single_byte']}"
        f"/{init_stats['non_special_tokens']} 个 token 恰好等于某个旧 token"
        f"（{init_stats['overlap_rate'] * 100:.1f}%；其中多字节 token 里 "
        f"{init_stats['overlap_exact_old_token']}/{init_stats['multi_byte_tokens']} = "
        f"{init_stats['multi_byte_overlap_rate'] * 100:.1f}%）"
    )
    print(
        "embedding 初始化：special "
        f"{init_stats['special']}，单字节 {init_stats['single_byte']}，"
        f"合法 UTF-8 均值 {init_stats['multi_byte_utf8_mean']}，"
        f"非法 UTF-8 按字节均值 {init_stats['multi_byte_invalid_utf8_byte_mean']}"
    )
    if after_swap["bpb"] > baseline["bpb"]:
        closure = (after_swap["bpb"] - after_training["bpb"]) / (
            after_swap["bpb"] - baseline["bpb"]
        )
        print(
            f"BPB gap closure：从「换后未训练」到「训练后」追回了与 baseline 差距的 "
            f"{closure * 100:.1f}%（仍差 {after_training['bpb'] - baseline['bpb']:.4f} BPB）"
        )
    print(f"相位1 {phase_times['phase1_seconds']}s（{args.steps_phase1} 步），"
          f"相位2 {phase_times['phase2_seconds']}s（{args.steps_phase2} 步），"
          f"总耗时 {result['timings_seconds']['total']}s，峰值内存 {result['peak_rss_mb']}MB")
    print("自检：" + "；".join(f"{k}={v}" for k, v in sanity.items()))
    print("=" * 78)

    ok = all(sanity.values())
    if not ok:
        print("[warn] 有自检项未通过，请看上面 sanity 输出", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
