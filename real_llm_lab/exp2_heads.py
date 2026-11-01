#!/usr/bin/env python3
"""exp2：真实 GPT-2 的注意力头 —— 特别是 induction head（复制头）。

前三个实验文件的分工：
    exp1_parity.py  证明「我们自己写的 step07 结构 + 官方权重」= 官方输出
    exp2_heads.py   （本文件）打开 12 层 x 12 头 = 144 个注意力矩阵，看它们各自在干什么
    exp3/exp4/...   其它实验

本文件只用到你已经学过的两样东西：
    1) step03/step05 的注意力：weights[i, j] = 位置 i 的 query 分给位置 j 的权重
    2) step04 的因果掩码：i 只能看 j <= i

要观察的现象叫 **induction head**（归纳头 / 复制头），一句话解释：

    如果序列里出现重复片段 S S，那么在第二个 S 的某个位置 i，
    有一个头会去看「上一次出现同一个 token 的位置的下一个 token」，
    也就是位置 i - period + 1（period = 重复片段的长度）。
    这样它就把「上次这里后面接了什么」搬过来，模型于是能接着往下复读。

举例（真实 GPT-2 里确实存在这种头）：
    "The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the"
    period = 10（第一句 10 个 token）。最后的位置 16 是第二个 "the"，
    它的「上一次出现」在位置 6，所以 induction 目标是位置 7 = " lazy"。
    如果模型预测 " lazy"，那就是复制行为的直接证据。

运行（从仓库根目录）：

    MPLCONFIGDIR=$PWD/real_llm_lab/.mplcache \
        .venv/bin/python real_llm_lab/exp2_heads.py

自测（权重还没下完时，用随机小模型把所有代码路径跑一遍）：

    .venv/bin/python real_llm_lab/exp2_heads.py --selftest

输出：
    real_llm_lab/runs/exp2.txt                      stdout 日志（用 tee 存）
    real_llm_lab/figs/exp2_induction_heads.png      两个子图
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
# matplotlib 的缓存目录默认在 ~/.matplotlib，这里改到仓库里，免得污染家目录
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mplcache"))

import matplotlib

matplotlib.use("Agg")            # 无窗口后端，只往文件里画
import matplotlib.pyplot as plt  # noqa: E402

import torch                     # noqa: E402

sys.path.insert(0, str(HERE))
from gpt2_real import GPT2, GPT2Config, load_gpt2  # noqa: E402

FIG_DIR = HERE / "figs"
RUNS_DIR = HERE / "runs"

# ---------------------------------------------------------------------------
# 两段固定的输入文本
# ---------------------------------------------------------------------------
# (1) 经典的「读句子、猜人名」补全：真实 GPT-2 应该会答 " Mary"
COMPLETION_TEXT = "When Mary and John went to the store, John gave a drink to"

# (2) 重复片段：第一句完整，第二句截断到 "over the"。
#     截断是故意的：这样模型必须靠 induction（回头看第一句）才能补出 " lazy"。
REPEAT_UNIT = "The quick brown fox jumps over the lazy dog."
REPEAT_TAIL = " The quick brown fox jumps over the"
REPEAT_TEXT = REPEAT_UNIT + REPEAT_TAIL


def line(char: str = "-", n: int = 72) -> None:
    print(char * n)


def fmt_ids(ids: list[int], limit: int = 10) -> str:
    """把很长的 id 列表缩成一行（自测时 period 有 45 个字符）。"""
    if len(ids) <= limit:
        return str(ids)
    return f"{ids[:5]} ... {ids[-2:]} (共 {len(ids)} 个)"


def fmt_pieces(tok, ids: list[int], limit: int = 10) -> str:
    """把 token 列表缩成一行。"""
    pieces = [tok.decode([i]) for i in ids]
    if len(pieces) <= limit:
        return str(pieces)
    return f"{pieces[:5]} ... {pieces[-2:]} (共 {len(pieces)} 个)"


def section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


# ---------------------------------------------------------------------------
# 小工具：只在 --selftest 里用，把字符映射成 id，行为像 step01 的 CharTokenizer
# ---------------------------------------------------------------------------
class ToyTokenizer:
    """随机小模型没有真正的词表，用字符级映射凑一个同接口的分词器。"""

    def __init__(self, *texts: str) -> None:
        chars = sorted(set("".join(texts)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        # 随机模型可能采样到词表里没见过的 id，用 <id> 兜底（只影响自测）
        return "".join(self.itos.get(i, f"<{i}>") for i in ids)

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        return [self.itos.get(i, f"<{i}>") for i in ids]


# ---------------------------------------------------------------------------
# [1] 注意力矩阵 + 因果性自检（对应 step04）
# ---------------------------------------------------------------------------
def causal_self_check(maps: torch.Tensor) -> dict:
    """maps: (n_layer, B, n_head, T, T)。返回几个自检数字供打印。"""
    L, B, H, T, _ = maps.shape
    assert B == 1, "本实验只喂 1 条序列"
    attn = maps[:, 0]                                  # (L, H, T, T)

    row_sums = attn.sum(dim=-1)                        # (L,H,T) 每行应该 = 1
    upper = torch.triu(attn, diagonal=1)               # j > i 的部分应该全是 0
    # 第 i 行有多少个「非零」位置：因果掩码下应该正好是 i+1 个
    nonzero_per_row = (attn > 0).sum(dim=-1)           # (L,H,T)

    return {
        "attn": attn,
        "row_sum_min": row_sums.min().item(),
        "row_sum_max": row_sums.max().item(),
        "upper_max": upper.abs().max().item(),
        "min_prob": attn.min().item(),
        "nonzero_row0": nonzero_per_row[..., 0].unique().tolist(),
        "nonzero_rowT": nonzero_per_row[..., T - 1].unique().tolist(),
        "T": T, "L": L, "H": H,
    }


# ---------------------------------------------------------------------------
# [2] 下一个 token 的 top-k
# ---------------------------------------------------------------------------
@torch.no_grad()
def next_token_topk(model: GPT2, tok, text: str, k: int = 5) -> list[tuple[str, float]]:
    ids = tok.encode(text)
    logits = model(torch.tensor([ids]))               # (1, T, 50257)
    probs = torch.softmax(logits[0, -1], dim=-1)      # 只看最后一个位置
    top_p, top_i = probs.topk(k)
    return [(tok.decode([i]), p) for p, i in zip(top_p.tolist(), top_i.tolist())]


# ---------------------------------------------------------------------------
# [3] induction head 检测
# ---------------------------------------------------------------------------
def find_repeat_period(ids: list[int],
                       min_aligned: int = 4) -> tuple[int, list[int]] | None:
    """找 period，并返回「真正重复的位置」。返回 (period, aligned_positions)。

    为什么不直接判断 ids[period:] == ids[:T-period]：
      1. 第二段是**截断**的，period 可能比后半段还长（字符级里 45 > 34）；
      2. BPE 在拼接边界上不一定对齐：第一段开头是 "The"，第二段开头是 " The"，
         BPE 会把它切成两个不同的 token（"The" vs "ĠThe"）。
    所以正确做法是**逐位置**比较 token，只统计真正重复的位置，其余位置在算
    induction score 时跳过（否则边界那个 token 会把平均值拉低）。
    """
    T = len(ids)
    best: tuple[int, list[int]] | None = None
    for period in range(1, T - min_aligned + 1):
        aligned = [i for i in range(period, T) if ids[i] == ids[i - period]]
        if len(aligned) < min_aligned:
            continue
        if best is None or len(aligned) > len(best[1]):
            best = (period, aligned)
    return best


def induction_scores(attn: torch.Tensor, period: int,
                     positions: list[int]) -> torch.Tensor:
    """对每个 (layer, head) 算 induction score，返回 (L, H)。

    score = 在给定的后半段位置上，注意力落在 i - period + 1 上的平均权重。

    为什么是 i - period + 1：induction head 看的是「上一次出现同一个 token 之后
    那一个 token」；token 在位置 i 的上一次出现在 i - period，它的下一个就是
    i - period + 1。注意 i - period + 1 <= i - 1，所以完全落在因果掩码允许的范围内。
    """
    pos = torch.as_tensor(positions)                  # (n,)
    targets = pos - period + 1                        # 对应的 induction 目标
    picked = attn[:, :, pos, targets]                 # (L, H, n)
    return picked.mean(dim=-1)                        # (L, H)


def offset_profile(attn_lh: torch.Tensor, period: int,
                   positions: list[int]) -> torch.Tensor:
    """单个头：对给定的后半段位置 i，统计「向左看 k 个位置」的平均注意力，k=1..period。

    返回值第 k-1 项 = mean_i attn[i, i-k]。
    induction head 应该在 k = period - 1 处出现尖峰（也就是偏移 period-1 的亮对角线）。
    """
    pos = torch.as_tensor(positions)
    out = [attn_lh[pos, pos - k].mean() for k in range(1, period + 1)]
    return torch.stack(out)


# ---------------------------------------------------------------------------
# [4] 画图（英文标签，避免中文字体缺失）
# ---------------------------------------------------------------------------
def plot_figure(scores: torch.Tensor, attn_best: torch.Tensor, best: tuple[int, int],
                ids: list[int], tok, period: int, positions: list[int],
                path: Path) -> None:
    L, H = scores.shape
    layer, head = best
    T = attn_best.shape[0]
    targets = [i - period + 1 for i in positions]

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.6),
                             gridspec_kw={"width_ratios": [1.0, 1.12]})

    # --- (a) 144 个头的 induction score 热图 ---------------------------------
    ax = axes[0]
    im = ax.imshow(scores.numpy(), cmap="viridis", aspect="equal")
    ax.set_xticks(range(H))
    ax.set_yticks(range(L))
    ax.set_xlabel("Head index")
    ax.set_ylabel("Layer index")
    ax.set_title(f"Induction score of all {L*H} heads\n"
                 f"(mean attention to position i-period+1, period={period})",
                 fontsize=10)

    vmax = scores.max().item()
    for l in range(L):
        for h in range(H):
            v = scores[l, h].item()
            if v >= 0.10:            # 只标明显的地方，否则 144 个数字太挤
                ax.text(h, l, f"{v:.2f}", ha="center", va="center",
                        fontsize=5.5, color="white" if v < vmax * 0.6 else "black")
    # 用红框圈出最强头
    ax.add_patch(plt.Rectangle((head - 0.5, layer - 0.5), 1, 1, fill=False,
                               edgecolor="red", linewidth=2.0))
    ax.text(head + 0.55, layer - 0.55, f"max L{layer}H{head} {vmax:.2f}",
            color="red", fontsize=8, va="bottom")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="induction score")

    # --- (b) 最强头的注意力矩阵 ---------------------------------------------
    ax = axes[1]
    im = ax.imshow(attn_best.numpy(), cmap="magma", aspect="equal")
    labels = [tok.decode([i]).replace(" ", "\u00b7") for i in ids]
    ax.set_xticks(range(T))
    ax.set_yticks(range(T))
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Key position (attended to)")
    ax.set_ylabel("Query position")
    ax.set_title(f"Attention of strongest head L{layer}H{head}\n"
                 f"circles = induction target (i-period+1)", fontsize=10)

    # 把 induction 目标位置用圆圈标出来，方便看「偏移 period-1 的斜线」
    ax.scatter(targets, positions, s=26, facecolors="none",
               edgecolors="cyan", linewidths=1.1,
               label=f"induction target (offset {period - 1})")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.85)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="attention weight")

    fig.suptitle("GPT-2 small: induction heads on a repeated prompt", fontsize=12)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    try:
        shown = path.relative_to(HERE.parent)
    except ValueError:          # --selftest 时写在临时目录，不在仓库里
        shown = path
    print(f"[OK] 图已保存: {shown}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def run(model: GPT2, tok, fig_path: Path, real: bool) -> None:
    cfg = model.config
    n_heads_total = cfg.n_layer * cfg.n_head
    n_params = sum(p.numel() for p in model.parameters())
    print(f"权重: {'真实 GPT-2 small（官方 safetensors）' if real else '随机初始化小模型（自测）'}")
    print(f"配置: n_layer={cfg.n_layer}, n_head={cfg.n_head}, n_embd={cfg.n_embd}, "
          f"block_size={cfg.block_size}, vocab_size={cfg.vocab_size}")
    print(f"参数量: {n_params:,}")

    # =====================================================================
    section(f"[1/4] {n_heads_total} 个注意力矩阵 + 因果性自检（对应 step04）")
    # =====================================================================
    ids = tok.encode(REPEAT_TEXT)
    idx = torch.tensor([ids])                          # (1, T)
    T = idx.shape[1]
    t0 = time.time()
    maps = model.attention_maps(idx)                   # (n_layer, B, n_head, T, T)
    dt = time.time() - t0
    print(f"输入文本: {REPEAT_TEXT!r}")
    print(f"BPE 分词: {T} 个 token -> {tok.convert_ids_to_tokens(ids)}")
    print()
    print(f"idx.shape      = {str(tuple(idx.shape)):<14}  <- (B, T)")
    print(f"maps.shape     = {str(tuple(maps.shape)):<14}  <- (n_layer, B, n_head, T, T)")
    print(f"共 {n_heads_total} 个头，每个头一张 {T}x{T} 的注意力矩阵"
          f"（耗时 {dt*1000:.0f} ms）")

    chk = causal_self_check(maps)
    attn = chk["attn"]                                 # (L, H, T, T)
    print()
    print(f"每行之和: min={chk['row_sum_min']:.8f}, max={chk['row_sum_max']:.8f}   （应 = 1）")
    print(f"上三角(j>i)最大绝对值: {chk['upper_max']:.1e}   （应 = 0）")
    print(f"最小注意力权重: {chk['min_prob']:.1e}   （应 >= 0，softmax 不会出负数）")
    print(f"第 0 行非零个数: {chk['nonzero_row0']}（全部 {n_heads_total} 个头都只看得见自己）")
    print(f"第 {T-1} 行非零个数: {chk['nonzero_rowT']}（最后一个位置看得见全部）")
    assert abs(chk["row_sum_min"] - 1) < 1e-5 and abs(chk["row_sum_max"] - 1) < 1e-5
    assert chk["upper_max"] == 0.0, "因果掩码没生效！"
    print("[OK] 注意力是概率分布，并且严格因果（位置 i 看不到 i 之后的 token）")

    # =====================================================================
    section("[2/4] 经典补全：真实 GPT-2 会填谁的名字？")
    # =====================================================================
    print(f"prompt = {COMPLETION_TEXT!r}")
    top5 = next_token_topk(model, tok, COMPLETION_TEXT, k=5)
    print()
    print("下一个 token 的 top-5（概率来自 softmax(logits[-1])）：")
    for rank, (piece, p) in enumerate(top5, 1):
        print(f"  {rank}. {piece!r:<12} p = {p:.4f}")
    best_piece, best_p = top5[0]
    print()
    print(f"-> 模型最想说的是 {best_piece!r}（p={best_p:.4f}）")
    if best_piece.strip() == "Mary" or best_piece == " Mary":
        print("[OK] 和预期一致：它知道要给 Mary 喝的东西（John 给 Mary 饮料）")
    else:
        print("[!] 和预期不一致，如实记录")

    # =====================================================================
    section("[3/4] induction head 检测：重复片段上的注意力")
    # =====================================================================
    found = find_repeat_period(ids)
    assert found is not None, "输入里找不到重复片段，induction 检测没有意义"
    period, aligned = found
    candidates = list(range(period, T))
    skipped = [i for i in candidates if i not in aligned]
    print(f"重复片段长度 period = {period} 个 token")
    print(f"前半段 tokens = {fmt_pieces(tok, ids[:period])}")
    print(f"后半段 tokens = {fmt_pieces(tok, ids[period:])}")
    print(f"候选位置（后半段）{len(candidates)} 个: {candidates}")
    print(f"其中「和 period 个位置之前是同一个 token」的位置 {len(aligned)} 个: {aligned}")
    if skipped:
        print(f"[!] 跳过不对齐的位置 {skipped}：BPE 把拼接处切成了不同的 token"
              f"（位置 0 是 {tok.decode([ids[0]])!r}，位置 {period} 是 "
              f"{tok.decode([ids[period]])!r}），这一个位置不能用来测 induction")
    assert len(aligned) >= 4, "对齐的位置太少，统计不可靠"
    assert T - 1 in aligned, "最后一个位置没对齐，换个输入"

    print()
    print("induction 目标（每个位置 i 应该看 i-period+1）：")
    pairs = [(i, i - period + 1) for i in aligned]
    rows = pairs if len(pairs) <= 8 else pairs[:4] + [None] + pairs[-2:]
    for pair in rows:
        if pair is None:
            print(f"  ... 中间省略 {len(pairs) - 6} 个位置 ...")
            continue
        i, t = pair
        print(f"  i={i:>2} token={tok.decode([ids[i]])!r:<9} 上一次出现在 i-period={i-period:>2}"
              f" -> 它的下一个 token 是位置 {t:>2} = {tok.decode([ids[t]])!r}")

    scores = induction_scores(attn, period, aligned)   # (L, H)
    flat = scores.flatten()
    top_val, top_flat = flat.topk(min(5, flat.numel()))
    print()
    print(f"{n_heads_total} 个头的 induction score 排名前 5：")
    print(f"  {'rank':<5}{'layer':<7}{'head':<6}{'score':<8}")
    ranking = []
    for rank, (v, f) in enumerate(zip(top_val.tolist(), top_flat.tolist()), 1):
        l, h = divmod(f, scores.shape[1])
        ranking.append((l, h, v))
        print(f"  {rank:<5}{l:<7}{h:<6}{v:<8.4f}")
    print()
    print(f"全部 {n_heads_total} 个头: 均值 {scores.mean():.4f}, 中位数 {scores.median():.4f}, "
          f"最大 {scores.max():.4f}, 最小 {scores.min():.4f}")
    n_high = (scores >= 0.20).sum().item()
    print(f"score >= 0.20 的头有 {n_high} 个（0.20 是随便定的「看起来明显」的线）")

    # 前 5 个头逐个检查：注意力峰值是不是在 offset period-1（也就是那条偏移的亮对角线）
    print()
    print(f"前 5 个头的「偏移对角线」检查（峰值偏移 k 应该 = period-1 = {period - 1}）：")
    print(f"  {'layer':<7}{'head':<6}{'score':<8}{'peak k':<8}{'peak weight':<13}{'k == period-1 ?'}")
    for l, h, v in ranking:
        prof_i = offset_profile(attn[l, h], period, aligned)
        pk = prof_i.argmax().item() + 1
        print(f"  {l:<7}{h:<6}{v:<8.4f}{pk:<8}{prof_i[pk-1].item():<13.4f}"
              f"{'yes' if pk == period - 1 else 'NO'}")

    # 最强头到底是不是在看 offset period-1？
    best_layer, best_head, best_score = ranking[0]
    best_attn = attn[best_layer, best_head]            # (T, T)
    prof = offset_profile(best_attn, period, aligned)  # (period,)
    peak_k = prof.argmax().item() + 1
    print()
    print(f"最强头 L{best_layer}H{best_head}（score={best_score:.4f}）的行为检查：")
    print(f"  向左偏移 k 个位置的平均注意力（只在上面 {len(aligned)} 个位置上平均）：")
    for k in range(1, period + 1):
        mark = ""
        if k == period - 1:
            mark = "  <- induction 目标（偏移 period-1）"
        if k == period:
            mark = "  <- 同一个 token 的上一次出现"
        print(f"    k={k:>2}  {prof[k-1].item():.4f}{mark}")
    print(f"  -> 峰值在 k={peak_k}，"
          f"{'正好是 period-1，符合 induction head' if peak_k == period - 1 else '不是 period-1'}")

    # 大图里最直观的一行：最后一个位置看了谁
    last = T - 1
    row = best_attn[last]
    top_p, top_i = row.topk(3)
    print()
    print(f"  最强头的最后一行（query=位置 {last} = {tok.decode([ids[last]])!r}）看了谁：")
    for p, i in zip(top_p.tolist(), top_i.tolist()):
        tag = "  <- induction 目标 i-period+1" if i == last - period + 1 else ""
        print(f"    位置 {i:>2} {tok.decode([ids[i]])!r:<10} 权重 {p:.4f}{tag}")
    if 0 in top_i.tolist():
        print("  （位置 0 是序列第一个 token；这个头固定分了一部分注意力给它，"
              "是 GPT-2 里常见的现象，不影响上面 induction 的结论）")

    # 机制 -> 行为：这个重复序列的最后一个 token 后面，模型会接什么？
    # （要求 induction 才能答对：第一句里 "over the" 后面是 " lazy"）
    preds = next_token_topk(model, tok, REPEAT_TEXT, k=3)
    print()
    print("接上第 [2] 步的验证：这段重复文本的下一个 token 是什么？")
    for rank, (piece, p) in enumerate(preds, 1):
        print(f"  {rank}. {piece!r:<12} p = {p:.4f}")
    # 最后一个位置的上一次出现在 T-1-period，它的下一个 token 就是 induction 目标
    expected = tok.decode([ids[T - period]])
    print(f"第一句里同样位置后面的 token 是 {expected!r}（induction 猜的就是它）")

    # =====================================================================
    section("[4/4] 画图")
    # =====================================================================
    plot_figure(scores, best_attn, (best_layer, best_head), ids, tok, period,
                aligned, fig_path)

    # =====================================================================
    section("结论（只写数据支持的部分）")
    # =====================================================================
    ratio = best_score / max(scores.mean().item(), 1e-9)
    print(f"1) 因果性：{n_heads_total} 个头全部严格下三角，每行和 = 1 —— 和 step04 学的一致。")
    print(f"2) 补全：{COMPLETION_TEXT!r} -> {best_piece!r} (p={best_p:.4f})，"
          f"top-5 = {[(s, round(p, 4)) for s, p in top5]}")
    print(f"3) 最强头 L{best_layer}H{best_head}: induction score = {best_score:.4f}，"
          f"是全体均值 {scores.mean():.4f} 的 {ratio:.1f} 倍；"
          f"注意力峰值偏移 k={peak_k}（期望 period-1={period-1}）。")
    if peak_k == period - 1 and ratio >= 3:
        print(f"   -> 数据支持：GPT-2 small 里存在明确的 induction head，"
              f"它在重复片段上把注意力放到「上一次出现的下一个 token」上。")
    elif peak_k == period - 1:
        print(f"   -> 数据只能说：最强头的峰值偏移对了，但相对均值不够突出"
              f"（{ratio:.1f}x），不能说「存在很干净的 induction head」。")
    else:
        print(f"   -> 数据不支持：最强头的峰值偏移不是 period-1，"
              f"按这个定义算不上 induction head（如实报告）。")
    print(f"4) 机制到行为：重复文本的最后一个 token 后面模型预测 "
          f"{[(s, round(p, 4)) for s, p in preds[:3]]}，"
          f"第一句对应的 token 是 {expected!r}。")
    if n_high >= 1:
        print(f"5) 有 {n_high} 个头的 score >= 0.20：复制行为不一定是单一头的专利，"
              f"可能有多头冗余（这也是 GPT-2 能工作的原因之一）。")
    else:
        print(f"5) 没有任何头的 score 达到 0.20（最高 {scores.max():.4f}）："
              f"在这个序列上没看到集中的复制行为。")
    print(f"   注：induction score 只在对齐的 {len(aligned)} 个位置上平均"
          f"（跳过了 {skipped}，它们是 BPE 拼接边界上没对齐的 token）。")


def main() -> None:
    parser = argparse.ArgumentParser(description="exp2: GPT-2 注意力头 / induction head")
    parser.add_argument("--selftest", action="store_true",
                        help="用随机初始化的 2 层小模型跑通代码路径（不读真实权重）")
    args = parser.parse_args()

    print("=" * 72)
    print("exp2：真实 GPT-2 的注意力头 —— 找 induction head（复制头）")
    print("=" * 72)
    print(f"PyTorch {torch.__version__}, device=cpu, dtype=float32")

    t_start = time.time()
    if args.selftest:
        # 随机小模型：只验证代码路径和形状，结论没有意义
        cfg = GPT2Config(n_layer=2, n_head=2, n_embd=64, block_size=128, vocab_size=100)
        model = GPT2(cfg).eval()
        tok = ToyTokenizer(COMPLETION_TEXT, REPEAT_TEXT)
        # 自测图写到临时目录，不污染 figs/
        fig_path = Path(tempfile.mkdtemp(prefix="exp2_selftest_")) / "exp2_selftest.png"
        print("[selftest] 随机小模型，形状/流程检查用")
    else:
        t_load = time.time()
        model, tok = load_gpt2()
        print(f"[加载] 真实权重 + BPE 分词器，耗时 {time.time() - t_load:.1f} s")
        fig_path = FIG_DIR / "exp2_induction_heads.png"

    with torch.no_grad():
        run(model, tok, fig_path, real=not args.selftest)

    print()
    line("=")
    print(f"[OK] exp2 完成，总耗时 {time.time() - t_start:.1f} s")
    line("=")


if __name__ == "__main__":
    main()
