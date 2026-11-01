"""exp3: 真实 GPT-2 上的 KV cache 实测——生成到底快几倍？

step08 在 2 层、64 维的玩具模型上说过：

    没有缓存（naive）：生成 N 个 token 是 O(N^3)，
        因为每一步都要把整段前缀重新喂进模型算一遍；
    有缓存（cached）：是 O(N^2)，
        因为每步只算「新 token 的 q/k/v」，历史 K、V 直接查表。

这一节把这句话放到**真实权重**（GPT-2 small，124M 参数）上，测出具体数字。

测四件事：
    1. 正确性  两种实现必须给出逐 token 相同的序列，每一步的 logits 也要几乎相同
    2. 端到端  prompt 16/64/128 个 token，各生成 32 个 token，比总墙钟时间
    3. 单步    context 16/64/256/512 时，「一次全量前向」vs「一次 cache decode」
    4. 缓存    公式算出来的大小 vs 实际分配，看看到底吃掉多少内存

机器：Apple M5 / CPU / fp32（不用 GPU、不用半精度），
所以下面的毫秒数只代表这台机器；换设备绝对值会变，趋势不变。

⚠️ 先说清楚一件事，免得看完数字得出错误结论：
    KV cache 省掉的是**重复的矩阵乘**，但它省不掉每一步的固定开销
    （Python 解释、算子调度、把 124M 个参数从内存里读一遍……）。
    真实模型 + 短上下文 + CPU 上，这部分固定开销占比很大，
    所以实测加速比会明显小于「计算量之比」——这正是本实验想让你看到的。

跑法（在仓库根目录）：

    MPLCONFIGDIR=$PWD/real_llm_lab/.mplcache \
        .venv/bin/python real_llm_lab/exp3_kv_cache.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# matplotlib 默认想往 ~/.matplotlib 写缓存，这里指定一个仓库内可写目录，
# 必须在 import matplotlib 之前设置。
HERE = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mplcache"))

# 让 `from gpt2_real import ...` 能找到同目录的模块
sys.path.insert(0, str(HERE))

import matplotlib

matplotlib.use("Agg")           # 无窗口后端，直接存 PNG
import matplotlib.pyplot as plt  # noqa: E402
import torch                     # noqa: E402

from gpt2_real import (GPT2Config, KVCache, generate_cached,  # noqa: E402
                       generate_naive, load_gpt2)

RUNS = HERE / "runs"
FIGS = HERE / "figs"
DEVICE = torch.device("cpu")

# 用来造不同长度 prompt 的原料：只要够长、能切成任何 token 数就行
BASE_TEXT = (
    "The old library on the corner of Ninth Street had been closed for years. "
    "Nobody remembered who locked the door, and nobody had asked. "
    "On the third floor, behind a shelf of newspapers from 1962, a small machine "
    "was still running. Every night it printed one page, and every morning the "
    "page was gone. The neighbours said the building was quiet, which was true, "
    "and that it was empty, which was not. "
)


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------
def timeit(fn, warmup: int = 1, repeat: int = 3) -> float:
    """返回 best-of-repeat 的耗时（秒）。

    warmup 次用来吃掉「第一次」的额外成本：
    第一次往往会触发内存分配、缓存冷启动、算子库首次编译等，
    不剔除的话测出来的不是稳态性能。
    取 best 而不是 mean：这台机器上可能有别的进程在抢 CPU，
    best 更接近「这段代码本身有多快」。
    """
    for _ in range(warmup):
        fn()
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def fmt_ms(sec: float) -> str:
    """秒 -> 毫秒字符串。"""
    return f"{sec * 1000:8.1f}"


def prompt_of_length(tok, length: int) -> torch.Tensor:
    """造一个正好 length 个 token 的 prompt，形状 (1, length)。

    反复拼 BASE_TEXT 再切前 length 个 token；
    BPE 下文本长度和 token 数不成比例，所以必须按 token 切。
    """
    ids: list[int] = []
    while len(ids) < length:
        ids.extend(tok.encode(BASE_TEXT))
    return torch.tensor([ids[:length]])


def text_of_length(tok, length: int) -> str:
    """取 prompt 的原文，方便打印出来看一眼。"""
    return tok.decode(prompt_of_length(tok, length)[0].tolist())


# ---------------------------------------------------------------------------
# 1. 正确性：缓存版 == 全量重算版
# ---------------------------------------------------------------------------
@torch.no_grad()
def step_logits_max_diff(model, idx: torch.Tensor, n_steps: int) -> tuple:
    """逐步比较两种实现给出的 logits。

    返回 (最大绝对差, 最大相对差, 最大 logits 绝对值)。

    第 k 步比较的是「用来预测第 L+k 个 token 的那份 logits」：
        naive  -> model(前 L+k 个 token)[:, -1, :]      形状 (1, V)
        cached -> prefill 给出预测第 L 个 token 的 logits，
                  之后每 decode 一次给出预测下一个 token 的 logits
    两份 logits 含义相同、形状相同，直接相减取绝对值。

    为什么允许有微小差异（而不是完全等于 0）：
    naive 一次算 T 个位置的注意力，cached 一次只算 1 个位置的，
    矩阵乘的分块方式不同 -> 浮点加法顺序不同 -> 最后几位有效数字会差一点。
    这不叫 bug，叫浮点误差。所以要同时看**绝对差**和**相对差**：
    GPT-2 的 logits 能到 300 左右，在 300 这个量级上 fp32 的最小间隔
    (ulp) 就有 3e-5，所以「绝对差 < 1e-4」这种要求在小模型上合理，
    在真实 GPT-2 上其实低于 fp32 的分辨率。真正该看的是相对差。
    """
    L = idx.size(1)
    cache = KVCache(model.config, idx.size(0), DEVICE,
                    dtype=model.wte.weight.dtype)
    # prefill：把 prompt 一次性喂进去，缓存好所有层的 K、V
    logits_cached = model(idx, kv_cache=cache, start_pos=0)[:, -1, :]
    pos = L
    cur = idx
    max_abs = 0.0
    max_rel = 0.0
    max_mag = 0.0

    print(f"    {'step':>4} | {'绝对差':>10} | {'相对差':>10} | "
          f"{'max|logit|':>11}")
    print("    " + "-" * 44)
    for step in range(n_steps):
        # naive：整段前缀重算一遍，只取最后一个位置的 logits
        logits_naive = model(cur)[:, -1, :]                  # (1, V)
        diff = (logits_naive - logits_cached).abs().max().item()
        mag = logits_naive.abs().max().item()                # logits 的量级
        max_abs = max(max_abs, diff)
        max_mag = max(max_mag, mag)
        max_rel = max(max_rel, diff / mag)
        print(f"    {step:>4} | {diff:>10.2e} | {diff / mag:>10.2e} | "
              f"{mag:>11.1f}")

        # 两条路的贪心选择必须一致，不一致说明缓存写错了位置
        nxt = logits_cached.argmax(dim=-1, keepdim=True)
        assert torch.equal(nxt, logits_naive.argmax(dim=-1, keepdim=True)), \
            f"第 {step} 步两种实现的 argmax 不一致"

        cur = torch.cat((cur, nxt), dim=1)
        if pos < model.config.block_size:
            # decode：只喂 1 个新 token，写进缓存的下一个位置
            logits_cached = model(nxt, kv_cache=cache, start_pos=pos)[:, -1, :]
            pos += 1

    return max_abs, max_rel, max_mag


@torch.no_grad()
def check_correctness(model, tok, n_prompt: int = 16,
                      n_new: int = 16) -> tuple:
    """跑一遍完整检查：token 序列一致 + 逐步 logits 差。"""
    print("\n--- 1. 正确性：naive 与 cached 必须生成同一段文字 ---")
    idx = prompt_of_length(tok, n_prompt)
    print(f"  prompt = {n_prompt} 个 token，生成 {n_new} 个 token，greedy 解码")
    print(f"  prompt 原文: {text_of_length(tok, n_prompt)!r}")

    t0 = time.perf_counter()
    out_naive = generate_naive(model, idx.clone(), max_new_tokens=n_new,
                               greedy=True)          # (1, L+n_new)
    t_naive = time.perf_counter() - t0
    t0 = time.perf_counter()
    out_cached = generate_cached(model, idx.clone(), max_new_tokens=n_new,
                                 greedy=True)
    t_cached = time.perf_counter() - t0

    # 逐个 token 比：不是「差不多」，是必须一模一样
    assert torch.equal(out_naive, out_cached), "两种实现生成的序列不一致！"
    print(f"  naive  输出: {tok.decode(out_naive[0, n_prompt:].tolist())!r}")
    print(f"  cached 输出: {tok.decode(out_cached[0, n_prompt:].tolist())!r}")
    print(f"  两种实现给出的 token ids 完全相同 "
          f"({n_new} 个: {out_cached[0, n_prompt:].tolist()})")

    print(f"\n  逐步比较 logits（naive 全量重算 vs cached 单步 decode）：")
    max_abs, max_rel, max_mag = step_logits_max_diff(model, idx.clone(),
                                                     n_steps=n_new)
    print(f"\n  {n_new} 步汇总：绝对差 {max_abs:.2e}，相对差 {max_rel:.2e}，"
          f"logits 量级 {max_mag:.0f}")
    assert max_rel < 1e-5, f"相对差太大（{max_rel:.2e}），缓存逻辑有问题"
    print(f"  [OK] 序列完全一致，logits 相对差 {max_rel:.1e} < 1e-5")
    print(f"       绝对差 {max_abs:.1e} 看似大于 1e-4，但 logits 量级是 "
          f"{max_mag:.0f}，")
    ulp = max_mag * 2 ** -23        # fp32 在 logits 量级上的最小间隔
    print(f"       fp32 在 {max_mag:.0f} 这个量级的 ulp 就有 {ulp:.1e}，"
          f"所以 {max_abs:.1e} 只相当于 {max_abs / ulp:.0f} 个 ulp —— 正常浮点误差。")
    print(f"       下面第 5 节会用 fp64 再验证一次「两路算法完全等价」。")

    # 顺便把这一次「有/无缓存」的时间差记下来（规模小，只是热身）
    print(f"  （附带：这次 L={n_prompt} 的 naive {t_naive * 1000:.0f} ms，"
          f"cached {t_cached * 1000:.0f} ms）")
    return max_abs, max_rel, max_mag


# ---------------------------------------------------------------------------
# 2. 端到端：不同 prompt 长度下，生成 32 个 token 的总时间
# ---------------------------------------------------------------------------
@torch.no_grad()
def bench_end_to_end(model, tok, contexts, n_new: int,
                     warmup: int = 1, repeat: int = 2) -> list[dict]:
    """对每个 prompt 长度，分别给 naive / cached 计时（best-of-repeat）。

    每次调用都传入 idx.clone()，避免上一次生成把 idx 接长了污染下一次。

    「计算量之比」是怎么算的：
        naive 第 k 步要处理 L+k 个 token（k=0..n_new-1），
        总共 sum_{t=L}^{L+n_new-1} t 次「token 前向」；
        cached 只有 prefill 的 L 个 + 每次 decode 的 1 个，共 L + n_new。
        计算量大致正比于 token 前向次数，这个比值就是理论加速比的上限
        （上限，因为前提是「每个 token 的处理时间都一样」，而实际不是）。
    """
    rows = []
    for L in contexts:
        idx = prompt_of_length(tok, L)
        t_naive = timeit(lambda: generate_naive(
            model, idx.clone(), n_new, greedy=True), warmup, repeat)
        t_cached = timeit(lambda: generate_cached(
            model, idx.clone(), n_new, greedy=True), warmup, repeat)

        work_naive = sum(range(L, L + n_new))    # token-前向次数
        work_cached = L + n_new
        rows.append({
            "context": L,
            "naive": t_naive,
            "cached": t_cached,
            "speedup": t_naive / t_cached,
            "ideal": work_naive / work_cached,   # 理论上限（计算量之比）
            "tok_per_s": (L + n_new) / t_cached,  # cached 的端到端吞吐
        })
    return rows


def print_e2e_table(rows: list[dict], n_new: int) -> None:
    ctx_text = "/".join(str(r["context"]) for r in rows)
    print(f"\n--- 2. 端到端：prompt {ctx_text}，各生成 {n_new} 个 token ---")
    print(f"  计时方式：每个配置先预热 1 次，再取 2 次中的最好成绩（best-of-2）")
    print()
    head = (f"  {'prompt':>7} | {'naive 总耗时':>12} | {'cached 总耗时':>13} | "
            f"{'加速比':>7} | {'理论比':>7} | {'cached 吞吐':>12}")
    print(head)
    print("  " + "-" * (len(head) - 2))
    for r in rows:
        print(f"  {r['context']:>7} | {r['naive'] * 1000:>10.1f} ms | "
              f"{r['cached'] * 1000:>11.1f} ms | "
              f"x{r['speedup']:>6.2f} | x{r['ideal']:>6.1f} | "
              f"{r['tok_per_s']:>8.1f} tok/s")
    print()
    print("  加速比 = naive / cached（同一个 prompt、同样生成这么多 token）")
    print("  理论比 = 两种实现各自的「token 前向次数」之比，也就是纯算法的上限")
    print("  -> 实测加速比只有理论上限的一小部分，原因见最后的讨论")


# ---------------------------------------------------------------------------
# 3. 单步延迟：全量前向 vs cache decode
# ---------------------------------------------------------------------------
@torch.no_grad()
def bench_single_step(model, contexts, warmup: int = 1,
                      repeat: int = 3) -> list[dict]:
    """测「一步」的耗时，分两种情况：

        全量前向  model(x)，x 长度 = context
                  生成时 naive 的第 k 步就是这一下（T 随上下文变长）
        cache decode  model(1 个 token, kv_cache, start_pos=context)
                  生成时 cached 的第 k 步就是这一下（T 永远是 1）

    注意 decode 要先 prefill 把缓存填到 context 那么长，
    否则测的就不是「上下文很长时每步要多久」了。
    """
    rows = []
    for L in contexts:
        # 内容无所谓，形状对就行（真实生成时这里是 prompt 的 token）
        x = torch.randint(0, model.config.vocab_size, (1, L))
        t_full = timeit(lambda: model(x), warmup, repeat)

        cache = KVCache(model.config, 1, DEVICE,
                        dtype=model.wte.weight.dtype)
        # prefill：把 L 个 token 的 K、V 写进缓存（这一步仍然是全量的）
        t_prefill = timeit(lambda: model(x, kv_cache=cache, start_pos=0),
                           warmup, repeat)
        nxt = model(x, kv_cache=cache, start_pos=0)[:, -1, :].argmax(
            dim=-1, keepdim=True)                    # (1,1)
        # decode：只喂 1 个 token，start_pos = L -> 注意力看 L+1 个位置
        t_dec = timeit(lambda: model(nxt, kv_cache=cache, start_pos=L),
                       warmup, repeat)

        rows.append({
            "context": L,
            "full": t_full,
            "per_token": t_full / L,
            "prefill": t_prefill,
            "decode": t_dec,
            "speedup": t_full / t_dec,
        })
    return rows


def print_single_step_table(rows: list[dict]) -> None:
    print("\n--- 3. 单步前向延迟：一次全量重算 vs 一次 cache decode ---")
    print("  计时方式：预热 1 次 + best-of-3")
    print()
    head = (f"  {'context':>7} | {'全量前向 T=ctx':>14} | {'折合每 token':>12} | "
            f"{'cached decode T=1':>17} | {'单步加速比':>10} | {'prefill 参考':>12}")
    print(head)
    print("  " + "-" * (len(head) - 2))
    for r in rows:
        print(f"  {r['context']:>7} | {r['full'] * 1000:>12.1f} ms | "
              f"{r['per_token'] * 1000:>10.2f} ms | "
              f"{r['decode'] * 1000:>15.2f} ms | "
              f"x{r['speedup']:>9.1f} | {r['prefill'] * 1000:>10.1f} ms")
    print()
    print("  怎么读这张表：")
    print("    「全量前向 T=ctx」这一列随 context 变大而变长——")
    print("      因为注意力是 O(T^2)：每个位置都要和前面所有位置算相似度，")
    print("      所以 naive 生成到后面，每一步都在变慢。")
    print("    「cached decode T=1」这一列几乎是平的——")
    print("      因为每步只算 1 个新 token 的 q/k/v，历史 K、V 直接从缓存里切出来，")
    print("      注意力对 T 个位置做加权是 O(T)，T 从 16 长到 512 只多一点点。")
    # 用实测数字说明「固定开销」有多重要：大上下文下的边际成本 vs 一次 T=1 调用
    marginal = ((rows[-1]["full"] - rows[-2]["full"])
                / (rows[-1]["context"] - rows[-2]["context"]))
    print(f"    「折合每 token」是关键：它在大上下文时趋近于 "
          f"~{marginal * 1000:.2f} ms/token，")
    print(f"      而一次 T=1 的调用却要 ~{rows[-1]['decode'] * 1000:.2f} ms"
          f"——两者差 {rows[-1]['decode'] / marginal:.0f} 倍，这就是固定开销。")


# ---------------------------------------------------------------------------
# 4. 缓存大小
# ---------------------------------------------------------------------------
def report_cache_size(model) -> int:
    """2 * n_layer * B * T_max * C * 4 字节，和实际分配对比。"""
    print("\n--- 4. KV cache 到底占多少内存 ---")
    config = model.config
    B, T_max, C = 1, config.block_size, config.n_embd
    formula_bytes = 2 * config.n_layer * B * T_max * C * 4      # 2 = K 和 V

    cache = KVCache(config, B, DEVICE, dtype=model.wte.weight.dtype)
    actual_bytes = cache.memory_bytes()
    n_params = sum(p.numel() for p in model.parameters())
    weight_bytes = n_params * 4

    print(f"  公式: 2 * n_layer * B * T_max * C * 4 字节")
    print(f"      = 2 * {config.n_layer} * {B} * {T_max} * {C} * 4")
    print(f"      = {formula_bytes:,} 字节 "
          f"= {formula_bytes / 1024**2:.1f} MiB "
          f"= {formula_bytes / 1e6:.1f} MB")
    print(f"  KVCache(...).memory_bytes() = {actual_bytes:,} 字节 "
          f"= {actual_bytes / 1024**2:.1f} MiB")
    assert actual_bytes == formula_bytes, "缓存实际大小和公式对不上"

    print(f"\n  对照：模型权重 {n_params:,} 个参数 * 4 字节 "
          f"= {weight_bytes / 1024**2:.0f} MiB")
    print(f"        缓存 / 权重 = {formula_bytes / weight_bytes * 100:.1f}%")
    print(f"  注意：这里是按 T_max={T_max}（block_size）**预分配**的，")
    print(f"        只用到 {T_max // 2} 个位置的话实际只需要 "
          f"{formula_bytes / 2 / 1024**2:.1f} MiB。")
    print(f"        长度换算：每多 1 个 token 的上下文，缓存多 "
          f"{2 * config.n_layer * B * C * 4 / 1024:.0f} KiB。")
    print("  [OK] 公式与实际分配一致")
    return actual_bytes


# ---------------------------------------------------------------------------
# 5. 画图
# ---------------------------------------------------------------------------
def make_figure(e2e: list[dict], step: list[dict], n_new: int) -> Path:
    RUNS.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.8))
    c_naive, c_cached = "#d95f02", "#1b9e77"

    # ---- (a) 端到端总耗时柱状图 ----
    xs = list(range(len(e2e)))
    w = 0.36
    naive_ms = [r["naive"] * 1000 for r in e2e]
    cached_ms = [r["cached"] * 1000 for r in e2e]
    ax1.bar([i - w / 2 for i in xs], naive_ms, w, label="naive (recompute prefix)",
            color=c_naive)
    ax1.bar([i + w / 2 for i in xs], cached_ms, w, label="cached (KV cache)",
            color=c_cached)
    top = max(naive_ms)
    for i, r in enumerate(e2e):
        ax1.text(i, max(naive_ms[i], cached_ms[i]) + top * 0.02,
                 f"x{r['speedup']:.1f}", ha="center", va="bottom",
                 fontsize=11, fontweight="bold")
        ax1.text(i - w / 2, naive_ms[i] + top * 0.005, f"{naive_ms[i]:.0f}",
                 ha="center", va="bottom", fontsize=7.5, color=c_naive)
        ax1.text(i + w / 2, cached_ms[i] + top * 0.005, f"{cached_ms[i]:.0f}",
                 ha="center", va="bottom", fontsize=7.5, color=c_cached)
    ax1.set_xticks(xs)
    ax1.set_xticklabels([f"{r['context']}" for r in e2e])
    ax1.set_xlabel("prompt length (tokens)")
    ax1.set_ylabel(f"total wall time for {n_new} new tokens (ms)")
    ax1.set_title(f"(a) End-to-end generation of {n_new} tokens\n"
                  "GPT-2 small, Apple M5 CPU, fp32, greedy", fontsize=11)
    ax1.set_ylim(0, top * 1.16)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(axis="y", alpha=0.3, linestyle="--")
    ax1.set_axisbelow(True)

    # ---- (b) 单步延迟 vs context ----
    ctx = [r["context"] for r in step]
    full_ms = [r["full"] * 1000 for r in step]
    dec_ms = [r["decode"] * 1000 for r in step]
    ax2.plot(ctx, full_ms, "o-", color=c_naive, linewidth=2,
             label="full forward (T = context), O(T^2) attention")
    ax2.plot(ctx, dec_ms, "s-", color=c_cached, linewidth=2,
             label="cached decode (T = 1), flat")
    for i, (x, y) in enumerate(zip(ctx, full_ms)):
        # 倒数第二个点的标注放到左上方，免得压到往上冲的那段线
        off = (-12, 10) if i == len(ctx) - 2 else (0, 8)
        ha = "right" if i == len(ctx) - 2 else "center"
        ax2.annotate(f"{y:.0f} ms", (x, y), textcoords="offset points",
                     xytext=off, ha=ha, fontsize=8.5, color=c_naive)
    ax2.annotate(f"{dec_ms[-1]:.1f} ms", (ctx[-1], dec_ms[-1]),
                 textcoords="offset points", xytext=(-8, -16), ha="right",
                 fontsize=8.5, color=c_cached)
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(ctx)
    ax2.set_xticklabels([str(c) for c in ctx])
    ax2.minorticks_off()
    ax2.set_xlabel("context length (tokens)")
    ax2.set_ylabel("latency of one step (ms)")
    ax2.set_title("(b) One decoding step: recompute vs cached\n"
                  "the gap widens as context grows", fontsize=11)
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(alpha=0.3, linestyle="--")
    ax2.set_axisbelow(True)
    ax2.set_ylim(0, max(full_ms) * 1.18)

    fig.tight_layout()
    out = FIGS / "exp3_kv_cache.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\n  [OK] 图已保存: {out}  (150 dpi, {out.stat().st_size / 1024:.0f} KiB)")
    return out


# ---------------------------------------------------------------------------
# 6. 附：用 fp64 验证「两路算法完全等价」
# ---------------------------------------------------------------------------
@torch.no_grad()
def check_fp64_equivalence(model, tok, n_prompt: int = 16,
                           n_steps: int = 8) -> float:
    """把同一份代码换成 fp64（双精度）再跑一遍，看两路是不是真的等价。

    为什么要这一步：
        fp32 只有 ~7 位有效数字，上面测到 7e-4 的绝对差，肉眼没法判断
        「这是浮点舍入」还是「缓存写错了位置」。换成 fp64 就一目了然：
          - 如果缓存写错了位置（比如 start_pos 差 1），fp64 下也会差得很离谱；
          - 如果 fp64 下两路差到 1e-12，说明两路的**算法**完全一样，
            之前那点差纯粹是 fp32 加法顺序不同带来的舍入。
    代价是 fp64 在 CPU 上没有硬件加速，慢好几倍，所以只跑 8 步、只作验证。
    """
    print("\n--- 5. 附：换成 fp64 再跑一遍，验证「两路算法完全等价」---")
    idx = prompt_of_length(tok, n_prompt)
    model.to(torch.float64)                 # 12 层、124M 参数全变成双精度
    cache = KVCache(model.config, idx.size(0), DEVICE, dtype=torch.float64)
    logits_cached = model(idx, kv_cache=cache, start_pos=0)[:, -1, :]
    pos = n_prompt
    cur = idx
    max_abs = 0.0
    for _ in range(n_steps):
        logits_naive = model(cur)[:, -1, :]
        max_abs = max(max_abs, (logits_naive - logits_cached).abs().max().item())
        nxt = logits_cached.argmax(dim=-1, keepdim=True)
        cur = torch.cat((cur, nxt), dim=1)
        logits_cached = model(nxt, kv_cache=cache, start_pos=pos)[:, -1, :]
        pos += 1
    model.to(torch.float32)                 # 换回 fp32（值不变：fp32 -> fp64 -> fp32 无损）

    print(f"  fp64 下 {n_steps} 步 logits 最大绝对差 = {max_abs:.2e}")
    assert max_abs < 1e-9, f"fp64 下还差 {max_abs:.2e}，缓存逻辑真的有问题"
    print(f"  [OK] fp64 下两路一致到 {max_abs:.0e}")
    print(f"       => 两路的算法完全等价；fp32 下那点差是浮点舍入，不是 bug。")
    assert next(model.parameters()).dtype == torch.float32
    return max_abs


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("exp3: 真实 GPT-2 上的 KV cache 实测")
    print("=" * 78)

    # ---- 0. 加载真实权重 ----
    t0 = time.perf_counter()
    model, tok = load_gpt2(device="cpu")
    load_s = time.perf_counter() - t0
    config = model.config
    n_params = sum(p.numel() for p in model.parameters())

    print(f"\n机器      : Apple M5 / CPU / fp32（torch {torch.__version__}，"
          f"线程 {torch.get_num_threads()}/{os.cpu_count()}）")
    print(f"模型      : GPT-2 small，{n_params:,} 参数"
          f"（官方 124,439,808），权重 {n_params * 4 / 1024**2:.0f} MiB")
    print(f"配置      : {config}")
    print(f"分词器    : GPT-2 BPE，词表 {len(tok):,}")
    print(f"加载耗时  : {load_s:.2f} s")
    assert n_params == 124_439_808, "参数量不对，权重加载有问题"
    assert (config.n_layer, config.n_head, config.n_embd, config.block_size,
            config.vocab_size) == (12, 12, 768, 1024, 50257)
    print("  [OK] 真实权重 + 真实分词器就绪")

    torch.manual_seed(1337)

    # ---- 1. 正确性 ----
    N_CHECK = 16
    logit_abs, logit_rel, logit_mag = check_correctness(
        model, tok, n_prompt=16, n_new=N_CHECK)

    # ---- 2. 端到端 ----
    NEW_TOKENS = 32
    e2e = bench_end_to_end(model, tok, contexts=[16, 64, 128],
                           n_new=NEW_TOKENS, warmup=1, repeat=2)
    print_e2e_table(e2e, NEW_TOKENS)

    # ---- 3. 单步延迟 ----
    step = bench_single_step(model, contexts=[16, 64, 256, 512],
                             warmup=1, repeat=3)
    print_single_step_table(step)

    # 用实测数字拆一下「一步到底花在哪」：
    #   大 batch 下每 token 的边际成本 vs 一次 T=1 调用的成本
    marginal = (step[-1]["full"] - step[-2]["full"]) / (
        step[-1]["context"] - step[-2]["context"])          # 秒/token
    fixed = step[-1]["decode"]
    bw = n_params * 4 / fixed / 1e9                          # GB/s（把权重读一遍）
    print(f"\n  把数字拆开看（context=512 这组）：")
    print(f"    在大上下文下多处理 1 个 token 的边际成本 ≈ {marginal * 1000:.2f} ms")
    print(f"    而一次独立的 T=1 调用要 {fixed * 1000:.2f} ms")
    print(f"    => 每次调用有约 {(fixed - marginal) * 1000:.1f} ms 的固定开销")
    print(f"    这部分主要是「把 {n_params * 4 / 1024**2:.0f} MiB 权重从内存读一遍」"
          f"（等效带宽 {bw:.0f} GB/s）+ Python 层算子调度。")
    print(f"    KV cache 能省掉重复的矩阵乘，但省不掉这 {(fixed - marginal) * 1000:.1f} ms。")

    # ---- 4. 缓存大小 ----
    cache_bytes = report_cache_size(model)

    # ---- 5. 附：fp64 等价性验证 ----
    fp64_diff = check_fp64_equivalence(model, tok, n_prompt=16, n_steps=8)

    # ---- 6. 画图 ----
    print("\n--- 6. 画图 ---")
    png = make_figure(e2e, step, NEW_TOKENS)

    # ---- 6. 结论 ----
    print("\n" + "=" * 78)
    print("结论（请连同上面的数字一起看）")
    print("=" * 78)
    r_first, r_last = e2e[0], e2e[-1]
    print(f"1) 正确性：两种实现逐 token 完全相同。fp32 下 {N_CHECK} 步 logits "
          f"相对差 {logit_rel:.1e}、绝对差 {logit_abs:.1e}"
          f"（logits 量级 {logit_mag:.0f}）；")
    print(f"   换成 fp64 后两路只差 {fp64_diff:.0e} -> 算法完全等价，"
          f"缓存写的位置没错。")
    print(f"2) 端到端：prompt {r_first['context']} -> {r_last['context']} token 时，"
          f"加速比从 x{r_first['speedup']:.2f} 涨到 x{r_last['speedup']:.2f}；")
    print(f"   上下文越长，省掉的重复计算越多（这正是 O(N^3) -> O(N^2) 的含义）。")
    print(f"3) 但实测加速比远小于「计算量之比」"
          f"（x{r_first['ideal']:.0f} ~ x{r_last['ideal']:.0f}）：")
    print(f"   naive 每步都要处理越来越长的序列，但这一步的 FLOPs 里注意力只占一部分，")
    print(f"   剩下的矩阵乘（FFN、lm_head 输出层）无论有没有缓存都得照算；")
    print(f"   而 cached 每步虽有 ~{fixed * 1000:.1f} ms 固定开销（读权重 + 调度），")
    print(f"   所以在 CPU fp32 上加速比只能到几倍，而不是几十倍。")
    print(f"4) 单步：全量前向随 context 线性上升"
          f"（{step[0]['full'] * 1000:.0f} -> {step[-1]['full'] * 1000:.0f} ms），")
    print(f"   cached decode 基本恒定"
          f"（{step[0]['decode'] * 1000:.1f} ~ {step[-1]['decode'] * 1000:.1f} ms），"
          f"这就是长上下文生成时缓存的价值。")
    print(f"5) 缓存大小 {cache_bytes / 1024**2:.1f} MiB"
          f"（权重 {n_params * 4 / 1024**2:.0f} MiB 的 "
          f"{cache_bytes / (n_params * 4) * 100:.0f}%），按 T_max 预分配，"
          f"随层数/上下文线性增长。")
    print()
    print("一句话：KV cache 让「每步的计算量」不再随上下文增长（步骤 3 的平线），")
    print("但每步的固定开销它省不掉，所以在小模型/短上下文/CPU 上，")
    print("端到端加速比是「几倍」这个量级，而不是复杂度公式给出的「几十倍」。")
    print()
    print("思考题：")
    print("  1. 如果把 batch 从 1 改成 8，cached 的固定开销会被摊薄，加速比会怎么变？")
    print("  2. 为什么 decode 时慢的其实是「读权重」（内存带宽），而不是「算数」？")
    print("     这对量化（fp16/int8）意味着什么？")
    print("  3. 换成 GPU，为什么加速比会比 CPU 上大得多？（固定开销占比不同）")
    print(f"\n图: {png}")
    print(f"日志: {RUNS / 'exp3.txt'}")


if __name__ == "__main__":
    main()
