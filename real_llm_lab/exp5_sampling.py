"""exp5: 采样策略 —— greedy / temperature / top_k（真实 GPT-2 small 权重）。

学过的东西怎么用上：

    step08 的生成函数里有两个参数：temperature 和 top_k。
    那里模型是 2 层 64 维的玩具模型，生成出来的东西本来就不成句子，
    所以「温度高低有什么区别」看不出来。
    换成真实的 GPT-2 small 之后，同一个 prompt、同一个随机种子，
    不同温度生成的句子质量差别一眼就能看出来。

    温度做的事，其实就是你学过的 softmax 加了一个除法：
        p_i = softmax(logits / T)_i
    T < 1：把 logits 的差距放大，分布变尖（更保守、更像 greedy）
    T > 1：把差距缩小，分布变平（更随机、更容易说胡话）
    T -> 0 就是 greedy（argmax），只留概率最大的那个。

这个脚本跑两件事：

    1. 固定 prompt，用 greedy / T=0.5 / T=1.0 / T=1.5（top_k=40）各生成 40 个 token，
       再补一个 T=1.0 且不设 top_k，看看「截断」的作用
    2. 打印同一个位置 top-5 token 在三种温度下的 softmax 概率，
       把「温度如何改变分布」直接算出来（顺便看熵和 top-40 概率质量）

运行方式（在仓库根目录，约 20 秒）：

    MPLCONFIGDIR=$PWD/real_llm_lab/.mplcache \
        .venv/bin/python real_llm_lab/exp5_sampling.py
"""

from __future__ import annotations

import math
import os
import sys
import textwrap
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
# 和 exp4 一样：给 matplotlib 指一个可写的缓存目录（本脚本其实不画图，
# 但 gpt2_real 以外的地方可能间接 import，保持环境一致）
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mplcache"))
(HERE / ".mplcache").mkdir(parents=True, exist_ok=True)

import torch
import torch.nn.functional as F

sys.path.insert(0, str(HERE))
from gpt2_real import generate_cached, load_gpt2   # noqa: E402

PROMPT = "The future of artificial intelligence is"
# 第二个 context：模型在这里非常确定下一个词（" 6"），用来和上面「平」的分布对比
PEAKED_CONTEXT = "1, 2, 3, 4, 5,"
MAX_NEW_TOKENS = 40
TOP_K = 40
SEED = 1234            # 每个策略都用同一个种子，保证比较公平


def banner(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def wrap(text: str, width: int = 96) -> str:
    """把生成的长文本折行，方便在终端里看。"""
    return "\n".join(textwrap.wrap(text, width=width)) or "(空)"


@torch.no_grad()
def show_temperature_probs(model, tok, context: str, k: int = 5):
    """打印 context 最后一个位置 top-k token 在 T=0.5/1.0/1.5 下的概率。

    这就是 softmax(logits / T) 的直接计算，不涉及采样。
    """
    ids = torch.tensor([tok.encode(context)], dtype=torch.long)   # (1, T)
    logits = model(ids)[0, -1]                                    # (V,) 最后一个位置
    temperatures = [0.5, 1.0, 1.5]
    probs = {t: F.softmax(logits / t, dim=-1) for t in temperatures}

    print(f"\ncontext = {context!r}   (最后一个位置的分布)")
    print(f"  {'token':<14} {'logit':>8} "
          f"{'p(T=0.5)':>10} {'p(T=1.0)':>10} {'p(T=1.5)':>10}")
    print("  （logit 的绝对值不重要——softmax 对整体平移不变，只看相对差）")
    print("  " + "-" * 56)

    top = torch.topk(logits, k)                                   # 按 T=1.0 的 logits 排序
    for i in range(k):
        tid = top.indices[i].item()
        piece = repr(tok.decode([tid]))                            # repr 能看出前导空格
        print(f"  {piece:<14} {logits[tid].item():>8.2f} "
              f"{probs[0.5][tid].item():>10.4f} "
              f"{probs[1.0][tid].item():>10.4f} "
              f"{probs[1.5][tid].item():>10.4f}")

    # 熵：分布的「不确定程度」，单位 bit。均匀分布时 = log2(50257) ≈ 15.6
    entropy = {t: -(p * p.clamp_min(1e-12).log()).sum().item() / math.log(2)
               for t, p in probs.items()}
    top_mass = {t: torch.topk(p, TOP_K).values.sum().item() for t, p in probs.items()}
    max_prob = {t: p.max().item() for t, p in probs.items()}

    print("  " + "-" * 56)
    print(f"  {'max prob':<14} {'':>8} "
          f"{max_prob[0.5]:>10.4f} {max_prob[1.0]:>10.4f} {max_prob[1.5]:>10.4f}")
    print(f"  {'熵(bit)':<12} {'':>8} "
          f"{entropy[0.5]:>10.2f} {entropy[1.0]:>10.2f} {entropy[1.5]:>10.2f}")
    print(f"  {'top-40 质量':<12} {'':>8} "
          f"{top_mass[0.5]:>9.1%} {top_mass[1.0]:>10.1%} {top_mass[1.5]:>10.1%}")

    # 温度升高 -> 分布变平 -> 熵变大、最大概率变小
    assert entropy[0.5] < entropy[1.0] < entropy[1.5], "温度越高熵应该越大"
    assert max_prob[0.5] > max_prob[1.0] > max_prob[1.5], "温度越高最大概率应该越小"

    # 把关键数字交回给 main，用来写「看图说话」的结论（不硬编码）
    top_piece = tok.decode([top.indices[0].item()])
    return top_piece, max_prob, entropy


def run_generation(model, tok, label: str, prompt_ids: torch.Tensor,
                   **gen_kwargs) -> tuple[str, list[int], float]:
    """跑一次生成，返回 (解码后的新文本, 新 token 列表, 耗时秒)。"""
    torch.manual_seed(SEED)          # 每个策略都从同一个随机数状态开始
    t0 = time.time()
    out = generate_cached(model, prompt_ids.clone(), MAX_NEW_TOKENS, **gen_kwargs)
    elapsed = time.time() - t0

    new_ids = out[0, prompt_ids.size(1):].tolist()
    assert len(new_ids) == MAX_NEW_TOKENS, "生成的 token 数不对"
    return tok.decode(new_ids), new_ids, elapsed


def main() -> None:
    t_start = time.time()

    # ------------------------------------------------------------------
    # 0. 加载真实权重
    # ------------------------------------------------------------------
    banner("0. 加载真实 GPT-2 small 权重")
    model, tok = load_gpt2()
    prompt_ids = torch.tensor([tok.encode(PROMPT)], dtype=torch.long)   # (1, T)
    print(f"prompt = {PROMPT!r}")
    print(f"prompt 长度 = {prompt_ids.size(1)} 个 token，"
          f"每种策略生成 {MAX_NEW_TOKENS} 个新 token")

    # ------------------------------------------------------------------
    # 1. 五种解码策略，同一个 prompt
    # ------------------------------------------------------------------
    banner("1. 同一个 prompt，五种解码策略（各生成 40 个 token）")
    strategies = [
        ("greedy (T→0, argmax)", dict(greedy=True)),
        ("T=0.5, top_k=40", dict(temperature=0.5, top_k=TOP_K)),
        ("T=1.0, top_k=40", dict(temperature=1.0, top_k=TOP_K)),
        ("T=1.5, top_k=40", dict(temperature=1.5, top_k=TOP_K)),
        ("T=1.0, 不设 top_k", dict(temperature=1.0, top_k=None)),
    ]

    results = []
    for label, kwargs in strategies:
        text, new_ids, elapsed = run_generation(
            model, tok, label, prompt_ids, **kwargs)
        diversity = len(set(new_ids)) / len(new_ids)
        results.append((label, new_ids, elapsed, diversity))

        print(f"\n[{label}]   用时 {elapsed:.2f}s，"
              f"不同 token 数 {len(set(new_ids))}/{MAX_NEW_TOKENS}")
        print(wrap(PROMPT + text))

    print(f"""
怎么读这五段：
  greedy：每一步都选概率最高的 token，完全确定。
    读起来最「稳」，但容易一直重复同一小段（因为没有随机性把句子推出去）。
  T=0.5：分布被压尖，几乎只在最有把握的几个词里选，
    句子通顺、保守，接近 greedy 但不会死锁。
  T=1.0：用的就是模型输出的原始分布（softmax 不除任何东西），
    最接近「模型本来的想法」，多样性和通顺度比较平衡。
  T=1.5：分布被压平，低概率的怪词也有机会被选中，
    句子开始跑偏、语法变差——但偶尔会有更有创意的搭配。
  T=1.0 不设 top_k：从全部 50257 个 token 里抽，
    长尾里那些几乎不可能的 token 被抽中的概率很小，但一旦抽中，
    后面就会顺着这个错误往下写。top_k=40 的作用就是直接把长尾切掉。
""")

    # 温度越高，生成结果越多样（不同 token 的比例越高）——趋势判断，允许相等
    assert results[0][3] <= results[3][3], "greedy 的多样性不该高于 T=1.5"

    # ------------------------------------------------------------------
    # 2. 温度对 softmax 分布的影响
    # ------------------------------------------------------------------
    banner("2. 温度如何改变 softmax 概率（同一个位置，top-5 token）")
    print("softmax(logits / T)：温度只改 logits 的「尺度」，不改排序，")
    print("但 softmax 对尺度很敏感——除以 0.5 相当于把差距放大一倍，")
    print("除以 1.5 相当于把差距缩小到 2/3。")
    flat_tok, flat_max, flat_ent = show_temperature_probs(model, tok, PROMPT, k=5)
    print()
    peak_tok, peak_max, peak_ent = show_temperature_probs(
        model, tok, PEAKED_CONTEXT, k=5)

    print(f"""
这两张表放在一起看（下面的数字都是刚才实测的，不是写死的）：
  · 第一个 context 的分布比较平：T=1.0 时最大的 token 只占 {flat_max[1.0]:.1%}，
    T=0.5 时被放大到 {flat_max[0.5]:.1%}，T=1.5 时又被摊薄到 {flat_max[1.5]:.1%}。
    温度确实在改变「候选集的有效大小」（熵：{flat_ent[0.5]:.2f} < {flat_ent[1.0]:.2f} < {flat_ent[1.5]:.2f} bit）。
  · 第二个 context 的分布很尖：模型基本确定下一个词是 {peak_tok!r}，
    T=1.0 时概率 {peak_max[1.0]:.1%}，即使 T=1.5 也还有 {peak_max[1.5]:.1%}。
    对比两个 context 的熵：{flat_ent[1.0]:.2f} bit vs {peak_ent[1.0]:.2f} bit（都是 T=1.0）。
  · 所以温度不是「乱说」的开关：模型越确定的地方，温度的影响越小；
    模型越犹豫的地方，温度调高越容易跑偏。
  · 熵是最好用的直观指标：熵越大 = 模型越犹豫 = 生成越随机。
    均匀分布（完全瞎猜）的熵是 log2(50257) ≈ {math.log2(50257):.2f} bit。
""")

    # ------------------------------------------------------------------
    # 总结
    # ------------------------------------------------------------------
    banner("总结")
    print(f"  {'策略':<24} {'不同 token 数':>12} {'耗时(s)':>9}")
    print("  " + "-" * 48)
    for label, new_ids, elapsed, diversity in results:
        print(f"  {label:<24} {len(set(new_ids)):>8}/{MAX_NEW_TOKENS:<3} "
              f"{elapsed:>9.2f}")
    print(f"\n总耗时 {time.time() - t_start:.1f}s")
    print("\n[OK] exp5 完成：解码策略对比 + 温度对 softmax 分布的影响")


if __name__ == "__main__":
    main()
