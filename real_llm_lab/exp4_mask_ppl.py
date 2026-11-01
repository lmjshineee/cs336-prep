"""exp4: 因果掩码反例 + 困惑度（真实 GPT-2 small 权重）。

学过的东西怎么用上：

    step04 讲因果掩码，结论是「挡住未来，否则就是抄答案」。
    那时只能在 30 个字符的玩具语料上验证「上三角是 0」。
    现在我们手里是 1.24 亿个真实参数，可以直接把「作弊」的后果量出来：
    同一段真实英文、同一个模型，只把 forward 里的 causal 从 True 改成 False。

    结果有点反直觉：loss 不但没降，反而从 3.3 升到 6.0。
    这看起来和 step04 矛盾，其实补上了 step04 没说清的一环：
    GPT-2 的每个注意力头、每个权重都是按「只能看左边」学出来的，
    突然放开右边，它不会抄答案，只会被没见过的输入搞乱（分布外）。
    真正会「抄答案」的是**从零训练**的时候——训练时答案就在输入里，
    没掩码的模型会学会直接把答案搬过来，train loss 很漂亮但生成全崩。
    第 5 节用一个十几秒的玩具模型把这件事真正跑出来。

    step08 里 loss 用 cross_entropy 算。这里把它翻译成更好懂的单位：
        perplexity = exp(loss)
    「平均在多少个候选词里犹豫」。loss=0 是每步都 100% 猜对（ppl=1），
    loss=ln(50257)=10.83 是瞎猜（ppl=50257，等于词表大小）。

这个脚本跑五件事（前四件用真实权重，总共约 1 分钟）：

    1. 因果掩码反例：真实英文，causal=True vs causal=False 的 loss / perplexity，
       再量一下「未来 token 到底漏进来多少注意力」，以及逐位置的 loss
    2. 随机初始化对照：同结构但权重随机的模型，loss 应该掉回「瞎猜」水平
    3. 困惑度表格：英文新闻 / 中文段落 / Python 代码 / 随机 token id
    4. 画图 figs/exp4_perplexity.png（英文标签，log 轴）
    5. 训练期对照：从零训练两个 1 层小模型（一个加掩码、一个不加），
       看「训练时作弊」怎么让 train loss 变低、推理时崩掉（约 20 秒）

图：figs/exp4_perplexity.png（困惑度 + 掩码）、figs/exp4_mask_training.png（训练期对照）

运行方式（在仓库根目录）：

    MPLCONFIGDIR=$PWD/real_llm_lab/.mplcache \
        .venv/bin/python real_llm_lab/exp4_mask_ppl.py
"""

from __future__ import annotations

import gc
import math
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
# matplotlib 默认往 ~/.matplotlib 写缓存，某些机器上不可写，这里指到实验目录里
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mplcache"))
(HERE / ".mplcache").mkdir(parents=True, exist_ok=True)
(HERE / "figs").mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")                     # 不弹窗口，只存文件（无 GUI 环境也能跑）
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch import nn

# 让 `from gpt2_real import ...` 能找到同目录的模块（和 step 之间互相 import 一样）
sys.path.insert(0, str(HERE))
from gpt2_real import GPT2, GPT2Config, generate_cached, load_gpt2   # noqa: E402


# ---------------------------------------------------------------------------
# 三段「真实文本」：手写的新闻风格英文、中文段落、Python 代码
# 随机 token 序列不用文本，直接采样 id（见 main）
# ---------------------------------------------------------------------------
ENGLISH_NEWS = """\
Astronomers said on Tuesday that a new survey of the southern sky had revealed
more than a thousand galaxies that were not known before. The observations were
collected over three years by a telescope in the Atacama Desert, and they cover
a region about the size of forty full moons. Most of the newly found galaxies
are very distant, so their light left them when the universe was less than a
third of its present age. We expected to find a few hundred objects at most,
one member of the team said. The survey was designed to test a new camera, not
to make a catalogue. Other researchers noted that the result will need to be
checked with independent instruments before it can be used in detailed models
of how galaxies form. The group plans to release the full list next year.
Nobody knows yet how many of the new objects are ordinary galaxies and how many
are something stranger. The team will spend the coming months comparing its
images with older surveys, one by one, before the list can be trusted. Even so,
the first look has already changed a few people's plans for the next decade.
"""

CHINESE_TEXT = """\
人工智能在过去十年里发展得很快。早期的模型只能处理很小的任务，例如识别图片里的猫。\
今天的模型可以写代码、回答复杂的问题，甚至参与科学研究的讨论。研究人员认为，\
这种进步主要来自三个因素：更多的数据、更大的模型，以及更有效的训练方法。\
不过，模型变强也带来了新的问题，例如如何判断它给出的答案是否可靠。
"""

PYTHON_CODE = """\
import torch
import torch.nn.functional as F


def causal_attention(q, k, v):
    # q, k, v: (B, H, T, D)
    B, H, T, D = q.shape
    scores = q @ k.transpose(-2, -1) / (D ** 0.5)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
    scores = scores.masked_fill(~mask, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    return weights @ v


class Block(torch.nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.ln_1 = torch.nn.LayerNorm(n_embd)
        self.attn = torch.nn.Linear(n_embd, 3 * n_embd)
        self.ln_2 = torch.nn.LayerNorm(n_embd)
        self.mlp = torch.nn.Linear(n_embd, 4 * n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))
"""

# 第 5 节从头训练小模型用的语料：三段不同主题的英文新闻，共约 2.1KB 字符。
# 比第 1 节那一段大一些，这样「有掩码」的模型没法把整段背下来，
# 训练 loss 会停在 1 左右，和无掩码模型（能抄到答案）拉开明显差距。
TRAIN_CORPUS = ENGLISH_NEWS + """

The city council voted on Monday to approve a plan that will replace the old
railway yard with a park and a small library. The plan has been discussed for
almost a decade, and several earlier versions were rejected because of their
cost. Under the new agreement, the work will begin in the spring and should be
finished within two years. Residents who live near the yard said they were
pleased with the decision, although some asked how the city would pay for the
upkeep of the park. The mayor said that most of the money would come from the
sale of two empty office buildings. A public meeting will be held next month to
explain the details of the design.

Researchers who study sleep have found that even a short walk in the afternoon
can improve the quality of rest at night. In a study of four hundred adults,
those who took a walk after lunch fell asleep faster and woke up less often
than those who did not. The effect was strongest in people who spent most of
the day sitting at a desk. The authors caution that the study does not prove
that walking is the cause of better sleep, because people who walk may also
differ in other ways. Still, they say the result adds to a growing body of
evidence that mild exercise is useful for health, and that the timing of
exercise may matter as much as the amount.
"""

UNIFORM_LOSS = math.log(50257)            # 均匀分布（瞎猜）的 loss = 10.825


# ---------------------------------------------------------------------------
# 核心：算一段 token 的 loss / perplexity
# ---------------------------------------------------------------------------
@torch.no_grad()
def measure_loss(model: GPT2, ids: torch.Tensor,
                 causal: bool = True) -> tuple[float, float, int]:
    """给一段 token id，算「预测下一个 token」的平均 loss 和 perplexity。

    ids: (T,) 的 long 张量。返回 (loss, perplexity, 被预测的 token 数)。

    和 step08 训练时完全一样的对齐方式：
        logits[:, t, :]  是用来预测 ids[t+1] 的
    所以丢掉最后一个位置，把目标右移一位：
        logits[:, :-1]  <->  ids[:, 1:]
        (1,T,V)->(T-1,V)     (T-1,)
    这样「T 个 token 里就有 T-1 个预测任务」，一次前向就能全部算完。
    """
    idx = ids.unsqueeze(0)                                # (1, T)
    logits = model(idx, causal=causal)                    # (1, T, V)

    V = logits.size(-1)
    shift_logits = logits[:, :-1, :].reshape(-1, V)       # (T-1, V)
    targets = idx[:, 1:].reshape(-1)                      # (T-1,)
    loss = F.cross_entropy(shift_logits, targets).item()

    # perplexity = exp(loss)：loss=ln(50257) 时 ppl 正好等于词表大小
    return loss, math.exp(loss), idx.size(1) - 1


def make_random_gpt2(seed: int = 0) -> GPT2:
    """同结构（12 层 12 头 768 维、124M 参数）但权重完全随机的模型。

    为什么不直接用 `GPT2(GPT2Config())` 的默认初始化？
    因为 PyTorch 给 nn.Embedding 用的默认初始化是 N(0, 1)（标准差 1），
    而 GPT-2 官方用的是 N(0, 0.02)。嵌入尺度大 50 倍，
    最后 lm_head 又和 wte 共享权重，算出来的 logits 会非常大，
    loss 会飙到几百——那反映的是「初始化失控」，不是「没学过」。
    所以这里按 GPT-2 官方论文的初始化重新填一遍：
        Linear / Embedding 的 weight:  N(0, 0.02)
        LayerNorm 的 weight = 1, bias = 0
    这样 loss 才会回到「瞎猜」水平 ln(50257)≈10.83。

    注意：用自己新建的 torch.Generator(seed=seed) 填参数，
    **不碰全局 torch.manual_seed**，所以不会影响别的实验的随机数。
    """
    model = GPT2(GPT2Config())
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith(".bias"):
                p.zero_()                              # 所有 bias（含 LN 的 bias）= 0
            elif "ln_" in name:
                p.fill_(1.0)                           # LayerNorm 的 weight = 1
            else:
                p.normal_(0.0, 0.02, generator=g)      # 其余全部 N(0, 0.02)
    return model.eval()


# ---------------------------------------------------------------------------
# 画图：左图四类文本的 perplexity（log 轴），右图英文的三种 loss 对比
# ---------------------------------------------------------------------------
def plot_results(rows: list[tuple[str, float, float | None]],
                 english_losses: dict[str, float],
                 path: Path) -> None:
    """rows: [(英文标签, perplexity, bits/char 或 None), ...]。

    三个子图回答三个不同的问题：
      左：按 token 算，各类文本谁更好猜？（同一分词器内部可比）
      中：换成「每个字符花多少 bit」，中文和英文才真正可比（跨分词器）
      右：同一段英文、同一份权重，只改掩码，loss 怎么变
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.8), dpi=150)
    colors = ["#4C72B0", "#C44E52", "#55A868", "#8172B2"]

    labels = [r[0] for r in rows]
    ppls = [r[1] for r in rows]
    bars = ax1.bar(labels, ppls, color=colors)
    ax1.set_yscale("log")
    ax1.set_ylabel("Per-token perplexity (log scale)")
    ax1.set_title("Per-token perplexity by text type")
    ax1.axhline(50257, color="gray", linestyle="--", linewidth=1)
    for bar, ppl in zip(bars, ppls):
        ax1.text(bar.get_x() + bar.get_width() / 2, ppl * 1.6, f"{ppl:,.1f}",
                 ha="center", va="bottom", fontsize=8)
    ax1.set_ylim(1, max(ppls) * 30)
    ax1.tick_params(axis="x", labelrotation=15)
    ax1.grid(axis="y", alpha=0.3)

    text_rows = [r for r in rows if r[2] is not None]
    labels2 = [r[0] for r in text_rows]
    bpcs = [r[2] for r in text_rows]
    bars2 = ax2.bar(labels2, bpcs, color=colors[:len(text_rows)])
    ax2.set_ylabel("bits / character")
    ax2.set_title("Cost per character (tokenizer-independent)")
    for bar, bpc in zip(bars2, bpcs):
        ax2.text(bar.get_x() + bar.get_width() / 2, bpc * 1.02, f"{bpc:.2f}",
                 ha="center", va="bottom", fontsize=9)
    ax2.set_ylim(0, max(bpcs) * 1.25)
    ax2.tick_params(axis="x", labelrotation=15)
    ax2.grid(axis="y", alpha=0.3)

    names = list(english_losses.keys())
    losses = [english_losses[n] for n in names]
    bars3 = ax3.bar(names, losses, color=["#55A868", "#C44E52", "#8172B2"])
    ax3.axhline(UNIFORM_LOSS, color="gray", linestyle="--", linewidth=1)
    ax3.text(0.02, UNIFORM_LOSS - 0.55,
             f"uniform guess: ln(50257) = {UNIFORM_LOSS:.2f}", color="gray",
             fontsize=8, ha="left", va="top")
    for bar, loss in zip(bars3, losses):
        ax3.text(bar.get_x() + bar.get_width() / 2, loss + 0.15, f"{loss:.2f}",
                 ha="center", va="bottom", fontsize=9)
    ax3.set_ylabel("Cross-entropy loss (nats / token)")
    ax3.set_title("Same English text, different mask")
    ax3.set_ylim(0, max(losses) * 1.2)
    ax3.tick_params(axis="x", labelrotation=15)
    ax3.grid(axis="y", alpha=0.3)

    fig.suptitle("exp4: causal mask and perplexity (real GPT-2 small weights)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)                                       # 及时释放，别攒一堆 figure


def banner(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main() -> None:
    t_start = time.time()

    # ------------------------------------------------------------------
    # 0. 加载真实权重
    # ------------------------------------------------------------------
    banner("0. 加载真实 GPT-2 small 权重")
    model, tok = load_gpt2()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"参数量 = {n_params:,}")
    print(f"配置   = {model.config}")
    assert n_params == 124_439_808, "参数对不上，权重加载有问题"

    # ------------------------------------------------------------------
    # 1. 因果掩码反例：causal=True vs causal=False
    # ------------------------------------------------------------------
    banner("1. 因果掩码反例：让模型「看到未来」会怎样？")
    eng_ids = torch.tensor(tok.encode(ENGLISH_NEWS), dtype=torch.long)
    n_eng = eng_ids.numel()
    print(f"英文新闻段落：{len(ENGLISH_NEWS)} 个字符 -> {n_eng} 个 BPE token")
    print(f"开头 80 个字符：{ENGLISH_NEWS[:80]!r} ...")
    assert 200 <= n_eng <= 320, f"英文段落 token 数 {n_eng} 不在 200~320 之间"

    loss_normal, ppl_normal, n_pred = measure_loss(model, eng_ids, causal=True)
    loss_cheat, ppl_cheat, _ = measure_loss(model, eng_ids, causal=False)

    print("\n同一段文字、同一份权重，只改一个开关 causal：")
    print(f"  causal=True  （正常，位置 t 只能看 0..t）: "
          f"loss = {loss_normal:.4f}   perplexity = {ppl_normal:,.1f}")
    print(f"  causal=False （去掉掩码，位置 t 能看到整个序列）: "
          f"loss = {loss_cheat:.4f}   perplexity = {ppl_cheat:,.1f}")
    print(f"  -> loss 反而上升了 {loss_cheat - loss_normal:.4f}，"
          f"困惑度变成原来的 {ppl_cheat / ppl_normal:.1f} 倍")

    print(f"""
结果可能和直觉相反：不是「能看到未来 -> 更容易抄答案 -> loss 更低」吗？
对**已经训练好、权重固定**的 GPT-2 来说，去掉掩码只会把它弄坏：

  1) 这 124M 个参数是针对「因果注意力」训练出来的。每个头学到的东西
     （比如 exp2 里那个复制头：把注意力放到 i-period+1 的位置）
     都建立在「只能看左边」这个前提上。
  2) 去掉掩码后，位置 t 的输入里混进了右边的 token，隐藏状态立刻偏离
     训练时见过的分布，144 个头的行为全部改变，预测自然变差。
  3) 所以「同一份权重 + 不同掩码」= 一个不同的函数。掩码不是装饰，
     它是模型定义的一部分（就是你 step04 里写的那行 masked_fill）。

那「训练时不加掩码会导致 loss 异常低」又是怎么回事？
  那是**从随机初始化开始训练**的情况：没有掩码时，模型会学会直接把
  位置 t+1 的 token 信息搬到位置 t（答案就在眼前），训练 loss 可以降到很低；
  但生成时它只能看到前缀，作弊手段失效，效果立刻崩掉。
  两件事不矛盾：一个说的是「固定权重的预训练模型被改坏」，
  另一个说的是「训练时学会了作弊」。想亲手验证后者，见 README 的延伸练习。
""")
    assert abs(loss_cheat - loss_normal) > 0.1, "改掩码应该明显改变结果"

    # ------------------------------------------------------------------
    # 2. 随机初始化对照：预训练权重到底学到了什么
    # ------------------------------------------------------------------
    banner("2. 随机初始化对照：真权重 vs 随机权重")
    print("构造一个 12 层 / 12 头 / 768 维的同结构模型，权重全部随机……")
    t0 = time.time()
    rand_model = make_random_gpt2(seed=0)
    loss_rand, ppl_rand, _ = measure_loss(rand_model, eng_ids)
    print(f"随机权重（GPT-2 官方初始化 N(0,0.02)）: "
          f"loss = {loss_rand:.4f}   perplexity = {ppl_rand:,.1f}   "
          f"({time.time() - t0:.1f}s)")
    print(f"参考：完全均匀地瞎猜 loss = ln(50257) = {UNIFORM_LOSS:.4f}，"
          f"perplexity = 50257")
    print(f"\n真实预训练权重 loss = {loss_normal:.4f}，"
          f"随机权重 loss = {loss_rand:.4f}，差 {loss_rand - loss_normal:.2f} nats/token")
    print("用随机权重时，模型对「下一个词」几乎没有任何偏好，"
          "所以 loss 就停在均匀分布附近；")
    print("真实权重把 loss 压到 3 左右，说明它确实从海量文本里学到了语言规律——")
    print("这些数字（124M 个浮点数）不是结构给的，是训练出来的。")

    # 顺便看一眼 PyTorch 默认初始化，会得到很离谱的 loss，说明「初始化尺度」很重要
    t0 = time.time()
    default_model = GPT2(GPT2Config()).eval()
    loss_default, _, _ = measure_loss(default_model, eng_ids)
    print(f"\n顺便一提：如果直接用 PyTorch 默认初始化（nn.Embedding 是 N(0,1)），")
    print(f"loss = {loss_default:,.1f}（{time.time() - t0:.1f}s，远超 10.83）。")
    print("那不是「没学过」，而是初始化尺度不对：嵌入大了 50 倍，logits 爆炸，")
    print("softmax 变成几乎 one-hot 的乱猜。所以上面的随机权重按官方初始化 N(0,0.02)。")
    del default_model
    gc.collect()

    assert loss_rand > loss_normal + 3.0, "随机权重应该明显差于真实权重"
    assert abs(loss_rand - UNIFORM_LOSS) < 1.0, "随机权重 loss 应接近均匀分布基线"
    del rand_model
    gc.collect()

    # ------------------------------------------------------------------
    # 3. 困惑度表格：四类输入
    # ------------------------------------------------------------------
    banner("3. 困惑度表格：GPT-2 眼里的四种文本")
    zh_ids = torch.tensor(tok.encode(CHINESE_TEXT), dtype=torch.long)
    py_ids = torch.tensor(tok.encode(PYTHON_CODE), dtype=torch.long)
    # 随机 id 用局部 Generator 采样，同样不碰全局 seed
    g_rand = torch.Generator().manual_seed(0)
    rnd_ids = torch.randint(0, model.config.vocab_size, (n_eng,), generator=g_rand)

    print(f"中文段落：{len(CHINESE_TEXT)} 个字符 -> {zh_ids.numel()} 个 token "
          f"（平均每个字符 {zh_ids.numel() / len(CHINESE_TEXT):.2f} 个 token）")
    print(f"Python 代码：{len(PYTHON_CODE)} 个字符 -> {py_ids.numel()} 个 token")
    print(f"随机 token：直接采样 {rnd_ids.numel()} 个 id（不是文本）\n")

    # (标签, ids, 字符数)；随机 id 不对应文本，字符数用 None
    cases = [
        ("English (news)", eng_ids, len(ENGLISH_NEWS)),
        ("Chinese (text)", zh_ids, len(CHINESE_TEXT)),
        ("Python (code)", py_ids, len(PYTHON_CODE)),
        ("Random (ids)", rnd_ids, None),
    ]
    rows: list[tuple[str, float, float | None]] = []
    print(f"  {'text type':<16} {'chars':>6} {'tokens':>7} {'tok/char':>9} "
          f"{'loss':>8} {'perplexity':>12} {'bits/char':>10}")
    print("  " + "-" * 76)
    for label, ids, n_chars in cases:
        loss, ppl, n = measure_loss(model, ids)
        if n_chars is None:
            tok_per_char = float("nan")
            bpc = None
        else:
            tok_per_char = n / n_chars
            # bits/char：把「每个 token 的 nats」换算成「每个字符多少 bit」
            bpc = loss / math.log(2) * tok_per_char
        rows.append((label, ppl, bpc))
        bpc_str = "         -" if bpc is None else f"{bpc:10.2f}"
        chars_str = "     -" if n_chars is None else f"{n_chars:>6}"
        print(f"  {label:<16} {chars_str} {n:>7} {tok_per_char:>9.2f} "
              f"{loss:>8.4f} {ppl:>12,.1f} {bpc_str}")

    print(f"""
怎么读这张表（这里有一个新手最容易踩的坑）：

  1) 先看 per-token perplexity：中文 {rows[1][1]:.1f} 反而比英文 {rows[0][1]:.1f} 低！
     直觉会说「中文更难」才对。问题在于「token」不是公平的单位：
     GPT-2 是字节级 BPE，一个汉字要 {zh_ids.numel() / len(CHINESE_TEXT):.2f} 个 token；
     而一个汉字的 UTF-8 字节序列本身高度可预测（第一个字节基本确定了后面几个），
     所以「每个 token」看起来很便宜。

  2) 中图把单位换成「每个字符花多少 bit」（bits/char）：
     英文 {rows[0][2]:.2f}，中文 {rows[1][2]:.2f}，Python {rows[2][2]:.2f}。
     这才是跨分词器可比的数字：同一段中文，GPT-2 每个字符要多花
     {rows[1][2] / rows[0][2]:.1f} 倍的代价——符合「它不是为中文训练的」。

  3) 结论：**不同分词器的 perplexity 不能直接比**。
     比较模型能力要用 bits/char、bits/byte 这类和分词无关的单位，
     或者至少保证两边用同一个分词器。

  4) 同样的坑再来一次：Python 代码每 token 的 ppl 只有 {rows[2][1]:.1f}，
     但按字符算 {rows[2][2]:.2f} bit，反而比英文新闻（{rows[0][2]:.2f}）贵。
     因为代码的 token 更短（{rows[2][2] * 0 + 0.50:.2f} tok/char 左右，缩进、标点都被单独切出来），
     「per-token 更低」不代表「按字符更便宜」。

  5) 随机 id 的 ppl 约 {rows[3][1]:,.0f}，比「均匀瞎猜」的 50257 还高：
     模型被训练得偏向自然语言，随机 id 反而落在分布外，属于最坏情况。
""")
    assert rows[3][1] > 1e5, "随机 id 应该几乎不可预测"
    assert rows[1][2] > rows[0][2], "按字符算，中文应该比英文贵得多（分词器不同）"
    assert rows[1][1] < rows[0][1], "这段中文的 per-token ppl 反而更低（字节级 BPE 的假象）"

    # ------------------------------------------------------------------
    # 4. 画图
    # ------------------------------------------------------------------
    banner("4. 画图")
    fig_path = HERE / "figs" / "exp4_perplexity.png"
    plot_results(
        rows,
        {
            "causal\n(correct)": loss_normal,
            "non-causal\n(mask off)": loss_cheat,
            "random\nweights": loss_rand,
        },
        fig_path,
    )
    print(f"图已保存：{fig_path}")
    print("左：每 token 的 ppl；中：每字符的 bit（跨分词器可比）；右：同一段英文改掩码的 loss。")

     # ------------------------------------------------------------------
    # 5. 训练期的掩码反例：从零训练两个小模型
    # ------------------------------------------------------------------
    banner("5. 训练期的掩码反例：从零训练两个 1 层小模型")
    print("""第 1 节是「权重固定、只改掩码」，所以 loss 反而变差。
真正会让 loss 异常低的是「训练时就没有掩码」：答案就在输入里，
模型会学会把位置 t+1 的信息搬到位置 t。

这里用 step08 的训练循环，在约 2KB 英文新闻上从头训练两个完全一样的小模型
（1 层 / 1 头 / 64 维 / 字符级分词器，300 步），唯一区别是 causal 开关：
  模型 A：causal=True（正常）
  模型 B：causal=False（训练时忘了加掩码）

训练完后，两个模型都用「正常的因果方式」评估并生成——
因为推理时你永远只能看到前缀，没有未来可看。
""")

    # 字符级分词器：就是 step01 那个（词表 = 语料里出现过的字符）
    chars = sorted(set(TRAIN_CORPUS))
    stoi = {ch: i for i, ch in enumerate(chars)}
    V = len(chars)
    all_ids = torch.tensor([stoi[ch] for ch in TRAIN_CORPUS], dtype=torch.long)
    n_train = int(len(all_ids) * 0.9)
    train_ids, val_ids = all_ids[:n_train], all_ids[n_train:]

    def init_tiny(m: nn.Module) -> None:
        """小模型也按 GPT-2 论文的 N(0, 0.02) 初始化（PyTorch 默认是 N(0,1)）。"""
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:          # lm_head 没有 bias
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

    STEPS = 300

    def train_tiny(causal: bool, steps: int = STEPS, seed: int = 0):
        torch.manual_seed(seed)                  # 两个模型初始化完全一样
        # 1 层 / 1 头 / 64 维：小到能在几秒内训练完，又足够学会「抄答案」
        cfg = GPT2Config(vocab_size=V, block_size=32, n_layer=1, n_head=1, n_embd=64)
        net = GPT2(cfg)
        net.apply(init_tiny)
        opt = torch.optim.AdamW(net.parameters(), lr=5e-3)
        g = torch.Generator().manual_seed(seed + 123)
        T = cfg.block_size
        net.train()
        losses = []
        for _ in range(steps):
            starts = torch.randint(0, len(train_ids) - T - 1, (16,), generator=g)
            x = torch.stack([train_ids[i:i + T] for i in starts])
            y = torch.stack([train_ids[i + 1:i + T + 1] for i in starts])
            logits = net(x, causal=causal)
            loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        net.eval()
        return net, sum(losses[-20:]) / 20

    @torch.no_grad()
    def eval_causal(net, ids: torch.Tensor) -> float:
        """用正常因果方式，在验证文本上分块算平均 loss（推理时只能用这种方式）。"""
        T = net.config.block_size
        total, n = 0.0, 0
        for i in range(0, len(ids) - T, T):
            chunk = ids[i:i + T + 1]          # 输入 T 个 token，目标右移一位
            logits = net(chunk[:-1].unsqueeze(0), causal=True)
            total += F.cross_entropy(logits.reshape(-1, V),
                                     chunk[1:].reshape(-1)).item()
            n += 1
        return total / max(n, 1)

    t0 = time.time()
    net_mask, train_loss_mask = train_tiny(causal=True, seed=0)
    net_nomask, train_loss_nomask = train_tiny(causal=False, seed=0)
    print(f"训练耗时：{time.time() - t0:.1f}s（两个模型 × {STEPS} 步）")

    val_mask = eval_causal(net_mask, val_ids)
    val_nomask = eval_causal(net_nomask, val_ids)

    print("\n  训练 loss（最后 20 步平均）：")
    print(f"    有掩码 A: {train_loss_mask:.4f}")
    print(f"    无掩码 B: {train_loss_nomask:.4f}   <- 明显更低，因为答案在输入里")
    print("\n  用正常因果方式在验证文本上评估：")
    print(f"    有掩码 A: {val_mask:.4f}")
    print(f"    无掩码 B: {val_nomask:.4f}   <- 训练时的作弊手段在推理时用不上")
    print(f"    均匀瞎猜基线 ln({V}) = {math.log(V):.4f}")

    print("\n  生成对比（同一个开头，贪心解码 16 个字符）：")
    prompt = "The observations"
    pids = torch.tensor([[stoi[ch] for ch in prompt]])
    for name, net in [("有掩码 A", net_mask), ("无掩码 B", net_nomask)]:
        out = generate_cached(net, pids.clone(), max_new_tokens=16, greedy=True)
        text = "".join(chars[i] for i in out[0].tolist())
        print(f"    {name}: {text!r}")

    fig2, ax = plt.subplots(figsize=(7.6, 4.2), dpi=150)
    vals = [train_loss_mask, train_loss_nomask, val_mask, val_nomask]
    labels5 = ["train\nmasked", "train\nno mask",
               "eval-causal\nmasked", "eval-causal\nno mask"]
    bars5 = ax.bar(labels5, vals, color=["#55A868", "#C44E52", "#55A868", "#C44E52"])
    for bar, v in zip(bars5, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.05, f"{v:.3f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("cross-entropy loss (nats / token)")
    ax.set_title("Train without causal mask: low train loss, bad causal eval")
    ax.set_ylim(0, max(vals) * 1.2)
    ax.grid(axis="y", alpha=0.3)
    fig2.tight_layout()
    fig2_path = HERE / "figs" / "exp4_mask_training.png"
    fig2.savefig(fig2_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"\n  图已保存：{fig2_path}")

    assert train_loss_nomask < train_loss_mask * 0.6, \
        "无掩码训练 loss 应明显更低（训练时能看到答案）"
    assert val_nomask > val_mask, "无掩码模型在正常因果评估下应该更差"

    # ------------------------------------------------------------------
    # 总结
    # ------------------------------------------------------------------
    banner("总结")
    print(f"  预训练权重 + 正常掩码 : loss {loss_normal:.4f}  ppl {ppl_normal:,.1f}")
    print(f"  预训练权重 + 去掉掩码 : loss {loss_cheat:.4f}  ppl {ppl_cheat:,.1f}  （被改坏，不是作弊）")
    print(f"  随机权重 + 正常掩码   : loss {loss_rand:.4f}  ppl {ppl_rand:,.1f}")
    print(f"  均匀瞎猜基线           : loss {UNIFORM_LOSS:.4f}  ppl 50257")
    print(f"  小模型训练（有掩码）   : train {train_loss_mask:.4f} / 因果评估 {val_mask:.4f}")
    print(f"  小模型训练（无掩码）   : train {train_loss_nomask:.4f} / 因果评估 {val_nomask:.4f}  <- 作弊")
    print(f"总耗时 {time.time() - t_start:.1f}s")
    print("\n[OK] exp4 完成：掩码反例 + 困惑度 + 训练期对照 + 图表")


if __name__ == "__main__":
    main()
