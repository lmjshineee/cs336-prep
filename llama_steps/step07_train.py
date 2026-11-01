"""Step 7: 同预算训练 GPT-2 与 Llama —— 这几处改造，到底值不值？

GPT-2 的做法（本文件内联了这个对照组，就是你在 transformer_steps 写过的那个）：
    LayerNorm + GELU 的 4C 宽 FFN + 标准 MHA + 学习式绝对位置嵌入，
    lm_head 和 token_embedding 共享权重。

Llama 的做法（从 step06_model 导入）：
    RMSNorm + SwiGLU（隐藏维 8/3 C）+ GQA（4 个 Q 头共享 2 组 KV）+ RoPE，
    完全没有位置嵌入，lm_head 独立。

为什么要这样比：
    1) 问题：前面每一步都只演示「一个改造」，没人回答四个改造一起上到底有没有用；
    2) 方案：同一份语料、同一串 batch、同一优化器配方、同样 500 步，分别训一遍，
       唯一的变量是架构本身（层数、宽度、头数、训练预算都相同）。注意「同预算」=
       同样的训练 token 数（500 × 16 × 32），不是同参数量：实测 Llama 反而少 8.2%；
    3) 代价/风险：玩具规模（2 层、64 维、438 字符、500 步）远小于真实 LLM，
       架构优势很可能被「两个模型都只是背下这段语料」淹没。如果没有差距，本文件
       会照实说没有差距，并把原因写清楚，不挑好听的说。

关键公式：
    参数预算  GPT-2 block = 4C²(attn) + 8C²(FFN) + 4C(两个 LayerNorm)
              Llama block = 3C²(GQA, g=2) + 3·C·F(SwiGLU) + 2C(两个 RMSNorm)
    KV/每 token = 2 · n_layer · n_kv_head · head_dim · sizeof(dtype)
    随机初始化时 loss ≈ ln(V)，V=30 时约 3.4012

和上一步的 diff：
    - 只有 forward 和 KV cache，没有 loss、没有优化器、没有生成
    + 两个模型各训 500 步的真实 loss、ms/step、KV 字节换算、生成样例
    + 一段只基于实测数字的结论（不好看也照实写）

跑法（在仓库根目录）：

    .venv/bin/python llama_steps/step07_train.py
"""

from __future__ import annotations

import math
import os
import statistics
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

# matplotlib 默认往 ~/.matplotlib 写缓存，这里指到仓库内可写目录，必须在 import 之前。
os.environ.setdefault("MPLCONFIGDIR", str(REPO / ".tmp" / "mpl"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")           # 无窗口后端，直接存 PNG
import matplotlib.pyplot as plt  # noqa: E402

from step06_model import (CORPUS, CharTokenizer, KVCache,  # noqa: E402
                          LlamaConfig, MiniLlama)

# ---------------------------------------------------------------------------
# 实验配置：两个模型共享的一切都在这里，唯一自由的是各自的结构
# ---------------------------------------------------------------------------
SEED = 0
MAX_SEQ = 64          # 模型最大序列长度（GPT-2 位置表大小 / RoPE 缓存长度）
TRAIN_BLOCK = 32      # 训练样本长度（= 前面 spec 要求的 block）
BATCH_SIZE = 16
STEPS = 500
LR = 3e-3
BETAS = (0.9, 0.95)
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
WARMUP_STEPS = 50     # 10% 的步数做线性 warmup
N_LAYER = 2
N_HEAD = 4
N_KV_HEAD = 2         # Llama：4 个 Q 头分成 2 组，共享 K/V
N_EMBD = 64
FFN_HIDDEN = 176      # 8/3 * 64 ≈ 170.7，取 8 的倍数
GEN_NEW = 40          # 生成 40 个新字符，加上 20 字符 prompt 正好 60
TOKENS_PER_STEP = BATCH_SIZE * TRAIN_BLOCK


def _width(text: str) -> int:
    """终端里中文占 2 列，按显示宽度算才能对齐。"""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def _pad(text: str, width: int, right: bool = False) -> str:
    """按终端显示宽度补空格；right=True 时右对齐（用于数字列）。"""
    blanks = " " * max(0, width - _width(text))
    return blanks + text if right else text + blanks


def show(name: str, value: object, note: str = "") -> None:
    """打印一行对齐的「名字 = 值」，note 是行尾的人话注释。"""
    if isinstance(value, torch.Tensor):
        value = tuple(value.shape)
    if isinstance(value, int):
        value = f"{value:,}"
    pad = " " * max(0, 22 - _width(name))
    tail = f"   <- {note}" if note else ""
    print(f"{name}{pad} = {value}{tail}")


# ---------------------------------------------------------------------------
# 对照组：这是你之前写的 GPT-2，原样搬来做对照
# 来源 transformer_steps/step07_model.py，只去掉了 KV cache 生成路径：
# 这一节让两个模型共用同一个「全量重算」的生成函数，生成对比才公平。
# ---------------------------------------------------------------------------
@dataclass
class GPT2Config:
    vocab_size: int
    block_size: int = 64
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 64
    dropout: float = 0.0      # toy 配置是 0；两个模型都不加 dropout


class GPT2Attention(nn.Module):
    """标准 MHA：Q、K、V 都是 n_head 个头，每个头 head_dim 维。"""

    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)        # 各 (B, T, C)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
        scores = scores.masked_fill(~mask, float("-inf"))
        out = (F.softmax(scores, dim=-1) @ v).transpose(1, 2).contiguous()
        return self.out_proj(out.view(B, T, C))


class GPT2FFN(nn.Module):
    """GPT-2 的 FFN：C -> 4C -> C，中间夹 GELU，两个矩阵共 8C²。"""

    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.up = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.down = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.gelu(self.up(x)))


class GPT2Block(nn.Module):
    """Pre-Norm：先归一化再进子层，最后加回残差（和 Llama 的方向一致）。"""

    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)     # LayerNorm 每个 2C 参数
        self.attn = GPT2Attention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ffn = GPT2FFN(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.ffn(self.ln2(x))


class MiniGPT2(nn.Module):
    """GPT-2 玩具版：学习式位置嵌入 + 与 lm_head 共享权重的 token 嵌入。"""

    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.ModuleList(
            [GPT2Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight   # 权重共享，省 V*C
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)    # 绝对位置，0..T-1
        x = self.token_embedding(idx) + self.position_embedding(pos)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.ln_f(x))          # (B, T, V)


# ---------------------------------------------------------------------------
# 共用的训练 / 评测 / 生成工具
# ---------------------------------------------------------------------------
def get_batch(data: torch.Tensor, block_size: int, batch_size: int,
              generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """随机取 batch_size 个起点，切出等长 x 和右移一位的 y。

    两个模型共用这个函数，而且共用同一个 generator：
    先把 500 个 batch 全部生成好，再分别训练，保证两边看到的 batch 逐位相同。
    """
    starts = torch.randint(0, len(data) - block_size - 1, (batch_size,),
                           generator=generator)
    x = torch.stack([data[i:i + block_size] for i in starts])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in starts])
    return x.to(torch.long), y.to(torch.long)


def lr_scale(step: int) -> float:
    """前 10% 步线性 warmup，之后余弦退火到 0（step 从 0 开始）。"""
    if step < WARMUP_STEPS:
        return (step + 1) / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / max(1, STEPS - WARMUP_STEPS)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def train_model(model: nn.Module, batches: list[tuple[torch.Tensor,
                                                      torch.Tensor]],
                tag: str) -> tuple[list[float], list[float]]:
    """在同一串 batch 上训练，返回 (每步 loss, 每步耗时 ms)。"""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=BETAS,
                                  weight_decay=WEIGHT_DECAY)
    losses: list[float] = []
    step_ms: list[float] = []

    for step, (xb, yb) in enumerate(batches):
        for group in optimizer.param_groups:
            group["lr"] = LR * lr_scale(step)

        t0 = time.perf_counter()
        logits = model(xb)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               yb.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        step_ms.append((time.perf_counter() - t0) * 1000.0)
        losses.append(loss.item())

        if (step + 1) % 100 == 0:
            print(f"  [{tag:<5}] step {step + 1:3d} | loss {loss.item():.4f}")

    return losses, step_ms


@torch.no_grad()
def eval_loss(model: nn.Module, batches: list[tuple[torch.Tensor,
                                                    torch.Tensor]]) -> float:
    """在固定的 8 个评测 batch 上算平均 loss（这些 batch 不参与训练）。"""
    model.eval()
    total = 0.0
    for xb, yb in batches:
        logits = model(xb)
        total += F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                 yb.reshape(-1)).item()
    return total / len(batches)


def _next_token(logits: torch.Tensor, greedy: bool, temperature: float,
                top_k: int) -> torch.Tensor:
    """从 (B, V) 的 logits 里选下一个 token；两个模型共用这段逻辑。"""
    if greedy:
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits / temperature
    values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
    logits[logits < values[:, [-1]]] = float("-inf")
    return torch.multinomial(F.softmax(logits, dim=-1), 1)


@torch.no_grad()
def generate(model: nn.Module, idx: torch.Tensor, n_new: int,
             tok: CharTokenizer, greedy: bool, temperature: float = 0.8,
             top_k: int = 20) -> str:
    """全量重算的生成（不用 KV cache，两个模型用同一份代码）。"""
    model.eval()
    if idx.size(1) + n_new > MAX_SEQ:
        raise ValueError(f"prompt {idx.size(1)} + new {n_new} > MAX_SEQ {MAX_SEQ}")
    for _ in range(n_new):
        logits = model(idx)
        nxt = _next_token(logits[:, -1, :], greedy, temperature, top_k)
        idx = torch.cat((idx, nxt), dim=1)
    return tok.decode(idx[0].tolist())


# ---------------------------------------------------------------------------
# 参数量：先逐模块实测（sum(p.numel())），再用公式交叉验证，表格只报实测值
# ---------------------------------------------------------------------------
PARAM_KEYS = ("token_embedding", "position_embedding", "lm_head",
              "attention", "ffn", "norm")


def gpt2_param_parts(config: GPT2Config, vocab: int) -> dict[str, int]:
    """GPT-2 各组件参数量的公式值（token 嵌入与 lm_head 共享，只算一次）。"""
    C, L = config.n_embd, config.n_layer
    return {
        "token_embedding": vocab * C,
        "position_embedding": config.block_size * C,
        "lm_head": 0,                       # 与 token_embedding 共享
        "attention": 4 * C * C * L,         # qkv 3C² + out C²
        "ffn": 8 * C * C * L,               # C -> 4C -> C
        "norm": 4 * C * L + 2 * C,          # 每层两个 LN（各 2C）+ 最后的 LN
    }


def llama_param_parts(config: LlamaConfig, vocab: int,
                      tied: bool) -> dict[str, int]:
    """Llama 各组件参数量的公式值；GQA 的 K/V 只有 n_kv_head 组。"""
    C, L, F_ = config.n_embd, config.n_layer, config.ffn_hidden
    head_dim = C // config.n_head
    kv_dim = config.n_kv_head * head_dim
    attn = (C * C + 2 * C * kv_dim + C * C) * L   # q / k / v / o
    return {
        "token_embedding": vocab * C,
        "position_embedding": 0,            # RoPE 没有位置表
        "lm_head": 0 if tied else vocab * C,
        "attention": attn,                  # g=2 时一共 3C²
        "ffn": 3 * C * F_ * L,              # SwiGLU 三个矩阵
        "norm": 2 * C * L + C,              # 每层两个 RMSNorm（各 C）+ 最后
    }


def measured_param_parts(model: nn.Module, is_gpt2: bool) -> dict[str, int]:
    """直接数每个子模块的参数量，不经过任何公式。"""
    parts = dict.fromkeys(PARAM_KEYS, 0)
    parts["token_embedding"] = model.token_embedding.weight.numel()
    if is_gpt2:
        parts["position_embedding"] = model.position_embedding.weight.numel()
        norm_names, final_norm = ("ln1", "ln2"), model.ln_f
    else:
        norm_names, final_norm = ("attn_norm", "ffn_norm"), model.norm
    if model.lm_head.weight is not model.token_embedding.weight:
        parts["lm_head"] = model.lm_head.weight.numel()
    for block in model.blocks:
        parts["attention"] += sum(p.numel() for p in block.attn.parameters())
        parts["ffn"] += sum(p.numel() for p in block.ffn.parameters())
        for name in norm_names:
            parts["norm"] += sum(p.numel()
                                 for p in getattr(block, name).parameters())
    parts["norm"] += sum(p.numel() for p in final_norm.parameters())
    return parts


def print_param_table(rows_gpt2: dict[str, int], rows_llama: dict[str, int],
                      total_gpt2: int, total_llama: int) -> None:
    """打印 GPT-2 vs Llama 的参数量分解表（实测值 + 差值）。"""
    labels = {
        "token_embedding": "token_embedding",
        "position_embedding": "position_embedding",
        "lm_head": "lm_head",
        "attention": "attention（2 层）",
        "ffn": "ffn（2 层）",
        "norm": "norm（2 层 + 最终）",
    }
    w = 26
    print(_pad("item", w) + _pad("GPT-2", 10, right=True)
          + _pad("Llama", 10, right=True) + _pad("delta", 10, right=True))
    print("-" * 56)
    for key in rows_gpt2:
        a, b = rows_gpt2[key], rows_llama[key]
        print(_pad(labels[key], w) + f"{a:>10,}{b:>10,}{b - a:>+10,}")
    print("-" * 56)
    print(_pad("total", w) + f"{total_gpt2:>10,}{total_llama:>10,}"
          f"{total_llama - total_gpt2:>+10,}")


# ---------------------------------------------------------------------------
# 画图：两条 loss 曲线（英文标注，macOS 没有中文字体）
# ---------------------------------------------------------------------------
def make_figure(gpt2_losses: list[float], llama_losses: list[float],
                vocab: int) -> Path:
    figs = HERE / "figs"
    figs.mkdir(parents=True, exist_ok=True)
    out = figs / "step07_loss.png"

    steps = list(range(1, len(gpt2_losses) + 1))
    g50 = statistics.mean(gpt2_losses[-50:])
    l50 = statistics.mean(llama_losses[-50:])
    c_g, c_l = "#1f77b4", "#d95f02"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.8))
    ax1.plot(steps, gpt2_losses, color=c_g, linewidth=1.3,
             label=f"GPT-2 baseline (final-50 mean {g50:.3f})")
    ax1.plot(steps, llama_losses, color=c_l, linewidth=1.3,
             label=f"Llama (final-50 mean {l50:.3f})")
    ax1.axhline(math.log(vocab), color="grey", linestyle="--", linewidth=1.0,
                label=f"ln(V) = {math.log(vocab):.2f} at init")
    ax1.set_xlabel("training step")
    ax1.set_ylabel("training loss (cross-entropy)")
    ax1.set_title("Same data, same budget, only architecture differs\n"
                  "char-level, 2 layers, C=64, 500 steps", fontsize=11)
    ax1.legend(fontsize=8.5, loc="upper right")
    ax1.grid(alpha=0.3, linestyle="--")
    ax1.set_axisbelow(True)

    # 右图放大最后 120 步：两条曲线到后面几乎重合，需要放大才能看清
    zoom = 120
    ax2.plot(steps[-zoom:], gpt2_losses[-zoom:], color=c_g, linewidth=1.6,
             marker="o", markersize=2.2, markevery=10, label="GPT-2")
    ax2.plot(steps[-zoom:], llama_losses[-zoom:], color=c_l, linewidth=1.6,
             marker="s", markersize=2.2, markevery=10, label="Llama")
    ax2.axhline(g50, color=c_g, linestyle=":", linewidth=1.0, alpha=0.8)
    ax2.axhline(l50, color=c_l, linestyle=":", linewidth=1.0, alpha=0.8)
    ax2.set_xlabel("training step")
    ax2.set_ylabel("training loss (cross-entropy)")
    ax2.set_title(f"(b) last {zoom} steps  (final-50 mean: GPT-2 {g50:.3f}, "
                  f"Llama {l50:.3f})", fontsize=10)
    ax2.legend(fontsize=9, loc="upper right")
    ax2.grid(alpha=0.3, linestyle="--")
    ax2.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\n[OK] 图已保存: {out.relative_to(REPO)} "
          f"(150 dpi, {out.stat().st_size / 1024:.0f} KiB)")
    return out


def main() -> None:
    torch.manual_seed(SEED)
    device = torch.device("cpu")

    tok = CharTokenizer(CORPUS)
    data = torch.tensor(tok.encode(CORPUS), dtype=torch.long)
    vocab = tok.vocab_size
    ln_vocab = math.log(vocab)

    gpt2_config = GPT2Config(vocab_size=vocab, block_size=MAX_SEQ,
                             n_layer=N_LAYER, n_head=N_HEAD, n_embd=N_EMBD)
    llama_config = LlamaConfig(vocab_size=vocab, block_size=MAX_SEQ,
                               n_layer=N_LAYER, n_head=N_HEAD,
                               n_kv_head=N_KV_HEAD, n_embd=N_EMBD,
                               ffn_hidden=FFN_HIDDEN)

    # 两个模型用同一个种子初始化：参数总量不同，但初始化分布可比
    torch.manual_seed(SEED)
    gpt2 = MiniGPT2(gpt2_config).to(device)
    torch.manual_seed(SEED)
    llama = MiniLlama(llama_config).to(device)

    # ---- 先实测参数量：§1 要引用差额，§2 要列表 ----
    gpt2_rows = measured_param_parts(gpt2, is_gpt2=True)
    llama_rows = measured_param_parts(llama, is_gpt2=False)
    gpt2_total = sum(p.numel() for p in gpt2.parameters())
    llama_total = sum(p.numel() for p in llama.parameters())
    # 公式只做交叉检查；表里报的是上面实测出来的数字
    assert gpt2_rows == gpt2_param_parts(gpt2_config, vocab)
    assert llama_rows == llama_param_parts(
        llama_config, vocab, llama.lm_head.weight is llama.token_embedding.weight)
    assert sum(gpt2_rows.values()) == gpt2_total
    assert sum(llama_rows.values()) == llama_total

    print("\n--- 1. 实验设计：唯一的变量是架构 ---")
    show("corpus", f"{len(data)} chars", "8 句英文重复两遍，字符级分词")
    show("vocab_size", vocab, "字符级，空格和标点都算一个 token")
    show("budget", f"{STEPS} steps x batch {BATCH_SIZE} x block {TRAIN_BLOCK}",
         f"{TOKENS_PER_STEP} tokens/step")
    print("两个模型共用同一串 batch 和同一套 AdamW 配方，唯一不同的是结构。")
    print(f"这里的「同预算」指 {STEPS} 步 x batch {BATCH_SIZE} x block {TRAIN_BLOCK} = 同样的")
    print(f"训练 token 数，不是同参数量：Llama 反而少 {gpt2_total - llama_total:,}"
          f"（{1 - llama_total / gpt2_total:.1%}）。")

    print("\n--- 2. 参数量分解（实测 sum(p.numel())，C=64, 2 层, 词表 30）---")
    print_param_table(gpt2_rows, llama_rows, gpt2_total, llama_total)
    print(f"Llama 是 GPT-2 的 {llama_total / gpt2_total:.1%}：attention 少 "
          f"{1 - llama_rows['attention'] / gpt2_rows['attention']:.0%}"
          f"（4C² → 3C²），FFN 多 "
          f"{llama_rows['ffn'] / gpt2_rows['ffn'] - 1:.1%}（SwiGLU 三矩阵）。")
    print("[OK] 实测分解与参数量公式逐项一致，两个 total 也等于 params 总数")

    # ---- 3. 训练 ----
    print("\n--- 3. 分别训练（同一串 batch，同一优化器配方）---")
    print(f"AdamW lr={LR:g} betas={BETAS} wd={WEIGHT_DECAY} "
          f"clip={GRAD_CLIP}，warmup {WARMUP_STEPS} 步 + cosine")
    print("warmup 是前 10% 步把学习率从 0 线性拉到最大值，避免一开始大步长搅乱随机初始化；")
    print("GPT-2 当年用 Adam + 手工 warmup，这里统一 AdamW + cosine，只为只留下架构一个变量。")
    # 训练前先确认两个模型都没写错：改最后一个 token，不应该影响前面位置的 logits
    with torch.no_grad():
        probe = torch.randint(0, vocab, (1, 12))
        probe_changed = probe.clone()
        probe_changed[:, -1] = (probe_changed[:, -1] + 1) % vocab
        for model in (gpt2, llama):
            before = model(probe)
            after = model(probe_changed)
            assert torch.allclose(before[:, :-1], after[:, :-1], atol=1e-6)
    print("[OK] 两个模型都满足因果性：改未来 token，不影响过去位置的 logits")
    gen_batches = torch.Generator().manual_seed(SEED)
    batches = [get_batch(data, TRAIN_BLOCK, BATCH_SIZE, gen_batches)
               for _ in range(STEPS)]
    eval_gen = torch.Generator().manual_seed(SEED + 7)
    eval_batches = [get_batch(data, TRAIN_BLOCK, 8, eval_gen) for _ in range(8)]

    gpt2_losses, gpt2_ms = train_model(gpt2, batches, "GPT-2")
    llama_losses, llama_ms = train_model(llama, batches, "Llama")

    # ---- 4. loss 对比 ----
    g50 = statistics.mean(gpt2_losses[-50:])
    l50 = statistics.mean(llama_losses[-50:])
    g_first = statistics.mean(gpt2_losses[:50])
    l_first = statistics.mean(llama_losses[:50])
    g_eval = eval_loss(gpt2, eval_batches)
    l_eval = eval_loss(llama, eval_batches)
    assert abs(gpt2_losses[0] - ln_vocab) < 0.5, gpt2_losses[0]
    assert abs(llama_losses[0] - ln_vocab) < 0.5, llama_losses[0]
    assert g50 < g_first and l50 < l_first, "两个模型都应该在 500 步里明显收敛"

    print("\n--- 4. loss（真实数字，没有挑好看的）---")
    show("GPT-2 last-50 mean", f"{g50:.4f}", "最后 50 步训练 loss 均值")
    show("Llama last-50 mean", f"{l50:.4f}", "最后 50 步训练 loss 均值")
    show("delta (Llama-GPT-2)", f"{l50 - g50:+.4f}", "负数 = Llama 更低")
    show("fixed eval batches", f"GPT-2 {g_eval:.4f} | Llama {l_eval:.4f}",
         "8 个固定评测 batch，不参与训练")
    print("说明：这 8 个 batch 仍来自同一份 438 字符语料，固定下来只为减少抖动，不是泛化")
    print("指标（模型基本背下了语料，别把它当验证集）。")
    print("[OK] 两个模型都从 ln(V) 附近开始，并在 500 步内明显下降")

    make_figure(gpt2_losses, llama_losses, vocab)

    # ---- 5. 速度 ----
    g_med = statistics.median(gpt2_ms)
    l_med = statistics.median(llama_ms)
    g_tps = TOKENS_PER_STEP / (g_med / 1000.0)
    l_tps = TOKENS_PER_STEP / (l_med / 1000.0)
    print("\n--- 5. 速度（CPU fp32，含前向 + 反向 + clip + step）---")
    show("GPT-2", f"{g_med:.1f} ms/step = {g_tps:,.0f} tokens/s")
    show("Llama", f"{l_med:.1f} ms/step = {l_tps:,.0f} tokens/s")
    print("（玩具模型上 Python/调度开销占比很大，ms/step 只代表这台机器）")

    # ---- 6. 生成样例 ----
    prompt = CORPUS[:20]
    assert len(prompt) == 20, len(prompt)
    idx = torch.tensor([tok.encode(prompt)], dtype=torch.long)
    gpt2_greedy = generate(gpt2, idx.clone(), GEN_NEW, tok, greedy=True)
    llama_greedy = generate(llama, idx.clone(), GEN_NEW, tok, greedy=True)
    torch.manual_seed(SEED + 100)      # 两个模型用同一个随机流，采样才可比
    gpt2_temp = generate(gpt2, idx.clone(), GEN_NEW, tok, greedy=False)
    torch.manual_seed(SEED + 100)
    llama_temp = generate(llama, idx.clone(), GEN_NEW, tok, greedy=False)
    assert len(gpt2_greedy) == len(llama_greedy) == 60, "总长应为 60 字符"
    assert len(gpt2_temp) == len(llama_temp) == 60

    print(f"\n--- 6. 生成样例（同一 prompt，总长 60 字符）---")
    print(f'prompt = "{prompt}"   -> 各自续写 {GEN_NEW} 个字符')
    print(f"{'GPT-2 greedy':<15}| {gpt2_greedy}")
    print(f"{'Llama greedy':<15}| {llama_greedy}")
    print(f"{'GPT-2 T=0.8':<15}| {gpt2_temp}")
    print(f"{'Llama T=0.8':<15}| {llama_temp}")
    print("注意：两边都超出了训练长度（训练 block=32）：GPT-2 在查没训练过的位置行，RoPE 不越界")
    print("      但远距离同样会退化；这段样例对两者都偏苛刻（loss 对照只发生在长度 32 内，不受影响）。")

    # ---- 7. KV cache 每 token 字节数 ----
    head_dim = N_EMBD // N_HEAD
    g_toy_fp32 = 2 * N_LAYER * N_HEAD * head_dim * 4        # MHA：K/V 都是 4 组
    l_toy_fp32 = 2 * N_LAYER * N_KV_HEAD * head_dim * 4     # GQA：K/V 只有 2 组
    g_toy_fp16 = g_toy_fp32 // 2
    l_toy_fp16 = l_toy_fp32 // 2
    # 真实规模：Llama-3-8B（32 层、32 个 Q 头、8 组 KV、head_dim 128，fp16）
    real_layers, real_kv, real_head_dim = 32, 8, 128
    l_real = 2 * real_layers * real_kv * real_head_dim * 2
    g_real = 2 * real_layers * 32 * real_head_dim * 2       # 同规模 MHA（32 组 KV）
    toy_cache = KVCache(llama_config, 1, device)
    expected = l_toy_fp32 * MAX_SEQ
    assert toy_cache.nbytes() == expected, (toy_cache.nbytes(), expected)

    print("\n--- 7. KV cache：每生成一个 token 要多存多少字节 ---")
    print("公式：2 · n_layer · n_kv_head · head_dim · sizeof(dtype)（K、V 各一份）")
    print(f"toy 配置 fp32：GPT-2 MHA(4 组 KV) {g_toy_fp32:,} B/token | "
          f"Llama GQA(2 组 KV) {l_toy_fp32:,} B/token")
    print(f"toy 配置 fp16：GPT-2 {g_toy_fp16:,} B/token | Llama {l_toy_fp16:,} B/token")
    print(f"真实 Llama-3-8B（32 层, 8 组 KV, head_dim 128, fp16）："
          f"{l_real:,} B/token ({l_real / 1024:.0f} KiB)")
    print(f"  若用 GPT-3 式 MHA（32 组 KV）同规模：{g_real:,} B/token "
          f"({g_real / 1024:.0f} KiB)，正好是 GQA 的 {g_real // l_real} 倍")
    print(f"  8192 token 上下文：GQA {l_real * 8192 / 1024**3:.0f} GiB | "
          f"MHA {g_real * 8192 / 1024**3:.0f} GiB")
    print(f"[OK] step06 的 KVCache.nbytes() = {toy_cache.nbytes():,} B，"
          f"与公式（{l_toy_fp32:,} B/token x {MAX_SEQ} 位置）一致")

    # ---- 8. 结论 ----
    gap = g50 - l50      # 正数 = Llama 的 loss 更低
    print("\n--- 8. 结论（只基于上面的实测数字）---")
    print(f"参数量（实测）：GPT-2 {gpt2_total:,} → Llama {llama_total:,}"
          f"（{llama_total - gpt2_total:+,}，{llama_total / gpt2_total - 1:+.1%}）。")
    print(f"玩具规模 loss（最后 50 步均值）：GPT-2 {g50:.4f} → Llama {l50:.4f}"
          f"（{l50 - g50:+.4f}）；固定评测 batch：{g_eval:.4f} → {l_eval:.4f}。")
    print(f"每 token KV（公式）：toy fp32 {g_toy_fp32:,} B → {l_toy_fp32:,} B；"
          f"真实 8B 规模 fp16 512 KiB → 128 KiB（8K 上下文 4 GiB → 1 GiB）。")
    print(f"玩具速度：{g_med:.1f} → {l_med:.1f} ms/step（CPU，只代表这台机器）。")
    if gap > 0.05:
        print(f"判断：这次 Llama 赢了 {gap:.4f}（> 0.05），四个改造一起上是有效的。")
    elif gap > 0.01:
        print(f"判断：Llama 略好 {gap:.4f}，方向对但幅度小，离「显著」还差得远。")
    else:
        print(f"判断：Llama 没有明显赢（差 {gap:+.4f}），玩具规模上看不出架构优势。")
    print("两件事要分开：玩具 loss 受 438 字符/500 步限制不能外推；Llama 在真实规模的"
          "主要收益是 KV 显存与长上下文吞吐（上面 4 倍），不是这 0.0x 的 loss。")

    print("\n一句话记住：")
    print("  同预算下 Llama 赢在参数更少、KV 显存更小，玩具 loss 差距别当真；但玩具规模上")
    print("  training 反而略慢（GQA 的 repeat + SwiGLU 三矩阵固定开销），真实规模会被吞掉。")
    print("\n[OK] 全部检查通过")


if __name__ == "__main__":
    main()
