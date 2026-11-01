"""Step 6: Transformer Block —— 注意力 + 前馈网络 + 残差 + LayerNorm。

一个 block 做两件事，每件都包着「残差 + 归一化」：

    x = x + Attention(LayerNorm(x))
    x = x + FeedForward(LayerNorm(x))

三个设计点：

1) 残差连接 x + f(x)
   反向传播时梯度有一条「高速公路」直接流回去：
       dL/dx = dL/dout * (1 + df/dx)
   那项 1 保证即使 f 的梯度很小，信号也能穿过几十层。没有残差，
   深层网络基本训不动。

2) Pre-LN（先归一化再进子层）
   原始 Transformer 是 Post-LN：x = LayerNorm(x + f(x))。
   Pre-LN 训练更稳定，通常不需要 learning-rate warmup。
   现代模型（GPT-2/3、Llama）都用 Pre-LN，最后再补一个 ln_f。

3) LayerNorm
   对每个 token 自己的 C 维向量做归一化（减均值、除标准差），
   不是对 batch 做。所以 batch 大小、序列长度都不影响它。
   再乘可学习的 gamma、加 beta，保留表达能力。

前馈网络 FFN：C -> 4C -> C，中间 GELU。
   先升维再降维，相当于对每个位置独立做一次「思考」。
   参数量 2 * 4C² = 8C²，是注意力（4C²）的两倍 ——
   大模型里约 2/3 的参数都在 FFN 里。
"""

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

CORPUS = """\
Spring arrives, and the wind moves through the valley.
A small cat sits by the window and watches the clouds.
The sun goes down, the sky turns gold, and the birds return.
Night comes, the moon rises, and the stars shine above.
The cat closes its eyes and listens to the leaves.
A new day begins, and light fills the room again.
Spring arrives, and the wind moves through the valley.
A small cat sits by the window and watches the clouds.
"""


class CharTokenizer:
    def __init__(self, text: str) -> None:
        chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(chars)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 64
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 64
    dropout: float = 0.0


class Embeddings(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        tok = self.token_embedding(idx)
        pos = torch.arange(T, device=idx.device)
        return tok + self.position_embedding(pos)


class MultiHeadAttention(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.n_embd = config.n_embd
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        H, D = self.n_head, self.head_dim
        q, k, v = self.qkv(x).split(self.n_embd, dim=2)
        q = q.view(B, T, H, D).transpose(1, 2)
        k = k.view(B, T, H, D).transpose(1, 2)
        v = v.view(B, T, H, D).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / (D ** 0.5)
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool,
                                     device=x.device))
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = self.dropout(F.softmax(scores, dim=-1))
        out = (weights @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.dropout(self.out_proj(out))


class FeedForward(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-LN 结构：先归一化，再进子层，最后加残差。"""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ffn = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))   # 残差 1：token 之间通信
        x = x + self.ffn(self.ln2(x))    # 残差 2：每个 token 内部计算
        return x


if __name__ == "__main__":
    torch.manual_seed(0)
    tok = CharTokenizer(CORPUS)
    config = GPTConfig(vocab_size=tok.vocab_size, n_head=4, n_embd=64)
    embed = Embeddings(config)
    block = TransformerBlock(config)

    ids = torch.tensor([tok.encode("cat sits")])
    x = embed(ids)
    y = block(x)
    print("block 输入 =", tuple(x.shape))
    print("block 输出 =", tuple(y.shape), "  <- 形状不变，可以直接串联")

    def count(m: nn.Module) -> int:
        return sum(p.numel() for p in m.parameters())

    print(f"\nattention 参数 = {count(block.attn):,}   (~4*C^2 = {4*64*64:,})")
    print(f"ffn 参数       = {count(block.ffn):,}   (~8*C^2 = {8*64*64:,})")
    print(f"ln 参数        = {count(block.ln1) + count(block.ln2):,}")
    print(f"block 合计     = {count(block):,}")

    # 残差意味着「默认什么都不做」：把子层输出置零，输出应等于输入
    with torch.no_grad():
        block.attn.out_proj.weight.zero_()
        block.ffn.net[2].weight.zero_()
        block.ffn.net[2].bias.zero_()
        y0 = block(x)
    assert torch.allclose(y0, x, atol=1e-6)
    print("\n[OK] 子层输出为 0 时，block 退化为恒等映射（残差的作用）")

    # 思考题：如果把 LayerNorm 挪到残差之后（Post-LN），
    #        上面这个「恒等映射」的性质还在吗？
