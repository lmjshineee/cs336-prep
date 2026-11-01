"""Step 5: 多头注意力 —— 把 C 维切成 H 份，每份独立做一次注意力。

动机：
    一个注意力头只能学一种「关注模式」（比如"看前一个词"或"看句子的主语"）。
    语言里同时存在很多种关系，所以并行开 H 个头，各学各的。

形状变化（最容易绕晕的地方，逐行看）：

    x        (B, T, C)
    qkv(x)   (B, T, 3C)         一次矩阵乘同时算出 q/k/v，比三次更省
    split    (B, T, C) x3
    view     (B, T, H, D)       C = H * D，把通道拆成「头」和「头内维度」
    transpose(B, H, T, D)       把 head 挪到 batch 后面，让每个头独立做矩阵乘
    scores   (B, H, T, T)       每个头一张 T x T 的相关性矩阵
    out      (B, H, T, D)
    transpose(B, T, H, D)
    view     (B, T, C)          拼回来
    out_proj (B, T, C)          ← 关键：让不同头的信息混合

为什么最后还要 out_proj？
    拼回来的 C 维只是 H 段互不相干的子空间。out_proj 是一个
    (C,C) 的线性层，负责把各头的结果重新组合。没有它，多头就只是
    把单头拆成 H 份再做平均，表达力不会提升。

参数量：qkv 3C² + out_proj C² = 4C²（和单头同维度时一样多）。
计算量：O(B * H * T² * D) = O(B * T² * C)，和 T 的平方成正比。
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
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

        q, k, v = self.qkv(x).split(self.n_embd, dim=2)   # 各 (B, T, C)

        # (B,T,C) -> (B,T,H,D) -> (B,H,T,D)
        q = q.view(B, T, H, D).transpose(1, 2)
        k = k.view(B, T, H, D).transpose(1, 2)
        v = v.view(B, T, H, D).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(D)  # (B,H,T,T)

        mask = torch.tril(torch.ones(T, T, dtype=torch.bool,
                                     device=x.device))
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = self.dropout(F.softmax(scores, dim=-1))

        out = weights @ v                                  # (B,H,T,D)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.dropout(self.out_proj(out))


if __name__ == "__main__":
    torch.manual_seed(0)
    tok = CharTokenizer(CORPUS)
    config = GPTConfig(vocab_size=tok.vocab_size, n_head=4, n_embd=64)
    embed = Embeddings(config)
    attn = MultiHeadAttention(config)

    ids = torch.tensor([tok.encode("cat sits")])   # (1, 9)
    x = embed(ids)
    print("x                  =", tuple(x.shape))

    B, T, C = x.shape
    H, D = attn.n_head, attn.head_dim
    print("n_head H           =", H)
    print("head_dim D = C / H =", D, f"(C={C})")

    q, k, v = attn.qkv(x).split(C, dim=2)
    print("qkv(x).split       = 3 x", tuple(q.shape))
    print("view+transpose     =", tuple(q.view(B, T, H, D).transpose(1, 2).shape),
          "  <- (B,H,T,D)")
    scores = (q.view(B, T, H, D).transpose(1, 2)
              @ k.view(B, T, H, D).transpose(1, 2).transpose(-2, -1))
    print("scores             =", tuple(scores.shape), "  <- (B,H,T,T)")

    y = attn(x)
    print("out                =", tuple(y.shape), "  <- (B,T,C)")

    # 因果性在多头下依然成立
    ids2 = ids.clone()
    ids2[:, -1] = (ids2[:, -1] + 1) % tok.vocab_size
    assert torch.allclose(attn(embed(ids))[:, :-1],
                          attn(embed(ids2))[:, :-1], atol=1e-6)
    print("\n[OK] 多头 + 因果掩码，未来 token 不影响过去")

    # 思考题：H 越大越好吗？给定 C，H 越大 D 越小，
    #        每个头能表达的子空间变窄，且小矩阵乘法的硬件效率下降。
