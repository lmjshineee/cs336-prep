"""Step 3: 单头自注意力 —— token 之间开始交换信息。

核心公式（缩放点积注意力）：

    Attention(Q, K, V) = softmax(Q K^T / sqrt(D)) V

三个投影的含义（用「查字典」类比）：
    Q (query)  "我在找什么样的信息"
    K (key)    "我身上有什么标签，可以被别人匹配"
    V (value)  "如果真的被选中，我实际提供的内容"

计算过程：
    scores = Q K^T           (B,T,C) @ (B,C,T) -> (B,T,T)
             scores[i,j] = 第 i 个 token 的 query 和第 j 个 token 的 key 的点积
    weights = softmax(scores) 每一行归一化，表示「i 应该从 j 那里取多少」
    output  = weights @ V     每个位置得到所有位置 value 的加权平均

为什么除以 sqrt(D)？
    假设 q、k 各维独立、均值 0 方差 1，点积的方差约为 D。
    D 越大，点积绝对值越大 -> softmax 越尖锐（接近 one-hot）
    -> 梯度趋近于 0，训练不动。除以 sqrt(D) 把方差拉回 1。

注意：这一步还没有因果掩码，每个 token 能看到「未来」。
"""

import math
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
    n_embd: int = 64


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


class SingleHeadAttention(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.scale = 1.0 / math.sqrt(config.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        q = self.q_proj(x)                    # (B, T, C)
        k = self.k_proj(x)                    # (B, T, C)
        v = self.v_proj(x)                    # (B, T, C)

        # (B,T,C) @ (B,C,T) -> (B,T,T)
        scores = (q @ k.transpose(-2, -1)) * self.scale
        weights = F.softmax(scores, dim=-1)   # 沿最后一维（key 方向）归一化
        out = weights @ v                     # (B,T,T) @ (B,T,C) -> (B,T,C)
        return out


if __name__ == "__main__":
    torch.manual_seed(0)
    tok = CharTokenizer(CORPUS)
    config = GPTConfig(vocab_size=tok.vocab_size, n_embd=8)
    embed = Embeddings(config)
    attn = SingleHeadAttention(config)

    text = "cat sits"
    ids = torch.tensor([tok.encode(text)])   # (1, T)
    x = embed(ids)
    print("x.shape       =", tuple(x.shape), "  <- (B, T, C)")

    q, k, v = attn.q_proj(x), attn.k_proj(x), attn.v_proj(x)
    scores = (q @ k.transpose(-2, -1)) * attn.scale
    weights = F.softmax(scores, dim=-1)

    print("scores.shape  =", tuple(scores.shape), "  <- (B, T, T)")
    print("weights.shape =", tuple(weights.shape), "  <- (B, T, T)")
    print("weights 每行和 =", weights.sum(-1).tolist()[0][:3], "...")
    assert torch.allclose(weights.sum(-1),
                          torch.ones(1, weights.size(1)), atol=1e-5)

    y = attn(x)
    print("out.shape     =", tuple(y.shape), "  <- (B, T, C)，和输入同形")

    # 观察：这里每个 token 都能看到全部 T 个 token（还没有因果掩码）。
    # 打印第一行权重，看看 "c" 把注意力分给了谁
    print("\n第 0 个 token 的注意力权重：")
    for j, w in enumerate(weights[0, 0].tolist()):
        print(f"  -> token {tok.itos[ids[0, j].item()]!r:>5}  {w:.4f}")

    # 思考题：如果所有 token 的 x 完全相同，attention 输出会怎样？
    #        （权重变成均匀 1/T，输出 = V 的均值。）
