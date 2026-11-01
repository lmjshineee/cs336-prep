"""Step 4: 因果掩码 —— 每个 token 只能看自己和左边的 token。

上一版里，位置 0 的 token 能看到位置 5 的 token。
但语言模型的任务是「根据前缀预测下一个词」，
训练时如果能看到未来，就是抄答案，学不到东西。

做法：把 score 矩阵的上三角（j > i 的部分）设成 -inf。
    softmax 之后 exp(-inf) = 0，这些位置的权重就变成 0。

    mask = tril(ones(T, T))        下三角为 True
    例如 T=4：
        [[1, 0, 0, 0],
         [1, 1, 0, 0],
         [1, 1, 1, 0],
         [1, 1, 1, 1]]

实现细节：
    - 用 bool 张量做 masked_fill，比乘法更快，也不会在 -inf 处产生 NaN
    - 先 mask 再 softmax（顺序不能反）
    - 用 float("-inf") 而不是 -1e9：后者在 fp16 下仍可能算出一个很小的权重
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


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.scale = 1.0 / math.sqrt(config.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        scores = (q @ k.transpose(-2, -1)) * self.scale      # (B,T,T)

        # 下三角为 True：位置 i 只能被 j <= i 看到
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool,
                                     device=x.device))
        scores = scores.masked_fill(~mask, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        return weights @ v


if __name__ == "__main__":
    torch.manual_seed(0)
    tok = CharTokenizer(CORPUS)
    config = GPTConfig(vocab_size=tok.vocab_size, n_embd=8)
    embed = Embeddings(config)
    attn = CausalSelfAttention(config)

    ids = torch.tensor([tok.encode("cat sits")])   # (1, T)
    T = ids.size(1)
    x = embed(ids)

    mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
    print(f"T={T} 的掩码（1=可见）：")
    for row in mask.int().tolist():
        print("   ", " ".join(map(str, row)))

    scores = (attn.q_proj(x) @ attn.k_proj(x).transpose(-2, -1)) * attn.scale
    weights = F.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1)
    print("\n掩码后第 2 个 token 的权重（只应非零在前 3 个）：")
    print("   ", [round(w, 4) for w in weights[0, 2].tolist()])
    assert torch.all(weights[0].triu(1) == 0)
    print("[OK] 上三角全部为 0")

    # 关键验证：只改最后一个 token，前面位置的输出必须完全不变
    ids2 = ids.clone()
    ids2[:, -1] = (ids2[:, -1] + 1) % tok.vocab_size
    out_a = attn(embed(ids))
    out_b = attn(embed(ids2))
    assert torch.allclose(out_a[:, :-1], out_b[:, :-1], atol=1e-6)
    print("[OK] 修改未来 token，不影响过去位置的输出（因果性）")

    # 思考题：为什么 mask 是 (T,T) 而不是跟 batch 有关？
    #        （每个样本、每个 head 用的是同一张掩码，所以可以共享/缓存。）
