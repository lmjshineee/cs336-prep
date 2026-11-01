"""Step 2: 词嵌入 + 位置嵌入 —— (B,T) 的 ID 变成 (B,T,C) 的向量。

记号约定（后面每一步都用）：
    B = batch size        一次喂几条序列
    T = sequence length   每条序列多少个 token（也叫 context length）
    C = n_embd            每个 token 的向量维度
    V = vocab_size        词表大小

两个嵌入都是「查表」：
    token_embedding:    (V, C)  第 i 行 = 第 i 个 token 的向量
    position_embedding: (T, C)  第 t 行 = 第 t 个位置的向量

为什么需要位置嵌入？
    注意力机制本质上是「加权求和」，它看到的是一堆 token，没有顺序感。
    "cat sat on mat" 和 "mat on sat cat" 在纯注意力眼里完全一样。
    所以必须显式把位置信息加进去。
"""

from dataclasses import dataclass

import torch
from torch import nn

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
    block_size: int = 64   # 最大上下文长度 = T 的上限
    n_embd: int = 64       # C


class Embeddings(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape

        tok = self.token_embedding(idx)              # (B, T) -> (B, T, C)
        pos = torch.arange(T, device=idx.device)     # [0, 1, ..., T-1]
        posv = self.position_embedding(pos)          # (T,)   -> (T, C)

        # (B,T,C) + (T,C)：广播，每个 batch 共享同一套位置向量
        return tok + posv


if __name__ == "__main__":
    torch.manual_seed(0)
    tok = CharTokenizer(CORPUS)
    model = Embeddings(GPTConfig(vocab_size=tok.vocab_size, n_embd=16))

    text = "A small cat"
    ids = torch.tensor([tok.encode(text)])  # (1, T)
    print("ids.shape        =", tuple(ids.shape), "  <- (B, T)")
    print("token_emb.weight =", tuple(model.token_embedding.weight.shape),
          "  <- (V, C)")
    print("pos_emb.weight   =", tuple(model.position_embedding.weight.shape),
          "  <- (block_size, C)")

    x = model(ids)
    print("x.shape          =", tuple(x.shape), "  <- (B, T, C)")

    # 手动验证：x[0, 2] 应该 = 第 2 个 token 的向量 + 第 2 个位置的向量
    manual = (model.token_embedding(ids)[0, 2]
              + model.position_embedding(torch.tensor(2)))
    assert torch.allclose(x[0, 2], manual)
    print("\n[OK] 嵌入 = token 向量 + 位置向量")

    # 思考题：block_size 设成 512，但输入 T=1000 会发生什么？
    #        （位置表只有 512 行，越界。）
