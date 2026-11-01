"""Step 7: 完整模型 —— 嵌入 + N 个 block + 输出层。

前向流程：

    idx            (B, T)          整数 ID
    embeddings     (B, T, C)       词向量 + 位置向量
    N x block      (B, T, C)       每一层都在「交换信息 + 独立计算」
    ln_f           (B, T, C)       最后一层归一化
    lm_head        (B, T, V)       每个位置输出整个词表的分数（logits）
    cross_entropy  scalar          和右移一位的 targets 算损失

两个值得注意的点：

1) lm_head 为什么不加 bias、并且和 token_embedding 共享权重（weight tying）
   输入嵌入是 V x C 的查表，输出投影是 C x V 的矩阵。
   它们描述的是同一件事：「第 i 个 token 对应哪个向量」。
   共享之后省下 V*C 个参数，小模型效果通常更好（GPT-2 就这么做）。

2) 为什么最后一层要 ln_f
   Pre-LN 结构里，残差会把未经归一化的量一路加到底，
   所以输出前要统一归一化一次，否则 logits 的尺度不可控。

参数量估算（忽略 LN 和 bias）：
    embeddings  V*C (共享) + T*C
    each block  4C² (attn) + 8C² (ffn) = 12C²
    total       V*C + T*C + N*12C²
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

    def forward(self, x: torch.Tensor,
                k_cache: torch.Tensor | None = None,
                v_cache: torch.Tensor | None = None,
                start_pos: int = 0) -> torch.Tensor:
        """k_cache/v_cache/start_pos 只在推理时用；训练时全传默认值即可。"""
        B, T, C = x.shape
        H, D = self.n_head, self.head_dim

        q, k, v = self.qkv(x).split(self.n_embd, dim=2)   # 各 (B,T,C)
        q = q.view(B, T, H, D).transpose(1, 2)            # (B,H,T,D)
        k = k.view(B, T, H, D).transpose(1, 2)
        v = v.view(B, T, H, D).transpose(1, 2)

        if k_cache is None:
            # 训练 / 全量重算：所有 token 一起进来
            k_all, v_all = k, v
            total = T
        else:
            # 推理：把这次新算的 k/v 写进缓存，再对「全部历史 + 自己」做注意力
            assert v_cache is not None, "k_cache 和 v_cache 必须成对传入"
            total = start_pos + T
            k_cache[:, :, start_pos:total] = k
            v_cache[:, :, start_pos:total] = v
            k_all = k_cache[:, :, :total]
            v_all = v_cache[:, :, :total]

        # 通用因果掩码：query 在 start_pos..start_pos+T-1，
        # key 在 0..total-1，只允许 key_pos <= query_pos。
        # T=1 时这张掩码全是 True（历史 token 都在左边），所以不花什么钱。
        q_pos = torch.arange(start_pos, total, device=x.device).unsqueeze(1)
        k_pos = torch.arange(total, device=x.device).unsqueeze(0)
        mask = k_pos <= q_pos                              # (T, total)

        scores = (q @ k_all.transpose(-2, -1)) / (D ** 0.5)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = self.dropout(F.softmax(scores, dim=-1))
        out = (weights @ v_all).transpose(1, 2).contiguous().view(B, T, C)
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
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ffn = FeedForward(config)

    def forward(self, x: torch.Tensor,
                k_cache: torch.Tensor | None = None,
                v_cache: torch.Tensor | None = None,
                start_pos: int = 0) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), k_cache, v_cache, start_pos)
        return x + self.ffn(self.ln2(x))


class KVCache:
    """每一层一份 K、V，形状固定为 (B, H, block_size, D)。

    为什么可以固定大小：位置表最多 block_size 个位置，
    所以任何时刻缓存的长度都不会超过 block_size。
    预分配的好处是不用每步 torch.cat，避免反复分配显存。
    """

    def __init__(self, config: GPTConfig, batch_size: int,
                 device: torch.device, dtype: torch.dtype = torch.float32):
        H = config.n_head
        D = config.n_embd // H
        shape = (batch_size, H, config.block_size, D)
        self.k = [torch.zeros(shape, device=device, dtype=dtype)
                  for _ in range(config.n_layer)]
        self.v = [torch.zeros(shape, device=device, dtype=dtype)
                  for _ in range(config.n_layer)]

    def memory_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.k + self.v)


class MiniGPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layer)]
        )
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # 权重共享：输出投影和输入嵌入用同一张表
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor,
                targets: torch.Tensor | None = None,
                kv_cache: KVCache | None = None,
                start_pos: int = 0
                ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """推理时传 kv_cache + start_pos，就只处理新 token。"""
        T = idx.shape[1]
        if start_pos + T > self.config.block_size:
            raise ValueError(
                f"start_pos {start_pos} + seq_len {T} "
                f"> block_size {self.config.block_size}")

        # 位置用「绝对位置」：缓存模式下新 token 的位置不是 0
        pos = torch.arange(start_pos, start_pos + T, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(pos)
        x = self.dropout(x)

        for i, block in enumerate(self.blocks):
            if kv_cache is None:
                x = block(x)
            else:
                x = block(x, kv_cache.k[i], kv_cache.v[i], start_pos)

        logits = self.lm_head(self.ln_f(x))      # (B, T, V)

        loss = None
        if targets is not None:
            # 展平成 (B*T, V) 和 (B*T,)，一次算所有位置的交叉熵
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   targets.reshape(-1))
        return logits, loss


if __name__ == "__main__":
    torch.manual_seed(1337)
    tok = CharTokenizer(CORPUS)
    config = GPTConfig(vocab_size=tok.vocab_size, block_size=48,
                       n_layer=2, n_head=4, n_embd=64)
    model = MiniGPT(config)

    total = sum(p.numel() for p in model.parameters())
    print(f"vocab_size = {config.vocab_size}")
    print(f"params     = {total:,}")
    print("  embeddings+head =",
          f"{model.token_embedding.weight.numel() + model.position_embedding.weight.numel():,}")
    print("  blocks          =",
          f"{sum(p.numel() for p in model.blocks.parameters()):,}")
    print("  ln_f            =", f"{sum(p.numel() for p in model.ln_f.parameters()):,}")

    ids = torch.tensor([tok.encode("A small cat")])
    targets = torch.tensor([tok.encode(" small cat ")])
    logits, loss = model(ids, targets)
    print("\nidx.shape     =", tuple(ids.shape), "  <- (B, T)")
    print("logits.shape  =", tuple(logits.shape), "  <- (B, T, V)")
    print("loss          =", f"{loss.item():.4f}",
          f"  (随机初始化时约等于 ln(V) = {torch.log(torch.tensor(float(config.vocab_size))).item():.4f})")

    assert logits.shape == (1, ids.size(1), config.vocab_size)
    assert model.lm_head.weight is model.token_embedding.weight
    print("\n[OK] 输出形状正确，且 lm_head 与 token_embedding 共享权重")

    # 思考题：如果把 targets 换成 ids 本身会怎样？
    #        （模型学会「复制输入」，loss 能降到接近 0，但没有泛化意义。）
