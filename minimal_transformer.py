"""Minimal decoder-only Transformer, implemented from scratch with PyTorch."""
from __future__ import annotations
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    """Causal multi-head self-attention."""
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.n_embd = config.n_embd
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, channels = x.shape
        q, k, v = self.qkv(x).split(self.n_embd, dim=2)
        shape = (batch, seq_len, self.n_head, self.head_dim)
        q = q.view(shape).transpose(1, 2)
        k = k.view(shape).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool,
                                     device=x.device))
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = self.attn_dropout(F.softmax(scores, dim=-1))
        out = weights @ v
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, channels)
        return self.resid_dropout(self.out_proj(out))

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
    """Pre-LN block: x = x + Attn(LN(x)); x = x + FFN(LN(x))."""
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ffn = FeedForward(config)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.ffn(self.ln2(x))

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
        self.lm_head.weight = self.token_embedding.weight  # weight tying
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
                targets: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch, seq_len = idx.shape
        if seq_len > self.config.block_size:
            raise ValueError(f"seq_len {seq_len} > block_size {self.config.block_size}")
        pos = torch.arange(seq_len, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(pos)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                targets.reshape(-1),
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int,
                 temperature: float = 0.8, top_k: int | None = 20
                 ) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.config.block_size:])
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = float("-inf")
            next_token = torch.multinomial(F.softmax(logits, dim=-1), 1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx

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

def get_batch(data: torch.Tensor, block_size: int, batch_size: int,
              device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    starts = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in starts])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in starts])
    return x.to(device), y.to(device)

def train_demo() -> None:
    torch.manual_seed(1337)
    device = torch.device("cpu")
    tokenizer = CharTokenizer(CORPUS)
    data = torch.tensor(tokenizer.encode(CORPUS), dtype=torch.long)
    config = GPTConfig(vocab_size=tokenizer.vocab_size, block_size=48,
                       n_layer=2, n_head=4, n_embd=64)
    model = MiniGPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    print(f"device={device} vocab={tokenizer.vocab_size} "
          f"params={sum(p.numel() for p in model.parameters()):,}")
    model.train()
    for step in range(1, 301):
        xb, yb = get_batch(data, config.block_size, 8, device)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            print(f"step {step:3d} | loss {loss.item():.4f}")
    idx = torch.tensor([tokenizer.encode("Spring arrives, ")])
    out = model.generate(idx, max_new_tokens=60, temperature=0.8, top_k=20)
    print("\n--- generated ---")
    print(tokenizer.decode(out[0].tolist()))
    model.eval()
    with torch.no_grad():
        x = torch.randint(0, tokenizer.vocab_size, (2, 16))
        logits, _ = model(x)
        assert logits.shape == (2, 16, tokenizer.vocab_size)
        a = torch.randint(0, tokenizer.vocab_size, (1, 12))
        b = a.clone()
        b[:, -1] = (b[:, -1] + 1) % tokenizer.vocab_size
        logits_a, _ = model(a)
        logits_b, _ = model(b)
        assert torch.allclose(logits_a[:, :-1], logits_b[:, :-1], atol=1e-6)
    print("\nshape + causality checks passed")

if __name__ == "__main__":
    train_demo()
