"""最小可用 Llama：一个文件看懂从 GPT-2 到 Llama 的全部改动。

和 `minimal_transformer.py`（GPT-2 版）对照着看，差异只有五处：

    GPT-2 写法                                  Llama 写法
    ------------------------------------------  ------------------------------------------
    nn.LayerNorm(C)                             RMSNorm(C)           # 不再减均值，去掉 β
    pos_emb(idx)  学习式绝对位置                  RoPE 转 q/k           # 0 参数，可外推
    FFN: C → 4C → C, GELU                       SwiGLU: C → 8/3C → C # 多一个门控矩阵
    K/V 头和 Q 头一样多                          GQA: K/V 头更少       # KV cache 小 g 倍
    Linear(bias=True)  + dropout 0.1            全部 bias=False，dropout 0.0

其余全部不变：Pre-Norm 残差结构、因果掩码、softmax(QKᵀ/√D)V、交叉熵、自回归生成。

这个文件是「压缩版」：把 `llama_steps/step01–06` 拆开讲的每一步合在一起，
跑一次就能训练 + 生成 + 验证 KV cache。

运行（CPU 十几秒）：
    .venv/bin/python minimal_llama.py
"""

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class LlamaConfig:
    vocab_size: int
    block_size: int = 64       # 最大序列长度
    n_layer: int = 2
    n_head: int = 4            # query 头数
    n_kv_head: int = 2         # key/value 头数（GQA）
    n_embd: int = 64
    ffn_hidden: int = 176      # 8/3 * 64 ≈ 171，向上取到 8 的倍数
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    tie_word_embeddings: bool = False


    def __post_init__(self) -> None:
        for name in ("vocab_size", "block_size", "n_layer", "n_head", "n_kv_head", "n_embd", "ffn_hidden"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} 必须是正整数")
        if self.n_embd % self.n_head:
            raise ValueError("n_embd 必须能被 n_head 整除")
        if self.n_head % self.n_kv_head:
            raise ValueError("n_head 必须是 n_kv_head 的倍数")
        if (self.n_embd // self.n_head) % 2:
            raise ValueError("RoPE 的 head_dim 必须为偶数")


class RMSNorm(nn.Module):
    """Llama 的归一化：只除以均方根，不减均值、没有 β、没有 bias。"""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 与 step06 一致：乘 gamma 后统一转回输入精度；主验证路径是 CPU fp32
        x32 = x.float()
        rms = torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x32 * rms * self.weight).to(x.dtype)


class SwiGLU(nn.Module):
    """门控 FFN：silu(gate(x)) 逐元素决定 up(x) 的哪些通道通过。"""

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        C, F = config.n_embd, config.ffn_hidden
        self.gate_proj = nn.Linear(C, F, bias=False)
        self.up_proj = nn.Linear(C, F, bias=False)
        self.down_proj = nn.Linear(F, C, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """前一半和后一半配对：(x1, x2) → (-x2, x1)。"""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def build_rope_cache(head_dim: int, max_len: int, theta: float = 10000.0,
                     device: torch.device | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """预先算好每个位置、每个通道对的 cos/sin，形状都是 (max_len, head_dim)。"""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    freqs = torch.outer(torch.arange(max_len, dtype=torch.float32), inv_freq)   # (T, D/2)
    emb = torch.cat([freqs, freqs], dim=-1)                                     # (T, D)
    return emb.cos().to(device), emb.sin().to(device)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """旋转 q/k；固定内容向量的旋转点积取决于相对位置，真实打分还取决于内容。"""
    cos, sin = cos[None, None], sin[None, None]        # (1,1,T,D)
    return x * cos + rotate_half(x) * sin


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """(B, n_kv_head, T, D) → (B, n_head, T, D)，交错复制给共享它的 Q 头。"""
    return x if n_rep == 1 else x.repeat_interleave(n_rep, dim=1)


class Attention(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int = 0) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_rep = config.n_head // config.n_kv_head
        self.head_dim = config.n_embd // config.n_head
        self.layer_idx = layer_idx
        C = config.n_embd
        self.q_proj = nn.Linear(C, self.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(C, self.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(C, self.n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_head * self.head_dim, C, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                kv_cache: "KVCache | None" = None, start_pos: int = 0) -> torch.Tensor:
        B, T, _ = x.shape
        H, Kv, D = self.n_head, self.n_kv_head, self.head_dim
        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)      # (B,H,T,D)
        k = self.k_proj(x).view(B, T, Kv, D).transpose(1, 2)     # (B,Kv,T,D)
        v = self.v_proj(x).view(B, T, Kv, D).transpose(1, 2)     # (B,Kv,T,D)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)  # 只转 q 和 k，v 不动

        if kv_cache is not None:                                  # decode：把新 k/v 写进缓存
            k, v = kv_cache.append(self.layer_idx, k, v, start_pos)
        k = repeat_kv(k, self.n_rep)                              # 复制成 H 份给每个 Q 头
        v = repeat_kv(v, self.n_rep)

        scores = q @ k.transpose(-2, -1) / math.sqrt(D)           # (B,H,T,total)
        if T > 1:                                                 # 掩掉未来位置
            total = scores.size(-1)
            mask = torch.triu(torch.ones(T, total, dtype=torch.bool, device=x.device),
                              diagonal=start_pos + 1)
            scores = scores.masked_fill(mask, float("-inf"))
        out = F.softmax(scores, dim=-1) @ v                       # (B,H,T,D)
        return self.o_proj(out.transpose(1, 2).reshape(B, T, H * D))


class LlamaBlock(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int = 0) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.n_embd, config.norm_eps)
        self.attn = Attention(config, layer_idx)
        self.ffn_norm = RMSNorm(config.n_embd, config.norm_eps)
        self.ffn = SwiGLU(config)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                kv_cache: "KVCache | None" = None, start_pos: int = 0) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin, kv_cache, start_pos)  # 残差 + Pre-Norm
        return x + self.ffn(self.ffn_norm(x))


class KVCache:
    """预分配 (B, n_kv_head, block_size, head_dim) 的缓冲，decode 时逐位追加。"""

    def __init__(self, config: LlamaConfig, batch_size: int, device: torch.device,
                 dtype: torch.dtype = torch.float32) -> None:
        head_dim = config.n_embd // config.n_head
        shape = (batch_size, config.n_kv_head, config.block_size, head_dim)
        self.config = config
        self.k = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(config.n_layer)]
        self.v = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(config.n_layer)]

    def append(self, layer_idx: int, k_new: torch.Tensor, v_new: torch.Tensor,
               start_pos: int) -> tuple[torch.Tensor, torch.Tensor]:
        end = start_pos + k_new.size(2)
        self.k[layer_idx][:, :, start_pos:end] = k_new
        self.v[layer_idx][:, :, start_pos:end] = v_new
        return self.k[layer_idx][:, :, :end], self.v[layer_idx][:, :, :end]

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.k + self.v)


class MiniLlama(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)  # 唯一的嵌入
        self.blocks = nn.ModuleList([LlamaBlock(config, i) for i in range(config.n_layer)])
        self.norm = RMSNorm(config.n_embd, config.norm_eps)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        head_dim = config.n_embd // config.n_head
        cos, sin = build_rope_cache(head_dim, config.block_size, config.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init)

    def _init(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, kv_cache: KVCache | None = None,
                start_pos: int = 0) -> torch.Tensor:
        B, T = idx.shape
        x = self.token_embedding(idx)                                # (B,T,C)，没有位置嵌入
        cos = self.rope_cos[start_pos:start_pos + T].to(idx.device)
        sin = self.rope_sin[start_pos:start_pos + T].to(idx.device)
        for block in self.blocks:
            x = block(x, cos, sin, kv_cache, start_pos)
        return self.lm_head(self.norm(x))                            # (B,T,V)

    def loss(self, idx: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = self(idx)
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int,
                 temperature: float = 0.8, top_k: int = 20) -> torch.Tensor:
        """prefill 一次，之后每步只喂 1 个 token（KV cache）。"""
        self.eval()
        cache = KVCache(self.config, idx.size(0), idx.device)
        logits = self(idx, kv_cache=cache, start_pos=0)
        pos = idx.size(1)
        for _ in range(max_new_tokens):
            step_logits = logits[:, -1, :] / temperature
            values, _ = torch.topk(step_logits, min(top_k, step_logits.size(-1)))
            step_logits[step_logits < values[:, [-1]]] = float("-inf")
            nxt = torch.multinomial(F.softmax(step_logits, dim=-1), 1)
            idx = torch.cat((idx, nxt), dim=1)
            if pos >= self.config.block_size:
                break
            logits = self(nxt, kv_cache=cache, start_pos=pos)
            pos += 1
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
    config = LlamaConfig(vocab_size=tokenizer.vocab_size, block_size=64,
                         n_layer=2, n_kv_head=2)
    model = MiniLlama(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, betas=(0.9, 0.95),
                                  weight_decay=0.1)
    print(f"device={device} vocab={tokenizer.vocab_size} "
          f"params={sum(p.numel() for p in model.parameters()):,}")

    model.train()
    for step in range(1, 301):
        xb, yb = get_batch(data, config.block_size, 8, device)
        loss = model.loss(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 50 == 0:
            print(f"step {step:3d} | loss {loss.item():.4f}")

    idx = torch.tensor([tokenizer.encode("Spring arrives, ")])
    out = model.generate(idx, max_new_tokens=60, temperature=0.8, top_k=20)
    print("\n--- generated (KV cache) ---")
    print(tokenizer.decode(out[0].tolist()))
    print("（生成停止于当前实现的上下文边界：RoPE 表与 KV cache 都按 block_size 预分配。"
          "\n  RoPE 公式可计算更远位置，但本实现的表和缓存有限；这不证明长上下文质量。）")

    # 自检 1：KV cache 与全量重算等价
    model.eval()
    with torch.no_grad():
        probe = torch.randint(0, tokenizer.vocab_size, (1, 16))
        full = model(probe)
        cache = KVCache(config, 1, device)
        cached = model(probe[:, :6], kv_cache=cache, start_pos=0)
        worst = (cached - full[:, :6]).abs().max().item()
        for t in range(6, 16):
            step = model(probe[:, t:t + 1], kv_cache=cache, start_pos=t)
            worst = max(worst, (step[:, -1] - full[:, t]).abs().max().item())
        assert worst < 1e-5, worst
        print(f"\n[OK] KV cache 与全量重算一致：max|diff| = {worst:.2e}")
        print(f"[OK] KV cache 占用 = {cache.nbytes() / 1024:.1f} KiB "
              f"（2 × n_layer × B × T_max × n_kv_head × head_dim × 4B）")

    # 自检 2：因果性
    with torch.no_grad():
        a = torch.randint(0, tokenizer.vocab_size, (1, 12))
        b = a.clone()
        b[:, -1] = (b[:, -1] + 1) % tokenizer.vocab_size
        assert torch.allclose(model(a)[:, :-1], model(b)[:, :-1], atol=1e-6)
    print("[OK] 因果性：改最后一个 token 不影响前面的输出")


if __name__ == "__main__":
    train_demo()
