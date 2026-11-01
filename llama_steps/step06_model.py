"""Step 6: 完整 MiniLlama + KV cache —— 把三个改造拼成一个能生成文字的模型。

GPT-2 的做法（前面 step07_model.py 的前向）：
    tok_emb(idx) + pos_emb(0..T-1)  ->  N × block  ->  ln_f  ->  lm_head
    位置只存在于 embedding 里，注意力本身不知道 token 排在第几位。

Llama 的做法：
    tok_emb(idx)  ->  N × block(cos, sin)  ->  norm  ->  lm_head
    位置由 RoPE 直接注入 q/k：cos/sin 从缓冲区按 start_pos 切片。

为什么要改：
    1) 没有位置嵌入：整模型没有 position_embedding，所以不随长度加参数，
       也不存在「位置表用完了」这种失败模式。
    2) 没有 bias：所有 Linear 都是 bias=False。去 bias 是 PaLM/Llama 的经验选择：
       质量基本不降，参数少一点，也更容易把 Linear 和归一化融进一个 kernel。
    3) 没有权重共享（默认）：Llama 的 lm_head 和 token_embedding 是两张表；
       GPT-2 共享（省 V·C 个参数），SmolLM2-135M 这类小模型会重新打开共享。

KV cache：
    prefill：prompt 一次算完，每层的 K/V 写进预分配缓存 (B, n_kv_head, T_max, D)。
    decode ：每步只喂上一个新 token（T=1），start_pos = 已生成 token 数；
        RoPE 按 start_pos 取 cos/sin，历史 K/V 从缓存里读，不再重算。
    缓存大小 = 2 · n_layer · B · T_max · n_kv_head · D · sizeof(dtype)。

关键公式：
    loss（随机初始化）≈ ln(vocab_size)，因为此时模型输出接近均匀分布
    缓存字节数 = 2 · n_layer · B · T_max · n_kv_head · D · 4B（fp32）

和上一步的 diff（只列变化的行）：
    - self.ln_f = nn.LayerNorm(C)                   + self.norm = RMSNorm(C)
    - self.position_embedding = nn.Embedding(...)   + 没有它；位置来自 rope_cos/sin
    - forward(...) -> (logits, loss)                + forward(...) -> logits
    + 每层 block 都要传 cos、sin、kv_cache、start_pos
"""

import math
import unicodedata
from dataclasses import dataclass, replace

import torch
from torch import nn
import torch.nn.functional as F


def _width(text: str) -> int:
    """终端里中文占 2 列，按显示宽度算才能对齐。"""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


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
# 本步用到的「当前进度」代码：从前面步骤逐字复制过来，可以扫一眼跳过
# ---------------------------------------------------------------------------

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
class LlamaConfig:
    vocab_size: int
    block_size: int = 64          # 最大序列长度（= 训练长度上限）
    n_layer: int = 2
    n_head: int = 4               # query 头数
    n_kv_head: int = 2            # key/value 头数，n_head % n_kv_head == 0
    n_embd: int = 64              # C
    ffn_hidden: int = 176         # 8/3 * C 向上取整到 8 的倍数
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    dropout: float = 0.0
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
    """x / RMS(x) * γ，其中 RMS(x) = √(mean(x²) + eps)；float32 里算，再转回原 dtype。"""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(..., C) -> (..., C)，只沿最后一维归一化。"""
        dtype = x.dtype
        x32 = x.float()
        rms = torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        # 先乘 γ 再统一转回：fp32 下逐位不变，bf16/fp16 输入也能得到原 dtype
        return (x32 * rms * self.weight).to(dtype)


class SwiGLU(nn.Module):
    """down(SiLU(gate(x)) * up(x))：SiLU(gate) 逐元素决定 up 的哪些维度通过。"""

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.n_embd, config.ffn_hidden, bias=False)
        self.up_proj = nn.Linear(config.n_embd, config.ffn_hidden, bias=False)
        self.down_proj = nn.Linear(config.ffn_hidden, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, C) -> (B, T, C)。"""
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """(..., D) -> (..., D)：后一半取负挪到前面，实现「配对旋转」。"""
    d = x.size(-1) // 2
    return torch.cat((-x[..., d:], x[..., :d]), dim=-1)


def build_rope_cache(head_dim: int, max_len: int, theta: float = 10000.0,
                     device: torch.device | None = None
                     ) -> tuple[torch.Tensor, torch.Tensor]:
    """返回 (cos, sin)，形状都是 (max_len, head_dim)，float32。

    half-split 写法：通道 i 和 i+D/2 配成一对，共用一个角频率
    θ_i = theta^(-2i/D)，位置 m 上的旋转角是 m·θ_i。
    """
    i = torch.arange(0, head_dim, 2, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (theta ** (i / head_dim))                      # (D/2,)
    angles = torch.arange(max_len, device=device, dtype=torch.float32)[:, None]
    angles = angles * inv_freq[None, :]                             # (max_len, D/2)
    emb = torch.cat((angles, angles), dim=-1)                       # (max_len, D)
    return emb.cos(), emb.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor,
               sin: torch.Tensor) -> torch.Tensor:
    """x: (B, H, T, D)；cos/sin: (T, D)。输出同形状：只旋转，不换位置。"""
    cos = cos.unsqueeze(0).unsqueeze(0).to(x.dtype)                 # (1,1,T,D)
    sin = sin.unsqueeze(0).unsqueeze(0).to(x.dtype)
    return x * cos + rotate_half(x) * sin


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """(B, Hkv, T, D) -> (B, Hkv*n_rep, T, D)：每个 KV 头原地复制 n_rep 份。"""
    if n_rep == 1:
        return x
    return x.repeat_interleave(n_rep, dim=1)


class KVCache:
    """推理时的 KV 缓存：每层预分配 (B, n_kv_head, block_size, head_dim)。

    为什么可以一次分配好：模型只处理前 block_size 个绝对位置，缓存不会更长。
    存进去的是「已经旋转过」的 k，所以 decode 时不用重算历史，只按 start_pos 切片。
    """

    def __init__(self, config: LlamaConfig, batch_size: int,
                 device: torch.device,
                 dtype: torch.dtype = torch.float32) -> None:
        self.n_layer = config.n_layer
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.n_embd // config.n_head
        self.block_size = config.block_size
        shape = (batch_size, self.n_kv_head, config.block_size, self.head_dim)
        self.k = [torch.zeros(shape, device=device, dtype=dtype)
                  for _ in range(config.n_layer)]
        self.v = [torch.zeros(shape, device=device, dtype=dtype)
                  for _ in range(config.n_layer)]

    def append(self, layer_idx: int, k_new: torch.Tensor, v_new: torch.Tensor,
               start_pos: int) -> tuple[torch.Tensor, torch.Tensor]:
        """把新 token 的 k/v 写进缓存，返回 0..start_pos+T 的全部 k/v。"""
        T = k_new.size(2)
        end = start_pos + T
        if end > self.block_size:
            raise ValueError(
                f"KV cache 装不下 start_pos={start_pos} + T={T} = {end} > "
                f"block_size={self.block_size}")
        self.k[layer_idx][:, :, start_pos:end] = k_new
        self.v[layer_idx][:, :, start_pos:end] = v_new
        return self.k[layer_idx][:, :, :end], self.v[layer_idx][:, :, :end]

    def nbytes(self) -> int:
        """所有层 K、V 缓冲加起来占多少字节。"""
        return sum(t.numel() * t.element_size() for t in self.k + self.v)


class Attention(nn.Module):
    """GQA 注意力：Q 有 n_head 个头，K/V 只有 n_kv_head 个，多出的 Q 头共享 K/V。"""

    def __init__(self, config: LlamaConfig, layer_idx: int = 0) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0, "C 必须能被 n_head 整除"
        assert config.n_head % config.n_kv_head == 0, "n_head 必须是 n_kv_head 的倍数"
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_rep = config.n_head // config.n_kv_head
        self.head_dim = config.n_embd // config.n_head
        self.n_embd = config.n_embd
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        kv_dim = config.n_kv_head * self.head_dim
        self.k_proj = nn.Linear(config.n_embd, kv_dim, bias=False)
        self.v_proj = nn.Linear(config.n_embd, kv_dim, bias=False)
        self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                kv_cache: KVCache | None = None,
                start_pos: int = 0) -> torch.Tensor:
        """x: (B,T,C) -> (B,T,C)。cos/sin 对应当前这 T 个 token 的绝对位置。"""
        B, T, C = x.shape
        H, Hkv, D = self.n_head, self.n_kv_head, self.head_dim

        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)         # (B,H,T,D)
        k = self.k_proj(x).view(B, T, Hkv, D).transpose(1, 2)       # (B,Hkv,T,D)
        v = self.v_proj(x).view(B, T, Hkv, D).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)   # 只旋转 q、k，v 不转：位置只影响打分

        if kv_cache is not None:
            # 把这步新算的 k/v 写进缓存，拿回 0..start_pos+T 的全部历史
            k, v = kv_cache.append(self.layer_idx, k, v, start_pos)

        total = k.size(2)                         # 参与打分的 key 个数
        k = repeat_kv(k, self.n_rep)              # (B,H,total,D)，复制后才对齐 q
        v = repeat_kv(v, self.n_rep)

        scores = q @ k.transpose(-2, -1) / math.sqrt(D)             # (B,H,T,total)
        # 因果掩码：query 的绝对位置是 start_pos..start_pos+T-1，key 从 0 开始
        q_pos = torch.arange(start_pos, start_pos + T, device=x.device)[:, None]
        k_pos = torch.arange(total, device=x.device)[None, :]
        scores = scores.masked_fill(k_pos > q_pos, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        out = (weights @ v).transpose(1, 2).reshape(B, T, C)
        return self.o_proj(out)


class LlamaBlock(nn.Module):
    """Pre-Norm 残差块：先归一化，再进子层，最后加回残差。"""

    def __init__(self, config: LlamaConfig, layer_idx: int = 0) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.n_embd, config.norm_eps)
        self.attn = Attention(config, layer_idx)
        self.ffn_norm = RMSNorm(config.n_embd, config.norm_eps)
        self.ffn = SwiGLU(config)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                kv_cache: KVCache | None = None,
                start_pos: int = 0) -> torch.Tensor:
        """x: (B,T,C) -> (B,T,C)，输入输出同形。"""
        x = x + self.attn(self.attn_norm(x), cos, sin, kv_cache, start_pos)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class MiniLlama(nn.Module):
    """token_embedding -> N × LlamaBlock(cos, sin) -> norm -> lm_head。"""

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList(
            [LlamaBlock(config, layer_idx=i) for i in range(config.n_layer)]
        )
        self.norm = RMSNorm(config.n_embd, config.norm_eps)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # 位置不是参数，而是一张所有层共享的 cos/sin 表；
        # persistent=False：不进 state_dict，换 block_size 时自动重建。
        head_dim = config.n_embd // config.n_head
        rope_cos, rope_sin = build_rope_cache(head_dim, config.block_size,
                                              config.rope_theta)
        self.register_buffer("rope_cos", rope_cos, persistent=False)
        self.register_buffer("rope_sin", rope_sin, persistent=False)

        self.apply(self._init_weights)
        if config.tie_word_embeddings:
            # 共享：lm_head 直接用 embedding 那张表（GPT-2 / SmolLM2 的做法）
            self.lm_head.weight = self.token_embedding.weight

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, kv_cache: KVCache | None = None,
                start_pos: int = 0) -> torch.Tensor:
        """idx: (B,T) -> logits (B,T,V)。传 kv_cache 时 T 通常为 1（decode）。"""
        T = idx.shape[1]
        if start_pos + T > self.config.block_size:
            raise ValueError(
                f"序列超长：start_pos {start_pos} + T {T} = {start_pos + T} > "
                f"block_size {self.config.block_size}（RoPE 表只建到 block_size）")

        x = self.token_embedding(idx)                        # (B,T,C)，无位置嵌入
        cos = self.rope_cos[start_pos:start_pos + T]         # (T,D) 按绝对位置切片
        sin = self.rope_sin[start_pos:start_pos + T]
        for block in self.blocks:
            x = block(x, cos, sin, kv_cache, start_pos)
        return self.lm_head(self.norm(x))                    # (B,T,V)


def count(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


@torch.no_grad()
def generate_naive(model: MiniLlama, idx: torch.Tensor,
                   max_new_tokens: int) -> torch.Tensor:
    """没有缓存：每生成一个 token，都把整段前缀重新喂一遍。"""
    model.eval()
    for _ in range(max_new_tokens):
        logits = model(idx[:, -model.config.block_size:])
        nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)    # 贪心
        idx = torch.cat((idx, nxt), dim=1)
    return idx


@torch.no_grad()
def generate_cached(model: MiniLlama, idx: torch.Tensor,
                    max_new_tokens: int) -> torch.Tensor:
    """KV cache：prompt 走一次 prefill，之后每步只喂 1 个新 token。"""
    model.eval()
    config = model.config
    if idx.size(1) + max_new_tokens > config.block_size:
        raise ValueError("prompt + 新 token 超过 block_size，缓存版不能滑动窗口")
    cache = KVCache(config, idx.size(0), idx.device,
                    dtype=model.token_embedding.weight.dtype)

    logits = model(idx, kv_cache=cache, start_pos=0)           # prefill
    pos = idx.size(1)
    for _ in range(max_new_tokens):
        nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        idx = torch.cat((idx, nxt), dim=1)
        if pos >= config.block_size:
            break
        logits = model(nxt, kv_cache=cache, start_pos=pos)     # decode：T=1
        pos += 1
    return idx


@torch.no_grad()
def check_kv_cache(model: MiniLlama, prompt: torch.Tensor,
                   n_steps: int = 8) -> list[float]:
    """逐 token 对比「全量重算」和「KV cache」的 logits，返回每步的 max|diff|。"""
    model.eval()
    cache = KVCache(model.config, prompt.size(0), prompt.device,
                    dtype=model.token_embedding.weight.dtype)

    full = model(prompt)
    cached = model(prompt, kv_cache=cache, start_pos=0)       # prefill
    diffs = [(full - cached).abs().max().item()]

    seq = prompt
    nxt = full[:, -1, :].argmax(dim=-1, keepdim=True)         # 贪心，去掉采样随机性
    for _ in range(n_steps):
        seq = torch.cat((seq, nxt), dim=1)
        full = model(seq)                                     # 整段重算
        cached = model(nxt, kv_cache=cache, start_pos=seq.size(1) - 1)
        diffs.append((full[:, -1] - cached[:, -1]).abs().max().item())
        nxt = full[:, -1, :].argmax(dim=-1, keepdim=True)
    return diffs


def main() -> None:
    torch.manual_seed(0)
    device = torch.device("cpu")
    tok = CharTokenizer(CORPUS)
    config = LlamaConfig(vocab_size=tok.vocab_size)
    model = MiniLlama(config).to(device)
    model.eval()
    C, V = config.n_embd, config.vocab_size
    D = C // config.n_head

    print("\n--- 1. 结构：没有位置嵌入 ---")
    show("vocab_size", V)
    show("block_size", config.block_size, "RoPE 表的最大长度")
    show("head_dim", D, "= C / n_head")
    show("rope_cos", model.rope_cos, "(block_size, head_dim)，不进 checkpoint")
    show("位置嵌入参数", 0, "整个模型一个都没有")
    assert "rope_cos" not in model.state_dict()
    assert "rope_sin" not in model.state_dict()
    assert not any("position" in n for n, _ in model.named_parameters())

    # 锁死 RoPE 表的定义：half-split + cat(freqs, freqs)，θ_i = theta^(-2i/D)。
    # 换 theta 或改成 interleaved 排列，step08 和 HuggingFace 就对不上了。
    inv_freq = [config.rope_theta ** (-2 * k / D) for k in range(D // 2)]
    rows = [0, 1, 37]
    cos_ref = torch.tensor([[math.cos(m * f) for f in inv_freq] * 2
                            for m in rows], dtype=torch.float32)
    sin_ref = torch.tensor([[math.sin(m * f) for f in inv_freq] * 2
                            for m in rows], dtype=torch.float32)
    assert torch.allclose(model.rope_cos[rows], cos_ref, atol=1e-6, rtol=0)
    assert torch.allclose(model.rope_sin[rows], sin_ref, atol=1e-6, rtol=0)
    print("[OK] RoPE 表 = half-split + cat(freqs, freqs)，theta=10000，与 HF 定义一致")

    print("\n--- 2. 参数分解 ---")
    show("token_embedding", count(model.token_embedding))
    per_attn = count(model.blocks[0].attn)
    per_ffn = count(model.blocks[0].ffn)
    per_norm = count(model.blocks[0]) - per_attn - per_ffn
    show("每层 attn", per_attn, f"× {config.n_layer} 层")
    show("每层 ffn", per_ffn, f"× {config.n_layer} 层")
    show("每层 norm", per_norm, "2 个 RMSNorm 的 γ")
    show("blocks 合计", count(model.blocks),
         f"= {config.n_layer} × ({per_attn:,} + {per_ffn:,} + {per_norm})")
    show("norm", count(model.norm), "最后一层 RMSNorm")
    show("lm_head", count(model.lm_head))
    total = count(model)
    show("总参数 (不共享)", total, "Llama 默认 tie=False")
    tied = MiniLlama(replace(config, tie_word_embeddings=True))
    show("总参数 (共享)", count(tied), "GPT-2 / SmolLM2 用这个")
    assert count(model.blocks) == config.n_layer * (per_attn + per_ffn + per_norm)
    assert total == (count(model.token_embedding) + count(model.blocks)
                     + count(model.norm) + count(model.lm_head))
    assert count(tied) == total - count(model.token_embedding)
    assert tied.lm_head.weight is tied.token_embedding.weight

    print("\n--- 3. KV cache：每层预分配 (B, n_kv_head, block_size, head_dim) ---")
    cache = KVCache(config, batch_size=1, device=device, dtype=torch.float32)
    show("每层 k 的形状", cache.k[0])
    show("nbytes()", cache.nbytes(), "字节")
    esize = cache.k[0].element_size()
    formula = (2 * config.n_layer * 1 * config.block_size
               * config.n_kv_head * D * esize)
    show("公式", formula, "2*n_layer*B*T_max*n_kv_head*D*4B")
    show("每 token", formula // config.block_size, "字节/token，只有 MHA 的 1/2")
    assert cache.nbytes() == formula

    print("\n--- 4. 缓存 vs 全量重算：逐步比较 logits ---")
    prompt = torch.tensor([tok.encode("Spring arrives, ")])
    diffs = check_kv_cache(model, prompt, n_steps=8)
    print(f"  {'prefill':<8} max|diff| = {diffs[0]:.2e}")
    for step, d in enumerate(diffs[1:], start=1):
        print(f"  {'step ' + str(step):<8} max|diff| = {d:.2e}")
    worst = max(diffs)
    assert worst < 1e-5
    print(f"[OK] 每步 logits 都一致（最大 {worst:.2e} < 1e-5）")

    print("\n--- 5. 两条贪心解码路径，结果必须逐 token 相同 ---")
    out_naive = generate_naive(model, prompt.clone(), max_new_tokens=16)
    out_cached = generate_cached(model, prompt.clone(), max_new_tokens=16)
    assert torch.equal(out_naive, out_cached)
    show("序列长度", out_naive.size(1), "prompt 16 + 新生成 16")
    print("  生成：" + tok.decode(out_cached[0].tolist()))
    print("[OK] 有 cache 与无 cache 的贪心解码结果完全一致")

    print("\n--- 6. 其他自检 ---")
    ids = torch.tensor([tok.encode("A small cat")])
    logits = model(ids)
    show("logits", logits, "(B, T, V)")
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, V), ids[:, 1:].reshape(-1))
    ln_v = math.log(V)
    show("loss (随机初始化)", f"{loss.item():.4f}", "还没训练，应接近均匀分布")
    show("ln(vocab_size)", f"{ln_v:.4f}", f"V = {V}")
    assert abs(loss.item() - ln_v) < 0.05
    print(f"[OK] 随机初始化时 loss ≈ ln(V)（差 {abs(loss.item() - ln_v):.4f} < 0.05）")

    # 换一条完全不同的路径对照：每一步只喂 1 个 token、start_pos 递增地走缓存，
    # 每个位置的 logits 都必须和整段前向（start_pos=0、无 cache）一致。
    cache_step = KVCache(config, ids.size(0), device, dtype=logits.dtype)
    step_logits = [model(ids[:, p:p + 1], kv_cache=cache_step, start_pos=p)[:, -1]
                   for p in range(ids.size(1))]
    diff_step = (logits - torch.stack(step_logits, dim=1)).abs().max().item()
    assert diff_step < 1e-5
    print(f"[OK] 逐 token 走缓存 prefill 与整段前向一致（max|diff| = {diff_step:.1e}）")

    bf16_norm = model.norm(model.token_embedding(ids).to(torch.bfloat16))
    assert bf16_norm.dtype == torch.bfloat16
    print("[OK] RMSNorm 对 bf16 输入返回 bf16（内部 float32 计算，最后统一转回）")

    ids2 = ids.clone()
    ids2[:, -1] = (ids2[:, -1] + 1) % V
    logits2 = model(ids2)
    assert torch.allclose(logits[:, :-1], logits2[:, :-1], atol=1e-6, rtol=0)
    print("[OK] 改未来 token，前面位置的 logits 不变（端到端因果性）")

    try:
        model(torch.zeros(1, config.block_size + 1, dtype=torch.long))
    except ValueError as err:
        print(f"[OK] 超过 block_size 抛 ValueError：{err}")
    else:
        raise AssertionError("序列超过 block_size 时应该抛 ValueError")

    print("\n一句话记住：")
    print("  Llama = GPT-2 的骨架 - 位置嵌入 - bias - 权重共享")
    print("       + RMSNorm + RoPE + GQA + SwiGLU + KV cache。")
    print("\n[OK] 全部检查通过")


if __name__ == "__main__":
    main()
