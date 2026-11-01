"""Step 4: 多头注意力（MHA）→ 分组查询注意力（GQA, Grouped-Query Attention）。

GPT-2 的做法（MHA）：
    q、k、v 三个投影都是 C -> C，也就是 H 个头各有一套自己的 K、V：
        q = q_proj(x).view(B, T, H, D)      # H = n_head
        k = k_proj(x).view(B, T, H, D)
        v = v_proj(x).view(B, T, H, D)
    参数量 = C²(q) + C²(k) + C²(v) + C²(o) = 4C²。

Llama 的做法（GQA）：
    Q 仍然有 H 个头，K/V 只有 Hkv 个（H % Hkv == 0），
    每 g = H / Hkv 个 Q 头共享一组 K/V：
        q = q_proj(x).view(B, T, H, D)          # C -> C
        k = k_proj(x).view(B, T, Hkv, D)        # C -> Hkv*D，比 q 小 g 倍
        v = v_proj(x).view(B, T, Hkv, D)
        k, v = repeat_kv(k, g), repeat_kv(v, g) # 复制 g 份去对齐 q，不新增参数
    n_kv_head = n_head 时退回 MHA；n_kv_head = 1 时叫 MQA（最省）。

为什么要改：
    1) 问题：推理时要把历史 K、V 存进 KV cache。MHA 每个 token 每层存 2C 个数，
       decode 每步都要把整个 cache 读一遍，长上下文时显存带宽全花在读写 cache 上。
    2) 方案：让多个 Q 头共享同一组 K/V，cache 直接缩小 g 倍。
    3) 代价：K/V 的表达能力被压缩，几个 Q 头只能看到同一份「历史摘要」；
       g 取 4~8 时质量几乎不掉，所以 Llama-2/3、Mistral 都用 GQA。

关键公式（C = n_embd，D = C / n_head，g = n_head / n_kv_head）：
    attn 参数 = C·C + 2·C·(n_kv_head·D) + C·C = 2C² + 2C²/g
    KV cache 每 token 每层 = 2 · n_kv_head · D 个元素
    g=2 -> 3C²（比 MHA 的 4C² 少 25%）
    g=4 -> 2.5C²（少 37.5%，Llama-3-8B 就是 32 个 Q 头 / 8 个 KV 头）

和 GPT-2 版多头注意力的 diff（只列变化的行）：
    - self.qkv = nn.Linear(C, 3*C, bias=False)                # 一次算出 q/k/v
    + self.q_proj = nn.Linear(C, C, bias=False)               # q 还是 C 维
    + self.k_proj = nn.Linear(C, n_kv_head * D, bias=False)   # k/v 只有 Hkv 个头
    + self.v_proj = nn.Linear(C, n_kv_head * D, bias=False)
    + k = repeat_kv(k, g)                                     # 用之前先复制 g 份
"""

import math
import unicodedata
from dataclasses import dataclass

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


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """(..., D) -> (..., D)：把后一半取负挪到前面，实现「配对旋转」。"""
    d = x.size(-1) // 2
    return torch.cat((-x[..., d:], x[..., :d]), dim=-1)


def build_rope_cache(head_dim: int, max_len: int, theta: float = 10000.0,
                     device: torch.device | None = None
                     ) -> tuple[torch.Tensor, torch.Tensor]:
    """返回 (cos, sin)，形状都是 (max_len, head_dim)，float32。

    第 i 对通道共用角频率 θ_i = theta^(-2i/D)，位置 m 处的旋转角是 m·θ_i。
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


class Attention(nn.Module):
    """GQA 注意力：Q 有 n_head 个头，K/V 只有 n_kv_head 个，多出来的 Q 头共享 K/V。"""

    def __init__(self, config: LlamaConfig, layer_idx: int = 0) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0, "C 必须能被 n_head 整除"
        assert config.n_head % config.n_kv_head == 0, "n_head 必须是 n_kv_head 的倍数"
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_rep = config.n_head // config.n_kv_head   # 每个 KV 头被几个 Q 头共用
        self.head_dim = config.n_embd // config.n_head
        self.n_embd = config.n_embd

        # Llama 所有 Linear 都是 bias=False；q 依旧是 C 维，k/v 只有 Hkv*D 维
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        kv_dim = config.n_kv_head * self.head_dim
        self.k_proj = nn.Linear(config.n_embd, kv_dim, bias=False)
        self.v_proj = nn.Linear(config.n_embd, kv_dim, bias=False)
        self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                kv_cache=None, start_pos: int = 0) -> torch.Tensor:
        """x: (B,T,C) -> (B,T,C)。

        cos/sin: (T,D)，对应当前这 T 个 token 的绝对位置。
        kv_cache: step06 才定义的 KVCache；这一步不传，只给下面那三行占位。
        """
        B, T, C = x.shape
        H, Hkv, D = self.n_head, self.n_kv_head, self.head_dim

        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)        # (B,H,T,D)
        k = self.k_proj(x).view(B, T, Hkv, D).transpose(1, 2)      # (B,Hkv,T,D)
        v = self.v_proj(x).view(B, T, Hkv, D).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)   # 只旋转 q、k，v 不转：位置只影响打分

        if kv_cache is not None:
            # 推理时把这步新算的 k/v 写进缓存，拿回 0..start_pos+T 的全部历史
            k, v = kv_cache.append(self.layer_idx, k, v, start_pos)

        total = k.size(2)                     # 参与打分的 key 个数（可能是历史长度）
        k = repeat_kv(k, self.n_rep)          # (B,H,total,D)，复制后才有 H 个头
        v = repeat_kv(v, self.n_rep)

        scores = q @ k.transpose(-2, -1) / math.sqrt(D)             # (B,H,T,total)
        # 因果掩码：query 的绝对位置是 start_pos..start_pos+T-1，key 从 0 开始
        q_pos = torch.arange(start_pos, start_pos + T, device=x.device)[:, None]
        k_pos = torch.arange(total, device=x.device)[None, :]
        scores = scores.masked_fill(k_pos > q_pos, float("-inf"))
        weights = F.softmax(scores, dim=-1)                         # (B,H,T,total)
        out = (weights @ v).transpose(1, 2).reshape(B, T, C)
        return self.o_proj(out)


def main() -> None:
    torch.manual_seed(0)
    tok = CharTokenizer(CORPUS)
    config = LlamaConfig(vocab_size=tok.vocab_size)
    attn = Attention(config)
    C, H, Hkv = config.n_embd, config.n_head, config.n_kv_head
    D = C // H
    g = H // Hkv

    print("\n--- 1. GQA 到底改了什么 ---")
    show("C (n_embd)", C)
    show("n_head", H, "Q 头的个数")
    show("n_kv_head", Hkv, "K/V 头的个数，只有 Q 头的 1/g")
    show("head_dim", D, "= C / n_head，每个头的维度")
    show("g", g, "= n_head / n_kv_head，每 g 个 Q 头共享一组 K/V")
    print("  n_kv_head = n_head -> MHA（每个 Q 头独占自己的 K/V）")
    print("  n_kv_head = 1      -> MQA（所有 Q 头共享同一组 K/V）")

    print("\n--- 2. 形状流水线：只有 K/V 的头数变了 ---")
    ids = torch.tensor([tok.encode("cat sits")])
    x = torch.randn(1, ids.size(1), C)     # 假装这是 embedding 的输出
    B, T = 1, ids.size(1)
    cos, sin = build_rope_cache(D, T, config.rope_theta)

    q = attn.q_proj(x).view(B, T, H, D).transpose(1, 2)
    k = attn.k_proj(x).view(B, T, Hkv, D).transpose(1, 2)
    v = attn.v_proj(x).view(B, T, Hkv, D).transpose(1, 2)
    q = apply_rope(q, cos, sin)
    k = apply_rope(k, cos, sin)
    k_rep, v_rep = repeat_kv(k, g), repeat_kv(v, g)
    scores = q @ k_rep.transpose(-2, -1) / math.sqrt(D)

    show("x", x, "(B,T,C)")
    show("q", q, "(B,H,T,D)，已过 RoPE，H 个 query 头")
    show("k", k, "(B,Hkv,T,D)，已过 RoPE，只有 Hkv 个 KV 头")
    show("repeat_kv(k)", k_rep, "(B,H,T,D)，复制出来给每个 Q 头")
    show("scores", scores, "(B,H,T,T)，这里还没加因果掩码")
    show("out", attn(x, cos, sin), "(B,T,C)，和输入同形")

    assert k_rep.shape == q.shape
    assert torch.equal(k_rep[:, 1], k[:, 0])   # 第 2 个 Q 头拿到的就是第 1 组 K/V
    assert torch.equal(v_rep[:, 1], v[:, 0])

    print("\n--- 3. 参数量：K/V 投影变小了 ---")

    def count(m: nn.Module) -> int:
        return sum(p.numel() for p in m.parameters())

    show("q_proj", count(attn.q_proj), "C * C")
    show("k_proj", count(attn.k_proj), "C * C/g")
    show("v_proj", count(attn.v_proj), "C * C/g")
    show("o_proj", count(attn.o_proj), "C * C")
    gqa = count(attn)
    mha = 4 * C * C
    show("attn GQA", gqa, "2C² + 2C²/g")
    show("attn MHA", mha, "4C²")
    show("saved", mha - gqa, f"{(gqa / mha - 1):+.1%} vs MHA")
    assert gqa == 2 * C * C + 2 * C * C // g
    mqa = 2 * C * C + 2 * C * C // H
    show("MQA (kv=1)", mqa, f"{(mqa / mha - 1):+.1%} vs MHA")

    print("\n--- 4. KV cache：GQA 真正省钱的地方 ---")
    per_tok_gqa = 2 * Hkv * D
    per_tok_mha = 2 * H * D
    show("per-token GQA", per_tok_gqa, "每 token 每层的元素数")
    show("per-token MHA", per_tok_mha, "每 token 每层的元素数")
    n_layer, t_max = config.n_layer, config.block_size
    toy_gqa = 2 * n_layer * 1 * t_max * Hkv * D * 4
    toy_mha = 2 * n_layer * 1 * t_max * H * D * 4
    show("toy cache GQA", toy_gqa, "字节，B=1，fp32")
    show("toy cache MHA", toy_mha, "字节，B=1，fp32")
    ctx, d_real = 8192, 128                      # 真实 Llama 的上下文和 head_dim
    l8_gqa = 2 * 32 * ctx * 8 * d_real * 2       # 32 层、8 个 KV 头、fp16
    l8_mha = 2 * 32 * ctx * 32 * d_real * 2      # 同规模但 32 个 KV 头（MHA）
    l70_gqa = 2 * 80 * ctx * 8 * d_real * 2      # 80 层、8 个 KV 头、fp16
    show("Llama-3-8B GQA", f"{l8_gqa / 2**30:.1f} GiB", "32 层 × 8 KV 头 × 8192 × fp16")
    show("Llama-3 MHA", f"{l8_mha / 2**30:.1f} GiB", "同层数，但 32 个 KV 头")
    show("Llama-3-70B GQA", f"{l70_gqa / 2**30:.1f} GiB", "80 层 × 8 KV 头 × 8192 × fp16")
    print("  公式：2 * n_layer * B * T_max * n_kv_head * head_dim * sizeof(dtype)")
    assert l8_gqa == 2**30 and l8_mha == 4 * 2**30
    assert l70_gqa == int(2.5 * 2**30)
    assert toy_mha == g * toy_gqa

    print("\n--- 5. 三个自检 ---")
    # (a) n_kv_head = n_head 时，GQA 必须和手写的 MHA 完全一样
    attn_mha = Attention(LlamaConfig(vocab_size=tok.vocab_size,
                                     n_head=H, n_kv_head=H))
    y_gqa = attn_mha(x, cos, sin)             # n_rep=1，等价于普通 MHA
    qh = apply_rope(attn_mha.q_proj(x).view(B, T, H, D).transpose(1, 2), cos, sin)
    kh = apply_rope(attn_mha.k_proj(x).view(B, T, H, D).transpose(1, 2), cos, sin)
    vh = attn_mha.v_proj(x).view(B, T, H, D).transpose(1, 2)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
    scores_h = qh @ kh.transpose(-2, -1) / math.sqrt(D)
    scores_h = scores_h.masked_fill(~mask, float("-inf"))
    weights_h = F.softmax(scores_h, dim=-1)
    y_manual = (weights_h @ vh).transpose(1, 2).reshape(B, T, C)
    y_manual = attn_mha.o_proj(y_manual)
    diff = (y_gqa - y_manual).abs().max().item()
    assert diff < 1e-6
    print(f"[OK] n_kv_head = n_head 时 GQA 退化成 MHA，max|diff| = {diff:.1e}")

    # (b) 因果性：改最后一个 token，前面位置的输出不能变
    x2 = x.clone()
    x2[:, -1] = torch.randn(C)
    past_a = attn(x, cos, sin)[:, :-1]
    past_b = attn(x2, cos, sin)[:, :-1]
    assert torch.allclose(past_a, past_b, atol=1e-6, rtol=0)
    print("[OK] 改未来 token，不影响过去位置的输出（因果掩码）")

    # (c) repeat_kv：形状对，而且第 2 个 Q 头拿到的就是第 1 组 K/V（复制，不是新参数）
    assert k_rep.shape == (B, H, T, D)
    assert torch.equal(k_rep[:, 1], k[:, 0]) and torch.equal(v_rep[:, 1], v[:, 0])
    print("[OK] repeat_kv 只复制、不新增参数：多出来的 Q 头共用同一组 K/V")

    print("\n一句话记住：")
    print("  GQA 让 g 个 Q 头共享一组 K/V：cache 小 g 倍，参数从 4C² 降到 2C²+2C²/g。")
    print("\n[OK] 全部检查通过")


if __name__ == "__main__":
    main()
