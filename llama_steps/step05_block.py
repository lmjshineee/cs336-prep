"""Step 5: 把 Llama 的零件拼成一个 Block（RMSNorm + RoPE + GQA + SwiGLU + 残差）。

GPT-2 的做法（前面 step06 的 block）：
    x = x + attn(ln1(x))
    x = x + ffn(ln2(x))

Llama 的做法（骨架完全相同，只换了零件）：
    x = x + attn(attn_norm(x), cos, sin)   # 位置信息改从 cos/sin 传进来
    x = x + ffn(ffn_norm(x))

为什么要改：
    1) 归一化：LayerNorm -> RMSNorm，只除均方根，省掉减均值和 β（step01）
    2) 位置：学习式位置嵌入 -> RoPE，不存位置表，按绝对位置旋转 q/k（step03）
    3) 注意力：MHA -> GQA，K/V 头数少 g 倍（step04）
    4) 前馈：GELU-FFN -> SwiGLU，门控 + 3 个矩阵，hidden 取 8/3·C（step02）
    5) 正则：去掉 bias 和 dropout。去 bias 是 PaLM/Llama 的经验选择：质量基本
       不降，参数少一点，也更容易把 Linear 和归一化融进一个 kernel；dropout 在
       大数据量下拖慢收敛，推理时又是恒等运算。GPT-2 把 dropout 用在注意力
       权重和残差上，p=0.1。
    骨架不变：Pre-Norm + 两条残差，保证输入输出同形、深层也训得动。

关键公式（C = n_embd，F = ffn_hidden，g = n_head / n_kv_head，均不含 Linear bias）：
    GPT-2 block = 4C²(attn) + 8C²(ffn) + 4C(2 个 LayerNorm 的 γ、β)
    Llama block = (2C² + 2C²/g)(attn) + 3·C·F(SwiGLU) + 2C(2 个 RMSNorm 的 γ)

和 GPT-2 block 的 diff（只列变化的行）：
    - x = x + self.attn(self.ln1(x))
    + x = x + self.attn(self.attn_norm(x), cos, sin)
    - x = x + self.ffn(self.ln2(x))
    + x = x + self.ffn(self.ffn_norm(x))
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


class RMSNorm(nn.Module):
    """x / RMS(x) * γ，其中 RMS(x) = √(mean(x²) + eps)。

    和 LayerNorm 的区别：不减均值、没有 β，所以每个 token 只省下一半参数。
    计算在 float32 里做，再转回原 dtype（和 HuggingFace 的写法一致）。
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))     # 初始 γ = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(..., C) -> (..., C)，只在最后一维上归一化。"""
        dtype = x.dtype
        x32 = x.float()
        rms = torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        # 先乘 γ 再统一转回：fp32 下逐位不变，bf16/fp16 输入也能得到原 dtype
        return (x32 * rms * self.weight).to(dtype)


class SwiGLU(nn.Module):
    """down(SiLU(gate(x)) * up(x))：SiLU(gate) 是一个逐元素的「门」。"""

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.n_embd, config.ffn_hidden, bias=False)
        self.up_proj = nn.Linear(config.n_embd, config.ffn_hidden, bias=False)
        self.down_proj = nn.Linear(config.ffn_hidden, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, C) -> (B, T, C)，先升到 F 维再降回来。"""
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """(..., D) -> (..., D)：把后一半取负挪到前面，实现「配对旋转」。"""
    d = x.size(-1) // 2
    return torch.cat((-x[..., d:], x[..., :d]), dim=-1)


def build_rope_cache(head_dim: int, max_len: int, theta: float = 10000.0,
                     device: torch.device | None = None
                     ) -> tuple[torch.Tensor, torch.Tensor]:
    """返回 (cos, sin)，形状都是 (max_len, head_dim)，float32。"""
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
                kv_cache=None, start_pos: int = 0) -> torch.Tensor:
        """x: (B,T,C) -> (B,T,C)；cos/sin: (T,D)；kv_cache 到 step06 才用。"""
        B, T, C = x.shape
        H, Hkv, D = self.n_head, self.n_kv_head, self.head_dim

        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)         # (B,H,T,D)
        k = self.k_proj(x).view(B, T, Hkv, D).transpose(1, 2)       # (B,Hkv,T,D)
        v = self.v_proj(x).view(B, T, Hkv, D).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)   # 只旋转 q、k，v 不转：位置只影响打分

        if kv_cache is not None:
            k, v = kv_cache.append(self.layer_idx, k, v, start_pos)

        total = k.size(2)
        k = repeat_kv(k, self.n_rep)              # 复制成 H 个头才能和 q 对齐
        v = repeat_kv(v, self.n_rep)

        scores = q @ k.transpose(-2, -1) / math.sqrt(D)             # (B,H,T,total)
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
                kv_cache=None, start_pos: int = 0) -> torch.Tensor:
        """x: (B,T,C) -> (B,T,C)。残差保证输入输出同形，可以无限串联。"""
        x = x + self.attn(self.attn_norm(x), cos, sin, kv_cache, start_pos)
        x = x + self.ffn(self.ffn_norm(x))
        return x


# ---------------------------------------------------------------------------
# 只用来数参数的 GPT-2 block 样品（Linear 都带 bias，和真实 GPT-2 一致）
# ---------------------------------------------------------------------------

class GPT2Block(nn.Module):
    """LayerNorm + 带 bias 的 MHA + GELU-FFN，只用来数参数量，不跑前向。"""

    def __init__(self, c: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(c)
        self.qkv = nn.Linear(c, 3 * c)
        self.o_proj = nn.Linear(c, c)
        self.ln2 = nn.LayerNorm(c)
        self.up = nn.Linear(c, 4 * c)
        self.down = nn.Linear(4 * c, c)


def count(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def main() -> None:
    torch.manual_seed(0)
    C, F_hidden = 64, 176
    config = LlamaConfig(vocab_size=30)   # block 不碰词表，vocab_size 只是占位
    block = LlamaBlock(config)
    H, Hkv = config.n_head, config.n_kv_head
    D, g = C // H, H // Hkv
    T = 8

    print("\n--- 1. 一个 block 只有两行，各自包着残差 ---")
    print("  x = x + attn(attn_norm(x), cos, sin)")
    print("  x = x + ffn(ffn_norm(x))")
    x = torch.randn(1, T, C) * 3.0        # 故意把残差流放大 3 倍
    cos, sin = build_rope_cache(D, T, config.rope_theta)
    y = block(x, cos, sin)
    show("x", x, "(B,T,C)，残差流的初始值")
    show("y", y, "(B,T,C)，和输入同形")
    assert y.shape == x.shape

    print("\n--- 2. Pre-Norm：归一化在残差分支入口，不在残差之后 ---")
    seen: dict[str, torch.Tensor] = {}

    def grab(tag: str):
        def hook(module, args, output):
            seen[tag] = args[0].detach().clone()
        return hook

    h1 = block.attn_norm.register_forward_hook(grab("attn"))
    h2 = block.ffn_norm.register_forward_hook(grab("ffn"))
    y = block(x, cos, sin)
    h1.remove()
    h2.remove()

    attn_out = block.attn(block.attn_norm(x), cos, sin)
    assert torch.equal(seen["attn"], x)                       # 拿到的是残差流本身
    assert torch.allclose(seen["ffn"], x + attn_out, atol=1e-6, rtol=0)
    y_manual = x + attn_out + block.ffn(block.ffn_norm(x + attn_out))
    assert torch.allclose(y, y_manual, atol=1e-6, rtol=0)

    rms_in = x.pow(2).mean(-1).sqrt().mean().item()
    rms_out = block.attn_norm(x).pow(2).mean(-1).sqrt().max().item()
    show("residual RMS", f"{rms_in:.2f}", "没被归一化，故意放大 3 倍")
    show("attn_norm out RMS", f"{rms_out:.4f}", "≈ 1，尺度被拉回来了")
    show("manual vs block", f"max|diff| = {(y - y_manual).abs().max().item():.1e}")
    assert abs(rms_out - 1) < 1e-4
    assert block.attn_norm(x.to(torch.bfloat16)).dtype == torch.bfloat16
    print("[OK] RMSNorm 在 float32 里算，最后转回输入 dtype：bf16 进 -> bf16 出")

    print("\n--- 3. 没有 bias、没有 dropout ---")
    linears = [m for m in block.modules() if isinstance(m, nn.Linear)]
    show("Linear 个数", len(linears), "q/k/v/o + gate/up/down")
    show("bias 参数数", sum(m.bias.numel() for m in linears if m.bias is not None))
    show("Dropout 个数", sum(1 for m in block.modules()
                             if isinstance(m, nn.Dropout)))
    show("config.dropout", config.dropout, "GPT-2 是 0.1，Llama 是 0")
    assert all(m.bias is None for m in linears)
    assert not any(isinstance(m, nn.Dropout) for m in block.modules())

    print(f"\n--- 4. 参数量：GPT-2 block vs Llama block（C={C}, F={F_hidden}, "
          f"g={g}） ---")
    gpt2 = GPT2Block(C)
    g2_attn = count(gpt2.qkv) + count(gpt2.o_proj)
    g2_ffn = count(gpt2.up) + count(gpt2.down)
    g2_ln = count(gpt2.ln1) + count(gpt2.ln2)
    l_attn = count(block.attn)
    l_ffn = count(block.ffn)
    l_norm = count(block.attn_norm) + count(block.ffn_norm)
    assert g2_attn == 4 * C * C + 4 * C and g2_ffn == 8 * C * C + 5 * C
    assert l_ffn == 3 * C * F_hidden
    show("GPT-2 attn", g2_attn, "4C² + 4C（含 bias）")
    show("GPT-2 ffn", g2_ffn, "8C² + 5C（含 bias）")
    show("GPT-2 LayerNorm", g2_ln, "2 × (γ,β) = 4C")
    show("GPT-2 block", count(gpt2), "12C² + 13C")
    show("Llama attn", l_attn, "2C² + 2C²/g（无 bias）")
    show("Llama ffn", l_ffn, "3·C·F")
    show("Llama RMSNorm", l_norm, "2 × γ = 2C")
    show("Llama block", count(block),
         f"{(count(block) / count(gpt2) - 1):+.1%} vs GPT-2")
    show("GPT-2 若不带 bias", 12 * C * C + 4 * C, "4C² + 8C² + 4C（LN 的 γ、β）")
    l_attn_g4 = 2 * C * C + 2 * C * C // 4
    show("Llama 若 g=4", l_attn_g4 + l_ffn + l_norm, "2.5C² + 3CF + 2C")

    assert count(block) == l_attn + l_ffn + l_norm
    assert count(gpt2) == g2_attn + g2_ffn + g2_ln
    assert count(block) < count(gpt2)

    print("\n--- 5. block 内参数占比 ---")
    total = count(block)
    show("attn", f"{l_attn / total:.1%}", "GQA 之后只占约 1/4")
    show("ffn", f"{l_ffn / total:.1%}", "大头依然在 FFN")
    show("norm", f"{l_norm / total:.1%}", "只剩 γ，几乎可以忽略")
    assert l_attn + l_ffn + l_norm == total

    print("\n--- 6. 三个自检 ---")
    assert y.shape == x.shape
    print("[OK] 输入输出同形：block 可以原样叠 N 层")

    with torch.no_grad():
        block.attn.o_proj.weight.zero_()      # 注意力分支输出 0
        block.ffn.down_proj.weight.zero_()    # FFN 分支输出 0
        y0 = block(x, cos, sin)
    d = (y0 - x).abs().max().item()
    assert d < 1e-6
    print(f"[OK] 两个子层输出为 0 时，block 退化成恒等映射（max|diff| = {d:.1e}）")

    print("[OK] 先 norm 再进子层、最后加残差：hook 抓到的是未归一化的残差流")

    print("\n一句话记住：")
    print("  Llama block = GPT-2 block 的骨架 + 四个零件：")
    print("  RMSNorm / RoPE / GQA / SwiGLU，外加去掉 bias 和 dropout。")
    print("\n[OK] 全部检查通过")


if __name__ == "__main__":
    main()
