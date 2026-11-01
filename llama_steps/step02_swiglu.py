"""Step 2: GELU 前馈网络 → SwiGLU 门控前馈网络。

GPT-2 的 FFN（每个 token 独立做一次「升维 → 激活 → 降维」）：
    h = GELU(x @ W_up)        # C -> 4C
    y = h @ W_down            # 4C -> C
    两个矩阵，参数量 2 · C · 4C = 8C²

Llama 的 FFN（SwiGLU）：
    h = SiLU(x @ W_gate) * (x @ W_up)   # C -> F，两个升维结果逐元素相乘
    y = h @ W_down                      # F -> C
    三个矩阵，参数量 3 · C · F

为什么要改：
    1) 问题：GELU 对每个通道单独做非线性，通道之间没有「商量」；一个通道
       该输出多少，只能看它自己，无法根据输入去「筛掉」另一组通道。
    2) 方案：门控（gating）。SiLU(gate) 像一扇门：负数把对应维度压小，
       正数近似线性放行。门由输入自己算出，每个 token 可以决定哪些通道
       多传递、哪些少传递（注意：矩阵乘法仍是 dense 的，计算量不变）。
    3) 代价：多一个矩阵。为了不让参数量爆炸，把 hidden 从 4C 缩到 8/3·C
       （3 · C · (8/3)C = 8C²），与 GPT-2 的 FFN 对齐。

关键公式：
    8/3·C 的来历：令 3·C·F ≈ 8C²，得 F ≈ 8C/3。
    toy C=64：8/3·64 = 170.67，向上取到 8 的倍数 → 176。
    Llama-7B C=4096：8/3·4096 = 10922.67，取 256 的倍数 → 11008（GPU 对齐）。
    门：SiLU(g) = g · sigmoid(g)，值域 ≈ (-0.28, +∞)。

和上一步的 diff（只列变化的行）：
    - self.up   = nn.Linear(C, 4C)          # 一个升维矩阵
    - self.down = nn.Linear(4C, C)
    + self.gate = nn.Linear(C, F)           # 多一个门控矩阵
    + self.up   = nn.Linear(C, F)           # hidden 从 4C 缩小到 8/3·C
    + self.down = nn.Linear(F, C)
    - y = down(GELU(up(x)))
    + y = down(SiLU(gate(x)) * up(x))
"""

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


@dataclass
class LlamaConfig:
    """和 step04 起完全相同的配置（后面每一步都用这套字段名）。"""

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


class GPT2FFN(nn.Module):
    """GPT-2 的 FFN：C -> 4C -> C，中间 GELU，参数 8C²。"""

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        c = config.n_embd
        self.up_proj = nn.Linear(c, 4 * c, bias=False)
        self.down_proj = nn.Linear(4 * c, c, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C) -> (B, T, C)。"""
        return self.down_proj(F.gelu(self.up_proj(x)))


class SwiGLU(nn.Module):
    """Llama 的 FFN：C -> F -> C，gate 和 up 并行升维，相乘后降维。"""

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        c, f = config.n_embd, config.ffn_hidden
        # 三个矩阵都不带 bias：Llama 全线 Linear(bias=False)
        self.gate_proj = nn.Linear(c, f, bias=False)
        self.up_proj = nn.Linear(c, f, bias=False)
        self.down_proj = nn.Linear(f, c, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C) -> (B, T, C)。"""
        # 门控 = 用 SiLU(gate) 逐元素缩放 up 的每个通道，再压回 C 维
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


def main() -> None:
    torch.manual_seed(0)
    config = LlamaConfig(vocab_size=30)
    c = config.n_embd
    f = config.ffn_hidden

    print("\n--- 1. hidden 为什么是 8/3·C ---")
    raw = 8 * c / 3
    show("C (隐藏维度)", c)
    show("8/3·C", f"{raw:.2f}", "与 4C 的 GPT-2 FFN 参数量对齐")
    show("向上取整到 8 的倍数", f, "170.67 → 176（8 的倍数，方便切块）")
    assert f % 8 == 0 and f >= raw
    print("  如果 hidden 还保持 4C=256，三个矩阵就是 12C²，比 GPT-2 的 8C² 多 50%；")
    print("  缩到 8/3·C 正好把「多出来的那个门控矩阵」的预算还回去。")

    print("\n--- 2. 参数量对照 ---")
    old_ffn = GPT2FFN(config)
    new_ffn = SwiGLU(config)
    n_old = sum(p.numel() for p in old_ffn.parameters())
    n_new = sum(p.numel() for p in new_ffn.parameters())
    show("GPT-2 FFN 参数", n_old, "2·C·4C = 8C²")
    show("SwiGLU 参数", n_new, "3·C·F")
    show("每个矩阵", c * f, "C·F = 64·176")
    show("比值", f"{n_new / n_old:.4f}", "≈1.03，基本对齐")
    assert 1.0 <= n_new / n_old <= 1.1
    print("[OK] SwiGLU 在参数量基本不变的前提下多了一个门控矩阵")

    print("\n--- 3. 前向形状：可以直接替换旧的 FFN ---")
    x = torch.randn(2, 16, c)                       # (B, T, C)
    gate = new_ffn.gate_proj(x)                     # (B, T, F)
    up = new_ffn.up_proj(x)                         # (B, T, F)
    hidden = F.silu(gate) * up                      # (B, T, F)
    y = new_ffn(x)                                  # (B, T, C)
    show("x", x)
    show("gate_proj(x)", gate)
    show("up_proj(x)", up)
    show("SiLU(gate) * up", hidden, "逐元素相乘")
    show("down_proj(hidden)", y, "回到 (B,T,C)")
    assert y.shape == (2, 16, c) == old_ffn(x).shape
    print("[OK] 输出形状和 GPT-2 FFN 完全一致，可以原地替换")

    print("\n--- 4. 门控直觉：SiLU 把 gate 值翻译成「开多大」---")
    vals = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    silu = F.silu(vals)
    print("  gate =", " ".join(f"{v:+7.2f}" for v in vals.tolist()))
    print("  SiLU =", " ".join(f"{v:+7.3f}" for v in silu.tolist()))
    print("  负数 → 压到 0 附近（门关上）；正数 → 近似原样通过（门打开）。")
    assert torch.allclose(silu, vals * torch.sigmoid(vals), atol=1e-6)
    assert silu[2].item() == 0.0
    print("[OK] SiLU(x) = x·sigmoid(x)，且 SiLU(0) = 0（0 门 = 完全关闭）")

    print("\n--- 5. 有多少通道对 down_proj 的贡献被压到接近 0 ---")
    with torch.no_grad():
        xb = torch.randn(8, 64, c)                  # (B, T, C)
        gates = F.silu(new_ffn.gate_proj(xb))
        low = (gates.abs() < 0.05).float().mean().item()
    show("|SiLU(gate)| < 0.05 的比例", f"{low * 100:.1f}%", "对 down 的贡献很小")
    print("  注意：门控 ≠ 稀疏计算——矩阵乘法仍是 dense 的，FLOPs 一个不少；")
    print("  收益是每个 token 能动态地少依赖一些通道（有效容量/表达），不是省算力。")

    print("\n一句话记住：")
    print("  SwiGLU 用一扇 SiLU 门逐元素筛一遍 up 的通道，再把 hidden 缩到 8/3·C。")
    print("\n[OK] 全部检查通过")


if __name__ == "__main__":
    main()
