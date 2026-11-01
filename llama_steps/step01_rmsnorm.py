"""Step 1: LayerNorm → RMSNorm —— 归一化时不再减均值。

GPT-2 的做法（对每个 token 自己的 C 维向量归一化）：
    mu  = x.mean(-1, keepdim=True)
    var = x.var(-1, unbiased=False, keepdim=True)
    y   = (x - mu) / sqrt(var + eps) * gamma + beta   # 减均值 + 除标准差 + 两组参数

Llama 的做法：
    rms = sqrt(x.pow(2).mean(-1, keepdim=True) + eps) # 均方根，不减均值
    y   = x / rms * gamma                             # 只有 gamma，没有 beta

为什么要改：
    1) 问题：LayerNorm 要算均值、方差，还带 gamma、beta 两组参数。可是在残差
       流里，每个 token 的「中心」本来就在 0 附近漂移，硬把均值拉回 0 未必有收益。
    2) 方案：RMSNorm 只除以「均方根」（root mean square，RMS = sqrt(mean(x²))），
       保留向量的整体方向（包括均值偏移），只把尺度统一到 1。
       结果：少一次减均值、少一次加法、参数量减半。
    3) 代价：丢掉了「重新居中」的能力。实测大模型上几乎不掉点，
       而省下的参数和计算是实打实的，所以 Llama 全系列默认用它。

关键公式：
    LayerNorm: y = (x - μ) / √(σ² + ε) * γ + β,   μ = mean(x), σ² = mean((x-μ)²)
    RMSNorm:   y = x / √(mean(x²) + ε) * γ

和上一步的 diff（只列变化的行）：
    - self.ln = nn.LayerNorm(C)        # 2C 个参数，输出均值 0
    + self.ln = RMSNorm(C)             # C 个参数，输出均方根 1
"""

import time
import unicodedata

import torch
from torch import nn


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


class RMSNorm(nn.Module):
    """Llama 的 RMSNorm：y = x / sqrt(mean(x²) + eps) * gamma。"""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))   # 只有 gamma，没有 beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C) -> (B, T, C)，在最后一维上做归一化。"""
        dtype = x.dtype
        # 先在 fp32 里算：小 dtype 下 mean(x²) 容易溢出，HF 的 LlamaRMSNorm 也这样
        x = x.float()
        # eps 加在 mean(x²) 里面再开方，保证分母永远大于 0（加在开方后就不保证了）
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return ((x / rms) * self.weight).to(dtype)


def timeit(fn, warmup: int = 20, iters: int = 200) -> float:
    """先 warmup 再计时，返回中位数（微秒）。"""
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    samples.sort()
    return samples[len(samples) // 2] * 1e6


def main() -> None:
    torch.manual_seed(0)
    C = 64

    print("\n--- 1. 公式与代码对照 ---")
    print("  LayerNorm:  μ = mean(x),  σ² = mean((x-μ)²)")
    print("              y = (x - μ) / √(σ² + ε) * γ + β")
    print("  RMSNorm :   y = x / √(mean(x²) + ε) * γ      # 不碰均值，没有 β")
    print("  写成代码：")
    print("    LN : (x - x.mean(-1, keepdim=True))")
    print("         / (x.var(-1, keepdim=True, unbiased=False) + eps).sqrt() * g + b")
    print("    RMS: x / (x.pow(2).mean(-1, keepdim=True) + eps).sqrt() * g")

    print("\n--- 2. 均值偏移的例子：LN 抹掉它，RMSNorm 保留它 ---")
    # 故意造一个整体偏向 +3 的向量：LN 会把它拉回 0，RMSNorm 不会
    x = torch.randn(2, C) * 0.5 + 3.0
    ln = nn.LayerNorm(C)
    rms = RMSNorm(C)
    y_ln = ln(x)
    y_rms = rms(x)

    show("输入 mean", f"{x.mean().item():+.3f}", "整体偏向 +3")
    show("输入 std", f"{x.std(unbiased=False).item():.3f}")
    show("LN 输出 mean", f"{y_ln.mean().item():+.3f}", "被减均值拉回 0")
    show("LN 输出 std", f"{y_ln.std(unbiased=False).item():.3f}", "除以标准差后 ≈ 1")
    show("RMS 输出 mean", f"{y_rms.mean().item():+.3f}", "均值方向保留下来")
    show("RMS 输出 std", f"{y_rms.std(unbiased=False).item():.3f}", "它不是 1：没有重新居中")
    show("RMS 输出均方根", f"{y_rms.pow(2).mean().sqrt().item():.3f}", "被拉到 1 的是均方根")

    per_token_rms = y_rms.pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(per_token_rms, torch.ones_like(per_token_rms), atol=1e-5)
    assert abs(y_ln.mean().item()) < 1e-6
    assert abs(y_ln.std(unbiased=False).item() - 1.0) < 1e-3
    assert y_rms.mean().item() > 0.5         # 偏移还在，没有被抹掉
    print("[OK] LN 输出均值≈0；RMSNorm 输出均方根≈1，但保留了均值方向")

    print("\n--- 3. 参数量：C → 2C vs C ---")
    n_ln = sum(p.numel() for p in nn.LayerNorm(C).parameters())
    n_rms = sum(p.numel() for p in RMSNorm(C).parameters())
    show("C (隐藏维度)", C)
    show("LayerNorm 参数", n_ln, "γ (C) + β (C)")
    show("RMSNorm 参数", n_rms, "只有 γ (C)")
    assert n_ln == 2 * C and n_rms == C
    print(f"[OK] RMSNorm 参数正好是 LayerNorm 的一半（{n_rms} vs {n_ln}）")

    print("\n--- 4. 实测耗时（warmup 20 次 + 200 次取中位数）---")
    xb = torch.randn(8, 64, C)               # (B, T, C) = (8, 64, 64) fp32
    gamma = torch.ones(C)
    eps = 1e-5

    def ln_formula(x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        return (x - mu) / torch.sqrt(var + eps) * gamma

    def rms_formula(x: torch.Tensor) -> torch.Tensor:
        return x / torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps) * gamma

    with torch.no_grad():
        t_ln = timeit(lambda: ln_formula(xb))
        t_rms = timeit(lambda: rms_formula(xb))
        t_ln_fused = timeit(lambda: ln(xb))
        t_rms_mod = timeit(lambda: rms(xb))
    show("手写 LN 公式", f"{t_ln:.1f} µs")
    show("手写 RMS 公式", f"{t_rms:.1f} µs")
    show("公式版比值", f"{t_ln / t_rms:.2f}x", "少一次减均值/方差，快不少")
    show("nn.LayerNorm（融合）", f"{t_ln_fused:.1f} µs")
    show("RMSNorm 模块", f"{t_rms_mod:.1f} µs")
    print("  但 nn.LayerNorm 是单个 C++ 融合 kernel，而 RMSNorm 模块由几个小算子拼成，")
    print("  这个 8×64×64 的规模下启动开销占大头，差异很小（甚至可能反过来）。")
    print("  省下的是「少一次逐元素计算 + 一半参数」，大张量/大模型上才明显。")
    print("  注意：上面几行耗时每次运行都会小幅抖动，属正常现象。")

    print("\n--- 5. 数值稳定性：eps 加在均方根里面 ---")
    z = torch.zeros(2, C)
    naive = z / torch.sqrt(z.pow(2).mean(dim=-1, keepdim=True))   # 0/0
    y_zero = rms(z)
    show("全零输入 + 朴素除法", f"有 NaN = {bool(torch.isnan(naive).any())}")
    show("全零输入 + RMSNorm", f"全零 = {bool((y_zero == 0).all())}", "eps 兜住分母")
    assert torch.isnan(naive).any()
    assert torch.isfinite(y_zero).all() and (y_zero == 0).all()
    # 半精度输入在内部用 fp32 算，再转回原 dtype：和 HF 的 LlamaRMSNorm 一致
    xh = torch.randn(2, 8, C).to(torch.bfloat16)
    yh = rms(xh)
    show("bf16 输入 → 输出 dtype", yh.dtype, "内部 fp32 计算，再转回")
    assert yh.dtype == torch.bfloat16 and torch.isfinite(yh).all()
    print("[OK] eps 在开方前加进 mean(x²)：全零输入输出仍是 0 而不是 NaN")

    print("\n--- 6. 梯度检查 ---")
    xg = torch.randn(2, 8, C, requires_grad=True)
    # 用随机权重加权求和：如果只用 mean(y²)，归一化后它恒等于 1，梯度是 0
    w = torch.randn(2, 8, C)
    (rms(xg) * w).sum().backward()
    assert xg.grad is not None and xg.grad.shape == xg.shape
    assert torch.isfinite(xg.grad).all()
    show("x.grad 形状", xg.grad, "与输入同形 (B,T,C)")
    show("max|x.grad|", f"{xg.grad.abs().max().item():.4f}")

    xz = torch.zeros(2, C, requires_grad=True)
    rms(xz).sum().backward()
    assert torch.isfinite(xz.grad).all()
    print("[OK] 反向传播走通：梯度形状正确、全部有限（全零输入也不炸）")

    print("\n--- 7. 在 block 里怎么用 ---")
    print("  GPT-2:  self.attn_norm = nn.LayerNorm(C)   # 2C 个参数")
    print("  Llama:  self.attn_norm = RMSNorm(C, eps=1e-5)  # C 个参数")
    print("  结构不变（还是 Pre-Norm：先归一化再进子层），只把 LN 换成 RMSNorm。")

    print("\n一句话记住：")
    print("  RMSNorm 不重新居中、参数少一半、少一次减均值；小规模实测未必更快。")
    print("\n[OK] 全部检查通过")


if __name__ == "__main__":
    main()
