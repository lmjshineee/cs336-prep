"""Step 3: 学习式位置嵌入 → RoPE 旋转位置编码。

GPT-2 的做法：
    pos_emb = nn.Embedding(block_size, n_embd)   # 一张 T × C 的查找表
    x = token_emb(idx) + pos_emb(arange(T))      # 位置信息「加」进输入

Llama 的做法：
    q = apply_rope(q_proj(x), cos, sin)          # 只旋转 q 和 k，v 不转
    k = apply_rope(k_proj(x), cos, sin)
    scores = q @ k.transpose(-2, -1)             # 位置信息藏在「旋转角」里

为什么要改：
    1) 问题一：学习式位置嵌入有 T·C 个参数，而且是一张定长的表 ——
       查询位置 > T 直接 IndexError，训练长度之外完全失效。
    2) 问题二：它把位置「加」进输入，模型只看得到绝对位置；
       但语言里起作用的是相对距离（「猫 追 狗」和「猫 昨天 追 狗」）。
    3) 方案：RoPE 用旋转代替加法。位置 m 的向量转 m·θ 弧度，
       两个向量的点积就只依赖 m-n —— 相对位置，且不占一个参数。
    4) 代价：每个位置都要算 sin/cos（纯公式，想算多长都行）；
       注意旋转只作用于 q、k，value 不参与。

关键公式：
    θ_i = 10000^(-2i/D),  i = 0 .. D/2-1     # D = head_dim，每对通道一个频率
    第 m 个位置、第 i 对通道转 m·θ_i 弧度（一个二维旋转）
    性质：<R(m)q, R(n)k> = <R(m-n)q, k>      # 只依赖相对位移 m-n
    保范数：|R(m)q| = |q|   # 旋转是正交变换（只转方向、不改长度）

和上一步的 diff（只列变化的行）：
    - self.pos_emb = nn.Embedding(block_size, n_embd)
    - x = token_emb(idx) + self.pos_emb(arange(T))
    + self.register_buffer("rope_cos", cos, persistent=False)   # 不占参数
    + self.register_buffer("rope_sin", sin, persistent=False)
    + q = apply_rope(q, cos, sin)
    + k = apply_rope(k, cos, sin)
"""

import math
import os
import unicodedata
from pathlib import Path

import torch
from torch import nn

# matplotlib 默认缓存目录可能不可写，先指到仓库里的 .tmp/mpl；Agg 后端不弹窗
_MPL_DIR = Path(__file__).resolve().parent.parent / ".tmp" / "mpl"
_MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_DIR))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """x: (..., D) -> (..., D)。把前一半和后一半配对：(x1, x2) -> (-x2, x1)。"""
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    return torch.cat((-x2, x1), dim=-1)


def build_rope_cache(head_dim: int, max_len: int, theta: float = 10000.0,
                     device: torch.device | None = None
                     ) -> tuple[torch.Tensor, torch.Tensor]:
    """返回 (cos, sin)，形状都是 (max_len, head_dim)，float32。

    inv_freq[i] = theta^(-2i/D)：第 i 对通道每前进一个位置转过 inv_freq[i] 弧度。
    cos/sin 的后一半复制了前一半，这样 rotate_half 的前后两半能用同一组角度。
    这段和 HuggingFace 的 LlamaRotaryEmbedding 完全一致（step08 要逐位对齐）。
    """
    idx = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (theta ** (idx / head_dim))
    freqs = torch.outer(torch.arange(max_len, dtype=torch.float32, device=device),
                        inv_freq)                        # (max_len, D/2)
    emb = torch.cat((freqs, freqs), dim=-1)              # (max_len, D)
    return emb.cos(), emb.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor,
               sin: torch.Tensor) -> torch.Tensor:
    """x: (B, H, T, D)；cos/sin: (T, D)，内部 unsqueeze 成 (1,1,T,D)。"""
    cos = cos.to(x.dtype).unsqueeze(0).unsqueeze(0)
    sin = sin.to(x.dtype).unsqueeze(0).unsqueeze(0)
    return x * cos + rotate_half(x) * sin


def interleave(x: torch.Tensor) -> torch.Tensor:
    """half-split 的通道顺序 → interleaved 的通道顺序：前一半与后一半交错拼。"""
    half = x.shape[-1] // 2
    return torch.stack([x[..., :half], x[..., half:]], dim=-1).flatten(-2)


def rotate_interleaved(x: torch.Tensor) -> torch.Tensor:
    """interleaved 版 rotate_half：相邻两维 (2i, 2i+1) 配对，(x1, x2) -> (-x2, x1)。"""
    out = torch.empty_like(x)
    out[..., 0::2] = -x[..., 1::2]
    out[..., 1::2] = x[..., 0::2]
    return out


def make_figure(cos: torch.Tensor, sin: torch.Tensor, out_path: Path) -> None:
    """(a) 分数随相对距离的变化（竖线标训练长度 64）；(b) 长度 256 的分数矩阵。"""
    d = cos.shape[-1]
    gen = torch.Generator().manual_seed(0)
    u = torch.randn(64, d, generator=gen)
    u = u / u.norm(dim=-1, keepdim=True)          # 64 个随机单位方向

    # (a) 每个偏移 off，用同一组方向算 <R(off)u, u> 再取平均，看包络如何衰减
    offsets = list(range(256))
    curve = []
    for off in offsets:
        rot = apply_rope(u.view(64, 1, 1, d), cos[off:off + 1], sin[off:off + 1])
        curve.append((rot.view(64, d) * u).sum(-1).mean().item())

    # (b) 一个固定方向，长度 256 的分数矩阵：只由 |i-j| 决定，所以是衰减的带状
    v = u[0].view(1, 1, 1, d).expand(1, 1, 256, d)
    q_rot = apply_rope(v, cos[:256], sin[:256])
    mat = (q_rot @ q_rot.transpose(-2, -1))[0, 0].numpy()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.plot(offsets, curve, color="tab:blue")
    ax.axvline(64, color="black", linestyle="--", linewidth=1.2,
               label="trained length 64")
    ax.set_xlabel("relative offset (m - n)")
    ax.set_ylabel("attention score")
    ax.set_title("(a) score vs relative offset")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    im = ax.imshow(mat, cmap="RdBu_r", vmin=0, vmax=1, origin="lower")
    ax.set_xlabel("key position")
    ax.set_ylabel("query position")
    ax.set_title("(b) score matrix at length 256 (built from length-64 RoPE)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="score")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    torch.manual_seed(0)
    D = 16                    # head_dim（toy 配置 n_embd=64 / n_head=4）
    theta = 10000.0

    print("\n--- 1. 学习式位置嵌入的两个毛病 ---")
    learned = nn.Embedding(64, 64)        # 训练长度 T=64，隐藏维度 C=64
    show("学习式位置嵌入参数", sum(p.numel() for p in learned.parameters()),
         "T·C = 64×64，纯查表")
    try:
        learned(torch.tensor([200]))
    except IndexError as exc:
        print(f"  查位置 200（训练长度只有 64）→ {type(exc).__name__}: {exc}")
    else:
        raise AssertionError("位置 200 超出表长，应该报 IndexError")
    print("  毛病二：位置 id 超出表长就查不到，训练长度之外完全没法用。")

    print("\n--- 2. RoPE：把「加位置向量」换成「按位置旋转 q、k」---")
    print("  θ_i = 10000^(-2i/D)，第 m 个位置第 i 对通道旋转 m·θ_i 弧度")
    print("  q_rot = R(m·θ) q，k_rot = R(n·θ) k，v 不旋转")
    cos, sin = build_rope_cache(D, 320, theta=theta)
    show("head_dim D", D, "toy: n_embd 64 / n_head 4")
    show("cos", cos, "(max_len, D) float32，不占参数")
    show("sin", sin, "(max_len, D) float32，不占参数")
    inv_freq = 1.0 / (theta ** (torch.arange(0, D, 2, dtype=torch.float32) / D))
    show("θ_0 ... θ_3", " ".join(f"{v:.3f}" for v in inv_freq[:4].tolist()),
         "越靠后的通道转得越慢")
    show("cos 前后两半相同",
         str(bool(torch.allclose(cos[:, : D // 2], cos[:, D // 2:]))),
         "half-split 配对的前提")
    assert torch.allclose(cos[:, : D // 2], cos[:, D // 2:])
    print("  rotate_half 把 (第 i 维, 第 i+D/2 维) 配成一对，用同一组角度旋转。")

    print("\n--- 3. 核心性质：整段序列平移，注意力分数矩阵不变 ---")
    T = 64
    q = torch.randn(1, 1, T, D)          # 先看单头 (B, H, T, D)
    k = torch.randn(1, 1, T, D)
    scale = math.sqrt(D)

    def scores_of(q_: torch.Tensor, k_: torch.Tensor) -> torch.Tensor:
        return (q_ @ k_.transpose(-2, -1)) / scale

    base = scores_of(apply_rope(q, cos[:T], sin[:T]),
                     apply_rope(k, cos[:T], sin[:T]))
    shift = scores_of(apply_rope(q, cos[37:37 + T], sin[37:37 + T]),
                      apply_rope(k, cos[37:37 + T], sin[37:37 + T]))
    diff = (base - shift).abs().max().item()
    show("平移 +37 后分数最大差", f"{diff:.2e}")
    assert diff < 1e-5
    print("[OK] 整个序列位置 +37，q@kᵀ 完全不变（RoPE 只认相对位置）")

    # 所有位置放同一个 q/k：此时分数只可能由相对距离 i-j 决定
    q_same = torch.randn(1, 1, 1, D).expand(1, 1, T, D)
    k_same = torch.randn(1, 1, 1, D).expand(1, 1, T, D)
    s = scores_of(apply_rope(q_same, cos[:T], sin[:T]),
                  apply_rope(k_same, cos[:T], sin[:T]))
    delta = torch.arange(T)[:, None] - torch.arange(T)[None, :]
    spread = 0.0
    for d in range(-(T - 1), T):
        group = s[0, 0][delta == d]
        spread = max(spread, (group.max() - group.min()).item())
    show("同一相对距离内最大差", f"{spread:.2e}", "分数只是 i-j 的函数")
    assert spread < 1e-5
    print("[OK] 按 i-j 分组后组内完全一致：分数只依赖相对距离")

    print("\n--- 4. 保范数：旋转不改变向量长度 ---")
    dnorm = (q.norm(dim=-1) - apply_rope(q, cos[:T], sin[:T]).norm(dim=-1))
    show("apply_rope 前后模长最大差", f"{dnorm.abs().max().item():.2e}")
    assert dnorm.abs().max().item() < 1e-5
    print("[OK] 每个 token 的 q 模长不变（正交 = 只转方向、不改长度）")

    print("\n--- 5. 两种配对写法：half-split vs interleaved ---")
    # interleaved 的 cos/sin：同一频率重复两遍，对应相邻两维 (2i, 2i+1)
    cos_i = cos[:T, : D // 2].repeat_interleave(2, dim=-1)
    sin_i = sin[:T, : D // 2].repeat_interleave(2, dim=-1)

    # 正确做法：先把 half-split 的通道顺序重排成 interleaved，再相邻配对旋转
    q_i = interleave(q) * cos_i + rotate_interleaved(interleave(q)) * sin_i
    k_i = interleave(k) * cos_i + rotate_interleaved(interleave(k)) * sin_i
    d_score = (scores_of(q_i, k_i) - base).abs().max().item()

    # 反例：不重排、直接按相邻配对旋转，就是另一个东西了
    q_raw = q * cos_i + rotate_interleaved(q) * sin_i
    k_raw = k * cos_i + rotate_interleaved(k) * sin_i
    d_raw = (scores_of(q_raw, k_raw) - base).abs().max().item()
    show("重排通道后 interleaved", f"{d_score:.2e}", "与 half-split 等价")
    show("不重排直接 interleaved", f"{d_raw:.2e}", "反例：通道配错了")
    assert d_score < 1e-5 and d_raw > 1e-2
    print("  两种写法只在「先把通道重排成对应顺序」之后才等价；模型的投影权重")
    print("  是配套的，混用就错（step08 会用真实 logits 演示这一点）。")
    print("  我们选 half-split：HuggingFace 的 Llama 就是这么写的，step08 要对齐。")

    print("\n--- 6. 外推：长度 256 的分数矩阵照算不误 ---")
    T_long = 256
    ql = torch.randn(1, 1, T_long, D)
    kl = torch.randn(1, 1, T_long, D)
    ql_rot = apply_rope(ql, cos[:T_long], sin[:T_long])
    s_long = scores_of(ql_rot, apply_rope(kl, cos[:T_long], sin[:T_long]))
    show("长度 256 的分数矩阵", s_long, "没有越界，也没有 NaN")
    assert torch.isfinite(s_long).all()
    assert (ql.norm(dim=-1) - ql_rot.norm(dim=-1)).abs().max().item() < 1e-5
    dlong = torch.arange(T_long)[:, None] - torch.arange(T_long)[None, :]
    near = s_long[0, 0][dlong.abs() < 64].abs().mean().item()
    far = s_long[0, 0][dlong.abs() >= 64].abs().mean().item()
    show("|offset| < 64 平均 |分数|", f"{near:.3f}")
    show("|offset| ≥ 64 平均 |分数|", f"{far:.3f}", "仍有定义，量级相当")
    print("[OK] RoPE 靠公式：训练长度 64 之外也能算，不像查表会越界")
    print("  注意：能算 ≠ 效果好；远距离外推质量要靠 NTK/YaRN 之类缩放，这里只验证不崩。")

    print("\n--- 7. 对 KV cache 友好：存已旋转的 k ---")
    start = 200
    q_new = torch.randn(1, 1, 1, D)
    rot_alone = apply_rope(q_new, cos[start:start + 1], sin[start:start + 1])
    q_seq = torch.randn(1, 1, start + 1, D)
    q_seq[..., start:start + 1, :] = q_new
    rot_in_seq = apply_rope(q_seq, cos[:start + 1], sin[:start + 1])
    diff_cache = (rot_alone - rot_in_seq[..., start:start + 1, :]).abs().max().item()
    show("单 token 旋转 vs 放进长序列", f"{diff_cache:.2e}", "历史 k 不用重算")
    assert diff_cache < 1e-6
    print("  decode 新 token 时只需 cos[start_pos:start_pos+T]；历史 k 已旋转好，原样复用。")

    print("\n--- 8. 画图 ---")
    out_path = Path(__file__).resolve().parent / "figs" / "step03_rope.png"
    make_figure(cos, sin, out_path)
    print(f"图片已保存：{out_path}")
    print("  图 (a) 的上下波动不是 bug：它是不同频率 θ_i 叠加的拍频；")
    print("  关键不变量是：分数只取决于相对距离 m-n。")
    assert out_path.is_file()

    print("\n一句话记住：")
    print("  RoPE 用「按位置旋转 q、k」代替位置查表：不占参数，分数只认相对距离。")
    print("\n[OK] 全部检查通过")


if __name__ == "__main__":
    main()
