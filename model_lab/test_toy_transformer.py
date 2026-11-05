"""test_toy_transformer.py —— `toy_transformer` 的双模式测试（覆盖 SPEC §6 的 43 项）。

两种跑法都支持：

    # 1) 直接跑（不需要 pytest；打印 PASS/FAIL/SKIP 汇总，有失败时 exit 1）
    .venv/bin/python model_lab/test_toy_transformer.py

    # 2) pytest（如果临时环境里有）
    python -m pytest model_lab/test_toy_transformer.py -q

测试的分层（与 SPEC §6 对应）：

    A. 手算/公式（10 项）：softmax、sdpa、rmsnorm、rope、silu/swiglu
    B. 形状/接口（8 项）：每个模块的 in/out 形状、d_ff 默认值、越界报错
    C. 性质（8 项）：因果性、RoPE 只转 q/k、位置平移、尺度不变、tie、反传
    D. 初始化（4 项）：trunc_normal 的 std 与截断、无 bias、state_dict 键名
    E. 官方快照 parity（12 项 / 13 个 npz）：与 `assignment1-basics` 的快照逐元素对齐
    F. 端到端（1 项）：小 LM 反传若干步，loss 下降

另外有 1 个**回归测试**不计入 43 项：`test_C27_mha_accepts_batch_token_positions`
（MHA 的 token_positions 支持 (T,)/(1,T)/(B,T)/(1,1,T)，曾因只判断 `ndim == x.ndim`
而在 (B,T) 上报错、(1,T) 恰好广播成功把 bug 掩盖）。

关于 parity：官方仓库默认路径 `/Users/yunni/Projects/assignment1-basics`，可用环境变量
`A1_REPO` 覆盖。**只读**，不写不改。找不到时打印 SKIP（不报错退出），每个 parity 都会
打印 maxdiff，方便抄进 LAB_NOTES / 教程。
"""

from __future__ import annotations

import math
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

# 本文件的所有测试都很小，但统一限制线程数，避免和别的教程脚本抢 CPU
torch.set_num_threads(2)

# 允许 `python model_lab/test_toy_transformer.py` 直接跑（此时 model_lab 不在 sys.path 上）
sys.path.insert(0, str(Path(__file__).resolve().parent))

import toy_transformer as tt  # noqa: E402

# ---------------------------------------------------------------------------
# 0. 基础设施：pytest 兼容的 skip、断言工具、官方 fixture 加载
# ---------------------------------------------------------------------------
A1_REPO = Path(os.environ.get("A1_REPO", "/Users/yunni/Projects/assignment1-basics"))

try:  # pytest 可选：没有 pytest 时用自定义异常实现「跳过」
    import pytest as _pytest
except Exception:  # pragma: no cover - 取决于环境
    _pytest = None


class SkipTest(Exception):
    """直接运行时用来表示「这个测试跳过」。"""


def _skip(reason: str):
    """跳过当前测试：pytest 模式转发给 pytest.skip，直接运行模式抛 SkipTest。"""
    if _pytest is not None:
        _pytest.skip(reason)
    raise SkipTest(reason)


_CACHE: dict[str, object] = {}


def _official_state_dict() -> dict[str, Tensor]:
    """官方参考实现的 state_dict（去掉 torch.compile 留下的 `_orig_mod.` 前缀）。"""
    if "state_dict" not in _CACHE:
        path = A1_REPO / "tests/fixtures/ts_tests/model.pt"
        if not path.exists():
            _skip(f"缺少官方权重：{path}（可用 A1_REPO 指定仓库路径）")
        raw = torch.load(path, map_location="cpu", weights_only=True)
        _CACHE["state_dict"] = {
            k.replace("_orig_mod.", ""): v for k, v in raw.items()
        }
    return _CACHE["state_dict"]  # type: ignore[return-value]


def _official_config() -> dict:
    """官方参考模型的几何配置（vocab/ctx/d_model/...）。"""
    if "config" not in _CACHE:
        path = A1_REPO / "tests/fixtures/ts_tests/model_config.json"
        if not path.exists():
            _skip(f"缺少官方配置：{path}")
        import json

        _CACHE["config"] = json.loads(path.read_text(encoding="utf-8"))
    return _CACHE["config"]  # type: ignore[return-value]


def _snapshot(name: str) -> Tensor:
    """加载官方快照 `tests/_snapshots/{name}.npz` 里的 `array`。"""
    key = f"snapshot:{name}"
    if key not in _CACHE:
        path = A1_REPO / "tests" / "_snapshots" / f"{name}.npz"
        if not path.exists():
            _skip(f"缺少官方快照：{path}")
        _CACHE[key] = torch.from_numpy(np.load(path)["array"])
    return _CACHE[key]  # type: ignore[return-value]


def _official_subset(prefix: str) -> dict[str, Tensor]:
    """取官方 state_dict 里以 `prefix` 开头的键，并去掉前缀（给单个模块 load 用）。"""
    return {
        k[len(prefix) :]: v
        for k, v in _official_state_dict().items()
        if k.startswith(prefix)
    }


def _maxdiff(actual: Tensor, expected: Tensor) -> float:
    """两个张量的最大绝对误差（统一按 float32 比，避免 dtype 干扰）。"""
    return (actual.detach().float() - expected.detach().float()).abs().max().item()


def _assert_close(
    actual: Tensor,
    expected: Tensor,
    *,
    atol: float,
    rtol: float = 1e-4,
    label: str = "",
) -> float:
    """断言两 tensors 接近，返回 maxdiff（parity 测试靠它打印数字）。"""
    actual = actual.detach().cpu()
    expected = expected.detach().cpu()
    if actual.shape != expected.shape:
        raise AssertionError(f"{label}: 形状不一致 {tuple(actual.shape)} vs {tuple(expected.shape)}")
    assert actual.dtype == expected.dtype, f"{label}: dtype 不一致 {actual.dtype} vs {expected.dtype}"
    diff = _maxdiff(actual, expected)
    if label:
        print(f"    {label}: maxdiff={diff:.3e}  (atol={atol:g}, rtol={rtol:g})")
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol, msg=label or "assert_close")
    return diff


def _assert_raises(exc_type: type[BaseException], fn, label: str = "") -> None:
    """断言 `fn()` 抛 `exc_type`（避免依赖 pytest.raises，双模式都能用）。"""
    try:
        fn()
    except exc_type:
        return
    except Exception as exc:  # noqa: BLE001 - 要把「抛错但类型不对」也报出来
        raise AssertionError(
            f"{label}: 期望 {exc_type.__name__}，实际抛了 {type(exc).__name__}: {exc}"
        ) from exc
    raise AssertionError(f"{label}: 期望抛 {exc_type.__name__}，但没有抛任何异常")


# ---------------------------------------------------------------------------
# 固定输入（与官方 conftest.py 的 fixture 逐位一致；每个都单独 seed）
# ---------------------------------------------------------------------------
def _x() -> Tensor:  # in_embeddings
    torch.manual_seed(4)
    return torch.randn(4, 12, 64)


def _q() -> Tensor:
    torch.manual_seed(1)
    return torch.randn(4, 12, 64)


def _k() -> Tensor:
    torch.manual_seed(2)
    return torch.randn(4, 16, 64)


def _v() -> Tensor:
    torch.manual_seed(3)
    return torch.randn(4, 16, 64)


def _mask() -> Tensor:
    torch.manual_seed(5)
    return torch.randn(4, 12, 16) > 0.5


def _idx() -> Tensor:
    torch.manual_seed(6)
    return torch.randint(0, 10_000, (4, 12))


def _heads(t: Tensor, num_heads: int) -> Tensor:
    """`(..., T, C)` -> `(..., H, T, D)`（测试里复用手写参考用）。"""
    d_head = t.shape[-1] // num_heads
    return t.reshape(*t.shape[:-1], num_heads, d_head).transpose(-3, -2)


def _causal_mask(seq_len: int) -> Tensor:
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))


# ===========================================================================
# A. 手算 / 公式（SPEC §6 A：第 1–10 项）
# ===========================================================================
def test_A01_softmax_hand_computed_and_negative_dim():
    """[1] softmax 每行和为 1、与手算一致、负 dim 正确。"""
    x = torch.tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 1.0]])

    out = tt.softmax(x, dim=-1)
    # 每行和为 1
    torch.testing.assert_close(out.sum(dim=-1), torch.ones(3), atol=1e-6, rtol=0)
    # 与手算公式一致：exp(x - max) / Σ exp(x - max)
    manual = torch.exp(x - x.max(dim=-1, keepdim=True).values)
    manual = manual / manual.sum(dim=-1, keepdim=True)
    _assert_close(out, manual, atol=1e-6)

    # 负 dim 与正 dim 等价：-1 等价于 2，-2 等价于 1
    y = torch.randn(2, 3, 4)
    _assert_close(tt.softmax(y, -1), tt.softmax(y, 2), atol=0.0)
    _assert_close(tt.softmax(y, -2), tt.softmax(y, 1), atol=0.0)
    _assert_close(tt.softmax(y, 0), torch.softmax(y, 0), atol=1e-6)
    # 沿 dim=0 归一：每列和为 1
    torch.testing.assert_close(tt.softmax(y, 0).sum(dim=0), torch.ones(3, 4), atol=1e-6, rtol=0)


def test_A02_softmax_numerical_stability():
    """[2] softmax 数值稳定：大输入不溢出、整体 +1e4 结果不变、无 inf/nan。"""
    # (a) 数值很大的输入：不先减 max 的话 exp 会直接 inf，减了就不该出现 inf/nan
    torch.manual_seed(0)
    x = torch.randn(8, 16) * 1e4
    assert not torch.isfinite(torch.exp(x)).all(), "这个测试的输入应该能让裸 exp 溢出（否则测不出东西）"
    out = tt.softmax(x, dim=-1)
    assert torch.isfinite(out).all(), "softmax 输出出现 inf/nan"
    torch.testing.assert_close(out.sum(-1), torch.ones(8), atol=1e-6, rtol=0)
    # 与 PyTorch 的稳定实现对照（两者都减 max，应当几乎逐位一致）
    _assert_close(out, torch.softmax(x, dim=-1), atol=1e-6)

    # (b) 整体 +1e4 结果不变。用 0.25 的倍数，保证 x 与 x+1e4 在 fp32 里都能精确表示
    base = torch.arange(16, dtype=torch.float32).mul(0.25).sub(2.0).unsqueeze(0)
    shifted = base + 1e4
    assert torch.equal(shifted - 1e4, base), "测试前提：+1e4 再减回去必须精确"
    _assert_close(tt.softmax(shifted, -1), tt.softmax(base, -1), atol=1e-6)


def test_A03_softmax_shift_invariance():
    """[3] softmax 平移不变性：每行加不同的常数，结果不变。"""
    torch.manual_seed(1)
    x = torch.randn(6, 10) * 3.0
    shift = torch.randn(6, 1) * 20.0  # 每一行加自己的常数
    _assert_close(tt.softmax(x + shift, dim=-1), tt.softmax(x, dim=-1), atol=1e-4)


def test_A04_sdpa_matches_manual_formula_3d():
    """[4] sdpa 3D 与手写公式一致（且 d_v 可以 != d_k）。"""
    torch.manual_seed(2)
    q = torch.randn(3, 5, 16)  # (B, T_q, d_k)
    k = torch.randn(3, 7, 16)  # (B, T_k, d_k)
    v = torch.randn(3, 7, 8)  # (B, T_k, d_v)，故意和 d_k 不同
    scores = torch.einsum("bqd,bkd->bqk", q, k) / math.sqrt(q.shape[-1])
    manual = torch.einsum("bqk,bkd->bqd", torch.softmax(scores, dim=-1), v)

    out = tt.scaled_dot_product_attention(q, k, v)
    assert out.shape == (3, 5, 8), out.shape
    _assert_close(out, manual, atol=1e-6)


def test_A05_sdpa_mask_semantics():
    """[5] sdpa mask 语义：False 位置权重为 0，True 位置权重和为 1。"""
    torch.manual_seed(3)
    q = torch.randn(2, 5, 4)
    k = torch.randn(2, 5, 4)
    v = torch.randn(2, 5, 4)

    # (a) 极端情况：每行只允许看 key 0，输出必须恰好等于 v[:, 0, :]
    mask = torch.zeros(5, 5, dtype=torch.bool)
    mask[:, 0] = True
    out = tt.scaled_dot_product_attention(q, k, v, mask)
    expected = v[:, 0, :].unsqueeze(1).expand(2, 5, 4)
    _assert_close(out, expected, atol=1e-6)

    # (b) 一般 mask：False 位置权重恰好为 0，True 位置权重和为 1
    torch.manual_seed(4)
    mask = torch.randn(5, 5) > 0.2
    mask[:, 0] = True  # 保证每行至少一个可见 key
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    scores = scores.masked_fill(~mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    assert torch.all(probs[..., ~mask] == 0), "被 mask 的位置权重必须恰好是 0"
    torch.testing.assert_close(probs.sum(-1), torch.ones(2, 5), atol=1e-6, rtol=0)
    _assert_close(tt.scaled_dot_product_attention(q, k, v, mask), probs @ v, atol=1e-6)

    # (c) mask 需要能广播：3D 输入 + (T_q, T_k) 的 2D mask
    #     （官方 fixture 是 (B, T_q, T_k)，但 2D mask 也应该能用）
    out_2d = tt.scaled_dot_product_attention(q, k, v, mask)
    _assert_close(out_2d, tt.scaled_dot_product_attention(q, k, v, mask.expand(2, 5, 5)), atol=0.0)


def test_A06_sdpa_4d_matches_per_head_loop():
    """[6] sdpa 4D 与逐 head 的 3D 循环一致；3D mask 可广播到 4D。"""
    torch.manual_seed(5)
    q3 = torch.randn(8, 6, 16)
    k3 = torch.randn(8, 7, 16)
    v3 = torch.randn(8, 7, 16)
    mask3 = torch.randn(8, 6, 7) > 0.3
    mask3[:, :, 0] = True  # 保证每行至少一个 True

    # 官方 conftest 的 rearrange："(batch head) seq d -> batch head seq d"，head=2
    q4 = q3.reshape(4, 2, 6, 16)
    k4 = k3.reshape(4, 2, 7, 16)
    v4 = v3.reshape(4, 2, 7, 16)
    mask4 = mask3.reshape(4, 2, 6, 7)

    out4 = tt.scaled_dot_product_attention(q4, k4, v4, mask4)
    assert out4.shape == (4, 2, 6, 16)

    expected = torch.empty(4, 2, 6, 16)
    for b in range(4):
        for h in range(2):
            expected[b, h] = tt.scaled_dot_product_attention(q4[b, h], k4[b, h], v4[b, h], mask4[b, h])
    _assert_close(out4, expected, atol=1e-6)

    # mask 广播：2D mask (T_q, T_k) 与手工 expand 成 (B, H, T_q, T_k) 等价
    mask2d = mask3[0]  # (6, 7)，所有 batch/head 共用同一个 mask
    out4_broadcast = tt.scaled_dot_product_attention(q4, k4, v4, mask2d)
    out4_explicit = tt.scaled_dot_product_attention(
        q4, k4, v4, mask2d.expand(4, 2, 6, 7)
    )
    _assert_close(out4_broadcast, out4_explicit, atol=0.0)


def test_A07_rmsnorm_matches_formula():
    """[7] rmsnorm 与公式一致（随机 gain，带 eps）。"""
    torch.manual_seed(6)
    d_model, eps = 64, 1e-5
    norm = tt.RMSNorm(d_model, eps=eps)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(d_model))  # 随机 gain，验证不是「只对全 1 成立」

    x = torch.randn(4, 12, d_model) * 2.0
    expected = x / torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps) * norm.weight
    _assert_close(norm(x), expected, atol=1e-6)

    # 输出每行的 RMS（除以 gain 后）应当约等于 1
    rms = norm(x).div(norm.weight).pow(2).mean(dim=-1).sqrt()
    torch.testing.assert_close(rms, torch.ones_like(rms), atol=1e-5, rtol=0)


def test_A08_rope_2d_hand_computed():
    """[8] rope 二维手算旋转正确（交错配对，含位置 0 是恒等）。"""
    theta, d_k, seq_len = 10000.0, 4, 3
    rope = tt.RotaryPositionalEmbedding(theta, d_k, max_seq_len=seq_len)
    x = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [-1.0, 0.5, 2.0, -3.0],
        ]
    )
    pos = torch.arange(seq_len)
    out = rope(x, pos)
    assert out.shape == x.shape

    # 位置 0 的角度全是 0 -> 恒等变换
    _assert_close(out[0], x[0], atol=1e-6)

    # 位置 1：k=0 的角度是 1*theta^0 = 1，k=1 的角度是 1*theta^(-2/4) = 0.01
    a0, a1 = 1.0, theta ** (-2.0 / d_k)
    c0, s0 = math.cos(a0), math.sin(a0)
    c1, s1 = math.cos(a1), math.sin(a1)
    x0, x1, x2, x3 = x[1].tolist()
    expected = torch.tensor(
        [x0 * c0 - x1 * s0, x0 * s0 + x1 * c0, x2 * c1 - x3 * s1, x2 * s1 + x3 * c1]
    )
    _assert_close(out[1], expected, atol=1e-6)

    # 位置 2：角度 = 2 * (theta^(-2k/d_k))，即「角度与位置成正比」
    a0, a1 = 2.0, 2.0 * theta ** (-2.0 / d_k)
    c0, s0 = math.cos(a0), math.sin(a0)
    c1, s1 = math.cos(a1), math.sin(a1)
    x0, x1, x2, x3 = x[2].tolist()
    expected = torch.tensor(
        [x0 * c0 - x1 * s0, x0 * s0 + x1 * c0, x2 * c1 - x3 * s1, x2 * s1 + x3 * c1]
    )
    _assert_close(out[2], expected, atol=1e-6)


def test_A09_rope_relative_position_only():
    """[9] rope 相对性：`dot(R_i q, R_j k)` 只依赖 `i-j`。"""
    torch.manual_seed(7)
    d_k, seq_len = 8, 6
    rope = tt.RotaryPositionalEmbedding(10000.0, d_k, max_seq_len=seq_len + 8)
    q0 = torch.randn(d_k)
    k0 = torch.randn(d_k)
    # 所有位置用同一个底层向量：旋转后的点积只应该跟位置差有关
    q = q0.expand(1, seq_len, d_k)
    k = k0.expand(1, seq_len, d_k)
    pos = torch.arange(seq_len)
    q_rot = rope(q, pos)[0]  # (T, d_k)
    k_rot = rope(k, pos)[0]

    dots = q_rot @ k_rot.transpose(-2, -1)  # (T, T)，dots[i, j] = <R_i q, R_j k>
    for delta in range(seq_len):
        # 同一条对角线上（i-j = delta）的点积必须完全相同
        vals = [dots[i, i + delta] for i in range(seq_len - delta)]
        ref = vals[0]
        for value in vals[1:]:
            torch.testing.assert_close(value, ref, atol=1e-5, rtol=0, msg=f"delta={delta}")

    # 与「把 q/k 同时平移 s」等价：<R_{i+s} q, R_{j+s} k> == <R_i q, R_j k>
    shift = 5
    dots_shift = rope(q, pos + shift)[0] @ rope(k, pos + shift)[0].transpose(-2, -1)
    _assert_close(dots_shift, dots, atol=1e-5)

    # 与「只旋转 j-i」的形式一致：<R_i q, R_j k> == <q, R_{j-i} k>
    for i in (0, 1, 2):
        for j in (i, i + 1, seq_len - 1):
            left = dots[i, j]
            rotated_k = rope(k0.reshape(1, 1, d_k), torch.tensor([j - i]))[0, 0]
            right = torch.dot(q0, rotated_k)
            torch.testing.assert_close(left, right, atol=1e-5, rtol=0, msg=f"i={i}, j={j}")


def test_A10_silu_and_swiglu_match_formula():
    """[10] silu 与 F.silu 一致；swiglu 与手写公式（含显式 d_ff）一致。"""
    torch.manual_seed(8)
    x = torch.randn(3, 5, 16) * 3.0
    _assert_close(tt.silu(x), F.silu(x), atol=1e-6)
    _assert_close(tt.silu(x), x * torch.sigmoid(x), atol=0.0)

    d_model, d_ff = 16, 40
    ffn = tt.SwiGLU(d_model, d_ff)
    with torch.no_grad():  # 随机权重，避免「实现刚好等于初始化值」的巧合
        for p in ffn.parameters():
            p.normal_(0.0, 0.1)
    manual = ffn.w2(F.silu(ffn.w1(x)) * ffn.w3(x))
    _assert_close(ffn(x), manual, atol=1e-6)


# ===========================================================================
# B. 形状 / 接口（SPEC §6 B：第 11–18 项）
# ===========================================================================
def test_B11_linear_weight_layout_and_forward():
    """[11] Linear.weight 形状 (out,in)、无 bias、forward == x @ W.T。"""
    layer = tt.Linear(8, 5)
    assert layer.weight.shape == (5, 8), layer.weight.shape
    assert isinstance(layer.weight, nn.Parameter)
    names = {n for n, _ in layer.named_parameters()}
    assert names == {"weight"}, names  # 没有 bias
    assert not hasattr(layer, "bias")

    with torch.no_grad():
        layer.weight.copy_(torch.randn(5, 8))
    x = torch.randn(2, 3, 8)
    out = layer(x)
    assert out.shape == (2, 3, 5), out.shape
    _assert_close(out, x @ layer.weight.transpose(-2, -1), atol=0.0)

    # 二维输入也支持：(4, 8) -> (4, 5)
    row_input = torch.randn(4, 8)
    assert layer(row_input).shape == (4, 5)

    # device / dtype 参数透传
    layer64 = tt.Linear(4, 2, dtype=torch.float64)
    assert layer64.weight.dtype == torch.float64
    assert layer64(torch.randn(3, 4, dtype=torch.float64)).dtype == torch.float64


def test_B12_embedding_lookup_arbitrary_rank():
    """[12] Embedding 查表等于 weight[ids]；1D/2D/3D ids 都支持。"""
    emb = tt.Embedding(13, 5)
    weight = emb.weight

    ids1 = torch.tensor([0, 3, 7, 12])
    out1 = emb(ids1)
    assert out1.shape == (4, 5)
    _assert_close(out1, weight[ids1], atol=0.0)

    ids2 = torch.tensor([[0, 1], [2, 3], [4, 5]])
    out2 = emb(ids2)
    assert out2.shape == (3, 2, 5)
    _assert_close(out2, weight[ids2], atol=0.0)

    ids3 = torch.randint(0, 13, (2, 3, 4))
    out3 = emb(ids3)
    assert out3.shape == (2, 3, 4, 5)
    _assert_close(out3, weight[ids3], atol=0.0)


def test_B13_rmsnorm_preserves_dtype():
    """[13] RMSNorm 保持输入 dtype（bf16/fp16 进、同 dtype 出），且数值合理。"""
    torch.manual_seed(9)
    x_fp32 = torch.randn(2, 3, 16) * 2.0
    ref = tt.RMSNorm(16)(x_fp32)  # fp32 参考

    for dtype, atol in ((torch.bfloat16, 3e-2), (torch.float16, 1e-2)):
        norm = tt.RMSNorm(16, dtype=dtype)
        assert norm.weight.dtype == dtype
        out = norm(x_fp32.to(dtype))
        assert out.dtype == dtype, f"{dtype}: 输出 dtype 变成 {out.dtype}"
        assert torch.isfinite(out.float()).all()
        # bf16/fp16 精度有限，只要求和 fp32 版本近似
        _assert_close(out.float(), ref, atol=atol)


def test_B14_swiglu_default_d_ff():
    """[14] SwiGLU 默认 d_ff：8/3·d 取最近 64 倍数（half-up，最小 64）。"""
    # 1) handout/SPEC 给定的参考值
    assert tt._default_d_ff(512) == 1344
    assert tt._default_d_ff(1600) == 4288
    assert tt._default_d_ff(8) == 64  # 极小 d_model 也至少 64

    # 2) GPT-2 四个配置的 d_model 都要符合「最近 64 倍数」
    for d_model, expected in ((512, 1344), (768, 2048), (1024, 2752), (1280, 3392), (1600, 4288)):
        assert tt._default_d_ff(d_model) == expected, (d_model, tt._default_d_ff(d_model), expected)

    # 3) 边界/tie 规则：8/3*60 = 160，160/64 = 2.5 恰好落在中间。
    #    half-up 取 3 -> 192；Python round() 会做银行家舍入得到 128。
    #    这里和 accounting.py 的 d_ff_default 公式（floor(x/64+0.5)）保持一致。
    assert tt._default_d_ff(60) == 192
    assert tt._default_d_ff(12) == 64  # 0.5 -> 1，但仍被 max(64, ...) 抬到 64
    for d_model in (24, 60, 100, 512, 768, 1024, 1280, 1600):
        accounting_style = max(64, 64 * math.floor(8.0 * d_model / 3.0 / 64.0 + 0.5))
        assert tt._default_d_ff(d_model) == accounting_style, d_model

    # 4) 实例化后形状正确（用 d_model=768 顺便验证 2048）
    for d_model, expected in ((512, 1344), (768, 2048), (1600, 4288)):
        ffn = tt.SwiGLU(d_model)
        assert ffn.w1.weight.shape == (expected, d_model), ffn.w1.weight.shape
        assert ffn.w3.weight.shape == (expected, d_model)
        assert ffn.w2.weight.shape == (d_model, expected)
        assert expected % 64 == 0

    # 显式 d_ff 时不覆盖用户的选择
    ffn = tt.SwiGLU(64, 100)
    assert ffn.w1.weight.shape == (100, 64)


def test_B15_mha_shape_and_head_divisibility():
    """[15] MHA 输出形状；num_heads 不整除 d_model 时抛错。"""
    mha = tt.CausalMultiHeadSelfAttention(64, 4)
    x = torch.randn(4, 12, 64)
    out = mha(x)
    assert out.shape == (4, 12, 64), out.shape

    # 任意前导 batch 维都可以（head 维当 batch）
    out = mha(torch.randn(2, 3, 12, 64))
    assert out.shape == (2, 3, 12, 64)

    _assert_raises(ValueError, lambda: tt.CausalMultiHeadSelfAttention(10, 3), "num_heads=3 不整除 d_model=10")
    _assert_raises(ValueError, lambda: tt.CausalMultiHeadSelfAttention(64, 5), "num_heads=5 不整除 d_model=64")


def test_B16_block_shape_preserved():
    """[16] Block 输出形状与输入一致（有无 RoPE 都试一遍）。"""
    x = torch.randn(2, 7, 64)

    block = tt.TransformerBlock(64, 4, 128)
    assert block(x).shape == (2, 7, 64)

    rope = tt.RotaryPositionalEmbedding(10000.0, 16, max_seq_len=16)
    block_rope = tt.TransformerBlock(64, 4, 128, rope=rope)
    assert block_rope(x).shape == (2, 7, 64)

    # block 内部的属性名是 state_dict 键名的一部分，不能变
    assert hasattr(block_rope, "attn") and hasattr(block_rope, "ffn")
    assert hasattr(block_rope, "ln1") and hasattr(block_rope, "ln2")


def test_B17_lm_logits_shape_and_truncated():
    """[17] LM logits 形状 (B,T,V)；截断输入 T=6 也可以跑。"""
    torch.manual_seed(10)
    model = tt.TransformerLM(
        vocab_size=101, context_length=16, d_model=32, num_layers=2, num_heads=4, d_ff=64
    )
    ids = torch.randint(0, 101, (3, 12))
    logits = model(ids)
    assert logits.shape == (3, 12, 101), logits.shape

    truncated = model(ids[:, :6])
    assert truncated.shape == (3, 6, 101), truncated.shape

    # 正好用满 context_length 也可以
    full = model(torch.randint(0, 101, (2, 16)))
    assert full.shape == (2, 16, 101)


def test_B18_lm_rejects_sequence_over_context():
    """[18] LM 超长输入 T > context_length 抛错。"""
    model = tt.TransformerLM(
        vocab_size=32, context_length=8, d_model=16, num_layers=1, num_heads=2, d_ff=32
    )
    _assert_raises(
        ValueError,
        lambda: model(torch.randint(0, 32, (2, 9))),
        "T=9 超过 context_length=8",
    )
    _assert_raises(
        ValueError,
        lambda: model(torch.randint(0, 32, (1, 17))),
        "T=17 超过 context_length=8",
    )
    # 边界：正好 8 是允许的
    assert model(torch.randint(0, 32, (2, 8))).shape == (2, 8, 32)


# ===========================================================================
# C. 性质（SPEC §6 C：第 19–26 项）
# ===========================================================================
def test_C19_causal_future_tokens_do_not_change_past():
    """[19] causal：改未来 token 不影响过去位置的输出。"""
    torch.manual_seed(11)
    rope = tt.RotaryPositionalEmbedding(10000.0, 16, max_seq_len=16)
    block = tt.TransformerBlock(64, 4, 128, rope=rope)
    x = torch.randn(2, 10, 64)
    x_changed = x.clone()
    x_changed[:, 5:] += torch.randn(2, 5, 64) * 3.0  # 只改第 5 个 token 及之后

    out1 = block(x)
    out2 = block(x_changed)
    _assert_close(out1[:, :5], out2[:, :5], atol=1e-6)

    # 反过来：未来位置的输出应该被改动了（否则说明 mask 把一切都挡住了）
    assert _maxdiff(out1[:, 5:], out2[:, 5:]) > 1e-4


def test_C20_causal_attention_weights_upper_triangle_zero():
    """[20] causal：注意力权重上三角为 0（用 monkeypatch 抓 MHA 内部真实用的 mask）。"""
    torch.manual_seed(12)
    d_model, num_heads, seq_len = 32, 4, 9
    mha = tt.CausalMultiHeadSelfAttention(d_model, num_heads)
    x = torch.randn(2, seq_len, d_model)

    captured: dict[str, Tensor | None] = {}
    original = tt.scaled_dot_product_attention

    def spy(q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None = None):
        captured["mask"] = None if mask is None else mask.detach().clone()
        return original(q, k, v, mask)

    tt.scaled_dot_product_attention = spy  # type: ignore[assignment]
    try:
        mha(x)
    finally:
        tt.scaled_dot_product_attention = original  # type: ignore[assignment]

    mask = captured["mask"]
    assert mask is not None, "MHA 必须内置 causal mask"
    assert mask.dtype == torch.bool
    assert mask.shape == (seq_len, seq_len), mask.shape
    tril = _causal_mask(seq_len)
    assert torch.equal(mask, tril), "MHA 传给 SDPA 的 mask 必须是下三角 True"

    # 用这个 mask 重算注意力权重：上三角必须恰好是 0，每行和是 1
    d_head = d_model // num_heads
    q = _heads(mha.q_proj(x), num_heads)
    k = _heads(mha.k_proj(x), num_heads)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_head)
    probs = torch.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1)
    upper = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
    assert torch.all(probs[..., upper] == 0), "上三角的注意力权重必须为 0"
    torch.testing.assert_close(probs.sum(-1), torch.ones(2, num_heads, seq_len), atol=1e-6, rtol=0)


def test_C21_rope_only_applied_to_qk():
    """[21] RoPE 只作用于 q/k：v 路径与「不旋转」的手写参考一致；旋转 v 则明显不同。"""
    torch.manual_seed(13)
    d_model, num_heads, seq_len = 32, 4, 7
    d_head = d_model // num_heads
    rope = tt.RotaryPositionalEmbedding(10000.0, d_head, max_seq_len=16)
    mha = tt.CausalMultiHeadSelfAttention(d_model, num_heads, rope=rope)
    x = torch.randn(2, seq_len, d_model)
    pos = torch.arange(seq_len)
    out = mha(x)

    mask = _causal_mask(seq_len)
    q = rope(_heads(mha.q_proj(x), num_heads), pos)  # 旋转 q
    k = rope(_heads(mha.k_proj(x), num_heads), pos)  # 旋转 k
    v = _heads(mha.v_proj(x), num_heads)  # v 不旋转
    ref = mha.output_proj(
        tt.scaled_dot_product_attention(q, k, v, mask).transpose(-3, -2).reshape(*x.shape)
    )
    _assert_close(out, ref, atol=1e-6)

    # 反例：如果连 v 也旋转，结果会明显不同 -> 证明实现确实没转 v
    v_rot = rope(v, pos)
    wrong = mha.output_proj(
        tt.scaled_dot_product_attention(q, k, v_rot, mask).transpose(-3, -2).reshape(*x.shape)
    )
    assert _maxdiff(out, wrong) > 1e-3, "v 也旋转时结果应当不同"


def test_C22_rope_position_shift_keeps_attention():
    """[22] RoPE 位置平移：q/k 同时平移 s，attention 输出不变。"""
    torch.manual_seed(14)
    rope = tt.RotaryPositionalEmbedding(10000.0, 16, max_seq_len=32)
    mha = tt.CausalMultiHeadSelfAttention(64, 4, rope=rope)
    x = torch.randn(2, 10, 64)

    out_zero = mha(x)  # 默认位置 0..9
    out_shift = mha(x, token_positions=torch.arange(7, 17))  # 整体平移 7
    _assert_close(out_zero, out_shift, atol=1e-5)


def test_C23_single_head_equals_manual_attention():
    """[23] 多头拆分等价：num_heads=1 与手写单头注意力一致。"""
    torch.manual_seed(15)
    d_model, seq_len = 16, 5
    mha = tt.CausalMultiHeadSelfAttention(d_model, 1)
    x = torch.randn(2, seq_len, d_model)

    out = mha(x)
    ref = mha.output_proj(
        tt.scaled_dot_product_attention(
            mha.q_proj(x), mha.k_proj(x), mha.v_proj(x), _causal_mask(seq_len)
        )
    )
    _assert_close(out, ref, atol=1e-6)


def test_C24_rmsnorm_is_scale_invariant():
    """[24] RMSNorm 尺度不变：x 与 x*3 的输出相同（同一 gain）。"""
    torch.manual_seed(16)
    norm = tt.RMSNorm(32)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(32))
    x = torch.randn(3, 8, 32) * 2.0
    _assert_close(norm(x), norm(x * 3.0), atol=1e-5)

    # 注意：只在「正缩放」下成立。RMSNorm(-2x) 的分子变成 -2x、分母仍是正的，
    # 输出会整体反号 —— 这是 RMSNorm 与 LayerNorm 都有的性质，不是 bug。


def test_C25_tie_weights_shares_parameters():
    """[25] tie_weights=True 时 lm_head 与 token_embeddings 共享存储，num_parameters 变少。"""
    kwargs = dict(
        vocab_size=50, context_length=8, d_model=16, num_layers=1, num_heads=2, d_ff=32
    )
    untied = tt.TransformerLM(**kwargs, tie_weights=False)
    tied = tt.TransformerLM(**kwargs, tie_weights=True)

    assert tied.lm_head.weight is tied.token_embeddings.weight
    assert tied.lm_head.weight.data_ptr() == tied.token_embeddings.weight.data_ptr()
    assert untied.lm_head.weight.data_ptr() != untied.token_embeddings.weight.data_ptr()

    # state_dict 里两个键指向同一块存储
    sd = tied.state_dict()
    assert sd["lm_head.weight"].data_ptr() == sd["token_embeddings.weight"].data_ptr()

    n_untied = untied.num_parameters()
    n_tied = tied.num_parameters()
    assert n_untied - n_tied == kwargs["vocab_size"] * kwargs["d_model"], (n_untied, n_tied)

    # tied 模型照样能前向，且改动 embedding 会同步影响 logits
    ids = torch.randint(0, 50, (2, 8))
    assert tied(ids).shape == (2, 8, 50)
    with torch.no_grad():
        tied.token_embeddings.weight.mul_(1.0)
    assert tied.lm_head.weight.data_ptr() == tied.token_embeddings.weight.data_ptr()


def test_C26_all_parameters_receive_finite_gradients():
    """[26] 反向传播：最后一个 token 的 loss 反传，所有参数 grad 非 None 且有限。"""
    torch.manual_seed(17)
    model = tt.TransformerLM(
        vocab_size=64, context_length=16, d_model=32, num_layers=2, num_heads=4, d_ff=64
    )
    ids = torch.randint(0, 64, (2, 16))
    target = torch.randint(0, 64, (2,))
    logits = model(ids)
    loss = F.cross_entropy(logits[:, -1, :], target)
    loss.backward()

    n_with_grad = 0
    for name, param in model.named_parameters():
        assert param.grad is not None, f"{name} 没有梯度"
        assert torch.isfinite(param.grad).all(), f"{name} 的梯度有 inf/nan"
        n_with_grad += 1
    assert n_with_grad == len(list(model.parameters()))
    # 至少有参数拿到了非零梯度（否则说明计算图断了）
    assert any(float(p.grad.abs().sum()) > 0 for p in model.parameters())


def test_C27_mha_accepts_batch_token_positions():
    """[回归，不计入 43 项] MHA 的 token_positions 支持 (T,)/(1,T)/(B,T)/(1,1,T)。

    背景：一度只处理「positions.ndim == x.ndim」的情况，于是 x=(B,T,C) + positions=(B,T)
    不会插 head 维而报错，而 (1,T) 恰好靠右对齐广播成功、把 bug 掩盖了（scribe/Lead
    复现）。这里对四种形状做回归，并验证 (B,T) 是**逐样本**生效而不是「所有样本都用
    第 0 行」。
    """
    torch.manual_seed(21)
    d_model, num_heads, seq_len, batch = 64, 4, 5, 2
    # max_seq_len 要够大：下面会用 pos*2，最大位置是 2*(seq_len-1) = 8
    rope = tt.RotaryPositionalEmbedding(10000.0, d_model // num_heads, max_seq_len=16)
    mha = tt.CausalMultiHeadSelfAttention(d_model, num_heads, rope=rope)
    x = torch.randn(batch, seq_len, d_model)
    pos = torch.arange(seq_len)

    # 四种合法形状都不许报错，且共享同一组位置时结果一致
    out_t = mha(x, pos)  # (T,)
    out_1t = mha(x, pos[None, :])  # (1,T)
    out_bt = mha(x, pos[None, :].expand(batch, seq_len))  # (B,T)
    out_11t = mha(x, pos.reshape(1, 1, seq_len))  # (1,1,T)
    for name, out in (("(T,)", out_t), ("(1,T)", out_1t), ("(B,T)", out_bt), ("(1,1,T)", out_11t)):
        assert out.shape == (batch, seq_len, d_model), (name, out.shape)
        _assert_close(out, out_t, atol=1e-6)

    # (B,T) 必须逐样本用自己的位置：两行给不同的位置间隔（0,1,2,… vs 0,2,4,…），
    # 再与「逐样本用 (1,T) 分别算」的结果逐元素对齐。
    pos_bt = torch.stack([pos, pos * 2])
    out_bt2 = mha(x, pos_bt)
    per_sample = torch.cat([mha(x[i : i + 1], pos_bt[i : i + 1]) for i in range(batch)], dim=0)
    _assert_close(out_bt2, per_sample, atol=1e-6)

    # 排除「所有样本都用了第 0 行位置」的实现：两行位置不同，第 1 个样本输出必须不同
    out_row0 = mha(x, pos_bt[0:1].expand(batch, seq_len))
    _assert_close(out_bt2[0], out_row0[0], atol=1e-6)
    assert _maxdiff(out_bt2[1], out_row0[1]) > 1e-4, "(B,T) 位置没有逐样本生效"


# ===========================================================================
# D. 初始化（SPEC §6 D：第 27–30 项）
# ===========================================================================
def test_D27_linear_init_scale():
    """[27] Linear 初始化 std ≈ sqrt(2/(in+out))（±25%），值在 ±3σ 内。"""
    torch.manual_seed(18)
    in_features, out_features = 512, 256
    layer = tt.Linear(in_features, out_features)
    expected_std = math.sqrt(2.0 / (in_features + out_features))

    measured = layer.weight.std().item()
    assert abs(measured / expected_std - 1.0) <= 0.25, (measured, expected_std)
    assert layer.weight.abs().max().item() <= 3.0 * expected_std + 1e-6
    # 均值应该接近 0
    assert abs(layer.weight.mean().item()) <= 0.1 * expected_std


def test_D28_embedding_init_scale():
    """[28] Embedding 初始化 std ≈ 1（±25%），值在 ±3 内。"""
    torch.manual_seed(19)
    emb = tt.Embedding(4000, 32)
    measured = emb.weight.std().item()
    assert abs(measured - 1.0) <= 0.25, measured
    assert emb.weight.abs().max().item() <= 3.0 + 1e-6
    assert abs(emb.weight.mean().item()) <= 0.1


def test_D29_rmsnorm_gain_ones_and_no_bias():
    """[29] RMSNorm gain 全 1；Linear/Embedding 没有 bias 参数。"""
    norm = tt.RMSNorm(32)
    assert torch.equal(norm.weight, torch.ones(32))
    assert {n for n, _ in norm.named_parameters()} == {"weight"}

    # Linear/Embedding 都只有 weight 一个参数
    assert {n for n, _ in tt.Linear(4, 6).named_parameters()} == {"weight"}
    assert {n for n, _ in tt.Embedding(7, 3).named_parameters()} == {"weight"}

    # 整个 LM 里不能出现任何 bias
    model = tt.TransformerLM(
        vocab_size=20, context_length=8, d_model=16, num_layers=1, num_heads=2, d_ff=32
    )
    assert not any("bias" in n for n, _ in model.named_parameters())


def test_D30_state_dict_keys_match_official():
    """[30] state_dict 键集合与官方参考一致（官方仓库可得时；否则 SKIP）。"""
    config = _official_config()
    model = tt.TransformerLM(
        vocab_size=config["vocab_size"],
        context_length=config["context_length"],
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        rope_theta=config["rope_theta"],
    )
    official_keys = set(_official_state_dict().keys())
    ours = set(model.state_dict().keys())
    assert ours == official_keys, f"缺少 {official_keys - ours}；多出 {ours - official_keys}"

    # 官方权重必须能 strict=True 直接 load 进来
    model.load_state_dict(_official_state_dict(), strict=True)


# ===========================================================================
# E. 官方快照 parity（SPEC §6 E：第 31–42 项，13 个 npz）
# ===========================================================================
def test_E31_parity_linear():
    """[31] 官方快照 test_linear：layers.0.ffn.w1.weight + in_embeddings。"""
    layer = tt.Linear(64, 128)
    layer.load_state_dict({"weight": _official_state_dict()["layers.0.ffn.w1.weight"]}, strict=True)
    _assert_close(layer(_x()), _snapshot("test_linear"), atol=1e-4, label="test_linear")


def test_E32_parity_embedding():
    """[32] 官方快照 test_embedding：token_embeddings.weight + in_indices。"""
    emb = tt.Embedding(10_000, 64)
    emb.load_state_dict({"weight": _official_state_dict()["token_embeddings.weight"]}, strict=True)
    _assert_close(emb(_idx()), _snapshot("test_embedding"), atol=1e-4, label="test_embedding")


def test_E33_parity_swiglu():
    """[33] 官方快照 test_swiglu：layers.0.ffn.w1/w2/w3 + in_embeddings。"""
    ffn = tt.SwiGLU(64, 128)
    ffn.load_state_dict(_official_subset("layers.0.ffn."), strict=True)
    _assert_close(ffn(_x()), _snapshot("test_swiglu"), atol=1e-5, label="test_swiglu")


def test_E34_parity_rmsnorm():
    """[34] 官方快照 test_rmsnorm：layers.1.ln1.weight + in_embeddings。"""
    norm = tt.RMSNorm(64, eps=1e-5)
    norm.load_state_dict({"weight": _official_state_dict()["layers.1.ln1.weight"]}, strict=True)
    _assert_close(norm(_x()), _snapshot("test_rmsnorm"), atol=1e-4, label="test_rmsnorm")


def test_E35_parity_rope():
    """[35] 官方快照 test_rope：无权重，x + pos=arange(12)，d_k=64。"""
    rope = tt.RotaryPositionalEmbedding(theta=10000.0, d_k=64, max_seq_len=12)
    out = rope(_x(), torch.arange(12))
    _assert_close(out, _snapshot("test_rope"), atol=1e-5, label="test_rope")


def test_E36_parity_sdpa():
    """[36] 官方快照 test_scaled_dot_product_attention：q/k/v + 3D mask。"""
    out = tt.scaled_dot_product_attention(_q(), _k(), _v(), _mask())
    _assert_close(out, _snapshot("test_scaled_dot_product_attention"), atol=1e-5, label="test_sdpa")


def test_E37_parity_sdpa_4d():
    """[37] 官方快照 test_4d_scaled_dot_product_attention：按 head=2 变 4D。"""
    q4 = _q().reshape(2, 2, 12, 64)
    k4 = _k().reshape(2, 2, 16, 64)
    v4 = _v().reshape(2, 2, 16, 64)
    mask4 = _mask().reshape(2, 2, 12, 16)
    out = tt.scaled_dot_product_attention(q4, k4, v4, mask4)
    _assert_close(out, _snapshot("test_4d_scaled_dot_product_attention"), atol=1e-5, label="test_4d_sdpa")


def test_E38_parity_mha():
    """[38] 官方快照 test_multihead_self_attention：layers.0.attn.*，无 RoPE。"""
    mha = tt.CausalMultiHeadSelfAttention(64, 4)
    mha.load_state_dict(_official_subset("layers.0.attn."), strict=True)
    _assert_close(mha(_x()), _snapshot("test_multihead_self_attention"), atol=1e-5, label="test_mha")


def test_E39_parity_mha_rope():
    """[39] 官方快照 test_multihead_self_attention_with_rope：token_positions=(1,12)。"""
    rope = tt.RotaryPositionalEmbedding(theta=10000.0, d_k=16, max_seq_len=16)
    mha = tt.CausalMultiHeadSelfAttention(64, 4, rope=rope)
    mha.load_state_dict(_official_subset("layers.0.attn."), strict=True)
    out = mha(_x(), token_positions=torch.arange(12).reshape(1, 12))
    _assert_close(
        out, _snapshot("test_multihead_self_attention_with_rope"), atol=1e-5, label="test_mha_rope"
    )


def test_E40_parity_positionwise_feedforward():
    """[40] 官方快照 test_positionwise_feedforward：与 swiglu 同输入同权重。"""
    ffn = tt.SwiGLU(64, 128)
    ffn.load_state_dict(_official_subset("layers.0.ffn."), strict=True)
    out = ffn(_x())
    _assert_close(out, _snapshot("test_positionwise_feedforward"), atol=1e-5, label="test_pwff")


def test_E41_parity_transformer_block():
    """[41] 官方快照 test_transformer_block：layers.0.*（去掉前缀）+ RoPE。"""
    rope = tt.RotaryPositionalEmbedding(theta=10000.0, d_k=16, max_seq_len=16)
    block = tt.TransformerBlock(64, 4, 128, rope=rope)
    block.load_state_dict(_official_subset("layers.0."), strict=True)
    _assert_close(block(_x()), _snapshot("test_transformer_block"), atol=1e-4, label="test_block")


def test_E42_parity_transformer_lm():
    """[42a] 官方快照 test_transformer_lm：全量 state_dict + in_indices（ctx=16）。"""
    config = _official_config()
    model = tt.TransformerLM(
        vocab_size=config["vocab_size"],
        context_length=config["context_length"],
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        rope_theta=config["rope_theta"],
    )
    model.load_state_dict(_official_state_dict(), strict=True)
    _assert_close(
        model(_idx()),
        _snapshot("test_transformer_lm"),
        atol=1e-4,
        rtol=1e-2,
        label="test_transformer_lm",
    )


def test_E42_parity_transformer_lm_truncated():
    """[42b] 官方快照 test_transformer_lm_truncated_input：idx[:, :6]。"""
    config = _official_config()
    model = tt.TransformerLM(
        vocab_size=config["vocab_size"],
        context_length=config["context_length"],
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        rope_theta=config["rope_theta"],
    )
    model.load_state_dict(_official_state_dict(), strict=True)
    _assert_close(
        model(_idx()[:, :6]),
        _snapshot("test_transformer_lm_truncated_input"),
        atol=1e-4,
        label="test_transformer_lm_truncated",
    )


# ===========================================================================
# F. 端到端（SPEC §6 F：第 43 项）
# ===========================================================================
def test_F43_one_step_training_reduces_loss():
    """[43] 小 LM 前向 + CrossEntropy 反传若干步，loss 下降（端到端可训练）。"""
    torch.manual_seed(20)
    vocab_size, context_length, d_model = 64, 16, 32
    model = tt.TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=2,
        num_heads=4,
        d_ff=64,
    )
    ids = torch.randint(0, vocab_size, (2, context_length))
    targets = torch.roll(ids, shifts=-1, dims=-1)  # 语言建模：预测下一个 token

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    losses: list[float] = []
    for _ in range(30):
        logits = model(ids)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert all(math.isfinite(loss) for loss in losses), "loss 出现 inf/nan"
    assert losses[-1] < losses[0], f"loss 没有下降：{losses[0]:.4f} -> {losses[-1]:.4f}"
    print(f"    loss: {losses[0]:.4f} -> {losses[-1]:.4f}（30 步 AdamW, lr=3e-3）")


# ===========================================================================
# 双模式入口
# ===========================================================================
def _all_tests() -> list[tuple[str, object]]:
    """收集本模块所有 `test_*` 函数（按定义顺序，A→F 编号方便对照 SPEC）。"""
    return [
        (name, obj)
        for name, obj in globals().items()
        if name.startswith("test_") and callable(obj)
    ]


def main() -> int:
    torch.set_num_threads(2)
    tests = _all_tests()
    passed = failed = skipped = 0
    skipped_names: list[str] = []
    start = time.perf_counter()

    print(f"运行 {len(tests)} 个测试")
    print(f"  A1_REPO = {A1_REPO}（存在: {A1_REPO.exists()}）")
    print(f"  torch = {torch.__version__}，线程数 = {torch.get_num_threads()}\n")

    skip_exc: tuple[type[BaseException], ...] = (
        (_pytest.skip.Exception,) if _pytest is not None else ()
    )
    for name, fn in tests:
        try:
            fn()  # type: ignore[operator]
        except SkipTest as exc:
            skipped += 1
            skipped_names.append(name)
            print(f"SKIP {name}: {exc}")
        except Exception as exc:  # noqa: BLE001 - 测试运行器要抓住所有异常
            if skip_exc and isinstance(exc, skip_exc):
                skipped += 1
                skipped_names.append(name)
                print(f"SKIP {name}: {exc}")
                continue
            failed += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
        else:
            passed += 1
            print(f"PASS {name}")

    elapsed = time.perf_counter() - start
    print(f"\n汇总: {passed} passed, {failed} failed, {skipped} skipped")
    print(f"耗时: {elapsed:.2f}s，线程数: {torch.get_num_threads()}")
    if skipped:
        print(f"跳过的测试: {', '.join(skipped_names)}")
        print("（parity 测试需要官方仓库；用 A1_REPO=/path/to/assignment1-basics 指定）")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
