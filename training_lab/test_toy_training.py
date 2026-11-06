"""test_toy_training.py —— `toy_training` 的双模式测试（覆盖 SPEC §4 的 A–H 共 35 项）。

两种跑法都支持：

    # 1) 直接跑（不需要 pytest；打印 PASS/FAIL 汇总，有失败时 exit 1）
    .venv/bin/python training_lab/test_toy_training.py

    # 2) pytest（如果环境里有）
    python -m pytest training_lab/test_toy_training.py -q

测试分层（编号与 SPEC §4 的表格一一对应，方便对照）：

    A CE（5 项）：与 F.cross_entropy 的数值/梯度一致、超大 logits 稳定
    B AdamW（7 项）：与 torch.optim.AdamW 一致、weight decay、param groups、
                     state_dict 续训、非法 lr、closure
    C schedule（4 项）：官方 25 个期望值、三段边界、warmup=0、set_lr
    D clip（4 项）：与 torch.clip_grad_norm_ 一致、不动小梯度、裁剪后≈max、
                    None grad 跳过
    E data（5 项）：形状/dtype/偏移、起点均匀性（5σ）、memmap、rng 可复现、
                    非法 device
    F ckpt（4 项）：官方往返口径、中断恢复等价、BytesIO、返回 iteration
    G decode（5 项）：极低温=argmax、top_p=1、手工 top-p 核、熵随温度下降、
                     generate 的 max_new_tokens/eos/context/seed/prompt
    H 集成（1 项）：小模型 50 步 loss 下降 > 50%

所有断言失败都会打印「label + maxdiff + 期望/实际」，方便定位；不需要官方仓库，
也不联网。整个文件应 < 5 秒跑完。
"""

from __future__ import annotations

import copy
import io
import math
import sys
import tempfile
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

# 与仓库其它脚本一致：限制线程数，避免测试和训练脚本抢 CPU。
torch.set_num_threads(2)

# 允许 `python training_lab/test_toy_training.py` 从任意 cwd 直接跑。
sys.path.insert(0, str(Path(__file__).resolve().parent))

import toy_training as tt  # noqa: E402

# 官方 test_optimizer.py 里的 25 个 cosine 期望值（逐元素对齐口径）。
OFFICIAL_COSINE_LRS = [
    0,
    0.14285714285714285,
    0.2857142857142857,
    0.42857142857142855,
    0.5714285714285714,
    0.7142857142857143,
    0.8571428571428571,
    1.0,
    0.9887175604818206,
    0.9554359905560885,
    0.9018241671106134,
    0.8305704108364301,
    0.7452476826029011,
    0.6501344202803414,
    0.55,
    0.44986557971965857,
    0.3547523173970989,
    0.26942958916356996,
    0.19817583288938662,
    0.14456400944391146,
    0.11128243951817937,
    0.1,
    0.1,
    0.1,
    0.1,
]


# ===========================================================================
# 0. 基础设施：断言工具（失败信息要能直接定位）
# ===========================================================================
def _detach(value):
    """比较数值时先脱离计算图/梯度，避免 as_tensor 对 requires_grad 张量告警。"""
    return value.detach() if isinstance(value, Tensor) else value


def _maxdiff(actual, expected) -> float:
    """两个 tensor/标量在 float64 下的最大绝对差（统一升精度避免误判）。"""
    a = torch.as_tensor(_detach(actual), dtype=torch.float64)
    b = torch.as_tensor(_detach(expected), dtype=torch.float64)
    if a.numel() == 0:
        return 0.0
    return float((a - b).abs().max())


def _assert_close(actual, expected, *, atol: float, label: str, rtol: float = 0.0) -> None:
    """逐元素 allclose，失败时打印 label 与 maxdiff，而不是干巴巴的 AssertionError。"""
    a = torch.as_tensor(_detach(actual), dtype=torch.float64)
    b = torch.as_tensor(_detach(expected), dtype=torch.float64)
    if a.shape != b.shape:
        raise AssertionError(f"{label}: 形状不一致，actual={tuple(a.shape)} expected={tuple(b.shape)}")
    diff = (a - b).abs()
    limit = atol + rtol * b.abs()
    bad = diff > limit
    if bool(bad.any()):
        idx = int(bad.reshape(-1).nonzero()[0])
        raise AssertionError(
            f"{label}: maxdiff={_maxdiff(a, b):.3e} 超过 atol={atol:.1e}；"
            f"首个越界索引={idx}，actual={a.reshape(-1)[idx].item():.8f}，"
            f"expected={b.reshape(-1)[idx].item():.8f}"
        )


def _assert(cond: bool, label: str) -> None:
    """普通条件断言，label 直接说明哪条约定被违反。"""
    if not cond:
        raise AssertionError(label)


def _expect_raises(exc_types, fn, *, label: str):
    """断言 ``fn()`` 抛出指定异常（可以是 tuple），返回该异常方便看消息。"""
    try:
        fn()
    except exc_types as exc:  # type: ignore[misc]
        return exc
    except Exception as exc:  # noqa: BLE001 - 测试辅助要区分「异常类型不对」
        raise AssertionError(
            f"{label}: 期望 {exc_types}，实际抛出 {type(exc).__name__}: {exc}"
        ) from exc
    raise AssertionError(f"{label}: 期望抛出 {exc_types}，但没有抛异常")


def _optimizer_state_dicts_equal(sd1: dict, sd2: dict, atol: float = 1e-8) -> tuple[bool, str]:
    """比较两个 optimizer.state_dict()（官方 test_serialization.py 的口径）。"""
    if set(sd1) != set(sd2):
        return False, f"顶层键不同：{set(sd1)} vs {set(sd2)}"
    if sd1["param_groups"] != sd2["param_groups"]:
        return False, "param_groups 不同（lr/betas/params 顺序等）"
    state1, state2 = sd1["state"], sd2["state"]
    if set(state1) != set(state2):
        return False, f"state 的 param 索引不同：{set(state1)} vs {set(state2)}"
    for key in state1:
        if set(state1[key]) != set(state2[key]):
            return False, f"param {key} 的子键不同"
        for sub_key, item1 in state1[key].items():
            item2 = state2[key][sub_key]
            if torch.is_tensor(item1) and torch.is_tensor(item2):
                if not torch.allclose(item1, item2, atol=atol, rtol=1e-5):
                    return False, f"param {key}.{sub_key} 的 maxdiff={_maxdiff(item1, item2):.3e}"
            elif item1 != item2:
                return False, f"param {key}.{sub_key} 不相等：{item1!r} vs {item2!r}"
    return True, "ok"


class _TestNet(nn.Module):
    """官方 test_serialization.py 的同款小 MLP（100 -> 200 -> 100 -> 10）。"""

    def __init__(self, d_input: int = 100, d_output: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(d_input, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, d_output)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class _ScriptedLM(nn.Module):
    """generate 测试用的假 LM：最后一位 logits = arange(vocab)，并记录输入长度。

    有真实 ``Parameter``，这样 ``generate`` 能推断出设备；logits 与输入内容无关，
    因此采样结果只由 torch.Generator 决定，适合验证「seed 可复现」。
    """

    def __init__(self, vocab_size: int = 12):
        super().__init__()
        self.vocab_size = vocab_size
        self.dummy = nn.Parameter(torch.zeros(1))
        self.input_lengths: list[int] = []

    def forward(self, token_ids: Tensor) -> Tensor:
        batch, seq_len = token_ids.shape
        self.input_lengths.append(int(seq_len))
        logits = torch.zeros(batch, seq_len, self.vocab_size, device=token_ids.device)
        logits[:, -1, :] = torch.arange(
            self.vocab_size, dtype=torch.float32, device=token_ids.device
        )
        return logits


# ===========================================================================
# A. 交叉熵（SPEC §4 A1–A5）
# ===========================================================================
def test_A1_cross_entropy_matches_torch():
    """[A1] (2,3,7) 随机 logits 与 F.cross_entropy 一致（含前导维平均口径）。"""
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 7)
    targets = torch.randint(0, 7, (2, 3))
    ours = tt.cross_entropy(logits, targets)
    official = F.cross_entropy(logits.reshape(-1, 7), targets.reshape(-1))
    _assert_close(ours, official, atol=1e-6, label="A1 cross_entropy")


def test_A2_cross_entropy_stable_with_huge_logits():
    """[A2] logits×1000 时不 NaN/Inf，且仍与 F.cross_entropy 一致（max 平移有效）。"""
    torch.manual_seed(1)
    logits = torch.randn(2, 3, 7) * 1000.0
    targets = torch.randint(0, 7, (2, 3))
    ours = tt.cross_entropy(logits, targets)
    official = F.cross_entropy(logits.reshape(-1, 7), targets.reshape(-1))
    _assert(bool(torch.isfinite(ours)), f"A2 loss 出现 NaN/Inf：{ours.item()}")
    # 大 logits 下 float32 的相对误差会被放大，用 1e-4 绝对容差（官方也按相对误差看）。
    _assert_close(
        ours, official, atol=1e-4, rtol=1e-6, label="A2 cross_entropy(huge logits)"
    )


def test_A3_cross_entropy_gradient_matches_torch():
    """[A3] 反向梯度与 F.cross_entropy 一致（1e-6）：等价于 softmax - onehot。"""
    torch.manual_seed(2)
    base = torch.randn(3, 5) * 2.0
    targets = torch.randint(0, 5, (3,))
    ours_logits = base.clone().requires_grad_(True)
    official_logits = base.clone().requires_grad_(True)
    tt.cross_entropy(ours_logits, targets).backward()
    F.cross_entropy(official_logits, targets).backward()
    _assert_close(
        ours_logits.grad, official_logits.grad, atol=1e-6, label="A3 cross_entropy grad"
    )


def test_A4_cross_entropy_high_dim_batch():
    """[A4] 高维 batch (5,4,9) 与 F.cross_entropy 一致（前导维全部参与平均）。"""
    torch.manual_seed(3)
    logits = torch.randn(5, 4, 9)
    targets = torch.randint(0, 9, (5, 4))
    ours = tt.cross_entropy(logits, targets)
    official = F.cross_entropy(logits.reshape(-1, 9), targets.reshape(-1))
    _assert_close(ours, official, atol=1e-6, label="A4 cross_entropy (5,4,9)")


def test_A5_cross_entropy_returns_float_scalar():
    """[A5] 返回 0 维浮点标量（不是 (1,)），并验证 perplexity=exp(loss)。"""
    torch.manual_seed(4)
    logits = torch.randn(2, 5)
    targets = torch.randint(0, 5, (2,))
    loss = tt.cross_entropy(logits, targets)
    _assert(isinstance(loss, Tensor), f"A5 返回值不是 Tensor：{type(loss)}")
    _assert(loss.ndim == 0, f"A5 应为 0 维标量，实际 ndim={loss.ndim}, shape={tuple(loss.shape)}")
    _assert(torch.is_floating_point(loss), f"A5 应为浮点，实际 dtype={loss.dtype}")
    _assert_close(
        tt.perplexity(loss),
        math.exp(float(loss)),
        atol=0.0,
        rtol=1e-12,
        label="A5 perplexity=exp(loss)",
    )


# ===========================================================================
# B. AdamW（SPEC §4 B1–B7）
# ===========================================================================
def _run_official_adamw_regression(opt_cls) -> Tensor:
    """官方 test_optimizer.py 的 1000 步线性回归（方法体逐行对齐官方）。"""
    torch.manual_seed(42)
    model = torch.nn.Linear(3, 2, bias=False)
    opt = opt_cls(
        model.parameters(),
        lr=1e-3,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    for _ in range(1000):
        opt.zero_grad()
        x = torch.rand(model.in_features)
        y_hat = model(x)
        y = torch.tensor([x[0] + x[1], -x[2]])
        loss = ((y - y_hat) ** 2).sum()
        loss.backward()
        opt.step()
    return model.weight.detach()


def test_B1_adamw_matches_torch_1000_steps():
    """[B1] 官方 1000 步线性回归：与 torch.optim.AdamW 逐元素 atol 1e-4 内一致。"""
    ours = _run_official_adamw_regression(tt.AdamW)
    official = _run_official_adamw_regression(torch.optim.AdamW)
    maxdiff = _maxdiff(ours, official)
    print(f"    [B1] AdamW vs torch.optim.AdamW 1000 步 maxdiff = {maxdiff:.3e}")
    _assert_close(ours, official, atol=1e-4, label="B1 AdamW 1000-step weights")


def test_B2_adamw_zero_grad_shrinks_by_weight_decay():
    """[B2] 梯度为 0 时，参数只按 (1 - lr*wd) 收缩（moment 不产生更新）。"""
    p = nn.Parameter(torch.tensor([1.0, -2.0, 0.5]))
    lr, wd = 0.1, 0.1
    opt = tt.AdamW([p], lr=lr, weight_decay=wd, betas=(0.9, 0.999))
    before = p.detach().clone()
    p.grad = torch.zeros_like(p)
    opt.step()
    expected = before * (1 - lr * wd)
    _assert_close(p, expected, atol=1e-7, label="B2 weight decay only")


def test_B3_adamw_other_betas_match_torch():
    """[B3] betas=(0.9, 0.95) 下 50 步与 torch.optim.AdamW 一致（1e-6）。"""
    torch.manual_seed(5)
    init = torch.randn(6)

    def run(opt_cls) -> Tensor:
        p = nn.Parameter(init.clone())
        opt = opt_cls([p], lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=1e-3)
        # 固定梯度序列，保证两边看到完全相同的 grad。
        for step in range(50):
            p.grad = torch.full_like(p, 0.01 * (step + 1)) * torch.linspace(-1, 1, 6)
            opt.step()
        return p.detach()

    ours = run(tt.AdamW)
    official = run(torch.optim.AdamW)
    _assert_close(ours, official, atol=1e-6, label="B3 AdamW betas=(0.9,0.95)")


def test_B4_adamw_param_groups_use_own_lr():
    """[B4] param groups 各自 lr 生效：两组分别与 torch 的同结构优化器一致。"""
    torch.manual_seed(6)
    g1 = torch.randn(4)
    g2 = torch.randn(4)

    def run(opt_cls):
        p1 = nn.Parameter(torch.ones(4))
        p2 = nn.Parameter(torch.ones(4))
        opt = opt_cls(
            [
                {"params": [p1], "lr": 1e-3},
                {"params": [p2], "lr": 1e-2},
            ],
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )
        p1.grad = g1.clone()
        p2.grad = g2.clone()
        opt.step()
        return p1.detach(), p2.detach()

    ours = run(tt.AdamW)
    official = run(torch.optim.AdamW)
    _assert_close(ours[0], official[0], atol=1e-6, label="B4 group1 (lr=1e-3)")
    _assert_close(ours[1], official[1], atol=1e-6, label="B4 group2 (lr=1e-2)")
    _assert(
        _maxdiff(ours[0], ours[1]) > 1e-3,
        "B4 两组 lr 不同，但更新量几乎一样：param group 的 lr 没有独立生效",
    )


def test_B5_adamw_state_dict_resume_is_exact():
    """[B5] state_dict 存取后继续训练，与不中断训练的最终参数逐位一致。"""
    torch.manual_seed(7)
    init = torch.randn(5)
    grads = [torch.randn(5) for _ in range(8)]

    def make():
        p = nn.Parameter(init.clone())
        return p, tt.AdamW([p], lr=1e-2, weight_decay=0.01)

    # 参考：连续跑 8 步。
    p_ref, opt_ref = make()
    for g in grads:
        p_ref.grad = g.clone()
        opt_ref.step()

    # 中断：跑 4 步 -> 存 -> 新优化器读回 -> 再跑 4 步。
    p_run, opt_run = make()
    for g in grads[:4]:
        p_run.grad = g.clone()
        opt_run.step()
    saved_state = copy.deepcopy(opt_run.state_dict())
    p_new = nn.Parameter(p_run.detach().clone())
    opt_new = tt.AdamW([p_new], lr=1e-2, weight_decay=0.01)
    opt_new.load_state_dict(saved_state)
    for g in grads[4:]:
        p_new.grad = g.clone()
        opt_new.step()

    _assert_close(p_new, p_ref, atol=0.0, label="B5 resume 后参数应逐位一致")


def test_B6_adamw_invalid_lr_raises_value_error():
    """[B6] 非法超参抛 ValueError（负 lr / 负 weight_decay / 非法 beta）。"""
    p = nn.Parameter(torch.zeros(1))
    _expect_raises(ValueError, lambda: tt.AdamW([p], lr=-1e-3), label="B6 lr=-1e-3")
    _expect_raises(ValueError, lambda: tt.AdamW([p], weight_decay=-0.1), label="B6 wd<0")
    _expect_raises(ValueError, lambda: tt.AdamW([p], betas=(0.9, 1.0)), label="B6 beta=1.0")
    # lr=0 是合法边界（冻结更新），不应该报错。
    tt.AdamW([p], lr=0.0)


def test_B7_adamw_closure_called_and_returns_loss():
    """[B7] closure 被调用一次，step 返回它的 loss；closure 内反传有效。"""
    torch.manual_seed(8)
    p = nn.Parameter(torch.tensor([1.0, -2.0]))
    opt = tt.AdamW([p], lr=1e-2)
    calls = {"n": 0}

    def closure():
        calls["n"] += 1
        opt.zero_grad()
        loss = (p**2).sum()
        loss.backward()
        return loss

    returned = opt.step(closure)
    _assert(calls["n"] == 1, f"B7 closure 应被调用 1 次，实际 {calls['n']} 次")
    _assert(returned is not None, "B7 有 closure 时 step 应返回 loss，实际 None")
    _assert_close(returned, 5.0, atol=1e-6, label="B7 closure loss")
    _assert(p.grad is not None, "B7 closure 内 backward 后 p.grad 不应是 None")


# ===========================================================================
# C. 学习率调度（SPEC §4 C1–C4）
# ===========================================================================
def test_C1_cosine_matches_official_25_values():
    """[C1] 官方 test_optimizer.py 的 25 个期望值逐元素一致（1e-12）。"""
    actual = [tt.cosine_lr(it, 1.0, 0.1, 7, 21) for it in range(25)]
    maxdiff = max(abs(a - b) for a, b in zip(actual, OFFICIAL_COSINE_LRS))
    print(f"    [C1] cosine 25 点 maxdiff = {maxdiff:.3e}")
    _assert_close(
        actual, OFFICIAL_COSINE_LRS, atol=1e-12, label="C1 official cosine schedule"
    )


def test_C2_cosine_boundaries():
    """[C2] t=0→0、t=Tw→max、t=Tc→min、t>Tc→min（含 t=Tc+1）。"""
    lr0 = tt.cosine_lr(0, 1.0, 0.1, 7, 21)
    lr_w = tt.cosine_lr(7, 1.0, 0.1, 7, 21)
    lr_c = tt.cosine_lr(21, 1.0, 0.1, 7, 21)
    lr_after = tt.cosine_lr(22, 1.0, 0.1, 7, 21)
    lr_far = tt.cosine_lr(1000, 1.0, 0.1, 7, 21)
    _assert_close(lr0, 0.0, atol=1e-12, label="C2 it=0")
    _assert_close(lr_w, 1.0, atol=1e-12, label="C2 it=Tw (应与 cosine 段连续=max)")
    _assert_close(lr_c, 0.1, atol=1e-12, label="C2 it=Tc")
    _assert_close(lr_after, 0.1, atol=1e-12, label="C2 it=Tc+1")
    _assert_close(lr_far, 0.1, atol=1e-12, label="C2 it=1000")


def test_C3_cosine_zero_warmup_returns_max_at_zero():
    """[C3] warmup_iters=0 时 t=0 走 cosine 分支 progress=0，返回 max（不是除零）。"""
    _assert_close(
        tt.cosine_lr(0, 2.0, 0.2, 0, 10), 2.0, atol=1e-12, label="C3 warmup=0, it=0"
    )
    _assert_close(
        tt.cosine_lr(10, 2.0, 0.2, 0, 10), 0.2, atol=1e-12, label="C3 warmup=0, it=Tc"
    )


def test_C4_set_lr_updates_all_param_groups():
    """[C4] set_lr 原地修改所有 param_group 的 lr（包括自定义不同初值的组）。"""
    p1, p2 = nn.Parameter(torch.zeros(2)), nn.Parameter(torch.zeros(2))
    opt = torch.optim.AdamW(
        [
            {"params": [p1], "lr": 0.1},
            {"params": [p2], "lr": 0.2},
        ]
    )
    tt.set_lr(opt, 0.123)
    lrs = [group["lr"] for group in opt.param_groups]
    _assert(
        lrs == [0.123, 0.123],
        f"C4 set_lr 后应全部为 0.123，实际 {lrs}",
    )


# ===========================================================================
# D. 梯度裁剪（SPEC §4 D1–D4）
# ===========================================================================
def _make_grad_params(shapes: list[tuple[int, ...]], scale: float, seed: int):
    """构造一组参数并把同一份随机梯度同时挂到两组参数上，用于对比实现。"""
    torch.manual_seed(seed)
    grads = [torch.randn(shape) * scale for shape in shapes]
    ours = [nn.Parameter(torch.zeros(shape)) for shape in shapes]
    official = [nn.Parameter(torch.zeros(shape)) for shape in shapes]
    for p, q, g in zip(ours, official, grads):
        p.grad = g.clone()
        q.grad = g.clone()
    return ours, official


def test_D1_clip_matches_torch_clip_grad_norm():
    """[D1] norm > max 时，与 torch.nn.utils.clip_grad_norm_ 的裁剪结果一致。"""
    ours, official = _make_grad_params([(12,), (4,)], scale=3.0, seed=9)
    pre_norm = tt.clip_gradients(ours, max_l2_norm=1.0, eps=1e-6)
    torch_norm = float(torch.nn.utils.clip_grad_norm_(official, max_norm=1.0))
    for i, (p, q) in enumerate(zip(ours, official)):
        _assert_close(p.grad, q.grad, atol=1e-6, label=f"D1 clipped grad[{i}]")
    _assert_close(pre_norm, torch_norm, atol=1e-5, label="D1 返回的裁剪前 norm")


def test_D2_clip_no_op_when_norm_below_max():
    """[D2] norm <= max 时梯度一个 bit 不动，返回原始全局 norm。"""
    ours, _ = _make_grad_params([(5,), (3,)], scale=0.1, seed=10)
    before = [p.grad.detach().clone() for p in ours]
    expected_norm = math.sqrt(sum(float(g.double().pow(2).sum()) for g in before))
    returned = tt.clip_gradients(ours, max_l2_norm=100.0, eps=1e-6)
    _assert_close(returned, expected_norm, atol=1e-6, label="D2 返回的 norm")
    for i, (p, ref) in enumerate(zip(ours, before)):
        _assert(
            torch.equal(p.grad, ref),
            f"D2 norm 未超阈值，grad[{i}] 不应被修改（maxdiff={_maxdiff(p.grad, ref):.3e}）",
        )


def test_D3_clipped_norm_close_to_max():
    """[D3] 裁剪后全局 norm ≈ max（略小，因为缩放因子是 max/(norm+eps)）。"""
    ours, _ = _make_grad_params([(20,), (7,)], scale=5.0, seed=11)
    max_norm = 0.5
    pre = tt.clip_gradients(ours, max_l2_norm=max_norm, eps=1e-6)
    _assert(pre > max_norm, f"D3 构造的裁剪前 norm={pre:.4f} 应大于 max={max_norm}")
    post = math.sqrt(sum(float(p.grad.double().pow(2).sum()) for p in ours))
    _assert(post <= max_norm + 1e-12, f"D3 裁剪后 norm={post:.10f} 不应超过 max={max_norm}")
    # 理论值 max * pre / (pre + eps)：验证 eps 用的是加法而不是省略。
    theoretical = max_norm * pre / (pre + 1e-6)
    _assert_close(post, theoretical, atol=1e-8, label="D3 post-clip norm 与 max/(norm+eps) 一致")
    _assert(
        (max_norm - post) / max_norm < 1e-5,
        f"D3 裁剪后 norm 应非常接近 max，实际 post={post:.10f}, max={max_norm}",
    )


def test_D4_clip_skips_none_grads():
    """[D4] None grad 被跳过，返回裁剪前 norm（只统计有梯度的参数）。"""
    p1 = nn.Parameter(torch.zeros(2))
    p2 = nn.Parameter(torch.zeros(3))
    p1.grad = torch.tensor([3.0, 4.0])  # norm=5
    p2.grad = None
    returned = tt.clip_gradients([p1, p2], max_l2_norm=1.0, eps=1e-6)
    _assert_close(returned, 5.0, atol=1e-12, label="D4 返回裁剪前 norm")
    _assert(p2.grad is None, "D4 p2.grad 应保持 None，而不是被写成 0")
    post = math.sqrt(float(p1.grad.double().pow(2).sum()))
    _assert(
        abs(post - 1.0 * 5.0 / (5.0 + 1e-6)) < 1e-6,
        f"D4 p1 应按 1/(5+eps) 缩放，实际 norm={post:.10f}",
    )


# ===========================================================================
# E. 数据加载（SPEC §4 E1–E5）
# ===========================================================================
def test_E1_get_batch_shapes_and_offset():
    """[E1] 返回 (B,T) int64、y == x+1（语言建模的 next-token 偏移）。"""
    dataset = np.arange(0, 200, dtype=np.uint16)
    x, y = tt.get_batch(dataset, batch_size=8, context_length=7, rng=np.random.default_rng(0))
    _assert(tuple(x.shape) == (8, 7), f"E1 x.shape 应为 (8,7)，实际 {tuple(x.shape)}")
    _assert(tuple(y.shape) == (8, 7), f"E1 y.shape 应为 (8,7)，实际 {tuple(y.shape)}")
    _assert(x.dtype == torch.int64, f"E1 x.dtype 应为 int64，实际 {x.dtype}")
    _assert(y.dtype == torch.int64, f"E1 y.dtype 应为 int64，实际 {y.dtype}")
    _assert(x.device.type == "cpu", f"E1 默认 device 应为 cpu，实际 {x.device}")
    _assert(bool(torch.equal(y, x + 1)), "E1 y 应恒等于 x+1")


def test_E2_get_batch_start_uniformity():
    """[E2] 1000 次采样的起点覆盖 [0, len-T-1] 且服从均匀分布（官方 5σ 口径）。"""
    dataset = np.arange(0, 100)
    context_length, batch_size = 7, 32
    num_iters = 1000
    np.random.seed(1234)  # 固定种子：保持测试确定性，但仍能暴露实现偏差
    counter: Counter[int] = Counter()
    for _ in range(num_iters):
        x, _ = tt.get_batch(dataset, batch_size, context_length)
        counter.update(int(v) for v in x[:, 0].tolist())

    num_possible = len(dataset) - context_length  # 93
    observed_min, observed_max = min(counter), max(counter)
    _assert(
        observed_min == 0,
        f"E2 最小起点应为 0（上界应是 exclusive 的 0），实际 {observed_min}",
    )
    _assert(
        observed_max == num_possible - 1,
        f"E2 最大起点应为 len-T-1={num_possible - 1}，实际 {observed_max}（上界写错？）",
    )
    expected_count = (num_iters * batch_size) / num_possible
    std = math.sqrt(
        (num_iters * batch_size) * (1 / num_possible) * (1 - 1 / num_possible)
    )
    lo, hi = expected_count - 5 * std, expected_count + 5 * std
    bad = {k: v for k, v in counter.items() if not lo <= v <= hi}
    _assert(
        not bad,
        f"E2 起点计数超出 5σ 区间 [{lo:.1f}, {hi:.1f}]：{sorted(bad.items())[:5]}",
    )


def test_E3_memmap_dataset_readable_and_in_bounds():
    """[E3] load_tokens 读裸 uint16 文件为 memmap；采样起点不越界、内容正确。"""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "tokens.bin"
        raw = (np.arange(0, 50, dtype=np.uint16) * 2).astype(np.uint16)
        raw.tofile(path)
        dataset = tt.load_tokens(path)
        _assert(
            isinstance(dataset, np.memmap),
            f"E3 load_tokens 应返回 np.memmap，实际 {type(dataset)}",
        )
        _assert(
            dataset.dtype == np.uint16,
            f"E3 默认 dtype 应为 uint16，实际 {dataset.dtype}",
        )
        _assert(len(dataset) == 50, f"E3 长度应为 50，实际 {len(dataset)}")
        x, y = tt.get_batch(dataset, batch_size=16, context_length=7, rng=np.random.default_rng(3))
        # raw[i] = 2*i，所以由每行第一个值反推起始索引（用来验证上界）。
        starts = [int(v) // 2 for v in x[:, 0].tolist()]
        _assert(
            max(starts) <= len(dataset) - 7 - 1,
            f"E3 起点越界：max start={max(starts)} > {len(dataset) - 7 - 1}",
        )
        # 每行都必须是原序列的连续窗口（memmap 切片读取正确）。
        for row in range(x.shape[0]):
            start = starts[row]
            expected = raw[start : start + 7]
            _assert(
                np.array_equal(x[row].numpy(), expected.astype(np.int64)),
                f"E3 第 {row} 行不是 dataset[{start}:{start + 7}]（memmap 读取错位）",
            )
            _assert(
                np.array_equal(y[row].numpy(), raw[start + 1 : start + 8].astype(np.int64)),
                f"E3 第 {row} 行 y 不是 dataset[{start + 1}:{start + 8}]",
            )


def test_E4_get_batch_rng_reproducible():
    """[E4] 给定相同 seed 的 Generator，两次采样完全一致；不同 seed 应不同。"""
    dataset = np.arange(0, 500, dtype=np.uint16)
    x1, y1 = tt.get_batch(dataset, 8, 16, rng=np.random.default_rng(42))
    x2, y2 = tt.get_batch(dataset, 8, 16, rng=np.random.default_rng(42))
    _assert(torch.equal(x1, x2) and torch.equal(y1, y2), "E4 相同 seed 的 rng 应完全可复现")
    x3, _ = tt.get_batch(dataset, 8, 16, rng=np.random.default_rng(43))
    _assert(not torch.equal(x1, x3), "E4 不同 seed 的 rng 不应产生完全相同的 batch")


def test_E5_get_batch_invalid_device_raises():
    """[E5] 非法 device 抛异常（官方用 cuda:99 口径）。"""
    dataset = np.arange(0, 100)
    exc = _expect_raises(
        (RuntimeError, AssertionError, ValueError),
        lambda: tt.get_batch(dataset, 2, 7, device="cuda:99"),
        label="E5 invalid device=cuda:99",
    )
    _assert(bool(str(exc)), "E5 非法 device 应抛出带消息的异常")


# ===========================================================================
# F. checkpoint（SPEC §4 F1–F4）
# ===========================================================================
def test_F1_checkpoint_roundtrip_matches_official():
    """[F1] 官方 test_serialization.py 口径：model/optimizer/iteration 往返一致。"""
    torch.manual_seed(12)
    model = _TestNet(d_input=20, d_output=5)
    optimizer = tt.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    it = 0
    for _ in range(10):
        optimizer.zero_grad()
        x = torch.rand(20)
        y = torch.rand(5)
        loss = ((y - model(x)) ** 2).sum()
        loss.backward()
        optimizer.step()
        it += 1

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "checkpoint.pt"
        tt.save_checkpoint(model, optimizer, iteration=it, out=path)
        new_model = _TestNet(d_input=20, d_output=5)
        new_optimizer = tt.AdamW(new_model.parameters(), lr=1e-3, weight_decay=0.01)
        loaded_it = tt.load_checkpoint(path, new_model, new_optimizer)

    _assert(loaded_it == it, f"F1 返回的 iteration 应为 {it}，实际 {loaded_it}")
    for key, value in model.state_dict().items():
        _assert(
            torch.equal(value, new_model.state_dict()[key]),
            f"F1 model state[{key}] 往返后不一致（maxdiff={_maxdiff(value, new_model.state_dict()[key]):.3e}）",
        )
    equal, reason = _optimizer_state_dicts_equal(
        optimizer.state_dict(), new_optimizer.state_dict()
    )
    _assert(equal, f"F1 optimizer state_dict 往返后不一致：{reason}")


def test_F2_interrupted_resume_is_equivalent():
    """[F2] 6 步 vs 3 步+存+读+3 步：最终参数应逐位一致（<1e-7）。"""
    torch.manual_seed(13)
    init_state = copy.deepcopy(nn.Linear(4, 3).state_dict())
    torch.manual_seed(14)
    batches = [(torch.randn(5, 4), torch.randn(5, 3)) for _ in range(6)]

    def new_model() -> nn.Module:
        model = nn.Linear(4, 3)
        model.load_state_dict(copy.deepcopy(init_state))
        return model

    def train_step(model, optimizer, x, y):
        optimizer.zero_grad()
        loss = ((model(x) - y) ** 2).mean()
        loss.backward()
        optimizer.step()

    # 参考：连续 6 步。
    model_ref = new_model()
    opt_ref = tt.AdamW(model_ref.parameters(), lr=1e-2, weight_decay=0.01)
    for x, y in batches:
        train_step(model_ref, opt_ref, x, y)

    # 中断：3 步 -> BytesIO 存 -> 全新 model/optimizer 读 -> 3 步。
    model_run = new_model()
    opt_run = tt.AdamW(model_run.parameters(), lr=1e-2, weight_decay=0.01)
    for x, y in batches[:3]:
        train_step(model_run, opt_run, x, y)
    buffer = io.BytesIO()
    tt.save_checkpoint(model_run, opt_run, iteration=3, out=buffer)
    buffer.seek(0)

    model_resumed = nn.Linear(4, 3)  # 故意用随机初始化的新模型，靠 checkpoint 覆盖
    opt_resumed = tt.AdamW(model_resumed.parameters(), lr=1e-2, weight_decay=0.01)
    it = tt.load_checkpoint(buffer, model_resumed, opt_resumed)
    _assert(it == 3, f"F2 恢复的 iteration 应为 3，实际 {it}")
    for x, y in batches[3:]:
        train_step(model_resumed, opt_resumed, x, y)

    for (name_a, param_a), (name_b, param_b) in zip(
        model_ref.named_parameters(), model_resumed.named_parameters()
    ):
        _assert(
            _maxdiff(param_a, param_b) < 1e-7,
            f"F2 恢复等价性失败：{name_a} vs {name_b} maxdiff={_maxdiff(param_a, param_b):.3e}",
        )


def test_F3_checkpoint_bytesio_supported():
    """[F3] file-like（BytesIO）也能保存/读取，且状态与路径保存等价。"""
    torch.manual_seed(15)
    model = _TestNet(d_input=8, d_output=3)
    optimizer = tt.AdamW(model.parameters(), lr=1e-2)
    buffer = io.BytesIO()
    tt.save_checkpoint(model, optimizer, iteration=99, out=buffer)
    buffer.seek(0)

    model2 = _TestNet(d_input=8, d_output=3)
    optimizer2 = tt.AdamW(model2.parameters(), lr=1e-2)
    loaded_it = tt.load_checkpoint(buffer, model2, optimizer2)
    _assert(loaded_it == 99, f"F3 从 BytesIO 读出的 iteration 应为 99，实际 {loaded_it}")
    for key, value in model.state_dict().items():
        _assert(
            torch.equal(value, model2.state_dict()[key]),
            f"F3 BytesIO 往返后 model state[{key}] 不一致",
        )


def test_F4_load_checkpoint_returns_saved_iteration():
    """[F4] load_checkpoint 返回保存时的 iteration（不是 0 / 不是当前值）。"""
    model = _TestNet(d_input=4, d_output=2)
    optimizer = tt.AdamW(model.parameters(), lr=1e-3)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ckpt.pt"
        tt.save_checkpoint(model, optimizer, iteration=7, out=path)
        fresh_model = _TestNet(d_input=4, d_output=2)
        fresh_optimizer = tt.AdamW(fresh_model.parameters(), lr=1e-3)
        loaded = tt.load_checkpoint(path, fresh_model, fresh_optimizer)
    _assert(loaded == 7, f"F4 应返回 7，实际 {loaded}")


# ===========================================================================
# G. 解码（SPEC §4 G1–G5）
# ===========================================================================
def test_G1_low_temperature_is_argmax():
    """[G1] 极低温度下 softmax 近似 one-hot，采样结果恒为 argmax。"""
    logits = torch.tensor([0.3, 2.0, -1.0, 1.999, 0.0])
    expected = int(torch.argmax(logits))
    probs = tt.softmax_temperature(logits, temperature=1e-6)
    _assert_close(
        probs, torch.eye(5)[expected], atol=1e-6, label="G1 低温 softmax≈one-hot"
    )
    generated = {
        tt.sample_token(logits, temperature=1e-6, generator=torch.Generator().manual_seed(s))
        for s in range(10)
    }
    _assert(
        generated == {expected},
        f"G1 极低温应恒为 argmax={expected}，实际采样到 {sorted(generated)}",
    )


def test_G2_top_p_one_keeps_all_tokens():
    """[G2] top_p=1.0 时保留全部正概率 token，输出与输入一致且和为 1。"""
    torch.manual_seed(16)
    probs = torch.softmax(torch.randn(20) * 2.0, dim=-1)
    filtered = tt.top_p_filter(probs, top_p=1.0)
    _assert_close(filtered, probs, atol=1e-7, label="G2 top_p=1 输出")
    _assert_close(filtered.sum(), 1.0, atol=1e-6, label="G2 归一化")
    _assert(
        bool((filtered[probs > 0] > 0).all()),
        "G2 top_p=1 不应把任何正概率 token 置 0",
    )


def test_G3_top_p_keeps_minimal_nucleus():
    """[G3] 手工概率 [0.5,0.3,0.15,0.05]：p=0.7 保留前 2 个、p=0.9 保留前 3 个。"""
    probs = torch.tensor([0.5, 0.3, 0.15, 0.05])

    def kept(p: float) -> list[int]:
        return [
            i
            for i, v in enumerate(tt.top_p_filter(probs, top_p=p).tolist())
            if v > 0
        ]

    # 累计和：0.5, 0.8, 0.95, 1.0。p=0.7：前 2 个（第 2 个使累计首次 >=0.7）。
    _assert(kept(0.7) == [0, 1], f"G3 p=0.7 应保留 [0,1]，实际 {kept(0.7)}")
    out_07 = tt.top_p_filter(probs, top_p=0.7)
    _assert_close(
        out_07,
        torch.tensor([0.5 / 0.8, 0.3 / 0.8, 0.0, 0.0]),
        atol=1e-7,
        label="G3 p=0.7 归一化后的概率",
    )
    # p=0.9：前 3 个（累计 0.8 < 0.9，第 3 个 0.95 >= 0.9）。
    _assert(kept(0.9) == [0, 1, 2], f"G3 p=0.9 应保留 [0,1,2]，实际 {kept(0.9)}")
    # 极小 p：至少保留第一个（cumsum-sorted_probs 口径保证）。
    _assert(kept(0.01) == [0], f"G3 p=0.01 应至少保留 [0]，实际 {kept(0.01)}")


def test_G4_temperature_controls_entropy():
    """[G4] 温度降低 ⇒ 分布熵下降；温度升高 ⇒ 熵上升。"""
    torch.manual_seed(17)
    logits = torch.randn(30) * 2.0

    def entropy(temperature: float) -> float:
        probs = tt.softmax_temperature(logits, temperature)
        return float(-(probs * probs.log()).sum())

    cold = entropy(0.5)
    neutral = entropy(1.0)
    hot = entropy(2.0)
    _assert(cold < neutral < hot, f"G4 熵应随温度单调上升，实际 {cold:.4f} < {neutral:.4f} < {hot:.4f} 不成立")


def test_G5_generate_respects_limits_and_seed():
    """[G5] generate：返回含 prompt、seed 可复现、遵守 max_new_tokens/eos/context 截断。"""
    vocab = 12
    model = _ScriptedLM(vocab)
    prompt = [0, 1, 2, 3]

    # (1) seed 可复现 + 返回含 prompt + 长度 = prompt + max_new_tokens。
    gen1 = torch.Generator().manual_seed(7)
    out1 = tt.generate(model, prompt, max_new_tokens=5, temperature=1.0, generator=gen1)
    gen2 = torch.Generator().manual_seed(7)
    out2 = tt.generate(model, prompt, max_new_tokens=5, temperature=1.0, generator=gen2)
    _assert(out1 == out2, f"G5 相同 seed 的 generate 应完全一致：{out1} vs {out2}")
    _assert(out1[: len(prompt)] == prompt, f"G5 返回值应包含 prompt 前缀，实际 {out1}")
    _assert(
        len(out1) == len(prompt) + 5,
        f"G5 应生成 5 个 token（共 {len(prompt) + 5} 个），实际 {len(out1)}",
    )
    _assert(
        all(0 <= t < vocab for t in out1), f"G5 生成 token 超出词表范围：{out1}"
    )

    # (2) eos 立刻截断，且 eos 本身包含在返回值里。
    eos = out1[len(prompt)]  # 第一次新生成的 token
    gen3 = torch.Generator().manual_seed(7)
    out_eos = tt.generate(
        model, prompt, max_new_tokens=10, temperature=1.0,
        eos_token_id=eos, generator=gen3,
    )
    _assert(
        len(out_eos) == len(prompt) + 1,
        f"G5 第一 token 即 eos 时应停在 prompt+1，实际长度 {len(out_eos)}",
    )
    _assert(out_eos[-1] == eos, f"G5 eos 应包含在返回值末尾，实际 {out_eos}")

    # (3) context 截断：prompt 长 8、context=4，模型每次只能看到最近 4 个 token。
    model.input_lengths.clear()
    long_prompt = list(range(8))
    tt.generate(
        model, long_prompt, max_new_tokens=3, temperature=1.0, context_length=4,
        generator=torch.Generator().manual_seed(3),
    )
    _assert(
        model.input_lengths == [4, 4, 4],
        f"G5 每步输入长度应被截到 context_length=4，实际 {model.input_lengths}",
    )


# ===========================================================================
# H. 集成（SPEC §4 H1）
# ===========================================================================
def test_H1_small_model_trains_with_toolbox():
    """[H1] Embedding+Linear 用本工具箱训 50 步，loss 下降 > 50%（端到端可训练）。"""
    torch.manual_seed(18)
    np.random.seed(18)
    vocab_size, context_length, d_model = 32, 8, 16
    # 确定性序列：dataset[i] = i % 32，则 next token 是当前 token+1，模型可学。
    dataset = (np.arange(0, 400, dtype=np.uint16) % vocab_size).astype(np.uint16)

    model = nn.Sequential(
        nn.Embedding(vocab_size, d_model),
        nn.Linear(d_model, vocab_size),
    )
    optimizer = tt.AdamW(model.parameters(), lr=2e-2, weight_decay=0.0)
    rng = np.random.default_rng(0)
    losses: list[float] = []

    for step in range(50):
        tt.set_lr(optimizer, tt.cosine_lr(step, 2e-2, 2e-4, warmup_iters=5, cosine_cycle_iters=50))
        x, y = tt.get_batch(dataset, batch_size=16, context_length=context_length, rng=rng)
        logits = model(x)
        loss = tt.cross_entropy(logits, y)
        optimizer.zero_grad()
        loss.backward()
        tt.clip_gradients(model.parameters(), max_l2_norm=1.0)
        optimizer.step()
        losses.append(float(loss.item()))

    initial, final = losses[0], losses[-1]
    print(f"    [H1] 50 步 loss: {initial:.4f} -> {final:.4f}")
    _assert(all(math.isfinite(x) for x in losses), "H1 loss 出现 NaN/Inf")
    _assert(
        final < initial * 0.5,
        f"H1 loss 应下降 > 50%，实际 {initial:.4f} -> {final:.4f}"
        f"（下降 {100 * (initial - final) / initial:.1f}%）",
    )


# ===========================================================================
# 双模式入口
# ===========================================================================
def _all_tests() -> list[tuple[str, object]]:
    """按定义顺序收集本模块所有 `test_*` 函数（A→H 编号方便对照 SPEC）。"""
    return [
        (name, obj)
        for name, obj in globals().items()
        if name.startswith("test_") and callable(obj)
    ]


def main() -> int:
    torch.set_num_threads(2)
    tests = _all_tests()
    passed = failed = 0
    start = time.perf_counter()

    print(f"运行 {len(tests)} 个测试（torch={torch.__version__}，线程数={torch.get_num_threads()}）\n")
    for name, fn in tests:
        try:
            fn()  # type: ignore[operator]
        except Exception as exc:  # noqa: BLE001 - 测试运行器要抓住所有异常
            failed += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
        else:
            passed += 1
            print(f"PASS {name}")

    elapsed = time.perf_counter() - start
    print(f"\n汇总: {passed} passed, {failed} failed, 共 {len(tests)} 项")
    print(f"耗时: {elapsed:.2f}s，线程数: {torch.get_num_threads()}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
