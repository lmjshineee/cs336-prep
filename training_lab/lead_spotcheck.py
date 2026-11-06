#!/usr/bin/env python
"""training_lab/lead_spotcheck.py — Lead 的独立复核脚本（不 import impl 的测试代码）。

目的：只按 SPEC §3 的契约，用**官方口径 / PyTorch 参考实现**重算一遍关键数值，
确认 `toy_training.py` 不是「测试自己证明自己」。每一项都打印 maxdiff，
全过 exit 0，任何一项失败 exit 1。

用法::

    .venv/bin/python training_lab/lead_spotcheck.py
"""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(2)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import toy_training as tt  # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, fn) -> None:
    try:
        ok, detail = fn()
    except Exception as exc:  # noqa: BLE001 - 复核脚本要把异常变成 FAIL
        ok, detail = False, f"{type(exc).__name__}: {exc}"
    RESULTS.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


# ---------------------------------------------------------------------------
# 1. cross-entropy：与 F.cross_entropy 对拍（含大 logits 稳定性 + 梯度）
# ---------------------------------------------------------------------------
def check_ce() -> tuple[bool, str]:
    torch.manual_seed(0)
    logits = torch.randn(4, 6, 11)
    targets = torch.randint(0, 11, (4, 6))
    got = tt.cross_entropy(logits, targets)
    want = F.cross_entropy(logits.reshape(-1, 11), targets.reshape(-1))
    d1 = (got - want).abs().item()

    big = logits * 1000.0
    got_big = tt.cross_entropy(big, targets)
    want_big = F.cross_entropy(big.reshape(-1, 11), targets.reshape(-1))
    d2 = abs(got_big.item() - want_big.item())

    # 梯度对拍
    x1 = logits.clone().requires_grad_(True)
    x2 = logits.clone().requires_grad_(True)
    tt.cross_entropy(x1, targets).backward()
    F.cross_entropy(x2.reshape(-1, 11), targets.reshape(-1)).backward()
    d3 = (x1.grad - x2.grad).abs().max().item()

    ok = d1 < 1e-5 and d2 < 1e-3 and d3 < 1e-6 and math.isfinite(got_big.item())
    return ok, f"maxdiff={d1:.2e}, large={d2:.2e}, grad={d3:.2e}"


# ---------------------------------------------------------------------------
# 2. AdamW：官方 test_optimizer.py 的 1000 步线性回归口径
# ---------------------------------------------------------------------------
def _optimize(opt_cls) -> torch.Tensor:
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


def check_adamw() -> tuple[bool, str]:
    ours = _optimize(tt.AdamW)
    ref = _optimize(torch.optim.AdamW)
    d = (ours - ref).abs().max().item()
    return d < 1e-4, f"1000-step weight maxdiff={d:.2e} (atol=1e-4)"


# ---------------------------------------------------------------------------
# 3. cosine schedule：官方 25 个期望值
# ---------------------------------------------------------------------------
EXPECTED_LRS = [
    0, 0.14285714285714285, 0.2857142857142857, 0.42857142857142855,
    0.5714285714285714, 0.7142857142857143, 0.8571428571428571, 1.0,
    0.9887175604818206, 0.9554359905560885, 0.9018241671106134,
    0.8305704108364301, 0.7452476826029011, 0.6501344202803414, 0.55,
    0.44986557971965857, 0.3547523173970989, 0.26942958916356996,
    0.19817583288938662, 0.14456400944391146, 0.11128243951817937,
    0.1, 0.1, 0.1, 0.1,
]


def check_cosine() -> tuple[bool, str]:
    got = [
        tt.cosine_lr(it, 1.0, 0.1, warmup_iters=7, cosine_cycle_iters=21)
        for it in range(25)
    ]
    d = float(np.max(np.abs(np.array(got) - np.array(EXPECTED_LRS))))
    return d < 1e-12, f"25-point maxdiff={d:.2e}"


# ---------------------------------------------------------------------------
# 4. gradient clipping：与 torch.nn.utils.clip_grad_norm_ 对拍
# ---------------------------------------------------------------------------
def check_clip() -> tuple[bool, str]:
    torch.manual_seed(1)
    shapes = [(3, 4), (5,), (2, 2)]
    grads = [torch.randn(s) * 10 for s in shapes]

    params_a = [torch.nn.Parameter(torch.zeros(s)) for s in shapes]
    params_b = [torch.nn.Parameter(torch.zeros(s)) for s in shapes]
    for p, g in zip(params_a, grads):
        p.grad = g.clone()
    for p, g in zip(params_b, grads):
        p.grad = g.clone()

    norm = tt.clip_gradients(params_a, max_l2_norm=1.0)
    ref_norm = torch.nn.utils.clip_grad_norm_(params_b, max_norm=1.0)
    d = max((a.grad - b.grad).abs().max().item() for a, b in zip(params_a, params_b))
    dn = abs(norm - ref_norm.item())
    ok = d < 1e-7 and dn < 1e-6 and norm > 1.0
    return ok, f"grad maxdiff={d:.2e}, norm diff={dn:.2e}, norm={norm:.4f}"


# ---------------------------------------------------------------------------
# 5. get_batch：形状 / offset / start 范围 / 分布
# ---------------------------------------------------------------------------
def check_get_batch() -> tuple[bool, str]:
    dataset = np.arange(0, 100, dtype=np.uint16)
    x, y = tt.get_batch(dataset, batch_size=32, context_length=7, device="cpu")
    shape_ok = x.shape == (32, 7) and y.shape == (32, 7)
    offset_ok = bool(torch.equal(x + 1, y))
    n_possible = len(dataset) - 7  # 93
    starts = set()
    for _ in range(200):
        xb, _ = tt.get_batch(dataset, 32, 7, "cpu")
        starts.update(xb[:, 0].tolist())
    range_ok = min(starts) == 0 and max(starts) == n_possible - 1
    # 93 个起点应全部覆盖（200*32 次抽样，几乎必然）
    coverage_ok = len(starts) == n_possible
    dtype_ok = x.dtype == torch.int64
    ok = shape_ok and offset_ok and range_ok and coverage_ok and dtype_ok
    return ok, (
        f"shape={shape_ok}, offset={offset_ok}, start_range={range_ok}, "
        f"covered={len(starts)}/{n_possible}, dtype={dtype_ok}"
    )


# ---------------------------------------------------------------------------
# 6. checkpoint：往返 + 恢复等价
# ---------------------------------------------------------------------------
def _train_steps(model, opt, steps: int, seed: int = 0) -> None:
    torch.manual_seed(seed)
    for _ in range(steps):
        opt.zero_grad()
        x = torch.randn(8, 4)
        loss = (model(x) ** 2).mean()
        loss.backward()
        opt.step()


def check_checkpoint() -> tuple[bool, str]:
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Tanh(), torch.nn.Linear(8, 2))
    opt = tt.AdamW(model.parameters(), lr=1e-2)

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "ckpt.pt"
        _train_steps(model, opt, 4)
        tt.save_checkpoint(model, opt, iteration=4, out=path)

        model2 = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Tanh(), torch.nn.Linear(8, 2))
        opt2 = tt.AdamW(model2.parameters(), lr=1e-2)
        it = tt.load_checkpoint(path, model2, opt2)

        same_model = all(
            torch.equal(a, b) for a, b in zip(model.state_dict().values(), model2.state_dict().values())
        )
        same_opt = opt.state_dict()["param_groups"] == opt2.state_dict()["param_groups"]

        # 恢复后继续 2 步 vs 原模型继续 2 步，参数应逐位一致
        _train_steps(model, opt, 2)
        _train_steps(model2, opt2, 2)
        same_after = all(
            torch.equal(a, b) for a, b in zip(model.state_dict().values(), model2.state_dict().values())
        )
    ok = it == 4 and same_model and same_opt and same_after
    return ok, f"iteration={it}, model={same_model}, opt={same_opt}, resume={same_after}"


# ---------------------------------------------------------------------------
# 7. 解码：greedy（低温）+ top-p 最小核
# ---------------------------------------------------------------------------
def check_decode() -> tuple[bool, str]:
    logits = torch.tensor([0.1, 2.0, -1.0, 0.5])
    greedy = tt.sample_token(logits, temperature=1e-3)
    greedy_ok = greedy == 1

    probs = torch.tensor([0.05, 0.05, 0.6, 0.3])
    kept = tt.top_p_filter(probs, top_p=0.6)
    # 排序后 0.6 / 0.3 / 0.05 / 0.05；累计 0.6 >= p，最小核 = {2}
    nucleus_ok = (
        kept[2] > 0.999 and kept.sum().item() == 1.0 and kept[0].item() == 0.0
    )

    probs2 = tt.softmax_temperature(torch.tensor([0.0, 1.0, 2.0]), 0.5)
    ent = -(probs2 * probs2.clamp_min(1e-12).log()).sum().item()
    ok = greedy_ok and nucleus_ok and ent < 0.5
    return ok, f"greedy={greedy_ok}, nucleus={nucleus_ok}, sharp_entropy={ent:.3f}"


# ---------------------------------------------------------------------------
# 8. 端到端：小模型 40 步 loss 下降（我们的 CE + AdamW + clip）
# ---------------------------------------------------------------------------
def check_train_loop() -> tuple[bool, str]:
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Embedding(32, 16),
        torch.nn.Flatten(),
        torch.nn.Linear(16 * 4, 4 * 32),
    )
    opt = tt.AdamW(model.parameters(), lr=1e-2)
    x = torch.randint(0, 32, (16, 4))
    y = torch.randint(0, 32, (16, 4))

    def loss_fn() -> torch.Tensor:
        logits = model(x).view(16, 4, 32)
        return tt.cross_entropy(logits, y)

    first = loss_fn().item()
    for _ in range(40):
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        tt.clip_gradients(model.parameters(), 1.0)
        opt.step()
    last = loss_fn().item()
    ok = last < first * 0.5
    return ok, f"loss {first:.4f} -> {last:.4f} (drop {100 * (1 - last / first):.1f}%)"


def main() -> int:
    print("=" * 70)
    print("Lead 独立复核：training_lab/toy_training.py")
    print("=" * 70)
    check("cross_entropy vs F.cross_entropy", check_ce)
    check("AdamW vs torch.optim.AdamW (1000 steps)", check_adamw)
    check("cosine_lr vs 官方 25 个期望值", check_cosine)
    check("clip_gradients vs torch.clip_grad_norm_", check_clip)
    check("get_batch 形状/offset/范围", check_get_batch)
    check("checkpoint 往返 + 恢复等价", check_checkpoint)
    check("解码 greedy / top-p 最小核", check_decode)
    check("小模型 40 步 loss 下降", check_train_loop)

    failed = [name for name, ok, _ in RESULTS if not ok]
    print("-" * 70)
    print(f"{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    if failed:
        print("FAILED: " + ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
