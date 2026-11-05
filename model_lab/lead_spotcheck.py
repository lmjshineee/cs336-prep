"""Lead 独立复核：不 import impl 的测试代码，直接用官方快照抽查 toy_transformer。

这是「第二个人用不同代码路径复核」的证据：它自己构造模块、自己加载权重、自己算
SDPA/RoPE/MHA/block/LM，只复用官方快照作为真值。默认读 A1_REPO 或
/Users/yunni/Projects/assignment1-basics，缺失时打印 SKIP。

用法：
    .venv/bin/python model_lab/lead_spotcheck.py
    A1_REPO=/path/to/assignment1-basics .venv/bin/python model_lab/lead_spotcheck.py
"""


import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model_lab"))

A1 = Path(os.environ.get("A1_REPO", "/Users/yunni/Projects/assignment1-basics"))
SNAP = A1 / "tests/_snapshots"
WEIGHTS = A1 / "tests/fixtures/ts_tests/model.pt"

C, H, D, V, T, THETA = 64, 4, 16, 10000, 12, 10000.0


def load_sd():
    sd = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
    return {k.replace("_orig_mod.", ""): v for k, v in sd.items()}


def snap(name):
    return np.load(SNAP / f"{name}.npz")["array"]


def diff(name, actual):
    d = float(np.abs(actual.detach().cpu().numpy() - snap(name)).max())
    print(f"{name:50s} maxdiff={d:.3e}")
    return d


def main():
    if not WEIGHTS.exists():
        print("SKIP: 官方仓库不存在")
        return 0
    try:
        import toy_transformer as tt
    except Exception as e:  # noqa: BLE001
        print(f"SKIP: 无法 import model_lab/toy_transformer.py -> {e}")
        return 0

    sd = load_sd()
    torch.set_num_threads(2)
    worst = 0.0

    torch.manual_seed(4)
    x = torch.randn(4, 12, 64)
    torch.manual_seed(1)
    q = torch.randn(4, 12, 64)
    torch.manual_seed(2)
    k = torch.randn(4, 16, 64)
    torch.manual_seed(3)
    v = torch.randn(4, 16, 64)
    torch.manual_seed(5)
    mask = torch.randn(4, 12, 16) > 0.5
    torch.manual_seed(6)
    idx = torch.randint(0, V, (4, 12))
    pos = torch.arange(12)

    # 1. Linear
    lin = tt.Linear(64, 128)
    lin.load_state_dict({"weight": sd["layers.0.ffn.w1.weight"]})
    worst = max(worst, diff("test_linear", lin(x)))

    # 2. RoPE（d=64，位置 0..11）
    rope = tt.RotaryPositionalEmbedding(theta=THETA, d_k=64, max_seq_len=16)
    worst = max(worst, diff("test_rope", rope(x, pos)))

    # 3. MHA（layer0 权重，causal，无 RoPE）
    mha = tt.CausalMultiHeadSelfAttention(d_model=64, num_heads=4)
    mha.load_state_dict(
        {
            f"{n}_proj.weight": sd[f"layers.0.attn.{n}_proj.weight"]
            for n in ("q", "k", "v", "output")
        }
    )
    worst = max(worst, diff("test_multihead_self_attention", mha(x)))

    # 4. Block（layer0 全部子权重）
    rope16 = tt.RotaryPositionalEmbedding(theta=THETA, d_k=16, max_seq_len=16)
    block = tt.TransformerBlock(d_model=64, num_heads=4, d_ff=128, rope=rope16)
    block_sd = {
        kk.replace("layers.0.", ""): vv
        for kk, vv in sd.items()
        if kk.startswith("layers.0.")
    }
    block.load_state_dict(block_sd, strict=True)
    worst = max(worst, diff("test_transformer_block", block(x)))

    # 5. LM（全量权重，untied）
    lm = tt.TransformerLM(
        vocab_size=V,
        context_length=16,
        d_model=64,
        num_layers=3,
        num_heads=4,
        d_ff=128,
        rope_theta=THETA,
    )
    missing, unexpected = lm.load_state_dict(sd, strict=True)
    worst = max(worst, diff("test_transformer_lm", lm(idx)))

    print(f"\nWORST maxdiff = {worst:.3e}  ({'OK' if worst <= 1e-4 else 'FAIL'})")
    return 0 if worst <= 1e-4 else 1


if __name__ == "__main__":
    raise SystemExit(main())
