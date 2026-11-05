#!/usr/bin/env python
"""model_lab/accounting.py — Transformer 参数量 / fp32 显存 / 前向 FLOPs 解析式核算.

对应 SPEC §7.1 与 handout §3.5 ``transformer_accounting``.

设计原则
--------
1. **闭式公式，不实例化大模型**。所有数字由配置直接推出，GPT-2 XL 也不会真的分配 6.5GB。
2. 所有结果先落 ``model_lab/results/accounting.json``，再打印人类可读表格。
3. 对账（checks）：
   - 官方 ``tests/fixtures/ts_tests/model.pt`` 的真实 ``sum(numel)``（只读，若可得）；
   - 若 ``model_lab/toy_transformer.py`` 已存在，用小配置实例 ``sum(p.numel())`` 对账，
     并顺带验证它能否 ``load_state_dict`` 官方权重（键名契约的一致性检查）。

公式（SPEC §7.1，``2mnp`` 规则）
--------------------------------
参数量::

    token_embeddings = V*d
    每层 attn        = 4*d*d
    每层 ffn         = 3*d*d_ff
    每层 ln1/ln2     = 2*d
    ln_final         = d
    lm_head          = V*d              (untied)

    total = V*d + L*(4d² + 3*d*d_ff + 2d) + d + V*d

前向 FLOPs（T = 序列长度）::

    每层投影 (q,k,v,o)   = 8*T*d²
    attention 分数+加权  = 4*T²*d
    FFN (w1,w3,w2)       = 6*T*d*d_ff
    每层合计             = 8Td² + 4T²d + 6T d d_ff
    lm_head              = 2*T*d*V
    embedding 查表       = 0 FLOPs

用法::

    .venv/bin/python model_lab/accounting.py
    .venv/bin/python model_lab/accounting.py --out model_lab/results/accounting.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import resource
import sys
import time
from pathlib import Path

import torch

# SPEC §1：所有可能超过 1 秒的脚本必须 2 线程
torch.set_num_threads(2)

MIB = 1024 * 1024
BYTES_PER_PARAM = 4  # fp32

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "results" / "accounting.json"
DEFAULT_A1_REPO = Path("/Users/yunni/Projects/assignment1-basics")

# handout §3.5 给出的唯一 d_ff 参考值，用于校验我们的「最近 64 倍数」规则
HANDOUT_D_FF = {"gpt2_xl": 4288}

# SPEC §7.1 的四 + 一个配置
CONFIGS = {
    "gpt2_small": {
        "vocab_size": 50257,
        "context_length": 1024,
        "d_model": 768,
        "num_layers": 12,
        "num_heads": 12,
    },
    "gpt2_medium": {
        "vocab_size": 50257,
        "context_length": 1024,
        "d_model": 1024,
        "num_layers": 24,
        "num_heads": 16,
    },
    "gpt2_large": {
        "vocab_size": 50257,
        "context_length": 1024,
        "d_model": 1280,
        "num_layers": 36,
        "num_heads": 20,
    },
    "gpt2_xl": {
        "vocab_size": 50257,
        "context_length": 1024,
        "d_model": 1600,
        "num_layers": 48,
        "num_heads": 25,
    },
    "gpt2_xl_ctx16384": {
        "vocab_size": 50257,
        "context_length": 16384,
        "d_model": 1600,
        "num_layers": 48,
        "num_heads": 25,
    },
}

D_FF_RULE = "nearest multiple of 64 to 8/3*d_model (minimum 64)"


def d_ff_default(d_model: int) -> int:
    """SPEC §3 / §7.1：d_ff = 8/3 * d_model 取最近的 64 的倍数（最小 64）。

    512 -> 1344, 1600 -> 4288（handout §3.5 给定）。
    使用 floor(x/64 + 0.5) 而非 Python round()，避免银行家舍入。
    """
    value = 8.0 * d_model / 3.0
    n = math.floor(value / 64.0 + 0.5)
    return max(64, 64 * n)


def peak_rss_mib() -> float:
    """进程峰值 RSS（MiB）。macOS 的 ru_maxrss 单位是字节，Linux 是 KiB。"""
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return ru / MIB
    return ru / 1024.0


def param_breakdown(cfg: dict) -> dict:
    """闭式参数量分解。cfg 必须含 d_ff（显式给出，不依赖默认规则）。"""
    V = cfg["vocab_size"]
    L = cfg["num_layers"]
    d = cfg["d_model"]
    d_ff = cfg["d_ff"]

    token_embeddings = V * d
    attn_per_layer = 4 * d * d
    ffn_per_layer = 3 * d * d_ff
    ln_per_layer = 2 * d
    per_layer = attn_per_layer + ffn_per_layer + ln_per_layer

    attn_total = L * attn_per_layer
    ffn_total = L * ffn_per_layer
    ln_total = L * ln_per_layer
    ln_final = d
    lm_head = V * d

    total = token_embeddings + attn_total + ffn_total + ln_total + ln_final + lm_head
    return {
        "token_embeddings": token_embeddings,
        "attention_per_layer": attn_per_layer,
        "ffn_per_layer": ffn_per_layer,
        "ln_per_layer": ln_per_layer,
        "per_layer_total": per_layer,
        "attention_total": attn_total,
        "ffn_total": ffn_total,
        "ln_total": ln_total,
        "ln_final": ln_final,
        "lm_head": lm_head,
        "non_embedding_total": attn_total + ffn_total + ln_total + ln_final,
        "total": total,
        "total_billions": total / 1e9,
    }


def flop_breakdown(cfg: dict) -> dict:
    """闭式前向 FLOPs 分解（2mnp 规则，只计矩阵乘）。"""
    V = cfg["vocab_size"]
    T = cfg["context_length"]
    L = cfg["num_layers"]
    d = cfg["d_model"]
    d_ff = cfg["d_ff"]

    projections_per_layer = 8 * T * d * d
    attention_per_layer = 4 * T * T * d
    ffn_per_layer = 6 * T * d * d_ff
    layer_total = projections_per_layer + attention_per_layer + ffn_per_layer

    projections_total = L * projections_per_layer
    attention_total = L * attention_per_layer
    ffn_total = L * ffn_per_layer
    layers_total = L * layer_total
    lm_head = 2 * T * d * V

    total = layers_total + lm_head
    breakdown = {
        "projections": projections_total,
        "attention": attention_total,
        "ffn": ffn_total,
        "lm_head": lm_head,
        "total": total,
    }
    proportions = {
        key: (value / total) if total else 0.0
        for key, value in breakdown.items()
        if key != "total"
    }
    return {
        "total": total,
        "total_tflops": total / 1e12,
        "breakdown": breakdown,
        "proportions": proportions,
        "per_layer_total": layer_total,
        "formulas": {
            "projections_per_layer": "8*T*d^2",
            "attention_per_layer": "4*T^2*d",
            "ffn_per_layer": "6*T*d*d_ff",
            "lm_head": "2*T*d*V",
        },
    }


def summarize_config(cfg: dict) -> dict:
    params = param_breakdown(cfg)
    flops = flop_breakdown(cfg)
    return {
        "config": dict(cfg),
        "params": params,
        "memory_fp32_mib": params["total"] * BYTES_PER_PARAM / MIB,
        "memory_fp32_gib": params["total"] * BYTES_PER_PARAM / (1024 * MIB),
        "flops": flops,
    }


# ---------------------------------------------------------------- checks


def check_official_fixture(a1_repo: Path) -> dict:
    """用官方 model.pt 的真实参数量对账（只读；文件不存在则 skipped）。"""
    pt = a1_repo / "tests" / "fixtures" / "ts_tests" / "model.pt"
    cfg_path = a1_repo / "tests" / "fixtures" / "ts_tests" / "model_config.json"
    result = {"name": "official_model_pt_param_count", "path": str(pt)}
    if not pt.exists():
        result.update(skipped=True, reason="model.pt not found")
        return result
    if not cfg_path.exists():
        result.update(skipped=True, reason="model_config.json not found")
        return result

    try:
        cfg_raw = json.loads(cfg_path.read_text())
        try:
            state = torch.load(pt, map_location="cpu", weights_only=True)
        except Exception:
            # 官方 fixture 是 OrderedDict，weights_only=True 在部分 torch 版本下也可读，
            # 失败时退回 False（本地只读文件，风险可控）。
            state = torch.load(pt, map_location="cpu", weights_only=False)
        actual = int(sum(v.numel() for v in state.values() if hasattr(v, "numel")))

        # 配置里 d_ff 是显式值（128），不是 8/3 规则推导值
        cfg = {
            "vocab_size": cfg_raw["vocab_size"],
            "context_length": cfg_raw["context_length"],
            "d_model": cfg_raw["d_model"],
            "num_layers": cfg_raw["num_layers"],
            "num_heads": cfg_raw["num_heads"],
            "d_ff": cfg_raw["d_ff"],
        }
        analytic = param_breakdown(cfg)["total"]
        result.update(
            skipped=False,
            config=cfg,
            actual_params=actual,
            analytic_params=analytic,
            diff=actual - analytic,
            match=actual == analytic,
            num_state_dict_entries=len(state),
            state_dict_has_orig_mod_prefix=any(k.startswith("_orig_mod.") for k in state),
        )
    except Exception as exc:  # pragma: no cover - 防御性
        result.update(skipped=True, reason=f"{type(exc).__name__}: {exc}")
    return result


def check_toy_transformer(cfg: dict | None = None) -> dict:
    """若 toy_transformer.py 存在，用小 config 实例化并 sum(p.numel()) 对账。"""
    result = {"name": "toy_transformer_sum_numel", "skipped": True}
    toy_path = HERE / "toy_transformer.py"
    if not toy_path.exists():
        result["reason"] = "model_lab/toy_transformer.py not found (impl 未完成)"
        return result

    sys.path.insert(0, str(HERE))
    try:
        import toy_transformer  # noqa: F401  (延迟导入，避免 impl 未完成时报错)
    except Exception as exc:
        result["reason"] = f"import failed: {type(exc).__name__}: {exc}"
        return result

    if not hasattr(toy_transformer, "TransformerLM"):
        result["reason"] = "toy_transformer.TransformerLM missing"
        return result

    # 用小配置即可；显式 d_ff=128，避免依赖 impl 的默认值实现
    toy_cfg = cfg or {
        "vocab_size": 256,
        "context_length": 64,
        "d_model": 64,
        "num_layers": 2,
        "num_heads": 4,
        "d_ff": 128,
    }
    analytic = param_breakdown(toy_cfg)["total"]
    try:
        torch.manual_seed(0)
        model = toy_transformer.TransformerLM(
            vocab_size=toy_cfg["vocab_size"],
            context_length=toy_cfg["context_length"],
            d_model=toy_cfg["d_model"],
            num_layers=toy_cfg["num_layers"],
            num_heads=toy_cfg["num_heads"],
            d_ff=toy_cfg["d_ff"],
        )
        actual = int(sum(p.numel() for p in model.parameters()))
        num_parameters = None
        if hasattr(model, "num_parameters"):
            try:
                num_parameters = int(model.num_parameters())
            except Exception as exc:
                num_parameters = f"error: {type(exc).__name__}: {exc}"
        result.update(
            skipped=False,
            config=toy_cfg,
            actual_params=actual,
            analytic_params=analytic,
            diff=actual - analytic,
            match=actual == analytic,
            num_parameters_returned=num_parameters,
            num_parameters_match=(num_parameters == actual),
        )
    except Exception as exc:
        result.update(skipped=True, reason=f"instantiate failed: {type(exc).__name__}: {exc}")
    return result


def check_default_d_ff() -> dict:
    """校验 d_ff 规则：512->1344, 1600->4288, 且全部是 64 倍数。"""
    cases = {512: 1344, 768: 2048, 1024: 2752, 1280: 3392, 1600: 4288}
    got = {str(d): d_ff_default(d) for d in cases}
    per_config = {
        name: {
            "d_model": cfg["d_model"],
            "d_ff": d_ff_default(cfg["d_model"]),
            "is_multiple_of_64": d_ff_default(cfg["d_model"]) % 64 == 0,
        }
        for name, cfg in CONFIGS.items()
    }
    return {
        "name": "d_ff_default_rule",
        "rule": D_FF_RULE,
        "expected": {str(k): v for k, v in cases.items()},
        "actual": got,
        "match": got == {str(k): v for k, v in cases.items()},
        "all_multiple_of_64": all(v["is_multiple_of_64"] for v in per_config.values()),
        "per_config": per_config,
        "handout_reference": {
            name: {"expected": expected, "actual": d_ff_default(CONFIGS[name]["d_model"])}
            for name, expected in HANDOUT_D_FF.items()
        },
    }


def check_xl_context_ratio(accounting: dict) -> dict:
    """XL ctx 16384 vs ctx 1024 的 FLOPs 变化与各组件占比变化（handout §3.5(e)）。"""
    base = accounting["configs"]["gpt2_xl"]["flops"]
    long = accounting["configs"]["gpt2_xl_ctx16384"]["flops"]
    return {
        "name": "xl_ctx16384_vs_1024",
        "total_flops_ratio": long["total"] / base["total"],
        "total_flops_ctx16384": long["total"],
        "total_flops_ctx1024": base["total"],
        "proportions_ctx1024": base["proportions"],
        "proportions_ctx16384": long["proportions"],
        "attention_share_delta": long["proportions"]["attention"] - base["proportions"]["attention"],
    }


def check_scale_trend(local: dict) -> dict:
    """小->大模型各组件 FLOPs 占比趋势（handout §3.5(d) 的量化版）。"""
    order = ["gpt2_small", "gpt2_medium", "gpt2_large", "gpt2_xl"]
    trend = {
        name: {
            "params_total": local["configs"][name]["params"]["total"],
            "d_ff": local["configs"][name]["config"]["d_ff"],
            "proportions": local["configs"][name]["flops"]["proportions"],
        }
        for name in order
    }
    return {
        "name": "scale_trend",
        "order": order,
        "per_model": trend,
    }


# ---------------------------------------------------------------- main


def print_table(accounting: dict) -> None:
    print()
    print("=" * 118)
    print("Transformer 资源核算（解析式，2mnp 规则；无模型实例化）")
    print("=" * 118)
    header = (
        f"{'config':<20}{'params':>14}{'non-emb':>13}{'fp32 MiB':>11}"
        f"{'FLOPs':>14}{'proj%':>7}{'attn%':>7}{'ffn%':>7}{'head%':>7}"
    )
    print(header)
    print("-" * 118)
    for name, entry in accounting["configs"].items():
        p = entry["params"]
        f = entry["flops"]
        prop = f["proportions"]
        print(
            f"{name:<20}{p['total']/1e6:>12.1f}M{p['non_embedding_total']/1e6:>11.1f}M"
            f"{entry['memory_fp32_mib']:>11.1f}{f['total']/1e12:>12.2f}T"
            f"{prop['projections']*100:>6.1f}%{prop['attention']*100:>6.1f}%"
            f"{prop['ffn']*100:>6.1f}%{prop['lm_head']*100:>6.1f}%"
        )
    print("-" * 118)
    print("注：FLOPs 为单次前向、T=context_length、batch=1；params 为 untied 配置。")
    print()


def main(argv: list[str] | None = None) -> int:
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="JSON 输出路径")
    parser.add_argument("--a1-repo", type=Path, default=Path(os.environ.get("A1_REPO", DEFAULT_A1_REPO)))
    parser.add_argument("--skip-toy-check", action="store_true", help="跳过 toy_transformer 对账")
    parser.add_argument("--quiet", action="store_true", help="不打印表格")
    parser.add_argument("--strict", action="store_true", help="任一 check 失败则 exit 1")
    args = parser.parse_args(argv)

    accounts: dict[str, dict] = {}
    for name, cfg in CONFIGS.items():
        full_cfg = dict(cfg)
        full_cfg["d_ff"] = d_ff_default(cfg["d_model"])
        accounts[name] = summarize_config(full_cfg)

    local = {"configs": accounts}

    checks: dict[str, dict] = {}
    checks["official_fixture"] = check_official_fixture(args.a1_repo)
    checks["toy_transformer"] = (
        {"name": "toy_transformer_sum_numel", "skipped": True, "reason": "--skip-toy-check"}
        if args.skip_toy_check
        else check_toy_transformer()
    )
    checks["d_ff_rule"] = check_default_d_ff()
    checks["xl_ctx16384"] = check_xl_context_ratio(local)
    checks["scale_trend"] = check_scale_trend(local)

    elapsed = time.perf_counter() - started
    threads = torch.get_num_threads()
    payload = {
        "meta": {
            "spec": "model_lab/SPEC.md §7.1",
            "source": "handout §3.5 transformer_accounting (a)-(e)",
            "vocab_size": 50257,
            "d_ff_rule": D_FF_RULE,
            "bytes_per_param": BYTES_PER_PARAM,
            "flops_rule": "2*m*n*p for A(m,n) @ B(n,p); embedding lookup not counted",
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "configs": accounts,
        "checks": checks,
        "runtime": {
            "elapsed_s": round(elapsed, 4),
            "threads": threads,
            "peak_rss_mib": round(peak_rss_mib(), 1),
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

    if not args.quiet:
        print_table(payload)
        ch = checks
        off = ch["official_fixture"]
        if off.get("skipped"):
            print(f"[check] 官方 model.pt 对账: SKIP ({off.get('reason')})")
        else:
            verdict = "PASS" if off["match"] else "FAIL"
            print(
                f"[check] 官方 model.pt 对账: {verdict} "
                f"actual={off['actual_params']} analytic={off['analytic_params']} diff={off['diff']}"
            )
        toy = ch["toy_transformer"]
        if toy.get("skipped"):
            print(f"[check] toy_transformer 对账: SKIP ({toy.get('reason')})")
        else:
            verdict = "PASS" if toy["match"] else "FAIL"
            print(
                f"[check] toy_transformer 对账: {verdict} "
                f"actual={toy['actual_params']} analytic={toy['analytic_params']} "
                f"num_parameters()={toy['num_parameters_returned']}"
            )
        dff = ch["d_ff_rule"]
        print(
            f"[check] d_ff 规则: {'PASS' if dff['match'] else 'FAIL'} "
            f"{dff['actual']} (64 倍数: {dff['all_multiple_of_64']})"
        )
        xl = ch["xl_ctx16384"]
        print(
            f"[check] XL ctx 16384 / 1024 FLOPs 比: {xl['total_flops_ratio']:.3f}x，"
            f"attention 占比 {xl['proportions_ctx1024']['attention']*100:.2f}% -> "
            f"{xl['proportions_ctx16384']['attention']*100:.2f}%"
        )
        print(f"[runtime] elapsed_s={elapsed:.3f} threads={threads} peak_rss_mib={peak_rss_mib():.1f}")
        print(f"[out] {args.out}")

    failed = []
    off = checks["official_fixture"]
    if not off.get("skipped") and not off.get("match"):
        failed.append("official_fixture")
    toy = checks["toy_transformer"]
    if not toy.get("skipped") and not toy.get("match"):
        failed.append("toy_transformer")
    if not checks["d_ff_rule"]["match"] or not checks["d_ff_rule"]["all_multiple_of_64"]:
        failed.append("d_ff_rule")
    if failed:
        print(f"[FAIL] checks failed: {failed}", file=sys.stderr)
        return 1 if args.strict else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
