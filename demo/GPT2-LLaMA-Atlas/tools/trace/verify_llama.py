#!/usr/bin/env python3
"""Independent verification for the SmolLM2 LLaMA trace (atlas.trace.case/2).

Recomputes operations from exported payloads and compares against the HF reference.
Writes data/llama/verification.json only when all required checks pass. Never edits
the case/manifest and never falls back to random values.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import platform
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
MATERIAL_DEFAULT = Path("/Users/yunni/Projects/temp/cs336-prep-code")
CASE_ID = "smollm2-main-002"
THREADS = 2
TOL = {
    "hf_logits_max_abs": 1e-3,
    "module_recompute_max_abs": 1e-5,
    "attention_max_abs": 1e-5,
    "residual_max_abs": 1e-6,
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def decode_tensor(t: dict) -> np.ndarray:
    payload = t["payload"]
    raw = base64.b64decode(payload["data"], validate=True)
    if len(raw) != payload["nbytes"]:
        raise AssertionError(f"payload byte length mismatch: {t['id']}")
    if sha256_bytes(raw) != payload["sha256"]:
        raise AssertionError(f"payload sha mismatch: {t['id']}")
    count = int(np.prod(t["shape"])) if t["shape"] else 1
    if t["dtype"] == "float32":
        values = np.frombuffer(raw, dtype="<f4").copy()
        if values.size != count:
            raise AssertionError(f"element count mismatch: {t['id']}")
        if not np.isfinite(values).all():
            raise AssertionError(f"non-finite payload: {t['id']}")
        return values.reshape(t["shape"])
    if t["dtype"] == "uint8":
        values = np.frombuffer(raw, dtype=np.uint8).copy()
        if values.size != count:
            raise AssertionError(f"element count mismatch: {t['id']}")
        return values.reshape(t["shape"])
    raise AssertionError(f"unknown dtype: {t['id']}")


def tensor_map(case: dict) -> dict[str, dict]:
    return {t["id"]: t for t in case["tensors"]}


def values(tensors: dict, case: dict, tensor_id: str) -> np.ndarray:
    return decode_tensor(tensors[tensor_id])


def max_abs(a, b) -> float:
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)))) if np.size(a) else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--material-root", default=str(MATERIAL_DEFAULT))
    parser.add_argument("--data-dir", default="data/llama")
    parser.add_argument("--case-id", default=CASE_ID)
    args = parser.parse_args()
    material = Path(args.material_root)
    data_dir = Path(args.data_dir)
    case_path = data_dir / f"{args.case_id}.json"
    manifest_path = data_dir / "manifest.json"
    verification_path = data_dir / "verification.json"
    checks = []

    def check(check_id: str, passed: bool, observed, expected=None, required=True, tolerance=None, details=None):
        checks.append({
            "id": check_id,
            "status": "passed" if passed else "failed",
            "required": required,
            "observed": observed,
            "expected": expected,
            "tolerance": tolerance,
            "details": details or {},
        })

    case = json.loads(case_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    tensors = tensor_map(case)
    source_map = json.loads((ROOT / "sources" / "llama" / "source-map.json").read_text())
    instance = next(i for i in manifest["instances"] if i["case_id"] == args.case_id)

    check("structure.case_schema", case["schema"] == "atlas.trace.case/2", case["schema"], "atlas.trace.case/2")
    check("structure.manifest_schema", manifest["schema"] == "atlas.trace/2", manifest["schema"], "atlas.trace/2")
    check("structure.model_id", case["model_id"] == "smollm2-135m" and manifest["model"]["model_id"] == "smollm2-135m", [case["model_id"], manifest["model"]["model_id"]], "smollm2-135m")
    case_sha = sha256_file(case_path)
    check("integrity.case_sha256", case_sha == instance["case_sha256"], case_sha, instance["case_sha256"])
    check("integrity.case_bytes", case_path.stat().st_size == instance["case_bytes"], case_path.stat().st_size, instance["case_bytes"])
    check("integrity.source_map_version", source_map["map_version"] == case["source_map_version"], source_map["map_version"], case["source_map_version"])
    check("integrity.source_file_hashes", all(sha256_file(ROOT / "sources" / "llama" / name) == meta["sha256"] for name, meta in source_map["sources"].items()), "checked", "two files")

    for implementation in manifest["provenance"]["implementation"]["files"]:
        name = Path(implementation["snapshot"]).name
        check(f"integrity.upstream.{name}", sha256_file(material / "llama_steps" / name) == implementation["upstream_sha256"], "read-only upstream hash", implementation["upstream_sha256"])

    # Source-map spans are recomputed from the snapshots, independent of the map.
    source_texts = {name: (ROOT / "sources" / "llama" / name).read_text() for name in source_map["sources"]}
    bad_spans = []
    for group in ("symbols", "operations"):
        for item_id, entry in source_map[group].items():
            lines = source_texts[entry["file"]].splitlines(keepends=True)
            snippet = "".join(lines[entry["lines"][0] - 1:entry["lines"][1]])
            if sha256_bytes(snippet.encode("utf-8")) != entry["span_sha256"]:
                bad_spans.append(item_id)
    check("integrity.source_spans", not bad_spans, bad_spans, [])

    # Event and tensor references.
    event_ids = [e["id"] for e in case["events"]]
    check("structure.event_seq", [e["seq"] for e in case["events"]] == list(range(len(event_ids))), len(event_ids), "0..n-1")
    check("structure.event_count", len(case["events"]) == 238, len(case["events"]), 238)
    check("structure.sample_count", len([e for e in case["events"] if e["kind"] == "sample"]) == 7, len([e for e in case["events"] if e["kind"] == "sample"]), 7)
    check("structure.block_count", len([e for e in case["events"] if e["kind"] == "block"]) == 210, len([e for e in case["events"] if e["kind"] == "block"]), 210)
    missing_refs = []
    for event in case["events"]:
        for tid in event.get("tensor_ids", []):
            if tid not in tensors:
                missing_refs.append((event["id"], tid))
        for op in [event.get("primary_op"), (event.get("source") or {}).get("symbol")]:
            pass
    check("structure.tensor_refs", not missing_refs, missing_refs[:5], [])

    # Decode every payload and verify shape/basics.
    decoded = {}
    for tid, t in tensors.items():
        decoded[tid] = decode_tensor(t)
        if t["dtype"] == "float32":
            check(f"payload.finite.{tid}", bool(np.isfinite(decoded[tid]).all()), True, True, required=False)
    check("payload.all_decoded", len(decoded) == len(tensors), len(decoded), len(tensors))

    # Attention: row sums, mask, shape, and per-layer reference comparison later.
    sample_events = [e for e in case["events"] if e["kind"] == "sample"]
    generated_ids = [e["token_id"] for e in sample_events]
    generation = case["execution"]["generation"]
    check("generation.ids", generated_ids == generation["generated_ids"], generated_ids, generation["generated_ids"])
    check("generation.forward_count", generation["forward_count"] == 7, generation["forward_count"], 7)
    check("generation.full_sequence", generation["full_sequence_ids"] == case["input"]["ids"] + generated_ids, generation["full_sequence_ids"], case["input"]["ids"] + generated_ids)
    check("generation.no_unused_forward", all(e["logits_event"]["used_by_sampling"] for e in case["events"] if e["kind"] == "logits"), True, True)
    check("generation.stopped", case["events"][-1]["id"] == "sample6" and generation.get("stop_reason") == "max_new_tokens", case["events"][-1]["id"], "sample6")
    check("structure.wrapper_equivalence", case["execution"]["wrapper_equivalence_max_abs"] < 1e-6, case["execution"]["wrapper_equivalence_max_abs"], "< 1e-6")

    # HF reference: logits/generation/attention for all forwards.
    from transformers import AutoTokenizer, LlamaForCausalLM
    weights_dir = material / "llama_steps" / "weights" / "smollm2-135m"
    hf = LlamaForCausalLM.from_pretrained(
        str(weights_dir), local_files_only=True, torch_dtype=torch.float32,
        attn_implementation="eager",
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(str(weights_dir), local_files_only=True)
    prompt = case["input"]["text"]
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids
    check("hf.prompt_ids", prompt_ids[0].tolist() == case["input"]["ids"], prompt_ids[0].tolist(), case["input"]["ids"])

    vocabulary = case.get("vocabulary", {})
    expected_decoded = [tokenizer.decode([i], clean_up_tokenization_spaces=False) for i in range(hf.config.vocab_size)]
    expected_raw = [tokenizer.convert_ids_to_tokens(i) for i in range(hf.config.vocab_size)]
    check("hf.full_vocabulary", vocabulary.get("decoded") == expected_decoded and vocabulary.get("raw") == expected_raw, len(vocabulary.get("decoded", [])), hf.config.vocab_size)

    # Rerun the actual project generator without the export wrapper. Count model calls.
    sys.path.insert(0, str(ROOT / "sources" / "llama"))
    import step06_model as core
    import step09_real_llama as real
    from safetensors.torch import load_file
    config = real.load_config(json.loads((weights_dir / "config.json").read_text()))
    model = core.MiniLlama(config).eval()
    mapped = real.hf_to_ours(load_file(str(weights_dir / "model.safetensors")), config)
    if config.tie_word_embeddings:
        mapped["lm_head.weight"] = mapped["token_embedding.weight"]
    model.load_state_dict(mapped, strict=True)
    calls = []
    handle = model.register_forward_hook(lambda _model, args, output: calls.append(args[0].shape[1]))
    try:
        actual, _ = real.generate(model, prompt_ids, 7, greedy=True)
    finally:
        handle.remove()
    check("implementation.greedy_ids", actual[0].tolist() == generation["full_sequence_ids"], actual[0].tolist(), generation["full_sequence_ids"])
    check("implementation.no_unused_forward", calls == [len(case["input"]["ids"])] + [1] * 6, calls, [len(case["input"]["ids"])] + [1] * 6)
    del model

    hf_last_rows = []
    hf_attn = {}
    prefixes = []
    for forward in range(7):
        prefix_ids = prompt_ids[0].tolist()[:] + generated_ids[:forward]
        prefixes.append(prefix_ids)
        ids = torch.tensor([prefix_ids], dtype=torch.long)
        with torch.no_grad():
            out = hf(ids, output_attentions=True)
        hf_last_rows.append(out.logits[0, -1, :].float().clone())
        hf_attn[forward] = []
        for a in out.attentions:
            item = a.detach().clone()
            if item.dim() == 3:
                item = item.unsqueeze(0)
            hf_attn[forward].append(item)

    for forward in range(7):
        tensor_id = f"fwd{forward}.logits.last_row"
        exported = torch.tensor(decoded[tensor_id][0], dtype=torch.float32)
        diff = float((exported - hf_last_rows[forward]).abs().max().item())
        check(f"hf.logits_fwd{forward}", diff < TOL["hf_logits_max_abs"], diff, f"< {TOL['hf_logits_max_abs']}", tolerance=TOL["hf_logits_max_abs"])
        top_exported = int(torch.argmax(exported).item())
        top_hf = int(torch.argmax(hf_last_rows[forward]).item())
        check(f"hf.top1_fwd{forward}", top_exported == top_hf, [top_exported, top_hf], "equal")

    with torch.no_grad():
        hf_gen = hf.generate(prompt_ids, max_new_tokens=7, do_sample=False, eos_token_id=None, pad_token_id=tokenizer.eos_token_id)
    hf_new = hf_gen[0, prompt_ids.shape[1]:].tolist()
    check("hf.greedy_ids", hf_new == generated_ids, hf_new, generated_ids)

    # Raw selected-layer recomputation.
    for forward in range(7):
        prefix = f"fwd{forward}.block0"
        x_in = decoded[f"{prefix}.x_in"].astype(np.float32)
        gamma1 = decoded[f"{prefix}.norm1_gamma"]
        norm1 = decoded[f"{prefix}.norm1_out"]
        mean1 = decoded[f"{prefix}.norm1_mean_sq"]
        inv1 = decoded[f"{prefix}.norm1_inv_rms"]
        calc_mean = (x_in.astype(np.float64) ** 2).mean(-1, keepdims=True)
        calc_inv = 1.0 / np.sqrt(calc_mean + 1e-5)
        calc_norm1 = (x_in.astype(np.float64) * calc_inv * gamma1.astype(np.float64)).astype(np.float32)
        check(f"recompute.mean1_f{forward}", max_abs(mean1, calc_mean) < TOL["module_recompute_max_abs"], max_abs(mean1, calc_mean), f"< {TOL['module_recompute_max_abs']}")
        check(f"recompute.inv1_f{forward}", max_abs(inv1, calc_inv) < TOL["module_recompute_max_abs"], max_abs(inv1, calc_inv), f"< {TOL['module_recompute_max_abs']}")
        check(f"recompute.rmsnorm1_f{forward}", max_abs(norm1, calc_norm1) < TOL["module_recompute_max_abs"], max_abs(norm1, calc_norm1), f"< {TOL['module_recompute_max_abs']}")

        q = decoded[f"{prefix}.q_proj"].astype(np.float32)
        k = decoded[f"{prefix}.k_proj"].astype(np.float32)
        cos = decoded[f"{prefix}.cos"]
        sin = decoded[f"{prefix}.sin"]
        D = 64
        Hq, Hkv = 9, 3
        q4 = q.reshape(1, q.shape[1], Hq, D).transpose(0, 2, 1, 3)
        k4 = k.reshape(1, k.shape[1], Hkv, D).transpose(0, 2, 1, 3)
        def rope(x):
            d = x.shape[-1] // 2
            rot = np.concatenate([-x[..., d:], x[..., :d]], axis=-1)
            return x * cos[None, None, :, :] + rot * sin[None, None, :, :]
        calc_q = rope(q4).astype(np.float32)
        calc_k = rope(k4).astype(np.float32)
        check(f"recompute.rope_q_f{forward}", max_abs(decoded[f"{prefix}.q_rope"], calc_q) < 2e-5, max_abs(decoded[f"{prefix}.q_rope"], calc_q), "< 2e-5")
        check(f"recompute.rope_k_f{forward}", max_abs(decoded[f"{prefix}.k_rope"], calc_k) < 2e-5, max_abs(decoded[f"{prefix}.k_rope"], calc_k), "< 2e-5")

        k_cache = decoded[f"{prefix}.k_cache"]
        v_cache = decoded[f"{prefix}.v_cache"]
        check(f"cache.k_rotated_f{forward}", max_abs(k_cache[:, :, -calc_k.shape[2]:, :], calc_k) < 2e-5, max_abs(k_cache[:, :, -calc_k.shape[2]:, :], calc_k), "< 2e-5")
        check(f"cache.v_unrotated_f{forward}", max_abs(v_cache[:, :, -decoded[f"{prefix}.v_heads"].shape[2]:, :], decoded[f"{prefix}.v_heads"]) < 1e-6, max_abs(v_cache[:, :, -decoded[f"{prefix}.v_heads"].shape[2]:, :], decoded[f"{prefix}.v_heads"]), "< 1e-6")
        k_exp = decoded[f"{prefix}.k_expanded"]
        calc_k_exp = np.repeat(k_cache, 3, axis=1)
        check(f"gqa.k_repeat_f{forward}", max_abs(k_exp, calc_k_exp) < 1e-6, max_abs(k_exp, calc_k_exp), "< 1e-6")

        q_rope = decoded[f"{prefix}.q_rope"]
        scores = decoded[f"{prefix}.scores_scaled"]
        calc_scores = np.matmul(q_rope.astype(np.float64), k_exp.astype(np.float64).transpose(0, 1, 3, 2)) / math.sqrt(64)
        mask = decoded[f"{prefix}.causal_mask"].astype(bool)
        event = next(e for e in case["events"] if e["id"] == f"fwd{forward}.block0")
        qpos = np.array(event["positions"]["query"])[None, :, None]
        kpos = np.array(event["positions"]["key"])[None, None, :]
        expected_mask = (kpos <= qpos)[0].astype(bool)
        check(f"recompute.mask_f{forward}", mask.shape == expected_mask.shape and bool((mask == expected_mask).all()), mask.shape, expected_mask.shape)
        check(f"recompute.scores_f{forward}", max_abs(scores, calc_scores) < 2e-5, max_abs(scores, calc_scores), "< 2e-5")
        masked = np.where(mask[None, None, :, :], calc_scores, -np.inf)
        weights = torch.softmax(torch.tensor(masked, dtype=torch.float64), dim=-1).numpy()
        check(f"recompute.weights_f{forward}", max_abs(decoded[f"{prefix}.attn_weights"], weights) < 2e-5, max_abs(decoded[f"{prefix}.attn_weights"], weights), "< 2e-5")

        gate = decoded[f"{prefix}.gate_proj"]
        up = decoded[f"{prefix}.up_proj"]
        silu = gate / (1.0 + np.exp(-gate))
        check(f"recompute.silu_f{forward}", max_abs(decoded[f"{prefix}.silu_gate"], silu) < 1e-6, max_abs(decoded[f"{prefix}.silu_gate"], silu), "< 1e-6")
        check(f"recompute.gated_f{forward}", max_abs(decoded[f"{prefix}.gated"], silu * up) < 1e-6, max_abs(decoded[f"{prefix}.gated"], silu * up), "< 1e-6")
        check(f"recompute.residual1_f{forward}", max_abs(decoded[f"{prefix}.x_after_attn"], x_in + decoded[f"{prefix}.attn_out"]) < TOL["residual_max_abs"], max_abs(decoded[f"{prefix}.x_after_attn"], x_in + decoded[f"{prefix}.attn_out"]), "< 1e-6")
        check(f"recompute.residual2_f{forward}", max_abs(decoded[f"{prefix}.x_out"], decoded[f"{prefix}.x_after_attn"] + decoded[f"{prefix}.down_out"]) < TOL["residual_max_abs"], max_abs(decoded[f"{prefix}.x_out"], decoded[f"{prefix}.x_after_attn"] + decoded[f"{prefix}.down_out"]), "< 1e-6")

        # All 30 layers' attention weights versus HF last-query row.
        for layer in range(30):
            exported_w = decoded[f"fwd{forward}.block{layer}.attn_weights"]
            ref = hf_attn[forward][layer].numpy()
            ref_last = ref if forward == 0 else ref[:, :, -1:, :]
            diff = max_abs(exported_w, ref_last)
            if diff >= TOL["attention_max_abs"]:
                check(f"hf.attention_f{forward}_l{layer}", False, diff, f"< {TOL['attention_max_abs']}")
                break
        else:
            check(f"hf.attention_all_layers_f{forward}", True, "< tolerance", f"< {TOL['attention_max_abs']}")

    # Event/sequence and tensor role invariants.
    for event in case["events"]:
        if event["kind"] == "sample":
            expected_pos = event["output_position"]
            check(f"sample.position.{event['id']}", expected_pos == event["forward_index"] + len(case["input"]["ids"]), expected_pos, event["forward_index"] + len(case["input"]["ids"]))
            check(f"sample.rank.{event['id']}", event["rank"] >= 1, event["rank"], ">=1")
    check("cache.allocated_bytes", case["events"][1]["cache"]["allocated"]["total_bytes"] == 11796480, case["events"][1]["cache"]["allocated"]["total_bytes"], 11796480)
    check("cache.hkv_shape", case["events"][1]["cache"]["allocated"]["per_layer_k_shape"] == [1, 3, 256, 64], case["events"][1]["cache"]["allocated"]["per_layer_k_shape"], [1, 3, 256, 64])

    failed = [c for c in checks if c["status"] != "passed" and c["required"]]
    summary = {
        "total": len(checks),
        "passed": sum(1 for c in checks if c["status"] == "passed"),
        "failed": sum(1 for c in checks if c["status"] != "passed"),
        "required": sum(1 for c in checks if c["required"]),
        "required_failed": sum(1 for c in checks if c["required"] and c["status"] != "passed"),
    }
    verification = {
        "schema": "atlas.trace.verification/2",
        "case_id": args.case_id,
        "case_sha256": case_sha,
        "tool": "verify_llama.py",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu", "dtype": "float32", "threads": THREADS},
        "reference": {"framework": "transformers", "version": __import__("transformers").__version__, "class": "LlamaForCausalLM", "device": "cpu", "dtype": "float32"},
        "summary": summary,
        "checks": checks,
        "tolerances": TOL,
        "generated_ids": generated_ids,
        "notes": [
            "载荷全部重新解码；RMSNorm/RoPE/GQA/SwiGLU/残差从导出张量重算，不调用导出器函数。",
            "logits/生成/注意力与本地 HF 权重参考对照；HuggingFace 参考只用于验证。",
            "失败不会写 verification.json；不得失败后放宽阈值。",
        ],
    }
    if failed:
        print(json.dumps({"summary": summary, "failed": failed}, ensure_ascii=False, indent=2)[:8000])
        return 1
    tmp = verification_path.with_name(verification_path.name + ".tmp")
    tmp.write_text(json.dumps(verification, ensure_ascii=False, indent=2) + "\n")
    os.replace(tmp, verification_path)
    print(f"wrote {verification_path}: {summary['passed']}/{summary['total']} passed, required_failed={summary['required_failed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
