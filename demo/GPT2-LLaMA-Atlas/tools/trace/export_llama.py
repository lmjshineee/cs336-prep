#!/usr/bin/env python3
"""Export one verified SmolLM2-135M LLaMA trace (atlas.trace.case/2).

Read-only with respect to the material directory. Copies the two frozen source files,
loads local weights/tokenizer, runs a real prefilled 7-token greedy generation with
capture, and writes data/llama/. The exporter deliberately does NOT write a passing
verification: verify_llama.py must independently check the payload.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import platform
import struct
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
MATERIAL_DEFAULT = Path("/Users/yunni/Projects/temp/cs336-prep-code")
CASE_ID = "smollm2-main-002"
MODEL_ID = "smollm2-135m"
PROMPT = "To make a cup of tea, first"
N_NEW = 7
THREADS = 2

sys.path[:0] = [str(ROOT / "tools" / "trace"), str(ROOT / "sources" / "llama")]


# ------------------------------------------------------------------ utilities
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def b64_tensor(tensor: torch.Tensor, dtype: str = "float32") -> dict:
    if dtype == "uint8":
        arr = tensor.detach().to(torch.uint8).cpu().contiguous()
        raw = arr.numpy().tobytes(order="C")
    else:
        arr = tensor.detach().to(torch.float32).cpu().contiguous()
        raw = arr.numpy().tobytes(order="C")
    return {
        "encoding": "u8-base64" if dtype == "uint8" else "f32le-base64",
        "nbytes": len(raw),
        "sha256": sha256_bytes(raw),
        "data": base64.b64encode(raw).decode("ascii"),
    }


def shape_of(tensor: torch.Tensor) -> list[int]:
    return [int(v) for v in tensor.shape]


def axes_for(shape: list[int], role: str) -> list[str]:
    templates = {
        "attn_weights": ["B", "Hq", "Tq", "Tk"],
        "scores_scaled": ["B", "Hq", "Tq", "Tk"],
        "causal_mask": ["Tq", "Tk"],
        "q_rope": ["B", "Hq", "T", "D"],
        "k_rope": ["B", "Hkv", "T", "D"],
        "v_heads": ["B", "Hkv", "T", "D"],
        "k_cache": ["B", "Hkv", "Tk", "D"],
        "v_cache": ["B", "Hkv", "Tk", "D"],
        "k_expanded": ["B", "Hq", "Tk", "D"],
        "v_expanded": ["B", "Hq", "Tk", "D"],
        "k_proj": ["B", "T", "Hkv·D"],
        "v_proj": ["B", "T", "Hkv·D"],
        "cos": ["T", "D"],
        "sin": ["T", "D"],
        "gamma": ["C"],
        "mean_sq": ["B", "T", "1"],
        "inv_rms": ["B", "T", "1"],
        "gate_proj": ["B", "T", "F"],
        "silu_gate": ["B", "T", "F"],
        "up_proj": ["B", "T", "F"],
        "gated": ["B", "T", "F"],
        "logits_row": ["B", "V"],
    }
    if role in templates and len(templates[role]) == len(shape):
        return templates[role]
    if len(shape) == 4:
        return ["B", "H*", "T*", "D"]
    if len(shape) == 3:
        return ["B", "T", "C*"]
    if len(shape) == 2:
        return ["A", "B"]
    return [f"X{i}" for i in range(len(shape))]


# ------------------------------------------------------------------ capture
class Recorder:
    def __init__(self):
        self.enabled = False
        self.forward = None
        self.tensors = []
        self.attention = {}
        self.hooks = []

    def begin(self, forward: int) -> None:
        self.enabled = True
        self.forward = forward
        self.tensors = []
        self.attention = {}

    def end(self):
        self.enabled = False
        result = {"tensors": list(self.tensors), "attention": dict(self.attention)}
        self.tensors = []
        self.attention = {}
        return result

    def add(self, event_id: str, role: str, tensor: torch.Tensor, op: str,
            scope: str = "full", derivation: str = "observed", dtype: str = "float32") -> str:
        if not self.enabled:
            return ""
        tensor_id = f"{event_id}.{role}"
        self.tensors.append({
            "id": tensor_id,
            "event_id": event_id,
            "role": role,
            "tensor": tensor.detach().clone(),
            "op": op,
            "scope": scope,
            "derivation": derivation,
            "dtype": dtype,
        })
        return tensor_id


REC = Recorder()


def install_capture(AttentionClass, apply_rope, repeat_kv, KVCacheClass, math_sqrt):
    original = AttentionClass.forward

    def captured_forward(self, x, cos, sin, kv_cache=None, start_pos=0):
        B, T, C = x.shape
        H, Hkv, D = self.n_head, self.n_kv_head, self.head_dim
        f = REC.forward
        block = f"fwd{f}.block{self.layer_idx}"

        q_flat = self.q_proj(x)
        k_flat = self.k_proj(x)
        v_flat = self.v_proj(x)
        q = q_flat.view(B, T, H, D).transpose(1, 2)
        k = k_flat.view(B, T, Hkv, D).transpose(1, 2)
        v = v_flat.view(B, T, Hkv, D).transpose(1, 2)
        q_rope = apply_rope(q, cos, sin)
        k_rope = apply_rope(k, cos, sin)
        v_heads = v

        if kv_cache is not None:
            k_all, v_all = kv_cache.append(self.layer_idx, k_rope, v_heads, start_pos)
        else:
            k_all, v_all = k_rope, v_heads
        k_exp = repeat_kv(k_all, self.n_rep)
        v_exp = repeat_kv(v_all, self.n_rep)
        scores_pre = q_rope @ k_exp.transpose(-2, -1) / math_sqrt(D)
        total = k_all.size(2)
        q_pos = torch.arange(start_pos, start_pos + T, device=x.device)[:, None]
        k_pos = torch.arange(total, device=x.device)[None, :]
        mask = (k_pos <= q_pos)
        scores = scores_pre.masked_fill(~mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        out = (weights @ v_exp).transpose(1, 2).reshape(B, T, C)
        attn_out = self.o_proj(out)

        if REC.enabled and REC.forward is not None:
            REC.attention[(f, self.layer_idx)] = weights.detach().clone()
            if self.layer_idx == 0:
                REC.add(block, "q_proj", q_flat, "llama.attn.q_proj")
                REC.add(block, "k_proj", k_flat, "llama.attn.k_proj")
                REC.add(block, "v_proj", v_flat, "llama.attn.v_proj")
                REC.add(block, "q_rope", q_rope, "llama.rope.apply_q")
                REC.add(block, "k_rope", k_rope, "llama.rope.apply_k")
                REC.add(block, "v_heads", v_heads, "llama.attn.v_heads")
                REC.add(block, "k_cache", k_all, "llama.cache.append")
                REC.add(block, "v_cache", v_all, "llama.cache.append")
                REC.add(block, "k_expanded", k_exp, "llama.gqa.repeat_kv")
                REC.add(block, "v_expanded", v_exp, "llama.gqa.repeat_kv")
                REC.add(block, "scores_scaled", scores_pre.detach(), "llama.attn.scores")
                REC.add(block, "causal_mask", mask.to(torch.uint8).expand(T, total), "llama.attn.mask", dtype="uint8")
                REC.add(block, "attn_out", attn_out, "llama.attn.o_proj")
        return attn_out

    AttentionClass.forward = captured_forward
    return original


def install_hooks(model, rec: Recorder):
    handles = []

    def hook_module(module, role, op, *, dtype="float32", scope="full", derivation="observed"):
        def fn(_module, inputs, output):
            if rec.enabled and rec.forward is not None:
                f = rec.forward
                rec.add(f"fwd{f}.block0", role, output, op, scope=scope, derivation=derivation, dtype=dtype)
        handles.append(module.register_forward_hook(fn))

    def hook_input(module, role, op):
        def fn(_module, inputs):
            if rec.enabled and rec.forward is not None:
                rec.add(f"fwd{rec.forward}.block0", role, inputs[0], op)
        handles.append(module.register_forward_pre_hook(fn))

    def hook_top(module, role, op, event_suffix):
        def fn(_module, inputs, output):
            if rec.enabled and rec.forward is not None:
                rec.add(f"fwd{rec.forward}.{event_suffix}", role, output, op)
        handles.append(module.register_forward_hook(fn))

    block0 = model.blocks[0]

    def block_pre(_module, inputs):
        if rec.enabled and rec.forward is not None:
            rec.add(f"fwd{rec.forward}.block0", "x_in", inputs[0], "llama.block.forward")

    def block_post(_module, inputs, output):
        if rec.enabled and rec.forward is not None:
            rec.add(f"fwd{rec.forward}.block0", "x_out", output, "llama.block.forward")

    block0.register_forward_pre_hook(block_pre)
    block0.register_forward_hook(block_post)
    hook_input(block0.attn_norm, "norm1_in", "llama.rmsnorm")
    hook_module(block0.attn_norm, "norm1_out", "llama.rmsnorm")
    hook_module(block0.ffn_norm, "norm2_out", "llama.rmsnorm")
    hook_input(block0.ffn_norm, "norm2_in", "llama.rmsnorm")
    hook_module(block0.ffn.gate_proj, "gate_proj", "llama.swiglu.gate")
    hook_module(block0.ffn.up_proj, "up_proj", "llama.swiglu.up")
    hook_module(block0.ffn.down_proj, "down_out", "llama.swiglu.down")
    hook_top(model.token_embedding, "token_embed", "llama.embed", "embed")
    hook_top(model.norm, "out", "llama.rmsnorm", "ln_f")
    return handles


# ------------------------------------------------------------------ model/data
def load_model(material: Path):
    import step06_model as core
    import step09_real_llama as real

    raw = json.loads((material / "llama_steps" / "weights" / "smollm2-135m" / "config.json").read_text())
    config = real.load_config(raw)
    from safetensors.torch import load_file
    sd = load_file(str(material / "llama_steps" / "weights" / "smollm2-135m" / "model.safetensors"))
    mapped = real.hf_to_ours(sd, config)
    if config.tie_word_embeddings and "lm_head.weight" not in mapped:
        mapped["lm_head.weight"] = mapped["token_embedding.weight"]
    model = core.MiniLlama(config)
    model.load_state_dict(mapped, strict=True)
    model.eval()
    return model, config, raw


def token_record(tok, ids: list[int], position: int, ordinal: int, byte_offset: int) -> dict:
    raw_token = tok.convert_ids_to_tokens(ids[position])
    decoded = tok.decode([ids[position]], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    raw_bytes = decoded.encode("utf-8")
    return {
        "position": position,
        "ordinal": ordinal,
        "id": int(ids[position]),
        "raw_token": raw_token,
        "standalone_decode": decoded,
        "byte_complete": True,
        "bytes_hex": raw_bytes.hex(),
        "byte_start": byte_offset,
        "byte_end": byte_offset + len(raw_bytes),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--material-root", default=str(MATERIAL_DEFAULT))
    parser.add_argument("--data-dir", default="data/llama")
    parser.add_argument("--case-id", default=CASE_ID)
    args = parser.parse_args()
    material = Path(args.material_root)
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(THREADS)
    torch.manual_seed(0)

    import step06_model as core
    import step09_real_llama as real
    from transformers import AutoTokenizer
    import transformers

    weights_dir = material / "llama_steps" / "weights" / "smollm2-135m"
    config_raw = json.loads((weights_dir / "config.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(weights_dir), local_files_only=True)
    model, config, _ = load_model(material)
    block0 = model.blocks[0]
    original_attention = install_capture(core.Attention, core.apply_rope, core.repeat_kv, core.KVCache, math.sqrt)
    handles = install_hooks(model, REC)

    prompt_ids = tokenizer(PROMPT, return_tensors="pt").input_ids
    input_ids = prompt_ids[0].tolist()
    tokens = []
    byte_offset = 0
    for pos, token_id in enumerate(input_ids):
        rec = token_record(tokenizer, input_ids, pos, pos + 1, byte_offset)
        byte_offset = rec["byte_end"]
        tokens.append(rec)
    if byte_offset != len(PROMPT.encode("utf-8")):
        raise RuntimeError(f"token byte coverage mismatch: {byte_offset} vs {len(PROMPT.encode('utf-8'))}")

    source_map = json.loads((ROOT / "sources" / "llama" / "source-map.json").read_text())

    def source_for(op: str) -> dict:
        entry = source_map["operations"].get(op)
        if not entry:
            raise RuntimeError(f"source-map missing operation {op}")
        return {"file": entry["file"], "symbol": entry["symbol"], "lines": entry["lines"], "span_sha256": entry["span_sha256"]}

    cache = core.KVCache(config, 1, torch.device("cpu"), dtype=torch.float32)
    generated_ids: list[int] = []
    events: list[dict] = []
    tensors: list[dict] = []
    forward_summaries = []
    all_logits = []
    next_input = prompt_ids
    sequence_ids = list(input_ids)

    def add_event(event: dict) -> dict:
        event["seq"] = len(events)
        events.append(event)
        return event

    def cache_state(forward: int, layer: int | None, phase: str) -> dict:
        start = 0 if forward == 0 else len(prompt_ids[0]) + forward - 1
        T = len(input_ids) if forward == 0 else 1
        total = start + T
        effective = [start] * config.n_layer
        written = [0] * config.n_layer
        if phase == "block":
            for l in range((layer or 0) + 1):
                effective[l] = total
                written[l] = T
        elif phase in ("ln_f", "logits"):
            effective = [total] * config.n_layer
        elif phase == "embed":
            effective = [start] * config.n_layer
        read_window = {"start": 0, "end_exclusive": total}
        write_window = {"start": start, "end_exclusive": total} if phase in ("block", "logits", "ln_f") else {"start": start, "end_exclusive": start}
        return {
            "capacity_positions": config.block_size,
            "max_position_embeddings": 8192,
            "allocated": {
                "per_layer_k_shape": [1, config.n_kv_head, config.block_size, config.n_embd // config.n_head],
                "per_layer_v_shape": [1, config.n_kv_head, config.block_size, config.n_embd // config.n_head],
                "per_layer_bytes": 2 * config.n_kv_head * config.block_size * (config.n_embd // config.n_head) * 4,
                "total_bytes": cache.nbytes(),
                "dtype": "float32",
                "note": "预分配容量，不是已写入内容",
            },
            "layers_effective_positions": effective,
            "layers_written_positions": written,
            "synchronized_positions": min(effective),
            "read_window": read_window,
            "write_window": write_window if phase != "sample" else None,
            "selected_layer_read_refs": None,
        }

    with torch.no_grad():
        for forward in range(N_NEW):
            start_pos = 0 if forward == 0 else len(prompt_ids[0]) + forward - 1
            REC.begin(forward)
            logits = model(next_input, kv_cache=cache, start_pos=start_pos)
            captured = REC.end()
            logits_row = logits[0, -1, :].detach().float().clone()
            all_logits.append(logits_row)
            T = next_input.shape[1]
            total = start_pos + T
            positions = {"query": list(range(start_pos, total)), "key": list(range(total))}
            cos = model.rope_cos[start_pos:total].detach().clone()
            sin = model.rope_sin[start_pos:total].detach().clone()

            events_for_forward = []
            event_prefix = f"fwd{forward}"
            add_event({
                "id": f"{event_prefix}.embed", "kind": "embed", "phase": "prefill" if forward == 0 else "decode",
                "forward_index": forward, "layer": None, "position_start": start_pos, "position_count": T,
                "total_positions": total, "input_ids": next_input[0].tolist(), "positions": positions,
                "cache": cache_state(forward, None, "embed"), "tensor_ids": [], "primary_op": "llama.embed",
                "source": source_for("llama.embed"),
                "replay": {"phase_complete": False, "sequence_length_after": len(sequence_ids)},
            })
            # selected layer intermediates are produced by hooks; attach after block events.
            selected = {item["role"]: item for item in captured["tensors"] if item["event_id"].endswith("block0")}
            selected_ids = []
            for role, item in selected.items():
                if role not in ("q_proj", "k_proj", "v_proj", "q_rope", "k_rope", "v_heads", "k_cache", "v_cache", "k_expanded", "v_expanded", "scores_scaled", "causal_mask", "attn_out", "x_in", "x_out", "norm1_out", "norm2_out", "gate_proj", "up_proj", "down_out"):
                    continue
                meta = tensor_meta(item, source_for(item["op"]))
                tensors.append(meta)
                selected_ids.append(meta["id"])
            # derived selected tensors
            def add_derived(role, tensor, op, derivation="derived", dtype="float32"):
                item = {"role": role, "tensor": tensor.detach().clone(), "op": op, "event_id": f"{event_prefix}.block0", "scope": "full", "derivation": derivation, "dtype": dtype}
                meta = tensor_meta(item, source_for(op))
                tensors.append(meta)
                selected_ids.append(meta["id"])

            # RMSNorm gamma is a real parameter; export it with each selected layer event.
            for role, module in [("norm1_gamma", block0.attn_norm), ("norm2_gamma", block0.ffn_norm)]:
                item = {"role": role, "tensor": module.weight, "op": "llama.rmsnorm", "event_id": f"{event_prefix}.block0", "scope": "full", "derivation": "observed", "dtype": "float32"}
                meta = tensor_meta(item, source_for("llama.rmsnorm"))
                tensors.append(meta)
                selected_ids.append(meta["id"])
            if "norm1_in" in selected:
                x_in = selected["norm1_in"]["tensor"]
                add_derived("norm1_mean_sq", x_in.float().pow(2).mean(-1, keepdim=True), "llama.rmsnorm")
                add_derived("norm1_inv_rms", torch.rsqrt(x_in.float().pow(2).mean(-1, keepdim=True) + config.norm_eps), "llama.rmsnorm")
            if "norm2_in" in selected:
                x2 = selected["norm2_in"]["tensor"]
                add_derived("norm2_mean_sq", x2.float().pow(2).mean(-1, keepdim=True), "llama.rmsnorm")
                add_derived("norm2_inv_rms", torch.rsqrt(x2.float().pow(2).mean(-1, keepdim=True) + config.norm_eps), "llama.rmsnorm")
            if "gate_proj" in selected and "up_proj" in selected:
                gate = selected["gate_proj"]["tensor"]
                up = selected["up_proj"]["tensor"]
                add_derived("silu_gate", F.silu(gate), "llama.swiglu.gate")
                add_derived("gated", F.silu(gate) * up, "llama.swiglu.mul")
            if "x_in" in selected and "attn_out" in selected:
                add_derived("x_after_attn", selected["x_in"]["tensor"] + selected["attn_out"]["tensor"], "llama.block.residual_attn")
            # per-forward RoPE table slice; attach to the selected block event.
            for role, tensor, op in [("cos", cos, "llama.rope.build"), ("sin", sin, "llama.rope.build")]:
                item = {"role": role, "tensor": tensor, "op": op, "event_id": f"{event_prefix}.block0", "scope": "slice", "derivation": "observed", "dtype": "float32"}
                meta = tensor_meta(item, source_for(op))
                tensors.append(meta)
                selected_ids.append(meta["id"])
            embed_ids = []
            for role, tensor, op in [("token_embed", captured.get("embed", None), "llama.embed")]:
                if tensor is None:
                    # hook stores token_embed in captured tensors with event fwd0.embed; use it if present
                    found = [item for item in captured["tensors"] if item["role"] == "token_embed"]
                    tensor = found[0]["tensor"] if found else None
                if tensor is not None:
                    item = {"role": role, "tensor": tensor, "op": op, "event_id": f"{event_prefix}.embed", "scope": "full", "derivation": "observed", "dtype": "float32"}
                    meta = tensor_meta(item, source_for(op))
                    tensors.append(meta)
                    embed_ids.append(meta["id"])
            if not embed_ids:
                # fall back to observed model embedding output from a fresh lookup
                embed_val = model.token_embedding(next_input).detach()
                item = {"role": "token_embed", "tensor": embed_val, "op": "llama.embed", "event_id": f"{event_prefix}.embed", "scope": "full", "derivation": "observed", "dtype": "float32"}
                meta = tensor_meta(item, source_for("llama.embed"))
                tensors.append(meta)
                embed_ids.append(meta["id"])
            events[-1]["tensor_ids"] = embed_ids

            # attention weights all layers + selected layer IDs
            for layer in range(config.n_layer):
                weight_id = f"{event_prefix}.block{layer}.attn_weights"
                weight = captured["attention"].get((forward, layer))
                if weight is None:
                    raise RuntimeError(f"missing attention weight f{forward} L{layer}")
                item = {"role": "attn_weights", "tensor": weight, "op": "llama.attn.softmax", "event_id": f"{event_prefix}.block{layer}", "scope": "full", "derivation": "observed", "dtype": "float32"}
                meta = tensor_meta(item, source_for("llama.attn.softmax"))
                tensors.append(meta)
                ids = [meta["id"]]
                if layer == 0:
                    ids.extend(selected_ids)
                cache_info = cache_state(forward, layer, "block")
                cache_info["selected_layer_read_refs"] = {"k": f"{event_prefix}.block0.k_cache", "v": f"{event_prefix}.block0.v_cache"} if layer == 0 else None
                add_event({
                    "id": f"{event_prefix}.block{layer}", "kind": "block", "phase": "prefill" if forward == 0 else "decode",
                    "forward_index": forward, "layer": layer, "position_start": start_pos, "position_count": T,
                    "total_positions": total, "input_ids": next_input[0].tolist(), "positions": positions,
                    "cache": cache_info, "tensor_ids": ids, "primary_op": "llama.block.forward",
                    "source": source_for("llama.block.forward"),
                    "replay": {"phase_complete": False, "sequence_length_after": len(sequence_ids)},
                })

            lnf = [item for item in captured["tensors"] if item["role"] == "out"]
            lnf_ids = []
            item_gamma = {"role": "norm_gamma", "tensor": model.norm.weight, "op": "llama.rmsnorm", "event_id": f"{event_prefix}.ln_f", "scope": "full", "derivation": "observed", "dtype": "float32"}
            meta_gamma = tensor_meta(item_gamma, source_for("llama.rmsnorm"))
            tensors.append(meta_gamma)
            lnf_ids.append(meta_gamma["id"])
            if lnf:
                item = {"role": "out", "tensor": lnf[0]["tensor"], "op": "llama.rmsnorm", "event_id": f"{event_prefix}.ln_f", "scope": "full", "derivation": "observed", "dtype": "float32"}
                meta = tensor_meta(item, source_for("llama.rmsnorm"))
                tensors.append(meta)
                lnf_ids.append(meta["id"])
            add_event({
                "id": f"{event_prefix}.ln_f", "kind": "ln_f", "phase": "prefill" if forward == 0 else "decode",
                "forward_index": forward, "layer": None, "position_start": start_pos, "position_count": T,
                "total_positions": total, "input_ids": next_input[0].tolist(), "positions": positions,
                "cache": cache_state(forward, None, "ln_f"), "tensor_ids": lnf_ids, "primary_op": "llama.rmsnorm",
                "source": source_for("llama.rmsnorm"),
                "replay": {"phase_complete": True, "sequence_length_after": len(sequence_ids)},
            })

            used = forward < N_NEW
            full_shape = list(logits.shape)
            per_position_argmax = logits[0].argmax(-1).tolist()
            per_position_max = logits[0].max(-1).values.float().tolist()
            logits_ids = []
            if used:
                item = {"role": "last_row", "tensor": logits_row.unsqueeze(0), "op": "llama.lm_head", "event_id": f"{event_prefix}.logits", "scope": "slice", "derivation": "observed", "dtype": "float32"}
                meta = tensor_meta(item, source_for("llama.lm_head"))
                tensors.append(meta)
                logits_ids.append(meta["id"])
            add_event({
                "id": f"{event_prefix}.logits", "kind": "logits", "phase": "prefill" if forward == 0 else "decode",
                "forward_index": forward, "layer": None, "position_start": start_pos, "position_count": T,
                "total_positions": total, "input_ids": next_input[0].tolist(), "positions": positions,
                "cache": cache_state(forward, None, "logits"), "tensor_ids": logits_ids, "primary_op": "llama.lm_head",
                "source": source_for("llama.lm_head"),
                "logits_event": {
                    "full_output_shape": full_shape, "full_output_dtype": "float32",
                    "last_row_index": total - 1, "used_by_sampling": used,
                    "used_by_sample": f"sample{forward}" if used else None,
                    "per_position_argmax": per_position_argmax, "per_position_max_logit": per_position_max,
                    "note": "只导出被采样使用的最后一行完整 logits；其余位置仅摘要。",
                },
                "replay": {"phase_complete": True, "sequence_length_after": len(sequence_ids)},
            })

            forward_summaries.append({
                "forward_index": forward, "phase": "prefill" if forward == 0 else "decode",
                "start_pos": start_pos, "T": T, "total_positions": total,
                "logits_shape": full_shape, "used_by_sampling": used,
                "input_ids": next_input[0].tolist(),
            })

            if used:
                row = logits_row.detach().to(torch.float64)
                probs = torch.softmax(row, dim=-1)
                token_id = int(row.argmax().item())
                probability = float(probs[token_id].item())
                rank = int((probs > probs[token_id]).sum().item()) + 1
                entropy = float(-(probs * torch.log(probs.clamp_min(1e-300))).sum().item())
                top = torch.topk(probs, 10)
                top_k = [{"id": int(i), "probability": float(p), "decoded_text": tokenizer.decode([int(i)], clean_up_tokenization_spaces=False), "raw_token": tokenizer.convert_ids_to_tokens(int(i))} for p, i in zip(top.values.tolist(), top.indices.tolist())]
                other = float(1.0 - top.values.sum().item())
                generated_ids.append(token_id)
                sequence_ids.append(token_id)
                raw_token = tokenizer.convert_ids_to_tokens(token_id)
                decoded = tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
                add_event({
                    "id": f"sample{forward}", "kind": "sample", "phase": "sample",
                    "forward_index": forward, "output_position": total, "strategy": "greedy",
                    "position_start": total, "position_count": 0, "total_positions": total,
                    "cache": cache_state(forward, None, "logits"), "positions": {"query": [total], "key": list(range(total))},
                    "token_id": token_id, "raw_token": raw_token, "decoded_text": decoded,
                    "standalone_decode_complete": True, "probability": probability, "rank": rank,
                    "top_k": top_k, "other_probability_mass": other, "entropy_nats": entropy,
                    "probability_basis": "对完整 vocab 行做 float64 softmax 后统计；top_k 仅展示，未重新归一化",
                    "tensor_ids": [], "primary_op": "llama.generate.sample",
                    "source": source_for("llama.generate.sample"),
                    "replay": {"phase_complete": True, "sequence_length_after": len(sequence_ids)},
                })
                next_input = torch.tensor([[token_id]], dtype=torch.long)

    for h in handles:
        h.remove()
    core.Attention.forward = original_attention

    # Reference without wrapper for wrapper-equivalence evidence.
    with torch.no_grad():
        ref_logits = model(prompt_ids)[0, -1, :].float()
    wrapper_diff = float((ref_logits - all_logits[0]).abs().max().item())

    tensors.sort(key=lambda item: item["id"])
    execution = {
        "batch_size": 1,
        "config": {
            "n_layer": config.n_layer, "n_head": config.n_head, "n_kv_head": config.n_kv_head,
            "n_embd": config.n_embd, "head_dim": config.n_embd // config.n_head,
            "ffn_hidden": config.ffn_hidden, "vocab_size": config.vocab_size,
            "block_size": config.block_size, "rms_norm_eps": config.norm_eps,
            "rope_theta": config.rope_theta, "rope_style": "half_split",
            "tie_word_embeddings": config.tie_word_embeddings,
            "max_position_embeddings": 8192,
        },
        "generation": {
            "strategy": "greedy", "max_new_tokens": N_NEW, "prompt_len": len(input_ids),
            "forward_count": N_NEW, "sample_count": N_NEW,
            "stop_reason": "max_new_tokens",
            "generated_ids": generated_ids,
            "full_sequence_ids": sequence_ids,
            "note": "prefill 一次；仅为后续选词执行 decode，达到生成上限立即结束。",
        },
        "selected_layer": 0,
        "wrapper_equivalence_max_abs": wrapper_diff,
        "wrapper_equivalence_tolerance": 1e-6,
        "collection_budget_note": "所有层×9 Q 头注意力；仅第 0 层完整中间张量；导出全部被采样最后行 logits；缓存不导出零区",
    }
    case = {
        "schema": "atlas.trace.case/2",
        "case_id": args.case_id,
        "model_id": MODEL_ID,
        "source_map_version": source_map["map_version"],
        "provenance_ref": "manifest.json#/provenance",
        "vocabulary": {"decoded": [tokenizer.decode([i], clean_up_tokenization_spaces=False) for i in range(config.vocab_size)], "raw": [tokenizer.convert_ids_to_tokens(i) for i in range(config.vocab_size)]},
        "input": {
            "text": PROMPT,
            "byte_length": len(PROMPT.encode("utf-8")),
            "tokenizer_note": "真实 SmolLM2 BPE tokenizer；单 token 不一定能独立解码",
            "tokens": tokens,
            "ids": input_ids,
        },
        "execution": execution,
        "events": events,
        "forward_summaries": forward_summaries,
        "tensors": tensors,
    }
    case_text = json.dumps(case, ensure_ascii=False, separators=(",", ":"), sort_keys=False)
    case_path = data_dir / f"{args.case_id}.json"
    tmp = case_path.with_name(case_path.name + ".tmp")
    tmp.write_text(case_text + "\n", encoding="utf-8")
    os.replace(tmp, case_path)
    case_sha = sha256_file(case_path)

    impl_files = []
    for name in ("step06_model.py", "step09_real_llama.py"):
        snapshot = ROOT / "sources" / "llama" / name
        impl_files.append({"role": "model" if name.startswith("step06") else "real_loader", "sha256": sha256_file(snapshot), "upstream_sha256": sha256_file(material / "llama_steps" / name), "snapshot": f"sources/llama/{name}"})
    config_path = weights_dir / "config.json"
    tokenizer_path = weights_dir / "tokenizer.json"
    tokenizer_cfg_path = weights_dir / "tokenizer_config.json"
    manifest = {
        "schema": "atlas.trace/2",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "tool": {"name": "export_llama.py", "version": "2.1.0", "command": " ".join(sys.argv)},
        "model": {
            "model_id": MODEL_ID, "family": "llama", "architecture": "LlamaForCausalLM",
            "display_name": "SmolLM2-135M（Llama 架构）", "weights_origin": "HuggingFaceTB/SmolLM2-135M",
            "not_meta_llama": True, "license": "Apache-2.0",
            "upstream": {"model_url": "https://huggingface.co/HuggingFaceTB/SmolLM2-135M", "config_url": "https://huggingface.co/HuggingFaceTB/SmolLM2-135M/raw/main/config.json", "revision": None, "revision_note": "本地副本使用可变 main；载荷身份以本地 SHA-256 为准"},
            "config": execution["config"],
            "weights_storage_dtype": "bfloat16", "runtime_dtype": "float32",
        },
        "provenance": {
            "implementation": {"main_file": "step06_model.py", "real_file": "step09_real_llama.py", "files": impl_files, "note": "项目内个人实现；由外部材料派生，仅调整生成终止条件；外部材料只读。"},
            "weights": {"path": "llama_steps/weights/smollm2-135m/model.safetensors", "sha256": sha256_file(weights_dir / "model.safetensors"), "bytes": (weights_dir / "model.safetensors").stat().st_size, "tensor_count": 272, "storage_dtype": "bfloat16", "parameter_elements": 134515008},
            "config": {"path": "llama_steps/weights/smollm2-135m/config.json", "sha256": sha256_file(config_path)},
            "tokenizer": {"tokenizer_json_sha256": sha256_file(tokenizer_path), "tokenizer_config_sha256": sha256_file(tokenizer_cfg_path), "vocab_size": len(tokenizer), "kind": "SmolLM2 BPE tokenizer", "special_ids": {"bos": 0, "eos": 0}},
            "reference_implementation": {"framework": "transformers", "version": transformers.__version__, "class": "LlamaForCausalLM", "attn_implementation": "eager", "device": "cpu", "note": "仅用于核对，不参与导出"},
            "runtime": {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__, "platform": platform.platform(), "device": "cpu", "dtype": "float32", "threads": THREADS, "eval": True, "grad_mode": "no_grad", "seed": 0},
        },
        "instances": [{
            "case_id": args.case_id, "case_file": case_path.name, "case_sha256": case_sha, "case_bytes": case_path.stat().st_size,
            "model_id": MODEL_ID, "label": "主实例", "input_text": PROMPT, "prompt_token_count": len(input_ids),
            "generated_token_ids": generated_ids, "selected_layer": 0, "status": "exported",
            "validation_file": "verification.json", "validation_status": "not_run",
        }],
        "source_snapshots": [{"path": "sources/llama/step06_model.py", "sha256": sha256_file(ROOT / "sources" / "llama" / "step06_model.py")}, {"path": "sources/llama/step09_real_llama.py", "sha256": sha256_file(ROOT / "sources" / "llama" / "step09_real_llama.py")}],
        "source_map": {"path": "sources/llama/source-map.json", "sha256": sha256_file(ROOT / "sources" / "llama" / "source-map.json"), "version": source_map["map_version"]},
    }
    manifest_path = data_dir / "manifest.json"
    tmp = manifest_path.with_name(manifest_path.name + ".tmp")
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, manifest_path)

    print(f"wrote {case_path} ({case_path.stat().st_size / 1024:.1f} KiB, {len(events)} events, {len(tensors)} tensors)")
    print(f"wrote {manifest_path}")
    print(f"generated_ids={generated_ids} wrapper_equiv={wrapper_diff:.3e} case_sha256={case_sha[:16]}…")
    print(f"selected tensors per forward={len([t for t in tensors if t['event_id'].endswith('block0')])}")
    return 0


def tensor_meta(item: dict, source: dict) -> dict:
    tensor = item["tensor"]
    shape = shape_of(tensor)
    dtype = item.get("dtype", "float32")
    payload = b64_tensor(tensor, dtype)
    return {
        "id": item.get("id") or f"{item['event_id']}.{item['role']}",
        "event_id": item["event_id"],
        "role": item["role"],
        "shape": shape,
        "axes": axes_for(shape, item["role"]),
        "dtype": dtype,
        "scope": item.get("scope", "full"),
        "derivation": item.get("derivation", "observed"),
        "op": item["op"],
        "payload": payload,
        "source": source,
    }


if __name__ == "__main__":
    raise SystemExit(main())
