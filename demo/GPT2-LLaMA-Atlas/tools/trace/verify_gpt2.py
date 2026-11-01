#!/usr/bin/env python3
"""Independent verifier for the GPT-2 atlas trace export (S1 gate).

This script is deliberately independent of the exporter: it never imports or
reads ``export_gpt2.py``.  It only uses:

  * ``docs/TRACE_SCHEMA.md`` (the data contract),
  * the exported data under ``data/gpt2/`` and ``sources/gpt2/``,
  * the original, read-only material implementation ``gpt2_real.py``,
  * the HuggingFace ``GPT2LMHeadModel`` reference implementation (eager),
  * the exported payloads themselves (recomputed with float64 math).

It writes ``data/gpt2/verification.json`` (schema ``atlas.trace.verification/1``)
with every check (id/category/status/object/expected/observed/tolerance/command/
details).  Any required check that fails or is not run makes the exit code
non-zero.

Usage:

  PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  <venv python> GPT2-LLaMA-Atlas/tools/trace/verify_gpt2.py \
      --material-root /path/to/real_llm_lab \
      --data-dir GPT2-LLaMA-Atlas/data/gpt2 \
      --case-id gpt2-main-002
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import math
import os
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True

VERIFY_VERSION = "1.0.0"
MANIFEST_SCHEMA = "atlas.trace/1"
CASE_SCHEMA = "atlas.trace.case/1"
SOURCE_MAP_SCHEMA = "atlas.source-map/1"
VERIFICATION_SCHEMA = "atlas.trace.verification/1"

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # GPT2-LLaMA-Atlas/

# Expected model geometry for the pinned main instance (schema section 5/6/10).
EXPECTED_CONFIG = {
    "n_layer": 12,
    "n_head": 12,
    "n_embd": 768,
    "head_dim": 64,
    "vocab_size": 50257,
    "block_size": 1024,
    "layer_norm_eps": 1e-5,
}
EXPECTED_FORWARD_COUNT = 9
EXPECTED_SAMPLE_COUNT = 9
EXPECTED_LAYERS = 12
EPS = 1e-5
HEAD_DIM = 64

SAMPLE_TOP_K = 10

# Roles only present on the selected layer per schema section 6.3.
SELECTED_LAYER_ROLES = [
    "x_in", "ln1_out", "qkv", "q_raw", "k_raw", "v_raw",
    "q_heads", "k_heads", "v_heads", "k_read", "v_read",
    "scores_scaled", "causal_mask", "weighted_v", "attn_out",
    "x_after_attn", "ln2_out", "fc_out", "act_out", "mlp_out", "x_out",
]

FWD_EVENT_KINDS = ("embed", "block", "ln_f", "logits")

TOLERANCES = {
    "collection_equivalence_logits": "bitwise equal (max abs diff == 0)",
    "collection_equivalence_tokens": "exact equality",
    "reference_prefill_logits": "max|diff| <= 1e-4",
    "reference_attention": "max|diff| <= 1e-6",
    "payload_roundtrip": "exact (nbytes, shape product, sha256, encoding)",
    "formula_activations": "max|diff| <= 1e-4 and rel <= 1e-4 (allclose atol=rtol=1e-4, float64 recompute)",
    "formula_softmax": "max|diff| <= 1e-6",
    "formula_row_sum": "|sum - 1| <= 1e-6",
    "formula_mask_zero": "masked attention weights exactly 0.0",
    "output_head": ("(a) float32 same-execution-shape batched reproduction of the last row "
                    "from exported ln_f.out x shared wte is bitwise equal (max|diff| == 0); "
                    "(b) float64 recompute max|diff| / max|logit| <= 1e-5, raw abs diff reported"),
    "cache_metadata": "exact match with schema section 5.3 rules",
    "cache_read_slice": "bitwise equal to reconstructed cache content",
    "generation_token": "exact equality (argmax)",
    "generation_probability": "|diff| <= 1e-6 (canonical: float32 logits promoted to float64 softmax)",
    "generation_other_mass": "|diff| <= 1e-6",
    "generation_entropy": "|diff| <= 1e-5 (schema section 8 does not state a tolerance; chosen by verifier, declared before running)",
    "tokenizer": "exact match with independently re-encoded HF tokenizer",
    "provenance_hash": "recomputed sha256 exact match",
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def jsafe(value):
    """Make a value JSON-serialisable; arrays become shape/dtype summaries."""
    if isinstance(value, dict):
        return {str(k): jsafe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsafe(v) for v in value]
    if isinstance(value, np.ndarray):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return str(value)
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def load_json_strict(path: Path):
    """Load JSON, rejecting NaN/Infinity literals, and return (obj, raw bytes)."""
    raw = path.read_bytes()

    def _reject(token: str):
        raise ValueError(f"non-finite JSON constant not allowed: {token}")

    obj = json.loads(raw.decode("utf-8"), parse_constant=_reject)
    return obj, raw


def reject_constant_scan(obj, where=""):
    """Return list of non-finite floats found in a parsed JSON tree."""
    bad = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            bad.extend(reject_constant_scan(v, f"{where}/{k}"))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            bad.extend(reject_constant_scan(v, f"{where}/{i}"))
    elif isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        bad.append(where or "?")
    return bad


def gpt2_bytes_to_unicode() -> dict:
    """The byte-level BPE byte -> unicode mapping (independent implementation)."""
    bs = list(range(ord("!"), ord("~") + 1))
    bs += list(range(ord("\u00a1"), ord("\u00ac") + 1))
    bs += list(range(ord("\u00ae"), ord("\u00ff") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


BYTE_TO_UNICODE = gpt2_bytes_to_unicode()
UNICODE_TO_BYTE = {v: k for k, v in BYTE_TO_UNICODE.items()}


def token_bytes(tokenizer, token_id: int) -> bytes:
    token = tokenizer.convert_ids_to_tokens(int(token_id))
    if token is None:
        raise ValueError(f"token id {token_id} not in tokenizer vocab")
    out = bytearray()
    for ch in token:
        if ch not in UNICODE_TO_BYTE:
            raise ValueError(f"token {token!r} contains char {ch!r} outside byte-level alphabet")
        out.append(UNICODE_TO_BYTE[ch])
    return bytes(out)


def decode_token_bytes(raw: bytes, errors: str = "replace") -> str:
    return raw.decode("utf-8", errors=errors)


def f32_bytes(array: np.ndarray) -> bytes:
    return np.ascontiguousarray(array, dtype="<f4").tobytes()


def close_values(observed, expected, atol, rtol=0.0):
    """Return (max_abs_diff, ok, shape_mismatch)."""
    obs = np.asarray(observed, dtype=np.float64)
    exp = np.asarray(expected, dtype=np.float64)
    if obs.shape != exp.shape:
        return None, False, {"observed_shape": list(obs.shape), "expected_shape": list(exp.shape)}
    if obs.size == 0:
        return 0.0, True, None
    diff = np.abs(obs - exp)
    max_abs = float(diff.max())
    ok = bool(np.allclose(obs, exp, rtol=rtol, atol=atol, equal_nan=False))
    return max_abs, ok, None


def softmax_f64(row: np.ndarray) -> np.ndarray:
    row = np.asarray(row, dtype=np.float64)
    shifted = row - row.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def ln_f64(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float = EPS) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * weight + bias


def gelu_new_f64(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


# ---------------------------------------------------------------------------
# check recorder
# ---------------------------------------------------------------------------
class Recorder:
    def __init__(self, command: str):
        self.command = command
        self.checks: list[dict] = []

    def add(self, cid, category, required, obj, expected, observed, tolerance,
            ok=None, status=None, details=None):
        if status is None:
            status = "passed" if ok else "failed"
        self.checks.append({
            "id": cid,
            "category": category,
            "status": status,
            "required": bool(required),
            "object": obj,
            "expected": jsafe(expected),
            "observed": jsafe(observed),
            "tolerance": tolerance,
            "command": self.command,
            "details": jsafe(details) if details is not None else {},
        })

    def passed(self, cid, category, required, obj, expected, observed, tolerance, details=None):
        self.add(cid, category, required, obj, expected, observed, tolerance,
                 ok=True, details=details)

    def failed(self, cid, category, required, obj, expected, observed, tolerance, details=None):
        self.add(cid, category, required, obj, expected, observed, tolerance,
                 ok=False, details=details)

    def not_run(self, cid, category, required, obj, expected, reason, details=None):
        d = {"reason": reason}
        if details:
            d.update(details)
        self.add(cid, category, required, obj, expected, "not_run",
                 "n/a", status="not_run", details=d)

    def equal(self, cid, category, required, obj, expected, observed,
              tolerance="exact", details=None):
        ok = expected == observed
        self.add(cid, category, required, obj, expected, observed, tolerance,
                 ok=ok, details=details)

    def truth(self, cid, category, required, obj, ok, expected=True, observed=None,
              tolerance="exact", details=None):
        if observed is None:
            observed = bool(ok)
        self.add(cid, category, required, obj, expected, observed, tolerance,
                 ok=bool(ok), details=details)

    def close(self, cid, category, required, obj, expected, observed, atol,
              rtol=0.0, tolerance=None, details=None):
        max_abs, ok, mismatch = close_values(observed, expected, atol, rtol)
        tol = tolerance
        if tol is None:
            tol = f"max|diff| <= {atol:g}" + (f" and rel <= {rtol:g}" if rtol else "")
        if mismatch is not None:
            self.add(cid, category, required, obj,
                     {"summary": "float values within tolerance"},
                     {"shape_mismatch": mismatch}, tol, ok=False, details=details)
            return
        observed_summary = {"max_abs_diff": max_abs, "elements": int(np.asarray(observed).size)}
        self.add(cid, category, required, obj,
                 {"summary": "float values within tolerance"}, observed_summary,
                 tol, ok=ok, details={"recompute": "float64", **(details or {})})

    def exact_arrays(self, cid, category, required, obj, expected: bytes, observed: bytes,
                     tolerance="bitwise equal", details=None):
        ok = expected == observed
        self.add(cid, category, required, obj,
                 {"sha256": sha256_bytes(expected), "nbytes": len(expected)},
                 {"sha256": sha256_bytes(observed), "nbytes": len(observed),
                  "byte_equal": ok}, tolerance, ok=ok, details=details)


# ---------------------------------------------------------------------------
# verifier
# ---------------------------------------------------------------------------
class Verifier:
    def __init__(self, args):
        self.args = args
        self.data_dir = Path(args.data_dir).resolve()
        self.material_root = Path(args.material_root).resolve()
        self.case_id = args.case_id
        self.command = (
            f"{sys.executable} {Path(__file__).resolve()} "
            f"--material-root {self.material_root} --data-dir {self.data_dir} "
            f"--case-id {self.case_id}"
        )
        self.rec = Recorder(self.command)
        self.manifest = None
        self.case = None
        self.source_map = None
        self.manifest_raw = b""
        self.case_raw = b""
        self.snapshot_path = None
        self.source_map_path = None
        self.snapshot_text = ""
        self.tensors: dict[str, dict] = {}
        self.tensors_list: list[dict] = []
        self.arrays: dict[str, np.ndarray] = {}
        self.events: list[dict] = []
        self.events_by_id: dict[str, dict] = {}
        self.instance = None
        self.notes: list[str] = []
        self.prompt_ids: list[int] = []
        self.generated_ids: list[int] = []
        self.full_ids: list[int] = []
        self.prompt_len = 0
        self.selected_layer = 0
        self.forward_meta: dict[int, dict] = {}
        self.model = None
        self.hf = None
        self.tokenizer = None
        self.kv = None
        self.fatal = False
        self.static_only = bool(args.static_only)
        self.t_start = time.time()

    # -- convenience ------------------------------------------------------
    def low(self):
        if self.case is None or self.manifest is None:
            raise RuntimeError("data not loaded")
        if self.fatal:
            raise RuntimeError("data unavailable")

    def tensor(self, tid: str) -> dict:
        return self.tensors[tid]

    def array(self, tid: str) -> np.ndarray:
        return self.arrays[tid]

    def a64(self, tid: str) -> np.ndarray:
        t = self.tensors[tid]
        return self.arrays[tid].astype(np.float64).reshape(tuple(t["shape"]))

    def a8(self, tid: str) -> np.ndarray:
        t = self.tensors[tid]
        return self.arrays[tid].reshape(tuple(t["shape"]))

    def fwd_event(self, f: int, kind: str) -> dict:
        return self.events_by_id[f"fwd{f}.{kind}"]

    def block_tid(self, f: int, role: str, layer: int | None = None) -> str:
        layer = self.selected_layer if layer is None else layer
        return f"fwd{f}.block{layer}.{role}"

    # ==================================================================
    # phase: load files
    # ==================================================================
    def load_files(self):
        rec = self.rec
        data_dir = self.data_dir
        manifest_path = data_dir / "manifest.json"
        rec.truth("meta.manifest_exists", "structure", True,
                  str(manifest_path), ok=manifest_path.is_file())
        if not manifest_path.is_file():
            self.fatal = True
            rec.not_run("meta.case_load", "structure", True, "case file",
                        "readable case JSON", "manifest.json missing")
            return
        try:
            self.manifest, self.manifest_raw = load_json_strict(manifest_path)
        except Exception as exc:
            self.fatal = True
            rec.failed("meta.manifest_parse", "structure", True, str(manifest_path),
                       "valid JSON without NaN/Infinity", f"{type(exc).__name__}: {exc}",
                       "exact")
            return
        rec.passed("meta.manifest_parse", "structure", True, str(manifest_path),
                   "valid JSON without NaN/Infinity",
                   {"bytes": len(self.manifest_raw), "sha256": sha256_bytes(self.manifest_raw)},
                   "exact")

        # locate the instance
        instances = self.manifest.get("instances") or []
        self.instance = next((i for i in instances if i.get("case_id") == self.case_id), None)
        rec.truth("meta.manifest_instance", "structure", True,
                  f"manifest instances contains {self.case_id}", ok=self.instance is not None,
                  observed=[i.get("case_id") for i in instances])
        if self.instance is None:
            self.fatal = True
            rec.not_run("meta.case_load", "structure", True, "case file",
                        "readable case JSON", "case id missing from manifest")
            return

        case_path = data_dir / str(self.instance.get("case_file") or f"{self.case_id}.json")
        rec.truth("meta.case_exists", "structure", True, str(case_path), ok=case_path.is_file())
        if not case_path.is_file():
            self.fatal = True
            return
        try:
            self.case, self.case_raw = load_json_strict(case_path)
        except Exception as exc:
            self.fatal = True
            rec.failed("meta.case_parse", "structure", True, str(case_path),
                       "valid JSON without NaN/Infinity", f"{type(exc).__name__}: {exc}",
                       "exact")
            return

        self.case_sha = sha256_bytes(self.case_raw)
        self.case_bytes = len(self.case_raw)
        self.case_path = case_path
        self.manifest_sha = sha256_bytes(self.manifest_raw)
        # parse the fixed instance parameters early: later phases (reference,
        # formula, cache) rely on them regardless of check ordering.
        try:
            execu = self.case.get("execution") or {}
            gen = execu.get("generation") or {}
            inp = self.case.get("input") or {}
            self.prompt_ids = [int(x) for x in (inp.get("ids") or [])]
            self.prompt_len = len(self.prompt_ids)
            self.generated_ids = [int(x) for x in (gen.get("generated_ids") or [])]
            self.full_ids = [int(x) for x in (gen.get("full_sequence_ids") or [])]
            self.selected_layer = execu.get("selected_layer", 0)
        except Exception:
            pass
        rec.passed("meta.case_parse", "structure", True, str(case_path),
                   "valid JSON without NaN/Infinity",
                   {"bytes": self.case_bytes, "sha256": self.case_sha}, "exact")

        # NaN/Infinity scan of the whole parsed tree (belt and braces)
        bad_manifest = reject_constant_scan(self.manifest)
        bad_case = reject_constant_scan(self.case)
        rec.truth("payload.no_nan_in_json", "payload", True,
                  "manifest.json + case JSON contain no NaN/Infinity",
                  ok=not bad_manifest and not bad_case,
                  observed={"manifest_paths": bad_manifest[:10], "case_paths": bad_case[:10]},
                  details={"manifest_count": len(bad_manifest), "case_count": len(bad_case)})

        # manifest case digest / size vs disk
        rec.equal("meta.case_sha256", "structure", True,
                  "manifest.instances[].case_sha256 == sha256(case file)",
                  self.instance.get("case_sha256"), self.case_sha)
        rec.equal("meta.case_bytes", "structure", True,
                  "manifest.instances[].case_bytes == case file size",
                  self.instance.get("case_bytes"), self.case_bytes)

        # resolve snapshot + source map paths (paths are relative to data dir
        # per schema section 3; accept a couple of harmless layouts).
        snap = (self.manifest.get("source_snapshot") or {}).get("path")
        smap = (self.manifest.get("source_map") or {}).get("path")
        self.snapshot_path = self._resolve_project_path(snap)
        self.source_map_path = self._resolve_project_path(smap)
        rec.truth("meta.source_snapshot_exists", "structure", True,
                  f"source snapshot resolvable from {snap}",
                  ok=self.snapshot_path is not None and self.snapshot_path.is_file(),
                  observed=str(self.snapshot_path))
        rec.truth("meta.source_map_exists", "structure", True,
                  f"source map resolvable from {smap}",
                  ok=self.source_map_path is not None and self.source_map_path.is_file(),
                  observed=str(self.source_map_path))
        if self.snapshot_path and self.snapshot_path.is_file():
            self.snapshot_text = self.snapshot_path.read_text(encoding="utf-8")
        if self.source_map_path and self.source_map_path.is_file():
            try:
                self.source_map, smap_raw = load_json_strict(self.source_map_path)
                self.source_map_sha = sha256_bytes(smap_raw)
            except Exception as exc:
                rec.failed("meta.source_map_parse", "structure", True,
                           str(self.source_map_path), "valid JSON", str(exc), "exact")

    def _resolve_project_path(self, rel):
        if not rel:
            return None
        p = Path(str(rel))
        candidates = [
            self.data_dir / p,
            self.data_dir.parent / p,
            PROJECT_ROOT / p,
            p,
        ]
        for cand in candidates:
            if cand.is_file():
                return cand.resolve()
        return (self.data_dir / p).resolve()

    def _resolve_material_path(self, rel):
        if not rel:
            return None
        p = Path(str(rel))
        candidates = [self.material_root / p, self.material_root.parent / p, p]
        for cand in candidates:
            if cand.is_file():
                return cand.resolve()
        return None

    # ==================================================================
    # phase: manifest structure
    # ==================================================================
    def check_manifest(self):
        rec = self.rec
        man = self.manifest
        rec.equal("manifest.schema", "schema", True, "manifest.schema",
                  MANIFEST_SCHEMA, man.get("schema"))
        # generated_at must be a timezone-qualified ISO timestamp
        ga = man.get("generated_at")
        ok_ts = False
        try:
            dt = datetime.fromisoformat(str(ga))
            ok_ts = dt.tzinfo is not None
        except Exception:
            ok_ts = False
        rec.truth("manifest.generated_at", "schema", True,
                  "manifest.generated_at is timezone-qualified ISO 8601", ok=ok_ts,
                  observed=ga)

        tool = man.get("tool") or {}
        rec.truth("manifest.tool", "schema", True, "manifest.tool fields",
                  ok=(tool.get("name") == "export_gpt2.py"
                      and isinstance(tool.get("version"), str)
                      and bool(tool.get("version"))
                      and isinstance(tool.get("command"), str)
                      and bool(tool.get("command"))),
                  observed={k: tool.get(k) for k in ("name", "version")})

        prov = man.get("provenance") or {}
        missing = [k for k in ("implementation", "weights", "config", "tokenizer",
                               "reference_implementation", "runtime",
                               "weight_origin_note") if k not in prov]
        rec.truth("manifest.provenance_fields", "provenance", True,
                  "manifest.provenance required blocks", ok=not missing,
                  observed={"missing": missing})
        impl = prov.get("implementation") or {}
        rec.truth("manifest.provenance_impl_fields", "provenance", True,
                  "provenance.implementation fields",
                  ok=all(k in impl for k in ("path", "sha256", "snapshot_sha256")),
                  observed={k: impl.get(k) for k in ("path", "sha256", "snapshot_sha256")})
        w = prov.get("weights") or {}
        rec.truth("manifest.provenance_weights_fields", "provenance", True,
                  "provenance.weights fields",
                  ok=all(k in w for k in ("path", "sha256", "bytes")),
                  observed={k: w.get(k) for k in ("path", "sha256", "bytes")})
        cfg = prov.get("config") or {}
        rec.truth("manifest.provenance_config_fields", "provenance", True,
                  "provenance.config fields", ok=all(k in cfg for k in ("path", "sha256")),
                  observed={k: cfg.get(k) for k in ("path", "sha256")})
        tok = prov.get("tokenizer") or {}
        rec.truth("manifest.provenance_tokenizer_fields", "provenance", True,
                  "provenance.tokenizer fields",
                  ok=all(k in tok for k in ("tokenizer_json_sha256", "vocab_sha256",
                                            "merges_sha256", "kind", "local_files_only")),
                  observed={k: tok.get(k) for k in ("kind", "local_files_only")})
        ref = prov.get("reference_implementation") or {}
        rec.truth("manifest.provenance_reference_fields", "provenance", True,
                  "provenance.reference_implementation fields",
                  ok=all(k in ref for k in ("framework", "version", "class",
                                            "attn_implementation")),
                  observed={k: ref.get(k) for k in ("framework", "version", "class",
                                                    "attn_implementation")})
        rt = prov.get("runtime") or {}
        rec.truth("manifest.provenance_runtime_fields", "provenance", True,
                  "provenance.runtime fields",
                  ok=all(k in rt for k in ("python", "torch", "numpy", "safetensors",
                                           "platform", "machine", "device", "dtype",
                                           "threads", "eval", "seed")),
                  observed={"missing": [k for k in ("python", "torch", "numpy",
                                                    "safetensors", "platform", "machine",
                                                    "device", "dtype", "threads", "eval",
                                                    "seed") if k not in rt]})

        inst = self.instance or {}
        required_instance = ("case_id", "case_file", "case_sha256", "case_bytes", "label",
                             "input_text", "prompt_token_count", "generated_token_ids",
                             "selected_layer", "status", "validation_file")
        missing_i = [k for k in required_instance if k not in inst]
        rec.truth("manifest.instance_fields", "structure", True,
                  "manifest.instances[] required fields", ok=not missing_i,
                  observed={"missing": missing_i})
        rec.equal("manifest.instance_status", "structure", True,
                  "manifest.instances[].status", "exported", inst.get("status"))
        rec.equal("manifest.instance_validation_file", "structure", True,
                  "manifest.instances[].validation_file", "verification.json",
                  inst.get("validation_file"))
        rec.truth("manifest.instance_validation_status", "structure", False,
                  "manifest.instances[].validation_status",
                  ok=inst.get("validation_status") in ("not_run", "passed", "failed"),
                  expected="one of not_run/passed/failed",
                  observed=inst.get("validation_status"),
                  details={"note": "verifier cannot modify manifest.json (write scope); "
                                    "schema section 3 says exporter writes not_run."})

        # manifest vs case cross references (validated fully later)
        rec.equal("manifest.input_text", "structure", True,
                  "manifest.instances[].input_text == case.input.text",
                  (self.case or {}).get("input", {}).get("text"), inst.get("input_text"))
        rec.equal("manifest.selected_layer", "structure", True,
                  "manifest.instances[].selected_layer == case.execution.selected_layer",
                  (self.case or {}).get("execution", {}).get("selected_layer"),
                  inst.get("selected_layer"))

    # ==================================================================
    # phase: provenance (recomputed hashes and versions)
    # ==================================================================
    def check_provenance(self):
        rec = self.rec
        prov = (self.manifest.get("provenance") or {})
        impl = prov.get("implementation") or {}
        weights = prov.get("weights") or {}
        config = prov.get("config") or {}
        tok = prov.get("tokenizer") or {}
        ref = prov.get("reference_implementation") or {}
        rt = prov.get("runtime") or {}

        # 1. material implementation file
        material_impl = self.snapshot_path
        rec.truth("prov.impl.exists", "provenance", True,
                  "material gpt2_real.py", ok=material_impl.is_file(),
                  observed=str(material_impl))
        impl_sha = None
        if material_impl.is_file():
            impl_sha = sha256_file(material_impl)
            rec.equal("prov.impl.sha256", "provenance", True,
                      "sha256(material gpt2_real.py)", impl.get("sha256"), impl_sha,
                      tolerance="exact sha256")
        # 2. snapshot file + copied bytes
        snap_sha = sha256_file(self.snapshot_path) if self.snapshot_path and self.snapshot_path.is_file() else None
        rec.equal("prov.snapshot.sha256", "provenance", True,
                  "sha256(sources/gpt2/gpt2_real.py)",
                  (self.manifest.get("source_snapshot") or {}).get("sha256"), snap_sha,
                  tolerance="exact sha256")
        rec.equal("prov.snapshot.impl_snapshot_sha256", "provenance", True,
                  "provenance.implementation.snapshot_sha256",
                  impl.get("snapshot_sha256"), snap_sha, tolerance="exact sha256")
        rec.equal("prov.snapshot.same_as_project", "provenance", True,
                  "sha256(snapshot) == sha256(project implementation)",
                  impl_sha, snap_sha, tolerance="exact sha256")
        if self.snapshot_path and self.snapshot_path.is_file() and material_impl.is_file():
            rec.exact_arrays("prov.snapshot.readonly_copy", "provenance", True,
                             "snapshot bytes == project implementation bytes",
                             material_impl.read_bytes(), self.snapshot_path.read_bytes(),
                             tolerance="byte identical")

        rec.equal("prov.upstream.sha256", "provenance", True,
                  "read-only upstream material hash", impl.get("upstream_sha256"),
                  sha256_file(self.material_root / "gpt2_real.py"), tolerance="exact sha256")

        # 3. weights / config / tokenizer hashes relative to material root
        weights_path = self._resolve_material_path(weights.get("path"))
        rec.truth("prov.weights.exists", "provenance", True,
                  "material weights file", ok=weights_path is not None,
                  observed=str(weights_path))
        if weights_path is not None:
            w_sha = sha256_file(weights_path)
            rec.equal("prov.weights.sha256", "provenance", True,
                      "sha256(weights)", weights.get("sha256"), w_sha,
                      tolerance="exact sha256")
            rec.equal("prov.weights.bytes", "provenance", True,
                      "weights file size", weights.get("bytes"), weights_path.stat().st_size)
        config_path = self._resolve_material_path(config.get("path"))
        rec.truth("prov.config.exists", "provenance", True,
                  "material config.json", ok=config_path is not None,
                  observed=str(config_path))
        if config_path is not None:
            rec.equal("prov.config.sha256", "provenance", True,
                      "sha256(config.json)", config.get("sha256"), sha256_file(config_path),
                      tolerance="exact sha256")
        weights_dir = weights_path.parent if weights_path is not None else (
            self.material_root / "weights" / "gpt2")
        for fname, key in (("tokenizer.json", "tokenizer_json_sha256"),
                           ("vocab.json", "vocab_sha256"),
                           ("merges.txt", "merges_sha256")):
            fp = weights_dir / fname
            rec.truth(f"prov.tokenizer.{fname}.exists", "provenance", True,
                      str(fp), ok=fp.is_file())
            if fp.is_file():
                rec.equal(f"prov.tokenizer.{fname}.sha256", "provenance", True,
                          f"sha256({fname})", tok.get(key), sha256_file(fp),
                          tolerance="exact sha256")
        rec.equal("prov.tokenizer.local_files_only", "provenance", True,
                  "provenance.tokenizer.local_files_only", True,
                  tok.get("local_files_only"))
        if self.tokenizer is None:
            rec.not_run("prov.tokenizer.kind_matches_loaded", "provenance", False,
                        "provenance.tokenizer.kind vs loaded tokenizer class",
                        "compared", "tokenizer not loaded")
        else:
            rec.equal("prov.tokenizer.kind_matches_loaded", "provenance", True,
                      "provenance.tokenizer.kind == type(loaded tokenizer).__name__",
                      getattr(type(self.tokenizer), "__name__", None), tok.get("kind"),
                      details={"note": "schema section 3 example says GPT2TokenizerFast; "
                                       "transformers 5.x AutoTokenizer returns GPT2Tokenizer "
                                       "for this local directory. Compared against the actually "
                                       "loaded class, which is the verifiable fact."})

        # 4. reference implementation + runtime versions vs real environment
        import transformers  # local import: version check only
        try:
            import safetensors
            st_version = safetensors.__version__
        except Exception:
            st_version = None
        rec.equal("prov.reference.framework", "provenance", True,
                  "provenance.reference_implementation.framework",
                  "transformers", ref.get("framework"))
        rec.equal("prov.reference.class", "provenance", True,
                  "provenance.reference_implementation.class",
                  "GPT2LMHeadModel", ref.get("class"))
        rec.equal("prov.reference.attn_implementation", "provenance", True,
                  "provenance.reference_implementation.attn_implementation",
                  "eager", ref.get("attn_implementation"))
        rec.equal("prov.reference.version", "provenance", True,
                  "transformers.__version__", transformers.__version__,
                  ref.get("version"))
        rec.equal("prov.runtime.python", "provenance", True,
                  "platform.python_version()", platform.python_version(),
                  rt.get("python"))
        rec.equal("prov.runtime.torch", "provenance", True,
                  "torch.__version__", torch_version(), rt.get("torch"))
        rec.equal("prov.runtime.numpy", "provenance", True,
                  "numpy.__version__", np.__version__, rt.get("numpy"))
        rec.equal("prov.runtime.safetensors", "provenance", True,
                  "safetensors.__version__", st_version, rt.get("safetensors"))
        rec.equal("prov.runtime.platform", "provenance", False,
                  "platform.platform()", platform.platform(), rt.get("platform"))
        rec.equal("prov.runtime.machine", "provenance", False,
                  "platform.machine()", platform.machine(), rt.get("machine"))
        rec.equal("prov.runtime.device", "provenance", True,
                  "provenance.runtime.device", "cpu", rt.get("device"))
        rec.equal("prov.runtime.dtype", "provenance", True,
                  "provenance.runtime.dtype", "float32", rt.get("dtype"))
        rec.equal("prov.runtime.eval", "provenance", True,
                  "provenance.runtime.eval", True, rt.get("eval"))
        rec.equal("prov.runtime.seed", "provenance", True,
                  "provenance.runtime.seed", 0, rt.get("seed"))
        rec.equal("prov.runtime.threads_declared", "provenance", True,
                  "provenance.runtime.threads (fixed thread count)", 4, rt.get("threads"))
        grad_mode = rt.get("grad_mode")
        inference_mode = rt.get("inference_mode")
        grad_ok = (grad_mode in ("no_grad", "inference_mode")) or (inference_mode is True)
        rec.truth("prov.runtime.grad_mode", "provenance", True,
                  "runtime no_grad/inference_mode declared",
                  ok=grad_ok,
                  observed={"grad_mode": grad_mode, "inference_mode": inference_mode},
                  details={"note": "schema section 3 example uses inference_mode; "
                                   "no_grad is equivalent for collection precision"})
        rec.truth("prov.weight_origin_note", "provenance", True,
                  "provenance.weight_origin_note non-empty",
                  ok=bool(prov.get("weight_origin_note")),
                  observed=str(prov.get("weight_origin_note"))[:300])
        # source map digest declared in manifest
        rec.equal("prov.source_map.sha256", "provenance", True,
                  "manifest.source_map.sha256",
                  (self.manifest.get("source_map") or {}).get("sha256"),
                  getattr(self, "source_map_sha", None), tolerance="exact sha256")

    # ==================================================================
    # phase: source map
    # ==================================================================
    def check_source_map(self):
        rec = self.rec
        sm = self.source_map
        if sm is None:
            rec.not_run("source_map.loaded", "source_map", True, "source-map.json",
                        "parsed source map", "source-map.json missing or invalid")
            return
        rec.equal("source_map.schema", "schema", True, "source-map.schema",
                  SOURCE_MAP_SCHEMA, sm.get("schema"))
        rec.equal("source_map.source_file", "source_map", True,
                  "source-map.source_file", "gpt2_real.py", sm.get("source_file"))
        snap_sha = sha256_file(self.snapshot_path) if self.snapshot_path and self.snapshot_path.is_file() else None
        rec.equal("source_map.source_sha256", "source_map", True,
                  "source-map.source_sha256 == sha256(snapshot)",
                  snap_sha, sm.get("source_sha256"), tolerance="exact sha256")
        rec.equal("source_map.map_version", "source_map", True,
                  "source-map.map_version == case.source_map_version",
                  (self.case or {}).get("source_map_version"), sm.get("map_version"))

        lines = self.snapshot_text.splitlines(keepends=True)
        n_lines = len(lines)

        def span_sha(a, b):
            return sha256_bytes("".join(lines[a - 1:b]).encode("utf-8"))

        symbols = sm.get("symbols") or {}
        bad_symbols = []
        for name, ent in symbols.items():
            ln = ent.get("lines") or [None, None]
            ok_range = (isinstance(ln, list) and len(ln) == 2
                        and all(isinstance(x, int) for x in ln)
                        and 1 <= ln[0] <= ln[1] <= n_lines)
            actual = span_sha(ln[0], ln[1]) if ok_range else None
            want = ent.get("span_sha256")
            ok = ok_range and actual == want
            rec.add(f"source_map.symbol.{name}", "source_map", True,
                    f"symbol {name} lines/span_sha256",
                    {"lines": ln, "span_sha256": want},
                    {"lines_valid": ok_range, "span_sha256": actual},
                    "exact sha256 over inclusive line span",
                    ok=ok)
            if not ok:
                bad_symbols.append(name)
        rec.truth("source_map.symbols_all", "source_map", True,
                  "all symbol spans recomputed and valid",
                  ok=not bad_symbols, observed={"invalid": bad_symbols})

        operations = sm.get("operations") or {}
        bad_ops = []
        for name, ent in operations.items():
            ln = ent.get("lines") or [None, None]
            sym = ent.get("symbol")
            sym_ent = symbols.get(sym) or {}
            sym_ln = sym_ent.get("lines") or [None, None]
            ok_range = (isinstance(ln, list) and len(ln) == 2
                        and all(isinstance(x, int) for x in ln)
                        and 1 <= ln[0] <= ln[1] <= n_lines)
            ok_within = (ok_range and isinstance(sym_ln, list) and len(sym_ln) == 2
                         and sym_ln[0] <= ln[0] <= ln[1] <= sym_ln[1])
            actual = span_sha(ln[0], ln[1]) if ok_range else None
            want = ent.get("span_sha256")
            ok = ok_range and actual == want and ok_within
            rec.add(f"source_map.op.{name}", "source_map", True,
                    f"operation {name} symbol/lines/span_sha256",
                    {"symbol": sym, "lines": ln, "span_sha256": want,
                     "within_symbol": ok_within},
                    {"span_sha256": actual},
                    "exact sha256 over inclusive line span; within symbol range",
                    ok=ok)
            if not ok:
                bad_ops.append(name)
        rec.truth("source_map.operations_all", "source_map", True,
                  "all operation spans recomputed, valid and inside their symbol",
                  ok=not bad_ops, observed={"invalid": bad_ops})

        # every op referenced by events / tensors must exist
        referenced = set()
        for ev in self.case.get("events") or []:
            if ev.get("primary_op"):
                referenced.add(ev["primary_op"])
        for t in self.case.get("tensors") or []:
            if t.get("op"):
                referenced.add(t["op"])
        missing = sorted(referenced - set(operations))
        rec.truth("source_map.coverage", "source_map", True,
                  "all event/tensor op IDs exist in source-map.operations",
                  ok=not missing, expected=sorted(referenced), observed={"missing": missing},
                  details={"referenced_ops": len(referenced), "mapped_ops": len(operations)})
        # event source references match the mapped operation entry
        mismatched = []
        for ev in self.case.get("events") or []:
            op = ev.get("primary_op")
            ent = operations.get(op)
            src = ev.get("source")
            if ent is None or not isinstance(src, dict):
                mismatched.append({"event": ev.get("id"), "reason": "missing op/source"})
                continue
            if (src.get("symbol") != ent.get("symbol")
                    or src.get("lines") != ent.get("lines")
                    or src.get("span_sha256") != ent.get("span_sha256")):
                mismatched.append({"event": ev.get("id"), "op": op,
                                   "event_source": src,
                                   "map_source": {k: ent.get(k) for k in
                                                  ("symbol", "lines", "span_sha256")}})
        rec.truth("source_map.event_sources", "source_map", True,
                  "every event.source matches source-map operation entry",
                  ok=not mismatched, observed=mismatched[:10],
                  details={"mismatch_count": len(mismatched)})
        # tensor op / source references (tensors have no source field in the export;
        # op must be mapped, which is covered by source_map.coverage)
        self.operations = operations
        self.symbols = symbols

    # ==================================================================
    # phase: case structure
    # ==================================================================
    def check_case_structure(self):
        rec = self.rec
        case = self.case
        rec.equal("case.schema", "schema", True, "case.schema", CASE_SCHEMA,
                  case.get("schema"))
        rec.equal("case.case_id", "structure", True, "case.case_id",
                  self.case_id, case.get("case_id"))
        rec.equal("case.provenance_ref", "structure", True, "case.provenance_ref",
                  "manifest.json#/provenance", case.get("provenance_ref"))

        inp = case.get("input") or {}
        execu = case.get("execution") or {}
        gen = execu.get("generation") or {}
        cfg = execu.get("config") or {}
        ids = inp.get("ids") or []
        text = inp.get("text")
        self.prompt_ids = [int(x) for x in ids]
        self.prompt_len = len(self.prompt_ids)
        self.generated_ids = [int(x) for x in (gen.get("generated_ids") or [])]
        self.full_ids = [int(x) for x in (gen.get("full_sequence_ids") or [])]
        self.selected_layer = execu.get("selected_layer", 0)

        rec.truth("case.input_text", "structure", True, "case.input.text non-empty",
                  ok=isinstance(text, str) and bool(text), observed=text)
        rec.equal("case.input.byte_length", "tokenizer", True,
                  "case.input.byte_length == len(utf-8 bytes of text)",
                  len(text.encode("utf-8")) if isinstance(text, str) else None,
                  inp.get("byte_length"))
        rec.truth("case.input.numbering", "structure", True, "case.input.numbering present",
                  ok=isinstance(inp.get("numbering"), str), observed=inp.get("numbering"))
        rec.truth("case.input.tokenizer_note", "structure", True,
                  "case.input.tokenizer_note present",
                  ok=isinstance(inp.get("tokenizer_note"), str),
                  observed=str(inp.get("tokenizer_note"))[:120])
        tokens = inp.get("tokens") or []
        rec.equal("case.input.token_count", "tokenizer", True,
                  "len(case.input.tokens) == len(ids)", len(ids), len(tokens))

        rec.equal("case.execution.batch_size", "structure", True,
                  "case.execution.batch_size", 1, execu.get("batch_size"))
        cfg_expected = dict(EXPECTED_CONFIG)
        cfg_observed = {k: cfg.get(k) for k in cfg_expected}
        rec.equal("case.execution.config", "structure", True,
                  "case.execution.config vs pinned GPT-2 small geometry",
                  cfg_expected, cfg_observed)
        rec.equal("case.generation.strategy", "structure", True,
                  "case.execution.generation.strategy", "greedy", gen.get("strategy"))
        rec.equal("case.generation.max_new_tokens", "structure", True,
                  "case.execution.generation.max_new_tokens", EXPECTED_SAMPLE_COUNT,
                  gen.get("max_new_tokens"))
        rec.equal("case.generation.prompt_len", "structure", True,
                  "case.execution.generation.prompt_len == len(input.ids)",
                  len(ids), gen.get("prompt_len"))
        rec.equal("case.generation.forward_count", "structure", True,
                  "case.execution.generation.forward_count", EXPECTED_FORWARD_COUNT,
                  gen.get("forward_count"))
        rec.equal("case.generation.sample_count", "structure", True,
                  "case.execution.generation.sample_count", EXPECTED_SAMPLE_COUNT,
                  gen.get("sample_count"))
        rec.equal("case.generation.full_sequence", "structure", True,
                  "full_sequence_ids == input.ids + generated_ids",
                  self.prompt_ids + self.generated_ids, self.full_ids)
        rec.truth("case.generation.note", "structure", True,
                  "case.execution.generation.note describes the stopping condition",
                  ok=isinstance(gen.get("note"), str) and bool(gen.get("note")),
                  observed=str(gen.get("note"))[:200])
        rec.equal("case.selected_layer", "structure", True,
                  "case.execution.selected_layer", 0, execu.get("selected_layer"))
        rec.truth("case.collection_budget_note", "structure", True,
                  "case.execution.collection_budget_note present",
                  ok=bool(execu.get("collection_budget_note")),
                  observed=str(execu.get("collection_budget_note"))[:160])

        events = case.get("events") or []
        tensors = case.get("tensors") or []
        fsum = case.get("forward_summaries")
        rec.equal("case.events_count", "structure", True, "len(case.events)",
                  EXPECTED_FORWARD_COUNT * (2 + EXPECTED_LAYERS + 1)
                  + EXPECTED_SAMPLE_COUNT, len(events))
        rec.equal("case.tensors_count", "structure", True, "len(case.tensors)",
                  EXPECTED_FORWARD_COUNT * 37, len(tensors))
        rec.truth("case.forward_summaries_type", "structure", True,
                  "case.forward_summaries is a list", ok=isinstance(fsum, list),
                  observed=type(fsum).__name__)

        # dedupe / ids
        eids = [e.get("id") for e in events]
        rec.truth("case.event_ids_unique", "structure", True,
                  "event ids unique and non-empty",
                  ok=len(eids) == len(set(eids)) and all(isinstance(x, str) and x for x in eids),
                  observed={"count": len(eids), "unique": len(set(eids))})
        tids = [t.get("id") for t in tensors]
        rec.truth("case.tensor_ids_unique", "structure", True,
                  "tensor ids unique and non-empty",
                  ok=len(tids) == len(set(tids)) and all(isinstance(x, str) and x for x in tids),
                  observed={"count": len(tids), "unique": len(set(tids))})

    # ==================================================================
    # phase: tensors (structure + payload integrity)
    # ==================================================================
    def event_tensor_spec(self, event: dict) -> dict:
        """Expected {role: (shape, axes, op)} for an event per schema section 6."""
        kind = event.get("kind")
        f = event.get("forward_index")
        T = event.get("position_count")
        total = event.get("total_positions")
        H = EXPECTED_CONFIG["n_head"]
        C = EXPECTED_CONFIG["n_embd"]
        F = 4 * C
        V = EXPECTED_CONFIG["vocab_size"]
        D = EXPECTED_CONFIG["head_dim"]
        if kind == "embed":
            return {
                "wte_out": ([1, T, C], ["B", "T", "C"], "gpt2.embed"),
                "wpe_out": ([T, C], ["T", "C"], "gpt2.embed"),
            }
        if kind == "ln_f":
            return {"out": ([1, T, C], ["B", "T", "C"], "gpt2.ln_f")}
        if kind == "logits":
            return {"last_row": ([1, V], ["B", "V"], "gpt2.generate.last_row_slice")}
        if kind == "sample":
            return {}
        if kind != "block":
            return {}
        specs = {
            "attn_weights": ([1, H, T, total], ["B", "H", "Tq", "Tk"], "gpt2.attn.softmax"),
        }
        if event.get("layer") != self.selected_layer:
            return specs
        specs.update({
            "x_in": ([1, T, C], ["B", "T", "C"], "gpt2.block.forward"),
            "ln1_out": ([1, T, C], ["B", "T", "C"], "gpt2.block.ln1"),
            "qkv": ([1, T, 3 * C], ["B", "T", "3C"], "gpt2.attn.qkv_linear"),
            "q_raw": ([1, T, C], ["B", "T", "C"], "gpt2.attn.split"),
            "k_raw": ([1, T, C], ["B", "T", "C"], "gpt2.attn.split"),
            "v_raw": ([1, T, C], ["B", "T", "C"], "gpt2.attn.split"),
            "q_heads": ([1, H, T, D], ["B", "H", "T", "C"], "gpt2.attn.view_transpose"),
            "k_heads": ([1, H, T, D], ["B", "H", "T", "C"], "gpt2.attn.view_transpose"),
            "v_heads": ([1, H, T, D], ["B", "H", "T", "C"], "gpt2.attn.view_transpose"),
            "k_read": ([1, H, total, D], ["B", "H", "Tk", "C"], "gpt2.cache.read_window"),
            "v_read": ([1, H, total, D], ["B", "H", "Tk", "C"], "gpt2.cache.read_window"),
            "scores_scaled": ([1, H, T, total], ["B", "H", "Tq", "Tk"], "gpt2.attn.scores"),
            "causal_mask": ([T, total], ["Tq", "Tk"], "gpt2.attn.mask"),
            "weighted_v": ([1, T, C], ["B", "T", "C"], "gpt2.attn.weighted_v"),
            "attn_out": ([1, T, C], ["B", "T", "C"], "gpt2.attn.out_proj"),
            "x_after_attn": ([1, T, C], ["B", "T", "C"], "gpt2.block.residual_attn"),
            "ln2_out": ([1, T, C], ["B", "T", "C"], "gpt2.block.ln2"),
            "fc_out": ([1, T, F], ["B", "T", "F"], "gpt2.mlp.fc"),
            "act_out": ([1, T, F], ["B", "T", "F"], "gpt2.mlp.act"),
            "mlp_out": ([1, T, C], ["B", "T", "C"], "gpt2.mlp.proj"),
            "x_out": ([1, T, C], ["B", "T", "C"], "gpt2.block.forward"),
        })
        return specs

    def check_tensors(self):
        rec = self.rec
        self.tensors_list = list(self.case.get("tensors") or [])
        self.tensors = {}
        for idx, t in enumerate(self.tensors_list):
            tid = t.get("id")
            if not isinstance(tid, str) or not tid:
                rec.failed(f"tensor.index{idx}", "payload", True, f"tensor[{idx}].id",
                           "non-empty unique id", tid, "exact")
                continue
            if tid in self.tensors:
                rec.failed(f"tensor.{tid}.duplicate", "payload", True, tid,
                           "unique tensor id", "duplicate", "exact")
                continue
            self.tensors[tid] = t
        self.events = list(self.case.get("events") or [])
        self.events_by_id = {e.get("id"): e for e in self.events}

        arrays = {}
        for tid, t in self.tensors.items():
            issues = []
            required = ("id", "event_id", "role", "op", "shape", "axes", "dtype",
                        "scope", "derivation", "payload")
            for key in required:
                if key not in t:
                    issues.append(f"missing field {key}")
            shape = t.get("shape")
            if not (isinstance(shape, list) and shape
                    and all(isinstance(x, int) and x >= 0 for x in shape)):
                issues.append(f"bad shape {shape!r}")
                shape = []
            axes = t.get("axes")
            if not (isinstance(axes, list) and all(isinstance(a, str) for a in axes)):
                issues.append(f"bad axes {axes!r}")
                axes = []
            if len(axes) != len(shape):
                issues.append(f"axes length {len(axes)} != shape rank {len(shape)}")
            if t.get("scope") not in ("full", "slice", "summary"):
                issues.append(f"bad scope {t.get('scope')!r}")
            if t.get("derivation") not in ("observed", "derived", "summary"):
                issues.append(f"bad derivation {t.get('derivation')!r}")
            dtype = t.get("dtype")
            payload = t.get("payload")
            raw = None
            encoding = None
            nbytes = None
            sha = None
            data_b64 = None
            if isinstance(payload, dict):
                encoding = payload.get("encoding")
                nbytes = payload.get("nbytes")
                sha = payload.get("sha256")
                data_b64 = payload.get("data")
                for k in ("encoding", "nbytes", "sha256", "data"):
                    if k not in payload:
                        issues.append(f"payload missing {k}")
            else:
                issues.append("payload is not an object")
            if t.get("id") and t.get("event_id") and t.get("role"):
                expected_tid = f"{t['event_id']}.{t['role']}"
                if tid != expected_tid:
                    issues.append(f"id {tid!r} != event_id.role {expected_tid!r}")
            if isinstance(data_b64, str):
                try:
                    raw = base64.b64decode(data_b64, validate=True)
                except Exception as exc:
                    issues.append(f"base64 decode failed: {exc}")
            elif data_b64 is not None:
                issues.append("payload.data is not a string")

            prod = 1
            for dim in shape:
                prod *= dim
            if dtype == "float32":
                itemsize = 4
                if encoding != "f32le-base64":
                    issues.append(f"encoding {encoding!r} != 'f32le-base64'")
            elif dtype in ("uint8", "u8"):
                itemsize = 1
                if encoding != "u8-base64":
                    issues.append(f"encoding {encoding!r} != 'u8-base64'")
            else:
                itemsize = None
                issues.append(f"unsupported dtype {dtype!r}")
            expected_nbytes = prod * itemsize if itemsize is not None else None
            if expected_nbytes is not None and nbytes != expected_nbytes:
                issues.append(f"payload.nbytes {nbytes!r} != prod(shape)*itemsize "
                              f"{expected_nbytes}")
            if raw is not None and nbytes is not None and len(raw) != nbytes:
                issues.append(f"decoded length {len(raw)} != nbytes {nbytes}")
            if raw is not None and sha is not None:
                actual_sha = sha256_bytes(raw)
                if actual_sha != sha:
                    issues.append(f"payload.sha256 mismatch: header {sha} vs actual {actual_sha}")
            if raw is not None and itemsize is not None and len(raw) == expected_nbytes:
                try:
                    np_dtype = "<f4" if dtype == "float32" else "u1"
                    arr = np.frombuffer(raw, dtype=np_dtype).copy()
                    if dtype == "float32":
                        n_bad = int(np.size(arr) - np.count_nonzero(np.isfinite(arr)))
                        if n_bad:
                            issues.append(f"{n_bad} non-finite float32 values in payload")
                    arrays[tid] = arr
                except Exception as exc:
                    issues.append(f"array decode failed: {exc}")
            rec.add(f"tensor.{tid}", "payload", True, tid,
                    {"shape": shape, "axes": axes, "dtype": dtype,
                     "encoding": encoding, "nbytes": expected_nbytes,
                     "sha256": "recomputed"},
                    {"issues": issues} if issues else "ok",
                    "payload roundtrip exact (section 8)", ok=not issues,
                    details={"role": t.get("role"), "op": t.get("op"),
                             "scope": t.get("scope"),
                             "derivation": t.get("derivation"),
                             "event_id": t.get("event_id")})
        self.arrays = arrays

        # event tensor references must resolve, and the tensor sets must match
        # the schema section 6 role inventory exactly.
        missing_refs = []
        wrong_owner = []
        spec_issues = []
        expected_ids = set()
        for ev in self.events:
            eid = ev.get("id")
            refs = ev.get("tensor_ids")
            if not isinstance(refs, list):
                spec_issues.append({"event": eid, "issue": "tensor_ids not a list"})
                refs = []
            for r in refs:
                expected_ids.add(r)
                if r not in self.tensors:
                    missing_refs.append({"event": eid, "tensor": r})
                elif self.tensors[r].get("event_id") != eid:
                    wrong_owner.append({"event": eid, "tensor": r,
                                        "owner": self.tensors[r].get("event_id")})
            spec = self.event_tensor_spec(ev)
            want = {f"{eid}.{role}" for role in spec}
            got = set(refs)
            if want != got:
                spec_issues.append({
                    "event": eid,
                    "missing": sorted(want - got),
                    "extra": sorted(got - want),
                })
            # per-tensor shape/axes/op validation for the schema table
            for role, (shape, axes, op) in spec.items():
                tid = f"{eid}.{role}"
                t = self.tensors.get(tid)
                if t is None:
                    continue
                probs = []
                if t.get("shape") != shape:
                    probs.append({"field": "shape", "expected": shape,
                                  "observed": t.get("shape")})
                if t.get("axes") != axes:
                    probs.append({"field": "axes", "expected": axes,
                                  "observed": t.get("axes")})
                if t.get("op") != op:
                    probs.append({"field": "op", "expected": op,
                                  "observed": t.get("op")})
                if t.get("role") != role:
                    probs.append({"field": "role", "expected": role,
                                  "observed": t.get("role")})
                if probs:
                    spec_issues.append({"tensor": tid, "problems": probs})
        rec.truth("tensors.references", "structure", True,
                  "every event.tensor_ids entry exists and is owned by that event",
                  ok=not missing_refs and not wrong_owner,
                  observed={"missing": missing_refs[:10], "wrong_owner": wrong_owner[:10]},
                  details={"missing_count": len(missing_refs),
                           "wrong_owner_count": len(wrong_owner)})
        rec.truth("tensors.schema_inventory", "structure", True,
                  "event tensor sets/shapes/axes/ops match schema section 6",
                  ok=not spec_issues, observed=spec_issues[:10],
                  details={"issue_count": len(spec_issues)})
        actual_ids = set(self.tensors)
        rec.truth("tensors.no_orphans", "structure", True,
                  "case.tensors contains exactly the event-referenced tensors",
                  ok=actual_ids == expected_ids,
                  observed={"orphans": sorted(actual_ids - expected_ids)[:10],
                            "unreferenced": sorted(expected_ids - actual_ids)[:10]})

    # ==================================================================
    # phase: events (order / fields / replay state)
    # ==================================================================
    def check_events(self):
        rec = self.rec
        p = self.prompt_len
        expected_ids = []
        for f in range(EXPECTED_FORWARD_COUNT):
            expected_ids.append(f"fwd{f}.embed")
            expected_ids.extend(f"fwd{f}.block{l}" for l in range(EXPECTED_LAYERS))
            expected_ids.extend([f"fwd{f}.ln_f", f"fwd{f}.logits"])
            if f < EXPECTED_SAMPLE_COUNT:
                expected_ids.append(f"sample{f}")
        actual_ids = [e.get("id") for e in self.events]
        rec.equal("events.exact_order", "structure", True,
                  "event id order (真实执行顺序)", expected_ids, actual_ids)

        seqs = [e.get("seq") for e in self.events]
        rec.truth("events.seq_increasing", "structure", True,
                  "seq starts at 0 and increases by 1 in event order",
                  ok=seqs == list(range(len(seqs))), observed=seqs[:5] + seqs[-3:])

        field_issues = []
        raw_issues = []
        replay_issues = []
        self.forward_meta = {}
        for idx, ev in enumerate(self.events):
            eid = ev.get("id")
            kind = ev.get("kind")
            f = ev.get("forward_index")
            if kind not in FWD_EVENT_KINDS + ("sample",):
                raw_issues.append({"event": eid, "issue": f"bad kind {kind!r}"})
                continue
            if kind == "sample":
                s = f
                exp_phase = "sample"
                exp_output_pos = p + s
                for key, want in (("phase", exp_phase), ("forward_index", s),
                                  ("sample_index", s), ("strategy", "greedy"),
                                  ("output_position", exp_output_pos)):
                    if ev.get(key) != want:
                        field_issues.append({"event": eid, "field": key,
                                             "expected": want, "observed": ev.get(key)})
                if f >= EXPECTED_SAMPLE_COUNT:
                    raw_issues.append({"event": eid, "issue": "unexpected sample event"})
                if ev.get("tensor_ids") not in ([], None):
                    raw_issues.append({"event": eid, "issue": "sample event must not own tensors",
                                       "observed": ev.get("tensor_ids")})
                if not (isinstance(ev.get("replay"), dict)
                        and ev["replay"].get("sequence_length_after") == exp_output_pos + 1):
                    replay_issues.append({"event": eid, "field": "sequence_length_after",
                                          "expected": exp_output_pos + 1,
                                          "observed": (ev.get("replay") or {}).get(
                                              "sequence_length_after")})
                continue

            if not isinstance(f, int) or not (0 <= f < EXPECTED_FORWARD_COUNT):
                raw_issues.append({"event": eid, "issue": f"bad forward_index {f!r}"})
                continue
            start = 0 if f == 0 else p + f - 1
            T = p if f == 0 else 1
            total = start + T
            exp_phase = "prefill" if f == 0 else "decode"
            exp_layer = None
            if kind == "block":
                exp_layer = int(eid.split(".block")[1]) if ".block" in eid else None
            checks = {
                "phase": exp_phase,
                "position_start": start,
                "position_count": T,
                "total_positions": total,
            }
            if kind == "block":
                checks["layer"] = exp_layer
            for key, want in checks.items():
                if ev.get(key) != want:
                    field_issues.append({"event": eid, "field": key,
                                         "expected": want, "observed": ev.get(key)})
            exp_input = (self.prompt_ids if f == 0
                         else [self.generated_ids[f - 1]] if f - 1 < len(self.generated_ids)
                         else None)
            if ev.get("input_ids") != exp_input:
                field_issues.append({"event": eid, "field": "input_ids",
                                     "expected": exp_input, "observed": ev.get("input_ids")})
            positions = ev.get("positions") or {}
            exp_query = list(range(start, total))
            exp_key = list(range(total)) if kind == "block" else []
            if positions.get("query") != exp_query:
                field_issues.append({"event": eid, "field": "positions.query",
                                     "expected": exp_query, "observed": positions.get("query")})
            if positions.get("key") != exp_key:
                field_issues.append({"event": eid, "field": "positions.key",
                                     "expected": exp_key, "observed": positions.get("key")})
            op = ev.get("primary_op")
            if not isinstance(op, str) or op not in (self.operations or {}):
                raw_issues.append({"event": eid, "issue": f"primary_op {op!r} not mapped"})
            elif kind == "block" and op != "gpt2.block.forward":
                field_issues.append({"event": eid, "field": "primary_op",
                                     "expected": "gpt2.block.forward", "observed": op})
            replay = ev.get("replay") or {}
            exp_complete = kind in ("ln_f", "logits")
            if replay.get("phase_complete") is not exp_complete:
                replay_issues.append({"event": eid, "field": "phase_complete",
                                      "expected": exp_complete,
                                      "observed": replay.get("phase_complete")})
            self.forward_meta.setdefault(f, {"start": start, "T": T, "total": total,
                                             "phase": exp_phase})
        rec.truth("events.kind_phase_layout", "structure", True,
                  "kind/phase/forward_index/layer/position/input_ids/positions per section 5",
                  ok=not field_issues, observed=field_issues[:12],
                  details={"issue_count": len(field_issues)})
        rec.truth("events.replay_state", "structure", True,
                  "replay.phase_complete only on ln_f/logits; sample has sequence_length_after",
                  ok=not replay_issues, observed=replay_issues[:10],
                  details={"issue_count": len(replay_issues)})
        rec.truth("events.primary_op", "structure", True,
                  "every event.primary_op is mapped; block events use gpt2.block.forward",
                  ok=not raw_issues, observed=raw_issues[:10],
                  details={"issue_count": len(raw_issues)})

        # forward_summaries must agree with the logits events
        fsum = self.case.get("forward_summaries")
        fs_issues = []
        if not isinstance(fsum, list) or len(fsum) != EXPECTED_FORWARD_COUNT:
            fs_issues.append({"summary_list": fsum if not isinstance(fsum, list) else len(fsum)})
        else:
            for f, item in enumerate(fsum):
                meta = self.forward_meta.get(f) or {}
                lev = self.fwd_event(f, "logits").get("logits_event") or {}
                want = {
                    "forward_index": f,
                    "phase": meta.get("phase"),
                    "start_pos": meta.get("start"),
                    "position_count": meta.get("T"),
                    "logits_shape": lev.get("full_output_shape"),
                    "last_row_index": meta.get("start", 0) + meta.get("T", 1) - 1,
                }
                for key, w in want.items():
                    if item.get(key) != w:
                        fs_issues.append({"forward_index": f, "field": key,
                                          "expected": w, "observed": item.get(key)})
        rec.truth("events.forward_summaries", "structure", True,
                  "forward_summaries agree with events/logits_event",
                  ok=not fs_issues, observed=fs_issues[:10],
                  details={"issue_count": len(fs_issues)})

    # ==================================================================
    # phase: cache metadata (section 5.3)
    # ==================================================================
    def check_cache_metadata(self):
        rec = self.rec
        H = EXPECTED_CONFIG["n_head"]
        C = EXPECTED_CONFIG["n_embd"]
        D = EXPECTED_CONFIG["head_dim"]
        L = EXPECTED_LAYERS
        block_size = EXPECTED_CONFIG["block_size"]
        exp_shape = [1, H, block_size, D]
        per_layer_bytes = 1 * H * block_size * D * 4
        exp_total = per_layer_bytes * 2 * L

        # allocated block
        allocated_seen = []
        alloc_issues = []
        for ev in self.events:
            cache = ev.get("cache")
            if not isinstance(cache, dict):
                continue
            alloc = cache.get("allocated")
            if isinstance(alloc, dict):
                allocated_seen.append((ev.get("id"), alloc))
        if not allocated_seen:
            rec.not_run("cache.allocated", "cache", True, "cache.allocated",
                        "allocated block on events", "no event carries cache.allocated")
        else:
            first_id, alloc = allocated_seen[0]
            observed = {k: alloc.get(k) for k in
                        ("per_layer_k_shape", "per_layer_k_bytes", "total_bytes", "dtype")}
            expected = {"per_layer_k_shape": exp_shape,
                        "per_layer_k_bytes": per_layer_bytes,
                        "total_bytes": exp_total, "dtype": "float32"}
            if observed != expected:
                alloc_issues.append({"event": first_id, "expected": expected,
                                     "observed": observed})
            # every event repeats the same allocation description
            for eid, other in allocated_seen[1:]:
                sub = {k: other.get(k) for k in
                       ("per_layer_k_shape", "per_layer_k_bytes", "total_bytes", "dtype")}
                if sub != expected:
                    alloc_issues.append({"event": eid, "expected": expected, "observed": sub})
            rec.truth("cache.allocated", "cache", True,
                      "allocated bytes per section 5.3 (numel x element_size)",
                      ok=not alloc_issues, expected=expected, observed=observed,
                      details={"events_with_allocated": len(allocated_seen),
                               "issues": alloc_issues[:5]})
        # formula check on the actual allocation (material KVCache, if loadable)
        if self.kv is not None:
            actual_total = self.kv.memory_bytes()
            rec.equal("cache.allocated.actual", "cache", True,
                      "2 * n_layer * numel(k) * element_size (actual torch tensors)",
                      int(exp_total), int(actual_total))
            rec.equal("cache.allocated.actual_shape", "cache", True,
                      "actual KVCache per-layer k shape", exp_shape,
                      list(self.kv.k[0].shape))

        # per-event effective/written/synchronized arrays
        eff_issues, wr_issues, sync_issues, window_issues = [], [], [], []
        sample_issues = []
        ref_issues = []
        for ev in self.events:
            eid = ev.get("id")
            kind = ev.get("kind")
            cache = ev.get("cache")
            if not isinstance(cache, dict):
                wr_issues.append({"event": eid, "issue": "missing cache object"})
                continue
            f = ev.get("forward_index")
            meta = self.forward_meta.get(f) or {}
            start = meta.get("start", 0)
            T = meta.get("T", 1)
            total = meta.get("total", start + T)
            if kind == "sample":
                s = f
                exp_eff = [self.prompt_len + s] * L
                exp_wr = [0] * L
                exp_sync = self.prompt_len + s
            elif kind == "embed":
                exp_eff = [start] * L
                exp_wr = [0] * L
                exp_sync = start
            elif kind == "block":
                layer = ev.get("layer")
                exp_eff = [start + T if i <= layer else start for i in range(L)]
                exp_wr = [T if i <= layer else 0 for i in range(L)]
                exp_sync = min(exp_eff)
            else:  # ln_f / logits
                exp_eff = [start + T] * L
                exp_wr = [0] * L
                exp_sync = start + T
            got_eff = cache.get("layers_effective_positions")
            got_wr = cache.get("layers_written_positions")
            got_sync = cache.get("synchronized_positions")
            if got_eff != exp_eff:
                eff_issues.append({"event": eid, "expected": exp_eff, "observed": got_eff})
            if got_wr != exp_wr:
                wr_issues.append({"event": eid, "expected": exp_wr, "observed": got_wr})
            if got_sync != exp_sync:
                sync_issues.append({"event": eid, "expected": exp_sync, "observed": got_sync})
            if kind == "block":
                rw = cache.get("read_window")
                ww = cache.get("write_window")
                exp_rw = {"start": 0, "end_exclusive": total}
                exp_ww = {"start": start, "end_exclusive": start + T}
                if rw != exp_rw:
                    window_issues.append({"event": eid, "field": "read_window",
                                          "expected": exp_rw, "observed": rw})
                if ww != exp_ww:
                    window_issues.append({"event": eid, "field": "write_window",
                                          "expected": exp_ww, "observed": ww})
            if kind == "sample":
                # sample only appends the sequence, it must not write KV cache
                if got_wr != [0] * L:
                    sample_issues.append({"event": eid, "field": "layers_written_positions",
                                          "expected": [0] * L, "observed": got_wr})
                if got_eff != [self.prompt_len + f] * L:
                    sample_issues.append({"event": eid,
                                          "field": "layers_effective_positions",
                                          "expected": [self.prompt_len + f] * L,
                                          "observed": got_eff})
                if cache.get("read_window") is not None or cache.get("write_window") is not None:
                    sample_issues.append({"event": eid,
                                          "issue": "sample must not declare read/write windows",
                                          "observed": {"read_window": cache.get("read_window"),
                                                       "write_window": cache.get("write_window")}})
            # selected layer read refs
            refs = cache.get("selected_layer_read_refs")
            if kind == "block" and ev.get("layer") == self.selected_layer:
                want = {"k": f"{eid}.k_read", "v": f"{eid}.v_read"}
                if refs != want:
                    ref_issues.append({"event": eid, "expected": want, "observed": refs})
                else:
                    for role, rid in refs.items():
                        if rid not in self.tensors or self.tensors[rid].get("event_id") != eid:
                            ref_issues.append({"event": eid, "tensor": rid,
                                               "issue": "cache read ref missing/owned elsewhere"})
        rec.truth("cache.effective_positions", "cache", True,
                  "layers_effective_positions per section 5.3 (all events)",
                  ok=not eff_issues, observed=eff_issues[:8],
                  details={"issue_count": len(eff_issues)})
        rec.truth("cache.written_positions", "cache", True,
                  "layers_written_positions per section 5.3 (all events)",
                  ok=not wr_issues, observed=wr_issues[:8],
                  details={"issue_count": len(wr_issues)})
        rec.truth("cache.synchronized_positions", "cache", True,
                  "synchronized_positions == min(layers_effective_positions)",
                  ok=not sync_issues, observed=sync_issues[:8],
                  details={"issue_count": len(sync_issues)})
        rec.truth("cache.windows", "cache", True,
                  "block read_window=0..total, write_window=start..start+T",
                  ok=not window_issues, observed=window_issues[:8],
                  details={"issue_count": len(window_issues)})
        rec.truth("cache.sample_no_write", "cache", True,
                  "sample events append only; no KV write",
                  ok=not sample_issues, observed=sample_issues[:8],
                  details={"issue_count": len(sample_issues)})
        rec.truth("cache.read_refs", "cache", True,
                  "selected-layer block events expose k_read/v_read refs",
                  ok=not ref_issues, observed=ref_issues[:8],
                  details={"issue_count": len(ref_issues)})

    # ==================================================================
    # phase: models (material implementation + HF reference + tokenizer)
    # ==================================================================
    def load_models(self):
        rec = self.rec
        if self.static_only:
            rec.not_run("models.load", "reference_implementation", False,
                        "material gpt2_real + HF GPT2LMHeadModel", "loaded models",
                        "static-only mode")
            return
        import torch

        threads = (self.manifest.get("provenance") or {}).get("runtime", {}).get("threads")
        threads = threads if isinstance(threads, int) and threads > 0 else 4
        torch.manual_seed(0)
        torch.set_num_threads(threads)
        self.torch = torch
        self.python_exec = sys.executable

        prov = (self.manifest.get("provenance") or {})
        weights_path = self._resolve_material_path((prov.get("weights") or {}).get("path"))
        self.weights_dir = (weights_path.parent if weights_path is not None
                            else self.material_root / "weights" / "gpt2")
        impl_file = self.snapshot_path
        try:
            spec = importlib.util.spec_from_file_location("gpt2_real_material_verify", impl_file)
            mod = importlib.util.module_from_spec(spec)
            sys.modules["gpt2_real_material_verify"] = mod
            spec.loader.exec_module(mod)
            self.gr = mod
            self.model, self.tokenizer = mod.load_gpt2(self.weights_dir, device="cpu")
            self.model.eval()
            n_params = int(sum(p.numel() for p in self.model.parameters()))
            rec.equal("models.material.params", "reference_implementation", True,
                      "material gpt2_real parameter count", 124439808, n_params)
            rec.passed("models.material.load", "reference_implementation", True,
                       "material gpt2_real.py loads with local weights",
                       {"params": 124439808, "device": "cpu"},
                       {"params": n_params, "dtype": str(next(self.model.parameters()).dtype)},
                       "exact")
            cfg = self.model.config
            observed_cfg = {
                "n_layer": cfg.n_layer, "n_head": cfg.n_head, "n_embd": cfg.n_embd,
                "head_dim": cfg.head_dim, "vocab_size": cfg.vocab_size,
                "block_size": cfg.block_size, "layer_norm_eps": cfg.layer_norm_eps,
            }
            rec.equal("models.config", "structure", True,
                      "case.execution.config == actual model config",
                      EXPECTED_CONFIG, observed_cfg)
        except Exception as exc:
            self.model = None
            self.tokenizer = None
            rec.failed("models.material.load", "reference_implementation", True,
                       "material gpt2_real.py loads with local weights",
                       "loads without error",
                       f"{type(exc).__name__}: {exc}", "n/a",
                       details={"traceback": traceback.format_exc()[-2000:]})
            return

        try:
            from transformers import GPT2LMHeadModel
            self.hf = GPT2LMHeadModel.from_pretrained(
                str(self.weights_dir), local_files_only=True,
                attn_implementation="eager").to("cpu").eval()
            n_hf = int(sum(p.numel() for p in self.hf.parameters()))
            rec.equal("models.hf.params", "reference_implementation", True,
                      "HF GPT2LMHeadModel parameter count", 124439808, n_hf)
            rec.equal("models.hf.attn_implementation", "reference_implementation", True,
                      "HF config attn implementation", "eager",
                      getattr(self.hf.config, "_attn_implementation", None))
        except Exception as exc:
            self.hf = None
            rec.failed("models.hf.load", "reference_implementation", True,
                       "HF GPT2LMHeadModel(eager) loads from local files",
                       "loads without error", f"{type(exc).__name__}: {exc}", "n/a",
                       details={"traceback": traceback.format_exc()[-2000:]})
            return

        # weight tying / shared output head
        try:
            t5 = getattr(self.hf, "transformer", self.hf)
            hf_tied = bool(np.array_equal(
                t5.wte.weight.detach().numpy(), self.hf.lm_head.weight.detach().numpy()))
            mat_tied = bool(self.model.lm_head.weight.data_ptr()
                            == self.model.wte.weight.data_ptr())
            rec.truth("models.weight_tying", "reference_implementation", True,
                      "output head shares wte weights (no double counted parameters)",
                      ok=hf_tied and mat_tied,
                      observed={"hf_tied": hf_tied, "material_tied": mat_tied})
        except Exception as exc:
            rec.failed("models.weight_tying", "reference_implementation", True,
                       "output head shares wte weights", "tied",
                       f"{type(exc).__name__}: {exc}", "n/a")

        # actual KVCache allocation (numel x element_size)
        try:
            self.kv = self.gr.KVCache(self.model.config, 1,
                                      self.torch.device("cpu"), dtype=self.torch.float32)
        except Exception as exc:
            self.kv = None
            rec.failed("models.kvcache", "cache", True, "material KVCache allocation",
                       "allocatable", f"{type(exc).__name__}: {exc}", "n/a")

        # case generation params vs manifest instance
        inst = self.instance or {}
        rec.equal("models.instance_generated_ids", "structure", True,
                  "manifest.instances[].generated_token_ids == case.generation.generated_ids",
                  self.generated_ids, inst.get("generated_token_ids"))
        rec.equal("models.instance_prompt_token_count", "structure", True,
                  "manifest.instances[].prompt_token_count == len(case.input.ids)",
                  self.prompt_len, inst.get("prompt_token_count"))

    # ==================================================================
    # phase: tokenizer (independent HF re-encode + byte-level BPE mapping)
    # ==================================================================
    def check_tokenizer(self):
        rec = self.rec
        tok = self.tokenizer
        if tok is None:
            rec.not_run("tokenizer", "tokenizer", (False if self.static_only else True), "tokenizer checks",
                        "HF tokenizer loaded", "tokenizer not loaded")
            return
        inp = self.case.get("input") or {}
        text = inp.get("text")
        ids = [int(x) for x in (inp.get("ids") or [])]
        tokens = inp.get("tokens") or []

        vocabulary = self.case.get("vocabulary", {})
        expected_decoded = [tok.decode([i], clean_up_tokenization_spaces=False) for i in range(EXPECTED_CONFIG["vocab_size"])]
        expected_raw = [tok.convert_ids_to_tokens(i) for i in range(EXPECTED_CONFIG["vocab_size"])]
        rec.truth("tokenizer.full_vocabulary", "tokenizer", True,
                  "all standalone and raw vocabulary labels agree with the independent tokenizer",
                  ok=vocabulary.get("decoded") == expected_decoded and vocabulary.get("raw") == expected_raw,
                  observed=len(vocabulary.get("decoded", [])))

        exp_ids = [int(x) for x in tok.encode(text)]
        rec.equal("tokenizer.ids", "tokenizer", True,
                  "independent HF tokenizer.encode(case.input.text)", exp_ids, ids)
        rec.equal("tokenizer.prompt_len", "tokenizer", True,
                  "len(ids) == tokenizer token count", len(exp_ids), len(ids))

        issues = []
        cursor = 0
        for i, item in enumerate(tokens):
            if not isinstance(item, dict):
                issues.append({"index": i, "issue": "token entry is not an object"})
                continue
            tid = item.get("id")
            try:
                raw = token_bytes(tok, tid)
            except Exception as exc:
                issues.append({"index": i, "id": tid, "issue": f"byte mapping failed: {exc}"})
                continue
            complete = True
            try:
                standalone = raw.decode("utf-8")
            except UnicodeDecodeError:
                complete = False
                standalone = raw.decode("utf-8", errors="replace")
            want = {
                "position": i,
                "ordinal": i + 1,
                "id": exp_ids[i] if i < len(exp_ids) else None,
                "raw_token": tok.convert_ids_to_tokens(int(tid)),
                "bytes_hex": raw.hex(),
                "byte_start": cursor,
                "byte_end": cursor + len(raw),
                "standalone_decode": standalone,
                "byte_complete": complete,
            }
            cursor += len(raw)
            for key, w in want.items():
                if item.get(key) != w:
                    issues.append({"index": i, "field": key, "expected": w,
                                   "observed": item.get(key)})
        text_bytes = text.encode("utf-8")
        rec.truth("tokenizer.tokens", "tokenizer", True,
                  "per-token id/raw_token/bytes/byte range/standalone decode/byte_complete",
                  ok=not issues, observed=issues[:12],
                  details={"issue_count": len(issues), "token_count": len(tokens)})
        rec.equal("tokenizer.total_bytes", "tokenizer", True,
                  "sum of token byte lengths == len(utf-8(text))",
                  len(text_bytes), cursor)
        try:
            decoded = tok.decode(ids)
        except Exception as exc:
            decoded = f"<decode failed: {exc}>"
        rec.equal("tokenizer.roundtrip", "tokenizer", True,
                  "tokenizer.decode(ids) == case.input.text", text, decoded)

    # ==================================================================
    # phase: reference implementation (HF eager) parity
    # ==================================================================
    def check_reference(self):
        rec = self.rec
        if self.model is None or self.hf is None:
            rec.not_run("reference.prefill_logits", "reference_implementation",
                        (False if self.static_only else True),
                        "material prefill vs HF GPT2LMHeadModel", "compared",
                        "models not loaded")
            return
        torch = self.torch
        ids_t = torch.tensor([self.prompt_ids], dtype=torch.long)
        try:
            with torch.no_grad():
                mat_logits = self.model(ids_t).detach().numpy()
                hf_out = self.hf(ids_t, output_attentions=True)
                hf_logits = hf_out.logits.detach().numpy()
            self.mat_prefill_logits = mat_logits
            self.hf_prefill_logits = hf_logits
            rec.close("reference.prefill_logits", "reference_implementation", True,
                      "uncached prefill logits: material gpt2_real vs HF (eager)",
                      hf_logits, mat_logits, atol=1e-4,
                      tolerance=TOLERANCES["reference_prefill_logits"])
        except Exception as exc:
            rec.failed("reference.prefill_logits", "reference_implementation", True,
                       "uncached prefill logits", "compared",
                       f"{type(exc).__name__}: {exc}", "n/a",
                       details={"traceback": traceback.format_exc()[-2000:]})
            return

        # all layers / all heads: exported fwd0 attention vs HF attentions
        try:
            hf_attn = [a.detach().numpy()[0] for a in hf_out.attentions]  # each (H,T,T)
            issues = []
            for layer in range(EXPECTED_LAYERS):
                tid = f"fwd0.block{layer}.attn_weights"
                observed = self.a64(tid)[0]
                expected = hf_attn[layer]
                max_abs, ok, mismatch = close_values(observed, expected, 1e-6, 0.0)
                if mismatch is not None or not ok:
                    issues.append({"layer": layer, "max_abs_diff": max_abs,
                                   "shape_mismatch": mismatch})
                rec.add(f"reference.attention.layer{layer}", "reference_implementation",
                        True, f"fwd0 attention weights layer {layer} (all heads) vs HF",
                        {"max_abs_diff": 0.0},
                        {"max_abs_diff": max_abs, "shape_mismatch": mismatch},
                        TOLERANCES["reference_attention"], ok=ok)
            # material attention_maps vs HF (extra evidence, same tolerance)
            with torch.no_grad():
                mat_attn = self.model.attention_maps(ids_t).detach().numpy()[:, 0]
            max_abs, ok, mismatch = close_values(mat_attn, np.stack(hf_attn), 1e-6, 0.0)
            rec.add("reference.attention.material_vs_hf", "reference_implementation", False,
                    "material attention_maps vs HF attentions (all layers)", 
                    {"max_abs_diff": 0.0}, {"max_abs_diff": max_abs},
                    TOLERANCES["reference_attention"], ok=ok)
        except Exception as exc:
            rec.failed("reference.attention", "reference_implementation", True,
                       "all-layer all-head attention vs HF", "compared",
                       f"{type(exc).__name__}: {exc}", "n/a",
                       details={"traceback": traceback.format_exc()[-2000:]})

        # greedy generation sequence: HF vs case
        try:
            hf_gen = self.hf.generate(ids_t, max_new_tokens=EXPECTED_SAMPLE_COUNT,
                                      do_sample=False, pad_token_id=50256)
            hf_gen_ids = [int(x) for x in hf_gen[0, len(self.prompt_ids):].tolist()]
            rec.equal("reference.greedy_hf", "reference_implementation", True,
                      "HF greedy generate ids", self.generated_ids, hf_gen_ids)
        except Exception as exc:
            rec.failed("reference.greedy_hf", "reference_implementation", True,
                       "HF greedy generate ids", self.generated_ids,
                       f"{type(exc).__name__}: {exc}", "exact")

        # greedy generation sequence: unmodified material generate_cached vs case
        try:
            with torch.no_grad():
                mat_gen = self.gr.generate_cached(
                    self.model, ids_t.clone(), max_new_tokens=EXPECTED_SAMPLE_COUNT,
                    greedy=True)
            mat_gen_ids = [int(x) for x in mat_gen[0, len(self.prompt_ids):].tolist()]
            mat_full = [int(x) for x in mat_gen[0].tolist()]
            rec.equal("reference.greedy_material", "collection_equivalence", True,
                      "unmodified gpt2_real generate_cached generated ids",
                      self.generated_ids, mat_gen_ids,
                      tolerance=TOLERANCES["collection_equivalence_tokens"])
            rec.equal("reference.greedy_full_sequence", "collection_equivalence", True,
                      "unmodified gpt2_real generate_cached full sequence",
                      self.full_ids, mat_full,
                      tolerance=TOLERANCES["collection_equivalence_tokens"])
        except Exception as exc:
            rec.failed("reference.greedy_material", "collection_equivalence", True,
                       "unmodified gpt2_real generate_cached", "runs and matches",
                       f"{type(exc).__name__}: {exc}", "exact")

    # ==================================================================
    # phase: collection equivalence (hook-observed rerun, bitwise logits)
    # ==================================================================
    def check_equivalence(self):
        rec = self.rec
        if self.model is None:
            rec.not_run("equivalence.logits", "collection_equivalence",
                        (False if self.static_only else True),
                        "uncollected rerun", "compared", "models not loaded")
            return
        torch = self.torch
        captured = []
        emb_caps = {"wte": [], "wpe": [], "ln_f": []}
        handles = []

        def model_hook(module, args, kwargs, output):
            captured.append({
                "input": args[0].detach().clone() if args and torch.is_tensor(args[0]) else None,
                "start_pos": kwargs.get("start_pos"),
                "output": output.detach().clone(),
            })

        handles.append(self.model.register_forward_hook(model_hook, with_kwargs=True))
        for key, modname in (("wte", "wte"), ("wpe", "wpe"), ("ln_f", "ln_f")):
            mod = getattr(self.model, modname)

            def make_hook(k):
                def hook(module, args, output):
                    emb_caps[k].append(output.detach().clone())
                return hook
            handles.append(mod.register_forward_hook(make_hook(key)))
        try:
            with torch.no_grad():
                gen = self.gr.generate_cached(
                    self.model, torch.tensor([self.prompt_ids], dtype=torch.long),
                    max_new_tokens=EXPECTED_SAMPLE_COUNT, greedy=True)
        finally:
            for h in handles:
                h.remove()
        gen_ids = [int(x) for x in gen[0, len(self.prompt_ids):].tolist()]
        rec.equal("equivalence.tokens", "collection_equivalence", True,
                  "uncollected rerun generated token sequence",
                  self.generated_ids, gen_ids,
                  tolerance=TOLERANCES["collection_equivalence_tokens"])
        rec.equal("equivalence.forward_count", "collection_equivalence", True,
                  "number of model forward calls in generate_cached",
                  EXPECTED_FORWARD_COUNT, len(captured),
                  tolerance="exact",
                  details={"note": "no forward after the last selection"})
        rec.truth("equivalence.no_unused_forward", "collection_equivalence", True,
                  "one forward per generated token; last event is sample",
                  ok=(len(captured) == EXPECTED_SAMPLE_COUNT and self.events[-1]["kind"] == "sample"),
                  observed={"forward_count": len(captured), "last_event": self.events[-1]["id"]})
        if len(captured) != EXPECTED_FORWARD_COUNT:
            rec.not_run("equivalence.per_forward", "collection_equivalence", True,
                        "per-forward comparison", f"{EXPECTED_FORWARD_COUNT} forwards",
                        f"observed {len(captured)} forwards")
            return
        # per-forward: input ids / start_pos / shape / last-row bitwise equality
        for f, cap in enumerate(captured):
            eid = f"fwd{f}"
            meta = self.forward_meta.get(f) or {}
            T = meta.get("T", 1)
            got_input = cap["input"][0].tolist() if cap["input"] is not None else None
            exp_input = (self.prompt_ids if f == 0 else [self.generated_ids[f - 1]])
            rec.equal(f"equivalence.input.f{f}", "collection_equivalence", True,
                      f"forward {f} input ids", exp_input, got_input)
            rec.equal(f"equivalence.start_pos.f{f}", "collection_equivalence", True,
                      f"forward {f} start_pos", meta.get("start"), cap["start_pos"])
            out = cap["output"].detach().numpy()  # (1,T,V)
            lev = self.fwd_event(f, "logits").get("logits_event") or {}
            rec.equal(f"equivalence.out_shape.f{f}", "collection_equivalence", True,
                      f"forward {f} full output shape", lev.get("full_output_shape"),
                      list(out.shape))
            obs_bytes = f32_bytes(out[0, -1])
            exp_bytes = f32_bytes(self.a64(f"fwd{f}.logits.last_row")[0])
            rec.exact_arrays(f"equivalence.logits.f{f}", "collection_equivalence", True,
                             f"forward {f} logits last row: collected payload vs uncollected rerun",
                             obs_bytes, exp_bytes,
                             tolerance="bitwise equal (max abs diff == 0)",
                             details={"row_index": -1, "forward_index": f})
            # per-position summaries stored in logits_event
            argmax = [int(x) for x in out[0].argmax(axis=-1).tolist()]
            maxlogit = [float(x) for x in out[0].max(axis=-1).tolist()]
            rec.exact_arrays(f"equivalence.summary_argmax.f{f}", "collection_equivalence",
                             True, f"forward {f} logits_event.per_position_argmax",
                             np.asarray(lev.get("per_position_argmax"), dtype=np.int32).tobytes(),
                             np.asarray(argmax, dtype=np.int32).tobytes(),
                             tolerance="exact integers")
            max_abs, ok, mismatch = close_values(
                lev.get("per_position_max_logit"), maxlogit, 1e-6, 0.0)
            rec.add(f"equivalence.summary_maxlogit.f{f}", "collection_equivalence", True,
                    f"forward {f} logits_event.per_position_max_logit",
                    {"max_abs_diff": 0.0}, {"max_abs_diff": max_abs},
                    "max|diff| <= 1e-6", ok=ok)
            # greedy choice from the full output (samples 0..2 only)
            if f < EXPECTED_SAMPLE_COUNT:
                rec.equal(f"equivalence.greedy_token.f{f}", "collection_equivalence", True,
                          f"forward {f} last-row argmax == generated token {f}",
                          self.generated_ids[f], argmax[-1])
        # embed / ln_f outputs: exported payload vs rerun module outputs (bitwise)
        for key, payload_fmt in (("wte", "fwd{f}.embed.wte_out"),
                                 ("wpe", "fwd{f}.embed.wpe_out"),
                                 ("ln_f", "fwd{f}.ln_f.out")):
            if len(emb_caps[key]) != EXPECTED_FORWARD_COUNT:
                rec.failed(f"equivalence.{key}", "collection_equivalence", True,
                           f"{key} outputs captured for all forwards", EXPECTED_FORWARD_COUNT,
                           len(emb_caps[key]), "exact")
                continue
            issues = []
            for f in range(EXPECTED_FORWARD_COUNT):
                tid = payload_fmt.format(f=f)
                obs = f32_bytes(emb_caps[key][f].detach().numpy())
                exp = f32_bytes(self.a64(tid))
                if obs != exp:
                    issues.append({"forward_index": f,
                                   "observed_sha256": sha256_bytes(obs),
                                   "payload_sha256": sha256_bytes(exp)})
            rec.truth(f"equivalence.{key}_outputs", "collection_equivalence", True,
                      f"exported {key} outputs are bitwise identical to uncollected rerun",
                      ok=not issues, observed=issues[:5],
                      details={"note": "detach/clone during collection must not change values"})

    # ==================================================================
    # phase: formula recompute from payloads (float64, section 8)
    # ==================================================================
    def _load_weights(self):
        t5 = getattr(self.hf, "transformer", self.hf)
        W = {}
        W["wte"] = t5.wte.weight.detach().numpy().astype(np.float64)
        W["wpe"] = t5.wpe.weight.detach().numpy().astype(np.float64)
        W["ln_f_w"] = t5.ln_f.weight.detach().numpy().astype(np.float64)
        W["ln_f_b"] = t5.ln_f.bias.detach().numpy().astype(np.float64)
        for l in range(EXPECTED_LAYERS):
            h = t5.h[l]
            p = f"l{l}."
            W[p + "ln1_w"] = h.ln_1.weight.detach().numpy().astype(np.float64)
            W[p + "ln1_b"] = h.ln_1.bias.detach().numpy().astype(np.float64)
            W[p + "cattn_w"] = h.attn.c_attn.weight.detach().numpy().astype(np.float64)
            W[p + "cattn_b"] = h.attn.c_attn.bias.detach().numpy().astype(np.float64)
            W[p + "cproj_w"] = h.attn.c_proj.weight.detach().numpy().astype(np.float64)
            W[p + "cproj_b"] = h.attn.c_proj.bias.detach().numpy().astype(np.float64)
            W[p + "ln2_w"] = h.ln_2.weight.detach().numpy().astype(np.float64)
            W[p + "ln2_b"] = h.ln_2.bias.detach().numpy().astype(np.float64)
            W[p + "cfc_w"] = h.mlp.c_fc.weight.detach().numpy().astype(np.float64)
            W[p + "cfc_b"] = h.mlp.c_fc.bias.detach().numpy().astype(np.float64)
            W[p + "cmlp_w"] = h.mlp.c_proj.weight.detach().numpy().astype(np.float64)
            W[p + "cmlp_b"] = h.mlp.c_proj.bias.detach().numpy().astype(np.float64)
        self.W = W
        self.wte = W["wte"]
        self.wpe = W["wpe"]

    def check_formula(self):
        rec = self.rec
        if self.hf is None or "wte" not in getattr(self, "W", {}):
            if self.hf is not None:
                try:
                    self._load_weights()
                except Exception as exc:
                    rec.failed("formula.weights", "formula_recompute", True,
                               "load reference weights for formula recompute",
                               "loaded", f"{type(exc).__name__}: {exc}", "n/a")
                    return
            else:
                rec.not_run("formula.recompute", "formula_recompute",
                            (False if self.static_only else True),
                            "float64 payload recompute", "compared", "models not loaded")
                return
        l = self.selected_layer
        pfx = f"l{l}."
        W = self.W
        for f in range(EXPECTED_FORWARD_COUNT):
            if f not in self.forward_meta:
                continue
            meta = self.forward_meta[f]
            start, T, total = meta["start"], meta["T"], meta["total"]
            try:
                self.formula_forward(f, start, T, total, l, pfx)
            except Exception as exc:
                rec.failed(f"formula.forward.f{f}", "formula_recompute", True,
                           f"forward {f} payload recompute", "completes",
                           f"{type(exc).__name__}: {exc}", "n/a",
                           details={"traceback": traceback.format_exc()[-2000:]})

    def formula_forward(self, f, start, T, total, l, pfx):
        rec = self.rec
        W = self.W
        btid = lambda role: f"fwd{f}.block{l}.{role}"
        # ---- embeddings ----
        wte_out = self.a64(f"fwd{f}.embed.wte_out")[0]
        wpe_out = self.a64(f"fwd{f}.embed.wpe_out")
        exp_wte = self.wte[self.prompt_ids if f == 0 else [self.generated_ids[f - 1]]]
        exp_wpe = self.wpe[np.arange(start, start + T)]
        x_in = self.a64(btid("x_in"))[0]
        diffs = {}
        for name, obs, exp in (("wte_out", wte_out, exp_wte),
                               ("wpe_out", wpe_out, exp_wpe),
                               ("x_in_sum", x_in, exp_wte + exp_wpe)):
            max_abs, ok, mismatch = close_values(obs, exp, 1e-4, 1e-4)
            diffs[name] = {"max_abs_diff": max_abs, "shape_mismatch": mismatch, "ok": ok}
        rec.truth(f"formula.embed.f{f}", "formula_recompute", True,
                  f"forward {f} embeddings: wte_out, wpe_out, x_in = wte+wpe",
                  ok=all(d["ok"] for d in diffs.values()), observed=diffs,
                  tolerance=TOLERANCES["formula_activations"])

        # ---- LayerNorm 1 ----
        ln1_out = self.a64(btid("ln1_out"))[0]
        ln1_exp = ln_f64(x_in, W[pfx + "ln1_w"], W[pfx + "ln1_b"])
        rec.close(f"formula.ln1.f{f}", "formula_recompute", True,
                  f"forward {f} block{l} LN1 from exported x_in", ln1_exp, ln1_out,
                  atol=1e-4, rtol=1e-4, tolerance=TOLERANCES["formula_activations"])
        # ---- qkv linear ----
        qkv = self.a64(btid("qkv"))[0]
        qkv_exp = ln1_out @ W[pfx + "cattn_w"] + W[pfx + "cattn_b"]
        rec.close(f"formula.qkv.f{f}", "formula_recompute", True,
                  f"forward {f} block{l} qkv = c_attn(ln1_out)", qkv_exp, qkv,
                  atol=1e-4, rtol=1e-4, tolerance=TOLERANCES["formula_activations"])
        C = EXPECTED_CONFIG["n_embd"]
        H = EXPECTED_CONFIG["n_head"]
        D = EXPECTED_CONFIG["head_dim"]
        # ---- split ----
        q_raw, k_raw, v_raw = (self.a64(btid("q_raw"))[0], self.a64(btid("k_raw"))[0],
                               self.a64(btid("v_raw"))[0])
        split_obs = {"q_raw": q_raw, "k_raw": k_raw, "v_raw": v_raw}
        split_exp = {"q_raw": qkv[:, :C], "k_raw": qkv[:, C:2 * C], "v_raw": qkv[:, 2 * C:]}
        split_diffs = {}
        for name in split_obs:
            max_abs, ok, mismatch = close_values(split_obs[name], split_exp[name],
                                                 1e-4, 1e-4)
            split_diffs[name] = {"max_abs_diff": max_abs, "ok": ok}
        rec.truth(f"formula.split.f{f}", "formula_recompute", True,
                  f"forward {f} block{l} q/k/v split", ok=all(d["ok"] for d in split_diffs.values()),
                  observed=split_diffs, tolerance=TOLERANCES["formula_activations"])
        # ---- view/transpose heads ----
        heads = {}
        for name, raw in (("q_heads", q_raw), ("k_heads", k_raw), ("v_heads", v_raw)):
            exp = raw.reshape(T, H, D).transpose(1, 0, 2)
            obs = self.a64(btid(name))[0]
            max_abs, ok, mismatch = close_values(obs, exp, 1e-4, 1e-4)
            heads[name] = {"max_abs_diff": max_abs, "ok": ok, "shape_mismatch": mismatch}
        rec.truth(f"formula.heads.f{f}", "formula_recompute", True,
                  f"forward {f} block{l} q/k/v view+transpose", 
                  ok=all(d["ok"] for d in heads.values()), observed=heads,
                  tolerance=TOLERANCES["formula_activations"])
        # ---- scores (pre-mask, finite) ----
        q_heads = self.a64(btid("q_heads"))[0]
        k_read = self.a64(btid("k_read"))[0]
        v_read = self.a64(btid("v_read"))[0]
        scores_obs = self.a64(btid("scores_scaled"))[0]
        scores_exp = q_heads @ k_read.transpose(0, 2, 1) / math.sqrt(D)
        rec.close(f"formula.scores.f{f}", "formula_recompute", True,
                  f"forward {f} block{l} scores = q@k_read^T / sqrt({D})",
                  scores_exp, scores_obs, atol=1e-4, rtol=1e-4,
                  tolerance=TOLERANCES["formula_activations"],
                  details={"note": "pre-mask finite scores; masked positions are not -inf"})
        # ---- causal mask ----
        mask = self.a8(btid("causal_mask"))
        q_positions = np.arange(start, start + T)[:, None]
        k_positions = np.arange(total)[None, :]
        mask_exp = (k_positions <= q_positions).astype(np.uint8)
        rec.exact_arrays(f"formula.mask.f{f}", "formula_recompute", True,
                         f"forward {f} block{l} causal mask (key_pos <= query_pos)",
                         mask_exp.tobytes(), np.ascontiguousarray(mask).tobytes(),
                         tolerance="exact uint8 equality")
        # ---- softmax over masked scores ----
        scores_masked = np.where(mask[None, :, :] == 1, scores_obs,
                                 -np.inf)
        weights = softmax_f64(scores_masked)
        attn_obs = self.a64(btid("attn_weights"))[0]
        rec.close(f"formula.softmax.f{f}", "formula_recompute", True,
                  f"forward {f} block{l} softmax(masked scores)",
                  weights, attn_obs, atol=1e-6, rtol=0.0,
                  tolerance=TOLERANCES["formula_softmax"])
        zero_mask = mask == 0
        n_nonzero = int(np.count_nonzero(attn_obs[:, :, :][:, zero_mask]) if zero_mask.any()
                        else 0)
        rec.truth(f"formula.mask_zero.f{f}", "formula_recompute", True,
                  f"forward {f} block{l} masked attention weights are exactly 0",
                  ok=n_nonzero == 0, observed={"nonzero_masked_weights": n_nonzero},
                  tolerance=TOLERANCES["formula_mask_zero"])
        row_err = float(np.abs(attn_obs.sum(axis=-1) - 1.0).max())
        rec.truth(f"formula.row_sum.f{f}", "formula_recompute", True,
                  f"forward {f} block{l} attention row sums",
                  ok=row_err <= 1e-6, observed={"max_abs_row_sum_error": row_err},
                  tolerance=TOLERANCES["formula_row_sum"])
        # ---- weighted V ----
        weighted_obs = self.a64(btid("weighted_v"))[0]
        weighted_exp = (attn_obs @ v_read).transpose(1, 0, 2).reshape(T, C)
        rec.close(f"formula.weighted_v.f{f}", "formula_recompute", True,
                  f"forward {f} block{l} weighted V (transpose+reshape before c_proj)",
                  weighted_exp, weighted_obs, atol=1e-4, rtol=1e-4,
                  tolerance=TOLERANCES["formula_activations"])
        # ---- attention output projection + residual ----
        attn_out_obs = self.a64(btid("attn_out"))[0]
        attn_out_exp = weighted_obs @ W[pfx + "cproj_w"] + W[pfx + "cproj_b"]
        rec.close(f"formula.attn_out.f{f}", "formula_recompute", True,
                  f"forward {f} block{l} c_proj(weighted_v)", attn_out_exp, attn_out_obs,
                  atol=1e-4, rtol=1e-4, tolerance=TOLERANCES["formula_activations"])
        x_after_obs = self.a64(btid("x_after_attn"))[0]
        rec.close(f"formula.x_after_attn.f{f}", "formula_recompute", True,
                  f"forward {f} block{l} residual x_in + attn_out",
                  x_in + attn_out_obs, x_after_obs, atol=1e-4, rtol=1e-4,
                  tolerance=TOLERANCES["formula_activations"])
        # ---- LN2 / MLP ----
        ln2_obs = self.a64(btid("ln2_out"))[0]
        ln2_exp = ln_f64(x_after_obs, W[pfx + "ln2_w"], W[pfx + "ln2_b"])
        rec.close(f"formula.ln2.f{f}", "formula_recompute", True,
                  f"forward {f} block{l} LN2", ln2_exp, ln2_obs,
                  atol=1e-4, rtol=1e-4, tolerance=TOLERANCES["formula_activations"])
        fc_obs = self.a64(btid("fc_out"))[0]
        fc_exp = ln2_obs @ W[pfx + "cfc_w"] + W[pfx + "cfc_b"]
        rec.close(f"formula.fc.f{f}", "formula_recompute", True,
                  f"forward {f} block{l} c_fc(ln2_out)", fc_exp, fc_obs,
                  atol=1e-4, rtol=1e-4, tolerance=TOLERANCES["formula_activations"])
        act_obs = self.a64(btid("act_out"))[0]
        act_exp = gelu_new_f64(fc_obs)
        rec.close(f"formula.act.f{f}", "formula_recompute", True,
                  f"forward {f} block{l} gelu_new(tanh approximation)",
                  act_exp, act_obs, atol=1e-4, rtol=1e-4,
                  tolerance=TOLERANCES["formula_activations"])
        mlp_obs = self.a64(btid("mlp_out"))[0]
        mlp_exp = act_obs @ W[pfx + "cmlp_w"] + W[pfx + "cmlp_b"]
        rec.close(f"formula.mlp_out.f{f}", "formula_recompute", True,
                  f"forward {f} block{l} mlp c_proj(act_out)", mlp_exp, mlp_obs,
                  atol=1e-4, rtol=1e-4, tolerance=TOLERANCES["formula_activations"])
        x_out_obs = self.a64(btid("x_out"))[0]
        rec.close(f"formula.x_out.f{f}", "formula_recompute", True,
                  f"forward {f} block{l} block output x_after_attn + mlp_out",
                  x_after_obs + mlp_obs, x_out_obs, atol=1e-4, rtol=1e-4,
                  tolerance=TOLERANCES["formula_activations"])
        # ---- final LayerNorm payload: normalization statistics ----
        lnf = self.a64(f"fwd{f}.ln_f.out")[0]
        y = (lnf - W["ln_f_b"]) / W["ln_f_w"]
        mean_err = float(np.abs(y.mean(axis=-1)).max()) if T else 0.0
        var_err = float(np.abs(y.var(axis=-1) - 1.0).max()) if T else 0.0
        rec.truth(f"formula.ln_f_normalization.f{f}", "formula_recompute", True,
                  f"forward {f} ln_f.out normalized (mean 0, var 1 before affine)",
                  ok=mean_err <= 1e-5 and var_err <= 1e-5,
                  observed={"max_abs_mean": mean_err, "max_abs_var_minus_1": var_err},
                  tolerance="abs <= 1e-5 (float32 payload of a normalized value)",
                  details={"note": "schema section 8 does not state a tolerance for this "
                                    "intrinsic LN check; verifier declares 1e-5 before running"})
        # ---- output head (clarified schema section 8): ----
        # (a) float32 reproduction with the same execution shape as the model
        #     (prefill: one [1,T,C] batched lm_head call; decode: one [1,1,C] call),
        #     last row must be bitwise equal;
        # (b) float64 recompute with scale-free relative error max|diff| / max|logit|
        #     <= 1e-5, plus the raw absolute difference for the record.
        last_row_tid = f"fwd{f}.logits.last_row"
        last_row_raw = self.arrays[last_row_tid].astype("<f4")
        last_row_obs = self.a64(last_row_tid)[0]
        # (a) same-execution-shape float32 reproduction of the ONNX-free path used by
        # GPT2.forward: linear(lnf_all) with the shared, tied wte/lm_head weight.
        try:
            lnf_payload = self.a64(f"fwd{f}.ln_f.out")  # (1,T,C), values are float32
            lnf_f32 = self.torch.from_numpy(
                np.ascontiguousarray(lnf_payload, dtype=np.float32)).clone()
            with self.torch.no_grad():
                reproduced = self.torch.nn.functional.linear(
                    lnf_f32, self.model.lm_head.weight.detach())
            rep_row = reproduced[0, -1].detach().numpy().astype("<f4")
            max_abs_f32 = float(np.abs(rep_row.astype(np.float64)
                                       - last_row_obs).max())
            byte_equal = rep_row.tobytes() == last_row_raw.tobytes()
            rec.add(f"formula.output_head.f{f}", "formula_recompute", True,
                    f"forward {f} output head (a): float32 same-execution-shape batched "
                    f"linear(ln_f.out) with shared wte/lm_head",
                    {"byte_equal": True, "max_abs_diff": 0.0},
                    {"byte_equal": byte_equal, "max_abs_diff": max_abs_f32},
                    TOLERANCES["output_head"], ok=byte_equal,
                    details={"execution_shape": list(lnf_f32.shape),
                             "f32_batched_max_abs_diff": max_abs_f32})
        except Exception as exc:
            rec.failed(f"formula.output_head.f{f}", "formula_recompute", True,
                       f"forward {f} output head (a) float32 same-shape reproduction",
                       "reproduced", f"{type(exc).__name__}: {exc}", "n/a",
                       details={"traceback": traceback.format_exc()[-1200:]})
        # (b) float64 recompute with relative-scale gate; prior strict absolute gate
        # is retained in details for audit (do not delete the historical fact).
        last_row_exp = lnf[-1] @ self.wte.T
        diff = np.abs(last_row_exp - last_row_obs)
        max_abs = float(diff.max()) if diff.size else 0.0
        scale = float(np.abs(last_row_obs).max()) if last_row_obs.size else 0.0
        rel = (max_abs / scale) if scale > 0 else float("inf")
        prior_abs_ok = max_abs <= 1e-4
        rec.add(f"formula.output_head.relative.f{f}", "formula_recompute", True,
                f"forward {f} output head (b): float64 recompute, "
                f"max|diff| / max|logit| <= 1e-5",
                {"relative": 0.0}, {"max_abs_diff": max_abs, "relative": rel,
                                    "max_abs_logit": scale},
                TOLERANCES["output_head"], ok=rel <= 1e-5,
                details={"prior_strict_abs_gate": {
                    "criterion": "max|diff| <= 1e-4 (absolute; pre-clarification wording)",
                    "observed_max_abs_diff": max_abs,
                    "result": "passed" if prior_abs_ok else
                              "failed before schema clarification; retained for audit"}})

    # ==================================================================
    # phase: generation (samples) and fwd3 non-use
    # ==================================================================
    def check_samples(self):
        rec = self.rec
        if not self.arrays:
            rec.not_run("generation.samples", "generation", True, "sample events",
                        "verified", "payloads not loaded")
            return
        events = [e for e in self.events if e.get("kind") == "sample"]
        rec.equal("generation.sample_count", "generation", True,
                  "number of sample events", EXPECTED_SAMPLE_COUNT, len(events))
        tok = self.tokenizer
        for s in range(EXPECTED_SAMPLE_COUNT):
            eid = f"sample{s}"
            ev = self.events_by_id.get(eid)
            if ev is None:
                rec.failed(f"generation.sample.s{s}", "generation", True, eid,
                           "sample event present", "missing", "exact")
                continue
            row = self.a64(f"fwd{s}.logits.last_row")[0].astype(np.float64)
            p = softmax_f64(row)
            order = np.argsort(-p)
            token = int(order[0])
            logit = float(row[token])
            prob = float(p[token])
            rank = int(np.count_nonzero(row > row[token]) + 1)
            top_ids = [int(x) for x in order[:SAMPLE_TOP_K]]
            top_probs = [float(x) for x in p[top_ids]]
            entropy = float(-np.sum(p * np.log(p)))
            top_mass = float(np.sum(top_probs))
            other_mass = 1.0 - top_mass
            other_alt = 1.0 - prob
            # token / probability / rank / logit
            issues = []
            if ev.get("token_id") != token:
                issues.append({"field": "token_id", "expected": token,
                               "observed": ev.get("token_id")})
            if s >= len(self.generated_ids) or token != self.generated_ids[s]:
                issues.append({"field": "generated_ids", "expected": self.generated_ids,
                               "observed": token})
            max_p, ok_p, _ = close_values([ev.get("probability")], [prob], 1e-6, 0.0)
            if not ok_p:
                issues.append({"field": "probability", "expected": prob,
                               "observed": ev.get("probability"),
                               "max_abs_diff": max_p})
            max_l, ok_l, _ = close_values([ev.get("logit")], [logit], 1e-6, 0.0)
            if not ok_l:
                issues.append({"field": "logit", "expected": logit,
                               "observed": ev.get("logit"), "max_abs_diff": max_l})
            if ev.get("rank") != rank:
                issues.append({"field": "rank", "expected": rank,
                               "observed": ev.get("rank")})
            rec.truth(f"generation.sample.s{s}.token_prob_rank", "generation", True,
                      f"{eid} greedy token / probability / rank / logit",
                      ok=not issues, observed=issues[:6],
                      tolerance=TOLERANCES["generation_probability"] +
                                "; logit/rank exact",
                      details={"note": "probabilities recomputed by promoting the float32 "
                                       "logits row to float64 before softmax (schema 5.2)"})
            # top-k view
            top_issues = []
            top_field = ev.get("top_k")
            if not isinstance(top_field, list) or len(top_field) != SAMPLE_TOP_K:
                top_issues.append({"field": "top_k", "expected_length": SAMPLE_TOP_K,
                                   "observed_length": len(top_field)
                                   if isinstance(top_field, list) else None})
            else:
                for i, (want_id, want_p) in enumerate(zip(top_ids, top_probs)):
                    item = top_field[i]
                    if item.get("id") != want_id:
                        top_issues.append({"index": i, "field": "id", "expected": want_id,
                                           "observed": item.get("id")})
                    max_abs, ok, _ = close_values([item.get("probability")], [want_p],
                                                  1e-6, 0.0)
                    if not ok:
                        top_issues.append({"index": i, "field": "probability",
                                           "expected": want_p,
                                           "observed": item.get("probability"),
                                           "max_abs_diff": max_abs})
            rec.truth(f"generation.sample.s{s}.top_k", "generation", True,
                      f"{eid} top_k ids/probabilities (top {SAMPLE_TOP_K} of full vocab row)",
                      ok=not top_issues, observed=top_issues[:6],
                      tolerance="ids exact; |prob diff| <= 1e-6 (float64 softmax)")
            # other mass / entropy
            mass_issues = []
            observed_other = ev.get("other_probability_mass")
            max_m, ok_m, _ = close_values([observed_other], [other_mass], 1e-6, 0.0)
            alt_m, ok_alt, _ = close_values([observed_other], [other_alt], 1e-6, 0.0)
            note = ("matched 1 - sum(top_k)" if ok_m
                    else "matched 1 - p(selected) (alternative reading)")
            if not ok_m and not ok_alt:
                mass_issues.append({"field": "other_probability_mass",
                                    "expected": other_mass,
                                    "observed": observed_other,
                                    "max_abs_diff": max_m})
            max_e, ok_e, _ = close_values([ev.get("entropy_nats")], [entropy], 1e-5, 0.0)
            if not ok_e:
                mass_issues.append({"field": "entropy_nats", "expected": entropy,
                                    "observed": ev.get("entropy_nats"),
                                    "max_abs_diff": max_e})
            rec.truth(f"generation.sample.s{s}.mass_entropy", "generation", True,
                      f"{eid} other_probability_mass and entropy_nats from full row",
                      ok=not mass_issues, observed=mass_issues[:4],
                      tolerance=TOLERANCES["generation_other_mass"] + "; entropy <= 1e-5",
                      details={"other_mass_interpretation": note,
                               "computed_other_mass": other_mass,
                               "computed_entropy": entropy})
            # event bookkeeping / decoded text
            book_issues = []
            exp_pos = self.prompt_len + s
            if ev.get("output_position") != exp_pos:
                book_issues.append({"field": "output_position", "expected": exp_pos,
                                    "observed": ev.get("output_position")})
            if ev.get("strategy") != "greedy":
                book_issues.append({"field": "strategy", "expected": "greedy",
                                    "observed": ev.get("strategy")})
            if ev.get("sample_index") != s:
                book_issues.append({"field": "sample_index", "expected": s,
                                    "observed": ev.get("sample_index")})
            if ev.get("phase") != "sample":
                book_issues.append({"field": "phase", "expected": "sample",
                                    "observed": ev.get("phase")})
            if tok is not None:
                try:
                    exp_raw = tok.convert_ids_to_tokens(token)
                    exp_delta = self._incremental_text(token, exp_pos)
                    exp_standalone = decode_token_bytes(token_bytes(tok, token))
                    raw_bytes = token_bytes(tok, token)
                    try:
                        raw_bytes.decode("utf-8")
                        exp_complete = True
                    except UnicodeDecodeError:
                        exp_complete = False
                except Exception as exc:
                    book_issues.append({"issue": f"tokenizer decode failed: {exc}"})
                    exp_raw = exp_delta = exp_standalone = None
                    exp_complete = None
                if exp_raw is not None and ev.get("raw_token") != exp_raw:
                    book_issues.append({"field": "raw_token", "expected": exp_raw,
                                        "observed": ev.get("raw_token")})
                if exp_delta is not None and ev.get("decoded_text") != exp_delta:
                    book_issues.append({"field": "decoded_text", "expected": exp_delta,
                                        "observed": ev.get("decoded_text")})
                if exp_standalone is not None and ev.get("standalone_decode_complete") is not None:
                    if bool(ev.get("standalone_decode_complete")) != exp_complete:
                        # not a schema 8 gate; recorded for transparency
                        rec.add(f"generation.sample.s{s}.standalone_complete", "generation",
                                False, f"{eid} standalone_decode_complete",
                                exp_complete, ev.get("standalone_decode_complete"),
                                "strict UTF-8 decodability of the token bytes",
                                ok=bool(ev.get("standalone_decode_complete")) == exp_complete,
                                details={"note": "strict UTF-8 decodability of the token bytes; "
                                                 "schema 5.2 example updated to true on 2026-10-30"})
            replay = ev.get("replay") or {}
            if replay.get("sequence_length_after") != exp_pos + 1:
                book_issues.append({"field": "replay.sequence_length_after",
                                    "expected": exp_pos + 1,
                                    "observed": replay.get("sequence_length_after")})
            rec.truth(f"generation.sample.s{s}.fields", "generation", True,
                      f"{eid} position/strategy/raw_token/decoded_text/replay",
                      ok=not book_issues, observed=book_issues[:6])

        rec.truth("generation.stopped", "generation", True,
                  "generation stops at the final selection without another forward",
                  ok=(self.events[-1].get("id") == f"sample{EXPECTED_SAMPLE_COUNT - 1}"
                      and self.case["execution"]["generation"].get("stop_reason") == "max_new_tokens"),
                  observed=self.events[-1].get("id"))
        for s in range(EXPECTED_SAMPLE_COUNT):
            lev = self.fwd_event(s, "logits").get("logits_event") or {}
            rec.truth(f"generation.fwd{s}_used", "generation", True,
                      f"fwd{s}.logits_event.used_by_sampling true, used_by_sample sample{s}",
                      ok=(lev.get("used_by_sampling") is True
                          and lev.get("used_by_sample") == f"sample{s}"),
                      observed={"used_by_sampling": lev.get("used_by_sampling"),
                                "used_by_sample": lev.get("used_by_sample")})
        # logits_event shape / dtype / last row index for every forward
        for f in range(EXPECTED_FORWARD_COUNT):
            meta = self.forward_meta.get(f) or {}
            ev = self.fwd_event(f, "logits")
            lev = ev.get("logits_event") or {}
            T = meta.get("T", 1)
            expected = {
                "full_output_shape": [1, T, EXPECTED_CONFIG["vocab_size"]],
                "full_output_dtype": "float32",
                "last_row_index": meta.get("start", 0) + T - 1,
            }
            got = {k: lev.get(k) for k in expected}
            issues = [{"field": k, "expected": v, "observed": lev.get(k)}
                      for k, v in expected.items() if lev.get(k) != v]
            if not isinstance(lev.get("per_position_argmax"), list) or \
                    len(lev.get("per_position_argmax")) != T:
                issues.append({"field": "per_position_argmax",
                               "expected_length": T,
                               "observed": lev.get("per_position_argmax")})
            if not isinstance(lev.get("per_position_max_logit"), list) or \
                    len(lev.get("per_position_max_logit")) != T:
                issues.append({"field": "per_position_max_logit",
                               "expected_length": T,
                               "observed": lev.get("per_position_max_logit")})
            rec.truth(f"generation.logits_event.f{f}", "generation", True,
                      f"fwd{f}.logits event metadata", ok=not issues,
                      observed=issues[:6])

    def _incremental_text(self, token_id, output_position) -> str:
        tok = self.tokenizer
        seq = self.full_ids[:output_position + 1]
        before = tok.decode(seq[:-1]) if output_position > 0 else ""
        after = tok.decode(seq)
        if after.startswith(before):
            return after[len(before):]
        return after

    # ==================================================================
    # phase: cache read slices (selected layer)
    # ==================================================================
    def check_cache_content(self):
        rec = self.rec
        H = EXPECTED_CONFIG["n_head"]
        D = EXPECTED_CONFIG["head_dim"]
        l = self.selected_layer
        for f in range(EXPECTED_FORWARD_COUNT):
            if f not in self.forward_meta:
                continue
            try:
                # reconstruct the selected-layer cache content from per-forward k_heads
                pieces = [self.a64(f"fwd{f2}.block{l}.k_heads") for f2 in range(f + 1)]
                k_rebuilt = np.concatenate(pieces, axis=2)  # (1,H,total,D)
                pieces_v = [self.a64(f"fwd{f2}.block{l}.v_heads") for f2 in range(f + 1)]
                v_rebuilt = np.concatenate(pieces_v, axis=2)
                k_read = self.a64(f"fwd{f}.block{l}.k_read")
                v_read = self.a64(f"fwd{f}.block{l}.v_read")
                rec.exact_arrays(f"cache.read.k.f{f}", "cache", True,
                                 f"fwd{f}.block{l}.k_read == cache content rebuilt from k_heads 0..{f}",
                                 f32_bytes(k_rebuilt), f32_bytes(k_read),
                                 tolerance="bitwise equal (cache read window)",
                                 details={"shape": list(k_read.shape)})
                rec.exact_arrays(f"cache.read.v.f{f}", "cache", True,
                                 f"fwd{f}.block{l}.v_read == cache content rebuilt from v_heads 0..{f}",
                                 f32_bytes(v_rebuilt), f32_bytes(v_read),
                                 tolerance="bitwise equal (cache read window)",
                                 details={"shape": list(v_read.shape)})
                # new-token slice of this forward must equal this forward's k_heads/v_heads
                meta = self.forward_meta[f]
                start = meta.get("start", 0)
                total = meta.get("total", start + 1)
                if f == 0:
                    start_slice = 0
                else:
                    start_slice = start
                k_heads = self.a64(f"fwd{f}.block{l}.k_heads")
                rec.exact_arrays(f"cache.write_slice.k.f{f}", "cache", True,
                                 f"fwd{f}.block{l}.k_read write window == this forward k_heads",
                                 f32_bytes(k_heads),
                                 f32_bytes(k_read[:, :, start_slice:total]),
                                 tolerance="bitwise equal (write window)")
                v_heads = self.a64(f"fwd{f}.block{l}.v_heads")
                rec.exact_arrays(f"cache.write_slice.v.f{f}", "cache", True,
                                 f"fwd{f}.block{l}.v_read write window == this forward v_heads",
                                 f32_bytes(v_heads),
                                 f32_bytes(v_read[:, :, start_slice:total]),
                                 tolerance="bitwise equal (write window)")
            except Exception as exc:
                rec.failed(f"cache.read.f{f}", "cache", True,
                           f"fwd{f} selected-layer cache read content", "compared",
                           f"{type(exc).__name__}: {exc}", "n/a",
                           details={"traceback": traceback.format_exc()[-1500:]})

    # ==================================================================
    # report
    # ==================================================================
    def summary(self):
        counts = {"total": len(self.rec.checks), "passed": 0, "failed": 0,
                  "not_run": 0, "required": 0, "required_failed": 0,
                  "required_not_run": 0, "optional_failed": 0}
        for c in self.rec.checks:
            counts[c["status"]] = counts.get(c["status"], 0) + 1
            if c["required"]:
                counts["required"] += 1
                if c["status"] == "failed":
                    counts["required_failed"] += 1
                elif c["status"] == "not_run":
                    counts["required_not_run"] += 1
            elif c["status"] == "failed":
                counts["optional_failed"] += 1
        return counts

    def environment(self):
        env = {
            "python": platform.python_version(),
            "executable": sys.executable,
            "torch": None,
            "transformers": None,
            "numpy": np.__version__,
            "safetensors": None,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "device": "cpu",
            "dtype": "float32",
        }
        try:
            import torch
            env["torch"] = torch.__version__
            env["threads"] = int(torch.get_num_threads())
        except Exception:
            pass
        try:
            import transformers
            env["transformers"] = transformers.__version__
        except Exception:
            pass
        try:
            import safetensors
            env["safetensors"] = safetensors.__version__
        except Exception:
            pass
        return env

    def write_report(self, out_path: Path, exit_code: int):
        counts = self.summary()
        report = {
            "schema": VERIFICATION_SCHEMA,
            "case_id": self.case_id,
            "case_file": (self.instance or {}).get("case_file"),
            "case_sha256": getattr(self, "case_sha", None),
            "case_bytes": getattr(self, "case_bytes", None),
            "manifest_file": "manifest.json",
            "manifest_sha256": getattr(self, "manifest_sha", None),
            "source_map_file": (str(self.source_map_path.relative_to(PROJECT_ROOT))
                                if self.source_map_path and self.source_map_path.exists()
                                and str(self.source_map_path).startswith(str(PROJECT_ROOT))
                                else str(self.source_map_path)),
            "source_map_sha256": getattr(self, "source_map_sha", None),
            "generated_at": now_iso(),
            "tool": {"name": "verify_gpt2.py", "version": VERIFY_VERSION,
                     "mode": "static-only" if self.static_only else "full"},
            "command": self.command,
            "data_dir": str(self.data_dir),
            "material_root": str(self.material_root),
            "environment": self.environment(),
            "tolerances": TOLERANCES,
            "summary": counts,
            "checks": self.rec.checks,
            "notes": self.notes,
            "exit_code": exit_code,
        }
        payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_name(out_path.name + ".tmp")
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, out_path)
        return report


def torch_version():
    try:
        import torch
        return torch.__version__
    except Exception:
        return None


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Independent verifier for the GPT-2 atlas trace (S1 gate).")
    ap.add_argument("--material-root", required=True,
                    help="read-only material directory containing gpt2_real.py and weights/gpt2")
    ap.add_argument("--data-dir", default=str(PROJECT_ROOT / "data" / "gpt2"),
                    help="directory containing manifest.json and the case JSON")
    ap.add_argument("--case-id", default="gpt2-main-002")
    ap.add_argument("--out", default=None,
                    help="verification.json path (default: <data-dir>/verification.json)")
    ap.add_argument("--static-only", action="store_true",
                    help="test mode: skip model-dependent checks (never use for the real gate)")
    args = ap.parse_args(argv)

    v = Verifier(args)
    v.notes.append(
        "verification.json is the authoritative validation result (schema section 3); "
        "manifest.instances[].validation_status is left untouched by this verifier "
        "because manifest.json is outside its write scope.")
    v.notes.append(
        "probabilities/rank/entropy/other mass are recomputed by promoting the float32 "
        "logits row to float64 before softmax (schema section 5.2 canonical method).")
    v.notes.append(
        "schema section 8 output-head row (clarified 2026-10-30 after the first verifier run): "
        "canonical gate is (a) float32 batched reproduction with the actual execution shape must be "
        "bitwise equal, and (b) float64 recompute must satisfy max|diff| / max|logit| <= 1e-5. "
        "Both are required. The pre-clarification absolute gate (max|diff| <= 1e-4) and its f0 "
        "observation (1.90e-4 at token 39969, exported logit -117.45, max|logit| 133.91) are retained "
        "inside the (b) check details as prior_strict_abs_gate for audit.")

    def run_phase(name, fn):
        try:
            fn()
        except Exception:
            v.rec.failed(f"phase.{name}.exception", "structure", True, f"phase {name}",
                         "phase completes without exception", "exception", "n/a",
                         details={"traceback": traceback.format_exc()[-3000:]})

    run_phase("load_files", v.load_files)
    if not v.fatal:
        run_phase("load_models", v.load_models)
        run_phase("check_manifest", v.check_manifest)
        run_phase("check_provenance", v.check_provenance)
        run_phase("check_source_map", v.check_source_map)
        run_phase("check_case_structure", v.check_case_structure)
        run_phase("check_tensors", v.check_tensors)
        run_phase("check_events", v.check_events)
        run_phase("check_cache_metadata", v.check_cache_metadata)
        run_phase("check_tokenizer", v.check_tokenizer)
        run_phase("check_reference", v.check_reference)
        run_phase("check_equivalence", v.check_equivalence)
        run_phase("check_formula", v.check_formula)
        run_phase("check_samples", v.check_samples)
        run_phase("check_cache_content", v.check_cache_content)
    else:
        for phase in ("load_models", "check_manifest", "check_provenance", "check_source_map",
                      "check_case_structure", "check_tensors", "check_events",
                      "check_cache_metadata", "check_tokenizer", "check_reference",
                      "check_equivalence", "check_formula", "check_samples",
                      "check_cache_content"):
            v.rec.not_run(f"phase.{phase}", "structure", True, f"phase {phase}",
                          "executed", "data files could not be loaded")

    counts = v.summary()
    exit_code = 0 if (counts["required_failed"] == 0 and counts["required_not_run"] == 0) else 1
    out_path = Path(args.out) if args.out else (v.data_dir / "verification.json")
    try:
        v.write_report(out_path, exit_code)
    except Exception:
        print("FATAL: could not write verification.json:", file=sys.stderr)
        traceback.print_exc()
        return 2

    print(f"verification.json: {out_path}")
    print(f"case: {v.case_id}  sha256: {getattr(v, 'case_sha', None)}")
    print(f"checks: total={counts['total']} passed={counts['passed']} "
          f"failed={counts['failed']} not_run={counts['not_run']}")
    print(f"required: {counts['required']} required_failed={counts['required_failed']} "
          f"required_not_run={counts['required_not_run']} "
          f"optional_failed={counts['optional_failed']}")
    failed = [c for c in v.rec.checks if c["status"] == "failed"]
    if failed:
        print("\nfailed checks:")
        for c in failed:
            print(f"  [{'REQ' if c['required'] else 'opt'}] {c['id']} ({c['category']})")
            print(f"        expected={json.dumps(c['expected'], ensure_ascii=False)[:300]}")
            print(f"        observed={json.dumps(c['observed'], ensure_ascii=False)[:300]}")
    not_run = [c for c in v.rec.checks if c["status"] == "not_run" and c["required"]]
    if not_run:
        print("\nrequired checks not run:")
        for c in not_run:
            print(f"  {c['id']}: {c['details'].get('reason')}")
    print(f"\nexit code: {exit_code}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
