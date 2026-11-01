#!/usr/bin/env python3
"""Read-only LLaMA material audit (L0).

Freezes the source/weights/config/tokenizer identity and checks that the current
llama_steps files are the reviewed post-fix versions. It does not load torch,
does not run the model, does not write into the material directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from tokenizers import Tokenizer

PROMPT = "The capital of France is"
EXPECTED_PROMPT_IDS = [504, 3575, 282, 4649, 314]
EXPECTED_CONFIG = {
    "architectures": ["LlamaForCausalLM"],
    "hidden_size": 576,
    "intermediate_size": 1536,
    "num_hidden_layers": 30,
    "num_attention_heads": 9,
    "num_key_value_heads": 3,
    "vocab_size": 49152,
    "rms_norm_eps": 1e-5,
    "rope_theta": 100000,
    "rope_interleaved": False,
    "tie_word_embeddings": True,
    "max_position_embeddings": 8192,
    "torch_dtype": "bfloat16",
}
SOURCE_FILES = [
    "llama_steps/step06_model.py",
    "llama_steps/step09_real_llama.py",
    "llama_steps/step08_parity.py",
    "llama_steps/tests/test_review_regressions.py",
]
CONCEPT_FILES = [
    "llama_steps/step01_rmsnorm.py",
    "llama_steps/step02_swiglu.py",
    "llama_steps/step03_rope.py",
    "llama_steps/step04_gqa.py",
    "llama_steps/step05_block.py",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_safetensors_header(path: Path) -> dict:
    with path.open("rb") as fh:
        header_len = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(header_len).decode("utf-8"))


def audit(material_root: Path, applied_path: Path | None) -> dict:
    llama = material_root / "llama_steps"
    weights = llama / "weights" / "smollm2-135m"
    checks: list[dict] = []

    def check(name: str, passed: bool, observed, expected=None) -> None:
        checks.append({
            "id": name,
            "status": "passed" if passed else "failed",
            "observed": observed,
            "expected": expected,
        })

    files = {}
    for rel in SOURCE_FILES + CONCEPT_FILES:
        path = material_root / rel
        files[rel] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    for rel in ["config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json", "model.safetensors"]:
        path = weights / rel
        files[f"llama_steps/weights/smollm2-135m/{rel}"] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    config = json.loads((weights / "config.json").read_text())
    for key, expected in EXPECTED_CONFIG.items():
        check(f"config.{key}", config.get(key) == expected, config.get(key), expected)

    header = read_safetensors_header(weights / "model.safetensors")
    metadata = header.pop("__metadata__", None)
    dtypes = Counter(item["dtype"] for item in header.values())
    elements = 0
    for item in header.values():
        count = 1
        for dim in item["shape"]:
            count *= dim
        elements += count
    check("weights.tensor_count", len(header) == 272, len(header), 272)
    check("weights.all_bf16", dict(dtypes) == {"BF16": 272}, dict(dtypes), {"BF16": 272})
    check("weights.parameter_elements", elements == 134515008, elements, 134515008)
    check("weights.no_lm_head", not any("lm_head" in key for key in header), [key for key in header if "lm_head" in key], [])
    check("weights.tie_note", metadata == {"format": "pt"}, metadata, {"format": "pt"})

    tokenizer = Tokenizer.from_file(str(weights / "tokenizer.json"))
    encoded = tokenizer.encode(PROMPT)
    check("tokenizer.vocab_size", tokenizer.get_vocab_size() == 49152, tokenizer.get_vocab_size(), 49152)
    check("tokenizer.prompt_ids", encoded.ids == EXPECTED_PROMPT_IDS, encoded.ids, EXPECTED_PROMPT_IDS)
    check("tokenizer.prompt_tokens", encoded.tokens == ["The", "Ġcapital", "Ġof", "ĠFrance", "Ġis"], encoded.tokens,
          ["The", "Ġcapital", "Ġof", "ĠFrance", "Ġis"])
    tokenizer_config = json.loads((weights / "tokenizer_config.json").read_text())
    check("tokenizer.bos_eos_config", config.get("bos_token_id") == 0 and config.get("eos_token_id") == 0,
          [config.get("bos_token_id"), config.get("eos_token_id")], [0, 0])
    check("tokenizer.has_endoftext", tokenizer_config.get("added_tokens_decoder", {}).get("0", {}).get("content") == "<|endoftext|>",
          tokenizer_config.get("added_tokens_decoder", {}).get("0", {}).get("content"), "<|endoftext|>")

    applied = None
    applied_checks = []
    if applied_path and applied_path.exists():
        applied = json.loads(applied_path.read_text())
        for rel, expected_hash in applied.get("files", {}).items():
            if rel in files:
                actual = files[rel]["sha256"]
                applied_checks.append({
                    "file": rel,
                    "expected_sha256": expected_hash,
                    "actual_sha256": actual,
                    "match": actual == expected_hash,
                })
        check("sources.match_applied_fix", all(item["match"] for item in applied_checks),
              sum(item["match"] for item in applied_checks), len(applied_checks))

    failed = [item for item in checks if item["status"] != "passed"]
    return {
        "schema": "atlas.llama-material-audit/1",
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "material_root": str(material_root),
        "prompt": PROMPT,
        "expected_prompt_ids": EXPECTED_PROMPT_IDS,
        "config": config,
        "weights": {
            "tensor_count": len(header),
            "dtypes": dict(dtypes),
            "parameter_elements": elements,
            "has_lm_head": any("lm_head" in key for key in header),
            "safetensors_metadata": metadata,
        },
        "tokenizer": {
            "vocab_size": tokenizer.get_vocab_size(),
            "prompt_ids": encoded.ids,
            "prompt_tokens": encoded.tokens,
        },
        "files": files,
        "applied_fix": {"path": str(applied_path) if applied_path else None, "checks": applied_checks},
        "checks": checks,
        "failed": failed,
        "ok": not failed,
        "notes": [
            "只读审计：未加载 torch、未运行模型、未写材料目录。",
            "当前文件哈希与 output/qa/llama-fix/applied.json 的修正后哈希一致，未使用修正前片段。",
            "权重文件是本地副本；上游 main 链接可变，文件 SHA-256 才是载荷身份。",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--material-root", default="/Users/yunni/Projects/temp/cs336-prep-code")
    parser.add_argument("--applied", default="")
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    root = Path(args.material_root)
    applied = Path(args.applied) if args.applied else None
    if applied is None:
        candidate = Path("output/qa/llama-fix/applied.json")
        applied = candidate if candidate.exists() else None
    result = audit(root, applied)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n")
        print(f"wrote {out}")
    else:
        print(text)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
