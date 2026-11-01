#!/usr/bin/env python3
"""Copy the frozen LLaMA sources and build atlas.source-map/2.

Only reads the material directory; writes snapshots/map under sources/llama.
No model loading, no training, no network.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
from pathlib import Path

SOURCE_FILES = {
    "step06_model.py": "model",
    "step09_real_llama.py": "real_loader",
}

WANTED_SYMBOLS = {
    "RMSNorm", "RMSNorm.forward", "SwiGLU", "SwiGLU.forward",
    "Attention", "Attention.forward", "LlamaBlock", "LlamaBlock.forward",
    "MiniLlama", "MiniLlama.forward", "KVCache", "KVCache.append",
    "build_rope_cache", "apply_rope", "repeat_kv", "rotate_half",
    "load_config", "hf_to_ours", "generate", "assert_same_generation",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def split_keepends(text: str) -> list[str]:
    return text.splitlines(keepends=True)


def span_text(text: str, start: int, end: int) -> str:
    lines = split_keepends(text)
    if start < 1 or end < start or end > len(lines):
        raise ValueError(f"line range out of bounds: [{start}, {end}], file has {len(lines)} lines")
    return "".join(lines[start - 1:end])


def span_sha(text: str, start: int, end: int) -> str:
    return sha256_bytes(span_text(text, start, end).encode("utf-8"))


def find_line(text: str, needle: str, *, after: int = 1) -> int:
    lines = text.splitlines()
    for idx, line in enumerate(lines, 1):
        if idx >= after and needle in line:
            return idx
    raise ValueError(f"needle not found: {needle}")


def symbol_spans(text: str) -> dict[str, tuple[int, int]]:
    """Top-level functions/classes and their methods."""
    tree = ast.parse(text)
    result: dict[str, tuple[int, int]] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            result[node.name] = (node.lineno, node.end_lineno)
        elif isinstance(node, ast.ClassDef):
            result[node.name] = (node.lineno, node.end_lineno)
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    result[f"{node.name}.{child.name}"] = (child.lineno, child.end_lineno)
    return result


def single_line(text: str, needle: str) -> tuple[int, int]:
    line = find_line(text, needle)
    return (line, line)


def operation_spans(text: str) -> dict[str, tuple[int, int]]:
    sym = symbol_spans(text)
    ops = {
        "llama.rmsnorm": sym["RMSNorm.forward"],
        "llama.rope.build": sym["build_rope_cache"],
        "llama.rope.apply_q": sym["apply_rope"],
        "llama.rope.apply_k": sym["apply_rope"],
        "llama.attn.q_proj": single_line(text, "q = self.q_proj(x)"),
        "llama.attn.k_proj": single_line(text, "k = self.k_proj(x)"),
        "llama.attn.v_proj": single_line(text, "v = self.v_proj(x)"),
        "llama.attn.v_heads": single_line(text, "v = self.v_proj(x)"),
        "llama.cache.append": sym["KVCache.append"],
        "llama.gqa.repeat_kv": single_line(text, "k = repeat_kv(k"),
        "llama.attn.scores": single_line(text, "scores = q @ k"),
        "llama.attn.mask": single_line(text, "scores = scores.masked_fill"),
        "llama.attn.softmax": single_line(text, "weights = F.softmax"),
        "llama.attn.o_proj": single_line(text, "return self.o_proj(out)"),
        "llama.swiglu.gate": single_line(text, "self.down_proj(F.silu(self.gate_proj(x))"),
        "llama.swiglu.up": single_line(text, "self.down_proj(F.silu(self.gate_proj(x))"),
        "llama.swiglu.mul": single_line(text, "self.down_proj(F.silu(self.gate_proj(x))"),
        "llama.swiglu.down": single_line(text, "self.down_proj(F.silu(self.gate_proj(x))"),
        "llama.block.forward": sym["LlamaBlock.forward"],
        "llama.block.residual_attn": single_line(text, "x = x + self.attn"),
        "llama.embed": single_line(text, "x = self.token_embedding(idx)"),
        "llama.lm_head": single_line(text, "return self.lm_head(self.norm(x))"),
    }
    return ops


def build_map(material_root: Path, source_dir: Path, source_root: Path | None = None) -> dict:
    sources = {}
    texts = {}
    for name, role in SOURCE_FILES.items():
        src = (source_root or material_root / "llama_steps") / name
        raw = src.read_bytes()
        snapshot = source_dir / name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        tmp = snapshot.with_name(name + ".tmp")
        tmp.write_bytes(raw)
        os.replace(tmp, snapshot)
        snapshot.chmod(0o444)
        texts[name] = raw.decode("utf-8")
        sources[name] = {"sha256": sha256_bytes(raw), "snapshot": f"sources/llama/{name}", "role": role}

    symbols = {}
    spans_by_file = {}
    for name, text in texts.items():
        spans_by_file[name] = symbol_spans(text)
        for symbol, (start, end) in spans_by_file[name].items():
            if symbol not in WANTED_SYMBOLS:
                continue
            if symbol in symbols:
                raise ValueError(f"duplicate symbol name across sources: {symbol}")
            symbols[symbol] = {
                "file": name,
                "lines": [start, end],
                "span_sha256": span_sha(text, start, end),
            }

    operations = {}
    for name, text in texts.items():
        if name != "step06_model.py":
            continue
        for op_id, (start, end) in operation_spans(text).items():
            candidates = [(symbol, (a, b)) for symbol, (a, b) in spans_by_file[name].items() if a <= start <= b]
            enclosing = min(candidates, key=lambda item: item[1][1] - item[1][0])[0] if candidates else ""
            operations[op_id] = {
                "file": name,
                "symbol": enclosing,
                "lines": [start, end],
                "span_sha256": span_sha(text, start, end),
            }

    real_text = texts["step09_real_llama.py"]
    operations.update({
        "llama.generate.prefill": {"file": "step09_real_llama.py", "symbol": "generate", "lines": [find_line(real_text, "logits = model(ids"), find_line(real_text, "logits = model(ids")], "span_sha256": span_sha(real_text, find_line(real_text, "logits = model(ids"), find_line(real_text, "logits = model(ids"))},
        "llama.generate.sample": {"file": "step09_real_llama.py", "symbol": "generate", "lines": [find_line(real_text, "nxt = logits[:, -1, :].argmax"), find_line(real_text, "nxt = logits[:, -1, :].argmax")], "span_sha256": span_sha(real_text, find_line(real_text, "nxt = logits[:, -1, :].argmax"), find_line(real_text, "nxt = logits[:, -1, :].argmax"))},
        "llama.generate.decode": {"file": "step09_real_llama.py", "symbol": "generate", "lines": [find_line(real_text, "logits = model(nxt"), find_line(real_text, "logits = model(nxt")], "span_sha256": span_sha(real_text, find_line(real_text, "logits = model(nxt"), find_line(real_text, "logits = model(nxt"))},
    })
    # Sanity: every operation is inside its declared symbol and file.
    for op_id, entry in operations.items():
        text = texts[entry["file"]]
        assert span_sha(text, *entry["lines"]) == entry["span_sha256"], op_id

    return {
        "schema": "atlas.source-map/2",
        "map_version": "smollm2-main@3",
        "sources": sources,
        "symbols": symbols,
        "operations": operations,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--material-root", default="/Users/yunni/Projects/temp/cs336-prep-code")
    parser.add_argument("--source-dir", default="sources/llama")
    parser.add_argument("--source-root", default="sources/llama")
    parser.add_argument("--output", default="sources/llama/source-map.json")
    args = parser.parse_args()
    material_root = Path(args.material_root)
    source_dir = Path(args.source_dir)
    result = build_map(material_root, source_dir, Path(args.source_root))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {out} ({len(result['symbols'])} symbols, {len(result['operations'])} operations)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
