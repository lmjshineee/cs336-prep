"""gpt2_real.py 源码快照与操作映射表。

S1 数据契约要求：轨迹中的每个张量都能追溯到快照源码的具体行区间。
本模块维护稳定的符号/操作 ID 到行号区间的映射，并在导出时：

1. 复制材料目录中的 ``gpt2_real.py`` 到 ``sources/gpt2/gpt2_real.py``（原文，不改写）；
2. 校验每个行区间落在快照内；
3. 计算行区间原文的 SHA-256（``span_sha256``），写入 ``source-map.json``。

源码文件变化时，本表必须显式更新（行号与哈希会一起变化），否则导出失败，
避免把旧映射标到新源码上。

当前模型计算来自项目内 sources/gpt2/gpt2_real.py；仅修正了生成终止条件。
外部材料保持只读，上游与项目来源哈希分别记录于 manifest。
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

SCHEMA = "atlas.source-map/1"
SOURCE_FILE = "gpt2_real.py"
MAP_VERSION = "gpt2-real@2"

# 符号：稳定的代码单元名 -> 行区间（含首尾，1-based）。
SYMBOLS: dict[str, tuple[int, int]] = {
    "GPT2Config": (49, 72),
    "GPT2Config.from_json": (61, 72),
    "KVCache": (78, 93),
    "KVCache.__init__": (84, 90),
    "KVCache.memory_bytes": (92, 93),
    "CausalSelfAttention": (99, 151),
    "CausalSelfAttention.forward": (114, 151),
    "MLP": (157, 165),
    "MLP.forward": (164, 165),
    "Block": (171, 186),
    "Block.forward": (179, 186),
    "GPT2": (192, 248),
    "GPT2.forward": (204, 223),
    "GPT2.attention_maps": (226, 239),
    "GPT2.from_pretrained": (241, 248),
    "load_weights": (260, 319),
    "load_tokenizer": (322, 330),
    "load_gpt2": (333, 342),
    "_next_token": (348, 356),
    "generate_naive": (359, 369),
    "generate_cached": (372, 396),
}

# 操作：轨迹张量/事件使用的稳定操作 ID。
OPERATIONS: dict[str, dict] = {
    "gpt2.embed": {"symbol": "GPT2.forward", "lines": (214, 215)},
    "gpt2.block.forward": {"symbol": "Block.forward", "lines": (179, 186)},
    "gpt2.block.ln1": {"symbol": "Block.forward", "lines": (184, 184)},
    "gpt2.block.attn": {"symbol": "Block.forward", "lines": (184, 184)},
    "gpt2.block.residual_attn": {"symbol": "Block.forward", "lines": (184, 184)},
    "gpt2.block.ln2": {"symbol": "Block.forward", "lines": (185, 185)},
    "gpt2.block.residual_mlp": {"symbol": "Block.forward", "lines": (185, 185)},
    "gpt2.attn.qkv_linear": {"symbol": "CausalSelfAttention.forward", "lines": (122, 122)},
    "gpt2.attn.split": {"symbol": "CausalSelfAttention.forward", "lines": (122, 122)},
    "gpt2.attn.view_transpose": {"symbol": "CausalSelfAttention.forward", "lines": (123, 125)},
    "gpt2.cache.write": {"symbol": "CausalSelfAttention.forward", "lines": (132, 133)},
    "gpt2.cache.read_window": {"symbol": "CausalSelfAttention.forward", "lines": (130, 135)},
    "gpt2.attn.scores": {"symbol": "CausalSelfAttention.forward", "lines": (137, 137)},
    "gpt2.attn.mask": {"symbol": "CausalSelfAttention.forward", "lines": (139, 144)},
    "gpt2.attn.softmax": {"symbol": "CausalSelfAttention.forward", "lines": (146, 146)},
    "gpt2.attn.weighted_v": {"symbol": "CausalSelfAttention.forward", "lines": (150, 150)},
    "gpt2.attn.out_proj": {"symbol": "CausalSelfAttention.forward", "lines": (151, 151)},
    "gpt2.mlp.fc": {"symbol": "MLP.forward", "lines": (165, 165)},
    "gpt2.mlp.act": {"symbol": "MLP.forward", "lines": (165, 165)},
    "gpt2.mlp.proj": {"symbol": "MLP.forward", "lines": (165, 165)},
    "gpt2.ln_f": {"symbol": "GPT2.forward", "lines": (223, 223)},
    "gpt2.lm_head": {"symbol": "GPT2.forward", "lines": (223, 223)},
    "gpt2.kv_cache.alloc": {"symbol": "KVCache.__init__", "lines": (84, 90)},
    "gpt2.kv_cache.memory": {"symbol": "KVCache.memory_bytes", "lines": (92, 93)},
    "gpt2.generate.prefill": {"symbol": "generate_cached", "lines": (386, 387)},
    "gpt2.generate.sample": {"symbol": "generate_cached", "lines": (390, 390)},
    "gpt2.generate.last_row_slice": {"symbol": "generate_cached", "lines": (390, 390)},
    "gpt2.generate.decode": {"symbol": "generate_cached", "lines": (394, 395)},
    "gpt2.generate.loop": {"symbol": "generate_cached", "lines": (389, 395)},
    "gpt2.generate.continue": {
        "symbol": "generate_cached",
        "lines": (394, 395),
        "note": "仅在还需生成下一个 token 时执行 decode；达到上限即停止。",
    },
}


def _split_lines(text: str) -> list[str]:
    return text.splitlines(keepends=True)


def span_text(text: str, start: int, end: int) -> str:
    """返回 1-based 闭区间行文本；越界即抛错。"""
    lines = _split_lines(text)
    if start < 1 or end < start or end > len(lines):
        raise ValueError(f"行区间越界: [{start}, {end}]，文件只有 {len(lines)} 行")
    return "".join(lines[start - 1:end])


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_source_map(source_text: str) -> dict:
    """基于快照文本构建 source-map.json 内容，并校验所有区间。"""
    source_bytes = source_text.encode("utf-8")
    symbols: dict[str, dict] = {}
    for name, (start, end) in SYMBOLS.items():
        snippet = span_text(source_text, start, end)
        symbols[name] = {"lines": [start, end], "span_sha256": _sha256(snippet.encode("utf-8"))}

    operations: dict[str, dict] = {}
    for op_id, entry in OPERATIONS.items():
        start, end = entry["lines"]
        snippet = span_text(source_text, start, end)
        item = {
            "symbol": entry["symbol"],
            "lines": [start, end],
            "span_sha256": _sha256(snippet.encode("utf-8")),
        }
        if "note" in entry:
            item["note"] = entry["note"]
        operations[op_id] = item

    return {
        "schema": SCHEMA,
        "source_file": SOURCE_FILE,
        "source_sha256": _sha256(source_bytes),
        "map_version": MAP_VERSION,
        "symbols": symbols,
        "operations": operations,
    }


def write_snapshot(material_source: Path, snapshot_path: Path) -> tuple[str, int]:
    """把材料源码原文复制为只读快照，返回 (sha256, 字节数)。不修改材料文件。"""
    material_source = Path(material_source)
    raw = material_source.read_bytes()
    snapshot_path = Path(snapshot_path)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = snapshot_path.with_name(snapshot_path.name + ".tmp")
    tmp_path.write_bytes(raw)
    os.replace(tmp_path, snapshot_path)  # 覆盖旧快照时不受旧文件只读属性影响
    snapshot_path.chmod(0o444)
    return _sha256(raw), len(raw)


def snapshot_readme(source_sha256: str, map_sha256: str) -> str:
    return (
        "# gpt2_real.py 源码快照\n\n"
        "- 来源：项目内 `sources/gpt2/gpt2_real.py`；从外部材料派生，外部材料保持只读（本机路径见 `data/gpt2/manifest.json` 的 `tool.command`；只用于复现，不进入页面流程）。\n"
        f"- 快照 SHA-256：`{source_sha256}`\n"
        f"- 映射文件：`source-map.json`（SHA-256 `{map_sha256}`）\n"
        "- 说明：仅调整生成终止条件：最后一次选词后不再前向；上游原文件哈希另见 manifest 的 implementation.upstream_sha256。\n"
        "- 行号映射的时效性由 `source-map.json` 中每个区间的 `span_sha256` 保证。\n"
    )
