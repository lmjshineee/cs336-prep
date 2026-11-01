#!/usr/bin/env python3
"""S1：从真实 GPT-2 实现导出一条可核对、可回放的最小轨迹。

数据契约见 ``docs/TRACE_SCHEMA.md``。本工具只读材料目录，不修改原实现、权重或实验记录；
所有采集通过「与源码逐行等价的观察包装」完成，并在写出前证明：

- 采集运行与未采集运行的每个 forward logits 逐位相等；
- greedy 生成序列与原始 ``generate_cached`` 完全一致；
- 载荷 base64 往返、shape/axes、哈希、缓存区间、事件顺序全部自检通过。

失败时不发布半份轨迹：只写 ``export-failure.json``，不更新 manifest。

用法（在项目根目录，使用材料环境自带的 venv）：

    /Users/yunni/Projects/temp/cs336-prep-code/.venv/bin/python \\
        GPT2-LLaMA-Atlas/tools/trace/export_gpt2.py

默认材料根目录：``/Users/yunni/Projects/temp/cs336-prep-code/real_llm_lab``
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import tempfile
import time
import traceback
import types
from datetime import datetime
from pathlib import Path

sys.dont_write_bytecode = True
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
DEFAULT_MATERIAL_ROOT = Path("/Users/yunni/Projects/temp/cs336-prep-code/real_llm_lab")
DEFAULT_CASE_ID = "gpt2-main-002"
TOOL_VERSION = "1.1.0"
MANIFEST_SCHEMA = "atlas.trace/1"
CASE_SCHEMA = "atlas.trace.case/1"
EXPECTED_CACHE_BYTES = 75_497_472  # 2 * 12 层 * 1 * 12 头 * 1024 位置 * 64 维 * 4 字节
CASE_SIZE_TARGET = 6 * 1024 * 1024
CASE_SIZE_MAX = 10 * 1024 * 1024

# 骨架导入在 main 中完成（先设置 dont_write_bytecode / 离线环境）。
torch = None  # type: ignore[assignment]
F = None  # type: ignore[assignment]
gpt2_real = None  # type: ignore[assignment]

RECORDED_AT = datetime.now().astimezone().isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def json_bytes(obj: object) -> bytes:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def write_bytes_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp-", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def write_failure(data_dir: Path, case_id: str, stage: str, exc: BaseException, args: argparse.Namespace) -> None:
    record = {
        "schema": "atlas.trace.failure/1",
        "case_id": case_id,
        "stage": stage,
        "error": f"{type(exc).__name__}: {exc}",
        "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "generated_at": RECORDED_AT,
        "published": False,
        "note": "导出失败，未写入案例或 manifest；修复后重跑。",
    }
    write_bytes_atomic(data_dir / "export-failure.json", json_bytes(record))


# ---------------------------------------------------------------------------
# GPT-2 字节级 BPE：token 字符串 -> 原始字节
# ---------------------------------------------------------------------------
def bytes_to_unicode() -> dict[int, str]:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


def token_bytes(raw_token: str, unicode_to_byte: dict[str, int]) -> bytes:
    try:
        return bytes(unicode_to_byte[ch] for ch in raw_token)
    except KeyError as exc:  # 反射字符不在 GPT-2 映射表内，说明 tokenizer 与预期不符
        raise ValueError(f"token 含非 GPT-2 字节映射字符: {raw_token!r}") from exc


# ---------------------------------------------------------------------------
# 载荷编码
# ---------------------------------------------------------------------------
class PayloadError(RuntimeError):
    pass


def encode_payload(tensor, encoding: str) -> dict:
    import numpy as np

    if encoding == "f32le-base64":
        array = np.ascontiguousarray(tensor.detach().to("cpu", torch.float32).numpy(), dtype="<f4")
        if not np.isfinite(array).all():
            raise PayloadError("载荷含 NaN/Inf，拒绝写出")
    elif encoding == "u8-base64":
        array = np.ascontiguousarray(tensor.detach().to("cpu", torch.uint8).numpy(), dtype="u1")
    else:
        raise PayloadError(f"未知载荷编码: {encoding}")

    raw = array.tobytes(order="C")
    # 往返自检：解码回来必须与原始字节完全一致。
    rebuilt = np.frombuffer(base64.b64decode(b64(raw)), dtype=array.dtype)
    if rebuilt.tobytes() != raw:
        raise PayloadError("载荷 base64 往返不一致")
    return {
        "encoding": encoding,
        "nbytes": len(raw),
        "sha256": sha256_bytes(raw),
        "data": b64(raw),
    }


# ---------------------------------------------------------------------------
# 张量角色表
# ---------------------------------------------------------------------------
C = 768
F_ = 3072
QKV = 2304
V_ = 50257
H_ = 12
D_ = 64


def role_axes(role: str) -> list[str]:
    table = {
        "attn_weights": ["B", "H", "Tq", "Tk"],
        "x_in": ["B", "T", "C"],
        "ln1_out": ["B", "T", "C"],
        "qkv": ["B", "T", "3C"],
        "q_raw": ["B", "T", "C"],
        "k_raw": ["B", "T", "C"],
        "v_raw": ["B", "T", "C"],
        "q_heads": ["B", "H", "T", "C"],
        "k_heads": ["B", "H", "T", "C"],
        "v_heads": ["B", "H", "T", "C"],
        "k_read": ["B", "H", "Tk", "C"],
        "v_read": ["B", "H", "Tk", "C"],
        "scores_scaled": ["B", "H", "Tq", "Tk"],
        "causal_mask": ["Tq", "Tk"],
        "weighted_v": ["B", "T", "C"],
        "attn_out": ["B", "T", "C"],
        "x_after_attn": ["B", "T", "C"],
        "ln2_out": ["B", "T", "C"],
        "fc_out": ["B", "T", "F"],
        "act_out": ["B", "T", "F"],
        "mlp_out": ["B", "T", "C"],
        "x_out": ["B", "T", "C"],
        "wte_out": ["B", "T", "C"],
        "wpe_out": ["T", "C"],
        "out": ["B", "T", "C"],
        "last_row": ["B", "V"],
    }
    if role not in table:
        raise KeyError(f"未登记的张量角色: {role}")
    return table[role]


def expected_shape(role: str, ctx: dict) -> list[int]:
    B, T, total = ctx["B"], ctx["T"], ctx["total"]
    mapping = {
        "attn_weights": [B, H_, T, total],
        "scores_scaled": [B, H_, T, total],
        "causal_mask": [T, total],
        "q_heads": [B, H_, T, D_],
        "k_heads": [B, H_, T, D_],
        "v_heads": [B, H_, T, D_],
        "k_read": [B, H_, total, D_],
        "v_read": [B, H_, total, D_],
        "qkv": [B, T, QKV],
        "fc_out": [B, T, F_],
        "act_out": [B, T, F_],
        "last_row": [B, V_],
        "wpe_out": [T, C],
    }
    return mapping.get(role, [B, T, C])


ROLE_DTYPE = {"causal_mask": "uint8"}


ROLE_ENCODING = {"causal_mask": "u8-base64"}


# ---------------------------------------------------------------------------
# 采集记录器
# ---------------------------------------------------------------------------
class Recorder:
    def __init__(self, source_map: dict, allocated: dict, selected_layer: int, n_layer: int, tok, prompt_ids: list[int]):
        self.source_map = source_map
        self.allocated = allocated
        self.selected_layer = selected_layer
        self.n_layer = n_layer
        self.tok = tok
        self.prompt_ids = list(prompt_ids)
        self.events: list[dict] = []
        self.tensors: list[dict] = []
        self.seq = 0
        self.current: dict | None = None
        self.pending: list[tuple[str, object, str, str, str, str | None]] = []
        self.last_logits_event: dict | None = None
        self.last_logits: object | None = None
        self.bytes_state: list[bytes] = []
        self.unicode_to_byte: dict[str, int] = {}
        self.samples: list[dict] = []
        self.active_forward_logits: object | None = None

    # -- 源码引用 --------------------------------------------------------
    def source_of(self, op_id: str) -> dict:
        entry = self.source_map["operations"].get(op_id)
        if entry is None:
            raise KeyError(f"source-map 缺少操作: {op_id}")
        return {"symbol": entry["symbol"], "lines": list(entry["lines"]), "span_sha256": entry["span_sha256"]}

    # -- 事件 ------------------------------------------------------------
    def _new_event(self, event_id: str, kind: str, **kw) -> dict:
        event = {"id": event_id, "seq": self.seq, "kind": kind}
        event.update(kw)
        self.seq += 1
        self.events.append(event)
        return event

    def cache_state(self, kind: str, layer: int | None, start_pos: int, T: int, output_position: int | None = None,
                    selected_layer_refs: dict | None = None) -> dict:
        n = self.n_layer
        if kind == "embed":
            effective = [start_pos] * n
            written = [0] * n
            sync = start_pos
            read_window = None
            write_window = None
        elif kind == "block":
            assert layer is not None
            effective = [start_pos + T if i <= layer else start_pos for i in range(n)]
            written = [T if i <= layer else 0 for i in range(n)]
            sync = min(effective)
            read_window = {"start": 0, "end_exclusive": start_pos + T}
            write_window = {"start": start_pos, "end_exclusive": start_pos + T}
        elif kind in ("ln_f", "logits"):
            effective = [start_pos + T] * n
            written = [0] * n
            sync = start_pos + T
            read_window = None
            write_window = None
        elif kind == "sample":
            assert output_position is not None
            effective = [output_position] * n
            written = [0] * n
            sync = output_position
            read_window = None
            write_window = None
        else:
            raise ValueError(f"未知缓存事件类型: {kind}")
        state = {
            "allocated": self.allocated,
            "layers_effective_positions": effective,
            "layers_written_positions": written,
            "synchronized_positions": sync,
            "read_window": read_window,
            "write_window": write_window,
            "selected_layer_read_refs": selected_layer_refs,
            "note": {
                "embed": "嵌入阶段尚未写入任何层缓存",
                "block": "第 0—layer 层已写入本轮 K/V；其余层仍停留在上一轮",
                "ln_f": "本轮各层 K/V 已写入完成",
                "logits": "本轮各层 K/V 已写入完成",
                "sample": "采样只追加序列，不写 KV 缓存",
            }[kind],
        }
        return state

    def begin_forward(self, index: int, phase: str, start_pos: int, T: int, input_ids: list[int]) -> None:
        self.current = {"index": index, "phase": phase, "start_pos": start_pos, "T": T, "input_ids": list(input_ids)}

    def add_tensor(self, event: dict, role: str, tensor, op_id: str, *, scope: str | None = None,
                   derivation: str = "observed", note: str | None = None) -> dict:
        axes = role_axes(role)
        shape = list(tensor.shape)
        if len(shape) != len(axes):
            raise PayloadError(f"{event['id']}.{role}: shape {shape} 与 axes {axes} 维数不符")
        ctx = self.current
        assert ctx is not None
        want = expected_shape(role, {"B": 1, "T": ctx["T"], "total": ctx["start_pos"] + ctx["T"]})
        if shape != want:
            raise PayloadError(f"{event['id']}.{role}: shape {shape} 与固定语义 {want} 不符")
        encoding = ROLE_ENCODING.get(role, "f32le-base64")
        tensor_id = f"{event['id']}.{role}"
        entry = {
            "id": tensor_id,
            "event_id": event["id"],
            "role": role,
            "op": op_id,
            "shape": shape,
            "axes": axes,
            "dtype": ROLE_DTYPE.get(role, "float32"),
            "scope": scope or "full",
            "derivation": derivation,
            "payload": encode_payload(tensor, encoding),
        }
        if note:
            entry["note"] = note
        self.tensors.append(entry)
        event["tensor_ids"].append(tensor_id)
        return entry

    def pending_tensor(self, role: str, tensor, op_id: str, *, scope: str = "full",
                       derivation: str = "observed", note: str | None = None) -> None:
        self.pending.append((role, tensor.detach().clone(), op_id, scope, derivation, note))

    def flush_pending(self, event: dict) -> None:
        for role, tensor, op_id, scope, derivation, note in self.pending:
            self.add_tensor(event, role, tensor, op_id, scope=scope, derivation=derivation, note=note)
        self.pending = []

    # -- 事件构造 --------------------------------------------------------
    def make_embed_event(self) -> dict:
        ctx = self.current
        assert ctx is not None
        f, start_pos, T = ctx["index"], ctx["start_pos"], ctx["T"]
        event_id = f"fwd{f}.embed"
        event = self._new_event(
            event_id, "embed",
            phase=ctx["phase"], forward_index=f,
            position_start=start_pos, position_count=T, total_positions=start_pos + T,
            input_ids=list(ctx["input_ids"]),
            positions={"query": list(range(start_pos, start_pos + T)), "key": []},
            cache=self.cache_state("embed", None, start_pos, T),
            tensor_ids=[], primary_op="gpt2.embed", source=self.source_of("gpt2.embed"),
            replay={"phase_complete": False, "note": "嵌入完成；尚未写任何层缓存"},
        )
        return event

    def make_block_event(self, layer: int, attn_weights) -> dict:
        ctx = self.current
        assert ctx is not None
        f, start_pos, T = ctx["index"], ctx["start_pos"], ctx["T"]
        total = start_pos + T
        event_id = f"fwd{f}.block{layer}"
        refs = None
        if layer == self.selected_layer:
            refs = {"k": f"{event_id}.k_read", "v": f"{event_id}.v_read"}
        event = self._new_event(
            event_id, "block",
            phase=ctx["phase"], forward_index=f, layer=layer,
            position_start=start_pos, position_count=T, total_positions=total,
            input_ids=list(ctx["input_ids"]),
            positions={"query": list(range(start_pos, start_pos + T)), "key": list(range(total))},
            cache=self.cache_state("block", layer, start_pos, T, selected_layer_refs=refs),
            tensor_ids=[], primary_op="gpt2.block.forward", source=self.source_of("gpt2.block.forward"),
            replay={
                "phase_complete": False,
                "note": f"第 {layer} 层执行完成；本轮已写该层 K/V" if layer == self.selected_layer
                        else f"第 {layer} 层执行完成；注意力权重已采集",
            },
        )
        self.add_tensor(event, "attn_weights", attn_weights, "gpt2.attn.softmax", scope="full")
        if layer == self.selected_layer:
            self.flush_pending(event)
        return event

    def make_lnf_event(self, lnf_out) -> dict:
        ctx = self.current
        assert ctx is not None
        f, start_pos, T = ctx["index"], ctx["start_pos"], ctx["T"]
        event = self._new_event(
            f"fwd{f}.ln_f", "ln_f",
            phase=ctx["phase"], forward_index=f,
            position_start=start_pos, position_count=T, total_positions=start_pos + T,
            input_ids=list(ctx["input_ids"]),
            positions={"query": list(range(start_pos, start_pos + T)), "key": []},
            cache=self.cache_state("ln_f", None, start_pos, T),
            tensor_ids=[], primary_op="gpt2.ln_f", source=self.source_of("gpt2.ln_f"),
            replay={"phase_complete": True, "note": "最终 LayerNorm 完成；所有位置都经过输出头"},
        )
        self.add_tensor(event, "out", lnf_out, "gpt2.ln_f")
        return event

    def make_logits_event(self, logits) -> dict:
        ctx = self.current
        assert ctx is not None
        f, start_pos, T = ctx["index"], ctx["start_pos"], ctx["T"]
        full = logits.detach()
        if full.dim() != 3 or full.shape[0] != 1:
            raise PayloadError(f"logits 形状异常: {tuple(full.shape)}")
        if int(full.shape[1]) != T or int(full.shape[2]) != V_:
            raise PayloadError(f"logits 形状与执行事实不符: {tuple(full.shape)}，期望 (1,{T},{V_})")
        last_row = full[:, -1, :].clone()
        event = self._new_event(
            f"fwd{f}.logits", "logits",
            phase=ctx["phase"], forward_index=f,
            position_start=start_pos, position_count=T, total_positions=start_pos + T,
            input_ids=list(ctx["input_ids"]),
            positions={"query": list(range(start_pos, start_pos + T)), "key": []},
            cache=self.cache_state("logits", None, start_pos, T),
            tensor_ids=[], primary_op="gpt2.lm_head", source=self.source_of("gpt2.lm_head"),
            logits_event={
                "full_output_shape": list(full.shape),
                "full_output_dtype": "float32",
                "last_row_index": start_pos + T - 1,
                "used_by_sampling": False,
                "used_by_sample": None,
                "per_position_argmax": [int(x) for x in full.argmax(dim=-1)[0].tolist()],
                "per_position_max_logit": [float(x) for x in full.max(dim=-1).values[0].tolist()],
                "note": "GPT2.forward 返回全部位置 logits；生成函数再取 logits[:, -1, :]。其余位置数值仅在验证中重算核对。",
            },
            replay={"phase_complete": True,
                    "note": "本前向完成；最后一行 logits 可能用于采样，也可能未被使用（见 used_by_sampling）"},
        )
        self.add_tensor(event, "last_row", last_row, "gpt2.generate.last_row_slice", scope="slice")
        self.last_logits_event = event
        self.last_logits = full.clone()
        self.active_forward_logits = full.clone()
        return event

    def make_sample_event(self, sample_index: int, output_position: int) -> dict:
        import torch as _torch

        assert self.last_logits is not None and self.last_logits_event is not None
        row = self.last_logits[:, -1, :].reshape(-1).to(_torch.float64)
        probs = _torch.softmax(row, dim=-1)
        token_id = int(_torch.argmax(row).item())
        logit = float(self.last_logits[:, -1, :].reshape(-1)[token_id].item())
        probability = float(probs[token_id].item())
        rank = int((probs > probs[token_id]).sum().item()) + 1
        topv, topi = _torch.topk(probs, 10)
        top_k = [{"id": int(i), "probability": float(v), "decoded_text": self.tok.decode([int(i)], clean_up_tokenization_spaces=False), "raw_token": self.tok.convert_ids_to_tokens(int(i))} for v, i in zip(topv.tolist(), topi.tolist())]
        other_mass = float(1.0 - topv.sum().item())
        entropy = float(-(probs * _torch.log(probs)).sum().item())

        raw_token = self.tok.convert_ids_to_tokens([token_id])[0]
        token_raw_bytes = token_bytes(raw_token, self.unicode_to_byte)
        try:
            token_raw_bytes.decode("utf-8", errors="strict")
            standalone_complete = True
        except UnicodeDecodeError:
            standalone_complete = False
        prev_bytes = b"".join(self.bytes_state)
        next_bytes = prev_bytes + token_raw_bytes
        try:
            next_text = next_bytes.decode("utf-8")
            prev_text = prev_bytes.decode("utf-8")
            decoded_delta = next_text[len(prev_text):] if next_text.startswith(prev_text) else next_text
        except UnicodeDecodeError:
            decoded_delta = next_bytes.decode("utf-8", errors="replace")
        self.bytes_state.append(token_raw_bytes)

        event_id = f"sample{sample_index}"
        event = self._new_event(
            event_id, "sample",
            phase="sample",
            forward_index=self.last_logits_event["forward_index"],
            sample_index=sample_index,
            output_position=output_position,
            strategy="greedy",
            token_id=token_id,
            raw_token=raw_token,
            decoded_text=decoded_delta,
            standalone_decode_complete=standalone_complete,
            logit=logit,
            probability=probability,
            rank=rank,
            top_k=top_k,
            other_probability_mass=other_mass,
            entropy_nats=entropy,
            probability_basis="对完整 vocab_size 行做 float64 softmax 后统计；top_k 仅展示，未重新归一化",
            cache=self.cache_state("sample", None, 0, 0, output_position=output_position),
            tensor_ids=[], primary_op="gpt2.generate.sample", source=self.source_of("gpt2.generate.sample"),
            replay={"phase_complete": True, "sequence_length_after": output_position + 1,
                    "note": "采样完成；token 已追加到序列，但尚未写入 KV 缓存"},
        )
        self.last_logits_event["logits_event"]["used_by_sampling"] = True
        self.last_logits_event["logits_event"]["used_by_sample"] = event_id
        self.samples.append({"token_id": token_id, "probability": probability, "rank": rank})
        return event


# ---------------------------------------------------------------------------
# 观察包装：与源码逐行等价，只增加 detach().clone() 采集
# ---------------------------------------------------------------------------
def install_capture(model, rec: Recorder):
    import torch.nn.functional as _F

    block = model.h[rec.selected_layer]
    orig_model_forward = model.forward
    orig_block_forward = block.forward
    orig_attn_forward = block.attn.forward
    hook_handles = []

    def observed_attention(self, x, k_cache=None, v_cache=None, start_pos: int = 0, causal: bool = True):
        B, T, Cc = x.shape
        H, D = self.n_head, self.head_dim

        qkv = self.c_attn(x)
        q_split, k_split, v_split = qkv.split(self.n_embd, dim=2)
        q = q_split.view(B, T, H, D).transpose(1, 2)
        k = k_split.view(B, T, H, D).transpose(1, 2)
        v = v_split.view(B, T, H, D).transpose(1, 2)

        if k_cache is None:
            k_all, v_all, total = k, v, T
        else:
            assert v_cache is not None
            total = start_pos + T
            k_cache[:, :, start_pos:total] = k
            v_cache[:, :, start_pos:total] = v
            k_all = k_cache[:, :, :total]
            v_all = v_cache[:, :, :total]

        scores = (q @ k_all.transpose(-2, -1)) / (D ** 0.5)
        scores_pre_mask = scores
        if causal:
            q_pos = torch.arange(start_pos, total, device=x.device).unsqueeze(1)
            k_pos = torch.arange(total, device=x.device).unsqueeze(0)
            mask = k_pos <= q_pos
            scores = scores.masked_fill(~mask, float("-inf"))
        else:
            mask = torch.ones((T, total), dtype=torch.bool, device=x.device)

        weights = _F.softmax(scores, dim=-1)
        if self.record:
            self.last_attn = weights.detach()

        out = (weights @ v_all).transpose(1, 2).contiguous().view(B, T, Cc)
        proj = self.c_proj(out)

        rec.pending_tensor("qkv", qkv, "gpt2.attn.qkv_linear")
        rec.pending_tensor("q_raw", q_split, "gpt2.attn.split")
        rec.pending_tensor("k_raw", k_split, "gpt2.attn.split")
        rec.pending_tensor("v_raw", v_split, "gpt2.attn.split")
        rec.pending_tensor("q_heads", q, "gpt2.attn.view_transpose")
        rec.pending_tensor("k_heads", k, "gpt2.attn.view_transpose")
        rec.pending_tensor("v_heads", v, "gpt2.attn.view_transpose")
        rec.pending_tensor("k_read", k_all, "gpt2.cache.read_window",
                           note="本次注意力实际读取的 k_all（含历史缓存与本次新 K）")
        rec.pending_tensor("v_read", v_all, "gpt2.cache.read_window",
                           note="本次注意力实际读取的 v_all（含历史缓存与本次新 V）")
        rec.pending_tensor("scores_scaled", scores_pre_mask, "gpt2.attn.scores",
                           note="掩码前的有限分数；掩码位置不存 -inf，改用 causal_mask 表达")
        rec.pending_tensor("causal_mask", mask.to(torch.uint8), "gpt2.attn.mask",
                           note="1=可读，0=不可读；对 B/H 广播")
        rec.pending_tensor("weighted_v", out, "gpt2.attn.weighted_v",
                           note="softmax 权重加权 V 后、c_proj 前的输出")
        rec.pending_tensor("attn_out", proj, "gpt2.attn.out_proj")
        return proj

    def observed_block_forward(self, x, k_cache=None, v_cache=None, start_pos: int = 0, causal: bool = True):
        rec.pending_tensor("x_in", x, "gpt2.block.forward")
        ln1 = self.ln_1(x)
        rec.pending_tensor("ln1_out", ln1, "gpt2.block.ln1")
        attn_out = self.attn(ln1, k_cache, v_cache, start_pos, causal)
        x1 = x + attn_out
        rec.pending_tensor("x_after_attn", x1, "gpt2.block.residual_attn")
        ln2 = self.ln_2(x1)
        rec.pending_tensor("ln2_out", ln2, "gpt2.block.ln2")
        mlp_out = self.mlp(ln2)
        rec.pending_tensor("mlp_out", mlp_out, "gpt2.mlp.proj")
        x2 = x1 + mlp_out
        rec.pending_tensor("x_out", x2, "gpt2.block.forward")
        return x2

    def observed_gpt2_forward(self, idx, kv_cache=None, start_pos: int = 0, causal: bool = True):
        B, T = idx.shape
        pos = torch.arange(start_pos, start_pos + T, device=idx.device)
        wte_out = self.wte(idx)
        wpe_out = self.wpe(pos)
        x = wte_out + wpe_out

        embed_event = rec.make_embed_event()
        rec.add_tensor(embed_event, "wte_out", wte_out, "gpt2.embed")
        rec.add_tensor(embed_event, "wpe_out", wpe_out, "gpt2.embed")

        for i, layer_block in enumerate(self.h):
            if kv_cache is None:
                x = layer_block(x, causal=causal)
            else:
                x = layer_block(x, kv_cache.k[i], kv_cache.v[i], start_pos, causal)
            if layer_block.attn.last_attn is None:
                raise RuntimeError(f"第 {i} 层注意力权重未记录，检查 record 开关")
            rec.make_block_event(i, layer_block.attn.last_attn)

        lnf = self.ln_f(x)
        rec.make_lnf_event(lnf)
        logits = self.lm_head(lnf)
        rec.make_logits_event(logits)
        return logits

    def fc_hook(_module, _inputs, output):
        rec.pending_tensor("fc_out", output, "gpt2.mlp.fc")

    def act_hook(_module, _inputs, output):
        rec.pending_tensor("act_out", output, "gpt2.mlp.act")

    hook_handles.append(block.mlp.c_fc.register_forward_hook(fc_hook))
    hook_handles.append(block.mlp.act.register_forward_hook(act_hook))

    model.forward = types.MethodType(observed_gpt2_forward, model)
    block.forward = types.MethodType(observed_block_forward, block)
    block.attn.forward = types.MethodType(observed_attention, block.attn)

    def restore():
        model.forward = orig_model_forward
        block.forward = orig_block_forward
        block.attn.forward = orig_attn_forward
        for handle in hook_handles:
            handle.remove()

    return restore


# ---------------------------------------------------------------------------
# 运行
# ---------------------------------------------------------------------------
def run_plain(model, prompt_ids: list[int], max_new_tokens: int):
    """原始路径：与 generate_cached 相同的调用顺序，记录每个 forward 的 logits。"""
    gpt2_real_model = model
    config = model.config
    idx = torch.tensor([prompt_ids], dtype=torch.long)
    cache = gpt2_real.KVCache(config, 1, torch.device("cpu"), dtype=model.wte.weight.dtype)
    logits_list = []
    sampled = []
    with torch.no_grad():
        logits = gpt2_real_model(idx, kv_cache=cache, start_pos=0)
        logits_list.append(logits.detach().clone())
        pos = idx.size(1)
        for step in range(max_new_tokens):
            nxt = gpt2_real._next_token(logits[:, -1, :], 1.0, None, True)
            sampled.append(int(nxt.item()))
            idx = torch.cat((idx, nxt), dim=1)
            if step + 1 == max_new_tokens or idx.size(1) >= config.block_size:
                break
            logits = gpt2_real_model(nxt, kv_cache=cache, start_pos=pos)
            logits_list.append(logits.detach().clone())
            pos += 1
    return logits_list, sampled, idx


def run_captured(model, prompt_ids: list[int], max_new_tokens: int, rec: Recorder, cache=None):
    config = model.config
    idx = torch.tensor([prompt_ids], dtype=torch.long)
    if cache is None:
        cache = gpt2_real.KVCache(config, 1, torch.device("cpu"), dtype=model.wte.weight.dtype)
    logits_list = []
    sampled = []
    with torch.no_grad():
        rec.begin_forward(0, "prefill", 0, idx.size(1), idx[0].tolist())
        logits = model(idx, kv_cache=cache, start_pos=0)
        logits_list.append(logits.detach().clone())
        pos = idx.size(1)
        for step in range(max_new_tokens):
            rec.make_sample_event(step, pos)
            nxt = gpt2_real._next_token(logits[:, -1, :], 1.0, None, True)
            sampled.append(int(nxt.item()))
            idx = torch.cat((idx, nxt), dim=1)
            if step + 1 == max_new_tokens or idx.size(1) >= config.block_size:
                break
            rec.begin_forward(step + 1, "decode", pos, 1, [int(nxt.item())])
            logits = model(nxt, kv_cache=cache, start_pos=pos)
            logits_list.append(logits.detach().clone())
            pos += 1
    return logits_list, sampled, idx


# ---------------------------------------------------------------------------
# 案例组装
# ---------------------------------------------------------------------------
def prompt_token_table(tok, ids: list[int], text: str) -> tuple[list[dict], dict]:
    unicode_to_byte = {v: k for k, v in bytes_to_unicode().items()}
    tokens = []
    byte_cursor = 0
    raw_bytes_all = b""
    for position, token_id in enumerate(ids):
        raw_token = tok.convert_ids_to_tokens([token_id])[0]
        raw = token_bytes(raw_token, unicode_to_byte)
        raw_bytes_all += raw
        try:
            standalone = raw.decode("utf-8")
            complete = True
        except UnicodeDecodeError:
            standalone = None
            complete = False
        tokens.append({
            "position": position,
            "ordinal": position + 1,
            "id": int(token_id),
            "raw_token": raw_token,
            "bytes_hex": raw.hex(),
            "byte_start": byte_cursor,
            "byte_end": byte_cursor + len(raw),
            "standalone_decode": standalone,
            "byte_complete": complete,
        })
        byte_cursor += len(raw)
    text_bytes = text.encode("utf-8")
    info = {
        "byte_roundtrip_ok": raw_bytes_all == text_bytes,
        "decoded_roundtrip_ok": raw_bytes_all.decode("utf-8", errors="strict") == text if raw_bytes_all else True,
    }
    return tokens, info


def build_allocated(cache) -> dict:
    k0 = cache.k[0]
    per_layer_k_bytes = int(k0.numel() * k0.element_size())
    total = int(cache.memory_bytes())
    return {
        "per_layer_k_shape": list(k0.shape),
        "per_layer_k_bytes": per_layer_k_bytes,
        "total_bytes": total,
        "dtype": str(k0.dtype).replace("torch.", ""),
        "note": "预分配容量，不是已写入内容；75,497,472 B = 72 MiB = 75.5 MB，源码注释的 '75 MiB' 单位表述有误",
    }


def infer_unicode_map(tok) -> dict[str, int]:
    # 从 tokenizer 的原始 token 表推导 GPT-2 字节映射；GPT-2 的 unicode->byte 映射是固定表，
    # 这里逐字符验证所有 token 字符都在映射表内，避免猜测。
    mapping = {v: k for k, v in bytes_to_unicode().items()}
    vocab = tok.get_vocab()
    for token in vocab:
        for ch in token:
            if ch not in mapping:
                raise RuntimeError(f"tokenizer 词表含未知字符 {ch!r}，与 GPT-2 字节级 BPE 不符")
    return mapping


def build_tensor_checks(events, tensors, allocated, source_map, cache) -> list[dict]:
    checks = []

    def check(cid, category, ok, object_, expected, observed, details=None):
        checks.append({
            "id": cid, "category": category, "status": "passed" if ok else "failed",
            "object": object_, "expected": expected, "observed": observed,
            "details": details or {},
        })

    n_layer = 12
    count_by_kind = {}
    for event in events:
        count_by_kind[event["kind"]] = count_by_kind.get(event["kind"], 0) + 1
    check("event_structure", "structure", count_by_kind == {"embed": 9, "block": 108, "ln_f": 9, "logits": 9, "sample": 9},
          "event kinds", {"embed": 9, "block": 108, "ln_f": 9, "logits": 9, "sample": 9}, count_by_kind)

    ordered = all(events[i]["seq"] == i for i in range(len(events)))
    check("event_seq", "structure", ordered, "seq 连续递增", True, ordered)

    tensor_ids = {t["id"] for t in tensors}
    missing_refs = [e["id"] for e in events for tid in e["tensor_ids"] if tid not in tensor_ids]
    check("event_tensor_refs", "structure", not missing_refs, "事件引用的张量都存在", [], missing_refs[:10])

    for t in tensors:
        payload = t["payload"]
        import numpy as np
        raw = base64.b64decode(payload["data"])
        expected_nbytes = 1 if t["dtype"] == "uint8" else 4
        expected_nbytes *= int(np.prod(t["shape"]))
        ok = (
            len(raw) == expected_nbytes
            and payload["nbytes"] == expected_nbytes
            and sha256_bytes(raw) == payload["sha256"]
        )
        if not ok:
            check(f"payload:{t['id']}", "payload", False, t["id"], expected_nbytes, len(raw))
            break
    else:
        check("payloads_roundtrip", "payload", True, "所有载荷", "nbytes/sha 一致", f"{len(tensors)} 个张量")

    check("cache_allocated_bytes", "cache", allocated["total_bytes"] == EXPECTED_CACHE_BYTES,
          "KV 预分配总字节", EXPECTED_CACHE_BYTES, allocated["total_bytes"],
          {"formula": "2 * n_layer * B * n_head * block_size * head_dim * 4"})

    for f in sorted({e["forward_index"] for e in events}):
        roles = {t["role"] for t in tensors if t["event_id"] == f"fwd{f}.block0"}
        missing = {
            "x_in", "ln1_out", "qkv", "q_raw", "k_raw", "v_raw", "q_heads", "k_heads", "v_heads",
            "k_read", "v_read", "scores_scaled", "causal_mask", "weighted_v", "attn_out",
            "x_after_attn", "ln2_out", "fc_out", "act_out", "mlp_out", "x_out", "attn_weights",
        } - roles
        check(f"selected_layer_tensors_fwd{f}", "structure", not missing, f"fwd{f}.block0 张量集合",
              "完整", sorted(missing))
        weights = {(t["event_id"], t["role"]) for t in tensors}
        missing_attn = [f"fwd{f}.block{l}" for l in range(n_layer) if (f"fwd{f}.block{l}", "attn_weights") not in weights]
        check(f"all_layer_attention_fwd{f}", "structure", not missing_attn, f"fwd{f} 全部层注意力权重",
              "12 层", missing_attn)

    sampled_forward = {e["forward_index"] for e in events if e["kind"] == "sample"}
    logits_events = [e for e in events if e["kind"] == "logits"]
    check("no_unused_forward", "structure",
          all(e["logits_event"]["used_by_sampling"] for e in logits_events)
          and {e["forward_index"] for e in logits_events} == sampled_forward
          and events[-1]["kind"] == "sample",
          "每次前向都用于选词，最后一步为选词", True, events[-1]["kind"] == "sample")

    used_forwards = {
        e["logits_event"]["used_by_sample"] for e in events
        if e["kind"] == "logits" and e["logits_event"]["used_by_sampling"]
    }
    check("sample_links", "structure", used_forwards == {e["id"] for e in events if e["kind"] == "sample"},
          "采样与 logits 关联", sorted(e["id"] for e in events if e["kind"] == "sample"), sorted(x for x in used_forwards if x))

    for event in events:
        if event["kind"] != "block":
            continue
        cache_state = event["cache"]
        l, start_pos, T = event["layer"], event["position_start"], event["position_count"]
        want_eff = [start_pos + T if i <= l else start_pos for i in range(n_layer)]
        want_written = [T if i <= l else 0 for i in range(n_layer)]
        ok = (
            cache_state["layers_effective_positions"] == want_eff
            and cache_state["layers_written_positions"] == want_written
            and cache_state["synchronized_positions"] == min(want_eff)
        )
        if not ok:
            check("cache_block_rules", "cache", False, event["id"], {"effective": want_eff}, cache_state)
            break
    else:
        check("cache_block_rules", "cache", True, "全部 block 事件的缓存区间", "按层推导一致", "48 个事件")

    check("source_map_ops", "source", set(source_map["operations"]) >= {
        t["op"] for t in tensors
    } | {e["primary_op"] for e in events},
        "操作 ID 覆盖", "全部存在", "通过")

    return checks


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出 GPT-2 真实前向/生成轨迹（S1）")
    parser.add_argument("--material-root", type=Path, default=DEFAULT_MATERIAL_ROOT)
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "data" / "gpt2")
    parser.add_argument("--sources-dir", type=Path, default=PROJECT_ROOT / "sources" / "gpt2")
    parser.add_argument("--case-id", default=DEFAULT_CASE_ID)
    parser.add_argument("--label", default="主实例")
    parser.add_argument("--source-root", type=Path, default=PROJECT_ROOT / "sources" / "gpt2")
    parser.add_argument("--prompt", default="The cat sat on the")
    parser.add_argument("--max-new-tokens", type=int, default=9)
    parser.add_argument("--selected-layer", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    started = time.time()
    stage = "import"

    try:
        import numpy as np  # noqa: F401
        import torch as _torch
        import torch.nn.functional as _F

        global torch, F, gpt2_real
        torch = _torch
        F = _F
        torch.set_num_threads(max(1, args.threads))
        torch.manual_seed(args.seed)

        if not args.material_root.is_dir():
            raise FileNotFoundError(f"材料目录不存在: {args.material_root}")
        material_source = args.source_root / "gpt2_real.py"
        if not material_source.is_file():
            raise FileNotFoundError(f"缺少材料源码: {material_source}")
        if not 1 <= args.max_new_tokens <= 16:
            raise ValueError("max_new_tokens 必须在 1..16")
        if args.selected_layer < 0:
            raise ValueError("selected_layer 必须 >= 0")

        sys.path.insert(0, str(args.source_root))
        stage = "load-model"
        import gpt2_real as _gpt2_real  # type: ignore

        gpt2_real = _gpt2_real

        weights_dir = args.material_root / "weights" / "gpt2"
        model, tok = gpt2_real.load_gpt2(weights_dir=weights_dir, device="cpu")
        model.eval()
        config = model.config
        if args.selected_layer >= config.n_layer:
            raise ValueError(f"selected_layer 超出层数 {config.n_layer}")
        prompt_ids = [int(x) for x in tok.encode(args.prompt)]
        if not prompt_ids:
            raise ValueError("prompt 编码结果为空")

        stage = "baseline"
        # A. 未采集的原始路径（与 generate_cached 同样的调用顺序）
        plain_logits, plain_sampled, plain_idx = run_plain(model, prompt_ids, args.max_new_tokens)
        # B. 使用原始 generate_cached 复核同一序列（不依赖我们自己重写的驱动）
        with torch.no_grad():
            base_idx = torch.tensor([prompt_ids], dtype=torch.long)
            generated_idx = gpt2_real.generate_cached(model, base_idx.clone(), args.max_new_tokens, greedy=True)
        baseline_generated_ids = [int(x) for x in generated_idx[0, len(prompt_ids):].tolist()]
        gen_match = baseline_generated_ids == plain_sampled
        if not gen_match:
            raise RuntimeError(f"基线驱动与 generate_cached 生成不一致: {plain_sampled} vs {baseline_generated_ids}")

        # C. 缓存 prefill 与无缓存前向的差异（记录，要求 <= 1e-6）
        stage = "cache-nocache-check"
        with torch.no_grad():
            nocache_logits = model(torch.tensor([prompt_ids], dtype=torch.long))
        prefill_diff = float((plain_logits[0] - nocache_logits).abs().max().item())

        stage = "source-map"
        source_text = material_source.read_text(encoding="utf-8")
        from gpt2_source_map import build_source_map, snapshot_readme, SCHEMA as SOURCE_MAP_SCHEMA

        source_map = build_source_map(source_text)
        unicode_map = infer_unicode_map(tok)

        stage = "capture"
        cache = gpt2_real.KVCache(config, 1, torch.device("cpu"), dtype=model.wte.weight.dtype)
        allocated = build_allocated(cache)
        rec = Recorder(source_map, allocated, args.selected_layer, config.n_layer, tok, prompt_ids)
        rec.unicode_to_byte = unicode_map
        rec.bytes_state = []
        for token_id in prompt_ids:
            raw_token = tok.convert_ids_to_tokens([token_id])[0]
            rec.bytes_state.append(token_bytes(raw_token, unicode_map))

        for layer_block in model.h:
            layer_block.attn.record = True
        restore = install_capture(model, rec)
        try:
            captured_logits, captured_sampled, captured_idx = run_captured(
                model, prompt_ids, args.max_new_tokens, rec, cache=cache)
        finally:
            restore()
            for layer_block in model.h:
                layer_block.attn.record = False

        if captured_sampled != plain_sampled:
            raise RuntimeError(f"采集运行生成不一致: {captured_sampled} vs {plain_sampled}")
        if len(captured_logits) != len(plain_logits):
            raise RuntimeError("采集运行与基线前向次数不一致")
        capture_diffs = [float((a - b).abs().max().item()) for a, b in zip(captured_logits, plain_logits)]
        if any(d != 0.0 for d in capture_diffs):
            raise RuntimeError(f"采集改变了输出: max diff {capture_diffs}")

        stage = "assemble"
        generated_ids = [int(x) for x in captured_idx[0, len(prompt_ids):].tolist()]
        prompt_tokens, byte_info = prompt_token_table(tok, prompt_ids, args.prompt)
        case = {
            "schema": CASE_SCHEMA,
            "case_id": args.case_id,
            "source_map_version": source_map["map_version"],
            "provenance_ref": "manifest.json#/provenance",
            "vocabulary": {"decoded": [tok.decode([i], clean_up_tokenization_spaces=False) for i in range(config.vocab_size)], "raw": [tok.convert_ids_to_tokens(i) for i in range(config.vocab_size)]},
            "input": {
                "text": args.prompt,
                "byte_length": len(args.prompt.encode("utf-8")),
                "numbering": "position 从 0 开始；ordinal 从 1 开始",
                "tokenizer_note": "GPT-2 byte-level BPE；单 token 不一定能独立解码成完整字符",
                "tokens": prompt_tokens,
                "ids": prompt_ids,
                "byte_roundtrip_ok": byte_info["byte_roundtrip_ok"],
                "decoded_roundtrip_ok": byte_info["decoded_roundtrip_ok"],
            },
            "execution": {
                "batch_size": 1,
                "config": {
                    "n_layer": config.n_layer, "n_head": config.n_head, "n_embd": config.n_embd,
                    "head_dim": config.head_dim, "vocab_size": config.vocab_size,
                    "block_size": config.block_size, "layer_norm_eps": config.layer_norm_eps,
                },
                "generation": {
                    "strategy": "greedy",
                    "max_new_tokens": args.max_new_tokens,
                    "prompt_len": len(prompt_ids),
                    "forward_count": len(captured_logits),
                    "sample_count": len(captured_sampled),
                    "generated_ids": generated_ids,
                    "full_sequence_ids": [int(x) for x in captured_idx[0].tolist()],
                    "stop_reason": "max_new_tokens",
                    "note": "prefill 后逐次选词；达到生成上限即停止，不执行无用的最后一次前向。",
                },
                "selected_layer": args.selected_layer,
                "collection_budget_note": "所有层、所有头导出注意力权重；仅选定层导出完整中间张量；只导出生成使用的一行 logits",
            },
            "events": rec.events,
            "forward_summaries": [
                {
                    "forward_index": i,
                    "phase": "prefill" if i == 0 else "decode",
                    "start_pos": 0 if i == 0 else len(prompt_ids) + i - 1,
                    "position_count": len(prompt_ids) if i == 0 else 1,
                    "logits_shape": list(logits.shape),
                    "last_row_index": (len(prompt_ids) - 1) if i == 0 else len(prompt_ids) + i - 1,
                }
                for i, logits in enumerate(captured_logits)
            ],
            "tensors": rec.tensors,
        }

        stage = "checks"
        checks = build_tensor_checks(rec.events, rec.tensors, allocated, source_map, cache)
        checks.extend([
            {"id": "generation_matches_source", "category": "generation",
             "status": "passed" if gen_match else "failed", "object": "generated ids",
             "expected": baseline_generated_ids, "observed": plain_sampled, "details": {}},
            {"id": "capture_outputs_bitwise_equal", "category": "capture",
             "status": "passed" if all(d == 0.0 for d in capture_diffs) else "failed",
             "object": "逐 forward logits", "expected": [0.0] * len(capture_diffs),
             "observed": capture_diffs, "details": {}},
            {"id": "capture_ids_equal", "category": "capture",
             "status": "passed" if captured_sampled == plain_sampled else "failed",
             "object": "生成 token", "expected": plain_sampled, "observed": captured_sampled, "details": {}},
            {"id": "prefill_cache_vs_nocache", "category": "capture",
             "status": "passed" if prefill_diff <= 1e-6 else "failed",
             "object": "prefill logits 最大差", "expected": "<= 1e-6", "observed": prefill_diff, "details": {}},
            {"id": "tokenizer_bytes_roundtrip", "category": "tokenizer",
             "status": "passed" if byte_info["byte_roundtrip_ok"] and byte_info["decoded_roundtrip_ok"] else "failed",
             "object": "token 字节拼接", "expected": "与原文 UTF-8 完全一致",
             "observed": byte_info, "details": {}},
        ])
        for event in rec.events:
            if event["kind"] == "sample":
                probs_ok = abs(sum(item["probability"] for item in event["top_k"]) + event["other_probability_mass"] - 1.0) < 1e-9
                checks.append({
                    "id": f"probability:{event['id']}", "category": "generation",
                    "status": "passed" if probs_ok else "failed", "object": f"{event['id']} 概率质量",
                    "expected": "top_k + other = 1", "observed": probs_ok, "details": {},
                })

        failed = [c for c in checks if c["status"] != "passed"]
        if failed:
            raise RuntimeError(f"自检失败 {len(failed)} 项: {[c['id'] for c in failed]}")

        stage = "serialize"
        case_bytes = json_bytes(case)
        if len(case_bytes) > CASE_SIZE_MAX:
            raise RuntimeError(f"案例体积 {len(case_bytes)} 超过上限 {CASE_SIZE_MAX}")
        size_warning = len(case_bytes) > CASE_SIZE_TARGET
        case_sha = sha256_bytes(case_bytes)

        # 先在内存中完成快照、映射与 manifest 序列化；任何失败都发生在发布之前，不会留下半份轨迹。
        stage = "manifest"
        snapshot_bytes = material_source.read_bytes()
        source_sha = sha256_bytes(snapshot_bytes)
        if source_sha != source_map["source_sha256"]:
            raise RuntimeError("源码快照哈希与映射不一致")
        source_size = len(snapshot_bytes)
        map_bytes = json_bytes(source_map)
        map_sha = sha256_bytes(map_bytes)
        readme_bytes = snapshot_readme(source_sha, map_sha).encode("utf-8")
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "generated_at": RECORDED_AT,
            "tool": {
                "name": "export_gpt2.py",
                "version": TOOL_VERSION,
                "command": " ".join([sys.executable, *sys.argv]),
                "material_root_local": str(args.material_root),
                "note": "绝对路径仅用于本机复现，不进入用户必须理解的页面流程",
            },
            "provenance": {
                "implementation": {
                    "path": "sources/gpt2/gpt2_real.py",
                    "upstream_sha256": sha256_path(args.material_root / "gpt2_real.py"),
                    "sha256": source_sha, "snapshot_sha256": source_sha,
                    "note": "项目内个人实现；由外部材料派生，仅调整生成终止条件；外部材料只读。",
                },
                "weights": {
                    "path": "weights/gpt2/model.safetensors",
                    "sha256": sha256_path(weights_dir / "model.safetensors"),
                    "bytes": (weights_dir / "model.safetensors").stat().st_size,
                },
                "config": {"path": "weights/gpt2/config.json",
                           "sha256": sha256_path(weights_dir / "config.json")},
                "tokenizer": {
                    "tokenizer_json_sha256": sha256_path(weights_dir / "tokenizer.json"),
                    "vocab_sha256": sha256_path(weights_dir / "vocab.json"),
                    "merges_sha256": sha256_path(weights_dir / "merges.txt"),
                    "kind": type(tok).__name__,
                    "local_files_only": True,
                },
                "reference_implementation": {
                    "framework": "transformers",
                    "version": importlib.metadata.version("transformers"),
                    "class": "GPT2LMHeadModel",
                    "attn_implementation": "eager",
                    "note": "仅由独立核对器使用，不参与导出",
                },
                "runtime": {
                    "python": platform.python_version(),
                    "torch": torch.__version__,
                    "numpy": __import__("numpy").__version__,
                    "safetensors": importlib.metadata.version("safetensors"),
                    "platform": platform.platform(),
                    "machine": platform.machine(),
                    "device": "cpu",
                    "dtype": str(model.wte.weight.dtype).replace("torch.", ""),
                    "threads": torch.get_num_threads(),
                    "eval": not model.training,
                    "grad_mode": "no_grad",
                    "seed": args.seed,
                },
                "weight_origin_note": "本地 weights/gpt2；来源记录见材料 download_weights.py；本轮未联网重新下载或与上游逐字节比对",
            },
            "instances": [{
                "case_id": args.case_id,
                "case_file": f"{args.case_id}.json",
                "case_sha256": case_sha,
                "case_bytes": len(case_bytes),
                "label": args.label,
                "input_text": args.prompt,
                "prompt_token_count": len(prompt_ids),
                "generated_token_ids": generated_ids,
                "selected_layer": args.selected_layer,
                "status": "exported",
                "validation_file": "verification.json",
                "validation_status": "not_run",
            }],
            "source_snapshot": {"path": "../../sources/gpt2/gpt2_real.py", "sha256": source_sha},
            "source_map": {"path": "../../sources/gpt2/source-map.json", "sha256": map_sha},
            "export_checks": checks,
            "export_notes": {
                "size_warning": "案例超过 6 MiB 目标" if size_warning else None,
                "source_size_bytes": source_size,
                "duration_seconds": round(time.time() - started, 2),
            },
        }
        manifest_bytes = json_bytes(manifest)

        stage = "publish"
        args.out_dir.mkdir(parents=True, exist_ok=True)
        args.sources_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = args.sources_dir / "gpt2_real.py"
        map_path = args.sources_dir / "source-map.json"
        write_bytes_atomic(snapshot_path, snapshot_bytes)
        snapshot_path.chmod(0o444)
        write_bytes_atomic(map_path, map_bytes)
        map_path.chmod(0o444)
        write_bytes_atomic(args.sources_dir / "README.md", readme_bytes)
        case_path = args.out_dir / f"{args.case_id}.json"
        write_bytes_atomic(case_path, case_bytes)
        write_bytes_atomic(args.out_dir / "manifest.json", manifest_bytes)
        failure_path = args.out_dir / "export-failure.json"
        if failure_path.exists():
            failure_path.unlink()

        print(f"[OK] 案例: {case_path}  ({len(case_bytes):,} B, sha256 {sha256_bytes(case_bytes)[:16]}…)")
        print(f"[OK] 清单: {args.out_dir / 'manifest.json'}")
        print(f"[OK] 源码: {snapshot_path}  (sha256 {source_sha[:16]}…)")
        print(f"[OK] 前向 {len(captured_logits)} 次 / 采样 {len(captured_sampled)} 次 / 事件 {len(rec.events)} 个 / 张量 {len(rec.tensors)} 个")
        print(f"[OK] 生成 token: {generated_ids}  解码: {tok.decode(generated_ids)!r}")
        print(f"[OK] 采集等价 max|diff| = {capture_diffs}；prefill 缓存/无缓存差 = {prefill_diff:.2e}")
        print(f"[OK] 自检 {len(checks)} 项全部通过；耗时 {time.time() - started:.1f}s")
        if size_warning:
            print(f"[WARN] 案例体积超过 6 MiB 目标: {len(case_bytes):,} B")
        return 0

    except BaseException as exc:  # noqa: BLE001
        data_dir = args.out_dir
        try:
            data_dir.mkdir(parents=True, exist_ok=True)
            write_failure(data_dir, args.case_id, stage, exc, args)
        except Exception as failure_error:  # 失败记录本身也失败时，至少打印
            print(f"[ERROR] 无法写失败记录: {failure_error}", file=sys.stderr)
        print(f"[FAIL] 导出失败于阶段 {stage}: {type(exc).__name__}: {exc}", file=sys.stderr)
        if os.environ.get("ATLAS_EXPORT_DEBUG") == "1":
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
