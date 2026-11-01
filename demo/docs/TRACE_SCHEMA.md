# GPT-2 真实轨迹数据契约（S0 定稿）

> 2026-10-31 主例更新：当前案例为 `gpt2-main-002`（9 次前向、9 次选词）与 `smollm2-main-002`（7 次前向、7 次选词）。最后事件为选词，`stop_reason=max_new_tokens`；没有未使用的额外前向。新增 `vocabulary.decoded/raw` 为完整词表标签，ID 为数组下标。下文旧案例 ID、数字与额外前向示例属于原始契约记录，当前值以案例 JSON、独立核对报告及 [工具说明](../GPT2-LLaMA-Atlas/tools/trace/README.md) 为准。


- 日期：2026-10-30
- 状态：S0 交付物。本文件定义 `GPT2_SOURCE_GROUNDED_PLAN.md` 中 S1—S7 使用的数据格式；格式变更必须提升版本号并在交接中说明。
- 适用范围：`GPT2-LLaMA-Atlas/tools/trace/export_gpt2.py`（写出方）、`tools/trace/verify_gpt2.py`（核对方）、S2—S7 的读取与回放模块。
- 本文件不表示真实轨迹已经导出；实际核对结果与未验证项见 `docs/S0_S1_HANDOFF.md`。

## 1. 版本与文件

| 标识 | 值 | 用于 |
|---|---|---|
| `atlas.trace/1` | 清单格式 | `data/gpt2/manifest.json` |
| `atlas.trace.case/1` | 单实例轨迹格式 | `data/gpt2/<case-id>.json` |
| `atlas.source-map/1` | 源码映射格式 | `sources/gpt2/source-map.json` |
| `atlas.trace.verification/1` | 独立核对结果格式 | `data/gpt2/verification.json` |

目录职责沿用计划 §10.1：

```text
GPT2-LLaMA-Atlas/
  tools/trace/export_gpt2.py      导出适配器（S1 新增）
  tools/trace/verify_gpt2.py      独立核对（S1 新增）
  tools/trace/gpt2_source_map.py  源码快照到操作 ID 的映射表与校验（S1 新增）
  data/gpt2/manifest.json         实例索引、provenance、文件摘要
  data/gpt2/<case-id>.json        事件、张量、数值载荷
  data/gpt2/verification.json     独立核对结果（不与案例数据混写）
  sources/gpt2/gpt2_real.py       与案例对应的只读源码快照（原文复制）
  sources/gpt2/source-map.json    操作 ID → 源码行区间与片段摘要
  sources/gpt2/README.md          快照来源与哈希说明
```

`data/gpt2/` 与 `sources/gpt2/` 均由工具生成。禁止手工编辑生成文件；修复必须改工具或材料清单后重跑。

## 2. 通用约定

### 2.1 位置、序号与形状

- 位置一律从 **0** 开始：`position 7` 指索引 7。正文若写“第 8 个位置”，必须同时给出从 0 开始的索引，禁止混用。
- `position_start` 是本次前向的第一个输入位置；`position_count` 是本次新输入长度 T；`total_positions = position_start + T` 是注意力可读取的 key 位置数。
- 形状一律保留 batch 维，B=1 也写出。轴名限定为：`B, H, Tq, Tk, T, C, 3C, F`（`F=4C`）、`V`、`P`（位置轴）。T=7 的前向里 `Tq` 与 `Tk` 不同时，分别标注。
- 一个 token 的“位置”是它在序列中的索引；“第几个新 token”只在采样记录中用 `ordinal`（从 1 开始）表达。

### 2.2 数值编码

| 项目 | 规定 |
|---|---|
| 模型与采集精度 | float32（CPU、eval、`torch.no_grad` / `inference_mode`） |
| 浮点载荷 | 小端 float32 原始字节，标准 base64（含 `=` 填充），`encoding = "f32le-base64"` |
| 掩码载荷 | 每元素 1 字节，`0`=不可读，`1`=可读，`encoding = "u8-base64"` |
| 摘要 | `payload.sha256` 对解码前的原始字节做 SHA-256，小写十六进制 |
| 长度 | `payload.nbytes` 必须等于 `prod(shape) × 每元素字节数`；解码后必须精确等于该长度 |
| 禁止 | JSON 内不写 `NaN`/`Infinity`；掩码分数用「有限分数 + 掩码」表达；载荷中也不出现 NaN/Inf |
| 字节序 | 固定小端；不接受运行环境字节序推断 |

模型输出的完整 logits 为 float32；页面载荷只保留生成实际使用的一行（见 §5）。不得把 Top-N 重新归一化后当作全词表概率。

### 2.3 数据状态与缺失处理

每个张量必须声明：

| 字段 | 取值 | 含义 |
|---|---|---|
| `scope` | `full` / `slice` / `summary` | 完整采集 / 明确切片 / 统计摘要 |
| `derivation` | `observed` / `derived` / `summary` | 采集所得 / 由采集值按公式重算 / 统计 |
| `axes` | 数组 | 与 `shape` 等长，轴名见 §2.1 |
| `op` / `source` | 见 §6 | 生产该张量的源码操作 |

未采集的层或张量必须**不出现**，不得用零、旧层数据或模拟值填充。无法提供时在事件中写 `unavailable_reason`。

### 2.4 时间与确定性

- 案例文件不含墙钟时间戳；同一环境、同一参数重跑应得到逐字节相同的案例文件（载荷与哈希可复现）。时间戳只写入 `manifest.json` 的 `generated_at` 与 `verification.json`。
- 随机性：主实例使用 greedy，无采样随机数；仍记录 `seed=0` 并固定线程数。任何采样实验另立实例与种子。

## 3. `manifest.json`

```json
{
  "schema": "atlas.trace/1",
  "generated_at": "2026-10-30T18:00:00+08:00",
  "tool": {
    "name": "export_gpt2.py",
    "version": "1.0.0",
    "command": "具体的完整命令",
    "note": "本机路径仅用于定位，不进入页面流程"
  },
  "provenance": {
    "implementation": {"path": "材料目录内 gpt2_real.py 的相对说明", "sha256": "…", "snapshot_sha256": "…", "note": "个人实现，不是官方代码"},
    "weights": {"path": "weights/gpt2/model.safetensors", "sha256": "…", "bytes": 548105171},
    "config": {"path": "weights/gpt2/config.json", "sha256": "…"},
    "tokenizer": {"tokenizer_json_sha256": "…", "vocab_sha256": "…", "merges_sha256": "…", "kind": "GPT2TokenizerFast", "local_files_only": true},
    "reference_implementation": {"framework": "transformers", "version": "…", "class": "GPT2LMHeadModel", "attn_implementation": "eager", "note": "仅用于核对，不参与导出"},
    "runtime": {"python": "…", "torch": "…", "numpy": "…", "safetensors": "…", "platform": "…", "machine": "…", "device": "cpu", "dtype": "float32", "threads": 4, "eval": true, "grad_mode": "no_grad", "seed": 0},
    "weight_origin_note": "本地权重来源记录；本轮是否与上游重新比对，如实写"
  },
  "instances": [
    {
      "case_id": "gpt2-main-001",
      "case_file": "gpt2-main-001.json",
      "case_sha256": "…",
      "case_bytes": 0,
      "label": "主实例",
      "input_text": "The cat sat on the mat.",
      "prompt_token_count": 7,
      "generated_token_ids": [198, 198, 1],
      "selected_layer": 0,
      "status": "exported",
      "validation_file": "verification.json",
      "validation_status": "passed"
    }
  ],
  "source_snapshot": {"path": "../../sources/gpt2/gpt2_real.py", "sha256": "…"},
  "source_map": {"path": "../../sources/gpt2/source-map.json", "sha256": "…"}
}
```

规则：

- `manifest.json` 必须最后写入；写出流程失败时不更新清单，旧清单与旧案例保持可读。
- `instances[]` 的 `case_sha256` 必须与磁盘文件一致；`source_snapshot.sha256` 必须与快照文件一致。
- 导出器写 `validation_status: "not_run"`。独立核对结果以 `verification.json` 为权威，S7 构建必须读取该文件；若核对失败，`manifest` 保持原样，由交接或 S7 构建标记失败，不要求核对器改写 `manifest.json`。

## 4. `source-map.json`

```json
{
  "schema": "atlas.source-map/1",
  "source_file": "gpt2_real.py",
  "source_sha256": "…",
  "map_version": "gpt2-real@1",
  "symbols": {
    "GPT2Config": {"lines": [49, 72], "span_sha256": "…"},
    "KVCache": {"lines": [78, 93], "span_sha256": "…"},
    "CausalSelfAttention.forward": {"lines": [114, 151], "span_sha256": "…"},
    "MLP.forward": {"lines": [164, 165], "span_sha256": "…"},
    "Block.forward": {"lines": [179, 186], "span_sha256": "…"},
    "GPT2.forward": {"lines": [204, 223], "span_sha256": "…"},
    "attention_maps": {"lines": [226, 239], "span_sha256": "…"},
    "load_weights": {"lines": [260, 319], "span_sha256": "…"},
    "load_tokenizer": {"lines": [322, 330], "span_sha256": "…"},
    "_next_token": {"lines": [348, 356], "span_sha256": "…"},
    "generate_naive": {"lines": [359, 369], "span_sha256": "…"},
    "generate_cached": {"lines": [372, 396], "span_sha256": "…"}
  },
  "operations": {
    "gpt2.embed": {"symbol": "GPT2.forward", "lines": [214, 215], "span_sha256": "…"},
    "gpt2.attn.qkv_linear": {"symbol": "CausalSelfAttention.forward", "lines": [122, 122], "span_sha256": "…"},
    "gpt2.attn.mask": {"symbol": "CausalSelfAttention.forward", "lines": [139, 144], "span_sha256": "…"},
    "…": {}
  }
}
```

- `span_sha256` 是快照文件对应行区间（含首尾行，保留换行）的 SHA-256，用于证明映射仍指向原文。
- 事件与张量通过 `op` 引用 `operations` 的键；不得引用不存在的操作 ID。
- 映射表在 `tools/trace/gpt2_source_map.py` 中维护；源码快照哈希变化时必须显式更新并重新核对行号。

## 5. 案例文件 `<case-id>.json`

```json
{
  "schema": "atlas.trace.case/1",
  "case_id": "gpt2-main-001",
  "source_map_version": "gpt2-real@1",
  "provenance_ref": "manifest.json#/provenance",
  "input": {
    "text": "The cat sat on the mat.",
    "byte_length": 24,
    "numbering": "position 从 0 开始；ordinal 从 1 开始",
    "tokenizer_note": "GPT-2 byte-level BPE；单 token 不一定能独立解码成完整字符",
    "tokens": [
      {"position": 0, "ordinal": 1, "id": 464, "raw_token": "The", "bytes_hex": "546865",
       "byte_start": 0, "byte_end": 3, "standalone_decode": "The", "byte_complete": true}
    ],
    "ids": [464, 3797, 3332, 319, 262, 2603, 13]
  },
  "execution": {
    "batch_size": 1,
    "config": {"n_layer": 12, "n_head": 12, "n_embd": 768, "head_dim": 64, "vocab_size": 50257, "block_size": 1024, "layer_norm_eps": 0.00001},
    "generation": {
      "strategy": "greedy", "max_new_tokens": 3, "prompt_len": 7,
      "forward_count": 4, "sample_count": 3,
      "generated_ids": [198, 198, 1],
      "full_sequence_ids": [464, 3797, 3332, 319, 262, 2603, 13, 198, 198, 1],
      "note": "generate_cached 在最后一次采样后仍执行一次前向；该前向 logits 未被后续选择使用，见 forward 3 与 logits 事件"
    },
    "selected_layer": 0,
    "collection_budget_note": "所有层、所有头导出注意力权重；仅选定层导出完整中间张量；只导出生成使用的一行 logits"
  },
  "events": [],
  "forward_summaries": [],
  "tensors": []
}
```

### 5.1 前向语义（固定）

| forward_index | phase | start_pos | T | 输入 | 输出使用 |
|---|---:|---:|---:|---|---|
| 0 | prefill | 0 | 7 | prompt 全部 token | `sample0` 使用最后一行 logits |
| 1 | decode | 7 | 1 | 生成的 token 0（位置 7） | `sample1` 使用 |
| 2 | decode | 8 | 1 | 生成的 token 1（位置 8） | `sample2` 使用 |
| 3 | decode | 9 | 1 | 生成的 token 2（位置 9） | **未使用**：循环在采样后仍执行一次前向 |

- 每个前向固定产生事件：`fwd{f}.embed → fwd{f}.block0 … fwd{f}.block11 → fwd{f}.ln_f → fwd{f}.logits`。
- `sample{s}` 事件紧跟在产生其输入 logits 的前向之后，`sample0..2` 分别对应 forward 0..2。
- `fwd3.logits.used_by_sampling = false`；不得为了“好看”删除该前向，也不得改写生成流程。

### 5.2 事件结构

```json
{
  "id": "fwd0.block0",
  "seq": 1,
  "kind": "block",
  "phase": "prefill",
  "forward_index": 0,
  "layer": 0,
  "position_start": 0,
  "position_count": 7,
  "total_positions": 7,
  "input_ids": [464, 3797, 3332, 319, 262, 2603, 13],
  "positions": {"query": [0, 1, 2, 3, 4, 5, 6], "key": [0, 1, 2, 3, 4, 5, 6]},
  "cache": { "…": "见 §5.3" },
  "tensor_ids": ["fwd0.block0.attn_weights", "fwd0.block0.x_in", "…"],
  "primary_op": "gpt2.block.forward",
  "source": {"symbol": "Block.forward", "lines": [179, 186], "span_sha256": "…"},
  "replay": {"phase_complete": false, "note": "该层执行完成；其余层尚未写入本轮 K/V"}
}
```

- `seq` 从 0 递增，事件顺序即真实执行顺序。
- `kind` 限定：`embed` / `block` / `ln_f` / `logits` / `sample`。
- `phase` 取值：前向事件为 `prefill` / `decode`；`sample` 事件为 `sample`（不是前向阶段，仅用于保持字段统一）。
- `phase` 取值：前向事件为 `prefill` / `decode`；`sample` 事件为 `sample`（不是前向阶段，仅用于保持字段统一）。
- `sample` 事件字段：

```json
{
  "id": "sample0",
  "seq": 16,
  "kind": "sample",
  "forward_index": 0,
  "output_position": 7,
  "strategy": "greedy",
  "token_id": 198,
  "raw_token": "Ċ",
  "decoded_text": "\n",
  "standalone_decode_complete": true,
  "logit": 0.0,
  "probability": 0.0,
  "rank": 1,
  "top_k": [{"id": 198, "probability": 0.0}, "… top 10 …"],
  "other_probability_mass": 0.0,
  "entropy_nats": 0.0,
  "probability_basis": "对完整 vocab_size 行做 softmax 后统计；top_k 仅展示，未重新归一化"
}
```

- `decoded_text` 是当前 token 追加后的增量文本（若该 token 单独无法解码，则填 `standalone_decode_complete=false` 并在 `decoded_text` 写可显示的转义形式）。完整文本由连续 token 字节拼接后解码，不逐 token 拼接字符串。
- 概率、排名、熵、其余质量必须由完整 logits 行计算；top_k 只是展示子集。
- 统一计算口径：把 float32 logits 行提升到 **float64** 后做 softmax，再算概率、排名、熵与其余质量。float32 softmax 只作为方法对照；两者在 50257 维行上可差到 5e-5 量级（S0 审计已观测），属方法差，不得据此判数据失败。

### 5.3 缓存结构

```json
"cache": {
  "allocated": {
    "per_layer_k_shape": [1, 12, 1024, 64],
    "per_layer_k_bytes": 3145728,
    "total_bytes": 75497472,
    "dtype": "float32",
    "note": "预分配容量，不是已写入内容；75,497,472 B = 72 MiB = 75.5 MB，源码注释的 '75 MiB' 单位表述有误"
  },
  "layers_effective_positions": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  "layers_written_positions": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  "synchronized_positions": 0,
  "read_window": {"start": 0, "end_exclusive": 7},
  "write_window": {"start": 0, "end_exclusive": 7},
  "selected_layer_read_refs": {"k": "fwd0.block0.k_read", "v": "fwd0.block0.v_read"},
  "note": "layers_effective_positions 是该事件发生时刻各层已写入位置数；synchronized_positions 是所有层的最小值"
}
```

- `layers_effective_positions` 必须由「当前事件之前真实执行过的层」推导，不能依赖播放历史。
- 规则：`fwd{f}.embed` 全层等于 `position_start`、`written=0`；`fwd{f}.block{l}` 中第 0—l 层等于 `position_start + T`、其余层等于 `position_start`，`written` 对被本轮执行过的层为 T、未执行为 0；`ln_f`/`logits` 全层等于 `position_start + T`、`written=0`（写入已在各 block 完成）；`sample` 不写缓存。
- `allocated` 的字节数必须按实际张量 `numel() × element_size()` 求和，不得套用 fp16 公式。
- 选定层的 `k_read`/`v_read` 是对应注意力实际读取的 `k_all`/`v_all` 切片；其它层不导出预分配零区。

## 6. 张量与事件—张量清单

### 6.1 事件与张量 ID 规则

- 事件 ID：`fwd{forward_index}.embed`、`fwd{forward_index}.block{layer}`、`fwd{forward_index}.ln_f`、`fwd{forward_index}.logits`、`sample{sample_index}`。
- 张量 ID：`{event_id}.{role}`。所有层都有的张量：`fwd{f}.block{l}.attn_weights`（`role="attention_weights"`）。
- 选定层额外张量使用相同事件前缀，见下表。层号在 S0 固定为 `0`；若以后更换，必须换实例或提升 schema 版本并说明。

### 6.2 所有层（每个 block 事件）

| role | shape | axes | op | 说明 |
|---|---|---|---|---|
| `attn_weights` | `[1,12,Tq,Tk]` | B,H,Tq,Tk | `gpt2.attn.softmax` | 该层该前向的 softmax 后注意力权重；由原生 `record` 开关取得；Tk=total_positions |

### 6.3 选定层（每个 block 事件，layer=selected）

| role | shape | axes | op | 说明 |
|---|---|---|---|---|
| `x_in` | `[1,T,768]` | B,T,C | `gpt2.block.forward` | 残差主干输入；block0 的 prefill 输入即嵌入相加结果 |
| `ln1_out` | `[1,T,768]` | B,T,C | `gpt2.block.ln1` | LN_1 输出 |
| `qkv` | `[1,T,2304]` | B,T,3C | `gpt2.attn.qkv_linear` | `c_attn(x)` 原始输出 |
| `q_raw` / `k_raw` / `v_raw` | `[1,T,768]` | B,T,C | `gpt2.attn.split` | split 三段 |
| `q_heads` / `k_heads` / `v_heads` | `[1,12,T,64]` | B,H,T,C | `gpt2.attn.view_transpose` | view+transpose 后 |
| `k_read` | `[1,12,Tk,64]` | B,H,Tk,C | `gpt2.cache.read_window` | 实际读取的 k_all |
| `v_read` | `[1,12,Tk,64]` | B,H,Tk,C | `gpt2.cache.read_window` | 实际读取的 v_all |
| `scores_scaled` | `[1,12,Tq,Tk]` | B,H,Tq,Tk | `gpt2.attn.scores` | **掩码前**的有限分数 `q@k^T/√D` |
| `causal_mask` | `[Tq,Tk]` | Tq,Tk | `gpt2.attn.mask` | u8；`1` 可读，`0` 不可读；对所有 B/H 广播 |
| `weighted_v` | `[1,T,768]` | B,T,C | `gpt2.attn.weighted_v` | softmax 权重加权 V、transpose+reshape 后、c_proj 前 |
| `attn_out` | `[1,T,768]` | B,T,C | `gpt2.attn.out_proj` | `c_proj` 输出，即注意力子层输出 |
| `x_after_attn` | `[1,T,768]` | B,T,C | `gpt2.block.residual_attn` | `x + attn_out` |
| `ln2_out` | `[1,T,768]` | B,T,C | `gpt2.block.ln2` | LN_2 输出 |
| `fc_out` | `[1,T,3072]` | B,T,F | `gpt2.mlp.fc` | c_fc 输出 |
| `act_out` | `[1,T,3072]` | B,T,F | `gpt2.mlp.act` | `gelu_new` 输出 |
| `mlp_out` | `[1,T,768]` | B,T,C | `gpt2.mlp.proj` | c_proj 输出 |
| `x_out` | `[1,T,768]` | B,T,C | `gpt2.block.forward` | `x_after_attn + mlp_out`，block 输出 |

若 T>1，`x_in` 与 `x_out` 包含所有位置；不另做整层切片。

### 6.4 embed / ln_f / logits 事件

| 事件 | role | shape | op | 说明 |
|---|---|---|---|---|
| `fwd{f}.embed` | `wte_out` | `[1,T,768]` | `gpt2.embed` | token 嵌入 |
| | `wpe_out` | `[T,768]` | `gpt2.embed` | 位置嵌入（`self.wpe(pos)` 实际形状为 (T,C)，batch 由广播得出） |
| `fwd{f}.ln_f` | `out` | `[1,T,768]` | `gpt2.ln_f` | 最终 LayerNorm 输出（所有位置） |
| `fwd{f}.logits` | `last_row` | `[1,50257]` | `gpt2.generate.last_row_slice` | 生成实际使用的最后一行；`scope="slice"`，并记录完整形状 |

`fwd{f}.logits` 事件额外字段：

```json
"logits_event": {
  "full_output_shape": [1, 7, 50257],
  "full_output_dtype": "float32",
  "last_row_index": 6,
  "used_by_sampling": true,
  "used_by_sample": "sample0",
  "per_position_argmax": [ "…T 个整数…" ],
  "per_position_max_logit": ["…T 个 float…"],
  "note": "GPT2.forward 返回全部位置 logits；生成函数再取 logits[:, -1, :]。其余位置数值不进入页面载荷，仅在验证中重算核对。"
}
```

- `full_output_shape` 必须是真实执行形状（含 B、T、V），即使只导出最后一行。
- `per_position_argmax` / `per_position_max_logit` 是摘要，不得被当作概率或完整数据。

### 6.5 事件—张量顺序与回放状态

- 事件 `seq` 顺序即回放顺序；播放/跳转只能根据目标事件重新推导状态（缓存、当前层、当前序列），不得累加修改。
- 前端状态至少可由以下信息派生：`forward_index`、`phase`、`layer`、`position_start/count`、`positions`、`cache`、`sample` 的 `token_id`、`input_ids`。
- `replay.phase_complete` 仅在 ln_f/logits 事件为 true；`replay.sequence_length_after` 在 sample 事件写出。

## 7. 导出预算

| 内容 | 采集范围 | 理由 |
|---|---|---|
| 注意力权重 | 全部 12 层 × 12 头，每个前向 | 主实例核心机制；每前向原始 float32 约 28 KB（prefill），base64 后约 57 KB |
| 选定层中间张量 | 第 0 层全部主要张量 | 支撑 S4 逐步展开与源码对应 |
| logits | 只保留生成使用的一行（完整 50257 维）；全部位置只留摘要 | 保持概率质量完整，控制体积 |
| 缓存 | 每层容量 + 有效/写入区间元数据；选定层实际读取 K/V | 不导出预分配零区 |
| 源码 | `gpt2_real.py` 原文快照 | 离线、可追溯 |
| 其它位置/层激活 | 不导出 | 未采集不伪装成可查看 |

体积目标：单案例 JSON（base64 后）不超过 6 MiB；两页单 HTML 合计目标 ≤ 15 MiB。超出 10 MiB 先报告组成再决定是否调整预算。

## 8. 容差（运行前声明）

> 2026-10-30 S1 实测后的口径补充（不改变数据格式）：① sample 概率等统一用 float64 softmax，float32 对照允许 1e-4 方法差；② 输出头重算同时要求 float32 同形状逐位相等与 float64 尺度相对误差 ≤1e-5。两处均在实测方法差出现后补充，原因与观测值如实记录，不涉及修改数据或事后放宽数据错误判定。

独立核对必须按本节容差判定；不得在失败后放宽容差以制造通过。实际观测值写入 `verification.json`。

| 类别 | 检查对象 | 容差 | 说明 |
|---|---|---|---|
| 采集等价 | 采集运行 vs 未采集运行的每个 forward logits | **逐位相等**（max abs diff = 0） | 同一代码路径、同一输入；采集只做 `detach().clone()` |
| 采集等价 | 生成 token 序列 | 完全相等 | greedy |
| 参考实现 | 未缓存前向 logits vs HF `GPT2LMHeadModel`（eager） | `max|diff| ≤ 1e-4` | 历史 parity 记录为 0；这是本机同权重对照 |
| 参考实现 | 全层注意力权重 vs HF attentions | `max|diff| ≤ 1e-6` | 历史记录为 0；eager 注意力 |
| 载荷往返 | base64 解码、shape 乘积、nbytes、sha256 | 精确相等 | 任何不符即失败 |
| 公式重算 | 由导出张量重算的激活（LN、残差、MLP、输出投影） | `max|diff| ≤ 1e-4`，`rel ≤ 1e-4` | 重算在 float64，原值为 float32 |
| 公式重算 | softmax 权重与行和 | `max|diff| ≤ 1e-6`；行和 `|Σ-1| ≤ 1e-6` | 掩码位置权重必须严格为 0 |
| 输出头 | 由 `ln_f.out` × 共享嵌入权重重算最后一行 logits | 两条都满足：(a) 与执行同形状的 float32 batched 重算**逐位相等**（max\|diff\|=0）；(b) float64 重算的 `max|diff| / max|logit| ≤ 1e-5`；同时记录原始绝对差 | 验证共享权重与输出头；(a) 证明载荷与执行一致，(b) 吸收 fp32 GEMM 累加顺序差。S1 实测 f0 绝对差 1.90e-4 @ \|logit\|max 133.9（相对 1.42e-6），f1–f3 为 8.5e-6/2.05e-5/3.43e-5，(a) 均为 0.0 |
| 缓存 | 分配字节、有效区间、写入区间、读取切片 | 与公式和实际张量精确一致 | 见 §5.3 |
| 生成 | argmax / 概率 / 排名 / 剩余质量 | token 完全相等；同一 float64 口径下概率 `|diff| ≤ 1e-6`；若用 float32 softmax 对照，允许 `≤1e-4` 并必须注明方法差 | 概率基于完整 vocab 行；不重新归一化 Top-N |

本表是 S1 的通过门槛；任何一项失败都阻止该实例进入产品接入，并按 `GPT2_SOURCE_GROUNDED_PLAN.md` 的关口记录阻碍。

## 9. 失败与中断

- 导出失败：在 `data/gpt2/export-failure.json` 写入 `{schema, case_id, stage, error, traceback, args, generated_at, published: false}`；不得发布半份案例、不得更新 manifest。成功后再删除该失败记录。
- 校验失败：`verify_gpt2.py` 退出码非 0，`verification.json` 仍完整写出失败项与观测值；`manifest.instances[].validation_status` 更新为 `failed`。
- 数据缺失或 schema 不匹配：读取方必须暂停该实例并显示原因，不得回退为模拟数据。
- 不修改材料目录内任何文件；导出器设置 `sys.dont_write_bytecode = True` 并只读材料。
- 不自动联网下载权重；`local_files_only=True`。

## 10. 待定项（S0 已定，S1 执行时记录）

| 项目 | 决定 |
|---|---|
| 主实例文本 | `The cat sat on the mat.`（真实 tokenizer 长度为 7，不硬编码九词） |
| 选定层 | 第 0 层 |
| 生成策略 | greedy，3 个新 token；一次真实前向不被使用也要保留 |
| 辅助实例 | 计数序列 `1, 2, 3, 4, 5, 6, 7, 8,` 留到 S6/后续；S1 不导出 |
| 数据体积 | 单案例目标 ≤ 6 MiB（base64 后） |
| 参考实现 | `transformers.GPT2LMHeadModel`，`attn_implementation="eager"`，`local_files_only=True` |

若 S1 实测与上述决定不符（例如体积超限、范围外失败），记录原因后停止并请求确认，不自行扩大范围。
