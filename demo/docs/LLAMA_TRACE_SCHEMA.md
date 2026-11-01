# Llama 真实轨迹数据契约 v2（L0 定稿）

> 2026-10-31 主例更新：当前案例为 `gpt2-main-002`（9 次前向、9 次选词）与 `smollm2-main-002`（7 次前向、7 次选词）。最后事件为选词，`stop_reason=max_new_tokens`；没有未使用的额外前向。新增 `vocabulary.decoded/raw` 为完整词表标签，ID 为数组下标。下文旧案例 ID、数字与额外前向示例属于原始契约记录，当前值以案例 JSON、独立核对报告及 [工具说明](../GPT2-LLaMA-Atlas/tools/trace/README.md) 为准。


- 日期：2026-10-30
- 状态：**L0 交付物**。本文件固定 `SmolLM2-135M` 主实例的数据格式、身份、采集预算与失败边界；L1 按本契约导出，L2 读取。
- 适用方案：[LLAMA_SOURCE_GROUNDED_PLAN.md](./LLAMA_SOURCE_GROUNDED_PLAN.md)（L0—L7）。
- 与 GPT-2 的关系：GPT-2 使用已发布的 `atlas.trace/1`、`atlas.trace.case/1`、`atlas.source-map/1`；**本文件不修改 GPT-2 v1**。LLaMA 使用独立的 `/2` 版本与 `data/llama/`、`sources/llama/` 目录。
- 本文件不表示轨迹已经导出。实际导出、独立核对与限制见 L1 交接。

## 1. 版本与文件

| 标识 | 值 | 文件 |
|---|---|---|
| `atlas.trace/2` | LLaMA 清单 | `data/llama/manifest.json` |
| `atlas.trace.case/2` | LLaMA 单实例轨迹 | `data/llama/<case-id>.json` |
| `atlas.source-map/2` | 多源文件行映射 | `sources/llama/source-map.json` |
| `atlas.trace.verification/2` | 独立核对结果 | `data/llama/verification.json` |

目录职责：

```text
GPT2-LLaMA-Atlas/
  tools/trace/export_llama.py          导出适配器（L1 新增）
  tools/trace/verify_llama.py          独立核对（L1 新增）
  tools/trace/llama_source_map.py      源码快照与符号/行映射（L1 新增）
  data/llama/manifest.json             实例索引、身份与 provenance
  data/llama/smollm2-main-001.json     事件、张量、数值载荷
  data/llama/verification.json         独立核对结果
  sources/llama/step06_model.py        主模型源码原文快照（L1 复制）
  sources/llama/step09_real_llama.py   真权重加载/生成源码原文快照（L1 复制）
  sources/llama/source-map.json        操作 ID → 文件/符号/行区间/片段摘要
  sources/llama/README.md              快照来源与哈希说明
```

规则：

- `data/llama/` 与 `sources/llama/` 由工具生成；不得手工编辑。源材料、权重和旧实验记录只读。
- `manifest.json` 最后写入；失败时不覆盖上一份已验证实例。
- **版本分派必须精确匹配 `schema` 字符串**。读取器不得把 v1 字段猜成 v2，也不得把 v2 数据填入 GPT-2 页面。GPT-2 的 v1 文件、导出器与页面不因本契约改变。

## 2. 通用约定

### 2.1 位置、序号、形状

- 位置从 **0** 开始；`position_start` 是本次输入第一个位置；`position_count = T` 是本次新输入长度；`total_positions = start + T` 是注意力可读取的 key 数。
- `ordinal` 只用于采样/生成记录，从 1 开始；不得与位置混用。
- shape 一律保留 batch 维（B=1 也写出）。
- 轴名限定：`B, T, Tq, Tk, C, D, V, F, Hq, Hkv, P`。
  - `C=576`，`D=64`，`V=49152`，`F=1536`，`Hq=9`，`Hkv=3`，`P` 为位置/缓存轴。
  - **禁止把 F 写成 4C 或 3C**：SwiGLU 的 1536 来自配置，不是 GPT-2 的 4×扩张公式。
  - GQA 下 `Hq ≠ Hkv`；缓存 shape 用 `Hkv`，展开后的注意力张量用 `Hq`，两者都是必须区分的轴。
- Q/K 必须区分 `pre_rope` / `post_rope`；缓存中的 K 是 `post_rope`，V 不旋转。

### 2.2 数值编码

| 项目 | 规定 |
|---|---|
| 模型运行精度 | float32（CPU、`eval()`、`torch.no_grad` / inference mode） |
| 权重存储精度 | bfloat16（本地文件）；**不把存储精度写成计算精度** |
| 浮点载荷 | 小端 float32 原始字节 → 标准 base64；`encoding="f32le-base64"` |
| 掩码 | 每元素 1 字节；`0`=不可读，`1`=可读；`encoding="u8-base64"` |
| 摘要 | `payload.sha256` 对解码前原始字节做 SHA-256 |
| 长度 | `payload.nbytes = prod(shape) × 每元素字节数`；解码必须精确一致 |
| 禁止 | JSON 内 NaN/Infinity；掩码分数单独存；载荷内也不得出现 NaN/Inf |
| 字节序 | 固定小端，不依赖运行环境推断 |

概率、排名、熵、其余质量统一用 **float64 softmax** 对完整词表行计算；float32 softmax 只能作为方法对照，不得据此判数据失败。展示 Top-N 时保留“其他”质量，不重新归一化。

### 2.3 数据状态与缺失

每个张量必须声明：

| 字段 | 取值 | 含义 |
|---|---|---|
| `scope` | `full` / `slice` / `summary` | 完整 / 明确切片 / 统计摘要 |
| `derivation` | `observed` / `derived` / `summary` | 直接观察 / 由观察值按公式重算 / 统计 |
| `axes` | 与 shape 等长 | 轴名见 §2.1 |
| `dtype` | `float32` / `uint8` | 运行时载荷类型 |
| `op` | source-map 的 operation ID | 生产该张量的操作 |
| `source` | `{file, symbol, lines, span_sha256}` | 源码定位 |

未采集的层/张量**必须缺席**并给出 `unavailable_reason`；不得用零、第 0 层数据、模拟值或随机数补齐。缺失不是错误，伪造才是。

### 2.4 时间与确定性

- case 文件不含墙钟时间；时间戳只在 manifest/verification 中。
- 主实例 greedy，`seed=0`，固定两条计算线程；记录环境与 no-grad/eval。
- 同一环境、同一参数重跑应得到相同 generation 与相同载荷摘要。
- 温度/top-k 实验若在 L6 出现，必须另立实例，不修改主轨迹。

## 3. 模型身份

页面与数据必须一起表达，不能只替换标题文字：

```json
"model": {
  "model_id": "smollm2-135m",
  "family": "llama",
  "architecture": "LlamaForCausalLM",
  "display_name": "SmolLM2-135M（Llama 架构）",
  "weights_origin": "HuggingFaceTB/SmolLM2-135M",
  "not_meta_llama": true,
  "license": "Apache-2.0",
  "upstream": {
    "model_url": "https://huggingface.co/HuggingFaceTB/SmolLM2-135M",
    "config_url": "https://huggingface.co/HuggingFaceTB/SmolLM2-135M/raw/main/config.json",
    "revision": null,
    "revision_note": "本地副本使用可变的 main 下载；不声称固定 commit，载荷身份以本地 SHA-256 为准"
  },
  "config": {
    "n_layer": 30, "n_embd": 576, "n_head": 9, "n_kv_head": 3, "head_dim": 64,
    "ffn_hidden": 1536, "vocab_size": 49152, "rms_norm_eps": 1e-5,
    "rope_theta": 100000, "rope_style": "half_split", "rope_interleaved": false,
    "tie_word_embeddings": true, "max_position_embeddings": 8192,
    "demo_context_capacity": 256
  },
  "weights_storage_dtype": "bfloat16",
  "runtime_dtype": "float32"
}
```

- 模型身份字段不得互相替代：`model_id` 固定 `smollm2-135m`；节点前缀、记录命名空间、标题均使用该 ID。
- 旧 `llama2` LLaMA-2 7B 模拟图**只作资料**：不新建 `llama2.L*` 节点承载 SmolLM2 数据，不把 32 层/4096 宽混入 30 层/576 宽。
- 发布配置最大位置 8192 与首版演示容量 256 必须分开显示；256 不是训练长度。
- 权重 BF16 存储不等于 bf16/MPS/GPU 推理已验证；主验证路径是 CPU fp32。

## 4. `manifest.json`

```json
{
  "schema": "atlas.trace/2",
  "generated_at": "2026-10-30T…+08:00",
  "tool": {"name": "export_llama.py", "version": "2.0.0", "command": "…"},
  "model": {},
  "provenance": {
    "implementation": {
      "main_file": "step06_model.py",
      "real_file": "step09_real_llama.py",
      "files": [
        {"role": "model", "sha256": "…", "snapshot": "sources/llama/step06_model.py"},
        {"role": "real_loader", "sha256": "…", "snapshot": "sources/llama/step09_real_llama.py"}
      ],
      "note": "个人实现，不是 Meta/HuggingFace 官方代码"
    },
    "weights": {"path": "llama_steps/weights/smollm2-135m/model.safetensors", "sha256": "…", "bytes": 269060552, "tensor_count": 272, "storage_dtype": "bfloat16"},
    "config": {"path": "…/config.json", "sha256": "…"},
    "tokenizer": {"tokenizer_json_sha256": "…", "tokenizer_config_sha256": "…", "vocab_size": 49152, "kind": "BPE tokenizer from local_files_only", "special_ids": {"bos": 0, "eos": 0}},
    "reference_implementation": {"framework": "transformers", "version": "…", "class": "LlamaForCausalLM", "attn_implementation": "eager", "device": "cpu", "note": "仅用于对照，不参与导出"},
    "runtime": {"python": "…", "torch": "…", "transformers": "…", "device": "cpu", "dtype": "float32", "threads": 2, "eval": true, "grad_mode": "no_grad", "seed": 0}
  },
  "instances": [
    {
      "case_id": "smollm2-main-001",
      "case_file": "smollm2-main-001.json",
      "case_sha256": "…",
      "case_bytes": 0,
      "model_id": "smollm2-135m",
      "label": "主实例",
      "input_text": "The capital of France is",
      "prompt_token_count": 5,
      "generated_token_ids": [],
      "selected_layer": 0,
      "status": "exported",
      "validation_file": "verification.json",
      "validation_status": "not_run"
    }
  ],
  "source_snapshots": [
    {"path": "sources/llama/step06_model.py", "sha256": "…"},
    {"path": "sources/llama/step09_real_llama.py", "sha256": "…"}
  ],
  "source_map": {"path": "sources/llama/source-map.json", "sha256": "…", "version": "smollm2-main@2"}
}
```

规则：

- `instances[].case_sha256` 必须与磁盘案例一致；`source_snapshots[].sha256` 与快照文件一致。
- 导出器写 `validation_status:"not_run"`；独立核对结果以 `verification.json` 为权威。失败不覆盖旧案例/清单。
- `generated_token_ids` 由 L1 的真实运行冻结；本契约不预先写死“Paris”。

## 5. `source-map.json`（v2）

两个源码文件，因此 v2 不再只有一个 `source_file`：

```json
{
  "schema": "atlas.source-map/2",
  "map_version": "smollm2-main@2",
  "sources": {
    "step06_model.py": {"sha256": "…", "snapshot": "sources/llama/step06_model.py"},
    "step09_real_llama.py": {"sha256": "…", "snapshot": "sources/llama/step09_real_llama.py"}
  },
  "symbols": {
    "RMSNorm": {"file": "step06_model.py", "lines": [0, 0], "span_sha256": "…"},
    "RMSNorm.forward": {"file": "step06_model.py", "lines": [0, 0], "span_sha256": "…"},
    "Attention.forward": {"file": "step06_model.py", "lines": [0, 0], "span_sha256": "…"},
    "SwiGLU.forward": {"file": "step06_model.py", "lines": [0, 0], "span_sha256": "…"},
    "LlamaBlock.forward": {"file": "step06_model.py", "lines": [0, 0], "span_sha256": "…"},
    "MiniLlama.forward": {"file": "step06_model.py", "lines": [0, 0], "span_sha256": "…"},
    "build_rope_cache": {"file": "step06_model.py", "lines": [0, 0], "span_sha256": "…"},
    "apply_rope": {"file": "step06_model.py", "lines": [0, 0], "span_sha256": "…"},
    "KVCache.append": {"file": "step06_model.py", "lines": [0, 0], "span_sha256": "…"},
    "generate": {"file": "step09_real_llama.py", "lines": [0, 0], "span_sha256": "…"}
  },
  "operations": {
    "llama.rmsnorm": {"file": "step06_model.py", "symbol": "RMSNorm.forward", "lines": [0, 0], "span_sha256": "…"},
    "llama.rope.build": {"file": "step06_model.py", "symbol": "build_rope_cache", "lines": [0, 0], "span_sha256": "…"},
    "llama.rope.apply_q": {"file": "step06_model.py", "symbol": "apply_rope", "lines": [0, 0], "span_sha256": "…"},
    "llama.rope.apply_k": {"file": "step06_model.py", "symbol": "apply_rope", "lines": [0, 0], "span_sha256": "…"},
    "llama.attn.q_proj": {"file": "step06_model.py", "symbol": "Attention.forward", "lines": [0, 0], "span_sha256": "…"},
    "llama.attn.k_proj": {"file": "step06_model.py", "symbol": "Attention.forward", "lines": [0, 0], "span_sha256": "…"},
    "llama.attn.v_proj": {"file": "step06_model.py", "symbol": "Attention.forward", "lines": [0, 0], "span_sha256": "…"},
    "llama.attn.v_heads": {"file": "step06_model.py", "symbol": "Attention.forward", "lines": [0, 0], "span_sha256": "…"},
    "llama.attn.scores": {"file": "step06_model.py", "symbol": "Attention.forward", "lines": [0, 0], "span_sha256": "…"},
    "llama.attn.mask": {"file": "step06_model.py", "symbol": "Attention.forward", "lines": [0, 0], "span_sha256": "…"},
    "llama.cache.append": {"file": "step06_model.py", "symbol": "KVCache.append", "lines": [0, 0], "span_sha256": "…"},
    "llama.gqa.repeat_kv": {"file": "step06_model.py", "symbol": "Attention.forward", "lines": [0, 0], "span_sha256": "…"},
    "llama.attn.softmax": {"file": "step06_model.py", "symbol": "Attention.forward", "lines": [0, 0], "span_sha256": "…"},
    "llama.attn.o_proj": {"file": "step06_model.py", "symbol": "Attention.forward", "lines": [0, 0], "span_sha256": "…"},
    "llama.swiglu.gate": {"file": "step06_model.py", "symbol": "SwiGLU.forward", "lines": [0, 0], "span_sha256": "…"},
    "llama.swiglu.up": {"file": "step06_model.py", "symbol": "SwiGLU.forward", "lines": [0, 0], "span_sha256": "…"},
    "llama.swiglu.mul": {"file": "step06_model.py", "symbol": "SwiGLU.forward", "lines": [0, 0], "span_sha256": "…"},
    "llama.swiglu.down": {"file": "step06_model.py", "symbol": "SwiGLU.forward", "lines": [0, 0], "span_sha256": "…"},
    "llama.block.forward": {"file": "step06_model.py", "symbol": "LlamaBlock.forward", "lines": [0, 0], "span_sha256": "…"},
    "llama.block.residual_attn": {"file": "step06_model.py", "symbol": "LlamaBlock.forward", "lines": [0, 0], "span_sha256": "…"},
    "llama.embed": {"file": "step06_model.py", "symbol": "MiniLlama.forward", "lines": [0, 0], "span_sha256": "…"},
    "llama.lm_head": {"file": "step06_model.py", "symbol": "MiniLlama.forward", "lines": [0, 0], "span_sha256": "…"},
    "llama.generate.prefill": {"file": "step09_real_llama.py", "symbol": "generate", "lines": [0, 0], "span_sha256": "…"},
    "llama.generate.sample": {"file": "step09_real_llama.py", "symbol": "generate", "lines": [0, 0], "span_sha256": "…"},
    "llama.generate.decode": {"file": "step09_real_llama.py", "symbol": "generate", "lines": [0, 0], "span_sha256": "…"}
  }
}
```

- `lines` 是 1-based 闭区间；`span_sha256` 对快照对应行（含换行，与 GPT-2 v1 相同算法）计算。真实行号由 L1 生成，**L0 不填写占位行号**；示例的 `[0,0]` 表示待生成，不是可发布数据。
- 页面/事件/张量只能引用存在的 operation ID；源码哈希变化必须显式更新行号并重验。
- 快照是个人实现原文；中文解释是项目讲解，不得把改写后的代码标为原文。

## 6. `case` 文件与执行语义

```json
{
  "schema": "atlas.trace.case/2",
  "case_id": "smollm2-main-001",
  "model_id": "smollm2-135m",
  "source_map_version": "smollm2-main@2",
  "provenance_ref": "manifest.json#/provenance",
  "input": {
    "text": "The capital of France is",
    "byte_length": 0,
    "tokenizer_note": "真实 SmolLM2 BPE tokenizer；单 token 不一定能独立解码",
    "tokens": [
      {"position": 0, "ordinal": 1, "id": 504, "raw_token": "The", "bytes_hex": "…",
       "standalone_decode": "The", "byte_complete": true}
    ],
    "ids": [504, 3575, 282, 4649, 314]
  },
  "execution": {
    "batch_size": 1,
    "config": {"n_layer": 30, "n_head": 9, "n_kv_head": 3, "n_embd": 576, "head_dim": 64, "ffn_hidden": 1536, "vocab_size": 49152, "block_size": 256, "rms_norm_eps": 1e-5, "rope_theta": 100000, "rope_style": "half_split", "tie_word_embeddings": true, "max_position_embeddings": 8192},
    "generation": {
      "strategy": "greedy",
      "max_new_tokens": 3,
      "prompt_len": 5,
      "forward_count": 4,
      "sample_count": 3,
      "generated_ids": [],
      "full_sequence_ids": [],
      "note": "prefill 一次；每次采样后仍有下一轮 decode 前向，最后一次 logits 未被后续采样消费；L1 冻结实际数量与 token"
    },
    "selected_layer": 0,
    "collection_budget_note": "全部层×全部 Q 头注意力权重；仅第 0 层完整中间张量；只导出被采样消费的最后一行完整 logits；缓存不导出预分配零区"
  },
  "events": [],
  "forward_summaries": [],
  "tensors": []
}
```

### 6.1 前向语义（计划值，L1 冻结）

| forward_index | phase | start_pos | T | 输入 | 输出使用 |
|---|---:|---:|---:|---|---|
| 0 | prefill | 0 | 5 | prompt 全部 token | `sample0` 使用最后一行 |
| 1 | decode | 5 | 1 | 生成 token 0（位置 5） | `sample1` 使用 |
| 2 | decode | 6 | 1 | 生成 token 1（位置 6） | `sample2` 使用 |
| 3 | decode | 7 | 1 | 生成 token 2（位置 7） | **未使用**；`used_by_sampling=false` |

- 每前向事件固定：`fwd{f}.embed → fwd{f}.block0 … fwd{f}.block29 → fwd{f}.ln_f → fwd{f}.logits`。
- 事件顺序即真实执行顺序；不得为“好看”删除未消费前向。
- 若 L1 实际数量不同（例如提前停止），必须在 manifest 与 `used_by_sampling` 中如实记录，并更新本表；不得改数据迎合本节。

### 6.2 事件

```json
{
  "id": "fwd0.block0",
  "seq": 1,
  "kind": "block",
  "phase": "prefill",
  "forward_index": 0,
  "layer": 0,
  "position_start": 0,
  "position_count": 5,
  "total_positions": 5,
  "input_ids": [504, 3575, 282, 4649, 314],
  "positions": {"query": [0,1,2,3,4], "key": [0,1,2,3,4]},
  "cache": {},
  "tensor_ids": [],
  "primary_op": "llama.block.forward",
  "source": {"file": "step06_model.py", "symbol": "LlamaBlock.forward", "lines": [0,0], "span_sha256": "…"},
  "replay": {"phase_complete": false, "note": "…"}
}
```

- `kind`：`embed` / `block` / `ln_f` / `logits` / `sample`。
- `phase`：前向为 `prefill` / `decode`；sample 为 `sample`。
- `sample{s}` 字段：`forward_index`、`output_position`、`strategy:"greedy"`、`token_id`、`raw_token`、`decoded_text`、`standalone_decode_complete`、`probability`、`rank`、`top_k`、`other_probability_mass`、`entropy_nats`、`probability_basis`。
- `fwd{f}.logits.logits_event`：`full_output_shape=[B,T,V]`、`last_row_index`、`used_by_sampling`、`used_by_sample`、`per_position_argmax`、`per_position_max_logit`、说明。只导出被采样使用的最后一行载荷；其余位置仅摘要。
- `replay.sequence_length_after` 仅在 sample 事件写出。

### 6.3 缓存

```json
"cache": {
  "capacity_positions": 256,
  "max_position_embeddings": 8192,
  "allocated": {
    "per_layer_k_shape": [1, 3, 256, 64],
    "per_layer_v_shape": [1, 3, 256, 64],
    "per_layer_bytes": 196608,
    "total_bytes": 11796480,
    "dtype": "float32",
    "note": "预分配容量，不是已写入内容；2×30×1×3×256×64×4 = 11,796,480 B = 11.25 MiB"
  },
  "layers_effective_positions": [],
  "layers_written_positions": [],
  "synchronized_positions": 0,
  "read_window": {"start": 0, "end_exclusive": 5},
  "write_window": {"start": 0, "end_exclusive": 5},
  "selected_layer_read_refs": {"k": "fwd0.block0.k_cache", "v": "fwd0.block0.v_cache"}
}
```

- `layers_effective_positions` 与 `layers_written_positions` 的长度必须等于 `n_layer=30`，由 L1 填实际值；上面的 `[]` 只是字段占位。
- `allocated` 必须由实际张量 `numel() × element_size()` 求和；不得套用 fp16 或 8192 估算。
- K 缓存保存 **RoPE 后**的 K；V 不旋转；缓存按 `Hkv` 存，不按 `Hq` 存。
- 规则：`embed` 全层等于 `position_start`，written=0；`block{l}` 中第 0—l 层等于 `position_start+T`，其余等于 `position_start`；`ln_f`/`logits` 全层等于 `start+T`；`sample` 不写缓存。
- 8192/fp16 的 180 MiB 只是配置容量估算，不是本实例实测，也不承诺加速。

## 7. 事件—张量清单

### 7.1 所有层（每个 block）

| role | shape | axes | op | 说明 |
|---|---|---|---|---|
| `attn_weights` | `[1,9,Tq,Tk]` | B,Hq,Tq,Tk | `llama.attn.softmax` | 真实 softmax 权重；Tk=total_positions |

### 7.2 选定层（layer=0，完整中间值）

| role | shape | axes | op | 说明 |
|---|---|---|---|---|
| `x_in` | `[1,T,576]` | B,T,C | `llama.block.forward` | 残差流输入 |
| `norm1_out` | `[1,T,576]` | B,T,C | `llama.rmsnorm` | pre-attention RMSNorm |
| `norm1_mean_sq` | `[1,T,1]` | B,T,1 | `llama.rmsnorm` | derived：mean(x²) |
| `norm1_inv_rms` | `[1,T,1]` | B,T,1 | `llama.rmsnorm` | derived：rsqrt(mean_sq+eps) |
| `q_proj` | `[1,T,576]` | B,T,C | `llama.attn.q_proj` | Q 投影（未旋转） |
| `k_proj` | `[1,T,192]` | B,T,Hkv·D | `llama.attn.k_proj` | K 投影（未旋转） |
| `v_proj` | `[1,T,192]` | B,T,Hkv·D | `llama.attn.v_proj` | V 投影（不旋转） |
| `q_rope` | `[1,9,T,64]` | B,Hq,T,D | `llama.rope.apply_q` | RoPE 后 Q |
| `k_rope` | `[1,3,T,64]` | B,Hkv,T,D | `llama.rope.apply_k` | RoPE 后 K |
| `v_heads` | `[1,3,T,64]` | B,Hkv,T,D | `llama.attn.v_heads` | 不旋转 |
| `k_cache` | `[1,3,Tk,64]` | B,Hkv,Tk,D | `llama.cache.append` | 缓存读窗口，已旋转 |
| `v_cache` | `[1,3,Tk,64]` | B,Hkv,Tk,D | `llama.cache.append` | 缓存读窗口，未旋转 |
| `k_expanded` | `[1,9,Tk,64]` | B,Hq,Tk,D | `llama.gqa.repeat_kv` | repeat 后的临时张量，不是新参数 |
| `v_expanded` | `[1,9,Tk,64]` | B,Hq,Tk,D | `llama.gqa.repeat_kv` | 同上 |
| `scores_scaled` | `[1,9,Tq,Tk]` | B,Hq,Tq,Tk | `llama.attn.scores` | 掩码前有限分数 |
| `causal_mask` | `[Tq,Tk]` | Tq,Tk | `llama.attn.mask` | u8；1 可读，0 不可读 |
| `attn_out` | `[1,T,576]` | B,T,C | `llama.attn.o_proj` | 输出投影后 |
| `x_after_attn` | `[1,T,576]` | B,T,C | `llama.block.residual_attn` | 残差相加 ① |
| `norm2_out` | `[1,T,576]` | B,T,C | `llama.rmsnorm` | pre-MLP RMSNorm |
| `norm2_mean_sq` / `norm2_inv_rms` | `[1,T,1]` | B,T,1 | `llama.rmsnorm` | derived |
| `gate_proj` | `[1,T,1536]` | B,T,F | `llama.swiglu.gate` | gate 投影 |
| `silu_gate` | `[1,T,1536]` | B,T,F | `llama.swiglu.gate` | SiLU(gate)；可为负或 >1 |
| `up_proj` | `[1,T,1536]` | B,T,F | `llama.swiglu.up` | up 投影 |
| `gated` | `[1,T,1536]` | B,T,F | `llama.swiglu.mul` | 逐元素乘，不是矩阵乘 |
| `down_out` | `[1,T,576]` | B,T,C | `llama.swiglu.down` | down 投影 |
| `x_out` | `[1,T,576]` | B,T,C | `llama.block.forward` | 残差相加 ② / Block 输出 |

### 7.3 embed / ln_f / logits 事件

| 事件 | role | shape | op | 说明 |
|---|---|---|---|---|
| `fwd{f}.embed` | `token_embed` | `[1,T,576]` | `llama.embed` | token 嵌入；无位置表 |
| `fwd{f}.ln_f` | `out` | `[1,T,576]` | `llama.rmsnorm` | 最终 RMSNorm 全部位置 |
| `fwd{f}.logits` | `last_row` | `[1,49152]` | `llama.lm_head` | 被采样的最后一行；scope=slice |

`logits_event` 必须记录完整 `[1,T,49152]` 形状、最后行索引、是否被采样、被哪个 sample 使用、每位置摘要；完整 logits 其余位置不进入页面载荷。

### 7.4 注意力权重

- 全部 30 层 × 9 个 Q 头，每个前向都导出 `attn_weights`；默认只深入第 0 层。
- 其他层只显示真实注意力权重；其 RMSNorm/MLP 中间值未采集必须缺席，不能用第 0 层数据替代。
- 注意力 head 与 KV 分组映射固定 `[0,0,0,1,1,1,2,2,2]`；页面切换头时应展示同一真实权重，不伪造未采集头。

## 8. 参数与共享权重

- 独立参数量 = 134,515,008；权重文件 272 个张量，全部 BF16，文件内无 `lm_head`（与 `token_embedding` 共享）。
- 参数账本按共享 Parameter 身份去重；不得沿用旧 LLaMA-2 7B 的独立输出头账本。
- `tie_word_embeddings=true`；页面必须说明"共享"不等于"没有输出计算"。

## 9. 验证文件

```json
{
  "schema": "atlas.trace.verification/2",
  "case_id": "smollm2-main-001",
  "case_sha256": "…",
  "tool": "verify_llama.py",
  "environment": {},
  "reference": {"framework": "transformers", "version": "…", "class": "LlamaForCausalLM", "device": "cpu", "dtype": "float32"},
  "summary": {"total": 0, "passed": 0, "failed": 0, "required": 0},
  "checks": [],
  "tolerances": {},
  "generated_ids": [],
  "notes": []
}
```

独立核对至少覆盖：

1. 来源/载荷/字节序/尺寸/shape/事件/source 引用；
2. 严格来源键映射、共享权重、参数总量；
3. 未采样与采样 logits、完整生成序列（长度+全部 ID）与参考一致；
4. HF 参考前向与选定中间结果；
5. 从载荷独立重算 RMSNorm、RoPE、GQA 展开、attention、SwiGLU，不只调用导出器同一函数；
6. K 缓存已旋转、V 未旋转、Hkv 存储与 Hq 展开关系；
7. 负例：长度不一致、非法头配置、未映射来源键、损坏载荷；
8. 误差按比较对象声明；不得失败后放宽阈值。

## 10. 采集预算与体积

| 内容 | 范围 | 估算 |
|---|---|---|
| 注意力权重 | 30 层 × 9 头 × 4 前向 | prefill 5×5 + decode 1×6/7/8；约 1.2 万 float，base64 后 < 0.1 MiB |
| 选定层中间张量 | 第 0 层，4 前向 | 主要是 `[1,T,1536]` 与 4 个 `[1,9,T,Tk]`；约 0.5 MiB 量级 |
| logits | 3 个被采样行 × 49,152 float | 约 0.6 MiB 原始，base64 后约 0.8 MiB |
| 缓存 | 仅容量/区间元数据 + 选定层读窗口 | 不导出零区 |
| 源码 | 两个文件快照 | 约 45 KiB |
| 合计 | 单案例 JSON | 预计 < 2.5 MiB；单 HTML 目标 ≤ 15 MiB |

若超过 6 MiB（case）或 15 MiB（GPT-2+Llama 交付总量），先报告体积组成再调整；不得为体积修改 token 排名、logits 或 dtype。

## 11. 失败与回退

- 数据损坏、版本不符、哈希不符、shape/长度/有限值错误：**显式失败**；禁止随机/模拟值回退后仍显示 SmolLM2 真实实例。
- 未采集张量：缺席 + `unavailable_reason`；页面显示未采集，不用零填充。
- 页面可保留旧 LLaMA-2 模拟资料入口，但必须明确身份，不能自动当作 SmolLM2 数据。
- 单文件离线构建若无法内联，显示明确失败，不请求外部资源后假装离线。

## 12. L1 必须冻结的开放项

1. 真实 prompt token ids、生成 token ids、实际 forward/sample 数量；
2. 第 0 层完整张量清单与最终 `role/shape`；
3. 两个源码快照的 SHA-256 与真实行号/source-map；
4. 与 HF 的 logits/完整生成对照值和容差；
5. RoPE/GQA/SwiGLU 独立重算的容差与负例；
6. 单文件体积实测。

在这些值冻结前，本契约只规定字段和边界，不宣称任何具体 token、行号或误差已通过。
