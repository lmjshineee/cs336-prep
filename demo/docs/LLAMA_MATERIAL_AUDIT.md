# Llama L0 材料核对与来源冻结

- 日期：2026-10-30
- 状态：**L0 交付物**；来源、身份与采集边界已冻结，尚未导出 LLaMA 轨迹，未改产品页面/数据。
- 依据：[LLAMA_SOURCE_GROUNDED_PLAN.md](./LLAMA_SOURCE_GROUNDED_PLAN.md)、[LLAMA_IMPLEMENTATION_REVIEW.md](./LLAMA_IMPLEMENTATION_REVIEW.md)、[LLAMA_TRACE_SCHEMA.md](./LLAMA_TRACE_SCHEMA.md)、[LLAMA_COURSE_IDS.md](./LLAMA_COURSE_IDS.md)。
- 证据：`output/qa/llama-l0/audit.json`（只读审计）、`output/qa/llama-l0/regressions.log`（14 项回归复跑）；历史修正证据 `output/qa/llama-fix/`。
- 备份：`backup/20261030T224403-L0-llama/`（修改前方案/审查/v1 契约/修正证据）。

## 1. 三种对象，三种身份

| 对象 | 用途 | 不是 |
|---|---|---|
| MiniLlama（`step06_model.py`，修正后） | 机制讲解、局部修改、缓存/GQA/RoPE 小例；L1 的唯一模型主干 | 不是预训练大模型；不宣称语言能力 |
| SmolLM2-135M（本地真实权重） | 主课程的真实轨迹、真实 logits、完整生成 | 不是 Meta LLaMA-2/3 权重；不能代表整个 Llama 系列 |
| 既有 LLaMA-2 7B 模拟图（`llama.html` 现有资料） | 原始结构资料入口 | 不能用 SmolLM2 数据填充 32 层/4096 宽节点 |

页面身份固定：

- 模型 ID：`smollm2-135m`
- 架构：`LlamaForCausalLM`
- 显示名：`SmolLM2-135M（Llama 架构）`
- 发布方：`HuggingFaceTB/SmolLM2-135M`，Apache-2.0
- 主实例：`smollm2-main-001`
- 首条学习路径：`The capital of France is`，3 个 greedy 新 token（L1 冻结实际值）

## 2. 来源冻结（当前文件）

材料根：`/Users/yunni/Projects/temp/cs336-prep-code`（只读；本轮未改）。

| 文件 | 字节 | SHA-256 | 说明 |
|---|---:|---|---|
| `llama_steps/step06_model.py` | 24,660 | `33fab3eecd388ae9eb474eac1931a5c9ae4897f05ba9ecc2d6c92b0f5ab79506` | 主模型实现（修正后） |
| `llama_steps/step09_real_llama.py` | 14,592 | `5af8867dbe24dd9f10b9709811e5da1739a789c7f296446df1a1a306de985ff3` | 真权重加载与生成（修正后） |
| `llama_steps/step08_parity.py` | 11,651 | `074e0683bc044abf21d53e61cda76a02ae658272152c97ecf800d5b94d01f024` | HF 对齐工具（参考，不是导出主干） |
| `llama_steps/step01_rmsnorm.py` | 9,305 | `8a7a92bc222dc2d5cc7d3e770fbbb4d769e09d0c4633256fab36e44e2927d240` | 机制小例 |
| `llama_steps/step02_swiglu.py` | 7,663 | `811c1a594911ed558d98c260a8889f1f98af06a09bf01ba436e3406e7024df73` | 机制小例 |
| `llama_steps/step03_rope.py` | 14,190 | `57b478c353b185d2bdc859d055b43f83194cae1862fb123cf5df70d1cdf05345` | 机制小例；图中的长度/训练结论不当作真实证据 |
| `llama_steps/step04_gqa.py` | 14,999 | `9256e4fd8469b1b711cfcf33e74ea1d8af08949bc7ea3d4b44b24dc173f8f65e` | 机制小例 |
| `llama_steps/step05_block.py` | 15,273 | `3e4bee8a48596b6bbaaf56b08d0db6e1ab8608357bd9ddfe09ae13368c784b1c` | 机制小例 |
| `llama_steps/tests/test_review_regressions.py` | 4,920 | `5895c1fcdde00253e468353b010b6a611903a170edfc34622060a63a5c00710e` | 审查修正回归 |

**版本核对**：上述 `step06_model.py`、`step09_real_llama.py`、`step08_parity.py`、回归测试的 SHA-256 与 `output/qa/llama-fix/applied.json` 的“修正后”哈希逐项一致；不是修正前版本。修正前哈希保存在 `original-hashes.json`，本轮未使用其片段或行号。

- L0 **不复制源码快照**：L1 才生成 `sources/llama/`，并据快照生成真实行号/source-map。
- L0 文档不引用具体行号，避免把修正前行号带入产品。
- 旧 `runs/`、`figs/`、`OUTPUT.md`、权重未改；历史日志只作历史记录。

## 3. 权重、配置与 tokenizer 身份

| 文件 | 字节 | SHA-256 |
|---|---:|---|
| `weights/smollm2-135m/model.safetensors` | 269,060,552 | `80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1` |
| `config.json` | 704 | `1d556eab73b69c7f11f64c557a2f9c6f440bd4c6b89bb2584a6b498c92603843` |
| `tokenizer.json` | 2,104,556 | `9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c` |
| `tokenizer_config.json` | 3,658 | `4bb9af56a342753d39374f4016a16574cab299fe088e896f425ce3c433f61424` |
| `generation_config.json` | 111 | `2056c988e990b0d13670f63f2f3b87b3b6d07edaf7a3416998ba27dab2d8a059` |

权重头验证（不加载 torch，只读 safetensors 头）：

- 272 个张量，全部 `BF16`；元素总数 **134,515,008**，与独立参数量一致；
- 文件内没有 `lm_head` 张量：输出头与 `model.embed_tokens.weight` 共享，不能重复计参数；
- `__metadata__ = {"format": "pt"}`；文件存储 bf16，运行时 CPU fp32，两个字段分开记录。

配置逐字段对齐（本地 `config.json` → `step06.LlamaConfig`）：

| 字段 | 值 |
|---|---|
| hidden_size / n_embd | 576 |
| intermediate_size / ffn_hidden | 1536 |
| num_hidden_layers / n_layer | 30 |
| num_attention_heads / n_head | 9 |
| num_key_value_heads / n_kv_head | 3 |
| head_dim | 64（576÷9；偶数） |
| vocab_size | 49,152 |
| rms_norm_eps / norm_eps | 1e-5 |
| rope_theta | 100,000（当前 SmolLM2 配置值，不代表其他 Llama 版本） |
| rope_interleaved | false（half-split） |
| tie_word_embeddings | true |
| max_position_embeddings | 8,192（配置上限；首版演示容量 256，两者分开） |
| torch_dtype | bfloat16（文件存储） |

Tokenizer（本地 `tokenizer.json`）：

- 词表 49,152；BOS/EOS id 均为 0（`<|endoftext|>`）；
- 真实复现 `The capital of France is` → token `["The","Ġcapital","Ġof","ĠFrance","Ġis"]`，ids `[504,3575,282,4649,314]`；
- 单 token 不一定能独立解码成完整字符；页面按字节/原始 token/解码文本分别记录。

**上游来源**：

- 模型说明：<https://huggingface.co/HuggingFaceTB/SmolLM2-135M>
- 发布配置：<https://huggingface.co/HuggingFaceTB/SmolLM2-135M/raw/main/config.json>
- 本地下载器使用可变 `main`，未固定 commit；因此**不声称固定上游 revision**，载荷身份以本地文件 SHA-256 为准。L1 页面来源需明确写这一点，不能只写“来自 HF main”。

## 4. 独立核对证据与边界

### 4.1 本轮复跑（L0）

| 检查 | 结果 |
|---|---|
| 材料只读审计（本文件 §2/§3） | 通过；所有字段/哈希/计数一致（`audit.json`） |
| `tests/test_review_regressions.py`（14 项） | **14/14 通过**（本轮重新执行，约 0.05 s） |
| 修正后源码哈希与 `applied.json` | 4/4 一致 |
| 配置 → `LlamaConfig` 字段 | 逐项一致；非法配置负例在回归中拒绝 |
| tokenizer prompt ids | 与历史 `runs/step09.txt` 一致 `[504,3575,282,4649,314]` |

### 4.2 已有历史证据（修正后，未在本轮重跑完整模型）

| 检查 | 结果 | 来源 |
|---|---|---|
| SmolLM2 前向 vs HF logits | 最大绝对差 **4.768e-05**（<1e-3） | `llama-fix/step09` 证据 |
| top-5 | 完全一致 | 同上 |
| 12-token greedy 完整序列 | 长度 + 全部 ID 一致 | 修正后 `step09` |
| B=2、前缀分块 5/3/7 | 最大差 1.49e-07 | 审查修正复验 |
| GQA 9:3 映射 | `[0,0,0,1,1,1,2,2,2]` | 审查修正复验 |
| 回归负例 | 长度不一致、非法头配置、未知源键、tied 头不一致均拒绝 | 14 项回归 |

### 4.3 明确未验证（不得写成已通过）

- 尚未导出 LLaMA 轨迹；尚未生成第 0 层中间张量/全部层注意力权重。
- 尚未独立重算 RMSNorm、RoPE、GQA 展开、SwiGLU（L1 验证器任务）。
- 尚未验证 bf16/MPS/GPU 推理；主路径仅 CPU fp32。
- 尚未验证 8,192 长度质量、速度或显存；180 MiB 只是公式估算。
- 尚未验证浏览器页面、真实轨迹回放、课程与离线单文件；当前 `llama-atlas.html` 仍是旧模拟内容。
- 尚未进行用户试学；不能宣称教学效果。

## 5. 采集预算（L1 预冻结）

| 内容 | 范围 | 目的 |
|---|---|---|
| 注意力权重 | 30 层 × 9 头 × 4 前向 | 真实 GQA/因果注意力；约 1.2 万 float |
| 第 0 层中间张量 | RMSNorm/Q/K/V/RoPE 前后/共享/展开 KV/mask/SwiGLU/残差 | 支撑 L4a/L4b 局部计算 |
| Logits | 3 个被采样最后行（49,152 维/行）；其余位置仅摘要 | 完整概率质量，不重归一化 |
| 缓存 | 容量、逐层有效/写入区间、读窗口；不导出零区 | prefill/decode 教学 |
| 源码 | `step06_model.py`、`step09_real_llama.py` 原文快照 | 离线追溯 |
| 不导出 | 整份权重、全部层 MLP 激活、零缓存区、未使用位置完整 logits | 体积与真实性边界 |

体积估计：case 预计 <2.5 MiB；单 HTML 目标 ≤15 MiB。超 6 MiB（case）或 15 MiB（交付总量）先报告组成，不通过修改 token 排名/概率来缩体积。

## 6. L0 检查对照

| 方案要求 | 完成情况 |
|---|---|
| 核对修正后源指纹与权重/配置对应 | 完成：源码/权重/配置/tokenizer 全哈希；4/4 修正后哈希一致；权重 272 BF16、参数 134,515,008 |
| 上游来源可核对 | 完成：发布方链接、本地文件哈希；明确未固定 commit，不假装不可变 |
| 不复用修正前源码片段和行号 | 完成：L0 文档不引用行号；L1 才基于快照生成 source-map |
| 模型身份/验证边界/采集预算明确 | 完成：本文件 + schema + 课程 ID 表 |
| v1/v2 共存方式明确 | 完成：schema v2 独立目录/版本精确分派；GPT-2 v1 不改 |
| 不覆盖 GPT-2/旧 LLaMA 资料，不导出 | 完成：仅新增文档/审计工具与证据；产品代码/数据未改 |

## 7. 风险与 L1 输入

1. **上游可变**：下载器未固定 commit；若未来需要严格复现，应记录 HF commit/ETag，或只依赖本地 SHA。L1 manifest 必须写 `revision:null`/本地身份。
2. **运行时/配置版本不同**：config 标注 `transformers_version 4.40.1`，审查运行环境为 Transformers 5.21.0；L1 以实际运行版本记录 reference，不把 config 字段当运行版本。
3. **行号未生成**：L1 复制快照后由 `llama_source_map.py` 按真实修正后文件生成；不得凭旧审查行号填写。
4. **中间值未核对**：L1 验证器必须从载荷独立重算 RMSNorm/RoPE/GQA/SwiGLU，不能只调用导出器同一函数。
5. **体积**：首版只导出 3 个采样最后行；若增加位置 logits 需先报告体积。

**L1 建议命令/产物**（L1 实现时冻结）：

```bash
# 在 GPT2-LLaMA-Atlas 目录，新增工具后：
.venv/bin/python tools/trace/export_llama.py --material-root /Users/yunni/Projects/temp/cs336-prep-code --data-dir data/llama --case-id smollm2-main-001
.venv/bin/python tools/trace/verify_llama.py --material-root /Users/yunni/Projects/temp/cs336-prep-code --data-dir data/llama --case-id smollm2-main-001
```

L1 完成标准：一条真实轨迹与独立核对通过；不改页面；不覆盖 GPT-2 数据；源快照/source-map v2 与 schema 对齐。
