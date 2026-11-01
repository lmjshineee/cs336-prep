# Llama 课程与稳定 ID 表（L0 定稿）

- 日期：2026-10-30
- 状态：**L0 交付物**。本文件固定课程 ID、步骤 ID、模型 ID、实例 ID 与记录命名空间；L3—L5 按此实现，不得随意改名。
- 依据：[LLAMA_SOURCE_GROUNDED_PLAN.md](./LLAMA_SOURCE_GROUNDED_PLAN.md) §4、§10、§11；[LLAMA_TRACE_SCHEMA.md](./LLAMA_TRACE_SCHEMA.md)。
- 本文件只固定 ID 与教学意图，不表示课程已经实现。实际数据映射在 L1 导出、L2 接入后冻结；若必须改 ID，必须提供旧 ID→新 ID 的迁移记录，不能静默让旧进度失效。

## 1. 命名空间

| 项目 | 值 |
|---|---|
| 模型 ID | `smollm2-135m` |
| 页面入口 | `llama.html`（HTTP 开发）/ `llama-atlas.html`（离线单文件） |
| 主实例 ID | `smollm2-main-001` |
| 新课程记录键 | `llama.progress.v1` |
| 课程修订 | `llama-r1` |
| 旧 LLaMA-2 模拟资料 | 保留为资料入口，不写入新课程/真实节点 |
| GPT-2 记录 | `gpt2.progress.v2` 等保持原样，不读写、不继承 |

规则：

- 旧 `gpt2llama.lessons`、`gpt2llama.progress.v1` 与 GPT-2 新键只读或保持原状；不因用户读过 GPT-2 就推断 Llama 课通过。
- 新记录包含 `model: smollm2-135m`、revision、稳定 lessonId/stepId、实例 ID；导入格式沿用 S2b 的 `atlas.progress.export/1` 思想，L5 实现适配。
- 不把 SmolLM2 课程写入 `llama2.progress.v2`（那是旧 LLaMA-2 页面命名空间）。
- 模型切换/数据缺失时暂停课程，不把模拟内容接到真实课程。

## 2. 课程总览

| # | lessonId | 标题 | 主要问题 | 步骤数 |
|---|---|---|---|---:|
| 01 | `overview` | 整体回路：一座 30 层塔 | 输入如何走到输出与缓存？ | 4 |
| 02 | `tokens` | 输入与形状：没有位置表 | token ID 怎样变成 576 维？ | 4 |
| 03 | `rmsnorm` | RMSNorm：不减均值的尺度 | RMS、eps、gamma 各做什么？ | 4 |
| 04 | `rope` | RoPE：位置怎样进入 Q/K | 没有位置嵌入时，绝对位置如何进入？ | 5 |
| 05 | `gqa` | GQA 与因果注意力 | 9 个 Q 头怎样读 3 组 KV？ | 5 |
| 06 | `swiglu` | SwiGLU 与残差 | gate/up 两支路怎样相乘并加回？ | 5 |
| 07 | `output` | 输出、概率与参数 | 最后一行 logits 怎样变成 token？ | 4 |
| 08 | `cache` | 生成与 KV cache | prefill/decode 分别读写了什么？ | 5 |
| 09 | `summary` | 复述与验证 | 一次真实生成怎样连起来？ | 4 |

合计 **40 个稳定步骤**，九道单题测验（每课一道，ID 为 `${lessonId}.quiz`，不计入步骤）。旧混合课程的 37 步编号不用于本课程。

## 3. 步骤 ID 与教学意图

### 01 `overview`（4 步）

| stepId | 教学意图 | 计划数据/源码落点 |
|---|---|---|
| `overview.s1` | 单塔 30 层；SmolLM2 与旧 LLaMA-2 7B 资料分开 | 场景/模型身份；`MiniLlama.forward` |
| `overview.s2` | 一个 Block：RMSNorm→Attention→Add→RMSNorm→SwiGLU→Add | `fwd0.block0`；`LlamaBlock.forward` |
| `overview.s3` | 输入、输出头、KV cache 三个锚点 | `fwd0.embed` / `fwd0.logits` / cache 元数据 |
| `overview.s4` | 一次真实生成的全回路 | `sample0` 与完整生成摘要 |

### 02 `tokens`（4 步）

| stepId | 教学意图 | 计划数据/源码落点 |
|---|---|---|
| `tokens.s1` | 真实 BPE；`The capital of France is` 的 token/id | `fwd0.embed` 输入；tokenizer |
| `tokens.s2` | token 嵌入 `[1,5,576]`；没有可学习位置表 | `fwd0.embed.token_embed`；`token_embedding` |
| `tokens.s3` | 输出头与嵌入共享权重，但有实际输出计算 | 参数身份/lm_head；`MiniLlama.forward` |
| `tokens.s4` | 核对一个真实 token 的 id、位置与向量切片 | `token_embed` 切片；`sample0` 序列位置 |

### 03 `rmsnorm`（4 步）

| stepId | 教学意图 | 计划数据/源码落点 |
|---|---|---|
| `rmsnorm.s1` | pre-norm：注意力前、MLP 前各一次 | `fwd0.block0.norm1_out` / `norm2_out` |
| `rmsnorm.s2` | mean(x²)→+eps→rsqrt→逐元素缩放 | `norm1_mean_sq`/`norm1_inv_rms`；`RMSNorm.forward` |
| `rmsnorm.s3` | gamma 逐通道；输出 RMS 不保证为 1、均值不保证保留 | `gamma`、`norm1_out`；数值/统计 |
| `rmsnorm.s4` | 完整向量 RMS 与切片 RMS 的区别 | 切片范围与全维统计 |

### 04 `rope`（5 步）

| stepId | 教学意图 | 计划数据/源码落点 |
|---|---|---|
| `rope.s1` | 没有位置嵌入；位置 m 进入 Q/K | `fwd0.block0` positions；`build_rope_cache` |
| `rope.s2` | half-split 通道对 `(i,i+32)`；cos/sin | `cos`/`sin` 表；`apply_rope` |
| `rope.s3` | 旋转 Q、K；V 不旋转 | `q_proj`→`q_rope`、`k_proj`→`k_rope`、`v_heads` |
| `rope.s4` | 缓存 K 是旋转后的；绝对位置随 decode 前进 | `k_cache`、`write_window`；`KVCache.append` |
| `rope.s5` | “固定内容仅改位置”的对照条件与边界 | 机制小例（玩具/切片），不写入真实轨迹 |

### 05 `gqa`（5 步）

| stepId | 教学意图 | 计划数据/源码落点 |
|---|---|---|
| `gqa.s1` | Q 头 9、KV 头 3；每 3 个 Q 共享一组 KV | `attn_weights` head/轴；配置 |
| `gqa.s2` | 固定映射 `[0,0,0,1,1,1,2,2,2]` | `repeat_kv`；head 映射 |
| `gqa.s3` | 缓存按 Hkv 存；repeat 后按 Hq 读，不是新增参数 | `k_cache`/`v_cache` vs `k_expanded`/`v_expanded` |
| `gqa.s4` | 因果 mask、softmax、输出投影 | `causal_mask`、`scores_scaled`、`attn_weights`、`attn_out` |
| `gqa.s5` | 形状边界：`[1,3,Tk,64]`→`[1,9,Tk,64]`→`[1,9,Tq,Tk]` | shape 链与切片 |

### 06 `swiglu`（5 步）

| stepId | 教学意图 | 计划数据/源码落点 |
|---|---|---|
| `swiglu.s1` | 576→1536 两条支路 gate/up | `gate_proj`/`up_proj`；`SwiGLU.forward` |
| `swiglu.s2` | SiLU(gate) 可正可负、可大于 1，不是概率/开关 | `silu_gate` 数值分布 |
| `swiglu.s3` | 与 up 逐元素相乘，不是矩阵乘 | `gated` |
| `swiglu.s4` | down 1536→576，再残差相加 | `down_out`、`x_out` |
| `swiglu.s5` | 三个 dense 投影都完整执行；不以“变暗”暗示稀疏跳过 | 事件/操作顺序；源码 |

### 07 `output`（4 步）

| stepId | 教学意图 | 计划数据/源码落点 |
|---|---|---|
| `output.s1` | 所有输入位置都产生 `[1,T,49152]` logits | `fwd0.logits` 完整 shape |
| `output.s2` | 生成只取最后一行 | `last_row`；`generated_ids` |
| `output.s3` | 完整词表 softmax；Top-N 不重新归一化 | `sample0` 概率/余量 |
| `output.s4` | greedy 选中 token；共享 lm_head 不等于无计算 | `sample0`；参数身份 |

### 08 `cache`（5 步）

| stepId | 教学意图 | 计划数据/源码落点 |
|---|---|---|
| `cache.s1` | prefill 逐层写入 5 个位置 | `fwd0.block0`→`block29`；cache 层数组 |
| `cache.s2` | decode 只来 1 个新 token，读全部历史 | `fwd1.block0`；`read_window`/`write_window` |
| `cache.s3` | 预分配 11,796,480 B 与有效字节的区别；Hkv | `allocated`、层有效长度；`KVCache` |
| `cache.s4` | 采样只追加序列；下一轮前向才写 KV | `sample0` 与 `fwd1.block0` |
| `cache.s5` | 前进/倒退后缓存状态由事件重建，历史不被改写 | 回放一致性 |

### 09 `summary`（4 步）

| stepId | 教学意图 | 计划数据/源码落点 |
|---|---|---|
| `summary.s1` | 从真实 prompt 到 sample 的完整链路 | 40 步/事件总览 |
| `summary.s2` | 模型身份、实例与限制（SmolLM2，不是 Meta；256 演示容量 vs 8192 配置） | manifest/model 身份 |
| `summary.s3` | 数值—事件—源码的追溯与验证边界 | source-map、verification |
| `summary.s4` | 用户试学三任务：RoPE 后 K 来源、一个 Q 头读哪组缓存、末 token 何时写缓存 | 真实轨迹/源码 |

## 4. 每步必须记录的教学/交互字段（L5 实现）

每个步骤后续至少定义：

- `id`：本表固定值；
- `title` / `text`：问题→设置→操作→结果→为什么，不先给结论；
- `event` / `seek`：真实事件 ID 或事件谓词（L1/L2 后填）；
- `focus`：`smollm2-135m` 节点，不使用 `llama2.L*`；
- `sourceOps`：source-map operation ID（可多个）；
- `viewer`：真实张量/切片角色（第 0 层）或机制小例标记；
- `quiz`：只考已讲概念，答错可查看当前实例并重试；
- `isToy`：玩具/切片/真实观测的显式标识，禁止把玩具结果写成真实轨迹。

## 5. 记录与恢复要求（L5）

- 新键 `llama.progress.v1`；旧键只读；不覆盖 GPT-2 键。
- 稳定步骤 ID 用于刷新恢复、导入/导出与旧数字步骤映射；修订后不得误继承旧答题通过。
- 课程只在真实实例就绪时启动；缺失/损坏时明确提示，可进入独立机制资料，不自动回退到随机数据。
- 用户实际试学前，状态写“功能验证通过，学习体验待验收”。

## 6. 变更规则

- 本表 ID 自 L0 起冻结。若 L1 真实数据无法支撑某一步，可调整该步的 `event`/文案，但不得改 `stepId`；确实需要改 ID 时，必须提交旧→新映射、记录迁移说明并提升课程修订。
- 新增步骤只能追加到对应课程末尾，使用下一个递增 `sN`，不能插入中间复用旧编号。
- 任何改动同步更新本文件与 L5 交接，不同步视为未完成。
