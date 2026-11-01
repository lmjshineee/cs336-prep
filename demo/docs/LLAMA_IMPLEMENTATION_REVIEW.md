# LLaMA 学习代码独立审查

- 日期：2026-10-30。
- 范围：`/Users/yunni/Projects/temp/cs336-prep-code/minimal_llama.py` 与 `llama_steps/`；当前可视化及其 GPT-2 实施不作改动。
- 结论：`step06_model.py` 的默认 CPU fp32 主干和 `step09_real_llama.py` 的当前真实权重实例复验通过；仍存在验证、边界和教学表述问题，不能给整套材料“无问题”的结论。
- 用户要求审查通过后再写 LLaMA 详细方案。本轮因此先交付审查，不生成已通过前提下的详细实施方案；未修正原代码。

> 后续更新（2026-10-30）：用户已授权直接修正。R1—R3 及 C1—C3 中的契约、映射和主要教学表述已修正，14 项新增回归检查与 7 个原有自检通过。以下保留首次审查发现作为历史；当前状态以文末“修正复验”及材料目录 `llama_steps/REVIEW_FIXES.md` 为准。

## 1. 实际模型身份

真实权重来自 `HuggingFaceTB/SmolLM2-135M`，配置的实现类型为 `LlamaForCausalLM`。这是采用 Llama 架构的 SmolLM2 实例，不能作为现有 LLaMA-2 7B 或 Meta Llama-3 权重的真实运行数据。

本地与发布方配置一致的主要字段：30 层、隐藏宽度 576、9 个 Q 头、3 个 KV 头、head_dim 64、FFN 1536、词表 49,152、共享嵌入/输出权重、theta 100,000、配置最大上下文 8,192。本地 step09 另将演示上下文设为 256；权重载荷标为 bf16，当前模型载入后在 CPU fp32 中运行。

来源：

- [发布方模型说明](https://huggingface.co/HuggingFaceTB/SmolLM2-135M)。
- [发布方 config.json](https://huggingface.co/HuggingFaceTB/SmolLM2-135M/raw/main/config.json)。

后续页面应表达为“Llama 架构：SmolLM2-135M 实例”。概念课程可使用 MiniLlama；其小模型、SmolLM2 真实实例与原 LLaMA-2 7B 模拟图必须有独立配置和数据身份。

## 2. 本次实际运行

环境：CPU fp32，PyTorch 2.14.1、Transformers 5.21.0、Python 3.14.3，限制两条计算线程；禁用联网和 Python 字节码写入，未训练模型、未启动三维页面。

| 检查 | 结果 |
|---|---|
| step01 RMSNorm 自检 | 通过 |
| step02 SwiGLU 自检 | 通过 |
| step04 GQA 自检 | 通过 |
| step05 Block 自检 | 通过 |
| step06 完整模型、缓存、因果等自检 | 通过 |
| step08 HF 对齐与三种错误对照 | 通过，正确实现 logits 最大差 1.192e-07 |
| step09 真实权重实例、top-5 和当前生成实例 | 通过，logits 最大差 4.768e-05 |
| 额外 B=2，前缀分块 5/3/7 个 token 的缓存/全量对照 | 通过，最大差 1.4901161193847656e-07 |
| 额外 GQA 3 组 KV、每组映射 3 个 Q 头的复制顺序 | `[0,0,0,1,1,1,2,2,2]` |
| 源材料、runs、figs 与本地权重前后哈希 | 已记录文件无变化 |

结构化结果、源哈希、复验程序与原始日志见 `output/qa/llama-review/`。本轮没有重跑 step03 绘图或 step07 训练，没有覆写原始实验记录，也没有复验长期性能和质量结论。

## 3. 需要修正的问题

### R1（P2）：生成一致性断言只比较共同前缀

- 位置：`llama_steps/step09_real_llama.py:234–239`。
- 原因：取 `n=min(ours_length,hf_length)` 后只比较 `:n`，没有要求完整长度相同。任意一边提前结束，只要已有前缀相同，也会打印“12 个 token 完全相同”。
- 验证：独立构造长序列 `[1,2,3,4]` 和短序列 `[1,2]`，当前谓词返回 True，完整序列相等返回 False。当前真实默认 prompt 两边实际相同，并未因此被否定；缺陷在验证可能误报。
- 修正建议：先明确 EOS 和长度限制策略，断言生成长度/新 token 数符合预期，再比较完整 token ID 张量；额外保留提前停止、长度不同而共同前缀相同的负例。

### R2（P2）：单文件版对非法头配置静默向下取整

- 位置：`minimal_llama.py:100–111`。
- 原因：`head_dim=n_embd//n_head` 没有检查整除，Q 投影输出又使用 `n_head*head_dim`，因此不能整除时会悄悄缩窄，而不是拒绝配置。
- 验证：`n_embd=12,n_head=5,n_kv_head=1` 实际能前向输出 `[1,3,31]`，但 Q 投影宽度为 10。主版本 `step06_model.py` 在构造时已有整除检查。
- 修正建议：单文件版补正整数、`C%H==0`、`H%Hkv==0`、RoPE head_dim 为偶数的配置验证；非法配置应有明确错误。保留合法 MHA/GQA/MQA 与非法配置的检查。

### R3（P2）：生成提前停止时，速度分母仍为请求 token 数

- 位置：`llama_steps/step09_real_llama.py:140–154`。
- 原因：循环会因上下文限制提前结束，却始终除以 `n_new`；请求数不等于实际生成数时低估每 token 耗时。计时还包含采样/拼接与循环内的最后一次前向，不宜直接称为纯 decode 前向耗时。
- 验证：独立使用 `block_size=8`、7 个 prompt token、请求 12 个新 token，函数实际生成 2 个，但仍除以 12；此条件下每生成 token 的平均耗时被低估 6 倍。
- 修正建议：分母取实际完成的生成/前向次数，并明确测量口径；对零 token 请求、上下文不足和提前停止单独处理。是否允许最后选出的 token 超出可继续计算的上下文范围，必须明确，而不能把不同函数的策略混用。

## 4. 接入前还应统一的内容

### C1：两份 RMSNorm 的 dtype 行为不同

`minimal_llama.py:56` 先转回输入精度，再乘 fp32 weight；`step06_model.py:120` 在乘 weight 后统一转回输入精度。

独立检查：默认 fp32 参数、bf16 输入时，单文件返回 fp32，step06 返回 bf16。默认 CPU fp32 自检不会发现这个差异。不同训练/自动混合精度策略可能有不同 dtype 约定，本轮不把其中一种笼统定为所有参考实现的唯一行为；但它们不能在可视化中被当成完全等价的同一源码版本。

建议以 step06 作为后续导出的唯一模型主干，明确支持的 dtype；单文件若继续称压缩版，应对齐其契约并补充相应检查。

### C2：strict 加载不等于检查了所有来源键

`step09_real_llama.py:107` 先过滤未在 mapping 中的来源键。后面的 `ours.load_state_dict(...,strict=True)` 只能验证过滤后的目标结构。

独立检查：传入一个未映射的来源张量，`hf_to_ours` 直接返回空映射。不能仅凭这里的 strict=True 声称所有任意来源张量都检查过。不过当前 main() 还会把原始 sd 加载进 HF 并检查 unexpected keys，对当前固定模型提供了另一道检查，不能把局部不足误报为当前加载已失败。

导出适配器若独立复用该映射函数，应断言全部源键被映射或进入明确白名单，并验证共享权重关系和指纹。

### C3：应修订的教学表述

- “真实 Llama 权重”应明确为 SmolLM2 的真实权重，避免与 Meta Llama 产品版本混淆。
- `step09_real_llama.py:174` 的 theta=1e5 只来自当前 SmolLM2 配置，不应写成所有“Llama-3 系”的固定值。
- RoPE 的相对位置性质针对给定 q/k 内容向量的旋转点积；不能写成真实注意力分数只由距离决定。把相同内容向量铺到各位置得到的 Toeplitz 图也需交代构造条件。
- 当前模型预计算的 RoPE 表和 KV cache 均有长度限制；不能说这一实现只受缓存约束。公式能延伸不证明训练外长度上的模型质量。
- RMSNorm 含 epsilon 和可学习 gamma，输出 RMS 不保证始终严格为 1；无减均值也不代表数值均值保持不变。
- SwiGLU 的 SiLU 门可以为负或大于 1，不是 0/1 开关或概率。三个 dense 投影仍完整执行。
- “GQA 被所有新模型采用”“任何一处不一致就立刻乱码”等断言应去掉普遍性；是否对齐应由数值与结构检查支持。
- 逐元素近似对齐不应写成逐位相同；当前复验结果已显示非零浮点误差。
- step08 的 GPT-2 8k KV 行仅是外推容量算式，不能当作本地 GPT-2 支持 8k 输入的运行结果。
- 训练 loss 下降和流畅文本不构成实现正确性的独立证明；单个玩具训练对照不能外推真实模型质量或架构优越性。

## 5. 后续处理建议

先确认一轮限定修正：R1—R3、单文件与主版本的契约一致性、相关说明文字及必要负例；原始旧日志保留为旧版本记录，重新运行结果另行存放。避免为了修复历史措辞而直接改写旧实验输出。

修正后复核默认实例及上述反例，再写 LLaMA 详细可视化方案。该方案应沿用现有 GPT-2 的离线轨迹和源码映射思想，但必须扩展 schema 中的 Hkv、RoPE 前后 Q/K、未复制 KV 与复制后读取张量、gate/up/down 等字段；不能把 SmolLM2 数据塞入原有 32 层 LLaMA-2 7B 配置。

模型代码位于上级目录；本轮没有修改它们。可视化目录已有其他 GPT-2 批次的新进展，本轮仅阅读，不覆盖其交接文件、产品代码或数据契约。

## 6. 修正复验（用户授权后）

- 已修改：`minimal_llama.py`、`llama_steps/step06_model.py`、`step08_parity.py`、`step09_real_llama.py`、`README.md`；新增 `llama_steps/tests/test_review_regressions.py` 和 `REVIEW_FIXES.md`。
- R1：完整生成 shape、新 token 数和全部 token IDs 一起核对，长度不一致负例被拒绝。
- R2：两份配置均验证正整数、整除和 RoPE 偶数维；原 C=12/H=5 反例明确报错，合法 MHA/GQA/MQA 继续可用。
- R3：计时分母为实际新 token 数；固定 6 秒、实际生成 2 个 token 的可控检查得到 3 秒/token；零请求无前向，负请求和超长 prompt 拒绝。标清保留的末 token 和循环计时语义。
- C1：单文件 RMSNorm 与主版本对齐，float32/bfloat16 输入的本地精度契约和输出数值一致。
- C2：未知来源键报错，共享输出头别名只在与嵌入一致时接受。
- C3：修订当前 README、step08/09 的关键输出和单文件注释，原历史 runs/figs/OUTPUT 不重写；剩余早期措辞的适用边界集中标在 REVIEW_FIXES.md。
- 14 项回归检查和 7 个既有非训练自检全部通过；真实 SmolLM2 实例 logits 最大差 4.768e-05，top-5 和完整 12-token 贪心序列一致。
- 证据：`output/qa/llama-fix/verification.json`、各脚本新日志、`applied.json`、差异文件与回归程序。修改前源文件及摘要在 `backup/20261030-llama-review-fix-before/`，不含重复权重。
- 结论：已发现的修正项通过复验，主版本可作为后续可视化导出的实现基础。这不表示完整低精度推理、GPU/MPS、模型质量、训练对照或任意输入均已验证。
- 已据此编写 [LLaMA 详细可视化方案](./LLAMA_SOURCE_GROUNDED_PLAN.md)。仅编写方案，未接入 LLaMA 真实轨迹，未改当前产品代码或 HTML。
