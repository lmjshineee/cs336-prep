# 从 cs336-prep-code 到 CS336：26 周进阶学习计划

> 版本：2026-10-29
> 适用起点：已完成本仓库 `transformer_steps/` + `real_llm_lab/`，尚未开始 CS336 作业
> 默认环境：Apple M5 / 16 GB 统一内存 / 无 NVIDIA CUDA / 约 229 GB 可用磁盘
> 默认节奏：12–14 小时/周 × 26 周（2026-11-02 → 2027-05-02）
> 默认预算：Modal 现金支出 $50–220（典型 $100–200），硬上限 $300；优先用每月 $30 免费额度
> 本文件只有计划，不含任何作业实现、伪代码或答案。

---

## 0. 一页纸结论

1. **下一步不是再学一个框架，也不是再读几篇论文，而是按 CS336 2026 的顺序做五份作业：A1 → A2 → A3 → A4 → A5。**
   你的 prep 已经把「Transformer 长什么样、为什么这样算」打通了；CS336 考的是「从零实现 + 性能/规模/数据/对齐」，这两件事之间还有很大的工程 gap。

2. **A1 必须自己完整做，不能跳。**
   你已经写过 GPT-2 风格的模型，但 2026 版 A1 要求的是 Llama 风格（BPE、RMSNorm、SwiGLU、RoPE、AdamW、checkpoint、leaderboard 45 分钟训练上限），其中 BPE 和训练工程对你仍然是全新的。

3. **A2 是整个计划的分水岭，也是你机器最不够用的地方。**
   FlashAttention-2（PyTorch + Triton）、DDP、FSDP、sharded optimizer 都需要 NVIDIA GPU；Mac 上只能写代码和跑单测，必须租云 GPU。A2 预计占总时间的 1/4，超期优先从 A3/A4 借时间，不砍 A2 的核心。

4. **你的已有优势在「推理/缓存直觉」，不在「分布式/内核」。**
   `mini-kv-cache` 的 1.39×→3.78×、logit 误差 ~5e-7，加上 `real_llm_lab/exp3`，说明你已经真正量过 KV cache；A2 的 inference 部分会轻松一些，但 Triton 和 DDP/FSDP 是从零开始。

5. **默认方向：偏系统（训练/推理基础设施），研究素养作为第二条腿。**
   从你的仓库分布看，系统类投入最多；但绝大多数是脚手架/文档/TODO，真正跑通的代码很少。计划里每个交付物都必须是「测试通过 / benchmark 数字 / commit」，不是又一份设计文档。

6. **本周（10-29 → 11-08）只做 P0：环境、基线、读 handout、跑一次 GPU 冒烟。不写模型代码。**

---

## 1. 起点盘点（有证据的部分）

### 1.1 已经证明的能力

| 能力 | 证据 | 判断 |
|---|---|---|
| PyTorch 基础、张量形状、训练循环 | `python_basics/py01–py05`、`transformer_steps/step01–08` | 够用，但只到「能跑通」 |
| GPT-2 风格 Transformer 从零实现 | `minimal_transformer.py`、`step03–07` | 已掌握单机、小规模版本 |
| 真实权重加载与数值对齐 | `real_llm_lab/exp1`：logits 最大差 0.00e+00，124,439,808 参数 | **强证据**，少见 |
| KV cache 正确性与收益 | `exp3` 2.19×→4.64×；`mini-kv-cache` 自带 28 个测试、benchmark 1.39×→3.78×、logit 误差 ~5e-7 | 推理侧直觉已建立，但只在 CPU/玩具规模 |
| 注意力行为分析 | `exp2` induction head：L5H5 得分 0.7484，均值 0.0771 | 会做可解释性小实验 |
| 因果掩码/困惑度诊断 | `exp4` 无掩码训练 loss 0.087、推理 7.88 | 理解「训练目标 ≠ 推理行为」 |
| 采样策略 | `exp5` 温度/top-k 对熵的影响 | 入门 |
| 训练断点/恢复的工程概念 | `MicroLM` 里自己改过 checkpoint resume（+132 行）和预训练脚本 | 有训练循环工程感，但非 CS336 规模 |
| C++/CMake 基础 | `cs106l-assignments` A1–A2 有实现 | 基础够，系统课还没真正做 |

### 1.2 还没有证明的能力（计划要补的）

| 缺口 | 现状证据 | 对应阶段 |
|---|---|---|
| BPE 分词器（训练/编码/特殊 token/内存效率） | 只做过字符级分词；`MyTransformer` 是第三方脚手架，339 行全是 `NotImplementedError`，**不算进度** | A1 W2–W3 |
| 现代架构（RMSNorm、SwiGLU、RoPE） | prep 用的是 LayerNorm/GELU/学习式位置编码 | A1 W4–W5 |
| AdamW、cosine schedule、grad clip、checkpoint 细节 | 训练循环会写，但 2026 handout 的 AdamW 顺序/权重衰减细节没做过 | A1 W6 |
| GPU 性能工程（profiling、roofline、mixed precision、activation checkpointing） | 全部实验在 CPU fp32；无 CUDA | A2 W8–W14 |
| Triton / FlashAttention-2 | 仓库里没有任何 Triton/CUDA kernel | A2 W9–W11 |
| 分布式（DDP/FSDP/sharded optimizer） | 所有仓库 grep 不到 DDP/FSDP/torchrun/NCCL/deepspeed | A2 W12–W14 |
| Scaling laws | 未做过 | A3 |
| 数据管线（Common Crawl、语言识别、PII、质量过滤、MinHash 去重） | 未做过 | A4 |
| SFT / GRPO / DPO / 评测 | 未做过（MicroLM 的 SFT/LoRA 是别人代码） | A5 |
| 工程习惯（git、实验日志、可复现） | `cs336-prep-code` 没有 git；不少仓库只有文档和 TODO | 全程 |

### 1.3 环境与硬约束

- **机器**：MacBook Air M5，10 核 CPU，16 GB 统一内存，无 NVIDIA GPU；MPS 可用，但 **Triton 不支持 macOS**，A2 的 kernel 部分必须在云端做。
- **磁盘**：约 229 GB 可用。TinyStories 约 2 GB，OWT 样本约 12 GB，A4 的 WET 子集要额外留 30–50 GB。
- **Python**：系统 3.14，prep 的 `.venv` 是 3.14 + torch 2.14.1；CS336 作业用 `uv` + Python 3.13。**不要混用环境**，每个作业一个独立 `uv` 环境。
- **已有作业仓库**：`/Users/yunni/Projects/assignment1-basics` 是官方 26.0.3 学生版，工作区干净（只有 `.DS_Store` 未跟踪），还没有实现。直接在这里做，不要重新下载。
- **云端**：目前没有 Modal/AWS 等 CLI 或账号，P0 要开通 Modal 并设预算上限。
- **课程状态**：CS336 Spring 2026 已结课，讲义、录像、作业全部公开；2026 版 A1 的架构和 A2 的 B200 假设都以仓库 `26.0.x` 为准。

---

## 2. 目标、路线与节奏

### 2.1 26 周后要达到什么

不是「看完 19 讲」，而是能拿出下面这些可验证的东西：

1. **A1**：在空白文件里从零写出 BPE + Llama 风格 LM + 训练工具，测试全绿；训练出一个能生成连贯 TinyStories 的模型（低资源 40M token 或云上完整配置），并给出 loss/吞吐/显存数字；leaderboard 45 分钟为可选。
2. **A2**：自己实现 FlashAttention-2（先 PyTorch、再 Triton），自己实现 DDP 和 FSDP，能在 2 张 NVIDIA GPU 上跑通并给出 scaling efficiency、显存、带宽的 benchmark。
3. **A3**：用 4 个以上模型规模拟合 scaling law，能做 compute-optimal 预测，并解释误差来源。
4. **A4**：把原始 WET 数据走完「抽取 → 语言识别 → PII → 质量/安全过滤 → 去重 → 训练」全流程，用下游 loss 证明过滤/去重有用（或诚实地证明在你的规模下没用）。
5. **A5**：完成 SFT + GRPO，GSM8K 指标相对 SFT 有提升；能解释 advantage、importance ratio、clip、reward hacking。
6. **Capstone**：一个 3 周内可完成、有数字的系统/研究小项目（默认：把 nano-vllm 从「只读」变成「实测 + 自己实现 paged attention / continuous batching」）。
7. **写作**：每阶段一篇 1–2 页的英文/中文 writeup，包含问题、设计、数字、失败原因。

### 2.2 三条节奏（里程碑相同，只改日期）

| 轨道 | 周期 | 每周投入 | 适合 | 预计结束 |
|---|---|---|---|---|
| 冲刺 | 18–20 周 | 18–20 h | 脱产/假期/课程压力小 | 2027-03 中旬 |
| **基准（默认）** | **26 周** | **12–14 h** | **在职或同时有课** | **2027-05-02** |
| 稳健 | 36 周 | 8–10 h | 工作忙/基础薄弱 | 2027-07 中旬 |

> 如果第 8 周时你发现实际用时比计划多 50% 以上，直接切稳健轨。**不要压缩 A1/A2 的质量去追日期。**

### 2.3 方向选择

- **默认：系统方向（训练 + 推理基础设施）**。理由：你的仓库兴趣集中在系统侧，且 A2 是最稀缺、最能拉开差距的能力。
- **偏研究**：A3/A4/A5 多花时间，capstone 换成小规模复现（scaling law、数据消融、GRPO vs DPO）。
- **偏应用**：A1/A5 认真做，A2 做到「能 profile 能解释」即可，capstone 换成评测/RAG/agent；但这会浪费你已有的系统积累，不作为默认。
- **最低可接受路径（时间不够时）**：A1 + A2 完整做；A3/A4/A5 选一个做到能讲清；capstone 缩到 1 周。宁可两份深，不要五份浅。

### 2.4 明确不做的事

- 不先系统学 CUDA/C++/编译原理再开始；这些在 A2 遇到瓶颈时再补。
- 不并行开两个 assignment；A1 没全绿不碰 A2。
- 不把 `MyTransformer`、`LLMs-from-scratch`、`nano-vllm` 的现成代码当答案；它们只能在**自己实现完之后**用来对照。
- 不让 AI 写作业代码、补 TODO、给伪代码；不装 AI autocomplete。
- 不追 leaderboard 名次；第一次只求「完整跑通 + 数字可信」。
- 不在 16 GB Mac 上训练超过 100M 参数；不做全量 Common Crawl。
- 不把「写文档 / 列 TODO / clone 仓库」当学习成果；每周必须有测试或数字。

---

## 3. 总路线图

| 阶段 | 周次 | 日期 | 主题 | 关键交付物 | GPU 预算（单卡当量） |
|---|---|---|---|---|---|
| P0 准备 | W1（10-29 起） | 10-29 → 11-08 | 环境、基线、读 A1 handout、Modal 冒烟 | 测试失败清单、预算上限、实验日志模板 | 0.5 h |
| P1 A1 | W2–W7 | 11-09 → 12-20 | BPE + Llama 风格 LM + 训练 | 测试全绿、TinyStories 训练 + 实验、report（leaderboard 可选） | 0–4 h（本地优先） |
| P2 A2 | W8–W14 | 12-21 → 02-07 | Profiling + FlashAttention-2 + DDP/FSDP | Triton kernel、2 卡 scaling 数据、report | 8–14 h |
| P3 A3 | W15–W17 | 02-08 → 02-28 | Scaling laws | loss-vs-FLOPs 曲线、拟合/外推、report | 10–20 h |
| P4 A4 | W18–W20 | 03-01 → 03-21 | 数据管线 | 过滤/去重管线、下游对比、report | 2–4 h + 8×1 h |
| P5 A5 | W21–W23 | 03-22 → 04-11 | SFT + GRPO | GSM8K 提升、稳定性分析、report | 4–8 h |
| P6 Capstone | W24–W26 | 04-12 → 05-02 | 系统小项目 | 可复现 repo + benchmark + writeup | 3–6 h |

> 每个阶段最后 20% 时间是 buffer；W14（春节周）和 W8（圣诞周）默认降速，只做整理/补测试。

### 讲座与作业的对应关系

| 周 | 讲座 | 作业 |
|---|---|---|
| W1 | L1 Overview/tokenization；L2 PyTorch/einops/资源核算 | 读 A1 handout |
| W2–W3 | L1 复习 | A1 tokenizer |
| W4–W5 | L3 Architectures/hyperparameters | A1 model |
| W6–W7 | L2 复习；L4 Attention alternatives/MoE（选看） | A1 training/leaderboard |
| W8 | L5 GPUs/TPUs | A2 profiling |
| W9–W11 | L6 Kernels/Triton | A2 FlashAttention-2 |
| W12–W14 | L7–L8 Parallelism；L10 Inference | A2 DDP/FSDP |
| W15–W17 | L9、L11 Scaling laws；L12 Evaluation | A3 |
| W18–W20 | L12–L14 Data | A4 |
| W21–W23 | L15–L16 Post-training/RLVR；L17 Alignment | A5 |
| W24–W26 | L18–L19 Guest（选看） | Capstone |

---

## 4. 分阶段详细计划

### P0（W1，10-29 → 11-08）：只准备，不写模型

**目标**：把「怎样算学会」定义清楚，把工具链和预算跑通。

任务：

1. **A1 基线**：在 `/Users/yunni/Projects/assignment1-basics` 建 `self-study` 分支，`uv sync`，`uv run pytest -q`，把失败清单和数量记下来（预期大量 `NotImplementedError`）；确认 `data/`、`weights/` 在 `.gitignore`；记录 `git rev-parse HEAD`、Python/torch 版本，提交一次 baseline commit。
2. **读 handout**：只读 A1 的 tokenization 和 model 两节，写 10 行摘要 + 5 个问题；不要读实现提示。
3. **云 GPU 冒烟**：注册 Modal，设 monthly spend limit（建议先 $50），跑一个最小 GPU 任务确认 `torch.cuda.is_available()`、显卡型号、计费；记录启动/停止命令。
4. **实验日志**：建 `notes/`，第一页写清：本周目标、命令、结果、意外、下周动作。以后每次 GPU run 都必须有记录。
5. **日历**：把 12–14 h/周拆成固定块（例如工作日 2×1.5 h + 周末 5 h），写进日历；GPU 长任务放在周末。

验收：

- [ ] 能说清 A1 的 4 个模块和测试入口
- [ ] 有一次成功的 GPU 冒烟记录（型号、耗时、费用）
- [ ] Modal 预算上限已设置
- [ ] baseline 测试失败数量已记录

**不要做**：不要在这一周开始写 BPE 或模型；不要同时读 A2。

---

### P1（W2–W7，11-09 → 12-20）：Assignment 1 — Basics

> 逐步到每天的计划、资料路径和验收清单见 `CS336_A1_DETAILED_PLAN.md`；本节保留阶段目标。

**目标**：BPE 分词器 + Llama 风格 Transformer + 训练工具链，全部通过官方测试，并完成一次有数字的训练。

#### 周安排

| 周 | 重点 | 本周必须交付 |
|---|---|---|
| W2 | BPE 训练：字节级、预分词、特殊 token、tie-break | `test_train_bpe` + 速度测试；TinyStories 10K tokenizer |
| W3 | Tokenizer 类 + tokenizer 实验；Linear/Embedding/RMSNorm | tokenizer 全绿；4 个实验答案；基础模块测试绿 |
| W4 | RoPE、softmax、attention、causal MHA、SwiGLU、block、完整 LM、资源核算 | `test_model.py` 全绿；accounting 答案 |
| W5 | 训练工具：CE/AdamW/schedule/clip/data/checkpoint/loop；TinyStories smoke | 工具测试全绿；训练脚本可跑 |
| W6 | TinyStories 训练 + LR/batch 实验 + 生成 + 4 个消融 | val loss 达标/接近；曲线与生成文本 |
| W7 | OWT（可选 leaderboard）+ writeup + 全量测试；buffer | `writeup.pdf`；提交包；可选 leaderboard |

#### 验收标准

- [ ] `uv run pytest` 全绿（macOS 会 skip 两个 Linux-only 内存测试，记录 skip 原因）
- [ ] BPE 速度测试 <1.5 s；`test_train_bpe` 的 merges 与参考完全一致
- [ ] tokenizer 在 TinyStories/OWT 样本上 round-trip，且与 `tiktoken` 的 GPT-2 编码一致
- [ ] 全模型测试通过，包括 RoPE + 截断输入
- [ ] 一次 TinyStories 训练（低资源 40M token 或云上完整配置）：loss 曲线、吞吐、显存、生成样本；leaderboard 可选
- [ ] 能脱稿解释：BPE 的 tie-breaking、RoPE 的相对位置性质、RMSNorm vs LayerNorm、AdamW vs Adam、Pre-LN 为什么稳
- [ ] 60 分钟内从空白文件写出 MHA + RoPE（自测，不查代码）

#### 常见坑

- BPE 的 merge 顺序/并列时字典序、特殊 token 不能被切开、按字节而不是按字符处理 Unicode、预分词正则。
- RoPE 的位置索引、截断输入、dtype；mask 的形状和 `-inf` 的位置；softmax 的 dim。
- AdamW 的 weight decay 与梯度更新的顺序（2026 handout 明确改过）；cosine schedule 的 off-by-one；grad clip 用全局范数。
- `get_batch` 的 dtype（输入应是 `int64`）、checkpoint 的 key 对齐、memmap 的路径。
- macOS 跳过内存测试 ≠ 内存效率合格；要在 Linux（Modal）上补跑。
- 测试没全绿之前不要调超参；调参不是实现。

#### 交付物

- `assignment1-basics` 私有仓库，按周 commit；`notes/a1.md`；`reports/a1.md`（配置、loss、吞吐、显存、失败尝试）；每日安排见 `CS336_A1_DETAILED_PLAN.md`。

#### 算力

- 本地：BPE、单测、TinyStories 40M token 训练；约 20–40 h，$0。
- 云端（可选）：OWT 32K BPE（CPU 高内存）、1 次 OWT 训练、leaderboard；B200 0–4 h，现金约 $0–25。官方完整实验上限 19 B200 h（约 $120），第一次不必用完。

---

### P2（W8–W14，12-21 → 02-07）：Assignment 2 — Systems

**目标**：从「会写模型」升级到「会测、会优化、会分布式」。这是全计划最硬的一段。

#### 周安排

| 周 | 重点 | 本周必须交付 |
|---|---|---|
| W8 | L5；搭 benchmark/profiling 框架：warmup、计时、显存、tokens/s；在 GPU 上复现 A1 模型基线 | 可复现的 benchmark 脚本 + 第一张表 |
| W9 | FlashAttention-2 的 PyTorch 版：online softmax、分块、causal、前向 | 前向与参考实现数值一致 |
| W10 | PyTorch 版反向；Triton 前向 | 反向梯度一致；Triton 前向正确 |
| W11 | Triton 反向 + 调优；与 `torch.nn.functional.scaled_dot_product_attention` 对比 | 正确性 + 速度/显存对比表 |
| W12 | DDP：参数广播、梯度 all-reduce、bucket、异步重叠、`no_sync` | 2 卡 DDP 正确性；与单卡 loss 对齐 |
| W13 | DDP benchmark；FSDP：参数分片、all-gather、reduce-scatter、混合精度 | FSDP 正确性 + 显存下降数据 |
| W14 | sharded optimizer；汇总 benchmark；写 report；春节 buffer | 完整 report（正确性/速度/显存/扩展效率） |

#### 验收标准

- [ ] FlashAttention-2 PyTorch 版前向/反向与参考实现 `allclose`（记录容差）
- [ ] Triton 版前向/反向通过同一组测试，且相对 baseline 有可解释的加速
- [ ] DDP：相同 seed/数据下，2 卡与单卡 loss 曲线一致；梯度同步正确
- [ ] FSDP：能 all-gather 出完整参数，显存显著低于 DDP；sharded optimizer 状态正确
- [ ] 至少 2 张 NVIDIA GPU 上跑通 `torchrun` 多进程
- [ ] report 里有：时间、显存、tokens/s、扩展效率、瓶颈分析
- [ ] 能解释：arithmetic intensity、online softmax 为什么省显存、all-reduce vs reduce-scatter 的通信量、FSDP 的 shard 数学

#### 常见坑

- Mac 上写 Triton 一定失败；先在云端跑通最小 kernel，再写完整的。
- Triton 版本、`tl.dot` 的 shape/dtype、mask 的边界、causal 的对角线。
- 多卡测试用 `torchrun`，注意 `LOCAL_RANK`、NCCL 初始化、每张卡的数据切分。
- DDP 的 bucket 大小、`no_sync` 与 gradient accumulation 的配合、buffer（如 RoPE cache）的同步。
- FSDP 的 wrap policy、mixed precision 的 master weights、checkpoint 的 state dict 格式。
- benchmark 必须 warmup、固定 clocks/功率、重复取中位数；否则数字不可信。
- A2 是本计划最可能超期的阶段：**超期就从 A3/A4 借 1–2 周，不砍 Triton/DDP 核心。**

#### 交付物

- `cs336_systems/` 实现 + 测试；`reports/a2.md`（含 4 张表：FA2 正确性、FA2 性能、DDP scaling、FSDP 显存）。

#### 算力

- 1×B200 做 FA2：约 4–6 h；2×B200 做 DDP/FSDP：约 4–8 h；合计 8–14 单卡当量（约 $50–88）。
- 省钱：用 A100/H100 替代 B200，正确性/扩展趋势一致；最终数字注明硬件。

---

### P3（W15–W17，02-08 → 02-28）：Assignment 3 — Scaling

**目标**：理解「模型变大时 loss 怎么变」，会做 compute-optimal 预测，而不是只会跑单点训练。

#### 周安排

| 周 | 重点 | 本周必须交付 |
|---|---|---|
| W15 | L9/L11；读 handout；搞清 compute 核算（6ND）和可用训练路径 | 实验设计：模型规模、token 预算、seed、指标 |
| W16 | 跑 sweep（4–6 个规模 × 2 seeds；算力不够就减点） | 原始结果表（loss、FLOPs、tokens、超参） |
| W17 | 拟合 power law、外推、误差分析；做一个消融（宽度 vs 深度 / LR / batch） | 曲线 + 拟合参数 + 预测 vs 实测 + report |

#### 验收标准

- [ ] 至少 4 个模型规模、2 个 seed；记录每个 run 的 FLOPs 与最终 loss
- [ ] 在对数空间拟合，给出指数和系数；报告拟合残差
- [ ] 用拟合预测一个没跑过的规模，并实测对比（误差要解释）
- [ ] 一个消融实验：例如固定参数量下深度 vs 宽度，或 LR 随规模的变化
- [ ] 能解释：6ND 的来历、不可约 loss、compute-optimal 分配、为什么小规模结论不能直接外推

#### 常见坑

- 官方 hosted API 只对 8 位学号的学生开放；如果你有课程 API key，直接用，A3 可省约 1 周；没有就走 non-student 路径，自己跑训练，成本要提前算。
- 如果服务端配置成本太高：用 A1 的模型在 TinyStories 上做 4 个规模的缩小复现，FLOPs 核算、log 拟合、外推的方法完全一致；在 report 里注明规模不同，不要假装是官方 API 的结果。
- 不同规模要用匹配的 LR/batch/token 预算，否则拟合的是「调参差异」不是 scaling。
- eval 噪声可能大于规模间差异；固定 eval set、多种子、报告置信区间。
- 不要用 A1 的最终模型直接当 scaling 数据点；训练 token 数要可控。

#### 交付物

- `reports/a3.md`：数据表、拟合图、外推验证、消融、成本记录。

#### 算力

- 10–20 个 B200 GPU 小时（单卡当量，约 $63–125）；压缩版：4 个规模 × 1 seed、更小模型，约 $30–50。

---

### P4（W18–W20，03-01 → 03-21）：Assignment 4 — Data

**目标**：把「数据质量」从口号变成可测量的管线。

#### 周安排

| 周 | 重点 | 本周必须交付 |
|---|---|---|
| W18 | HTML 抽取、语言识别、PII（邮箱/电话/IP）掩码、NSFW/毒性/质量分类、Gopher 过滤 | 各模块单测通过；在 10–50 个 WET 文件上跑通 |
| W19 | 精确行去重 + MinHash/LSH 去重；统计保留率、重复率 | 去重前后统计；参数（ngram/bands/hashes）记录 |
| W20 | 用过滤后的数据训练 tokenizer + LM，和 baseline 对比；写 report | 下游 loss/perplexity 对比；失败也如实写 |

#### 验收标准

- [ ] 所有 adapter 测试通过
- [ ] 在缩减版 WET（例如 100–300 个文件）上端到端跑通，记录吞吐和保留率
- [ ] MinHash 的 Jaccard 阈值、band 数、hash 数与理论匹配；给出假阳性/假阴性讨论
- [ ] 过滤/去重后的数据训练出的模型，在固定 held-out 集上不差于 baseline（或解释为什么）
- [ ] 能解释：Gopher 规则、质量分类器的训练数据从哪来、去重为什么能省 compute、数据泄漏怎么防

#### 常见坑

- 非学生路径必须先实现 `is_english`，否则全量 WET 下载会被挡住；先用小 `n_files` 验证。
- 全量 WET 默认约 2.5k 文件，磁盘和时间都吃不消；先改小 `n_files`。
- HTML 抽取库的版本差异、编码错误、正则回溯爆炸。
- MinHash 的 band 设计算错会导致「看起来去了很多，其实没去掉」。
- 过滤器和 tokenizer 的训练集要分开；否则下游评估被污染。
- 在 16 GB Mac 上处理大文件要流式；不要 `read()` 整个文件。

#### 交付物

- `cs336_data/` 实现 + 测试；`reports/a4.md`（管线图、保留率、下游数字、成本）。

#### 算力

- CPU：20–40 h（本地或 Modal CPU）；GPU：2–4 单卡小时做小规模训练；若按官方 8×B200 跑最终训练，1 h ≈ $50，可缩到 2 卡。

---

### P5（W21–W23，03-22 → 04-11）：Assignment 5 — Alignment

**目标**：把「后训练」跑通：SFT → GRPO，理解每个 loss 项在做什么。

#### 周安排

| 周 | 重点 | 本周必须交付 |
|---|---|---|
| W21 | SFT：prompt/response 拼接、response mask、packing、loss；小模型训练 + GSM8K 评测 | SFT baseline 准确率；训练稳定 |
| W22 | GRPO：rollout、reward（格式+答案）、group 归一化 advantage、importance ratio、clip、loss 聚合 | `test_grpo.py` 全绿；单步训练能跑 |
| W23 | 完整 GRPO 训练；SFT vs GRPO 对比；可选 DPO；写 report | 指标提升 + 稳定性曲线（entropy、clip fraction、长度） |

#### 验收标准

- [ ] `uv run pytest tests/test_grpo.py` 全绿
- [ ] SFT 在 GSM8K 上有一个可信的 baseline（不是 0，也不是随机）
- [ ] GRPO 相对 SFT 有提升（哪怕几个点），并报告方差/种子
- [ ] 训练稳定：reward 不崩、entropy 不爆炸、clip fraction 可解释
- [ ] 能解释：policy gradient、advantage、importance sampling、PPO clip、GRPO 的去 critic 设计、reward hacking
- [ ] 可选：DPO loss 与偏好对构造

#### 常见坑

- `flash-attn`/vLLM 的安装对 CUDA/编译器版本敏感；先按 README 的两步装。
- prompt/response 的 tokenizer 边界、response mask 对齐、padding 的 loss 屏蔽。
- reward 只看答案会诱导格式 hack；格式 reward 权重过高会抑制推理。
- GRPO 的 group size、normalization、长度偏置；不要只报最好 seed。
- 显存：1B–1.5B 级模型全量微调需要 80 GB 级 GPU；预算不够就用更小模型/LoRA，并在报告里说明。

#### 交付物

- `cs336_alignment/` 实现 + 测试；`reports/a5.md`（SFT/GRPO 曲线、指标、失败模式）。

#### 算力

- 4–8 个 B200 GPU 小时（单卡当量，约 $25–50）；0.5B 模型 + LoRA 可压到 $10–20。

---

### P6（W24–W26，04-12 → 05-02）：Capstone

**目标**：把五份作业里的零件组装成一个「别人能复现」的系统，而不是第六份作业。

默认项目（系统方向）：

> **给 A1 的模型加一个真正的推理引擎**：continuous batching + paged KV cache + prefix cache + 简单 scheduler；对比 naive decode 的吞吐/延迟/显存，画出 batch size–throughput 曲线。

为什么选它：你已经量过 KV cache，nano-vllm 只读过没改过；这个项目能把「读懂」变成「实现 + 数字」。

其他选项：

| 方向 | 项目 | 关键数字 |
|---|---|---|
| Kernel | fused RMSNorm+SwiGLU+RoPE，Triton autotune | 融合前后延迟/带宽 |
| 研究 | 在 TinyStories 上复现 Chinchilla 式 scaling | 预测误差、compute-optimal 点 |
| 数据 | 构建 1B–10B token 的过滤子集，训练 50M 模型 | 下游 loss/成本 |
| 对齐 | DPO vs GRPO 小规模对比 | GSM8K/生成质量/稳定性 |
| 可解释 | induction head 形成过程（训练 checkpoint 序列） | 指标随训练步的变化 |

验收：

- [ ] 全新环境按 README 能在 30 分钟内复现主要数字
- [ ] 有 baseline 和至少 3 个配置的对比（batch size / cache / scheduler）
- [ ] 1500 字 writeup + 10 分钟自讲（录音或给朋友讲）
- [ ] 代码私有或注明「学习用途，非作业提交」

---

## 5. 每周怎么学（比计划表更重要）

### 5.1 时间块模板（12–14 h/周）

| 时间 | 内容 | 产出 |
|---|---|---|
| 周一 30 min | 定本周 1 个主目标 + 1 个验收 | 写进 `notes/` |
| 周二/周四 各 2 h | 实现/读 handout，手机静音 | commit 或测试结果 |
| 周六 4–5 h | GPU run / 长实验 / 卡点攻坚 | 实验记录 + 数字 |
| 周日 1.5 h | 复盘：这周学到什么、哪里卡住、下周砍什么 | 周报 10 行 |
| 每天 15 min | 回看昨天的代码/笔记 | 修正注释 |

### 5.2 实验日志格式（每次 GPU run 必填）

```
日期：
目标（一句话）：
命令/commit：
配置（模型、数据、GPU、dtype）：
结果（loss/时间/显存/吞吐）：
意外：
下一步：
```

没有这 7 行的 run 视为没做。

### 5.3 调试协议

1. 先打印 shape/dtype/device/范围，再怀疑算法。
2. 用最小输入（B=1、T=3、1 层、1 头）复现。
3. 和参考实现比单个中间张量，不只看最终 loss。
4. 固定 seed；一次只改一个变量。
5. 卡住 45 分钟就写下「我已知/我假设/我下一步验证」，再求助。
6. 性能问题先 profile 再改；没有 profile 的优化视为猜测。

### 5.4 AI 使用协议（和 CS336 的规则一致）

**允许**：

- 让 AI 解释报错、概念、论文段落；
- 让 AI 对你的代码做 review，指出边界/不变量/测试缺口；
- 让 AI 出测试题、检查你的理解；
- 让 AI 帮你规划时间、整理笔记。

**禁止**：

- 让 AI 写作业代码、补 TODO、给伪代码；
- 让 AI 直接给某道题的解法；
- 开 Cursor Tab/Copilot 自动补全；
- 把 AI 生成的代码贴进提交仓库。

`assignment1-basics/AGENTS.md` 已经写明了这套规则；把该文件保留在仓库根目录，任何 AI 工具先读它。**本计划文件也不含任何实现提示；后续你让我执行时，我只会按这份协议给提示和 review，不代写。**

### 5.5 讲座怎么用

- 作业是主线，讲座是工具：只看到能推动当前作业的部分，不按 1→19 顺序硬刷。
- 1.25–1.5× 播放，遇到公式/图暂停，自己推一遍再继续。
- 每讲只记 5 行：解决什么问题、关键公式、一个数字、一个疑问、对应代码位置。
- 看完立刻在代码里找一个对应点；找不到就说明还没懂，别急着看下一讲。

### 5.6 证据优先

每周日问自己：

- 这周有没有一个**测试从红变绿**？
- 有没有一个**数字**（loss、tokens/s、显存、准确率）？
- 有没有一个**commit**？
- 如果没有，下周先补证据，不学新内容。

---

## 6. 算力与预算

### 6.1 为什么必须上云

- A2 的 Triton/FlashAttention-2 需要 NVIDIA GPU；macOS 没有 CUDA，Triton 不支持 MPS。
- A2 的 DDP/FSDP 测试至少需要 2 张 GPU。
- A1 的 leaderboard 45 分钟训练上限、A4 的最终训练、A5 的 rollout，在 16 GB Mac 上都不现实。

### 6.2 推荐方案：Modal

- 课程页给出的参考价（2026-03-28）：Modal B200 **$6.25/h**，每月 **$30 免费额度**（以当前政策为准）；RunPod $4.99/h、Lambda $6.69/h、Nebius $5.50/h（可抢占 $3.05/h）。下单前以官网实时价格为准。
- 建议：先设 **$50/月** 上限；跑完一个阶段就把 artifact 下载到本地，停掉 app。
- 调试在本地 CPU；只有「测试全绿 + 需要真实 GPU 数字」时才开 GPU。
- 多卡实验用 2 卡，不用 8 卡；只有 A4 的最终训练可考虑 8 卡 1 小时。
- 零预算备选：Kaggle 2×T4 / Colab 可跑 A1 smoke 和部分 A5；A2 的 Triton/DDP 不建议在免费环境硬撑，环境问题会吃掉学习时间。

### 6.3 预算表（估算，非报价）

| 阶段 | 单卡当量小时 | 费用 @ $6.25/h | 省钱做法 |
|---|---|---|---|
| P0 | 0.5 | ~$3 | 只做 `nvidia-smi` 冒烟 |
| A1 | 0–4 | $0–25 | 本地跑 BPE/训练；云只做 OWT/leaderboard |
| A2 | 8–14 | $50–88 | 用 A100/H100；2 卡而非 8 卡 |
| A3 | 10–20 | $63–125 | 4 个规模 × 1 seed；小模型 |
| A4 | 2–4 + 8×1（=10–12） | $63–75 | 缩到 100–300 个 WET 文件；2 卡训练 |
| A5 | 4–8 | $25–50 | 0.5B + LoRA |
| Capstone | 3–6 | $19–38 | 单卡；小 batch |
| **合计** | **36–65** | **$222–403** | 用满 $30/月额度后，现金约 **$42–223** |

> A4 的 8×B200 只按 1 小时估；官方全量若跑 2–3 小时，另加 $50–100。A3/A4 的「压缩版」是默认推荐，先保证方法跑通，再决定要不要加规模。
> A1 行按最小云训练估；官方完整实验上限 19 B200 h（约 $120），详见 `CS336_A1_DETAILED_PLAN.md`。

**硬规则**：单次实验超过 $25 先停下来算（A4 的 8×B200 最终训练提前单列）；单周超过 $50 就降规模；总现金超过 $300 停止，改稳健轨。

### 6.4 本地能做什么

- BPE、tokenizer、单测、tiny 训练（<10M 参数）、数据预处理、写报告：本地 CPU。
- MPS 可用于快速 smoke test，但不要用它报性能数字；也不要假设 MPS 和 CUDA 的数值/算子完全一致。
- 16 GB 内存：不要同时开浏览器 50 个标签 + 训练；大文件流式处理。

### 6.5 磁盘管理

- TinyStories + OWT 原始数据约 14 GB；预处理后的 `.bin` 可能再翻倍。
- A4 的 WET 子集先在云端处理，只把统计和最终 tokenized 数据拉回本地；原始压缩包用完即删。
- 保持至少 30 GB 空闲；`df -h` 每周看一次。

---

## 7. 每阶段自测题（能答出来才算过）

### A1

1. BPE 在多个 pair 频次相同时怎么选？为什么这个顺序会影响最终 vocab？
2. 特殊 token 为什么不能参与 merge？编码时怎么保证不被切开？
3. RoPE 的「相对位置」性质具体指什么？它真的能直接外推到训练长度之外吗？
4. RMSNorm 和 LayerNorm 的差别？为什么 LLM 常用 RMSNorm？
5. AdamW 的 decoupled weight decay 和 Adam + L2 的区别？为什么 2026 handout 强调更新顺序？
6. Pre-LN 为什么比 Post-LN 好训？残差路径上的 norm 会不会影响梯度？

### A2

1. 注意力层的 arithmetic intensity 怎么算？为什么 decode 阶段是 memory-bound？
2. FlashAttention 的 online softmax 解决了什么问题？它省的是显存还是 FLOPs？
3. causal mask 在分块 kernel 里怎么处理对角线块？
4. DDP 的 all-reduce 和 FSDP 的 reduce-scatter 各通信多少字节？为什么 FSDP 更省显存？
5. DDP 的 bucket 为什么要 overlap？`no_sync` 在 gradient accumulation 里起什么作用？
6. FSDP 的 shard 之后，forward 前为什么要 all-gather？backward 后为什么可以丢掉分片？

### A3

1. 6ND 是怎么来的？attention 的 FLOPs 为什么在长序列下不可忽略？
2. 为什么 scaling law 要在 log 空间拟合？不可约 loss 怎么估计？
3. compute-optimal 的模型大小和 token 数为什么近似等比例增长？
4. 小模型上拟合的指数，为什么不能直接外推到 100B？
5. 你的 sweep 里，eval 噪声和规模效应哪个更大？怎么验证？

### A4

1. Gopher 质量规则有哪些？哪些规则对代码/中文会误杀？
2. MinHash 的 collision probability 公式是什么？band 数怎么影响 recall/precision？
3. 精确行去重和 MinHash 去重各自删掉了什么？为什么两个都要？
4. 语言识别模型在短文本/代码上的失败模式？
5. 过滤后训练集变小，为什么下游还可能更好？什么情况下会变差？

### A5

1. SFT 的 response mask 为什么只对 response token 算 loss？
2. policy gradient 里 baseline 为什么能降方差而不引入偏差？
3. PPO/GRPO 的 importance ratio 和 clip 分别在防什么？
4. GRPO 去掉 critic 后，group 内归一化起了什么作用？
5. reward hacking 的典型表现？格式 reward 权重过高会怎样？
6. DPO 的 loss 在做什么？它和 RLHF 的关系？

---

## 8. 风险与应对

| 风险 | 触发信号 | 应对 |
|---|---|---|
| 没有 CUDA，进度被云卡住 | A2 一直没跑过 GPU | P0 就开通 Modal；先跑最小 kernel；预算按月分配 |
| A2 超期 | W14 时 FA2/DDP 还没全绿 | 从 A3/A4 借 1–2 周；不砍 Triton/DDP 核心；capstone 缩到 1 周 |
| BPE 卡住 | W3 结束速度/正确性没过 | 先正确后优化；用 profiler 找热点；不要提前看答案 |
| 多卡调试崩溃 | NCCL hang、loss 不一致 | 降到 2 卡、单机；先跑 1 步；开 `TORCH_DISTRIBUTED_DEBUG` |
| 成本超支 | 单周 >$50 | 停 GPU，改小模型/短 run；用免费额度月；下月再跑 |
| 数据太大/磁盘满 | A4 下载/处理爆盘 | 缩 `n_files` 到 100–300；云端处理；删原始包 |
| 文档/计划代替实现 | 这周只有笔记没有测试/数字 | 执行 §5.6「证据优先」；下周只做能验收的事 |
| AI 依赖 | 代码能跑但讲不清 | 关 autocomplete；做 blank-file challenge；AI 只做 review |
| 版本/环境错乱 | A1 的 `uv sync` 和 prep 的 `.venv` 混用 | 每个作业独立目录 + `uv`；记录 Python/torch 版本 |
| 动力下降 | 连续 2 周没有 demo | 每周日录 2 分钟 demo；把大目标拆成当天可完成的小测试 |

---

## 9. 资源清单

### 课程与作业

- 课程主页：<https://cs336.stanford.edu>
- 课程组织：<https://github.com/stanford-cs336>
- 讲座材料：<https://github.com/stanford-cs336/lectures>
- YouTube 播放列表：<https://www.youtube.com/watch?v=JuoVZkPBiKk&list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV>
- A1：<https://github.com/stanford-cs336/assignment1-basics>
- A2：<https://github.com/stanford-cs336/assignment2-systems>
- A3：<https://github.com/stanford-cs336/assignment3-scaling>
- A4：<https://github.com/stanford-cs336/assignment4-data>
- A5：<https://github.com/stanford-cs336/assignment5-alignment>

### 工具

- uv：<https://docs.astral.sh/uv/>
- Modal：<https://modal.com/docs>
- PyTorch profiler：<https://pytorch.org/docs/stable/profiler.html>
- Triton tutorials：<https://triton-lang.org/main/getting-started/tutorials/>
- PyTorch FSDP：<https://pytorch.org/docs/stable/fsdp.html>
- How to Scale Your Model：<https://jax-ml.github.io/scaling-book/>

### 分阶段论文（每阶段 2–3 篇，先读摘要/方法，不啃证明）

- **A1**：BPE (Sennrich 2016, arXiv:1508.07909)；RoPE (Su 2021, arXiv:2104.09864)；RMSNorm (Zhang & Sennrich 2019, arXiv:1910.07467)；SwiGLU (Shazeer 2020, arXiv:2002.05202)；Pre-LN (Xiong 2020, arXiv:2002.04745)；AdamW (Loshchilov 2017, arXiv:1711.05101)
- **A2**：FlashAttention (Dao 2022, arXiv:2205.14135)；FlashAttention-2 (Dao 2023, arXiv:2307.08691)；ZeRO (Rajbhandari 2019, arXiv:1910.02054)；Megatron-LM (Shoeybi 2019, arXiv:1909.08053)；GPipe (Huang 2018, arXiv:1811.06965)
- **A3**：Kaplan 2020 (arXiv:2001.08361)；Chinchilla (Hoffmann 2022, arXiv:2203.15556)
- **A4**：C4 (Raffel 2019, arXiv:1910.10683)；Gopher (Rae 2021, arXiv:2112.11446)；Dedup (Lee 2021, arXiv:2107.06499)；FineWeb (Penedo 2024, arXiv:2406.17557)；DCLM (Li 2024, arXiv:2406.11794)
- **A5**：InstructGPT (Ouyang 2022, arXiv:2203.02155)；PPO (Schulman 2017, arXiv:1707.06347)；DPO (Rafailov 2023, arXiv:2305.18290)；DeepSeekMath/GRPO (Shao 2024, arXiv:2402.03300)；DeepSeek-R1 (2025, arXiv:2501.12948)；Tulu 3 (Lambert 2024, arXiv:2411.15124)

### 对照代码（**自己实现完之后**再看）

- nanoGPT：<https://github.com/karpathy/nanoGPT>
- FlashAttention：<https://github.com/Dao-AILab/flash-attention>
- nano-vllm：<https://github.com/GeeeekExplorer/nano-vllm>
- vLLM：<https://github.com/vllm-project/vllm>

> 注意：CS336 的规矩是「不要看现成实现」。自学者可以放宽到「实现完成、测试绿了之后，用它们做 code review 对照」，但不要边看边写。

---

## 10. 接下来 72 小时（10-29 → 11-01）

只做这 4 件事，每件 ≤2 小时：

1. **冻结基线**：在 `assignment1-basics` 建 `self-study` 分支，`uv sync`，`uv run pytest -q`，把失败数量/首个报错记到 `notes/a1-baseline.md`；commit 一次。
2. **读 handout**：只读 A1 的 tokenization + model 两节，写 10 行摘要 + 5 个问题；不读 A2。
3. **开通云 GPU**：注册 Modal，设 $50/月上限，跑一次 `nvidia-smi`/`torch.cuda.is_available()` 冒烟，记录显卡型号、启动耗时、预估费用。
4. **排日历**：把 12–14 h/周拆成固定时间块，写进日历；把本文件的 §3 表格复制到 `notes/plan.md`，每完成一行打勾。

> 72 小时后你应该有：一个干净的 baseline、一个能用的 GPU 通道、一份可执行的时间表。**不要提前写 BPE。**

---

## 11. CS336 之后往哪走

26 周结束后，按兴趣选一条主线，不要三条同时开：

1. **系统/推理**（最匹配你现在的仓库）：CUDA/Triton 深入、paged attention、continuous batching、speculative decoding、量化、KV cache 压缩；项目：把 capstone 的推理引擎做到能服务一个小模型，压测 QPS/延迟。
2. **训练/数据**：分布式并行（TP/PP/EP）、混合精度、激活检查点、数据配比、合成数据；项目：在固定算力下训练一个 100M–300M 模型，复现一个 scaling 结论。
3. **研究/对齐**：scaling laws、数据消融、RLVR、可解释性；项目：复现一篇 2024–2026 的小论文，或做 induction head 的形成过程。
4. **应用**：评测、RAG、agent；CS336 的系统能力会变成你的差异化，但不要把主线换成「调 API」。

**判断标准**：做完 A2 后，如果你享受「profile → 改 kernel → 数字变好」的过程，走系统；如果你更享受「设计实验 → 拟合曲线 → 解释现象」，走研究。不要用「哪个更好找工作」来选，两个方向都需要能落地的人。

---

## 附录 A：A1 组件清单（按测试对齐，不含实现）

- **Tokenizer**：`train_bpe`（字节级、特殊 token、速度）、`Tokenizer`（vocab/merges 加载、encode、decode、`encode_iterable`、特殊 token 处理）
- **Model**：Linear、Embedding、RMSNorm、SwiGLU、RoPE、softmax、scaled dot-product attention、MHA（含 RoPE）、Pre-LN block、完整 LM、截断输入
- **Training**：cross entropy、AdamW、cosine LR schedule、gradient clipping、`get_batch`、checkpoint save/load
- **Run**：训练循环、验证、生成、leaderboard 45 分钟限制、report

## 附录 B：26 周检查表

| 周 | 阶段 | 打勾条件 |
|---|---|---|
| W1 | P0 | baseline + GPU 冒烟 + 日历 |
| W2 | A1 | `test_train_bpe` 绿 |
| W3 | A1 | tokenizer 全绿 + 10k vocab 训练完 |
| W4 | A1 | 基础组件测试绿 |
| W5 | A1 | 全模型测试绿 |
| W6 | A1 | 训练工具测试绿 + CPU smoke |
| W7 | A1 | leaderboard 45 分钟训练 + report |
| W8 | A2 | benchmark 脚本 + 基线表 |
| W9 | A2 | FA2 PyTorch 前向正确 |
| W10 | A2 | FA2 反向正确 + Triton 前向 |
| W11 | A2 | Triton 反向 + 性能表 |
| W12 | A2 | 2 卡 DDP 正确 |
| W13 | A2 | FSDP 正确 + 显存数据 |
| W14 | A2 | sharded optimizer + report |
| W15 | A3 | 实验设计 |
| W16 | A3 | sweep 原始数据 |
| W17 | A3 | 拟合 + 外推 + report |
| W18 | A4 | 过滤模块测试绿 |
| W19 | A4 | 去重统计 |
| W20 | A4 | 下游对比 + report |
| W21 | A5 | SFT baseline |
| W22 | A5 | GRPO 测试绿 |
| W23 | A5 | GRPO 结果 + report |
| W24 | Capstone | 设计 + baseline |
| W25 | Capstone | 核心实现 |
| W26 | Capstone | 复现 + writeup |

---

## 需要你确认的 3 个参数（确认后我再改排期）

1. **每周实际可投入小时数**：默认 12–14 h；如果 <8 h，切 36 周稳健轨。
2. **GPU 现金预算**：默认 $50–220；如果只能 $0–50，A3 改成「小规模复现」，A4 缩到 100 个文件，A5 用 0.5B + LoRA。
3. **方向偏好**：默认系统/推理；如果更想走研究，把 A3/A4/A5 各加 1 周，capstone 换成论文复现。

> 这份计划只定义「做什么、做到什么程度、怎么验收」，不替你写任何作业代码。下一步从 §10 的 4 件事开始。
