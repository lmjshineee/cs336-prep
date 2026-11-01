# CS336 课程学习 + Assignment 1 逐步方案

> 版本：2026-10-29  
> 起点：已完成 `cs336-prep-code`；本地已有 `Courses/cs336-notes` 的讲义、英文笔记、中文笔记和中文导读  
> 目标：6 周完成 Assignment 1（代码 + 实验 + writeup），同时建立一套能复用到 A2–A5 的课程学习法  
> 默认投入：12–14 h/周；本地 Apple M5 / 16 GB；A1 云预算 $0–25（可选加 leaderboard）  
> 原则：**官方 handout 和官方测试是唯一标准；先自己做，后看导读；不让 AI 写作业代码。**

---

## 0. 一页纸结论

1. **A1 不是「把模型写出来」就结束。** 按题目分值粗算，代码/单测约 55%，实验、写作和资源核算约 45%。很多人卡在「测试全绿但训练不出东西」，因为实验和写作没做。
2. **你的机器能完成 A1 的 80%。** TinyStories 的 BPE、模型、40M token 训练都可以在 M5 上做；只有 **OWT 的 32K BPE（需要 ~100 GB 内存）**、**OWT 训练** 和 **leaderboard（B200、45 分钟）** 需要云。
3. **课程不要从第 1 讲顺序刷到最后一讲。** A1 只依赖 L1（tokenization）、L2（PyTorch/einsum/资源核算）、L3（架构/超参）。L4 之后的内容在 A2–A5 再学。
4. **你已经有很好的资料，不要再重复造轮子。** `stanford-cs336-2026-notes-main/lecture-01..03.md` 是英文详细笔记；`course_notes/lecture_03_notes.md` 是中文架构笔记；官方 handout 是 spec。
5. **你手上那本《跟着 stanford 动手学大模型》含完整代码块（至少 Linear 等节是完整实现），不是纯练习册。** 它适合做「讲解版教材」，不适合边看边抄。计划里所有任务都要求「先空白实现 → 跑测试 → 再看导读」。
6. **6 周、每周一个 gate（Gate 0–6）。** 本周先做 Week 0 的 3 件事：跑 baseline 测试、读 L1 的 tokenization、写 unicode 两题。不要提前写 BPE。

---

## 1. 资料地图：你已经有什么，什么时候用

| 资料 | 位置 | 作用 | 使用时机 / 注意 |
|---|---|---|---|
| 官方作业仓库 | `/Users/yunni/Projects/assignment1-basics` | 代码骨架、`adapters.py`、测试、`make_submission.sh` | 全程；**不要改测试** |
| 官方 handout | 同目录 `cs336_assignment1_basics.pdf`（47 页） | 唯一 spec：接口、题目、评分、算力、低资源建议 | 每做一块前先读对应节 |
| 2026 英文详细笔记 | `Courses/cs336-notes/stanford-cs336-2026-notes-main/lecture-01.md` ~ `lecture-18.md` | 讲座内容的逐节整理；A1 主要看 01/02/03 | 讲座前预习、卡住时查 |
| 中文架构笔记 | `Courses/cs336-notes/course_notes/lecture_03/lecture_03_notes.md` | L3 的中文压缩版：pre-norm、RMSNorm、RoPE、SwiGLU、超参 | 学模型前/复习时看 |
| 官方课件 | `Courses/cs336-notes/course_notes/lecture_XX/` | 原始 `.py` 可执行讲义 / `.pdf` 课件 | 想看原始推导/动画时 |
| 中文导读 | `Courses/cs336-notes/跟着 stanford 动手学大模型/跟着 stanford 动手学大模型.md` | 按 problem 讲 A1，含代码块 | **只在做完题后对照**；不要边看边写 |
| Karpathy BPE 笔记 | `Courses/karpathy-gpt-roadmap/minbpe-comment/notes.md`、`tokenizer-comment/notes.md` | BPE 的另一种直觉解释 | 只在 BPE 卡住时补充；不替代 handout |
| Karpathy nanoGPT | `Courses/karpathy-gpt-roadmap/nanoGPT-comment/` | 训练循环/GPT 结构的参考 | **A1 完成之后**再看，避免架构被带偏 |
| 你的 prep 代码 | `/Users/yunni/Projects/temp/cs336-prep-code` | 已实现的 GPT-2 风格模型、KV cache、真实权重实验 | 当「直觉参考」，不要直接复制成 A1（A1 是 RoPE/RMSNorm/SwiGLU） |
| 课程主页/录像 | `https://cs336.stanford.edu`、YouTube playlist | 讲座视频、schedule、assignments | 按需；不必全看 |

### 使用规则（很重要）

1. **一个概念只找一个主 source。** A1 的主 source 是 handout；讲座笔记是解释；中文导读是第二解释。不要同时读四份。
2. **官方测试 > 任何笔记。** 笔记可能转写错、导读可能代码有 bug；测试失败时以 handout 接口和测试输入为准。
3. **导读的「三遍法」：**
   - 第一遍：只看它的标题/图/文字解释，**不看代码块**；
   - 第二遍：自己从空白实现、跑测试；
   - 第三遍：卡住超过 45 分钟，才看它的代码，看完关掉，**从空白重写**，再用测试验证。
4. **AI 使用边界（课程明文规定）：** 可以问概念、API 签名、报错解释；**不可以让 AI 实现任何 assignment 部分**，也不开 Cursor Tab/Copilot。你让我后续执行时，我只会做解释、review、测试建议，不代写。

---

## 2. 课程怎么学（适用于 L1–L17 + guest 讲座）

### 2.1 课程地图

| 阶段 | 讲座 | 作业 | 你本地对应的笔记 |
|---|---|---|---|
| A1 基础 | L1 tokenization；L2 PyTorch/einsum/资源核算；L3 架构/超参 | Assignment 1 | `lecture-01/02/03.md`；`lecture_03_notes.md` |
| A2 系统 | L4 注意力替代/MoE（选）；L5 GPU/TPU；L6 kernel/Triton；L7–L8 并行；L10 推理 | Assignment 2 | `lecture-04..08/10`（A1 后再看） |
| A3 规模 | L9/L11 scaling laws；L12 evaluation | Assignment 3 | `lecture-09/11/12` |
| A4 数据 | L12–L14 数据 | Assignment 4 | `lecture-12..14` |
| A5 后训练 | L15 SFT/RLHF；L16 RLVR；L17 对齐/多模态 | Assignment 5 | `lecture-15..17` |

**A1 只学 L1–L3。** L4 可以在 A1 结束后当拓展；L5 之后现在看会分散注意力。

### 2.2 单讲学习法：3-pass

**Before（20–30 分钟）**

- 看 handout 对应节 + 笔记的标题/小标题，写下 3 个问题：
  - 这一讲解决什么问题？
  - 哪个公式/形状我必须能推？
  - 它对应 A1 的哪个 problem？
- 不提前看代码。

**During（60–80 分钟）**

- 视频 1.25–1.5×；遇到公式、图、代码演示暂停，自己推一遍/在纸上写形状。
- 每讲至少看一个「演示」：lecture 1 的 BPE trace、lecture 2 的 einsum、lecture 3 的 RoPE 旋转，至少手推一个。

**After（20–30 分钟）**

写 5 行增量笔记（不是抄讲义）：

```
本讲解决：
关键公式/形状：
一个数字/例子：
和 A1 的对应：
我还想问：
```

你已经有一份很详细的 `lecture-01..18` 笔记，**不要重抄**；只写「我原来不懂、现在懂」的增量，以及和 A1 的映射。

### 2.3 每周节奏（12–14 h）

| 时间 | 内容 | 产出 |
|---|---|---|
| 周一 30 min | 定本周 gate + 读 handout 对应节标题 | 本周计划 |
| 周二/周四 各 2.5 h | 讲座笔记 + 实现/测试 | commit + 测试结果 |
| 周三 2 h | 只实现，不看新内容 | 代码 |
| 周六 4–5 h | 长训练/实验/卡点 | 曲线/数字 |
| 周日 1.5 h | 复盘 + 写实验日志 + 下周计划 | 周报 10 行 |

A1 期间讲座量要少：每周最多 1–2 讲，作业实现优先。

### 2.4 A1 前诊断（30 分钟，决定你跳过什么）

能脱稿回答下面 8 题，就跳过对应笔记；不能就回去读对应节：

1. 为什么 byte-level BPE 既避免 OOV 又比纯字节短？压缩比怎么定义？
2. BPE 训练时 pair 频次并列怎么处理？特殊 token 为什么不能参与 merge 统计？
3. 矩阵乘 `(m,n)@(n,p)` 的 FLOPs 是多少？Transformer 前向的 FLOPs 主要花在哪？
4. AdamW 比 SGD 多存什么？为什么权重衰减要 decoupled？
5. Pre-norm 和 post-norm 的公式差在哪？为什么 pre-norm 更稳？
6. RMSNorm 和 LayerNorm 的差别？为什么输入要 upcast 到 fp32？
7. RoPE 对 q/k 做什么？为什么 value 不做？head 维度为什么要当 batch 维？
8. causal mask 里 True 表示「能看」还是「不能看」？mask 加在 softmax 前还是后？

答不出的题，记录到 `notes/a1-gaps.md`，在对应周补。

---

## 3. Assignment 1 全景：你要交什么

### 3.1 四块实现

1. **BPE tokenizer**：训练（字节级、预分词、特殊 token、速度）、Tokenizer 类（encode/decode/`encode_iterable`/`from_files`）。
2. **Transformer LM**：Linear、Embedding、RMSNorm、SwiGLU、RoPE、softmax、scaled dot-product attention、causal MHA、pre-norm block、完整 LM。
3. **训练工具**：cross entropy、AdamW、cosine schedule + warmup、gradient clipping、data loader、checkpoint、training loop。
4. **生成与实验**：解码（temperature/top-p）、TinyStories 训练、LR/ batch size 实验、4 个消融、OWT、leaderboard（可选）。

### 3.2 交付物与分值（按 handout 题目粗算）

| 模块 | 主要内容 | 大约分值 | 交付 |
|---|---|---|---|
| Tokenizer | unicode、BPE 训练、Tokenizer、tokenizer 实验 | ~42 | 代码 + 测试 + 文字回答 |
| 模型 | 组件、block、LM、资源核算 | ~29 | 代码 + 测试 + 核算 |
| 训练 | loss/AdamW/schedule/clip/data/checkpoint/loop | ~15 | 代码 + 测试 + 脚本 |
| 生成 | decoding | ~3 | 代码 + 生成文本 |
| 实验 | log、LR、batch、4 个消融、OWT、leaderboard | ~20 | 曲线 + 实验日志 + 写作 |

> 分值相加只是帮你分配时间，不是官方总分；以 handout 为准。重点：**实验/写作接近一半**，不要只刷单测。

### 3.3 官方算力与低资源路径

- 官方给 A1 的实验预算约 **19 B200 小时**（LR 2 + batch 1 + 4 个消融各 0.5 + OWT 2 + leaderboard 10）。
- TinyStories 基准配置：vocab 10,000、context 256、d_model 512、d_ff 1344、RoPE theta 10000、4 层 16 头、约 17M 非嵌入参数、总 token 约 327,680,000；1×B200 约 20–30 分钟，目标 val loss ≤ 1.45。
- **低资源路径**：把总 token 降到 40M（例如 32×5000×256），目标 val loss ≤ 2.00。Handout 给的参考：M4 Max 36 GB 上 40.96M token，MPS 约 36 分钟、CPU 约 1 小时 22 分，val loss 1.80。你的 M5 16 GB 会更慢，预计 1.5–4 小时，取决于 batch 和编译；以实测为准。
- Leaderboard：OWT 数据、B200 上 ≤45 分钟、val loss 至少优于 5.0 的 baseline；10 B200 小时预算，**第一次做可以跳过**。

### 3.4 Definition of Done（A1 完成的标准）

- [ ] `uv run pytest` 全绿（记录 macOS skip 的两个 Linux-only 内存测试）
- [ ] 所有 written problems 有答案（unicode1/2、tokenizer experiments、transformer accounting、LR toy、AdamW accounting、实验分析）
- [ ] TinyStories 上训练出一个模型：低资源 val loss ≤2.0，或云上 ≤1.45
- [ ] 至少 256 token 的生成文本 + 流畅度分析
- [ ] LR sweep、batch size、4 个消融都有曲线和结论
- [ ] 实验日志（日期、配置、命令、结果、结论）完整
- [ ] `writeup.pdf` + `code.zip` 能按 `make_submission.sh` 生成
- [ ] （可选）OWT 训练 + leaderboard 提交
- [ ] 能脱稿解释：BPE、RoPE、RMSNorm、MHA、AdamW、cosine schedule、checkpoint 的作用与坑

---

## 4. 每个 problem 的学习闭环

每道题都按这 6 步走，不要跳：

1. **读 spec**：handout 对应节 + 测试文件的名字/输入形状。写下接口、输入输出、边界条件。
2. **纸上推导**：公式、张量形状、复杂度；用一个小例子手算（BPE、attention、RoPE 各一个）。
3. **空白实现**：新建自己的文件；不看导读代码、不看 AI 代码；先写能跑的正确版本，不优化。
4. **针对性测试**：只跑这一块的测试，例如 `uv run pytest -k test_linear`；读失败信息，不急着改测试。
5. **解释 + 记录**：用 5 行话说明「为什么这样、哪里容易错」；写进 `notes/a1.md`；commit。
6. **优化/扩展**：正确后再 profile、优化、做实验；把数字写进实验日志。

### 卡住时的 hint ladder（按顺序，不跳级）

1. 重读 handout 对应段落 + 测试输入形状；
2. 看 `stanford-cs336-2026-notes-main` 对应节；
3. 看中文笔记/中文导读的**文字解释**（不看代码）；
4. 用最小例子手算/打印中间张量；
5. 看导读的代码结构（只看结构，看完关掉重写）；
6. 问 AI/同学：只问「我的理解哪里错」，不要要代码。

**45 分钟规则**：同一问题卡 45 分钟，必须写下「我已知/我假设/我下一步验证」，然后才允许看导读或求助。

### 测试即 spec

- `tests/adapters.py` 只是 glue：把测试输入接到你的实现；**不要把逻辑写进 adapter**。
- 不要改 `tests/`；测试失败先看是哪一维、哪个 dtype、哪个边界。
- snapshot 测试有 `rtol/atol`；数值差一点通常是 dtype、初始化、或公式顺序，不是「测试太严」。
- macOS 会 skip 两个 Linux-only 内存测试；要证明内存效率，在 Linux（Modal/Docker）补跑。

---

## 5. 六周逐步计划（A1）

> 默认从 **2026-11-09** 开始（与主计划 A1 的 W2–W7 对齐）；如果你从 11-02 开始，把 Week 0 压缩到 10-29 → 11-01，Week 1 从 11-02 开始。  
> 每周 5 个学习日 + 周末长跑/缓冲；每天 2.5–3 h。

### Week 0（10-29 → 11-08）：orientation + setup，不写模型

| Day | 学习 | 动手 | 完成标准 |
|---|---|---|---|
| D1 | handout §1–2.2；`lecture-01.md` §5–6 标题 | `uv sync`；`uv run pytest -q` 记录失败数量；建 `notes/` | baseline 数字 + 首个报错 |
| D2 | `lecture-01.md` §6 BPE；handout 的 `bpe_example` | 手算 `low/lower/widest/newest` 的 merge 顺序；写 unicode1/2 答案草稿 | 能解释 tie-break 和特殊 token |
| D3 | `lecture-02.md` §5–9（einsum、FLOPs、memory） | 在 Python 里复现 2 个 einsum 例子；算一次 GPT-2 参数/显存 | 手算和代码对上 |
| D4 | `lecture-03.md` §2–6 + `lecture_03_notes.md` | 画 pre-norm block 图；列出 A1 组件清单和形状 | 一张自己的架构图 |
| D5（可选） | Modal/云账号、预算上限 | 跑一次 GPU 冒烟；记录型号/价格 | 云通道可用 |

**Gate 0**：baseline 清楚、L1–L3 的问题清单、A1 组件清单、云预算上限。**不要写 BPE。**

---

### Week 1（11-09 → 11-15）：BPE 训练

| Day | 学习 | 动手 | 完成标准 |
|---|---|---|---|
| D1 | handout §2.3–2.4；`lecture-01.md` §6.1–6.2 | 实现最朴素的 BPE（字节、预分词、特殊 token） | 小语料上 merge 序列与手算一致 |
| D2 | handout §2.4 的 tie-break/特殊 token；`test_train_bpe.py` | 跑 `uv run pytest -k test_train_bpe`；修正确性 | `test_train_bpe` 通过 |
| D3 | §2.5 性能提示；用 cProfile/py-spy | 优化到 `test_train_bpe_speed` <1.5 s；记录朴素版/优化版耗时 | 速度测试通过 |
| D4 | §2.5 TinyStories 数据 | 先用验证集 22K 文档做 debug 数据集；再全量 10K vocab；序列化 vocab/merges | 时间/内存/最长 token 有数字 |
| D5 | §2.5 OWT 的要求；中文导读补1.2 | 规划 OWT：云 CPU 100 GB 或子集；写 2 句实验结论 | Week 1 实验日志 |

**Gate 1**：`test_train_bpe` + `test_train_bpe_speed` 绿；TinyStories 10K tokenizer 已保存；知道 OWT 32K 怎么跑。  
**如果超期**：周末补；不要为速度牺牲正确性。

---

### Week 2（11-16 → 11-22）：Tokenizer 类 + 实验 + 模型基础

| Day | 学习 | 动手 | 完成标准 |
|---|---|---|---|
| D1 | handout §2.6；`lecture-01.md` §6.3 | 实现 encode/decode/特殊 token/`encode_iterable`/`from_files` | 小例子 round-trip |
| D2 | `test_tokenizer.py` | 跑 tokenizer 测试；修 unicode/特殊 token/流式边界 | `test_tokenizer.py` 全绿 |
| D3 | handout §2.7 | 做 10 文档实验：压缩比、跨 tokenizer、吞吐、uint16；tokenize TinyStories 为 `.bin` | 4 个问题有答案；`.bin` 可加载 |
| D4 | handout §3.3；`lecture-02.md` §5 | 实现 Linear/Embedding；跑 `-k test_linear`/`test_embedding` | 两个测试绿 |
| D5 | §3.4.1；`lecture-03.md` §2；中文笔记 §2 | 实现 RMSNorm；跑 `-k test_rmsnorm`；读 RoPE 推导 | RMSNorm 绿 + RoPE 笔记 |

**Gate 2**：tokenizer 全绿 + 4 个实验答案 + Linear/Embedding/RMSNorm 绿 + TinyStories `.bin`。  
**注意**：macOS 会 skip 两个内存测试；不要以为 skip 就等于通过。

---

### Week 3（11-23 → 11-29）：模型核心 + 资源核算

| Day | 学习 | 动手 | 完成标准 |
|---|---|---|---|
| D1 | handout §3.4.3；`lecture-03.md` §5；中文笔记 §4 | 实现 RoPE；跑 `-k test_rope`；手推相对位置 | RoPE 绿 |
| D2 | §3.4.4；`lecture-03.md` §6 | 实现 softmax + scaled dot-product attention；跑两个测试 | softmax/3D/4D attention 绿 |
| D3 | §3.4.5；中文笔记 §6 | 实现 causal MHA + RoPE；跑 `-k test_multihead_self_attention` | MHA 绿 |
| D4 | §3.4.2/3.5 | SwiGLU、block、完整 LM；跑 `-k test_swiglu`/`test_transformer_block`/`test_transformer_lm` | 全模型测试绿 |
| D5 | §3.5 资源核算；`lecture-02.md` §6–9 | 做 transformer accounting 五问；写进 writeup 草稿；overfit 一个 minibatch | 核算答案 + loss 接近 0 |

**Gate 3**：`test_model.py` 全绿 + accounting 答案 + 单 batch overfit。  
**如果卡在 RoPE/MHA**：回到 `lecture-03` 的图，先把 q/k 形状写成 `(B,H,T,D)`，再检查 mask 和 head 维。

---

### Week 4（11-30 → 12-06）：训练工具 + 训练循环

| Day | 学习 | 动手 | 完成标准 |
|---|---|---|---|
| D1 | §4.1 | 实现 cross entropy；跑 `-k test_cross_entropy`；算 perplexity | CE 绿 |
| D2 | §4.2–4.3；`lecture-02.md` §10 | 实现 AdamW；跑 `-k test_adamw`；跑 SGD 学习率 toy 实验 | AdamW 绿 + toy 结论 |
| D3 | §4.4–4.5 | cosine schedule + warmup、grad clip；跑两个测试 | 两个测试绿 |
| D4 | §5.1–5.2 | data loader + checkpoint；跑 `test_get_batch`/`test_checkpointing` | 两个测试绿 |
| D5 | §5.3 | 训练脚本（CLI 超参、memmap、日志、checkpoint）；CPU/MPS 小跑 100 步 | loss 下降；脚本可复现 |

**Gate 4**：所有单元测试绿 + 训练脚本能在 TinyStories 小数据上跑起来。  
**注意**：AdamW 的 t 从 1 开始、weight decay 与 moment 更新的顺序、cosine 的边界，是最常见的三个坑。

---

### Week 5（12-07 → 12-13）：TinyStories 训练 + 实验

| Day | 学习 | 动手 | 完成标准 |
|---|---|---|---|
| D1 | §7.2；`lecture-03.md` §6 | 用 10K tokenizer 准备数据；跑 200–500 步 smoke；确认吞吐/显存 | 1 条 loss 曲线 |
| D2 | §7.2.1 | LR sweep（小 token 预算：2–3 个 LR）；记录曲线 | 选出一个稳定 LR |
| D3 | 低资源路径 | 40M token 训练（batch 8–16，context 256，MPS/CPU）；周期验证 | val loss ≤2.0（目标） |
| D4 | §6 | 实现/完善 decoding；生成 256 token；调 temperature/top-p | 生成文本 + 分析 |
| D5 | §7.2.1 batch size | batch size 实验（至少 3 个）；记录吞吐/曲线；开始消融 | batch 结论 + 第一组消融 |

**Gate 5（部分）**：TinyStories 模型可生成、val loss 达标或接近；LR/batch 实验有曲线。  
**MPS 注意**：不要设 `torch.set_float32_matmul_precision('high')`；`torch.compile` 用 `backend="aot_eager"`；和 CPU 小样本对比数值。

---

### Week 6（12-14 → 12-20）：消融 + OWT + writeup

| Day | 学习 | 动手 | 完成标准 |
|---|---|---|---|
| D1 | §7.3 | 完成 4 个消融：去 RMSNorm、post-norm、NoPE、SwiGLU vs SiLU；画对比曲线 | 4 条曲线 + 结论 |
| D2 | §7.4；中文导读 3.4 | OWT 32K BPE（云 CPU 100 GB 或子集）；tokenize OWT | OWT tokenizer + `.bin` |
| D3 | §7.4 | OWT 训练（1×B200 约 45 分钟，或本地子集）；对比 TinyStories loss | OWT 曲线 + 生成文本 |
| D4 | §7.5（可选） | leaderboard 45 分钟 run（可选）；记录 val loss/曲线 | 可选：提交/记录 |
| D5 | writeup | 补全所有文字题、图表、实验日志；跑全量测试；打包 | `writeup.pdf` + `code.zip` |

**Gate 6（最终）**：§3.4 的 Definition of Done 全部打勾；全量测试绿；提交包能生成。

---

## 6. 测试地图：每块该跑什么

| 模块 | 命令 | 覆盖 | Gate |
|---|---|---|---|
| BPE 训练 | `uv run pytest -k test_train_bpe` | 正确性、特殊 token、速度 | W1 |
| Tokenizer | `uv run pytest tests/test_tokenizer.py` | round-trip、特殊 token、与 tiktoken 一致、流式 | W2 |
| Model | `uv run pytest tests/test_model.py` | Linear/Embedding/SwiGLU/RMSNorm/RoPE/softmax/attention/MHA/block/LM | W3 |
| nn utils | `uv run pytest tests/test_nn_utils.py` | softmax、cross entropy、grad clip | W4 |
| optimizer | `uv run pytest tests/test_optimizer.py` | AdamW、cosine schedule | W4 |
| serialization | `uv run pytest tests/test_serialization.py` | checkpoint save/load | W4 |
| data | `uv run pytest tests/test_data.py` | get_batch 形状/dtype/device | W4 |
| 全量 | `uv run pytest` | 所有单测 | 每周 gate / 最终 |

使用方式：

- 实现完一个组件，**只跑对应的 `-k`**；不要每次全量，反馈太慢。
- 失败时先看测试的输入 shape 和断言；不要改测试。
- 通过后 commit：`feat(a1): <component> passes tests`。
- macOS 跳过 Linux-only 内存测试是正常的；要验证内存，用 Linux 容器/Modal 补跑。

---

## 7. 本地算力方案（M5 / 16 GB）

### 7.1 哪些能本地做

| 任务 | 本地 | 方案 | 预计 |
|---|---|---|---|
| TinyStories 10K BPE | ✅ | 先验证集 22K 文档，再全量；预分词分块/多进程；控制内存 | 10–30 分钟（优化后） |
| Tokenizer 测试/实验 | ✅ | CPU | 分钟级 |
| 模型单测/小训练 | ✅ | CPU/MPS | 分钟级 |
| TinyStories 40M token 训练 | ✅ | MPS，batch 8–16，context 256；`torch.compile(aot_eager)` | 1.5–4 小时 |
| TinyStories 328M token（官方基准） | ⚠️ | 云 1×B200 约 20–30 分钟 | 本地太慢，不推荐 |
| OWT 32K BPE | ❌ | 需 ~100 GB 内存；Modal CPU 或子集 | 云 |
| OWT 训练 | ❌/⚠️ | 云 1×B200 约 45 分钟；或本地小数据子集 | 云 |
| 4 个消融 | ✅/⚠️ | 本地 10–20M token 小 run；或云 0.5 B200 每个 | 本地数小时 |
| Leaderboard | ❌ | B200 45 分钟；可选 | 云 |

### 7.2 本地训练建议

- 基准配置按 handout：d_model 512、d_ff 1344、4 层、16 头、context 256、vocab 10K、RoPE theta 10000。
- 低资源：总 token 40M（如 32×5000×256）；目标 val loss ≤2.0。M5 16GB 可以把 batch 降到 8/16，保持总 token 不变。
- 用 `np.memmap` 加载 token 数组；验证集单独切；每 N 步评估。
- MPS：不要用 TF32；`torch.compile(model, backend="aot_eager")`；先在 CPU 上跑 20 步确认数值正常。
- 训练前先 **overfit 一个 minibatch**（loss 接近 0），再上全量；否则调 LR 是浪费时间。
- 监控：loss、grad norm、tokens/s、峰值内存；数据加载/验证/checkpoint 频率不要拖慢训练。

### 7.3 云端只花在刀刃上

| 用途 | 建议 | 估算 |
|---|---|---|
| OWT 32K BPE | Modal CPU，≥100 GB 内存；或先 1/10 子集 | <$5 |
| 1 次 OWT 训练 | 1×B200，≤45 分钟 | ~$5 |
| Leaderboard（可选） | 2–3 次 45 分钟 run | $10–20 |
| 官方完整实验预算 | 19 B200 小时 | ~$120（第一次不必） |

**A1 总预算建议：$0–25；上限 $50。** 先用每月 $30 免费额度；设 spend limit；跑完下载 artifact 并停实例。

### 7.4 没有云也能完成什么

- 可以完成：BPE（TinyStories）、Tokenizer、模型、训练工具、TinyStories 40M 训练、生成、4 个消融（小规模）、大部分 writeup。
- 可以跳过/标注：OWT BPE 全量、OWT 训练、leaderboard。
- 不要假装做了 OWT；在 writeup 里写明「self-study low-resource path」。学习价值在方法和数字，不在假装全量。

---

## 8. 写作与实验记录（容易被忽略的一半）

### 8.1 要写的题

| 时间 | 题目 | 写作要点 |
|---|---|---|
| W0–W2 | unicode1、unicode2 | 一句话回答 + 反例；不要抄定义 |
| W2 | tokenizer_experiments (a–d) | 压缩比、跨 tokenizer、吞吐、uint16 理由；给数字 |
| W3 | transformer_accounting (a–e) | 列矩阵乘、FLOPs、参数量、显存；给公式和代入 |
| W4 | learning_rate_tuning | SGD 三个 LR 的 loss 行为；一句话结论 |
| W4 | adamw_accounting (a–d) | 参数/激活/梯度/优化器状态；batch size 表达式；MFU 估算 |
| W5 | learning_rate (a–b) | 多条 LR 曲线；发散点与最佳 LR 的关系 |
| W5 | batch_size_experiment | 多条曲线；吞吐 vs 稳定性 |
| W5–W6 | generate | 256 token 文本 + 流畅度 + 两个影响因素 |
| W6 | 4 个消融 | 每个：曲线 + 2–3 句解释 |
| W6 | main_experiment | OWT 曲线 + 与 TinyStories 的 loss 解释 + 生成文本 |
| W6 | experiment_log | 所有实验的表格/链接 |
| 可选 | leaderboard | final val loss + <45 min 曲线 + 做了什么 |

### 8.2 实验日志模板

每次 run 必填：

```
日期：
目标/假设：
配置（模型、数据、token 数、batch、LR、schedule、GPU）：
命令/commit：
结果（final loss、val loss、tokens/s、峰值显存、wall-clock）：
曲线位置：
意外/失败原因：
下一步：
```

没有这 7 行，视为没做。实验日志就是 writeup 的素材。

### 8.3 writeup 结构

1. **Tokenizer**：unicode 答案、BPE 实现与性能、tokenizer 实验。
2. **Model**：架构图、组件、资源核算（表 + 公式）。
3. **Training**：cross entropy/AdamW、schedule/clip、data/checkpoint、训练循环。
4. **Experiments**：TinyStories LR/batch、生成、4 个消融、OWT；每张图配 2–3 句结论。
5. **Reflection**：你遇到的 3 个 bug、怎么定位、下次怎么避免。

图表要求：

- loss vs step 和 loss vs wall-clock 都要有（handout 明确要求）。
- 同一张图固定颜色/坐标范围；标注模型配置和 token 数。
- 不要只放「最好看」的 run；发散/失败的 run 也是结论。

### 8.4 提交检查

- [ ] `uv run pytest` 全绿
- [ ] `writeup.pdf` 回答所有题目，图表清晰
- [ ] `code.zip` 含 `cs336_basics/` 和必要的脚本；不含大数据/权重
- [ ] `make_submission.sh` 能生成包
- [ ] checkpoint 能 load 并生成
- [ ] 实验日志完整

---

## 9. 调试手册

| 症状 | 可能原因 | 先检查 |
|---|---|---|
| `test_train_bpe` merges 不一致 | tie-break 不是字典序更大；特殊 token 参与了统计；预分词边界 | 手算小例子；打印每轮选中的 pair |
| BPE 速度测试失败 | 每轮全量重扫；Python 循环太多；未用多进程 | cProfile；先正确再优化 |
| tokenizer round-trip 失败 | 特殊 token 被切开；decode 未用 `errors="replace"`；流式跨 chunk | 用 fixture 小文本；打印 token/bytes |
| 模型 snapshot 差一点 | dtype/float32、初始化、einsum 标签、mask 位置 | 打印 shape/dtype；对照公式 |
| RoPE 测试失败 | 位置索引错；旋转配对错；q/k 维度顺序；对 v 也做了 | 用 T=2 手算一个向量 |
| attention 测试失败 | mask True/False 语义反了；mask 加在 softmax 后；scale 错 | 打印 mask 和 softmax 前分数 |
| MHA 测试失败 | head 拆分/合并错；transpose 后未 contiguous；out_proj 顺序 | 打印 `(B,T,C)->(B,H,T,D)` 每步 |
| AdamW 失败 | t 从 0 开始；weight decay 顺序错；bias correction 错；state 未初始化 | 对照 handout Algorithm 1 |
| schedule 失败 | warmup/cosine 边界；`t > Tc`；步数 off-by-one | 打印 t=0/Tw/Tc 附近的值 |
| checkpoint 失败 | 只存 model 没存 optimizer/iteration；`map_location` | 读回后继续训练一步 |
| get_batch 失败 | dtype 不是 int64；device 未搬；切片越界 | 打印 shape/dtype/device |
| loss 不下降 | LR 太大/太小；数据未 tokenize 对；mask 漏；loss reduction 错 | 先 overfit 一个 minibatch |
| loss 是 NaN | 未减 max；LR 太大；fp16 溢出；RoPE 位置越界 | 缩小模型/LR；检查 softmax |
| MPS 结果不对 | TF32、部分算子不支持、compile 后端 | 同输入 CPU vs MPS；不要设 TF32 |
| 训练比预期慢 | 数据加载/验证/checkpoint 太频繁；batch 太小；没 compile | profiler；先排除数据加载 |

**通用顺序**：shape → dtype → device → 公式 → 测试容差。不要一上来改算法。

---

## 10. A1 之后怎么接 A2–A5

- A1 的 gate 全过、writeup 提交后，再开 A2；不要边做 A1 边看 Triton。
- A2 需要 NVIDIA GPU；先补 L4（注意力替代，选）、L5（GPU/TPU）、L6（kernel/Triton）、L7–L8（并行）、L10（推理）。本地笔记 `lecture-04..08/10` 已就绪。
- 保持「作业驱动讲座」：每讲只看到能推动当前作业的部分。
- 完整 26 周排期见 `CS336_ROADMAP.md`；A1 只是第一段。

---

## 11. 今天开始（2–3 小时）

1. **读**：handout §1–2.2 + `lecture-01.md` §5–6 的标题；写下 3 个问题。
2. **跑**：`cd /Users/yunni/Projects/assignment1-basics && uv sync && uv run pytest -q`；记录失败数量和第一个报错。
3. **写**：unicode1 的三问 + unicode2 的三问草稿；不要求完美，先写自己的答案。
4. **不要做**：不要打开中文导读的代码块；不要开始写 BPE；不要看 A2。

> 72 小时后你应该有：baseline 失败清单、L1 的 5 行笔记、unicode 答案草稿、A1 组件清单。

---

## 附录 A：Problem → 资料 → 测试/交付 映射

### A.1 Tokenizer

| Problem | Handout | 先读 | 测试/交付 |
|---|---|---|---|
| unicode1 | §2.1 | `lecture-01.md` §6 开头 | writeup 3 问 |
| unicode2 | §2.2 | `lecture-01.md` §6 | writeup 3 问 + 反例 |
| train_bpe | §2.4 | `lecture-01.md` §6.1–6.4；导读 补充1.2 | `test_train_bpe` |
| train_bpe_tinystories | §2.5 | `lecture-01.md` §6；导读 补充1.3 | 时间/内存/最长 token 答案 |
| train_bpe_expts_owt | §2.5 | 同上 | OWT tokenizer + 对比 |
| tokenizer | §2.6 | `lecture-01.md` §6.3；导读 补充1.4 | `test_tokenizer.py` |
| tokenizer_experiments | §2.7 | `lecture-01.md` §5；导读 补充1.4 | 4 问 + TinyStories `.bin` |

### A.2 Model

| Problem | Handout | 先读 | 测试/交付 |
|---|---|---|---|
| linear | §3.3.2 | `lecture-02.md` §5 | `test_linear` |
| embedding | §3.3.3 | `lecture-02.md` §5 | `test_embedding` |
| rmsnorm | §3.4.1 | `lecture-03.md` §2；中文笔记 §2 | `test_rmsnorm` |
| positionwise_feedforward | §3.4.2 | `lecture-03.md` §3；中文笔记 §3 | `test_swiglu` |
| rope | §3.4.3 | `lecture-03.md` §5；中文笔记 §4 | `test_rope` |
| softmax | §3.4.4 | `lecture-02.md` §5；`lecture-03.md` §6 | `test_softmax_matches_pytorch` |
| scaled_dot_product_attention | §3.4.4 | `lecture-03.md` §6 | `test_scaled_dot_product_attention` / `test_4d_...` |
| multihead_self_attention | §3.4.5 | `lecture-03.md` §6；中文笔记 §6 | `test_multihead_self_attention` |
| transformer_block | §3.5 | `lecture-03.md` §2–3 | `test_transformer_block` |
| transformer_lm | §3.5 | `lecture-03.md` §2–6 | `test_transformer_lm` / truncated |
| transformer_accounting | §3.5 | `lecture-02.md` §6–9 | writeup 5 问 |

### A.3 Training / generation

| Problem | Handout | 先读 | 测试/交付 |
|---|---|---|---|
| cross_entropy | §4.1 | `lecture-02.md` §10 | `test_cross_entropy` |
| learning_rate_tuning | §4.2.1 | `lecture-02.md` §10；导读 2.2.1 | writeup |
| adamw | §4.3 | `lecture-02.md` §10 | `test_adamw` |
| adamw_accounting | §4.3 | `lecture-02.md` §9 | writeup 4 问 |
| learning_rate_schedule | §4.4 | `lecture-02.md` §10 | `test_get_lr_cosine_schedule` |
| gradient_clipping | §4.5 | `lecture-02.md` §10 | `test_gradient_clipping` |
| data_loading | §5.1 | `lecture-02.md` §9 | `test_get_batch` |
| checkpointing | §5.2 | `lecture-02.md` §9 | `test_checkpointing` |
| training_together | §5.3 | `lecture-02.md` §10；导读 2.3.3 | 训练脚本 |
| decoding | §6 | `lecture-01.md` §6.3；导读 3.3 | 生成文本 |

### A.4 Experiments

| Problem | Handout | 先读 | 交付 |
|---|---|---|---|
| experiment_log | §7.1 | 导读 3.1 | 日志 + 曲线 |
| learning_rate | §7.2.1 | 导读 3.2.3 | LR 曲线 + 结论 |
| batch_size_experiment | §7.2.1 | 导读 3.2.3 | batch 曲线 + 结论 |
| generate | §7.2.2 | 导读 3.3 | 256 token 文本 + 分析 |
| layer_norm_ablation | §7.3 | 导读 3.2.4 | 去 RMSNorm 曲线 |
| pre_norm_ablation | §7.3 | 导读 3.2.4 | post-norm 曲线 |
| no_pos_emb | §7.3 | 导读 3.2.4 | RoPE vs NoPE |
| swiglu_ablation | §7.3 | 导读 3.2.4 | SwiGLU vs SiLU |
| main_experiment | §7.4 | 导读 3.4 | OWT 曲线 + 文本 |
| leaderboard | §7.5 | 导读 补充2 | 可选：45 分钟 run |

---

## 附录 B：六周检查表

| 周 | 日期 | Gate | 关键产出 | 云 |
|---|---|---|---|---|
| W0 | 10-29 → 11-08 | baseline + L1–L3 诊断 | 失败清单、unicode 草稿、架构图 | 冒烟 |
| W1 | 11-09 → 11-15 | BPE tests + 10K tokenizer | BPE 代码、时间/内存数字 | 0 |
| W2 | 11-16 → 11-22 | tokenizer 全绿 + 实验 + 基础模块 | `.bin`、4 个实验答案、Linear/Embedding/RMSNorm | 0 |
| W3 | 11-23 → 11-29 | 模型全绿 + accounting | 全模型、核算答案、overfit | 0 |
| W4 | 11-30 → 12-06 | 训练工具全绿 | AdamW/schedule/clip/data/checkpoint/loop | 0 |
| W5 | 12-07 → 12-13 | TinyStories 训练 + 实验 | 模型、生成、LR/batch 曲线 | 0 |
| W6 | 12-14 → 12-20 | 消融 + OWT + writeup | 4 条消融、OWT、`writeup.pdf` | ~$10–20 |

---

## 附录 C：提交前 10 项检查

- [ ] `uv run pytest` 全绿；记录 skip 项
- [ ] 所有 written problems 有答案，不是空白/占位
- [ ] 每张图有轴标签、单位、配置、结论
- [ ] 实验日志能对应到每张图/每个数字
- [ ] checkpoint 能 load，生成 256 token
- [ ] `writeup.pdf` 能打开，页码/公式正常
- [ ] `code.zip` 不含大数据/权重，含运行说明
- [ ] `make_submission.sh` 能跑通
- [ ] 低资源/云路径在 writeup 里说明清楚
- [ ] 能脱稿讲 10 分钟：你做了什么、数字说明什么、哪里失败过

---

> 这份方案只给顺序、资料、验收和调试方法，不含任何作业实现或伪代码。  
> 下一步：从 §11 的 3 件事开始；做完 Week 1 的 Gate 再来看 Week 2。
