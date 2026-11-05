# Transformer 模型：从形状到数值对齐的从零实现

> 版本：2026-11-05
> 适用：读完本仓库 `transformer_steps/`（或等价材料）、准备做 CS336 A1 §3.3–3.5，或想系统搞懂 pre-norm Transformer 每个组件“为什么长这样”的自学者
> 配套代码：`model_lab/`（教学实现 + 43 项测试 + 资源核算 + 过拟合实验）
> 事实底座：`model_lab/SPEC.md`（接口、数值约定、官方快照配方）、A1 handout §3.3–3.5（`.tmp/a1_handout.txt:684–1030`）
> 一句话：**写一个 forward 能跑通的 Transformer，代码量并不大；这个 lab 真正难的是“证明它和官方实现逐元素一致”，以及“把参数、显存、FLOPs 算清楚”。**

> **定稿状态**：本文与冻结版 `model_lab/toy_transformer.py`、`model_lab/test_toy_transformer.py`、`model_lab/results/accounting.json`、`model_lab/results/overfit.json` 对齐（2026-11-05）。所有实测数字均在所在表格/段落标注来源命令；没有跑过的写「未测」。

---

## 目录

- [0. 一页纸结论与使用说明](#0-一页纸结论与使用说明)
- [1. 为什么还要写一个模型 lab](#1-为什么还要写一个模型-lab)
- [2. 形状语言：B T C H D V](#2-形状语言b-t-c-h-d-v)
- [3. 逐组件实现](#3-逐组件实现)
- [4. 组装与完整数据流](#4-组装与完整数据流)
- [5. 怎么证明写对了](#5-怎么证明写对了)
- [6. 资源核算](#6-资源核算)
- [7. 过拟合实验](#7-过拟合实验)
- [8. 与 A1 和 training_lab 的关系](#8-与-a1-和-training_lab-的关系)
- [9. 学习计划与自测](#9-学习计划与自测)
- [10. FAQ 20 问](#10-faq-20-问)
- [附录 A 命令速查](#附录-a-命令速查)
- [附录 B 术语表](#附录-b-术语表)
- [附录 C 材料地图](#附录-c-材料地图)
- [附录 D 检查表](#附录-d-检查表)

---

## 0. 一页纸结论与使用说明

### 0.1 这份教程解决什么问题

市面上的 Transformer 教程大致分两类：一类把论文公式贴一遍，另一类直接甩一份完整代码。两者都跳过了新手真正卡住的地方——**公式里的维度到底对应张量的哪一维，以及“我觉得写对了”和“数值上确实写对了”之间的差距**。

这份教程只回答三个问题，每个问题都对应可验证的产出：

| 问题 | 在哪一章回答 | 产出（可检查） |
| --- | --- | --- |
| 每个组件到底在算什么？ | 第 2–4 章 | 能默写公式、形状和参数矩阵的维度；能解释每个 `transpose` 为什么存在 |
| 我怎么知道写对了？ | 第 5 章 | `model_lab` 的 43 项测试全绿；12 个官方快照的 maxdiff 打得出来 |
| 怎么算账、怎么证明可训练？ | 第 6–7 章 | `results/accounting.json`、`results/overfit.json` 里的数字能复现 |

它**不**做的事：

- 不替代 handout。接口、数值约定、测试清单以 `model_lab/SPEC.md` 和官方 handout 为准；本文负责“把规格翻译成直觉和排错路径”。
- 不是 A1 答案。`model_lab` 的接口层与官方 `tests/adapters.py` 故意不同，代码更慢、更啰嗦、带自检；直接复制去交 A1 既不符合学术诚信，也过不了 adapter 测试。
- 不重讲论文。RoPE、SwiGLU 的出处只在必要处点名，重点放在“实现时哪一行会错”。

### 0.2 一页纸结论

如果你只读一页，记住这八条：

1. **结构是固定的**：`ids → token_embeddings → N × TransformerBlock → ln_final → lm_head → logits`。每个 block 做两次“归一化 → 子层 → 残差”：`y = x + attn(rms(x))`，`z = y + ffn(rms(y))`。残差流本身**不**被归一化，这是 pre-norm 的核心。
2. **全程只有三种形状**：`(B,T,C)`、`(B,H,T,D)`、`(B,H,T,T)`；`C = H·D`；最后变成 `(B,T,V)`。多头不是“并行跑 H 个模型”，而是把 `C` 切成 H 份、把 head 当成 batch 维，算完再拼回来过一个 `output_proj`。
3. **三个最常错的数值契约**：mask 的 `True` 表示**可见**；RoPE 用**交错配对**且 `d_k = C/H`（不是 `C`）；权重用 `trunc_normal_` 按 handout 的 `σ = sqrt(2/(d_in+d_out))` 初始化。前两项一旦写错，loss 可能照样下降，但官方快照一定过不了；初始化则是另一个独立问题——parity 会 `load_state_dict` 覆盖它，必须单独测（见 §1.3.1）。
4. **验证有优先级**：官方快照（13 个 `.npz`，按 12 项 parity 计）> 手算小例子 > PyTorch 独立参考 > 自洽性质测试。本 lab 实测 45 个测试函数全绿（44 个覆盖 43 项 + 1 个 C27 回归；`.venv/bin/python model_lab/test_toy_transformer.py`，1.04–1.23 s，2 线程）。“loss 降了”“生成像人话”都不是正确性证据。
5. **官方快照的实测 maxdiff**：13 个快照的最大绝对误差在 `0`～`1.049e-05`，worst 是 `test_transformer_lm_truncated_input` 的 `1.049e-05`；全部低于本 lab 自设的 `1e-4` 门槛一个量级（命令同上，完整表见 §5.6）。Lead 独立抽查的 LM maxdiff 为 `1.001e-05`，与 impl 报告一致。若你的 parity 明显大于 `1e-4`，先查 mask 方向、RoPE 配对、per-head `d_k`、`lm_head` 是否被误 tie、block 的 pre-norm 顺序。
6. **资源账可以解析地算**：参数量 = `V·C`（输入嵌入）+ 每层 `4C² + 3C·F + 2C` + `C`（final norm）+ `V·C`（untied lm_head）；前向 FLOPs 每层约 `8T·C² + 4T²·C + 6T·C·F`，lm_head 再加 `2T·C·V`。本 lab 核算 GPT-2 XL 为 1,640,452,800 参数、6257.8 MiB fp32 权重、3.52e12 前向 FLOPs（ctx 1024）；ctx 16384 时 FLOPs 为 1.34e14，是 1024 的 37.98 倍，attention 占比从 9.16% 升到 61.73%（`results/accounting.json`）。`T` 变大时 `T²` 的注意力项占比上升——这是长上下文变贵的主因。
7. **过拟合实验只能证明“实现可训练”**：单 batch、字节级、玩具规模。本 lab 实测 300 步把固定 batch 的 loss 从 `5.673745` 压到 `0.003972`（1.7172 s，2 线程，`results/overfit.json`）——它证明梯度通路、loss、优化器接口没接错，**不能**证明模型学到了语言能力，也不能外推到真实训练。
8. **数字纪律**：本文每个实测数字都标注来源（`results/*.json`、测试输出或命令）；没跑过的写「未测」，不写推测值。这条比任何结论都重要。

### 0.3 三条阅读路线

时间都是**粗略估计**，不是实测；按你的目标选一条，不要从头到尾线性读。

| 路线 | 适合谁 | 读哪些 | 动手 | 预计时间 | 验收 |
| --- | --- | --- | --- | --- | --- |
| A. 结论速览 | 复习/面试/只想判断值不值得学 | 0、1、5.6、6.5、7.2 | 无 | 约 1 小时 | 能脱稿解释三个数值契约和“为什么 loss 不能当证据” |
| B. 跟着写一遍 | 第一次实现 Transformer | 2 → 3 → 4 → 5 | 自己写 `toy_transformer.py`，再对照；跑 43 项测试 | 2–3 天 | 测试全绿；12 个 maxdiff 打得出来；能解释每处 transpose |
| C. A1 备考 | 要交 A1 §3.3–3.5 | 全篇 + handout §3.3–3.5 + 官方测试源码 | 写 `tests/adapters.py`，用官方仓库跑 pytest | 约 1 周 | 官方 `test_linear` 到 `test_transformer_lm` 全绿；写 writeup 时能引用本文的核算推导 |

> **不要做的事**：一上来读 RoPE/SwiGLU 论文；先手算一个 2×2 的 attention，先把形状跑通，先让一个组件对齐官方快照。论文放到第 5 章之后再读，收益高得多。

### 0.4 前置知识

| 必须会 | 为什么需要 | 5 分钟自检 |
| --- | --- | --- |
| PyTorch 的 `Tensor` / `nn.Module` / `nn.Parameter` / `state_dict` | 所有组件都是 `nn.Module`，parity 靠 `load_state_dict` | 能解释 `model.state_dict()` 里的键从哪来、`load_state_dict(strict=True)` 在检查什么 |
| 矩阵乘、转置、广播 | 第 2 章的全部内容 | 能手算 `(2,3) @ (3,4)` 的形状；能说出 `(4,12,16)` 和 `(2,2,12,16)` 能否直接相加 |
| softmax、交叉熵、`logsumexp` | SDPA 的数值稳定性和 loss | 能解释 softmax 为什么要先减最大值；能写出 `-log softmax(o)[y]` 的稳定形式 |
| 读过 `transformer_steps/` step01–07（或等价材料） | 知道 Q/K/V、残差、head 是什么 | 不看代码写出 `softmax(QKᵀ/√d_k)V`，并说出每个矩阵的形状 |

**不需要**：GPU、CUDA、分布式训练、FlashAttention、KV cache、tokenizer 细节、大规模数据管线。这个 lab 全部在 CPU 上跑，模型是玩具规模。

### 0.5 与 A1 的关系（这不是 A1 答案）

| 项 | CS336 A1 | 本 lab（`model_lab/`） |
| --- | --- | --- |
| 架构 | handout §3.3–3.5 的 pre-norm Transformer | 相同 |
| 权重与 `state_dict` 键 | 官方 `model.pt` 的键名 | **相同**（parity 要直接 `load_state_dict`，键名不能改） |
| 接口层 | `tests/adapters.py` 里的 `run_*` 函数，权重以参数传入 | 直接构造 `Linear`/`TransformerLM` 等模块，再用 `load_state_dict` 装官方权重 |
| 参考实现 | 官方测试 + 固定种子的 `.npz` 快照 | 相同快照，只读，不修改官方仓库 |
| 测试 | 官方 pytest | 自写 45 个测试函数：44 个覆盖 43 项 + 1 个 C27 回归（双模式：直接跑 + pytest 兼容），其中 12 项 parity 对应 13 个官方快照 |
| 额外内容 | 无 | 形状断言与自检、`accounting.py` 资源核算、`exp_overfit.py` 单 batch 过拟合 |
| 学术诚信 | A1 要求自己写 | 本文是“读参考实现”的材料；读完请合上重写，不要复制提交 |

**边界**：官方仓库 `/Users/yunni/Projects/assignment1-basics` 只读。本 lab 不写它、不改它的测试和 fixtures，只用它的权重与快照做数值对齐。找不到官方仓库时，parity 测试打印 `SKIP` 并继续，不能报错退出（`SPEC.md` §5）。

### 0.6 环境与成本

环境以 `model_lab/SPEC.md` §1 为准：

- 解释器：`/Users/yunni/Projects/cs336-prep-code/.venv/bin/python`（Python 3.14.3，torch 2.14.1）。
- 纯 CPU；所有可能超过 1 秒的脚本开头 `torch.set_num_threads(2)`；单条命令 **< 120 秒**。
- 不联网、不安装依赖；画图时设 `MPLCONFIGDIR=$PWD/.tmp/mpl`。

脚本就绪后，三个入口分别是：

```bash
# 0) 自检：形状 / 参数量 / state_dict 键名 / 反向传播（约 0.9 s）
.venv/bin/python model_lab/toy_transformer.py

# 1) 45 个测试函数 = 44 个覆盖 43 项 + 1 个 C27 回归（parity 缺失时 SKIP）
.venv/bin/python model_lab/test_toy_transformer.py

# 2) 资源核算 -> model_lab/results/accounting.json（任一 check 失败 exit 1）
.venv/bin/python model_lab/accounting.py --strict

# 3) 单 batch 过拟合 -> model_lab/results/overfit.json
.venv/bin/python model_lab/exp_overfit.py --max-seconds 90
```

> 实测：自检 0.9 s；测试 45 passed / 0 failed / 0 skipped、1.04–1.23 s；`accounting.py --strict` 约 0.5 s 墙钟（JSON `runtime` 随运行小幅波动：实测 `elapsed_s` ≈0.006、`peak_rss_mib` ≈212.5）；`exp_overfit.py` 1.7172 s、peak RSS 316.1 MiB。以上均为 2 线程。

---

## 1. 为什么还要写一个模型 lab

### 1.1 仓库里已经有什么

本仓库不是没有 Transformer 材料。恰恰相反，“再写一个”必须先回答“和已有的差在哪”：

| 材料 | 一句话定位 | 它讲清楚了什么 | 它不覆盖什么 |
| --- | --- | --- | --- |
| `transformer_steps/`（8 步） | 从零实现 GPT-2 骨架 | 字符分词、学习式位置嵌入、LayerNorm、GELU-FFN、残差、head 拆分、训练与生成；每步一个可单独运行的文件 | A1 的初始化、官方快照、per-head RoPE、资源核算；注意力用融合 `qkv` 线性层，不是 A1 的独立投影 |
| `llama_steps/`（9 步） | 把上面的 GPT-2 改成 Llama | RMSNorm、SwiGLU、RoPE、GQA、KV cache、与 HuggingFace `LlamaForCausalLM` 逐元素对齐、装真实 SmolLM2-135M 权重 | A1 的精确接口与 3D mask、官方 13 个快照（12 项 parity）、参数/FLOPs 核算；RoPE 用前后半配对（`rotate_half`，HF 约定），不是 A1 的交错配对 |
| `minimal_transformer.py` / `minimal_llama.py` | 单文件压缩版 | 训练 + 生成 + KV cache 的全貌 | 没有分步讲解，不适合第一次学 |
| `bpe_lab/` + `BPE_TOKENIZER_TUTORIAL.md` | 分词器专项 | 文本 ↔ token ID 的全过程；tokenizer 与模型的接口 | 模型内部的形状、数值、算账 |
| `/Users/yunni/Projects/assignment1-basics` | 官方骨架 + 测试 + fixtures | 唯一权威的测试与参考快照；`tests/adapters.py` 定义接口 | 没有实现（骨架里 `raise NotImplementedError`） |
| `model_lab/` + 本文 | A1 §3.3–3.5 的契约式实现 | 精确接口、数值约定、13 个快照对齐、资源核算、过拟合实验 | 不是 A1 答案；不训练有语言能力的模型 |

一句话总结：**`transformer_steps/` 教你“概念如何一步步长出来”，`llama_steps/` 教你“Llama 改了哪几处”，`model_lab/` 教你“怎么证明一套实现和官方逐元素一致，并且算得清账”。** 三者目标不同，不是替代关系。

### 1.2 差异表：同一套 Transformer，目标完全不同

| 维度 | `transformer_steps/` | `llama_steps/` | `model_lab/`（本 lab） |
| --- | --- | --- | --- |
| 教学主线 | 8 步，每步只加一个概念 | 9 步，从 GPT-2 改到 Llama | 一次给全 A1 §3.3–3.5 的契约，再逐项证明 |
| 架构 | GPT-2 骨架：学习式位置嵌入、LayerNorm、GELU-FFN、tied embeddings、字符分词 | Llama：RMSNorm、SwiGLU、RoPE、GQA、KV cache | Llama 组件 + A1 精确接口：MHA（无 GQA）、无 KV cache、默认 untied |
| 位置编码 | 学习式 `(T,C)` 表，加在输入上 | RoPE，前后半配对（`rotate_half`，与 HF 一致） | RoPE，**交错配对**，`d_k = C/H`，`token_positions` 支持 `(T,)`/`(1,T)`/`(B,T)`/`(1,1,T)` |
| 初始化 | `nn.init.normal_(std=0.02)` | 同左 | `trunc_normal_`：Linear `σ=sqrt(2/(d_in+d_out))`，Embedding `σ=1`，RMSNorm gain 全 1 |
| 权重绑定 | tied（`lm_head.weight` 与 `token_embedding.weight` 共享） | 默认 untied（SmolLM2 实例 tied） | 默认 untied；tie 只做单独的单元测试 |
| 对齐参考 | 自洽：打印形状、跑自检 | HuggingFace `LlamaForCausalLM`（+ 真实 SmolLM2 权重） | 官方 13 个 `.npz` 快照（12 项 parity）+ 固定 seed 输入 + `state_dict` 键集 |
| 接口与 `state_dict` | 自定义 `Config` + `MiniGPT`；注意力用融合 `qkv`；键名自定 | 自定义 `LlamaConfig` + `MiniLlama`；键名自定 | A1 契约：独立的 `q_proj/k_proj/v_proj/output_proj`，键名与官方 `model.pt` 完全一致，可直接 `load_state_dict` |
| 资源核算 | 无 | 只算 KV cache | `accounting.py`：参数量、fp32 显存、前向 FLOPs；GPT-2 四配置 + ctx 16384 |
| 训练实验 | 300 步字符级训练 + 生成 | 同预算 GPT-2 vs Llama 对照 | 单 batch 过拟合：验证“可训练”，不验证语言能力 |

这张表里，教学主线、位置编码、初始化、接口和对齐参考这几行最容易被人忽略，也最能解释“为什么跑通了还要再写一个”。下面逐条展开其中最关键的三个。

### 1.3 三个最容易被“能跑”掩盖的差异

#### 1.3.1 初始化：`load_state_dict` 会掩盖错的初始化

A1 handout §3.3.1 指定：

- Linear：`N(0, σ²)`，`σ² = 2/(d_in + d_out)`，截断在 `[-3σ, 3σ]`；
- Embedding：`N(0, 1)`，截断在 `[-3, 3]`；
- RMSNorm 的 gain 初始化为全 1。

用 `torch.nn.init.trunc_normal_` 实现。截断的作用是去掉正态分布尾部那些过大的初始值；pre-norm Transformer 对初始化不算敏感，但初始化会显著影响训练早期的收敛速度（handout 原话大意：pre-norm 异常鲁棒，但仍会影响训练速度与收敛）。

**关键陷阱**：parity 测试会把官方权重 `load_state_dict` 进来，**初始化瞬间被覆盖**。所以“13 个快照全过”根本不能证明你的初始化写对了。初始化必须单独测（`SPEC.md` §6 的 27–29 项：Linear 的 std 与 ±3σ 范围、Embedding 的 std 与 ±3 范围、gain 全 1 且无 bias）。这也是本 lab 与 `transformer_steps/`、`llama_steps/` 的一个实质差别：那两个 lab 用 `std=0.02` 的普通正态初始化，够训练，但不是 A1 契约。

#### 1.3.2 RoPE：交错配对 vs 前后半配对

A1 handout §3.4.3 的 RoPE 把 `d_k` 维向量按**相邻两维**配对：`(x_0, x_1), (x_2, x_3), …`，第 `k` 对的角度是 `pos · Θ^(-2k/d_k)`，`k = 0..d_k/2-1`。而 HuggingFace / `llama_steps/` 用的是 `rotate_half`：把**前一半和后一半**配对，`(x_i, x_{i+d_k/2})`。

两者都是“二维旋转”，差别只在通道顺序。如果你把 q/k 的通道做一个固定置换、同时把 `W_q/W_k` 的行做同样置换，attention 分数完全不变。问题在于：**官方快照里的权重是按交错顺序训练的**，你不能自己换顺序。`llama_steps/step03_rope.py` 的第 5 节专门演示了这件事：直接对未重排的通道套 `rotate_interleaved`，分数与 half-split 版本对不上；先把通道 `interleave` 重排再旋转，两者才等价。

还有两个连带细节：

- **per-head 的 `d_k`**：MHA 里 RoPE 作用在每个 head 的 `(B,H,T,D)` 上，`D = C/H`，不是整个 `C`。角度表形状是 `(max_seq_len, D/2)`；所有层共享同一个 RoPE 实例。官方 `test_rope` 用的 `d_k=64`（单头直接测），而 `test_multihead_self_attention_with_rope` 用的是 `d_k=16`——同一份实现必须两种都能跑。
- **只转 q/k，不转 v**：v 的路径不参与旋转；旋转也不改变向量模长（正交变换），所以 q/k 的范数在 RoPE 前后应一致。

#### 1.3.3 官方快照：13 个快照、12 项 parity，不是“跑通就行”

A1 的仓库里有一组固定种子的输入与对应输出快照（`.npz`，键名 `array`），覆盖从 `test_linear` 到 `test_transformer_lm_truncated_input` 的 13 个快照。`SPEC.md` §5/§6 把它们算成 **12 项 parity**：`test_swiglu` 与 `test_positionwise_feedforward` 同权重、同公式，算一项，但两个 `.npz` 都要跑。`test_positionwise_feedforward` 在官方 `test_model.py` 里没有同名测试函数（是遗留快照），本 lab 仍拿它做额外对照（测试 `E40`）。它们比“训练 loss 下降”强得多，因为：

1. **参考是独立的**：输出由官方参考实现生成，不是你的实现自己和自己对。
2. **输入是固定的**：`SPEC.md` §5 记录了每个输入对应的 `torch.manual_seed`（例如 `in_embeddings` 用 seed 4、`q` 用 seed 1、`k` 用 seed 2、`v` 用 seed 3、`mask` 用 seed 5、`idx` 用 seed 6），以及几何配置（`vocab=10000, ctx=16, d_model=64, num_layers=3, num_heads=4, d_ff=128, theta=10000`，`d_head=16`）。
3. **它逼你注意接口**：权重文件里的键带 `_orig_mod.` 前缀要先 strip；`lm_head.weight` 与 `token_embeddings.weight` 在参考里**不相等**（untied），所以默认不能 tie；SDPA 必须支持 3D mask（fixture 的 mask 是 `(4,12,16)`），4D 测试则把 q/k/v 和 mask 一起按 `head=2` 重排成 4D。

这也是为什么本 lab 的 parity 测试要打印每个快照的 maxdiff，而不是只报 PASS：**maxdiff 的量级本身就是诊断信息**。本 lab 实测 13 个快照的 maxdiff 在 `0`～`1.049e-05`（§5.6，命令 `.venv/bin/python model_lab/test_toy_transformer.py`）；如果某个组件跑到 `1e-2`，哪怕 assert 用的是宽松容差，也说明实现有系统性问题，只是被容差盖住了。

### 1.4 什么时候用哪个 lab

| 你的目标 | 用哪个 | 理由 |
| --- | --- | --- |
| 第一次理解 attention / 残差 / head | `transformer_steps/` | 8 步增量，形状和自检打印得最密 |
| 想知道 Llama 相对 GPT-2 改了什么、KV cache 怎么省 | `llama_steps/` | 有同预算对照和 HF 逐元素对齐 |
| 做 CS336 A1 §3.3–3.5，或需要“和官方数值一致” | `model_lab/` + 本文 | 唯一按 A1 契约、用官方快照对齐的实现 |
| 要 tokenizer | `bpe_lab/` + `BPE_TOKENIZER_TUTORIAL.md` | 分词是独立问题，本 lab 不重复 |
| 要训练、调参、数据管线、优化器 | 等下一阶段的 `training_lab` | 本 lab 只提供模型与两个诊断实验 |
| 想读 RoPE/SwiGLU 原论文 | 先读完本文第 5 章 | 没有数值手感时读论文，容易记住故事、记不住实现 |

### 1.5 本 lab 的边界（不做什么）

- **不训练有语言能力的模型**。`exp_overfit.py` 是单 batch 过拟合，目标是证明“能训”，不是“训得好”。
- **不覆盖推理优化**。没有 KV cache、GQA、FlashAttention、量化；这些在 `llama_steps/` 和真实推理栈里。
- **不覆盖训练基础设施**。没有数据加载器、学习率调度、梯度裁剪、检查点、分布式；这些属于后续 `training_lab` 和 A1 §4–5。
- **不是 A1 提交代码**。接口不同、测试不同、还多了自检与核算；请按 §0.5 的边界使用。
- **不做变长 batch**。LM 契约里 `T <= context_length`，一个 batch 内所有序列等长；padding/mask 的工程问题不在本 lab 范围内。

---

## 2. 形状语言：B T C H D V

Transformer 的 bug 里，超过一半是形状/维度语义错，而不是公式错。这一章先把“符号—形状—谁和谁能广播”这件事说死，后面第 3 章的每个组件都只是往这张骨架上填肉。

### 2.1 符号表

| 符号 | 含义 | 官方快照配置（只读 fixture） | 约束 |
| --- | --- | --- | --- |
| `B` | batch size | 4 | 任意正整数；SDPA/RoPE 还允许更多前导 batch 维 |
| `T` | sequence length（当前输入的 token 数） | 12（LM 的输入），SDPA 里 query 长度也是 12 | `T <= context_length`（LM 要求，`SPEC.md` §4） |
| `T_k` | key/value 的序列长度 | 16（SDPA 的 k/v） | 可与 `T` 不同；SDPA 必须支持 `T != T_k` |
| `C` | `d_model`，每个 token 的隐藏维 | 64 | 必须能被 `H` 整除 |
| `H` | head 数 | 4 | `C % H == 0`，否则 MHA 构造时报错 |
| `D` | `d_head = C / H` | 16 | RoPE 的 `d_k` 就是它（MHA 内） |
| `V` | 词表大小 | 10000 | 与 embedding / lm_head 的行数一致 |
| `F` | `d_ff`，FFN 内层维 | 128（fixture 显式给定） | `d_ff=None` 时按 8/3·C 取最近的 64 倍数、最小 64（`SPEC.md` §4） |
| `N` | Transformer block 层数 | 3 | 任意正整数 |

> 官方快照配置来自 `model_lab/SPEC.md` §5 与 `assignment1-basics/tests/fixtures/ts_tests/model_config.json`。`d_ff=128` 是 fixture 给定的值；不要把它和“8/3·C 取整”的默认规则混为一谈——默认规则用于你自己构造模型时。

三个派生关系，建议直接写成断言：

```
C == H * D
H 整除 C
x.shape[-1] == C    # 进入 block / LM 的隐藏维
```

### 2.2 三条形状规则

#### 规则一：batch-like 维永远在最左

handout §3.2 的约定：所有 batch-like 维度（batch、head、以及任何“并行独立计算”的维）都放在最左边，**最后一个或两个维度才是被操作的语义维**。所以：

- SDPA 接受 `(..., T, d_k)` 和 `(..., T_k, d_v)`，返回 `(..., T, d_v)`；`...` 可以是任意多个前导维。
- RoPE 接受 `(..., T, d_k)` 和 `(..., T)` 的位置，返回同形状。
- 本 lab 的 MHA / Block / LM 在 `(B,T,C)` 上工作；只有 SDPA 和 RoPE 需要“任意前导维”的通用性（因为 4D 测试会传 `(B,H,T,D)`）。

这条规则的价值是：**看到 `x.transpose(-2, -1)` 就知道在转“序列维和特征维”；看到 `transpose(1, 2)` 就知道在把 head 提到 batch 位置。** 如果代码里出现 `transpose(0, 1)` 或 `permute` 之后接 `view`，先停下来检查是不是把 batch 维当成了语义维。

#### 规则二：head 当 batch，不是“H 个独立模型”

多头注意力的正确心智模型：

```
(B, T, C) --q_proj--> (B, T, C) --view/transpose--> (B, H, T, D)
```

每个 head 在自己的 `D` 维子空间里算 attention，**head 之间在 attention 内部没有任何交互**；它们只在最后 `transpose + reshape` 回 `(B,T,C)`、再过一个 `output_proj` 时才混合。因此：

- SDPA 的 batch 维是 `B*H`（或显式 `(B,H)`），不需要为每个 head 写循环。
- causal mask 是 `(T,T)`，广播到 `(B,H,T,T)`；每个 head 用同一张 mask。
- RoPE 对每个 head 用同一组角度；`token_positions` 广播到 `(B,H,T)`。

#### 规则三：广播靠右对齐，不要手工 expand

PyTorch 的广播从**最右维**开始对齐。这条规则解释了几个“看起来能加、其实不能”的例子：

| 想做的事 | 形状 | 结果 |
| --- | --- | --- |
| causal mask 加到 scores | `(T,T)` + `(B,H,T,T)` | ✅ 广播成 `(B,H,T,T)` |
| 3D mask 加到 3D scores | `(B,T,T_k)` + `(B,T,T_k)` | ✅ 逐元素；SDPA 的 3D 测试就是这一路 |
| 3D mask 加到 4D scores | `(4,12,16)` + `(2,2,12,16)` | ❌ 右对齐成 `(1,4,12,16)`，第 `-3` 维 `4 ≠ 2`；官方 4D 测试会先把 mask 也重排成 `(2,2,12,16)` |
| 位置 `(T,)` 的 cos 表加到 `(B,H,T,D/2)` | `(T,D/2)` + `(B,H,T,D/2)` | ✅ 广播；等价于先 `unsqueeze` 成 `(1,1,T,D/2)` |
| 位置 `(B,T)` 的 cos 表加到 `(B,H,T,D/2)` | `(B,T,D/2)` + `(B,H,T,D/2)` | ❌ 右对齐成 `(1,B,T,D/2)`，`B` 落到 head 维；应先 `unsqueeze(-2)` 成 `(B,1,T)` 再取表 |

实践建议：**在写实现前先把每个中间张量的形状写成注释**；对“本来就应该广播”的地方，宁可显式 `unsqueeze` 一次，也不要把“能跑”当成“语义对”。尤其是在 RoPE 里，`(B,T)` 的位置张量用高级索引取表得到的是 `(B,T,D/2)`，不是 `(B,H,T,D/2)`——你必须自己决定在 `H` 前面还是后面补维。

### 2.3 一次前向的完整形状表

下表是一次标准 LM 前向（训练时再接 cross-entropy）。`B,T,C,H,D,V,F` 见 §2.1；`T_k` 只在 SDPA 的通用描述里出现，LM 里 `T_k = T`。

| # | 操作 | 输入形状 | 输出形状 | 备注 |
| --- | --- | --- | --- | --- |
| 0 | token ids | — | `(B,T)` | `torch.long`，值域 `[0,V)` |
| 1 | `token_embeddings(ids)` | `(B,T)` | `(B,T,C)` | 查表，不改形状的语义 |
| 2 | `ln1(x)`（RMSNorm） | `(B,T,C)` | `(B,T,C)` | 只在最后一维上归一化 |
| 3 | `q_proj/k_proj/v_proj` | `(B,T,C)` | 各 `(B,T,C)` | 三个 `(C,C)` 权重；也可以融合成一次 `(3C,C)`，但 state_dict 键会变 |
| 4 | 拆 head | `(B,T,C)` | `(B,T,H,D)` | `view(B,T,H,D)`；第 `h` 个 head 取最后维的 `[h*D:(h+1)*D]` |
| 5 | 提到 batch 位 | `(B,T,H,D)` | `(B,H,T,D)` | `transpose(1,2)`；此后序列维是 `-2`，特征维是 `-1` |
| 6 | RoPE(q), RoPE(k) | `(B,H,T,D)` + 位置 `(...,T)` | `(B,H,T,D)` | 只转 q/k；v 不动 |
| 7 | `q @ kᵀ / sqrt(D)` | `(B,H,T,D)`、`(B,H,T_k,D)` | `(B,H,T,T_k)` | `k.transpose(-2,-1)`；LM 里 `T_k=T` |
| 8 | 加 causal mask | `(B,H,T,T_k)`、`(T,T_k)` | `(B,H,T,T_k)` | `True`=可见；`masked_fill(~mask, -inf)` |
| 9 | softmax(dim=-1) | `(B,H,T,T_k)` | `(B,H,T,T_k)` | 每行和为 1；被 mask 的位置权重为 0 |
| 10 | `weights @ v` | `(B,H,T,T_k)`、`(B,H,T_k,D)` | `(B,H,T,D)` | 对 value 加权求和 |
| 11 | 合并 head | `(B,H,T,D)` | `(B,T,C)` | `transpose(1,2)` 后接 `contiguous().view(B,T,C)` |
| 12 | `output_proj` | `(B,T,C)` | `(B,T,C)` | `(C,C)`；混合各 head 的信息 |
| 13 | 第一次残差 | `(B,T,C)` + `(B,T,C)` | `(B,T,C)` | `y = x + attn(ln1(x))` |
| 14 | `ln2(y)` | `(B,T,C)` | `(B,T,C)` | 第二个 RMSNorm |
| 15 | `w1(y), w3(y)` | `(B,T,C)` | 各 `(B,T,F)` | 两个升维投影 |
| 16 | `silu(w1(y)) * w3(y)` | `(B,T,F)`、`(B,T,F)` | `(B,T,F)` | 门控逐元素相乘 |
| 17 | `w2(...)` | `(B,T,F)` | `(B,T,C)` | 降回 `C` |
| 18 | 第二次残差 | `(B,T,C)` + `(B,T,C)` | `(B,T,C)` | `z = y + ffn(ln2(y))` |
| 19 | 重复 2–18 共 `N` 次 | `(B,T,C)` | `(B,T,C)` | 形状完全不变——这是 block 的契约 |
| 20 | `ln_final` | `(B,T,C)` | `(B,T,C)` | 最后一个 RMSNorm，不在 block 内 |
| 21 | `lm_head` | `(B,T,C)` | `(B,T,V)` | `(V,C)` 权重；logits，不是概率 |
| 22 | cross-entropy（训练时，不属于 `LM.forward`） | logits `(B,T,V)`、targets `(B,T)` | 标量 `()` | 通常 shift：`logits[:, :-1]` vs `ids[:, 1:]` |

两个值得单独记住的“形状不变”点：

1. **block 的输入输出都是 `(B,T,C)`**。如果你在 block 内部把形状改成了别的，最后必须改回来；否则残差加不上，或者 LM 的 `lm_head` 维度对不上。
2. **从 `(B,T,C)` 到 `(B,T,V)` 只发生在最后一步**。注意力分数 `(B,H,T,T)` 是中间量，不是输出；不要把它和 logits 混为一谈。

### 2.4 参数矩阵的形状

| 参数（`state_dict` 键） | 形状 | 元素个数 | 说明 |
| --- | --- | --- | --- |
| `token_embeddings.weight` | `(V,C)` | `V·C` | 每一行是一个 token 的输入向量 |
| `layers.{i}.attn.q_proj.weight` | `(C,C)` | `C²` | 四个投影各一份 |
| `layers.{i}.attn.k_proj.weight` | `(C,C)` | `C²` | |
| `layers.{i}.attn.v_proj.weight` | `(C,C)` | `C²` | |
| `layers.{i}.attn.output_proj.weight` | `(C,C)` | `C²` | `SPEC.md` 的键名是 `output_proj`，不是 `o_proj` |
| `layers.{i}.ln1.weight` | `(C,)` | `C` | RMSNorm 的 gain，初始全 1 |
| `layers.{i}.ln2.weight` | `(C,)` | `C` | |
| `layers.{i}.ffn.w1.weight` | `(F,C)` | `F·C` | 升维 / 门控分支 |
| `layers.{i}.ffn.w3.weight` | `(F,C)` | `F·C` | |
| `layers.{i}.ffn.w2.weight` | `(C,F)` | `C·F` | 降维 |
| `ln_final.weight` | `(C,)` | `C` | |
| `lm_head.weight` | `(V,C)` | `V·C` | 默认 untied；tie 时共享存储 |

untied 情况下的总参数量（推导见 §6.2）：

```
params = 2·V·C + N·(4·C² + 3·C·F + 2·C) + C
```

注意 `Linear` 存的是 `(out_features, in_features)`，所以 `w1` 是 `(F,C)`、`w2` 是 `(C,F)`；`forward` 走 `x @ W.T`。这一点和 PyTorch `nn.Linear` 的存储一致，但和你脑子里“`y = Wx`”的写法差一个转置——第 3.1 节会专门讲这个坑。

### 2.5 广播与索引的三个实例

#### 实例 A：causal mask

scores 是 `(B,H,T,T)`，causal mask 是 `(T,T)` 的下三角（`True`=可见）。`scores.masked_fill(~mask, -inf)` 时：

```
(T,T)  ->  右对齐  ->  (1,1,T,T)  ->  广播到 (B,H,T,T)
```

`mask` 的语义是 **`True` 表示“信息可以流动”**（handout §3.4.4 明确说，`True` 稍微反直觉）。上三角为 `False`，softmax 之后未来位置的权重为 0。用 `torch.tril(torch.ones(T,T, dtype=torch.bool))` 构造，不要用 `triu` 搞反方向。

#### 实例 B：3D mask 与 4D scores

官方 fixture 的 mask 是 `(4,12,16)`（3D，`B=4, T=12, T_k=16`），配套的 `test_scaled_dot_product_attention` 也是 3D。到了 `test_4d_scaled_dot_product_attention`，测试会做：

```
q,k,v: (4,12,64)/(4,16,64)  --rearrange head=2-->  (2,2,12,64)/(2,2,16,64)
mask:  (4,12,16)            --rearrange head=2-->  (2,2,12,16)
```

也就是说，**4D 测试不仅把 q/k/v 变 4D，也把 mask 变 4D**。你的 SDPA 只需要支持“mask 与 scores 可广播”即可，不需要自己把 3D mask 猜成 4D。反过来说，如果实现里写了 `mask = mask.unsqueeze(1)` 之类的硬编码，4D 测试会挂——因为形状已经被测试改好了。

#### 实例 C：`token_positions` 与 RoPE 表

RoPE 的 cos/sin 缓存形状是 `(max_seq_len, D/2)`（`SPEC.md` §4）。取表时用高级索引：

- 位置 `(T,)` → 取表得到 `(T,D/2)`；对 `(B,H,T,D/2)` 的 cos 需要 `(1,1,T,D/2)`，靠右对齐直接广播。
- 位置 `(1,T)` → 取表得到 `(1,T,D/2)`；广播到 `(B,H,T,D/2)`。
- 位置 `(B,T)` → 直接取表得到 `(B,T,D/2)`，**不能**直接广播到 `(B,H,T,D/2)`：右对齐会把 `B` 补到 head 维。应先 `positions.unsqueeze(-2)` 得到 `(B,1,T)`，取表后是 `(B,1,T,D/2)`，再广播到 `H`。这一步是 RoPE 实现里最容易“形状对了但语义错了”的地方；MHA 内部现在会自动做这个 `unsqueeze`，直接调用 RoPE 时仍要自己补。

再强调一次：**RoPE 只转 q/k，v 不转**；cos/sin 缓存是 fp32，乘之前要 cast 到 `x.dtype`（`SPEC.md` §4 的 dtype 约定）。在本 lab 的 MHA 里，`token_positions` 支持 `(T,)`（默认）、`(1,T)`、`(B,T)`、`(1,1,T)`；MHA 在 `token_positions.ndim == x.ndim - 1` 时会自动 `unsqueeze(-2)`，把 `(B,T)` 变成 `(B,1,T)` 再广播到 `(B,H,T,D)`。`(B,H,T)` 对 3D 输入不是受支持形状：`H == num_heads` 时可能因形状恰好匹配而广播（未测试，不要依赖），`H != num_heads` 会报错；4D+ 输入若要逐 batch 位置，请显式补成 `(B1,B2,1,T)`。

### 2.6 形状断言清单

把下面这些断言写进 `forward`（或用 `assert` / `if ... raise ValueError`），比事后 debug 便宜得多：

| 位置 | 断言 | 失败时说明什么 |
| --- | --- | --- |
| Embedding | `ids.dtype == torch.long`；`ids.shape == (B,T)`；`0 <= ids < V` | 传了 float、传了 `(T,B)`、ID 越界 |
| Linear | `x.shape[-1] == in_features` | 权重形状和输入对不上 |
| RMSNorm | `x.shape[-1] == d_model` | 把 `(B,T)` 或 `(B,T,H,D)` 直接喂进来了 |
| SwiGLU | `w1/w3` 输出 `F`、`w2` 输入 `F`；默认 `d_ff` 是 64 的倍数 | 三个矩阵顺序接错 |
| RoPE | `x.shape[-1] == d_k`；`token_positions.shape[-1] == x.shape[-2]` | 把 `C` 当 `d_k`；位置和序列长度没对齐 |
| SDPA | `q.shape[-1] == k.shape[-1]`；`k.shape[-2] == v.shape[-2]`；`mask.dtype == torch.bool` | q/k 特征维不等；k/v 序列长不等；mask 传了 0/1 整数 |
| MHA | `C % H == 0`；输出形状与输入一致 | head 不整除；transpose/reshape 顺序错 |
| Block | 输出形状 == 输入形状 | 残差接错 / 忘记改回 `(B,T,C)` |
| LM | `T <= context_length`；`logits.shape == (B,T,V)` | 超长输入；lm_head 维度不对 |
| `state_dict` | 键集合等于 `SPEC.md` §3 列出的集合 | 模块属性名不一致（parity 会直接失败） |

### 2.7 三个自测

1. **内存布局**：`(B,T,C) -> view(B,T,H,D) -> transpose(1,2)` 之后，张量在内存里还是连续的吗？为什么合并 head 时要先 `transpose(1,2)` 再 `contiguous()`，而不能直接 `view(B,T,C)`？
   <details><summary>要点</summary>`transpose` 只改 stride，不搬数据；此时 `view` 会报错或得到错误布局，必须先 `contiguous()` 把 head 维搬回原位。`view(B,T,H,D)` 的语义是“第 h 个 head 占最后维的一段连续通道”，这个约定决定了 `q_proj` 输出的通道如何分配给 head。</details>

2. **广播判断**：mask `(4,12,16)` 能不能直接加到 scores `(2,2,12,16)` 上？如果不能，官方 4D 测试是怎么处理的？
   <details><summary>要点</summary>不能。右对齐把 3D mask 补成 `(1,4,12,16)`，第 `-3` 维 `4` 与 scores 的 `2` 不匹配。官方测试用 `rearrange("(batch head) query key -> batch head query key", head=2)` 把 mask 也变成 `(2,2,12,16)`，所以实现只需支持“可广播的 mask”，不必猜维度。</details>

3. **位置张量**：`x` 是 `(B,H,T,D)`、`token_positions` 是 `(B,T)`，取完 cos 表得到什么形状？要广播到 `(B,H,T,D/2)`，应该在哪一维 `unsqueeze`？
   <details><summary>要点</summary>取表得到 `(B,T,D/2)`；在 `T` 前补一维得到 `(B,1,T,D/2)`，即可广播到 `(B,H,T,D/2)`。如果错补在最后（`(B,T,D/2,1)`），`T` 会落到 `-3` 维，和 `(B,H,T,D/2)` 的 `H` 维冲突。（MHA 现在会自动完成这个 `unsqueeze(-2)`；这里说的是直接调用 RoPE 的情况。）</details>

---

## 3. 逐组件实现

> 本章代码摘自冻结版 `model_lab/toy_transformer.py`（2026-11-05）。为了可读性删掉了 docstring、`extra_repr`、部分空行与部分类型标注，**逻辑与最终文件一致**；完整文件请直接对照源码。每节固定五件事：公式 → 直觉 → 代码 → 三个坑 → 对应测试。

### 3.1 Linear

**公式**：`y = x Wᵀ`，`W ∈ R^{d_out×d_in}`；`forward` 的输入最后一维是 `d_in`，输出最后一维是 `d_out`。没有 bias。

**直觉**：只做一次线性变换。现代 LLM 通常去掉 bias，少一组参数、便于算子融合；handout §3.3.2 要求“存 W 而不是 Wᵀ”，这样官方 `state_dict` 里的 `(d_out,d_in)` 权重能原样装进来。

```python
def _truncated_normal_(param: Tensor, std: float) -> None:
    nn.init.trunc_normal_(param, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std)


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int,
                 device=None, dtype=None) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # 参数形状是 (out, in)：前向时 x @ weight.T 才是 (..., in) -> (..., out)
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        std = math.sqrt(2.0 / (in_features + out_features))
        _truncated_normal_(self.weight, std)

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight.transpose(-2, -1)
```

**三个坑**：

1. **存成 `(in,out)` 再接 `x @ W`**：方阵时形状检查发现不了，parity 直接偏；正确约定是 `(out,in)` + `x @ W.T`。
2. **顺手加 bias**：`state_dict` 会多出 `bias` 键，官方权重 `strict=True` 装载失败；A1 明确禁止 bias。
3. **初始化 std 写成 `sqrt(2/in)` 或 `1/sqrt(in)`**：形状和 parity 都发现不了（官方权重会覆盖初始化），只有初始化测试（D27，`±25%` 容差）能抓到。

**对应测试**：11（权重布局/无 bias/前向等价）、27（std 与 ±3σ）、29（无 bias 参数）；官方快照 `test_linear`（`E31`，用 `layers.0.ffn.w1.weight`）。

### 3.2 Embedding

**公式**：`E[ids]`，`E ∈ R^{V×C}`；`ids` 可以是任意形状 `(...)`，输出 `(..., C)`。

**直觉**：查表。每个 token ID 对应一行向量；`d_model` 是最后一维，前面的维度只是“多套一层形状”。

```python
class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int,
                 device=None, dtype=None) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        _truncated_normal_(self.weight, std=1.0)

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.weight[token_ids]
```

**三个坑**：

1. **用 `nn.Embedding` / `F.embedding`**：A1 禁止；而且内置模块的初始化、参数名与契约不同，parity 会偏。
2. **忘了 ids 必须是整数**：`weight[ids]` 对 float 会直接报索引错误；`(T,B)` 和 `(B,T)` 也不会报错，只会悄悄转置语义——先断言 `ids.dtype == torch.long`。
3. **初始化 std 跟着 Linear 用 `sqrt(2/(in+out))`**：embedding 的契约是 `std=1`、截断 `±3`（比 Linear 大很多），D28 会抓。

**对应测试**：12（查表 = `weight[ids]`，1D/2D/3D ids）、28（初始化）；官方快照 `test_embedding`（`E32`）。

### 3.3 RMSNorm

**公式**：`RMS(a) = sqrt(mean(a²) + eps)`，`RMSNorm(a) = a / RMS(a) * g`，`g ∈ R^C`，`eps=1e-5`。没有减均值、没有 bias。

**直觉**：只按“均方根”缩放，再把每个通道乘一个可学习 gain。它不改变方向，只控制尺度；参数只有 `C` 个（LayerNorm 是 `2C`）。

```python
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5,
                 device=None, dtype=None) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        in_dtype = x.dtype
        x32 = x.to(torch.float32)  # 平方前 upcast，避免 fp16/bf16 溢出
        rms = torch.sqrt(x32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        out = x32 / rms * self.weight.to(torch.float32)
        return out.to(in_dtype)
```

**三个坑**：

1. **`mean` 写成 `sum`**：分母多乘了 `C`，输出整体缩小；手算测试（A07）能抓到。
2. **`eps` 放在 `sqrt` 外面**：`sqrt(mean(x²)) + eps` 与 `sqrt(mean(x²)+eps)` 只在数值上差一点，但官方快照会偏；公式要求放里面。
3. **忘了 dtype 透传**：内部 fp32 算完后必须 `.to(in_dtype)`；bf16 进、fp32 出会让后续 matmul 的 dtype 不一致（B13 专门测）。

**对应测试**：7（与公式一致，随机 gain）、13（dtype 保持）、24（正尺度不变）；官方快照 `test_rmsnorm`（`E34`，用 `layers.1.ln1.weight`）。

### 3.4 SwiGLU

**公式**：`FFN(x) = W2( SiLU(W1 x) ⊙ (W3 x) )`，`W1,W3 ∈ R^{F×C}`，`W2 ∈ R^{C×F}`；`SiLU(x)=x·sigmoid(x)`。`d_ff=None` 时取 8/3·C 最近的 64 倍数，最小 64。

**直觉**：一条门控分支（`W3`）决定另一条分支（`W1` 过 SiLU）哪些通道通过；三个矩阵让参数量与 4C 的 ReLU-FFN 大致对齐，因此内层宽度取 8/3·C。

```python
def _default_d_ff(d_model: int) -> int:
    # 用与 accounting.d_ff_default 完全相同的 half-up 公式，保证两处结果一致
    n = math.floor(8.0 * d_model / (3.0 * 64.0) + 0.5)
    return max(64, 64 * n)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int | None = None,
                 device=None, dtype=None) -> None:
        super().__init__()
        if d_ff is None:
            d_ff = _default_d_ff(d_model)
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))
```

**三个坑**：

1. **`w1` 与 `w3` 接反**：交换后 `SiLU(W3x)⊙(W1x)` 与公式不同，手算测试（A10）会偏；三个矩阵名字必须与 `state_dict` 一致。
2. **SiLU 写成 sigmoid**：少了乘 `x`，输出量级完全变了；`silu(x) = x * torch.sigmoid(x)`。
3. **`d_ff` 取整**：必须是“最近的 64 倍数”而不是向下取整；`_default_d_ff` 与 `accounting.py` 现在统一用 half-up（`floor(8d/(3·64)+0.5)`），避免 Python `round` 的银行家舍入。`d_model=60` 是 tie 边界的例子：8/3·60=160，160/64=2.5，half-up 取 3 → 192；旧写法会得 128。显式传了 `d_ff` 就不要再覆盖用户的值。

**对应测试**：10（`silu` 与手写公式、`swiglu` 与手写公式）、14（默认 `d_ff`：512→1344、1600→4288、最小 64、显式值不覆盖）；官方快照 `test_swiglu`（`E33`）与 `test_positionwise_feedforward`（`E40`，同权重同公式的额外对照）。

### 3.5 softmax

**公式**：`softmax(v)_i = exp(v_i) / Σ_j exp(v_j)`；实现上先减该维最大值：`softmax(v) = softmax(v - max(v))`。

**直觉**：softmax 对整体平移不变，减最大值只是把最大项变成 0，避免 `exp` 溢出成 `inf`；代价是多一次 `max`。

```python
def softmax(x: Tensor, dim: int) -> Tensor:
    x_max = x.max(dim=dim, keepdim=True).values
    shifted = x - x_max  # 数值稳定：最大值变成 0
    exp = torch.exp(shifted)
    return exp / exp.sum(dim=dim, keepdim=True)
```

**三个坑**：

1. **忘了 `keepdim=True`**：`(B,H,T,T) - (B,H,T)` 会按最后一维广播成错误形状，且不报错。
2. **整行被 mask 掉**：`max=-inf`，`x-max` 产生 `nan`；正常因果 mask 每行至少对角线可见，但写通用 SDPA 时要意识到这个病态输入。
3. **在 fp16/bf16 里直接 `exp`**：即使减了 max，极端值仍可能溢出；本 lab 的 SDPA 在输入 dtype 下算，靠减 max 保证稳定。

**对应测试**：1（行和为 1、手算、负 `dim`）、2（大输入不溢出、整体加常数不变、无 inf/nan）、3（平移不变性）。

### 3.6 scaled dot-product attention

**公式**：`Attention(Q,K,V) = softmax(QKᵀ/√d_k, dim=-1) V`。`Q:(...,T_q,d_k)`、`K:(...,T_k,d_k)`、`V:(...,T_k,d_v)`；mask 是 bool，`True`=可见，`False` 在 softmax 前填 `-inf`。

**直觉**：用每个 query 去匹配所有 key 得到分数，softmax 成权重，再对 value 加权求和。除以 `√d_k` 是因为点积方差随 `d_k` 增长，不缩放会让 softmax 饱和、梯度接近 0。

```python
def scaled_dot_product_attention(q: Tensor, k: Tensor, v: Tensor,
                                 mask: Tensor | None = None) -> Tensor:
    d_k = q.shape[-1]
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        if mask.dtype != torch.bool:
            raise TypeError(f"mask 必须是 bool 张量，得到 {mask.dtype}")
        scores = scores.masked_fill(~mask, float("-inf"))
    return softmax(scores, dim=-1) @ v
```

**三个坑**：

1. **mask 方向反了**：`True` 是“可见”，`False` 才填 `-inf`。写反后模型能看到未来，训练 loss 异常低、生成完全不能用。
2. **用 `-1e9` 代替 `-inf`**：大多数时候 softmax 后权重接近 0 但不严格为 0；A05 断言被 mask 的位置权重**恰好**是 0。
3. **把 `d_k` 写成 `d_model`**：MHA 里缩放必须用每个 head 的 `D=C/H`；用 `C` 会让分数尺度偏小。

**对应测试**：4（3D 与手写公式一致，`d_v≠d_k`）、5（mask 语义、2D mask 广播）、6（4D 与逐头 3D 循环一致）；官方快照 `test_scaled_dot_product_attention`（`E36`）、`test_4d_scaled_dot_product_attention`（`E37`）。

### 3.7 RotaryPositionalEmbedding（RoPE）

**公式**：把 `d_k` 维向量按**相邻两维**配对 `(x_0,x_1),(x_2,x_3),…`，第 `k` 对在位置 `pos` 旋转角度 `pos·θ^{-2k/d_k}`：

```
[x'_{2k}  ]   [ cos θ  -sin θ ] [ x_{2k}   ]
[x'_{2k+1}] = [ sin θ   cos θ ] [ x_{2k+1} ]
```

**直觉**：位置信息编码在旋转角里；`<R_i q, R_j k>` 只依赖相对位移 `i-j`，所以同一段序列整体平移不改变注意力分数。只转 q/k，不转 v；没有可学习参数。

```python
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int,
                 device=None) -> None:
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError(f"RoPE 要求 d_k 为偶数（交错配对），得到 {d_k}")
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        k = torch.arange(0, d_k, 2, dtype=torch.float32, device=device)
        inv_freq = theta ** (-k / d_k)  # (d_k/2,)
        positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)
        angles = positions[:, None] * inv_freq[None, :]  # (max_seq_len, d_k/2)

        self.register_buffer("cos_cached", torch.cos(angles), persistent=False)
        self.register_buffer("sin_cached", torch.sin(angles), persistent=False)

    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        if x.shape[-1] != self.d_k:
            raise ValueError(f"输入最后一维应为 d_k={self.d_k}，得到 {x.shape[-1]}")
        idx = token_positions.to(device=self.cos_cached.device, dtype=torch.long)
        cos = self.cos_cached[idx].to(x.dtype)  # (..., T, d_k/2)
        sin = self.sin_cached[idx].to(x.dtype)

        x_even = x[..., 0::2]  # (..., T, d_k/2)
        x_odd = x[..., 1::2]
        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos
        return torch.stack((out_even, out_odd), dim=-1).flatten(-2)
```

**三个坑**：

1. **配对方式写成前后半**（`rotate_half`）：那是 HF/Llama 的通道约定；A1 快照是交错配对。两者只差一个固定的通道置换，但官方权重是按交错顺序训练的，直接换会偏。
2. **`d_k` 用了 `C`**：MHA 里每个 head 的 `d_k = C/H`；RoPE 表只有 `d_k/2` 列，用 `C` 会在拆 head 后形状不匹配，或把不同 head 的通道混在一起转。
3. **位置张量的维度对不上**：MHA 传进来的是 `(B,H,T,D)` 的 q/k，`(B,T)` 的位置要先在 head 维处插 1 变成 `(B,1,T)`。现在 MHA 在 `token_positions.ndim == x.ndim - 1` 时会自动 `unsqueeze(-2)`，所以 `(T,)`、`(1,T)`、`(B,T)`、`(1,1,T)` 都能用；但如果写成 `(B,H,T)`，对 3D 输入不是受支持形状（`H == num_heads` 可能恰好广播，`H != num_heads` 会报错），不要依赖。4D+ 输入若要逐 batch 位置，请显式补成 `(B1,B2,1,T)`。

**对应测试**：8（二维手算旋转、位置 0 恒等）、9（相对性：只依赖 `i-j`、整体平移不变）、21（只作用于 q/k）、22（位置平移后 attention 输出不变）；官方快照 `test_rope`（`E35`，`d_k=64`）与 `test_multihead_self_attention_with_rope`（`E39`，`d_k=16`）。

### 3.8 CausalMultiHeadSelfAttention

**公式**：`MultiHead(x) = W_O · Concat(head_1,…,head_H)`，`head_i = Attention(W_Q x, W_K x, W_V x)`；每个 head 的 `D=C/H`，缩放用 `√D`；内置 causal mask。

**直觉**：把 `C` 切成 `H` 份并行做 attention，head 之间互不交互，最后 `output_proj` 混合。head 维当 batch，所以一次 matmul 就能算完所有头。

```python
    def _split_heads(self, x: Tensor) -> Tensor:
        *leading, seq_len, _ = x.shape
        return x.reshape(*leading, seq_len, self.num_heads, self.d_k).transpose(-3, -2)

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        seq_len = x.shape[-2]
        q = self._split_heads(self.q_proj(x))  # (..., H, T, D)
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)
            elif token_positions.ndim == x.ndim - 1:
                # (..., T) 的 leading 维与 x 的 batch 维一一对应；q/k 多一个 head 维，
                # 在 head 维处插 1 变成 (..., 1, T)，才能与 (..., H, T, D) 广播
                token_positions = token_positions.unsqueeze(-2)
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device)
        )
        attn_out = scaled_dot_product_attention(q, k, v, causal_mask)

        merged = attn_out.transpose(-3, -2).reshape(*x.shape)
        return self.output_proj(merged)
```

**三个坑**：

1. **缩放用 `√C` 而不是 `√D`**：`D=C/H`，两者差 `√H` 倍；分数尺度变了，parity 会偏，训练也会更难。
2. **head 拆分顺序与官方权重不一致**：`view(B,T,H,D)` 的语义是“第 h 个 head 占最后维 `[h*D:(h+1)*D]`”；若先 `split` 到 `(B,H,T,D)` 再按别的顺序拼回，权重语义就变了。
3. **causal mask 方向/位置错**：必须 `tril`（下三角 True）并作用在 `(T,T)` 的 scores 上；用 `triu` 或加到错误维度，会让模型看到未来或整行全遮。

**对应测试**：15（输出形状、任意前导维、head 不整除抛错）、19（改未来 token 不影响过去）、20（抓出 MHA 内部真实 mask，断言上三角权重为 0）、23（`num_heads=1` 与手写单头一致）、C27（`token_positions` 的 `(B,T)` 逐样本回归，新增、不计入 43 项）；官方快照 `test_multihead_self_attention`（`E38`）、`test_multihead_self_attention_with_rope`（`E39`）。

### 3.9 TransformerBlock

**公式**：`y = x + attn(rms1(x))`，`out = y + ffn(rms2(y))`；pre-norm，残差流不归一化；最后 LM 再套 `ln_final`。

**直觉**：归一化放在子层输入侧，残差流从 embedding 到输出一路不被 norm 打断，梯度可以直接回传；代价是输出尺度可能偏大，所以 LM 末尾补一个 final norm。

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 rope: RotaryPositionalEmbedding | None = None,
                 device=None, dtype=None) -> None:
        super().__init__()
        self.attn = CausalMultiHeadSelfAttention(
            d_model, num_heads, rope=rope, device=device, dtype=dtype
        )
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        y = x + self.attn(self.ln1(x), token_positions)
        return y + self.ffn(self.ln2(y))
```

**三个坑**：

1. **写成 post-norm**：`ln(x + f(x))` 是原始 Transformer 的顺序；本 lab 契约是 pre-norm，残差加在 norm **外面**。
2. **两个子层共用一个 RMSNorm**：`ln1` 和 `ln2` 是两组独立 gain，共享会让 `state_dict` 少键、parity 失败。
3. **残差加错对象**：是 `x + attn(...)`，不是 `attn(...)` 自己；`y` 再进第二个子层，`out = y + ffn(ln2(y))`。

**对应测试**：16（输出形状不变，有无 RoPE 都测）、26（反传后所有参数 grad 非 None 且有限）；官方快照 `test_transformer_block`（`E41`，权重去掉 `layers.0.` 前缀）；端到端 `F43`。

### 3.10 TransformerLM

**公式**：`logits = lm_head(ln_final(block_N(…block_1(token_embeddings(ids)))))`；`T <= context_length`；默认 untied；所有层共享一个 RoPE 实例。

**直觉**：embedding 把离散 ID 变成向量，N 个 block 做上下文混合，final norm + lm_head 把隐藏向量映射回词表分布。RoPE 的 cos/sin 是常数，没必要每层各算一份。

```python
class TransformerLM(nn.Module):
    def __init__(self, vocab_size, context_length, d_model, num_layers,
                 num_heads, d_ff, rope_theta: float = 10000.0,
                 tie_weights: bool = False, device=None, dtype=None) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} 必须能被 num_heads={num_heads} 整除")
        self.vocab_size = vocab_size
        self.context_length = context_length
        # ...（省略 num_layers/num_heads 等属性赋值，完整见源码）
        self.rope = RotaryPositionalEmbedding(
            theta=rope_theta, d_k=d_model // num_heads,
            max_seq_len=context_length, device=device,
        )
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff,
                             rope=self.rope, device=device, dtype=dtype)
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

        if tie_weights:
            self.lm_head.weight = self.token_embeddings.weight

    def forward(self, token_ids: Tensor) -> Tensor:
        seq_len = token_ids.shape[-1]
        if seq_len > self.context_length:
            raise ValueError(
                f"输入序列长度 {seq_len} 超过 context_length={self.context_length}"
            )
        x = self.token_embeddings(token_ids)  # (B,T,C)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)  # (B,T,C)
        return self.lm_head(x)  # (B,T,V)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
```

**三个坑**：

1. **漏掉 `ln_final`**：直接 `lm_head(x)` 也能跑，但 pre-norm 的残差流尺度偏大，parity 会偏、训练更不稳。
2. **默认 tie 了**：官方 fixture 的 `lm_head.weight` 与 `token_embeddings.weight` 不相等；默认必须 untied，tie 只走单独测试（C25）。
3. **每层各自 new 一个 RoPE**：参数不变，但每层位置表可能不一致，且浪费内存；契约是共享一个实例。另外 `T > context_length` 必须在 LM 层报错，而不是让 RoPE 索引越界。

---

## 4. 组装与完整数据流

### 4.1 一次前向的数据流

把第 3 章的零件按 `TransformerLM.forward` 串起来，形状只在三个位置变化：

```text
ids (B,T) int64
  │
  ├─ Embedding ──────────────────────────────► x (B,T,C)
  │
  ├─ block 1 ──┐
  ├─ block 2   │  每个 block 内部：
  ├─ ...       │    y = x + attn(rms1(x))       (B,T,C) → (B,T,C)
  └─ block N ──┘    out = y + ffn(rms2(y))      (B,T,C) → (B,T,C)
  │
  ├─ ln_final ───────────────────────────────► (B,T,C)
  │
  └─ lm_head (V,C) ──────────────────────────► logits (B,T,V)
                                                  │
                                  训练时：shift 后接 cross-entropy → 标量 ()
```

关键点：**block 不改变形状**；`(B,T,C)` 只在 `lm_head` 处变成 `(B,T,V)`；`(B,H,T,T)` 的注意力分数只是中间量，不会出现在 `forward` 的返回值里。

### 4.2 残差流：为什么是 `x + f(rms(x))`

pre-norm 的直觉可以拆成三句话：

1. **残差流是一条“干净”的通道**：从 embedding 到最后一层，`x` 只做加法，不被 norm 或子层改写。梯度可以沿这条通道直接回传，深层模型更容易训。
2. **归一化只作用于“进入子层之前”的副本**：`rms(x)` 是给 attention/FFN 看的，不改变残差流本身；post-norm 把 norm 放在残差相加之后，等于每层都重新缩放残差流，梯度路径更长。
3. **代价是输出尺度可能偏大**：所以 LM 在 block 之后、`lm_head` 之前再放一个 `ln_final`，把最终表示拉回稳定尺度。

这也是为什么“漏掉 `ln_final`”不会让代码报错，却会让 parity 偏、训练更抖——它破坏的是数值尺度约定，不是形状。

### 4.3 装配顺序与 `state_dict`

`TransformerLM.__init__` 的装配顺序是：

```text
RoPE（共享实例，无参数）
  → token_embeddings
  → layers[0..N-1]（每层 attn / ln1 / ln2 / ffn）
  → ln_final
  → lm_head
  → （可选）tie_weights：lm_head.weight = token_embeddings.weight
```

顺序本身不影响前向，但**模块属性名决定 `state_dict` 键名**，而键名是 parity 的硬契约（`SPEC.md` §3）。本 lab 实测：

| 检查 | 数字 | 来源 |
| --- | --- | --- |
| 官方 `model.pt` 参数总数 | 1,403,328 | `results/accounting.json` 的 `checks.official_fixture`（`actual_params`，与解析值 `diff=0`） |
| 官方 `state_dict` 条目数 | 30 | 同上（`num_state_dict_entries`） |
| 小 LM（V=256, C=64, N=2, d_ff=128）参数 | 115,008 | `checks.toy_transformer`；`sum(p.numel())` 与 `num_parameters()` 一致 |
| 自检小模型（V=257, C=32, N=2, d_ff=64）参数 | 37,088 | `.venv/bin/python model_lab/toy_transformer.py` 输出 |
| 自检 `state_dict` 键数 | 21 | 同上（不含 RoPE 的非持久 buffer） |

装载官方权重的做法：

```python
sd = {k.replace("_orig_mod.", ""): v for k, v in torch.load(
    f"{A1_REPO}/tests/fixtures/ts_tests/model.pt",
    map_location="cpu", weights_only=True).items()}
model.load_state_dict(sd, strict=True)   # 缺失/多余键都会报错
```

用 `strict=True` 是故意的：只要多一个 `bias`、少一个 `ln2`，立刻失败，而不是等到 maxdiff 变大才发现。

### 4.4 三个“能跑但错了”的组装错误

| 错误 | 症状 | 定位 |
| --- | --- | --- |
| 漏 `ln_final` | 形状全对、loss 能降；`test_transformer_lm` maxdiff 明显变大 | 打印 `model.state_dict().keys()`，对比 `ln_final.weight` 是否存在 |
| 默认 tie 了 lm_head | 参数量比官方少 `V·C`；`strict=True` 可能不报错（键都在），但共享存储让两张表互相覆盖，logits 偏 | 检查 `lm_head.weight is token_embeddings.weight`；看 `num_parameters()` 差值 |
| 每层独立 RoPE / 位置表长度错 | 单层 block parity 过、整 LM parity 偏；位置表短于 `context_length` 时越界 | 检查 `self.rope` 是否 `is` 同一个对象；`max_seq_len` 是否等于 `context_length` |

### 4.5 从 logits 到 loss

语言模型的目标是“给定前缀预测下一个 token”。两种等价写法：

```python
# 写法 A：targets 右移一位，logits 全用
targets = torch.roll(ids, shifts=-1, dims=-1)
loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))

# 写法 B：丢掉最后一个位置的 logits（它没有下一个 token 可预测）
loss = F.cross_entropy(logits[:, :-1].reshape(-1, vocab_size), ids[:, 1:].reshape(-1))
```

`test_F43` 用的是写法 A（`torch.roll`），30 步 AdamW（`lr=3e-3`）后 loss 从 4.8600 降到 0.2942（命令 `.venv/bin/python model_lab/test_toy_transformer.py` 的 F43 输出）。注意这只是一个小 LM 在一个固定 batch 上的端到端冒烟测试，不是训练实验。

### 4.6 先跑自检，再跑测试

```bash
.venv/bin/python model_lab/toy_transformer.py
# shape ok: (2, 12, 257)
# num_parameters ok: 37,088 (解析式一致)
# state_dict keys ok: 21 个
# loss: 5.6815（反向传播成功）
# OK: toy_transformer 自检通过
```

自检覆盖“形状 + 参数量 + 键名 + 反向传播”四件事，约 0.9 秒。它通过之后再跑完整测试，能省很多“其实是装配错”的 debug 时间。

---

## 5. 怎么证明写对了

### 5.1 四层验证金字塔

| 层级 | 方法 | 能发现什么 | 不能发现什么 |
| --- | --- | --- | --- |
| 1. 手算 | 2×2 softmax、2 维 RoPE、2×2 attention | 公式符号、旋转方向、mask 方向 | 大规模形状/广播 |
| 2. PyTorch 独立参考 | `F.softmax` / `F.silu` / 显式 `torch.matmul` 重写 | 实现与框架语义的差异 | 与官方“特定权重布局”的差异 |
| 3. 官方快照 | 13 个 `.npz` + 固定 seed | 数值契约、`state_dict` 布局、per-head 细节 | 初始化（权重被覆盖） |
| 4. 性质测试 | causal、RoPE 相对性、尺度不变、tie、反传 | 语义性质与边界条件 | 绝对数值正确性 |

结论：**快照 parity 是最强的一层，但它不覆盖初始化**；性质测试覆盖语义，但不证明“和官方逐位一致”。四层都要有。

### 5.2 手算哪些、怎么手算

- **softmax**：取 `[1,2,3]`，手算 `exp(0),exp(1),exp(2)` 归一化；再对整体 `+1e4` 算一遍，确认结果不变。
- **RoPE**：取 `d_k=2`、位置 1，角度 `θ=1`，手算 `[x0 cos1 - x1 sin1, x0 sin1 + x1 cos1]`；再取位置 0，确认恒等。
- **attention**：取 `T=2`、`d_k=1`，手算 2×2 分数、softmax、加权和；把 mask 设成只有对角线可见，确认输出等于 `v` 本身。

手算只验证“符号和方向”，不追求覆盖所有形状；形状和广播交给第 2 章的表与性质测试。

### 5.3 与 PyTorch 对照

- `softmax` ↔ `torch.softmax`（A01/A02 都对照过）。
- `silu` ↔ `F.silu`（A10）。
- `swiglu` ↔ 用 `F.silu` 手写的 `w2(F.silu(w1(x)) * w3(x))`（A10）。
- attention 用显式 `einsum`/`matmul` 重写（A04/A06），**不要**直接用 `F.scaled_dot_product_attention` 当参考：它的 mask 语义、后端选择和数值路径都可能不同，容易把实现错误掩盖成“后端差异”。

### 5.4 官方快照配方

路径（`A1_REPO` 可覆盖，默认 `/Users/yunni/Projects/assignment1-basics`，只读）：

- 权重：`tests/fixtures/ts_tests/model.pt`（键带 `_orig_mod.` 前缀，装载前 strip）
- 配置：`tests/fixtures/ts_tests/model_config.json`
- 快照：`tests/_snapshots/{test_name}.npz`，数组键是 `array`

固定输入（与官方 `conftest.py` 的 fixture 逐位一致，每个单独 seed）：

| 变量 | seed | 形状 |
| --- | --- | --- |
| `in_embeddings` / `x` | 4 | `(4,12,64)` |
| `q` | 1 | `(4,12,64)` |
| `k` | 2 | `(4,16,64)` |
| `v` | 3 | `(4,16,64)` |
| `mask` | 5 | `(4,12,16)`，`randn > 0.5` |
| `idx` | 6 | `(4,12)`，`randint(0,10000)` |

几何配置：`vocab=10000, ctx=16, d_model=64, num_layers=3, num_heads=4, d_ff=128, theta=10000`，`d_head=16`。

> 注意：`mask` 是 3D；4D SDPA 测试会把 `q,k,v` 与 `mask` 一起按 `head=2` 重排成 `(2,2,12,64)/(2,2,16,64)/(2,2,12,16)`（源码 `test_4d_scaled_dot_product_attention` 的 `rearrange`）。

跳过机制：找不到官方仓库或快照时，parity 打印 `SKIP` 并继续；**全 SKIP 不构成验证**，必须设置 `A1_REPO` 后重跑。

### 5.5 测试运行方式与覆盖

```bash
.venv/bin/python model_lab/test_toy_transformer.py
# 汇总: 45 passed, 0 failed, 0 skipped
# 耗时: 1.23s，线程数: 2
```

（2026-11-05 多次复跑为 1.04–1.23 s；2 线程；官方仓库存在，因此无 SKIP。）

45 个测试函数与 43 项的关系：

| 组 | 测试函数数 | 对应 SPEC §6 | 说明 |
| --- | --- | --- | --- |
| A 手算/公式 | 10 | 1–10 | softmax、sdpa、rmsnorm、rope、silu/swiglu |
| B 形状/接口 | 8 | 11–18 | 每个模块的 in/out、d_ff、越界报错 |
| C 性质 | 8 | 19–26 | causal、RoPE、尺度、tie、反传 |
| C27 回归 | 1 | 不计入 43 项 | MHA 的 `token_positions` 支持 `(B,T)`（新增回归） |
| D 初始化 | 4 | 27–30 | std/截断、gain、键集合 |
| E parity | 13 | 31–42 | 13 个快照；`swiglu`+`pwff` 算一项，LM 全量+截断算一项 |
| F 端到端 | 1 | 43 | 小 LM 30 步 loss 下降 |
| 合计 | 45 | 43 项 + 1 回归 | 44 个原有函数覆盖 43 项，加 1 个 C27 |

所以是 **45 个函数 = 44 个原有函数（覆盖 43 项）+ 1 个 C27 回归**；C27 不计入 43 项，不是“多测了两项”。C27 还验证 `(B,T)` 是逐样本生效；Lead 复核它与逐样本 `(1,T)` 的 maxdiff 为 `9.5e-7`。

### 5.6 实测 maxdiff 表

命令：`.venv/bin/python model_lab/test_toy_transformer.py`（2026-11-05，torch 2.14.1，2 线程，`A1_REPO=/Users/yunni/Projects/assignment1-basics`）。下表 13 行全部来自该命令的 stdout。

| # | 快照 | maxdiff | 官方容差 | 本 lab 断言容差 |
| --- | --- | --- | --- | --- |
| 31 | `test_linear` | 8.941e-07 | `rtol=1e-4, atol=1e-2`（默认） | `atol=1e-4` |
| 32 | `test_embedding` | 0.000e+00 | `rtol=1e-4, atol=1e-2`（默认） | `atol=1e-4` |
| 33 | `test_swiglu` | 1.431e-06 | `atol=1e-5` | `atol=1e-5` |
| 34 | `test_rmsnorm` | 1.252e-06 | `atol=1e-4` | `atol=1e-4` |
| 35 | `test_rope` | 1.475e-06 | `atol=1e-5` | `atol=1e-5` |
| 36 | `test_scaled_dot_product_attention` | 9.537e-07 | `atol=1e-5` | `atol=1e-5` |
| 37 | `test_4d_scaled_dot_product_attention` | 9.537e-07 | `atol=1e-5` | `atol=1e-5` |
| 38 | `test_multihead_self_attention` | 5.364e-07 | `atol=1e-5` | `atol=1e-5` |
| 39 | `test_multihead_self_attention_with_rope` | 2.980e-07 | `atol=1e-5` | `atol=1e-5` |
| 40 | `test_positionwise_feedforward` | 1.431e-06 | 官方无同名测试（遗留快照） | `atol=1e-5` |
| 41 | `test_transformer_block` | 2.384e-06 | `atol=1e-4` | `atol=1e-4` |
| 42a | `test_transformer_lm` | 1.001e-05 | `atol=1e-4, rtol=1e-2` | `atol=1e-4, rtol=1e-2` |
| 42b | `test_transformer_lm_truncated_input` | 1.049e-05 | `atol=1e-4` | `atol=1e-4` |

**怎么读这张表**：

1. 全部 13 项 ≤ `1.049e-05`，比本 lab 自设的 `1e-4` 门槛低约一个量级；worst 是截断输入的 LM。
2. **官方容差和本 lab 容差不是一回事**：官方 `NumpySnapshot.assert_match` 默认 `rtol=1e-4, atol=1e-2`，`linear`/`embedding` 没有覆盖默认值；本 lab 对这两项用了更紧的 `atol=1e-4`。其余项目本 lab 与官方一致（`sdpa`/`mha`/`rope`/`swiglu` 用 `atol=1e-5`；`rmsnorm`/`block`/`lm` 用 `atol=1e-4`；LM 另给 `rtol=1e-2`）。
3. **官方容差只写显式覆盖的 `atol`**；未写 `rtol` 的沿用默认 `rtol=1e-4`（LM 全量为 `1e-2`）。本 lab 的 `_assert_close` 默认也是 `rtol=1e-4`。
4. **Lead 独立复查**：用 `model_lab/lead_spotcheck.py`（不 import 本 lab 测试，自己构造模块并复算）抽查 LM 的 maxdiff 为 `1.001e-05`，与 impl 报告一致；13 项全部通过。这是“第二个人用不同代码路径复核”的证据，不是同一份测试的自证。
5. `test_embedding` 的 maxdiff 恰好为 `0` 是正常的：查表是纯索引，没有浮点运算。
6. `test_positionwise_feedforward` 与 `test_swiglu` 同权重同公式，数值也应相同（都是 `1.431e-06`）；它没有官方同名测试函数，本 lab 仍跑，作为额外对照。
7. 新增的 `C27` 只验证 MHA 的位置形状，不改变上面 13 个 parity 数值；修复后复跑逐位相同（同一命令）。

### 5.7 症状 → 病因 → 定位

| 症状 | 最可能的原因 | 定位方法 |
| --- | --- | --- |
| LM maxdiff ~1e-2，单层组件都正常 | mask 方向反了、`ln_final` 漏了、RoPE 只给第一层加了 | 先跑 `E41`（单 block）；再对比 `E42a` 与 `E42b`，看误差是否随层数放大 |
| 所有 parity 都偏、量级 ~1e-1 | `d_k` 用了 `C`、缩放用了 `√C`、RMSNorm 少 `mean` | 检查 `MHA.d_k`、`sqrt(d_k)`、RMSNorm 公式；用 A07/A10 手算对照 |
| RoPE 相关项偏、其他正常 | 交错配对写成前后半、位置表索引错、`token_positions` 没插 head 维 | 直接跑 `E35`（`d_k=64`）和 `E39`（`d_k=16`）；再跑 A08/A09 |
| `load_state_dict(strict=True)` 失败 | 多/少 `bias`、模块属性名不对、键拼写不一致 | 打印键集合差集；对照 `SPEC.md` §3 |
| parity 全过但初始化测试失败 | 初始化写错（被 `load_state_dict` 掩盖） | 跑 D27–D29，看 std、截断范围、gain |
| 全 SKIP | 官方仓库路径不对 / 权限 | 设 `A1_REPO`；确认 `tests/_snapshots/*.npz` 可读 |

### 5.8 一个容易自欺的点

“13 个快照全过”不等于“实现全对”：它不覆盖初始化、不覆盖 `tie_weights`、不覆盖 `T > context_length` 的报错、不覆盖反传。反过来说，如果快照不过，也不要靠调大容差“修好”——先按 §5.7 定位，再重跑。容差是判据，不是调试工具。

---

## 6. 资源核算

> 本章所有结果来自 `model_lab/results/accounting.json`（命令 `.venv/bin/python model_lab/accounting.py --strict`，2026-11-05）。脚本只做解析式核算，不实例化大模型；`runtime` 字段随每次运行小幅波动（实测 `elapsed_s` ≈0.006，2 线程，`peak_rss_mib` ≈212.5）。

### 6.1 `2mnp` 规则

对 `A ∈ R^{m×n}`、`B ∈ R^{n×p}`，矩阵乘 `AB` 需要 `2mnp` FLOPs：每个输出元素是 `n` 次乘加（`2n` FLOPs），共 `m·p` 个输出。这个规则忽略了加法、转置、softmax、norm 等非 matmul 运算。

### 6.2 参数量推导

| 组件 | 每个的数量 | 说明 |
| --- | --- | --- |
| `token_embeddings` | `V·d` | `(V,d)` 查表 |
| 每层 attention | `4d²` | q/k/v/output 四个 `(d,d)` |
| 每层 FFN | `3·d·d_ff` | w1/w3 `(d_ff,d)` + w2 `(d,d_ff)` |
| 每层 RMSNorm | `2d` | ln1 + ln2 的 gain |
| `ln_final` | `d` | |
| `lm_head` | `V·d` | untied 时独立 |

untied 总参数量：

```
params = 2·V·d + N·(4·d² + 3·d·d_ff + 2·d) + d
```

tied 时最后一个 `V·d` 与 `token_embeddings` 共享，总数减去 `V·d`。

**对账（`checks` 字段，均来自 `results/accounting.json`）**：

- 官方 fixture（`V=10000, d=64, N=3, d_ff=128`）：实际 1,403,328 = 解析 1,403,328，`diff=0`；`state_dict` 30 个条目、带 `_orig_mod.` 前缀。
- 小 LM（`V=256, d=64, N=2, d_ff=128`）：`sum(p.numel())` = 115,008 = `num_parameters()`，`diff=0`。
- 自检模型（`V=257, d=32, N=2, d_ff=64`）：37,088（`.venv/bin/python model_lab/toy_transformer.py` 输出），与手算解析式一致。

### 6.3 前向 FLOPs 推导

一次前向、序列长度 `T`、batch 维不进入公式（`2mnp` 只算单样本；batch 乘上去即可）：

| 部分 | 每层 FLOPs | 推导 |
| --- | --- | --- |
| q/k/v/output 投影 | `8T d²` | 四个 `2·T·d·d` |
| attention 分数 + 加权 | `4T² d` | `2·T·T·d`（QKᵀ）+ `2·T·T·d`（权重·V） |
| FFN（SwiGLU） | `6T d d_ff` | w1/w3 各 `2T d d_ff`，w2 `2T d_ff d` |
| 每层合计 | `8Td² + 4T²d + 6T d d_ff` | |
| `lm_head` | `2T d V` | 只在最后算一次 |
| embedding 查表 | 0 | 纯索引，不算 matmul |

**没算进去的**：softmax、RMSNorm、SiLU、残差、mask、逐元素乘。它们相对 matmul 很小，但不是 0；解析式与 profiler 的差异主要来自这里。

**训练 FLOPs**：按常见近似，反向约为前向的 2 倍，所以训练一步约 `3×` 前向 FLOPs（handout §4.3 的 MFU 题也用了这个假设）。本 lab 只核前向，不把 3× 写进 JSON。

### 6.4 显存口径与 sharding

`accounting.py` 只算**推理时 fp32 权重**：

```
memory = 4 bytes × params
```

这**不是**训练显存。全 fp32 + AdamW 训练时，每个参数至少还要：梯度 4 B、一阶矩 4 B、二阶矩 4 B，加上参数本身 4 B，共约 **16 B/param**（解析下界，不含激活、临时 buffer、通信）。按 GPT-2 XL 的 1,640,452,800 参数：

```
1,640,452,800 × 16 B = 26,247,244,800 B ≈ 26.25 GB（24.44 GiB）
```

若把优化器状态按数据并行/ZeRO 分片到 `k` 张卡，每卡约 `16/k B/param`（忽略激活与通信）；`k=8` 时约为 `3.28 GB/卡`。这是**口径推导**，不是本 lab 的实测；本 lab 没跑多卡，也没有训练大模型。

> 结论：看到“GPT-2 XL 只要 6.1 GiB”时，要分清是**推理权重**还是**训练状态**；两者差 4 倍以上，且训练还有激活。

### 6.5 GPT-2 四配置 + 长上下文

来源：`results/accounting.json` 的 `configs.*`。`d_ff` 按 8/3·d 取最近 64 倍数；FLOPs 是单样本、`T=context_length` 的前向值。

| 配置 | 参数量 | fp32 权重 (MiB) | 前向 FLOPs | 投影 | attention | FFN | lm_head |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GPT-2 small（12 层 / 768 / 12 头 / ctx 1024） | 162,148,608 | 618.5 | 2.916e11 | 19.9% | 13.3% | 39.8% | 27.1% |
| GPT-2 medium（24 / 1024 / 16 / 1024） | 406,539,264 | 1550.8 | 8.302e11 | 24.8% | 12.4% | 50.1% | 12.7% |
| GPT-2 large（36 / 1280 / 20 / 1024） | 833,591,040 | 3179.9 | 1.769e12 | 27.3% | 10.9% | 54.3% | 7.4% |
| GPT-2 XL（48 / 1600 / 25 / 1024） | 1,640,452,800 | 6257.8 | 3.517e12 | 28.6% | 9.2% | 57.5% | 4.7% |
| GPT-2 XL，ctx 16384 | 1,640,452,800 | 6257.8 | 1.336e14 | 12.1% | 61.7% | 24.2% | 2.0% |

**读表**：

- 从 small 到 XL：参数量约 10.1×，fp32 权重约 10.1×，前向 FLOPs 约 12.1×；FFN 的 FLOPs 占比从 39.8% 升到 57.5%，`lm_head` 从 27.1% 降到 4.7%。
- ctx 1024→16384（16× 长度）：attention 项按 `T²` 增长 256×，投影/FFN/lm_head 只按 `T` 增长 16×；总 FLOPs 变成 37.98×，attention 占比从 9.16% 升到 61.73%。
- 这是“长上下文为什么贵”的最直接账本：**不是参数变多，是 `T²` 项变成主项**。

### 6.6 `d_ff` 取整

来源：`checks.d_ff_rule`（`expected` 与 `actual` 完全一致，且全为 64 的倍数）。

| `d_model` | 8/3·d | 最近 64 倍数 |
| ---: | ---: | ---: |
| 512 | 1365.33 | 1344 |
| 768 | 2048.00 | 2048 |
| 1024 | 2730.67 | 2752 |
| 1280 | 3413.33 | 3392 |
| 1600 | 4266.67 | 4288 |

`1600→4288` 与 handout §3.5 给出的 GPT-2 XL `d_ff` 一致；这也是 `checks.d_ff_rule.handout_reference` 的对照点。

### 6.7 结论与局限

- 解析式核算能让你**不跑模型**就回答“多少参数、多少显存、多少 FLOPs”，并且能和官方权重总数对上（`diff=0`）。
- 它**不能**替代 profiler：真实耗时受 kernel 融合、内存带宽、cache、softmax/norm 等非 matmul 运算影响；本 lab 的 FLOPs 是“矩阵乘下界”，不是 wall-clock。
- 它**不包含**训练显存（优化器状态、激活、梯度）和通信；§6.4 的 16 B/param 只是解析下界。
- 所有数字都可以用 `accounting.py --strict` 重跑；任一 check 失败会 exit 1。

---

## 7. 过拟合实验

> 本章数字来自 `model_lab/results/overfit.json`（命令 `.venv/bin/python model_lab/exp_overfit.py`，2026-11-05，2 线程）。实验只回答“实现能不能训”，不回答“模型好不好”。

### 7.1 设置

| 项 | 值 | 来源 |
| --- | --- | --- |
| 数据 | `tinystories_sample.txt`（3794 字节，只读） | `overfit.json` 的 `data.path` / `data.bytes` |
| tokenizer | 字节级，`vocab=256` | `overfit.json` 的 `config.vocab_size` |
| 模型 | `d_model=64, num_layers=2, num_heads=4, d_ff=128, ctx=64`，untied，115,008 参数 | `config` / `num_parameters` |
| batch | 固定 `B=8, T=64`（每步 512 token） | `data.batch_shape` / `tokens_per_step` |
| 优化器 | AdamW，`lr=3e-3`，`weight_decay=0.0` | `config` |
| 预算 | 最多 300 步或 90 秒硬停 | `config.max_steps` / `max_seconds` |
| 记录 | 每 10 步在固定 batch 上算 eval loss；结束后 greedy 生成 64 字节 | `curve` / `sample_bytes` |

### 7.2 loss 曲线（每 10 步采样）

来源：`overfit.json` 的 `curve`；下表把 31 个点两两并排，保留 4 位小数。

| step | loss | step | loss |
| ---: | ---: | ---: | ---: |
| 0 | 5.6737 | 10 | 2.8142 |
| 20 | 1.7734 | 30 | 0.8259 |
| 40 | 0.2056 | 50 | 0.0496 |
| 60 | 0.0216 | 70 | 0.0137 |
| 80 | 0.0107 | 90 | 0.0091 |
| 100 | 0.0082 | 110 | 0.0075 |
| 120 | 0.0070 | 130 | 0.0066 |
| 140 | 0.0062 | 150 | 0.0059 |
| 160 | 0.0057 | 170 | 0.0055 |
| 180 | 0.0053 | 190 | 0.0051 |
| 200 | 0.0049 | 210 | 0.0048 |
| 220 | 0.0047 | 230 | 0.0045 |
| 240 | 0.0044 | 250 | 0.0043 |
| 260 | 0.0043 | 270 | 0.0042 |
| 280 | 0.0041 | 290 | 0.0040 |
| 300 | 0.0040 | | |

形态：前 50 步从 5.67 快速降到 0.05，之后进入长尾；300 步时 `final_loss=0.003972`（JSON 原始值，不是表中四舍五入后的 0.0040）。这个“先陡后平”的形态是单 batch 记忆的典型曲线，不是泛化曲线。

### 7.3 吞吐、内存与样例

| 指标 | 值 | 来源 |
| --- | --- | --- |
| 总耗时 | 1.7172 s | `overfit.json` 的 `elapsed_s` |
| 纯训练耗时 | 1.2927 s | `train_elapsed_s` |
| 吞吐 | 118,818.7 tokens/s | `tokens_per_s` |
| 线程数 | 2 | `threads` |
| 峰值 RSS | 316.1 MiB | `peak_rss_mib` |
| `initial_loss` → `final_loss` | 5.673745 → 0.003972 | 同名字段 |
| `checks` | 9 项全 true（finite、final<initial、≤120 s、2 线程、steps≥1、sample 非空、final<0.5、未超时、参数量一致） | `checks` |
| 生成耗时 | 0.0227 s | `meta.generation_elapsed_s` |

greedy 样例（prompt = `"\nOnce up"`，续写 64 字节）：

```text
\nOnce upon a time there was a little boy named Ben. Ben loved to ame to 
```

对应 `sample` 字段是 `"on a time there was a little boy named Ben. Ben loved to ame to "`（前面接上 prompt 的 `up` 就是 `upon`）。它看起来“像故事开头”，是因为训练数据只有这一小段 TinyStories，模型在**记忆**这个固定 batch；不要把它当作语言能力证据。生成在 64 字节处截断，`ame` 只是这段输出的一部分，不必过度解读。

### 7.4 结论

- **能训**：300 步内 loss 单调下降（`checks.all_losses_finite`、`final_loss_below_initial` 均 true），说明前向、反向、优化器接口、`state_dict` 都接对了。
- **梯度通路完整**：C26 另外验证了所有参数 `grad` 非 None 且有限；本实验的 loss 下降是端到端证据。
- **达到目标**：`final_loss=0.003972 < 0.5`，`checks.final_loss_below_0_5_target=true`。
- **不能说明**：泛化、语言能力、超参优劣、真实数据上的收敛速度；单 batch、字节级、玩具模型都不支持这些外推。

### 7.5 局限

1. **单 batch**：模型只见过 512 个 token 的固定窗口，loss 低是记忆，不是学习。
2. **字节级 + 小词表**：没有 BPE，序列长、语义粒度粗；结论不能直接搬到真实 tokenizer。
3. **玩具规模**：2 层、64 维；吞吐/内存数字不能外推到真实模型。
4. **CPU 2 线程**：`118,818.7 tokens/s` 只描述这个环境，不是硬件上限。
5. **判定标准固定**：`final_loss<0.5` 是目标，不是调参后的结论；达不到就如实记录。

### 7.6 复现与失败处理

```bash
.venv/bin/python model_lab/exp_overfit.py --max-seconds 90
```

- 超时：`--max-seconds` 是硬停，已跑的曲线会保存，`checks.timed_out` 会标 true。
- 数据缺失：自动回退到内置合成文本，`data.fallback_reason` 会写明原因；此时数字只能和同回退条件的结果比。
- 线程数不是 2：`checks.two_threads` 会失败；先确认脚本开头的 `torch.set_num_threads(2)`。
- 想换配置：`--max-steps`、`--lr`、`--batch-size`、`--seq-len`、`--seed` 都可调，但**改完要重新落 JSON**，不能把新数字写回旧文件。

---

## 8. 与 A1 和 training_lab 的关系

### 8.1 A1 §3.3–3.5 题目映射

| A1 题目 | handout | 分值 | 本 lab 对应 | 本 lab 的验证 |
| --- | --- | ---: | --- | --- |
| `linear` | §3.3.2 | 1 | `Linear`（3.1） | B11、D27、E31 |
| `embedding` | §3.3.3 | 1 | `Embedding`（3.2） | B12、D28、E32 |
| `rmsnorm` | §3.4.1 | 1 | `RMSNorm`（3.3） | A07、B13、C24、E34 |
| `positionwise_feedforward` | §3.4.2 | 2 | `SwiGLU`（3.4） | A10、B14、E33、E40 |
| `rope` | §3.4.3 | 2 | `RotaryPositionalEmbedding`（3.7） | A08、A09、C21、C22、E35、E39 |
| `softmax` | §3.4.4 | 1 | `softmax`（3.5） | A01–A03 |
| `scaled_dot_product_attention` | §3.4.4 | 5 | `scaled_dot_product_attention`（3.6） | A04–A06、E36、E37 |
| `multihead_self_attention` | §3.4.5 | 5 | `CausalMultiHeadSelfAttention`（3.8） | B15、C19、C20、C23、E38、E39 |
| `transformer_block` | §3.5 | 3 | `TransformerBlock`（3.9） | B16、C26、E41 |
| `transformer_lm` | §3.5 | 3 | `TransformerLM`（3.10） | B17、B18、C25、D30、E42a/b、F43 |
| `transformer_accounting` | §3.5 | 5 | `accounting.py`（第 6 章） | `results/accounting.json` 的 `checks` |

合计 29 分（handout §3.3–3.5 的分值之和）。本 lab 不覆盖 A1 的 §4–5（训练、优化器、数据加载、checkpoint）以及 BPE 部分。

### 8.2 哪些是 A1 要求，哪些是本 lab 扩展

**A1 要求（必须自己写）**：上述 11 个模块/函数、`state_dict` 键名、无 bias、`trunc_normal` 初始化、RMSNorm fp32、RoPE 交错配对、mask 语义、untied 默认、`T <= context_length`。

**本 lab 扩展（不是 A1 提交内容）**：

- 45 个测试函数（44 个覆盖 43 项 + 1 个 C27 回归；官方 pytest 只有 13 个 model 测试），额外的形状断言、性质测试、初始化测试。
- `toy_transformer.py` 的 `__main__` 自检（形状/参数量/键名/反传一次跑完）。
- `accounting.py` 与 `results/accounting.json`：A1 只要求一段 writeup，本 lab 把它做成可复跑脚本。
- `exp_overfit.py` 与 `results/overfit.json`：A1 §4–5 才涉及训练，本 lab 只做单 batch 冒烟。
- 教学化的接口（类名/参数名、`d_model` 而不是 `d` 等）与错误信息。

### 8.3 学术诚信

`model_lab` 的接口层与官方 `tests/adapters.py` 故意不同，就是为了避免“复制即提交”。如果你在修 CS336 学分：

- 可以读本文和 `toy_transformer.py` 来理解概念、对照形状；
- **不要**把类名/函数体直接搬进 A1 的 `adapters.py`；
- 提交前删掉对 `toy_transformer` 的 import，确保 A1 代码只依赖 `torch`；
- writeup 里引用本文的核算推导时，注明来源。

### 8.4 给 training_lab 的接口

`TransformerLM` 的契约足以直接接到下一阶段：

```python
model = TransformerLM(
    vocab_size=..., context_length=..., d_model=..., num_layers=...,
    num_heads=..., d_ff=..., rope_theta=10000.0, tie_weights=False,
)
logits = model(ids)          # (B,T) -> (B,T,V)
n = model.num_parameters()   # 与 sum(p.numel()) 一致
```

training_lab 需要自己补：数据加载（变长 batch/padding 不在本 lab 范围）、`F.cross_entropy` 的 shift、AdamW/学习率调度/梯度裁剪、checkpoint、评估。本 lab 不提供 KV cache、dropout、GQA、FlashAttention；这些在 `llama_steps/` 或真实训练栈里。

### 8.5 下一步学什么

- **训练侧**：A1 §4–5（cross-entropy、AdamW、学习率调度、梯度裁剪、数据加载、checkpoint）。
- **推理侧**：`llama_steps/` 的 KV cache、GQA、真实权重对齐；再往上是量化、FlashAttention。
- **系统侧**：`accounting.py` 的 FLOPs/显存公式 → 分布式训练（DP/TP/PP/ZeRO）的账本。

---

## 9. 学习计划与自测

### 9.1 五天计划（每天 2–3 小时）

| Day | 读 | 写/跑 | 验收 |
| --- | --- | --- | --- |
| D1 | 第 2 章 + 3.1–3.3 | 自己写 `Linear`/`Embedding`/`RMSNorm`；跑 B11–B13、D27–D29 | 手算 A07 能对上；RMSNorm dtype 测试过 |
| D2 | 3.4、3.7 + handout §3.4.2–3.4.3 | 写 `SwiGLU`/RoPE；跑 B14、A08–A10、C21–C22、E33/E35 | 能说清交错配对和 `rotate_half` 的差别 |
| D3 | 3.5–3.8 + handout §3.4.4–3.4.5 | 写 `softmax`/SDPA/MHA；跑 A01–A06、B15、C19–C20、E36–E39 | mask 语义、3D/4D 广播、`d_k` 缩放都能脱稿 |
| D4 | 3.9–3.10 + 第 4 章 | 写 `Block`/`LM`；装官方权重跑 E41/E42；跑 D30 | `strict=True` 能加载；13 个 maxdiff 打得出来 |
| D5 | 第 5–7 章 | 跑 45 个测试、`accounting.py --strict`、`exp_overfit.py` | 三份结果都能复现；能解释每个数字的来源 |

### 9.2 自测 12 题

<details>
<summary>先自己写答案，再展开对照</summary>

1. **一次前向里，张量形状有哪三种？** `(B,T,C)`（残差流）、`(B,H,T,D)`（每个 head 的 q/k/v 与注意力输出）、`(B,H,T,T)`（注意力分数/权重）；最后 `(B,T,V)`（logits）。`C=H·D`，block 不改变形状。
2. **为什么要先 `view(B,T,H,D)` 再 `transpose(1,2)`？** 前一步把最后一维切成 H 个连续通道（第 h 个 head 占 `[hD:(h+1)D]`）；后一步把 head 提到序列维前，使 attention 变成 `(B,H,·,·)` 的批量运算。合并时反向 `transpose(-3,-2)` 后 `reshape` 回 `(...,T,C)`（非连续时 `reshape` 会自动 copy）。
3. **mask 的 `True` 是什么语义？为什么填 `-inf`？** `True`=可见/信息可流动；`False` 在 softmax 前填 `-inf`，softmax 后权重恰好为 0。用 `-1e9` 只能得到“很小但不为 0”的权重。
4. **3D mask `(4,12,16)` 能直接加到 4D scores `(2,2,12,16)` 吗？** 不能，右对齐会补成 `(1,4,12,16)`，第 `-3` 维 `4≠2`。官方 4D 测试用 `rearrange` 把 mask 也变成 `(2,2,12,16)`；实现只需支持可广播 mask。
5. **RoPE 的交错配对和 HF 的 `rotate_half` 能互换吗？** 数学上只差一个固定的通道置换：同时置换 q/k 的通道和 `W_q/W_k` 的行，分数不变；但官方权重是按交错顺序训练的，直接换会偏。`llama_steps/step03_rope.py` 第 5 节有验证。
6. **RoPE 的 `d_k` 用 `C` 还是 `C/H`？** 用 `C/H`（每个 head 的维度）。角度表是 `(max_seq_len, d_k/2)`；用 `C` 会把不同 head 的通道混在一起。
7. **为什么 `<R_i q, R_j k>` 只依赖 `i-j`？** `R_i` 是角度 `i·θ` 的旋转；`R_iᵀR_j = R_{j-i}`，所以内积 = `qᵀR_{j-i}k`。所有位置共享同一组频率，相对位移决定结果。
8. **RMSNorm 为什么要 upcast 到 fp32？** 平方在 fp16/bf16 下容易溢出或丢精度；内部用 fp32 算 `mean(x²)+eps` 再开方，最后 cast 回输入 dtype，保证 dtype 透传。
9. **为什么 pre-norm 还要 `ln_final`？** pre-norm 的残差流一路不被归一化，越往后尺度可能越大；`ln_final` 在 `lm_head` 前把最终表示拉回稳定尺度。block 内部的 norm 只作用于子层输入。
10. **参数量的公式是什么？** `2VC + N(4C² + 3C·d_ff + 2C) + C`（untied）。tied 时减 `VC`。代入 V=256,C=64,N=2,d_ff=128 得 115,008，与 `num_parameters()` 和 `sum(p.numel())` 一致（`results/accounting.json`）。
11. **FLOPs 的公式是什么？长上下文时哪项主导？** 每层 `8Td² + 4T²d + 6T·d·d_ff`，lm_head `2TdV`。ctx 16384 时 attention 占 61.73%，因为 `4T²d` 按 `T²` 增长，而其他项按 `T` 线性增长（`results/accounting.json`）。
12. **13 个 maxdiff 都 ≤1.05e-5，能说明什么？不能说明什么？** 说明前向数值、`state_dict` 布局、RoPE/mask 约定与官方一致（在官方权重和固定输入上）；不能说明初始化正确（权重被覆盖）、不能说明训练动态/泛化、也不能外推到更大模型。

</details>

### 9.3 完成标准

- 45 个测试函数全绿（不是全 SKIP）；13 个 parity maxdiff 都打印出来且 ≤1e-4。
- `accounting.py --strict` 无失败；`exp_overfit.py` 的 `checks` 全 true。
- 能不看代码解释 §9.2 的 12 题，并能指出每个数字来自哪个 JSON/命令。

---

## 10. FAQ 20 问

**Q1. 为什么不能直接用 `nn.Linear` / `nn.Embedding`？**
A1 要求自己实现，内置模块的初始化、参数布局和错误行为都与契约不同；而且 `nn.Embedding` 的初始化不是 `trunc_normal(std=1)`。

**Q2. `Linear` 的权重为什么存 `(out,in)` 而不是 `(in,out)`？**
与 PyTorch `nn.Linear` 和官方 `state_dict` 一致；前向 `x @ W.T` 得到 `(..., out)`。存反了在方阵时不会报错，只会数值错。

**Q3. 为什么所有 Linear 都没有 bias？**
现代 LLM（PaLM/LLaMA 等）通常去掉 bias，少参数、便于融合；A1 handout §3.3.2 明确无 bias，官方权重也没有 `bias` 键。

**Q4. `trunc_normal_` 的 `std` 为什么是 `sqrt(2/(d_in+d_out))`？**
Xavier 风格的尺度，让前向激活方差大致稳定；截断在 `±3σ` 去掉过大初值。Embedding 单独用 `std=1`，因为查表不经过矩阵乘。

**Q5. 13 个 parity 全过，能说明初始化写对了吗？**
不能。`load_state_dict` 会把官方权重覆盖到你的参数上，初始化错误完全被掩盖；必须靠 D27–D29 单独测。

**Q6. RMSNorm 为什么先 upcast 到 fp32？**
平方会放大 fp16/bf16 的溢出与精度损失；内部 fp32 算完再 cast 回输入 dtype，兼顾稳定与 dtype 透传。

**Q7. RMSNorm 为什么没有减均值、没有 bias？**
RMSNorm 只按均方根缩放，保留均值方向；少一次均值计算和一组 bias，参数只有 `C` 个 gain。

**Q8. `eps` 放在 `sqrt` 里面还是外面？**
里面：`sqrt(mean(x²)+eps)`。放外面是 `sqrt(mean(x²))+eps`，数值和官方不一致，`E34` 会偏。

**Q9. SwiGLU 的 `d_ff` 为什么取 8/3·C？**
SwiGLU 有三个矩阵而 ReLU-FFN 有两个；乘 2/3 再乘原来的 4 倍得到 8/3，使参数量大致对齐。

**Q10. `d_ff` 取整为什么要求是 64 的倍数？**
GPU 矩阵乘内核对齐到 64/128 更高效；本 lab 的默认规则是“最近的 64 倍数，最小 64”。

**Q11. RoPE 为什么只转 q/k，不转 v？**
位置信息通过 q·k 的相对角度进入注意力分数；v 只被加权求和，旋转它不会改变“谁看谁”，反而多一次无意义运算。

**Q12. 交错配对和 `rotate_half` 能不能直接互换？**
不能直接换；它们只差一个通道置换。要么按官方的交错顺序实现，要么在置换 q/k 通道的同时置换投影权重行，否则 parity 偏。

**Q13. RoPE 的 `d_k` 用 `C` 还是 `C/H`？**
`C/H`。MHA 里每个 head 独立旋转，角度表列数是 `d_k/2`；用 `C` 会把不同 head 的通道混在一起。

**Q14. 为什么 `token_positions` 要支持 `(B,T)`？现在支持哪些形状？**
RoPE 的通用接口允许每个样本、每个位置有自己的位置 id（例如 padding、片段拼接）。MHA 现在支持 `(T,)`、`(1,T)`、`(B,T)`、`(1,1,T)`：当 `token_positions.ndim == x.ndim - 1` 时自动 `unsqueeze(-2)`，把 `(B,T)` 变成 `(B,1,T)` 再广播到 `(B,H,T,D)`。caveat：`(B,H,T)` 对 3D 输入不是受支持形状（`H == num_heads` 可能恰好广播，`H != num_heads` 报错，不要依赖）；4D+ 输入逐 batch 位置请显式补成 `(B1,B2,1,T)`。`test_C27` 是这条的回归测试（新增，不计入 43 项）。

**Q15. mask 的 `True` 到底表示可见还是屏蔽？**
可见（信息可流动）。`masked_fill(~mask, -inf)`：`False` 被填 `-inf`，softmax 后权重为 0。

**Q16. 为什么 mask 用 `-inf`，不用 `-1e9`？**
`-inf` 经过 softmax 后严格为 0；`-1e9` 只能得到极小值，A05 会断言 `probs[~mask] == 0`，用 `-1e9` 过不了。

**Q17. pre-norm 和 post-norm 差在哪？**
pre-norm：`x + f(norm(x))`，norm 在子层输入侧，残差流不被归一化；post-norm：`norm(x + f(x))`，norm 在残差相加之后。前者深层更好训，是现代 LLM 的默认。

**Q18. 为什么 block 之外还要一个 `ln_final`？**
pre-norm 的残差流尺度可能偏大；`ln_final` 在 `lm_head` 前做最后一次归一化，稳定 logits 尺度。

**Q19. `lm_head` 默认 tie 吗？tie 了会怎样？**
默认 untied；官方 fixture 的两张表不相等。tie 会共享存储、少 `V·C` 参数；装官方权重时两个键写同一块存储，后加载的会覆盖先加载的，logits 偏（`strict=True` 不一定报错，因为键都在）。C25 专门测 tie 的路径。

**Q20. loss 下降了，能说明实现写对了吗？**
不能。loss 下降只说明梯度通路和优化器能工作；mask 方向、RoPE 配对、初始化等错误都可能让 loss 照样降。正确性要靠官方快照 maxdiff + 性质测试 + 初始化测试。

---

## 附录 A 命令速查

```bash
cd /Users/yunni/Projects/cs336-prep-code

# 自检（约 0.9 s）
.venv/bin/python model_lab/toy_transformer.py

# 45 个测试函数（2 线程，约 1.1 s）
.venv/bin/python model_lab/test_toy_transformer.py

# 官方仓库不在默认路径时
A1_REPO=/path/to/assignment1-basics .venv/bin/python model_lab/test_toy_transformer.py

# Lead 的独立复核（不 import 测试代码，自己复算 5 个模块）
.venv/bin/python model_lab/lead_spotcheck.py

# 资源核算（任一 check 失败 exit 1）
.venv/bin/python model_lab/accounting.py --strict

# 单 batch 过拟合（90 秒硬停）
.venv/bin/python model_lab/exp_overfit.py --max-seconds 90

# 画图时的 matplotlib 缓存目录
MPLCONFIGDIR=$PWD/.tmp/mpl .venv/bin/python model_lab/accounting.py
```

环境：`.venv/bin/python`（Python 3.14.3 + torch 2.14.1）；所有脚本 `torch.set_num_threads(2)`；单命令 < 120 秒；不联网。

## 附录 B 术语表

| 术语 | 含义 |
| --- | --- |
| `B / T / C / H / D / V / F / N` | batch、序列长、d_model、head 数、head 维、词表、d_ff、层数 |
| pre-norm | norm 放在子层输入侧：`x + f(norm(x))` |
| RMSNorm | 按均方根缩放 + gain，不减均值、无 bias |
| SwiGLU | `W2(SiLU(W1x) ⊙ (W3x))` 的门控 FFN |
| RoPE | 旋转位置编码，交错配对，只作用于 q/k |
| SDPA | scaled dot-product attention：`softmax(QKᵀ/√d_k)V` |
| causal mask | 下三角 `True`，禁止看未来 |
| `trunc_normal_` | 截断正态初始化，`±3σ` |
| tied / untied | `lm_head` 与 `token_embeddings` 是否共享权重 |
| parity | 与官方快照逐元素对齐 |
| maxdiff | 最大绝对误差，`(actual-expected).abs().max()` |
| `state_dict` | PyTorch 的参数字典；键名是 parity 契约 |
| `_orig_mod.` | `torch.compile` 留下的前缀，装载前要 strip |
| FLOPs / `2mnp` | 矩阵乘的浮点运算量；`A(m,n)@B(n,p)` 为 `2mnp` |
| MFU | model FLOPs utilization，实测吞吐 / 硬件峰值 |
| KV cache | 推理时缓存 k/v，本 lab 未实现 |

## 附录 C 材料地图

| 材料 | 覆盖 | 建议 |
| --- | --- | --- |
| `model_lab/SPEC.md` | 接口、数值约定、快照配方、测试清单、实验口径 | 写代码前读 §3–§6；实现后对照 §9 DoD |
| `.tmp/a1_handout.txt:684–1030`（§3.3–3.5） | 公式、题目、分值、初始化/RoPE/attention 原文 | 精读；公式以它为准 |
| `/Users/yunni/Projects/assignment1-basics/tests/` | 官方测试、`adapters.py`、fixtures、13 个快照 | 只读；对照 `test_model.py`、`conftest.py` |
| `transformer_steps/` | GPT-2 骨架的 8 步增量讲解 | 第 1 章前快速复习 |
| `llama_steps/` | Llama 改造、HF 对齐、KV cache | 读 1.3.2 时配合 `step03_rope.py` 第 5 节 |
| `BPE_TOKENIZER_TUTORIAL.md` | 分词器 | 与本 lab 解耦，按需读 |
| `model_lab/results/*.json` | 核算与过拟合的机器可读结果 | 论文/笔记引用数字时先查它 |

## 附录 D 检查表

**运行检查**

- [ ] `.venv/bin/python model_lab/toy_transformer.py` 打印 `OK: toy_transformer 自检通过`
- [ ] `.venv/bin/python model_lab/test_toy_transformer.py` 输出 `45 passed, 0 failed, 0 skipped`
- [ ] 13 行 `maxdiff=` 都出现，且每行 ≤ `1e-4`
- [ ] `accounting.py --strict` 退出码 0；`results/accounting.json` 的 `checks` 全 true
- [ ] `exp_overfit.py --max-seconds 90` 退出码 0；`checks` 全 true，`timed_out=false`
- [ ] 没有修改 `/Users/yunni/Projects/assignment1-basics`，没有提交 git

**学习检查**

- [ ] 能画出 `(B,T,C) → (B,H,T,D) → (B,H,T,T) → (B,T,C)` 的完整路径
- [ ] 能解释 `True`=可见、RoPE 交错配对、per-head `d_k`、untied 默认
- [ ] 能手推参数量公式与每层 FLOPs 公式
- [ ] 能解释 13 个 maxdiff 中任意一项在验证什么
- [ ] 能说明过拟合实验证明了什么、没证明什么

**43 项测试映射（组级）**

| 组 | 项 | 测试函数 |
| --- | --- | --- |
| A 手算 | 1–10 | `test_A01`–`test_A10` |
| B 形状 | 11–18 | `test_B11`–`test_B18` |
| C 性质 | 19–26 | `test_C19`–`test_C26` |
| C27 回归 | 不计入 43 项 | `test_C27` |
| D 初始化 | 27–30 | `test_D27`–`test_D30` |
| E parity | 31–42 | `test_E31`–`test_E42`（13 个快照） |
| F 端到端 | 43 | `test_F43` |

逐项对应见 `model_lab/README.md` §3。
