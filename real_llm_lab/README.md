# 真实 LLM 实验：把 GPT-2 的官方权重装进你自己写的代码

这个小实验把 `transformer_steps/` 里你从零写的那套东西（step01–08）换成**真实权重**：
OpenAI 的 **GPT-2 small**（124,439,808 个参数，官方 safetensors 文件），
然后做 5 个能在几分钟内跑完、每个都有明确结论的实验。

**为什么适合新手**：实验里用到的每个概念你都学过——分词器、嵌入、注意力、
因果掩码、多头、Pre-LN block、交叉熵、KV cache、温度采样。
没有任何一个新概念是「必须先学完 xx 才能做」的。

---

## 0. 一句话结论

| 实验 | 问题 | 真实结果 |
|---|---|---|
| exp1 | 我写的结构对不对？ | 与 HuggingFace **逐位一致**（logits 最大差 0.00e+00），124,439,808 参数 |
| exp2 | 真实权重里能看到「复制」行为吗？ | 找到 induction head：L5H5 分数 0.7484（均值 0.0771），偏移正好 = 重复周期 − 1 |
| exp3 | KV cache 到底快多少？ | 端到端 **2.19× → 4.64×**（不是复杂度公式说的几十倍）；单步 decode 恒定 ~7.5 ms |
| exp4 | 去掉因果掩码会怎样？ | 预训练模型反而**变差**（loss 3.30→5.97）；但**训练时**去掉掩码会让 train loss 降到 0.087、推理崩到 7.88 |
| exp5 | 温度和 top-k 在做什么？ | 同一位置熵 0.98↔7.55 bit；确定的地方温度影响小，犹豫的地方影响大 |

---

## 1. 准备（只需要做一次）

仓库自带的 `.venv` 里已有 torch / numpy。再补三个库：

```bash
uv pip install --python .venv/bin/python -r real_llm_lab/requirements.txt
```

下载 GPT-2 small 官方权重（约 550 MB，支持断点续传 + sha256 校验）：

```bash
.venv/bin/python real_llm_lab/download_weights.py
```

权重来自 HuggingFace 的 `openai-community/gpt2` 仓库，`model.safetensors` 的
sha256 是 `248dfc39…d3a707`（脚本会自动校验），和仓库 LFS 记录一致。

目录结构：

```
real_llm_lab/
├── gpt2_real.py        # 你的 GPT-2：加载真实权重 + KV cache + attention_maps
├── download_weights.py # 下载 / 校验权重
├── exp1_parity.py      # 实验 1：和 HuggingFace 逐位对比
├── exp2_heads.py       # 实验 2：注意力头 / induction head
├── exp3_kv_cache.py    # 实验 3：KV cache 实测加速
├── exp4_mask_ppl.py    # 实验 4：掩码反例 + 困惑度 + 训练期对照
├── exp5_sampling.py    # 实验 5：greedy / temperature / top-k
├── run_all.sh          # 一键跑完，输出存到 runs/
├── runs/               # 每个实验的真实运行记录
├── figs/               # 生成的图
└── weights/gpt2/       # 官方权重（550 MB，不要提交到 git）
```

一键跑完（约 2～3 分钟，CPU）：

```bash
bash real_llm_lab/run_all.sh
```

---

## 2. 五个实验，各看什么

### exp1：我写的 GPT-2 和官方实现一样吗？

`gpt2_real.py` 就是 step07 的 `MiniGPT` 换成 GPT-2 的 4 个实现细节：

| step07（你写的） | 真实 GPT-2 | 为什么要改 |
|---|---|---|
| `qkv = Linear(C, 3C)` 分开切 | `c_attn = Linear(C, 3C)` 一次算完 | 少一次矩阵乘，和官方权重对齐 |
| `nn.GELU()`（精确版） | `GELU(approximate="tanh")` | GPT-2 config 里叫 `gelu_new` |
| `Linear` 权重 `(out, in)` | 文件里是 `Conv1D` 的 `(in, out)` | 加载时 `.t()` 转置 |
| `ln` 默认 eps | eps=1e-5 | 其实一样，显式写出 |

**结果**（`runs/exp1.txt`）：
- 参数量 124,439,808，与 GPT-2 论文数字相同；
- 4 条 prompt 的 logits 与 HuggingFace 的最大差 **0.00e+00**；
- 12 层 × 12 头注意力矩阵最大差 **0.00e+00**；
- 63 token 英文段落 loss 2.8974 / perplexity 18.13，两边一致；
- 贪心生成 `"1, 2, 3, 4, 5, 6, 7, 8,"` → `" 9, 10, 11, 12, 13, 14,"`，两边逐 token 相同。

> 结论：**你已经写出了 GPT-2 本体**。后面 4 个实验都是在真实权重上观察它。

### exp2：真实权重里能看到「复制」行为（induction head）

step03–05 的注意力矩阵，在玩具模型上只能看到「上三角是 0」。
换成 GPT-2 后，同一张矩阵里能直接看到**结构**。

做法：给一句重复两遍的话
`"The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the"`
（period = 10 个 token），然后对 144 个头分别算：
**后半段每个位置 i，把多少注意力放在「上一次出现的下一个 token」（i−period+1）上**。

**结果**（`runs/exp2.txt`、`figs/exp2_induction_heads.png`）：
- 最强头 **L5H5**：得分 0.7484，是全体均值 0.0771 的 9.7 倍；
- 它的注意力峰值偏移正好是 k=9 = period−1，最后一行 51% 的注意力给了 `" lazy"`；
- 前 5 名（L5H5、L5H1、L7H10、L10H1、L10H7）全部在 k=9 处有峰值；
- 16 个头得分 ≥ 0.20 —— 复制行为不止一个头，有明显的多头冗余；
- 行为层面：这句重复文本的下一 token 是 `" lazy"`（p=0.9678），正好是第一遍里对应的那个词；
- 经典案例：`"When Mary and John went to the store, John gave a drink to"` → `" Mary"`（p=0.4460）。

> 结论：**induction head 是真的存在的**，而且它就是你学的「Q 匹配 K、再对 V 加权」的机制。

### exp3：KV cache 在真实权重上快多少？

step08 里你推导过：naive 每步重算整段前缀 O(N³)，KV cache 每步 O(N²)。
真实权重下把这个差距量出来。

**结果**（`runs/exp3.txt`、`figs/exp3_kv_cache.png`）：
- 正确性：贪心解码两种实现逐 token 完全相同；fp64 下 logits 差 8e-13（证明算法等价，fp32 的差只是舍入）；
- 端到端（生成 32 个 token）：prompt 16 → 128 时，加速比 **2.19× → 4.64×**；
- 单步前向：全量前向随 context 从 14.5 ms 涨到 131.0 ms（线性），cached decode 恒定 **7.3–8.2 ms**；
- 缓存大小：`2 × 12 层 × 1 × 1024 × 768 × 4B` = **72 MiB**，是权重（475 MiB）的 15%。

> 结论：缓存让「每步的计算量」不再随上下文增长，但每步的固定开销（读权重、调度）省不掉，
> 所以在 CPU + 小模型上端到端只有几倍。这正是真实推理优化的重点。

### exp4：因果掩码——两个反直觉的发现

**发现 1（权重固定）**：同一段真实英文、同一份权重，只把 `causal` 从 True 改成 False：

| 设置 | loss | perplexity |
|---|---|---|
| 正常因果掩码 | 3.2987 | 27.1 |
| 去掉掩码（能看到未来） | **5.9697** | **391.4** |

**loss 反而上升**。因为 124M 个参数都是按「只能看左边」训练的，放开右边只会把它弄坏——
掩码不是装饰，是模型定义的一部分（就是你 step04 写的那行 `masked_fill`）。

**发现 2（从零训练）**：那两个 1 层小模型（约 2KB 英文，300 步，只差 causal 开关）：

| 模型 | 训练 loss | 正常因果方式评估 |
|---|---|---|
| 有掩码 | 2.31 | **2.61**（比瞎猜 3.76 好） |
| 无掩码 | **0.087**（好得像收敛了） | **7.88**（比瞎猜还差，生成是乱码） |

无掩码的模型学会了「把位置 t+1 的答案搬到位置 t」，训练 loss 骗了你；
推理时它得不到未来 token，作弊手段失效，立刻崩掉。这就是 step04 那句
「训练时忘了加掩码，loss 会很好看，但生成时表现立刻崩掉」。

**发现 3（跨分词器比较）**：同一份 GPT-2 权重，四类文本的困惑度：

| 文本 | token 数 | 每 token ppl | bits/字符 |
|---|---|---|---|
| 英文新闻 | 234 | 27.1 | **1.02** |
| 中文段落 | 313 | **16.9**（看起来更低！） | **8.57** |
| Python 代码 | 379 | 14.4 | 1.92 |
| 随机 id | 234 | 259,839 | — |

中文每 token 的 ppl 反而更低，是因为 BPE 是字节级的：一个汉字要 2.10 个 token，
而汉字内部的 UTF-8 字节高度可预测。换成 **bits/字符**（跨分词器可比）后，
中文每个字符的代价是英文的 **8.4 倍**。

> 结论：**不同分词器的 perplexity 不能直接比**；比较模型能力要用 bits/char 这类单位。

### exp5：温度、top-k 到底在做什么

温度就是对 logits 做一次除法再 softmax：`p_i = softmax(logits/T)_i`。

| 策略 | 生成 40 token 的不同 token 数 | 耗时 |
|---|---|---|
| greedy（T→0） | 33/40 | 0.33 s |
| T=0.5, top_k=40 | 32/40 | 0.36 s |
| T=1.0, top_k=40 | 37/40 | 0.36 s |
| T=1.5, top_k=40 | 36/40 | 0.35 s |
| T=1.0, 不设 top_k | 32/40 | 0.35 s |

同一个位置换三个温度，分布的变化可以量化（`runs/exp5.txt`）：

| context | 最可能的 token | T=1.0 概率 | 熵（T=1.0） |
|---|---|---|---|
| `"The future of artificial intelligence is"` | `" uncertain"` | 7.7% | 7.55 bit |
| `"1, 2, 3, 4, 5,"` | `" 6"` | **91.0%** | **0.98 bit** |

模型越确定，温度的影响越小；越犹豫，温度调高越容易跑偏。
（均匀瞎猜的熵是 log2(50257) ≈ 15.62 bit。）

---

## 3. 你学的每一步 → 真实模型里的位置

| 你学的 | 在 GPT-2 里 | 大小 |
|---|---|---|
| step01 字符分词器 | BPE 分词器（`tokenizer.json`） | 50,257 个 token |
| step02 词嵌入 / 位置嵌入 | `wte` / `wpe` | (50257,768) / (1024,768) |
| step03 缩放点积注意力 | `attn.c_attn` + softmax | Q/K/V = 768→3×768 |
| step04 因果掩码 | `attn.bias`（官方文件里存成了 buffer） | (1,1,1024,1024) |
| step05 多头 | 12 层 × 12 头 | head_dim = 64 |
| step06 Pre-LN Block + FFN | `h.0` … `h.11` | 每层 ~7.09M 参数 |
| step07 lm_head 权重共享 | `lm_head.weight = wte.weight` | 省下 38.6M 参数 |
| step08 KV cache | exp3 的 72 MiB 缓存 | 2×12×1024×768×4B |

---

## 4. 新出现的东西（知道存在即可，不必先学会）

- **BPE 分词器**：step01 是字符级（几十个 token），GPT-2 是字节级 BPE（50,257 个）。
  所以「一个 token」不等于「一个字」，做任何比较前先看 token 数。
- **safetensors**：权重文件格式。8 字节 header 长度 + JSON 头 + 原始字节；
  `safetensors.torch.load_file` 一行读完。加载时注意 GPT-2 的 `Conv1D` 权重是转置存的。
- **HuggingFace transformers**：只当「参考实现」用（exp1）。我们不依赖它跑实验，
  但用它证明「自己写的实现和工业界实现逐位一致」。
- **matplotlib**：图都是 `Agg` 后端 + 英文标签（避免缺中文字体时出现方块）。
- **CPU / MPS**：本实验全程 CPU fp32，结果可复现；M5 的 MPS 可用，
  但小 batch 下不一定更快（CPU 已经很快，MPS 有调度开销）。

---

## 5. 已知限制（别把这当成「大模型结论」）

- GPT-2 small 只有 124M 参数，**事实知识很不可靠**。
  比如问它 `"the capital of France is"`，它更想接 `" the"` 而不是 `" Paris"`。
  本实验验证的是「结构和权重对得上」，不是「它什么都懂」。
- 计时数字是 Apple M5 CPU fp32、best-of-N（多次取最优），只用来比较相对差距。
- 中文/英文 perplexity 对比只在「同一分词器、同一段文本」内部有意义（见 exp4）。
- 训练期对照用的是 1 层 64 维字符级小模型，和 GPT-2 无关，只是为了把掩码的作用跑出来。

---

## 6. 延伸练习（做完 lab 之后）

1. 把 exp2 的重复片段换成 period=3/20，最强头还是 L5H5 吗？
2. exp3 里把 batch 改成 4，cached 的固定开销被摊薄，加速比会变成多少？
3. exp4 里把训练步数从 300 改到 100/1000，看「作弊」出现的临界点。
4. 给 exp5 加 top-p（nucleus）采样，和 top-k 对比。
5. 换一个真实权重：`distilgpt2`（82M）或 `TinyStories-33M`（33M），对比 exp1/ex2 的结论。
6. 把 `block_size` 调小、把 `causal=False` 的模型拿去生成，验证「作弊模型」在真实 token 数下何时崩溃。
