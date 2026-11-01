# 从零实现一个最简单的 Transformer（8 步）

这是一套**按顺序阅读**的代码：每一步在前一步的基础上只加一个新概念，
每个文件都能单独运行，并打印关键张量的形状和自检结果。

目标是让你能看着 `diff step03 step04` 就明白「因果掩码到底改了什么」。

## 怎么运行

在项目根目录执行（使用已有的 `.venv`）：

```bash
.venv/bin/python transformer_steps/step01_tokenizer.py
```

依次跑 01 到 08。`step08_train.py` 会真正训练 300 步并生成一段文本，
在 CPU 上大约几秒钟。

## 八个阶段

| 步骤 | 文件 | 新增概念 | 关键形状 |
|---|---|---|---|
| 1 | `step01_tokenizer.py` | 词表、字符 ↔ ID | `str ↔ list[int]` |
| 2 | `step02_embeddings.py` | 词嵌入、位置嵌入 | `(B,T) → (B,T,C)` |
| 3 | `step03_attention.py` | Q/K/V、缩放点积注意力 | `(B,T,C) → (B,T,T) → (B,T,C)` |
| 4 | `step04_causal_mask.py` | 因果掩码（下三角） | `(B,T,T)` |
| 5 | `step05_multihead.py` | 多头、输出投影 | `(B,T,C) → (B,H,T,D)` |
| 6 | `step06_block.py` | FFN、残差、Pre-LN | `(B,T,C)` 不变 |
| 7 | `step07_model.py` | 堆叠、lm_head、权重共享 | `(B,T) → (B,T,V)` |
| 8 | `step08_train.py` | 训练、自回归生成 | loss 标量 |

## 形状速查

```
B  batch size
T  序列长度（<= block_size）
C  n_embd，每个 token 的向量维度
H  n_head
D  head_dim = C / H
V  vocab_size

idx         (B, T)
embedding   (B, T, C)
q, k, v     (B, H, T, D)     每个头一份
scores      (B, H, T, T)     第 i 行 = 位置 i 对所有位置的注意力分数
weights     (B, H, T, T)     softmax 后，每行和为 1，上三角为 0
attn out    (B, T, C)
logits      (B, T, V)
loss        ()
```

## 每一步最该记住的一句话

1. **分词器**：模型只认整数；`vocab_size` 必须和模型输出维度一致。
2. **嵌入**：注意力本身没有顺序概念，位置信息必须显式加进去。
3. **注意力**：`softmax(QKᵀ/√D)V`，本质是用 Q 去匹配 K，再对 V 加权求和。
4. **掩码**：训练时必须挡住未来，否则就是抄答案；用 `-inf` 让 softmax 权重为 0。
5. **多头**：把通道切成 H 份并行做注意力，最后用 `out_proj` 混合各头。
6. **Block**：残差是梯度高速公路；Pre-LN 让深层训练稳定；FFN 占约 2/3 参数。
7. **输出层**：`lm_head` 把 C 维映射到词表；和输入嵌入共享权重可省参数。
8. **训练/生成**：目标是输入右移一位；生成时每次取最后一个位置的 logits 采样。

## 与 `minimal_transformer.py` 的对应关系

| 内容 | 本目录 | `minimal_transformer.py` |
|---|---|---|
| 分词器 | step01 | 9–18 |
| 配置 | step02 | 20–27 |
| 注意力 | step03–05 | 29–56 |
| FFN | step06 | 58–68 |
| Block | step06 | 70–80 |
| 完整模型 | step07 | 82–124 |
| 生成 | step08 | 126–139 |
| 训练循环 | step08 | 152–194 |

## Step 8 里的 KV cache

`step08_train.py` 现在有两种生成方式，并逐步验证它们等价：

| 函数 | 每一步喂什么 | 每步注意力计算量 | 生成 N 个 token 的总前向量 |
|---|---|---|---|
| `generate_naive` | 整段前缀 `(B, t)` | O(t²) | O(N³) |
| `generate_cached` | 只有新 token `(B, 1)` | O(t) | O(N²) |

关键改动在 `step07_model.py` 的 `MultiHeadAttention.forward`：
新增可选的 `k_cache / v_cache / start_pos`。缓存模式下，把新算出的 k/v
写进预分配的 `(B, H, block_size, D)` 缓冲区，再对「0..t 的全部 K/V」做注意力。

两个容易踩的坑：

1. **位置**：新 token 的绝对位置不是 0，必须通过 `start_pos` 传给位置嵌入。
2. **掩码**：prefill 时 query 有多个位置，仍然需要因果掩码；
   decode 时 T=1，所有缓存位置都在左边，掩码全为 True。

缓存大小：`2 * n_layer * B * T_max * C * sizeof(dtype)`。
本例 2 层、C=64、T=48、fp32 只有 48 KiB；
换成 32 层、C=4096、T=32768、fp16，单个样本就要约 16 GiB。

## 建议的练习顺序

1. 把 `step05` 的 `n_head` 改成 1，和 `step04` 对比输出（数学上等价，参数布局不同）。
2. 注释掉 `step04` 的 mask，观察 `step08` 的 loss 会异常低——因为模型能直接看到答案。
3. 把 `step06` 的 Pre-LN 改成 Post-LN（`x = ln(x + f(x))`），训练 300 步看看 loss 曲线。
4. 把 `step07` 的 `block_size` 改小到 8，生成时会看到模型只能利用很短的前文。
5. 给 `step08` 加验证集：从语料里留出最后 20%，观察训练 loss 和验证 loss 何时分叉。

## 之后可以往哪走

> 下面前四条已经做成了姊妹篇 **[`llama_steps/`](../llama_steps/README.md)**：
> 同样按步拆解、每个文件能单独运行，把这里讲的 GPT-2 一步步改造成 Llama
> （RMSNorm、SwiGLU、RoPE、GQA、KV cache，最后装上真实的 SmolLM2-135M 权重）。

- **KV cache**：推理时缓存 k/v，避免每步重算整段序列。
- **RoPE / ALiBi**：更好的位置编码，可外推到训练长度之外。
- **RMSNorm / SwiGLU**：Llama 系列的归一化和 FFN 变体。
- **BPE 分词器**：从字符级走向真实子词。
- **FlashAttention**：不显式物化 `(T,T)` 矩阵，省显存、更快。
- **混合精度 / 梯度累积 / 学习率调度**：真正训练大模型需要的工程手段。
