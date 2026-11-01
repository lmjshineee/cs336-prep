# llama_steps 真实运行记录

环境：Apple M5 / macOS 27.2 / CPU fp32（未用 MPS）、Python 3.14.3、torch 2.14.1、transformers 5.21.0（只作对照）。

以下都是 `bash llama_steps/run_all.sh` 存到 `llama_steps/runs/stepNN.txt` 的原文，只去掉了行首多余的空行；数字未改、错别字未改。

## 01

```text
$ .venv/bin/python llama_steps/step01_rmsnorm.py

--- 1. 公式与代码对照 ---
  LayerNorm:  μ = mean(x),  σ² = mean((x-μ)²)
              y = (x - μ) / √(σ² + ε) * γ + β
  RMSNorm :   y = x / √(mean(x²) + ε) * γ      # 不碰均值，没有 β
  写成代码：
    LN : (x - x.mean(-1, keepdim=True))
         / (x.var(-1, keepdim=True, unbiased=False) + eps).sqrt() * g + b
    RMS: x / (x.pow(2).mean(-1, keepdim=True) + eps).sqrt() * g

--- 2. 均值偏移的例子：LN 抹掉它，RMSNorm 保留它 ---
输入 mean              = +3.026   <- 整体偏向 +3
输入 std               = 0.519
LN 输出 mean           = -0.000   <- 被减均值拉回 0
LN 输出 std            = 1.000   <- 除以标准差后 ≈ 1
RMS 输出 mean          = +0.986   <- 均值方向保留下来
RMS 输出 std           = 0.169   <- 它不是 1：没有重新居中
RMS 输出均方根         = 1.000   <- 被拉到 1 的是均方根
[OK] LN 输出均值≈0；RMSNorm 输出均方根≈1，但保留了均值方向

--- 3. 参数量：C → 2C vs C ---
C (隐藏维度)           = 64
LayerNorm 参数         = 128   <- γ (C) + β (C)
RMSNorm 参数           = 64   <- 只有 γ (C)
[OK] RMSNorm 参数正好是 LayerNorm 的一半（64 vs 128）

--- 4. 实测耗时（warmup 20 次 + 200 次取中位数）---
手写 LN 公式           = 90.6 µs
手写 RMS 公式          = 27.2 µs
公式版比值             = 3.33x   <- 少一次减均值/方差，快不少
nn.LayerNorm（融合）   = 16.5 µs
RMSNorm 模块           = 28.1 µs
  但 nn.LayerNorm 是单个 C++ 融合 kernel，而 RMSNorm 模块由几个小算子拼成，
  这个 8×64×64 的规模下启动开销占大头，差异很小（甚至可能反过来）。
  省下的是「少一次逐元素计算 + 一半参数」，大张量/大模型上才明显。
  注意：上面几行耗时每次运行都会小幅抖动，属正常现象。

--- 5. 数值稳定性：eps 加在均方根里面 ---
全零输入 + 朴素除法    = 有 NaN = True
全零输入 + RMSNorm     = 全零 = True   <- eps 兜住分母
bf16 输入 → 输出 dtype = torch.bfloat16   <- 内部 fp32 计算，再转回
[OK] eps 在开方前加进 mean(x²)：全零输入输出仍是 0 而不是 NaN

--- 6. 梯度检查 ---
x.grad 形状            = (2, 8, 64)   <- 与输入同形 (B,T,C)
max|x.grad|            = 2.8939
[OK] 反向传播走通：梯度形状正确、全部有限（全零输入也不炸）

--- 7. 在 block 里怎么用 ---
  GPT-2:  self.attn_norm = nn.LayerNorm(C)   # 2C 个参数
  Llama:  self.attn_norm = RMSNorm(C, eps=1e-5)  # C 个参数
  结构不变（还是 Pre-Norm：先归一化再进子层），只把 LN 换成 RMSNorm。

一句话记住：
  RMSNorm 不重新居中、参数少一半、少一次减均值；小规模实测未必更快。

[OK] 全部检查通过
```

## 02

```text
$ .venv/bin/python llama_steps/step02_swiglu.py

--- 1. hidden 为什么是 8/3·C ---
C (隐藏维度)           = 64
8/3·C                  = 170.67   <- 与 4C 的 GPT-2 FFN 参数量对齐
向上取整到 8 的倍数    = 176   <- 170.67 → 176（8 的倍数，方便切块）
  如果 hidden 还保持 4C=256，三个矩阵就是 12C²，比 GPT-2 的 8C² 多 50%；
  缩到 8/3·C 正好把「多出来的那个门控矩阵」的预算还回去。

--- 2. 参数量对照 ---
GPT-2 FFN 参数         = 32,768   <- 2·C·4C = 8C²
SwiGLU 参数            = 33,792   <- 3·C·F
每个矩阵               = 11,264   <- C·F = 64·176
比值                   = 1.0312   <- ≈1.03，基本对齐
[OK] SwiGLU 在参数量基本不变的前提下多了一个门控矩阵

--- 3. 前向形状：可以直接替换旧的 FFN ---
x                      = (2, 16, 64)
gate_proj(x)           = (2, 16, 176)
up_proj(x)             = (2, 16, 176)
SiLU(gate) * up        = (2, 16, 176)   <- 逐元素相乘
down_proj(hidden)      = (2, 16, 64)   <- 回到 (B,T,C)
[OK] 输出形状和 GPT-2 FFN 完全一致，可以原地替换

--- 4. 门控直觉：SiLU 把 gate 值翻译成「开多大」---
  gate =   -2.00   -1.00   +0.00   +1.00   +2.00
  SiLU =  -0.238  -0.269  +0.000  +0.731  +1.762
  负数 → 压到 0 附近（门关上）；正数 → 近似原样通过（门打开）。
[OK] SiLU(x) = x·sigmoid(x)，且 SiLU(0) = 0（0 门 = 完全关闭）

--- 5. 有多少通道对 down_proj 的贡献被压到接近 0 ---
|SiLU(gate)| < 0.05 的比例 = 13.9%   <- 对 down 的贡献很小
  注意：门控 ≠ 稀疏计算——矩阵乘法仍是 dense 的，FLOPs 一个不少；
  收益是每个 token 能动态地少依赖一些通道（有效容量/表达），不是省算力。

一句话记住：
  SwiGLU 用一扇 SiLU 门逐元素筛一遍 up 的通道，再把 hidden 缩到 8/3·C。

[OK] 全部检查通过
```

## 03

```text
$ .venv/bin/python llama_steps/step03_rope.py

--- 1. 学习式位置嵌入的两个毛病 ---
学习式位置嵌入参数     = 4,096   <- T·C = 64×64，纯查表
  查位置 200（训练长度只有 64）→ IndexError: index out of range in self
  毛病二：位置 id 超出表长就查不到，训练长度之外完全没法用。

--- 2. RoPE：把「加位置向量」换成「按位置旋转 q、k」---
  θ_i = 10000^(-2i/D)，第 m 个位置第 i 对通道旋转 m·θ_i 弧度
  q_rot = R(m·θ) q，k_rot = R(n·θ) k，v 不旋转
head_dim D             = 16   <- toy: n_embd 64 / n_head 4
cos                    = (320, 16)   <- (max_len, D) float32，不占参数
sin                    = (320, 16)   <- (max_len, D) float32，不占参数
θ_0 ... θ_3            = 1.000 0.316 0.100 0.032   <- 越靠后的通道转得越慢
cos 前后两半相同       = True   <- half-split 配对的前提
  rotate_half 把 (第 i 维, 第 i+D/2 维) 配成一对，用同一组角度旋转。

--- 3. 核心性质：整段序列平移，注意力分数矩阵不变 ---
平移 +37 后分数最大差  = 4.11e-06
[OK] 整个序列位置 +37，q@kᵀ 完全不变（RoPE 只认相对位置）
同一相对距离内最大差   = 1.67e-06   <- 分数只是 i-j 的函数
[OK] 按 i-j 分组后组内完全一致：分数只依赖相对距离

--- 4. 保范数：旋转不改变向量长度 ---
apply_rope 前后模长最大差 = 4.77e-07
[OK] 每个 token 的 q 模长不变（正交 = 只转方向、不改长度）

--- 5. 两种配对写法：half-split vs interleaved ---
重排通道后 interleaved = 4.77e-07   <- 与 half-split 等价
不重排直接 interleaved = 6.09e+00   <- 反例：通道配错了
  两种写法只在「先把通道重排成对应顺序」之后才等价；模型的投影权重
  是配套的，混用就错（step08 会用真实 logits 演示这一点）。
  我们选 half-split：HuggingFace 的 Llama 就是这么写的，step08 要对齐。

--- 6. 外推：长度 256 的分数矩阵照算不误 ---
长度 256 的分数矩阵    = (1, 1, 256, 256)   <- 没有越界，也没有 NaN
|offset| < 64 平均 |分数| = 0.812
|offset| ≥ 64 平均 |分数| = 0.801   <- 仍有定义，量级相当
[OK] RoPE 靠公式：训练长度 64 之外也能算，不像查表会越界
  注意：能算 ≠ 效果好；远距离外推质量要靠 NTK/YaRN 之类缩放，这里只验证不崩。

--- 7. 对 KV cache 友好：存已旋转的 k ---
单 token 旋转 vs 放进长序列 = 0.00e+00   <- 历史 k 不用重算
  decode 新 token 时只需 cos[start_pos:start_pos+T]；历史 k 已旋转好，原样复用。

--- 8. 画图 ---
图片已保存：/Users/yunni/Projects/temp/cs336-prep-code/llama_steps/figs/step03_rope.png
  图 (a) 的上下波动不是 bug：它是不同频率 θ_i 叠加的拍频；
  关键不变量是：分数只取决于相对距离 m-n。

一句话记住：
  RoPE 用「按位置旋转 q、k」代替位置查表：不占参数，分数只认相对距离。

[OK] 全部检查通过
```

## 04

```text
$ .venv/bin/python llama_steps/step04_gqa.py

--- 1. GQA 到底改了什么 ---
C (n_embd)             = 64
n_head                 = 4   <- Q 头的个数
n_kv_head              = 2   <- K/V 头的个数，只有 Q 头的 1/g
head_dim               = 16   <- = C / n_head，每个头的维度
g                      = 2   <- = n_head / n_kv_head，每 g 个 Q 头共享一组 K/V
  n_kv_head = n_head -> MHA（每个 Q 头独占自己的 K/V）
  n_kv_head = 1      -> MQA（所有 Q 头共享同一组 K/V）

--- 2. 形状流水线：只有 K/V 的头数变了 ---
x                      = (1, 8, 64)   <- (B,T,C)
q                      = (1, 4, 8, 16)   <- (B,H,T,D)，已过 RoPE，H 个 query 头
k                      = (1, 2, 8, 16)   <- (B,Hkv,T,D)，已过 RoPE，只有 Hkv 个 KV 头
repeat_kv(k)           = (1, 4, 8, 16)   <- (B,H,T,D)，复制出来给每个 Q 头
scores                 = (1, 4, 8, 8)   <- (B,H,T,T)，这里还没加因果掩码
out                    = (1, 8, 64)   <- (B,T,C)，和输入同形

--- 3. 参数量：K/V 投影变小了 ---
q_proj                 = 4,096   <- C * C
k_proj                 = 2,048   <- C * C/g
v_proj                 = 2,048   <- C * C/g
o_proj                 = 4,096   <- C * C
attn GQA               = 12,288   <- 2C² + 2C²/g
attn MHA               = 16,384   <- 4C²
saved                  = 4,096   <- -25.0% vs MHA
MQA (kv=1)             = 10,240   <- -37.5% vs MHA

--- 4. KV cache：GQA 真正省钱的地方 ---
per-token GQA          = 64   <- 每 token 每层的元素数
per-token MHA          = 128   <- 每 token 每层的元素数
toy cache GQA          = 32,768   <- 字节，B=1，fp32
toy cache MHA          = 65,536   <- 字节，B=1，fp32
Llama-3-8B GQA         = 1.0 GiB   <- 32 层 × 8 KV 头 × 8192 × fp16
Llama-3 MHA            = 4.0 GiB   <- 同层数，但 32 个 KV 头
Llama-3-70B GQA        = 2.5 GiB   <- 80 层 × 8 KV 头 × 8192 × fp16
  公式：2 * n_layer * B * T_max * n_kv_head * head_dim * sizeof(dtype)

--- 5. 三个自检 ---
[OK] n_kv_head = n_head 时 GQA 退化成 MHA，max|diff| = 0.0e+00
[OK] 改未来 token，不影响过去位置的输出（因果掩码）
[OK] repeat_kv 只复制、不新增参数：多出来的 Q 头共用同一组 K/V

一句话记住：
  GQA 让 g 个 Q 头共享一组 K/V：cache 小 g 倍，参数从 4C² 降到 2C²+2C²/g。

[OK] 全部检查通过
```

## 05

```text
$ .venv/bin/python llama_steps/step05_block.py

--- 1. 一个 block 只有两行，各自包着残差 ---
  x = x + attn(attn_norm(x), cos, sin)
  x = x + ffn(ffn_norm(x))
x                      = (1, 8, 64)   <- (B,T,C)，残差流的初始值
y                      = (1, 8, 64)   <- (B,T,C)，和输入同形

--- 2. Pre-Norm：归一化在残差分支入口，不在残差之后 ---
residual RMS           = 3.04   <- 没被归一化，故意放大 3 倍
attn_norm out RMS      = 1.0000   <- ≈ 1，尺度被拉回来了
manual vs block        = max|diff| = 0.0e+00
[OK] RMSNorm 在 float32 里算，最后转回输入 dtype：bf16 进 -> bf16 出

--- 3. 没有 bias、没有 dropout ---
Linear 个数            = 7   <- q/k/v/o + gate/up/down
bias 参数数            = 0
Dropout 个数           = 0
config.dropout         = 0.0   <- GPT-2 是 0.1，Llama 是 0

--- 4. 参数量：GPT-2 block vs Llama block（C=64, F=176, g=2） ---
GPT-2 attn             = 16,640   <- 4C² + 4C（含 bias）
GPT-2 ffn              = 33,088   <- 8C² + 5C（含 bias）
GPT-2 LayerNorm        = 256   <- 2 × (γ,β) = 4C
GPT-2 block            = 49,984   <- 12C² + 13C
Llama attn             = 12,288   <- 2C² + 2C²/g（无 bias）
Llama ffn              = 33,792   <- 3·C·F
Llama RMSNorm          = 128   <- 2 × γ = 2C
Llama block            = 46,208   <- -7.6% vs GPT-2
GPT-2 若不带 bias      = 49,408   <- 4C² + 8C² + 4C（LN 的 γ、β）
Llama 若 g=4           = 44,160   <- 2.5C² + 3CF + 2C

--- 5. block 内参数占比 ---
attn                   = 26.6%   <- GQA 之后只占约 1/4
ffn                    = 73.1%   <- 大头依然在 FFN
norm                   = 0.3%   <- 只剩 γ，几乎可以忽略

--- 6. 三个自检 ---
[OK] 输入输出同形：block 可以原样叠 N 层
[OK] 两个子层输出为 0 时，block 退化成恒等映射（max|diff| = 0.0e+00）
[OK] 先 norm 再进子层、最后加残差：hook 抓到的是未归一化的残差流

一句话记住：
  Llama block = GPT-2 block 的骨架 + 四个零件：
  RMSNorm / RoPE / GQA / SwiGLU，外加去掉 bias 和 dropout。

[OK] 全部检查通过
```

## 06

```text
$ .venv/bin/python llama_steps/step06_model.py

--- 1. 结构：没有位置嵌入 ---
vocab_size             = 30
block_size             = 64   <- RoPE 表的最大长度
head_dim               = 16   <- = C / n_head
rope_cos               = (64, 16)   <- (block_size, head_dim)，不进 checkpoint
位置嵌入参数           = 0   <- 整个模型一个都没有
[OK] RoPE 表 = half-split + cat(freqs, freqs)，theta=10000，与 HF 定义一致

--- 2. 参数分解 ---
token_embedding        = 1,920
每层 attn              = 12,288   <- × 2 层
每层 ffn               = 33,792   <- × 2 层
每层 norm              = 128   <- 2 个 RMSNorm 的 γ
blocks 合计            = 92,416   <- = 2 × (12,288 + 33,792 + 128)
norm                   = 64   <- 最后一层 RMSNorm
lm_head                = 1,920
总参数 (不共享)        = 96,320   <- Llama 默认 tie=False
总参数 (共享)          = 94,400   <- GPT-2 / SmolLM2 用这个

--- 3. KV cache：每层预分配 (B, n_kv_head, block_size, head_dim) ---
每层 k 的形状          = (1, 2, 64, 16)
nbytes()               = 32,768   <- 字节
公式                   = 32,768   <- 2*n_layer*B*T_max*n_kv_head*D*4B
每 token               = 512   <- 字节/token，只有 MHA 的 1/2

--- 4. 缓存 vs 全量重算：逐步比较 logits ---
  prefill  max|diff| = 0.00e+00
  step 1   max|diff| = 1.19e-07
  step 2   max|diff| = 5.22e-08
  step 3   max|diff| = 8.94e-08
  step 4   max|diff| = 7.45e-08
  step 5   max|diff| = 8.94e-08
  step 6   max|diff| = 8.94e-08
  step 7   max|diff| = 7.45e-08
  step 8   max|diff| = 1.12e-07
[OK] 每步 logits 都一致（最大 1.19e-07 < 1e-5）

--- 5. 两条贪心解码路径，结果必须逐 token 相同 ---
序列长度               = 32   <- prompt 16 + 新生成 16
  生成：Spring arrives, TkrkceTfl. TfTfT
[OK] 有 cache 与无 cache 的贪心解码结果完全一致

--- 6. 其他自检 ---
logits                 = (1, 11, 30)   <- (B, T, V)
loss (随机初始化)      = 3.3997   <- 还没训练，应接近均匀分布
ln(vocab_size)         = 3.4012   <- V = 30
[OK] 随机初始化时 loss ≈ ln(V)（差 0.0015 < 0.05）
[OK] 逐 token 走缓存 prefill 与整段前向一致（max|diff| = 1.2e-07）
[OK] RMSNorm 对 bf16 输入返回 bf16（内部 float32 计算，最后统一转回）
[OK] 改未来 token，前面位置的 logits 不变（端到端因果性）
[OK] 超过 block_size 抛 ValueError：序列超长：start_pos 0 + T 65 = 65 > block_size 64（RoPE 表只建到 block_size）

一句话记住：
  Llama = GPT-2 的骨架 - 位置嵌入 - bias - 权重共享
       + RMSNorm + RoPE + GQA + SwiGLU + KV cache。

[OK] 全部检查通过
```

## 07

```text
$ .venv/bin/python llama_steps/step07_train.py

--- 1. 实验设计：唯一的变量是架构 ---
corpus                 = 438 chars   <- 8 句英文重复两遍，字符级分词
vocab_size             = 30   <- 字符级，空格和标点都算一个 token
budget                 = 500 steps x batch 16 x block 32   <- 512 tokens/step
两个模型共用同一串 batch 和同一套 AdamW 配方，唯一不同的是结构。
这里的「同预算」指 500 步 x batch 16 x block 32 = 同样的
训练 token 数，不是同参数量：Llama 反而少 8,640（8.2%）。

--- 2. 参数量分解（实测 sum(p.numel())，C=64, 2 层, 词表 30）---
item                           GPT-2     Llama     delta
--------------------------------------------------------
token_embedding                1,920     1,920        +0
position_embedding             4,096         0    -4,096
lm_head                            0     1,920    +1,920
attention（2 层）             32,768    24,576    -8,192
ffn（2 层）                   65,536    67,584    +2,048
norm（2 层 + 最终）              640       320      -320
--------------------------------------------------------
total                        104,960    96,320    -8,640
Llama 是 GPT-2 的 91.8%：attention 少 25%（4C² → 3C²），FFN 多 3.1%（SwiGLU 三矩阵）。
[OK] 实测分解与参数量公式逐项一致，两个 total 也等于 params 总数

--- 3. 分别训练（同一串 batch，同一优化器配方）---
AdamW lr=0.003 betas=(0.9, 0.95) wd=0.1 clip=1.0，warmup 50 步 + cosine
warmup 是前 10% 步把学习率从 0 线性拉到最大值，避免一开始大步长搅乱随机初始化；
GPT-2 当年用 Adam + 手工 warmup，这里统一 AdamW + cosine，只为只留下架构一个变量。
[OK] 两个模型都满足因果性：改未来 token，不影响过去位置的 logits
  [GPT-2] step 100 | loss 1.1628
  [GPT-2] step 200 | loss 0.3167
  [GPT-2] step 300 | loss 0.1523
  [GPT-2] step 400 | loss 0.1403
  [GPT-2] step 500 | loss 0.1088
  [Llama] step 100 | loss 0.3821
  [Llama] step 200 | loss 0.1771
  [Llama] step 300 | loss 0.1181
  [Llama] step 400 | loss 0.1120
  [Llama] step 500 | loss 0.0913

--- 4. loss（真实数字，没有挑好看的）---
GPT-2 last-50 mean     = 0.1131   <- 最后 50 步训练 loss 均值
Llama last-50 mean     = 0.0990   <- 最后 50 步训练 loss 均值
delta (Llama-GPT-2)    = -0.0141   <- 负数 = Llama 更低
fixed eval batches     = GPT-2 0.1107 | Llama 0.0996   <- 8 个固定评测 batch，不参与训练
说明：这 8 个 batch 仍来自同一份 438 字符语料，固定下来只为减少抖动，不是泛化
指标（模型基本背下了语料，别把它当验证集）。
[OK] 两个模型都从 ln(V) 附近开始，并在 500 步内明显下降

[OK] 图已保存: llama_steps/figs/step07_loss.png (150 dpi, 215 KiB)

--- 5. 速度（CPU fp32，含前向 + 反向 + clip + step）---
GPT-2                  = 2.6 ms/step = 194,529 tokens/s
Llama                  = 3.2 ms/step = 159,113 tokens/s
（玩具模型上 Python/调度开销占比很大，ms/step 只代表这台机器）

--- 6. 生成样例（同一 prompt，总长 60 字符）---
prompt = "Spring arrives, and "   -> 各自续写 40 个字符
GPT-2 greedy   | Spring arrives, and the wind moves throves moves moves throu
Llama greedy   | Spring arrives, and the wind moves through the ves, rind the
GPT-2 T=0.8    | Spring arrives, and the wind moves throves moves moves rooug
Llama T=0.8    | Spring arrives, and the wind moves through the valles the  c
注意：两边都超出了训练长度（训练 block=32）：GPT-2 在查没训练过的位置行，RoPE 不越界
      但远距离同样会退化；这段样例对两者都偏苛刻（loss 对照只发生在长度 32 内，不受影响）。

--- 7. KV cache：每生成一个 token 要多存多少字节 ---
公式：2 · n_layer · n_kv_head · head_dim · sizeof(dtype)（K、V 各一份）
toy 配置 fp32：GPT-2 MHA(4 组 KV) 1,024 B/token | Llama GQA(2 组 KV) 512 B/token
toy 配置 fp16：GPT-2 512 B/token | Llama 256 B/token
真实 Llama-3-8B（32 层, 8 组 KV, head_dim 128, fp16）：131,072 B/token (128 KiB)
  若用 GPT-3 式 MHA（32 组 KV）同规模：524,288 B/token (512 KiB)，正好是 GQA 的 4 倍
  8192 token 上下文：GQA 1 GiB | MHA 4 GiB
[OK] step06 的 KVCache.nbytes() = 32,768 B，与公式（512 B/token x 64 位置）一致

--- 8. 结论（只基于上面的实测数字）---
参数量（实测）：GPT-2 104,960 → Llama 96,320（-8,640，-8.2%）。
玩具规模 loss（最后 50 步均值）：GPT-2 0.1131 → Llama 0.0990（-0.0141）；固定评测 batch：0.1107 → 0.0996。
每 token KV（公式）：toy fp32 1,024 B → 512 B；真实 8B 规模 fp16 512 KiB → 128 KiB（8K 上下文 4 GiB → 1 GiB）。
玩具速度：2.6 → 3.2 ms/step（CPU，只代表这台机器）。
判断：Llama 略好 0.0141，方向对但幅度小，离「显著」还差得远。
两件事要分开：玩具 loss 受 438 字符/500 步限制不能外推；Llama 在真实规模的主要收益是 KV 显存与长上下文吞吐（上面 4 倍），不是这 0.0x 的 loss。

一句话记住：
  同预算下 Llama 赢在参数更少、KV 显存更小，玩具 loss 差距别当真；但玩具规模上
  training 反而略慢（GQA 的 repeat + SwiGLU 三矩阵固定开销），真实规模会被吞掉。

[OK] 全部检查通过
```

## 08

```text
$ .venv/bin/python llama_steps/step08_parity.py

W1030 18:24:03.093000 86513 torch/distributed/elastic/multiprocessing/redirects.py:35] NOTE: Redirects are currently not supported in MacOs.

--- 1. 同一套权重，两个实现，同一种输入 ---
我们的参数量           = 100,672
HF 的参数量            = 100,672
输入 idx               = (2, 16)   <- (B, T)
我们的 logits          = (2, 16, 64)   <- (B, T, V)
HF 的 logits           = (2, 16, 64)   <- (B, T, V)
max|diff|            = 1.192e-07
[OK] 逐位一致（剩下的 1e-7 只是浮点累加顺序不同，不是实现差异）

--- 2. 为什么「数学上等价」还不够（三个常见写错方式） ---
去掉 RoPE（不旋转 q/k）                       max|diff| = 5.323e-03
用 interleaved 配对（HF 用 half-split）      max|diff| = 6.281e-03
repeat_kv 用整块复制而不是交错                   max|diff| = 4.061e-01
[OK] 每个错误都被放大到 1e-3 以上，和对齐误差 1e-7 差了 4 个数量级

--- 3. 真实模型的 KV cache 账本（fp16，8192 上下文） ---
模型                       层   kv头    D    每 token     8k 上下文   说明
GPT-2 small                12     12   64   36.00 KiB    288.00 MiB   MHA，kv 头和 query 头一样多
Llama-3-8B                 32      8  128  128.00 KiB      1.00 GiB   GQA，8 组 kv 头
Llama-3-8B 若用 MHA        32     32  128  512.00 KiB      4.00 GiB   假设不改 GQA
Llama-3-70B                80      8  128  320.00 KiB      2.50 GiB   GQA，80 层
SmolLM2-135M               30      3   64   22.50 KiB    180.00 MiB   step09 要加载的真实模型
[OK] 代码算出的数字和公式一致（2 × 层数 × kv 头 × D × 上下文 × 2 字节）
     结论：8B 模型在 8k 上下文下，光 KV cache 就要 1 GiB；
     把 32 个 kv 头换成 8 个，直接省掉 3/4 —— 这就是 GQA 被所有新模型采用的原因。

--- 4. GPT-2 → Llama 改造清单（回顾八步） ---
位置        GPT-2                     Llama                       好处
归一化      LayerNorm (γ, β = 2C)     RMSNorm (只有 γ = C)        少一半参数、少一次减均值
FFN         GELU, C→4C→C              SwiGLU, C→8/3C→C            同参数量下门控更强
位置编码    学习式 wpe (T×C 参数)     RoPE (0 参数，转 q/k)       可外推、对 KV cache 友好
注意力      MHA (kv 头 = query 头)    GQA (kv 头 = 1/4~1/8)       KV cache 显存降 4–8 倍
线性层 bias 有                        全部去掉                    少参数、便于算子融合
dropout     0.1                       0.0                         数据量大时收益低，还省一次 mask
输出层      和输入嵌入共享            默认不共享                  小模型仍可共享（如 SmolLM2）

一句话记住：
  Llama = GPT-2 把「位置、归一化、门控、KV 头」四处换成更省的写法，
  换完参数量差不多，但同样显存能跑更长的上下文。

[OK] 全部检查通过
```

## 09

```text
$ .venv/bin/python llama_steps/step09_real_llama.py

W1030 18:24:04.603000 86515 torch/distributed/elastic/multiprocessing/redirects.py:35] NOTE: Redirects are currently not supported in MacOs.

--- 1. 官方 config.json → 我们的 LlamaConfig ---
hidden_size            = 576   <- → n_embd
intermediate_size      = 1,536   <- → ffn_hidden（8/3·C 取整）
num_hidden_layers      = 30   <- → n_layer
num_attention_heads    = 9   <- → n_head（不是 2 的幂也没关系）
num_key_value_heads    = 3   <- → n_kv_head（GQA 9:3）
head_dim               = 64   <- = n_embd / n_head
rope_theta             = 100000.0   <- Llama-3 系用 1e5（不是 GPT-2 时代的 1e4）
rms_norm_eps           = 1e-05
tie_word_embeddings    = True   <- 小模型省参数，输入输出共享
[OK] 每个字段都真的映射进 LlamaConfig（逐项断言），不是「看起来能对上」

--- 2. 把 272 个张量装进 MiniLlama ---
权重张量数             = 272   <- safetensors 里的张量个数
映射后的张量数         = 273   <- 多出来的是共享的 lm_head
我们的参数量           = 134,515,008
加载耗时               = 1.32 s
[OK] 没有 missing / unexpected 的键（strict=True 通过）

--- 3. 和 HuggingFace 的同款模型对比 ---
prompt                 = 'The capital of France is'
token ids              = [[504, 3575, 282, 4649, 314]]   <- (B, T)
max|diff|              = 4.768e-05
我们 top-5             = [' the', ' Paris', ' located', ' called', ' a']
官方 top-5             = [' the', ' Paris', ' located', ' called', ' a']
[OK] 30 层真实权重下，logits 差在 1e-4 量级，top-5 完全一致

--- 4. 用 KV cache 生成，和官方 generate 对比 ---
[我们的贪心生成] 'The capital of France is the capital of the country.\n\nThe capital of France'
[官方贪心生成] 'The capital of France is the capital of the country.\n\nThe capital of France'
[OK] 12 个 token 的贪心生成完全相同

[温度 0.8 + top-k 40 采样]
The capital of France is the French language. The language of the French, although not one-millionth as common as English, is a very useful and used language, and is used in almost all the areas of the world

--- 5. 真实速度与显存 ---
prefill 耗时           = 17.7 ms   <- prompt 5 个 token
decode 速度            = 10.3 ms/token   <- ≈ 97 token/s
KV cache @256          = 11.2 MiB   <- fp32，demo 上下文
KV cache @8192         = 180.0 MiB   <- fp16，真实上下文长度
     若 9 个 query 头各自持有 kv（MHA），同样 8k 上下文要 540.0 MiB，是 GQA 的 3 倍

一句话记住：
  真实 Llama 权重能直接装进你自己写的 MiniLlama，是因为前面八步每一步
  都和官方定义对上了；能生成通顺英文，就是「实现正确」最直观的证据。

[OK] 全部检查通过
```

## 全量跑一遍

```text
$ time bash llama_steps/run_all.sh
...
全部完成。输出在 llama_steps/runs/，图片在 llama_steps/figs/
TOTAL 13.995 s
```

各步单独计时（同一台机器、串行、各跑一次）：

| 步骤 | 脚本                          | 耗时（秒） |
| ---- | ----------------------------- | ---------- |
| 01   | `step01_rmsnorm.py`           | 0.54       |
| 02   | `step02_swiglu.py`            | 0.48       |
| 03   | `step03_rope.py`              | 0.78       |
| 04   | `step04_gqa.py`               | 0.49       |
| 05   | `step05_block.py`             | 0.48       |
| 06   | `step06_model.py`             | 0.49       |
| 07   | `step07_train.py`             | 4.25       |
| 08   | `step08_parity.py`            | 1.57       |
| 09   | `step09_real_llama.py`        | 5.03       |
| 合计 | `bash llama_steps/run_all.sh` | 14.00      |

说明：9 个脚本全部 exit 0；各步单独跑的总和约 14.1 秒，与 run_all 的 14.00 秒差 0.1 秒，
属于进程启动和机器抖动。计时受机器状态影响，只用来比较各步的相对快慢。
