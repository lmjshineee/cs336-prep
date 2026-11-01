## 01

cs336-prep-code ❯ python3 step01_tokenizer.py
vocab_size = 30
vocab      =
 ,.ANSTabcdefghiklmnoprstuvwy

text   -> 'A small cat'
ids    -> [4, 1, 24, 19, 8, 18, 18, 1, 10, 8, 25]
decode -> 'A small cat'

[OK] 编码 -> 解码 往返一致

## 02

temp/cs336-prep-code/transformer_steps
cs336-prep-code ❯ python3 step02_embeddings.py
ids.shape        = (1, 11)   <- (B, T)
token_emb.weight = (30, 16)   <- (V, C)
pos_emb.weight   = (64, 16)   <- (block_size, C)
x.shape          = (1, 11, 16)   <- (B, T, C)

[OK] 嵌入 = token 向量 + 位置向量

## 03
temp/cs336-prep-code/transformer_steps
cs336-prep-code ❯ python3 step03_attention.py
x.shape       = (1, 8, 8)   <- (B, T, C)
scores.shape  = (1, 8, 8)   <- (B, T, T)
weights.shape = (1, 8, 8)   <- (B, T, T)
weights 每行和 = [1.0, 1.0, 1.0] ...
out.shape     = (1, 8, 8)   <- (B, T, C)，和输入同形

第 0 个 token 的注意力权重：
  -> token   'c'  0.0736
  -> token   'a'  0.0791
  -> token   't'  0.2332
  -> token   ' '  0.0653
  -> token   's'  0.1981
  -> token   'i'  0.1099
  -> token   't'  0.1735
  -> token   's'  0.0673


## 04

temp/cs336-prep-code/transformer_steps
cs336-prep-code ❯ python3 step04_causal_mask.py
T=8 的掩码（1=可见）：
    1 0 0 0 0 0 0 0
    1 1 0 0 0 0 0 0
    1 1 1 0 0 0 0 0
    1 1 1 1 0 0 0 0
    1 1 1 1 1 0 0 0
    1 1 1 1 1 1 0 0
    1 1 1 1 1 1 1 0
    1 1 1 1 1 1 1 1

掩码后第 2 个 token 的权重（只应非零在前 3 个）：
    [0.2957, 0.3743, 0.33, 0.0, 0.0, 0.0, 0.0, 0.0]
[OK] 上三角全部为 0
[OK] 修改未来 token，不影响过去位置的输出（因果性）

## 05
temp/cs336-prep-code/transformer_steps
cs336-prep-code ❯ python3 step05_multihead.py
x                  = (1, 8, 64)
n_head H           = 4
head_dim D = C / H = 16 (C=64)
qkv(x).split       = 3 x (1, 8, 64)
view+transpose     = (1, 4, 8, 16)   <- (B,H,T,D)
scores             = (1, 4, 8, 8)   <- (B,H,T,T)
out                = (1, 8, 64)   <- (B,T,C)

[OK] 多头 + 因果掩码，未来 token 不影响过去


## 06

temp/cs336-prep-code/transformer_steps
cs336-prep-code ❯ python3 step06_block.py
block 输入 = (1, 8, 64)
block 输出 = (1, 8, 64)   <- 形状不变，可以直接串联

attention 参数 = 16,384   (~4*C^2 = 16,384)
ffn 参数       = 33,088   (~8*C^2 = 32,768)
ln 参数        = 256
block 合计     = 49,728

[OK] 子层输出为 0 时，block 退化为恒等映射（残差的作用）

## 07

temp/cs336-prep-code/transformer_steps
cs336-prep-code ❯ python3 step07_model.py
vocab_size = 30
params     = 104,576
  embeddings+head = 4,992
  blocks          = 99,456
  ln_f            = 128

idx.shape     = (1, 11)   <- (B, T)
logits.shape  = (1, 11, 30)   <- (B, T, V)
loss          = 3.3716   (随机初始化时约等于 ln(V) = 3.4012)

[OK] 输出形状正确，且 lm_head 与 token_embedding 共享权重

## 08

temp/cs336-prep-code/transformer_steps
cs336-prep-code ❯ python3 step08_train.py
device=cpu  vocab=30  params=104,576
语料长度=438 tokens

step  50 | loss 1.6446
step 100 | loss 1.0377
step 150 | loss 0.7236
step 200 | loss 0.4446
step 250 | loss 0.2215
step 300 | loss 0.1989

--- 等价性检查 ---
prefill: max|diff| = 0.00e+00
step  1:   max|diff| = 2.86e-06
step  2:   max|diff| = 2.86e-06
step  3:   max|diff| = 1.43e-06
step  4:   max|diff| = 1.91e-06
step  5:   max|diff| = 1.91e-06
step  6:   max|diff| = 3.34e-06
step  7:   max|diff| = 1.91e-06
step  8:   max|diff| = 2.15e-06
[OK] 缓存版与全量重算的 logits 逐步一致

[OK] 贪心解码下两种实现生成完全相同的序列

--- 生成结果（KV cache）---
Spring arrives, and the wind moves through the v

缓存大小 = 2 * n_layer * B * T_max * C * 4B = 49,152 bytes (48 KiB)

生成 24 个 token：naive 5.5 ms  |  cached 3.6 ms
（玩具模型上 Python 开销占比大，加速比远小于理论上限）

[OK] 形状检查 + 因果性检查通过
