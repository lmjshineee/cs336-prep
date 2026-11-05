# model_lab —— 从零写一个 pre-norm Transformer（CS336 A1 §3.3–3.5 教学版）

> **这是教学实现，不是 A1 的提交答案。** API 与官方 `tests/adapters.py` 故意不同（更「教学」的
> 类名/参数、额外的形状断言与自检），核心目的是「读得懂 + 能和官方参考权重逐位对上」，
> 而不是替交作业。官方仓库 `assignment1-basics` 只被**只读**地用来做数值对齐。

---

## 1. 文件表

| 文件 | 说明 |
| --- | --- |
| `toy_transformer.py` | 核心实现：`softmax` / `sdpa` / `silu` + `Linear` / `Embedding` / `RMSNorm` / `SwiGLU` / `RotaryPositionalEmbedding` / `CausalMultiHeadSelfAttention` / `TransformerBlock` / `TransformerLM`，`python model_lab/toy_transformer.py` 可跑自检 |
| `test_toy_transformer.py` | 双模式测试：45 个 `test_*` 函数 = 覆盖 SPEC §6 的 43 项（含 13 个官方快照 parity）+ 1 个 `token_positions` 形状回归测试 |
| `README.md` | 本文件 |
| `SPEC.md` | Lead 维护的实现契约（API、数值约定、测试清单） |
| `accounting.py` / `results/` | 另一位作者：参数量/显存/FLOPs 核算与结果 JSON |
| `exp_overfit.py` | 另一位作者：单 batch 过拟合实验 |
| `lead_spotcheck.py` | Lead 的独立复核脚本：不 import 本 lab 的测试，直接用官方快照抽查 5 个模块（`A1_REPO` 可指定官方仓库） |
| `LAB_NOTES.md` | Lead 维护：实测数字、命令、踩坑、限制 |

---

## 2. 快速开始

```bash
cd /Users/yunni/Projects/cs336-prep-code

# 自检：建个小模型，检查形状 / 参数量 / state_dict 键名 / 反向传播
.venv/bin/python model_lab/toy_transformer.py

# 全部测试（直接跑，不需要 pytest；失败会 exit 1）
.venv/bin/python model_lab/test_toy_transformer.py

# 官方仓库不在默认路径时，用 A1_REPO 指定（parity 会真跑而不是 SKIP）
A1_REPO=/path/to/assignment1-basics .venv/bin/python model_lab/test_toy_transformer.py

# Lead 的独立复核（不 import 测试代码，自己复算 5 个模块）
.venv/bin/python model_lab/lead_spotcheck.py

# 顺便跑一下实验脚本（绘图需要 MPLCONFIGDIR，避免写到只读的 HOME）
MPLCONFIGDIR=$PWD/.tmp/mpl .venv/bin/python model_lab/accounting.py
MPLCONFIGDIR=$PWD/.tmp/mpl .venv/bin/python model_lab/exp_overfit.py --max-seconds 90
```

环境：`/Users/yunni/Projects/cs336-prep-code/.venv/bin/python`（Python 3.14.3 + torch 2.14.1）。
所有脚本都用 `torch.set_num_threads(2)` 限流，单条命令在秒级完成。

最小用法：

```python
import sys; sys.path.insert(0, "model_lab")
import torch
from toy_transformer import TransformerLM

model = TransformerLM(
    vocab_size=10_000, context_length=16, d_model=64,
    num_layers=3, num_heads=4, d_ff=128, rope_theta=10_000.0,
)
logits = model(torch.randint(0, 10_000, (4, 12)))   # (4, 12, 10000)
print(model.num_parameters())
```

官方快照权重可以直接 `strict=True` 加载（键名逐字一致）：

```python
sd = {k.replace("_orig_mod.", ""): v for k, v in torch.load(
    "/Users/yunni/Projects/assignment1-basics/tests/fixtures/ts_tests/model.pt",
    map_location="cpu", weights_only=True).items()}
model.load_state_dict(sd, strict=True)   # 缺失/多余键都会报错
```

---

## 3. 测试覆盖点（SPEC §6 的 43 项 → 测试函数）

A. 手算 / 公式（10 项）

| # | 覆盖点 | 测试函数 |
| --- | --- | --- |
| 1 | softmax 行和为 1、手算一致、负 dim | `test_A01_softmax_hand_computed_and_negative_dim` |
| 2 | softmax 数值稳定（大输入 / +1e4 不变） | `test_A02_softmax_numerical_stability` |
| 3 | softmax 平移不变性 | `test_A03_softmax_shift_invariance` |
| 4 | sdpa 3D 与手写公式一致（d_v ≠ d_k） | `test_A04_sdpa_matches_manual_formula_3d` |
| 5 | sdpa mask 语义（False 权重 0、True 和为 1） | `test_A05_sdpa_mask_semantics` |
| 6 | sdpa 4D 与逐 head 3D 循环一致 | `test_A06_sdpa_4d_matches_per_head_loop` |
| 7 | rmsnorm 与公式一致（随机 gain） | `test_A07_rmsnorm_matches_formula` |
| 8 | rope 二维手算旋转（交错配对） | `test_A08_rope_2d_hand_computed` |
| 9 | rope 相对性：点积只依赖 `i-j` | `test_A09_rope_relative_position_only` |
| 10 | silu 对 `F.silu`；swiglu 对手写公式 | `test_A10_silu_and_swiglu_match_formula` |

B. 形状 / 接口（8 项）

| # | 覆盖点 | 测试函数 |
| --- | --- | --- |
| 11 | Linear 权重 `(out,in)`、无 bias、`x @ W.T` | `test_B11_linear_weight_layout_and_forward` |
| 12 | Embedding 查表 = `weight[ids]`，1D/2D/3D ids | `test_B12_embedding_lookup_arbitrary_rank` |
| 13 | RMSNorm 保持输入 dtype（bf16/fp16） | `test_B13_rmsnorm_preserves_dtype` |
| 14 | SwiGLU 默认 d_ff：512→1344、1600→4288、64 倍数 | `test_B14_swiglu_default_d_ff` |
| 15 | MHA 输出形状；num_heads 不整除抛错 | `test_B15_mha_shape_and_head_divisibility` |
| 16 | Block 输出形状不变（有无 RoPE） | `test_B16_block_shape_preserved` |
| 17 | LM logits `(B,T,V)`；截断 T=6 可跑 | `test_B17_lm_logits_shape_and_truncated` |
| 18 | LM `T > context_length` 抛错 | `test_B18_lm_rejects_sequence_over_context` |

C. 性质（8 项）

| # | 覆盖点 | 测试函数 |
| --- | --- | --- |
| 19 | causal：改未来 token 不影响过去输出 | `test_C19_causal_future_tokens_do_not_change_past` |
| 20 | causal：注意力权重上三角为 0 | `test_C20_causal_attention_weights_upper_triangle_zero` |
| 21 | RoPE 只转 q/k（v 路径与不旋转参考一致） | `test_C21_rope_only_applied_to_qk` |
| 22 | RoPE 位置平移，attention 输出不变 | `test_C22_rope_position_shift_keeps_attention` |
| 23 | 单头与手写单头注意力一致 | `test_C23_single_head_equals_manual_attention` |
| 24 | RMSNorm 正尺度不变（x 与 3x） | `test_C24_rmsnorm_is_scale_invariant` |
| 25 | tie_weights 共享存储、参数量变少 | `test_C25_tie_weights_shares_parameters` |
| 26 | 末 token loss 反传，所有参数梯度有限 | `test_C26_all_parameters_receive_finite_gradients` |

> 额外回归测试（**不计入 43 项**，编号 C27 只是排序方便）：`test_C27_mha_accepts_batch_token_positions`
> 覆盖 MHA 的 `token_positions` 四种形状 `(T,)` / `(1,T)` / `(B,T)` / `(1,1,T)`，并验证 `(B,T)`
> 是逐样本生效。这个形状 bug 一度被 `(1,T)` 的靠右对齐广播掩盖，故单独回归。

D. 初始化（4 项）

| # | 覆盖点 | 测试函数 |
| --- | --- | --- |
| 27 | Linear init std ≈ `sqrt(2/(in+out))`、±3σ | `test_D27_linear_init_scale` |
| 28 | Embedding init std ≈ 1、±3 | `test_D28_embedding_init_scale` |
| 29 | RMSNorm gain 全 1；无 bias 参数 | `test_D29_rmsnorm_gain_ones_and_no_bias` |
| 30 | state_dict 键集合 = 官方、strict 可加载 | `test_D30_state_dict_keys_match_official` |

E. 官方快照 parity（12 项 / 13 个 npz）

| # | 快照 | 测试函数 |
| --- | --- | --- |
| 31 | `test_linear` | `test_E31_parity_linear` |
| 32 | `test_embedding` | `test_E32_parity_embedding` |
| 33 | `test_swiglu` | `test_E33_parity_swiglu` |
| 34 | `test_rmsnorm` | `test_E34_parity_rmsnorm` |
| 35 | `test_rope` | `test_E35_parity_rope` |
| 36 | `test_scaled_dot_product_attention` | `test_E36_parity_sdpa` |
| 37 | `test_4d_scaled_dot_product_attention` | `test_E37_parity_sdpa_4d` |
| 38 | `test_multihead_self_attention` | `test_E38_parity_mha` |
| 39 | `test_multihead_self_attention_with_rope` | `test_E39_parity_mha_rope` |
| 40 | `test_positionwise_feedforward` | `test_E40_parity_positionwise_feedforward` |
| 41 | `test_transformer_block` | `test_E41_parity_transformer_block` |
| 42 | `test_transformer_lm` + `test_transformer_lm_truncated_input` | `test_E42_parity_transformer_lm`、`test_E42_parity_transformer_lm_truncated` |

F. 端到端（1 项）

| # | 覆盖点 | 测试函数 |
| --- | --- | --- |
| 43 | 小 LM + CrossEntropy 反传 30 步，loss 下降 | `test_F43_one_step_training_reduces_loss` |

> 45 个测试函数 = 43 项对应的 44 个函数（第 42 项合并了「全量」与「截断」两个快照，
> 所以 44 个函数恰好覆盖 43 项，无删减）+ 1 个回归测试 `test_C27_*`。

### 实测 parity maxdiff（本机，`A1_REPO=/Users/yunni/Projects/assignment1-basics`）

命令：`.venv/bin/python model_lab/test_toy_transformer.py`（2026-11-05，torch 2.14.1，2 线程）

| 快照 | maxdiff | 容差 |
| --- | --- | --- |
| `test_linear` | 8.94e-07 | atol 1e-4 |
| `test_embedding` | 0.00e+00 | atol 1e-4 |
| `test_swiglu` | 1.43e-06 | atol 1e-5 |
| `test_rmsnorm` | 1.25e-06 | atol 1e-4 |
| `test_rope` | 1.48e-06 | atol 1e-5 |
| `test_scaled_dot_product_attention` | 9.54e-07 | atol 1e-5 |
| `test_4d_scaled_dot_product_attention` | 9.54e-07 | atol 1e-5 |
| `test_multihead_self_attention` | 5.36e-07 | atol 1e-5 |
| `test_multihead_self_attention_with_rope` | 2.98e-07 | atol 1e-5 |
| `test_positionwise_feedforward` | 1.43e-06 | atol 1e-5 |
| `test_transformer_block` | 2.38e-06 | atol 1e-4 |
| `test_transformer_lm` | 1.00e-05 | atol 1e-4 / rtol 1e-2 |
| `test_transformer_lm_truncated_input` | 1.05e-05 | atol 1e-4 |

全部 ≤ 1.05e-05，比 SPEC 的 1e-4 门槛低一个量级；测试会打印每一行，方便写进 LAB_NOTES。

---

## 4. 两个实验看什么

> 实验由 `accounting.py` / `exp_overfit.py` 负责，**数字以 `results/*.json` 和实际命令输出为准**；
> 下面的「看什么」只描述目的，不写未跑过的数字。

1. **`accounting.py`（解析式核算）**：不看代码也能算的「账本」——
   参数量（embedding / 每层 attn `4d²` / 每层 ffn `3d·d_ff` / RMSNorm gain / lm_head）、
   fp32 权重显存、前向 FLOPs（`2mnp` 规则）。重点看三件事：
   - 与官方 `model.pt` 的参数量总数是否一致（对账 checks）；
   - GPT-2 small→XL 的 FLOPs 随 `d_model`/层数的增长比例，以及 ctx 1024→16384 时
     attention 项 `4T²d` 如何从「可忽略」变成「主导」；
   - `d_ff` 是否严格是 64 的倍数。
2. **`exp_overfit.py`（单 batch 过拟合）**：只回答一个问题——**这个实现能不能训**
   （loss 是否稳定下降、有没有 nan/爆炸、2 线程下吞吐与峰值内存）。
   固定 8×64 的一个 batch、AdamW lr=3e-3、最多 300 步或 90 秒，记录 loss 曲线并 greedy
   解码 64 字节。注意：loss 降到很低只证明「可训练 + 有记忆能力」，
   **不证明任何语言能力**（数据只有一个 batch）。

---

## 5. 实现要点 & 已知限制

### 5.1 已对齐的数值约定（细节见 `SPEC.md` §4）

- `Linear` 存 `(out, in)`，`x @ W.T`，无 bias；初始化 `trunc_normal(std=sqrt(2/(in+out)))`。
- `RMSNorm` 内部 fp32、`x/sqrt(mean(x²)+eps)*g`、eps=1e-5，输出回原 dtype。
- RoPE **交错配对** `(0,1),(2,3),…`，角度 `pos·theta^(-2k/d_k)`，`R=[[c,-s],[s,c]]`，
  只作用于 q/k；cos/sin 是 `persistent=False` 的 buffer，不进 state_dict。
- `sdpa` 的 mask 是 bool，**True=可见**，`masked_fill(~mask, -inf)`。
- MHA 缩放用 `sqrt(head_dim)`；`num_heads` 必须整除 `d_model`；causal mask 用 `torch.tril`。
- LM 默认 **untied**（`lm_head` 与 `token_embeddings` 独立）；`tie_weights=True` 时才共享。
- 所有层的 RoPE 是同一个实例；`token_positions` 默认 `0..T-1`，支持 `(T,)` / `(1,T)` / `(B,T)` /
  `(1,1,T)`（详见 §5.2 第 3 条）。
- `SwiGLU(d_model)` 的默认 `d_ff` 是 `8/3·d_model` 取最近的 64 倍数，取整用 **half-up**
  （`64 * floor(8d/(3·64) + 0.5)`，最小 64），与 `accounting.py` 的公式逐值一致；例如
  512→1344、768→2048、1024→2752、1280→3392、1600→4288，边界 `d_model=60`→192。

### 5.2 已知限制（故意保留，教学取舍）

1. **只支持 CPU/单卡的小规模场景**：没有 KV cache、没有 flash-attention、没有 `torch.compile`；
   推理是「每次重算整段」。
2. **RoPE 不支持超长外推**：cos/sin 表长度 = `context_length`，位置 ≥ 表长直接 IndexError；
   `T > context_length` 在 LM 层就报错，不做动态扩展/NTK 插值。
3. **`token_positions` 的形状**：`RoPE.forward` 本身是「靠右对齐广播」，MHA 额外把
   `(..., T)` 形式（leading 维与 x 的 batch 维一一对应）在 head 维前插一个 1，所以
   x=`(B,T,C)` 时支持 **`(T,)`、`(1,T)`、`(B,T)`、`(1,1,T)`** 四种位置形状。
   `(B,H,T)`（每个 head 不同位置）**不是这里支持的形状**：3 维输入 x=`(B,T,C)` 的 batch 维
   只有一个，位置要么全体共享 `(T,)`、要么逐样本 `(B,T)`；`(B,H,T)` 虽然能在 H 恰好等于
   `num_heads` 时「碰巧」广播成「每个 head 各用一组位置」，但这个语义没有测试覆盖，不要依赖
   （只要 H 与 `num_heads` 不等就会直接报错）。4 维及以上输入若要逐 batch 给位置，请自己补 1
   （如 `(B1,B2,1,T)`）。
4. **没有 dropout、没有 bias、没有 LayerNorm 备选**：只实现测验要求的最小集合。
5. **`num_heads=1` 的 MHA 与手写单头等价是测试项**，但这不代表「多头 = 单头」——
   多头是在 head 维上独立做 attention，表达力来自投影矩阵的切分。
6. 测试的 parity 依赖外部仓库；仓库不存在时会 **SKIP 而不是失败**，此时不构成数值验证，
   请用 `A1_REPO` 指过去后重跑。
7. 本目录**不修改** `assignment1-basics`（只读它的权重/快照/配置），也不提交 git。
