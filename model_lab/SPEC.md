# model_lab SPEC — A1 Transformer 教学实现契约

> 版本：2026-11-05
> 维护：Lead。**所有实现者先读本文件，再动手。**
> 本文件是唯一契约：接口、数值约定、官方快照对齐配方、测试清单、实验口径、文件所有权。

---

## 0. 目标与边界

做一套和 `bpe_lab/` 同规格的教学 lab：把 CS336 A1 handout §3.3–3.5 的 pre-norm Transformer
从零写出来，**能跑、能测、能算账**，并为下一步 training_lab 提供模型。

**不是** A1 提交答案：

- API 与官方 `tests/adapters.py` 故意不同（类名/参数名更"教学"，额外提供 shape 断言与自检）；
- **不修改** `/Users/yunni/Projects/assignment1-basics`（只读，用它的快照做数值对齐）；
- 不联网、不安装新依赖、不提交 git。

交付物：

| 文件 | 负责人 | 说明 |
| --- | --- | --- |
| `model_lab/toy_transformer.py` | impl | 核心实现 |
| `model_lab/test_toy_transformer.py` | impl | 双模式测试（直接跑 + pytest 兼容） |
| `model_lab/README.md` | impl | 快速开始、覆盖点、限制 |
| `model_lab/accounting.py` | exp | 参数量/显存/FLOPs 核算 |
| `model_lab/exp_overfit.py` | exp | 单 batch 过拟合实验 |
| `model_lab/results/*.json` | exp | 机器可读结果 |
| `model_lab/LAB_NOTES.md` | Lead | 实测数字、命令、踩坑、限制 |
| `MODEL_TRANSFORMER_TUTORIAL.md` | scribe | 长教程 |
| `README.md`（根） | Lead | 目录表加一行 |

---

## 1. 环境与热预算（硬约束）

- 解释器：`/Users/yunni/Projects/cs336-prep-code/.venv/bin/python`（Python 3.14.3，torch 2.14.1）。
- **所有可能超过 1 秒的脚本**开头必须 `torch.set_num_threads(2)`；单条命令 **< 120 秒**。
- `exp_overfit.py` 必须有 `--max-seconds`（默认 90）硬停，超时保存已跑结果。
- 不联网；`transformers` 只在必要时用，且不加载大模型。
- 画图（可选）需 `MPLCONFIGDIR=$PWD/.tmp/mpl`。
- 官方仓库默认路径 `/Users/yunni/Projects/assignment1-basics`，可用环境变量 `A1_REPO` 覆盖。
  **只读**：不写、不改测试、不改 fixtures。

---

## 2. 文件所有权（写入范围）

| 角色 | 可写文件 |
| --- | --- |
| impl | `model_lab/toy_transformer.py`、`model_lab/test_toy_transformer.py`、`model_lab/README.md` |
| exp | `model_lab/accounting.py`、`model_lab/exp_overfit.py`、`model_lab/results/` |
| scribe | `MODEL_TRANSFORMER_TUTORIAL.md` |
| Lead | `model_lab/SPEC.md`、`model_lab/LAB_NOTES.md`、根 `README.md` |

**不要越界写别人的文件。** 需要对方改接口时，先在报告里说明，由 Lead 协调。

---

## 3. 公共 API 契约

```python
# ---------- functional ----------
def softmax(x: Tensor, dim: int) -> Tensor                  # 数值稳定：先减 max
def scaled_dot_product_attention(q, k, v, mask=None) -> Tensor
def silu(x: Tensor) -> Tensor                               # x * sigmoid(x)

# ---------- modules（device=None, dtype=None 一律保留） ----------
class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None)
    # self.weight: nn.Parameter, shape (out_features, in_features)
    def forward(self, x) -> Tensor                          # x @ weight.T

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None)
    # self.weight: nn.Parameter, shape (num_embeddings, embedding_dim)
    def forward(self, token_ids) -> Tensor

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None)
    # self.weight: nn.Parameter, shape (d_model,)，初始化为 1
    def forward(self, x) -> Tensor

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int | None = None, device=None, dtype=None)
    # d_ff=None 时取「8/3 * d_model 最接近的 64 的倍数」，最小 64
    # self.w1: Linear(d_model, d_ff); self.w2: Linear(d_ff, d_model); self.w3: Linear(d_model, d_ff)
    def forward(self, x) -> Tensor                          # w2(silu(w1 x) * (w3 x))

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None)
    # cos/sin 用 register_buffer(..., persistent=False)，形状 (max_seq_len, d_k//2)
    def forward(self, x, token_positions) -> Tensor         # x: (..., T, d_k)

class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, rope=None, device=None, dtype=None)
    # self.q_proj/k_proj/v_proj/output_proj: Linear(d_model, d_model)
    # 默认 causal（下三角 True=可见）；RoPE 只加在 q/k；head 维当 batch
    def forward(self, x, token_positions=None) -> Tensor

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, rope=None, device=None, dtype=None)
    # self.attn; self.ln1; self.ln2; self.ffn
    # y = x + attn(rms(x)); out = y + ffn(rms(y))
    def forward(self, x, token_positions=None) -> Tensor

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, context_length, d_model, num_layers,
                 num_heads, d_ff, rope_theta=10000.0, tie_weights=False,
                 device=None, dtype=None)
    # self.token_embeddings; self.layers (ModuleList); self.ln_final; self.lm_head
    # 所有层共享同一个 RoPE 实例
    def forward(self, token_ids) -> Tensor                  # (B,T) -> (B,T,V)
    def num_parameters(self) -> int
```

**state_dict 键名必须与官方参考完全一致**（parity 测试要直接 `load_state_dict`）：

```
token_embeddings.weight
layers.{i}.attn.q_proj.weight / k_proj / v_proj / output_proj
layers.{i}.ln1.weight / ln2.weight
layers.{i}.ffn.w1.weight / w2 / w3
ln_final.weight
lm_head.weight
```

因此**模块属性名也必须一致**（`self.attn.q_proj`、`self.ffn.w1`、`self.ln1` …）。

---

## 4. 数值与形状约定（已用官方快照反推验证）

| 项 | 约定 |
| --- | --- |
| Linear | `W` 存 `(out, in)`；`forward = x @ W.T`；无 bias |
| Linear 初始化 | `trunc_normal_(std=sqrt(2/(in+out)), a=-3σ, b=3σ)` |
| Embedding 初始化 | `trunc_normal_(std=1.0, a=-3, b=3)` |
| RMSNorm | `x / sqrt(mean(x²) + eps) * g`；先 upcast 到 fp32，再回原 dtype；eps=1e-5 |
| SwiGLU | `W2(SiLU(W1 x) ⊙ (W3 x))`；d_ff 默认最近 64 倍数（512→1344，1600→4288） |
| RoPE | 交错配对 `(x0,x1),(x2,x3),...`；角度 `pos * theta^(-2k/d_k)`，`k=0..d_k/2-1`；MHA 里 `d_k = d_model // num_heads`；只加 q/k；cos/sin 缓存 `(max_seq_len, d_k//2)` |
| RoPE 位置 | `token_positions` 形状 `(..., T)`，可 `(T,)`、`(1,T)`、`(B,T)`；用高级索引按位置取表；靠右对齐广播 |
| SDPA | `softmax(q @ k.T / sqrt(d_k), dim=-1) @ v`；mask 为 bool，**True=可见**；`masked_fill(~mask, -inf)`；支持任意 batch 维 + mask 广播 |
| MHA | 内置 causal mask（`torch.tril`）；`num_heads` 必须整除 `d_model`；head 维当 batch |
| Block | pre-norm；`x + attn(rms(x))`，`y + ffn(rms(y))` |
| LM | emb → N×block → `ln_final` → `lm_head`；`T <= context_length` 需断言；**默认 untied**（`tie_weights=False`） |
| dtype | 运算保持输入 dtype；RoPE 缓存 fp32，用前 cast；RMSNorm 内部 fp32 |

**已验证的参考实现 maxdiff**（Lead 用官方权重手写参考跑出，可作为 parity 的预期量级）：

| 输出 | maxdiff |
| --- | --- |
| MHA（causal） | 4.8e-7 |
| SDPA（3D + mask） | 9.5e-7 |
| RoPE（交错） | 1.5e-6 |
| TransformerBlock | 2.0e-6 |
| TransformerLM | 8.6e-6 |

若你的 parity maxdiff 明显大于 `1e-4`，先查：mask 方向、RoPE 是否前后半拆分、per-head `d_k`、
lm_head 是否被错误 tie、block 的 pre-norm 顺序。

---

## 5. 官方快照对齐配方（只读）

路径（`A1_REPO` 可覆盖）：

- 权重：`{A1_REPO}/tests/fixtures/ts_tests/model.pt`（`_orig_mod.` 前缀要 strip）
- 配置：`{A1_REPO}/tests/fixtures/ts_tests/model_config.json`
- 快照：`{A1_REPO}/tests/_snapshots/{test_name}.npz`，键是 `array`

固定输入（与官方 conftest 一致，逐个 seed）：

```python
torch.manual_seed(4); x = torch.randn(4, 12, 64)        # in_embeddings
torch.manual_seed(1); q = torch.randn(4, 12, 64)        # q
torch.manual_seed(2); k = torch.randn(4, 16, 64)        # k
torch.manual_seed(3); v = torch.randn(4, 16, 64)        # v
torch.manual_seed(5); mask = torch.randn(4, 12, 16) > 0.5
torch.manual_seed(6); idx = torch.randint(0, 10_000, (4, 12))
```

几何配置：`vocab=10000, ctx=16, d_model=64, num_layers=3, num_heads=4, d_ff=128, theta=10000`。
`d_head=16`。`mask` 是 **3D**（官方 fixture 如此），SDPA 必须支持。

对应关系：

| 快照 | 权重来源 | 输入 | 本 lab 断言容差 |
| --- | --- | --- | --- |
| `test_linear` | `layers.0.ffn.w1.weight` | `x` | atol 1e-4 |
| `test_embedding` | `token_embeddings.weight` | `idx` | atol 1e-4 |
| `test_swiglu` | `layers.0.ffn.w{1,2,3}.weight` | `x` | atol 1e-5 |
| `test_rmsnorm` | `layers.1.ln1.weight` | `x` | atol 1e-4 |
| `test_rope` | 无 | `x`, `pos=arange(12)` | atol 1e-5 |
| `test_scaled_dot_product_attention` | 无 | `q,k,v,mask` | atol 1e-5 |
| `test_4d_scaled_dot_product_attention` | 无 | `q,k,v` 按 head=2 变 4D，mask 同 | atol 1e-5 |
| `test_multihead_self_attention` | `layers.0.attn.*` | `x` | atol 1e-5 |
| `test_multihead_self_attention_with_rope` | 同上 | `x`, `pos=(1,12)` | atol 1e-5 |
| `test_positionwise_feedforward` | 同 swiglu | `x` | 1e-5 |
| `test_transformer_block` | `layers.0.*`（去掉 `layers.0.` 前缀） | `x` | atol 1e-4 |
| `test_transformer_lm` | 全量 state_dict | `idx`, ctx=16 | atol 1e-4 / rtol 1e-2 |
| `test_transformer_lm_truncated_input` | 全量 | `idx[:, :6]` | atol 1e-4 |

**注意**：`lm_head.weight` 与 `token_embeddings.weight` 在参考里**不相等**（untied），
所以 `TransformerLM` 默认不能 tie；tie 只做单独单元测试。

跳过机制：找不到 `A1_REPO` 或快照时，parity 测试打印 `SKIP` 并继续，不能报错退出。
测试输出里要打印每个 parity 的 maxdiff，方便写进 LAB_NOTES 和教程。

---

## 6. 测试清单（impl 至少覆盖以下 43 项，可合并但不得删减）

**A. 手算/公式（10）**
1. `softmax` 每行和为 1、与手算一致、负 dim 正确
2. `softmax` 数值稳定：输入整体 +1e4 结果不变、无 inf/nan
3. `softmax` 平移不变性
4. `sdpa` 3D 与手写公式一致
5. `sdpa` mask 语义：False 位置权重为 0，True 位置和为 1
6. `sdpa` 4D 与逐头 3D 循环一致
7. `rmsnorm` 与公式一致（随机 gain）
8. `rope` 二维手算旋转正确
9. `rope` 相对性：`dot(R_i q, R_j k)` 只依赖 `i-j`
10. `silu` 与 `F.silu` 一致；`swiglu` 与手写公式一致

**B. 形状/接口（8）**
11. `Linear.weight` 形状 `(out,in)`、无 bias、`forward == x @ W.T`
12. `Embedding` 查表等于 `weight[ids]`、支持 1D/2D/3D ids
13. `RMSNorm` 保持输入 dtype（bf16/fp16 进、同 dtype 出）
14. `SwiGLU` 默认 d_ff：512→1344，1600→4288，且是 64 倍数
15. `MHA` 输出形状；`num_heads` 不整除时抛错
16. `Block` 输出形状与输入一致
17. `LM` logits 形状 `(B,T,V)`；截断输入 `T=6` 可跑
18. `LM` 超长输入 `T > context_length` 抛错

**C. 性质（8）**
19. causal：改未来 token 不影响过去位置的输出
20. causal：注意力权重上三角为 0（用 hook 或重算验证）
21. RoPE 只作用于 q/k：`v` 路径与「不旋转」的手写参考一致
22. RoPE 位置平移：q/k 同时平移 `s`，attention 输出不变
23. 多头拆分等价：`num_heads=1` 与手写单头一致
24. RMSNorm 尺度不变：`x` 与 `x*3` 输出相同（同一 gain）
25. `tie_weights=True` 时 `lm_head.weight` 与 `token_embeddings.weight` 共享存储，`num_parameters` 变少
26. 反向传播：最后一个 token 的 loss 反传，所有参数 `grad` 非 None 且有限

**D. 初始化（4）**
27. Linear 初始化 std ≈ `sqrt(2/(in+out))`（容差 ±25%），值在 ±3σ 内
28. Embedding 初始化 std ≈ 1（±25%），值在 ±3 内
29. RMSNorm gain 全 1；Linear/Embedding 无 bias 参数
30. state_dict 键集合与官方参考一致（在 `A1_REPO` 可得时，集合相等；否则跳过）

**E. 官方快照 parity（12 项 / 13 个快照，跳过机制见 §5）**
`swiglu` 与 `positionwise_feedforward` 是同权重同公式，算一项，但两个快照都要跑。
31–42：上表 13 个快照全部 `assert_close` 并打印 maxdiff。

**F. 端到端（1，可并入 C.26）**
43. 小 LM 前向 + `CrossEntropy`（用 `F.cross_entropy` 即可）反传 1 步，loss 下降

**运行方式**：双模式。`python model_lab/test_toy_transformer.py` 直接跑（自带
`PASS/FAIL/SKIP` 汇总，失败 exit 1）；函数命名 `test_*` 以便 pytest 兼容。**venv 里没有 pytest，
不要尝试安装或联网。**

---

## 7. 实验口径

### 7.1 `accounting.py`

解析式核算（不要靠跑大模型）：

- 参数量：
  - `token_embeddings`: `V*d`
  - 每层 attn：`4*d*d`
  - 每层 ffn：`3*d*d_ff`
  - 每层 ln1/ln2：`2*d`
  - `ln_final`: `d`；`lm_head`: `V*d`（untied）
- 前向 FLOPs（只算矩阵乘，`2mnp` 规则）：
  - 每层投影：`8*T*d²`
  - attention 分数 + 加权：`4*T²*d`
  - FFN：`6*T*d*d_ff`
  - 每层合计：`8Td² + 4T²d + 6T d d_ff`
  - lm_head：`2*T*d*V`；embedding 查表不计 FLOPs
- 配置（`d_ff` 一律按 8/3 取最近 64 倍数）：
  - GPT-2 small：12 层 / 768 / 12 头 / ctx 1024
  - GPT-2 medium：24 / 1024 / 16 / 1024
  - GPT-2 large：36 / 1280 / 20 / 1024
  - GPT-2 XL：48 / 1600 / 25 / 1024；另算 ctx 16384
  - 共同：`vocab=50257`
- 输出 JSON 字段（`model_lab/results/accounting.json`）：
  `configs.<name>.{params{...,"total"}, memory_fp32_mib, flops{total, breakdown, proportions}}`，
  外加 `checks`：与官方 `model.pt` 参数量总数对比（若可得）、`d_ff` 是否为 64 倍数。
- 用 `TransformerLM.num_parameters()` 或 ``sum(p.numel())`` 对账一次（小配置即可）。

### 7.2 `exp_overfit.py`

- 数据：`{A1_REPO}/tests/fixtures/tinystories_sample.txt`（约 3.8KB，只读）；
  找不到时用内置合成文本兜底。
- 字节级 tokenizer（`bytes` ↔ id，vocab=256），不依赖 bpe_lab。
- 模型：`d_model=64, num_layers=2, num_heads=4, d_ff=128, ctx=64`，untied。
- 训练：固定一个 batch（`B=8, T=64`），AdamW `lr=3e-3`、`weight_decay=0.0`，
  **最多 300 步或 90 秒**；每 10 步记录 loss；结束后 greedy 生成 64 字节。
- 输出 JSON（`model_lab/results/overfit.json`）：
  `config, steps_run, max_seconds, curve[[step,loss],...], final_loss, initial_loss,
  elapsed_s, tokens_per_s, threads, peak_rss_mib, sample, checks`。
- `checks` 至少包含：loss 全程 finite、`final_loss < initial_loss`、`elapsed_s <= 120`、
  `torch.get_num_threads() == 2`。
- 目标：`final_loss < 0.5`（这是过拟合演示；达不到就如实记录，不许改判定标准）。

### 7.3 数字规则

- **所有数字先落 JSON，再写文档。** 文档里的每个数字必须能在 `results/*.json` 或
  实际命令输出里找到；找不到就重跑，不许编。
- 未跑的实验在 LAB_NOTES 里写「未跑」，不许写推测值。
- 每个脚本打印 `elapsed_s`、线程数、峰值内存；JSON 里保留。

---

## 8. 文档要求

### 8.1 `model_lab/README.md`

首行声明「教学实现，不是 A1 提交答案」，然后是：文件表、快速开始（含 `.venv` 与
`MPLCONFIGDIR`）、测试覆盖点、两个实验看什么、已知限制。

### 8.2 `MODEL_TRANSFORMER_TUTORIAL.md`（scribe）

建议结构（可微调，但每章都要有）：

0. 一页纸结论 + 怎么用这份教程
1. 为什么还要写一个模型 lab（与 `transformer_steps/`、`llama_steps/` 的差别）
2. 形状语言（B,T,C,H,D,V、广播、head 当 batch）
3. 逐组件：Linear / Embedding / RMSNorm / SwiGLU / RoPE / softmax / SDPA / MHA / Block / LM
   （每节：公式 → 直觉 → 代码 → 3 个坑 → 对应测试）
4. 组装与一次前向的完整数据流
5. 怎么证明写对了：手算 / PyTorch 对照 / 官方快照 / 性质测试（附实测 maxdiff 表）
6. 资源核算：推导 + 代码 + GPT-2 四配置表 + 结论
7. 过拟合实验：设置、曲线、结论、局限
8. 和 A1/后续 training_lab 的关系（哪些是 A1 要求、哪些是本 lab 扩展）
9. 学习计划与自测 12 题
10. FAQ 20 问
附录：命令速查、术语表、材料地图

写作规则：

- 数字必须有出处（`results/*.json`、测试输出、命令），并写明命令；
- 代码片段必须和最终 `toy_transformer.py` 一致（以文件为准，不要凭记忆写）；
- 不吹结论：过拟合实验只能证明「实现可训练」，不能证明语言能力；
- 中文写作，术语保留英文。

---

## 9. Definition of Done

- [ ] `python model_lab/test_toy_transformer.py` 全绿（parity 在官方仓库存在时必须真跑，不许全 SKIP）
- [ ] 43 项测试覆盖清单无缺项（允许合并，但报告里逐项映射）
- [ ] 13 个 parity maxdiff 已打印，量级 ≤ 1e-4
- [ ] `python model_lab/accounting.py` 产出 JSON，GPT-2 四配置 + ctx 16384 齐全
- [ ] `python model_lab/exp_overfit.py` 在 120 秒内跑完并产出 JSON（2 线程）
- [ ] `LAB_NOTES.md` 含复现命令、实测数字、坑、限制
- [ ] 教程与最终代码/结果一致，无编造数字
- [ ] 根 README 目录表更新
- [ ] 未修改 `assignment1-basics`、未提交 git

## 10. 禁止

- 不装依赖、不联网、不跑大模型/大语料、不并行开多个 CPU 密集任务；
- 不只写「自己和自己对」的测试：至少要有官方快照或 PyTorch 独立参照；
- 不把 `assignment1-basics` 的任何代码复制进本仓库（它是空骨架，也没有可复制的实现）；
- 不在文档里写没有跑过的数字；
- 不提交 git（Lead 最后统一 review）。
