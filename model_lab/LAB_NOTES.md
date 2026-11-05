# model_lab —— 实测记录、复现命令与踩坑

> 版本：2026-11-05
> 环境：macOS 27.2 arm64（Apple Silicon，10 核），Python 3.14.3，torch 2.14.1。
> 约束：用户要求 **2 线程、单次脚本 < 120 秒**（无风扇机器会明显发热）。
> 声明：**这是教学实现，不是 CS336 A1 的提交答案**；API 与官方 `tests/adapters.py` 故意不同。
> 官方仓库 `assignment1-basics` 只被**只读**使用（权重 + 快照），全程未修改。

---

## 1. 文件与分工

| 文件 | 作用 | 作者 |
| --- | --- | --- |
| `model_lab/toy_transformer.py` | pre-norm Transformer 教学实现（673 行） | impl |
| `model_lab/test_toy_transformer.py` | 45 个双模式测试（44 个覆盖 43 项 + 1 个 MHA 位置回归） | impl |
| `model_lab/accounting.py` | 参数量 / fp32 显存 / 前向 FLOPs 解析式核算 | exp |
| `model_lab/exp_overfit.py` | 单 batch 过拟合 + greedy 生成 | exp |
| `model_lab/results/accounting.json` | 核算结果（含对账 checks） | exp |
| `model_lab/results/overfit.json` | 过拟合曲线 / 吞吐 / 内存 / 样例 | exp |
| `model_lab/SPEC.md` | 开工前定下的实现契约（API/数值/测试/实验口径） | Lead |
| `model_lab/lead_spotcheck.py` | Lead 的独立复核脚本（不 import 测试代码，直接用官方快照抽查 5 个模块） | Lead |
| `MODEL_TRANSFORMER_TUTORIAL.md` | 长教程 | scribe |

---

## 2. 复现命令（全部实测）

在仓库根目录执行，统一用仓库自带 `.venv`：

```bash
cd /Users/yunni/Projects/cs336-prep-code

# 1) 自检：形状 / 参数量 / state_dict / 反向传播（约 0.85s）
.venv/bin/python model_lab/toy_transformer.py

# 2) 全部测试：45 passed, 0 failed, 0 skipped（约 1.0–1.2s，2 线程）
.venv/bin/python model_lab/test_toy_transformer.py

# 3) 官方仓库不在默认路径时，用 A1_REPO 指定；找不到快照则 parity 打印 SKIP
A1_REPO=/path/to/assignment1-basics .venv/bin/python model_lab/test_toy_transformer.py

# 4) 资源核算：产出 results/accounting.json（约 0.5s，含 --strict 时失败 exit 1）
.venv/bin/python model_lab/accounting.py --strict

# 5) 过拟合实验：300 步 / 90s 硬停 / 2 线程，产出 results/overfit.json（约 1.7s）
.venv/bin/python model_lab/exp_overfit.py

# 6) 硬停路径自检（不覆盖默认结果时加 --out；42 步后 timed_out=true，exit 0）
.venv/bin/python model_lab/exp_overfit.py --max-seconds 0.2 --out /tmp/overfit_stop.json

# 7) 无官方仓库时的合成文本兜底
.venv/bin/python model_lab/exp_overfit.py --a1-repo /tmp/does_not_exist

# 8) Lead 的独立复核（自己复算 linear/rope/MHA/block/LM，不 import 测试）
.venv/bin/python model_lab/lead_spotcheck.py
```

环境变量（本 lab 不联网，一般不需要）：

```bash
export MPLCONFIGDIR=$PWD/.tmp/mpl   # 只有画图/导入 matplotlib 时需要
export A1_REPO=/Users/yunni/Projects/assignment1-basics  # 可选
```

---

## 3. 关键数字

### 3.1 测试与官方快照 parity

命令：`.venv/bin/python model_lab/test_toy_transformer.py`
结果：**45 passed / 0 failed / 0 skipped**，1.04–1.21s，2 线程（其中 1 个是 MHA 位置回归测试）。
SKIP 路径：`A1_REPO=/nonexistent/repo` → 31 passed / 14 skipped，exit 0。

| 官方快照 | maxdiff |
| --- | --- |
| `test_linear` | 8.941e-07 |
| `test_embedding` | 0 |
| `test_swiglu` | 1.431e-06 |
| `test_rmsnorm` | 1.252e-06 |
| `test_rope` | 1.475e-06 |
| `test_scaled_dot_product_attention` | 9.537e-07 |
| `test_4d_scaled_dot_product_attention` | 9.537e-07 |
| `test_multihead_self_attention` | 5.364e-07 |
| `test_multihead_self_attention_with_rope` | 2.980e-07 |
| `test_positionwise_feedforward` | 1.431e-06 |
| `test_transformer_block` | 2.384e-06 |
| `test_transformer_lm` | 1.001e-05 |
| `test_transformer_lm_truncated_input` | 1.049e-05 |

**口径说明**：

- 官方 `_snapshots/` 里有 **13 个**模型相关的 `.npz`，但官方 `test_model.py` 只调用 12 个；
  `test_positionwise_feedforward.npz` 没有同名测试函数（遗留快照）。本 lab 把它也跑了
  （与 SwiGLU 同权重、同公式），因此是「12 项 / 13 个快照」。
- 官方 `NumpySnapshot` 默认 `rtol=1e-4, atol=1e-2`；`test_linear` / `test_embedding` 没有显式覆盖，
  因此按默认值比较。显式收紧的是：`sdpa` 3D/4D、`mha`、`mha+rope`、`rope`、`swiglu` 用 `atol=1e-5`；
  `rmsnorm`、`block`、`lm`（含截断）用 `atol=1e-4`（`lm` 另给 `rtol=1e-2`）。
  本 lab 自己统一按更紧的 `atol` 断言（见 SPEC §5 表），与官方容差不是一回事。
- Lead 另写了一个**独立抽查脚本**（`model_lab/lead_spotcheck.py`，不 import impl 的测试），
  用官方权重复算 linear / rope / MHA / block / LM，worst maxdiff **1.001e-05**，与 impl 报告一致。

### 3.2 资源核算（`results/accounting.json`）

命令：`.venv/bin/python model_lab/accounting.py --strict`（2 线程；JSON 的 `runtime` 随运行小幅波动，实测 `elapsed_s` ≈0.006、峰值 RSS ≈212.5 MiB）。

| 配置 | 参数量 | fp32 显存 | 前向 FLOPs | proj / attn / ffn / head |
| --- | --- | --- | --- | --- |
| GPT-2 small (12L/768/12H) | 162,148,608 | 618.5 MiB | 2.92e11 | 19.9% / 13.3% / 39.8% / 27.1% |
| GPT-2 medium (24L/1024/16H) | 406,539,264 | 1550.8 MiB | 8.30e11 | 24.8% / 12.4% / 50.1% / 12.7% |
| GPT-2 large (36L/1280/20H) | 833,591,040 | 3179.9 MiB | 1.77e12 | 27.3% / 10.9% / 54.3% / 7.4% |
| GPT-2 XL (48L/1600/25H) | 1,640,452,800 | 6257.8 MiB | 3.52e12 | 28.6% / 9.2% / 57.5% / 4.7% |
| GPT-2 XL, ctx=16384 | 同上 | 同上 | 1.34e14 | 12.1% / 61.7% / 24.2% / 2.0% |

- `d_ff` 规则：`8/3*d_model` 取最近 64 倍数 → 768→2048、1024→2752、1280→3392、1600→4288
  （4288 与 handout §3.5 给出的值一致）。
- 对账 1：官方 `tests/fixtures/ts_tests/model.pt` 共 30 个张量、**1,403,328** 参数；
  解析式同为 **1,403,328**，diff=0。
- 对账 2：一个 115,008 参数的小 `TransformerLM`，`sum(p.numel())` 与 `num_parameters()` 都是 115,008。
- ctx 16384 / 1024 的 FLOPs 比：**37.98×**；attention 占比 9.16% → 61.73%。
  这是「长上下文先被 attention 平方项吃掉」的直观数字。

### 3.3 过拟合实验（`results/overfit.json`）

命令：`.venv/bin/python model_lab/exp_overfit.py`（默认 90s 硬停，实际 1.72s）。

| 项 | 值 |
| --- | --- |
| 数据 | `tinystories_sample.txt`（3794 B，只读），固定 batch 8×64=512 token |
| 模型 | 115,008 参数（d=64, L=2, H=4, d_ff=128, ctx=64, untied） |
| 优化器 | AdamW，lr=3e-3，weight_decay=0，300 步 |
| initial → final loss | **5.673745 → 0.003972** |
| 耗时 | 1.717s（纯训练 1.293s） |
| 吞吐 | 118,818.7 tokens/s |
| 线程 / 峰值 RSS | 2 / 316.1 MiB |
| greedy 样例（prompt `\nOnce up`） | `\nOnce upon a time there was a little boy named Ben. Ben loved to ame to ` |

选点曲线：0→5.6737，10→2.8142，20→1.7734，30→0.8259，50→0.0496，100→0.0082，
150→0.0059，200→0.0049，250→0.0043，300→0.0040（JSON 里有完整 31 点）。

额外验证（未覆盖默认结果文件）：

- 硬停：`--max-seconds 0.2` → 42 步、`timed_out=true`、exit 0，已跑结果仍落盘；
- 兜底：`--a1-repo /tmp/does_not_exist` → `synthetic_fallback`（832 B），300 步，final 0.0038；
- 可复现：同 seed 重跑，greedy 样例逐字节相同，loss 差 ~2e-7（2 线程 fp 归约的正常波动）。

---

## 4. 实现要点与踩坑（模型部分）

1. **mask 语义**：官方约定 `True = 可见`（信息能流动），所以是 `masked_fill(~mask, -inf)`。
   写反了不会报错，只会让 parity 差到 2～3。SDPA 的 mask 既可能是 `(T,T)`，也可能是
   `(B,T,T)`（官方 fixture 是 3D），实现要支持广播。
2. **RoPE 是交错配对**：`(x0,x1),(x2,x3),...`，不是「前半/后半」拆开（`rotate_half`）。
   用前后半拆分的实现能跑、能自洽，但和官方快照差 5.38；交错配对 maxdiff 1.5e-6。
   另外 `d_k = d_model // num_heads`（per-head），不是 `d_model`；用 `d_model` 当角度维度
   会和参考实现对不上。
3. **causal 在 MHA 内部**：官方参考的 `CausalMultiHeadSelfAttention` 自带下三角 mask，
   不是由调用方传。SDPA 本身不假设 causal。
4. **pre-norm 顺序**：`y = x + attn(rms(x))`，`out = y + ffn(rms(y))`；顺序反了形状照样对。
5. **lm_head 默认 untied**：官方参考权重里 `lm_head.weight` 与 `token_embeddings.weight`
   **不相等**。如果默认 tie，`load_state_dict(strict=True)` 会把两份权重写到同一个参数上，
   parity 直接崩。tie 作为单独开关 + 单独测试。
6. **d_ff 取「最近」64 倍数（half-up）**：`max(64, 64*floor(8d/(3*64)+0.5))`，不是向下取整，
   也不能用 Python `round`（银行家舍入，`round(2.5)=2`）。512→1344、1600→4288；
   tie 边界 `d_model=60` 时是 192 而不是 128。
7. **初始化**：Linear `std=sqrt(2/(in+out))`、Embedding `std=1`，都用 `trunc_normal_`，
   边界 `±3σ`；RMSNorm gain 初始化为 1。这些在 parity 里被 `load_state_dict` 掩盖，
   所以必须用独立统计测试（本 lab 的 D27/D28）。
8. **RMSNorm 先 upcast fp32** 再平方，最后回原 dtype；否则 fp16 输入可能溢出。
9. **state_dict 键名**：`token_embeddings`、`layers.{i}.attn.{q,k,v,output}_proj`、
   `layers.{i}.ffn.w{1,2,3}`、`layers.{i}.ln1/ln2`、`ln_final`、`lm_head` 必须与参考逐字一致，
   否则 parity 无法直接 `load_state_dict(strict=True)`。
10. **RoPE 的 cos/sin 用 `persistent=False` buffer**：不进 state_dict，但会跟随 `.to(device)`。
11. **positions 广播**：`(..., T)` 的位置张量用高级索引取表；`(1,12)` 对 `(B,H,12,D)`
    靠右对齐广播即可。`(B,12)` 需要先插 head 维：MHA 内部按
    `token_positions.ndim == x.ndim - 1` 自动 `unsqueeze(-2)`，支持 `(T,)`/`(1,T)`/`(B,T)`/`(1,1,T)`。
    `(B,H,T)` 对 3D 输入不是受支持形状（`H==num_heads` 时会「碰巧」广播，不要依赖）。
12. **超过 context_length 要显式报错**：官方不测，但教学实现里应 `raise ValueError`，
    否则 RoPE 表索引会给出莫名其妙的错误。

---

### 4.13 交付前 QA 发现并修复的两个真实缺陷（值得记住）

这两个都是「测试全绿、文档也写了」但实际行为不对的典型，靠**换一条代码路径复核**才抓出来：

1. **MHA 的 `token_positions` 形状分支写错**：原条件是 `token_positions.ndim == x.ndim`。
   对 `x=(B,T,C)` 传 `(1,T)` 时，靠 NumPy 式右对齐广播「碰巧」能跑通，于是测试和 README
   都以为 `(B,T)` 也支持；实际传 `(B,T)` 直接 `RuntimeError`。修复：改成
   `token_positions.ndim == x.ndim - 1` 再 `unsqueeze(-2)`；新增回归测试
   `test_C27_mha_accepts_batch_token_positions`（用两种不同位置间隔证明「每个样本用自己的位置」，
   旧代码下必 FAIL）。修复后 `(B,T)` 与逐样本 `(1,T)` 的 maxdiff 为 `9.5e-7`。
   教训：**形状「碰巧能广播」不等于语义对**；README 声称支持的行为必须有测试锁住。
2. **`_default_d_ff` 用了银行家舍入**：`round(160/64)=round(2.5)=2 → 128`，而 accounting 用
   half-up 得 192。当前配置不触发，但两套代码对同一规则给出不同答案就是隐患。修复：统一为
   `max(64, 64*floor(8d/(3*64)+0.5))`，交叉验证 `d=1..2048` 与 accounting 逐值一致。
   教训：**同一个公式在两处实现时，必须用交叉测试锁死边界**，不能只测「常用配置」。

## 5. 我们怎么证明「写对了」

三层，彼此独立：

1. **手算/公式**：softmax 平移不变、mask 语义、2D 旋转、RMSNorm 公式、SwiGLU 公式；
2. **PyTorch 独立参照**：`F.silu`、`F.scaled_dot_product_attention`、多头循环展开；
3. **官方快照**：用官方权重 + 固定 seed 输入，逐元素对齐 13 个 `.npz`；
4. **性质测试**：causal 不泄漏未来、RoPE 相对性/平移、tie 共享存储、反向梯度有限；
5. **Lead 独立抽查**：不复用 impl 的测试代码，单独复算 5 个模块，worst 1.001e-05。

只做第 3 层也能「过测试」，但只有 1/2/4 层能在没有官方快照时告诉你哪里错了。

---

## 6. 已知限制 / 未跑的事

1. **过拟合 ≠ 语言能力**：0.003972 只证明「实现可训练、能记住 512 个 token」，
   不能说明模型学到了语言结构；TinyStories 上真正的训练属于下一步 training_lab。
2. **没有训练/验证集划分、没有 LR sweep、没有消融**：这些是 A1 实验部分，本 lab 不做。
3. **accounting 只算前向矩阵乘**：不含 attention 的显存（softmax 矩阵 `B*H*T*T`）、
   不含反向（约前向 2×）、不含优化器状态；显存只算参数量的 fp32。
4. **单机 CPU、2 线程、单 seed、小配置**；数字不能外推到 GPU/大模型。
5. **官方快照 13 个 vs 官方测试 12 个**：`test_positionwise_feedforward.npz` 是遗留快照，
   没有同名测试函数；本 lab 仍拿它做额外对照。
6. **venv 没有 pytest**：只验证了直接跑模式；测试函数全部无参数、可被 pytest 收集，
   但没在 pytest 下实跑（不装包、不联网）。
7. **overfit 的 initial_loss 5.6737 略高于 ln(256)=5.545**：随机初始化 logits 非零均值，
   属正常现象。
8. **peak RSS 316 MiB 含 torch 导入基线**（约 200 MiB），不是模型自身。

---

## 7. 和 CS336 A1 的对应

| A1 组件 | 本 lab | 官方快照 |
| --- | --- | --- |
| Linear / Embedding | ✅（初始化也测） | ✅ |
| RMSNorm / SwiGLU | ✅（fp32 upcast、d_ff 规则） | ✅ |
| RoPE | ✅（交错、per-head、相对性） | ✅ |
| softmax / SDPA | ✅（3D/4D、mask 广播） | ✅（含 4D） |
| Causal MHA | ✅（内置 mask、per-head RoPE） | ✅ |
| Transformer block / LM | ✅（pre-norm、untied） | ✅（含截断） |
| 资源核算 | ✅（GPT-2 四配置 + 长上下文） | —（手写题） |
| 训练工具 / 真实训练 / 消融 | ❌ 下一步 training_lab | — |

---

## 8. 一句话总结

> 用 673 行、45 个测试、13 个官方快照把 A1 的 pre-norm Transformer 写到 worst maxdiff
> **1.05e-5**；资源核算与官方参数量 diff=0；单 batch 300 步把 loss 从 5.67 压到 0.0040，
> 全程 2 线程、1.7 秒、316 MiB——证明「实现正确且可训练」，但不证明语言能力。
