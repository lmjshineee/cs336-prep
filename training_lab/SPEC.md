# training_lab SPEC — CS336 A1 §4–6 训练工具教学实现契约

> 版本：2026-11-06
> 维护：Lead。**所有实现者先读本文件，再动手。**
> 本文件是唯一契约：公共 API、数值约定、测试清单、实验口径、文件所有权。
> 与 `model_lab/SPEC.md`、`bpe_lab/` 同规格：双模式测试、2 线程、单命令 < 120 秒。

---

## 0. 目标与边界

把 CS336 A1 handout §4–6 的**训练工具链**从零写出来，并用 `model_lab` 的模型 + `bpe_lab` 的
tokenizer 跑通一次**低资源冒烟训练**：

- cross-entropy（1 分）
- AdamW（2 分）
- cosine + warmup 学习率（1 分）
- 梯度裁剪（1 分）
- 数据加载 `get_batch`（2 分）
- checkpoint 保存/恢复（1 分）
- 训练脚本 `train_demo.py`（4 分）
- 温度 + top-p 解码（3 分）

合计对应 A1 的 15 分。**这不是 A1 的提交答案**：API 与官方 `tests/adapters.py` 故意不同，
代码优先可读性；官方仓库 `assignment1-basics` **只读**使用（fixtures / 对照测试行为），不修改。

**不做**（留给后续阶段）：完整 TinyStories 训练、LR/batch sweep、4 个消融实验、长教程。

---

## 1. 环境与热预算（硬约束）

- 解释器：`/Users/yunni/Projects/cs336-prep-code/.venv/bin/python`（Python 3.14.3，torch 2.14.1）。
- **所有可能超过 1 秒的脚本**开头必须 `torch.set_num_threads(2)`；单条命令 **< 120 秒**。
- 训练脚本必须有 `--max-seconds`（默认 90）硬停，超时先保存已跑结果。
- 不联网、不安装新依赖。数据只用本地
  `/Users/yunni/Projects/assignment1-basics/tests/fixtures/tinystories_sample_5M.txt`（只读）。
- 重活串行：exp 的数据准备与训练冒烟不要同时跑；任何一步超过 120 秒就改小规模。
- `training_lab/data/`、`training_lab/checkpoints/` 是生成产物，加入 `.gitignore`，不提交。

---

## 2. 文件所有权（写入范围）

| 角色 | 可写文件 |
| --- | --- |
| impl | `training_lab/toy_training.py`、`training_lab/test_toy_training.py` |
| exp | `training_lab/prep_data.py`、`training_lab/train_demo.py`、`training_lab/results/` |
| scribe | `TRAINING_TUTORIAL.md`（本期不做，后续阶段） |
| Lead | `training_lab/SPEC.md`、`training_lab/lead_spotcheck.py`、`training_lab/README.md`、`training_lab/LAB_NOTES.md`、根 `README.md`、`.gitignore` |

**不要越界写别人的文件。** 需要对方改接口时，先在报告里说明，由 Lead 协调。

---

## 3. 公共 API 契约（`training_lab/toy_training.py`）

只依赖 `torch` / `numpy` / 标准库，**不 import** `model_lab` 或 `bpe_lab`（保证工具箱独立可测）。

```python
# ---------- 1. 损失 ----------
def cross_entropy(logits: Tensor, targets: Tensor) -> Tensor
    # logits: (..., vocab_size) float; targets: (...) int64
    # 返回所有前导位置上的平均 loss（标量）。先减 max，logsumexp 与 logit 相消。
def perplexity(mean_loss: float | Tensor) -> float
    # exp(loss)；只接受「已经平均过的」loss，避免再对 batch 求均值

# ---------- 2. 优化器 ----------
class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    def step(self, closure=None)
    # 严格按 handout Algorithm 1：t 从 1 开始；先 decoupled weight decay（用 base lr），
    # 再更新一阶/二阶矩，最后用 alpha_t = lr * sqrt(1 - beta2**t) / (1 - beta1**t)。
    # 必须与 torch.optim.AdamW 在 1e-4 内逐元素一致（官方测试口径）。

# ---------- 3. 学习率调度 ----------
def cosine_lr(it: int, max_lr: float, min_lr: float,
              warmup_iters: int, cosine_cycle_iters: int) -> float
    # it < Tw:        it/Tw * max
    # Tw <= it <= Tc: min + 0.5*(1+cos((it-Tw)/(Tc-Tw)*pi))*(max-min)
    # it > Tc:        min
    # 边界：Tw=0 时 t=0 返回 max；it=Tc 返回 min。
def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None
    # 把 optimizer 里所有 param_group 的 lr 改成 lr（in place）

# ---------- 4. 梯度裁剪 ----------
def clip_gradients(parameters: Iterable[Parameter], max_l2_norm: float,
                   eps: float = 1e-6) -> float
    # 把所有 grad 的全局 L2 norm 算出来；若 > max，按 max/(norm+eps) 原地缩放。
    # 返回裁剪前的全局 norm（方便日志）。None grad 跳过。

# ---------- 5. 数据 ----------
def load_tokens(path: str | os.PathLike[str], dtype=np.uint16) -> np.memmap
def get_batch(dataset, batch_size: int, context_length: int, device: str = "cpu",
              rng: np.random.Generator | None = None) -> tuple[Tensor, Tensor]
    # dataset: 1D np.ndarray / np.memmap，int token id
    # start ∈ [0, len(dataset)-context_length-1]，x=dataset[start:start+T]，
    # y=dataset[start+1:start+T+1]；返回 (B,T) int64 torch.Tensor，放到 device。
    # rng 为 None 时用 np.random；给定 Generator 时必须完全可复现。

# ---------- 6. checkpoint ----------
def save_checkpoint(model, optimizer, iteration: int, out) -> None
    # dict: {"model": state_dict, "optimizer": state_dict, "iteration": int}
    # 支持 path 与 file-like（BytesIO）
def load_checkpoint(src, model, optimizer) -> int
    # torch.load(..., map_location="cpu", weights_only=True)，恢复两者，返回 iteration

# ---------- 7. 解码 ----------
def softmax_temperature(logits: Tensor, temperature: float = 1.0) -> Tensor
    # 沿最后一维；temperature 越小越尖；返回概率（和=1），数值稳定
def top_p_filter(probs: Tensor, top_p: float) -> Tensor
    # 核采样：按概率降序累加，保留累计和首次 >= p 的最小集合（至少保留 1 个），
    # 其余置 0，再归一化；返回同形状概率
def sample_token(logits: Tensor, temperature: float = 1.0,
                 top_p: float | None = None, generator: torch.Generator | None = None) -> int
    # logits: (vocab,)，返回 int
@torch.no_grad()
def generate(model, prompt_ids, *, max_new_tokens: int = 32, temperature: float = 1.0,
             top_p: float | None = None, eos_token_id: int | None = None,
             context_length: int | None = None,
             generator: torch.Generator | None = None) -> list[int]
    # prompt_ids: list[int] 或 1D Tensor；返回 **prompt + 新生成** 的完整 list[int]。
    # 每步取模型输出最后一位；超过 context_length 时只保留最近 context_length 个 token。
    # 生成到 max_new_tokens 或遇到 eos_token_id 时停止（eos 会包含在返回值里）。
```

### 数值约定（容易写错的地方）

- **CE**：`logsumexp(logits) - logits[target]`；不要先 softmax 再 log。
- **AdamW**：weight decay 必须在 moment 更新**之前**、且用 base `lr`（不是 `alpha_t`）；
  `m`/`v` 初值为 0，`t` 从 1 开始；更新 `p -= alpha_t * m / (sqrt(v) + eps)`。
- **cosine**：`it == warmup_iters` 时走 cosine 分支且 `cos(0)=1` → 返回 `max_lr`；
  `it == cosine_cycle_iters` → `cos(pi)=-1` → 返回 `min_lr`。
- **clip**：全局 norm 用 float 累加；缩放因子 `max/(norm+eps)` 而不是 `max/norm`。
- **get_batch**：上界用 `len(dataset) - context_length`（exclusive），所以最大 start 是
  `len - context_length - 1`（官方 `test_get_batch` 的断言口径）。
- **top-p**：先排序再累计；用 `cumsum - sorted_probs < p` 选集合，保证第一个 token 一定在内。

---

## 4. 测试清单（`training_lab/test_toy_training.py`）

双模式：`python test_toy_training.py` 直接跑（PASS/FAIL 汇总，失败 exit 1），也兼容 pytest。
统一 `torch.set_num_threads(2)`，全部测试应在 < 5 秒内跑完。

| 组 | # | 测试点 |
| --- | --- | --- |
| A CE | A1 | 与 `F.cross_entropy` 在随机 `(2,3,7)` logits 上一致 |
| | A2 | logits ×1000 时仍稳定（不 NaN/Inf，与 F 一致） |
| | A3 | 反向梯度与 F 一致（`allclose` 1e-6） |
| | A4 | 高维 batch（`(5,4,9)`）与 F 一致 |
| | A5 | 返回标量、dtype 为 float |
| B AdamW | B1 | 1000 步线性回归权重与 `torch.optim.AdamW` 一致（atol 1e-4，官方口径） |
| | B2 | grad=0 时参数按 `(1-lr*wd)` 收缩 |
| | B3 | `betas=(0.9,0.95)` 与 torch 一致 |
| | B4 | param groups 不同 lr 各自生效 |
| | B5 | `state_dict` 保存/加载后继续训练与不中断完全一致 |
| | B6 | 非法 lr 抛 `ValueError` |
| | B7 | closure 被调用且返回其 loss |
| C schedule | C1 | 官方 25 个期望值逐元素一致（见 `tests/test_optimizer.py`） |
| | C2 | `t=0→0`、`t=Tw→max`、`t=Tc→min`、`t>Tc→min` |
| | C3 | `warmup_iters=0` 时 `t=0` 返回 max |
| | C4 | `set_lr` 改全部 param group 的 lr |
| D clip | D1 | 与 `torch.nn.utils.clip_grad_norm_` 一致（norm > max） |
| | D2 | norm <= max 时梯度不变 |
| | D3 | 裁剪后 norm ≈ max（略小），eps=1e-6 |
| | D4 | 跳过 None grad，返回裁剪前 norm |
| E data | E1 | 形状 `(B,T)`、dtype int64、`y == x+1` |
| | E2 | start 范围与均匀性（1000 次采样，5σ 口径） |
| | E3 | `np.memmap` 数据可读、越界不采样 |
| | E4 | 给定 rng 时完全可复现 |
| | E5 | 非法 device 抛异常 |
| F ckpt | F1 | model/optimizer/iteration 往返一致（官方测试口径） |
| | F2 | 中断恢复等价：6 步 vs 3 步+存+读+3 步，参数逐位一致 |
| | F3 | `BytesIO` file-like 可用 |
| | F4 | 返回保存的 iteration |
| G decode | G1 | 极低温度 = argmax（greedy） |
| | G2 | `top_p=1.0` 保留全部 token |
| | G3 | 小词表手工概率下 `top_p` 保留最小核 |
| | G4 | 温度降低 ⇒ 分布熵下降 |
| | G5 | `generate` 遵守 max_new_tokens / eos / context 截断，seed 可复现，返回含 prompt |
| H 集成 | H1 | 小模型（Embedding+Linear）用本工具箱训 50 步，loss 下降 > 50% |

失败必须打印可定位信息（哪个断言、期望/实际），不要只报 `AssertionError`。

---

## 5. 实验契约（exp 负责）

### 5.1 `prep_data.py` — 数据准备

- 输入：TinyStories fixture 前 `--bytes`（默认 1_000_000）字节。
- 在**前 512KB** 上训练 byte-level BPE（`bpe_lab.train_bpe_from_file`，vocab 2048，
  special token `<|endoftext|>`），保存 `data/tokenizer/`。
- 用该 tokenizer 把整段文本编码成 uint16 token 序列，按 95/5 切成 `data/train.bin`、
  `data/val.bin`（裸 uint16，无 header），并写 `data/meta.json`：
  `{bytes, vocab_size, train_tokens, val_tokens, compression_ratio, elapsed_s, tokenizer_dir, ...}`。
- 单命令 < 120 秒；可加 `--force` 重跑，默认缓存命中直接退出。
- 输出打印：字节数、token 数、压缩比、耗时、峰值内存（`resource.getrusage`）。

### 5.2 `train_demo.py` — 端到端训练脚本

- CLI 至少支持：`--steps --batch-size --context-length --d-model --num-layers --num-heads
  --lr --min-lr --warmup --weight-decay --grad-clip --max-seconds --checkpoint-every
  --checkpoint-dir --resume --sample-tokens --temperature --top-p --seed --out`。
- 流程：`load_tokens`（memmap）→ `TransformerLM`（model_lab）→ `AdamW` + `cosine_lr`
  → 每步 `get_batch` / forward / `cross_entropy` / backward / `clip_gradients` / step /
  `set_lr` → 每 N 步记录 train loss、每 `--eval-every` 步在固定 val batch 上算 loss
  → 存 checkpoint → 结束用 `generate` 采样。
- 支持 `--resume`：从 checkpoint 恢复 model/optimizer/iteration 后继续，迭代号接续。
- 默认配置必须 **< 90 秒**跑完；`--max-seconds` 硬停并保存已跑结果。
- 结果写 `results/train_demo.json`：`config`、`curve`（step/train_loss/val_loss/lr/grad_norm/
  tokens_per_s）、`summary`（elapsed、tokens/s、final loss、参数量）、`checkpoints`、
  `resume_check`（中断恢复等价性的布尔 + maxdiff）、`samples`（prompt/输出/温度/top-p）。
- 额外提供 `--verify-resume` 模式：跑 A=6 步；再跑 3 步→存档→读档→3 步；比较最终参数
  与 val loss（要求逐位或 <1e-6），把结果写进 `resume_check`。

### 5.3 验收线

- 测试全绿（约 35 个，直接跑 + 可选 pytest）。
- `prep_data.py` 与 `train_demo.py` 单条命令均 < 120 秒。
- 冒烟训练 train loss 从初始值下降 ≥ 20%，val loss 有记录；采样不崩。
- 恢复等价性检查通过（maxdiff < 1e-6）。
- 所有数字写进 `LAB_NOTES.md`（Lead 维护），results JSON 可机器读。

---

## 6. 验收清单（2026-11-06 全部通过）

- [x] `training_lab/toy_training.py` 按 §3 契约实现，自检可独立跑
- [x] `training_lab/test_toy_training.py` 35 项全绿，0.44–0.59 秒
- [x] `training_lab/prep_data.py` 产出 train.bin/val.bin/meta.json，实测 0.40 秒
- [x] `training_lab/train_demo.py` 端到端可跑，实测 14.8 秒，产出 results/train_demo.json
- [x] `--verify-resume` 通过（max_param_diff 2.38e-07 < 1e-6）
- [x] `training_lab/lead_spotcheck.py` 独立复核 8/8 通过
- [x] `training_lab/README.md` 快速开始 + 覆盖点 + 限制
- [x] `training_lab/LAB_NOTES.md` 实测数字、命令、踩坑
- [x] 根 `README.md` 目录表加 `training_lab/` 一行
- [x] 根 `.gitignore` 忽略 `training_lab/data/`、`training_lab/checkpoints/`
