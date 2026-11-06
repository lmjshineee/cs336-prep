# training_lab —— 实测记录、复现命令与踩坑

> 版本：2026-11-06
> 环境：macOS 27.2 arm64（Apple Silicon），Python 3.14.3，torch 2.14.1，numpy 2.5.4。
> 约束：用户要求 **2 线程、单次脚本 < 120 秒、不要高温**（无风扇机器会明显发热）。
> 声明：**这是教学实现，不是 CS336 A1 的提交答案**；API 与官方 `tests/adapters.py` 故意不同。
> 官方仓库 `assignment1-basics` 只被**只读**使用（fixtures + 测试口径），全程未修改。

---

## 1. 文件与分工

| 文件 | 行数 | 作用 | 作者 |
| --- | --- | --- | --- |
| `training_lab/toy_training.py` | 560 | 训练工具箱（14 个公共 API） | impl |
| `training_lab/test_toy_training.py` | 978 | 35 个双模式测试（A–H） | impl |
| `training_lab/prep_data.py` | 343 | TinyStories 前缀 → uint16 token 切片 | exp |
| `training_lab/train_demo.py` | 665 | 端到端训练脚本（CLI / eval / ckpt / resume / 采样） | exp |
| `training_lab/lead_spotcheck.py` | 287 | Lead 独立复核（不 import 测试代码） | Lead |
| `training_lab/results/train_demo.json` | — | 冒烟训练结果（机器可读） | exp |
| `training_lab/SPEC.md` | 238 | 开工前契约 | Lead |
| `training_lab/README.md` / `LAB_NOTES.md` | — | 使用说明 / 本文件 | Lead |

---

## 2. 复现命令与实测结果

在仓库根目录执行，统一用仓库自带 `.venv`。

### 2.1 工具箱测试

```bash
.venv/bin/python training_lab/test_toy_training.py
```

实测（多次运行）：

```
汇总: 35 passed, 0 failed, 共 35 项
耗时: 0.44s，线程数: 2        # 进程 real 约 1.3 s
```

- 35 项与 SPEC §4 的 A–H 表逐项对应；失败时会打印 label + maxdiff + 期望/实际，不是裸
  `AssertionError`。
- impl 还做了负向自检：monkeypatch 成「cosine 恒返回 max」「get_batch 返回全零」后，
  测试能抓出来并给出定位信息。
- 依赖只有 `torch` / `numpy` / 标准库；从 `/tmp` 用绝对路径跑同样 35 passed（与 cwd 无关）。
- 本机 venv 没有 pytest，所以 pytest 模式未实跑；测试是纯 `test_*` 函数、无 fixture，
  按收集规则可直接兼容。

### 2.2 Lead 独立复核

```bash
.venv/bin/python training_lab/lead_spotcheck.py
```

```text
[PASS] cross_entropy vs F.cross_entropy: maxdiff=0.00e+00, large=1.22e-04, grad=5.59e-09
[PASS] AdamW vs torch.optim.AdamW (1000 steps): weight maxdiff=1.19e-07 (atol=1e-4)
[PASS] cosine_lr vs 官方 25 个期望值: maxdiff=2.78e-17
[PASS] clip_gradients vs torch.clip_grad_norm_: grad maxdiff=0.00e+00, norm diff=9.99e-07
[PASS] get_batch 形状/offset/范围: shape/offset/range 全 True, covered=93/93
[PASS] checkpoint 往返 + 恢复等价: iteration=4, model/opt/resume 全 True
[PASS] 解码 greedy / top-p 最小核: greedy/nucleus True, sharp_entropy=0.441
[PASS] 小模型 40 步 loss 下降: loss 3.6226 -> 0.0023 (drop 99.9%)
--------------------------------------------------------------
8/8 checks passed        # 进程 real 约 1.2 s
```

### 2.3 数据准备

```bash
/usr/bin/time -p .venv/bin/python training_lab/prep_data.py --force
```

```text
bytes=1,000,000  vocab=2048  bpe_train_bytes=524,288
tokens=271,029  train=257,477  val=13,552  compression_ratio=3.690 bytes/token
elapsed=0.40s (tokenizer 0.29s / encode 0.10s / write 0.00s)
peak_rss=228.7 MiB  threads=2
check: train max_id=2047  val max_id=2047  all < vocab_size=True
real 1.32
```

缓存命中路径（第二次运行）直接打印上次统计并退出，约 0.5 秒：

```text
prep_data：缓存命中，跳过重新生成（--force 可强制重跑）
bytes=1,000,000  vocab=2048  train_tokens=257,477  val_tokens=13,552  ratio=3.690
```

两次 `--force` 产物一致（271,029 token，split 257,477），说明流程可复现。

### 2.4 端到端冒烟训练

```bash
/usr/bin/time -p .venv/bin/python training_lab/train_demo.py --verify-resume
```

```text
data: train=257,477 tokens  val=13,552 tokens  vocab=2048
model: d_model=128 layers=2 heads=4 d_ff=512 ctx=64 params=1,049,216
steps=600 (this run 600)  batch=16  elapsed=14.81s  train_elapsed=13.76s  tokens/s=44643.5
train loss: initial=7.6663 -> final=2.9949 (60.93% down)
val loss: initial=7.6458 -> final=2.9788  timed_out=False  threads=2  peak_rss=386.3 MiB
verify-resume: max_param_diff=2.384e-07  val_loss_diff=0.000e+00  passed=True  loaded_iteration=3
sample (T=0.8, top_p=0.95): ' other animals who needed the food more.\nThey gave some food to a hungry bottle. They were very happy. They lo'
sample (T=0.8, top_p=0.95): ' Everywhere she looked there were trees, rocks and colourful flowers. Her mom saw them on the ground and said,'
checkpoints: [200, 400, 600]
[out] training_lab/results/train_demo.json
real 15.84
```

（exp 另一次同样配置实测 15.55 s / 41,656 tokens/s；差异来自机器负载波动，数值曲线一致。）

### 2.5 额外行为验证（exp）

| 场景 | 命令 | 结果 |
| --- | --- | --- |
| 断点续训 | `--resume /tmp/rt/latest.pt`（从 step 300 → 400） | iteration 正确接续；JSON `initial_train_loss_step=300`；`loss_reduction_pct` 为 null（resume 段不适用） |
| 硬停 | `--max-seconds 2 --steps 5000` | 2.4 s 停止，跑 104 步，`timed_out=true`，已保存 checkpoint 与结果，exit 0 |
| 缓存 | `prep_data.py`（不 `--force`） | 直接复用，0.5 s |

---

## 3. 数值对齐（官方 / PyTorch 口径）

| 项目 | 参照 | 实测差异 |
| --- | --- | --- |
| `cross_entropy` | `F.cross_entropy`（随机 `(2,3,7)`） | maxdiff 0；梯度 5.6e-09 |
| `cross_entropy`（大 logits） | `F.cross_entropy`（logits ×1000） | 1.22e-04（相对数值 ~1e4，属正常抵消误差；不 NaN/Inf） |
| `AdamW` | `torch.optim.AdamW`，官方 `test_optimizer.py` 的 1000 步线性回归 | **1.19e-07**（阈值 1e-4） |
| `cosine_lr` | 官方 25 个期望值 | **2.78e-17** |
| `clip_gradients` | `torch.nn.utils.clip_grad_norm_` | 裁剪后梯度 maxdiff **0**；裁剪前 norm diff 1.0e-06 |
| `get_batch` | 官方 `test_get_batch` 的起点范围/均匀性口径 | 93/93 起点覆盖，max start=92 |
| checkpoint | 官方 `test_serialization.py` 口径 | model/optimizer state 往返一致 |

> `AdamW` 与 `torch.optim.AdamW` 只差实现细节（`mul_/add_` vs `lerp_`），官方测试本身也允许
> 「匹配参考实现或匹配 PyTorch」两种口径。

---

## 4. 数据准备细节

- **来源**：`/Users/yunni/Projects/assignment1-basics/tests/fixtures/tinystories_sample_5M.txt`
  的前 1,000,000 字节（只读；UTF-8 截断时回退到合法边界）。
- **tokenizer**：`bpe_lab.train_bpe_from_file` 在前 512KB 上训练 byte-level BPE，
  `vocab_size=2048`（256 bytes + 1 special + **1791 merges**），special token `<|endoftext|>`；
  保存为 GPT-2 格式 `data/tokenizer/vocab.json` + `merges.txt`。
- **编码**：`encode_iterable` 按 64KB 文本块流式编码，得到 271,029 个 token（3.690 bytes/token）。
- **切分**：按 token 数 95/5，`split_token_index=257,477`；val 是文本尾部 5%（约最后 50KB）。
  验证集文本不在 BPE 训练语料（前 512KB）里，避免 tokenizer 直接见过验证文本；
  训练集前 512KB 同时用于 tokenizer 与训练是常规做法。
- **产物**：`data/train.bin` / `val.bin`（裸 uint16，无 header）+ `meta.json`。
  `meta.json` 含 bytes / vocab / token 数 / 压缩比 / 耗时 / 路径 / 自检 / 环境版本。
- **自检**：`np.memmap` 读回长度与 meta 一致；`max_id=2047 < vocab_size`。

---

## 5. 训练结果细节

配置：`d_model=128, layers=2, heads=4, d_ff=512, ctx=64, batch=16, steps=600,
lr=3e-3, min_lr=1e-4, warmup=20, wd=0.01, clip=1.0, seed=0`。

| step | train loss | val loss | lr | 裁剪前 grad norm |
| --- | --- | --- | --- | --- |
| 0 | 7.6663 | 7.6458 | 0 | — |
| 20 | 5.8876 | — | 2.85e-3 | 0.679 |
| 100 | 4.1629 | 3.9181 | 2.87e-3 | 0.576 |
| 200 | 3.3348 | 3.5275 | 2.37e-3 | 0.605 |
| 300 | 2.9851 | 3.2757 | 1.64e-3 | 0.622 |
| 400 | 2.9145 | 3.0848 | 8.78e-4 | 0.658 |
| 500 | 2.8819 | 2.9957 | 3.12e-4 | 0.687 |
| 600 | 2.9949 | 2.9788 | 1.00e-4 | 0.689 |

- 吞吐：614,400 token / 13.76 s = **44,643 tokens/s**（2 线程 CPU）。
- 内存：peak RSS **386.3 MiB**（含 checkpoint 序列化）。
- **clip 实际未触发**：全程裁剪前梯度范数 0.50–0.76，都小于 `--grad-clip 1.0`；
  默认配置里它只是安全网。想观察触发可跑 `--grad-clip 0.5`。
- checkpoint：`step 200/400/600` + `latest.pt` + `resume_verify.pt`，约 65MB（gitignored）。
- 采样：T=0.8、top-p=0.95、48 token。1MB 数据 + 1M 参数已能写出通顺的 TinyStories 风格句子；
  `results/train_demo.json` 里保存了 prompt / prompt_ids / output / generated_text / 参数。

---

## 6. 恢复等价性：2.4e-7 是哪里来的

`--verify-resume` 的判据是 `max_param_diff < 1e-6`，实测 **2.384e-07**，通过。但「为什么不是
逐位 0」值得记一笔，否则容易被误读成 checkpoint 有 bug。Lead 做了隔离实验：

1. **load 本身是逐位精确的**：把 3 步后的 model + optimizer 存盘，读进一个「不同初始化」的新
   实例，参数逐位相同（maxdiff 0.0）。
2. **两个同 seed、同初始化的模型，参数也逐位相同**（`torch.equal` 全 True），同一输入下
   forward logits 也逐位相同（maxdiff 0.0）。
3. **但 backward 不是完全确定**：`copy.deepcopy` 出来的两个相同模型，在同一 batch 上反向，
   grad 有 **1.46e-11** 的差异（例如权重梯度 GEMM 的归约顺序/内存对齐不同）。
4. 这个 1e-11 级别的差异会随训练步数累积：A 连续 6 步 vs B「3 步 + 存档 + 读档 + 3 步」，
   在 3 步处已有 **5.96e-08** 差异，6 步后放大到 **2.38e-07**。
5. 差异集中在 embedding / lm_head / 部分 LN 参数（见 `results/train_demo.json` 的
   `resume_check`），量级与 float32 单次归约误差一致。

结论：**checkpoint 逻辑正确，load 后状态逐位一致；剩余差异是 CPU float32 训练的固有
非确定性。** 因此判据用 1e-6 而不是逐位相等；如果要求逐位复现，需要固定 BLAS 线程/归约
顺序，或改用确定性算子（本 lab 不做）。

---

## 7. 踩坑清单

1. **AdamW 顺序**：必须先 decoupled weight decay（用 base lr），再更新 m/v，最后用
   `alpha_t` 更新参数。把 wd 加进梯度（Adam+L2）会被二阶矩归一化掉，和 `torch.optim.AdamW`
   对不上。
2. **cosine 边界**：`it == warmup_iters` 走 cosine 分支且 `cos(0)=1` → `max_lr`；
   `it == cosine_cycle_iters` → `min_lr`；`warmup=0` 时 `it=0` 不能进 warmup 分支（除零）。
   `Tw == Tc` 的退化配置 SPEC 未定义，实现返回 `min_lr`。
3. **get_batch 上界**：`randint(0, len - ctx)` 的上界是 exclusive，最大 start = `len-ctx-1`；
   写成 `len-ctx+1` 或闭区间会越界。官方 `test_get_batch` 就是盯这个。
4. **clip 的 eps**：缩放因子 `max/(norm+eps)`（eps=1e-6），不是 `max/norm`；只有
   `norm > max` 才动梯度，且要跳过 `grad is None` 的参数。
5. **`torch.load(weights_only=True)`**：从 torch 2.6 起默认就是 `True`，本 lab 显式写出，
   避免加载不可信 checkpoint 执行任意代码；optimizer state_dict 里的 int/tensor/list 都在
   白名单内。
6. **file-like checkpoint**：`BytesIO` 读之前要 `seek(0)`（`save_checkpoint` 之后指针在末尾）。
7. **`--max-seconds` 的边界**：它只约束训练循环，结束时的 eval/采样/存盘发生在循环外，
   所以总 wall 可能略超预算（2 s 预算实测 2.4 s）。默认 90 s 配置余量很大。
8. **resume 时的 loss 口径**：从 step 300 接续时，「loss 下降 20%」不适用（初始 loss 是
   step 300 的），脚本把该项标记为不适用，不误判失败。这是 exp 自测时发现并修掉的。
9. **临时 docstring 数字**：`prep_data.py` 最初写「默认实测约 3–6 秒」，实测 0.4 秒，
   Lead 收尾时改掉了。教学 lab 里数字要和实测一致，否则会误导读者。
10. **pytest 不在 venv**：测试写成纯函数，直接 `python` 跑；不要为了装 pytest 联网。
11. **`data/`、`checkpoints/`**：已加入 `.gitignore`；checkpoint 约 65MB，不要提交。

---

## 8. 限制与下一步

**限制**（细节见 README §7）：

- 规模是冒烟级：1MB 文本 / 1.05M 参数 / 600 步；val loss 2.98 证明链路通、能学，不是 A1 最终结果。
- 没做 LR sweep、batch size 对比、4 个消融；这些是 A1 §7 的实验部分。
- resume 判据 1e-6，不保证逐位复现（见 §6）。
- `get_batch` 用 Python 循环逐样本切片，冒烟规模足够；全量数据可改向量化 gather。
- 本机无 pytest；pytest 模式未实跑。

**下一步**：

1. 用 `bpe_lab` 在完整 TinyStories（~2GB）上训 10K/32K tokenizer，切 uint16 大文件；
2. 训 17M 参数模型（`d_model=512, layers=4, heads=16, ctx=512` 量级），跑 LR sweep；
3. 4 个消融（RMSNorm / RoPE / SwiGLU / pre-norm）短程训练比 val loss；
4. 生成与 perplexity 评测、写 report；leaderboard 可选。
