# training_lab —— CS336 A1 §4–6 训练工具教学实现

> **这是教学实现，不是 A1 的提交答案。** API 与官方 `tests/adapters.py` 故意不同（参数名更
> 「教学」，额外支持显式随机源 `rng` / `generator`，关键函数有中文 docstring 讲易错点），
> 核心目的是「写一遍、测一遍、跑一遍」，而不是替交作业。官方仓库 `assignment1-basics`
> 只被**只读**地用于 fixtures 与测试口径对照，全程未修改。

本 lab 把 A1 handout §4–6 的 8 个写代码的题（合计 15 分）从零实现，并接上
`bpe_lab` 的 tokenizer 与 `model_lab` 的 TransformerLM 跑一次**低资源冒烟训练**：
内存映射数据 → 前向 → cross-entropy → 反向 → clip → AdamW（cosine+warmup）→
周期 eval → checkpoint → 中断恢复 → 温度/top-p 采样。

---

## 1. 文件表

| 文件 | 说明 |
| --- | --- |
| `toy_training.py` | 训练工具箱：`cross_entropy` / `perplexity` / `AdamW` / `cosine_lr` / `set_lr` / `clip_gradients` / `load_tokens` / `get_batch` / `save_checkpoint` / `load_checkpoint` / `softmax_temperature` / `top_p_filter` / `sample_token` / `generate` |
| `test_toy_training.py` | 35 个双模式测试（A–H 八组，对应 SPEC §4；不需要 pytest 也能跑） |
| `prep_data.py` | 数据准备：TinyStories 前缀 → BPE（2048）→ uint16 `train.bin` / `val.bin` + `meta.json` |
| `train_demo.py` | 端到端训练脚本：memmap / AdamW / cosine / clip / eval / checkpoint / resume / 采样；`--max-seconds` 硬停 |
| `lead_spotcheck.py` | Lead 的独立复核：不 import 本 lab 的测试，直接用 `torch` / 官方期望值重算 8 项关键数值 |
| `results/train_demo.json` | 冒烟训练的机器可读结果（config / curve / summary / checkpoints / resume_check / samples / checks） |
| `SPEC.md` | 开工前定下的契约：API、数值约定、测试清单、实验口径、文件所有权 |
| `LAB_NOTES.md` | 实测数字、复现命令、恢复等价性分析、踩坑与限制 |
| `data/`、`checkpoints/` | 生成产物（已 gitignore，可随时用脚本重建） |

---

## 2. 快速开始

```bash
cd /Users/yunni/Projects/cs336-prep-code

# 0) 环境：仓库自带 .venv（Python 3.14.3 + torch 2.14.1），全部脚本 2 线程
#    所有命令都在仓库根执行，不需要联网、不需要安装新依赖

# 1) 工具箱自测：35 passed，约 0.6 秒
.venv/bin/python training_lab/test_toy_training.py

# 2) Lead 独立复核：8/8 checks，约 1.1 秒
.venv/bin/python training_lab/lead_spotcheck.py

# 3) 数据准备：1MB TinyStories 前缀 → uint16 token（实测约 0.4 秒）
.venv/bin/python training_lab/prep_data.py
.venv/bin/python training_lab/prep_data.py --force   # 强制重建
.venv/bin/python training_lab/prep_data.py --bytes 500000 --vocab-size 1024

# 4) 端到端冒烟训练：600 步 / 2 层 / 1.05M 参数，实测约 15 秒
.venv/bin/python training_lab/train_demo.py --verify-resume
.venv/bin/python training_lab/train_demo.py --steps 200 --max-seconds 30   # 更短
.venv/bin/python training_lab/train_demo.py --resume                       # 从 latest.pt 接续

# 5) 想换回较小模型/更快 demo
.venv/bin/python training_lab/train_demo.py --d-model 64 --num-layers 1 \
    --context-length 32 --batch-size 8 --steps 300
```

> 热预算：`torch.set_num_threads(2)`；所有单条命令实测 < 20 秒，`--max-seconds` 默认 90 秒硬停。
> 无风扇笔记本也能跑，不需要 MPS/CUDA。

---

## 3. 与 A1 的评分映射（15 分）

| handout 题 | 分值 | 本 lab 对应 |
| --- | --- | --- |
| `cross_entropy` | 1 | `cross_entropy`：先减 max 再 `logsumexp`，任意前导维求平均 |
| `adamw` | 2 | `AdamW`：严格按 Algorithm 1（先 decoupled WD、t 从 1、偏差修正步长） |
| `learning_rate_schedule` | 1 | `cosine_lr`：warmup / cosine / post-annealing 三段 |
| `gradient_clipping` | 1 | `clip_gradients`：全局 L2 norm、`max/(norm+eps)` 原地缩放 |
| `data_loading` | 2 | `load_tokens`（memmap）+ `get_batch`（exclusive 上界） |
| `checkpointing` | 1 | `save_checkpoint` / `load_checkpoint`（model + optimizer + iteration） |
| `training_together` | 4 | `train_demo.py`：CLI、memmap、周期日志/eval、checkpoint、resume、硬停 |
| `decoding` | 3 | `softmax_temperature` / `top_p_filter` / `sample_token` / `generate` |

未包含：`learning_rate_tuning`（§4.2.1 的 SGD 观察题）、`adamw_accounting`（§4.3 的显存/FLOPs
推导）、以及 §7 的 LR/batch sweep 与 4 个消融实验——这些属于「实验」而不是「写代码」，留给后续阶段。

---

## 4. 测试覆盖点（35 项，A–H）

| 组 | 项数 | 覆盖点 | 关键数值（实测） |
| --- | --- | --- | --- |
| A CE | 5 | 与 `F.cross_entropy` 的数值/梯度一致、`×1000` 大 logits 稳定、高维 batch、返回标量 | maxdiff 0；梯度 5.6e-9 |
| B AdamW | 7 | 官方 1000 步线性回归、decoupled WD、`betas=(0.9,0.95)`、param groups、state_dict 续训、非法 lr、closure | vs `torch.optim.AdamW` maxdiff **1.19e-07** |
| C schedule | 4 | 官方 25 个期望值、`Tw/Tc` 边界、`warmup=0`、`set_lr` | maxdiff **2.78e-17** |
| D clip | 4 | vs `clip_grad_norm_`、小梯度不动、裁剪后 ≈ max、None grad 跳过 | 梯度 maxdiff 0；norm diff 1.0e-06 |
| E data | 5 | 形状/偏移、起点均匀性（5σ）、memmap、rng 可复现、非法 device | 93/93 起点覆盖 |
| F ckpt | 4 | 官方往返口径、中断恢复等价、BytesIO、返回 iteration | 恢复后参数逐位一致 |
| G decode | 5 | 极低温=argmax、`top_p=1`、最小核、熵随温度下降、generate 的 limits/eos/seed | 手工核 {idx2} 保留 |
| H 集成 | 1 | 小模型 50 步 loss 下降 > 50% | 3.81 → 0.18（−95%） |

`lead_spotcheck.py` 是另一条独立路径：它只用 `torch` 的参考实现和官方 25 个期望值重算，
不 import 本 lab 的测试代码，避免「测试自己证明自己」。8 项全过（约 1.1 秒）。

---

## 5. 冒烟训练实测

默认配置（`--verify-resume`，实测 **14.8 秒**，2 线程，CPU）：

| 项 | 值 |
| --- | --- |
| 数据 | TinyStories 前 1MB → BPE 2048 → 257,477 train / 13,552 val token（3.69 bytes/token） |
| 模型 | `d_model=128, layers=2, heads=4, d_ff=512, ctx=64`，untied，**1,049,216 参数** |
| 训练 | batch 16 × ctx 64，600 步，lr 3e-3 → 1e-4，warmup 20，wd 0.01，clip 1.0 |
| 吞吐 | **44,643 tokens/s**（训练循环 13.8 秒，共 614,400 token） |
| train loss | **7.6663 → 2.9949（−60.93%）** |
| val loss（固定 batch） | **7.6458 → 2.9788**（step 100/200/300/400/500/600：3.918 / 3.528 / 3.276 / 3.085 / 2.996 / 2.979） |
| 采样 | T=0.8、top-p=0.95，48 token；已能写出通顺的 TinyStories 风格续写 |
| 显存/内存 | peak RSS **386 MiB** |
| checkpoint | step 200 / 400 / 600 + `latest.pt`（约 65MB，gitignored） |

`--verify-resume` 的结果：

| 项 | 值 |
| --- | --- |
| 方案 | A=连续 6 步；B=3 步 → 存 → **换一个初始化**的新模型读档 → 再 3 步 |
| `loaded_iteration` | 3 |
| 最终参数 maxdiff | **2.38e-07**（< 1e-6，通过） |
| val loss diff | 0.0 |
| 说明 | load 本身逐位精确；2.4e-7 来自两个独立模型实例在 CPU backward 中的浮点归约顺序差，会随步数累积，不是 checkpoint 逻辑错误（详见 `LAB_NOTES.md` §7） |

---

## 6. 关键实现点（最容易写错的地方）

1. **CE**：`logsumexp(logits) - logits[target]`，先减 max；不要先 `softmax` 再 `log`。
2. **AdamW 顺序**：weight decay 必须在 moment 更新**之前**、用 base lr；`t` 从 1 开始；
   `alpha_t = lr * sqrt(1-beta2**t) / (1-beta1**t)`。顺序错一处就会和 `torch` 对不上。
3. **cosine 边界**：`it == Tw` 走 cosine 分支（`cos(0)=1`）→ max；`it == Tc` → min；
   `Tw=0` 时 `it=0` 返回 max，不能除零。
4. **clip**：全局 norm 用 Python float 累加；缩放因子是 `max/(norm+eps)`（不是 `max/norm`）；
   只在 `norm > max` 时动梯度。
5. **get_batch**：上界是 `len(dataset) - context_length` 的 **exclusive** 区间，
   最大 start = `len - ctx - 1`；写成闭区间会在最后几个位置越界。
6. **checkpoint**：`torch.load(..., weights_only=True)` 更安全；file-like（BytesIO）要自己
   `seek(0)`；恢复时 iteration 决定 lr schedule 与日志接续。
7. **top-p**：先降序排序再累计；保留条件用 `cumsum - sorted_probs < p`（等价于「累计和首次
   ≥ p 的那个 token 也保留」），保证至少留 1 个；最后重新归一化。
8. **generate**：EOS 判断在 append 之后（EOS 保留在返回值里）；context 截断只影响喂模型的
   窗口，不影响返回的完整序列。

---

## 7. 已知限制

1. **规模是冒烟级**：1MB 文本 / 1.05M 参数 / 600 步。val loss 2.98 说明「链路通了、能学」，
   但远不是 A1 要求的 17M 模型 + 完整 TinyStories。正式实验请用全量数据与更大模型。
2. **clip 默认未触发**：实测梯度范数 0.5–0.76 < 1.0，所以 `--grad-clip 1.0` 实际是安全网；
   想观察裁剪行为可跑 `--grad-clip 0.5`。
3. **`--max-seconds` 只约束训练循环**：结束时的 eval/采样/存盘会让总 wall 略超预算
   （2 秒预算实测 2.4 秒）；默认配置余量很大。
4. **resume 不是逐位 0**：见 §5 与 LAB_NOTES §7；判据是 < 1e-6。
5. **本机没有 pytest**：测试是纯 `test_*` 函数，直接 `python` 跑已验证；pytest 收集规则兼容但
   未在仓库 venv 实跑。
6. **`data/`、`checkpoints/` 不提交**：体积大且可由 `prep_data.py` / `train_demo.py` 重建。

---

## 8. 下一步

按 A1 路线，本 lab 之后最顺的是「实验部分」：

1. 用 `bpe_lab` 在完整 TinyStories 上训 10K/32K tokenizer，切成 uint16 大文件；
2. 训 17M 模型，做 LR sweep（3 个量级）与 batch size 对比；
3. 4 个消融：RMSNorm / RoPE / SwiGLU / pre-norm，各自训短程并比 val loss；
4. 生成与 perplexity 评测，写进 report（leaderboard 可选）。

本 lab 的 `train_demo.py` 已经是这些实验的骨架：换数据路径、模型配置、步数即可复用。
