# bpe_lab：BPE 教学实现 + 两个可跑实验

这套代码是 [`BPE_TOKENIZER_TUTORIAL.md`](../BPE_TOKENIZER_TUTORIAL.md) 的配套 lab，目标是**让你把 BPE 从“看懂”变成“跑过、量过、改过”**。

> ⚠️ **这不是 CS336 A1 的提交答案**：API 与 `tests/adapters.py` 刻意不同（例如 `train_bpe` 直接吃字符串、参数名不同），代码优先可读性。如果你在修 CS336 学分，请按课程 AI 政策自己实现作业；把这里当参考实现来读，读完合上重写。

---

## 文件

| 文件 | 作用 |
| --- | --- |
| `toy_bpe.py` | 从零实现的 byte-level BPE：GPT-2 预分词、fast/naive 两条训练路径、`BPETokenizer`（encode / encode_iterable / decode / 保存加载）、GPT-2 格式序列化 |
| `test_toy_bpe.py` | 23 个测试，覆盖 handout 手算例子、tie-break、特殊 token、流式编码、大文件分块训练、roundtrip；无需 pytest 也能跑 |
| `demo_train.py` | 在小语料上训练 tokenizer，打印耗时/内存/压缩比/吞吐，输出 `results/toy_demo.json` |
| `transplant_gpt2.py` | 把自训 BPE 换到 GPT-2 上：embedding 投影初始化 → resize + tie → 继续训练 → 用 BPB 评估 |
| `control_original_tokenizer.py` | 对照：原 tokenizer、同预算继续训练（600 步全参；1 线程 + 降温约 11 分钟）。结果 0.7471，见 `results/control_original_tokenizer.json` |
| `results/` | 实验 JSON 结果与 BPB 曲线图 |
| `LAB_NOTES.md` | 运行记录、实测数字、踩坑（由实验执行者维护） |

---

## 快速开始

```bash
cd /Users/yunni/Projects/cs336-prep-code

# 离线运行所需的环境变量（home 缓存不可写时尤其重要）
export UV_CACHE_DIR=$PWD/.tmp/uvcache
export HF_HOME=$PWD/.tmp/hf
export MPLCONFIGDIR=$PWD/.tmp/mpl
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# 1) 测试（1.8 秒；纯 Python，不依赖 torch）
.venv/bin/python bpe_lab/test_toy_bpe.py

# 2) 训练一个小 tokenizer（默认 TinyStories 前 ~300KB、vocab 1000，约 1 秒）
.venv/bin/python bpe_lab/demo_train.py
.venv/bin/python bpe_lab/demo_train.py --bytes 1000000 --vocab-size 2000
.venv/bin/python bpe_lab/demo_train.py --compare-naive   # 顺带看 naive 慢多少

# 3) 换 tokenizer 移植实验（默认 vocab 4096、200+400 步；CPU 约 5 分钟）
#    ⚠️ 这会把 CPU 打满、机器明显发热；无风扇笔记本建议 --threads 2，或改小 --steps-phase2
.venv/bin/python bpe_lab/transplant_gpt2.py --threads 2
.venv/bin/python bpe_lab/transplant_gpt2.py --quick      # 几十秒冒烟版

# 4) 对照：原 tokenizer 同预算继续训练（600 步全参；1 线程 + 降温约 11 分钟）
#    默认 phase1=0、600 步全参 lr=1e-4；1 线程更慢但更凉
.venv/bin/python bpe_lab/control_original_tokenizer.py --threads 1 \
    --sleep-every 20 --sleep-seconds 0.4 --cooldown-every 200 --cooldown-seconds 60
```

> 环境：`.venv` 里是 torch 2.14 / transformers 5.21 / regex，**没有 pytest、没有 tiktoken**。跑 pytest 可用
> `uv run --with pytest python -m pytest bpe_lab/test_toy_bpe.py -q`，但直接 `python` 跑已经覆盖全部断言。

---

## 两个实验分别看什么

### `demo_train.py`

关注四件事：

1. **训练耗时 vs 语料/词表规模**（unique pre-token 数量比字节数更决定成本）；
2. **最长 token 是否合理**（TinyStories 上通常是 ` unexpected` 这类带前导空格的常见词）；
3. **压缩比**（1K 词表在 TinyStories 上约 3.2 bytes/token）；
4. **吞吐**（小语料缓存命中率高，数字偏乐观；真实语料会低不少）。

结果写入 `results/toy_demo.json`。

### `transplant_gpt2.py`

回答“**BPE 能不能装到已有模型上**”：

- 基线：GPT-2 + 原 tokenizer 在同一段 eval 文本上的 BPB；
- 换之后（未训练）：把 embedding 按“旧子词均值”投影到新词表，立刻测 BPB；
- 继续训练：先只训 embedding/LM head，再全参，逐步测 BPB。

实测结果（完整配置，见 `LAB_NOTES.md` §3.2–3.3）：基线 **0.956** → 换后未训练 **1.019**
（只差 +0.064 BPB，因为新旧词表 90.9% 的 token 完全相同）→ 600 步后 **0.782**；
同预算 control（原 tokenizer、600 步）是 **0.747**，所以换 tokenizer 的额外代价约 **+0.035 BPB**。
0.782 < 0.956 主要是 TinyStories 域适应，不是换词表的收益；结论口径见 `LAB_NOTES.md` §5.3。

结果写入 `results/transplant.json`、`results/control_original_tokenizer.json` 和
`results/transplant_bpb.png`（图：3 个完整 eval 检查点 + baseline + control 线）。

---

## 已知限制

- `demo_train.py` 默认语料是 TinyStories 的一个 5MB 采样文件，**不是**完整数据集；数字只用于教学对比。
- `transplant_gpt2.py` 在 CPU 上继续训练约 0.3M token，远小于 GPT-2 原始训练量；control 只覆盖这一种配置，结论不能外推到更大模型/跨语言词表。它演示的是流程与度量方法。
- 流式编码（`encode_iterable`）需要缓冲，不能按行切；细节见教程 §5.3。
