# bpe_lab —— 从零手写 byte-level BPE，并把它「换」到 GPT-2 上

> **先声明：这是教学代码，不是 CS336 A1 的提交答案。**
> `toy_bpe.py` 的 API 故意和 handout 的 `adapters.run_train_bpe` / `adapters.get_tokenizer` 不同
> （`train_bpe(corpus: str, ...)` 直接吃文本、不读文件）；也没有做并行 pretokenization 之类的
> 性能工程。语义对齐 handout §2.4–2.6，但如果你要交 A1，请自己按 handout 写一遍并跑官方
> `tests/`。这里的所有数字都来自本机实际运行，命令见 §2。

---

## 1. 文件与产物

| 文件 | 作用 |
| --- | --- |
| `bpe_lab/toy_bpe.py` | 教学版 byte-level BPE：训练（fast 增量 / naive 全量）、`BPETokenizer`（encode / encode_iterable / decode）、GPT-2 格式 `save`/`load` |
| `bpe_lab/test_toy_bpe.py` | 20 个测试，双模式（`python 直接跑` / `pytest`）；含 handout 手算例子、A1 官方 reference 对照、transformers GPT-2 逐 token 对照 |
| `bpe_lab/demo_train.py` | 在 TinyStories 前 ~300KB 上训练 vocab=1000，打印/保存耗时、内存、压缩比、吞吐 |
| `bpe_lab/transplant_gpt2.py` | 核心实验：训练 4096 词表 BPE，替换 GPT-2 tokenizer，测三点 bits-per-byte + 初始化消融 |
| `bpe_lab/results/toy_demo.json` | demo 的机器可读结果 |
| `bpe_lab/results/transplant.json` | 移植实验的全部结果（配置/曲线/BPB/耗时/机器信息/自检） |
| `bpe_lab/results/transplant_bpb.png` | 只用**完整 eval 三个点**画的 BPB 图（+ baseline 线 + 初始化消融 inset） |

环境：macOS 27.2 arm64（Apple Silicon，10 核），Python 3.14.3，torch 2.14.1，
transformers 5.21.0，tokenizers 0.23.3，regex，numpy，matplotlib。
**没有 pytest、没有 tiktoken**（pytest 通过 `uv run --with pytest` 临时环境跑）。

> ⚠️ 热约束：用户反馈无风扇机器发烫，所以按 Lead 要求停止了所有 CPU 密集任务。
> 因此 **`control_original_tokenizer`（同预算、原 tokenizer 的对照）没有跑**，
> 也没有补跑 512 词表的独立消融（只有早先 `--quick` 冒烟数字）。这会影响结论口径，见 §5。

---

## 2. 复现命令（全部实测）

所有命令都在仓库根目录执行，并先设置环境变量（沙箱里 home 不可写）：

```bash
cd /Users/yunni/Projects/cs336-prep-code
export UV_CACHE_DIR=$PWD/.tmp/uvcache \
       HF_HOME=$PWD/.tmp/hf \
       MPLCONFIGDIR=$PWD/.tmp/mpl \
       HF_HUB_OFFLINE=1 \
       TRANSFORMERS_OFFLINE=1
```

### 2.1 单文件自检（复现 handout 的 low/lower/widest/newest 例子，<1s）

```bash
.venv/bin/python bpe_lab/toy_bpe.py
# 输出：merges: ['s t', 'e st', 'o w', 'l ow', 'w est', 'n e', 'ne west', 'w i', 'wi d', 'wid est', 'low e', 'lowe r']
#       vocab_size: 269（256 字节 + 1 special + 12 merges）
```

### 2.2 测试（双模式，20 个测试，~1.6–1.8s）

```bash
# 直接跑（不需要 pytest，失败会 exit 1）
.venv/bin/python bpe_lab/test_toy_bpe.py
# 输出：汇总: 20 passed, 0 failed, 0 skipped

# pytest（会新建临时环境，慢一点；因为本仓库没有 pyproject，不要 uv sync）
uv run --with pytest python -m pytest bpe_lab/test_toy_bpe.py -q
# 输出：20 passed in 1.58s
```

覆盖点：12 条 merge 精确序列、6 merges 时 st/est/ow/low/west/ne 与 `newest → [ne, west]`、
tie-break（`(BA, A)`）、§2.6.1 `"the cat ate" → [9, 7, 1, 5, 10, 3]`、
与 A1 官方 `train-bpe-reference-{merges,vocab}.json`（corpus.en, vocab=500）**完全一致**、
与 transformers GPT-2 逐 token 一致（中文/emoji 也一致）、UTF-8 往返、special token 硬边界、
重叠 special 最长匹配、非法 id → U+FFFD、`encode_iterable` 惰性与一致性、词表数量关系与压缩比。

### 2.3 demo：小语料训练 + 指标（总耗时 < 1s）

```bash
# 默认配置（TinyStories 前 320KB，vocab 1000）
.venv/bin/python bpe_lab/demo_train.py
# 加 --compare-naive 会在 30KB 小片额外跑一次 naive 训练做速度对照（本目录 JSON 即此命令产出）
.venv/bin/python bpe_lab/demo_train.py --compare-naive
```

### 2.4 transplant：把 BPE 换到 GPT-2 上（本目录 JSON 对应的完整命令，CPU 约 5 分 13 秒）

```bash
.venv/bin/python -u bpe_lab/transplant_gpt2.py --ablation-steps 50
```

冒烟测试（vocab 512、40 篇 eval 文档、4+8 步，约 25s；注意会写默认输出路径）：

```bash
.venv/bin/python -u bpe_lab/transplant_gpt2.py --quick \
    --out bpe_lab/results/transplant_quick.json \
    --png bpe_lab/results/transplant_quick.png
```

---

## 3. 关键数字

### 3.1 toy BPE demo（`results/toy_demo.json`，`--compare-naive` 那次运行）

| 指标 | 数值 |
| --- | --- |
| 训练语料 | TinyStories fixture 前 299,896 字节（训练段 299,754 字符；按 `<|endoftext|>` 切在文档边界） |
| 留出段 | 20,248 字节 |
| 词表 | 1000（256 字节 + 1 special + **743 merges**） |
| 训练耗时 | **0.124 s**（fast 实现；unique pre-token 仅 2,923 个） |
| 峰值内存 | 训练后 53.4 MB；脚本结束 56.4 MB |
| 最长 token | `b' unexpected'`（11 字节） |
| top5 最长 token | `b' unexpected'`, `b' something'`, `b' happened'`, `b' remember'`, `b' together'` |
| 留出段压缩比 | **3.2063 bytes/token**（0.31188 tokens/byte） |
| roundtrip | OK（`decode(encode(x)) == x`） |
| encode 吞吐（冷缓存） | 11,835,372 bytes/s（0.0253 s / 300KB） |
| encode 吞吐（热缓存） | 17,831,052 bytes/s |
| encode_iterable 吞吐（冷缓存） | 11,060,151 bytes/s（0.0271 s，与整段 encode 一致） |
| naive vs fast（30KB, vocab 1000） | 0.0313 s vs **0.4542 s**，14.51× 加速，merges/vocab 完全一致 |
| 脚本总耗时 | 0.686 s（含 naive 对照） |

说明：该语料只有 2,923 个 unique pre-token，pre-token 缓存命中率极高，所以吞吐数字偏乐观；
换成长文本/更杂的语料会明显下降。

### 3.2 transplant：换 tokenizer（`results/transplant.json`）

语料：TinyStories fixture 6,458 篇文档，按 `<|endoftext|>` 切分。
**前 80% 文档训练新 tokenizer（4,199,316 字节），80%–85% 的文档做 eval（323 篇，259,849 字节）。**

| 指标 | 原 tokenizer（GPT-2） | 新 tokenizer（自训 BPE） |
| --- | --- | --- |
| 词表大小 | 50,257 | **4,096**（3,839 merges，special=`<|endoftext|>`） |
| tokenizer 训练 | — | **0.91 s**（4.2MB，单线程） |
| eval tokens | 63,744 | 65,132 |
| tokens/byte | **0.24531** | **0.25065**（+2.2%） |
| bytes/token | 4.076 | 3.990 |
| 最长 token | — | `b' uncomfortable'`（14 字节） |

**三点 BPB（完整 eval，滑窗 256、batch 4、无梯度；分母 = 被计分 token 的原始字节，覆盖率 99.60%）**

| 阶段 | BPB | 相对 baseline | 耗时 |
| --- | --- | --- | --- |
| ① 换之前：GPT-2 + 原 tokenizer | **0.95581** | — | 13.91 s |
| ② 换之后：未训练 | **1.01947** | **+0.06366**（+6.7%） | 11.05 s |
| ③ 相位 1 后（200 步，只训 embedding+lm_head） | **0.92350** | −0.03231 | 11.56 s（训练 47.9 s） |
| ④ 相位 1+2 后（600 步，全参） | **0.78218** | −0.17363 | 12.77 s（相位 2 训练 160.9 s） |

| 其他 | 数值 |
| --- | --- |
| 词表重叠（token 类型） | **3724 / 4095 = 90.94%** 的新 token 字节序列恰好等于某个旧 token；多字节 token 中 3468/3839 = 90.34% |
| 词表重叠（eval 出现次数） | **98.46%** 的 eval token 出现落在这 90.94% 的共享 token 上 |
| 初始化方式 | special 1 行复制 50256；单字节 256 行按字节反查；合法 UTF-8 的 3838 个多字节 token 取旧 tokenizer 编码的均值；1 个非法 UTF-8 token 取逐字节均值 |
| 训练数据 | 训练文档前 400,000 字节 → 100,109 个新 token id，随机 crop |
| 训练配置 | phase1 200 步 lr=1e-3（只训 tied embedding 矩阵，AdamW，clip 1.0）；phase2 400 步全参 lr=1e-4；batch 4×128；`torch.set_num_threads(8)` |
| 总耗时 / 峰值内存 | **313.4 s（5 分 13 秒）/ 3,298.5 MB**（峰值主要来自消融时同时加载两份 GPT-2） |
| 自检 | baseline 在 [0.5,3] ✓；换后变差 ✓；训练后变好 ✓；曲线无 NaN ✓；新旧 tokenizer `decode(encode(eval))==eval` ✓；`lm_head` 与 `wte` 共享存储 ✓ |
| 机器 | macOS-27.2-arm64，10 核；torch 2.14.1 / transformers 5.21.0；线程 8 |

**初始化消融（同一批 batch、同 seed、同 50 步，只训 embedding+lm_head，完整 eval）**

| 初始化 | 50 步后 BPB | 初始 train loss | 末 10 步 train loss |
| --- | --- | --- | --- |
| 均值初始化（旧子词 embedding 均值） | **1.00433** | 3.0383 | 2.9574 |
| resize 默认随机初始化（normal, std=0.02） | **2.17651** | 7.5254 | 6.2791 |
| 差值 | **1.17218 BPB** | | |

**quick 冒烟数字（vocab=512、40 篇/33,554 字节 eval、4+8 步，配置与主实验不同，仅量级示意）**

| 阶段 | BPB |
| --- | --- |
| 基线（原 tokenizer） | 0.9827 |
| 换后未训练 | **2.4309**（tokens/byte 0.4156，比旧的 0.2477 多 68%） |
| 4 步后 | 2.3648 |
| 12 步后 | 1.9968 |

这组数字说明：**词表太小 → 序列显著变长 → 未训练 BPB 明显变差**。但 eval 子集不同
（33.5KB vs 主实验 259.8KB），不能和主实验数字直接相减。

---

## 4. 实现要点 & 踩坑记录（教程可以引用）

### 4.1 handout 语义（容易写错的四处）

1. `vocab_size` **包含** special token 和所有 merge 产物：
   `len(merges) == vocab_size - 256 - len(special_tokens)`。
2. special token：训练时是**硬边界**（`re.split` 掉、不参与 pair 统计），编码时整体映射成一个 id；
   若不在给定 vocab 里，追加到末尾。
3. tie-break：频次相同取 **lexicographically greater** 的 pair（handout 的 `max(...)`）。
4. 编码不是「按 merge 列表从头重放」，而是**每步挑当前相邻 pair 中 merge rank 最小者**；
   否则像 `(a, b, c)` 这种词会合错。`BPETokenizer._encode_pretoken` 有注释说明。

### 4.2 预分词器必须可注入

handout 的 low/lower/widest/newest 手算例子假设「按空白切」；GPT-2 正则会把前导空格并进 token，
pair 频次完全不同。所以 `train_bpe(..., pretokenizer=whitespace_pretokenize)` 这个参数不是装饰。

### 4.3 训练加速：反向索引 + 懒删除堆（4.2MB / 4096 词表 = 0.91s）

- 先统计 `Counter[tuple[bytes,...] -> 频次]`（unique pre-token 只有 7,596 个，因为 TinyStories 重复度高）；
- `pair_counts` 存 pair 频次，`pair_to_words` 存「pair → 含它的 pre-token 集合」；
- 每轮 merge 只更新受影响的 pre-token；用 `heapq` + 懒删除取当前最高频 pair；
- 为满足 tie-break，堆里放 `(-count, _GreaterPair(pair))`，`_GreaterPair.__lt__` 把字典序反过来。
- `naive=True` 保留「每轮全量重数」版本做教学对照（30KB 语料上 14.5× 差距，merges 完全一致）。

### 4.4 `encode_iterable`：两个真实反例（都实测过）

不能「逐 chunk 独立 `encode` 再拼接」。除了大家熟知的「切在 pre-token 中间」，还有两个更隐蔽的：

1. **空白 run 跨块**：`"a\n\nb"` 整段切出来是 `['a', '\n', '\n', 'b']`；但如果 buffer 正好停在
   `"a\n\n"`（字符串结尾），正则 `\s+(?!\S)` 会把 `"\n\n"` 当成**一个** pre-token，
   单独编码就会得到 GPT-2 的 `'\n\n'` token，和整段结果不一致。
2. **撇号缩写跨块**：buffer 停在 `"I'l"` 时，`"'l"` 不构成 `'(?:[sdmt]|ll|ve|re)`，`"'"`
   会被当标点 token；而整段文本里 `"'ll"` 是一个 token。注意这种不稳定会**波及倒数第二个** pre-token。

正确规则（本实现）：先留出「尾部是某个 special token 真前缀」的部分，然后**只 flush 到最后一个词之前的
空白 run 起点**（即把「空白 run + 最后一个词」留在 buffer）；对 buffer 做一次预分词，**绝不把 safe 前缀
重新预分词**。这样每个编码单元的切法都来自同一份上下文。
验证：714 组（文本 × chunk size ∈ {1,2,3,5,7,13,64,997} + 随机切块，含连续空行、中文、emoji、
`"abc\n\n def"`）与整段 `encode` **逐 id 完全一致**；惰性用「会记录消费量的生成器」验证。

### 4.5 序列化

GPT-2 的 bytes↔unicode 双射（空格 → `Ġ`）写 `vocab.json` / `merges.txt`（首行 `#version: 0.2`），
`from_files`/`save`/`load` 可逆。测试里用官方 `gpt2_vocab.json` + `gpt2_merges.txt` 构造 tokenizer，
和 transformers 的 GPT-2 逐 token 对齐（含中文、emoji、德语变音）。

### 4.6 transplant：换 tokenizer 的工程细节

1. **单字节 token（新 id 0–255）**：从旧词表反查「字节序列恰好是该单字节」的 token id，直接复制行。
   （旧 GPT-2 词表 256 个单字节 token 齐全，脚本里 assert 了这一点。）
2. **多字节 token**：`bytes.decode("utf-8", strict)` 成功 → 用旧 tokenizer 编码成旧 id，取这些行的均值；
   失败（非法 UTF-8，通常是半个字符）→ 对它包含的每个字节取对应单字节 embedding 的均值。
3. **special token**：复制旧 `<|endoftext|>`（50256）那一行。
4. `resize_token_embeddings(4096, mean_resizing=False)` 后 **覆盖** `transformer.wte.weight`，
   再 `model.tie_weights()`，并断言 `lm_head.weight.data_ptr() == wte.weight.data_ptr()`
   —— GPT-2 是 tied，忘了 tie 会让输出头还是随机初始化。
5. `config.vocab_size/eos_token_id/bos_token_id` 改成新值。

### 4.7 BPB 公式与 eval 细节

`bpb = Σ CE(nats) / ln2 / n_bytes`，滑窗 256、batch 4、`no_grad`。
分母用**被计分 token 的原始字节数累加**（而不是 `decode` 片段后 `len(...encode())`）：
窗口边界可能切断多字节字符，`decode(errors="replace")` 会改变字节数；按 token 原始字节累加才精确。
每个窗口第 0 个 token 没有上下文、不参与计分（handout 的 `ids[:, :-1]` 写法），
所以覆盖率是 99.60%（不是 100%），JSON 里 `coverage` 字段可查。

### 4.8 环境/工具坑

- 必须设 `HF_HOME` / `MPLCONFIGDIR`（home 不可写）、`HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`（不联网）。
- 本仓库没有 `pyproject.toml`，**不要 `uv sync`**；pytest 用 `uv run --with pytest`。
- macOS 上 `resource.getrusage().ru_maxrss` 单位是**字节**（Linux 是 KB），脚本里做了区分。
- `safetensors` 加载有进度条噪声（transformers 5.x 的 `Loading weights`），属正常。
- 画图前 matplotlib 首次运行要建字体缓存（`MPLCONFIGDIR` 指到 `.tmp/mpl`）。

---

## 5. 结果解读（重要，别过度解读）

### 5.1 为什么「换 tokenizer」只掉 0.064 BPB？

因为**两个 tokenizer 的 vocab 高度重叠**：新 4096 词表里 **90.94%** 的 token 字节序列恰好等于某个
旧 GPT-2 token；在 eval 文本的 token **出现次数**里，**98.46%** 都落在这些共享 token 上。
对共享 token，我们把旧 embedding 行原样复制（或取均值），tied 输出头也同步，所以模型的
next-token 分布几乎没变；真正的代价来自 2.2% 的序列变长、9% 的没见过 token（用均值初始化）和 id 重映射。

这不是巧合：两边都是 byte-level BPE、都在英文上训练，最靠前的几千条 merge 高度一致
（3839 条新 merge 中 68.0% 与 GPT-2 的 merge 对完全相同）。**如果换成一个低重叠的 tokenizer
（例如 vocab=512，或不同语言/预分词方式），未训练 BPB 的退化会大得多**——quick 冒烟里
vocab=512 时是 2.43 vs 基线 0.98（不同 eval 子集，仅量级示意）。

### 5.2 为什么随机初始化这么差，均值初始化几乎无损？

50 步消融：mean-init **1.004** vs random-init **2.177**（同数据、同 batch、同 seed、同步数）。
随机初始化的 embedding 和已训练好的 transformer 不匹配，前向立刻崩掉（初始 train loss 7.53 vs 3.04），
50 步只能学回一部分；而均值初始化让新词表和旧模型的表示空间对齐，几乎不需要重新学。

### 5.3 为什么「训练后 0.782 < baseline 0.956」不能直接说「换 tokenizer 有收益」？

因为继续训练用的是 **TinyStories 训练文档**，而 eval 是同分布留出文档——这 600 步本身就带来了
**域适应收益**。要拆开「域适应」和「换 tokenizer 的代价」，需要一个对照：
**同一个 GPT-2、原 tokenizer、同样 600 步全参 lr=1e-4、同数据同 seed**，测同一 eval。

**这个 control 没有跑**（原因见 §1：用户要求立即停止 CPU 密集任务）。因此本目录的结论只到：

- 换 tokenizer 的**瞬时代价**：BPB **0.9558 → 1.0195（+0.0637，+6.7%）**，与新 tokenizer 的
  tokens/byte 只差 2.2% 一致；
- 在新 tokenizer 上继续训练 600 步后 BPB 为 **0.7822**；
- **不能**下结论说「移植后继续训练优于原 tokenizer」——需要上面那个 control 才能回答。

补 control 的配方（如果之后允许跑）：约 40 行脚本——加载原 GPT-2（不 resize），用**原 tokenizer**
把同一段 400KB 训练文档编码成 ids，batch 4×128、600 步全参、lr=1e-4、AdamW、clip 1.0、同 seed，
在同一个完整 eval 上按同一 BPB 公式评估；预计 CPU ~4 分钟。

### 5.4 图与表的口径

`transplant_bpb.png` 只画**完整 eval 的三个点**（step 0 / 200 / 600）+ baseline 水平线 +
初始化消融 inset；点之间是半透明虚线，表示「只是视觉引导，没有测中间点」。
JSON 里仍保留每 100 步的 `training.curve`，但它是在 **8KB eval 切片**上测的（约 10 篇文档、~500 token），
采样方差可达 0.08 BPB：例如 step600 的 slice BPB = 0.8675，而完整 eval = 0.7822（差 0.085）。
所以教程正文**引用完整 eval 的数字**，不要用 slice 曲线和 full eval 混着说。

---

## 6. 局限

1. **control 未跑**（见 §5.3）：不能回答「同预算下换 tokenizer 是收益还是损失」。
2. **512 词表消融只有 quick 冒烟**，eval 子集与主实验不同，只能示意量级。
3. 单机 CPU、单 seed（seed=0）、单模型（GPT-2 small）；换更大模型/更多步数，结论可能不同。
4. 随机初始化消融用的是 `resize_token_embeddings` 默认初始化（normal, std=0.02），
   不是 xavier/kaiming；不同随机初始化方案结论可能不同。
5. eval 是**非重叠**滑窗（窗口 256、stride 256），覆盖 99.60% 字节；与 stride=128 的常见做法
   在数值上会有小差异。
6. 继续训练数据与 tokenizer 训练数据同为「训练文档」（这是正常做法），eval 文档与二者都不相交；
   但 TinyStories 本身是儿童故事，分布简单，BPB 数字不能外推到真实网页文本。
7. 没做并行 pretokenization（本实现单线程 4.2MB/0.91s 已够用）；也没和 tiktoken 对比（未安装）。
8. 峰值内存 3.3GB 主要是初始化消融时同时持有两份 GPT-2，不是 tokenizer 训练本身（demo 只有 ~56MB）。
9. 所有数字都是在本机（macOS arm64）跑的，换机器会有小幅波动；但 tokenizer 训练/<br>
   BPB 的结论（重叠率、初始化重要性）不依赖具体机器。

---

## 7. 一句话总结（给教程正文）

> 用 4.2MB TinyStories 训出的 4096 词表 byte-level BPE，在 90.94% 词表重叠、压缩比 3.99 vs 4.08
> bytes/token 的条件下，直接换到 GPT-2 上只损失 **0.064 BPB（6.7%）**；新 embedding 用旧子词均值
> 初始化后，同 50 步比随机初始化好 **1.17 BPB**。600 步继续训练后 BPB 到 0.782，但因为缺少
> 「原 tokenizer 同预算」对照，不能把这部分归因于 tokenizer 本身。
