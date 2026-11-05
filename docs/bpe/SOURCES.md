# CS336 A1 · BPE 材料地图、精确规格、测试地图与自测题（SOURCES.md）

> 这份文件是教程正文的**事实与引用底座**，不写代码教程。正文里每个关键说法都应能在此按 `path:line` 回溯。
> 所有引用都在 2026-11-03 用 `grep -n` / `sed -n` / 小脚本现场核对过；无法核实的统一标「待核实」。

## 0. 约定

**路径缩写**（下文一律用缩写 + 行号，展开如下）：

| 缩写 | 绝对路径 | 性质 |
|---|---|---|
| `H` | `/Users/yunni/Projects/cs336-prep-code/.tmp/a1_handout.txt`（由 `A1/cs336_assignment1_basics.pdf` 抽取，§2 在 PDF p.3–12） | 一手：**唯一 spec** |
| `A1` | `/Users/yunni/Projects/assignment1-basics` | 一手：代码骨架 + 官方测试（**只读**） |
| `C` | `/Users/yunni/Courses/cs336-notes/course_notes/lecture_01/lecture_01.py` | 一手：2026 L1 可执行讲稿（toy 实现） |
| `EN` | `/Users/yunni/Courses/cs336-notes/stanford-cs336-2026-notes-main/lecture-01.md`（另有 `lecture-03.md`） | 一手（英文课程笔记，2026 版） |
| `CN` | `/Users/yunni/Courses/cs336-notes/跟着 stanford 动手学大模型/跟着 stanford 动手学大模型.md` | 二手（自述「部分内容可能由 AI 生成」，`CN:1972`） |
| `KV` | `/Users/yunni/Courses/karpathy-gpt-roadmap/{minbpe-comment,tokenizer-comment}/notes.md` | 二手 |
| `PLAN` | `CS336_A1_DETAILED_PLAN.md` | 计划 |
| `RM` | `CS336_ROADMAP.md` | 计划 |

**结论标记**：`核对无误` = 与一手材料逐字/逐数吻合；`勘误` = 与一手材料冲突，给出正确说法；`待核实` = 材料未给答案或本机缺数据，不许当结论用。

**本机环境限制（影响哪些结论可核）**：无 `tiktoken`、非 Linux（`rlimit` 不可用）、无 TinyStories/OWT 数据集。所以：与 tiktoken 逐 id 对比的结论只能靠 `A1/tests/fixtures/gpt2_*` 复现（已做）；「最长 token 是多少」类结论**本机无法复现**。

---

## 1. 材料地图

### 1.1 一手材料（spec / 测试 / 讲稿）

| 路径 | 覆盖内容 | 建议顺序 | 预计用时 | 备注 / 坑 |
|---|---|---|---|---|
| `H:88-481`（§2.1–2.7） | Unicode→编码→子词→BPE 训练→实验→编码/解码→实验题 | ① | 60–90 min 精读 | 唯一权威 spec；`vocab_size` 语义、tie-break、special token 规则都在这里。§2.1–2.2 只为 unicode1/2 两题存在，可先跳到 §2.3 |
| `H:191-284`（§2.4 + `bpe_example`） | 训练三步、tie-break 原文例子、12 条 merge 序列 | ② | 25 min + 手算 30 min | **必须手算一遍**，见本文 §4 |
| `A1/tests/test_train_bpe.py`（88 行） | 训练正确性/特殊 token/速度 3 个测试 | ③ | 10 min 读 + 反复跑 | 断言的是「merges 完全相等」→ tie-break 错一个就全挂 |
| `A1/tests/test_tokenizer.py`（464 行） | Tokenizer 25 个测试 + 1MB 内存测试 | ④ | 20 min | 两个内存测试在 macOS 被 skip（见 §2.7） |
| `A1/tests/common.py:10-51` | GPT-2 byte↔unicode remap | ④ 顺带 | 5 min | 序列化格式的唯一依据，见 §2.8 |
| `A1/tests/fixtures/*`（`gpt2_vocab.json`、`gpt2_merges.txt`、`train-bpe-reference-*`） | 真实 GPT-2 词表/合并 + 500 词表参考输出 | ④ | 15 min 抽查 | 实测：gpt2 词表 50,257 项、merges 50,000 行；参考 500 词表 / 243 merges |
| `A1/cs336_basics/pretokenization_example.py:5-49` | 并行预分词的 chunk 边界（按 special token 切） | ⑤ 做性能时 | 5 min | 可直接用；`mini_chunk_size=4096`（`:28`） |
| `A1/tests/adapters.py:542-592` | 两个适配器 + 参数 docstring | ⑤ | 5 min | `:580` 有与 handout 冲突的措辞，见 §2.11 |
| `C:484-758`（`tokenization()`） | 字符/字节/词/BPE 四种 tokenizer、toy `train_bpe`、`merge`、压缩比 | ⑥（先看视频再读） | 视频约 13–15 min + 精读 30 min | toy 实现**故意与作业规格不同**（无预分词、无 special token、tie-break 用插入序），见 §4.6 |
| `EN:119-183`（§5） | 压缩比定义、为什么字符/字节/词都不行 | ⑥ | 20 min | 压缩比 r=bytes/token（`EN:143-153`）最适合做正文定义引用 |
| `EN:184-332`（§6） | BPE 训练算法、`the cat in the hat` 三步 trace、编码、§6.4 工程清单 | ⑥ | 35 min | §6.4 是「作业比 toy 多了什么」的最佳清单（`:323-330`） |
| `EN:284-291`（lecture-03.md） | 词表规模 vs 模型规模、BPB 公式 | ⑦ 做实验题时 | 10 min | `bits-per-byte` 出处 |

### 1.2 二手笔记 / 计划

| 路径 | 覆盖内容 | 建议顺序 | 预计用时 | 备注 / 坑 |
|---|---|---|---|---|
| `CN:1764-1960`（§补充 1） | 按作业题号讲 BPE：unicode2、train_bpe 骨架、TinyStories/OWT、Tokenizer 类 | ⑧ **做完题之后**对照 | 25 min | **有 AI 生成成分**；实测 6 处需改/待核实，见 §6.2。`1.4` 只有标题没有正文（`:1960-1961`） |
| `KV/minbpe-comment/notes.md`（155 行） | minbpe 代码结构、`train(text, vocab_size)` 签名 | ⑨ 卡住时 | 15 min | 二手；`BasicTokenizer` 起始词表 256、无 special token（`notes.md:97`），与 A1 规格不同 |
| `KV/tokenizer-comment/notes.md`（154 行） | BPE 直觉、GPT-2 50,000 merges / 50,257 词表 | ⑨ | 15 min | 数字对得上 fixture（实测一致），但只作直觉补充 |
| `PLAN:213-239`（W1–W2） | 两天到一周的读/写/验收节奏 | 随时 | 5 min | `:239` 明确「macOS skip ≠ 通过」 |
| `PLAN:190,320`；`RM:194,208` | macOS skip 两个内存测试的提醒 | 随时 | 2 min | 与 `A1/tests/test_tokenizer.py:416-435` 一致 |
| `RM:49-53`（§1.2） | 缺口表：BPE 是「没有证明过的能力」 | 启动时 | 3 min | 用来说明为什么要单独练 BPE |

### 1.3 建议阅读顺序（一页版）

1. `H:88-150`（§2.1–2.2，只求能答 unicode1/2）→ 2. `H:168-284`（§2.3–2.4 + 例子，**必须手算**）→ 3. `A1/tests/test_train_bpe.py` 全文 → 4. `H:286-382`（§2.5 性能 + 三条数据规模）→ 5. `H:383-462`（§2.6 编码/解码 + 接口）→ 6. `A1/tests/test_tokenizer.py` 全文 → 7. `H:463-481`（§2.7 实验题）→ 8. `EN:184-332` 查漏 → 9. `C:698-758` 看 toy 与规格差在哪 → 10. `CN:1764-1960` 最后对照（只用来检查自己漏了什么）。

### 1.4 材料覆盖缺口（正文别假装材料里有）

| 问题 | 材料状态 |
|---|---|
| BPE vs WordPiece vs Unigram 对比 | **本机材料全无**。`EN:186` 只说 BPE 出自 Sennrich 2016、被 GPT-2 采用。要写对比必须外部来源，标「待核实」 |
| OWT 32K 的「最长 token 是什么」 | `H:374-382` 只**提问不给答案**；`CN:1956` 给了一个答案但无法核实（见 §6） |
| `test_ascii_string_matches_tiktoken` 为什么注释掉 id 断言 | `A1/tests/test_tokenizer.py:184` 有注释掉的 `# assert ids == reference_ids`，代码未说明原因 → 待核实 |
| `from_files` 是否被官方测试覆盖 | **没有**。测试走 `get_tokenizer(vocab, merges, special_tokens)`（`A1/tests/test_tokenizer.py:74`），从不读文件 → 序列化格式自选，见 §2.8 |
| 生产级 BPE 加速结构 | `EN:329` 只列方向（链表、堆 + 懒失效、Rust/C++）；`H:310-316` 只讲计数缓存 |

---

## 2. 精确规格

### 2.0 规格一览（可直接引用的一句话 + 出处）

| # | 规格 | 出处 |
|---|---|---|
| 1 | 初始词表 = 256 个字节值（不是 0–255 的字符，是 `bytes`） | `H:193-196`、`C:733` |
| 2 | 最终词表大小 = 256 + merges 数 + special token 数 | `H:231-233`、`H:335-336` |
| 3 | 同频 tie-break：取**字典序更大**的 pair | `H:234-238` |
| 4 | 统计 pair 时不跨 pre-token 边界；不数 special token 内部的 pair | `H:233-234`、`H:306-309` |
| 5 | 编码 = 先预分词，再在**每个 pre-token 内部**按 merge 创建顺序重放 | `H:389-395` |
| 6 | special token 编码时必须整体保留成**一个** id | `H:240-249`、`H:415-417` |
| 7 | decode = 拼接各 id 的 bytes 再 UTF-8 解码，失败用 `errors="replace"` | `H:424-432` |
| 8 | `encode_iterable` 内存必须是常数级（测试给 1 MB） | `H:418-423`、`A1/tests/test_tokenizer.py:449-455` |
| 9 | 训练速度门槛：fixture 语料上 < 1.5 s（参考实现 0.38 s） | `A1/tests/test_train_bpe.py:8-24` |
| 10 | 序列化格式自选；读 GPT-2 官方文件须做 byte→unicode remap | `A1/tests/common.py:10-51`、`A1/tests/test_tokenizer.py:44-59` |

### 2.1 `train_bpe(input_path, vocab_size, special_tokens)`

- 签名与三个参数的语义：`H:330-339`；适配器版本 `A1/tests/adapters.py:565-580`。
- `input_path`: 文本文件路径（`H:334`）。
- 返回：`vocab: dict[int, bytes]`（id → token 字节）与 `merges: list[tuple[bytes, bytes]]`（**按创建顺序**）。`H:340-346`、`A1/tests/adapters.py:584-591`。
- 训练三步：词表初始化 → 预分词（预分词 + 计数）→ 迭代合并。`H:192-238`。
- 每轮：数「当前表示下」所有相邻 pair → 取最高频 → 新 token = 两者 bytes 拼接 → 加入词表 → 替换所有（不重叠的）出现。`H:226-233`、`C:735-746`、`EN:202-214`。
- 不跨 pre-token 边界统计是**为效率**也是**为质量**：`H:197-204`（跨边界会产出 `dog!`/`dog.` 这种只差标点的 token）。脚注说明为何不加 end-of-word token：`H:242-245`。

### 2.2 `vocab_size` 语义（本条最易错）

`vocab_size` 是**最终词表上限，含 256 字节、含所有 special token、含所有 merge 产出**（`H:335-336`、`A1/tests/adapters.py:576`）。所以：
`merges 数 = vocab_size − 256 − len(special_tokens)`。

实测校验（本机 fixture，可复现）：

| 场景 | vocab_size | specials | 应有 merges | 实测文件 |
|---|---|---|---|---|
| `test_train_bpe` | 500 | 1 | 243 | `train-bpe-reference-merges.txt` 243 行、`train-bpe-reference-vocab.json` 500 键、id 0–499 |
| GPT-2 发布件 | 50,257 | 1（`<\|endoftext\|>`） | 50,000 | `gpt2_merges.txt` 50,000 行、`gpt2_vocab.json` 50,257 键 |

推论：`vocab_size` 若 ≤ 256 + specials，则一条 merge 都不做（不是 `max(0, …)` 就是直接返回）。`H:231-233` 的措辞是「final vocabulary … plus the number of BPE merge operations」，与上表一致。
`EN:321` 写成「merges ≈ |V| − 256」，那是**没有 special token** 的 toy 语境，别照搬到作业。

### 2.3 tie-break：同频取字典序更大的 pair

- 原文：`H:234-238` ——「deterministically break ties in pair frequency by preferring the lexicographically greater pair」，例子中 `("A","B")、("A","C")、("B","ZZ")、("BA","A")` 同频，合并 `("BA","A")`。
- 比较对象是**两个 token 的 bytes**（`bytes` 之间按无符号字节值比较）：`max([("A","B"),("A","C"),("B","ZZ"),("BA","A")]) == ("BA","A")`（`H:237-238`）。
- **坑 1**：讲稿 toy 版是 `pair = max(counts, key=counts.get)`（`C:740`），这是「最大频次 + dict 插入序」，**不是** handout 的规则。`EN:282` 也直说 toy trace 用的是 "implementation's insertion-order tie behavior"。照抄 → `test_train_bpe` 的 merges 对不上。
- **坑 2**：tie-break 必须发生在 pair 的 **bytes** 上。如果你的 `stats` 键是 int id，直接 `max(id_pair)` 比较的是 id 大小，与 bytes 序不等价（token id 与 bytes 字典序无单调关系）。写法之一：`max(stats, key=lambda k: (stats[k], vocab[k[0]], vocab[k[1]]))`（`CN:1848`，与 handout 等价）。
- **坑 3**：NaN 式「随机/哈希序」不可接受——`EN:205`、`:327` 都强调 determinism 是可复现的前提。

### 2.4 special token：训练期硬边界，编码期整体保留

| 阶段 | 规则 | 出处 |
|---|---|---|
| 训练-初始化 | 加进词表，占固定 id | `H:249-250`、`H:261-262` |
| 训练-预处理 | 先 `re.split` 切掉 special token（`"\|".join` + `re.escape`），只在片段内统计 | `H:300-309` |
| 训练-统计 | 是硬边界：**不参与 merge 统计**、不允许跨其合并 | `H:306-309`、`H:337-339` |
| 训练-并行分块 | chunk 边界落在 special token 开头（用 `find_chunk_boundaries`） | `H:290-296`、`A1/cs336_basics/pretokenization_example.py:5-49` |
| 编码 | 整体保留成一个 id，永不切开 | `H:240-249`、`H:415-417` |
| 编码-缺失时 | 构造 Tokenizer 时若 special 不在词表里，追加到末尾 | `H:435-436`、`A1/tests/test_tokenizer.py:60-65` |
| 编码-重叠 | 多个 special 互相包含时，必须**最长匹配**（`<\|endoftext\|><\|endoftext\|>` 优先整体） | **只有测试**：`A1/tests/test_tokenizer.py:248-262`；handout 未明说 → 正文若引用此规则，出处写测试文件 |

注意 id 自由度：测试只比较 vocab 的**键集合与值集合**，不比较 id→bytes 映射（`A1/tests/test_train_bpe.py:59-62`）；因此 special token 的 id 派发方式（先放/后放）不影响通过，但会影响「能否和 tiktoken 逐 id 相等」——这就是 `A1/tests/test_tokenizer.py:184` 把 id 断言注释掉的原因（待核实）。

### 2.5 `Tokenizer` 五个方法

出处：`H:433-459`（推荐接口）、`A1/tests/adapters.py:542-563`（适配器）。

| 方法 | 规格 | 出处 |
|---|---|---|
| `__init__(vocab, merges, special_tokens=None)` | 直接用给定 vocab/merges/specials；special 不在 vocab 则追加 | `H:441-446` |
| `from_files(cls, vocab_filepath, merges_filepath, special_tokens=None)` | 从「训练输出格式」加载 | `H:447-453` |
| `encode(text) -> list[int]` | 预分词 → 每段内按 merge 顺序合并 → id | `H:388-414`、`H:454` |
| `encode_iterable(iterable) -> Iterator[int]` | 惰性生成器；不能整体读入 | `H:455-458` |
| `decode(ids) -> str` | `b"".join(vocab[i])` 后 UTF-8 解码，`errors="replace"` | `H:424-432`、`EN:310-315` |

`decode` 为何必须容忍非法字节：输入是**任意整数 id**（用户可给任意序列），拼出来的 bytes 不一定是合法 UTF-8（`H:427-429`）；官方要求替换成 U+FFFD（`H:429-432`），也即 `errors="replace"`。`EN:330` 把「decode 策略必须显式选择」列为工程要点。

`decode` 不是 merge 的逆过程：直接查表拼 bytes（`H:425-427`、`EN:310-311`）。所以 `decode(encode(s)) == s` 成立，但 `encode(decode(ids)) == ids` 一般**不**成立（丢掉「怎样合并」的信息）。

### 2.6 `encode_iterable` 与 1 MB 内存要求

- 动机：文件大到不能整体读入，内存须为**常数**而非线性（`H:418-423`）。
- 关键约束：**token 不能跨 chunk 边界**，否则结果与「整段一次编码」不同（`H:422-423`）。
- 测试数值：`A1/tests/test_tokenizer.py:449-455` 把 limit 设为 `1e6`（1 MB），限制写在 `:24`：`setrlimit(RLIMIT_AS, (RSS + max_mem, -1))`。
- **测试实现坑（本机实测 + 语言语义，必读）**：
  1. macOS 上 `setrlimit(..., (x, -1))` 直接抛 `ValueError: Cannot convert negative int` —— 所以这两个测试在 macOS 是 skip 而不是 pass（与 `:416-419` 的 skip 理由「rlimit support … spotty」相符，但真因更具体）。
  2. 即使 Linux 上 limit 生效，`test_encode_iterable_memory_usage` 的限制**窗口也不覆盖迭代期**：`_encode_iterable` 是生成器函数，装饰器 wrapper 在**创建生成器**时就执行完 `finally` 恢复了 rlimit（我用打印顺序验证：`set limit → finally: restore → generator body START`）。结论：**别把这个测试通过当成内存合格的证据**，自己在 Linux 上量 RSS。
  3. `test_encode_memory_usage` 反过来是普通函数，limit 生效，并且官方直接标了 `xfail`（`:435`）：`Tokenizer.encode` 预期超 1 MB 是**正常的**。

### 2.7 macOS 上哪些测试 skip / xfail

| 测试 | 平台行为 | 出处 |
|---|---|---|
| `test_encode_iterable_memory_usage` | 非 Linux → **skip** | `A1/tests/test_tokenizer.py:416-420` |
| `test_encode_memory_usage` | 非 Linux → **skip**；Linux → **xfail**（预期失败） | `:431-436` |

也就是说 macOS 上 25 个 tokenizer 测试里实际跑 23 个。计划文档已把这点写成 gate 风险：`PLAN:239`、`PLAN:190`、`RM:208`。

### 2.8 序列化：为什么是 GPT-2 byte↔unicode remap

事实：
- 官方 fixture 用 **remap 后的 unicode 字符** 存 token：`gpt2_vocab.json` 是 `{remapped_token: id}`，`gpt2_merges.txt` 每行两个 remapped token，空格分隔。解码回 bytes：`A1/tests/test_train_bpe.py:40-49`、`A1/tests/test_tokenizer.py:44-52`。
- remap 规则（`A1/tests/common.py:10-51`）：188 个可打印字节原样；其余 68 个（0–32、127–160、173）映射到 `chr(256+n)`。所以**空格(32) → `Ġ`(U+0120)**、换行(10) → `Ċ`（`A1/tests/common.py:22-32` 专门解释空格这一例）。
- 数字对账：188 + 68 = 256；`chr(288)` 就是 `Ġ`。

为什么不能直接把任意 bytes 塞进 JSON/text：

1. JSON 字符串必须是合法 Unicode 文本；单字节 `0x80`–`0xFF` 往往不是合法 UTF-8 起始字节（如 `0xFF` 在 UTF-8 中永不合法），无法原样写进文件。
2. 即使能转义，`merges.txt` 是**空格分隔**的，token 里若含空格/换行会把行结构搞坏；remap 后每个 token 恰好一个可打印字符。
3. 控制字符（0x00–0x1F）会把「一行一对」的文本格式和肉眼检查毁掉（`H:239-250` 强调 special token 的可见性）。
4. 测试解析 `merges.txt` 时**只接受恰好两个字段的行**（`A1/tests/test_tokenizer.py:49-52`），这同时也容忍了发布件可能存在的 `#version` 头行（本仓库 fixture 无头行，实测首行即 `Ġ t`；官方下载件是否带头行 → 待核实）。

补充：序列化格式其实是**自选**的（`from_files` 不被官方测试调用；`H:447-449` 只说「same format that your BPE training code output」）。中文导读用 `v.hex()` 存 vocab、`f"{list(pair)}"` 存 merges（`CN:1893-1897`）——合法但**与 GPT-2 文件不互通**，别指望直接喂 tiktoken 或复用 `gpt2_vocab.json`。

### 2.9 速度门槛

- `test_train_bpe_speed`：`corpus.en`、`vocab_size=500`、1 个 special，**< 1.5 s**（`A1/tests/test_train_bpe.py:8-24`）。
- docstring 给的基准："takes 0.38 seconds with the reference implementation on my laptop… the toy implementation takes around 3 seconds"（`A1/tests/test_train_bpe.py:11-14`）→ 3 s ≈ 2 倍上限，0.38 s ≈ 4 倍余量。
- 语料实测量级：`corpus.en` 133,027 B / 1,015 行（本机 `stat`）。所以这是「小语料 + 500 词表」的门槛，不能外推到 TinyStories 10K。
- 官方对真实规模的预期：TinyStories 10K「≤ 30 min、≤ 30 GB RAM」，提示可「under 2 minutes」（`H:357-368`）；OWT 32K「≤ 12 h、≤ 100 GB RAM」（`H:374-378`）。

### 2.10 预分词正则（编码与训练共用）

```text
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
```

出处：`H:215-217`（说明来自 tiktoken PR #234），示例 `H:218-223`；必须用 `regex` 包（`\p{L}`），并建议 `re.finditer` 而不是 `findall` 以免把全部 pre-token 存下来（`H:224-225`）。
实测 `re.findall(PAT, "some text that i'll pre-tokenize")` = `['some', ' text', ' that', ' i', "'ll", ' pre', '-', 'tokenize']`（`H:223`，本机复现一致）。
两个易忽略的行为（本机实测，直接决定压测是否挂，见 §3.3）：

| 输入 | 预分词结果 | 原因 |
|---|---|---|
| `"abc\n\n def"` | `['abc', '\n\n', ' def']` | `\s+(?!\S)` 贪婪吃下 `\n\n`，后面是空格 → 负向前瞻成功 |
| `"abc\n\ndef"` | `['abc', '\n', '\n', 'def']` | 后面是 `d` 非空白 → 前瞻失败，回退成 `\s+` 各自吃一个 |
| `"<\|endoftext\|>\n\ntesting!"`（先切 special） | `['\n', '\n', 'testing', '!']` | 同上（后面是 `t`） |

### 2.11 规格冲突清单（照谁写？）

| # | 冲突 | 谁对 | 依据 |
|---|---|---|---|
| 1 | `A1/tests/adapters.py:580` 说 special token 在输入里「are treated as any other string」 | **handout 与测试** | `H:337-339`（不参与统计）+ `A1/tests/test_train_bpe.py:78-80`（任何 vocab 值不得含 `b"<\|"`） |
| 2 | `C:740` 的 `max(counts, key=counts.get)`（插入序） | **handout 的字典序更大** | `H:234-238`；`EN:282` 自认是 toy 行为 |
| 3 | `EN:206` / `C:743` 的「新 id = 256 + k」 | 只在无 special token 时成立 | `H:335-336`：id 空间被 specials 占掉；id 具体派发本身自由（§2.4） |
| 4 | `CN:1835`「下一轮是 we」 | **e st（9 次）** | `H:277` + 本文 §4 复算 |
| 5 | `CN:1934`「`<\|end_of_text\|>` 15 字节」 | TinyStories 用的是 `<\|endoftext\|>`（13 B） | `H:359`、`H:260`；实测 `len()` 13 / 15 |
| 6 | `CN:1936`「最耗时的是 max()」 | 预分词是主要瓶颈（朴素实现则是每轮全量重扫） | `H:291`、`H:311-313` |

---

## 3. 测试地图

### 3.1 `A1/tests/test_train_bpe.py`（3 个）

| 测试 | 断言什么 | 失败通常意味着 | 一句定位 |
|---|---|---|---|
| `test_train_bpe_speed` `:8-24` | `corpus.en` + `vocab_size=500` 训练耗时 < 1.5 s | 每轮全量重扫 pair；Python 内层循环过多；没做计数缓存/增量更新 | `cProfile` 看 `count_pairs`/`max` 占比；先正确后优化（`H:310-319`） |
| `test_train_bpe` `:27-62` | (a) merges **完全等于** 243 条参考 merge（remap 解码后逐项比）；(b) vocab 的 keys 集合与 values 集合相等 | tie-break 错（最常见）；预分词正则不对/没用 `regex`；special token 参与了统计；merge 顺序不是创建序 | 打印「第几条 merge 开始不同」+ 该轮 pair 计数前 10 名；先比前 10 条 merge |
| `test_train_bpe_special_tokens` `:65-88` | 5M TinyStories、`vocab_size=1000`：(a) 除 special 本身外，任何 vocab 值都不含 `b"<\|"`；(b) 整个结果与 `_snapshots/test_train_bpe_special_tokens.pkl` 快照一致 | 没先切 special 就统计 pair（`<\|` 被并进 token）；把 special 内部当普通字节；边界切错导致跨文档合并 | 先只跑 `:78-80` 的 `b"<\|"` 断言；通过了再查快照 |

补充：快照是 **pickle**（`A1/tests/_snapshots/test_train_bpe_special_tokens.pkl`，`A1/tests/conftest.py:95-133`），首次不通过时**不要**用 update 覆盖——那等于自己给自己判卷。

### 3.2 `A1/tests/test_tokenizer.py`（25 个，逐个）

固定件：`VOCAB_PATH`/`MERGES_PATH` = `fixtures/gpt2_vocab.json` / `gpt2_merges.txt`（`:15-16`），加载方式见 `:39-74`。

| # | 测试 | 断言 | 失败常见的实现错 |
|---|---|---|---|
| 1 | `test_roundtrip_empty` `:77` | `encode("")==[]` 且 decode 回 `""` | 空串走到异常路径；预分词对空串报错 |
| 2 | `test_empty_matches_tiktoken` `:88` | 空串 id 与 tiktoken 一致；`[decode([x]) for x in ids] == []` | 同上 |
| 3 | `test_roundtrip_single_character` `:107` | `"s"` round-trip | 单字节 token 被当成非法 |
| 4 | `test_single_character_matches_tiktoken` `:118` | `"s"` id 一致；逐 id decode == `["s"]` | 编码时把单字节扩展成特殊路径 |
| 5 | `test_roundtrip_single_unicode_character` `:137` | `"🙃"` round-trip | 按字符而非字节处理；decode 时逐字节解 |
| 6 | `test_single_unicode_character_matches_tiktoken` `:148` | `"🙃"` id 与 tiktoken 一致 | 同上 |
| 7 | `test_roundtrip_ascii_string` `:164` | `"Hello, how are you?"` round-trip | 标点被并进词 |
| 8 | `test_ascii_string_matches_tiktoken` `:175` | 逐 id decode == `["Hello", ",", " how", " are", " you", "?"]`；**id 断言被注释**（`:184`） | 预分词把 `,` 和 ` how` 切错（` ?\p{L}+` 要保留前导空格） |
| 9 | `test_roundtrip_unicode_string` `:193` | `"Héllò hôw are ü? 🙃"` round-trip | 非 ASCII 按字符处理 |
| 10 | `test_unicode_string_matches_tiktoken` `:204` | 带 `<\|endoftext\|>` 的 tokenizer，id 与 tiktoken 一致 | 同上 + special 处理 |
| 11 | `test_roundtrip_unicode_string_with_special_tokens` `:219` | 3 个 `<\|endoftext\|>` 原样保留、整体 decode 还原 | special 被切开（幽灵 token） |
| 12 | `test_unicode_string_with_special_tokens_matches_tiktoken` `:233` | 同上且 id 与 tiktoken（`allowed_special`）一致 | 切分位置不一致（把相邻空格吞进 special） |
| 13 | `test_overlapping_special_tokens` `:248` | specials = `["<\|endoftext\|>", "<\|endoftext\|><\|endoftext\|>"]`；双写那段必须是 **1 个** 长 token、且单写只出现 1 次；round-trip | 按列表顺序先匹配短的；用 `str.replace` 而非最长匹配/前缀扫描 |
| 14 | `test_address_roundtrip` `:265` | `address.txt`（1468 B，葛底斯堡演说）round-trip | 长文本边界/换行 |
| 15 | `test_address_matches_tiktoken` `:277` | 同文件 id 与 tiktoken 完全一致 | 预分词细节（数字、标点、大小写） |
| 16 | `test_german_roundtrip` `:294` | `german.txt`（594 B，含 `„…“` 弯引号）round-trip | 多字节标点被拆错 |
| 17 | `test_german_matches_tiktoken` `:306` | 同上 id 一致 | 同上 |
| 18 | `test_tinystories_sample_roundtrip` `:323` | 3,786 字符样本 round-trip（**不传 specials**，`<\|endoftext\|>` 当普通文本） | 把 special 字符串硬编码进 encode |
| 19 | `test_tinystories_matches_tiktoken` `:335` | 传 specials + `allowed_special`，id 完全一致 | special 与邻接空白/换行的切分 |
| 20 | `test_encode_special_token_trailing_newlines` `:351` | 文件内容 `<\|endoftext\|>\n\n`，id 与 tiktoken 一致且 round-trip | 尾部空白 run 被当 special 的一部分；EOF 处的 `\s+(?!\S)` 分支处理错 |
| 21 | `test_encode_special_token_double_newline_non_whitespace` `:367` | 文件内容 `<\|endoftext\|>\n\ntesting!`，id 一致 | 同一处：`\n\n` 应被切成两个 `\n` pretoken（§2.10 表） |
| 22 | `test_encode_iterable_tinystories_sample_roundtrip` `:383` | 用 `encode_iterable(file_handle)` 得到全部 id，decode 后等于整文件 | 流式分块把 token 切断 |
| 23 | `test_encode_iterable_tinystories_matches_tiktoken` `:397` | **流式结果必须与整段一次编码完全一致** | 见 §3.3：空白 run 跨 chunk |
| 24 | `test_encode_iterable_memory_usage` `:416-428` | Linux only：1 MB 限制下跑 5.24 MB 文件 | 见 §2.6：测试窗口本身不严谨 |
| 25 | `test_encode_memory_usage` `:431-446` | Linux only，**xfail**：`encode` 允许超 1 MB | 若它意外 pass 也不代表更好 |

### 3.3 补坑：`test_encode_iterable_*_matches_tiktoken` 的「空白 run 跨 chunk」

**问题**：把文件按行 `encode` 再拼接，遇到「空白 run 后面还是空白」就会与整段编码不一致。因为 `\s+(?!\S)` 的匹配长度取决于 run 后面第一个字符是否空白/EOF，而 chunk 末尾看不到下一个字符。

**实测（用 `A1/tests/fixtures/gpt2_vocab.json` + `gpt2_merges.txt` 自建编码器，不依赖 tiktoken）**：

| 输入 | 整段 encode | 逐行 encode | 结论 |
|---|---|---|---|
| `"abc\n\n def"` | `[39305, 628, 825]` | `[39305, 198, 198, 825]` | **不同**（`628` = `b"\n\n"`，`198` = `b"\n"`） |
| `"def f():\n\n    return 1"` | `[…, 628, 220,220,220, 1441, 352]` | `[…, 198, 198, 220,220,220, 1441, 352]` | 不同（代码缩进场景） |
| `"p\n\n"` | `[79, 628]` | `[79, 198, 198]` | 不同（EOF 处） |
| `"abc\n\ndef"` | `[39305, 198, 198, 4299]` | 同左 | **相同** |

最后一行是关键：**本仓库自带的 `tinystories_sample.txt` 里唯一的 `\n\n` 恰好后面跟的是 `Once`（非空白）**，所以在这个自带样本上，「逐行 encode」能通过 `test_encode_iterable_*` 却仍然是错的（5M 样本里有 1,771 处 `\n\n`，是否都落在非空白之前未逐处检查 → 待核实）。别用「测试绿了」证明流式正确。

**正确做法**：缓冲（buffer），只在「确定一个 pre-token 已经结束」时 flush；核心是**不要把空白 run 切开**——flush 到「空白 run 起点」之前，把这段空白连同下一个 chunk 一起处理。可证明的判据：只有当缓冲末尾不再是 `\s`，或已到输入末尾（EOF）时，才能安全 flush 掉尾部空白 run 之后的内容。

**怎么验证自己的实现**：构造 `"abc\n\n def"`、`"def f():\n\n    return 1"`、`"p\n\n"` 三个用例，比较 `encode_iterable([s1, s2])`（切在任意位置）与 `encode(s1+s2)` 的 id 序列；再对 `tinystories_sample_5M.txt` 做一次整体 encode 与流式 encode 的全序列比对。

### 3.4 断言层次（先炸哪个）

1. `test_train_bpe` 的 merges 相等 → 训练侧一切错误的**总闸**（tie-break/预分词/special 都会在这里现形）。
2. `test_train_bpe_special_tokens` 的 `b"<\|"` 断言 → 专查 special 泄漏。
3. `test_*_matches_tiktoken`（除 `:184` 那条被注释掉的 id 断言外，其余都逐 id 比对）→ 专查预分词和 special 切分。
4. `test_roundtrip_*` → 专查 decode 与字节/字符边界。
5. `test_encode_iterable_*` → 专查流式缓冲。

---

## 4. 手算示例核对（`bpe_example`，`H:254-284`）

### 4.1 复算方法

语料 = `low ×5 / lower ×2 / widest ×3 / newest ×6`（`H:255-260`）；预分词按空白（`H:263-267`，仅本例的简化）；词表 = 256 字节 + `<|endoftext|>`（`H:261-262`）；每轮按「频次降序，同频取 bytes 字典序更大的 pair」选一条，替换所有不重叠出现。脚本见附录 A（stdin 运行，未落盘）。

### 4.2 逐轮表（与 handout 完全一致）

| 轮 | pair 计数（降序；只列出现过的） | 最大频次并列 | 选中 pair | 合并后 pre-tokens（频次） | 合并后 token 总数 |
|---|---|---|---|---|---|
| 1 | es 9, st 9, we 8, lo 7, ow 7, ew 6, ne 6, de 3, id 3, wi 3, er 2 | es, st | **s t** | `l o w`(5), `l o w e r`(2), `w i d e st`(3), `n e w e st`(6) | 70 |
| 2 | e st 9, we 8, lo 7, ow 7, ew 6, ne 6, de 3, id 3, wi 3, er 2 | e st | **e st** | `l o w`(5), `l o w e r`(2), `w i d est`(3), `n e w est`(6) | 61 |
| 3 | lo 7, ow 7, ew 6, ne 6, w est 6, de 3, id 3, wi 3, er 2, we 2 | lo, ow | **o w** | `l ow`(5), `l ow e r`(2), `w i d est`(3), `n e w est`(6) | 54 |
| 4 | l ow 7, ew 6, ne 6, w est 6, de 3, id 3, wi 3, er 2, ow e 2 | l ow | **l ow** | `low`(5), `low e r`(2), `w i d est`(3), `n e w est`(6) | 47 |
| 5 | ew 6, ne 6, w est 6, de 3, id 3, wi 3, er 2, low e 2 | ew, ne, w est | **w est** | `low`(5), `low e r`(2), `w i d est`(3), `n e west`(6) | 41 |
| 6 | e west 6, ne 6, d est 3, id 3, wi 3, er 2, low e 2 | e west, ne | **n e** | `low`(5), `low e r`(2), `w i d est`(3), `ne west`(6) | 35 |
| 7 | ne west 6, d est 3, id 3, wi 3, er 2, low e 2 | ne west | **ne west** | `low`(5), `low e r`(2), `w i d est`(3), `newest`(6) | 29 |
| 8 | d est 3, id 3, wi 3, er 2, low e 2 | d est, id, wi | **w i** | `low`(5), `low e r`(2), `wi d est`(3), `newest`(6) | 26 |
| 9 | d est 3, wi d 3, er 2, low e 2 | d est, wi d | **wi d** | `low`(5), `low e r`(2), `wid est`(3), `newest`(6) | 23 |
| 10 | wid est 3, er 2, low e 2 | wid est | **wid est** | `low`(5), `low e r`(2), `widest`(3), `newest`(6) | 20 |
| 11 | er 2, low e 2 | er, low e | **low e** | `low`(5), `lowe r`(2), `widest`(3), `newest`(6) | 18 |
| 12 | lowe r 2 | — | **lowe r** | `low`(5), `lower`(2), `widest`(3), `newest`(6) | 16 |

得到的 12 条 merge 序列：
`s t, e st, o w, l ow, w est, n e, ne west, w i, wi d, wid est, low e, lowe r` —— 与 `H:279-280` 逐项相同（本机复算）。

注：最后一列是「合并**后**整份语料的 token 数」＝ Σ(pre-token 长度 × 频次)：初始 79（3·5 + 5·2 + 6·3 + 6·6）→ 第 1 轮后 70 → … → 第 12 轮后 16，正好说明 BPE 每步都在压缩序列（`H:184`、`EN:216`）。

### 4.3 第一轮为什么是 `('s','t')` 而不是 `('e','s')`

- 两者频次都是 9：`es` = widest(3, `d-e-s`) + newest(6, `e-s`)，`st` = widest(3) + newest(6)（`H:273`，本机复算一致）。
- tie-break 比较的是 **bytes 元组**：`(b's', b't')` vs `(b'e', b's')` → 首元素 `b's'=115 > b'e'=101` → 取 `('s','t')`（`H:234-238`、`H:273-274`）。
- 注意：不是「右元素更大」也不是「后出现者优先」；同样不能拿 id 比（`b's'`/`b't'` 的 id 恰好是 115/116，但这只在字节 token 区间成立，合并后就不成立了）。
- 同理可解释后面的并列：轮 5 三个并列取 `w est`（`b'w'` 最大）；轮 6 取 `n e`（`b'n' > b'e'`）；轮 9 取 `wi d`（`b'wi' > b'd'`，因为 `b'w' > b'd'` 是首个字节比较）；轮 11 取 `low e`（`b'low' > b'er'`）。

### 4.4 6 merges 与 12 merges 的词表、编码结果

| merges | 词表大小 | 新增 token（按创建序） | `newest` 编码 | `widest` | `lower` | `lowest`（语料外） |
|---|---|---|---|---|---|---|
| 6 | 263 = 256+1+6 | st, est, ow, low, west, ne | `[ne, west]`（`H:284`） | `[w, i, d, est]` | `[low, e, r]` | `[low, est]` |
| 12 | 269 = 256+1+12 | st, est, ow, low, west, ne, newest, wi, wid, widest, lowe, lower | `[newest]` | `[widest]` | `[lower]` | `[low, est]` |

（12 个新增 token 的完整列表见附录 A；6 merges 的 6 个 token 与 `H:281-283` 列出的完全一致；`newest → [ne, west]` 与 `H:284` 一致。）
`lowest` 说明一个反直觉点：**语料里没出现过的词也能被已有 merge 压短**（这里的 `low` + `est`），因为字母组合本身在语料里练过。

### 4.5 与 handout 的逐字对比

| handout 说法 | 本机复算 | 结论 |
|---|---|---|
| `{low:5, lower:2, widest:3, newest:6}`（`H:267`） | 一致 | 核对无误 |
| 第一轮 11 个 pair 计数（`H:273`） | 一致（11 项全对） | 核对无误 |
| 取 `('s','t')` 而非 `('e','s')`（`H:273-274`） | 一致 | 核对无误 |
| 第一轮合并后 4 个 pre-token（`H:275-276`） | 一致 | 核对无误 |
| 第二轮 `(e, st)` = 9（`H:277`） | 一致 | 核对无误 |
| 12 条 merge 序列（`H:279-280`） | 逐项一致 | 核对无误 |
| 6 merges 的词表与 `newest → [ne, west]`（`H:281-284`） | 一致 | 核对无误 |

### 4.6 讲稿 toy 实现在同一例子上的行为（对照用）

`C:729-750` 的 `train_bpe` 与作业规格有 4 处不同：① 不做预分词，直接在整串字节上数 pair；② 不处理 special token；③ `vocab` 只有 256 字节起步（`C:733`），第一轮新 id = 256（`C:743`）；④ tie-break 用 `max(counts, key=counts.get)`（`C:740`）→ 插入序。
`EN:244-292` 用 `the cat in the hat` 展示了 ④ 的后果：第三轮并列时它选了 `(257, 32)`（即 `"the "`，`EN:282`），而作业规格下同频必须取 bytes 更大的 pair。**教学价值**：正好用来讲「为什么 merge 顺序是训练出来的参数、为什么 tie-break 必须写进 spec」。

---

## 5. 自测题 20 道（答案要点均可回溯）

> 用法：先自己答，再对照「要点 + 出处」。带 ⚠ 的是本机材料**没有**覆盖的，答案里的对比只作背景、必须外部核实。

**1. 为什么用 UTF-8 字节而不是 Unicode 码点训练？**
要点：码点词表 ~15 万且稀疏（`H:119-121`）；字节词表固定 256、无 OOV（`H:145-150`）；UTF-8 是互联网主导编码（`H:122-124`）。⚠ 与 UTF-16/32 的对比细节（`H:151-155` 是题目不是答案）。

**2. 为什么是 256 而不是 65536？** 要点：一个字节 0–255；`bytes` 迭代即得整数序列（`H:125-137`、`H:147-150`）。

**3. 「逐字节 decode」函数错在哪？** 要点：一个 Unicode 字符可能占多字节，单字节 `bytes([b]).decode("utf-8")` 对 `0x80–0xFF` 会抛错或被拆坏；例子可用 `"你".encode("utf-8")` = `b'\xe4\xbd\xa0'`（`H:145-166`、`C:514-524`）。

**4. 举一个不能解码成任何字符的两字节序列。** 要点：UTF-8 里 `0xC0`/`0xC1` 是非法起始字节（overlong），如 `b'\xc0\xaf'`（`H:166-167` 只给题目；具体例子属背景知识 ⚠）。

**5. `chr(0)` 的 `repr` 与 `print` 差别？** 要点：`repr` 显示 `'\x00'`，`print` 什么都不显示；文本里出现会在拼接时"消失"（`H:104-117`）。

**6. `vocab_size=10000` 时做多少条 merge（1 个 special）？** 要点：10000 − 256 − 1 = 9,743；样例 `H:358-359`。GPT-2 发布件对账：50,257 = 256 + 50,000 + 1（§2.2 实测）。

**7. tie-break 规则，及它为什么影响最终词表？** 要点：同频取 bytes 字典序更大（`H:234-238`）；merge 顺序决定后续所有轮的 pair 统计，错一条后面全歪，所以测试直接断言 merges 序列（`A1/tests/test_train_bpe.py:50`）。

**8. special token 在训练期与编码期分别怎么处理？** 要点：训练期=硬边界、先切掉、不参与统计（`H:300-309`）；编码期=整体一个 id、不切开（`H:415-417`）；不在词表则追加（`H:435-436`）。

**9. 两个 special 互相包含时怎么办？** 要点：最长匹配，`<|endoftext|><|endoftext|>` 要整体成一个 token（`A1/tests/test_tokenizer.py:248-262`；handout 未明说）。

**10. 压缩比怎么定义？为什么不是越大越好？** 要点：r = UTF-8 bytes / tokens（`EN:143-153`、`C:567-571`）；越大序列越短（attention O(T²) 受益，`EN:155`）；但大词表→稀疏、embedding/LM head 参数涨、统计共享变弱（`EN:157`）。

**11. tokenize 结果为什么存 uint16？** 要点：词表 10K/32K < 65,536，id 放得下；存成 `uint16` 数组时每 token 2 字节（题目 `H:479-480` 只问「为什么合适」，理由需自答；词表规模依据见 §2.2）。

**12. BPE 与 WordPiece、Unigram 的区别？** ⚠ 本机材料**全无**覆盖（`EN:186` 只交代 BPE 出处）。最小背景：WordPiece 用似然而非频次选 merge、Unigram 做概率裁剪式词表。**写进正文前必须找外部来源核实**。

**13. 为什么 encode 不能「把 merges 当脚本对全文重放」？** 要点：① 必须先在 pre-token 边界分组，否则会造出跨词 token（`H:389-395`、`H:233-234`）；② special token 不能进入 merge 流程（`H:415-417`）；③ 逐条重放是 O(#merges × len)，50k merges 时不可行（`C:723` 明确说「Only loop over merges that matter」、`EN:321`）。

**14. `encode_iterable` 为什么能省内存，又为什么容易错？** 要点：惰性生成、常数内存（`H:418-421`）；但 chunk 边界不能切断 token（`H:422-423`），空白 run 是最常见断裂点（§3.3 实测）。

**15. decode 为什么要 `errors="replace"`？** 要点：输入的 id 序列任意，拼出的 bytes 可能非法 UTF-8；替换为 U+FFFD 而不是抛错（`H:424-432`、`EN:330`）。

**16. BPE 训练的复杂度瓶颈在哪？** 要点：预分词是主要瓶颈（`H:291`）；朴素合并每轮全量重扫（`H:311-313`）；优化=索引 pair 计数 + 增量更新（`H:313-316`）；合并部分在 Python 里不可并行（`H:316`）。

**17. 为什么 merges 必须按创建顺序存、按创建顺序用？** 要点：新 token 依赖旧 token 存在；顺序是训练参数的一部分（`H:344-346`、`EN:294`）；编码要「same order of creation」（`H:394-395`）。

**18. decode 为什么不「反向执行 merge」？** 要点：直接查 vocab 拼 bytes 即可（`H:425-427`、`EN:310-311`）；反向执行既无必要也会歧义（同一 bytes 可有多种 merge 路径）。

**19. 不做预分词直接在全文数字节对，会有哪两类问题？** 要点：性能（每次合并全文扫）与质量（`dog!`/`dog.` 拿到不同 id、语义相近却分裂）（`H:197-204`）。

**20. 我能不能要求自己的 special token id 等于 tiktoken 的 50256？** 要点：不必。测试只比 vocab 键/值集合，不比映射（`A1/tests/test_train_bpe.py:59-62`）；但一旦 id 派发不同，就**不能**再对含 special 的文本断言 id 与 tiktoken 相等（`A1/tests/test_tokenizer.py:184` 把该断言注释掉了，原因待核实）。

---

## 6. 对中文导读 §补充 1 的核对与勘误（`CN:1764-1960`）

### 6.1 逐条核对

| CN 行号 | 原文说法 | 结论 | 依据 / 正确说法 |
|---|---|---|---|
| `:1772` | `A`=65、`你`=20320（0x4F60） | 核对无误 | 0x4F60 = 20320；`H:96-99` 用 `牛`=29275 举例 |
| `:1774` | 有效码点约 15 万，词表过大且稀疏 | 核对无误 | `H:94-95`（Unicode 17.0：159,801 字符 / 172 脚本） |
| `:1776` | UTF-8 覆盖 98% 以上网页 | 核对无误 | `H:122-124` |
| `:1780` | 码点（21 位）→ 字节序列，词表 15 万 → 256 | 核对无误 | `H:145-150`（原文 "21-bit integers"） |
| `:1782` | 字节级无未登录词 | 核对无误 | `H:148-150` |
| `:1802` | 初始词表 = 256 字节 + 特殊 token | 核对无误 | `H:261-262` |
| `:1804` | 预分词两条理由（每次合并全文扫 / `dog!` vs `dog.`） | 核对无误 | `H:198-204` |
| `:1807` | 正则 `'(?:[sdmt]\|ll\|ve\|re)\| ?\p{L}+…` | 核对无误 | 与 `H:217` 逐字一致 |
| `:1809` | `findall` 输出 `['some',' text',' that',' i',"'ll",' pre','-','tokenize']` | 核对无误 | `H:222-223`；本机复现一致 |
| `:1816` | 平局取字典序更大的（`max()` 对元组恰好如此） | 核对无误（有条件） | `H:234-238`；条件=比较的是 **bytes 元组**，对 int id 的 dict 不成立（`C:740` 是反例） |
| `:1820` | 只更新受影响的 pair 计数（反向索引） | 核对无误 | `H:310-316` |
| `:1825` | pair 计数 es9 st9 we8 lo7 ow7 ne6 ew6 wi3 id3 de3 er2 | 核对无误 | `H:273`；本机 11 项全对 |
| `:1827-1828` | 第一轮 es/st 同频 9 → 取 st；新 token `b'st'` 进词表 **ID 257** | 核对无误（前提是 1 个 special） | `H:273-274`；256 字节 + 1 special = 257 项，故首个 merge=257；id 派发本身自由（§2.4） |
| `:1830-1834` | 合并后 `widest → w i d e (st)`、`newest → n e w e (st)` | 核对无误 | `H:275-276` |
| `:1835` | 「下一轮又会选出新的最高频 pair（**这里是 we**）」 | **勘误** | 第二轮是 `(e, st)`=9（`H:277`）；`we` 那一轮只有 8，不是最高频。正确：`['s t','e st','o w',…]` 见 §4.2 轮 2-3 |
| `:1838` | special token 训练时硬边界、不参与统计、整体保留 | 核对无误 | `H:300-309`、`H:415-417` |
| `:1842` | 返回 `vocab: dict[int,bytes]` 与按创建序的 `merges` | 核对无误 | `H:340-346` |
| `:1848` | `max(stats, key=lambda k: (stats[k], vocab[k[0]], vocab[k[1]]))` | 核对无误（等价写法） | 与 `H:234-238` 等价；注意 `vocab` 必须是 id→bytes |
| `:1850` | 合并后同步更新 pair 计数与反向索引 | 核对无误 | `H:310-316` |
| `:1893-1897` | 序列化：`v.hex()` 存 vocab、`f"{list(pair)}"` 存 merges | **格式不兼容** | 自选格式合法（`from_files` 不被官方测试调用），但无法与 `gpt2_vocab.json`/`gpt2_merges.txt` 互通，也不能直接喂 tiktoken；要互通须用 `A1/tests/common.py:10-51` 的 remap。另 `json.dump` 会把 int 键变成字符串 |
| `:1902` | 并行预分词只在 special token 边界切块，属性能优化 | 核对无误 | `H:290-296`、`A1/cs336_basics/pretokenization_example.py:5-49` |
| `:1916-1919` | TinyStories 10,000 / OWT 32,000，`<\|endoftext\|>` | 核对无误 | `H:358-359`、`H:375-376` |
| `:1934` | 最长 token 要排除 special token，因为「`<\|end_of_text\|>` 本身有 15 字节」 | **勘误（token 名）** | 作业用的是 `<\|endoftext\|>`（`H:359`），实测 **13** 字节；`<\|end_of_text\|>` 是另一种 15 字节的 token。另「只看 ID ≥ 257」隐含 1 个 special，通用式是 `≥ 256 + len(specials)` |
| `:1936` | 最耗时的是「选最高频 pair」那句 `max()` | **表述偏差** | `H:291` 明确说主要瓶颈是**预分词**；`H:311-313` 说朴素实现是「每轮全量重扫」。谁最慢取决于实现 → 待核实（需自己 profile） |
| `:1950` | 词表越大越容易把低频「脏」片段并成独立 token，正常现象 | 合理但**材料未给结论** | 待核实（`H:374-382` 只提问）。可作为假设写进实验，别当事实 |
| `:1954` | TinyStories(10k) 最长 token = `b' accomplishment'`（15 字节） | **待核实** | 材料未给答案；字节数自洽（1 空格 + 14 字母 = 15）。需自己在 10K 词表上验 |
| `:1956` | OWT(32k) 最长 token = 一段 64 字节的乱码 `ÃÂÃÂÃÂ…` | **待核实** | 同上；本机无 OWT 数据，无法复现。正文若引用必须标「社区经验，未复现」 |
| `:1960-1961` | `#### 1.4 分词器类：编码与解码`（只有标题） | **缺口** | 该节无正文，紧接着就是 `补充 2`（`:1962`）。编码/解码规格只能看 `H:383-462` |
| `:1972` | 「部分内容可能由 AI 生成」 | 事实 | 与上面 6 处问题一致，不可作为唯一来源 |

### 6.2 必须改的（如果要在教程里引用导读）

1. `:1835` 第二轮 merge 说成 `we` → 应为 `(e, st)`。
2. `:1934` 的 special token 名该是 `<|endoftext|>`（实测 13 字节），原文写成的是另一种 15 字节的 `<|end_of_text|>`。
3. `:1936` 把「最慢的步骤」直接说成 `max()` → 预分词才是 handout 点名的瓶颈。
4. `:1954`、`:1956` 的最长 token 结论无出处 → 标待核实。
5. `:1893-1897` 的序列化格式与官方 fixture 不兼容 → 想复用 GPT-2 词表必须换 remap。
6. `1.4` 整节缺失 → 编码/解码只能回到 handout。

### 6.3 它仍然有用的地方

按题号组织、把「作答时注意」写成清单，适合**做完题后自查**（`:1844-1850`、`:1932-1936`）；`train_bpe` 骨架的注释顺序（`:1878-1887`）对拆解步骤有帮助。**但不能作为 spec 或事实来源。**

---

## 7. 术语对照表

| 英文 | 中文 | 一句话定义（含出处） |
|---|---|---|
| token | 词元 / 标记 | 模型实际处理的最小单位，一个整数 id（`EN:121-137`） |
| token ID | 词元编号 | 词表里 token 的整数下标（`H:342-343`） |
| vocabulary / vocab | 词表 | token 字节串 ↔ id 的一一映射（`H:193-196`） |
| byte-level BPE | 字节级 BPE | 初始词表为 256 个字节值的 BPE（`H:186-196`） |
| pre-tokenization | 预分词 | 用正则把语料切成片段，只在片段内统计 pair（`H:197-225`） |
| pre-token | 预分词片段 | 预分词输出的一段字节串 + 频次（`H:205-210`） |
| merge | 合并（规则） | `(bytes, bytes) → new token`，一条一条按创建顺序记录（`H:226-233`、`H:344-346`） |
| merge rank | 合并次序 | merge 在列表中的下标；决定编码时先试哪条（`H:394-395`、`EN:327`） |
| tie-break | 并列打破规则 | 同频时取 bytes 字典序更大的 pair（`H:234-238`） |
| special token | 特殊 token | 永不切开的控制串（如 `<\|endoftext\|>`），占固定 id（`H:239-250`） |
| hard boundary | 硬边界 | 训练期不允许跨 special token 合并（`H:300-309`） |
| compression ratio | 压缩比 | UTF-8 字节数 / token 数（`EN:143-153`、`C:567-571`） |
| fertility | 生育率（切分率） | 平均每个词被切成几个 token；材料只出现术语（`EN:343`），定义属通行用法 ⚠ |
| bits-per-byte (BPB) | 每字节比特数 | `−log₂ p(x) / len_utf8(x)`，跨 tokenizer 比较 loss 的归一化指标（`EN:lecture-03.md:286-291`） |
| OOV / UNK | 未登录词 / 未知词 | 词表外 token 的占位符；字节级 BPE 不需要（`H:148-150`、`EN:192`） |
| GPT-2 byte↔unicode remap | GPT-2 字节重映射 | 把 256 个字节映射成可打印字符（空格→`Ġ`）以便文本序列化（`A1/tests/common.py:10-51`） |
| `encode_iterable` | 流式编码 | 惰性产出 id，常数内存（`H:455-458`） |
| tied embeddings | 权重共享 | Embedding 与 LM Head 共用矩阵；与 BPE 的交集仅是 `d_model × vocab_size` 参数量（`CN:1964`、`EN:lecture-03.md:284`） |
| adaptive computation | 自适应计算 | 常见片段 1 个 token、罕见内容多个 token，算力随内容变化（`EN:192`） |

---

## 附录 A：手算复算脚本与输出（截断）

脚本通过 stdin 运行（未写入任何文件），核心逻辑：

```python
from collections import Counter
corpus = "low low low low low\nlower lower widest widest widest\nnewest newest newest newest newest newest"
seqs = Counter(tuple(bytes([b]) for b in w.encode()) for w in corpus.split())   # 本例预分词 = 按空白切
vocab = {i: bytes([i]) for i in range(256)}          # 256 字节
vocab[256] = b"<|endoftext|>"; next_id = 257          # 1 个 special
for rnd in range(1, 13):
    pc = Counter()
    for seq, f in seqs.items():
        for a, b in zip(seq, seq[1:]): pc[(a, b)] += f
    (a, b), f = max(pc.items(), key=lambda kv: (kv[1], kv[0]))   # 频次, 再比 bytes
    vocab[next_id] = a + b; mid, next_id = next_id, next_id + 1
    # ... 把 (a,b) 替换成 a+b，重建 seqs ...
```

关键输出（节选）：

```text
--- round 1 ---
counts desc: {'e s': 9, 's t': 9, 'w e': 8, 'l o': 7, 'o w': 7, 'e w': 6, 'n e': 6, 'd e': 3, 'i d': 3, 'w i': 3, 'e r': 2}
tied at max: ['e s', 's t']
selected: s + t = st id 257
seqs: [('l o w', 5), ('l o w e r', 2), ('n e w e st', 6), ('w i d e st', 3)] | total tokens: 70

--- round 2 ---
counts desc: {'e st': 9, 'w e': 8, 'l o': 7, 'o w': 7, 'e w': 6, 'n e': 6, ...}
tied at max: ['e st']
selected: e + st = est id 258
seqs: [('l o w', 5), ('l o w e r', 2), ('n e w est', 6), ('w i d est', 3)] | total tokens: 61

--- round 5 ---
counts desc: {'e w': 6, 'n e': 6, 'w est': 6, ...}
tied at max: ['e w', 'n e', 'w est']
selected: w + est = west id 261
...
--- round 12 ---
counts desc: {'lowe r': 2}  selected: lowe + r = lower id 268

=== merges ===
['s t', 'e st', 'o w', 'l ow', 'w est', 'n e', 'ne west', 'w i', 'wi d', 'wid est', 'low e', 'lowe r']
handout: ['s t','e st','o w','l ow','w est','n e','ne west','w i','wi d','wid est','low e','lowe r']
12-merge new tokens: [(257,'st'),(258,'est'),(259,'ow'),(260,'low'),(261,'west'),(262,'ne'),
                      (263,'newest'),(264,'wi'),(265,'wid'),(266,'widest'),(267,'lowe'),(268,'lower')]
=== encode checks ===
newest   6 merges -> ['ne', 'west']   12 merges -> ['newest']
widest   6 merges -> ['w', 'i', 'd', 'est']  12 merges -> ['widest']
lower    6 merges -> ['low', 'e', 'r']       12 merges -> ['lower']
lowest   6 merges -> ['low', 'est']          12 merges -> ['low', 'est']
```

（脚本用「最短 rank 优先」的贪心编码，结果与 handout 的「按创建顺序重放」在全部 12 轮与这些例词上一致；两者是否在任意输入上恒等价，材料未讨论 → 不作断言。）

## 附录 B：本文件做过的核对动作（可复现）

| 核对项 | 命令要点 | 结果 |
|---|---|---|
| handout §2 行号 | `grep -n "2\.[1-7]" .tmp/a1_handout.txt` | 93/118/168/191/286/383/463 |
| 测试函数清单 | `grep -n "^def test_" A1/tests/test_tokenizer.py A1/tests/test_train_bpe.py` | 25 + 3 |
| 词表/merge 数量对账 | `python3 -c "json.load(...)"` | 500=256+1+243；50,257=256+50,000+1 |
| special token 字节数 | `python3 -c "print(len('<\|endoftext\|>'), len('<\|end_of_text\|>'))"` | 13 / 15 |
| 12 条 merge 复算 | 附录 A 脚本 | 与 `H:279-280` 逐项一致 |
| 逐行 vs 整段编码 | 用 fixture 自建编码器（无 tiktoken） | `"abc\n\n def"` → 628 vs 198,198（§3.3） |
| macOS rlimit 行为 | `resource.setrlimit(RLIMIT_AS, (rss+1e6, -1))` | `ValueError: Cannot convert negative int` |
| 生成器 + 装饰器时序 | 打印 wrapper/generator body 顺序 | limit 在迭代前已恢复（§2.6） |
