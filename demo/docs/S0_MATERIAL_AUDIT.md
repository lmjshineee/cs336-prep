# S0 材料核对与事实审计（独立）

- 日期：2026-10-30
- 审计者：materials-auditor（独立于 S0 数据契约作者与 S1 导出实现者）
- 审计对象：`/Users/yunni/Projects/temp/cs336-prep-code/real_llm_lab/`（只读）
- 依据：`docs/GPT2_SOURCE_GROUNDED_PLAN.md` §2、§3.3、§3.4、§10.3、§10.4、§12 S0/S1；`docs/TRACE_SCHEMA.md` §2、§5.1、§7、§8、§10
- 本轮边界：未修改材料目录任何文件，未改产品代码、文档、计划、备份；临时脚本与输出全部在 `output/s0-audit/`。未运行 exp2/exp3/exp4/exp5，未重跑 `exp1_parity.py` 全脚本，未运行 `download_weights.py`，未联网，未改动权重。

状态标签（全文只用这三种）：

| 标签 | 含义 |
|---|---|
| **已核对** | 本轮用只读脚本在本机实测，或有可直接复核的字节级/代码级证据 |
| **历史记录未复跑** | 只读取了 `runs/*.txt` 与源码，本轮没有重新执行该实验 |
| **未验证** | 材料不足以确认，或本轮明确没有做（需要外部核实或留到 S1） |

## 后续状态（2026-10-30 追加）

S1 已导出主实例并通过独立核对：`data/gpt2/verification.json` 496/496 通过（required 492），覆盖本文件 §9 的 U08（采集等价）、U10（逐字节可复现）、U11（体积）。证据与命令见 [`docs/S0_S1_HANDOFF.md`（已归档）](README.md#历史归档) §4.3。本文件仍保持 S0 时点的审计结论与“未验证”标签；S0 审计本身没有重跑 S1 的独立核对。

## 0. 结论摘要

1. **材料身份（已核对，限本地）**：`weights/gpt2/model.safetensors` 的 sha256 = `248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707`，与 `download_weights.py:40` 记录的期望值一致；文件头统计 160 个张量、其中 148 个参数共 124,439,808 个元素、12 个 `h.{i}.attn.bias` 掩码缓冲，无 `lm_head.weight`。**本轮没有联网重新下载或与上游逐字节比对（未验证）**。
2. **环境（已核对）**：`.venv` Python 3.14.3、torch 2.14.1、transformers 5.21.0、safetensors 0.8.0、numpy 2.5.4、tokenizers 0.23.3；macOS 27.2 / Apple M5（10 核，4 性能 + 6 能效，16 GB），CPU fp32，torch 线程 4。
3. **代码事实（已核对）**：配置 12 层 / 12 头 / 768 维 / 1024 位置 / 词表 50257 / eps 1e-5；参数量 124,439,808；`lm_head.weight = wte.weight`（`gpt2_real.py:202`）；`forward` 返回全部位置 `(B,T,V)`（`:223`）；`generate_cached(max_new_tokens=3)` 实测 **4 次前向、3 次采样，最后一次前向的 logits 未被使用**；KV cache 实际分配 **75,497,472 B = 72.0 MiB = 75.5 MB**，源码注释第 81 行写作「75 MiB」，属材料表述问题（见 D01，不改源码）。
4. **分词器（已核对）**：主实例 `The cat sat on the mat.` 是 **7 个 BPE token**（不是九词），ids `[464, 3797, 3332, 319, 262, 2603, 13]`，7 段原始字节连续覆盖 23 个 UTF-8 字节，逐 token 单独解码均可读。
5. **参考实现 parity（本轮复跑，已核对）**：与 `GPT2LMHeadModel`（`local_files_only=True`、`attn_implementation="eager"`）相比，参数量相等；主实例全输出 logits（7 个位置 × 50257）最大绝对差 **0.0（逐位相等）**；12 层 × 12 头 = 144 个注意力矩阵最大绝对差 **0.0**；greedy 3 个新 token 两边都是 `[198, 198, 1]`。
6. **主实例（建议，不代替决定）**：建议**保留**，理由与教学风险见 §7；greedy 续写是 `"\n\n\""`（换行、换行、双引号），不是语义上的“下一个词”，最终取舍由用户决定。
7. **历史实验（全部历史记录未复跑）**：exp2–exp5 的复杂度、耗时、induction head、跨语言 perplexity、掩码影响、采样分布均来自 `runs/*.txt`，本轮只读登记，不能当作本机当前速度或因果证明（§6）。
8. **发现的材料表述问题 13 项（D01–D13）**，其中 4 项会影响教学或 S1 表述（75 MiB、`download_weights.py` 的“参数总量”口径、吞吐量定义、每前向体积口径），均只登记、不改材料。

## 1. 审计范围与方法

### 1.1 只读与防污染措施

- 材料目录只读；所有 Python 命令带 `PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`，脚本内另设 `sys.dont_write_bytecode = True`。
- 运行前记录 `__pycache__` 基线（411 个目录，`output/s0-audit/pycache_baseline.txt`），结束后复查（416 个，`pycache_after.txt`）：**`real_llm_lab/` 下始终没有 `__pycache__` 或 `*.pyc`**（已核对）。新增的 5 个都在材料目录之外：`.tmp/review_snapshot/`（17:52:47，`step06_model.pyc`）、`.tmp/scratch_lead/`（17:35:37，`llama_proto.pyc`）、`.tmp/scratch_train-eval/`（17:42:59，`step06_model.pyc`）、`.venv/.../blip_2/`（17:35:55，`__init__.pyc`）、`llama_steps/`（17:43:39，`step03_rope/step06_model/step07_train.pyc`）；这些目录/文件都不是本轮脚本会导入的模块（本轮只导入 `gpt2_real` 与 `transformers`），且本轮所有命令均带 `PYTHONDONTWRITEBYTECODE=1`，判断来自并行工作的其它成员/其它步骤脚本，与本轮审计无关。
- 未修改任何既有文件；材料文件的修改时间仍为 2026-10-27/28（见 §2）。
- 复跑 parity 时没有修改 `gpt2_real.py`，前向计数器通过给**实例**挂 `model.forward` 包装实现（脚本退出即失效，磁盘文件未变）。

### 1.2 复现脚本（全部在 `output/s0-audit/`）

| 脚本 | 产出 | 用途 |
|---|---|---|
| `audit_code_facts.py` | `code_facts.json`、`code_facts_stdout.txt` | 配置、参数量、权重共享、前向形状、分词器字节映射、KV cache 字节、`generate_cached` 前向计数、注意力形状 |
| `audit_parity.py` | `parity.json`、`parity_stdout.txt` | 与 HF `GPT2LMHeadModel`(eager) 的参数量 / logits / 144 头注意力 / greedy 3 token |
| `audit_collect.py` | `collect_probe.json` | prefill/decode 的 `last_attn` 形状、`attention_maps` 签名、cache 有效区、非 ASCII 单 token 解码 |
| `audit_record_path.py` | `record_path.json` | `record=True` 采集路径是否改变输出；4 次前向逐次注意力形状与行和 |
| 材料哈希 | `materials_sha256.tsv` | 24 个材料文件的 sha256 / 字节数 / 修改时间 |
| `pycache_baseline.txt` / `pycache_after.txt` | — | 运行前后 `__pycache__` 对比 |

统一命令形式（下文不再重复）：

```bash
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /Users/yunni/Projects/temp/cs336-prep-code/.venv/bin/python \
  /Users/yunni/Projects/temp/demo/output/s0-audit/<script>.py
```

## 2. 材料清单与哈希

命令：`cd real_llm_lab && shasum -a 256 <file>; stat -f '%z %Sm' <file>`（结果见 `output/s0-audit/materials_sha256.tsv`）。状态：**已核对**（本地字节）。

### 2.1 代码与脚本

| 文件 | 字节 | 修改时间 | sha256 |
|---|---:|---|---|
| `gpt2_real.py` | 18,669 | 2026-10-28 09:49:40 | `f8201019db7459491a84b8ff73eb081c78e431d08990757ec8bcf81bf0c31ebe` |
| `exp1_parity.py` | 8,351 | 2026-10-28 10:01:49 | `74a1d67265807a4d91e5e2e95eb5496d134824b2550096c72836e63ec33b74cc` |
| `exp2_heads.py` | 25,710 | 2026-10-27 23:55:10 | `046ec76ff6fbf867d065eccfc99da5a4bca327506df7792f679f626772e598fb` |
| `exp3_kv_cache.py` | 30,638 | 2026-10-27 23:55:32 | `c2fffad55d6e31c56df14fcd3f3b2c4c9719ea0c0c80a0289cd5e65b2c15a578` |
| `exp4_mask_ppl.py` | 30,886 | 2026-10-28 10:15:37 | `8f82c0ffdfdddbedab8b04ea73decebbdc5798aa753a08f02fadd70d2a09bb8b` |
| `exp5_sampling.py` | 10,810 | 2026-10-28 10:10:50 | `392a7c543ec9cfb0f17a6985fcaf35a5c1e5c38958b4e3c5c440d61b1aa88979` |
| `download_weights.py` | 4,615 | 2026-10-28 10:10:50 | `33fcfecbb3992a6091c5a8983b646a979e31e5c6838eb3496177b0cff42d608b` |
| `requirements.txt` | 425 | 2026-10-27 23:52:48 | `be4d83a8bd883949170352cf6eaf355231dc3e2fcb88cfd7a638e0bc87d055a9` |
| `run_all.sh` | 1,293 | 2026-10-27 23:52:48 | `427282c66d1523991da8e1973be79806c190944145af8be3b021ccec7281f7f6` |

### 2.2 历史日志 `runs/`

| 文件 | 字节 | 修改时间 | sha256 |
|---|---:|---|---|
| `runs/gpt2.txt` | 619 | 2026-10-28 10:16:46 | `b86e8d934c9fe08dfc663adc166596597eadfb8f30474f8a507c99921310a962` |
| `runs/exp1.txt` | 4,374 | 2026-10-28 10:16:49 | `be98fcb44490385b6ae45bc291247b861c2a7f1fc852083fb1f6f9d35b9ef596` |
| `runs/exp2.txt` | 7,120 | 2026-10-28 10:16:51 | `ed09e26da207663af1e57d47f4ba796059f39619243aa170b055197b91c2a8dd` |
| `runs/exp3.txt` | 8,241 | 2026-10-28 10:17:07 | `c88a58f6d570088b27f142b437fca03e04cd4a0b57dde82d74199744dbd119b2` |
| `runs/exp4.txt` | 8,384 | 2026-10-28 10:17:14 | `a5805cba4012af2cc6124ba5447acd46c13fd3dd168d9a2fee7a25b1d5beff51` |
| `runs/exp5.txt` | 6,375 | 2026-10-28 10:17:17 | `2823a8f7cdbf5c2092d81bc9866b07e8ca3cd21fc8274b66c661cec00ce2469a` |

### 2.3 图 `figs/`

| 文件 | 字节 | 修改时间 | sha256 |
|---|---:|---|---|
| `figs/exp2_induction_heads.png` | 132,785 | 2026-10-28 10:16:51 | `5d5547cd9567b99fb891641326ce0be3049400ab1fb14fcb369b7ba4dd3e9ea5` |
| `figs/exp3_kv_cache.png` | 117,732 | 2026-10-28 10:17:07 | `953469d4dc9c7e6f475569b1f5f83df5737316c631306e02534df22aa2c7a63a` |
| `figs/exp4_mask_training.png` | 35,367 | 2026-10-28 10:17:14 | `d1a7626f9b4c96c083a680d48f1c7e42c1678cdc9637930b923bb5bafd48fd21` |
| `figs/exp4_perplexity.png` | 116,871 | 2026-10-28 10:17:12 | `28b30e5d00633bda55ddc629cd0dedbdc6080335cef6dbd9e7913601f8517730` |

### 2.4 权重 `weights/gpt2/`

| 文件 | 字节 | 修改时间 | sha256 |
|---|---:|---|---|
| `config.json` | 665 | 2026-10-27 23:37:07 | `0daed7749b4f02b8f76240d5444551d7b08712dab4d0adb8239c56ba823bb7b4` |
| `model.safetensors` | 548,105,171 | 2026-10-27 23:49:41 | `248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707` |
| `tokenizer.json` | 1,355,256 | 2026-10-27 23:39:03 | `8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6` |
| `vocab.json` | 1,042,301 | 2026-10-27 23:37:17 | `196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783` |
| `merges.txt` | 456,318 | 2026-10-27 23:37:20 | `1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5` |

### 2.5 其它文件（不在 S1 依赖路径，未做事实核对）

| 文件 | 字节 | 说明 |
|---|---:|---|
| `README.md` | 11,917 | 实验说明；**未验证**（本轮未逐条核对其中结论） |
| `OUTPUT.md` | 37,095 | 实验输出汇总；**未验证**（同上） |
| `.mplcache/fontlist-v3.11.0.json` | 453,268 | matplotlib 缓存，非材料 |
| `weights/.DS_Store` | 6,148 | macOS 元数据，非材料 |
| `weights/gpt2/.cache/huggingface/` | 4 个文件 | 下载元数据（见 §4.9 与 D11），`download/model.safetensors.metadata` 125 B |

材料目录共 32 个文件、约 527 MB（`du -sh`），其中权重 548,105,171 B。

## 3. 运行环境

状态：**已核对**（`audit_code_facts.py` 输出 `code_facts.json`；环境记录命令见附录 A.1）。

| 项 | 值 | 证据 |
|---|---|---|
| 解释器 | `/Users/yunni/Projects/temp/cs336-prep-code/.venv/bin/python`，Python **3.14.3**（Clang 22.1.1） | `sys.version` |
| 其它 Python（对照） | PATH `python3` = 3.14.10；`/usr/bin/python3` = 3.9.6 | `python3 --version` |
| 平台 | `macOS-27.2-arm64-arm-64bit-Mach-O`，macOS 27.2（build 26B123） | `platform.platform()`、`sw_vers` |
| 机型 | MacBook Air，Apple **M5**，10 核（4 性能 + 6 能效），16 GB | `system_profiler SPHardwareDataType` |
| torch | **2.14.1**，CPU fp32；CUDA 不可用，MPS 可用但本轮未用 | `torch.__version__` |
| torch 线程 | `get_num_threads()=4`，`get_num_interop_threads()=10` | 同上 |
| transformers | **5.21.0** | 同上 |
| safetensors | **0.8.0** | 同上 |
| numpy | **2.5.4** | 同上 |
| tokenizers | **0.23.3** | 同上 |

历史日志只记录了 `PyTorch 2.14.1`（`runs/exp2.txt:3`、`runs/exp3.txt:5`），与本机一致；但历史运行时 transformers 的版本**未验证**（日志未打印）。

## 4. 已核对事实

每条给出证据。`已核对` 表示本轮实测或字节/代码级可复核。

### 4.1 配置与参数量

| ID | 事实 | 状态 | 证据 |
|---|---|---|---|
| F01 | `GPT2Config` 实际值：`n_layer=12, n_head=12, n_embd=768, block_size=1024, vocab_size=50257, layer_norm_eps=1e-5`，`head_dim=64`；与 `weights/gpt2/config.json` 的 `n_layer/n_head/n_embd/n_positions/n_ctx/vocab_size/layer_norm_epsilon` 完全一致；`activation_function="gelu_new"`，代码用 `nn.GELU(approximate="tanh")`（`gpt2_real.py:161`） | 已核对 | `code_facts.json` → `config_from_class`、`config_json_raw_keys_of_interest`、`config_eps_equals_raw` |
| F02 | 参数量 **124,439,808**（`sum(p.numel())`），全部 float32，权重字节 497,759,232 = 474.7 MiB | 已核对 | `code_facts.json` → `param_count`、`param_dtypes`、`weight_mib` |
| F03 | safetensors 文件头独立统计：160 个张量 = 148 个参数（合计 124,439,808 元素）+ 12 个 `h.{i}.attn.bias`（各 `(1,1,1024,1024)`）缓冲；**无 `lm_head.weight`**；全部 `F32`；数据区 548,090,880 B + 头部 14,283 B + 8 B = 548,105,171 B = 文件大小 | 已核对 | 见附录 A.3 |
| F04 | 计划 §2 给出的 12 处行号引用（`GPT2Config:49`、`KVCache:78`、`CausalSelfAttention.forward:114`、`MLP:157`、`Block.forward:179`、`GPT2.forward:204`、`attention_maps:227`、`load_weights:260`、`load_tokenizer:322`、`_next_token:348`、`generate_naive:360`、`generate_cached:373`）与当前文件一致 | 已核对 | `code_facts.json` → `source_lines`；逐行 `grep -n` |

### 4.2 权重共享与输出头

| ID | 事实 | 状态 | 证据 |
|---|---|---|---|
| F05 | `lm_head` 与 `wte` 共享同一参数对象与同一存储：`model.lm_head.weight is model.wte.weight` 为真，`data_ptr()` 相同，形状都是 `(50257, 768)`，`lm_head.bias is None` | 已核对 | `gpt2_real.py:200-202`（`:202` 为 `self.lm_head.weight = self.wte.weight`）；`code_facts.json` → `tying` |
| F06 | HF 参考实现同样绑定（`hf.lm_head.weight.data_ptr() == hf.transformer.wte.weight.data_ptr()`，`tie_word_embeddings=true`） | 已核对 | `parity.json` → `hf_lm_head_tied_to_wte`、`hf_config.tie_word_embeddings` |
| F07 | 权重文件没有独立的 `lm_head.weight`，因此共享是唯一来源；`load_weights` 仅在文件里存在该键时才覆盖（`gpt2_real.py:317-318`） | 已核对 | 附录 A.3；代码阅读 |

### 4.3 前向输出：全部位置

| ID | 事实 | 状态 | 证据 |
|---|---|---|---|
| F08 | `GPT2.forward` 返回 `(B,T,V)` 全部位置 logits：主实例输入 `(1,7)` → 输出 `(1,7,50257)` float32；源码 `return self.lm_head(self.ln_f(x))`（`:223`），`x` 形状 `(B,T,C)`，没有任何“只取最后位置”的切片；docstring（`:208`）写明 `(B, T, vocab_size)` | 已核对 | `code_facts.json` → `forward`、`source_lines.223_forward_return` |
| F09 | 逐位置一致性：对每个前缀 `0..t` 单独前向，其最后一行与整段前向的第 `t` 行最大绝对差 3.66e-4（7 个位置分别 3.66e-4、8.39e-5、7.63e-5、0、8.39e-5、0、0），属不同 T 下 BLAS 归约顺序差异，不是语义差异 | 已核对 | `code_facts.json` → `all_positions_semantics` |
| F10 | 更强的证据：整段 `(1,7,50257)` logits 与 HF 参考实现**逐位相等**（max abs diff 0.0，7 个位置都为 0），若实现只算最后位置不可能得到该形状与结果 | 已核对 | `parity.json` → `logits.max_abs_diff_all_positions=0.0`、`bitwise_equal=true` |

### 4.4 生成边界：`generate_cached` 的前向次数与多余前向

用实例级 `model.forward` 包装计数（源码未改），主实例 7 token、`max_new_tokens=3`、greedy：

| ID | 事实 | 状态 | 证据 |
|---|---|---|---|
| F11 | **前向次数 = 4**：prefill `start_pos=0, T=7` → decode `start_pos=7, T=1` → `8, T=1` → `9, T=1`；采样 3 次；**多执行 1 次前向** | 已核对 | `code_facts.json` → `generate_cached.calls`、`generate_cached_forward_count_table` |
| F12 | 规律：`max_new_tokens=1/2/3` 时前向数分别为 **2/3/4**（= 1 + N），多余前向恒为 1；`start_pos` 序列 `[0,7]`、`[0,7,8]`、`[0,7,8,9]` | 已核对 | 同上 |
| F13 | 最后一次前向（第 4 次）输入是最后一个生成 token（id 1，位置 9），输出 logits 预测位置 10；而输出序列长度 10（位置 0..9），**该 logits 未被任何采样使用**；它的 argmax 是 40（与本例输出无关） | 已核对 | `code_facts.json` → `generate_cached.last_forward`、`sampling_trace`（3 个采样分别来自前向 0/1/2，`matches=true`） |
| F14 | 与 `generate_cached` 不同，`generate_naive` 对 3 个新 token 只做 **3 次**前向（每步重算整段），没有多余前向 | 已核对 | `code_facts.json` → `generate_cached.naive_forward_calls=3` |
| F15 | `generate_cached` 的循环结构决定上述行为：prefill 一次（`:386`），循环体内“先采样、追加、再前向”（`:389-395`），因此最后一次追加后仍会前向一次，然后循环结束 | 已核对 | `gpt2_real.py:386,389-395`；与 `TRACE_SCHEMA.md` §5.1 的 fwd3 表完全一致 |
| F16 | 该行为与 `docs/TRACE_SCHEMA.md` §5.1 的 `forward_index 0..3`、`fwd3.logits.used_by_sampling=false` 描述一致 | 已核对 | 对照 `TRACE_SCHEMA.md` §5.1 |

### 4.5 KV cache 容量

| ID | 事实 | 状态 | 证据 |
|---|---|---|---|
| F17 | 每个 K/V 张量形状 `(B=1, n_head=12, block_size=1024, head_dim=64)`，`numel=786,432`，fp32 每张量 3,145,728 B；12 层 × (K,V) = **24 个张量** | 已核对 | `code_facts.json` → `kv_cache.shape_per_tensor`、`numel_per_tensor`、`bytes_per_tensor`、`n_tensors` |
| F18 | `KVCache.memory_bytes()` = **75,497,472 B = 72.0 MiB = 75.497 MB**；公式 `2(K,V) × n_layer × B × n_head × block_size × head_dim × 4 = 2×12×1×12×1024×64×4 = 75,497,472`；等价写法 `2×12×1×1024×768×4`（768 = n_head×head_dim） | 已核对 | 同上 `memory_bytes`、`formula_full`、`as_mib=72.0`、`as_mb_decimal=75.497472`；独立算术复核见附录 A.4 |
| F19 | 占权重比例 15.17%；每多 1 个上下文位置缓存增加 73,728 B（72 KiB） | 已核对 | `code_facts.json` → `cache_over_weight=0.15167`、`bytes_per_token_ctx=73728` |
| F20 | 源码注释第 81 行原文：`真实尺寸：2 * 12 层 * 1 * 1024 * 768 * 4 字节 = 75 MiB（单个样本 fp32）。`——算式结果 75,497,472 正确，但单位写错：该值是 **72 MiB**（75.5 MB 十进制），不是 75 MiB | 已核对 | `code_facts.json` → `kv_cache.source_line_81`；见 D01 |
| F21 | 缓存按 `block_size` **预分配**，只写入 `[start_pos:total]` 切片；prefill+1 步 decode 后有效区为 8，`k[:, :, 8:]` 全为 0；8 个位置实际使用 589,824 B | 已核对 | `collect_probe.json` → `cache`；`code_facts.json` 中 `KVCache` 构造 |

### 4.6 分词器与主实例

| ID | 事实 | 状态 | 证据 |
|---|---|---|---|
| F22 | 分词器类 `transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer`，`is_fast=True`，后端 `tokenizers.Tokenizer`（tokenizers 0.23.3），基类 `TokenizersBackend`（`GPT2TokenizerFast` 与它是同一个类）；`vocab_size=50257`；特殊 token 只有 `<|endoftext|>`(50256)；`model_max_length` 是一个约 1e30 的哨兵值（不可当真实上限，见 D09） | 已核对 | `code_facts.json` → `tokenizer` |
| F23 | 主实例 `The cat sat on the mat.` → **7 个 token**，ids `[464, 3797, 3332, 319, 262, 2603, 13]`；不是九词 | 已核对 | `code_facts.json` → `n_tokens`、`ids` |
| F24 | 逐 token 的 token 字符串、原始字节、文本字节区间（连续覆盖 23 个 UTF-8 字节）与单独解码：见下表 | 已核对 | `code_facts.json` → `token_bytes`、`byte_ranges_cover_text=true`、`decode_all_matches=true` |

| 位置 | id | token 字符串 | 原始字节 | 文本字节区间 | 单独解码 |
|---:|---:|---|---|---|---|
| 0 | 464 | `The` | `b'The'` (`54 68 65`) | [0, 3) | `'The'` |
| 1 | 3797 | `Ġcat` | `b' cat'` (`20 63 61 74`) | [3, 7) | `' cat'` |
| 2 | 3332 | `Ġsat` | `b' sat'` | [7, 11) | `' sat'` |
| 3 | 319 | `Ġon` | `b' on'` | [11, 14) | `' on'` |
| 4 | 262 | `Ġthe` | `b' the'` | [14, 18) | `' the'` |
| 5 | 2603 | `Ġmat` | `b' mat'` | [18, 22) | `' mat'` |
| 6 | 13 | `.` | `b'.'` (`2e`) | [22, 23) | `'.'` |

`Ġ` 是 GPT-2 字节级 BPE 对空格（0x20）的可打印映射；上表的原始字节由 256 项 `bytes_to_unicode` 逆映射还原（transformers 5.21 已不导出该函数，脚本内按 OpenAI 原实现内联，见附录 A.5）。

| ID | 事实 | 状态 | 证据 |
|---|---|---|---|
| F25 | 文本共 23 个 UTF-8 字节、**6 个空格分隔的词**（`The/cat/sat/on/the/mat.`），对应 7 个 BPE token；旧教学实例 `The cat sat on the mat because it was tired.` 的 9 个词级 token（`GPT2-LLaMA-Atlas/js/spec.js:626` 的 `DEMO.prefix`，该对象自 `:625` 起，`:629` 自述“均为教学模拟”）是模拟数据，不能沿用其索引 | 已核对 | `code_facts.json` → `text_utf8_len`、`space_separated_words`；`js/spec.js:625-629` |
| F26 | 逐 token 单独解码在本主实例上全部可读（无 U+FFFD）；但字节级 BPE 的单个 token 不一定能独立解码：`我爱自然语言处理`（8 字）→ 19 个 token，19 个单 token 解码**全部**含 U+FFFD，整段解码才恢复原文 | 已核对 | `collect_probe.json` → `non_ascii_single_token_decode`、`main_instance_single_token_decode_ok=true` |

### 4.7 参考实现 parity（本轮复跑）

容差取自 `docs/TRACE_SCHEMA.md` §8：logits `max|diff| ≤ 1e-4`、注意力 `max|diff| ≤ 1e-6`、参数量与 greedy token 精确相等。**本轮实测值**（`audit_parity.py`，CPU fp32，threads=4，同一权重目录，HF `local_files_only=True`、`attn_implementation="eager"`）：

| ID | 检查 | 本轮实测 | 容差 | 状态 |
|---|---|---|---|---|
| F27 | 参数量：自己的实现 vs HF vs 官方 124,439,808 | `124,439,808` / `124,439,808` / `124,439,808`，相等 | 精确 | 已核对 |
| F28 | 主实例全输出 logits `(1,7,50257)` 最大绝对差 | **0.0**（逐位相等，7 个位置均为 0；logits 绝对值最大 133.91） | ≤1e-4 | 已核对 |
| F29 | 全 12 层 × 12 头 = 144 个注意力矩阵 `(12,1,12,7,7)` 最大绝对差 | **0.0**（逐位相等；每层 0.0） | ≤1e-6 | 已核对 |
| F30 | greedy 3 个新 token：自己的 `generate_cached` vs HF `generate(do_sample=False)` | 两边都是 `[198, 198, 1]` → `"\n\n\""`；全序列 `[464,3797,3332,319,262,2603,13,198,198,1]` 相同 | 精确 | 已核对 |
| F31 | 注意力基本性质：行和 0.99999982–1.00000024；严格上三角（未来位置）最大绝对值 0.0；最小权重 0.0 | 与 HF 相同 | — | 已核对 |
| F32 | 历史记录 `runs/exp1.txt`：权重逐张量 0.0、4 条 prompt logits 0.0、paragraph loss 两边 2.8974、12 token greedy 相同、注意力 0.0 | 未在本轮重跑该脚本 | — | **历史记录未复跑**（其主实例部分已由 F27–F31 独立复跑支持） |

补充说明：历史 `exp1.txt` 的注意力对比用的是 10-token 句子 `The quick brown fox jumps over the lazy dog.`，**不是主实例**；本轮复跑把注意力对比补到了主实例的 7 token 上。历史 exp1 的另外 3 条 prompt 与 paragraph loss **本轮未复跑（未验证）**。

### 4.8 采集可观测性（S1 前置检查）

| ID | 事实 | 状态 | 证据 |
|---|---|---|---|
| F33 | `record=True` 只额外保存 `weights.detach()`（`gpt2_real.py:147-148`），不改变计算：同一前向 record 开/关 logits **逐位相等**（max abs diff 0.0） | 已核对 | `record_path.json` → `record_off_vs_on` |
| F34 | 用 `record=True` + 每次 `last_attn.detach().clone()` 采集 4 次前向，greedy 序列仍为 `[198,198,1]`，与不采集完全一致；逐次注意力形状 `(1,12,7,7)`、`(1,12,1,8)`、`(1,12,1,9)`、`(1,12,1,10)`，各层同形，行和 1±1e-5，方形 prefill 严格上三角为 0 | 已核对 | `record_path.json` → `collect_path` |
| F35 | 逐次 `Tk` 满足 `Tk = start_pos + Tq`（0+7、7+1、8+1、9+1），即 decode 时不存在“未来 key” | 已核对 | `record_path.json` → `tk_equals_start_plus_tq` |
| F36 | `attention_maps(idx)` 只接受 `idx`，**不支持 kv_cache**，只能产出一条完整前向（T=Tq=Tk）的注意力；decode 步必须用 `record`+`last_attn` 或 hooks 采集 | 已核对 | `collect_probe.json` → `attention_maps_signature`、`attention_maps_accepts_cache=false` |
| F37 | `_next_token` 的 top_k 分支先 `logits / temperature`（产生新张量）再做原地 `-inf` 掩码，**不会修改调用方传入的 logits**；greedy 分支直接返回，也不修改 | 已核对 | `code_facts.json` → `_next_token_topk_mutates_input.mutated=false`、`_next_token_greedy_mutates_input=false` |

### 4.9 权重来源与指纹

`download_weights.py` 记录的来源与期望值（代码阅读，**已核对**）：

| 项 | 内容 | 行号 |
|---|---|---|
| 来源 URL | `https://huggingface.co/openai-community/gpt2/resolve/main` | `:23` |
| 下载文件 | `config.json`、`tokenizer.json`、`vocab.json`、`merges.txt`、`model.safetensors` | `:27-33` |
| 说明 | “这个文件就是 HuggingFace 上的 openai-community/gpt2 仓库” | `:9` |
| 期望大小 | `model.safetensors` = 548,105,171 B | `:36-38` |
| 期望 sha256 | `model.safetensors` = `248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707` | `:39-41` |
| 校验方式 | 下载后校验大小 + sha256；最后用 `safe_open` 解析文件并统计张量 | `:90-124` |

本地文件指纹（**已核对**，`materials_sha256.tsv`）：`model.safetensors` 的 sha256 与上表期望值**完全一致**，大小 548,105,171 B 也一致；文件头 160 个张量、148 个参数共 124,439,808 元素（F03）。

额外的本地来源痕迹（**已核对**，文件内容）：`weights/gpt2/.cache/huggingface/download/model.safetensors.metadata` 三行分别为上游 commit `607a30d783dfa663caf39e06633721c8d4cfcd7e`、sha256 `248dfc39…a707`、epoch `1793116184.6322188`（= 2026-10-27T23:49:44.632219，与文件修改时间 23:49:41 相差数秒，符合“下载完成后写元数据”）。

| ID | 事实 | 状态 | 证据 |
|---|---|---|---|
| F38 | 本地 `model.safetensors` 的 sha256/大小与 `download_weights.py` 记录的期望值一致；下载元数据记录的 sha256 也一致 | 已核对 | §2.4、`:36-41`、`.cache/.../model.safetensors.metadata` |
| F39 | **本轮没有联网重新下载，也没有与上游逐字节比对**；4 个小文件（config/tokenizer/vocab/merges）没有期望哈希可校验 | 未验证 | D10、U01–U03 |
| F40 | 上游 commit `607a30d…` 是否为当前上游内容，离线无法核实 | 未验证 | U02 |

## 5. 差异与风险

只登记，不修改材料；影响与建议供 S1/用户判断。

| ID | 差异 | 证据 | 影响 | 建议 |
|---|---|---|---|---|
| **D01** | `gpt2_real.py:81` 注释写「= 75 MiB」，实际 75,497,472 B = **72.0 MiB** = 75.5 MB。注释算式里的 `768` 其实是 `n_head×head_dim`（= n_embd），与上一行 docstring 写的形状 `(B, n_head, block_size, head_dim)` 用词不一致 | F18–F20；`runs/exp3.txt:89-90` 自己写的是 72.0 MiB | 若教学照抄注释，缓存体积会高估 4.2%，并与实验日志矛盾 | S1/课程按实测值表述；源码不改（计划 §3.4 已要求“记录，避免照抄”） |
| **D02** | `exp1_parity.py` 内部阈值口径不一：docstring（`:9-10`）说 1e-4；逐张量断言 `< 1e-5`（`:50`）；logits `allclose(atol=1e-4)`（`:106`）；注意力断言 `< 1e-5`（`:159`）。而 `TRACE_SCHEMA.md` §8 是 logits 1e-4、注意力 1e-6 | 代码阅读 | 本轮所有差值为 0，无实际影响；但 S1 若照抄 exp1 的断言会把注意力门槛放宽 10 倍 | S1 一律用 schema §8 的容差，不用 exp1 的内部阈值 |
| **D03** | `download_weights.py:110-124` 的「参数总量」把 160 个张量全部相加，含 12 个 `(1,1,1024,1024)` 的 `attn.bias` 掩码缓冲 → 会打印 **137,022,720**；而 docstring（`:32`）写「1.24 亿个浮点数」，`gpt2_real.py:420` 断言 124,439,808 | 代码阅读 + 文件头统计（附录 A.3）；**未执行该脚本** | 读者可能误以为权重文件与模型参数量对不上 | 以文件头/模型 `numel()` 为准；该脚本的打印口径标为“张量元素总数” |
| **D04** | `runs/exp4.txt` 同一段文本出现两套 token 数：正文 235/314/380（编码长度）与表格 234/313/379（被预测 token 数 = T−1），连带的 tok/char 也有 2.11 与 2.10 两种写法；日志未说明差异 | `runs/exp4.txt:14,62-65,116`；`exp4_mask_ppl.py:386,394,403` | 引用时容易差 1，或误以为代码有 bug | S1/课程引用时写清“token 数”指编码长度还是被预测数 |
| **D05** | `runs/exp3.txt` 的「cached 吞吐」= (prompt+new)/耗时，不是纯解码速度：16/64/128 prompt 分别 48/0.2573=186.6、96/0.2743=350.0、160/0.2863=558.9 tok/s，与日志逐项吻合 | `runs/exp3.txt:51-53`；算术复核 | 把 558.9 tok/s 当“生成速度”会高估（其中含 prefill token） | 引用时注明口径；S6 若要计时须重跑并写清分母 |
| **D06** | `attention_maps` 在 `try/finally` 之外设置/复位 `record`，且 `last_attn = weights.detach()` 不带 clone；`attention_maps` 也不支持 kv_cache | F36；`gpt2_real.py:226-239,147-148` | 若采集中途异常，`record` 会留在 True；跨步保存引用有别名风险 | S1 用独立适配器：`try/finally` 复位，逐次 `detach().clone()` |
| **D07** | `KVCache` 原地写入预分配张量，未用区域保持 0；采集时若直接保存缓存张量会保存大量零区，且后续写入会改变已保存视图 | F21；`gpt2_real.py:86-93,132-135` | 违背 `TRACE_SCHEMA.md` §10.4「detach 并复制到独立存储」 | 只导出容量元数据 + 有效/写入切片；切片必须 clone |
| **D08** | `generate_naive` 用 `idx[:, -block_size:]` 且不传 `start_pos`，序列超过 1024 后位置编码会从 0 重新开始（绝对位置错误） | `gpt2_real.py:366` | 主实例（10 个位置）不受影响；S6 若做长生成会得到错误位置 | 不在本轮范围；S6 需注意或另写实现 |
| **D09** | tokenizer 的 `model_max_length` 是约 1e30 的哨兵值；类名 `GPT2Tokenizer` 但 `is_fast=True`。transformers 5.21 中 `GPT2TokenizerFast` 就是 `GPT2Tokenizer` 本身（同一类对象），且基类含 `transformers.tokenization_utils_tokenizers.TokenizersBackend`，即快慢实现已合并 | F22；本轮实测 `GPT2TokenizerFast is GPT2Tokenizer`、`GPT2Tokenizer.__mro__` | 若按 `model_max_length` 截断会出错；按类名判断“慢分词器”会误判 | S1 用 `config.block_size`（1024）作为上限 |
| **D10** | `download_weights.py` 只对 `model.safetensors` 校验大小与 sha256，`config.json`/`tokenizer.json`/`vocab.json`/`merges.txt` 没有期望哈希 | `download_weights.py:36-41,90-97` | 4 个小文件的来源完整性只能靠功能验证（能加载、能分词、parity 通过） | 记为未验证项 U03；如需强证据要外部核对 |
| **D11** | `weights/gpt2/.cache/huggingface/download/model.safetensors.metadata` 记录上游 commit `607a30d783dfa663caf39e06633721c8d4cfcd7e` 与 sha256；本轮离线无法核实该 commit 是否仍是上游当前内容 | 文件内容 3 行：commit、sha256、epoch 时间 | 不影响本地可复现性 | 未验证项 U02 |
| **D12** | `TRACE_SCHEMA.md` §7 写「注意力权重每前向 < 30 KB」：原始 float32 成立（prefill 28,224 B），但 JSON 里 prefill 的 base64 是 37,632 B（约 36.8 KiB），48 个张量合计 base64 58,368 B | **S1 实测** `data/gpt2/gpt2-main-001.json`：48 张量原始 43,776 B、base64 58,368 B；算术复核见附录 A.6 | 若预算按 JSON 字节核对，措辞会不一致（仍远小于 6 MiB 目标） | S1 已按「原始字节 + base64 字节」分别报告（§10.3），口径统一 |
| **D13** | `runs/exp3.txt:83` 用「等效带宽 61 GB/s」解释固定开销，这是由 475 MiB / 7.9 ms 反推的估计值（≈63 GB/s，量级相符），不是独立测量 | 算术复核 | 若当作实测带宽会被误读 | 引用时标为“估计” |

## 6. 历史结论登记表（exp2–exp5 与 runs）

### 6.1 使用总则

- 下表所有实验结论的状态都是 **历史记录未复跑**：本轮只读代码与 `runs/*.txt`，没有执行 exp2/exp3/exp4/exp5。
- **不得把历史计时当作本机当前速度**：`runs/exp3.txt` 的毫秒/加速比、`runs/exp5.txt` 的耗时、`runs/exp2.txt` 的 16 ms 都是 2026-10-28 在 Apple M5 / CPU / fp32 / threads=4 上的一次性测量；本机当前环境（§3）虽与日志记录的 torch 2.14.1 一致，但没有重测，不能承诺同样的数字。
- **不得把热图当因果证明**：exp2 的 induction score 是“某个头在这条输入上把注意力放到 `i-period+1`”的相关性统计（`exp2_heads.py:190-205`），不能证明该头“负责”复制行为或决定输出；计划 §3.5 已明确这一点，S6 若要讲因果需另做消融。
- 每条都写明实验条件；跨实验/跨分词器的数字不可直接比较（尤其 perplexity）。

### 6.2 复杂度与耗时

| ID | 结论（可能被新计划引用） | 出处 | 实验条件 | 状态 | 使用限制 |
|---|---|---|---|---|---|
| R01 | naive 生成 N 个 token 是 `O(N^3)`，cached 是 `O(N^2)`（每步只算新 token 的 q/k/v，历史 K/V 查表） | `exp3_kv_cache.py:5-7`、`:593`；`runs/exp3.txt:121` | 生成 N 个 token，naive 每步重算整段前缀 | 历史记录未复跑 | 算法阶数结论；实测加速比远小于阶数比（见 R02），且 naive 实现有 `block_size` 切片上限（D08） |
| R02 | 端到端加速比：prompt 16/64/128 → **×2.19 / ×3.19 / ×4.64**；理论比 ×21.0/×26.5/×28.7 | `runs/exp3.txt:51-53`；`exp3_kv_cache.py:230-280` | 每配置生成 32 token，预热 1 次 + best-of-2；Apple M5 / CPU / fp32 / torch 2.14.1 / threads 4 | 历史记录未复跑 | 只代表那次运行；换机器、换线程、系统负载都会变；理论比 = 两实现的 token-前向总次数之比（已用算术复核：16+32 时 1008/48=21.0） |
| R03 | 单步：全量前向 T=16/64/256/512 → 14.5/22.5/63.3/131.0 ms；cached decode 约 7.41–8.21 ms 且几乎恒定；固定开销约 7.9 ms/次调用 | `runs/exp3.txt:56-84` | 预热 1 + best-of-3；同上机器 | 历史记录未复跑 | 7.9 ms 是“大上下文边际成本 vs T=1 调用”的差；不得当作本机当前 decode 延迟 |
| R04 | 144 个 17×17 注意力矩阵一次前向耗时 16 ms | `runs/exp2.txt:18`；`exp2_heads.py:315-316` | 17-token 输入，`attention_maps` 一次前向 | 历史记录未复跑 | 只是该次测量；不是本机承诺 |
| R05 | exp5 五种解码策略各生成 40 token 用时 0.33–0.35 s，总耗时 2.8 s | `runs/exp5.txt:12-32,111`；`exp5_sampling.py:121-125` | prompt 6 token，`SEED=1234`，每种策略重置随机状态 | 历史记录未复跑 | 历史耗时；采样本身有随机性，需同 seed/同环境才能复现文本 |

### 6.3 注意力模式 / “复制头”

| ID | 结论（可能被新计划引用） | 出处 | 实验条件 | 状态 | 使用限制 |
|---|---|---|---|---|---|
| R06 | 最强 induction 头 **L5H5**，score **0.7484**，为 144 头均值 0.0771 的 9.7 倍；注意力峰值偏移 k=9 = period−1 | `runs/exp2.txt:62-91,115`；`exp2_heads.py:190-211,379-434` | 输入 `The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the`（17 token，period=10），在 6 个对齐位置上平均（跳过 BPE 边界位置 10） | 历史记录未复跑 | 单条输入、单一定义（score 只在对齐位置平均）；是模式统计，不是因果；“period-1”依赖 BPE 对齐，换输入/换切分要重算 |
| R07 | score ≥ 0.20 的头有 **16 个**（0.20 是人为阈值） | `runs/exp2.txt:68-69,118` | 同上 | 历史记录未复跑 | 阈值是任意的；只能说明“不止一个头像”，不能证明冗余/因果 |
| R08 | 144 个头全部严格下三角、每行和 = 1；第 0 行只有 1 个非零、第 16 行 17 个非零 | `runs/exp2.txt:20-27`；`exp2_heads.py:128-164` | 同上 | 历史记录未复跑 | 本轮的 F31/F34 在 7-token 主实例上独立验证了同样的因果性质（不同输入） |
| R09 | 补全：`When Mary and John went to the store, John gave a drink to` → `' Mary'` p=0.4460，top-5 = `[(' Mary',0.4460),(' them',0.2116),(' the',0.0711),(' John',0.0600),(' her',0.0450)]` | `runs/exp2.txt:30-40,114` | 单条输入，greedy 前的分布 | 历史记录未复跑 | 单例；不能据此说模型“理解”共指 |
| R10 | 重复文本最后位置预测 `' lazy'` p=0.9678；最强头最后一行 0.5130 给位置 7（induction 目标），0.3324 给位置 0 | `runs/exp2.txt:94-100,117` | 同上 | 历史记录未复跑 | 位置 0 的固定注意力是常见现象（日志已注明）；机制到行为仍是相关性 |

### 6.4 跨语言 perplexity

| ID | 结论（可能被新计划引用） | 出处 | 实验条件 | 状态 | 使用限制 |
|---|---|---|---|---|---|
| R11 | per-token ppl：English 27.1、Chinese **16.9**、Python 14.4、Random 259,839；loss 3.2987/2.8272/2.6703/12.4678 | `runs/exp4.txt:62-66,136-141`；`exp4_mask_ppl.py:380-435` | 单段文本：英文新闻 1092 字符/234 预测 token、中文 149 字符/313、Python 761 字符/379、随机 234 个 id | 历史记录未复跑 | per-token ppl 跨分词器不可比（中文 2.10 tok/char）；单段文本；随机 id 用局部 `Generator(seed=0)` |
| R12 | bits/char：English 1.02、Chinese **8.57**、Python 1.92；结论“不同分词器的 perplexity 不能直接比，要用 bits/char 或同分词器” | `runs/exp4.txt:77-88`；`exp4_mask_ppl.py:409-433` | 同上；bits/char 由每 token nats 换算 | 历史记录未复跑 | 单段文本；bits/char 依赖字符计数口径（含换行/空格），换文本会变 |
| R13 | 随机权重 loss 10.9578 vs 均匀瞎猜 ln(50257)=10.8249；PyTorch 默认初始化 loss 484.9 | `runs/exp4.txt:42-45` | 同结构随机初始化（官方 N(0,0.02)，`Generator(seed=0)`） | 历史记录未复跑 | 依赖初始化尺度；不能当作“随机基线”的普适值 |

### 6.5 掩码影响

| ID | 结论（可能被新计划引用） | 出处 | 实验条件 | 状态 | 使用限制 |
|---|---|---|---|---|---|
| R14 | 同一段英文、同一权重，`causal=True` loss 3.2987/ppl 27.1；`causal=False` loss 5.9697/ppl 391.4（“被改坏”，不是作弊） | `runs/exp4.txt:15-16,136-137`；`exp4_mask_ppl.py:299-340` | 英文新闻段落，235 token；只改 `forward(causal=...)` | 历史记录未复跑 | 单段文本；`causal=False` 是故意改函数（作弊模式），不是正常推理路径 |
| R15 | 从零训练两个 1 层/1 头/64 维字符级小模型（300 步）：有掩码 train 2.3086 / 因果评估 2.6074；无掩码 train **0.0872** / 因果评估 **7.8801** | `runs/exp4.txt:116-141`；`exp4_mask_ppl.py:467-592` | 字符级分词器、约 2.1 KB 英文、`manual_seed(0)`、两模型同初始化、训练耗时 1.3 s | 历史记录未复跑 | 是 1 层玩具模型，**不是 GPT-2**；不能把它的 loss 与 GPT-2 的 loss 放同一张表比较；训练有随机性但已固定 seed |

### 6.6 采样分布

| ID | 结论（可能被新计划引用） | 出处 | 实验条件 | 状态 | 使用限制 |
|---|---|---|---|---|---|
| R16 | 温度改变 softmax 尺度：平坦 context 的熵 3.62/7.55/10.73 bit（T=0.5/1.0/1.5），最大概率 0.2858/0.0766/0.0213，top-40 质量 97.3%/61.2%/26.7%；尖锐 context 熵 0.01/0.98/8.91，最大概率 0.9992/0.9096/0.3471 | `runs/exp5.txt:55-93`；`exp5_sampling.py:74-112,187-210` | prompt `The future of artificial intelligence is` 与 `1, 2, 3, 4, 5,`，取最后一个位置的完整词表分布 | 历史记录未复跑 | 两个 context 的单位置观测；不是所有输入的规律；概率基于完整 50257 维 |
| R17 | 五种策略生成文本与“不同 token 数 33/40、32/40、37/40、36/40、32/40” | `runs/exp5.txt:12-32,105-111` | 40 个新 token，`SEED=1234`，每策略重置随机状态 | 历史记录未复跑 | 采样文本依赖 seed 与实现（`torch.multinomial`），换版本可能不同；多样性指标是启发式 |

### 6.7 代码与注释/日志的一致性检查

- 复杂度注释（`exp3_kv_cache.py:5-7`）与日志结论（`runs/exp3.txt:121`）一致；日志对“实测加速比远小于阶数比”的解释（`:594-599`）与代码打印一致。
- **不一致**：`gpt2_real.py:81` 的「75 MiB」与代码实际分配、`runs/exp3.txt:89-90` 的 72.0 MiB 矛盾 → D01。
- **不一致**：`exp1_parity.py` docstring 1e-4 与内部断言 1e-5、schema 1e-6 三种口径 → D02。
- **不一致**：`download_weights.py` 的「参数总量」口径会含 12 个掩码缓冲 → D03。
- **口径未说明**：`runs/exp4.txt` 的 token 数两套数字 → D04；`runs/exp3.txt` 的吞吐分母 → D05。
- `runs/exp2.txt` 的“16 个头的 score ≥ 0.20”与日志自述“0.20 是随便定的线”一致，属明确标注的启发式，不算不一致。
- `runs/exp3.txt:5` 记录机器 Apple M5 / 线程 4/10，与本机实测一致（但未重跑计时）。

## 7. 主实例与选定层

### 7.1 主实例实测（已核对）

| 项 | 值 |
|---|---|
| 文本 | `The cat sat on the mat.` |
| 编码 | 7 个 BPE token，23 个 UTF-8 字节，6 个空格分隔词 |
| ids | `[464, 3797, 3332, 319, 262, 2603, 13]` |
| 前向 | 4 次（prefill + 3 decode），第 4 次 logits 未使用 |
| greedy 3 新 token | `[198, 198, 1]` = `"\n\n\""`；完整解码 `The cat sat on the mat.\n\n"` |
| 每步被选 token 的概率（canonical float64） | 第 1 步 0.1887764578457502；第 2 步 0.9214682512711395；第 3 步 0.4010767584231396；（未使用的第 4 次前向 top1 0.09616257446513707） |
| 概率方法口径 | 上行为 `float32 logits 行 → 提升 float64 → softmax`（`TRACE_SCHEMA.md` §5.2 `:262` 规定的口径，§8 `:389` 定容差）。同一行的 float32 softmax 对照为 0.18878144025802612 / 0.9215197563171387 / 0.4010857045650482，差 4.98e-6 / **5.15e-5** / 8.95e-6；这是 50257 维 softmax 的 float32/float64 **方法差**，不是数据矛盾。S1 导出文件 `GPT2-LLaMA-Atlas/data/gpt2/gpt2-main-001.json` 的 `sample0..2.probability` 与本行 canonical 值**完全相等**（已核对，独立复算） |
| 第 1 步 top-5（float64） | `\n` 0.188776、` "` 0.084925、` He` 0.083714、` The` 0.083557、` She` 0.076465 |
| 第 1 步分布（float64） | 熵 **5.9578 bit**（均匀分布 15.617）；top-5 质量 0.517439；top-40 质量 0.803800；最大概率 0.188776（float32 对照：熵 5.9579、top-5 0.517452、top-40 0.803822、最大概率 0.188781） |
| 与 HF 一致性 | logits / 144 头注意力逐位相等；greedy 相同（F27–F30） |

### 7.2 适合性判断（**建议**，不代替用户决定）

**建议：保留 `The cat sat on the mat.` 作为主实例。** 理由：

1. 契约已固定：`TRACE_SCHEMA.md` §10 已把该文本、7 token、greedy 3 token、选定层第 0 层写死；本轮已按此完成 parity 与采集可行性核对（F27–F36），替换会使这些已核对结果与预算假设同时失效。
2. 规模合适：7 个输入位置 + 3 个新 token = 4 次前向，注意力载荷合计仅 43,776 B（原始），远低于预算（§10.3）。
3. 结果确定且可复核：greedy 无随机性，与 HF 逐位一致，适合作为“每个数字可追溯”的第一条轨迹。
4. 分布本身可教：第 1 步 top1 只有 18.9%、熵 5.96 bit、top-40 只占 80.4%，正好支撑 `TRACE_SCHEMA.md` §10.5「概率条保留其他 token 总质量」和“模型在此处并不确定”的教学点。

**风险（需要用户知情）**：greedy 续写是 `"\n\n\""`（两个换行 + 一个双引号），不是语义上的“下一个词”。如果课程首屏必须展示一个可读的词级续写（例如 ` mat`、` floor`），这条主实例的教学效果会偏弱。

**若决定替换（本轮不做）**：需要重新走同一套检查（tokenizer 长度、4 次前向、parity、注意力、体积预算），并同步修改 `TRACE_SCHEMA.md` §10 与 S1 期望值；本轮**没有测试任何替代候选（未验证）**。计划 §10.3 允许 S0 记录替换理由，但替换与否属于用户/Lead 的决定，本文只给建议。

### 7.3 选定层

- **选定层 = 第 0 层**（`TRACE_SCHEMA.md` §10 已定，本轮不改）。
- 依据：契约固定；S4 需要一个 Block 的完整 Q/K/V、LN、残差、MLP 数值，第 0 层最靠近嵌入、最容易与输入字节/位置对应；S1 的采集预算也只覆盖一个 Block。
- 未验证：第 0 层是否在所有教学点上“最有代表性”（例如注意力模式是否足够清晰）本轮没有评估；若 S4 发现第 0 层讲解效果差，需要另立变更并重跑采集，不能沿用其它层的数据（schema §2.3）。

## 8. 容差依据

### 8.1 本轮实测 vs 声明容差

| 检查 | 声明容差（`TRACE_SCHEMA.md` §8） | 本轮实测 | 结论 |
|---|---|---|---|
| 参数量 | 精确相等 | 124,439,808 = 124,439,808 | 通过 |
| 未缓存前向 logits vs HF eager | `max|diff| ≤ 1e-4` | **0.0**（逐位相等） | 通过（余量极大） |
| 全层注意力 vs HF attentions | `max|diff| ≤ 1e-6` | **0.0**（144 头） | 通过 |
| greedy token 序列 | 完全相等 | `[198,198,1]` 相同 | 通过 |
| 生成概率（canonical float64） | 同口径 `|diff| ≤ 1e-6`；float32 对照 `≤1e-4` 且必须注明方法差 | 本轮独立复算的 float64 概率与 S1 导出值完全相等；float32 对照差 4.98e-6 / 5.15e-5 / 8.95e-6 | 通过（方法差已记录） |
| softmax 行和 | `|Σ−1| ≤ 1e-6` | 偏差 ≤ 2.4e-7（0.99999982–1.00000024） | 通过 |
| 掩码位置权重 | 严格为 0 | 严格上三角 0.0 | 通过 |

### 8.2 为什么是这些值

- logits 容差 1e-4 与“logits 量级”配套：主实例 logits 绝对值最大 133.91，fp32 在该量级的 ulp 约 1.5e-5；1e-4 约 7 个 ulp，足以吸收不同 BLAS 归约顺序，又远小于影响 argmax/概率的尺度。历史 exp3 的 fp32 差异 7.02e-4（logits 量级 319）也是同一量级的舍入现象，说明**绝对差必须结合量级读**。
- 注意力容差 1e-6：softmax 输出在 [0,1]，7 个 key 的行和误差已到 2.4e-7；1e-6 既能吸收舍入又不会掩盖真正的实现差异（如掩码错位会带来 ≥1e-2 的差）。
- 以上是**本轮同机同权重的观测**；按计划 §12 S1 的要求，不能把 0 误差当作跨环境承诺。S1 若换机器/线程/库版本，必须重新声明并实测。
- 采集等价（`TRACE_SCHEMA.md` §8 第一行）要求“采集运行 vs 未采集运行逐位相等”；本轮已在主实例上用 `record=True` + clone 验证 logits 与 greedy 逐位/完全一致（F33–F34），但 hooks 版适配器尚未实现（S1 验证）。

## 9. 未验证项

| ID | 未验证内容 | 原因 | 建议由谁/何时验证 |
|---|---|---|---|
| U01 | 权重与上游逐字节一致（联网重新下载并比对） | 本轮离线；仅本地 sha256 与脚本内常量、HF 下载元数据一致 | 需要时由用户批准联网；S1 关口可引用本地指纹 |
| U02 | HF 上游 commit `607a30d783dfa663caf39e06633721c8d4cfcd7e` 仍是当前上游内容 | 离线 | 联网时核对 |
| U03 | `config.json`/`tokenizer.json`/`vocab.json`/`merges.txt` 的上游哈希 | 材料未记录期望值（D10） | 联网时核对；当前只有功能验证 |
| U04 | 历史实验运行时 transformers 的确切版本 | `runs/*.txt` 未打印 | 若要复现历史数字，先记录版本再重跑 |
| U05 | exp2–exp5 的全部数字在本机当前环境是否仍成立 | 本轮未执行（明确不跑重实验） | S6 引用前重跑 |
| U06 | exp1 的另外 3 条 prompt 与 paragraph loss（本轮只复跑主实例） | 控制范围 | S1 可加进验证器 |
| U07 | MPS/GPU 路径的行为与数值 | 本轮 CPU only；`TRACE_SCHEMA.md` §2.2 也规定 CPU fp32 | 不需要（首版 CPU） |
| U08 | hooks 版采集适配器是否也满足“采集=未采集逐位相等” | S1 导出器自检记录在 `GPT2-LLaMA-Atlas/data/gpt2/manifest.json` → `export_checks.capture_outputs_bitwise_equal` = `[0.0, 0.0, 0.0, 0.0]`（4 个 forward）；但 `instances[0].validation_status` 仍是 `"not_run"`，`verification.json` 尚未生成，**独立核对器未完成，不算最终通过** | 以 S1 独立核对器写出的 `data/gpt2/verification.json` 为准；在此之前不标为已核对 |
| U09 | 选定层第 0 层的教学代表性 | 本轮只核对形状/因果性，未评估模式可读性 | S4 前评估 |
| U10 | 案例文件逐字节可复现（同环境重跑两次） | S0 未做导出；**已由 S1 验证**：同环境连续两次导出案例 sha256 相同 `91e99a3474f32dd3ce0a523af07cab9b7fa24ebd3004c44ce4790096245317c9`（2,689,942 B） | 证据：`data/gpt2/gpt2-main-001.json` 与 `data/gpt2/manifest.json` → `instances[0].case_sha256` / `case_bytes`（S0 已读回核对一致）；S0 未独立重跑两次导出 |
| U11 | 实际 JSON 体积（attention + 选定层张量 + logits 行）是否 ≤6 MiB | S0 未生成载荷；**已由 S1 导出实测：2,689,942 B（2.57 MiB）**，低于 6 MiB | 证据：`data/gpt2/manifest.json` → `instances[0].case_bytes = 2689942`；组成见 §10.3（S0 已独立读回核对张量分类与字节） |

## 10. 对 S1 的输入与建议

### 10.1 固定输入与执行表（已核对，可直接作为 S1 期望值）

| forward | phase | start_pos | T | 输入 ids | 输出使用 | 注意力形状（每层） |
|---|---:|---:|---:|---|---|---|
| 0 | prefill | 0 | 7 | `[464,3797,3332,319,262,2603,13]` | sample0（`[198]`） | `(1,12,7,7)` |
| 1 | decode | 7 | 1 | `[198]` | sample1（`[198]`） | `(1,12,1,8)` |
| 2 | decode | 8 | 1 | `[198]` | sample2（`[1]`） | `(1,12,1,9)` |
| 3 | decode | 9 | 1 | `[1]` | **未使用** | `(1,12,1,10)` |

- 输出序列 `[464,3797,3332,319,262,2603,13,198,198,1]`，新 token 文本 `"\n\n\""`。
- 采样来自前向 0/1/2 的最后一行 logits（argmax 分别 198/198/1，已核对）；第 3 次前向的 logits 不得删除、不得标为“已使用”。
- 与 `TRACE_SCHEMA.md` §5.1 的表逐项一致。

### 10.2 采集实现要点

1. **注意力**：不能用 `attention_maps()` 采 decode 步（只接受 `idx`，F36）。用 `block.attn.record=True` + 每次前向后 `block.attn.last_attn.detach().clone()`（已核对不改变输出，F33–F34）；`try/finally` 复位 `record`（D06）。
2. **缓存**：只导出容量元数据（`(1,12,1024,64)`、75,497,472 B、每层/每 token 字节）与有效/写入区间（`start_pos..start_pos+T`），选定层可另存 clone 后的 K/V 切片；不要保存整块预分配张量（D07、schema §7）。
3. **logits**：保留生成实际使用的那一行完整 50257 维（fwd0/1/2 的最后一行）；其余位置只留摘要。采样前记录 logits 是安全的（`_next_token` 不改调用方张量，F37）。
4. **掩码**：prefill 的 softmax 之后严格上三角已经是 0（F31）；decode 前向不存在未来 key（`Tk = start_pos + Tq`，F35），因此 decode 没有需要掩码的位置。若要导出 masked scores，按 schema §2.2 用「有限分数 + u8 掩码」，不得写 `-inf`/NaN（schema 禁止）。
5. **确定性**：CPU、fp32、`eval()`、`torch.no_grad()`、`torch.set_num_threads(4)`；greedy 无采样随机数，仍按 schema 记录 `seed=0`。
6. **采集器状态**：`record` 开关、临时 hooks、输出目录都要在 `finally` 中恢复/清理；失败写 `export-failure.json`（schema §9），不发布半份轨迹。

### 10.3 体积预算（S1 已实测）

以下数字来自 S1 导出文件 `GPT2-LLaMA-Atlas/data/gpt2/gpt2-main-001.json`（2,689,942 B，148 个张量），由本轮独立读回核对：

| 载荷 | 张量数 | 原始字节 | base64 后 | 备注 |
|---|---:|---:|---:|---|
| 注意力权重（4 前向 × 12 层） | 48 | 43,776 | 58,368 | prefill 28,224；decode 4,608/5,184/5,760 |
| last_row logits（4 行 × 50,257） | 4 | 804,112 | 1,072,160 | fwd0..fwd3 各一行 |
| 选定层第 0 层 prefill 主要张量 | 22 | 585,361 | 780,484 | `fwd0.block0.*` |
| **单案例 JSON 总计** | 148 | — | **2,689,942 B = 2.57 MiB** | 文件大小；低于 6 MiB 目标 |

注意：`TRACE_SCHEMA.md` §7 的「每前向 < 30 KB」只在原始 float32 口径成立（prefill 28,224 B）；prefill 的 base64 是 37,632 B（约 36.8 KiB），48 个张量合计 base64 58,368 B（D12）。单案例 2.57 MiB 也低于 `GPT2_SOURCE_GROUNDED_PLAN.md` §10.3 的 15 MiB 单 HTML 目标。

### 10.4 S1 验证项映射（建议）

| `TRACE_SCHEMA.md` §8 检查 | 本轮已有的证据 | S1 还需做 |
|---|---|---|
| 采集等价（逐位相等） | record 路径在 4 次前向上已核对（F33–F34） | hooks 版适配器重做一次；两次运行逐字节比较 |
| 参考实现 logits / 注意力 | 本轮 0.0（F28–F29） | 在导出进程内再跑一次并写入 `verification.json` |
| 生成 token 序列 | 本轮 greedy 相同（F30） | 采集/未采集两次生成比较 |
| 载荷往返（shape/nbytes/sha256） | 未做 | 必须做（base64 解码 + sha256） |
| 公式重算（LN/残差/MLP/输出头） | 未做 | 用导出张量在 float64 重算 |
| softmax 行和 / 掩码严格 0 | 本轮行和与上三角已核（F31） | 在载荷上重算 |
| 缓存字节/区间 | 本轮容量与有效区已核（F18、F21） | 与导出元数据逐项比对 |
| 输出头（`ln_f.out × wte`） | 本轮权重共享已核（F05–F07） | 重算最后一行 logits |

注：S1 导出已完成，`data/gpt2/manifest.json` 的 `export_checks` 记录了结构、载荷往返（148 个张量）、缓存字节、采集等价（`capture_outputs_bitwise_equal=[0.0,0.0,0.0,0.0]`）等导出器自检；但 `instances[0].validation_status` 仍为 `not_run`。上表「S1 还需做」指**独立核对**，不因导出器自检而免除。

### 10.5 未决问题（S1 前需要确认或验证）

| ID | 问题 | 现状 | 建议 |
|---|---|---|---|
| I01 | 参考实现版本差异是否影响结论 | 当前 transformers 5.21.0 + eager 下逐位相等；历史版本未记录（U04） | S1 在 `manifest.provenance` 记录完整版本，并在同进程复跑 parity |
| I02 | hooks 采集是否会改变输出 | 本轮只验证了 `record` 路径（F33–F34） | S1 必须按 schema §8 做“采集 vs 未采集逐位相等” |
| I03 | 注意力 `record` 路径是否影响输出 | **已核对：不影响**（logits 逐位相等、greedy 相同） | 无需再议；但 S1 仍要记录该检查 |
| I04 | 体积预算 | 注意力 + logits 行理论值 ~347 KB（base64）；选定层张量未定（U11） | S1 实测单案例 JSON，超过 6 MiB 时按 schema §7 报告组成 |
| I05 | 主实例是否替换 | 本轮建议保留（§7.2），greedy 续写是标点 | 用户/Lead 决定；替换需重跑全部检查 |
| I06 | 逐字节可复现 | 未验证（U10） | S1 同环境重跑两次比较载荷 sha256 |
| I07 | 掩码分数如何表达 | schema 已规定「有限分数 + 掩码」，材料只暴露 softmax 后权重 | S1 用受限观察包装取 scores，或只导出权重并在 `unavailable_reason` 说明 |
| I08 | 选定层第 0 层要导出哪些中间张量 | schema §6.3 有清单，材料未实现采集 | S1 按清单实现并逐项校验 |
| I09 | 权重身份是否足以过 S0 关口 | 本地 sha256 与脚本常量、HF 下载元数据一致（已核对）；未联网（U01） | Lead/用户决定是否接受本地指纹作为身份证据 |
| I10 | 历史实验数字能否进课程 | 全部历史记录未复跑 | 若 S6 要引用，先在 S6 条件下重跑并记录新条件 |

## 11. 状态统计

- **已核对**：F01–F31、F33–F38（共 37 条事实，含本轮实测与字节/代码级证据）。
- **历史记录未复跑**：F32 与 R01–R17（共 18 条）。
- **未验证**：F39、F40 与 U01–U11（共 13 条），另加 §2.5 的 `README.md`/`OUTPUT.md` 内容。其中 U10、U11 已由 S1 提供证据（同环境两次导出 sha256 相同；单案例 2,689,942 B），U08 有 S1 导出器自检但独立核对器未完成；S0 未独立重跑这些检查，故仍列在“未验证项”，不改 S0 事实结论。
- **材料差异/风险**：D01–D13（13 条），其中 D01、D03、D05、D12 会影响教学或 S1 表述口径。

## 附录 A 复现命令

统一前缀：`cd /Users/yunni/Projects/temp/cs336-prep-code`，每条命令都加
`PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`。

### A.1 环境

```bash
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python - <<'EOF'
import platform, sys, torch, transformers, safetensors, numpy, tokenizers
print(sys.version.split()[0], sys.executable)
print(platform.platform())
print("torch", torch.__version__, "threads", torch.get_num_threads(), torch.get_num_interop_threads())
print("transformers", transformers.__version__, "safetensors", safetensors.__version__)
print("numpy", numpy.__version__, "tokenizers", tokenizers.__version__)
EOF
sw_vers; system_profiler SPHardwareDataType | grep -E 'Chip|Cores|Memory'
```

### A.2 材料哈希

```bash
cd real_llm_lab
for f in gpt2_real.py exp1_parity.py exp2_heads.py exp3_kv_cache.py exp4_mask_ppl.py \
         exp5_sampling.py download_weights.py requirements.txt run_all.sh \
         runs/*.txt figs/*.png \
         weights/gpt2/config.json weights/gpt2/model.safetensors \
         weights/gpt2/tokenizer.json weights/gpt2/vocab.json weights/gpt2/merges.txt; do
  printf '%s\t%s\t%s\t%s\n' "$f" "$(stat -f %z "$f")" \
    "$(stat -f '%Sm' -t '%Y-%m-%dT%H:%M:%S%z' "$f")" "$(shasum -a 256 "$f" | cut -d' ' -f1)"
done
```

### A.3 safetensors 文件头

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python - <<'EOF'
import struct, json, math
p = "real_llm_lab/weights/gpt2/model.safetensors"
with open(p, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n))
keys = [k for k in hdr if k != "__metadata__"]
bufs = [k for k in keys if k.split(".")[-2:] == ["attn", "bias"] or k.endswith("attn.masked_bias")]
params = [k for k in keys if k not in bufs]
print("keys", len(keys), "params", len(params), "buffers", len(bufs))
print("has_lm_head", "lm_head.weight" in hdr)
print("param_numel", sum(math.prod(hdr[k]["shape"]) for k in params))
print("data_bytes", sum(hdr[k]["data_offsets"][1]-hdr[k]["data_offsets"][0] for k in keys))
EOF
# 期望输出：keys 160 params 148 buffers 12 has_lm_head False param_numel 124439808
```

### A.4 KV cache 算术

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -c "
n_layer,B,n_head,T,D=12,1,12,1024,64
b=2*n_layer*B*n_head*T*D*4
print(b, b/1024**2, b/1e6, 2*n_layer*B*T*(n_head*D)*4)"
# 75497472 72.0 75.497472 75497472
```

### A.5 分词器字节映射

`output/s0-audit/audit_code_facts.py` 内联了 OpenAI 的 `bytes_to_unicode`（transformers 5.21 不再导出该函数），逐 token 还原原始字节并检查字节区间连续覆盖。

### A.6 载荷体积算术

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -c "
import math
fwd=[(12,12,7,7),(12,12,1,8),(12,12,1,9),(12,12,1,10)]
raw=sum(L*H*a*b*4 for L,H,a,b in fwd)
b64=sum(4*math.ceil(L*H*a*b*4/3) for L,H,a,b in fwd)
print('attn raw',raw,'b64',b64)
row=50257*4
print('logits row raw',row,'b64',4*math.ceil(row/3))
print('4 rows raw',4*row,'b64',4*4*math.ceil(row/3))"
# attn raw 43776 b64 58368
# logits row raw 201028 b64 268040
# 4 rows raw 804112 b64 1072160
```

### A.7 `__pycache__` 检查

```bash
find /Users/yunni/Projects/temp/cs336-prep-code -name __pycache__ | sort   # 运行前
find /Users/yunni/Projects/temp/cs336-prep-code/real_llm_lab \
  \( -name '__pycache__' -o -name '*.pyc' \)                              # 运行后应为空
```

## 附录 B 输出文件索引（`output/s0-audit/`）

| 文件 | 内容 |
|---|---|
| `audit_code_facts.py` / `code_facts.json` | §4.1–4.6 的全部实测值 |
| `audit_parity.py` / `parity.json` | §4.7 parity 实测值 |
| `audit_collect.py` / `collect_probe.json` | §4.8 采集相关探针 |
| `audit_record_path.py` / `record_path.json` | §4.8 record 路径不影响输出 |
| `materials_sha256.tsv` | §2 全部哈希 |
| `pycache_baseline.txt` / `pycache_after.txt` | §1.1 防污染证据 |
| `code_facts_stdout.txt` / `parity_stdout.txt` | 两个脚本的完整 stdout |

---

**审计声明**：本文只登记事实与差异，未修改材料目录、产品代码、文档、计划或备份。所有“已核对”均可由附录 A 的命令在只读条件下复现；所有“历史记录未复跑”“未验证”均未用推测替代实测。
