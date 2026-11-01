# LLaMA L1—L7 交付交接（SmolLM2-135M）

> 日期：2026-10-31 · 范围：L1—L5、L7；L6 可选局部对照未实施；S8 按范围排除。
> 本交接只描述已实际运行/核对的内容。未运行的验证不写成通过。

## 1. 最终状态

| 批次 | 状态 | 结果 |
|---|---|---|
| L0 来源冻结/格式 | ✅ | 修正后源哈希、v2 契约、九课/40 步 ID；见 L0 交接 |
| L1 主实例导出 | ✅ | 真实轨迹 + 独立核对 355/355；源快照与 source-map v2 |
| L2 读取/回放 | ✅ | 独立 v2 reader/replay；136 步；缺失/损坏明确失败 |
| L3 独立模型页面 | ✅ | `llama.html` / `llama-atlas.html` 默认 SmolLM2，真实尺寸/节点/标题/输入一致 |
| L4a RoPE/GQA | ✅ | 真实旋转、GQA 映射、KV 前后 shape、mask 可核对 |
| L4b RMSNorm/SwiGLU | ✅ | 真实数值、切片边界、gamma、门控、残差与源码行区间 |
| L5 课程/记录 | ✅ | 九课 40 步、独立 `llama.progress.v1`、答题/恢复/导入导出 |
| L6 局部对照（可选） | ⏭ | 未实施；主学习路径已稳定，可按需追加 |
| L7 交付 | ✅ | HTTP 源码/单文件、`file://`、复制到独立目录、无外部请求 |

**结论**：SmolLM2 的 L1—L5、L7 已完成并通过数据/页面/离线验收；L6 是可选项，未实施。GPT-2 数据与页面未回退；共享代码改动后 GPT-2 单文件已重建并回归。

## 2. 数据身份与载荷

- 模型：`smollm2-135m`，HuggingFaceTB/SmolLM2-135M（Llama 架构，**不是 Meta LLaMA-2/3**）。
- 配置：30 层、d=576、9 Q 头 / 3 KV 头、head_dim=64、FFN=1536、词表=49,152、RMSNorm eps=1e-5、RoPE θ=100,000、tied embeddings、配置最大位置 8192；独立参数 **134,515,008**。
- 权重：本地 `model.safetensors`，272 个 BF16 张量；SHA-256 `80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1`；运行 CPU fp32，存储 dtype 与运行 dtype 分开记录。
- 上游 revision 未固定：manifest `revision:null`，以本地 SHA-256 为载荷身份。
- 案例：`smollm2-main-001`，输入 `The capital of France is` → ids `[504,3575,282,4649,314]`；1 次 prefill + 3 次 decode；greedy 生成 `[260,3575,282]`（` the / capital / of`）；第 4 次 decode 的 logits 未被采样使用。
- 轨迹：**135 事件 / 259 张量**；30 层×9 Q 头注意力权重；仅第 0 层完整中间张量；只导出 3 个被采样最后行 logits；缓存零区不导出。
- 案例 SHA-256：`1b2978216eb3da9d3e2ea827bb43e40808588b61148de2ccbeea9770fa99b6f5`。
- Manifest SHA-256：`297a5fafb7e76dafd744dfa4be545fef2cbb824bebaf5b30eb7c1dbb659c2b09`。
- Verification SHA-256：`a76b877485e2541869013a43a4fa7284ce207b04e37df55fefa2efb63d6584ec`。

## 3. 独立核对（L1）

命令（venv Python；材料只读）：

```bash
cd GPT2-LLaMA-Atlas
PYTHONDONTWRITEBYTECODE=1 /Users/yunni/Projects/temp/cs336-prep-code/.venv/bin/python tools/trace/export_llama.py \
  --material-root /Users/yunni/Projects/temp/cs336-prep-code --data-dir data/llama --case-id smollm2-main-001
PYTHONDONTWRITEBYTECODE=1 /Users/yunni/Projects/temp/cs336-prep-code/.venv/bin/python tools/trace/verify_llama.py \
  --material-root /Users/yunni/Projects/temp/cs336-prep-code --data-dir data/llama --case-id smollm2-main-001
```

`verification.json`：**355/355 通过，100 项 required，0 required failed**。包括：

- 全部载荷重新 base64 解码、SHA-256、形状、有限值；事件/张量引用；
- 从载荷重算 RMSNorm（mean(x²)、invRMS、gamma）、RoPE half-split（Q/K）、GQA repeat、mask、scores、softmax、SiLU、逐元素门控、残差；
- 对照本地 HF `LlamaForCausalLM`（eager、CPU fp32）的 3 个被采样 logits 行、greedy 生成、30 层注意力（最后查询行）；
- 缓存 K 为旋转后、V 未旋转；容量 11,796,480 B；fwd3 不导出 last_row；
- 源快照哈希与每个行区间哈希；`wrapper_equivalence_max_abs=0`。

失败时 verifier 返回非 0 且不写 verification；没有模拟 fallback。

## 4. 数据与源码布局

- `data/llama/manifest.json`：`atlas.trace/2`，实例、来源、运行时、权重/配置/tokenizer 哈希。
- `data/llama/smollm2-main-001.json`：`atlas.trace.case/2`，135 事件、259 张量、forward summaries。
- `data/llama/verification.json`：`atlas.trace.verification/2`。
- `sources/llama/step06_model.py`（24,660 B，`33fab3ee…`）、`step09_real_llama.py`（14,592 B，`5af8867d…`）：L0 修正后快照，只读。
- `sources/llama/source-map.json`：`atlas.source-map/2`，20 符号 / 25 操作；哈希 `515e63fa…`。
- `js/llama-replay-data.js`：独立 v2 reader；`js/llama-replay.js`：136 步回放；`js/llama-local.js` + `js/llama-local-viewer.js`：局部计算；`js/llama-source-data.js`：多文件源码/行区间校验；`js/llama-lessons.js`：九课 40 步。
- GPT-2 v1 数据/读取器未改；GPT-2 页面重建后仍加载 64 步真实轨迹。

## 5. 页面与交互验收（L2—L7）

浏览器 QA 工具：`tools/qa/verify_llama_ui.mjs`；结果 **60/60，page_errors=0**，见 `output/qa/llama-l7/ui-checks.json`。

覆盖：

- HTTP `llama.html`：真实轨迹 ready、136 步、标题/模型名/节点数、旧 LLaMA-2 资料入口、来源弹窗身份；
- 局部查看器：源码已校验、注意力行和、RoPE 重算误差、RMSNorm/SwiGLU 重算、GQA 映射、第 4 层完整中间值明确不可用；
- 九课 40 步全部可定位；每课答题通过；`llama.progress.v1`/`llama-r1`；刷新恢复；GPT-2/旧键未被触碰；
- HTTP 单文件 `llama-atlas.html`；`file://` 单文件；复制到 `output/qa/llama-l7/` 后打开；无外部网络请求；GPT-2 单文件仍 64 步 ready。

Node 测试：`npm test` **111/111 通过**（含 5 项 LLaMA reader/replay、5 项 LLaMA local/source、2 项 LLaMA progress/records）。日志 `output/qa/llama-l7/npm-test.log`。

## 6. 产物哈希（最终）

| 文件 | 字节 | SHA-256 |
|---|---:|---|
| `llama-atlas.html` | 3,053,390 | `5cb1e3a3302605ae76a34c8a167b06c70272302b350cf89e719ae7989b94b0d9` |
| `atlas.html` | 4,037,702 | `506234e7ab6f2a2de44ed8ac54455154020b0310ad68d5a32c1e2b5411b72ad7` |
| `llama.html` | 13,078 | `8570cae3da43d38bdec8ee4494dd3d30d60a0a7405313c578fd6e5d1f12a8bf7` |
| `js/main.js` | 96,796 | `ac6a044ca678b04861022459febe24cb36c7e3a8f7339ca2120f37e7584067e7` |
| `js/spec.js` | 34,702 | `83e23f65119e9ef395e5d8e946d59ae3e8cb181ff0198c9a33b2dea78953505a` |
| `js/llama-lessons.js` | 15,661 | `5b27842c2b2f89c4875a77e1f598d32532c3328ebb2dddfe5394a1e8dcd4e536` |
| `js/llama-local-viewer.js` | 18,346 | `2841a09be2e16661899ce3624443583f3abb319cda799a50da05889e630bab05` |
| `README.md` | 18,001 | `782ca2069ed0e12578cd2b9088bb71aae9da4cf4cb7db5c9cd2437798712e625` |

## 7. 已知边界/未做

- **L6 未实施**（方案标为可选）：没有按问题打开的局部对照组件；默认仍不并排双塔。
- 旧 LLaMA-2 7B 只通过顶栏“旧 LLaMA-2 资料”弹窗查看配置摘要；完整旧页面在 `backup/20261030T214943-S7-final/` 的项目备份中，不在当前单文件内。当前页不把旧 32 层/6.7B 数据混入 SmolLM2。
- 局部查看器的完整中间张量只在第 0 层；其他层仅真实注意力权重；面板明确显示不可用，不补假值。
- 页面/课程不训练、不实时推理、不加载权重；单文件只内联已导出数据。
- 上游 revision 未固定；不要把本地哈希当成上游 commit。
- 长上下文/bf16/GPU 未验证；页面演示缓存容量 256，不宣称 8192 长度质量。
- 页面 KV cache 标签沿用共享 `formatBytes`（按 1024 换算但显示 “MB”），数值为 11.25 MiB；这是显示单位措辞，不是数据错误。

## 8. 备份

- L0 最终快照：`backup/20261030T231405-L0-llama-final/`。
- L1—L7 交付快照：`backup/20261031T094800-L1-L7-llama/`（数据、源码、工具、页面、证据、文档；不含只读材料/权重）。
- 旧 LLaMA-2 页面基线：`backup/20261030T214943-S7-final/GPT2-LLaMA-Atlas/`。
