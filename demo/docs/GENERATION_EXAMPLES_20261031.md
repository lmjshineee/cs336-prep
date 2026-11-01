# 2026-10-31 生成主例与温度示例

已批准范围：更换两个主例，修正终止/导出逻辑，结束时移除待预测占位，同步课程与离线文件；随后追加批准真实 logits 的温度调节和单步采样。

## 当前主例

- GPT-2：`The cat sat on the` → ` floor, and the cat was still asleep.`。5 个输入 token，9 次选词，1 次 prefill + 8 次 decode；145 个回放步骤。
- SmolLM2：`To make a cup of tea, first` → ` you need to boil the water.`。8 个输入 token，7 次选词，1 次 prefill + 6 次 decode；239 个回放步骤。

停止原因均为 `max_new_tokens`，并非声称模型触发 EOS。候选分别从每模型 8 组真实 greedy 试跑中选出；原先两个输入延长到 12 token 时，自写缓存实现与 HF 每步完整前缀重算的 token 也完全一致，所以本次证据不支持将原例质量差归因于这两次运行中的前向计算错误。

## 改动

项目内生成函数达到上限后直接返回，导出不再执行无用的末尾前向；末尾选出的 token 无需进入 KV，除非继续预测下一 token。时间轴最后一步为选词，结束没有 `?`，倒退恢复当时状态。完整文本与 token 位置分开呈现，并区分输入/生成部分。

底部“温度与采样”在完整真实 logits 上计算 softmax(logits/T)，显示 T=1/当前温度、熵与 Top-8 以外质量；支持逐轮查看和固定种子重复抽样。抽样也覆盖完整词表；全部 token 解码标签与原始分词符号经过独立 tokenizer 核对。它不修改主轨迹，不接用不同 token 后的旧轨迹，没有 top-k/top-p 截断。正温度不改变 greedy 的排名。

同步了课程中的输入、位置、形状、概率和缓存数字。课程/步骤 ID、记录键与修订号保留；progress.js、records.js、原始案例、Llama 模型核心与备份逐字节一致。外部课程代码与权重保持只读，来源哈希分别记录；项目派生代码仅调整生成终止条件及对应函数说明。

## 验证

- GPT-2 独立核对：879/879，包含 HF 参考、全部生成序列、缓存/采集等价、逐层注意力和完整词表标签。
- SmolLM2 独立核对：617/617，包含局部重算、HF 完整前缀 logits/注意力/生成，以及项目生成函数实际前向次数。
- Node 自动测试：123/123。
- 结束/倒退、温度/种子、HTTP/单文件/file://、1440/1200 宽度及学习记录不变：89/89，见 [浏览器报告](../output/playwright/generation-20261031/verification.json)。
- 原有九课与真实张量/源码浏览器检查：60/60，无页面错误。见 [课程报告](../output/qa/llama-generation-20261031/ui-checks.json)。课程操作仅发生在隔离浏览器中，不代表用户实际学完课程。
- `npm run build` 已重建 [GPT-2](../GPT2-LLaMA-Atlas/atlas.html) 与 [SmolLM2](../GPT2-LLaMA-Atlas/llama-atlas.html)，离线不发网络请求。

旧例、页面、源码、清单和核对报告保存在 [备份](../backup/20261031-generation-examples-before/GPT2-LLaMA-Atlas/README.md)。
