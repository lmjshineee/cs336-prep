# 2026-10-31 审查问题修正交接

- 范围：用户授权直接修复独立审查 R1—R7；不实施可选 S6/L6，不改真实轨迹或学习记录格式。
- 结论：7 项已修正，115 项 Node 检查和 22 项隔离浏览器检查通过，0 pageerror。两个单 HTML 已重建。
- 原始发现：[独立审查报告](./INDEPENDENT_REVIEW_20261031.md)。

## 修正与验证

| 项目 | 修改 | 验证 |
|---|---|---|
| R1 模型介绍串用 | 根节点说明从各自层数、归一化和 FFN 配置生成；GPT-2 恢复为 12 层 Transformer / LayerNorm / GELU | 三模型根节点文案回归；浏览器 GPT-2 默认说明核对 |
| R2 RoPE 错配 KV | `floor(head/(H/Hkv))` 代替截断索引 | 全部 9 个 Q 头与真实 K 投影切片核对，映射为 000111222；浏览器切头 |
| R3 课程数据缺口 | `rope.s4` 指向已完整采集的 decode L0；补齐 LLaMA 模板中已有处理逻辑所需的课内“真实张量”按钮 | 所有 viewer 课内步骤均读取实际可用张量；浏览器实际走到第 4 步并通过课内按钮打开 |
| R4 键盘与焦点 | 打开后聚焦内部关闭按钮、背景 inert、Tab 边界、关闭恢复触发控件；重绘选择框时保留焦点 | 从顶栏 Enter 打开、正反 Tab 循环、Escape 关闭并回到顶栏；课内打开后回到课内按钮 |
| R5 来源说明 | 按实际 traceMode 和当前模型生成来源说明，分别说明真实回放、手算示例与配置计算；单来源与总来源共用 | GPT-2 ready 后来源文案不再称 token/概率为模拟 |
| R6 位置选择 | RMSNorm 位置数来自本次前向张量，选项与数值明确显示绝对位置 | prefill 为 0—4，首轮 decode 仅有绝对位置 5；第三轮为 7；浏览器仅给一个有效选项 |
| R7 选中层跳转 | 层节点匹配覆盖新增 smollm2 | 浏览器选择 smollm2.L4 后显示本轮层跳转 |

R3 首次报告将查看器描述为自动打开，复核当前实现后确认实际由课内按钮触发。此次没有新增自动弹出行为；修正的是课程事件映射和模板遗漏，使原设计的查看操作可以完成。

## 文件范围

既有源文件：

- `GPT2-LLaMA-Atlas/js/spec.js`
- `GPT2-LLaMA-Atlas/js/llama-local.js`
- `GPT2-LLaMA-Atlas/js/llama-lessons.js`
- `GPT2-LLaMA-Atlas/js/llama-local-viewer.js`
- `GPT2-LLaMA-Atlas/js/main.js`
- `GPT2-LLaMA-Atlas/llama.html`

新增测试：`GPT2-LLaMA-Atlas/tests/review-20261031.test.mjs`。生成文件：`atlas.html`、`llama-atlas.html`。备份先保存前三个运行文件及生成物，再保存另外两个运行文件和 LLaMA 模板，未扩大到数据或模型实现。

## 验证证据

- `npm test`：115/115 通过。新增四项回归在修复前均失败，修复后通过，覆盖真实数值与课程路径而非仅匹配实现文本。
- `npm run build`：通过，两入口分别内联自己的轨迹和源码。
- `output/playwright/fix-20261031/fix20261031final.json`：22/22，0 pageerror；源码 HTTP、两个单 HTML HTTP、复制到独立目录后的两个 `file://` 冷启动均 ready。
- 浏览器另覆盖 1440×900、1280×720、390×844 查看器边界、源代码校验、RoPE 映射、decode 位置和课内查看操作。
- `output/playwright/fix-20261031/summary.json`：修改范围与 SHA-256；14 个已有数据/源码快照/记录模块文件保持原哈希。
- 真实数据未改，因此沿用同日刚完成的 GPT-2 496/496、SmolLM2 355/355 独立数值核对；本次未重新训练、推理导出或改写正式 verification。
- 测试使用隔离浏览器存储，没有操作用户正式记录。检查结束浏览器与本机临时服务器均已关闭。

第一轮浏览器复验因发现课内按钮实际缺失而停止；补齐后完整重跑。最终结论以 `fix20261031final.json` 为准，不把未完成的首轮算作通过。

## 当前单文件

| 文件 | 字节 | SHA-256 |
|---|---:|---|
| `GPT2-LLaMA-Atlas/atlas.html` | 4,039,431 | `494c1be6ba7730ef717791eb69a5c12f819ebe155c3d322368760f3fb6510d15` |
| `GPT2-LLaMA-Atlas/llama-atlas.html` | 3,055,194 | `5048a8616cf39cc2275875da24abe0692d8e95aaee401316ffc0aa8dfa18067a` |

旧 S7/L7 交接里的产物哈希作为当时记录保留，当前产物以本表为准。

## 备份和边界

修改前备份：`backup/20261031-review-fixes-before/`，只含本批 8 个被覆盖文件与摘要。按对应路径恢复源文件后重新构建可回退；新增回归测试应保留作问题记录。

此次结论限于修正的 7 项及相关回归；不宣称所有浏览器、真实触摸设备、屏幕阅读器、GPU 温度功耗或实际学习效果已验证。先前已记录的 MiB/MB 标签差异及可选扩展未包含在本批。
