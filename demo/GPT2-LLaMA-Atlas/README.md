# GPT-2 × Llama 架构 · 解码器学习图谱

一个可在浏览器中实时交互的三维架构学习页，面向第一次系统学习 Transformer 的人。

- **GPT-2 页面**（`index.html` / `atlas.html`）：GPT-2 Small（2019，12 层，124M），真实轨迹回放、九课与自由探索。
- **Llama 页面**（`llama.html` / `llama-atlas.html`）：HuggingFaceTB **SmolLM2-135M**（Llama 架构，30 层，134,515,008 参数），真实轨迹、九课、RoPE/GQA/RMSNorm/SwiGLU 局部计算与源码。
- 每个页面只创建并显示自身模型；LLaMA-2 7B 旧模拟资料只通过独立“旧 LLaMA-2 资料”入口打开，不混入 SmolLM2 数据、节点、缓存或课程。
- 每个模块都能看到**直觉、公式、张量形状、参数量**；两个页面各有九节真实轨迹课程。

设计文档见 [DESIGN.md（已归档）](../docs/README.md#历史归档)。

---

## 打开方式

### 双击打开（推荐）

**双击 `atlas.html`**（GPT-2）或 **`llama-atlas.html`**（SmolLM2）即可在浏览器里运行，不需要服务器、不需要联网。

它们是 `index.html` / `llama.html` 的单文件版本：CSS、全部 JS 与 Three.js 都已内联；两个文件还分别内联了已验证轨迹、源码快照、行映射与核对报告（约 7.4 MiB / 6.3 MiB）。两者不含外部引用、不使用 ES modules，所以 `file://` 不会被浏览器拦截；双击即可离线跑真实轨迹与课程。

### 开发模式

改代码时仍走原来的多文件结构。目录内已内置 Three.js 与所需 addon，不需要构建、不需要联网。

```bash
cd GPT2-LLaMA-Atlas
python3 -m http.server 8080
```

打开 <http://localhost:8080/index.html>（GPT-2）或 <http://localhost:8080/llama.html>（SmolLM2）。

> **真实轨迹**：开发模式分别通过 `data/gpt2/`、`sources/gpt2/` 与 `data/llama/`、`sources/llama/` 读取；单文件版已内联同样内容，浏览器会重新核对案例 SHA-256 与源码行区间哈希。开发模式的两个模板仍需要 HTTP 服务器。

> 源文件使用 ES modules，直接双击 `index.html` 会被 `file://` 拦截。改完源码后先跑参数检查，再重新生成单文件版：

```bash
npm install
npm test
npm run build
```

`npm run build` 会同时生成 `atlas.html` 与 `llama-atlas.html`，并校验内联案例/源码哈希。`npm test` 使用 Node 自带的 `node:test`（123 项），检查参数账本、形状语义、KV cache、来源标注、两套学习记录迁移/导入导出、真实轨迹解码/时间轴、LLaMA 局部重算与源码行区间、手算示例与按需渲染调度；不引入测试框架。

> 单文件版是构建产物，不要直接编辑 `atlas.html`。

推荐桌面 Chrome / Edge / Safari，窗口宽度 ≥ 1200px，开启 WebGL 2。

---

## 页面入口与学习记录（S2a–S2b）

**GPT-2 页面**首次进入会看到欢迎层：

1. **3 分钟看懂**：GPT-2 单塔概览 + 第一节导览。
2. **GPT-2 真实轨迹课**：9 节单模型课，从真实 BPE、L0 注意力、第 0 层张量一路走到 logits、greedy 与 KV cache；课文可打开真实显微镜/张量查看器。
3. **自由探索**：直接进入 GPT-2 结构。

**Llama 页面**（SmolLM2-135M）提供独立的九课真实轨迹课：整体回路、输入/无位置表、RMSNorm、RoPE、GQA、SwiGLU、输出/参数、生成/KV cache、总结。课程步骤 ID 与记录命名空间独立，不与 GPT-2 混用。旧 LLaMA-2 7B 结构只通过顶栏“旧 LLaMA-2 资料”弹窗查阅，不是当前实例。

学习记录（S2b）：
- 旧键 `gpt2llama.lessons` / `gpt2llama.progress.v1` **只读保留**，不迁移为新的“已学习/答题通过”。
- 新记录写入 `gpt2.progress.v2`（GPT-2 页，修订 `gpt2-r2`）或 `llama.progress.v1`（SmolLM2 页，修订 `llama-r1`），包含稳定课内步骤 ID，刷新或重开课程会恢复上次课程与步骤。旧 `llama2.progress.v2` 命名空间仍只作旧资料，不与 SmolLM2 混用。
- 当前 GPT-2 课程修订号为 `gpt2-r2`（S5 改写），SmolLM2 为 `llama-r1`。检测到旧 `gpt2-r1` 的 v2 记录时，GPT-2 页面会先把它逐字节备份到 `gpt2.progress.v2.superseded.gpt2-r1`，再从空白的新修订开始，不继承旧答题通过；SmolLM2 记录写入自己的键，不触碰 GPT-2/旧键。
- 顶栏「记录」可查看旧记录原文、导出当前记录、导入经校验的记录，并自动把导入前的记录备份到 `<键>.import-backup`；未知课程/步骤与外部格式会被拒绝。
- 旧完成记录不会让新课程显示为完成；存储被禁用或记录损坏时页面仍能打开并提示，不会因此丢课。

---

## 真实轨迹回放（S3）

GPT-2 页面读取 `data/gpt2/gpt2-main-002.json`：

- 输入 `The cat sat on the`（5 token）；真实 greedy 续写 ` floor, and the cat was still asleep.`（9 token），完整句子为 `The cat sat on the floor, and the cat was still asleep.`。
- 1 次 prefill + 8 次 decode，每次前向都用于选词；144 个事件、333 个张量，加分词步后共 145 步。最后一步为选词，达到 `max_new_tokens=9` 后立即结束，没有未使用的额外前向。
- 完整句子区分输入与生成内容；token 行保留位置编号，悬停显示 token id；回放结束不再显示待预测的 `?`。倒退会恢复对应步骤的序列与缓存。
- 输入、注意力、完整最后行 logits、候选概率、缓存与来源哈希均为真实载荷；缺失或损坏时拒绝加载，不回退为模拟数据。
- 输出区“温度与采样”支持逐轮查看真实 logits，在完整词表上计算 `softmax(logits/T)`，对照 T=1、熵及其余概率质量；可调温度并用固定种子单步抽样。正温度不改变 greedy 排名；抽样不修改原回放，也不接用不同 token 之后的旧轨迹。
- 旧 `gpt2-main-001`、旧页面、来源快照和报告保存在 `../backup/20261031-generation-examples-before/`；当前案例使用新 ID。

## 真实注意力、张量与源码（S4）

真实轨迹模式下可打开两类只读视图：

- **显微镜 → 注意力**：12 层 × 12 头的真实 softmax 权重，按当前前向取事件；可切换层/头/查询行，行和=1、未来位置严格为 0。其他显微面板（归一化/激活/采样等）在真实模式显示“尚未接入”，不展示模拟值。
- **真实张量**（顶栏）：第 0 层完整 Block 的 22 个中间张量与 4 个输入/输出张量；支持头、行、列切片、形状/统计/生产操作/载荷哈希。第 1–11 层只导出注意力，21 个未采集张量会列出并说明 S1 采集预算，不伪造中间值。
- **源码面板**：读取 `sources/gpt2/gpt2_real.py` 与 `source-map.json`，用浏览器 SHA-256 校验整文件与每个行区间；校验失败会禁用行号映射。当前操作行高亮，可切换到该步相关的其他操作。
- 视图全部只读：切片、缩放、点选都不改变真实轨迹、缓存或当前步骤。

## SmolLM2 真实轨迹、局部计算与源码（L1–L7）

Llama 页面读取 `data/llama/smollm2-main-002.json`：

- 输入 `To make a cup of tea, first`（8 token）；真实 greedy 续写 ` you need to boil the water.`（7 token），完整句子为 `To make a cup of tea, first you need to boil the water.`。
- 1 次 prefill + 6 次 decode；238 个事件、455 个张量，加分词步后共 239 步。达到 `max_new_tokens=7` 后在最后一次选词处结束。
- 全部 30 层、9 个 Q 头的注意力权重；第 0 层完整 RMSNorm/RoPE/GQA/SwiGLU/残差中间张量；7 个被选词使用的完整最后行 logits。缓存不导出零区。
- KV 按 Hkv=3、容量 256、CPU fp32 预分配 11,796,480 B；有效字节按实际写入位置计算。最后选出的句号尚未进入缓存，除非继续生成。
- “温度与采样”与 GPT-2 使用同一完整词表计算方式，词表标签由本地 tokenizer 导出并独立核对；原主轨迹仍为 greedy。
- 独立核对器重算局部操作，对照 HF 完整前缀 logits、全部层注意力及 greedy 序列；另外直接运行项目生成函数并检查前向次数。失败不会写通过报告。
- 项目内 `sources/` 的生成函数仅修正末尾终止条件；外部课程材料和权重只读，原文件哈希另存于 manifest。源码行区间与哈希已重新生成。
- 旧例及来源保存在上述备份中。课程/步骤 ID、学习记录键与修订号保留，未写入用户正式学习记录。

## 主要功能

### 三维解剖
- 鼠标左键拖动旋转、滚轮缩放、右键拖动平移；`R` 恢复整体视角。
- 「拆解」滑块分离层；「透视」把层壳变半透明；选中对象后可「隔离」。
- 点塔上的层或模块：选中、聚焦、右侧同步解释。
- 每个入口只构建自身模型；没有模型切换/对比开关。历史对照材料在「来源」弹窗中按需打开。

### 形状条
底部固定显示当前张量形状链：

```
ids → embed → norm → Q / 新 K/V → 读 K/V → scores → softmax → attn + add → MLP → +add → final norm → pick → logits → probs
```

Prefill 显示本轮全部 T 行的计算（scores 是 T×T）；decode 只显示 1 个新位置，同时标出注意力读取的 T 个 K/V（scores 是 1×T）。`pick` 表示从归一化输出中取最后一个位置。缓存条单独显示当前步骤已写入的 KV，不与本轮计算形状混用。

点击任意一格查看该步的说明；播放时整条链随步骤高亮。

### 参数账本
右侧「参数」页签只列出当前页面对应模型的：
注意力、MLP、Norm、每层合计、× 层数、嵌入、输出头、最终 Norm、总计。
点击任意一行可在本页塔上定位对应模块；两代账本的对照数字只保留在「来源」弹窗的历史材料中。

计算值（启动时会做断言）：

| 模型 | 计算总量 | 公布 |
|---|---|---|
| GPT-2 Small | 124,439,808 | 124M |
| SmolLM2-135M（Llama 架构） | 134,515,008 | 135M |
| LLaMA-2 7B（旧资料） | 6,738,415,616 | 6.7B |
| LLaMA-3 8B（对照） | 8,030,261,248 | 8.03B |

### 2D 显微镜
点「显微镜」或课程自动打开，五个面板：

- **注意力**：QKᵀ 热图、因果掩码、逐行 softmax；点行查看；切头；观察 “it → cat”。
- **位置**：GPT-2 可学习位置向量 vs LLaMA RoPE 旋转。
- **归一化**：LayerNorm / RMSNorm 前后条形图；拖动离群值看差异。
- **激活**：GELU / SiLU 曲线与 SwiGLU 门控。
- **采样**：温度、top-k、top-p 的实时过滤与采样。

### 可手算的注意力示例
顶栏「注意力示例」打开一个固定的小例子：3 个 token × 2 维，依次展示 Q、K、V、QKᵀ、÷√2、因果掩码、softmax 与加权输出。默认矩阵固定，可只改一个 Q 分量和一个 V 分量，观察下游数值如何变化；页脚实时核对行和、掩码处为 0、首行输出 = V₁。全部数值为教学指定，不是模型实测。

### 九课（GPT-2 真实轨迹，S5）
读图 → 真实 BPE 与位置 → 真实注意力 → 残差与 LayerNorm → MLP → 参数账本 → 生成 → KV cache → 同一实例总结。每节含定位镜头、课文与一道单选题；需要数值的步骤可打开真实显微镜或“真实张量”查看器。37 个课内步骤 ID 与 S2b 一致，课程修订号升至 `gpt2-r2`；旧 `gpt2-r1` 记录会先备份到 `gpt2.progress.v2.superseded.gpt2-r1`，旧的“已学习/答题通过”不会继承。

课程需要真实轨迹：`?trace=sim` 或真实数据失败时不会启动课程，并明确提示。自动化只证明流程与内容映射，**用户尚未试学**，学习体验仍待验收。

### 渲染与访问
- 按需渲染：静止、暂停且相机与拆解收敛后不保留 RAF 或计时器；播放时装饰动画上限 30fps；页面进入后台会暂停时间轴并取消绘制，返回后不追赶步骤。
- 系统「减少动态效果」会关闭相机阻尼、拆解过渡与装饰动画。
- WebGL 2 不可用时显示可操作的失败提示（“三维视图未能启动”与重新载入按钮），不会停在加载动画。
- 桌面验收覆盖 390×844、768×1024、1280×720、1440×900 四个视口；窄屏使用抽屉与键盘操作。

---

## 真实与模拟的边界

来源弹窗把每条材料标成 **论文原文 / 官方配置 / 本站计算 / 教学模拟** 四类，并提供原文或官方仓库链接。参数分项与总量是本站按公式计算，不是论文表格里的原始数字。

**事实（来自论文 / 官方配置）**
- 层数、`d_model`、头数、KV 头数、头维、FFN 维、激活、归一化、位置编码、bias、权重共享、词表、上下文。
- 参数量（整数算式）与公布总量。
- RoPE / LayerNorm / RMSNorm / GELU / SwiGLU / softmax 的公式。
- 注意力缩放、因果掩码、prefill / decode 与 KV cache 的基本机制。

**真实运行回放（S3，GPT-2 主实例）**
- 输入 token/id/字节、每步形状、缓存有效长度与容量、最后一行 logits、Top-N 概率与剩余质量、生成 token 序列。
- 来源：本地 `gpt2_real.py` + 官方权重的一次 CPU fp32 运行；离线核对报告与案例哈希随页面加载。

**教学模拟（界面以粉色「模拟」标注）**
- 演示句子的词级切分与 token id（真实分词器未内置）。
- 注意力权重、热图数值、top-k / top-p 与采样结果。
- 下一 token 的概率、排序与选择。
- 三维模块的位置、丝线数量与板宽缩放是教学表达，不代表物理布局。

**不做的事**：不加载模型权重、不做真实推理、不实现训练/MoE/MLA/FlashAttention。

---

## 文件

```
GPT2-LLaMA-Atlas/
├── atlas.html              GPT-2 单文件版（双击即开，构建产物）
├── index.html              GPT-2 页面结构（默认入口，window.ATLAS_ENTRY）
├── llama.html              LLaMA-2 独立页面（结构查阅与自由探索）
├── llama-atlas.html        LLaMA-2 单文件版（构建产物）
├── css/style.css           视觉系统
├── js/spec.js              两个模型配置、节点目录、来源、演示材料、参数公式
├── js/trace.js             时间轴、形状推导、模拟注意力、KV cache
├── js/scene.js             按入口构建的单塔 3D 场景、层/模块、拾取、数据流、相机
├── js/micro.js             2D 显微镜五个面板
├── js/attention-example.js     手算示例的纯数学模块
├── js/attention-example-ui.js  手算示例的界面、控件与核对项
├── js/progress.js          学习记录 v2：旧记录只读、稳定步骤 ID、状态与恢复位置
├── js/records.js           记录导出/导入校验、导入前备份
├── js/replay-data.js       真实轨迹读取、载荷解码与完整性校验
├── js/replay.js            真实事件 → 回放时间轴（形状/缓存/概率）
├── js/real-attention.js    真实注意力取层/头/查询行
├── js/real-tensors.js      第 0 层张量目录、切片与统计
├── js/source-data.js       源码快照/行映射哈希校验
├── js/real-viewer.js       真实张量 + 源码查看器
├── js/render-loop.js       按需渲染调度（静止停止、播放 30fps 上限）
├── js/lessons.js           9 节课程与测验
├── js/main.js              目录、检视器、播放、课程、弹窗
├── tests/                  参数账本、形状、缓存、来源、学习记录、手算示例、渲染调度检查
├── package.json            测试/构建脚本，锁定 esbuild 版本
├── tools/build-standalone.mjs  生成单文件版；GPT-2 版内联轨迹/源码/核对报告
├── data/gpt2/              已验证真实轨迹（manifest、案例、核对报告）
├── sources/gpt2/           真实实现快照与源码映射
├── vendor/                 内置 Three.js r170 与 OrbitControls / Bloom
├── DESIGN.md               设计方案
└── README.md               本文件
```

页面本身无 npm 依赖、无运行时构建、无外部字体、无网络请求；`tools/build-standalone.mjs` 只在生成单文件版时用一次 esbuild。

---

## 已知限制

- 三维场景以桌面为主；窄屏会隐藏左右面板，但尚未提供完整的 2D 阅读模式。
- 真实实例的上下文是 7 个输入 token + 3 个生成 token；真实模型支持更长上下文，但本轨迹只导出一个短实例。
- GPT-2 页面不再提供双塔对照或 LLaMA 课程步骤；相关材料只留在“来源”弹窗。
- 真实轨迹的注意力矩阵、第 0 层张量切片与源码行映射已接入；第 1–11 层完整中间张量未采集，查看器会明确标注；LLaMA 页仍为教学模拟。
- 单文件 `atlas.html` 已内联轨迹/源码/核对报告，`file://` 冷启动可用；复制到任意目录仍可打开，不发网络请求。
- Top-N 中未在导出 token 列表内的词表项仍显示为 `#id`（浏览器没有完整词表）；课程只依赖已导出的真实 token，不伪造文本。
- 2D 显微镜在真实轨迹模式下显示真实注意力；归一化/激活/采样等未接入真实数据的标签页会明确显示“尚未接入”，不展示模拟值。
- 组装三维塔需要 WebGL 2；集成显卡如不流畅，可关闭透视或减少拆解程度。
- 验收使用 Chromium 与独立测试存储；未验证 Firefox、Safari、真实手机触摸和屏幕阅读器。窄屏 ≤560px 会隐藏顶栏 3D 工具组，不承诺完整手机体验。
- 性能只记录绘制次数与收敛行为；未测真实 GPU 功耗、温度或跨设备表现，不能从绘制次数推断降温幅度。
