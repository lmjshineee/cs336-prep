# S7 交接记录：单文件内联与完整离线验收

- 日期：2026-10-30
- 批次状态：**S7 通过；首版 GPT-2 交付范围（S0—S5、S7）完成。**
- 说明：S6 为可选项、未实施；S8 按计划排除。未发布、未部署。
- 最终快照：`backup/20261030T214943-S7-final/`（8.6 MB；源码、两个单文件、轨迹、源码快照、S0—S7 交接、验收 JSON）。
- 证据：`output/qa/s7-browser.json`（28 项）；S5 回归 280 项；`npm test` 99 项。

## 1. 目标与结果

| 计划目标 | 结果 |
|---|---|
| 单 HTML HTTP 冷启动 | 完成：`atlas.html` 经 HTTP 加载真实轨迹、课程、查看器 |
| 单 HTML `file://` 冷启动 | 完成：直接双击打开，无服务器；真实 64 步回放、课程、源码均可用 |
| 关闭外网仍可读数据/源码/运行详情 | 完成：离线上下文零 http(s) 请求；运行详情含完整案例 sha256 与 496/496 |
| 复制单 HTML 到临时目录 | 完成：复制到 `output/qa/s7-copy/` 后 `file://` 仍可用 |
| 数据/源码可追溯 | 完成：内联 manifest、案例原文、verification、source-map、源码快照；浏览器重新核对案例 SHA-256 与源码行区间 |
| 两入口独立打包 | 完成：`atlas.html` 3,966,765 B（含轨迹/源码/核对）；`llama-atlas.html` 879,440 B（不含 GPT-2 数据） |
| 无外部引用 | 完成：`script[src]`、`link[rel=stylesheet]` 为 0；单文件不发网络请求 |
| 记录限制说明准确 | 完成：README/S5 交接写明 r1→r2 备份键、旧 v1 只读、导入限制 |

## 2. 文件

| 文件 | 字节 | SHA-256（前 16 位） | 说明 |
|---|---:|---|---|
| `GPT2-LLaMA-Atlas/atlas.html` | 3,966,765 | `aa16d014145fb267` | GPT-2 单文件：内联轨迹/源码/核对报告 |
| `GPT2-LLaMA-Atlas/llama-atlas.html` | 879,440 | `a29562203826b7dc` | LLaMA-2 单文件：独立轻量教学模拟 |
| `GPT2-LLaMA-Atlas/tools/build-standalone.mjs` | 6,548 | `51e42d5796581ec3` | 构建时校验并内联轨迹/源码 |
| `GPT2-LLaMA-Atlas/js/replay-data.js` | 10,619 | `2639d6f7e4f3ffdb` | 内联案例原文、浏览器 SHA-256 校验、内联 verification |
| `GPT2-LLaMA-Atlas/js/main.js` | 93,049 | `96e2be5f57d5181e` | 内联 verification 回退；S4 已完成文案修正 |
| `GPT2-LLaMA-Atlas/README.md` | 15,682 | `1ca3cad1919b25b7` | 离线打开、内联范围与限制 |
| `output/qa/s7-browser.json` | 1,520 | `b4a685602ca741fe` | 28 项离线验收记录 |

## 3. 内联机制

构建脚本 `npm run build` 对 GPT-2 入口：

1. 读取 `data/gpt2/manifest.json`、`gpt2-main-001.json`、`verification.json` 与 `sources/gpt2/source-map.json`、`gpt2_real.py`；
2. 构建时校验案例 SHA-256 = 清单、源码 SHA-256 = source-map，拒绝不一致；
3. 把 **案例原始文本**（不是重新序列化）注入 `window.__ATLAS_TRACE_BUNDLE__`，同时注入 manifest、sha256、verification 与 `window.__ATLAS_SOURCE_BUNDLE__`；
4. 对 `<`、U+2028/2029 做脚本安全转义，避免提前结束 `<script>`；
5. 运行时 `replay-data.js` 优先使用内联数据，重新哈希案例原文；`source-data.js` 重新校验源码整文件与每个行区间；不再 fetch `data/` 或 `sources/`；
6. LLaMA 入口不注入 GPT-2 数据，保持独立、轻量。

## 4. 验证

`output/qa/s7-browser.json`：**28 项通过，0 未捕获错误，0 外部请求。**

- 构建产物：GPT-2 含内联 trace/source 标记与多 MB 载荷；LLaMA 不含 GPT-2 trace；体积符合预期。
- `file://` 项目目录：真实轨迹 ready、64 步、案例 SHA-256 浏览器复核通过、WebCrypto 可用、无网络请求、无外部引用。
- 离线真实张量/源码：查看器 `sourceStatus=ready`、`sourceVerified=true`、softmax 行 146 映射正确。
- 离线运行详情：verification 496/496、完整案例 sha256 可见。
- 离线课程：第 3 课启动、真实显微镜可见且标题“真实运行”、记录弹窗可开。
- 复制到临时目录：64 步真实回放、无网络、源码校验通过。
- HTTP 单文件：不请求 `data/gpt2`/`sources/gpt2`，内联案例校验通过。
- LLaMA 单文件：入口 `llama2`、轨迹模式 `sim`、无 GPT-2 数据、无网络/外部引用。
- S5 回归 280 项继续通过（HTTP 开发模式、课程、记录迁移、查看器、移动布局）。

## 5. 复现命令

```bash
cd GPT2-LLaMA-Atlas
npm test                 # 99 项
npm run build            # 生成 atlas.html / llama-atlas.html
python3 -m http.server 8080   # 开发模式：http://localhost:8080/index.html
```

重新核对数据（只读材料，不修改）：

```bash
/Users/yunni/Projects/temp/cs336-prep-code/.venv/bin/python \
  tools/trace/verify_gpt2.py \
  --material-root /Users/yunni/Projects/temp/cs336-prep-code/real_llm_lab \
  --data-dir data/gpt2 --case-id gpt2-main-001
```

## 6. 材料与来源

- 轨迹：`data/gpt2/`，案例 sha256 `91e99a3474f32dd3ce0a523af07cab9b7fa24ebd3004c44ce4790096245317c9`；独立核对 496/496（required 492，failed 0）。
- 源码快照：`sources/gpt2/gpt2_real.py`，sha256 `f8201019db7459491a84b8ff73eb081c78e431d08990757ec8bcf81bf0c31ebe`；行映射 `gpt2-real@1`。
- 权重：GPT-2 Small 官方权重；实现/权重/参考实现哈希与运行环境见 manifest 与运行详情。
- 旧双塔版本：`backup/20261030T183140-S2a/`。

## 7. 未完成 / 限制

| 项 | 状态 |
|---|---|
| S6 采样实验（可选） | 未实施；不影响首版 GPT-2 交付范围 |
| S8 LLaMA 独立课程/真实轨迹 | 按计划排除 |
| 用户实际试学 | 未进行；S5 只完成功能验证，学习体验待验收 |
| Top-N 未导出词表项 | 仍显示 `#id`；不伪造文本 |
| 第 1–11 层完整中间张量 | 未采集；查看器明确缺失 |
| Firefox/Safari/真实手机/屏幕阅读器 | 未验证；验收为 headless Chromium |
| 单文件体积 | 约 3.8 MB，首次加载/解码有成本；未做跨设备性能测量 |
| `file://` localStorage | 不同浏览器策略可能不同；记录功能失败会提示，页面仍可学习 |
| 发布/部署 | 未执行（按要求） |

## 8. 最终快照

`backup/20261030T214943-S7-final/` 包含：

- 两个单文件入口、全部运行时源码、CSS、测试、构建脚本；
- 已验证轨迹（manifest/案例/核对报告）、源码快照与 source-map；
- `docs/` S0—S7 交接文档与计划；
- `output/qa/` S3b/S4a/S4b/S5/S7 浏览器验收 JSON。

恢复时以该目录覆盖对应文件；单文件为构建产物，恢复后建议重跑 `npm test` 与 `npm run build`。
