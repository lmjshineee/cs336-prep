# CS336 Prep Code

跟着 Stanford CS336《Language Modeling from Scratch》从零实现 Transformer / LLaMA 的学习代码、实验记录与可视化演示。

## 目录结构

| 路径 | 内容 |
| --- | --- |
| `transformer_steps/` | 从零实现最简 Transformer 的 8 步递进代码，按顺序阅读 |
| `python_basics/` | 配合 `transformer_steps` 的 Python / PyTorch 基础 |
| `llama_steps/` | 从 GPT-2 进阶到 Llama 架构的 9 步实现（SmolLM2-135M 实例） |
| `real_llm_lab/` | 真实 GPT-2 权重的实验：生成、注意力头、KV cache、mask PPL 等 |
| `minimal_transformer.py` | 单文件精简 Transformer |
| `minimal_llama.py` | 单文件精简 Llama |
| `CS336_ROADMAP.md` | CS336 学习路线图 |
| `CS336_A1_DETAILED_PLAN.md` | Assignment 1 详细计划 |
| `BPE_TOKENIZER_TUTORIAL.md` | BPE 分词器教程：学习路线、从零实现、实验与“移植到已有模型” |
| `bpe_lab/` | BPE 教学实现（含 20 个测试）、小语料训练 demo、GPT-2 换 tokenizer 的 BPB 实验 |
| `docs/bpe/SOURCES.md` | BPE 教程的材料地图、精确规格、测试地图与勘误 |
| `demo/docs/` | 演示页面的设计、审查与交接文档 |
| `demo/GPT2-LLaMA-Atlas/` | GPT-2 / LLaMA 交互式图谱；单文件版：`atlas.html`、`llama-atlas.html`，浏览器直接打开即可 |

## 单文件演示

以下 HTML 为自包含页面（数据与样式内联），无需启动服务，双击即可在浏览器中打开：

- `demo/GPT2-LLaMA-Atlas/atlas.html`
- `demo/GPT2-LLaMA-Atlas/llama-atlas.html`

## 未纳入版本控制

为控制仓库体积，以下内容不提交：

- 模型权重：`real_llm_lab/weights/`、`llama_steps/weights/`（从 Hugging Face 重新下载）
- 环境与缓存：`.venv/`、`.uvcache/`、`__pycache__/`
- 前端依赖与 demo 生成产物：`node_modules/`、`demo/output/`、`demo/backup/`、`demo/GPT2-LLaMA-Atlas/data/`

具体复现方式见各目录下的 `README.md`。

## 环境

Python 3 + PyTorch，依赖安装与运行方式见各目录 README；模型权重从 Hugging Face 下载（GPT-2 small / SmolLM2-135M）。
