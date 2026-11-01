# 真实生成轨迹与核对

当前案例为 `gpt2-main-002` 与 `smollm2-main-002`；旧例及其配套来源、清单、报告、离线页面保存在 `../../../backup/20261031-generation-examples-before/`。

| 案例 | 输入 | 新 token | 前向 | 事件 | 张量 |
|---|---|---:|---:|---:|---:|
| GPT-2 | `The cat sat on the` | 9 | 9 | 144 | 333 |
| SmolLM2 | `To make a cup of tea, first` | 7 | 7 | 238 | 455 |

两条轨迹均使用真实权重、CPU fp32、greedy。prefill 选出第一个 token，后续 decode 每轮输入上一轮选词；达到生成上限后直接返回，不再做未被使用的前向。最后选出的 token 不必写入 KV，除非继续预测下一个。

模型计算实现从项目内 `sources/gpt2/`、`sources/llama/` 加载；它们从课程材料派生，仅调整生成终止条件。外部材料、权重、个人实验记录保持只读；上游与项目代码哈希分别记录于 manifest。`llama_source_map.py` 默认从项目来源目录重建映射，GPT-2 映射在导出时重建。

在 `GPT2-LLaMA-Atlas/` 内，使用材料仓库的 `.venv/bin/python`，设置 `PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1`，依次运行：

```bash
python tools/trace/llama_source_map.py
python tools/trace/export_gpt2.py
python tools/trace/export_llama.py
python tools/trace/verify_gpt2.py --material-root /Users/yunni/Projects/temp/cs336-prep-code/real_llm_lab --data-dir data/gpt2 --case-id gpt2-main-002
python tools/trace/verify_llama.py
npm test
npm run build
npm run qa:session
npm run qa:browser
```

`python` 在这里指 `/Users/yunni/Projects/temp/cs336-prep-code/.venv/bin/python`。不下载权重，不训练。

只导出实际选词使用的最后行完整 logits，其他位置仅留摘要。全层注意力及第 0 层完整中间张量保持真实载荷。新增完整词表的单 token 解码与原始符号标签供温度面板使用，由独立 tokenizer 再核对一次。

温度面板在全部词表上做 softmax，然后截取 Top-8 显示，保留剩余概率质量。抽样也覆盖完整词表。它只改变当前一步的实验分布和抽取结果，不修改主轨迹或借用分支后的旧计算。

导出与核对失败时不得标记通过。浏览器测试使用隔离 context，检查开发 HTTP、单文件 HTTP、file://、终止/倒退、温度/种子及学习记录不变；报告和截图写入 `../../../output/playwright/generation-20261031/`。

浏览器验证默认使用实际核对为 Metal 的已安装 Chrome，两套检查共用锁串行执行；超时及异常统一清理。详见 [浏览器验证说明](../qa/README.md)。
