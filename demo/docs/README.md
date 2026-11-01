# 当前项目与文档入口

更新：2026-10-31。旧批次文档与中间产物已按确认清单整理。

## 当前交付

- [GPT-2 单文件](../GPT2-LLaMA-Atlas/atlas.html)与[Llama / SmolLM2 单文件](../GPT2-LLaMA-Atlas/llama-atlas.html)：已完成真实轨迹、独立学习路径与离线交付。
- [最新修正交接](REVIEW_FIXES_20261031.md)：7 项问题已修正；115 项 Node、22 项浏览器检查通过。
- [独立审查](INDEPENDENT_REVIEW_20261031.md)、[GPT-2 交付](S7_HANDOFF.md)、[Llama 交付](LLAMA_DELIVERY_HANDOFF.md)：保留历史时点，当前产物哈希以最新修正交接为准。
- [DeepSeek 项目](../DeepSeek-V4.1-Flash/README.md)：源码、原始 PDF 与既有验证结果保留。
- 学习效果、跨浏览器/真实触摸设备、GPU 温度功耗等仍按交接记录区分未验证；可选 S6/L6 未实施。

## 方案与数据依据

- [GPT-2 方案](GPT2_SOURCE_GROUNDED_PLAN.md)、[Llama 方案](LLAMA_SOURCE_GROUNDED_PLAN.md)。
- [GPT-2 轨迹契约](TRACE_SCHEMA.md)、[Llama 轨迹契约](LLAMA_TRACE_SCHEMA.md)、[Llama 课程 ID](LLAMA_COURSE_IDS.md)。
- [GPT-2 材料核对](S0_MATERIAL_AUDIT.md)、[Llama 材料核对](LLAMA_MATERIAL_AUDIT.md)、[Llama 实现审查与修正](LLAMA_IMPLEMENTATION_REVIEW.md)。
- 当前真实数据与源码快照分别在项目的 `data/`、`sources/`，未参与删除；测试与导出工具保持可用。

## 历史归档

[历史文档归档](history-20261031.tar.gz)按原相对路径保存早期设计、B 批次与执行交接、截图。解压到独立目录阅读；包内 `__archive__/SHA256SUMS.json` 提供逐文件校验。

已归档的原路径：

- `GPT2-LLaMA-Atlas/DESIGN.md`
- `VISUALIZATION_IMPROVEMENT_PLAN.md`
- `docs/B07_HANDOFF.md`
- `docs/B08_D_HANDOFF.md`
- `docs/B08_G_HANDOFF.md`
- `docs/B08_REVIEW.md`
- `docs/B09_HANDOFF.md`
- `docs/B10_HANDOFF.md`
- `docs/B11_HANDOFF.md`
- `docs/B12_HANDOFF.md`
- `docs/B13_HANDOFF.md`
- `docs/B14_HANDOFF.md`
- `docs/BATCH_STATUS.md`
- `docs/HOTFIX_RENDER_D_HANDOFF.md`
- `docs/HOTFIX_RENDER_G_HANDOFF.md`
- `docs/IMPLEMENTATION_PLAN.md`
- `docs/L0_LLAMA_HANDOFF.md`
- `docs/REMAINING_COORDINATION.md`
- `docs/S0_S1_HANDOFF.md`
- `docs/S2A_HANDOFF.md`
- `docs/S2B_HANDOFF.md`
- `docs/S3_HANDOFF.md`
- `docs/S4_HANDOFF.md`
- `docs/S5_HANDOFF.md`
- `docs/VERIFICATION.md`
- `docs/screenshots`

保留文档里的相关链接已转到本节。正文中原样出现的旧路径和旧批次备份路径属于历史记录，不代表仍存在于工作目录。

## 恢复与清理记录

- [当前交付恢复包](../backup/20261031-current-delivery.tar.gz)：清理前的已修正源码、两个 HTML、真实数据、vendor、DeepSeek 原始 PDF、文档与最新核对证据。包内逐文件 SHA-256 已重新读取校验；不含 node_modules 或上级模型权重。
- 初始基线保留在 `backup/20261029T200704-B00/`；模型代码初次修正前副本在 `backup/20261030-llama-review-fix-before/`。
- 旧 LLaMA-2 完整页面和源码恢复点保留在 `backup/20261030T214943-S7-final/`，与当前 SmolLM2 数据分开。
- [精确清单与执行结果](../output/qa/cleanup-20261031-manifest.json)。最新验收 JSON、修正截图与浏览器复现脚本保留；旧日志、测试夹具和重复 HTML 已删除。

清理不修改运行逻辑、正式学习记录或原始资料，不代表重新进行学习效果验收。
