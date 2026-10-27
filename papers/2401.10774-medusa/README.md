# Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads

**中文标题：** Medusa：基于多解码头的简单大语言模型推理加速框架

- arXiv：[2401.10774v3](https://arxiv.org/html/2401.10774v3)（2024-06-14）
- 作者：Tianle Cai、Yuhong Li、Zhengyang Geng、Hongwu Peng、Jason D. Lee、Deming Chen、Tri Dao
- 会议：ICML 2024，PMLR 235:5209–5235
- PDF：[arXiv v3](https://arxiv.org/pdf/2401.10774v3)（27 页，含附录）
- 官方代码：[FasterDecoding/Medusa](https://github.com/FasterDecoding/Medusa)
- 阅读进度：[progress.json](progress.json)

## 研究问题

如何在不维护独立 Draft Model 的情况下，让原始大模型基于同一个隐藏状态并行预测后续多个 Token，并通过 Tree Attention 将这些候选序列集中验证，从而减少自回归 Forward 轮次？

## 主要贡献（基于已精读 Abstract 与 Introduction；公式和实施细节仍须核对 Section 2）

1. 多个轻量化 Medusa Heads 分别预测不同未来位置的 Token。
2. Tree Attention 将候选续写合并成树，允许共享前缀并行验证。
3. Medusa-1 只训练解码头，Medusa-2 在特殊策略下联合训练主干和解码头。
4. Self-Distillation 缓解缺少训练数据的问题；Typical Acceptance 在允许输出分布变化的前提下提高接受效率。

摘要报告 Medusa-1 在维持生成质量时可超过 2.2× 加速，Medusa-2 为 2.3–2.8×。这些是作者的实验总结，不是普适服务性能保证。arXiv 摘要列表页与 v3 PDF 的数字并不完全一致，遵循 v3 PDF 原文。

## 章节导航

1. [Abstract：完整译文、机制解析和 EAGLE/DFlash 对比](00-Abstract.md) ✅
2. [1. Introduction：六段完整译文、Figure 1 全图注、带宽与多头推理分析](01-Introduction.md) ✅
3. 2. Methodology（待阅读）
   - 2.1 Key Components：Medusa Heads / Tree Attention（Figure 2）
   - 2.2 Training Strategies：Medusa-1 / Medusa-2 / Head 数量选择
   - 2.3 Extensions：Typical Acceptance / Self-Distillation / Optimized Tree
4. 3. Experiments（待阅读；Figure 3–5，Table 1–3）
   - 3.1 Vicuna-7B/13B；3.2 Vicuna-33B/Zephyr-7B；3.3 Ablation
5. 4. Discussion（待阅读）
6. A. Related Work（待阅读）
7. B. Experiment Settings（待阅读）
8. C. Visualization of Optimized Tree Attention（待阅读；Figure 6）
9. D. Results of Speculative Decoding（待阅读；Figure 7）
10. E. Additional Results for All Models（待阅读；Figure 8）
11. F. Additional Results on AlpacaEval（待阅读；Table 4）
12. G. Exploration and Modeling of Hardware Constraints and Medusa（待阅读；Figure 9–23，Table 5–8）

## 原图说明

本篇共 27 页。Figure 1 在 PDF 第 2 页，原图已在当前对话中裁图展示；GitHub 仓库尚未保存原始 PNG 二进制，故只引用[原版 PDF 第 2 页](https://arxiv.org/pdf/2401.10774v3#page=2)，不会生成虚假的相对路径。Figure 1 的完整图注已在 [Introduction](01-Introduction.md) 翻译，详细说明额外 Heads、Top-k 候选和最长可接受前缀。

## Section 1 Introduction 精读小结

[完整译文与 Figure 1 解析](01-Introduction.md) 已覆盖原文六段和 Vicuna-33B 原始脚注 1（PDF 第 1–2 页）。从系统层面解释 Batch=1 解码的 HBM 带宽瓶颈、为何提高 Arithmetic Intensity；从算法层面比较独立串行 Draft 与基于共享 Target 隐藏状态的多解码头、Tree Attention 共享前缀与分支隔离；讨论 Medusa-1/2、自蒸馏和 Typical Acceptance。原图 Figure 1 在对话裁图展示，**GitHub 未保存 PNG 二进制**；图注完整译文已归档。提醒：Introduction 对温度作为阈值的表述应结合后文 Section 2.3.1 的 ε/δ/Entropy 公式理解；Typical Acceptance 不是精确目标分布保真采样。下节为 Section 2.1。

## 已读论文关联

- [EAGLE](../2401.15077-eagle/README.md)：用 Feature-level autoregression + Shifted Token 提高 Draft 准确率，Target 树状验证。
- [DFlash](../2602.06036-dflash/README.md)：使用目标多层隐藏特征条件化的 Block Diffusion，一次前向并行预测一整个草稿块。
- Medusa 与二者共通之处是通过一次目标验证评估多个候选；主要不同在于 Medusa 直接利用多个 Token Heads 预测不同未来位置，且 Typical Acceptance 可不严格保持原采样分布。

**当前阅读状态：** Abstract + Section 1 Introduction 完成（2026-10-27），下一单元为 Section 2.1 Key Components（2.1.1 Medusa Heads、2.1.2 Tree Attention）。
