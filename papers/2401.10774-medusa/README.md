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

## 主要贡献（Abstract、Introduction、Methodology、Experiments、Discussion 均已完成；Appendices 待精读）

1. 多个轻量化 Medusa Heads 分别预测不同未来位置的 Token。
2. Tree Attention 将候选续写合并成树，允许共享前缀并行验证。
3. Medusa-1 只训练解码头，Medusa-2 在特殊策略下联合训练主干和解码头。
4. Self-Distillation 缓解缺少训练数据的问题；Typical Acceptance 在允许输出分布变化的前提下提高接受效率。

摘要报告 Medusa-1 在维持生成质量时可超过 2.2× 加速，Medusa-2 为 2.3–2.8×。这些是作者的实验总结，不是普适服务性能保证。arXiv 摘要列表页与 v3 PDF 的数字并不完全一致，遵循 v3 PDF 原文。

## 章节导航

1. [Abstract：完整译文、机制解析和 EAGLE/DFlash 对比](00-Abstract.md) ✅
2. [1. Introduction：六段完整译文、Figure 1 全图注、带宽与多头推理分析](01-Introduction.md) ✅
3. 2. Methodology（已完成）
   - [2.1 Key Components：Medusa Heads / Tree Attention 与 Figure 2 完整解析](02-Key-Components.md) ✅
   - [2.2 Training Strategies：完整译文、公式 (1)(2)、Heads Warmup、PyTorch 示例](03-Training-Strategies.md) ✅
   - [2.3 Extensions：Typical Acceptance、自蒸馏 KL 与优化树贪心搜索](04-Extensions.md) ✅
4. [3. Experiments：全节译文、Figure 3–5、Table 1–2、消融分析](05-Experiments.md) ✅
   - 3.1 Vicuna-7B/13B；3.2 Vicuna-33B/Zephyr-7B；3.3 Tree/Threshold/Two-stage Ablation（均已完成）
5. [4. Discussion：两段完整译文、Table 3 四阶段加速、Batch/Serving 限制](06-Discussion.md) ✅
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

## Section 2.1 Key Components 精读小结

[2.1.1 与 2.1.2 完整译文、原 Figure 2 和全部技术推导](02-Key-Components.md) 已完成（PDF 第 3–4 页）。重点分析 Medusa Head 的 $K$ 个时间索引、$W_1/W_2$ 权重初始化与残差；**v3 PDF 原公式声明 $W_2\in\mathbb R^{d\times V}$，与 $W_2(\cdots)$ 的左乘记法不一致**，章节完整保留原式并给出列向量/行向量两种自洽实现。Figure 2 六条候选路径、八个新增 Token 树节点、14 个 Tree Mask 可见关系，推导 $N=\sum_{k=1}^K\prod_{i=1}^ks_i$、祖先 Attention Mask、重复 Token 的分支区别及 Position IDs。**Figure 2 原图已在对话中由用户 PDF 裁取展示，GitHub 仍未保存 PNG 二进制。** 下一单元是 Section 2.2。

## Section 2.2 Training Strategies 精读小结

[Section 2.2 全部译文及公式 (1)(2) 推导](03-Training-Strategies.md) 已完成：2.2.1 Medusa-1 冻结 Backbone，用偏移 $k+1$ 的交叉熵监督未来 Token，采用 $\lambda_k=0.8^k$；2.2.2 Medusa-2 在原 LM Loss 基础上增加 $\lambda_0\mathcal L_{\mathrm{MEDUSA-1}}$，采用差异化学习率、Heads Warmup 与两阶段训练；2.2.3 建议训练 5 个 Heads，推理可只使用其中 3–4 个。**原文 Warmup 中将 $\lambda_0$ 称为 Backbone Loss 的权重，但式 (2) 将其乘在 Medusa Loss 上；章节忠实保留原句并指出不一致。** Appendix B 佐证 Vicuna-7B/13B 参数设置为 5 Heads、QLoRA、4 倍 Head LR、$\lambda_0=0.2$。提供[已测试 PyTorch 教学代码](examples/medusa_training_example.py)，验证 Backbone 冻结/联合训练时梯度行为（代码不是官方实现）。本节无新 Figure/Table。下一单元 Section 2.3 Extensions。

## Section 2.3 Extensions 精读小结

[完整译文、公式和数值分析](04-Extensions.md) 已完成（PDF 第 5–6 页）：2.3.1 Typical Acceptance 的原始阈值 $p(x)>\min(\epsilon,\delta e^{-H(p)})$、贪心首 Token 与最长合格前缀；2.3.2 Self-Distillation 的公开 Seed Prompt、自生成对话、Backbone KL($p_{\mathrm{teacher}}\|p_{\mathrm{student}}$) 和关闭 LoRA Adapter 获取原始 Teacher 分布；2.3.3 根据 Top-$i$ Acc 差分计算 $a_k^{(i)}$，使用 $\prod_j a_j^{(i_j)}$ 对树节点排序，贪心扩展固定预算树。**保留两点理论校核**：论文称 $p=q$ 时正确拒绝采样仍可能拒绝，这不适用于标准接受概率 $\min(1,p/q)$；树期望接受长度公式依赖排名准确率独立等近似，不能直接等价于 Typical Acceptance 真实接受长度或 Wall-Time 最优。阈值数值例子和树搜索算法已进行计算检查。本节无新的 Figure/Table；引用 Figure 5（Section 3）和 Figure 6（Appendix C）待后续正式精读。

## Section 3 Experiments 精读小结

[全文译文、Figure 3–5 / Table 1–2 原图页码与所有原表数值、实验归因分析](05-Experiments.md) 已完成（PDF 第 6–9 页）：Vicuna-7B/13B 对比 Medusa-1/Medusa-2，Figure 3 报告 7B 2.18×/2.83×、13B 2.33×/2.83×，八类 MT-Bench 最高 Extraction 3.62×、Coding 3.29×；Vicuna-33B 与 Zephyr-7B 的自蒸馏，Table 1 四组 Medusa-2 最终 Speedup 分别为 2.83×、2.66×、2.83×、2.35×，Quality 变化分别为 +0.01、−0.07、−0.14、+0.05；Figure 4：优化 64-node Sparse Tree 的 Acceleration Rate 优于 256-node Random Dense Tree，并证明候选增加不一定提高 Tokens/s；Figure 5：$T=0.7$、$\epsilon=0.01\ldots0.25$ 的典型接受阈值扫描，接受加速与 GPT-4 平均分呈整体权衡，原图没有逐阈值精确数据表；Table 2：Baseline 6.17、Direct Finetune 5.925、Medusa-1 6.23/2.18×、Medusa-2 6.18/2.83×。**校核**：用 Table 1 已展示值反算 Acc/Overhead 与论文 Speedup 相差约 0.001–0.024；原文未公开逐条计时统计，不能判定确切成因，也不可修改论文正式数据。Section 3 原图在本对话从原版 PDF 直接裁取，**GitHub 的 UTF-8-only Connector 不支持 PNG 上传**，章节只链接准确原版页码；额外整理五张 PNG 的 ZIP 下载包在当前聊天提供。Table 3 属于 Section 4 Discussion，已在 [本节译文](06-Discussion.md) 完整翻译。

## Section 4 Discussion 精读小结

[原文全部两段译文、Table 3 原表及四阶段加速分解](06-Discussion.md) 已完成（PDF 第 9 页）。Table 3 在论文排版中虽然浮动到 Section 4 标题上方，但正文明确引用；其四个**累计**净加速比为：Medusa-1 Heads 无树约 1.5×、加入 Tree Attention 约 1.9×、优化树配置约 2.2×、Medusa-2 联合训练约 2.8×。依次相对提升约 26.7%、15.8%、27.3%，均由原表约数独立推算，不能相乘或冒充独立因子消融。原文承认主要研究 Batch Size=1；高并发 Continuous Batching 需额外分析 GPU 算力、KV Cache 和调度开销，质量接近不等于分布精确。**2026-11-03 额外现状核实（非原文）**：[vLLM 主线 PR #61701](https://github.com/vllm-project/vllm/pull/61701) 已于 2026-10-15 移除内置 Medusa；vLLM-Ascend 单独仍列有；SGLang 官方文档目前主要支持 EAGLE、MTP、DFlash 等，并未列标准 Medusa 直接启动方式。Table 3 原图已从 v3 PDF 裁取并在聊天展示，GitHub Connector 不支持上传 PNG 二进制，故章节引用[原始 PDF 第 9 页](https://arxiv.org/pdf/2401.10774v3#page=9)。主文已完成，Appendix A–G 尚待正式阅读。

## 已读论文关联

- [EAGLE](../2401.15077-eagle/README.md)：用 Feature-level autoregression + Shifted Token 提高 Draft 准确率，Target 树状验证。
- [DFlash](../2602.06036-dflash/README.md)：使用目标多层隐藏特征条件化的 Block Diffusion，一次前向并行预测一整个草稿块。
- Medusa 与二者共通之处是通过一次目标验证评估多个候选；主要不同在于 Medusa 直接利用多个 Token Heads 预测不同未来位置，且 Typical Acceptance 可不严格保持原采样分布。

**当前阅读状态：** Abstract、Introduction、Section 2.1–2.3 Methodology、Section 3 Experiments、Section 4 Discussion 均已完成（2026-11-03）；下一单元 **Appendix A Related Work**。
