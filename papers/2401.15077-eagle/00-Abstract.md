# Abstract：完整中文翻译、Figure 1 原图解读与技术讲解

> 论文：EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty  
> arXiv：2401.15077v3（2025-03-04）  
> 原文：<https://arxiv.org/html/2401.15077v3>；PDF 第 1 页  
> 阅读日期：2026-10-13  
> **范围：仅 Abstract（以及 Abstract 前的 Figure 1）；尚未开始 Section 1 Introduction。**

## Abstract 全文翻译（依原文顺序）

自回归解码（Autoregressive Decoding）使大语言模型（Large Language Models，LLMs）的推理非常耗时。

在本文中，我们重新审视推测采样（Speculative Sampling），并推导出两个关键观察。

首先，在特征层面（Feature Level，即倒数第二层特征）进行自回归建模，比在 Token 层面进行自回归建模更加直接。

其次，特征层面（倒数第二层特征）自回归过程中固有的不确定性限制了其性能。

基于这些见解，我们提出了 EAGLE（Extrapolation Algorithm for Greater Language-model Efficiency，一种提升语言模型效率的外推算法），这是一个简单但高效的推测采样框架。

通过引入一个在时间步上提前一位的 Token 序列，EAGLE 有效地解决了这种不确定性，从而能够以极低的额外开销精确预测倒数第二层特征。

我们对 EAGLE 进行了全面的评估，涵盖了 Vicuna 和 LLaMA2-Chat 系列的所有模型、混合专家模型 Mixtral 8x7B Instruct，以及对话、代码生成、数学推理和指令遵循等任务。

对于 LLaMA2-Chat 70B，EAGLE 实现了 2.7–3.5 倍的延迟加速，将吞吐量提升至约两倍，同时保持生成文本的概率分布不变。

## Figure 1（论文原图，PDF 第 1 页）

**原图来源：** <https://arxiv.org/pdf/2401.15077v3#page=1>。本次对话已从用户提供的原始 PDF 直接裁图展示。**当前尚未将 PNG 二进制上传到 GitHub；不创建虚假的 figures/ 相对链接。**

**原图注完整翻译：** 在贪心（temperature=0）设置下，Vicuna 和 LLaMA2-Chat 在 MT-bench 上的推理延迟加速比。Medusa 和 Lookahead 的加速比取自它们各自的原始技术报告。对于推测采样，缺少合适的草稿模型来加速 7B 模型。将 7B 模型用作 13B 模型的草稿模型时，因为 7B 模型本身开销太高，导致整体速度缓慢，效果甚至不如普通自回归解码。相应情况标记为 N/A。本文仅与不必微调主干模型的推测采样方法进行比较，以保持目标模型原有的输出文本分布。

**原图数值（温度 0，MT-bench；按 EAGLE/Medusa/Lookahead/Speculative sampling/DistillSpec/Vanilla 顺序）：**

| Target | EAGLE | Medusa | Lookahead | Speculative sampling | DistillSpec | Vanilla |
|---|---:|---:|---:|---:|---:|---:|
| Vicuna 7B | 2.90× | 1.97× | — | N/A | — | 1.00× |
| Vicuna 13B | 3.07× | 1.92× | — | N/A | — | 1.00× |
| Vicuna 33B | 2.95× | 1.94× | — | 1.27× | 1.12× | 1.00× |
| LLaMA2-Chat 7B | 2.78× | — | 1.64× | N/A | — | 1.00× |
| LLaMA2-Chat 13B | 3.03× | — | 1.51× | N/A | — | 1.00× |
| LLaMA2-Chat 70B | 3.01× | — | 1.45× | 1.88× | 2.13× | 1.00× |

注：破折号表示该图未给出该方法的数值，不能理解为零；“N/A”按原图。Medusa 与 Lookahead 的严格非贪心分布保持能力需要区分，不宜将温度 0 的结果泛化至任意采样设置。

## 独立技术讲解（不是原文翻译）

### 核心问题

标准推测解码由低成本 Draft 与目标模型 Verification 两阶段组成，需要兼顾候选质量和 Draft 开销。若草稿模型太重，即使有较高接受率，也可能失去加速优势。

### 为什么预测特征

令 $f_i$ 表示目标模型对前缀 $T_{1:i}$ 得到的、LM Head 前的隐藏特征。普通自回归解码将其映射到下一 Token 分布：
\[
p_{i+1}=\mathrm{Softmax}(\mathrm{LMHead}(f_i)),\qquad t_{i+1}\sim p_{i+1}.
\]
EAGLE 的 Draft 不直接生成 Token，而是预测下一隐藏特征，继而借用目标模型的 LM Head 生成下一个 Token 分布。

### 什么叫 shifted-token

特征预测器的条件形式可写为：
\[
\hat f_{i+1}=g_\phi(F_{1:i},T_{2:i+1}), \quad
\hat p_{i+2}=\mathrm{Softmax}(\mathrm{LMHead}(\hat f_{i+1})).
\]
注意 $t_{i+1}$ 已从 $f_i$ 对应分布中得到；将它作为下一特征预测条件，**不是泄漏未知未来 Token**。如果同一 $f_i$ 可能采样出不同 $t_{i+1}$，这些采样结果可能对应不同的后续隐藏状态；只看历史特征会把多种后继状态混在一起。额外提供抽到的 Token，降低了预测歧义。

此处只作摘要导读；论文真正的 Draft Head 结构、训练损失与树验证算法分别详见 Section 3 和 Appendix A，后续按原文完整翻译。

### Figure 1 的解释边界

- Vicuna 7B：EAGLE 2.90×，Medusa 1.97×；相对速度比约 $2.90/1.97\approx1.47$。
- 图表为 MT-bench / temperature=0 的延迟加速，不可直接代表所有任务、GPU 或并发。
- 分布保持来自**正确的 Verification/拒绝采样**，不是说草稿预测的 Token 必须与目标单次采样的 Token 完全相同。

### 与 DFlash 的跨论文关联（独立对比）

[已精读 DFlash](../2602.06036-dflash/README.md) 用目标模型多层隐藏特征并通过逐层 KV Injection 条件化轻量块扩散 Draft，块内并行生成候选；EAGLE 是以目标模型隐藏状态为基础、用 shifted token 缓解不确定性的**自回归式**特征预测。相同研究范式是“复用 Target 信息提高 Draft 质量”，主要差异是自回归与并行 Draft 的时延结构。不要将本文 EAGLE 与后续 EAGLE-3 混为一篇。

## 阅读断点

Abstract + Figure 1 已完成。**下一阅读单元：1. Introduction**（从 PDF 第 1 页末跨至第 3 页，含 Figure 2、Figure 3、Figure 4 与贡献列表）。
