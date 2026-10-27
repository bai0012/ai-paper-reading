# Abstract — Medusa 完整译文与技术讲解

- 论文：Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads
- 版本：arXiv:2401.10774v3（2024-06-14）；ICML 2024 / PMLR 235
- 原文：[arXiv HTML](https://arxiv.org/html/2401.10774v3) · [PDF](https://arxiv.org/pdf/2401.10774v3#page=1)
- 范围：原 PDF 第 1 页 Abstract 全文。下面的「完整译文」与「技术讲解」严格分开。

## 一、Abstract 全文中文翻译

大型语言模型（Large Language Models，LLMs）采用自回归解码（auto-regressive decoding），这一过程需要顺序计算，每一步都依赖于前一步的输出。这形成了一个瓶颈，因为每一步都需要将完整的模型参数从高带宽显存（High-Bandwidth Memory，HBM）传输到加速器的缓存中。尽管已有推测解码（speculative decoding）等方法被提出以解决这一问题，但获取并维护独立草稿模型（draft model）所面临的困难阻碍了这些方法的实施。

在本文中，我们提出 Medusa，这是一种高效方法：通过增加额外的解码头（decoding heads），使大语言模型能够并行预测多个后续 Token，以加速其推理过程。Medusa 使用基于树的注意力机制（tree-based attention mechanism）构造多个候选续写，并在每个解码步骤中同时验证它们。通过利用并行处理，Medusa 大幅减少了所需的解码步骤数量。

为满足不同使用场景的需求，我们提出了两个层级的 Medusa 微调方法。**Medusa-1：** 直接在冻结主干大语言模型（backbone LLM）的基础上微调 Medusa，从而实现无损的推理加速。**Medusa-2：** 将 Medusa 与主干大语言模型一起微调，以提高 Medusa 解码头的预测准确率并获得更高的加速比，但这需要特殊的训练方案来保留模型原有的能力。

此外，我们提出了几项扩展，以提高 Medusa 的效率或拓展其实用性。其中包括：用于处理没有训练数据可用情形的自蒸馏（self-distillation），以及一种典型接受机制（typical acceptance scheme），它在维持生成质量的同时提高候选接受率。我们在不同规模、采用不同训练流程的模型上评估了 Medusa。实验表明，Medusa-1 能在不损害生成质量的情况下实现超过 2.2 倍的加速，而 Medusa-2 则进一步将加速比提高到 2.3–2.8 倍。

> 校核：上述数值遵循用户提供的 v3 PDF 和对应 v3 HTML。arXiv 的摘要元数据页目前另有 2.3–3.6× 表述，不能替换本版本 PDF 的结论。

## 二、技术讲解（非摘要原文）

### 1. 真正的瓶颈：单步解码的权重读取与低算术强度

Batch=1 的自回归解码通常是内存带宽受限，而不是算力耗尽。每次 Forward 仅新增一个 Token，矩阵运算的复用规模小，模型权重搬运成本难以摊薄。Medusa 用一次候选树并行验证换取平均多个已接受 Token，目的是减少目标模型前向迭代次数。不过是否加速取决于实际的验证开销和接受长度。

### 2. 多头输出的时间位置

设目标 Transformer 在位置 t 输出最后隐藏状态 h_t，原始 LM Head 给出位置 t+1 的概率分布 p_t^(0)，额外的第 k 个 Medusa Head 输出位置 t+k+1 的预测 p_t^(k)。因此，K 个新增头分别预测 t+2 到 t+K+1。这些预测**共享同一个 h_t**，并非逐个条件化在真实未来 Token 上；这一差异是预测越远越不准确的原因之一。

正文 Section 2.1.1 给出头结构：
$$
p_t^{(k)}=\operatorname{softmax}\left(W_2^{(k)}\left(\operatorname{SiLU}(W_1^{(k)}h_t)+h_t\right)\right).
$$
更严格地说，论文印刷式对 W_2/W_1 的维度标记存在需要结合括号顺序理解的情况；正式方法节会逐项核对。此处只预告这一结构，不把公式冒充摘要原文。

### 3. 为何要使用树

从不同预测头各选取多个 Top 候选并做组合，例如两个未来位置分别选择 Top-2 与 Top-3，则形成 2×3=6 条候选序列。如果逐条调用目标模型，额外成本很高；Tree Attention 通过祖先可见的注意力掩码和匹配的 position IDs 让同一棵树的不同分支一次并行验证。共享前缀只计算一次，不同分支不能互相看见。

### 4. Medusa-1 与 Medusa-2

| 维度 | Medusa-1 | Medusa-2 |
| --- | --- | --- |
| 主干 Backbone | 冻结 | 参与联合微调 |
| 训练对象 | 新增多个解码头 | 主干 + 多解码头 |
| 目标 | 低成本适配现有模型 | 更高预测准确率与接受长度 |
| 主要风险 | 远距离预测准确率受限 | 微调破坏原有 Next-Token 分布与能力 |
| 论文报告的加速 | 超过约 2.2× | 约 2.3–2.8×（摘要） |

### 5. 「无损」必须限定语义

冻结原始 Backbone 只能说明原模型参数未被改变，并不自动保证最终输出的**采样分布**与原模型完全一致。论文 Section 2.3.1 明确允许使用 Typical Acceptance；它以概率/熵阈值接受合理候选，不追求严格等分布。若目标是理论上与目标采样分布精确一致，需要合适的 Rejection Sampling/校正程序。温度为 0 的贪心一致性、温度大于 0 的分布等价性、人工评测质量接近是三种不同性质，不能混为一谈。此处是结合正文进行的必要限定，不是摘要作者的补充原话。

### 6. 一个有用的性能恒等式

论文 Appendix B.1 定义：Acceleration rate 为每次解码步骤平均产出的 Token 数，Overhead 为 Medusa 每步耗时与常规每步耗时之比。因此
$$
\mathrm{Speedup}=\frac{\mathrm{Acceleration\ Rate}}{\mathrm{Overhead}}.
$$
例如若每轮平均接受 3.47 个 Token，但每轮耗时为原来的 1.22 倍，则理论整体加速约 3.47/1.22≈2.84 倍。该算例使用论文 Table 1 中的 Vicuna-7B 统计（四舍五入误差导致文中报告 2.83×）。

## 三、原图位置与图注预告

本单元 Abstract 位于 PDF 第 1 页，无图。首张 Figure 1 位于 PDF 第 2 页、紧接 Introduction / Methodology 的过渡处。PDF：[第 2 页原图](https://arxiv.org/pdf/2401.10774v3#page=2)。该图展示共享主干隐藏状态、多个解码头、候选构造、验证和接受；原图已从用户 PDF 提取并在本次对话展示，**PNG 二进制尚未上传 GitHub**。Figure 1 的完整图注及逐层机制留待下一单元 Introduction 翻译。

## 四、与已读论文的联系（独立分析）

| 方法 | Draft 表示/形成方式 | Draft 并行程度 | 验证及生成分布 |
| --- | --- | --- | --- |
| **Medusa**（本篇） | 共享 Backbone 隐藏状态的多个未来 Token 解码头 | 多头一次并行预测、树状合并与验证 | 可选精确 Rejection Sampling；Typical Acceptance 属近似分布 |
| **EAGLE**（2401.15077） | 倒数第二层 Feature 的自回归预测 + Shifted Token | 多轮自回归 Draft、树状 Target 验证 | 使用正确的推测采样时保持目标分布 |
| **DFlash**（2602.06036） | 目标多层隐藏特征条件下的 Block Diffusion Draft | 一次前向并行生成块内候选 | 使用正确的 Target 验证时保持目标分布 |

这三种方法的核心权衡是草稿构造成本、候选质量/接受长度以及验证开销。比较论文速度时必须核对 Target 模型、硬件、Serving 后端、解码温度及 Batch Size，不能直接横比文章摘要的最高倍数。

## 本单元结论

Medusa 通过“共享主干 + 多头直接预测未来 Token + Tree Attention 并行验证”规避独立小草稿模型的管理与推理成本。Medusa-1 保守、易适配，Medusa-2 提高草稿可接受性。**下一阅读单元：Section 1 Introduction，完整译文及 Figure 1 原图/图注。**
