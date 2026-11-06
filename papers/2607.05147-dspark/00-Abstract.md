# Abstract — 完整翻译与技术讲解

> 论文：**DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation**  
> arXiv: [2607.05147v1](https://arxiv.org/abs/2607.05147)（2026-07-06）  
> 原文依据：用户提供的 v1 PDF，第 1 页，Abstract（正文为单个段落）。  
> 保存范围：仅 Abstract，下一单元为 **1. Introduction**。  
> 说明：以下 **原文翻译** 与 **独立技术讲解** 分开，后者不属于作者原文。

## 一、完整中文翻译（Abstract）

推测解码（Speculative Decoding）通过将草稿生成（Draft Generation）与目标模型验证（Target Verification）解耦，加速大语言模型（Large Language Model，LLM）的推理过程。

尽管近期的并行草稿模型（Parallel Drafters）能够在单次前向传播中高效地提出较长的 Token 序列，但由于缺乏 Token 之间的依赖关系建模，其接受率会迅速下降。此外，不加区分地验证这些扩展的 Token 块，会在高拒绝风险的 Token 上浪费宝贵的批处理容量，从而严重降低高并发服务系统的吞吐量。

我们提出 **DSpark**，一种将高吞吐并行生成与自适应、负载感知验证统一起来的推测解码框架。

为了维持草稿质量，DSpark 采用一种半自回归架构：将并行主干网络与轻量级顺序模块相结合，引入块内依赖关系建模，并缓解后缀质量衰减（Suffix Decay）。

为了优化系统效率，DSpark 采用置信度调度验证（Confidence-Scheduled Verification），根据估计的前缀存活概率（Prefix Survival Probabilities）和引擎特定的吞吐性能曲线，动态调整每个请求的验证长度。

在涵盖多个不同领域的离线基准测试中，相比当前最先进的自回归和并行草稿模型，DSpark 显著提升了接受长度。

当 DSpark 部署到 DeepSeek-V4 服务系统并处理真实用户流量时，它成功减少了无效验证开销。

与已有的生产环境基线 MTP-1 相比，在相同吞吐量水平下，DSpark 将单用户生成速度提高了 **60%～85%**。

更重要的是，通过避免在严格交互性能约束下出现严重的吞吐量下降，DSpark 实现了此前无法达到的性能等级，推动了服务系统帕累托前沿（Pareto Frontier）的提升。

为了促进社区研究进展，我们开源了 DSpark 模型检查点，同时发布了 DeepSpec——一个以算法为中心的推测解码训练代码仓库。

## 二、技术讲解（非摘要原文）

### 1. 研究动机

- **草稿质量问题**：传统自回归 Drafter 将每一步依赖于先前采样 Token，条件一致性较强但串行开销与草稿长度近似成正比；并行 Drafter 一次生成一个块，减少草稿延迟，却可能将不同的语义分支混合，尤其在后缀出现质量衰减。
- **验证预算问题**：并行生成长草稿很快，但 Target Model 必须占用算力/批处理槽位验证它们；高拒绝风险的后缀可能抢占其他活跃请求的资源。固定验证长度忽略请求差异与 GPU 实时负载。

### 2. DSpark 的技术路径

(1) 采用 DFlash 式并行主干提取所有位置的隐藏特征和基本 Logits，然后用轻量顺序头依据此前采样 Token 调整后续分布。这个设计不意味着 Target 的验证变成串行；目标仍可以一次并行验证被选中的连续前缀。

(2) 置信度头预测给定此前草稿 Token 均已通过验证的条件接受概率 `c_k`。连续前缀的存活概率由链式法则给出：
`a_k = P(1…k 均被接受) ≈ ∏_{i=1}^k c_i`。
公式与精确定义在后文 Section 3.2.1–3.2.2；这里是预读说明，不是摘要原有公式。

(3) 用设备实测的 SPS(B)（批量 B 下每秒执行的解码步数）评估候选验证预算，最大化预期吞吐。含一个目标模型生成的 bonus token 时，对单条请求验证长度 ℓ，期望生成 Token 数是 `1 + Σ_{k=1}^ℓ a_k`。Section 3.2 对多请求给出正式优化目标；负载会改变最优 ℓ。

### 3. 与已读论文的关系

| 方法 | Draft 建模 | 主要限制 | 与 DSpark 的关系 |
|---|---|---|---|
| [Medusa](../2401.10774-medusa/README.md) | 多个 Token Head，通常结合树状验证 | 头预测间的依赖与验证树的成本；Typical Acceptance 非严格同分布 | DSpark 重点是块内轻量因果修正 + 线上调度 |
| [EAGLE](../2401.15077-eagle/README.md) | 隐藏特征自回归、Shifted Token、树状验证 | 多轮 Draft 串行性 | DSpark 保留大部分并行计算，只让轻量头串行 |
| [DFlash](../2602.06036-dflash/README.md) | Target 特征条件下的并行 Block Draft、KV Injection | 块内 Token 独立预测，后缀验收衰减；固定长块可能造成验证浪费 | DSpark 的并行主干直接以 DFlash 为技术基础 |

注意：DSpark 论文 Section 4 的 Eagle3 是 EAGLE-3 基线，不等于此前阅读的原版 EAGLE；其离线接受长度数据不等于线上端到端吞吐数据。

### 4. 摘要结果的度量边界

- 离线 Qwen3-4B/8B/14B 宏平均接受长度，DSpark 相较 Eagle3 分别提升 30.9%/26.7%/30.0%，较 DFlash 分别提升 16.3%/18.4%/18.3%（Introduction，Section 4；不在 Abstract 逐项列出）。
- 实际部署 V4-Flash 相较 MTP-1 在**匹配吞吐容量**下的单用户 TPS 提高 60%–85%；V4-Pro 为 57%–78%（Introduction/Section 5）。
- 严格 SLA 下的高百分比吞吐提升是基线接近容量悬崖时的相对值，不应解释为一般负载下的平均提升（见 Section 5.4）。

## 三、后续阅读索引

1. 下一单元：[1. Introduction](https://arxiv.org/pdf/2607.05147v1#page=1)，从 Abstract 后开始，第 1–3 页。
2. Figure 1 位于 PDF 第 5 页，属于 Section 3 Architecture；本 Abstract 单元没有图表，不虚构插图链接。
3. 正式公式 (1)–(12) 将按后续原文顺序阅读并解释。

**当前状态：Abstract 已完成；Introduction 尚未开始。**
