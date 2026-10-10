# 2. Related Work（相关工作）

- 原文：[DFlash: Block Diffusion for Flash Speculative Decoding, arXiv:2602.06036v2](https://arxiv.org/html/2602.06036v2)
- PDF 页码：第 2–3 页（Section 2 跨页）
- 阅读日期：2026-10-10
- 进度：2.1、2.2、2.3 已完整翻译及讲解
- 图表：Section 2 不包含独立 Figure 或 Table；后续第 3 节 Figure 3 不应提前作为本节原图。
- 内容规范：下方**逐段翻译**来自论文，**技术讲解**属独立分析；涉及精确推测采样的公式为补充知识，不是论文 Section 2 的原始公式。

## 2.1 Speculative Decoding — 推测解码

### 原文逐段完整翻译

**第 1 段。** 推测解码通过缓解自回归生成的串行瓶颈，来加快大语言模型的推理。早期方法（Leviathan et al., 2023）采用一个较小的草稿模型提出候选 Token 序列，再由更大的目标模型并行验证。Medusa（Cai et al., 2024）通过在基础大语言模型上增加多个预测头，并采用树状注意力（Tree Attention）进行并行验证，消除了外部草稿模型的需要。EAGLE 系列（Li et al., 2025c; 2024; 2025b）进一步利用冻结目标模型的特征级上下文（Feature-level Context）改进推测解码。EAGLE-1 预测未来隐藏状态的分布以提高接受率；EAGLE-2 引入自适应草稿树；EAGLE-3 改进训练目标，从而进一步扩展加速效果。

**第 2 段。** 尽管有这些进展，现有方法大多数仍然依赖自回归草稿生成，而这种方式本质上是串行的，因而限制了加速效果。

### 技术讲解（独立分析）

#### 标准推测采样的工作机制

设目标模型为 \(p\)，草稿模型为 \(q\)。Draft 提出候选 Token \(x\)，目标模型采用接受概率

\[
\alpha(x)=\min\left(1,\frac{p(x)}{q(x)}\right).
\]

如果拒绝，必须使用恰当的残差分布修正；仅凭逐 Token 检查是否相同并不能保证分布无损。对标准单分支推测采样，修正分布的一个常见形式为

\[
r(x)=\frac{[p(x)-q(x)]_+}{\sum_y[p(y)-q(y)]_+}.
\]

上述公式是帮助理解精确推测采样的**补充知识**，不出现在原文 Section 2；Medusa/EAGLE 等具有不同候选结构，需按各自正确算法进行验收。

#### 方法演进

| 方法 | 本节原文说明的主要策略 | 有助于理解的设计维度 |
| --- | --- | --- |
| 传统推测解码（Leviathan et al.） | 小 Draft Model 提出候选，大 Target Model 并行验证 | 快速 Draft + 精确验证 |
| Medusa | 目标模型上的多个预测头与树状注意力 | 不再需要外部 Draft Model |
| EAGLE-1 | 预测未来隐藏状态分布 | 特征级预测，提高接受率 |
| EAGLE-2 | 自适应草稿树 | 对候选预算进行动态配置 |
| EAGLE-3 | 调整训练目标 | 提高接受与实际加速 |
| DFlash | 由目标上下文特征引导的并行块扩散 Draft | 避免自回归 Draft 的顺序依赖 |

这几条路线优化的并非相同环节：多头/树形结构关注候选覆盖与验证，特征级方法改善 Draft 质量，而 DFlash 还特别关注 Draft 自身的并行执行。

## 2.2 Diffusion Language Models — 扩散语言模型

### 原文逐段完整翻译

**第 1 段。** 扩散大语言模型（Diffusion Large Language Models，dLLMs）通过并行预测被掩码的 Token，提供了替代自回归生成的方式。LLaDA（Nie et al., 2025）首次将 dLLM 扩展至数十亿参数规模，其性能可与 LLaMA-3.1-8B（Grattafiori et al., 2024）相当。然而，完全并行的扩散模型存在固定长度生成的问题，且缺乏高效的 KV Cache 支持。块扩散模型（Block Diffusion Models；Arriola et al., 2025）通过逐块对序列进行去噪，结合并行性与自回归结构，解决了这些问题。沿着这一思路，Fast-dLLM v2（Wu et al., 2025）和 SDAR（Cheng et al., 2025）将预训练的自回归大语言模型适配为块扩散变体，既能并行生成，也能在特定任务上保持生成质量。不过，现有 dLLM 总体上仍不及先进的自回归模型，并且常常需要多个去噪步骤，限制了实际推理速度。

### 技术讲解（独立分析）

#### 自回归因子分解 vs. Masked Diffusion

自回归建模将序列概率分解为

\[
P(x_{1:n})=\prod_{i=1}^{n}P(x_i\mid x_{<i}).
\]

它意味着按 Token 顺序解码：第 \(i\) 个位置的采样须依赖已确定的前序 Token。

Masked Diffusion 则在受噪/Mask 的上下文中预测多个位置，可以通过多次更新逐步改善整个生成块的内容。在具体 dLLM 中，去噪更新、Mask 调度与训练目标可能不同，不能将一个简化过程当作所有模型的统一公式。

#### 固定长度与 KV Cache

全序列 Masked Diffusion 通常要预先指定待生成区域长度；迭代去噪又可能改变前一步的当前序列表示，在完全双向注意力下难以直接沿用传统 AR 解码的稳定 KV Cache。Block Diffusion 的关键是**块间按顺序推进、块内做并行去噪**：完成的历史块有机会被稳定缓存，当前块则允许多个 Token 共同建模。但能否高效利用 Cache 取决于具体模型的因果约束和实现方式，不能认为任意块扩散天然完全兼容 AR KV Cache。

#### 为什么 dLLM 适合当 Draft，而非一定要替换 Target

当扩散模型独立生成完整答案时，需要在自身计算成本与最终准确性之间权衡，通常要多步去噪。推测解码把**输出质量保障**交给自回归 Target，允许小型并行 Draft 采用更激进的单步预测策略：质量不足会降低接受长度，但正确的 Target 验证机制不会使最终分布偏离 Target。

## 2.3 Diffusion-based Speculative Decoding — 基于扩散的推测解码

### 原文逐段完整翻译

**第 1 段。** 近期研究开始探索在推测解码中将扩散模型作为草稿生成器。TiDAR（Liu et al., 2025）联合训练扩散目标与自回归目标，使得模型能够通过扩散进行并行“思考”（thinking），并通过自回归进行顺序“表达”（talking）；不过其最终生成质量尚未达到无损。

**第 2 段。** 其他方法将自回归模型改造为能够以扩散风格生成草稿的模型。Samragh et al.（2025）观察到，自回归大语言模型隐式编码了未来 Token 的信息，因此训练一个 LoRA 适配器来实现并行草稿生成，同时保留基础模型用于验证。

**第 3 段。** DiffuSpec（Li et al., 2025a）与 SpecDiff-2（Sandler et al., 2025）采用大型预训练扩散大语言模型（dLLMs）作为推测解码草稿模型，并通过推理时搜索（Inference-time Search）或训练—测试对齐（Train–Test Alignment）提高候选接受率。然而，这些方法依赖庞大的草稿模型（如 7B 参数规模），带来明显的显存与延迟开销。尽管它们能取得较长的接受长度，但高额 Draft 成本在真实服务场景中经常抵消了潜在加速收益。

### 技术讲解（独立分析）

#### 三类研究路线

| 路线 | 代表论文 | 原文强调的优势或问题 | 与 DFlash 的区别 |
| --- | --- | --- | --- |
| 联合训练扩散与 AR 目标 | TiDAR | 并行 thinking、顺序 talking；论文指出输出尚非 lossless | DFlash 将扩散限于 Draft，并保留 AR Target 的正确验证 |
| 原 AR 模型加适配器 | Your LLM Knows the Future（Samragh et al.） | 目标隐藏特征具有潜在多 Token 预测信息，借 LoRA 适配并行 Draft | DFlash 显式构建轻量块扩散 Draft 并使用 Target 特征作条件 |
| 大型预训练 dLLM 草稿 | DiffuSpec、SpecDiff-2 | 接受长度较高但额外显存和 Draft 延迟大 | DFlash 着重轻量化并通过逐层 KV 特征注入提高 Draft 质量 |

#### 接受长度不是唯一优化目标

在推测解码中，最终收益同时依赖 \(T_{\mathrm{draft}}\)、\(T_{\mathrm{verify}}\) 与平均每轮通过的 Token 数 \(\tau\)。下一节的论文公式将给出严谨定义；直观上，如果更大的 Draft Model 使 \(\tau\) 增长，但每轮 Draft 时间上涨更明显，整体加速反而可能减少。

**独立数值示例，非论文测量：** 假设方案 A 每轮 Draft 2ms、验证 5ms、接受 4 Token，则均摊成本约 1.75ms/Token；方案 B 每轮 Draft 10ms、验证 5ms、接受 6 Token，成本却是 2.5ms/Token。仅比较“接受 6 个比接受 4 个更多”就会得出错误结论。此例不包含可能的 Bonus Token，也未考虑实际的批处理与重叠调度，目的只是解释 trade-off。

#### DFlash 的创新切入点

- **目标特征条件化：** 不让小 Draft Model 自行重做大型 Target 的全部语义建模。
- **块级并行 Draft：** 不依赖串行自回归生成多步候选。
- **保留 Target 验证：** 借助正确验证/残差采样保持目标分布。
- **轻量结构：** 避免以 7B 参数大 Draft 换取较长接受长度的高代价。

以上是结合引言与 Section 2 进行的结构化解读；具体如何融合 Target 隐藏状态并注入 Draft 各层 KV，见 Section 4.1 与 Appendix A.3。

## 跨论文知识关联

当前仓库的知识索引只有 DFlash 的正式精读条目。以下论文均由 DFlash 原文参考文献提供，属于待扩展的关联，而不是已经完成独立核验的论文精读：

- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)：基础推测解码范式。
- [Medusa](https://arxiv.org/abs/2401.10774)、[EAGLE-1](https://arxiv.org/abs/2401.15077)、[EAGLE-2](https://arxiv.org/abs/2406.16858)、[EAGLE-3](https://arxiv.org/abs/2503.01840)：候选头、草稿树与特征级预测的演进。
- [LLaDA](https://arxiv.org/abs/2502.09992)、[Block Diffusion](https://arxiv.org/abs/2503.09573)、[Fast-dLLM v2](https://arxiv.org/abs/2509.26328)、[SDAR](https://arxiv.org/abs/2510.06303)：从全序列扩散到块扩散。
- [TiDAR](https://arxiv.org/abs/2511.08923)、[Your LLM Knows the Future](https://arxiv.org/abs/2507.11851)、[DiffuSpec](https://arxiv.org/abs/2510.02358)、[SpecDiff-2](https://arxiv.org/abs/2511.00606)：扩散与推测解码的融合思路。

## 本节总结与下节阅读断点

本节完成三条方法线的梳理：**推测解码**提升顺序模型效率，**块扩散**提供块内并行生成能力，**扩散推测解码**尝试合二为一。DFlash 瞄准的具体空缺是：“既保持草稿轻量与快速，又利用 Target 表示提高草稿质量，并采用 Target 验证保证最终输出分布。”

已完成：2. Related Work（2.1 / 2.2 / 2.3）。**下一阅读单元：3. Preliminaries**，将依原文逐段翻译并推导式 (1)–(3)，展示并解释论文原始 Figure 3。
