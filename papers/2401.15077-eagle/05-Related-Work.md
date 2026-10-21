# Section 5 · Related Work（完整译文与推测解码技术谱系）

> EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty  
> arXiv:2401.15077v3（原始 [HTML](https://arxiv.org/html/2401.15077v3) / [PDF](https://arxiv.org/pdf/2401.15077v3)），Section 5，PDF 第 8 页右栏—第 9 页左栏。阅读日期：2026-10-21。  
> **完整性：** 本节共三个正文段落，依原文顺序翻译，没有 Figure、Table 或论文新增公式。保留各方法与文献引用。下方分析为**独立分析或引自关联论文**，不能归入 EAGLE 原文结论。  
> **已读上文：** [4. Experiments](04-Experiments.md)；**下个单元：** 6. Conclusion。

## 一、Section 5 全文逐段中文翻译

### 第 1 段：一般性的模型加速路线（PDF 第 8 页）

目前，已经有大量关于语言模型加速的研究，涉及知识蒸馏（Distillation）（Hinton et al., 2015）、量化（Quantization）（Hubara et al., 2018；Shen et al., 2020；Kim et al., 2021；Zadeh et al., 2020；Zafrir et al., 2019）、剪枝（Pruning）（Gale et al., 2019；Sanh et al., 2020；Kurtic et al., 2022；Voita et al., 2019），以及创新的网络架构设计（Gu & Dao, 2023；Wu et al., 2020）。这些方法旨在降低每次前向传播（Forward Pass）的延迟。

### 第 2 段：从贪心加速到无损推测采样与有损替代方案（跨 PDF 第 8–9 页）

与我们的方法更为相似的是基于推测采样（Speculative Sampling）的框架。早期工作（Stern et al., 2018；Sun et al., 2021）加速了贪心解码（Greedy Decoding），而推测采样（Leviathan et al., 2023；Chen et al., 2023a）则将其扩展到非贪心采样（Non-greedy Sampling），并从理论上证明保持了原有的输出分布。

保证输出分布不变，使加速变得更加困难。许多研究探索了将有损加速作为一种折衷方案。例如，DistillSpec（Zhou et al., 2023）使用宽容函数（Lenience Function）修改接受概率；BiLD（Kim et al., 2023）在与目标 LLM 概率分布之间的距离指标低于某一阈值时接受草稿；Medusa（Cai et al., 2023）在截断时使用硬阈值（Hard Threshold）和依赖熵的阈值（Entropy-dependent Threshold）的最小值。

相比之下，EAGLE 不采用任何此类放宽条件（Relaxations），并且维持 LLM 的输出分布不变。

### 第 3 段：Drafting Phase 的分类（PDF 第 9 页）

基于推测采样的方法之间的主要区别，主要体现在草稿生成阶段（Drafting Phase）。

传统推测采样（Leviathan et al., 2023；Chen et al., 2023a）使用目标 LLM 的较小参数版本作为草稿模型。Self-Speculative Decoding（Zhang et al., 2023）在生成草稿时跳过目标 LLM 的某些层。SpecInfer（Miao et al., 2023）使用一组小型模型并行生成草稿。

Cascade Speculative Drafting（Chen et al., 2023b）和 Staged Speculative Decoding（Spector & Re, 2023）以级联方式组合具有不同开销的草稿模型。Online Speculative Decoding（Liu et al., 2023）使用查询分布训练草稿模型。

一些方法（Hooper et al., 2023；Fu et al., 2023；Yang et al., 2023b），以及 Medusa（Cai et al., 2023），不使用独立的目标 LLM，而是利用目标 LLM 的特征或权重生成草稿。REST（He et al., 2023）基于检索生成草稿。LLMA（Yang et al., 2023a）用于语法纠错等输入与输出重叠的任务，直接从输入中检索草稿。

**原文译文至此结束。原 PDF 紧接着为 Section 6 Conclusion。**

> 原文措辞注释：第三段写成“do not employ a separate target LLM”。为忠实呈现，以上按字面翻译为“不使用独立的目标 LLM”。但参照 Medusa、Lookahead 等实际方案，这不意味着没有主模型验证；更合理的技术意图是“不依赖另外一套独立的 Draft LLM”，疑似论文术语使用不精确。本注释是核对提示，不是对原句的静默修正。

---

## 二、技术谱系与五种重点方法（独立分析）

以下是根据 Section 5 的分类，并结合本篇前文和关联工作的公开原文进行的技术比较；不属于 Section 5 作者额外提供的实验数据。

| 方法 | Draft 来源 | 是否额外独立小 LLM | 主要预测方式 | Target 验证与分布保证 |
| --- | --- | --- | --- | --- |
| Standard Speculative Sampling | 较小同系列 LLM | 通常是 | 小模型按 Token 自回归 | 以 $\min(1,p/q)$ 接受，拒绝后残差采样；算法层面精确 |
| Self-Speculative Decoding | 原模型跳过部分层 | 否 | 共享主模型的跳层子网络 | 用完整目标模型验证；所引版本主张无损 |
| SpecInfer | 一组轻量推测模型 | 通常是，多模型 | 多模型产生候选、组织 Token Tree | 树式并行验证，论文主张保留生成质量 |
| Medusa | Target 的最后隐藏状态 | 否 | 多个附加预测头并行预测各未来位置 | Typical Acceptance 可放宽分布；若使用精确拒绝采样需另满足对应采样规则 |
| Lookahead | Target 自身及 Jacobi 迭代缓存 | 否 | 并行 Jacobi 生成 n-gram、并行验证 | 官方方案主要面向确定性/贪心精确解码；不要自动扩展为任意非贪心分布保持 |
| **EAGLE** | Target Feature + Shifted Token | 否（但需训练轻量预测头） | 单层 Decoder 自回归预测下一 Feature，通过 LM Head 生成 Token；多轮扩展树 | Tree Attention 一次目标 Forward 验证树；正确多轮精确推测采样保持分布 |

### 1. 标准 Speculative Sampling：独立小模型预测

令当前前缀为 $x_{1:t}$，Draft 提议 $q(\cdot\mid x_{1:t})$，Target 提供 $p(\cdot\mid x_{1:t})$。对采样到的 $x$，单步精确接受概率为：

\[
a(x)=\min(1,p(x)/q(x)).
\]

若拒绝，则从 $r(x)\propto[p(x)-q(x)]_+$ 重采样。分布守恒：

\[
q(x)a(x)+\Pr(\mathrm{reject})r(x)=p(x).
\]

这使 Draft 的分布误差**影响接受率和速度**，但不必改变最终生成分布。其工程瓶颈是需要加载、维护、运算一个足够快且能与 Target 对齐的辅助 Draft Model；若 Draft 太大，额外 Forward 可能抵消 Target 减少的步数。EAGLE 保留这一正确验证范式，把草稿生成改为低成本特征级适配器。

### 2. Self-Speculative Decoding：同一 LLM 跳层

[Draft & Verify: Lossless Large Language Model Acceleration via Self-Speculative Decoding](https://arxiv.org/abs/2309.08168)（Zhang et al., 2023）在 Draft 阶段选择性跳过目标模型的部分中间层，减少单次草稿前向延迟，再由未跳层的完整模型对候选验证。其思路是 **共享主模型权重，以运算较少的近似子网络替代独立小模型**；原论文描述为无需额外神经网络训练及额外独立模型显存。与 EAGLE 相比：
- Self-Spec 优势：没有单独训练一个 Draft Head 的前置工作；主要增加跳层路径和验证工程逻辑。
- Self-Spec 代价：跳层造成的预测分布偏差可能降低接受率，需优化跳层策略。
- EAGLE 优势：额外拟合 Target 的高层 Feature 转移；Shifted Token 特别缓解多 Token 分支不确定性。
- EAGLE 代价：需要训练、存储并执行附加的 Autoregression Head。

这里的“自推测”强调**模型本身兼任 Draft/Target**，不是“所有自回归草稿预测都叫 Self-Speculative Decoding”。

### 3. SpecInfer：候选树与批量验证

[SpecInfer](https://arxiv.org/abs/2305.09781)（Miao et al., 2023）使用多个轻量推测模型，构造包含不同候选续写的 Token Tree。目标 LLM 用**树式并行验证**减少每个续写序列单独跑完整大模型的成本。它与 EAGLE 最相关的共同点是 Tree-structured Speculative Verification 与树状 Attention：某个候选位置只应看到公共历史和自己路径的祖先节点，不能看到其他分支。

概念性 Tree Mask：

\[
M_{u,v}=\begin{cases}
0,&v\in\mathrm{Ancestors}(u)\cup\{u\}\ \text{或在该节点可见的公共因果前缀中},\\
-\infty,&\text{其他节点}.
\end{cases}
\]

这只是解释性结构，具体实现仍依赖 RoPE/Position、KV 分配、Tree Node 索引和 Verification 算法。**EAGLE 主要新意不在“第一次提出树验证”，而在草稿预测层**：它复用目标 Feature、在 Token 时间步上前移条件，并用一个轻量 Decoder 迭代预测下一 Feature。EAGLE 的 Appendix A.2 明确借鉴 SpecInfer 式多轮采样。

### 4. Medusa：多头独立预测 Token vs EAGLE 的逐步条件化 Feature

[Medusa](https://arxiv.org/abs/2401.10774)（Cai et al.）在主模型最后隐藏状态之上加多个 Token Prediction Heads，各 Head 负责不同未来位置，形成多种候选后组合成树并验证。其典型方案会用熵相关阈值和硬阈值选择“足够合理”的候选；这类放宽接受规则可以保持生成质量接近，但**不能凭此认定非贪心采样概率与 Target 完全一致**。

EAGLE Section 3 Figure 5 对比：Medusa 从同一 $f_i$ 独立预测后面几个 Token；EAGLE 先经 Feature+Shifted Token 预测 $\hat f_{i+1}$，由共享 LM Head 采样后续 Token，再把采样结果用于下一轮 Feature 预测。这意味着 EAGLE 能对**已经选定的采样分支**继续预测，而不是只对初始 Feature 为多个未来位置做独立 Head 预测。

重要版本边界：后续公开的 Medusa 论文也说明了可选择标准 Rejection Sampling 进行校正，并存在 Medusa-1（冻结主干）与 Medusa-2（联合训练）的不同设置；因此“Medusa 一律无法精确分布保持”是不准确的。**EAGLE Section 5 批评的是作者当时选取的 Typical Acceptance/截断模式**。更多资料：[Medusa 公开论文](https://proceedings.mlr.press/v235/cai24b.html)、[项目实现](https://github.com/FasterDecoding/Medusa)。

### 5. Lookahead：Jacobi + n-gram，无独立 Draft

[Lookahead Decoding](https://lmsys.org/blog/2023-11-21-lookahead-decoding/)（Fu et al., 2023）利用 Jacobi Iteration 并行探索多个未来位置的 Token 候选，从迭代轨迹中抽取、缓存 n-gram，再用同一主模型验证可用的 n-gram。它不需要独立小模型，也不需要专门训练 Draft Head。

和 EAGLE 的差异：
- Lookahead 的候选由 Jacobi 迭代轨迹及 n-gram 构造；EAGLE 在连续隐藏特征空间训练带 Shifted Token 的预测器。
- Lookahead 通过额外并行 FLOPs 换取较少顺序步数；EAGLE 通过低成本特征头和树候选提高目标验证的平均有效推进。
- Lookahead 的原始官方说明强调**精确的贪心结果**。不能据此推出一般 Temperature>0 场景也实现完全同分布采样；这正是 EAGLE 与其比较时需要分开 T=0 与 T=1 的原因之一。

### 6. 其他在 Section 5 被引用的方法

| 方法 | 作者/引用 | 此节实际明确指出的草稿思想 | 与 EAGLE 的主要差别 |
| --- | --- | --- | --- |
| DistillSpec | Zhou et al., 2023 | 使用 Lenience Function 修改接受概率 | 在该 Relaxed 模式中以分布偏差换取接受收益；EAGLE 保留精确拒绝采样 |
| BiLD | Kim et al., 2023 | 分布距离低于阈值时接受 | 阈值式接受而非严格 $p/q$ 校正 |
| Cascade Speculative Drafting | Chen et al., 2023b | 不同计算开销的草稿模型级联 | 多级计算预算决策，而非特征空间递推 |
| Staged Speculative Decoding | Spector & Re, 2023 | 将不同开销 Draft 分阶段使用 | 多级 Draft 设计 |
| Online Speculative Decoding | Liu et al., 2023 | 用真实查询分布训练 Draft | 主要优化训练数据适配，而非 EAGLE 的 Shifted Token 结构 |
| SPEED / 相关流水线工作 | Hooper et al., 2023 | 利用模型内部结构或权重辅助草稿 | 从执行/流水线角度降低依赖，论文此节没有展开更多细节 |
| Predictive Pipelined Decoding | Yang et al., 2023b | 利用模型内部预测与流水线降低延迟 | 强调流水线执行，不同于特征级 Draft 学习 |
| REST | He et al., 2023 | 检索式草稿 | 复用外部语料片段，不必训练未来 Feature 预测器 |
| LLMA | Yang et al., 2023a | 当输入与输出重叠时直接从输入检索草稿 | 擅长文本重合任务，不是通用 Feature 外推 |
| Greedy Blockwise Decoding | Stern et al., 2018 | 早期加速贪心解码 | 正确性对象是确定性解码路径，不自动代表随机分布保持 |
| Shallow Aggressive Decoding | Sun et al., 2021 | 早期贪心解码加速 | 面向特定任务/贪心过程，与 EAGLE 的泛化取向不同 |

此表中的技术区别是对源文的整理，不代表作者在 Section 5 对所有方法均进行了独立实验或得出普适排名。若进一步研究各论文技术细节，应重新阅读其原始方法和实验设置。

## 三、无损 / 有损：数学与评价指标边界（独立分析）

首先区分两种经常混淆的“无损”：

1. **Greedy / deterministic exactness**：固定输入和解码配置时，结果与 Target 贪心输出序列一致。
2. **Sampling distribution exactness**：对 Temperature、Top-p 等对应的有效随机采样分布，每个前缀下最终 Token 的分布与 Target 一致，因而完整生成序列的联合概率保持一致。

“生成质量看起来相同”不等于第二种数学保证。对于概率校正，单步 Draft $q$、Target $p$ 满足：

\[
a(x)=\min(1,p(x)/q(x)),\qquad r(x)\propto[p(x)-q(x)]_+.
\]

于是

\[
q(x)a(x)+(1-\alpha)r(x)=p(x),\quad
\alpha=\sum_x\min(p(x),q(x))=1-\mathrm{TV}(p,q).
\]

放宽 Accept 规则（如阈值而非 $p/q$）通常无法沿用上述证明；必须额外证明新规则在拒绝补偿后仍严格守恒，否则只能主张近似生成质量。这也解释 EAGLE 为何强调使用标准精确 Verification，尽管其 Draft 预测本身近似。

## 四、EAGLE 的真正研究定位

传统推测采样主要难点是“怎样找到一个小而好的独立 Draft LLM”；Self-Spec 解决独立模型的装载和训练，Medusa 解决并行预测多个 Token，Lookahead 解决无需训练额外 Draft 的候选来源，SpecInfer 提供树式验证框架。

**EAGLE 在此基础上提出不同的 Draft 建模层级：** 预测目标 LLM 的倒数第二层连续 Feature，利用已采样且向前平移一个时间步的 Token 明确分支，并通过冻结的目标 LM Head 变回 Token 分布。

这不是在否定前述方法，而是选择优化了一个新层次的组件：**条件信息更充分的低成本草稿生成器**。Tree Attention 和正确的概率校正与先前方法有继承关系。

结合[此前已读 DFlash](../2602.06036-dflash/README.md)，还能看到后续技术演进：EAGLE 的 Feature-level Autoregressive Draft 仍依赖草稿深度的顺序 Forward；DFlash 的 Block Diffusion Draft 进一步尝试消除块内顺序依赖，并通过 Target 多层 Feature 和 Layerwise KV Injection 改善 Draft 质量。两篇论文报告的加速数字不应跨模型、硬件、解码和 Serving 配置直接排序。

## 五、建议的跨论文精读入口（外部原始资料）

- [Leviathan et al. — Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)
- [Chen et al. — Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318)
- [Self-Speculative Decoding — Draft & Verify](https://arxiv.org/abs/2309.08168)
- [SpecInfer](https://arxiv.org/abs/2305.09781)
- [Medusa](https://arxiv.org/abs/2401.10774)
- [Lookahead Decoding](https://lmsys.org/blog/2023-11-21-lookahead-decoding/)
- [EAGLE-3（另一篇论文，不能与本篇混同）](https://arxiv.org/abs/2503.01840)

## 本节阅读进度

- 完成 Section 5 的三个完整段落（PDF 第 8–9 页）。
- 本节无新图、表或原始公式；额外公式属于讲解。
- 下一阅读单元：**6. Conclusion**（PDF 第 9 页），之后按论文顺序处理 Appendices A、B。
