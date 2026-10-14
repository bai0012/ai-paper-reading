# 1. Introduction：完整译文、Figure 2–4 原图图注与机制分析

> EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty  
> arXiv:2401.15077v3 · 原文 PDF 第 1–3 页 · 阅读日期：2026-10-14  
> 原文链接：<https://arxiv.org/html/2401.15077v3>  
> 本文件严格区分**逐段译文**和**独立技术分析**；Figure 2–4 来自用户上传的原始 PDF，在对话中直接裁图展示。二进制图像如未上传 GitHub，本文使用可验证的原始 PDF 页码链接而不使用虚假相对路径。

## 1. Introduction 逐段完整中文翻译

### 原文第 1 段：推测采样背景

自回归解码（Autoregressive Decoding）是大语言模型（LLMs）事实上的标准生成方式。由于它需要按顺序逐个生成 Token，因此生成过程缓慢且成本高昂。基于推测采样（Speculative Sampling）的方法（Leviathan et al., 2023；Chen et al., 2023a）通过将生成过程划分为低成本的草稿生成阶段（Draft Stage）和针对草稿 Token 的并行验证阶段（Verification Stage）来解决这一问题，使得一次 LLM 前向传播就能够验证多个 Token。这些方法通过在每次前向传播中生成多个 Token 来加速文本生成。更重要的是，验证阶段能够确保最终生成文本的概率分布与原始 LLM 直接解码的结果完全一致，从而保持生成内容的原有特性。

### 原文第 2 段：独立 Draft Model 的局限

应用推测采样的关键，在于找到一个能够模拟原始 LLM 功能、但具有更低推理延迟的 Draft Model。通常的做法是从同一个 LLM 系列中选择参数量更小的模型。例如，LLaMA2（Touvron et al., 2023）系列包含 7B、13B 和 70B 参数规模的模型。使用 7B 模型作为 70B 模型的 Draft Model 是可行的，但对于该系列中最小的 7B 模型，要找到一个合适的 Draft Model 就比较困难。一种替代方案是使用 TinyLLaMA（Zhang et al., 2024），但对于经过指令微调的模型而言，这并不可行，因为 LLaMA2-Chat 与 TinyLLaMA-Chat 的指令模板并不一致。尽管 7B 模型有作为 Draft Model 的潜力，但其自身较高的运行开销会削弱推测解码带来的加速收益。专门为推测采样训练一个新的、规模合适的 Draft Model 也不是理想方案，因为训练成本很高：TinyLLaMA 使用了 3,000B（3 万亿）Token 进行训练，而 EAGLE 仅使用 2–4B（20 亿–40 亿）Token 进行训练。

### 原文第 3 段：Lookahead、Medusa 与接受率问题

提高推测采样加速效果的关键，在于降低时间开销，并提高目标 LLM 对草稿 Token 的接受率（Acceptance Rate）（Chen et al., 2023b；Xia et al., 2023；Santilli et al., 2023）。许多方法都致力于降低草稿生成阶段的运行开销。Lookahead（Fu et al., 2023）使用 n-gram 和 Jacobi Iteration（雅可比迭代）；Medusa（Cai et al., 2023）则利用一组 MLP，基于原始 LLM 的倒数第二层特征来预测 Token。这些方法显著降低了草稿生成的延迟，从而改善了整体加速效果。但是，草稿预测结果的准确率较低限制了它们的效果：Medusa 的准确率约为 0.6，而 Lookahead 的准确率甚至更低。相比之下，我们的方法能够达到约 0.8 的准确率。

### Figure 2：非贪心解码

**原图：** [PDF 第 2 页 Figure 2](https://arxiv.org/pdf/2401.15077v3#page=2)；对话已提取并显示，未确认已将 PNG 提交到仓库。

**图注完整翻译：** 图 2：在非贪心解码（Non-greedy Decoding，temperature=1）设置下，MT-bench 上的加速比。Lookahead 仅适用于贪心解码，而 Medusa 的非贪心生成不能保证无损（Lossless）性能。因此，EAGLE 不与这两种方法进行比较。

图中准确数值（提取自原始图）：

| 目标模型 | EAGLE | Speculative Sampling | DistillSpec | Vanilla |
| --- | ---: | ---: | ---: | ---: |
| Vicuna 7B | 2.13× | N/A | 未绘制 | 1.00× |
| Vicuna 13B | 2.32× | N/A | 未绘制 | 1.00× |
| Vicuna 33B | 2.40× | 1.22× | 1.09× | 1.00× |
| LLaMA2-Chat 7B | 2.22× | N/A | 未绘制 | 1.00× |
| LLaMA2-Chat 13B | 2.68× | N/A | 未绘制 | 1.00× |
| LLaMA2-Chat 70B | 2.67× | 2.06× | 1.84× | 1.00× |

### 原文第 4 段：引出两项观察

为了克服上述局限，我们提出 EAGLE（Extrapolation Algorithm for Greater Language-model Efficiency），一种高效的推测采样方法。该方法建立在以下两个观察之上。

### 原文第 5 段：Observation 1——特征级自回归更简单

**首先，在特征层面进行自回归建模，比在 Token 层面进行自回归建模更加简单。**在本文中，“特征（Features）”指原始 LLM 的倒数第二层特征，位于 LM Head 之前。与 Token 序列相比，特征序列表现出更强的规律性。Token 序列本质上是自然语言的一种简单转换形式。在特征层面进行自回归预测，然后通过原始 LLM 的 LM Head 将特征转换为 Token，相比直接通过自回归方式预测 Token，可以取得更好的结果。如 Figure 4 所示，特征级自回归具有更好的性能，加速比从 1.5× 提升到了 1.9×。

### 原文第 6 段：Observation 2——采样不确定性

**其次，采样过程固有的不确定性会显著限制对下一个特征进行预测的性能。**在文本生成中，目标 LLM 首先预测 Token 的概率分布，然后根据该分布进行采样，因此会引入随机性。而特征是高维且连续的，不能以相同的方式处理。如 Figure 3 所示，采样不同的 Token，例如“am”或“always”，会产生不同的特征序列，从而使特征级自回归预测产生歧义。Medusa 在预测相隔多个位置的 Token 时也面临类似的问题：对于输入特征 $f_I$，无法确定其真正的预测目标应该是 $p_{\mathrm{am}}$ 还是 $p_{\mathrm{always}}$。为了解决这一问题，EAGLE 将提前一个时间步的 Token 序列作为 Draft Model 的输入，其中包含了实际的采样结果。在 Figure 3 的例子中，这意味着根据 $f_I$ 和 $t_{\mathrm{always}}$ 预测 $f_{\mathrm{always}}$，以及根据 $f_I$ 和 $t_{\mathrm{am}}$ 预测 $f_{\mathrm{am}}$。如 Figure 4 所示，在解决这种不确定性之后，加速比进一步从 1.9× 提高到了 2.8×。

### Figure 3：特征序列中的不确定性

**原图：** [PDF 第 2 页 Figure 3](https://arxiv.org/pdf/2401.15077v3#page=2)；在对话中已直接从原始 PDF 裁图展示。

**图注完整翻译：** 图 3：特征序列中的不确定性。在 $f_I$ 之后出现的下一个特征取决于采样结果，不能仅仅根据 $f_I$ 来确定。因为在 Token“I”后面既可能出现“always”，也可能出现“am”，从而产生两个不同的分支。

图中详细条件概率：$p_I(\mathrm{am})=0.6$，$p_I(\mathrm{always})=0.4$；若采样“always”，后续 $p_{\mathrm{always}}(\mathrm{begin})=0.8$、$p_{\mathrm{always}}(\mathrm{look})=0.2$；若采样“am”，后续 $p_{\mathrm{am}}(\mathrm{excited})=0.3$、$p_{\mathrm{am}}(\mathrm{ready})=0.7$。

### Figure 4：不同 Draft 输入的加速比与准确率

**原图：** [PDF 第 3 页 Figure 4](https://arxiv.org/pdf/2401.15077v3#page=3)；在对话中已直接从原始 PDF 裁图展示。

**图注完整翻译：** 图 4：在 Temperature=0 的条件下，基于 Token、Feature 和 Feature & Shifted-Token 的 Draft Model 的准确率与加速比。实验以 Vicuna 7B 作为目标 LLM，在 MT-bench 上进行测试。Feature & Shifted-Token 指同时使用特征序列以及提前一个时间步的 Token 序列作为输入。

### 原文第 7 段：评测、对比与互补技术

我们分别使用 MT-bench、HumanEval、GSM8K 和 Alpaca 数据集，在对话、代码生成、数学推理和指令遵循任务上进行了实验。测试的 LLM 包括 Vicuna 和 LLaMA2-Chat 系列的全部模型，以及 Mixtral 8x7B Instruct。对于 LLaMA2-Chat 70B，EAGLE 实现了 2.7–3.5 倍的加速比、约两倍的吞吐量提升，并从理论上保证生成文本的概率分布保持不变。Figure 1 和 Figure 2 展示了 EAGLE 在 MT-bench（Zheng et al., 2023）上的表现。MT-bench 是一个高度贴近真实应用的基准测试，模拟了实际使用场景，其中包括类似 ChatGPT 对话的多轮指令交互。我们选择这一基准，是因为当前最先进的方法（包括 Lookahead 和 Medusa）也使用它来展示其加速比。这使我们能够与这些方法进行公平、直接的比较。与最近提出的基于推测采样的框架 Lookahead 和 Medusa 相比，EAGLE 分别实现了 1.7–2.1 倍和 1.5–1.6 倍的相对加速。EAGLE 可以与量化（Quantization）、编译（Compilation）等其他加速或提高吞吐量的方法协同使用。将 EAGLE 与这些技术结合，可以进一步降低 LLM 系统的运行成本。例如，结合 gpt-fast（PyTorch Labs, 2023），EAGLE 能够在单张 RTX 3090 GPU 上将 LLaMA2-Chat 7B 的解码速度提升至 160.4 tokens/s。

### 原文第 8 段：训练成本

EAGLE 的训练成本很低。对于 LLaMA2-Chat 70B，EAGLE 使用 ShareGPT 数据集中的不超过 70,000 条对话，训练一个参数量小于 1B 的 Decoder Layer。整个训练过程可以在 4 张 A100（40GB）GPU 上于 1–2 天内完成。对于 7B、13B 和 33B 模型，EAGLE 的训练甚至可以在一个 RTX 3090 服务器节点上于 1–2 天内完成。在实际应用中，EAGLE 只需要完成一次训练，便能够持续为后续查询提供推理加速。随着查询数量增加，EAGLE 的摊销训练成本（Amortized Training Cost）将变得可以忽略不计。

### 原文第 9 段：通用性与可靠性

除性能优势之外，EAGLE 还具有以下优点。

- **通用性（Generality）：**EAGLE 原则上适用于任何自回归 LLM。我们已将 EAGLE 应用于 LLaMA2-Chat（7B、13B、70B）、Vicuna（7B、13B、33B）以及 Mixtral 8x7B Instruct，并在 MT-bench、GSM8K、HumanEval 和 Alpaca 数据集上进行了零样本评估。EAGLE 遵循 LLM 研究社区常用的 Zero-shot/Few-shot 评测设置。所有实验均使用同一组权重，这些权重仅在 ShareGPT 数据集上训练得到，没有在评测数据集上进行额外训练。该方法仅需向 LLM 添加一个轻量级插件（单个 Transformer Decoder Layer），因此能够方便地部署到生产环境中。
- **可靠性（Reliability）：**EAGLE 不需要对原始 LLM 进行任何微调。无论在贪心解码还是非贪心解码场景下，EAGLE 都能从理论上保证输出概率分布保持不变。这与 Lookahead 和 Medusa 有明显区别，因为这两种方法要么只关注贪心解码，要么无法在相应设置下保证输出分布不变。

> Section 1 正文译文结束。下一节为 2. Preliminaries（PDF 第 3 页）。

---

## 独立技术讲解：Feature Uncertainty、Shifted Token、Figure 4 消融

### 1. Feature Uncertainty 是一种条件多值映射

在 Figure 3 中，当前前缀为“I”，Target 的特征为 $f_I$。它可能采样后继 Token “am”（概率 0.6）或“always”（概率 0.4）。分别对应新的真实特征 $f_{\mathrm{am}}$ 和 $f_{\mathrm{always}}$，而后续的 LM Head 分布也截然不同。这意味着只给 Draft 输入 $f_I$，**一个输入对应两个监督目标**。

用 MSE 说明（仅为直观推导，论文实际训练使用 Smooth L1）：

\[
g^*(f_I)=\mathbb{E}[F_{\mathrm{next}}\mid f_I]
=0.6 f_{\mathrm{am}}+0.4 f_{\mathrm{always}}
\]

这个条件均值不一定对应任何一个真实分支。后续 LM Head 会放大某些隐藏特征偏差，导致 draft logits 和候选 token 不准确。该问题的关键不是普通噪声，而是采样后存在离散分支。

在随机变量 $H$=历史特征，$T$=实际采样 Token，$F$=下一特征的设定下：

\[
\operatorname{Cov}(F\mid H)=
\mathbb{E}[\operatorname{Cov}(F\mid H,T)\mid H]+
\operatorname{Cov}(\mathbb{E}[F\mid H,T]\mid H)
\]

增加 $T$ 作为条件能区分各个分支，对平方误差的理论最优预测而言，加入条件不会提高均方误差；工程中能否实现取决于网络与训练。这是独立数学解释，不是论文原文定理。

### 2. Shifted Token 的精确时间索引

目标模型在 $i$ 位置已有 $f_i$，能通过目标 LM Head 得到下一个 Token 分布：

\[
p_{i+1}=\operatorname{Softmax}(\mathrm{LMHead}(f_i)),
\qquad t_{i+1}\sim p_{i+1}
\]

此时 *已采样* 的 $t_{i+1}$ 被用作条件以预测 $f_{i+1}$，不属于提前窥视未产生的 Token：

\[
\hat f_{i+1} =
g_\phi(F_{1:i},T_{2:i+1}),
\qquad
\hat p_{i+2}=
\operatorname{Softmax}(\mathrm{LMHead}(\hat f_{i+1}))
\]

- 特征 $F_{1:i}$ 是到位置 $i$ 为止的历史；
- Token $T_{2:i+1}$ 相对于它**整体向前移动一个时间步**；
- 这是为了让 $t_{i+1}$ 与预测目标 $f_{i+1}$ 在同一训练位置形成条件，而不是让模型多看一个尚未采样的真实答案；
- 这依然是自回归式 Draft，新的预测特征与采样 Token 将用于下一迭代。

维度（进一步在 Section 3 验证）：令 batch=$B$、历史长度=$L$、hidden=$d$，$F\in\mathbb{R}^{B\times L\times d}$、$T\in\mathbb{Z}^{B\times L}$，共享 Embedding 得到 $E\in\mathbb{R}^{B\times L\times d}$，融合后 $[F;E]\in\mathbb{R}^{B\times L\times2d}$，经轻量预测头映射回 $\mathbb{R}^{B\times L\times d}$。

### 3. Figure 4 的消融实验：三种输入

原图在 MT-bench、Vicuna 7B、Temperature=0 下，比较三个方案，随着训练 Epoch 增加，曲线近似表现如下。下面为目测值或正文四舍五入结论，**非精确重算结果**：

| Draft 输入 | 约略最终速度加速比 | 约略最终 Acc | 说明 |
|---|---:|---:|---|
| token（橙） | ~1.5× | ~0.30 | 直接从 Token 建模，低成本但接受表现较差 |
| feature（绿） | ~1.9× | ~0.62 | 利用 Target 隐藏状态规律性，但不区分采样结果 |
| feature & shifted-token（蓝） | ~2.8× | ~0.78 | 在特征历史基础上显式条件化后继采样 Token |

**消融逻辑：** Token→Feature 的提升对应 Observation 1；Feature→Feature + shifted-token 对应 Observation 2。高精度特征预测并不自动保证最终输出分布；分布保持还依赖 Target 验证阶段的精确验收或拒绝后修正采样。

从图上可见蓝线在训练早期即明显优于其他方案，经过更多 Epoch 后渐进饱和。这能支持两项设计对 *该实验配置* 有效，但不足以证明所有任务、所有模型、所有温度和负载下收益相同。

### 4. 为什么 T=1 仍然能保持输出分布？（先修概念）

EAGLE 的 Draft 只是**候选提议分布** $q$，最终由 Target 得到真正分布 $p$ 并进行验证。标准推测采样对提议 Token $x$ 的接收概率：

\[
a(x)=\min\left(1,\frac{p(x)}{q(x)}\right).
\]

拒绝时应从修正分布 $r(x)\propto\max(0,p(x)-q(x))$ 重新采样。故 $q$ 不精确会造成更多拒绝、降低速度，但在正确实现并保持所需支持条件的情况下，不改变最终目标抽样分布。EAGLE 多分支树验证会在 Appendix A.2 进一步说明；此处只是动机，不替代 Section 2/3 的正式翻译。

### 5. 和 DFlash 的跨论文关联

[EAGLE 原始论文](https://arxiv.org/abs/2401.15077)：Target 隐藏特征 + Shifted Token，使用轻量**自回归**特征预测头；核心提升来自更可预测的表示和显式消除采样后继的不确定性。

[已精读的 DFlash](../2602.06036-dflash/README.md)：融合 Target 多层隐藏特征，以每层 KV Injection 条件化**并行 Block Diffusion Draft**；减少顺序草稿步数、通过 Target 验证保持正确分布。

两者不是同一实验条件，也不能将 EAGLE (2024) 与后续 [EAGLE-3](https://arxiv.org/abs/2503.01840) 混同。比较时同时关注 Draft step latency、Target verification cost、accepted tokens/target pass、GPU/后端/Batch 设置。

### 6. 可复现实验建议（独立建议）

以 Vicuna 7B 或兼容现代 decoder-only target 复制输入消融：固定 Target、LM Head、评测 prompt、树草稿策略和训练数据，分别训练 token-only、feature-only、feature+shifted-token 三种 Draft；跟踪候选接受率、平均接受长度、端到端 tokens/s、Verification overhead；在 T=0 与 T=1 下分别测量。正式复现时应依照 Section 3/4 和 Appendix A 的完整训练参数与树验证细节，而不是仅根据 Introduction 的曲线重建架构。

---

## 阅读进度

- **已完成：** Abstract；1. Introduction（全部段落，Figure 2、3、4 的图注与解释）。
- **下一阅读单元：** 2. Preliminaries（PDF 第 3 页），进入符号定义与标准推测采样的数学机制。
- **图像状态：** 原图已从用户提供的 PDF 裁图并在对话展示；二进制文件是否已提交 GitHub，见 progress.json 的 github_binary_saved 字段。
