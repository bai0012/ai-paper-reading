# Section 2 Background：推测解码与草稿模型架构

> 论文：[DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation](https://arxiv.org/abs/2607.05147)，v1，2026-07-06  
> 原始 PDF：[第 3 页](https://arxiv.org/pdf/2607.05147v1#page=3)、[第 4 页](https://arxiv.org/pdf/2607.05147v1#page=4)。已逐页核对原文与三个公式。  
> 本文覆盖 Section 2.1、2.2 全部正文，Equation (1)–(3)。本节不包含原始 Figure 或 Table。  
> 下文「原文完整翻译」与「独立技术分析」分开。不得将教学例子与实验数据混同。

## 一、原文完整翻译

### 2.1 Speculative Decoding（推测解码）

**第 1 段**

自回归语言模型在每次前向传播中生成一个 Token，因此推理延迟与输出长度成正比。推测解码（Speculative Decoding；Chen et al., 2023；Ge et al., 2022；Leviathan et al., 2023）使用一个轻量级草稿模型 $M_d$ 加速目标模型 $M_t$ 的推理。在每轮解码中，草稿模型提出 $\gamma$ 个候选 Token $x_1,\ldots,x_\gamma$。目标模型通过单次前向传播验证全部候选，并接受与自身分布一致的最长前缀。

**第 2 段**

具体来说，在每个草稿位置 $k$，目标模型计算自己的概率分布 $p_k^t$，并将它与草稿分布 $p_k^d$ 比较。Token $x_k$ 以概率 $\min(1,p_k^t(x_k)/p_k^d(x_k))$ 被接受。验证从左往右进行：一旦在位置 $k$ 首次拒绝，所有后续 Token $x_{k+1},\ldots,x_\gamma$ 都会被丢弃，无论它们本身的质量如何。

**第 3 段及公式 (1)**

设 $\tau$ 表示每轮解码所接受的 Token 数，$T_{\mathrm{draft}}$ 和 $T_{\mathrm{verify}}$ 分别表示草稿生成与验证过程的墙钟时间（Wall-Clock Time）。每生成一个 Token 的平均延迟为：

$$
L=\frac{T_{\mathrm{draft}}+T_{\mathrm{verify}}}{\tau}. \tag{1}
$$

因此，改善加速效果归结为三个切入点：降低 $T_{\mathrm{draft}}$（更快地生成草稿）、提高 $\tau$（生成更好的草稿），或者降低有效的 $T_{\mathrm{verify}}$（更智能地验证）。

### 2.2 Drafter Architectures（草稿模型架构）

**引言段**

草稿模型的设计决定了 $T_{\mathrm{draft}}$ 和 $\tau$ 如何权衡。现有方法分为两类。

**Autoregressive drafters / 自回归草稿模型**

自回归草稿模型按顺序生成 Draft Token，每个位置以此前采样出的 Token 为条件（DeepSeek-AI, 2024；Li et al., 2024b,c, 2026b；Zhang et al., 2025）。显式依赖提供了较强的建模能力，但生成草稿的开销随块长线性增长：$T_{\mathrm{draft}}\propto\gamma$。这迫使自回归草稿模型采用较小的 $\gamma$ 和较浅的网络结构，以保持较低的 $T_{\mathrm{draft}}$。为了弥补块长较短的不足，基于树的验证（Miao et al., 2024）将候选扩展为一棵树，并使用 Tree Attention 同时验证多条路径；但大量验证 Token 会降低整个服务系统的吞吐量。

**Parallel drafters / 并行草稿模型**

并行草稿模型在一次前向传播中生成全部 $\gamma$ 个 Draft Token，使 $T_{\mathrm{draft}}$ 几乎与块大小无关（Cai et al., 2024；Chen et al., 2026；Li et al., 2025a；Liu et al., 2026a；Sandler et al., 2026）。这允许使用大得多的块（例如 $\gamma=16$），而无需让延迟同比例增长。

在这些方法中，DFlash（Chen et al., 2026）是一种先进的并行草稿模型。它利用从目标模型提取的丰富上下文特征，对草稿模型进行条件化（KV Injection）。预填充（Prefill）期间，目标模型若干层 $\{l_1,\ldots,l_m\}$ 的隐藏状态被拼接并投影至草稿隐藏空间：

$$
H_{\mathrm{ctx}}=\operatorname{RMSNorm}\left(W_c[H^{(l_1)};\ldots;H^{(l_m)}]\right). \tag{2}
$$

其中 $W_c\in\mathbb R^{d\times md}$ 是共享投影矩阵。这些上下文特征通过沿序列维度将其与 Draft Block 的表示拼接为 Key 和 Value，注入每一个 Draft Layer：

$$
K_i=[W_i^K H_{\mathrm{ctx}};\ W_i^K H_d],\qquad
V_i=[W_i^V H_{\mathrm{ctx}};\ W_i^V H_d]. \tag{3}
$$

块内所有位置都可以双向关注块内其他位置以及注入的目标模型上下文。

草稿模型共享目标模型的 Embedding 层和语言模型输出头（LM Head）；这两个模块均冻结。它以一个锚点 Token（Anchor Token）的 Embedding，以及其后 $\gamma$ 个 Mask Token Embedding 为输入，在单次前向传播中输出所有 Mask 位置的 Logits。由于无论块长如何都只需要一次前向传播，与自回归草稿模型相比，DFlash 可以在相同延迟预算下采用更深的架构和更长的草稿块。

**Section 2 全文翻译结束。**

---

## 二、公式与实现讲解（独立分析，非原文）

### 1. 公式 (1)：性能优化的三种杠杆

- $T_{\mathrm{draft}}$：Draft Model 的 Wall-Clock Latency。更深的 AR Draft 可能提高准确性，但也增加 Draft 时间。
- $T_{\mathrm{verify}}$：Target Model 单轮验证消耗。长块对单请求可能较划算，对大量并发请求却会竞争验证容量。
- $\tau$：每轮有效产生 Token 的平均数量。论文 Section 4 的注脚明确说明，文中通常报告的 Accepted Length 包含 Target 生成的 Bonus Token；与单纯接受 Draft Token 数不同。

教学数值示例（非论文实测）：如果 Draft 耗时 4 ms、Target 验证耗时 12 ms、每轮输出平均 3 个 Token，则 $L=(4+12)/3=5.33$ ms/token。把 Draft 加深后，假设 Draft 耗时 8 ms、每轮输出变为 3.5，则 $L=(8+12)/3.5=5.71$ ms/token，尽管接受长度增加，平均输出延迟反而变差。

对连续前缀机制的先行推导（对应后续 Section 3.2 的概率定义，不是本节原文公式）：令 $c_k$ 是在此前 Token 全部接受条件下第 $k$ 个 Token 的接受概率，前缀存活概率 $a_k=\prod_{i=1}^k c_i$；期望输出长度 $\mathbb E[\tau]=1+\sum_{k=1}^\gamma a_k$。若 $\gamma=4$ 且所有 $c_k=0.8$，则 $\mathbb E[\tau]=1+0.8+0.64+0.512+0.4096=3.3616$。

标准推测采样被拒绝时需要从残差分布 $r(x)\propto[p(x)-q(x)]_+$ 抽样以保持目标分布，不能直接不加修正地从目标分布重新抽样。Target 可以用单次 Causal Forward 计算全部候选位置概率，但接受/拒绝的决策仍按有效前缀从左向右执行。标准理论证明已另存 [questions.md：Q2](questions.md)；残差式属于独立理论补充，不是 Section 2.1 原文逐字内容。

### 2. 公式 (2)：Target 多层 Hidden Features 融合

取目标模型 $m$ 个层的隐藏表示，在特征维进行拼接，经共享投影矩阵 $W_c$ 映射到 Draft 隐空间后再施加 RMSNorm。

假设单 Token 每层隐藏特征为 $d=4096$ 维、取 $m=3$ 层，则拼接后 $md=12288$ 维。作者写 $W_c\in\mathbb R^{d\times md}$，按列向量约定得到每 Token $4096$ 维的 $H_{\mathrm{ctx}}$；有 $n$ 个上下文位置时可以理解为 $n\times4096$。

示意 PyTorch 数据流（非原论文官方代码）：

    # m 个 target_hidden 均为 [B, seq_len, target_dim]
    features = torch.cat(target_hidden, dim=-1)
    h_ctx = rmsnorm(projection(features))

对于 Target 隐宽与 Draft 隐宽不同的实际模型，可以更一般地记作 $W_c\in\mathbb R^{d_d\times m d_t}$。这是**工程一般化**，不能冒充论文的原始矩阵尺寸。

### 3. 公式 (3)：逐层 KV Injection 的含义

区别两种拼接方式：公式 (2) 的方括号表示多层特征在**隐藏/特征维**的拼接，而公式 (3) 的方括号表示上下文 KV 和草稿 KV 在**序列维**的拼接；不能将两者混淆。

例如假设 1024 个目标上下文 Token、原始 DFlash 的一个 Anchor 和 8 个 Masks，共 9 个 Draft 输入位置、注意力 Head 维度 $d_k=128$：

| 张量 | 示例维度 |
| --- | --- |
| Draft Query $Q_d$ | $9\times128$ |
| Target Context Key | $1024\times128$ |
| Draft Block Key | $9\times128$ |
| $K_i$ 拼接结果 | $1033\times128$ |
| $V_i$ 拼接结果 | $1033\times128$ |
| 理论的 Attention Score | $9\times1033$ |

上述数值为**教学假设**，实际张量布局还受 Batch、Heads、GQA、KV Cache 以及 Attention Kernel 影响，未必显式计算完整的矩阵。

在单层 Draft Attention 中，Draft 查询通过：

$$
\operatorname{Attention}(Q_d,K_i,V_i)=
\operatorname{softmax}\left(\frac{Q_dK_i^\top}{\sqrt{d_k}}+M\right)V_i
$$

访问注入的上下文与块内 Key/Value，$M$ 表示所采用的注意力掩码。由于不是只在输入层做 Feature Fusion，而是在 Draft 的**每一层**提供 Target Context KV，因此草稿隐藏状态能够持续查询目标上下文。

**特别注意：块内双向 Attention 不等于真正依赖采样出来的前序 Token。** DFlash 的各位置在同一次 Backbone Forward 中看到的是 Anchor + Mask Embedding，虽然可相互传递隐藏特征，但并不能在第一个 Token 实际采样完成后重新计算第二个位置的条件 Logits。因此纯并行方式仍然存在 Suffix Decay；DSpark 在 Section 3.1 添加轻量顺序头以补充这一能力。

### 4. AR Draft 与树式验证的成本权衡

传统 AR Draft 通过 $q(x_k\mid x_{<k},C)$ 显式利用块内采样历史，而并行 Draft 在一次 pass 内形成多位置预测，可近似理解为基于共同上下文 $C$ 的并行预测。这是概念性区分，不等于说 DFlash 的 Transformer 完全没有块内信息交互。树式验证通过同时尝试多个分支，提高每轮有用输出的可能性，但 Target 需要处理的验证 Token 可能大量增加，对高并发吞吐不利。

### 5. 跨论文联系

| 已读论文 | 架构侧重点 | 与 DSpark 的关系 |
| --- | --- | --- |
| [EAGLE](../2401.15077-eagle/README.md) | Feature AR + Shifted Token + Tree Attention | 体现序列条件化带来的优势，但有连续 Draft Forward 成本。这里是原版 EAGLE，论文比较的 Eagle3 指后续 EAGLE-3。 |
| [Medusa](../2401.10774-medusa/README.md) | 多个 Future Token Head、稀疏 Tree Verification | 多路径 Target 验证带来额外 Token 成本；其 Typical Acceptance 与精确无损采样并不等价。 |
| [DFlash](../2602.06036-dflash/README.md) | 并行 Mask Block + Target 多层 Feature Fusion + KV Injection | DSpark 的并行 Backbone 以该设计为基础，在输出端增加顺序依赖并以置信度调度验证预算。 |

### 6. 小结与原始素材核验

- 原文第 3 页：2.1 完整，公式 (1)。
- 原文第 4 页：2.2 完整，公式 (2)–(3)。该页后半部分开始 Section 3 Architecture，不计入本节。
- **本节没有原始 Figure、Table、算法框。** 第一个 Figure 1 位于 PDF 第 5 页，属于下一单元 Section 3。
- 公式、注脚、排版已在 PDF 原始页面核对；没有将其他论文图片冒充 DSpark 原图。

**阅读进度：Section 2.1–2.2 完成。下一单元：3. Architecture 的总述、Figure 1 和 3.1 Semi-Autoregressive Generation；再推导公式 (4)–(6)。**
