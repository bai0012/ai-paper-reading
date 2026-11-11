# Section 3 Architecture：总体架构与 3.1 半自回归生成

> **论文：** DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation  
> **版本：** [arXiv:2607.05147v1](https://arxiv.org/abs/2607.05147)（2026-07-06）  
> **原始 PDF：** [第 4 页](https://arxiv.org/pdf/2607.05147v1#page=4)–[第 6 页](https://arxiv.org/pdf/2607.05147v1#page=6)。Figure 1 位于 PDF 第 5 页，Equation (4)–(6) 位于 PDF 第 6 页。  
> **阅读范围：** Section 3 总体介绍、Figure 1 与完整图注、Section 3.1 Semi-Autoregressive Generation 全部正文。Section 3.2 从 PDF 第 6 页下半部分开始，不在本文件翻译范围内。  
> **区分：** 第一部分为原文翻译；其后是独立技术分析、维度推导和伪代码。

## 一、Section 3 Architecture：原文完整翻译

### 3. Architecture

DSpark 的总体架构如 Figure 1 所示。回顾公式 (1)，推测解码的每 Token 延迟为：

$$
L=\frac{T_{\mathrm{draft}}+T_{\mathrm{verify}}}{\tau}.
$$

自回归草稿模型能够实现较高的 $\tau$，但需要付出 $T_{\mathrm{draft}}\propto\gamma$ 的代价；并行草稿模型将 $T_{\mathrm{draft}}$ 压缩为单次前向传播，但由于每个位置都是独立预测的，因此牺牲了 $\tau$。与此同时，固定长度验证会将 $T_{\mathrm{verify}}$ 浪费在低置信度、几乎注定会被拒绝的后缀 Token 上。

DSpark 通过两个互补组件解决这些限制：

- **半自回归生成（Semi-autoregressive generation，Section 3.1）。** 并行 Backbone 负责大部分草稿计算，使 $T_{\mathrm{draft}}$ 几乎不受 $\gamma$ 影响。随后，一个轻量级顺序模块在 Draft Token 之间引入依赖关系，以很小的额外延迟提高 $\tau$。
- **置信度调度验证（Confidence-scheduled verification，Section 3.2）。** 置信度预测头估计每个位置的接受概率，硬件感知调度器利用这些估计值裁剪低置信度后缀 Token，减少不必要的验证计算。

**原文脚注 1：** 本文使用 Anchor Token 和 Bonus Token 两个术语，指代同一个概念：上一轮解码过程中由目标模型生成的最后一个 Token。

### Figure 1 图注完整翻译

**图 1｜DSpark 架构与解码周期。**

给定提示词 Token ABC，目标模型执行一次解码步骤，生成下一个 Token D，该 Token 将作为草稿生成阶段的锚点。

以 D 为输入，DSpark 使用一个计算量较大的并行 Backbone 和一个轻量级顺序预测头，生成草稿 Token EFGH，以及与之对应的置信度分数 $c_1$–$c_4$。

随后，硬件感知前缀调度器（Hardware-Aware Prefix Scheduler）评估这些分数，保留前缀 EFG，丢弃低置信度 Token H。

最后，目标模型并行验证经过调度的前缀。如图所示，E 和 F 被接受，而 G 被拒绝，促使模型生成一个修正 Token $G^*$，以完成当前解码轮次。

### Figure 1 原图与素材状态

- 原图来源：用户提供的 DSpark v1 PDF，第 5 页，Figure 1。
- 原图已在阅读对话中从 PDF 页面裁剪并展示，包含完整原始图注。
- 原始 PDF 页面：[2607.05147v1，第 5 页](https://arxiv.org/pdf/2607.05147v1#page=5)。
- 图片二进制未提交到本仓库；具体状态与来源记录见 figures/README.md。没有使用重绘图冒充原图。

## 二、3.1 Semi-Autoregressive Generation：原文完整翻译

### 3.1.1 研究动机

并行草稿模型在一次前向传播中生成全部 $\gamma$ 个草稿 Logits，因此每个预测位置都无法以该块中其他位置实际采样出的 Token 为条件。

当上下文存在多种合理的后续续写时，例如 “of course” 和 “no problem”，并行草稿模型可能生成不连贯的组合，例如 “of problem” 或 “no course”。

这是因为每个位置都对所有可能的前序 Token 进行了边缘化，而不是以实际采样出的那个前序 Token 为条件（Gu et al., 2018；Huang et al., 2022a）。

因此，接受率会沿着草稿块迅速下降，浪费草稿生成和目标验证两方面的计算资源。为此，我们采用一种半自回归结构，将草稿生成划分为两个阶段。

### 3.1.2 Parallel stage（并行阶段）

一个并行 Backbone（在本文的具体实现中使用 DFlash，Chen et al., 2026）通过一次前向传播处理整个块，生成隐藏状态：

$$
h_1,\ldots,h_\gamma
$$

以及基础 Logits：

$$
U_1,\ldots,U_\gamma.
$$

我们仅对原始 DFlash Backbone 进行了一项小幅修改：

原始 DFlash 输入一个 Anchor Token 和 $\gamma$ 个 Mask Token，并且只预测 Mask 位置。

而在 DSpark 中，我们将 Anchor Token 本身也视为第一个预测位置。因此，使用 $\gamma$ 个输入 Token（一个 Anchor Token 加上 $\gamma-1$ 个 Mask Token），就可以得到 $\gamma$ 组 Draft Logits。

这种方式在保持草稿质量相近的同时，减少了草稿计算量。

### 3.1.3 Sequential stage（顺序阶段）

顺序阶段通过增加一个依赖于前缀的转移偏置（Prefix-dependent Transition Bias）：

$$
B_k(x_0,x_{<k},x_k)
$$

来补充基础 Logits，使每个草稿位置都能够以块内此前已经采样的 Token 为条件。

我们不定义一个全局归一化的能量模型（Globally Normalized Energy Model），而是通过自回归因子分解（Autoregressive Factorization）构造一个因果块分布：

$$
\boxed{
P(X\mid x_0)=
\prod_{k=1}^{\gamma}
p_k(x_k\mid x_0,x_{<k})
}
$$

其中：

$$
\boxed{
p_k(v\mid x_0,x_{<k})
=
\frac{
\exp\left(U_k(v)+B_k(x_0,x_{<k},v)\right)
}{
\displaystyle\sum_{u\in\mathcal V}
\exp\left(U_k(u)+B_k(x_0,x_{<k},u)\right)
}
}
\tag{4}
$$

这里：

- $x_0$ 表示上一轮验证周期产生的 Anchor Token。
- $U_k$ 表示并行 Backbone 在第 $k$ 个位置产生的基础 Logit 向量。
- $\mathcal V$ 表示词表（Vocabulary）。

在推理阶段，顺序模块按照以下条件分布，从左到右依次采样：

$$
p_k(\cdot\mid x_0,x_{<k}).
$$

由于这一采样过程本质上是串行的，因此该模块必须保持足够轻量：

$$
T_{\mathrm{sequential}}\ll T_{\mathrm{parallel}}.
$$

这样才能确保整体草稿生成延迟仍然主要由并行阶段决定。下面介绍顺序模块的两种具体实现。

### 3.1.4 Markov head（一阶马尔可夫预测头）

最简单的实现方式是限制 $B_k$ 只依赖于紧邻的前一个 Token，将其简化为一阶转移关系：

$$
B(x_{k-1},x_k).
$$

从理论上讲，$B$ 是一个完整的 $V\times V$ 矩阵。我们使用低秩分解（Low-rank Factorization）对其进行近似：

$$
B=W_1W_2,
$$

其中：

$$
W_1\in\mathbb R^{V\times r},\qquad
W_2\in\mathbb R^{r\times V}.
$$

给定前一个 Token $x_{k-1}$，位置 $k$ 的转移偏置为：

$$
\boxed{
B(x_{k-1},\cdot)
=
W_1[x_{k-1}]W_2
\in\mathbb R^V
}
\tag{5}
$$

其中 $W_1$ 充当 Embedding Lookup Table（嵌入查找表），$W_2$ 充当 Logit Projection（Logit 投影层）。

低秩分解默认使用 $r=256$，使存储开销和每一步的计算量都保持较小，即使面对较大的词表也能保持高效。

回到前面的例子：当第一个位置采样出 “of” 后，Markov Head 会在第二个位置提高 “course” 的分数，同时降低 “problem” 的分数，从而缓解不同续写模式之间的碰撞。

### 3.1.5 RNN head（循环神经网络预测头）

Markov Head 在超过一个时间步之后不再保留历史信息——位置 $k$ 无法访问 $x_{k-1}$ 之前的 Token。

RNN Head 通过维护一个循环状态（Recurrent State）$s_k$ 放宽了这一限制，该状态能够累积当前块内的完整前缀历史。

在每一步中，该模块将以下三个部分拼接：

1. 当前状态 $s_{k-1}\in\mathbb R^r$；
2. 前一个 Token 的嵌入 $W_1[x_{k-1}]\in\mathbb R^r$；
3. Backbone 隐藏状态 $h_k\in\mathbb R^d$。

形成输入向量：

$$
\boxed{
z_k=
[s_{k-1};W_1[x_{k-1}];h_k]
\in\mathbb R^{2r+d}
}
$$

随后，执行一次门控更新（Gated Update）：

$$
\boxed{
\begin{aligned}
s_k={}&
\sigma(W_gz_k)\odot s_{k-1}\\
&+\left(1-\sigma(W_gz_k)\right)
\odot\tanh(W_cz_k),\\[5pt]
B_k(x_{<k},\cdot)
={}&W_2^\top\tanh(W_oz_k).
\end{aligned}
}
\tag{6}
$$

其中：

$$
W_g,W_c,W_o
\in\mathbb R^{r\times(2r+d)}.
$$

这三个矩阵由一个共享的线性投影联合参数化，然后拆分为门控（Gate）、候选状态（Candidate）和输出（Output）三个部分。

初始状态 $s_0$ 被初始化为零向量。

---

## 三、公式与实现讲解（独立分析，非原文）

### 1. Figure 1 的解码周期

1. **Target 产生 Anchor：** 给定 Prompt ABC，Target Model 产生 D。D 是上一轮 Target 输出的最后一个 Token，也是当前 Draft Block 的起点。
2. **DSpark 产生 Draft：** 以 D 和若干 Mask Embedding 为输入，并行 Backbone 一次性产生全部 $h_k$ 和 $U_k$；随后 Sequential Head 逐位置采样 E、F、G、H。置信度头与硬件感知前缀调度器属于 Section 3.2，图中先行展示。
3. **Target 验证：** 调度器只保留 EFG，将其交给 Target 并行验证。图示结果是 E、F 接受，G 拒绝，Target 产生修正 Token $G^*$。H 并未送入 Target 验证；它是调度阶段被裁掉，而 G 是实际验证后被拒绝。这是两种不同的计算路径。

### 2. DSpark 对 DFlash 输入结构的修改

原始 DFlash 使用一个 Anchor + $\gamma$ 个 Mask 输入，预测 $\gamma$ 个 Mask 位置。

DSpark 将 Anchor 自身视为第一个预测位置，改为：

$$
\underbrace{[x_0,\mathrm{Mask},\ldots,\mathrm{Mask}]}_{\gamma\text{ 个输入 Token}}
\longrightarrow
\underbrace{[U_1,\ldots,U_\gamma]}_{\gamma\text{ 组 Draft Logits}}.
$$

即一个 Anchor + $\gamma-1$ 个 Mask，产生 $\gamma$ 个预测位置。其动机是减少一个输入位置的计算，同时保持相近的 Draft Quality。

这不是简单地删除最后一个预测 Token，而是改变了预测位置的对齐方式：Anchor 的 Backbone Hidden State 被用于预测块内第一个新 Token。

### 3. 公式 (4)：从并行 Logits 到因果块分布

将每个位置的条件分布写成：

$$
p_k(\cdot\mid x_0,x_{<k})
=
\operatorname{softmax}
\left(U_k+B_k(x_0,x_{<k},\cdot)\right).
$$

其中 $U_k$ 是并行 Backbone 预先计算的基础 Logits，$B_k$ 是依据已经采样的前缀计算出的 V 维修正向量。

注意这里的 $p_k$ 是 DSpark Sequential Block 定义的草稿条件分布；不要与 Section 2.1 中记作 $p_k^t$ 的 Target 分布混淆。符号复用但上下文不同。

整个块的联合分布：

$$
P(X\mid x_0)=\prod_{k=1}^{\gamma}p_k(x_k\mid x_0,x_{<k})
$$

是标准的自回归因子分解。由于每个条件分布分别在词表上归一化，联合分布自然归一化为 1，不需要对所有可能的 Token 序列计算一个全局 Partition Function。这是论文特别强调不采用 globally normalized energy model 的原因。

**教学例子（非论文实验数据）：** 假设第一个位置已经采样出 “of”，第二个位置的基础 Logits 对 course、problem、other 分别为 $1.0,1.2,0.0$；Markov Bias 分别为 $+2.0,-2.0,0.0$。修正后 Logits 是 $3.0,-0.8,0.0$，Softmax 会显著偏向 course。这里展示的是 Logit Bias 如何修正概率，不是论文训练出的真实参数。

### 4. 公式 (5)：Markov Head 的低秩分解

完整转移矩阵有 $V^2$ 个参数；低秩分解后只有 $2Vr$ 个参数。给定前一 Token，先用 $W_1$ 查出一个 $r$ 维向量，再通过 $W_2$ 投影为 V 维 Bias：

$$
\underbrace{W_1[x_{k-1}]}_{1\times r}
\underbrace{W_2}_{r\times V}
=
\underbrace{B(x_{k-1},\cdot)}_{1\times V}.
$$

因此参数量由 $O(V^2)$ 降为 $O(2Vr)$，单位置主要计算约为 $O(rV)$。论文给出的默认低秩维度为 $r=256$。

例如假设 $V=150000,r=256$，低秩参数量为 76.8M，而完整矩阵是 22.5B。该 V 值是演示假设，不代表论文模型的实际词表配置。

Markov Head 只显式利用上一个 Token，但它不是独立的 Markov Language Model：最终分布仍由 Backbone Logits $U_k$ 与转移 Bias 相加得到。它补充的是局部转移信息，而非替换整个上下文建模。

### 5. 公式 (6)：RNN Head 的维度与门控含义

输入维度：

$$
\underbrace{s_{k-1}}_{r}
\oplus
\underbrace{W_1[x_{k-1}]}_{r}
\oplus
\underbrace{h_k}_{d}
\Rightarrow z_k\in\mathbb R^{2r+d}.
$$

三个线性映射的输出都是 r 维：

- $W_gz_k$：经 Sigmoid 生成门控向量；
- $W_cz_k$：经 Tanh 生成候选状态；
- $W_oz_k$：经 Tanh 后再通过 $W_2^\top$ 投影为 V 维 Logit Bias。

门控更新可以改写为：

$$
g_k=\sigma(W_gz_k),\qquad
\tilde{s}_k=\tanh(W_cz_k),
$$

$$
s_k=g_k\odot s_{k-1}+(1-g_k)\odot\tilde{s}_k.
$$

当 $g_k$ 某个维度接近 1 时，该维度保留较多旧状态；接近 0 时，则更多采用当前候选状态。RNN 因而能够把此前生成的 Token 信息压缩进固定维度的状态向量。它保存的是前缀的压缩表示，不是无损存储全部历史 Token。

输出 Bias 为：

$$
B_k(x_{<k},\cdot)=W_2^\top\tanh(W_oz_k)\in\mathbb R^V.
$$

这里 $z_k$ 由旧状态、前一 Token Embedding 和当前 $h_k$ 组成，因此当前候选 Token 尚未采样前，顺序头就能产生该位置所有候选的偏置。

论文说明 $W_g,W_c,W_o$ 由单个线性投影拆分为三个分量。工程上可用一个输出维度为 $3r$ 的 Linear 一次计算，再按最后一维拆分，以减少 Kernel Launch。

### 6. Markov 与 RNN 的计算取舍

| 维度 | Markov Head | RNN Head |
| --- | --- | --- |
| 历史范围 | 仅最近一个 Token | 通过状态压缩更长前缀 |
| 核心输入 | 前一 Token ID | 旧状态、前一 Token Embedding、$h_k$ |
| 状态更新 | 无 | 门控循环更新 |
| 每位置 Bias | $W_1[x_{k-1}]W_2$ | $W_2^\top\tanh(W_oz_k)$ |
| 优势 | 结构简单、低秩转移 | 更丰富的前缀条件信息 |
| 代价 | 一阶转移表达能力有限 | 额外状态计算与顺序依赖 |

两者是 Sequential Block 的不同实现方式，不是先后叠加的两个模块。两者都要计算词表维度的 Bias；因此不能仅根据 RNN 的状态维度小，就断言完整 RNN Head 的延迟必然可忽略。论文的设计要求是 $T_{\mathrm{sequential}}\ll T_{\mathrm{parallel}}$，具体是否满足仍取决于词表、硬件、Batch 和 Kernel 实现。

### 7. 概念性 PyTorch 推理伪代码

以下代码依据公式 (4)–(6) 说明数据流，不是 DeepSpec 官方实现的逐行复现：

~~~python
# Parallel stage
hidden, base_logits = backbone(anchor, masks)
# hidden:      [B, gamma, d]
# base_logits: [B, gamma, vocab_size]

state = torch.zeros(B, rank, device=device)
previous_token = anchor
draft_tokens = []

# Sequential stage
for k in range(gamma):
    previous_emb = W1[previous_token]

    z = torch.cat(
        [state, previous_emb, hidden[:, k]],
        dim=-1
    )

    gate = torch.sigmoid(Wg(z))
    candidate = torch.tanh(Wc(z))
    state = gate * state + (1 - gate) * candidate

    bias = torch.tanh(Wo(z)) @ W2
    logits = base_logits[:, k] + bias

    probs = torch.softmax(logits, dim=-1)
    next_token = sample(probs)

    draft_tokens.append(next_token)
    previous_token = next_token
~~~

Markov 版本只需将 Bias 计算替换为：

~~~python
bias = W1[previous_token] @ W2
~~~

### 8. 与此前阅读论文的关系

| 论文 | 主要 Draft 计算 | Token 依赖方式 | DSpark 的区别 |
| --- | --- | --- | --- |
| EAGLE | 特征级自回归 Draft | 显式依赖此前采样 Token | DSpark 将重型计算放在并行 Backbone，只把依赖修正留给轻量头 |
| Medusa | 多个 Future Token Heads | 多位置候选并行预测、树状验证 | DSpark 生成单条顺序条件化 Draft，不依赖多个未来头形成候选树 |
| DFlash | 多层 Target Hidden Features + KV Injection + 并行 Mask Block | Backbone 内双向特征交互，但没有采样后的逐位置条件更新 | DSpark 继承并行 Backbone，在其后增加实际 Token 前缀条件化 |
| DSpark | 并行 Backbone + Sequential Head | 通过 Markov/RNN Bias 条件化 | 在 Draft 延迟与后缀一致性之间折中 |

**核心判断：** DSpark 不是把 DFlash 改造成一个小型 EAGLE，而是把自回归能力从重型 Backbone 中剥离出来，放到一个轻量级输出模块中。重型计算并行化，轻量模块负责采样后的条件修正。

### 9. 本节边界与原始素材

- Section 3 总述：PDF 第 4 页。
- Figure 1 及完整图注、Section 3.1 研究动机与 Parallel Stage：PDF 第 5 页。
- Sequential Stage、Equation (4)–(6)：PDF 第 6 页。
- PDF 第 6 页下半部分开始 Section 3.2 Confidence-Scheduled Verification，本文件没有提前翻译该节。
- Figure 1 原始图已在阅读会话中提取并展示；GitHub 的二进制图像文件未提交，图号、页码、来源记录见 figures/README.md。

## 四、小结与下一阅读位置

1. DSpark 的两个系统组件分别解决 Draft 质量和 Target 验证资源分配问题；本次完成的是第一个组件。
2. DFlash 并行 Backbone 一次计算所有 $h_k,U_k$；DSpark 将 Anchor 本身作为第一预测位置，使用 Anchor + $\gamma-1$ Masks 产生 $\gamma$ 组 Draft Logits。
3. Sequential Head 使用 $B_k$ 对基础 Logits 做前缀条件修正，使联合 Draft 分布成为合法的自回归因子分解。
4. Markov Head 使用前一 Token 的低秩转移 Bias；RNN Head 用循环状态保存更长前缀的压缩信息。
5. 关键设计约束是 $T_{\mathrm{sequential}}\ll T_{\mathrm{parallel}}$，否则串行头可能抵消并行 Backbone 的收益。

**已完成：Section 3 总体架构与 Section 3.1 Semi-Autoregressive Generation。下一单元：Section 3.2 Confidence-Scheduled Verification（3.2.1 Confidence Head、3.2.2 Hardware-Aware Prefix Scheduler），随后阅读 3.3 Training。**
