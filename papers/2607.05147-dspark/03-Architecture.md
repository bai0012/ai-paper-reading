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


---

## 五、Section 3.2 Confidence-Scheduled Verification（置信度调度验证）

> **原文范围：** PDF 第 6 页下半部分至第 9 页上半部分；Section 3.2、3.2.1、3.2.2；公式 (7)、(8) 与 Algorithm 1。  
> **原始素材：** 本节无 Figure 或 Table；包含原始 Algorithm 1，已从 PDF 第 8 页提取并在阅读对话展示。  
> **术语：** per-position conditional acceptance probability = 逐位置条件接受概率；prefix survival probability = 前缀存活概率；SPS = steps per second。

### 5.1 3.2 Confidence-Scheduled Verification：原文完整翻译

半自回归架构使 DSpark 能够高效地生成较大的草稿块。

然而，生成更多草稿 Token 并不会自动转化为更高的端到端加速比。无差别地验证整个草稿块，实际上可能降低整个系统的吞吐量，尤其是在高并发场景下（Hu et al., 2026b；Liu et al., 2024c）。

这一性能瓶颈来自两个相互作用的因素。

首先，在数据侧，不同领域的草稿接受率天然存在差异：代码等结构化文本通常具有较高的接受率，而开放式聊天的接受率则明显较低（Abramovich et al., 2026；Xia et al., 2024）。

其次，在系统侧，额外验证一个 Token 的实际成本严格取决于推理引擎的负载。在系统负载较低时，即使额外验证的 Token 最终被拒绝，其产生的额外惩罚也很小。然而，在高并发部署中，每一个不必要的验证 Token 都会占用目标模型的 Batch 容量，而这些资源原本可以用于服务其他活跃请求（Liu et al., 2024b；Wu et al., 2025）。

因此，要充分发挥大规模草稿块的潜力，需要一种统一机制，只将目标模型的计算资源分配给具有正向预期收益的 Token。DSpark 将一个预测前缀存活概率的 Confidence Head（Section 3.2.1），与一个根据当前系统负载动态决定最优验证长度的 Hardware-Aware Prefix Scheduler（Section 3.2.2）结合起来，实现这一目标。

### 5.2 3.2.1 Confidence Head：原文完整翻译

受到 Huang et al. (2024) 和 Wang et al. (2026b) 的启发，Confidence Head 为每个草稿位置 \(k\) 输出一个标量 \(c_k\in(0,1)\)。

关键在于，\(c_k\) 表示一个条件概率：在当前块中此前所有 Token 都已经被接受的前提下，位置 \(k\) 的草稿 Token 能够通过目标模型验证的概率。

该模块采用一个轻量级线性投影，后接 Sigmoid 函数：

\[
\boxed{
c_k=\sigma\left(w^\top[h_k;\ W_1[x_{k-1}]]\right).
}\tag{7}
\]

其中，\(h_k\) 是 Backbone 在位置 \(k\) 的隐藏状态，\(W_1[x_{k-1}]\) 是前一个草稿 Token 对应的 Markov Embedding。

我们使用解析得到的逐位置接受率 \(c_k^*\) 对 \(c_k\) 进行监督。该接受率由草稿分布 \(p_k^d\) 与目标分布 \(p_k^t\) 之间的总变差距离（Total Variation Distance）决定：

\[
\boxed{
c_k^*=1-\frac12\left\|p_k^d-p_k^t\right\|_1.
}\tag{8}
\]

#### Post-hoc Calibration / 事后校准

与基于阈值的验证启发式方法（Huang et al., 2024；Li et al., 2024b；Zhang et al., 2026b）不同，后者只要求置信度分数能够正确排列草稿 Token 质量的高低，而本文的硬件感知调度方法（详见 Section 3.2.2）需要置信度分数具有准确的绝对数值，以便计算期望接受长度 \(\tau\)。

由于神经网络的置信度估计往往存在过度自信（Overconfidence）问题（Guo et al., 2017；Ovadia et al., 2019），直接使用未经校准的原始置信度分数会扭曲吞吐量估计，进而导致次优的调度决策。

为解决这一问题，我们引入 Sequential Temperature Scaling（STS，序列温度缩放）。

由于每个 \(c_i\) 都表示一个条件概率，根据概率链式法则，一个草稿前缀被接受的联合概率可以分解为累计乘积：

\[
\prod_{i\leq k}c_i.
\]

使用一个留出的验证集，STS 从左到右依次校准这一联合概率。具体而言，对于每个位置 \(k\in\{1,\ldots,\gamma\}\)，我们执行一次简单的一维网格搜索，寻找一个最优温度标量，使累计乘积的 Expected Calibration Error（ECE，期望校准误差）最小，同时固定此前所有位置已经校准过的分数。

关键在于，温度缩放是一种保持顺序的变换（Order-preserving Transformation）：它能够修正预测概率，使之与经验接受率相匹配，同时不会破坏 Confidence Head 学到的草稿 Token 相对排序。

### 5.3 3.2.2 Hardware-Aware Prefix Scheduler：原文完整翻译

此前的方法（Huang et al., 2024；Li et al., 2024b）通常使用静态置信度阈值来决定验证长度。虽然在单请求、独立运行的假设下这种方法有效，但在高并发生产系统中，静态阈值可能并非最优，因为验证一个草稿 Token 的效用高度依赖当前系统负载。

为解决这一问题，我们将验证长度选择建模为一个全局吞吐量最大化问题（Algorithm 1）。

考虑一个包含 \(R\) 个活跃请求的 Batch。对于请求 \(r\)，令 \(c_{r,1},\ldots,c_{r,\gamma}\) 表示各位置的置信度估计，并令 \(\ell_r\in\{0,\ldots,\gamma\}\) 表示调度器为该请求选定的验证长度。

由于推测解码只会动态接受连续前缀中的草稿 Token，因此位置 \(j\) 的 Token 存活概率是累计乘积：

\[
\boxed{
a_{r,j}=\prod_{i\leq j}c_{r,i}.
}
\]

在一次验证步骤中，发送给目标模型的总 Batch 大小（以 Token 数量衡量）为：

\[
\boxed{
B=\sum_{r=1}^{R}(1+\ell_r).
}
\]

期望成功接受的 Token 数量为：

\[
\boxed{
\tau=\sum_{r=1}^{R}
\left(1+\sum_{j=1}^{\ell_r}a_{r,j}\right).
}
\]

在一个简化假设下²，令 \(SPS(B)\) 表示给定一次前向传播 Batch 大小时，推理引擎的吞吐量，以每秒执行的步骤数（Steps per Second）衡量。关键在于，这条容量曲线只需在推理引擎初始化期间进行一次 Profiling，并作为轻量级成本查找表保存。

随后，调度器通过动态选择验证长度 \(\ell_1,\ldots,\ell_R\)，最大化期望的系统级 Token 吞吐量：

\[
\boxed{
\Theta=\tau\cdot SPS(B).
}
\]

尽管寻找 \(\Theta\) 的全局最大值看起来是一个组合搜索问题，但目标函数的结构允许使用高效的贪心算法求解。

由于 \(a_{r,j}\) 关于 \(j\) 单调不增，即：

\[
a_{r,j}\leq a_{r,j-1},
\]

将请求 \(r\) 的验证长度从 \(j-1\) 扩展到 \(j\)，其期望接受 Token 数量的边际增益恰好是 \(a_{r,j}\)。

这种单调性保证了按照 \(a_{r,j}\) 对所有候选 Token 进行全局排序时，自然能够遵守每个请求内部的前缀依赖关系。

因此，如果总验证 Batch 大小 \(B\) 固定，那么最优分配 \(\{\ell_r\}\) 可以通过一种贪心方式确定：从所有请求的候选集合 \(\{a_{r,j}\}\) 中，选择存活概率最高的草稿 Token。

基于这一观察，可以沿着贪心准入路径评估优化目标。

首先，按照前缀存活概率从高到低，对所有有效的前缀扩展进行全局排序。为了动态确定最优的目标 Batch 大小 \(B\)，算法从排序后的候选池中逐个准入 Token，并通过成本查找表更新期望吞吐量 \(\Theta\)。

无损推测解码（Lossless Speculative Decoding）严格要求满足**非预知性（Non-anticipating Property）**：准入决策不能依赖未来的候选 Token（Chen et al., 2023；Leviathan et al., 2023）。

由于 Confidence Head 依赖前一个已采样 Token 的 Markov 特征，因此显式计算下一个位置的存活概率 \(a_{r,k+1}\) 需要已经实例化的候选 Token \(x_{r,k}\)。

因此，事后进行全局搜索会无意中将 \(x_{r,k}\) 泄漏到位置 \(k\) 的准入决策中，从而引入选择偏差（Selection Bias）。附录 A 提供了一个具体反例，展示这种理论违规。

为了强制保证严格的因果性，调度器（Algorithm 1）采用了提前停止（Early-stopping）机制。

当吞吐量下降时，即：

\[
\Theta\leq\Theta_{\mathrm{best}},
\]

算法立即终止贪心搜索。这样，截断决策只依赖截至当前确切步骤已经处理的前缀。

这使准入事件与未来 Token 隔离，从而确保精确恢复目标模型的概率分布。

需要注意，只有当目标函数 \(\Theta\) 是单峰函数（Unimodal）时，这种逐步提前停止才能得到全局最大吞吐量；这隐含地假设硬件容量曲线是平滑下降的。

对于真实系统中不平滑的 SPS 特性以及异步系统流水线所需的工程适配，作者将在 Section 5.2 中进行讨论。

**原文脚注 2：**

在实际服务场景中，平均上下文长度通常远低于极端情况（例如 100 万 Token），因此对于 DeepSeek-V4 这类高度优化的架构而言，上下文长度对解码延迟的影响较小。

此外，在 Prefill 与 Decode 分离部署的场景中，Decode 负载均衡器会使各数据并行（Data Parallel，DP）Rank 上的请求数量和上下文总长度大致保持均衡。

这实际上摊平了序列长度差异的影响，使我们可以合理地假设：推理引擎吞吐量主要取决于验证 Batch 大小 \(B\)。

### 5.4 Algorithm 1：Hardware-Aware Prefix Scheduler 原始算法

原始算法框来自 DSpark v1 PDF 第 8 页，已从用户提供的 PDF 提取并在阅读对话展示。完整 16 行伪代码的中文逐行对应如下。

**算法输入（Require）**
- 活跃请求 \(r\in\{1,\ldots,R\}\)；
- 每个请求的置信度序列 \(c_{r,1},\ldots,c_{r,\gamma}\)；
- 已 Profiling 的步骤吞吐曲线 \(SPS(B)\)。

**算法输出（Ensure）**

每个请求选定的前缀验证长度：

\[
\ell_1^*,\ldots,\ell_R^*.
\]

| 行 | 原始算法操作 | 中文含义 |
|---:|---|---|
| 1–3 | 对 \(r=1,\ldots,R\) 计算 \(a_{r,j}\leftarrow\prod_{i\leq j}c_{r,i}\)，再结束循环 | 遍历所有请求，计算每个位置的前缀存活概率 |
| 4 | \(\mathcal E\leftarrow\{(r,j)\mid a_{r,j}>0\}\)，按 \(a_{r,j}\) 降序排序 | 构建有效候选集合并进行全局排序 |
| 5 | \(\ell_r\leftarrow0\)，\(B\leftarrow R\)，\(\tau^*\leftarrow R\) | 初始不验证任何 Draft Token；每个请求仍贡献一个基础输出 Token |
| 6 | \(\Theta_{\mathrm{best}}\leftarrow R\cdot SPS(R)\)，\(\ell_r^*\leftarrow0\) | 初始化当前最佳吞吐量和最佳验证长度 |
| 7–8 | 遍历排序后的 \((r,j)\)，更新 \(\ell_r\leftarrow j\)、\(B\leftarrow B+1\)、\(\tau^*\leftarrow\tau^*+a_{r,j}\) | 尝试为对应请求增加一个候选位置 |
| 9 | \(\Theta\leftarrow\tau^*\cdot SPS(B)\) | 重新计算系统级期望吞吐量 |
| 10–11 | 若 \(\Theta>\Theta_{\mathrm{best}}\)，更新最佳吞吐量和各请求最佳长度 | 当前扩展带来吞吐量收益，保留它 |
| 12–14 | 否则立即 Break | 如果吞吐量不再增加，停止搜索 |
| 15–16 | 结束循环，返回 \((\ell_1^*,\ldots,\ell_R^*)\) | 输出最佳逐请求验证长度 |

### 5.5 原始公式与算法之间的对应关系

算法中初始化：

\[
B=R,\qquad \tau^*=R
\]

对应每个请求先占用一个基础 Token 的情况。

每准入一个草稿 Token：

\[
B\leftarrow B+1
\]

\[
\tau^*\leftarrow\tau^*+a_{r,j}
\]

因此每增加一个候选位置，Target 验证 Batch 的成本固定增加一个 Token，而期望产出增加量由其前缀存活概率决定。

最终：

\[
\Theta=\tau^*\cdot SPS(B),
\]

即系统期望 Token 吞吐量。

---

## 六、深入技术分析

### 6.1 逐位置条件接受概率与前缀存活概率不能混淆

令：

\[
c_{r,k}
=
P(\text{位置 }k\text{ 被接受}\mid
\text{位置 }1,\ldots,k-1\text{ 均被接受})
\]

它表示一个条件接受概率。

而：

\[
a_{r,j}=\prod_{i=1}^{j}c_{r,i}
\]

表示前 \(j\) 个 Token 全部被接受的联合概率：

\[
a_{r,j}
=
P(x_{r,1},\ldots,x_{r,j}
\text{ 全部通过验证}).
\]

例如某个请求的条件接受概率是：

\[
c=[0.9,0.8,0.5]
\]

那么前缀存活概率为：

\[
a=[0.9,0.72,0.36].
\]

这三个位置的条件接受率分别是 90%、80%、50%，但第三个 Token 真正进入有效接受前缀的概率只有 36%。

因此调度器按 \(a_{r,j}\) 排序，而不是直接按 \(c_{r,j}\) 排序。推测解码只接受连续前缀；后续 Token 必须先经过所有前序位置的验收。

### 6.2 为什么最大化的是吞吐量，而不是接受长度？

目标函数：

\[
\boxed{\Theta=\tau\cdot SPS(B)}
\]

其中：
- \(\tau\)：每次 Target 验证步骤的期望输出 Token 数；
- \(SPS(B)\)：验证 Batch 大小为 \(B\) 时，每秒执行的 Target Forward 步骤数；
- \(\Theta\)：系统每秒的期望输出 Token 数。

单位分析：

\[
\frac{\mathrm{tokens}}{\mathrm{step}}
\times
\frac{\mathrm{steps}}{\mathrm{second}}
=
\frac{\mathrm{tokens}}{\mathrm{second}}.
\]

如果只最大化 \(\tau\)，可能把大量低概率候选加入 Target Batch，导致 SPS 大幅下降；如果只最大化 SPS，又可能因为验证太少而使每轮输出 Token 数下降。两者必须联合优化。

### 6.3 双请求数值示例：完整执行贪心调度

下面模拟两个活跃请求，每个请求有三个候选 Token。所有数字均为教学假设，不是论文实验数据。

| 请求 | 位置 1 条件接受率 | 位置 2 | 位置 3 |
|---|---:|---:|---:|
| A | 0.90 | 0.80 | 0.50 |
| B | 0.70 | 0.60 | 0.90 |

换算为前缀存活概率：

| 请求 | 位置 1 | 位置 2 | 位置 3 |
|---|---:|---:|---:|
| A | 0.900 | 0.720 | 0.360 |
| B | 0.700 | 0.420 | 0.378 |

全局候选排序：

\[
A_1(0.900),A_2(0.720),B_1(0.700),
B_2(0.420),B_3(0.378),A_3(0.360).
\]

再假设引擎预先 Profiling 得到以下 SPS 曲线：

| Batch \(B\) | SPS（steps/s） |
|---:|---:|
| 2 | 100 |
| 3 | 98 |
| 4 | 95 |
| 5 | 90 |
| 6 | 82 |
| 7 | 70 |
| 8 | 50 |

从两个请求都不验证 Draft Token 开始：

\[
B=2,\quad \tau=2,\quad\Theta=2\times100=200.
\]

依次准入候选：

| 当前操作 | \(B\) | \(\tau\) | SPS | \(\Theta\) |
|---|---:|---:|---:|---:|
| 初始 | 2 | 2.00 | 100 | 200.00 |
| 加入 A1 | 3 | 2.90 | 98 | 284.20 |
| 加入 A2 | 4 | 3.62 | 95 | 343.90 |
| 加入 B1 | 5 | 4.32 | 90 | 388.80 |
| 尝试加入 B2 | 6 | 4.74 | 82 | 388.68 |

加入 B2 后，吞吐量从 388.80 降至 388.68，因此触发 Early Stopping。最终保留：

\[
\ell_A=2,\qquad \ell_B=1.
\]

这说明调度器可能宁愿让某个请求少验证几个 Token，也不愿为了较低的边际收益继续扩大 Batch。它优化的是系统级吞吐量，而不是每个请求单独追求最大接受长度。

### 6.4 为什么要使用 Sequential Temperature Scaling？

静态阈值方法只需要知道置信度排序是否合理。例如，置信度 0.9 的候选通常比 0.6 的候选更值得验证。

但 DSpark 的调度器要计算：

\[
a_k=\prod_{i=1}^{k}c_i
\]

并进一步估计：

\[
\tau=1+\sum_{k=1}^{\ell}a_k.
\]

这时，绝对概率值是否准确就非常重要。

假设模型对三个位置都预测：

\[
c=[0.95,0.95,0.95],
\]

预测的第三位置前缀存活率为：

\[
a_3=0.95^3\approx0.857.
\]

如果真实条件接受率实际上只有 0.8，那么真实存活率为：

\[
a_3^{\mathrm{real}}=0.8^3=0.512.
\]

调度器会明显高估这个长前缀的价值。

因此，STS 的重点不只是校准每个位置的独立置信度，而是逐位置校准累计前缀存活概率。

**ECE 的含义：** 将预测概率划分到若干置信度区间，比较每个区间内的平均预测置信度与实际事件发生频率，再按区间样本占比加权。ECE 越低，表示概率数值与经验频率越一致。

论文没有在本节给出温度变换的显式代数公式，只描述了逐位置的一维温度网格搜索。若采用标准二分类 Temperature Scaling，常见形式是：

\[
\widetilde c_k=
\sigma\left(\frac{\operatorname{logit}(c_k)}{T_k}\right),
\qquad
\widetilde a_k=\prod_{i=1}^{k}\widetilde c_i.
\]

在固定此前位置校准结果的条件下，搜索 \(T_k\) 使第 \(k\) 个累计前缀概率的 ECE 最小。以上显式公式是标准 Temperature Scaling 的解释性表达，不是论文在本节印出的原始公式。

### 6.5 Non-anticipating Property：为什么 Early Stopping 与无损性有关？

标准推测采样要求：是否验证或接受当前位置的决策，不能依赖未来还没有验证的候选 Token。

DSpark 的 Confidence Head 使用：

\[
c_k=\sigma(w^\top[h_k;W_1[x_{k-1}]]).
\]

因此：
- 计算 \(c_k\) 需要已知前一个候选 Token \(x_{k-1}\)；
- 计算 \(a_{k+1}\) 就需要知道 \(x_k\)；
- 后续位置的存活概率包含了当前 Token 的采样信息。

如果先把整块的所有存活概率都算出来，再做一次事后全局最优化，那么位置 \(k\) 的准入决策就可能受到 \(x_k\) 及其后续信息的影响。这会产生 Selection Bias。

论文采用的解决办法是：按照贪心路径逐步准入，并在吞吐量首次下降时立即停止。由于同一请求内部：

\[
a_{r,j+1}\leq a_{r,j},
\]

后续位置不会在全局排序中越过此前的前缀位置。Early Stopping 使当前截断点只由已处理的前缀决定，不再通过事后全局搜索利用未来 Token 的信息。

必须同时满足两个条件：
1. **拒绝采样规则正确**：拒绝时从残差分布修正，保持目标分布。
2. **验证长度选择满足非预知性**：不能让未来候选 Token 影响当前准入事件。

两者同时满足，才能维持精确的目标分布恢复。附录 A 将通过反例进一步说明不满足第二个条件会发生什么。

### 6.6 Early Stopping 的全局最优性条件

Algorithm 1 一旦发现：

\[
\Theta\leq\Theta_{\mathrm{best}}
\]

就停止。这意味着算法只检查贪心排序路径上的一个前缀，不继续搜索后续候选。

如果 \(\Theta\) 沿这条路径是单峰的，那么第一次下降后不会再次超过之前的峰值，因此 Early Stopping 可以得到全局最大值。

但如果 SPS 曲线存在明显的非平滑波动，吞吐量可能先下降、之后再次上升，算法就可能错过更好的解。

论文明确指出：其逐步提前停止只有在目标函数单峰时才能得到全局最大吞吐量，隐含假设是硬件容量曲线平滑下降。真实系统中的非平滑 SPS 和异步流水线问题将在 Section 5.2 讨论。

### 6.7 复杂度与工程实现分析

论文没有在本节给出 Algorithm 1 的渐近复杂度，下面是根据伪代码进行的独立分析。

令候选总数 \(N=R\gamma\)：
1. 计算每个请求的前缀存活概率：\(O(R\gamma)\)。
2. 构造并全局排序候选集合：\(O(R\gamma\log(R\gamma))\)。
3. 逐候选更新 Batch、期望接受量，并查询 SPS 表：最坏 \(O(R\gamma)\)。

因此主要复杂度为：

\[
\boxed{O(R\gamma\log(R\gamma))}
\]

存储候选和前缀概率需要 \(O(R\gamma)\)。

工程上，SPS 表应当按目标模型、推理引擎版本、GPU 配置和相关 Batch 形态进行 Profiling。若把一条 SPS 曲线直接复用于完全不同的硬件或模型配置，调度目标就会失真。

论文脚注 2 还给出了一个重要简化：在其 DeepSeek-V4 部署环境下，Decode 的请求数和上下文长度经负载均衡后大致均衡，因此用验证 Token Batch 大小 \(B\) 作为主要吞吐量索引。对于上下文长度高度不均衡、未进行 Prefill/Decode 分离或存在长序列偏斜的其他系统，SPS 可能还需要考虑上下文长度等额外维度。这是根据论文假设得出的工程启示，不是本文在本节中提出的新公式。

## 七、与已读论文的联系

| 方法 | 接受概率建模 | 验证长度选择 | 与 DSpark 的区别 |
| --- | --- | --- | --- |
| EAGLE / EAGLE-3 | 具体实现和版本不同 | 以树结构、候选策略为主 | DSpark 明确把条件接受概率校准后用于系统吞吐量调度 |
| Medusa | 可使用典型接受等策略 | 多头候选树及固定/配置的验证结构 | DSpark 不是扩展多条候选树，而是对单条半自回归草稿前缀动态分配验证预算 |
| DFlash | 主要依靠 Draft/Target 分布匹配提高接受率 | 通常验证给定 Draft Block | DSpark 增加 Confidence Head 和全局负载感知的逐请求前缀调度 |
| DSpark | 预测 \(c_{r,k}\)，校准累计 \(a_{r,j}\) | SPS 曲线驱动的全局贪心调度 | 同时考虑数据侧接受概率和系统侧 Batch 容量 |

DSpark 把两个通常分开处理的问题结合起来：模型层面，这个 Token 有多大概率通过验证？系统层面，在当前负载下，再验证这个 Token 是否值得？这也是它区别于单纯基于置信度阈值裁剪草稿的核心。

## 八、Section 3.2 小结与原始素材核对

1. Confidence Head 预测的是**条件接受概率** \(c_k\)，不是前缀联合存活概率。
2. 前缀存活概率 \(a_{r,j}\) 必须通过条件概率连乘得到。
3. STS 从左到右校准累计前缀概率，原因是调度器依赖概率的绝对数值，而不只是排序。
4. Scheduler 最大化 \(\Theta=\tau\cdot SPS(B)\)，而不是单纯最大化每个请求的 Draft 接受长度。
5. 全局贪心排序利用每个请求内部的存活概率单调性；Early Stopping 同时承担非预知性约束。
6. Early Stopping 的全局最优性依赖 \(\Theta\) 单峰假设；真实非平滑 SPS 曲线的工程处理留到 Section 5.2。

**原文核对：**
- PDF 第 6 页下半部分：Section 3.2 开始、Confidence Head、公式 (7)–(8)。
- PDF 第 7 页：Post-hoc Calibration、STS、3.2.2 开始。
- PDF 第 8 页：Algorithm 1、全局吞吐量目标及脚注 2。
- PDF 第 9 页上半部分：贪心准入、Non-anticipating Property、Early Stopping 和单峰性条件。
- 本节没有 Figure 或 Table；Algorithm 1 已提取原始 PDF 图像并在阅读对话展示。PNG 尚未提交到 GitHub 二进制目录，来源记录在 figures/README.md。

**已完成：Section 3.2（3.2.1 Confidence Head、3.2.2 Hardware-Aware Prefix Scheduler）。下一单元：Section 3.3 Training，完整翻译公式 (9)–(11)，分析 Cross-Entropy、Total Variation、Confidence 三项损失与位置权重。**
