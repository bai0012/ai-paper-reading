# Appendix C · Visualization of Optimized Tree Attention

**论文**：[Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774)  
**版本**：arXiv:2401.10774v3（2024-06-14）  
**原始 PDF**：[第 15 页：Figure 6](https://arxiv.org/pdf/2401.10774v3#page=15)  
**HTML 原文**：[Appendix C](https://arxiv.org/html/2401.10774v3)  
**阅读范围**：Appendix C 的全部正文（一个段落）及 Figure 6 图注；补充 Section 2.3.3、Section 2.1.2、官方实现和已读 EAGLE 的关联分析。  
**来源边界**：第一部分是 v3 原文完整翻译；公式推导、官方代码核对、实现解释和跨论文比较均单独标为技术分析，不混入原文。

## 一、Appendix C 原文完整翻译

### C. Visualization of optimized tree attention（优化树注意力的可视化）

Figure 6 展示了 Medusa-2 Vicuna-7B 模型所使用的一种稀疏构建树（sparsely constructed tree）的结构。

该树结构向下延伸了四个层级，这意味着计算过程中使用了四个 Medusa Heads。

这棵树最初通过 Cartesian Product（笛卡尔积）方法构建，随后根据 Alpaca-eval 数据集（Dubois et al., 2023）上测量得到的、各个 Medusa Head 的 Top-k 预测的统计期望，对树进行剪枝（Pruning），从而得到优化后的结构。

树结构在视觉上呈现出向左倾斜的特点，这代表算法倾向于选择各个 Head 上具有更高概率的节点。

### Figure 6 图注完整翻译

**图 6：Medusa-2 Vicuna-7B 的一种稀疏树配置的可视化。** 这棵树包含 64 个节点，每个节点代表一个候选 Token；树的深度为 4，表示计算过程中涉及 4 个 Medusa Heads。每个节点表示某个 Medusa Head 的 Top-k 预测中的一个 Token，边表示这些节点之间的连接关系。红色线条突出显示了能够正确预测未来 Token 的路径。

**原文翻译结束。** 本附录没有其他正文段落、公式、表格或额外图表。Figure 6 是本节唯一的原图。

原图位置：[arXiv v3 PDF 第 15 页](https://arxiv.org/pdf/2401.10774v3#page=15)。本次阅读已核对 PDF 页面截图；仓库未提交 PNG 二进制，具体原因和来源记录见 [figures/figure6-source.md](figures/figure6-source.md)，不创建虚假的图片相对路径。

## 二、先读懂 Figure 6：64 个节点、4 个 Head 与一条红色路径

### 2.1 64 个节点不等于 64 条候选序列

Figure 6 描述的是一棵候选 Token 树，而不是 64 条互相独立的完整 Token 序列。

- **节点（node）**：树中某个位置的一项 Token 候选。
- **边（edge）**：候选 Token 与其前驱 Token 之间的连接关系。
- **路径（path）**：从根节点出发、沿树向下到达某个节点或叶节点的一串候选 Token。
- **深度（depth）**：从根节点沿树向下的逻辑层级；Medusa 的 4 个 Head 对应 4 个 Medusa 候选层。

一棵树可以有 64 个节点，但完整根到叶路径的数量小于 64；不同路径共享前缀，Tree Attention 正是利用这种共享来避免把相同前缀重复计算。

### 2.2 官方代码对 64 节点的精确核对

论文图注报告 64 个候选节点、4 层深度。作者官方仓库的 [medusa/model/medusa_choices.py](https://github.com/FasterDecoding/Medusa/blob/main/medusa/model/medusa_choices.py) 将 Vicuna-7B 的树定义为 mc_sim_7b_63。该配置含 **63 个 Medusa 路径/节点定义**；[medusa/model/utils.py](https://github.com/FasterDecoding/Medusa/blob/main/medusa/model/utils.py) 中的 generate_medusa_buffers() 明确执行：

~~~python
medusa_len = len(sorted_medusa_choices) + 1
~~~

并将索引 0 留给根节点。官方 Notebook 也说明，tree_candidates 包含根节点，而根节点是原始模型 LM Head 的预测。因此，这套实现的计数关系是：

\[
\boxed{64=1\text{ 个原始 LM Head 根节点}+63\text{ 个 Medusa 候选节点}}
\]

这解释了为什么配置名带有 63，而论文图注报告 64 个节点：二者计数口径不同，不是矛盾。

官方配置中的 63 个 Medusa 节点还可以按深度精确统计：

| 树位置 | 节点数 | 说明 |
| --- | ---: | --- |
| 根节点（LM Head） | 1 | 原始模型预测的第一个 Token |
| Head 1（深度 1） | 10 | 该 Head 的 rank 0–9 候选 |
| Head 2（深度 2） | 28 | 连接在已保留 Head 1 节点下的候选 |
| Head 3（深度 3） | 23 | 连接在已保留 Head 2 节点下的候选 |
| Head 4（深度 4） | 2 | 最深层仅保留两个候选 |
| **总计** | **64** | 根节点 + 四层 Medusa 节点 |

这里的 10、28、23、2 是从作者发布的 mc_sim_7b_63 配置逐项统计得到的**代码事实**，不是 Appendix C 正文直接列出的数字。它也表明 Figure 6 不是一个每层都等宽的规则树：越往后保留的节点越少。

如果每个 Head 都保留 Top-10 并完整取 Cartesian Product，4 个 Head 会产生：

\[
10+10^2+10^3+10^4=11110
\]

个 Medusa 候选节点（另计 LM Head 根节点）。实际配置只保留 63 个 Medusa 节点，显著减少待验证的候选规模。

作者 Notebook 还指出，该配置产生 42 条根到叶候选路径。**42 是完整路径数，64 是树节点数**，二者不可混为一谈。

### 2.3 四层到底代表什么

Section 2.1.1 定义：给定 Backbone 在位置 \(t\) 的隐藏状态 \(h_t\)，第 \(k\) 个 Medusa Head 预测位置 \(t+k+1\) 的 Token，而原始 LM Head 预测 \(t+1\)。

因此，当前时刻的逻辑顺序为：

| 逻辑位置 | 预测模块 | 预测目标 |
| --- | --- | --- |
| 根 / 深度 0 | 原始 LM Head | \(t+1\) |
| 深度 1 | Medusa Head 1 | \(t+2\) |
| 深度 2 | Medusa Head 2 | \(t+3\) |
| 深度 3 | Medusa Head 3 | \(t+4\) |
| 深度 4 | Medusa Head 4 | \(t+5\) |

所以“深度 4”是指 4 个 Medusa Head 层，不是 4 个节点，也不是 64 个 Head。根节点是原始 LM Head 的预测；它与四层 Medusa 候选一起构成实现中的 64 个验证节点。

### 2.4 Figure 6 的红色正确路径

从原图可读到，红色路径展示了一个实际正确的连续候选序列，视觉上的 Token 依次为：

\[
\text{Once}\rightarrow\text{upon}\rightarrow\text{a}
\rightarrow\text{time}\rightarrow\text{,}
\]

其中根部的 Once 是原始 LM Head 的首个 Token，后续节点依次来自四个 Medusa Head。红线表示该示例中沿路径预测正确的 Token 序列。

要注意：

1. 红线不是树构造算法预先固定的“最优分支”，而是图中用于说明正确预测的示例路径。
2. 红线不表示每次推理都一定走这条路径；输入上下文变化后，Target 验证得到的正确路径也会变化。
3. 节点被纳入候选树，只表示它值得交给 Target 验证，不意味着它已经被接受。
4. 最终接受的是 Target 验证规则允许的最长连续前缀；在第一次不匹配处，后续节点不能越过错误父节点单独被接受。

## 三、从 Section 2.3.3 还原优化树贪心算法

Appendix C 只描述最终树的来源。具体优化目标和构造过程在主文 Section 2.3.3，下面将其与 Figure 6 对齐。

### 3.1 校准集测量的是“某个排名恰好正确”的频率

在校准数据集上，分别统计第 \(k\) 个 Medusa Head 的第 \(i\) 名预测的准确率，记为：

\[
a_k^{(i)}.
\]

论文脚注特别指出，这里的准确率是**单独第 \(i\) 名 Token 的准确率**，不是 Top-\(i\) 累积准确率。若记：

\[
A_k^{\mathrm{top}\text{-}i}
=
P(\text{正确 Token 出现在 Head }k\text{ 的前 }i\text{ 名}),
\]

则：

\[
\boxed{
a_k^{(i)}
=
A_k^{\mathrm{top}\text{-}i}
-
A_k^{\mathrm{top}\text{-}(i-1)}
}
\]

例如（以下数字仅用于解释，并非论文实验数据）：某 Head 的 Top-1 准确率为 0.65，Top-2 累积准确率为 0.80，则：

\[
a_k^{(1)}=0.65,\qquad
a_k^{(2)}=0.80-0.65=0.15.
\]

不能把 \(a_k^{(2)}\) 误写成 0.80。Top-2 累积准确率表示前两名中任一命中；rank-2 准确率只表示正确 Token 恰好排在第二名。

Appendix C 指明，这些统计量是在 **Alpaca-eval 数据集**上测量的。这里 Alpaca-eval 是树结构的校准数据来源；不能与 Appendix F 在 AlpacaEval 上报告最终推理速度的评测用途混为一谈。

### 3.2 Rank Path 的准确率估计

设某条路径依次选择四个 Head 的排名：

\[
[i_1,i_2,i_3,i_4].
\]

例如 \([1,2,1,3]\) 表示：
- Head 1 选择 rank 1；
- Head 2 选择 rank 2；
- Head 3 选择 rank 1；
- Head 4 选择 rank 3。

论文假设不同 Head 的预测准确事件相互独立，因此将整条候选路径正确的概率估计为：

\[
\boxed{
P(\text{path correct})
\approx
\prod_{j=1}^{4}a_j^{(i_j)}
}
\]

对任意深度 \(d\) 的节点，其路径前缀 \([i_1,\ldots,i_d]\) 的分数为：

\[
q(i_1,\ldots,i_d)
=
\prod_{j=1}^{d}a_j^{(i_j)}.
\]

这不是简单地把当前 Head 的 Top-1 概率作为节点分数，而是将**从根到该节点的全部排名选择**累乘。越深的节点，需要整条前缀都正确，因此其估计概率通常更低。

### 3.3 为什么期望接受长度等于树中所有节点路径概率之和

令树中每个节点 \(v\) 对应一个候选路径前缀，定义指示变量：

\[
X_v=
\begin{cases}
1,&\text{从根到 }v\text{ 的路径全部预测正确},\\
0,&\text{否则}.
\end{cases}
\]

一次验证中，树上能够连续接受的节点数可表示为：

\[
L=\sum_{v\in T}X_v.
\]

根据期望的线性性质：

\[
\mathbb E[L]
=
\sum_{v\in T}\mathbb E[X_v]
=
\sum_{v\in T}P(X_v=1).
\]

在论文的跨 Head 独立性假设下：

\[
\boxed{
\mathbb E[L]
\approx
\sum_{[i_1,\ldots,i_d]\in I}
\prod_{j=1}^{d}a_j^{(i_j)}
}
\]

其中 \(I\) 包含树中**所有节点对应的路径前缀**，不仅仅是叶节点。原因是一个正确的中间前缀本身就贡献一个已接受 Token；即使后续分支预测失败，这个前缀仍然有价值。

### 3.4 贪心地添加“当前可连接且分数最高”的节点

Section 2.3.3 的构造逻辑可以整理成如下步骤：

1. 从原始 LM Head 根节点开始。
2. 生成完整 Cartesian Product 候选空间中的可选子节点。
3. 对每个候选节点，计算其路径分数 \(q(i_1,\ldots,i_d)\)。
4. 只允许添加父节点已经在当前树中的节点，保证树连通。
5. 在当前可连接的 frontier 中，选择路径分数最高的节点加入树。
6. 展开该节点的下一层候选，更新 frontier。
7. 重复以上过程，直到达到目标节点预算（本例为 64 个验证节点，包含根节点）。

伪代码（对 Section 2.3.3 的独立整理，不是论文原始算法框）：

~~~python
tree = {root}
frontier = children(root)

while len(tree) < node_budget:
    node = max(frontier, key=path_accuracy)
    frontier.remove(node)
    tree.add(node)

    for child in children(node):
        if child not in tree:
            frontier.add(child)
~~~

这里 path_accuracy(node) 对应从根到该节点的排名准确率乘积。贪心策略的关键约束是：**不能跳过父节点直接加入一个更深的后代**。否则候选树不连通，后续 Tree Attention 也无法构造合法的祖先上下文。

### 3.5 为什么树会向左倾斜

Figure 6 的左偏不是一个独立的几何目标，而是排名准确率与贪心节点预算共同作用的结果：

- 在同一个 Head 内，rank 1 通常比 rank 2、rank 3 更容易命中真实 Token。
- 在较早 Head 上选择高排名候选后，整条路径的乘积通常更大。
- 贪心搜索优先扩展高分路径，因此这些早期、高排名分支会获得更多后续节点预算。
- 较低排名的分支虽然仍可能正确，但其路径分数更低，往往较早停止扩展。

因此，树不是对称的 \(s_1\times s_2\times s_3\times s_4\) 规则树，而是一个按预期收益分配预算的非均匀树。Figure 6 的左偏正是这一选择结果的可视化。

## 四、Tree Attention 如何验证这 64 个节点

树结构负责组织候选，Tree Attention 负责让 Target 模型在一次并行验证前向中正确处理它们。

### 4.1 每个节点只允许关注自己的祖先路径

对节点 \(v\)，其可见历史由以下部分组成：

\[
\mathrm{Visible}(v)
=
\mathrm{Prompt}
\cup
\{\mathrm{root}\}
\cup
\mathrm{Ancestors}(v)
\cup
\{v\}.
\]

节点不能看到兄弟节点、其他分支的 Token 或与自身无关的后代。否则，一个候选分支的 Token 会泄漏到另一个候选分支的上下文中，验证结果就不再对应各自的假设序列。

作者官方实现 [generate_medusa_buffers()](https://github.com/FasterDecoding/Medusa/blob/main/medusa/model/utils.py) 通过以下逻辑构造掩码：

- 先创建单位矩阵，保证节点能关注自己；
- 再令每个节点都能关注根节点；
- 根据其 rank path 找到所有祖先索引，并开放对应的 Attention 位置。

### 4.2 Position ID 使用树深度，而不是扁平化索引

把树节点摊平成一个长度为 64 的张量后，张量下标不等于 Token 在真实候选序列中的逻辑位置。

例如两个兄弟节点虽然在扁平化张量中的 index 不同，但它们都位于同一预测深度，因此应使用相同的逻辑 Position ID。作者的 generate_medusa_buffers() 按树深度设置 medusa_position_ids：

\[
\mathrm{position\_id}(v)=\mathrm{depth}(v).
\]

这保证 RoPE 等位置编码反映的是 Token 在候选序列中的时间位置，而不是树节点在内存中的存储顺序。

### 4.3 Target 一次 Forward 验证整棵树，随后重建候选路径

作者实现还生成 retrieve_indices，把扁平化的树节点输出重新组合成根到叶路径。官方 Notebook 的 Vicuna-7B 示例指出：
- tree_candidates 包含根节点，即原始 LM Head 的预测；
- 候选路径由树节点重建，示例中共有 42 条根到叶路径；
- 验证后选择最长的合格前缀，并将已接受路径对应的 KV 状态用于下一轮解码。

因此，**64 个节点不是 64 次 Target Forward，也不是 64 条独立 Batch 样本**。它们是一个树状验证输入中的节点，使用 Tree Attention Mask 共享公共前缀。

### 4.4 红色路径与贪心验收

在 Greedy Decoding 下，Target 对候选路径的每个位置计算下一 Token 的 argmax。只有当前路径上的候选 Token 与 Target 的贪心预测一致时，才能继续向下接受。

当某节点不匹配时，该分支的后代都不能越过错误前缀被接受；其他独立分支仍可继续检查。最终选择能够连续匹配的最长路径前缀。

Figure 6 的红色路径可视为一个成功匹配的示例。它说明了为什么稀疏树不仅要保留高概率候选，还必须保留这些候选之间的父子连接。

## 五、与 Section 2.3.3、Figure 2 规则树及 Figure 4 消融的关系

| 位置 | 解决的问题 | 树的特点 |
| --- | --- | --- |
| Section 2.1.2 / Figure 2 | 如何并行处理多个 Head 的 Top-\(s_k\) 组合 | 规则 Cartesian Product；第 \(k\) 层分支数为 \(s_k\) |
| Section 2.3.3 | 固定节点预算时，应该保留哪些候选 | 用校准准确率估计路径收益，贪心添加可连接的高分节点 |
| Appendix C / Figure 6 | 最终优化树长什么样 | Vicuna-7B 的 64 节点、4 Head 稀疏树；左偏，红色路径示例 |
| Section 3.3.1 / Figure 4 | 稀疏树是否值得 | 64 节点的优化稀疏树在 Acceleration Rate 上优于 256 节点的随机密集树；更大的树会增加计算开销 |

Figure 4 的结论并不是“节点越少越快”或“64 是所有硬件的普适最优值”。它表明在论文的 Vicuna-7B 实验设置中，优化候选结构比单纯扩大候选节点数更有效；更大的树可能使线性层和 Attention 的矩阵计算开销上升，抵消额外接受 Token 带来的收益。

## 六、与 EAGLE Tree Attention 的跨论文对比

两者都使用树结构组织多个候选，并利用 Target 的树状 Attention Mask 在一次 Target Forward 中验证多个节点。但树是如何生成的，两篇论文有本质区别。

| 维度 | Medusa（2401.10774v3） | EAGLE（2401.15077v3） |
| --- | --- | --- |
| Draft 的预测对象 | Token | Target 的隐藏 Feature |
| Draft 条件 | 多个 Head 共享当前 \(h_t\)，各自预测不同未来位置 | 历史预测 Feature + Shifted Token Embedding |
| 候选树生成 | 各 Head Top-k 的 Cartesian Product，再按校准准确率剪枝 | Feature-level Autoregression 逐层扩展树 |
| 树的选择 | Section 2.3.3 的 rank accuracy greedy search；Figure 6 是校准得到的静态稀疏树 | 原始 EAGLE Appendix A 的固定非均匀分支结构，按经验为高概率分支分配更多深度/宽度 |
| Draft 串行开销 | 多个 Head 并行产生候选，不需要每层单独执行一个自回归 Draft Forward | 需要多轮 Draft Forward；原论文 Appendix A 报告 Tree/Chain Draft 均为 5 次 Draft Forward |
| Target 验证 | 一次 Tree Attention Forward | 一次 Tree Attention Forward |
| Attention 可见范围 | Prompt、根节点、当前节点及祖先路径 | Prompt、当前节点及祖先路径 |
| Position ID | 按树深度设置，不能用扁平化下标替代 | 按树上逻辑位置设置，不能让无关分支互相泄漏 |
| 接受机制 | 可用 Rejection Sampling；Typical Acceptance 则放宽严格分布匹配 | 使用递归多轮 Speculative Sampling，按接受概率和残差分布保持 Target 分布 |

### 6.1 两者相同的是验证结构，不是 Draft 算法

Medusa 和 EAGLE 都需要为候选树构造拓扑感知的 Causal Mask：

- 每个节点可以看到共享 Prompt；
- 每个节点可以看到自己的祖先；
- 每个节点可以看到自己；
- 不可以看到兄弟分支和其他无关分支。

这使得多个候选节点能作为一组 Token 送入 Target 模型，并在一次 Target 前向中并行计算。

### 6.2 Medusa：并行 Token Heads，树由排名组合产生

Medusa 的第 \(k\) 个 Head 直接从同一个当前隐藏状态 \(h_t\) 预测未来位置 \(t+k+1\)。因此 Head 1、Head 2、Head 3、Head 4 可以并行输出 Top-k 列表；再通过笛卡尔积把这些排名组合成候选路径。

优化树算法解决的是：**给定节点预算，哪些 rank path 值得留下？**

它不需要先生成 Head 1 的某个 Token，再把这个 Token 送入一个 Draft Transformer 预测 Head 2 的 Feature。Medusa 用并行预测换取低 Draft 串行深度，但不同 Head 的预测在构造候选时并非像自回归 Draft 那样逐步条件化。

### 6.3 EAGLE：Feature-level Autoregression，树由逐步 Draft 扩展产生

EAGLE 使用目标模型倒数第二层 Feature，并将提前一个时间步的 Shifted Token Embedding 作为条件；轻量 Draft 网络预测下一个 Feature，再借用共享 Target LM Head 形成 Token。

因此：
1. EAGLE 先预测下一层 Feature；
2. 再根据该 Feature 产生 Token；
3. 将预测 Feature 与 Shifted Token 作为后续 Draft 的条件；
4. 对多个分支逐层扩展，构成树状 Draft；
5. Target 用 Tree Attention 一次验证整棵树。

这与 Medusa 的“多个 Head 共享同一个 \(h_t\)，各自直接猜未来 Token”不同。EAGLE 的 Draft 具有更强的逐步条件化能力，但要付出多次 Draft Forward 的串行开销。

### 6.4 原始 EAGLE 与 EAGLE-2 不应混为一谈

本节比较的是已读的原始 EAGLE（arXiv:2401.15077v3）及其 Appendix A Figure 9。原论文说明其非均匀树是基于直觉设计，并非通过 Medusa Section 2.3.3 那样的 AlpacaEval rank-accuracy greedy search 严格优化。

后续 EAGLE-2 的动态树机制是另一项改进，不能把它的动态树策略倒推成原始 EAGLE 的实现。

### 6.5 接受机制的区别

EAGLE 的递归多轮 Speculative Sampling 会按 \(p/q\) 接受候选，并在拒绝时使用残差分布，因此目标分布保持是其设计目标。

Medusa 也可以使用 Rejection Sampling；但论文重点提出的 Typical Acceptance 根据 Target 概率阈值接受合理候选，允许输出分布偏离原始 Target 分布，以换取更高的接受效率。因此，不能把“二者都用 Tree Attention”误解成“二者都采用相同的验收规则”。

## 七、实现复现与实验边界

### 7.1 官方实现中的关键结构

- [medusa_choices.py](https://github.com/FasterDecoding/Medusa/blob/main/medusa/model/medusa_choices.py)：mc_sim_7b_63 保存 63 个 rank-path 选择。
- [utils.py](https://github.com/FasterDecoding/Medusa/blob/main/medusa/model/utils.py)：generate_medusa_buffers() 负责构造树 Mask、Position IDs、Tree Indices 和 Retrieval Indices。
- [medusa_model_new.py](https://github.com/FasterDecoding/Medusa/blob/main/medusa/model/medusa_model_new.py)：Vicuna-7B 推理默认使用 mc_sim_7b_63。
- [官方 Notebook](https://github.com/FasterDecoding/Medusa/blob/main/notebooks/medusa_introduction.ipynb)：解释树候选、根节点、42 条候选路径和 Tree Attention 的实际调用。

复现时不要只保存 64 个 Token 排名，而要保存每个节点的父节点/路径、逻辑深度、Attention Mask、Position IDs 和 Retrieval Indices。仅有一份节点 Token 列表无法重建树式因果关系。

### 7.2 校准准确率独立性假设的局限

论文使用：

\[
P(\text{path correct})
\approx
\prod_{j=1}^{d}a_j^{(i_j)}
\]

这依赖不同 Head 排名命中事件独立的假设。实际中各 Head 共享同一个 Backbone 隐藏状态，预测目标又是相邻未来 Token，因此命中事件可能相关。乘积是用于树搜索的估计，不是对真实联合正确概率的严格恒等式。

因此，若在新模型或新数据分布上复现，建议：
- 用与部署输入分布相近的校准集重新测量 rank-specific accuracy；
- 同时测量完整 Rank Path 的真实联合命中率，检查独立性近似误差；
- 在固定节点预算下比较 greedy tree、规则 dense tree 与其他稀疏树；
- 分别记录平均接受长度、Tree Forward 延迟、显存和实际 tokens/s；
- 在不同 Batch Size、上下文长度、温度和任务上重新测量，而不是直接把 64 节点视为普适最优。

### 7.3 64 节点不是普适常数

论文 Figure 6 展示的是 Medusa-2 Vicuna-7B 在 Alpaca-eval 校准条件下的一种配置。Section 3.3.1 与 Figure 4 进一步指出，优化的 64 节点稀疏树在该实验中的 Acceleration Rate 优于最多 256 节点的随机密集树，但树增大后矩阵计算与 Attention 开销会抵消收益。

因此，64 是该模型与实验配置的结果，而不是 Medusa 理论要求每个模型都固定使用 64 个候选节点。

## 八、阅读小结与进度

- Appendix C 全部正文及 Figure 6 图注已按 v3 PDF 完整翻译。
- 原图已在原 PDF 第 15 页核对；GitHub 未保存 PNG 二进制，已记录准确图号、页码和来源，不创建虚假图片路径。
- 官方代码核对：63 个 Medusa rank-path 节点 + 1 个 LM Head 根节点 = 64；四个 Head 层的节点数为 10、28、23、2；该配置共有 42 条根到叶候选路径。
- 机制重点：校准集统计每个 Head 每个 rank 的独立命中准确率；用路径准确率乘积近似候选序列正确概率；在树连通约束下贪心添加收益最高的节点；Tree Attention 用祖先 Mask 和深度 Position ID 并行验证整棵树。
- 跨论文结论：Medusa 与 EAGLE 共用树状验证的核心思想，但 Medusa 是并行 Token Heads + 静态校准稀疏树，EAGLE 是 Feature + Shifted Token 的自回归 Draft + 树状验证。两者不能因都使用 Tree Attention 就视为相同 Draft 算法。
- **下一阅读单元：Appendix D Results of Speculative Decoding。** Medusa 主文与 Appendices A–C 已完成，D–G 待精读；整篇论文尚未完成。
