# Section 2.3 Extensions — 完整中文翻译、概率校核、LoRA 自蒸馏与优化树算法

- 论文：**Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads**
- 版本：arXiv:2401.10774v3（2024-06-14），27 页
- 阅读范围：**Section 2.3.1 Typical Acceptance / Section 2.3.2 Self-Distillation / Section 2.3.3 Searching for the Optimized Tree Construction（全文）**
- 原版 PDF：[第 5 页，2.3.1](https://arxiv.org/pdf/2401.10774v3#page=5) · [第 6 页，2.3.2/2.3.3](https://arxiv.org/pdf/2401.10774v3#page=6)
- [arXiv v3 HTML 的同一章节](https://arxiv.org/html/2401.10774v3#S2.SS3)
- 本节**没有新增的 Figure 或 Table**，但是 2.3.1 引用后续实验 Figure 5，2.3.3 引用 Appendix C 的 Figure 6；本阅读单元不把后续图表当成本节原图，也不提前声称那些实验图表已精读完成。

## 一、原文完整中文翻译（严格按段落、公式和脚注顺序）

### 2.3.1 Typical Acceptance — 典型接受机制

**第 1 段。** 在推测解码论文（Leviathan et al., 2022; Chen et al., 2023）中，作者使用拒绝采样（Rejection Sampling），以生成具有多样性、同时符合原始模型概率分布的输出。然而，后续实现（Joao Gante, 2023; Spector & Re, 2023）表明，随着采样温度增大，这种采样策略的效率会降低。直观来看，可以通过一种极端情况理解：假设草稿模型与原始模型完全相同。在贪心解码下，草稿模型的全部输出都会被接受，因此效率达到最大。相反，拒绝采样会引入额外开销，因为草稿模型与原始模型会独立进行采样。即使两者的概率分布完全相同，草稿模型的输出仍可能被拒绝。

**第 2 段。** 然而，在实际应用中，语言模型通常使用采样生成多样化回答，温度参数只是用于调节回答的“创造性”。因此，较高温度应该使原始模型有更多机会接受草稿模型的输出。我们认为，通常没有必要使输出分布完全匹配原始模型。因此，我们提出典型接受机制（Typical Acceptance），通过选择合理的候选，而不是使用拒绝采样完成验证。此方法受到截断采样（Truncation Sampling，Hewitt et al., 2022）的启发，进一步解释见 Appendix A。我们的目标是选择典型（Typical）的候选，也就是由原始模型生成的概率并非极低的候选。

**第 3 段。** 我们采用原始模型的预测概率来衡量候选的合理性，并根据预测分布设定接受阈值。具体地，给定上下文 $(x_1,x_2,\ldots,x_n)$，在评估候选序列 $(x_{n+1},\ldots,x_{n+K+1})$（由原始 LM Head 与 Medusa Heads 的最高概率预测构造）时，考虑以下条件：

$$
p_{\mathrm{original}}(x_{n+k}\mid x_1,x_2,\ldots,x_{n+k-1})
>
\min\left(
\epsilon,\,
\delta\exp\left[-H\left(p_{\mathrm{original}}(\cdot\mid x_1,x_2,\ldots,x_{n+k-1})\right)\right]
\right).
$$

其中 $H(\cdot)$ 是熵函数，$\epsilon$、$\delta$ 分别为硬阈值和依赖熵的阈值参数。该条件改编自 Hewitt et al.（2022），基于两个观察：（1）概率相对较高的 Token 是有意义的；（2）当分布具有较高熵时，多种后续 Token 都可能被视为合理。

**第 4 段。** 在解码过程中，我们用此标准评估每条候选序列，若候选的一个前缀满足条件，则接受该前缀。为确保每个解码步骤至少生成一个 Token，我们对第一个 Token 采用贪心解码并无条件接受它，对后续 Token 使用 Typical Acceptance。当前步骤的最终输出由全部候选序列中**最长的已接受前缀**决定。

**第 5 段。** 分析该方案可以得到若干认识。首先，当温度为 0 时，它退化为贪心解码，因为只有概率最高的 Token 具有非零概率。当温度大于 0 时，若 $\epsilon,\delta$ 选择得当，贪心解码的输出总能够被接受，因为这些 Token 具有最高概率，从而实现最大的加速效果。类似地，在一般场景中，更高温度会使接受序列更长，我们的实验也验证了这一点。

**第 6 段。** 实验表明，Typical Acceptance 在保持相近生成质量的同时能够获得更高加速，如 Figure 5 所示。

### 2.3.2 Self-Distillation — 自蒸馏

**第 1 段。** Section 2.2 假设存在一个与目标模型输出分布匹配的训练数据集。但这不一定成立，例如模型提供者可能只发布模型而不公开其训练数据；或者模型已经经过 RLHF，输出分布与其训练数据集不同。为解决这一问题，我们提出自动化自蒸馏流程，让模型自身生成用于训练 Medusa Heads 的数据，使训练样本匹配模型自身输出分布。

**第 2 段。** 生成数据的过程很简单。首先，从与目标模型领域相似的公开数据中获取 Seed Dataset，例如聊天模型使用 ShareGPT（ShareGPT, 2023）。然后只抽取 Prompt，让模型对这些 Prompt 生成回复。为了生成多轮对话样本，可以按顺序把种子数据集的多个 Prompt 输入模型。对于 Zephyr-7B（Tunstall et al., 2023）这类在对话双方角色上都进行训练、具备 Self-Talk 能力的模型，只需给出第一条 Prompt，就能够让模型自行产生多轮对话。

**第 3 段。** 对于 Medusa-1，这种数据集足够训练多个 Heads。但对 Medusa-2，我们发现仅使用这些数据联合训练 Backbone 和 Medusa Heads，通常会降低生成质量。事实上，即便不训练 Medusa Heads，单独用这些数据训练 Backbone，也会造成性能下降。这意味着训练 Backbone 时，需要用**原始模型的概率预测**而不是把生成 Token 当成单一的 Ground Truth Label，类似经典的知识蒸馏工作（Kim & Rush, 2016）。具体而言，Backbone Loss 是：

$$
\mathcal L_{\mathrm{LM-distill}}=
KL\left(p_{\mathrm{original},t}^{(0)}\parallel p_t^{(0)}\right),
$$

其中 $p_{\mathrm{original},t}^{(0)}$ 是原始模型在位置 $t$ 的预测分布。

**第 4 段。** 朴素地实现上述蒸馏需要训练时维护两份模型以获取原始模型的预测分布，从而增加显存要求。为了缓解这一问题，我们提出一个简单有效的做法：用 LoRA（Hu et al., 2021）这样的参数高效适配器微调 Backbone。这样，当关闭 Adapter 时，同一模型就成为原始模型，因此不需要额外的 Teacher 模型副本。通过这种自蒸馏流程，可以训练 Medusa-2、保持 Backbone 能力，而且几乎不引入额外模型参数存储。最后一项建议是：在这个自蒸馏场景中，**最好使用不量化的 LoRA**；否则 Teacher 会变成量化后的模型，可能使生成质量下降。

### 2.3.3 Searching for the Optimized Tree Construction — 搜索优化树结构

**第 1 段。** Section 2.1.2 展示了通过笛卡尔积构造候选树的最简单方法。但在树节点总数具有固定预算时，规则树并不一定最好。直观而言，由不同 Heads 的最高概率候选组成的序列，其准确率可能不同，因此可以利用准确率估计来构造树。

**第 2 段。** 具体而言，使用 Calibration Dataset 计算不同 Heads 各个排名位置预测结果的准确率。令 $a_k^{(i)}$ 为第 $k$ 个 Head 的第 $i$ 名预测 Token 的准确率。**原文脚注 2**：这里准确率针对排名恰好第 $i$ 的单个 Token，即 Top-$i$ Accuracy 减去 Top-$(i-1)$ Accuracy。假设各个 Head 的预测准确率彼此独立，则不同 Heads 的排名组合 $[i_1,\ldots,i_k]$ 所组成的候选序列，其准确率可以估算为：

$$
w(i_1,\ldots,i_k)=\prod_{j=1}^{k}a_j^{(i_j)}.
$$

令 $I$ 为所有可能的排名组合的集合。$I$ 中每个元素都可以映射到树的一个节点，既包括叶子，也包括所有内部节点。于是，一棵候选树的期望接受长度为：

$$
\mathbb E[L_{\mathrm{accept}}]
=
\sum_{[i_1,\ldots,i_k]\in I}
\prod_{j=1}^{k}a_j^{(i_j)}.
$$

**第 3 段。** 从逐个增加节点来构造树的角度看，每新增一个节点，它对期望接受长度的贡献恰好是对应节点的准确率。因此，我们可以贪心地选择与当前树连接、且准确率最大的节点加入树中。重复此过程，直到达到规定的节点预算。这样就构造了能够最大化期望接受长度的树。更多细节见 Appendix C。

## 二、独立技术推导：Typical Acceptance（非论文原文）

### 2.1 阈值是什么

在当前经过温度/采样处理的 Target 分布 $p(\cdot)$ 下，论文阈值为：

$$
\tau(p)=\min(\epsilon,\delta e^{-H(p)}),\qquad
H(p)=-\sum_{x\in\mathcal V}p(x)\log p(x).
$$

仅当候选 Token $x$ 满足严格不等式 $p(x)>\tau(p)$，才允许继续接受。$\epsilon$ 是阈值**上界**（因为 $\min$），不是概率的硬下界。$e^{H(p)}$ 可以看作熵有效支持规模 / 困惑度（Perplexity），所以：

$$
\delta e^{-H(p)}=\frac{\delta}{\operatorname{PPL}(p)}.
$$

分布越平坦，熵越高、PPL 越大，$\tau$ 越低，可接受的不同 Token 通常就越多。在较低熵下，阈值可能被 $\epsilon$ 截断；增大 $\epsilon$ 在其他条件不变时不会放宽阈值，可能使接受更严格。将阈值写成 min 的上界这一点不可看反。温度 $T$ 改变分布 $p_T(x)\propto\exp(z_x/T)$，并不是直接将 $T$ 当作阈值。

### 2.2 数值例子（独立计算）

设 $\epsilon=0.25,\delta=0.5$，自然对数熵：
- 低熵 $p=(0.8,0.1,0.1)$：$H\approx0.6390$，$e^{-H}\approx0.5278$，阈值 $\min(0.25,0.2639)=0.25$，只有 $0.8$ 对应 Token 通过。
- 较高熵 $p=(0.4,0.35,0.25)$：$H\approx1.0805$，$e^{-H}\approx0.3394$，阈值 $\min(0.25,0.1697)=0.1697$，三个 Token 均通过。

候选在两种分布下的接受集合可不同。但两个分布是说明阈值的人工样例，并非作者给出的实验数据。

### 2.3 与严格 Speculative Sampling 的本质差异

标准精确推测采样中，Target 概率 $p(x)$、Draft 条件提议 $q(x)$，采样候选 $x\sim q$ 后以 $a(x)=\min(1,p(x)/q(x))$ 接受，拒绝时从 $r\propto[p-q]_+$ 重新采样。该机制的概率校正可以确保每个生成位置的最终分布为 $p$（前提是按正确的条件提议模型执行）。一次接受概率是 $\alpha=1-\mathrm{TV}(p,q)$。

**必须校核原文中的一个论证瑕疵：** 2.3.1 作者说“即使草稿与目标分布完全一致，拒绝采样仍可能拒绝，因为是独立采样”。这对于**标准正确实现的接受比检验**并不成立。如果 $p=q$，则对所有 $q(x)>0$ 的候选均有 $a(x)=\min(1,1)=1$，不可能被拒绝。只有误把两次独立抽样的“采样结果恰好相等”当成接受条件，才会出现即使分布一样却不总一致的现象。因此应忠实保留作者原句，但不可由原句推出标准 Speculative Sampling 在 $p=q$ 时无法全接受。温度提高带来的实际接受率下降，可能来自 $p_T$ 与草稿 $q_T$ 的偏离变化，而不是同分布依旧会被拒绝。

Typical Acceptance 使用确定的概率阈值判断合理性，通常**不是**准确的 $p$-分布采样器，并且选择最长可接受路径还会进一步改变输出偏好。$T=0$ 退化贪心这一声明需要保证原始 Target argmax 路径确实存在于候选树且按正确规则保留；仅有 Tree Attention 本身不足以保证此性质。$T>0$ 时“更高温度总带来更长接受”是论文实验观察与直觉，非所有 logits、所有 $\epsilon,\delta$ 都满足的全称定理。

### 2.4 Typical Acceptance 教学伪代码

~~~python
def typical_accept(target_probs, candidate_id, epsilon, delta):
    import math
    H = -sum(p * math.log(p) for p in target_probs if p > 0)
    threshold = min(epsilon, delta * math.exp(-H))
    return target_probs[candidate_id] > threshold
~~~
注意对每个候选节点应使用**其特定历史分支下**的 Target 条件分布，不能为所有节点复用一个位置的概率；第一 Token 由论文规定用 Target 贪心生成并无条件接受；继续检查每条路径上的连续前缀，最终取最长前缀。

## 三、独立技术推导：Self-Distillation 和 LoRA Teacher Switch

### 3.1 硬标签与软标签的差异

从自生成回复取硬标签 $y$ 做 Backbone CE，只是使学生提高 $q_\theta(y)$：$\mathcal L_{\mathrm{CE}}=-\log q_\theta(y)$。但教师模型的原始分布可能同时对多个 Token 赋予合理概率；硬标签不能保留“其他正确选项”的相对概率结构，有限自生成样本还存在模式偏置。

使用 Teacher 分布 $p_T$、Student 分布 $p_S$ 的 KL：
$$
D_{\mathrm{KL}}(p_T\|p_S)=\sum_{v=1}^V p_T(v)\log\frac{p_T(v)}{p_S(v)}
=\underbrace{-\sum_v p_T(v)\log p_S(v)}_{\mathrm{soft\ target\ CE}}-H(p_T).
$$
对 Student logits $z_S$ 的导数（未使用额外温度缩放）为：
$$
\frac{\partial D_{\mathrm{KL}}(p_T\|p_S)}{\partial z_{S,v}}=p_S(v)-p_T(v).
$$
因此学生在每个词表位置收到教师分布而非单一 One-Hot Token 的监督。

### 3.2 LoRA Teacher / Student 切换

固定基础参数 $\theta_0$，LoRA 学生在部分层上增加 $\Delta W=BA\cdot(\alpha/r)$。
- Teacher：同一个 Backbone，**关闭 Adapter**，前向获得原始分布 $p_T$；不保存反向图。
- Student：**开启 Adapter**，同样输入获得 $p_S$；计算 $KL(p_T\|p_S)$ 并通过适配器参数反向传播。
- Medusa Heads：用 Teacher 自生成序列的真实未来 Token 作为多位置交叉熵监督。
- 对应 Section 2.2 式 (2) 的一种自然扩展（**独立合并推断，非本节正式新公式**）为：
$$
\mathcal L_{\mathrm{joint}}=\mathcal L_{\mathrm{LM-distill}}+\lambda_0\sum_{k=1}^K\lambda_k\mathcal L_k.
$$

**实现伪代码（PEFT 不同版本 API 需以本地安装为准）**：
~~~python
# source model frozen; adapter updates form the student
with torch.no_grad():
    with model.disable_adapter():
        teacher_logits = model(**batch).logits.detach()
student_logits = model(**batch).logits
teacher_logp = teacher_logits.float().log_softmax(-1)
student_logp = student_logits.float().log_softmax(-1)
teacher_p = teacher_logp.exp()
loss_kl = (teacher_p * (teacher_logp - student_logp)).sum(-1)
loss_kl = (loss_kl * loss_mask).sum() / loss_mask.sum()
loss = loss_kl + lambda0 * medusa_aux_loss
~~~
需保证 Teacher 前向的 Adapter 真正关闭，Target LoRA 初始化和模型保存状态正确，Teacher 的概率在相同输入条件/温度处理下计算。冻结原始权重不等于零计算/零额外内存：Teacher Forward、Teacher Logits、激活张量以及 KV 缓存仍有时间和内存成本；节省的是**一整份额外 Teacher 参数副本**。

### 3.3 量化与 Teacher 选择

论文特别建议这个场景避免量化 LoRA：若基础主干已经量化，关闭 Adapter 得到的是量化后的教师，其输出与原始未量化模型可能不同，容易导致自蒸馏目标退化。这与 Section 2.2 里“Medusa-1 可用 QLoRA/量化降低成本”不矛盾：**Medusa-1 Backbone 冻结并只训练 Heads**，而 Self-Distillation 的 Medusa-2 需要使用原始模型软概率作为 Teacher 监督。

## 四、独立技术推导：Optimized Tree Greedy Search

### 4.1 Rank Accuracy 与 Top-i Accuracy 的差别

若校准集第 $k$ 个 Head 的 Top-1 正确率是 0.55，Top-2 累计正确率 0.80，则：
$$
a_k^{(1)}=0.55,\qquad a_k^{(2)}=0.80-0.55=0.25.
$$
这表示真实目标恰好位于预测排名第 1 和排名第 2 的概率。**$a^{(2)}$ 不是 Top-2 累计准确率**，不可误用 0.80 替代 0.25。

### 4.2 联合路径评分与期望的推导

一条长度 $k$ 的 Rank Path $v=(i_1,\dots,i_k)$，根据论文的独立性假设：
$$
w(v)=\prod_{j=1}^k a_j^{(i_j)}.
$$
如果“正确下一个 Token”在每步是唯一事件，而且完整正确 Rank 组合在同一深度互斥，则能将某深度至少有一条正确候选路径的概率写成 $\sum_{v\text{ depth }k}w(v)$。又因为非负整数随机变量 $L$ 的 Tail-Sum 公式：
$$\mathbb E[L]=\sum_{k\ge1}\Pr(L\ge k),$$
可推得论文的节点概率和。这个表达式是**建模近似**：不同 Heads 的排名准确率一般不独立；如果采用 Typical Acceptance，多个不同 Token 可能同时被判断为合理，节点事件可能不互斥，因而该节点和未必严格等于实际最长可接受前缀期望。

### 4.3 Greedy 扩展为何在作者近似目标下合理

对非根节点 $v$，$w(v)=w(parent(v))\,a_k^{(i)}\le w(parent(v))$，因为每项排名准确率在 $[0,1]$ 内。这意味着**子节点的分数不会超过父节点分数**。在该固定校准权重模型、固定节点数且仅要求前缀闭包的条件下，按分数从高到低逐个选节点自然保持祖先先被选，贪心选择最高分的可连接 Frontier Node 就等价于选择 N 个高分节点（同分时可以有多个最优解），因此可最大化 $\sum w(v)$。这一最优性证明仅适用于作者假设下的加和目标，不等于实际 Wall-Time Speedup 最优。

### 4.4 两 Head 数值例子

假设 $a_1=(0.55,0.25)$、$a_2=(0.60,0.25)$，并给出 5 个**不含虚拟根**的树节点预算：

| Rank 路径 | 权重 w | 备注 |
| --- | ---: | --- |
| (1) | 0.55 | Head 1 的 rank-1 |
| (2) | 0.25 | Head 1 的 rank-2 |
| (1,1) | 0.33 | 0.55×0.60 |
| (1,2) | 0.1375 | 0.55×0.25 |
| (2,1) | 0.15 | 0.25×0.60 |
| (2,2) | 0.0625 | 0.25×0.25 |

贪心添加顺序：$(1)$ 0.55 → $(1,1)$ 0.33 → $(2)$ 0.25 → $(2,1)$ 0.15 → $(1,2)$ 0.1375。树总权重是 $1.4175$；这一数值是作者独立排名假设下的预测接受长度（只计额外 Head 对应节点，不计 LM 根 Token），不是测量速度或实际 Target 接受长度。

### 4.5 参考实现（教学算法，不是官方代码）

~~~python
import heapq

def optimized_tree(rank_accuracies, max_nodes):
    # rank_accuracies[k] = 第 k+1 个 Head 的逐排名准确率列表
    # 空前缀为虚拟根，实际不计入 max_nodes
    heap = []
    for rank, acc in enumerate(rank_accuracies[0], start=1):
        heapq.heappush(heap, (-acc, (rank,)))
    selected = []
    while heap and len(selected) < max_nodes:
        neg_score, path = heapq.heappop(heap)
        score = -neg_score
        selected.append((path, score))
        depth = len(path)
        if depth < len(rank_accuracies):
            for rank, acc in enumerate(rank_accuracies[depth], start=1):
                child = path + (rank,)
                heapq.heappush(heap, (-(score * acc), child))
    return selected

demo = optimized_tree([[.55, .25], [.60, .25]], 5)
assert [x[0] for x in demo] == [
    (1,), (1,1), (2,), (2,1), (1,2)
]
assert abs(sum(w for _, w in demo) - 1.4175) < 1e-12
~~~

如果有更多 Rank 候选、更多 Head，堆可以按需生成其子节点；理论上在预算 N 限制下堆操作大约 $O(Ns_{\max}\log(Ns_{\max}))$（若每个扩展节点最多入堆 $s_{\max}$ 个子节点），但这是此教学实现的实现复杂度，并非论文给出的复杂度结论。实际 Tree Optimization 还须考虑模型的有效接受分布、树规模 N 对 GPU 计算成本、Top-k Candidate Mapping 和 KV Cache 布局。

## 五、三篇已读论文的跨论文关联（额外分析）

| 对比 | Medusa | EAGLE（2401.15077） | DFlash（2602.06036） |
| --- | --- | --- | --- |
| 草稿来源 | 多未来位置的 Token Heads | Feature-level 自回归 + Shifted Token | Block Diffusion 并行预测 |
| 验证形态 | 多分支 Tree Attention | 多分支树式 Target Forward | Target 并行验证块 |
| 接受策略 | 可用精确分布校正或 Typical Acceptance | 正确 Speculative Sampling 下保持 Target 分布 | 正确校正下保持 Target 分布 |
| 树/块优化 | 利用排名准确率乘积做固定预算 Greedy Search | 树形并行验证，正式树多为启发式 | Block Size、并行 Draft 成本、锚点设计 |
| 原模型知识保护 | Medusa-2 的 KL 自蒸馏 + LoRA Teacher Switch | 冻结 Target、训练特征草稿 | 冻结 Target、通过多层特征和 KV Injection 提供条件 |

Medusa 的优化树机制使“候选数”与“候选结构”成为独立可调对象；相对 EAGLE 使用启发式树的路径，Medusa 为校准权重下的树形节点选择提出了明确的加和目标。不过该目标是**预期接受长度的近似**，不能直接等价于真实服务延迟的最优；论文 Section 3.3、Appendix C 与 G 中进一步研究树形验证开销。

## 六、原文与独立分析边界、阅读断点

- **完成**：2.3.1 全部六段及完整阈值式；2.3.2 全部四段及 KL 公式、原文量化 LoRA 提示；2.3.3 三段、脚注 2、路径权重与期望公式。
- **原文值得保留的校核提示**：标准正确拒绝采样在 $p=q$ 时应有 100% 接受，与 2.3.1 原文“同分布仍可能拒绝”的解释不一致；树的乘积模型基于预测准确率独立性，在 Typical Acceptance 下不是精确的期望接受长度定理。
- **本节无新的 Figure/Table**：Figure 5 是后续实验图，Figure 6 在 Appendix C，未在本节重新绘制冒充原图。
- **下一个阅读单元**：Section 3 Experiments（3.1–3.3，Figure 3–5、Table 1–2 和全部实验消融；表图所在页随原文核实）。 
