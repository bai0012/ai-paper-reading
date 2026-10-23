# Appendix A · Implementation Details — A.1 Tree Structure 与 A.2 Multi-Round Speculative Sampling

> **论文：** [EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/html/2401.15077v3)  
> **版本：** arXiv:2401.15077v3（2025-03-04）  
> **原始 PDF：** [第 12 页（Appendix A 全部正文、Figure 9）](https://arxiv.org/pdf/2401.15077v3#page=12)、[第 13 页（Algorithm 1）](https://arxiv.org/pdf/2401.15077v3#page=13)  
> **阅读日期：** 2026-10-23  
> **范围：** Appendix A 的 A.1、A.2 的每个完整段落、Figure 9 原图图注和 Algorithm 1 全部伪代码。后续同页的 Appendix B 属于另一阅读单元，不在此处提前混入译文。  
> **来源边界：** 第一部分是**原文完整翻译**；其余数学证明、代码、数值示例、工程建议、与 DFlash 的比较是**独立分析**。  
> **原图保存状态：** 本次对话已从用户上传的原版 PDF 裁出 Figure 9 和 Algorithm 1 原始截图，并在对话中展示；本文件提供原始 PDF 链接。当前 GitHub PNG 二进制未上传，不创建虚假的 figures/ 相对链接。

## 一、Appendix A 原文顺序完整翻译

### A. Implementation Details（实现细节）

### A.1 Tree Structure（树结构）

**原文第 1 段：**

利用树注意力（Tree Attention），EAGLE 能够生成树状结构的草稿（Tree-structured Draft）。Figure 9 左侧展示了草稿的树结构，右侧则展示了不使用 Tree Attention 时对应的链状草稿（Chain-structured Draft），后者用于 Section 4.3.1 中介绍的消融实验。

在贪心设置（Greedy Setting）下，我们选择概率最高的 Top-$k$ 个 Token 作为子节点。在非贪心设置（Non-greedy Setting）下，我们采样 $k$ 个 Token。每个节点的子节点数 $k$ 可由 Figure 9 推断，例如根节点的 $k=4$。

无论使用树状草稿还是链状草稿，Draft Model 都会在草稿生成阶段执行 **5 次前向传播**。在验证阶段，Target LLM 通过**一次前向传播**计算每个候选 Token 的概率。

**原文第 2 段：**

**为什么采用这样的树结构？** Figure 9 中的结构并没有经过严格优化，而是基于直觉设计：概率较高的 Token 分支应当更深、更宽。

本文所有模型的全部实验都采用 Figure 9 所展示的草稿结构。然而，最优树结构可能依赖上下文。例如 Batch Size 越大，冗余计算资源越少，使用较小的树可能更好。优化草稿结构有可能进一步改善性能。

### Figure 9 原图与图注

**原图所在位置：** [PDF 第 12 页 Figure 9](https://arxiv.org/pdf/2401.15077v3#page=12)。本次对话已直接裁出原始图片；左边为不均匀分叉的候选树，右边为深度对应的 Chain Draft。

**图注完整翻译：** 图 9：EAGLE 的草稿结构。左侧展示使用 Tree Attention 时的草稿结构，右侧展示不使用 Tree Attention 时的草稿结构。

### A.2 Multi-Round Speculative Sampling（多轮推测采样）

**原文唯一正文段落：**

与推测采样中的链状草稿不同，EAGLE 使用树状草稿，因此需要修改采样算法。

推测采样算法 A 可以简述为：如果一个 Token 被接受，就返回它；否则，从调整后的分布中采样一个 Token。

对于具有 $k$ 个候选 Token 的树状草稿，多轮推测采样递归调用算法 A。

与在一个 Token 被拒绝后直接从调整后的分布采样不同，多轮推测采样会再次调用算法 A。如果所有候选都被拒绝，才直接从调整后的分布采样。多轮推测采样伪代码见 Algorithm 1。

### Algorithm 1：多轮推测采样（完整翻译，保留原文印刷形式）

**原图所在位置：** [PDF 第 13 页 Algorithm 1](https://arxiv.org/pdf/2401.15077v3#page=13)。原始截图在本次对话直接展示。

**输入：** 目标分布 $p$；对于 $i=1,\ldots,k$，候选 $t_i$ 和分布 $\hat p_i$，其中 $t_i$ 从 $\hat p_i$ 中采样。

**输出：** 采样值 $x\sim p$。

~~~text
i ← 1
for i ≤ k do
    r ← U(0, 1)
    if r < p(t_i) / p̂_i(t_i) then
        return t_i
    end if
    p ← norm(max(0, p(t_i) − p̂_i(t_i)))
    i ← i + 1
end for
Sample t ∼ p
return t
~~~

**原文忠实性提醒：** 伪代码的残差更新行按 v3 PDF 原样保留，不能悄悄替换为向量公式。原文明确写为 $p(t_i)-\hat p_i(t_i)$，这两个量均是标量；下一次循环却要求 $p$ 是完整词表概率向量。因此，**逐字照写的 Algorithm 1 不能直接被实现为合法的概率采样器。** 下一节单独解释数学上合理的更正形式。

**Appendix A 原文翻译结束；接下来是独立技术讲解。**

---

## 二、Figure 9 具体结构与 Tree Attention（独立技术讲解）

### 1. 不均匀分叉的直觉

Figure 9 左侧是一棵非均匀分叉树：根节点分成 $k=4$ 个候选，高概率分支获得更深/更宽的后续扩展，低概率分支较早终止。右侧则是链状序列，每个位置一个后继。

重要的是，**一次 Target Verification 实际只会保留一条合法连续路径上的 Token 前缀**，不是把整棵树所有节点都计入接受长度 $\tau$。更宽的树提高命中可能性，但也增加单 Forward 的算力、显存及 KV 成本。

### 2. 同样 5 次 Draft Forward，为什么树可以更宽？

一轮 Draft Forward 能批量计算多个树分支的当前节点，而沿深度方向仍然需要顺序递推。因此 Tree Draft 和 Chain Draft 的 Draft Forward **轮数**都为 5，但每轮计算 Token 数量、对应 attention/KV 工作量并不相同。Section 3 Figure 6 的 3 次 Forward / 10 Token 只是另一幅**流程示意**；Appendix A Figure 9 才是正文声称“本文所有实验使用”的正式树拓扑，不能将两个示意的 3 与 5 当成实际超参数冲突。

### 3. 正确的 Tree Attention Mask

对于候选节点 $u$，仅允许注意到已经确认的公共历史、它自己以及沿合法分支的祖先节点：

\[
M_{uv}=\begin{cases}
0,&v\in\mathrm{Ancestors}(u)\cup\{u\}\text{ 或为因果可见的公共历史},\\
-\infty,&v\text{ 是其他候选分支节点}.
\end{cases}
\]

这里是抽象解释式，不是论文附录提供的逐元素 Matrix。模型计算时必须让一个分支的状态与单独运行 Target、处理相同路径前缀所得到的状态相符。逻辑位置相同的兄弟节点也应具有相符的 Position IDs/RoPE 对齐，而不是简单使用批量张量中不同存储位置作为逻辑位置。

### 4. 作者明确指出树拓扑没有严格优化

原作者通过直觉设计拓扑，并指出 Batch Size 变大可能需要小树。一个独立的优化形式是：

\[
\max_{\mathcal T}\ \frac{\mathbb E[\text{本轮实际输出 Token 数}]}{
 T_{\mathrm{draft}}(\mathcal T)+T_{\mathrm{verify}}(\mathcal T)+T_{\mathrm{overhead}}(\mathcal T)}.
\]

$\mathcal T$ 表示树结构；这是独立研究建议而非作者原式。接受率与树宽、树深和 GPU 负载的联合优化尚有研究空间。

---

## 三、Algorithm 1 印刷公式的类型问题与正确残差写法

论文**原样**写的是：

\[
p\leftarrow\operatorname{norm}(\max(0,p(t_i)-\hat p_i(t_i))).
\]

其中 $p(t_i),\hat p_i(t_i)$ 均为实数，无法直接形成所需的词表概率分布 $p\in\mathbb R^V$。这并不是仅有“是否写 min(1,p/q)”的简写问题，而是损失掉了全词表概率的信息。

根据 Section 2 的标准严格校正规则，**数学上合理的向量式修正**应为：

\[
\boxed{p\leftarrow\operatorname{Normalize}([p-\hat p_i]_+)}.
\]

这里 $[z]_+=\max(0,z)$ 是逐 Token 操作。

**不能不加条件地认定此修正就是完整的可直接部署树算法。** Algorithm 1 没有具体定义同一父节点多个候选的联合分布、条件相关性及贪心 Top-$k$ 的实际提议概率。而它的 Input 写“$t_i$ sampled from $\hat p_i$”。对确定性 Top-$k$、无放回多候选或树分支关联抽样，候选的真实条件提议分布可能不是一组彼此独立的原始 Draft Softmax 分布；需要对实际提议机制做正确校正，否则简单代入也不能自动声称采样精确。

在贪心解码下，可以直接对比目标模型的 argmax 来保证选择正确的确定性 Token；这不代表随机采样的严格分布证明自动适用于一组确定性 Top-$k$ 提议。

---

## 四、在明确采样条件下的 Multi-Round 数学证明（独立推导）

### 1. 定义逐次剩余分布

令 Target 初始分布为 $p_0=p$。第 $i$ 次候选在**已拒绝前 $i-1$ 个候选这一条件下**，合法服从提议分布 $q_i$，这是后面证明所必需的条件。

第 $i$ 轮以：

\[
a_i(x)=\min(1,p_{i-1}(x)/q_i(x))
\]

验收，平均接受率：

\[
\alpha_i=\sum_x q_i(x)a_i(x)=\sum_x\min(p_{i-1}(x),q_i(x)).
\]

发生拒绝时，剩余目标分布：

\[
\boxed{p_i(x)=\frac{[p_{i-1}(x)-q_i(x)]_+}{1-\alpha_i}}
\]

当 $1-\alpha_i=0$ 时该拒绝事件不可能出现，因而无需构造 $p_i$。当 $q_i(x)=0$ 时候选无法取到 $x$，不需计算 $p/q$ 的零分母。

### 2. 单次分解守恒

因为：

\[
\min(a,b)+[a-b]_+=a,
\]

所以：

\[
\boxed{
p_{i-1}(x)
=q_i(x)a_i(x)+(1-\alpha_i)p_i(x).
}
\]

这说明某轮被接受的概率质量与被拒绝后保留的概率质量恰好加回原目标分布。

### 3. 反向归纳证明 $D_1=p_0$

令 $D_i$ 表示从第 $i$ 轮开始完成所有剩余尝试后的输出分布。所有 $k$ 轮拒绝时直接从 $p_k$ 采样，所以 $D_{k+1}=p_k$。

对第 $i$ 轮：

\[
D_i(x)=q_i(x)a_i(x)+(1-\alpha_i)D_{i+1}(x).
\]

若 $D_{i+1}=p_i$，则：

\[
D_i(x)=\min(p_{i-1}(x),q_i(x))+[p_{i-1}(x)-q_i(x)]_+=p_{i-1}(x).
\]

由数学归纳：

\[
\boxed{D_1=p_0=p_{\rm Target}}.
\]

**精确性前提不能遗漏：** 每轮样本按对应合法的条件提议分布产生；正确的向量残差和归一化；Target 各分支 Causal logits 正确；位置与 KV 对齐；无自行放宽的近似接受规则。若一批 $t_i$ 事先联合相关地产生，简单把它们当作独立 $q_i$ 抽样套公式，证明未必成立。

### 4. 三 Token、两轮候选的完整算例

词表 $\{A,B,C\}$，Target：

\[
p_0=(0.40,0.35,0.25).
\]

令第一轮合法提议：

\[
q_1=(0.25,0.20,0.55).
\]

第一轮直接接受质量：

\[
\min(p_0,q_1)=(0.25,0.20,0.25),\quad \alpha_1=0.70.
\]

拒绝概率 0.30，剩余分布：

\[
p_1=\operatorname{Normalize}((0.15,0.15,0))=(0.5,0.5,0).
\]

第二轮合法提议：

\[
q_2=(0.60,0.10,0.30).
\]

第二轮条件接受质量 $(0.5,0.1,0)$，$\alpha_2=0.60$。再拒绝时残差 $p_2=(0,1,0)$。将三条输出来源的**无条件**概率质量相加：

| 输出途径 | A | B | C |
| --- | ---: | ---: | ---: |
| 第一轮接受 | 0.25 | 0.20 | 0.25 |
| 第一轮拒绝，第二轮接受 | 0.15 | 0.03 | 0 |
| 两次拒绝，残差最终采样 | 0 | 0.12 | 0 |
| **输出总概率** | **0.40** | **0.35** | **0.25** |

因此结果恰为 $p_0$。这里的 $q_2$ 假设是在第一轮被拒绝的条件下合法的候选分布，算例不能当作任意预生成相关候选的精确性保证。

### 5. 条件接受率与候选数量

若每轮残差分布和合法条件提议分布定义了 $\alpha_i$，那么没有任何候选被接受的概率（按链式条件概率）为：

\[
\prod_{i=1}^{k}(1-\alpha_i).
\]

因此至少一个候选被接受的概率是：

\[
1-\prod_{i=1}^{k}(1-\alpha_i).
\]

$\alpha_i$ 通常随剩余分布变化，不能随意用相同固定值取代；额外尝试也会带来计算成本。该表达是所述合法条件概率机制下的独立推导，非论文 Appendix A 原式。

---

## 五、树式验证与实际 Transformer 执行

1. **Draft 建树：** 历史 Target Features 与 Shifted Token 条件化的轻量 Decoder，按预设非均匀拓扑扩展候选。Greedy 使用 Top-$k$；非贪心按相应提议机制采样 $k$ 候选。
2. **一次 Target Forward：** 把候选树各节点批量送入 Target，以 Tree Attention 屏蔽非祖先兄弟分支；同时使用路径逻辑位置对应的 RoPE / Position ID。
3. **沿路径执行接受规则：** 在每个生成位置，同父候选依次尝试 Multi-Round 验证；遇拒绝时正确更新剩余条件目标分布。被接受 Token 决定下一位置进入哪一条树子分支。
4. **结束与提交：** 当候选用尽，从合法残差分布采样；只将最终已确认的 Token 路径及其 Target Features 写入下一轮上下文。已失效分支的 KV 不能作为有效前缀复用。

**Tree Attention 解决的是概率计算的并行性，不是用最大概率路径代替随机采样。** Token 输出的分布正确性由精确接受与剩余概率修正规则负责；在有相关提议的真实树采样器中还必须维护其正确的条件提议机制。

### 仅用于阐明向量残差的 PyTorch 示意（非 EAGLE 官方实现）

~~~python
import torch

def multi_round_from_conditionals(p0, proposal_distributions):
    """演示顺序合法条件提议；非预生成相关 Token Tree 的完整实现。"""
    p = p0.clone().to(dtype=torch.float64)

    for q in proposal_distributions:
        q = q.to(dtype=torch.float64)
        # 本轮候选在前轮拒绝条件下确实从 q 采样
        t = torch.multinomial(q, num_samples=1).item()
        accept_prob = min(1.0, (p[t] / q[t]).item())
        if torch.rand(()).item() < accept_prob:
            return t
        residual = (p - q).clamp_min(0)  # **逐 Token 向量相减**
        z = residual.sum()
        if z <= 0:
            raise RuntimeError("Reject branch with zero residual is inconsistent")
        p = residual / z

    return torch.multinomial(p, num_samples=1).item()

p_target = torch.tensor([0.4, 0.35, 0.25])
q1 = torch.tensor([0.25, 0.20, 0.55])
q2 = torch.tensor([0.60, 0.10, 0.30])
# 在上述独立条件提议的 toy 设定中，重复大量采样应接近 p_target
~~~

此代码特意在每次拒绝后按对应分布采样一个候选，而不是模拟 EAGLE 非均匀候选树或确定性 Top-$k$。它用于说明 **vector residual + conditional proposal** 的概率计算语义，不可将其冒充官方完整 Tree Verification 实现。

---

## 六、与 EAGLE 正文实验以及 DFlash 的研究联系

### Figure 9 与实验效果

Section 4.3.1 的消融比较 Tree vs Chain：平均接受长度 $\tau$ 增加约 0.6–0.8，延迟加速比绝对增加约 0.3–0.5。Appendix A.1 解释了该结果背后的结构设置：树更宽、候选覆盖更广，却也需要更多单轮 Token 计算。作者未严格搜索树拓扑；高并发负载下可能更适合较小树。

### EAGLE 与已读 DFlash

[本篇 EAGLE](README.md)：Draft Feature 仍随树的深度逐层自回归推进，树的宽度方向允许同一 Forward 扩展多个候选，Target 再使用树注意力和多轮验证复用计算。

[已精读 DFlash](../2602.06036-dflash/README.md)：将草稿生成转为块内并行的 Block Diffusion，使用 Target 多层隐藏特征和 Layerwise KV Injection 条件化草稿，优化草稿关键路径延迟。

统一的工程性能解释模型（独立推导）：

\[
\text{Average token latency}\approx
\frac{T_{\rm draft}+T_{\rm verification}+T_{\rm overhead}}
{\mathbb E[\text{每轮实际输出 Tokens}]}.
\]

- EAGLE 的 Figure 9 强调如何分配候选树宽度与深度。
- DFlash 强调如何减少多个候选 Token 的自回归草稿步骤。
- 两者都必须在正确的 Target 验证规则下才具有概率分布保持保证。
- DFlash 与 EAGLE-3 的实验不能未经对齐就视为与本篇原版 EAGLE 的直接性能对比。

---

## 七、阅读状态及原图保存说明

- 已完成：Abstract、Sections 1–6、Appendix A（A.1 全部 2 段、A.2 全部正文 + Algorithm 1）。
- **下一单元：Appendix B Detailed experimental results**（PDF 第 12–13 页；尤其是 Table 8，包括 HumanEval、GSM8K、Alpaca 上 Temperature=0 的接受率细节）。
- Figure 9：原版 PDF 第 12 页；已在本次对话裁图展示和翻译图注。Algorithm 1：原版 PDF 第 13 页；已在对话展示原始截图、按原样译出并单独指出标量残差印刷问题。
- PNG 图像二进制：本次未上传至 GitHub，需持续标记为 github_binary_saved=false。章节保留 PDF 精确页码链接，不生成不存在的图片相对路径。
- 本篇完整知识总结、summary.md 与索引最终状态，应在完成 Appendix B 后再生成和标记，避免提前宣告全文完成。
