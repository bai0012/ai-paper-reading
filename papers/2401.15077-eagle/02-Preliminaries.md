# Section 2 · Preliminaries（完整翻译与 Speculative Sampling 数学证明）

> 论文：**EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty**  
> arXiv：[2401.15077v3](https://arxiv.org/html/2401.15077v3)  
> **来源：** 用户提供的 v3 原始 PDF，第 3 页右栏，Section 2；阅读日期：2026-10-14。  
> **范围：** 第 2 节完整两个段落 Notations 与 Speculative sampling。本节无 Figure/Table。  
> **标注约定：**「一、逐段译文」对应论文原文；其余数学推导、示例、公式拓展、代码和跨论文讨论均属于**独立解释**，不可误写成 EAGLE 论文新增实验或定理。

## 一、逐段完整译文

### Notations（符号约定）

在本文中，“目标 LLM”（Target LLM）指需要进行推理加速的语言模型，而“草稿模型”（Draft Model）则指用于生成草稿的模型。

“特征”（Feature）通常指 LLM 倒数第二层的特征，即位于 LM Head 之前的隐藏状态（Hidden State）。

Token 使用小写字母 $t$ 表示，它们的 Embedding 使用 $e$ 表示，特征使用 $f$ 表示，概率分布使用 $p$ 表示。序列使用大写字母表示，例如：

\[
T_{i:j}=(t_i,t_{i+1},\ldots,t_j).
\]

在 LLM 中，输入 $T_{1:j}$ 首先经过 Embedding Layer 转换为 Embedding 序列 $E_{1:j}$，然后转换为特征序列 $F_{1:j}$。LM Head 将特征 $f_j$ 映射为概率分布：

\[
p_{j+1}=\mathrm{LMHead}(f_j),
\]

随后从这个概率分布中采样得到下一个 Token $t_{j+1}$。因此，对于任意整数 $j\ge1$，普通的 Token 级自回归生成过程可以表示为：

\[
T_{1:j}\to E_{1:j}\to f_j\to p_{j+1}\to t_{j+1}.
\]

### Speculative sampling（推测采样）

推测采样通过草稿生成（Drafting）和验证（Verification）两个阶段工作。在草稿生成阶段，使用一个较小的模型生成 $\gamma$ 个 Token：

\[
\hat T_{j+1:j+\gamma}
\]

以及它们对应的概率分布：

\[
\hat P_{j+1:j+\gamma}.
\]

在验证阶段，只需要对目标 LLM 执行一次前向传播，就可以得到对应的概率分布：

\[
P_{j+1:j+\gamma}.
\]

随后，对这些 Token 按顺序逐个进行评估。对于草稿 Token $\hat t_{j+i}$，其接受概率为：

\[
\min\left(1,\frac{p_{j+i}(\hat t_{j+i})}{\hat p_{j+i}(\hat t_{j+i})}\right).
\]

如果某个 Token $\hat t_{j+i}$ 被拒绝，则丢弃其后所有草稿 Token，并根据以下修正后的概率分布重新采样当前位置的 Token：

\[
\operatorname{norm}\left(\max(0,p_{j+i}-\hat p_{j+i})\right).
\]

正如推测采样论文（Leviathan et al., 2023）附录 A.1 所证明的，这种方法等价于直接从目标 LLM 中进行采样。EAGLE 采用这一方法，因此确保无论在贪心解码还是非贪心解码设置下，生成文本的概率分布都保持不变。

**译文到此结束；下一段原文为 Section 3 EAGLE。**

---

## 二、符号、维度及普通自回归

| 符号 | 类型/形状（解释性） | 含义 |
|---|---|---|
| $t_i$ | 整数 | 第 $i$ 位置的 Token ID |
| $e_i$ | $\mathbb R^d$ | Token Embedding |
| $f_i$ | $\mathbb R^d$ | LM Head 前的隐藏特征 |
| $p_{i+1}$ | $\mathbb R^V$ | Target 的有效下一 Token 概率分布 |
| $\hat p_{i+1}$ | $\mathbb R^V$ | Draft 的有效下一 Token 概率分布 |
| $T_{i:j}$ | 长度 $j-i+1$ 的序列 | Token 子序列 |
| $\gamma$ | 正整数 | 一轮 Draft 的 Token 数 |
| $V,d$ | 正整数 | 词表大小、Hidden Size（讲解补充） |

例如 $d=4096$、$V=32000$，$f_i\in\mathbb R^{4096}$，LM Head 线性映射可用 $W_{\rm head}\in\mathbb R^{32000\times4096}$，随后做 Softmax 得到词表分布。在实际实现里，LM Head 常只输出 logits，因此还需做 temperature/top-k/top-p 等采样变换；**采样和验证必须使用匹配的有效分布**。论文把这一阶段简写为 $p=\mathrm{LMHead}(f)$。

这里的 Feature 不是输入 Embedding，也不是任意中间层输出；作者所指为最终 LM Head 之前的隐藏状态。

## 三、单步 Speculative Sampling：接受概率推导

为去掉时间下标，记当前前缀条件下 Target 分布为 $p(x)$，Draft 提议分布为 $q(x)$（对应论文 $\hat p$），实际草稿候选 $X\sim q$。

接受候选 $x$ 的条件概率写成 $a(x)$。则“提议且接受”的概率质量为 $q(x)a(x)$。为了使这条路径分配给 $x$ 的概率不超过 Target 所需的 $p(x)$，必须满足：

\[
0\le a(x)\le1,\quad q(x)a(x)\le p(x).
\]

为了**最大化接受质量**（在候选本身直接输出、拒绝后独立修正的设计中），选择：

\[
\boxed{a(x)=\min(1,p(x)/q(x))}
\]

其中 $q(x)>0$ 只需在实际有机会采样到的 Token 上成立。若 $q(x)=0$ 则它不可能作为该轮候选，无需除法。

因此接受分支贡献：

\[
q(x)a(x)=\min(p(x),q(x)).
\]

全词表平均接受率：

\[
\alpha=\sum_x q(x)a(x)=\sum_x\min(p(x),q(x)).
\]

## 四、拒绝后的残差分布

在接受分支之后，Token $x$ 还欠缺的概率质量：

\[
p(x)-\min(p(x),q(x))=[p(x)-q(x)]_+
\]

其中 $[z]_+=\max(0,z)$。

由于所有 $p(x)$ 求和为 1，剩余质量恰为拒绝概率：

\[
Z=\sum_x[p(x)-q(x)]_+=1-\alpha.
\]

若 $Z>0$，定义拒绝时的条件采样分布：

\[
\boxed{r(x)=\frac{[p(x)-q(x)]_+}{\sum_y[p(y)-q(y)]_+}}.
\]

**为什么不能拒绝后直接从 $p$ 重抽：** 接受分支已经消耗了 $\min(p,q)$ 的概率质量，再抽 $p$ 会将不应补充的 Token 重复分配，从而改变输出分布。残差分布只补尚欠的部分。

当 $Z=0$ 时 $p=q$，拒绝事件零概率，无需构建 $r$；这是实现中应处理的边界。若 $q(x)=0$ 而 $p(x)>0$，该位置只可能经拒绝后的残差分支补足，不破坏结论。

## 五、严格概率证明

令 $Y$ 为最终输出的 Token。根据全概率公式：

\[
\begin{aligned}
\Pr(Y=x)
&=\Pr(X=x,\mathrm{accept})+\Pr(\mathrm{reject})\Pr(Y=x\mid\mathrm{reject})\\
&=q(x)a(x)+(1-\alpha)r(x)\\
&=\min(p(x),q(x))+[p(x)-q(x)]_+\\
&=p(x).
\end{aligned}
\]

故 $\boxed{Y\sim p}$，即最终一步的 Token 边缘概率精确等于 Target 采样。证明不是说 Draft 总会采样出与 Target 同一条随机轨迹的 Token，而是说**生成结果的分布**相同。

还可以推导：

\[
\begin{aligned}
\alpha
&=\sum_x\min(p(x),q(x))\\
&=\frac12\sum_x(p(x)+q(x)-|p(x)-q(x)|)\\
&=1-\frac12\sum_x|p(x)-q(x)|\\
&=\boxed{1-\operatorname{TV}(p,q)}.
\end{aligned}
\]

其中 $\operatorname{TV}(p,q)=\frac12\lVert p-q\rVert_1$。这说明随机采样的平均接受率与两个**有效概率分布**的重合程度严格相关，不能仅用 Draft 的 Top-1 Accuracy 代表。

## 六、可核算的三 Token 示例

设 $p=(0.6,0.3,0.1)$，$q=(0.4,0.4,0.2)$，词表仅有 A、B、C。

| Token | $p$ | $q$ | $a(x)=\min(1,p/q)$ | $q(x)a(x)$ | 正残差 $[p-q]_+$ |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 0.60 | 0.40 | 1.00 | 0.40 | 0.20 |
| B | 0.30 | 0.40 | 0.75 | 0.30 | 0 |
| C | 0.10 | 0.20 | 0.50 | 0.10 | 0 |
| **总和** | **1** | **1** | — | **0.80** | **0.20** |

平均接受率 $\alpha=0.8$，拒绝概率 $0.2$。归一化残差得 $r=(1,0,0)$。最终输出的概率质量：

- A：$0.4+0.2\cdot1=0.6$
- B：$0.3+0.2\cdot0=0.3$
- C：$0.1+0.2\cdot0=0.1$

最终恢复 $p$。若错误地在拒绝后直接从 $p$ 采样，则结果为 $(0.52,0.36,0.12)$，不等于 $p$。

## 七、多 Token 验证为何成立？单轮的时序、概率与 KV

1. 对当前前缀 $T_{1:j}$，Draft 自回归生成 $\gamma$ 个候选 $\hat t_{j+1},\ldots,\hat t_{j+\gamma}$，并保存每个候选对应的条件提议分布 $\hat p_{j+i}(\cdot\mid T_{1:j},\hat T_{j+1:j+i-1})$。
2. Target 在**同一次 Causal Forward** 中处理已有前缀和候选串，计算每个候选位置的目标条件分布 $p_{j+i}(\cdot\mid T_{1:j},\hat T_{j+1:j+i-1})$。并行的是大模型的计算；接受/拒绝决定仍须按位置顺序。
3. 依次为候选 $\hat t_{j+i}$ 做 Bernoulli 接受测试；若通过，则这个 Token 可追加到已验证前缀，继续检查下一个。
4. 遇到第一个拒绝，则用当前位置 Target $p_{j+i}$ 与 Draft $\hat p_{j+i}$ 形成残差分布重新采样该位置，并**丢弃后面所有草稿 Token**。因为后面这些 Token 的提议和 Target 条件都基于已被拒绝的错误前缀，不能继续复用。
5. 按照第 $i$ 个位置条件化的单步证明，输出的下一 Token 服从 Target 在已验证前缀上的条件分布。重复应用条件概率的链式法则，即得到完整 Token 序列的联合分布与逐 Token 从 Target 采样相同。这是将单步证明推广到多步的关键。

**关于一次 Forward 验证多个位置的细节（工程解释）：** Transformer 的因果 Attention Mask 确保候选位置只看到已经在它之前的 Token，故多个位置的 logits 可同时计算，却不发生未来 Token 信息泄漏。每个位置的 logits 必须与“对应下一个 Token”的索引准确对齐。发生拒绝时，Target 的已计算但无效的候选后缀 KV 不能作为新前缀复用，必须正确截断或重建所需状态。若连续 $\gamma$ 个候选全部接受，常见实现还能从 Target 验证结果额外采样一个 Token；这一扩展不是 Section 2 中显式给出的步骤，需要区分。

### 简化的接受长度与加速成本模型（独立分析）

若每个候选位置接受率近似常数 $\alpha$，各步条件可近似独立，设本轮草稿长度 $\gamma$，连续被接受的 Draft Token 个数 $K$，则：

\[
\Pr(K\ge k)\approx \alpha^k\quad (1\le k\le\gamma)
\]

\[
\boxed{\mathbb E K\approx\sum_{k=1}^{\gamma}\alpha^k
=\frac{\alpha(1-\alpha^\gamma)}{1-\alpha}}.
\]

若每轮对拒绝的第一个位置做残差采样；全部接受时再从 Target 取得一个额外 Token，那么本轮实际推进的 Token 数为 $N=K+1$，其期望近似：

\[
\boxed{\mathbb E N\approx\sum_{k=0}^\gamma \alpha^k
=\frac{1-\alpha^{\gamma+1}}{1-\alpha}}.
\]

例如 $\gamma=4,\alpha=0.8$ 时，$\mathbb E K\approx2.3616$，$\mathbb E N\approx3.3616$。这里的常数接受率近似与额外 Token 机制都是为了工程解释，不是原文此节给出的公式。**真正的速度还取决于草稿计算、验证计算、树展开、硬件瓶颈及 Cache 管理开销**。

## 八、可运行 PyTorch 验证示例（独立补充）

以下代码验证单 Token 的概率校正，并显示精确的输出分布。它不是 EAGLE 原仓库的官方实现。

~~~python
import torch

p = torch.tensor([0.6, 0.3, 0.1], dtype=torch.float64)
q = torch.tensor([0.4, 0.4, 0.2], dtype=torch.float64)

# 接受率仅在 q>0 的候选 Token 上需要定义
accept = torch.where(q > 0, torch.minimum(p / q.clamp_min(1e-12), torch.ones_like(p)), torch.zeros_like(p))
accepted_mass = q * accept
residual_mass = (p - q).clamp_min(0)
reject_prob = 1 - accepted_mass.sum()

print("a(x):", accept.tolist())
print("mean acceptance:", accepted_mass.sum().item())
if reject_prob > 0:
    residual_distribution = residual_mass / residual_mass.sum()
    corrected = accepted_mass + reject_prob * residual_distribution
else:
    corrected = accepted_mass
print("corrected:", corrected.tolist())
torch.testing.assert_close(corrected, p, rtol=0, atol=1e-12)
~~~

## 九、边界与实现注意事项（独立补充）

- **Greedy / temperature=0：** 有效目标分布是确定性 one-hot；若 Draft 候选与目标 argmax 不一致会被拒绝并修正为目标 Token。在正确实现的验证规则下，生成结果与 Target 贪心解码一致。
- **非贪心：** 精确性要求 $p$ 与 $q$ 为实际采样过程相应的有效分布（例如同样落实目标侧约定的 Temperature 与采样截断），不能拿原始 logits、归一化概率和不同的过滤结果混用。
- **$q(x)=0$：** 不存在该候选，直接避开 $p/q$ 的零分母；Target 独有的 Token 可通过残差修正出现。
- **$p=q$：** 拒绝概率为 0，残差归一化的分母为 0，但拒绝分支不会执行。
- **数值误差：** 浮点实现会有小误差；论文所指的“保持分布”是精确数学算法意义上的保证，并非任意错误的 CUDA/Kernel 实现也自动无损。
- **错误的 Draft 不影响分布、但影响速度：** 提议分布 $q$ 越偏离 $p$，平均接受率 $\alpha=1-\mathrm{TV}(p,q)$ 越低，后续草稿丢弃越多。
- **EAGLE 的差异在 Draft：** 本节介绍的是通用推测采样的单链基础；EAGLE 在 Section 3 和 Appendix A.2 把草稿扩展为 Tree，使用树注意力与多轮推测采样。不要将普通链式算法直接当作其完整 Tree 验证实现。

## 十、与先前 DFlash 论文的联系

[EAGLE](README.md) 的草稿模型通过 Feature + Shifted Token 提高提议质量；[DFlash](../2602.06036-dflash/README.md) 通过 Target Feature 条件化和 Block Diffusion 并行降低 Draft 延迟。两者在合法 Target 验证机制下都可以利用精确推测采样的分布校正思想。

- **模型质量**决定多少候选被接受（需要对比 $p,q$ 的有效分布、平均接受长度）。
- **Draft 计算成本**决定一次提出候选要花多少时间。
- **Target 验证成本**决定以一次较重 Forward 批量评估候选是否划算。
- 测得的 tokens/s 取决于三者共同作用，单独提高接受率不一定提高吞吐量。

## 阅读断点

- 已完成：Abstract、1. Introduction、2. Preliminaries。
- 下一阅读单元：**3. EAGLE**，依次为 **3.1 Drafting phase、3.2 Training of the draft models、3.3 Verification phase**（PDF 第 3–5 页；原图 Figure 5、Figure 6；附录 A.1/A.2 后续独立精读）。
- GitHub 保存状态须以成功提交并更新的 progress.json 为准。
