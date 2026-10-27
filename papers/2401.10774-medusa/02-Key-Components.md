# Section 2.1 Key Components — 完整翻译与 Figure 2 讲解

论文：Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads，arXiv:2401.10774v3（2024-06-14）。
原文位置：PDF 第 3–4 页。[第 3 页](https://arxiv.org/pdf/2401.10774v3#page=3) · [第 4 页与 Figure 2](https://arxiv.org/pdf/2401.10774v3#page=4)。
下面**「原文翻译」和「独立分析」严格分开**。保留原公式与论文印刷的权重维度，不暗自修正。

## 一、Section 2 Methodology 的两个开头自然段

**第 1 段（译文）**：Medusa 遵循与推测解码相同的框架，其中每个解码步骤主要由三个子步骤构成：（1）生成候选；（2）处理候选；（3）接受候选。对于 Medusa，（1）由 Medusa Heads 实现，（2）由 Tree Attention 实现。由于 Medusa Heads 位于原始模型之上，（2）中计算的 Logits 还可以用作下一轮（1）的输入。最后一步（3）既可以通过拒绝采样（Leviathan et al., 2022; Chen et al., 2023）实现，也可以通过 Typical Acceptance（Section 2.3.1）实现。总体流程参见 Figure 1。

**第 2 段（译文）**：本节首先介绍 Medusa 的核心组成，包括 Medusa Heads 和 Tree Attention；接着提出两级微调方法满足不同应用需要；最后介绍两项扩展，自蒸馏与典型接受机制，分别处理缺乏训练数据和解码效率改进的问题。

## 二、2.1.1 Medusa Heads — 全文译文

**自然段 1。** 在推测解码中，后续 Token 由辅助草稿模型预测。这个草稿模型必须足够小，又必须具备足够强的能力来生成原始模型愿意接受的续写。实现这两个要求并不容易，因此已有方法（Spector & Re, 2023; Miao et al., 2023）常常单独预训练一个小模型。这种预训练需要大量额外计算资源。例如 Miao et al.（2023）报告使用了 275 个 NVIDIA A100 GPU 小时。另外，单独的预训练可能使草稿模型和原始模型出现分布偏移，从而生成原模型不倾向于接受的续写。Chen et al.（2023）也强调了在分布式环境中服务多个模型的复杂性。

**自然段 2。** 为了简化并推广 LLM 推理加速，我们借鉴 Stern et al.（2018）的方法，后者在机器翻译、图像超分辨率等任务中使用并行解码。Medusa Heads 是附加于原始模型最后一层隐藏状态的额外解码头。具体而言，假设位置 $t$ 的最后隐藏状态为 $h_t$，我们在其上增加 $K$ 个解码头。第 $k$ 个 Head 预测位置 $t+k+1$ 的 Token；原始语言模型头预测位置 $t+1$ 的 Token。第 $k$ 个 Head 预测的词表分布记为 $p_t^{(k)}$，原始模型的预测记为 $p_t^{(0)}$。参考 Stern et al.（2018），每个 Head 使用带残差连接的单层前馈网络。实验发现，这个简单设计就可以达到令人满意的表现。第 $k$ 个 Head 定义为：

$$
p_t^{(k)}=\operatorname{softmax}\left(W_2^{(k)}\cdot\left(\operatorname{SiLU}(W_1^{(k)}\cdot h_t)+h_t\right)\right),
$$
$$
W_2^{(k)}\in\mathbb R^{d\times V},\quad W_1^{(k)}\in\mathbb R^{d\times d}.
$$

其中 $d$ 是 LLM 最后隐藏层的输出维度，$V$ 是词表大小。我们用与原始 LM Head 相同的权重初始化 $W_2^{(k)}$，将 $W_1^{(k)}$ 初始化为零，从而让 Medusa Heads 的初始预测与原始模型保持一致。SiLU 激活函数（Elfwing et al., 2017）遵循 Llama 模型（Touvron et al., 2023）的设计。

**自然段 3。** 与独立 Draft Model 不同，Medusa Heads 与原始主干模型一起训练；训练时主干既可以冻结（Medusa-1），也可以与 Heads 一同更新（Medusa-2）。该方法可以利用强大基础模型已经学到的表示，并允许在单张 GPU 上微调大型模型。同时，它有助于使 Medusa Heads 的分布与原始模型一致，减轻分布偏移问题。此外，由于额外 Head 只是类似原始 LM Head 的浅层结构，Medusa 不会对模型服务系统引入额外的设计复杂性，也适用于分布式部署。训练方案将在 Section 2.2 讨论。

## 三、2.1.2 Tree Attention — 全文译文

**自然段 1（跨 PDF 第 3–4 页）。** Medusa Heads 提供了未来 $K+1$ 个 Token 的概率预测，使我们能够构造长度为 $K+1$ 的候选续写。虽然以往的推测解码方法（Leviathan et al., 2022; Chen et al., 2023）建议仅采样一条候选续写，但是在解码过程中采用多条候选，可以提高一次解码步骤的预期接受长度。然而，更多候选也会提高计算需求。为了取得平衡，我们采用树结构注意力机制来同时处理多个候选。它不同于传统 Causal Attention：只有同一续写路径上的 Token 才能被视为历史。借鉴图神经网络领域将图结构融入 Attention 的思想（Ying et al., 2021），我们将树结构纳入注意力掩码，如 Figure 2 所示。类似思想也独立出现在 Miao et al.（2023）、Spector & Re（2023）中；那些工作采用自底向上的方式合并草稿模型生成的候选树，而我们基于 Medusa Heads 的候选结构，以自顶向下的方式构造树。第 $k$ 个 Head 的 Top-$s_k$ 预测用作构造候选的基础，其中 $s_k$ 是给定超参数。

**自然段 2。** 将各 Head 的 Top-$s_k$ 预测进行笛卡尔积（Cartesian Product）即可得到候选。例如 Figure 2 中 $s_1=2$、$s_2=3$，第一 Head 的每种预测后面都可以接上第二 Head 的任意一种预测。因此形成一棵树：第 $k$ 层每个节点具有 $s_k$ 个分支，虚拟根节点定义为第 0 层；实际情况下，第 0 层相当于原始 LM Head 的预测，可以独立采样。在这棵树上，只有 Token 的前驱节点会被视作历史上下文，掩码保证当前节点仅能关注前驱。通过这种掩码及正确调整的位置编码索引，我们无需增大 Batch Size 就能同时处理大量候选。新增树 Token 总数为：

$$
N=\sum_{k=1}^{K}\prod_{i=1}^{k}s_i.
$$

**自然段 3。** 本节展示的是通过笛卡尔积构造树的最简单、最规则的方法。但是也可以采用更复杂的结构，并利用各 Head 不同 Top 预测准确率不平衡的特点。我们将在 Section 2.3.3 中讨论这一点。

## 四、Figure 2 原图与完整图注

原始位置：[PDF 第 4 页左上角](https://arxiv.org/pdf/2401.10774v3#page=4)。已在本次对话从用户原始 PDF 裁取并展示高清原图；GitHub 目前没有该图 PNG 二进制，不能使用不存在的 figures/ 相对路径。

**图注完整译文**：我们演示了如何利用 Tree Attention 同时处理多个候选。示例中，第一个 Medusa Head 的 Top-2 预测与第二个 Medusa Head 的 Top-3 预测，共同形成 $2\times3=6$ 条候选序列。每条候选序列都对应树结构的一条独立分支。为保证每个 Token 只访问其前驱节点，我们设计了 Attention Mask，只允许注意力由当前节点流向祖先节点；位置编码索引也按照这种树结构进行调整。

图内标签翻译：Root＝根节点；Head 1/Head 2＝第一/第二个解码头；Query＝查询；Key＝键；Tree Mask＝树结构注意力掩码；打勾＝该 Query 可关注此 Key。

## 五、独立技术分析与详细推导（非原文）

### 5.1 多头预测的时间索引与条件缺失

设 Target 在位置 $t$ 得到 $h_t\in\mathbb R^d$。原始 LM Head 预测位置 $t+1$，第 $k$ 个 Medusa Head 预测位置 $t+k+1$。所有 Medusa Heads 的输入都是相同的 $h_t$，所以多头输出缺乏真实未来中间 Token 的信息；目标模型在验证候选分支时，则使用已给定候选前缀重新计算正确的自回归条件概率。

### 5.2 论文原式的矩阵形状问题

如果采用列向量约定 $h_t\in\mathbb R^{d\times1}$，则 $W_1h_t\in\mathbb R^{d\times1}$，残差结果同样是 $d\times1$。左乘输出矩阵要得到 $V\times1$ Logits，实际需要 $W_2\in\mathbb R^{V\times d}$；但论文写了 $W_2\in\mathbb R^{d\times V}$。原文在「左乘记号」和「输出权重形状」上不一致，应保留原式并明确标注。

两个数学上自洽的等价写法如下：

**列向量形式**：
$$u=\operatorname{SiLU}(A_1h_t)+h_t,\ A_1\in\mathbb R^{d\times d},\quad z=A_2u,\ A_2\in\mathbb R^{V\times d}.$$

**行向量形式（符合原文的 $d\times V$ 输出矩阵形状）**：
$$u=\operatorname{SiLU}(h_tW_1)+h_t,\quad z=uW_2,\quad W_2\in\mathbb R^{d\times V}.$$

输出 $p=\operatorname{softmax}(z)$。当 $W_1=0$，有 $\mathrm{SiLU}(0)=0$，即 $u=h_t$；若将 $W_2$ 初始化为原始 LM Head 参数，则初始化时 $p_t^{(k)}=p_t^{(0)}$。这只是初始化行为；它们分别训练不同未来位置，并不意味着参数必须永久与原 Head 绑定。

不计 bias，一个额外 Head 有 $d^2+dV$ 个矩阵参数。例如 $d=4096,V=32000$，每个 Head 约 147.85M 权重参数，BF16 约 295.7MB，五头约 1.48GB 仅矩阵权重。这是独立估算而非原论文实测；实际量化、权重共享和实现方式可能改变占用。

### 5.3 路径与树节点计数

若第 $k$ 个 Head 候选集合为 $S_k$，$|S_k|=s_k$，则第 $k$ 层节点对应组合：
$$S_1\times\cdots\times S_k,\qquad n_k=\prod_{i=1}^k s_i.$$
整条候选路径数（叶节点数量）为：
$$L_{\mathrm{leaves}}=\prod_{i=1}^K s_i.$$
实际树 Token 总数（不包括虚拟根和可独立采样的原始 LM Token）为：
$$N=\sum_{k=1}^K n_k.$$
若统一分叉数 $s>1$：$N=s+s^2+\dots+s^K=(s^{K+1}-s)/(s-1)$。

Figure 2 的 $K=2, s_1=2,s_2=3$，叶子有 $2\cdot3=6$ 条完整候选，树上只有 $2+6=8$ 个实际 Token 节点，虚拟根如果单算是第 9 个。若六条长度为 2 的路径分别完整处理，则共有 $6\cdot2=12$ 个包含重复前缀的 Token 位置。树合并节省了前缀重复。

举例 $K=5,s_i=3$ 时：叶子 $3^5=243$，树节点 $3+9+27+81+243=363$，将 243 条长度 5 的路径全部独立展开则是 1215 个路径 Token 位置。此处只是数量比较，不代表 Wall-time 直接按该比值缩短。

### 5.4 Figure 2 的 8×8 Tree Attention Mask

按图中展示顺序给节点编号：0=It (Head1), 1=I (Head1), 2=is (It 子节点), 3=' (It 子节点), 4=the (It 子节点), 5=is (I 子节点), 6=' (I 子节点), 7=the (I 子节点)。

如果 $\operatorname{Anc}(i)$ 为节点 $i$ 的祖先集合，则，不计前缀 Prompt KV：
$$M_{ij}=\begin{cases}0,&j=i\text{ 或 }j\in\operatorname{Anc}(i),\\-\infty,&\text{其余节点}.\end{cases}$$
对应的允许关注关系（行=Query，列=Key，1 表示允许）为：
$$
A=\begin{pmatrix}
1&0&0&0&0&0&0&0\\
0&1&0&0&0&0&0&0\\
1&0&1&0&0&0&0&0\\
1&0&0&1&0&0&0&0\\
1&0&0&0&1&0&0&0\\
0&1&0&0&0&1&0&0\\
0&1&0&0&0&0&1&0\\
0&1&0&0&0&0&0&1
\end{pmatrix}.
$$
第一层 2 个节点各看自己 1 个位置，第二层 6 个节点各看父节点 + 自己 2 个位置，故总共 14 个打勾，而不是完整的 64 个位置。原始上下文前缀仍可被所有查询访问。词面相同的 is 可以出现在不同分支，所处条件前缀不同，不能仅凭 token ID 相同就共享其整个 KV 状态。

### 5.5 Position IDs：深度相同，位置相同

展平索引与真实序列位置必须分开。把虚拟根（来自 LM Head 的 Token）视为第 $t+1$ 个位置，则 Head 1 的两个节点都处于位置 $t+2$，Head 2 的六个节点都处于位置 $t+3$。故 PDF Figure 2 对应的物理位置索引为：
$$[t+2,t+2,t+3,t+3,t+3,t+3,t+3,t+3].$$
从根开始的相对深度可记为 $[1,1,2,2,2,2,2,2]$，实现时需要对齐真实 Prefix 的位置偏移及所用 RoPE/Position IDs 约定。如果简单赋值 $[0,1,\dots,7]$，则分支上的相对位置编码不再与单条自回归序列一致；Attention Score 的位置关系将错误。

### 5.6 复杂度与注意力稀疏性

图中不含 Prompt 的逻辑可访问树内 Query-Key 对总数是：
$$E_{\mathrm{tree}}=\sum_{k=1}^K k\prod_{i=1}^k s_i.$$
Figure 2 得到 $2\cdot1+6\cdot2=14$。若共享 Prefix 长度 $L$，还存在约 $NL$ 个各树 Query 对历史 Prefix 的可访问关系。**注意逻辑可见边数量不等于真正 GPU 计算或内存开销**：如果采用稠密 $N\times N$ Mask，内存/计算仍可能包含 $O(N^2)$ 部分；Target 中线性层/MLP 也需要处理 $N$ 个 Token。随着树长大，矩阵乘法可能使验证逐渐由带宽限制转为算力限制，实际速度存在最佳树大小。论文 Section 2.3.3 和 Appendix G 后续分析这一问题。

### 5.7 同时验证与连续前缀接受

Tree Attention 允许一次 Target Forward 对树节点各自所属的条件前缀并行计算 Logits；接受过程仍需要保证连续的合法前缀。如果中间的 Token 不被接受，不能直接拼接下一层一个高概率节点。Tree Mask 仅保证分支条件上下文隔离，本身不能保证最终采样分布精确等于 Target。Medusa 的 Typical Acceptance 与精确的 Rejection Sampling 必须分别分析。

### 5.8 可运行的教学版 Head

以下 PyTorch 代码是独立示范，非官方源代码；Linear 的物理 weight 是 [out,in]，所以维度自洽：

~~~python
import torch
from torch import nn

class MedusaHead(nn.Module):
    def __init__(self, d, vocab, lm_head):
        super().__init__()
        self.fc = nn.Linear(d, d, bias=False)
        self.out = nn.Linear(d, vocab, bias=False)
        nn.init.zeros_(self.fc.weight)
        with torch.no_grad():
            self.out.weight.copy_(lm_head.weight)

    def forward(self, h):
        u = torch.nn.functional.silu(self.fc(h)) + h
        return self.out(u)  # logits: [batch, vocab]
~~~

## 六、EAGLE / DFlash 的交叉关联（独立分析）

| 项目 | Medusa | EAGLE（2401.15077） | DFlash（2602.06036） |
| --- | --- | --- | --- |
| Draft 生成 | 多个共享 $h_t$ 的 Token 预测头 | 自回归预测 Feature + Shifted Token | 基于 Target 多层特征的并行 Block Diffusion |
| Draft 内部串行 | Head 并行 | 多轮 Draft Forward | 一次块并行 Forward |
| 解决的瓶颈 | 免独立小 Draft Model | 特征/采样分支不确定性 | 减少 Draft 串行深度 |
| Target 验证 | 多分支 Tree Attention | Tree Attention + 精确采样 | Target 并行校验 |
| 分布 | Rejection Sampling 可精确；Typical Acceptance 为近似 | 正确精确采样下保持 | 正确精确采样下保持 |

Medusa 对未来更远位置没有真实中间 Token 的条件，草稿 Head 会受不确定性影响；EAGLE 通过 Shifted Token 帮助 Feature 预测，DFlash 则用上下文 Feature 与 KV Injection 条件化并行扩散。不能跨不同 GPU/任务/解码温度直接比较论文最大速度倍数。

## 已完成与下一单元

本次完成 Section 2 章首两段、2.1.1 完整正文和多头公式、2.1.2 完整正文、Figure 2 完整图注及独立推导；下一单元 **2.2 Training Strategies**，覆盖 2.2.1 Medusa-1，2.2.2 Medusa-2，2.2.3 Head 数量选择。
