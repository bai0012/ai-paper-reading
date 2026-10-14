# Section 3 EAGLE：完整译文、Figure 5–6、模型结构、损失和验证

> EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty · arXiv:2401.15077v3  
> 原文 PDF 第 3–5 页：[原文 PDF](https://arxiv.org/pdf/2401.15077v3)；2026-10-14 阅读。  
> Section 3、3.1、3.2、3.3 依序完整翻译；后面的技术分析不是论文原文。  
> 本次对话已从用户原始 PDF 提取 Figure 5/6 PNG 并展示。未实际提交原图二进制到仓库时，使用原 PDF 页码链接，不创建虚假的 figures/ 链接。

## 一、Section 3 完整中文翻译

### 3. EAGLE

与其他基于推测采样的方法一致，EAGLE 同时包含草稿生成阶段（Drafting Phase）和验证阶段（Verification Phase）。

### Figure 5：草稿生成方法对比

[Figure 5 原图：PDF 第 4 页](https://arxiv.org/pdf/2401.15077v3#page=4)

**图注完整翻译：** 图 5：不同方法在生成第 4 个和第 5 个 Token，即 $t_4$ 与 $t_5$ 时的草稿生成过程对比。其中 $t$（蓝色方块）表示 Token，$f$（橙色方块）表示特征，下标表示它们在序列中的位置。红色边框表示 Draft Model 的预测。为简化展示，图中 Lookahead 的 n-gram 参数 $n$ 设置为 2。

### 3.1 Drafting phase（草稿生成阶段）

**原文第 1 段。** EAGLE 与其他方法最主要的区别，在于草稿生成阶段。Figure 5 展示了不同方法的草稿生成过程。传统推测采样（Leviathan et al., 2023；Chen et al., 2023a）与 Lookahead（Fu et al., 2023）都是基于 Token 预测 Token。Medusa（Cai et al., 2023）则使用目标 LLM 的特征 $f_2$ 分别独立预测 $t_4$ 与 $t_5$。EAGLE 使用特征序列 $(f_1,f_2)$ 以及提前一个时间步的 Token 序列 $(t_2,t_3)$ 来预测 $f_3$。由 $p_4=\mathrm{LMHead}(f_3)$ 生成概率分布，并从中采样 $t_4$。随后，将 $f_3$ 和 $t_4$ 添加到输入序列中，以预测下一特征 $f_4$ 并采样下一 Token $t_5$。

**原文第 2 段。** 如 Figure 6 所示，EAGLE 的 Draft Model 由三个模块组成：Embedding Layer、LM Head 和 Autoregression Head。Embedding Layer 和 LM Head 直接复用目标 LLM 的参数，无需额外训练。Draft Model 输入特征序列，形状 $(\mathrm{bs},\mathrm{seq\_len},\mathrm{hidden\_dim})$，以及提前一个时间步的 Token 序列，形状 $(\mathrm{bs},\mathrm{seq\_len})$。模型将 Token 序列映射为 Token Embedding，形状 $(\mathrm{bs},\mathrm{seq\_len},\mathrm{hidden\_dim})$，再将其与特征序列拼接，形成 $(\mathrm{bs},\mathrm{seq\_len},2\times\mathrm{hidden\_dim})$ 的融合序列。

Autoregression Head 由一个全连接层（FC Layer）和一个 Decoder Layer 组成。FC 将融合序列降至 $(\mathrm{bs},\mathrm{seq\_len},\mathrm{hidden\_dim})$，再由 Decoder Layer 预测下一个特征。LM Head 根据特征计算 Token 分布，从中采样下一个 Token。最后，把预测特征和采样 Token 添加到输入中，以便继续自回归生成。

**原文第 3 段。** EAGLE 使用 Tree Attention 构造树状草稿，执行 $m$ 次前向传播，就能生成一棵深度为 $m$、包含超过 $m$ 个 Token 的草稿树。Figure 6 中，EAGLE 只用 3 次前向传播就生成了含 10 个 Token 的草稿树。实际采用的树结构详见 Appendix A.1。

### Figure 6：完整流程

[Figure 6 原图：PDF 第 4 页](https://arxiv.org/pdf/2401.15077v3#page=4)

**图注完整翻译：** 图 6：EAGLE 的处理流程。上半部分说明计算过程，下半部分说明各步骤的生成结果。在上半部分中，绿色方块表示 Token Embedding，橙色方块表示特征（Feature），红色框标示 Draft Model 的预测结果，带有雪花符号的蓝色模块代表复用 Target LLM 参数，该参数不参与训练。

### 3.2 Training of the draft models（草稿模型的训练）

**原文第 1 段。** 预测下一特征属于回归任务，因此我们采用 Smooth L1 Loss（参见 Figure 5 中的 EAGLE）：

\[
\boxed{\mathcal L_{\rm reg}=\mathrm{SmoothL1}\left(f_{i+1},\mathrm{DraftModel}(T_{2:i+1},F_{1:i})\right)}.
\]

**原文第 2 段。** 预测特征只是 Draft Model 的中间目标，其最终目标是预测 Token 以生成 Token 序列。因此，还采用分类损失以直接优化这一最终目标：

\[
p_{i+2}=\mathrm{Softmax}(\mathrm{LMHead}(f_{i+1})),
\]

\[
\hat p_{i+2}=\mathrm{Softmax}(\mathrm{LMHead}(\hat f_{i+1})),
\]

\[
\boxed{\mathcal L_{\rm cls}=\mathrm{CrossEntropy}(p_{i+2},\hat p_{i+2})}.
\]

通过结合回归损失与分类损失，我们用组合损失训练 Autoregression Head：

\[
\boxed{\mathcal L=\mathcal L_{\rm reg}+w_{\rm cls}\mathcal L_{\rm cls}}.
\]

通常分类损失的数值比回归损失大一个数量级，因此设 $w_{\rm cls}=0.1$。

**原文第 3 段。** 理想情况下，EAGLE Autoregression Head 应使用目标 LLM 自回归生成的文本训练，但这一过程成本较高。幸运的是，EAGLE 对训练数据敏感性较低（Section 4.3.3 的消融实验）。因此我们改用固定数据集，而不是目标 LLM 生成的文本，从而显著减少开销。草稿生成时，EAGLE 自回归处理特征；不准确的预测特征可能导致误差累积。为了减轻这个问题，训练阶段向目标模型的特征添加服从均匀分布 $\mathcal U(-0.1,0.1)$ 的随机噪声进行数据增强（Jain et al., 2023）。

### 3.3 Verification phase（验证阶段）

通过 Tree Attention，目标 LLM 在一次前向传播中计算树状草稿所有 Token 的概率。在草稿树的每个节点，我们递归应用推测采样算法以进行采样或调整概率分布（细节见 Appendix A.2），与 SpecInfer（Miao et al., 2023）的做法一致，从而保证最终输出文本的概率分布与目标 LLM 一致。同时，记录被接受的 Token 及对应特征，供下一轮草稿生成阶段使用。

**Section 3 原文翻译到此结束。下一节是 Section 4 Experiments。**

---

## 二、独立技术分析（不是论文翻译）

### 1. Figure 5：为什么 EAGLE 与 Medusa 不同

- 普通 Speculative Sampling：一个独立较小的 LLM 自回归生成多个后继 Token。
- Lookahead：基于 Token、n-gram 和 Jacobi 迭代构建候选。
- Medusa：从同一个 Target 隐藏特征 $f_2$ 独立使用多个 Head 预测未来 Token $t_4$、$t_5$；缺少每次真实采样结果决定的逐步特征转移条件。
- EAGLE：$(f_1,f_2)$ 与 $(t_2,t_3)$ 联合预测 $f_3$，再从 $p_4=\mathrm{LMHead}(f_3)$ 采样 $t_4$；之后联合新增的预测特征 $f_3$ 和 $t_4$，递推预测 $f_4$ 和 $t_5$。

Shifted Token 的精确索引形式：

\[
\hat f_{i+1}=g_\phi(F_{1:i},T_{2:i+1}),\qquad
\hat p_{i+2}=\mathrm{Softmax}(\mathrm{LMHead}(\hat f_{i+1})).
\]

这里 $t_{i+1}$ 是当前 $f_i$ 所给出分布下**已经采样完成**的 Token，而非泄漏未生成的未来标签。它消除 Feature-only 的多分支后继预测歧义。

### 2. Target/Draft 共享参数与张量维度

令 $B$ 为 Batch，$L$ 为历史长度，$d$ 为 Hidden Size，$V$ 为词表大小：

\[
F_{1:L}\in\mathbb R^{B\times L\times d},\quad
T_{2:L+1}\in\mathbb Z^{B\times L}.
\]

共享 Embedding：

\[
E=\mathrm{Embedding}_{\rm target}(T_{2:L+1})\in\mathbb R^{B\times L\times d}.
\]

拼接并投影：

\[
H=[F;E]\in\mathbb R^{B\times L\times 2d},\qquad
Z=\mathrm{FC}(H)\in\mathbb R^{B\times L\times d}.
\]

可训练 Decoder 预测未来特征：

\[
\hat F=\mathrm{Decoder}_\phi(Z),\qquad
\hat P=\mathrm{Softmax}(\mathrm{LMHead}_{\rm target}(\hat F)).
\]

可训练的是 FC 和 Decoder Layer，Target Embedding、Target LM Head 冻结复用。仅 FC 的权重规模为 $2d^2$（按列向量约定 $W_{\rm fc}\in\mathbb R^{d\times2d}$）。举例 $d=4096$ 时为 $33,554,432$ 参数，不含 Bias。由此可见额外开销远小于从零训练完整独立 LLM，但还有 Decoder 和推理 KV 等成本，不能将 FC 数量视为 EAGLE 总参数量。

### 3. Figure 6：3 次草稿 Forward ≠ 一次生成全部 Token

图示 Query 为 How can，Target 先用一次 Forward 给出真实特征并采样出 I。随后 Draft 模型：

| Draft Forward | 新增节点 | 节点数 |
| --- | --- | ---: |
| 1 | make，help | 2 |
| 2 | a、our（make 分支）；with、you（help 分支） | 4 |
| 3 | the、your（with 分支）；to、feel（you 分支） | 4 |
| **总计** | 图中示例草稿树 | **10** |

初始 I 是 Target 采样结果，**不属于这 10 个 Draft Token**。不同 Forward 之间保留跨深度依赖，但同一轮一次计算可以扩展多个并列节点；这解释了为什么 $m$ 次 Forward 能构造超过 $m$ 个候选，却不等于能同时接受所有不同分支的节点。实际最终输出只能沿合法的一条路径延伸。

### 4. Tree Attention：祖先可见、兄弟不可见

对树中节点 $u$ 的位置，需要允许其读取公共已确认前缀、当前分支的祖先节点，并避免读取其他候选分支。概念性注意力 Mask：

\[
M_{uv}=\begin{cases}
0,& v\in\mathrm{Ancestors}(u)\cup\{u\}\ \text{或在因果可见的公共前缀中},\\
-\infty,&\text{否则}.
\end{cases}
\]

图中分支 I→help→with→the 不应读到另一分支 I→make→our。否则 Target logits 不再等价于该分支在正常自回归上下文下的条件分布。实际还需配套正确的 RoPE/position IDs、KV 管理和 logits 位置对齐。

### 5. Smooth L1 Regression Loss

回归目标是预测特征 $\hat f_{i+1}$ 与 Target 真特征 $f_{i+1}$ 一致。解释性地，以 PyTorch 默认 Smooth L1 的 beta=1 为例：

\[
\ell_\beta(e)=\begin{cases}
e^2/(2\beta),&|e|<\beta,\\
|e|-\beta/2,&|e|\ge\beta.
\end{cases}
\]

小误差近似平方，大误差线性增长，因此相较纯 MSE，对异常偏差的梯度更稳健。该分段公式与 beta 默认值是工程补充，论文 Section 3.2 未写出具体 beta 参数。

### 6. Soft-label Classification Loss = Token 概率分布蒸馏

本节的 Cross Entropy 不是针对 One-Hot 真实 Token，而是将 Target 真实特征通过 LM Head 映射出的**完整词表分布** $p$ 作为软目标：

\[
\mathcal L_{\rm cls}=-\sum_{x\in\mathcal V}p(x)\log\hat p(x)=H(p)+D_{\rm KL}(p\Vert\hat p).
\]

由于 Target 被冻结，$H(p)$ 与 Draft 参数无关，故优化该交叉熵等价于优化正向 KL $D_{\rm KL}(p\Vert\hat p)$。这样训练同时控制隐藏空间误差和最终 Token 分布误差。

$\mathcal L=\mathcal L_{\rm reg}+0.1\mathcal L_{\rm cls}$ 中的权重 0.1 来自作者对损失量级的观察，不意味着两项的梯度幅值完全相等，也不是理论最优权重。

### 7. 为什么训练加入 Uniform Feature Noise

推理时 Draft 对历史位置可能使用自身预测的 $\hat f_i$ 而非 Target 精确的 $f_i$，多步递推可能导致误差累积。作者训练时对真实特征施加：

\[
\tilde f_i=f_i+\epsilon_i,\qquad\epsilon_i\sim\mathcal U(-0.1,0.1).
\]

使 Draft 在一定输入扰动下仍能预测下一特征。这里是特征加性噪声，**不是 Dropout**。文章未在 Section 3.2 证明任意扰动均有效，也没有给出复杂噪声调度；无需推测作者额外用了此类技巧。

### 8. Tree Verification 如何与精确推测采样结合

1. Draft 产生候选 Token 的树及其提议概率；不必只生成单条链。
2. Target 通过 Tree Attention 的**一次 Causal Forward**，为树上所有对应位置计算真实目标条件分布。并行计算 logits，不等于并行做所有接受决策。
3. 验证器在树节点递归应用正确的多轮 Speculative Sampling，接受或拒绝候选并修正残差概率，严格保留 Target 的生成分布；不能简单选得分最高的草稿路径来冒充无损采样。
4. 选中一条有效路径后，只保存被接受 Token 与该路径对应的 Target 特征，以供下一轮 Draft。被拒绝分支的无效 KV/Feature 不得复用到新路径。

**源码边界：** Section 3.3 仅描述树验证的基本原则，没有完整伪代码。树结构在 Appendix A.1 Figure 9，Multi-round Speculative Sampling 在 Appendix A.2 Algorithm 1，后续阅读时再逐条翻译与推导。不可把普通单链 $a(x)=\min(1,p/q)$ 直接当作已完成的树状实现。

### 9. PyTorch 结构示意（不是官方代码）

~~~python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DraftHead(nn.Module):
    def __init__(self, hidden, target_embedding, target_lm_head, heads=32):
        super().__init__()
        self.embedding = target_embedding  # 冻结共享组件
        self.lm_head = target_lm_head      # 冻结共享组件
        self.fc = nn.Linear(2 * hidden, hidden)
        self.decoder = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=heads, batch_first=True)

    def forward(self, features, shifted_tokens, causal_or_tree_mask):
        with torch.no_grad():
            token_embeds = self.embedding(shifted_tokens)
        merged = torch.cat([features, token_embeds], dim=-1)
        pred_features = self.decoder(
            self.fc(merged), src_mask=causal_or_tree_mask)
        draft_logits = self.lm_head(pred_features)
        return pred_features, draft_logits

def training_loss(pred_f, target_f, draft_logits, target_logits):
    reg = F.smooth_l1_loss(pred_f, target_f)
    target_probs = F.softmax(target_logits.detach().float(), dim=-1)
    cls = -(target_probs * F.log_softmax(
        draft_logits.float(), dim=-1)).sum(dim=-1).mean()
    return reg + 0.1 * cls
~~~

此代码只演示输入拼接、FC+Decoder、共享 LM Head、双损失的接口，并非可直接用于生产的 EAGLE：还需实现 Target 参数冻结、实际适配的 Decoder/RoPE、KV Cache、Token-Feature Shift、Tree Mask、精确 Speculative Sampling、Padding Mask 与在线 Draft 循环。

### 10. 与 DFlash 的技术关联

- [EAGLE 本篇](README.md)：利用 Target Hidden Features 和 Shifted Token，训练轻量**自回归**特征预测器；Tree Draft 的每一深度仍存在前一 Token 的依赖。
- [已精读 DFlash](../2602.06036-dflash/README.md)：以 Target 多层 Feature 融合及逐层 KV Injection 条件化轻量 Block Diffusion Draft，在 Block 内**并行**提议多个 Token，降低 Draft 顺序深度。
- 共同点：Target 表征提高 Draft 质量、由合法 Target 验证保证原生成分布。
- 比较边界：EAGLE 与后续 EAGLE-3（arXiv 2503.01840）不是同一算法；不同模型/硬件/并发与 Serving 后端下的加速比不可直接横比。

## 阅读断点

- 已完成：Abstract、1. Introduction、2. Preliminaries、3. EAGLE（3.1–3.3）。
- 下一单元：4. Experiments（PDF 第 5–8 页；4.1 Effectiveness、4.2 EAGLE+gpt-fast、4.3 Ablation、4.4 Batch size and throughput；Tables 1–7，Figures 7–8）。
- 原图：Figure 5、Figure 6 已从 PDF 裁图在对话显示，当前章节的 GitHub 版本使用原始 PDF 的可核验链接；若未实际上传二进制，progress.json 中 github_binary_saved 应维持 false。
