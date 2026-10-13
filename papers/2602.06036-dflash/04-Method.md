# 4. Method — 方法（完整中文翻译、原图与技术讲解）

- **论文**：[DFlash: Block Diffusion for Flash Speculative Decoding](https://arxiv.org/html/2602.06036v2)
- **版本**：arXiv:2602.06036v2（2026-05-28）
- **原文位置**：PDF 第 4–6 页，Section 4.1 / 4.2
- **Figure 2**：原 PDF 第 4 页；**Figure 4**：原 PDF 第 5 页
- **说明**：以下将论文原文的**完整中文翻译**与**独立技术讲解**分开。附录 A.3、A.1 为辅助理解而交叉引用，并不表示附录已按章节完整精读。原图已在当前对话中从 PDF 直接裁取，但目前不创建并不存在的 GitHub PNG 图片链接；可点击原 PDF 页码查看原图。

## 4.1 Inference — 完整翻译

**第 1 段（章节导言）。** DFlash 的系统设计如 Figure 2 所示。在本节中，我们解释使 DFlash 能够利用一个非常小且高效的草稿模型，实现较长草稿接受长度的关键设计选择。

### Figure 2 — DFlash 推理架构

**原图来源：** [论文原始 PDF 第 4 页，Figure 2](https://arxiv.org/pdf/2602.06036v2#page=4)（对话中已提取原始 PNG：dflash_Figure_2_original.png；仓库未保存二进制图文件）。

**原图注完整翻译：** 图 2. DFlash 的推理设计。从目标模型提取的隐藏上下文特征经过融合，被注入草稿模型每一层的 Key-Value Cache，从而实现条件化推测生成。

**原图标识：** 蓝色为 Fused Target Context Feature（融合的目标上下文特征），黄色为 Target Decode Token（目标解码 Token），绿色为 Mask Token；图中还标注 Target Model、Target Embedding、KV Cache、Bidirectional Attention、MLP、Draft Layer 1/2、Target LM Head。

**第 2 段（Context features from the target model）。** An et al.（2025）等先前工作直接使用小型扩散模型作为推测解码的草稿模型，但这种方法的接受长度较短，获得的加速效果也十分有限。为了验证这一现象，我们训练了一个五层块扩散草稿模型，不使用来自目标模型的任何条件信息，并在多个数学基准测试中进行评估。如 Table 10 所示，由此获得的加速比较为有限，通常只有约 2–3 倍。这一局限源于缺少丰富的上下文指导：如果无法访问目标模型的内部表示，扩散草稿模型实际上必须从零开始预测未来 Token。

**第 3 段（隐藏表示的价值）。** 相比之下，大型自回归目标模型的隐藏表示所编码的信息，远比 Token 级别的 Logits 更丰富。这些特征能够捕捉长距离依赖关系、与具体任务相关的语义信息，并且——至关重要的是——隐式编码了与未来 Token 预测相关的信息，这一现象也被 Samragh et al.（2025）观察到。

**第 4 段（Prefill / 跨层融合）。** 在 DFlash 中，给定一个输入 Prompt，目标模型首先执行标准 Prefill，并生成第一个 Token。在此过程中，我们从一组固定的层中提取隐藏表示，这些层从浅到深均匀选取。随后拼接这些隐藏状态，并通过一个轻量投影层融合跨层信息，得到紧凑的目标上下文特征，用于给草稿模型提供条件信息。

**第 5 段（Conditioning via KV injection enables acceptance scaling）。** EAGLE-3 等现有方法同样利用目标模型隐藏特征，但它们会将这些特征与草稿模型的 Token Embedding 融合，只在草稿模型输入端提供这些特征。随着草稿模型深度增加，目标模型的信息会逐渐被稀释，因此增加更多 Draft 层时，接受长度的收益会递减。

**第 6 段（逐层 KV 注入）。** DFlash 采用了根本不同的策略。我们将融合后的目标上下文特征视为持久上下文信息，并直接将其注入草稿模型每一层的 Key 和 Value 投影。投影特征存储在草稿模型的 KV Cache 中，可在多轮草稿生成中复用。更多 KV Injection 细节见附录 A.3。该设计在整个草稿模型中提供了强大且一致的条件信息，使接受长度能够有效地随草稿模型层数增加而提高。更多分析见第 5.5.2 节。

**第 7 段（Parallel diffusion drafting）。** DFlash 速度提升的另一重要因素是低 Draft 延迟。自回归草稿模型需要多次顺序前向传播才能生成草稿 Token 或草稿树，限制了并行性，并导致 GPU 利用率低。相比之下，DFlash 通过块级扩散过程预测下一个 Token 块。块内全部 Mask 位置在一次前向传播中并行解码。相比自回归 Draft，这种块级并行生成显著减少延迟、提高硬件利用率，即使使用更深 Draft 模型也是如此。

**第 8 段（小结）。** 总体而言，DFlash 将扩散式并行草稿生成与来自目标模型的紧密条件化结合起来，在显著降低草稿生成延迟的同时实现高质量草稿生成。

## 4.2 Training — 完整翻译

**第 1 段（章节导言）。** DFlash 草稿模型的训练目标是使块级扩散预测与冻结的自回归目标模型输出对齐。作者没有直接采用标准块扩散训练（Arriola et al., 2025），而是引入若干关键修改，以提升训练效率、可扩展性，并更好地匹配推理时推测解码的行为。

**第 2 段（KV injection）。** 按照推理流水线，给定一条含 Prompt 和 Response 的序列，首先将完整、干净（未经 Mask）的序列送入目标模型，为所有 Token 提取并融合隐藏特征；随后如 Figure 4 所示，将这些隐藏特征作为 Key 和 Value 投影注入草稿模型。

### Figure 4 — 训练 Attention

**原图来源：** [论文原始 PDF 第 5 页，Figure 4](https://arxiv.org/pdf/2602.06036v2#page=5)（对话中已提取原始 PNG：dflash_Figure_4_original.png；仓库未保存二进制图文件）。

**原图注完整翻译：** 图 4. DFlash 的训练 Attention。目标模型提供上下文特征（蓝色），用于给草稿模型提供条件信息。输入由干净的 Prompt Token \(\{p\}\) 与干净的 Response Token \(\{r\}\) 组成。对于每一个 Mask 块，随机选择一部分干净 Response Token 作为 Anchor（黄色），而 Mask Token \(m\)（绿色）标识需要并行预测的位置。不可见 Token（白色）表示 Attention Mask 中被屏蔽的位置，该 Mask 确保因果一致性，防止训练时不同块的信息泄漏。

**第 3 段（Standard block diffusion training）。** 在标准块扩散训练中，Response 被均匀划分成块，每个块内随机位置被 Mask，然后训练模型对这些位置去噪。

**第 4 段（Random sampling of masked blocks）。** DFlash 专门针对推测解码设置修改块构建方式：从 Response 中随机采样 Anchor Token，以每个 Anchor 作为块首位置，将剩余位置全部 Mask，训练草稿模型并行预测后续的 \(\text{Block Size}-1\) 个 Token。这正好匹配推理时的行为：草稿模型总是以目标模型产生的一个干净 Token 为条件，即上一轮验证的 Bonus Token。随机化 Anchor 位置也让模型接触更多样化的目标上下文特征，提高数据利用效率和覆盖率。Table 13 显示这项策略明显改善接受长度和加速比。

**第 5 段（联合训练与稀疏 Attention）。** 训练时，所有块被拼接为单个序列，并使用 Figure 4 所示的稀疏 Attention Mask 联合处理。同一块内 Token 之间允许双向 Attention，且能访问对应的已注入目标上下文特征；不同块之间的 Attention 被禁止。这使得多个 Draft Block 可以借助 Flex Attention（Dong et al., 2024）在一次 Forward 和 Backward 中高效训练。

**第 6 段（Efficient long-context training）。** 对 EAGLE-3 这样的模型，长上下文训练因昂贵的 Training-time Test 而有挑战。DFlash 通过固定每条序列的 Mask Block 数量、并在每个 Epoch 为每条序列随机选择 Anchor 位置，降低长上下文训练成本。这既形成有效的数据增强，又能使训练成本保持有界。

**第 7 段（Loss weighting for faster convergence）。** 推测解码中并非所有 Token 都同等重要：如果草稿块中较早位置预测错误，之后所有 Token 都会失效。因此较早位置的预测对接受长度特别重要。为了反映这种不对称性，训练时对交叉熵损失采用位置加权，更重视块内较早位置。

**第 8 段（公式 4）。** 具体而言，对于块内位置 \(k\) 的 Token，采用指数衰减权重：

\[
\boxed{w_k=\exp\left(-\frac{k-1}{\gamma}\right)}\tag{4}
\]

其中 \(\gamma\) 控制衰减速度。这种加权优先考虑靠前位置，有助于更快收敛；相较均匀加权，产生更长的接受长度（Figure 5）。

**第 9 段（Shared embedding and LM head；跨 PDF 第 6 页）。** 为提高训练效率，草稿模型与目标模型共享 Token Embedding 与 LM Head，并在训练时将它们保持冻结。只有 Draft Transformer 层被更新。这减少了可训练参数，使 Draft 模型成为一个与目标模型表示空间紧密对齐的轻量扩散适配器。

**Section 4 完整原文翻译至此结束；后文是 Section 5 Experiments。**

---

## 第三部分：深度技术讲解（不是逐段原文）

### 1. Target Hidden Features 与跨层融合

目标 Transformer 第 \(l\) 层的隐藏状态可记为 \(H^{(l)}\in\mathbb{R}^{T\times D}\)，其中 \(T\) 是上下文长度，\(D\) 是隐藏维度。本论文使用来自固定目标层的隐藏状态，拼接并压缩至草稿网络所需维度。附录 A.3 给出的核心公式为：

\[
\boxed{H_t=\operatorname{RMSNorm}\Big(W_c[H^{(l_1)};\dots;H^{(l_5)}]\Big)}
\]

附录示例令目标和草稿隐藏维度均为 \(D\)，故拼接后为 \(T\times5D\)，共享投影 \(W_c\in\mathbb{R}^{D\times5D}\)，输出 \(H_t\in\mathbb{R}^{T\times D}\)。当 \(D=2048\) 时，投影矩阵含 \(5\times2048^2\) 个参数；BF16 占用约 42 MB。这是**投影矩阵权重**的估计，并不包括草稿层本身、目标模型或全部 KV Cache。

使用多层特征，是为了保留目标模型在不同网络深度上的上下文表达，而不局限于最后一层 Token Logits；不意味着隐藏状态能无误预测所有未来位置。

### 2. 附录 A.3 的逐层 KV Injection 公式

定义 \(H_d\) 为草稿 Token 在当前 Draft 层参与注意力计算的隐藏表示。对第 \(i\) 个 Draft 层：

\[
\boxed{
Q_i=W_i^QH_d,\quad
K_i=[W_i^KH_t; W_i^KH_d]_{\mathrm{seq}},\quad
V_i=[W_i^VH_t; W_i^VH_d]_{\mathrm{seq}}
}
\]

\([\cdot;\cdot]_{\mathrm{seq}}\) 表示**沿序列维**拼接，而非特征维。目标特征只作为额外 KV，**不形成 Query**，不需要像普通草稿 Token 那样经过 Draft 的输出投影更新或 FFN。当前块的 Token 通过 Query 读取目标信息，且块内允许双向 Attention；KV 注入发生在每一草稿层。投影后的目标 KV 可以缓存复用，以减少每轮重复计算。

在忽略 Batch、Attention Head 与 GQA 细节的简化形式下，若当前 Draft Block 含 \(B\) 个 Token，允许访问的目标上下文 KV 长度为 \(T\)，每头维度为 \(d_h\)，则

\[
Q\in\mathbb{R}^{B\times d_h},\quad
K,V\in\mathbb{R}^{(T+B)\times d_h}
\]

及

\[
QK^\top\in\mathbb{R}^{B\times(T+B)}.
\]

这些形状是从论文公式推导的工程直觉，不是对实际实现的所有 Tensor 维度/缓存布局的完整规定；RoPE、位置偏置、Attention Mask 与具体 K/V 分组方式仍需在代码中处理。

### 3. KV Injection 与 EAGLE-3 Input Fusion 的本质差别

**Input Fusion**：目标特征主要在草稿输入端融合，并通过后续层传播。随着层数加深，作者认为目标特征信息会逐渐被稀释。

**DFlash KV Injection**：每一层都能直接读取目标上下文的 KV；增强草稿网络深度时，仍能在每层持续条件化。因此 DFlash 希望让接受长度与 Draft 网络容量共同提升。

这里的“输入信息稀释”是论文提出的解释，**不是**关于任何深层 Transformer 必然丢失条件信息的普适数学定理。需要结合第 5.5.5 节 KV-vs-Input 消融验证改善量。

### 4. 单步 Block Diffusion 和上一节的开销公式

草稿块包含一个正确 Anchor 加若干 Mask：

\[
[r_i,m_1,\dots,m_{B-1}]
\]

一次 Forward 预测多个后续位置，降低对逐 Token 串行 Forward 的依赖。与完整 dLLM 多轮去噪不同，DFlash 将扩散用于猜测未来块，之后由冻结 Target Model 验证，因此不要求 Draft 自己达到最终文本生成质量。即便单次 Forward 处理多个 Token，其计算量仍随上下文长度、Block Size、网络深度与 Attention 形状变化；低延迟并非数学常数。

### 5. 随机 Anchor 与 Train/Inference Alignment

设干净 Response 为 \(r_1,r_2,r_3,\dots\)，块长度 \(B=4\)，训练可分别取：

| Anchor | Draft 输入 | Mask 位置的监督目标 |
| --- | --- | --- |
| \(r_1\) | \([r_1,m,m,m]\) | \([r_2,r_3,r_4]\) |
| \(r_2\) | \([r_2,m,m,m]\) | \([r_3,r_4,r_5]\) |
| \(r_3\) | \([r_3,m,m,m]\) | \([r_4,r_5,r_6]\) |

每个 Anchor 均对应特定的目标上下文特征。随机选择多个 Anchor 可形成更多样的训练样本，并让每个块匹配推理中“已确定的正确 Token + 等待预测的 Token”形式。作者在附录 A.5.2 的 Table 13 展示此策略比标准块划分/Mask 改善接受长度与速度。

**记号提醒**：第 4 节的 Block Size \(B\) 含首个 Anchor 和 \(B-1\) 个 Mask；第 3 节用 \(\gamma\) 表示推测 token 预算；实验所报告的 DFlash Block Size 不应随意与第 3 节的 \(\gamma\) 等同。

### 6. Figure 4 的 Attention Mask 及防泄漏原则

训练过程会在同一个计算序列中组织多个块，但是同一块内允许双向 Attention，不同块之间禁止互相注意。对三个 \(B=4\) 的块，简化的草稿块间可见矩阵为：

\[
M=\begin{bmatrix}
\mathbf{1}_{4\times4}&0&0\\
0&\mathbf{1}_{4\times4}&0\\
0&0&\mathbf{1}_{4\times4}
\end{bmatrix}.
\]

这**只表示草稿块内部/块之间的简化关系**，并非论文 Figure 4 的完整 Attention Mask。真实 Mask 还需要允许各块读取对应的 Target 上下文 KV，并保证每个 Anchor 只能访问其因果前缀。训练时 Target 处理完整 Clean Sequence 并不代表允许给较早 Anchor 提供包含未来 Response 内容的上下文，否则将造成 Label Leakage。

Flex Attention 支持高效处理这种非标准稀疏 Attention 规则，在一次 Forward/Backward 中处理多个 Mask Blocks，并通过固定块数让长上下文训练成本相对有界。

### 7. 损失函数和前部 Token 权重

推测解码只接受从块首开始的连续正确前缀，较早预测失败会使后续候选不能继续提交。DFlash 因此对每个预测位置 \(k\) 使用：

\[
w_k=\exp(-(k-1)/\gamma).
\]

**特别注意**：式 (4) 的 \(\gamma\) 是**损失衰减超参数**。附录 A.1 设置为：

| Block Size | 衰减 \(\gamma\) |
| ---: | ---: |
| 8 | 4 |
| 10 | 5 |
| 16 | 7 |

即 Block Size 16 对应 \(\gamma=7\)，不是 16。这是对同一字母的另一种使用方式，与第 3 节的推测 Token 数记号相区别。

当 \(\gamma=7\)，第 1、8、16 个位置的权重分别为 1、约 0.368、约 0.117。较后位置仍参与训练，只是权重更小。

**补充的加权 CE 示意，而非论文给定的精确归一化实现：**

\[
\mathcal{L}=
\frac{\sum_{k\in\mathcal M}w_k\,\operatorname{CE}(p_k,y_k)}
{\sum_{k\in\mathcal M}w_k}.
\]

\(\mathcal M\) 表示需要监督的 Mask 位置。原文仅明确式 (4) 的位置权重，具体 Loss 的 Batch/Block 聚合方式需从官方代码核对。

### 8. PyTorch 伪代码（独立工程示意，非官方 DFlash 实现）

\`\`\`python
import torch
import torch.nn.functional as F

# 假设每个 block 含 1 个 clean anchor 和 B-1 个 mask target
B = 16
n_blocks = 8
decay_gamma = 7.0

# 对每个块独立建立 full bidirectional attention
ids = torch.arange(n_blocks * B)
block_ids = torch.div(ids, B, rounding_mode="floor")
block_attn = block_ids[:, None] == block_ids[None, :]
# block_attn 是 [n_blocks*B, n_blocks*B] 的布尔允许矩阵；
# 真正 DFlash 还需要因果对齐的 target context KV 可见规则。

# Draft 预测 B-1 个被 Mask 位置
# logits:  [n_blocks, B-1, vocab_size]
# targets: [n_blocks, B-1]
def illustrative_weighted_loss(logits, targets):
    mask_count = logits.size(1)
    ce = F.cross_entropy(
        logits.transpose(1, 2),  # [n_blocks, vocab, B-1]
        targets,
        reduction="none",
    )  # [n_blocks, B-1]
    k = torch.arange(
        1, mask_count + 1, device=logits.device, dtype=ce.dtype
    )
    w = torch.exp(-(k - 1) / decay_gamma)  # [B-1]
    return ((ce * w).sum(-1) / w.sum()).mean()
\`\`\`

这里没有实现真实目标特征抽取、逐层 KV 注入、Block-Local RoPE、完整稀疏 Attention 调度或精确 Speculative Sampling。仅用于理解 Attention 隔离和式 (4) 权重。

### 9. Shared Embedding + LM Head

目标模型的 Token Embedding 与 LM Head 被草稿模型复用并冻结，这避免额外训练庞大词表矩阵，也促使 Draft Hidden States 与 Target 的词表/表示空间保持一致。作者原文指出仅更新 Draft Transformer 层；同时附录 A.3 定义一个额外的投影矩阵 \(W_c\)。不可仅据原文那句便认定 \(W_c\) 一定冻结，需要以具体代码的 optimizer param groups 为准。

### 10. 工程实现关注点

1. **特征对齐**：提取的是目标模型特定层、特定位置的隐藏状态；KV 注入必须配合相应因果前缀与位置关系。
2. **缓存复用**：Draft 各层目标特征 KV 应尽可能复用，避免每轮重复编码整个 Prefix；但新生成的 Target 特征需及时更新。
3. **块内双向注意力**：与一般 Causal KV Cache 逻辑不同，当前预测块允许双向访问，却不能读取后续未发生的训练块标签。
4. **采样正确性**：Draft 提出候选不等于直接提交，仍需要 Target 的正确接受/拒绝逻辑以保持输出分布。
5. **块数和深度选择**：网络越深并非越快；接受长度、Draft 延迟、验证成本三者的综合最优点在第 5 节消融实验分析。
6. **Train/Inference gap**：Anchor 和 Loss 衰减都围绕推理中的连续前缀验收机制设计，不能简单替换为标准通用扩散训练策略。

### 11. 与其他论文的关联

- [EAGLE-3](https://arxiv.org/abs/2503.01840)：使用 Target Hidden Features 做 Draft 的基础；DFlash 主要区别在“逐层 KV conditioning + 并行块预测”，而不是只有输入融合和自回归草稿。
- [Your LLM Knows the Future](https://arxiv.org/abs/2507.11851)：为多层 Target 表示可能含未来 Token 预测信息提供研究动机。
- [Block Diffusion](https://arxiv.org/abs/2503.09573)：提供块内并行去噪的建模背景；DFlash 将其限制在推测解码 Draft 环节。
- [Flex Attention](https://arxiv.org/abs/2412.05496)：支持非标准块稀疏 Attention Pattern 的高效实现，用于训练时多 Mask Block 并行处理。

## 小结、未来阅读位置

本节两个主要创新协同：**推理时**通过多层 Target Hidden Features 的逐层 KV Injection 提高候选质量，同时单次前向并行预测 Mask Block；**训练时**随机 Anchor 匹配推理分布，稀疏 Mask 防止块间标签泄漏，指数位置权重优先优化影响接受长度的前部 Token，并冻结共享 Embedding/LM Head。

已完成：**Section 4 Method — 4.1、4.2**。下一阅读单元：**Section 5 Experiments**（模型、训练数据、基准、SGLang/vLLM 吞吐、消融 Table 1–9）；附录 A.1/A.3 已作交叉引用，仍将按原始顺序在附录阅读时全面处理。
