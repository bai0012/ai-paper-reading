# Section 1 — Introduction：完整译文、Figure 1 与深度技术讲解

- 论文：**Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads**
- 版本：arXiv:2401.10774v3，用户提供的 27 页 PDF（2024-06-14）
- 本节原文：PDF 第 1 页右栏（Section 1 标题起）至第 2 页右栏（Section 2 标题前）；Figure 1 在第 2 页
- [原版 PDF，第 1 页](https://arxiv.org/pdf/2401.10774v3#page=1) | [原版 PDF，第 2 页 / Figure 1](https://arxiv.org/pdf/2401.10774v3#page=2)
- 注意：正文 6 个自然段，跨页和分栏处的连字符已合并；原文图注与译文分开；后续推导均是额外分析。

## 一、Section 1 Introduction 完整译文（按原文顺序）

### 自然段 1：模型规模与内存带宽限制

近年来，大语言模型（Large Language Models，LLMs）的发展表明，随着模型规模增大至数十亿参数，语言生成质量也得到了显著提升（Brown et al., 2020; Chowdhery et al., 2022; Zhang et al., 2022; Hoffmann et al., 2022; OpenAI, 2023; Google, 2023; Touvron et al., 2023）。然而，模型规模的增长也导致了推理延迟的增加，为实际应用带来了重大挑战。从系统的角度来看，LLM 推理主要受到内存带宽限制（Memory-Bandwidth-Bound）（Shazeer, 2019; Kim et al., 2023），其主要延迟瓶颈来自加速器的内存带宽，而非算术计算能力。这一瓶颈是自回归解码（Auto-regressive Decoding）的顺序特性所固有的：每次前向传播都需要将完整的模型参数从高带宽显存（High-Bandwidth Memory，HBM）传输到加速器缓存。由于这一过程只生成单个 Token，现代加速器的算术计算潜力未能得到充分利用，从而导致效率低下。

### 自然段 2：传统推测解码

为了解决这一问题，加速 LLM 推理的一种方法是提高解码过程的算术强度（Arithmetic Intensity，即总浮点运算次数 FLOPs 与总数据传输量之比），同时减少所需的解码步骤数量。基于这一思路，研究者提出了推测解码（Speculative Decoding）（Leviathan et al., 2022; Chen et al., 2023; Xia et al., 2023; Miao et al., 2023）。该方法利用一个较小的草稿模型（Draft Model）生成一段 Token 序列，然后由原始的、更大的模型对这段序列进行处理，确定可以接受的续写内容。然而，获得一个合适的草稿模型仍然具有挑战性，而将草稿模型集成到分布式系统中则更加困难（Chen et al., 2023）。

### 自然段 3：多个解码头代替独立草稿模型

本文并不使用独立的草稿模型以顺序方式生成候选输出，而是重新审视并改进了在主干模型（Backbone Model）顶部使用多个解码头来加速推理的思想（Stern et al., 2018）。我们发现，只要得到有效应用，这一技术就能够克服推测解码所面临的困难，并且可以无缝集成到现有的 LLM 系统中。具体而言，我们提出了 Medusa：一种通过集成额外的解码头，使模型能够同时预测多个 Token、从而加速 LLM 推理的方法。这些解码头可以采用参数高效的方式进行微调，并且能够添加到任何现有模型之上。由于不需要独立的草稿模型，Medusa 可以轻松集成到现有 LLM 系统中，包括分布式部署环境，从而提供便于使用的方案。

### 自然段 4：Tree Attention 和 Typical Acceptance

我们进一步通过两个关键认识来增强 Medusa。首先，当前每个解码步骤只生成一条候选续写序列的做法，会造成计算资源利用效率低下。为了解决这一问题，我们提出使用 Medusa 解码头生成多条候选续写序列，并通过对 Attention Mask 进行简单调整，实现这些候选序列的并发验证。其次，我们可以复用推测解码中使用的拒绝采样（Rejection Sampling）机制（Leviathan et al., 2022; Chen et al., 2023），从而生成与原始模型具有相同概率分布的响应。然而，这种方式无法进一步提高加速率。作为替代方案，我们提出了典型接受机制（Typical Acceptance Scheme），用于从 Medusa 解码头的输出中选择合理的候选 Token。我们使用温度（Temperature）作为控制偏离原始模型预测程度的阈值依据，为拒绝采样方法提供一种高效的替代方案。我们的结果表明，所提出的典型接受机制能够在保持相近生成质量的同时，进一步提高解码速度。

### 自然段 5：两级微调和数据来源

为了使 LLM 具备能够执行预测的 Medusa 解码头，我们针对不同场景提出了两种不同的微调流程。对于计算资源有限的场景，或者希望在不影响现有模型性能的情况下将 Medusa 集成到原始模型中的场景，我们推荐使用 Medusa-1。这种方法所需的显存很少，并且可以通过类似 QLoRA（Dettmers et al., 2023）的量化技术进一步优化。由于主干模型保持冻结，因此不会因主干参数改变而损害原有的生成质量。然而，在 Medusa-1 中，主干模型的全部潜力尚未得到充分利用。我们还可以进一步微调主干模型，提高 Medusa 解码头的预测准确率，从而直接获得更大的加速效果。因此，我们提出 Medusa-2。它适用于计算资源充足的场景，也适用于从基础模型（Base Model）直接进行监督微调（Supervised Fine-Tuning，SFT）的场景。Medusa-2 的关键在于一种训练方案：它允许 Medusa 解码头和主干模型进行联合训练，同时不损害模型的下一个 Token 预测能力与输出质量。我们根据模型的训练流程以及训练数据集的可获得性，提出了不同的训练数据获取策略。当模型在公开数据集上进行微调时，可以直接使用该数据集来训练 Medusa。如果训练数据集无法获得，或者模型已经经历了人类反馈强化学习（Reinforcement Learning with Human Feedback，RLHF）（Ouyang et al., 2022）流程，我们建议采用自蒸馏（Self-Distillation）方法，为 Medusa 解码头生成训练数据集。

### 自然段 6：实验场景与主要结论

我们的实验主要关注 Batch Size 为 1 的场景，这可以代表将 LLM 部署在本地、供个人使用的典型情形。我们在不同规模、不同训练设置的模型上测试 Medusa，包括 Vicuna-7B、Vicuna-13B（使用公开数据集训练）、Vicuna-33B（Chiang et al., 2023，使用私有数据集训练），以及 Zephyr-7B（同时经过监督微调与对齐训练）。在不同类型的 Prompt 上，Medusa 可以实现 2.3–2.8 倍的加速，同时不损害生成质量。

**原文脚注 1：** 根据作者与 Vicuna 团队的联系，Vicuna-33B 这一版本属于实验性版本，使用的部分数据与 Vicuna-7B 和 13B 不同。

## 二、Figure 1 原图、全部图注与图内标注翻译

**原始图像：** [论文 PDF 第 2 页 Figure 1](https://arxiv.org/pdf/2401.10774v3#page=2)。原图已在当前聊天从用户原始 PDF 裁图展示，但截至本次 GitHub 保存，PNG 二进制尚未上传仓库；因此不提供虚构的 figures/ 路径。

**Figure 1 图注完整翻译：**

Medusa 在 LLM 最后一层隐藏状态之上引入多个解码头，使模型能够并行预测若干后续 Token（Section 2.1.1）。在推理过程中，每个解码头都会针对其指定的预测位置生成多个概率最高的候选 Token。这些预测结果被组合成候选序列，并利用基于树的注意力机制（Tree-based Attention Mechanism）进行并行处理（Section 2.1.2）。最后一步是验证候选序列，并接受一段续写内容。除了标准的拒绝采样机制以外，还可以采用典型接受机制（Section 2.3.1）来选择合理的续写内容，并使用所有候选序列中被接受的最长前缀进入下一个解码阶段。

**图内标签：**

- Original Model：原始模型；Embedding：词嵌入；Transformer Layers：Transformer 层；Last Hidden：最后隐藏状态；LM Head：原始语言模型头。
- Medusa Heads：Medusa 解码头；Top-k Predictions：概率最高的 k 个预测；Input：输入；Candidates：候选序列；Single step prediction：单次解码步骤输出。
- 原始输入 “What will happen if Medusa meets a llama?”：如果 Medusa 遇到一只羊驼，会发生什么？
- LM Head 候选：It / I / As；Head 1：is / ' / the；Head 2：difficult / is / '；Head 3：not / difficult / a。
- 示意候选 “It is difficult not” 被标为有效；“It' difficult a” 和 “It is' not” 标为无效；输出框展示 “It is difficult”。图为示意，不应用这一点反推出精确的接受实现或接受长度。

## 三、技术讲解：原文之外的机制解释

### 3.1 Roofline：为什么 Batch=1 解码容易受到 HBM 带宽限制

设模型有效读取参数量为 P、每参数 B 字节、GPU 有效带宽为 W 字节每秒；单步至少有模型权重读取量 PB，因此简化带宽下界为
$$
T_{\mathrm{memory}}\gtrsim\frac{P B}{W}.
$$
解码时矩阵乘法通常可以写成一个新 Token 的向量-矩阵乘，权重数据复用度不足，接近 Memory-Bandwidth-Bound；而大批量 Prefill 或多候选并行验证能够提高权重重用率，使 FLOPs / byte 增大。这个下界是理想化估算：不含 KV Cache 读取、Kernel、输出头、跨卡通信等，也不表示每次一定从 HBM 完整搬运所有权重。

具体示例（独立估算，非论文实验）：7B 参数、BF16 权重约 14GB，若理论 HBM 带宽 2TB/s，则“光权重读取”的理想下界约 7ms。实际推理可受 Attention 的 KV 缓存访问、Kernel 效率和批处理影响。

### 3.2 多个未来 Token Head 的数学索引及其信息瓶颈

设原模型在时间位置 t 的最后隐藏状态为 h_t∈R^d，原 LM Head 输出
$$
p_t^{(0)}=\operatorname{softmax}(W_{\mathrm{LM}}h_t), \quad x_{t+1}\sim p_t^{(0)}.
$$
额外第 k 个 Medusa Head 从相同 h_t 预测位置 t+k+1 的 Token：
$$
p_t^{(k)}=f_k(h_t),\quad k=1,\ldots,K.
$$
下一节 Section 2.1.1 的具体结构（提前作为解析，而非 Introduction 原文）是：
$$
p_t^{(k)}=\operatorname{softmax}\left[W_2^{(k)}\left(\operatorname{SiLU}(W_1^{(k)}h_t)+h_t\right)\right].
$$
原始 LM Head 预测 t+1，Medusa Head 1 预测 t+2，Head 2 预测 t+3，依次类推。所有 Medusa Head 在预测时只见 h_t，而没有真实未来 x_{t+1:t+k}；故 Head k 近似的是未来边际预测，而非在已知实际未来前缀下的条件预测。接受阶段必须回到 Target 校验条件概率。

### 3.3 为什么需要 Tree Attention 而不是独立序列逐一验证

设 Head 1、Head 2 分别提供 s_1、s_2 个候选，可组成 s_1 s_2 条长度为 2 的候选续写。所有候选共享前缀，适合合并为候选树。对当前树节点 i，令 Anc(i) 为其祖先集合，树内 Attention Mask 应当仅允许注意力访问祖先与自身：
$$
M_{ij}=\begin{cases}
0,&j\in\operatorname{Anc}(i)\cup\{i\}\text{ 或 }j\text{ 是原始 Prompt 的可见位置},\\
-\infty,&\text{其他}.
\end{cases}
$$
树的兄弟节点不能互看，否则会把不同续写历史混合并导致条件分布不正确。每个分支节点的 Position ID 应使用其在分支中的深度，而不是展平树数组中的索引。Target 在一次 Forward 中得到树上各节点对应的条件 Logits，然后根据验证/接受规则选择可保留的连续前缀。

### 3.4 独立 Draft vs Medusa：成本比较

独立 Draft Model 为形成长度 γ 的串行草稿，一般需要 γ 次小模型 Forward，再进行 Target Forward 验证。粗略写为
$$
T_{\text{traditional}}\approx\gamma T_{\text{draft-step}}+T_{\text{target-verify}}(\gamma)+T_{\text{post}}.
$$
Medusa Heads 从共享的 h_t 并行给出多位置候选，免去额外草稿 Transformer 的多轮顺序执行；一次 Target Tree Forward 验证 N 个树节点：
$$
T_{\text{Medusa}}\approx T_{\text{heads}}+T_{\text{target-tree}}(N)+T_{\text{post}}.
$$
这只是分析性成本拆分，并非原文给出的定量性能模型。Medusa 的瓶颈转为额外 Heads 的线性映射和 Top-k、树上候选数 N 对 Target GEMM/Attention 的放大、以及 KV Cache 管理开销。树越大不一定越快。

### 3.5 精确分布与 Typical Acceptance 不能混淆

标准 Speculative Sampling 使用 Target 分布 p、Draft 条件提议 q，接受概率 min(1,p(x)/q(x))，拒绝时对正部残差 (p−q)_+ 归一化采样，以得到目标分布；实现严格分布等价时必须满足条件提议分布假设。

Medusa 的 Typical Acceptance 不强调逐 Token 输出分布与 p 完全相等，而根据 Target 当前概率与熵等信息判断候选是否“足够合理”。其具体阈值在 Section 2.3.1 用 ε、δ、entropy 给出，并非简单直接拿温度 T 当固定概率阈值；温度会改变被验证的概率分布。Introduction 的原文表述已忠实保留，但此处按后文章节预先作语义限定。Medusa-1 冻结 Backbone 不代表采用 Typical Acceptance 时自动保持精确输出分布。

### 3.6 两种训练方案与业务模型适配

Medusa-1 仅训练新增的预测头，原 Backbone 冻结，可使用量化降低训练显存并在不用更改 Target 原参数的情况下部署；Medusa-2 联合训练主干 + Heads，需要保证原 Next-Token 任务不被多步预测任务破坏。缺乏数据或经过 RLHF 的模型可使用自生成回答构造蒸馏数据。后面 Section 2.2/2.3 再正式推导损失函数、训练权重和蒸馏策略。

### 3.7 与 EAGLE、DFlash 的关联（外部已读资料，不是本节作者结论）

| 比较维度 | Medusa（本篇） | EAGLE（2401.15077） | DFlash（2602.06036） |
| --- | --- | --- | --- |
| Draft 表示 | 多个 Token 预测头 | Shifted Token 条件下的自回归 Feature 预测 | 目标多层 Feature 条件化的 Block Diffusion |
| Draft 串行深度 | 额外头一次预测多个未来位置 | 多次 Draft Forward | 单次 Draft Block Forward |
| 主要取舍 | 多头无未来条件 → 预测准确率受限；树大小与计算开销 | 利用 Feature + Shifted Token 改善未来条件不确定性，但仍有自回归 Draft | 并行 Draft 降低串行成本，依赖多层特征和 KV Injection |
| Target 验证 | Tree Attention；支持精确采样或 Typical Acceptance | 树形 Target 验证 + 精确采样 | Target 并行验证 + 精确采样 |

相同术语“2.8× speedup”也可能对应不同模型、任务、温度、GPU 和 Serving 框架，不能直接推论方法的绝对优劣。Medusa Introduction 的实验范围以 Batch=1 为主；高并发的实际生产 Serving 需另行测量。

## 四、单元小结与待续位置

- 原文完整译文：Introduction 六段 + Vicuna-33B 脚注 1。
- Figure 1：原图在聊天展示，图注全译，图内标签/候选文字已说明。
- 核心思想：将外置自回归 Draft 替换为基于目标隐藏状态的多头预测；用树状 Target Attention 同时评估多个候选；按接受策略选择最长合格前缀。
- 严格边界：冻结 Target 与“生成分布不变”不是同一命题；树更大和 Head 更多也不保证更高 wall-time speedup。
- **下一阅读单元：2. Methodology，建议先读 2.1 Key Components（2.1.1 Medusa Heads 与 2.1.2 Tree Attention，Figure 2、公式与 Tree Mask）。**
