# Section 3 Experiments：完整译文、Figure 3–5 与 Table 1–2、实验归因分析

- 论文：Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads
- 版本：arXiv:2401.10774v3，2024-06-14，ICML 2024
- 本节范围：PDF **第 6 页右栏至第 9 页左栏 Section 4 标题之前**；Figure 3 PDF 第 7 页，Figure 4/Table 1/Figure 5 PDF 第 8 页，Table 2 PDF 第 9 页。
- [PDF 第 6 页](https://arxiv.org/pdf/2401.10774v3#page=6) | [第 7 页：Figure 3](https://arxiv.org/pdf/2401.10774v3#page=7) | [第 8 页：Figure 4/5、Table 1](https://arxiv.org/pdf/2401.10774v3#page=8) | [第 9 页：Table 2](https://arxiv.org/pdf/2401.10774v3#page=9)
- [arXiv v3 HTML Section 3](https://arxiv.org/html/2401.10774v3#S3)
- **原创图表状态**：Figure 3–5、Table 1–2 已从用户提供的 v3 PDF 原图裁图，在本次聊天展示；现有 GitHub Connector 仅允许创建/更新 UTF-8 文本文件，**没有上传 PNG 二进制功能**，本 Markdown 不使用不存在的 figures/ 路径，不虚称图像上传。图片在上面指定的 PDF 页码可获得；下面完整译出五个图注/表注以及原始数值。
- **边界**：Section 3 没有论文正式编号的数学公式；性能恒等式来自 Appendix B.1 的术语定义，单独标注。Table 3 位于 **Section 4 Discussion**，不算入本节。

## 一、原文完整中文翻译（依原始阅读顺序）

### 3 Experiments（实验）

为了证明 Medusa 在不同设置下的有效性，我们进行以下实验。首先，我们在 Vicuna-7B 和 Vicuna-13B 模型（Chiang et al., 2023）上评估 Medusa，以展示 Medusa-1 和 Medusa-2 的性能。然后，我们在 Vicuna-33B 和 Zephyr-7B 上评估该方法，证明自蒸馏在无法直接获得微调流程的情况下（例如 Vicuna-33B），以及在经过人类反馈强化学习（RLHF）的模型上（例如 Zephyr-7B），仍然具有实际可行性。评估使用 MT-Bench（Zheng et al., 2023），这是一个多轮对话形式的基准测试。详细设置参见 Appendix B。

### 3.1 Case Study: Medusa-1 v.s. Medusa-2 on Vicuna 7B and 13B

**Experimental Setup（完整译文）**：我们使用 Vicuna 系列模型（Chiang et al., 2023），包括从 Llama（Touvron et al., 2023）微调得到的不同规模（7B、13B、33B）的聊天模型。其中 7B 和 13B 使用 ShareGPT（ShareGPT, 2023）数据集训练，33B 是基于私有数据集训练的实验性模型。在这一节，我们使用 ShareGPT 数据集对 7B 和 13B 上的 Medusa Heads 训练 2 个 Epoch。我们使用 Vicuna v1.5 版本，它从 Llama-2 模型微调而来，序列长度为 4096。

**Results（完整译文）**：我们汇总结果并展示在 Figure 3 中。Baseline 是 Hugging Face 的默认实现。从 Figure 3(a) 可以看到，对于 7B 模型，Medusa-1 与 Medusa-2 都显著提升了以每秒 Token 数衡量的速度：Medusa-1 加速 2.18×，Medusa-2 进一步提高到 2.83×。对于更大的 13B 模型，Medusa-1 加速 2.33×，Medusa-2 仍保持相对于基线 2.83× 的增益。我们还展示了 Medusa-2 Vicuna-7B 按 MT-Bench 类别划分的加速比。Coding 类别达到 3.29×，说明 Medusa 对这类任务尤其有效，并显示在广泛应用于软件开发和编程任务的 Coding LLM 上进行优化的潜力。Extraction 类别的加速比最高，达到 3.62×，表明 Medusa 在该任务中具有特别好的优化效果。总体而言，这些结果说明 Medusa 在不同模型规模与不同任务上都能显著提高推理速度。

#### Figure 3 原图位置、图注和全部数据

原图：[PDF 第 7 页 Figure 3](https://arxiv.org/pdf/2401.10774v3#page=7)，原始裁图在当前聊天展示。

**图注完整翻译**：左图：在 Vicuna-7B/13B 上，比较 Baseline、Medusa-1 与 Medusa-2 的速度。Medusa-1 相比基线实现超过 2 倍的实际运行时间（Wall-time）加速，Medusa-2 进一步显著提高加速比。右图：展示 Medusa-2 Vicuna-7B 在 MT-Bench 八个类别上的详细加速性能。

**Figure 3(a) 明确标注的净速度加速（不是自行读出柱高）**：

| 模型 | Medusa-1 | Medusa-2 |
| --- | ---: | ---: |
| Vicuna-7B | 2.18× | 2.83× |
| Vicuna-13B | 2.33× | 2.83× |

注意原图纵轴为 Tokens per Second，虽有柱形高度但没有逐柱精确 TPS 数字标签，不能把目测值写成精确实验数据。

**Figure 3(b) 图上明确标注的全部 8 类净加速比**：

| Category | 中文 | Speedup |
| --- | --- | ---: |
| Humanities | 人文 | 2.58× |
| Reasoning | 推理 | 2.58× |
| Roleplay | 角色扮演 | 2.70× |
| Writing | 写作 | 2.72× |
| Stem | 科学、技术、工程与数学 | 2.77× |
| Math | 数学 | 3.01× |
| Coding | 编程 | 3.29× |
| Extraction | 信息提取 | 3.62× |

### 3.2 Case Study: Training with Self-Distillation on Vicuna-33B and Zephyr-7B

**Experimental Setup（完整译文）**：本案例关注需要自蒸馏的情形，以 Vicuna-33B（Chiang et al., 2023）与 Zephyr-7B（Tunstall et al., 2023）为例。按照 Section 2.3.2 的流程，我们先利用种子 Prompt 生成数据集。使用 ShareGPT（ShareGPT, 2023）和 UltraChat（Ding et al., 2023）作为种子数据集，为这两种情况分别收集约 100k 条样本。有趣的是，Zephyr 能在只收到一条 Prompt 后继续生成多轮对话，因此容易收集大型数据集。对于 Vicuna-33B，我们逐轮输入各个多轮种子对话中的 Prompt，以温度为 0.3 的随机采样生成多轮对话。两种模型的训练序列长度均为 2048，Batch Size 均为 128。

**Results（完整译文，跨 PDF 第 7–8 页）**：Table 1 补充这些发现，比较不同 Medusa-2 模型的加速率、额外开销和质量，其中使用 GPT-4 作为 MT-Bench 评判模型，得分范围为 0–10。我们报告 Medusa 相比原模型的质量变化。值得注意的是，尽管 Medusa-2 Vicuna-33B 的加速率较低，仍保持相当的质量。我们推测原因是原始模型未公开的训练数据与我们的自蒸馏数据之间存在不匹配。因此，模型生成质量可以经自蒸馏保持良好的一致性，但 Medusa Heads 从自蒸馏数据学习的分布可能与原始训练集有所偏离。研究中，我们也在 Vicuna 系列上使用开源 Draft Model 测试传统推测解码（Chen et al., 2023; Leviathan et al., 2022），详情见 Appendix D。

这些结果显示，在扩大模型规模及运用自蒸馏技术时，速度与模型表现间存在复杂相互作用。它们也凸显 Medusa-2 在谨慎维持模型输出质量的同时提升处理效率的潜力，为 LLM 与 Medusa Heads 的共同优化提供一个有前景的方向。

#### Table 1 原表位置、完整表注与原始数值

原表：[PDF 第 8 页 Table 1](https://arxiv.org/pdf/2401.10774v3#page=8)，原始裁图在当前聊天展示。

**表注完整翻译**：不同 Medusa-2 模型的比较。第一部分报告 Medusa-2 的细节，包括加速率、额外开销和生成质量（相对于原始模型的 MT-Bench 平均评分）。第二部分分别列出传统推测解码 SpecDecoding 与 Medusa 的加速比 $S$。

| 指标 | Vicuna-7B | Zephyr-7B | Vicuna-13B | Vicuna-33B |
| --- | ---: | ---: | ---: | ---: |
| Acc. rate | 3.47 | 3.14 | 3.51 | 3.01 |
| Overhead | 1.22 | 1.18 | 1.23 | 1.27 |
| Quality | 6.18 (+0.01) | 7.25 (-0.07) | 6.43 (-0.14) | 7.18 (+0.05) |
| $S_{\mathrm{SpecDecoding}}$ | 1.47 | – | 1.56 | 1.60 |
| $S_{\mathrm{MEDUSA}}$ | 2.83 | 2.66 | 2.83 | 2.35 |

原表未报告 Zephyr-7B 的 SpecDecoding Speedup，横线不是 0。

### 3.3 Ablation Study — 消融实验

#### 3.3.1 Configuration of Tree Attention

**自然段 1（全文翻译）**：我们使用 Medusa-2 Vicuna-7B，在 MT-Bench 的 Writing 和 Roleplay 类别上进行 Tree Attention 研究，旨在展示采用 Tree Attention 的动机及其性能。

**自然段 2（全文翻译，跨页/分栏）**：Figure 4(a) 将随机采样得到的稠密树配置（Section 2.1.2，以蓝色圆点表示）的加速率，与经过优化的稀疏树配置（Section 2.3.3，以红色星形表示）进行比较。具有 64 个节点的稀疏树比具有 256 个节点的稠密树获得更高的加速率。Figure 4(b) 中的速度下降，是计算受限（Compute-Bound）导致的额外开销增长造成的。虽然更复杂的树可以提高加速率，但由于线性层和自注意力中大量矩阵乘法所需的计算，整体速度会下降。Figure 4(a) 显示，加速率的增长呈现近似对数趋势，在树变大时增长逐渐变慢。然而初期增益十分可观，使 Medusa 能够取得显著加速。如果加速率提升不足以抵消额外开销，整体性能反而会降低。详细研究见 Appendix G。

#### Figure 4 原图位置与完整图注

原图：[PDF 第 8 页 Figure 4](https://arxiv.org/pdf/2401.10774v3#page=8)，原始裁图在当前聊天展示。

**图注完整翻译**：研究候选树引入的解码候选 Token 数量对性能的影响（采用 KV Cache 的普通解码每步默认只解码一个候选 Token）。左图：比较随机采样的稠密树（蓝色点）和优化后的稀疏树（红色星形）的 Acceleration Rate。右图：两类配置的实际解码速度（Tokens/s）。趋势线表明，尽管稀疏树的加速率相对稳定，随着候选 Token 数增加，实际速度会显著下降。

图内坐标翻译：Acc. Rate＝平均每步产生 Token 数；Speed (token/s)＝每秒生成 Token 数；Number of Candidate Tokens＝候选 Token 数；Sparse Tree Attention＝稀疏树注意力；w/o Medusa＝不使用 Medusa。蓝点、红星分别对应 Dense Random Tree 与 Optimized Sparse Tree，并**不是**训练时 Head 数量直接扫描。

#### 3.3.2 Thresholds of Typical Acceptance

**全文翻译**：我们在 MT-Bench（Zheng et al., 2023）的 Writing 和 Roleplay 类别上，使用 Medusa-2 Vicuna-7B 研究 Typical Acceptance 的阈值。遵循 Hewitt et al.（2022）所述方法，我们设置 $\alpha=\sqrt{\epsilon}$。Figure 5 比较不同采样方案下的性能：阈值 $\epsilon$ 从 $0.01$ 开始，以 $0.01$ 逐渐递增到 $0.25$。结果显示明显的权衡：$\epsilon$ 增大时，生成质量提高，但 Acceleration Rate 降低。此外，对于需要创造性的任务，默认的随机采样表现优于贪心采样；随着 $\epsilon$ 增大，我们提出的 Typical Sampling 能取得与随机采样相近的表现。

#### Figure 5 原图位置与完整图注

原图：[PDF 第 8 页 Figure 5](https://arxiv.org/pdf/2401.10774v3#page=8)，原始裁图在当前聊天展示。

**图注完整翻译**：使用本文提出的 Typical Sampling 的 Medusa 性能比较。模型由 Vicuna-7B 进行了完整微调。图在采样温度固定为 $0.7$ 的情况下，展示 Writing 和 Roleplay（MT-Bench）两类任务的 Acceleration Rate 与平均评分。共包括三种配置：贪心采样和随机采样（RS）分别绘制为星形与圆点；Typical Sampling 则绘制为不同阈值下的曲线。

**忠实标注边界**：Figure 5 使用左右双纵轴：蓝色 Acceleration Rate 与橙色 Scores；横轴 Posterior Thresholds。曲线没有提供 $25$ 个阈值下的逐点精确测量表，因此不应把图上近似读数包装成原始表格数据。Section 2.3.1 的公式使用 $\delta$，但 3.3.2 的实验使用 $\alpha=\sqrt{\epsilon}$，原文未明确说明 $\alpha$ 与 $\delta$ 的关系；可将其视作可能对应同一熵阈值系数的记号差异，不能擅自修改原文。

#### 3.3.3 Effectiveness of Two-stage Fine-tuning

**全文翻译**：Table 2 比较了 Vicuna-7B 的不同微调策略。Medusa-1 只训练 Medusa Heads，实现 2.18× 加速而不牺牲生成质量。Medusa-2 使用 Section 2.2.2 的两阶段微调，在维持生成质量的同时，相对 Medusa-1 取得更大的 2.83× 加速。相对地，直接将主干模型和 Medusa Heads 一起微调会损害生成质量。结果表明，使用我们的 Medusa-2 方案进行微调，可以同时保持模型质量，并进一步提高相对 Medusa-1 的速度。

#### Table 2 原图位置、完整表注与数值

原表：[PDF 第 9 页 Table 2](https://arxiv.org/pdf/2401.10774v3#page=9)，原始裁图在当前聊天展示。

**表注完整翻译**：Vicuna-7B 不同配置的比较。模型生成质量通过 MT-Bench 评估，由 GPT-4 充当评判模型；分数越高越好。

| 指标 | Baseline | Direct Fine-tuning | Medusa-1 | Medusa-2 |
| --- | ---: | ---: | ---: | ---: |
| Quality | 6.17 | 5.925 | 6.23 | 6.18 |
| Speedup | N/A | N/A | 2.18 | 2.83 |

N/A 表示作者未提供该列的加速比，不代表 0×。**论文 Section 3 到此结束。接下来的 Table 3 属于 Section 4 Discussion**，本单元不将其误归为 Section 3。

## 二、独立分析：三个性能指标的定义与反算（来源补充：Appendix B.1）

**Appendix B.1 原文定义**：

1. Acceleration Rate（加速率）＝每个解码步骤平均生成/接受的 Token 数。普通 AR 解码为 1.0。它是每步**Token 数量**，不是直接的时间加速比。
2. Overhead（额外开销）＝Medusa 平均单步延迟 / 原始模型平均单步延迟。它是无量纲的**时间比例**。
3. Speedup（净加速比）＝原始解码 Wall Time / Medusa Wall Time。

因此按作者定义：
$$
\boxed{\mathrm{Speedup}=\frac{\mathrm{Acceleration\ Rate}}{\mathrm{Overhead}}}.
$$

四个模型按 Table 1 **已展示两位小数**反算：

| 模型 | Acc/Overhead 反算值 | Table 1 官方 Speedup | 二者差（反算 − 报告） |
| --- | ---: | ---: | ---: |
| Vicuna-7B | 2.8443 | 2.83 | +0.0143 |
| Zephyr-7B | 2.6610 | 2.66 | +0.0010 |
| Vicuna-13B | 2.8537 | 2.83 | +0.0237 |
| Vicuna-33B | 2.3701 | 2.35 | +0.0201 |

**校核提示**：反算值与官方数据存在 0.001–0.024 的差异。不能默改 Table 1，也不能对全部差异都武断解释为只保留两位小数的舍入误差：例如 Vicuna-13B 的反算 2.854 与 2.83 偏差约 0.024，可能涉及不同层次的统计聚合、指标独立舍入或实现统计口径。现有 Section 3 只提供上述汇总，不给每个样本原始计时数据，**无法判定精确原因**。GitHub 和聊天均保留论文报告值。

假设平均每步接受 3.5 Token，Medusa 每步实际耗时是普通解码的 1.25 倍，则净加速约 3.5/1.25=2.8×，而非 3.5×。高接受长度不会自动保证更高 Tokens/s。

### 2.1 与传统 Speculative Decoding 的同表对比

用 **Table 1 中同一行同一模型**比较，Medusa/传统 SpecDecoding 的报告加速比为：Vicuna-7B 2.83/1.47≈1.93 倍；Vicuna-13B 2.83/1.56≈1.81 倍；Vicuna-33B 2.35/1.60≈1.47 倍。Zephyr-7B 没有对应 SpecDecoding 结果，不能横向补值。注意这是**论文同场比较的比值**，不是对所有 Draft Model/所有部署环境的普适结论。Appendix D 说明独立小 Draft 的选择、草稿长度 $\gamma$、推理实现会影响结果，后续正式阅读。

### 2.2 为什么模型更大不一定加速倍数更高

Table 1 中 Vicuna-13B 的 Acc Rate 3.51 为最高、Overhead 1.23；Vicuna-33B 的 Acc Rate 3.01 为最低、Overhead 1.27 为最高，净加速仅 2.35×。作者给出的解释是 Vicuna-33B 隐藏训练数据与自蒸馏样本之间分布偏差，影响 Heads 预测。**这只是作者的假设**，并没有在 Section 3 提供直接的分布距离或每 Head 接受率检验，不能将其升格为实验证明。

模型尺寸、KV Cache、Target Kernel 算术强度、候选树形、训练数据与输出文本长度共同影响延迟；因此仅凭一个“模型参数量”维度无法预测速度增益。Zephyr-7B 的质量 -0.07 和 Vicuna-13B -0.14 是 GPT-4 MT-Bench 平均分的微小变化，不是严格等分布保证，也没有显著性检验数据。

### 2.3 MT-Bench 质量分数和样本不确定性

原文 GPT-4 作为评分者，给出 0–10 分，报告均值，未在 Table 1/2 提供置信区间、标准误、重复次数或显著性检验。不能据 +0.01、+0.05 认定确定地提升了“真实质量”，也不能据 -0.07、-0.14 确定造成统计显著下降。Medusa-1 冻结 Backbone 不意味着在 Typical Acceptance 下采样分布严格不变。

## 三、独立分析：Figure 3 八类任务为何差异显著

Figure 3(b) 精确标注从 Humanities/Reasoning 2.58× 到 Extraction 3.62×、Coding 3.29×。可能解释包括：不同任务中局部 Token 的可预测性不同，公式化格式、关键词重复、短语结构可能使较远 Head 更容易预测；同时 Prompt 长度、输出长度、采样方式和接受率也会影响结果。**这些解释是独立假说**：作者在 Section 3 只把 Coding 和 Extraction 称为效果明显，并没有报告逐类的 Acceleration Rate、Per-head Accuracy 或平均输出长度来验证这些机制。若工程复现，建议按任务类别同时记录目标模型每步验证耗时、草稿节点数、接受前缀长度、TPS、长度分位数以及 P50/P95 的单请求完成时间。

Figure 3(a) 报告的是对默认 Hugging Face Baseline 的推理速度，研究以 Batch=1 为主要场景，不应直接推断到 vLLM/SGLang 的高并发吞吐。

## 四、独立分析：Figure 4 Tree Size / Compute-Bound 权衡

作者的强证据是：优化的 **64 节点稀疏树**在 Acceleration Rate 上优于**256 节点随机稠密树**。这说明更合理的候选节点选择，可以用较少的计算获得更长的有效预测前缀。需要区分三个量：

$$
N=\mathrm{tree\ candidate\ nodes},\quad
A(N)=\mathbb E[\mathrm{accepted\ tokens\ per\ step}],\quad
T(N)=\mathrm{wall\ time\ per\ step}.
$$

与普通 AR 解码的平均步时 $T_0$ 比较：
$$
S(N)=\frac{A(N)}{T(N)/T_0}.
$$
增加 N 时，A(N) 可以继续增大，但 T(N) 也增大，且边际 A(N) 增量通常递减。取导数给出内部最优条件（独立推导，非原文）：
$$
\frac{d\log A(N)}{dN}=\frac{d\log T(N)}{dN}.
$$
即接受长度的**相对边际收益**等于步时开销的**相对边际增长**，而非一味最大化 A(N)。矩阵乘法和 Attention 处理的新增 Token 节点数增加，系统可能从 HBM Bandwidth-Bound 逐渐转向 Compute-Bound。图 4(b) 体现该变化：红色稀疏树的 Acceleration Rate 随预算由约 64→256 提升，但实际 Tokens/s 有下降；不要将图上曲线的近似像素读数当成表格原始精确值。

图中蓝色随机稠密树与红色优化稀疏树的比较可能包含不止树密度一个差异：树节点排列/分布也不同，反映的是**优化配置的总体效果**，不是严格只改变“稀疏性”的单变量消融。尤其不能把 Tree Node 数等同于 Head 数。

## 五、独立分析：Figure 5 Typical Acceptance 阈值消融

原作者 Figure 5 采用固定温度 $T=0.7$，比较 Writing/Roleplay 的 GPT-4 平均评分（右侧橙色轴）和 Acceleration Rate（左侧蓝色轴）。论文仅报告 $\epsilon=0.01$ 到 $0.25$ 每隔 $0.01$ 的阈值扫描，并提到 $\alpha=\sqrt{\epsilon}$；它并**未提供各点精确表格**。

Section 2.3.1 的阈值是：
$$\tau(p)=\min(\epsilon,\delta e^{-H(p)}).$$
如果在保持另一个系数不变时增大 $\epsilon$，$\tau(p)$ 非减，因此通过阈值的候选 Token 集合不增。更严格阈值通常导致连续接受长度下降；相应质量可能更接近通常随机采样的评估质量，这与 Figure 5 的总体趋势一致。**但 3.3.2 将 $\alpha$ 设为随 $\epsilon$ 变化，且作者在原文没有明确说明 $\alpha$ 与 2.3.1 的 $\delta$ 的对应关系**，不可完全将一维固定 $\delta$ 的推导替代作者实测设置。

Figure 5 里 **Greedy 星形**和 **RS 圆点**是对照采样模式，不是加入了额外“默认 acceptance threshold”的两组同类曲线。右轴橙线有明显抖动，不是随每个 $\epsilon$ 单调上升；作者说的是总体权衡而非逐点数学单调性。质量评估只覆盖 Writing/Roleplay，不应自动外推到数学、代码和全部任务。

**分布保证区分**：Typical Acceptance 通过概率阈值选择合理候选，可能改变原 Target Sampling Distribution；它与严格维持 Target 采样分布的 Rejection Sampling 是不同目标。Greedy/T=0、T=0.7 下的评分也不是同一采样配置，比较时注意约束。

## 六、独立分析：Table 2 两阶段训练消融的可验证结论

原始分数：
- Baseline: 6.17
- Direct Fine-tuning：5.925（相对 Baseline **-0.245**）
- Medusa-1：6.23（+0.06），净加速 2.18×
- Medusa-2：6.18（+0.01），净加速 2.83×

在表面数值上，Medusa-2 相对 Medusa-1 净加速提高 2.83/2.18≈1.298 倍，即额外约 29.8%。但 Medusa-2 质量 6.18 比 Medusa-1 的 6.23 低 0.05，不能说它在该表上同时提高了质量。作者强调的是 **Medusa-2 相对原始 Baseline 基本保留质量**，相比 Medusa-1 有更高加速。Direct Fine-tuning 的 5.925 表明朴素联合训练会损害该配置下的 GPT-4 MT-Bench 分数；N/A 不意味着没有任何速度收益。

这组数据支持两阶段训练的必要性，但未报告种子重复、误差棒和学习率/参数量完全匹配的逐策略控制变量，因此还不能完全把差异归因于 Warmup 单个因素。Section 2.2 的联合 LM Loss、差异化学习率和 Heads Warmup 是作为一个整体组合的训练方案；**Table 2 不是这些子策略逐项关闭的详细消融**。

## 七、额外背景核对：Appendix B 的原始设置（不是 Section 3 正文译文）

为了理解前面的实验，附录 B.1–B.4 记录：
- 本文主要围绕 **Batch=1 推理**展开；第 3.1 实验 Vicuna-7B/13B 的训练使用 2 Epoch、ShareGPT、序列长度 4096；两模型在 Appendix B.3 采用 4-bit Backbone、QLoRA、Global Batch 64；Backbone 和 Heads 峰值 LR 分别 $5\times10^{-4}$、$2\times10^{-3}$，40 Warmup Steps、$\lambda_0=0.2$。
- 第 3.2 自蒸馏 33B、Zephyr-7B 生成约 100k 条样本、序列长度 2048、Batch 128。Appendix B.4 说这些实验采用 Medusa-2 而非两阶段训练，用一种 Sine Schedule 渐增 $\theta_0$，Backbone LoRA 峰值 LR $10^{-4}$、20 Warmup Steps、$\lambda_0=0.01$；注意这与 Table 2 针对 Vicuna-7B 的两阶段结论**不是同一训练配置**。
- 论文主文 2.3.1 的 $\delta$ 与实验 3.3.2 的 $\alpha$、附录 B.4 的 $\theta_0$ 和主文 2.2.2 的 $\lambda_0$ 存在不同记号。这里忠实保留原文，不擅自替换等同。

真正完整翻译 Appendix B 时将重新逐项核对其原文及范围。

## 八、与 EAGLE、DFlash 的跨论文比较（额外分析）

| 维度 | Medusa（本节） | EAGLE（2401.15077） | DFlash（2602.06036） |
| --- | --- | --- | --- |
| Draft 方式 | 同一个隐藏状态的多个 Future Token Heads | Feature 级自回归预测 + Shifted Token | 条件 Block Diffusion 并行草稿 |
| 主要优化 | Tree Nodes / Head 数 / Typical Threshold / M-1 vs M-2 | Draft Feature 预测质量与 Tree Attention；多轮 Spec Sampling | Block Size / KV Injection / 多层 Target Feature |
| 性能来源 | Batch=1、Hugging Face Baseline，Vicuna/Zephyr；2.35–2.83× 净加速，Figure 3 单类别最高 3.62× | 指定模型/任务/温度下的净延迟与吞吐加速，非统一基准 | Qwen 模型、不同 Serving 后端和并发下结果差异显著 |
| 对原分布的要求 | 可选精确推测采样；Typical Acceptance 可偏离 | 正确精确推测采样时保持 | 正确精确推测采样时保持 |
| 横比约束 | 同模型、相同 Prompt/输出长度、硬件、温度、Batch、后端才可有意义比较 | 同左 | 同左 |

Medusa 的 Figure 4 说明**候选树节点布局质量**是有独立价值的优化维度；EAGLE 则把更多研究努力放在提升 Draft Feature 的准确性；DFlash 把更多研究努力放在减少 Draft 的串行 Forward 深度。三者依旧共享“多预测 → Target 验证 → 以平均每轮生成 Token 数与开销共同决定速度”的基本框架。仅凭某篇论文的最大倍数，不能判定另一篇在同等条件下更慢或更快。

## 九、完成范围与下一位置

- 已逐段完整翻译 Section 3 开头、3.1 Setup/Results、3.2 Setup/Results、3.3.1–3.3.3 及五个 Figure/Table 注释。
- 已展示 Figure 3–5、Table 1–2 真实 PDF 裁图；图表内容严格保留原数据，Figure 3(b) 八类全部抄录，Table 1 四模型所有五行数据和 Table 2 所有四列数值全部抄录。
- 数学与工程分析：Speedup=Acceleration Rate / Overhead、报告值与反算值之间的差异、同表传统 SpecDecoding 对比、Tree Size 开销曲线、Typical Acceptance ε/α/δ 差异、两阶段训练消融与局限。
- **下一阅读单元：Section 4 Discussion**。注意 **Table 3 在 PDF 第 9 页，但属于 Section 4**，届时再正式展示并完整翻译该表。
