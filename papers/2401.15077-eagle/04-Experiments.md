# 4. Experiments：全文翻译、Table 1–7、Figure 7–8、性能分析

> 原文：EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty，arXiv:2401.15077v3  
> 来源：用户提供的 [原始 PDF](https://arxiv.org/pdf/2401.15077v3)，第 5–8 页；阅读时间：2026-10-16。  
> 范围：Section 4（实验设置、Metrics、Training、4.1–4.4，包含 4.3.1–4.3.3）、Table 1–7 和 Figure 7–8。  
> **“原文完整翻译”与“独立技术分析”明确分开。** 图表数字逐项核对 PDF 原表；聊天已从原 PDF 直接裁出 9 张原图/原表。尚未向 GitHub 上传二进制 PNG 时，以下仅引用可核验的原 PDF 页码，不创建不存在的 figures/ 路径。

## 一、Section 4 完整翻译（按原文阅读顺序）

### 4. Experiments — Models and tasks（模型与任务）

我们在 Vicuna 系列（7B、13B、33B）、LLaMA2-Chat 系列（7B、13B、70B），以及 Mixtral 8x7B Instruct 上进行实验，覆盖当前主流 LLM 的常见规模。

我们在多种任务上评估 EAGLE，包括多轮对话、代码生成、数学推理、指令遵循，分别采用 MT-bench（Zheng et al., 2023）、HumanEval（Chen et al., 2021）、GSM8K（Cobbe et al., 2021）和 Alpaca（Taori et al., 2023）数据集。

Speculative Sampling（Leviathan et al., 2023）使用 Batch Size=1 的实验设置，随后 DistillSpec（Zhou et al., 2023）和 BiLD（Kim et al., 2023）等工作也采用相同设置。类似地，我们的大多数实验使用 Batch Size=1。Batch Size 大于 1 时的实验见 Section 4.4。

### Metrics（评价指标）

类似其他基于推测采样的方法，EAGLE 主要关注的是延迟，而不是吞吐量。我们采用以下指标评估加速：

- **Walltime speedup ratio：** 相对于原始自回归解码的实际耗时加速比。
- **Average acceptance length $\tau$：** 目标 LLM 每次前向传播平均接受的 Token 数量。
- **Acceptance rate $\alpha$：** 草稿生成时已接受 Token 与已生成 Token 的比例，衡量草稿准确性。该指标不太适合树状草稿，因为同一位置会提出多个候选但只有一个会被接受。因此测量接受率时采用不使用 Tree Attention 的链状草稿，从而与 Speculative Sampling 和 DistillSpec 对齐。EAGLE 的 Draft Model 输入特征和 Token 序列；由于自回归特征预测会传播误差，我们将有 $n$ 个 Draft 预测的、可能存在偏差的 Feature 时的接受率记为 $n\text{-}\alpha$。

EAGLE 理论上保证保留目标 LLM 的输出分布。因此，作者认为评估 EAGLE 的生成质量既不必要也没有意义。（这是作者原文判断；在工程分析中仍需检查实现正确性及分布变换的一致性。）

### Training（训练）

我们固定 Target LLM。EAGLE 在 ShareGPT 数据集上进行训练，使用了 68,000 次对话迭代，学习率为 $3\times10^{-5}$。优化器为 AdamW，$\beta_1=0.9$、$\beta_2=0.95$，Gradient Clipping=0.5。

7B、13B、33B、70B 目标模型的 EAGLE 可训练参数量分别是 0.24B、0.37B、0.56B、0.99B。Mixtral 8x7B 的 EAGLE 可训练参数量为 0.28B。EAGLE 训练成本较低，70B 模型所用 Autoregression Head 在 A100 40GB GPU 服务器上可以在 1–2 天内完成训练。

### 4.1 Effectiveness（有效性）— 第一段

Figure 1、Figure 2 和 Table 1 给出了 EAGLE 的加速比。相比 Temperature=1，EAGLE 在 Temperature=0 下加速效果更好。例如 LLaMA2-Chat 13B 在 Temperature=0 时，加速比为 3.01×–3.76×，而在 Temperature=1 时为 2.66×–2.89×。

EAGLE 在代码生成任务（HumanEval）上获得最佳加速效果。这归因于代码存在大量固定模板，使得对这些模板生成草稿更加容易。

相比最近提出的 Lookahead 与 Medusa 等基于推测采样的方法，EAGLE 分别快 1.70×–2.08× 与 1.47×–1.60×。

在 Vicuna 和 LLaMA2-Chat 系列上使用传统推测采样存在挑战：7B 没有合适的草稿模型；对于其他规模，我们采用 7B 模型作为 Draft Model，依次测试从 2 到 10 的草稿长度并报告最高加速比。对于 13B，未观察到速度改善；对于 33B 和 70B，加速比分别为 1.12× 与 1.88×。

对于 DistillSpec，为保证公平，使用与 EAGLE 相同的训练数据，并使用 DistillSpec 论文 Appendix A.1 描述的 FKL 散度函数。尽管蒸馏使加速比略有提高，但这种增益仍然有限，因为蒸馏的目标是提高草稿模型的接受率，而传统 Speculative Sampling 的性能瓶颈在于 Draft Model 自身的高开销。

### Table 1：HumanEval、GSM8K 与 Alpaca

[原始 Table 1，PDF 第 6 页](https://arxiv.org/pdf/2401.15077v3#page=6)。

**表注完整翻译：** 表 1：在 HumanEval、GSM8K 和 Alpaca 上的加速比及平均接受长度 $\tau$。T 表示 Temperature，V 表示 Vicuna，LC 表示 LLaMA2-Chat。

下面每格为 **Speedup / $\tau$**，数据完全依据原表。

| T | Model | HumanEval | GSM8K | Alpaca |
| --- | --- | --- | --- | --- |
| 0 | V 7B | 3.33× / 4.29 | 3.01× / 4.00 | 2.79× / 3.86 |
| 0 | V 13B | 3.58× / 4.39 | 3.08× / 3.97 | 3.03× / 3.95 |
| 0 | V 33B | 3.67× / 4.28 | 3.25× / 3.94 | 2.97× / 3.61 |
| 0 | LC 7B | 3.17× / 4.24 | 2.91× / 3.82 | 2.78× / 3.71 |
| 0 | LC 13B | **3.76× / 4.52** | 3.20× / 4.03 | 3.01× / 3.83 |
| 0 | LC 70B | 3.52× / 4.42 | 3.03× / 3.93 | 2.97× / 3.77 |
| 1 | V 7B | 2.39× / 3.43 | 2.34× / 3.29 | 2.21× / 3.30 |
| 1 | V 13B | 2.65× / 3.63 | 2.57× / 3.60 | 2.45× / 3.57 |
| 1 | V 33B | 2.76× / 3.62 | 2.77× / 3.60 | 2.52× / 3.32 |
| 1 | LC 7B | 2.61× / 3.79 | 2.40× / 3.52 | 2.29× / 3.33 |
| 1 | LC 13B | 2.89× / 3.78 | 2.82× / 3.67 | 2.66× / 3.55 |
| 1 | LC 70B | 2.92× / 3.76 | 2.74× / 3.58 | 2.65× / 3.47 |

### Table 2：MT-bench 的接受长度与接受率

[原始 Table 2，PDF 第 6 页](https://arxiv.org/pdf/2401.15077v3#page=6)。

**表注完整翻译：** 表 2：MT-bench 上的平均接受长度 $\tau$ 与接受率 $\alpha$。T 表示 Temperature。

| T | Model | $\tau$ | $0\text{-}\alpha$ | $1\text{-}\alpha$ | $2\text{-}\alpha$ | $3\text{-}\alpha$ | $4\text{-}\alpha$ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | Vicuna 7B | 3.94 | 0.79 | 0.74 | 0.72 | 0.73 | 0.67 |
| 0 | Vicuna 13B | 3.98 | 0.79 | 0.74 | 0.72 | 0.74 | 0.70 |
| 0 | Vicuna 33B | 3.68 | 0.74 | 0.69 | 0.67 | 0.67 | 0.66 |
| 0 | LLaMA2-Chat 7B | 3.62 | 0.76 | 0.69 | 0.67 | 0.68 | 0.68 |
| 0 | LLaMA2-Chat 13B | 3.90 | 0.77 | 0.69 | 0.69 | 0.70 | 0.71 |
| 0 | LLaMA2-Chat 70B | 3.81 | 0.75 | 0.69 | 0.65 | 0.64 | 0.64 |
| 1 | Vicuna 7B | 3.17 | 0.71 | 0.68 | 0.66 | 0.66 | 0.65 |
| 1 | Vicuna 13B | 3.20 | 0.73 | 0.68 | 0.68 | 0.67 | 0.69 |
| 1 | Vicuna 33B | 3.22 | 0.71 | 0.67 | 0.64 | 0.64 | 0.64 |
| 1 | LLaMA2-Chat 7B | 3.30 | 0.71 | 0.66 | 0.66 | 0.66 | 0.64 |
| 1 | LLaMA2-Chat 13B | 3.45 | 0.73 | 0.69 | 0.66 | 0.67 | 0.67 |
| 1 | LLaMA2-Chat 70B | 3.46 | 0.73 | 0.67 | 0.64 | 0.66 | 0.65 |

### 4.1 Effectiveness — 第二、三段

Table 1 和 Table 2 表明，在 EAGLE 中，Target LLM 每次前向传播生成 3.2–4.5 个 Token，相比每次前向传播只生成一个 Token 的普通解码有明显提升，从而显著加快生成速度。

如 Figure 2 和 Appendix B 所示，输入特征序列完全准确时的接受率 $0\text{-}\alpha$ 显著高于序列含一个不准确特征时的接受率 $1\text{-}\alpha$，说明特征预测误差确实影响 Draft Model 性能。不过，从 $1\text{-}\alpha$ 到 $4\text{-}\alpha$ 的变化较小，说明 EAGLE 对特征误差比较稳健，能够较好地处理误差累积。

Table 3 表明，EAGLE 在 Mixtral 8x7B Instruct 上达到 1.5× 加速。与 LLaMA 等模型相比，加速较低的原因包括平均接受长度较短，以及通过推测采样加速 MoE 模型的复杂性。MoE 模型在普通自回归解码中通常每个 Token 只需读取两个 Expert 的权重；但在推测采样验证阶段，同时处理多个 Token 可能需要访问超过两个 Expert 的权重。相比之下，Dense Decoder-only 模型无论本轮处理几个 Token，通常都要读取相同的全部模型权重。

### Table 3：Mixtral 8x7B Instruct-v0.1

[原始 Table 3，PDF 第 6 页](https://arxiv.org/pdf/2401.15077v3#page=6)。

**表注完整翻译：** 表 3：Temperature=0 时，在 MT-bench 上使用 Mixtral 8x7B Instruct-v0.1 作为 Target LLM 的加速比、平均接受长度 $\tau$ 和接受率 $\alpha$。

| Speedup | $\tau$ | $0\text{-}\alpha$ | $1\text{-}\alpha$ | $2\text{-}\alpha$ | $3\text{-}\alpha$ | $4\text{-}\alpha$ |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.50× | 3.25 | 0.67 | 0.62 | 0.61 | 0.64 | 0.63 |

### 4.2 Case study: EAGLE + gpt-fast

EAGLE 可以与其他加速技术结合使用。我们将 EAGLE 与 gpt-fast 结合进行实验；gpt-fast 使用量化和编译加速推理。**原文这里写“如 Figure 4 所示”**，但本节相关生成速度数值实际列在 Table 4 中，因此保留作者原文的图号表述，同时提示表格定位。结合 EAGLE 和 gpt-fast 后，我们在单张 RTX 3090 上将 LLaMA2-Chat 7B 的生成速度提高到了 160.4 tokens/s。

### Table 4：结合 gpt-fast 的生成速度

[原始 Table 4，PDF 第 6 页](https://arxiv.org/pdf/2401.15077v3#page=6)。

**表注完整翻译：** 表 4：EAGLE 与 gpt-fast 结合的生成速度。在 MT-bench 上使用 LLaMA2-Chat 7B、Temperature=0 评测。

| 方法 | FP16 | int4 |
| --- | ---: | ---: |
| Vanilla（Hugging Face） | 24.5 tokens/s | N/A |
| gpt-fast | 55.1 tokens/s | 106.9 tokens/s |
| EAGLE + gpt-fast | **100.2 tokens/s** | **160.4 tokens/s** |

### 4.3 Ablation study（消融实验）

#### 4.3.1 Tree Attention（树注意力）

与 SpecInfer 和 Medusa 类似，EAGLE 使用 Tree Attention，因此草稿生成和草稿验证都采用树结构。相比之下，传统 Speculative Sampling 等方法不使用 Tree Attention，其 Draft 和 Verification 都是链状结构。

Table 5 和 Figure 7 对比了使用与不使用 Tree Attention 的效果。EAGLE 引入树状草稿和树状验证后，平均接受长度大约增加 0.6–0.8，加速比大约增加 0.3–0.5。

与链状草稿和链状验证相比，树状草稿和树状验证没有增加模型前向传播次数（无论 Target LLM 还是 Draft Model），但增加了每次 Forward 处理的 Token 数。因此，加速比的提升小于平均接受长度的提升。

值得注意的是，即使完全不使用树状草稿和验证，EAGLE 仍能获得约 2.3×–2.7× 的显著加速。

### Figure 7：Tree Attention 的加速效果

[原始 Figure 7，PDF 第 7 页](https://arxiv.org/pdf/2401.15077v3#page=7)。

**图注完整翻译：** 图 7：EAGLE 使用 Tree Attention 与不使用 Tree Attention 的加速比。评测数据集为 MT-bench，Temperature=0。

| Target | Tree | No tree | 绝对加速比差值 |
| --- | ---: | ---: | ---: |
| Vicuna 7B | 2.90× | 2.39× | +0.51 |
| Vicuna 13B | 3.07× | 2.56× | +0.51 |
| Vicuna 33B | 2.95× | 2.42× | +0.53 |
| LLaMA2-Chat 7B | 2.78× | 2.27× | +0.51 |
| LLaMA2-Chat 13B | 3.03× | 2.64× | +0.39 |
| LLaMA2-Chat 70B | 3.01× | 2.66× | +0.35 |

以上差值依据 Figure 7 已标注的数字计算，用于解释原图，非作者另列的表格。

### Table 5：使用 Tree Attention 前后的平均接受长度

[原始 Table 5，PDF 第 7 页](https://arxiv.org/pdf/2401.15077v3#page=7)。

**表注完整翻译：** 表 5：EAGLE 使用和不使用 Tree Attention 时的平均接受长度 $\tau$。在 MT-bench 上评测，Temperature=0。

| 系列 | Model | Chain $\tau$ | Tree $\tau$ | 增量 |
| --- | --- | ---: | ---: | ---: |
| Vicuna | 7B | 3.20 | 3.94 | +0.74 |
| Vicuna | 13B | 3.23 | 3.98 | +0.75 |
| Vicuna | 33B | 2.97 | 3.68 | +0.71 |
| LLaMA2-Chat | 7B | 3.00 | 3.62 | +0.62 |
| LLaMA2-Chat | 13B | 3.18 | 3.90 | +0.68 |
| LLaMA2-Chat | 70B | 3.12 | 3.81 | +0.69 |

#### 4.3.2 Inputs of draft models（草稿模型的输入）

与其他基于推测采样的方法相比，EAGLE 的关键创新是使用目标 LLM 计算的 Feature，并将采样结果纳入 Draft Model 输入，以应对随机性。

我们以 Vicuna 7B 为目标模型，采用不同的输入形式进行消融实验，共测试四种：feature&shifted-token（EAGLE）、feature&unshifted-token、token、feature。

feature&shifted-token（EAGLE）和 feature&unshifted-token 都融合了不同层次的语义信息。区别在于，前者输入的是提前一个时间步的 Token，因此能够有效处理随机采样带来的不确定性。

除 Feature+Token 模型需要增加一个 FC 以降低输入维度之外，Draft Model 的结构保持完全一致。

Figure 8 展示了在 MT-bench 上使用 Vicuna 7B 作为 Target 的实验结果。可以得到三个观察：

**第一，**在 Draft Model 参数量有限时，使用 Feature 比仅使用 Token 能得到略好的效果。

**第二，**融合 Feature 与 Token 能适度提升性能，主要原因在于离散、无误差的 Token 有助于减轻 Feature 误差累积。证据是 feature&unshifted-token 与 feature-only 的 $0\text{-}\alpha$ 接近，但前者的 $1\text{-}\alpha$ 明显改善。

**第三，**解决采样过程固有的随机性带来最大提升。与 feature&unshifted-token 相比，feature&shifted-token 方案并未增加模型复杂度，只需将 Token 前移一个时间步，就能让 Draft Model 感知采样随机性，从而显著改善能力。

### Figure 8：四种草稿输入的完整消融曲线

[原始 Figure 8，PDF 第 8 页](https://arxiv.org/pdf/2401.15077v3#page=8)。

**图注完整翻译：** 图 8：使用不同输入的 Draft Model 的性能。目标 LLM 为 Vicuna 7B，测试数据集为 MT-bench。Speed 指实际耗时加速比，$\tau$ 指平均接受长度，$0\text{-}\alpha$ 表示输入完全准确时的接受率，$1\text{-}\alpha$ 表示输入包含一个不准确 Feature 时的接受率，T 表示 Temperature。

图包含上下两排 T=0、T=1，横向四列 Speedup、$\tau$、$0\text{-}\alpha$、$1\text{-}\alpha$；每条曲线跨 Epoch 1–7。颜色/系列：蓝线 feature&shifted-token，橙线 feature&unshifted-token，绿线 token，红线 feature。**原图没有逐 Epoch 的完整数值表，因此章节不伪造精确折线点。** 定性排序：蓝线在两种温度、四项指标均整体占优；橙线较红线在 $1\text{-}\alpha$ 上改善尤为明显；绿线总体最差。

#### 4.3.3 Training data（训练数据）

EAGLE 使用固定数据集训练，避免使用 Target LLM 生成训练数据造成的额外开销。Table 6 的消融显示，Target LLM 生成的数据只能带来有限的性能改善，说明 EAGLE 对训练数据并不敏感，因此选择固定数据集有助于降低成本。

### Table 6：两种训练数据的比较

[原始 Table 6，PDF 第 7 页](https://arxiv.org/pdf/2401.15077v3#page=7)。

**表注完整翻译：** 表 6：使用不同训练数据时的加速比及平均接受长度 $\tau$。在 MT-bench 上评测，Target LLM 为 LLaMA2-Chat 7B、Temperature=0。“固定数据集（Fixed Dataset）”指问题与答案都来自 ShareGPT；“由目标 LLM 生成的数据”指问题来自 ShareGPT，而答案由目标 LLM 生成。

| 训练数据 | Speedup | $\tau$ |
| --- | ---: | ---: |
| 固定数据集（ShareGPT 问题+答案） | 2.78× | 3.62 |
| 目标 LLM 生成答案（ShareGPT 问题） | 2.88× | 3.75 |

### 4.4 Batch size and throughput（批大小与吞吐量）

**原文第 1 段。** LLM 推理是 Memory-bound（内存带宽受限）的（Patterson, 2004；Shazeer, 2019），因此 GPU 计算资源通常未得到充分利用。基于推测采样的方法之所以能够提高生成速度，是因为它能更有效地利用 GPU 计算资源。Batch Size 增加时，GPU 可用的计算资源减少，因此加速效果也会减弱。本节展示 Batch Size 大于 1 时的结果。

如 Table 7 所示，加速比通常随 Batch Size 增加而下降。对于 Vicuna 7B，bs=4 的加速比高于 bs=3。这是因为在 EAGLE 的验证阶段，Target LLM 一次前向传播处理多个 Token，而 bs=4 下的该处理反而快于 bs=3。相比之下，普通自回归解码每次 Forward 只处理一个 Token，因此 bs=3 和 bs=4 时速度几乎相同。

**原文第 2 段。** 尽管推测采样主要关注延迟，我们也调查了 Batch Size 大于 1 的吞吐量，这是 LLM 系统的另一项重要指标。

与普通自回归解码相比，EAGLE 需要稍多一些 CUDA 显存。对 Vicuna 7B，在单张 RTX 3090（24GB CUDA 显存）限制下，普通自回归解码最大 Batch Size 为 8，而 EAGLE 为 7。对于 LLaMA2-Chat 70B，在 4 张 A100（40GB）、总共 160GB CUDA 显存的限制下，普通解码最大 Batch Size 为 5，而 EAGLE 为 4。所有评估均在 FP16 精度下进行。

我们计算不同 Batch Size 下的吞吐量并选择其中最大值。普通自回归与 EAGLE 均在各自支持的最大 Batch Size 下获得最大吞吐量。Tree Attention 消耗更多计算资源；当 bs=7 时可用计算资源较少，此时不使用 Tree Attention 反而更加有利。如 Table 7 所示，EAGLE 实现了约两倍吞吐量提升。

### Table 7：Batch Size 和吞吐量

[原始 Table 7，PDF 第 8 页](https://arxiv.org/pdf/2401.15077v3#page=8)。

**表注完整翻译：** 表 7：EAGLE 在不同 Batch Size 下的加速比，以及其吞吐量表现。在 MT-bench 上评测，Temperature=0。

| Target | bs=1 | bs=2 | bs=3 | bs=4 | Throughput 相对提升 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Vicuna 7B | 2.90× | 2.87× | 2.65× | 2.76× | 1.97× |
| LLaMA2-Chat 70B | 3.01× | 2.81× | 2.50× | 2.40× | 1.99× |

**注意：** Table 7 最后一列不是绝对 tokens/s，也不是 bs=4 对应的吞吐数据，而是在可承受 Batch 范围内，分别选取 Vanilla 和 EAGLE 的最大吞吐量后得到的相对加速。不能将最后一列乘某个未经测量的 TPS 当成论文绝对性能。

**Section 4.4 完成；之后原文进入 5. Related Work。**

---

## 二、深入技术分析（独立补充；与上述译文分离）

### 1. 关键指标及加速模型

目标每次 Forward 平均推进 $\tau$ 个 Token，Draft 花费 $T_d$，Target 验证花费 $T_v$，其他调度/Cache/采样开销为 $T_o$。设普通解码每 Token $T_{\rm base}$：

\[
S\approx\frac{\tau T_{\rm base}}{T_d+T_v+T_o}.
\]

这是解释性的成本模型，不是作者在 Section 4 中报告的拟合公式。$\tau$ 增大不能保证 $S$ 等比例增大，因为更宽的树会抬高 Token 批量验证成本。

$0\text{-}\alpha$ 与 $1\text{-}\alpha$ 等指标是**基于链状草稿**定义的条件接受率：前者输入没有不准确的 Feature；后者输入已经包含一个 Draft 预测的、可能有偏差的 Feature。不能将 $0\text{-}\alpha$ 简化为“没有采样误差”，也不能把它直接等同于整棵树的平均 Token 接受率。

### 2. Table 1–2：Temperature 和任务差异

例如 LLaMA2-Chat 13B / HumanEval：T=0 下 Speedup=3.76×、$\tau=4.52$；T=1 下 Speedup=2.89×、$\tau=3.78$。Temperature=1 增加不确定性，Draft 与 Target 的采样分布不一定更重合，导致平均接受长度降低。**这是符合作者报告的现象及推测采样原理的解释，不是对所有数据集与温度的单调性定理**。

代码任务具有重复语法与模板，因此在作者实验条件下更容易获得较高的接受长度。应注意 Table 1 是各任务上针对特定模型、温度、推测草稿结构测得的 Speedup，不代表代码能力或精确率的提升；算法被验证的是采样分布不变。

Table 2 中 Vicuna 7B 在 T=0 下，从 $0\text{-}\alpha=0.79$ 降为 $1\text{-}\alpha=0.74$，再到 $4\text{-}\alpha=0.67$。大幅变化集中在加入第一个不准确 Feature 时；后续降幅整体相对有限，支持作者“对误差累积有一定稳健性”的结论，但不是说 Feature 误差可以完全忽略。

### 3. Tree Attention：接受长度增量与速度增量为何不同

以 Vicuna 7B 为例，Chain $\tau=3.20$，Tree $\tau=3.94$，接受长度绝对增加 0.74，相对增加约 $0.74/3.20=23.1\%$；但加速比仅从 2.39× 到 2.90×，绝对增加 0.51、相对增加约 21.3%。其他模型也可见这一不等价关系。原因是 Tree 使一次 Forward 计算更多位置，增加计算量/显存流量；草稿树更宽时 GPU 有限资源会更紧张。

从增量速度计算：论文 Figure 7 的树化相对于链化的实际吞吐倍数为 2.90/2.39≈1.213（Vicuna 7B）；不能将 0.51 误述为“提升 51%”。

### 4. Figure 8：四种输入究竟控制了什么

| Draft 输入 | 是否含 Target Feature | 是否含 Token | 是否有采样后继条件 | 研究问题 |
| --- | --- | --- | --- | --- |
| token | 否 | 是 | 无 Feature-to-next-Feature 的配对 | 直接 Token 级学习 |
| feature | 是 | 否 | 否 | 特征序列规律性是否足够 |
| feature&unshifted-token | 是 | 是 | 否（Token 未前移） | Token 有助减轻特征误差累积吗 |
| feature&shifted-token | 是 | 是 | **是** | 已采样 Token 能否消除 Feature 的多后继歧义 |

原图 T=0 和 T=1 上，蓝色 Shifted Token 曲线的 Speedup、$\tau$、$0\text{-}\alpha$、$1\text{-}\alpha$ 均高于其他输入的曲线。尤其橙色 Unshifted Token 相比红色 Feature Only，$0\text{-}\alpha$ 提升较小，但 $1\text{-}\alpha$ 提升明显：这支持“Token 条件对错误 Feature 的稳健性有效”；蓝色继续大幅领先橙色，支持“消除采样分支不确定性”是主要贡献。

不要把曲线末端的目测点精度当成原文的精确表格；论文没有逐 Epoch 数据表。

### 5. Table 6：固定数据与在线生成数据

使用 Target 自己生成 ShareGPT 回答，把 Speedup 从 2.78× 提升至 2.88×，**绝对加速比差** +0.10，相对推理速度改善约 $(2.88/2.78-1)\times100\%=3.60\%$。平均接受长度从 3.62 提升 3.75，绝对提升 0.13。如此改善不足以自动证明额外生成训练数据的成本划算；取决于训练规模、服务请求量和可复用生命周期。论文的结论仅对所测条件成立。

### 6. Mixtral / MoE：为何 $\tau=3.25$ 仍只有 1.50×

Dense Target 进行验证时，一次处理多个位置往往可更好地摊薄模型权重加载成本，因为全部层参数总要读取。对于 Top-2 MoE，单 Token 常只激活两个 Expert；多个不同候选 Token 可能路由到更多 Expert，导致整批访问更多 Expert 权重，抵消树验证的节省。这解释了 MoE 与 Dense Transformer 的硬件收益差异。确切机制还取决于 Expert Cache、路由模式、batching、内存带宽和 GPU Kernel，论文没有给出 Expert 访问数量的逐层测量；不应将这一解释当作完整的微观性能剖析。

### 7. gpt-fast 与 EAGLE：为何可以叠加？

Table 4 中 FP16：gpt-fast 55.1→EAGLE+gpt-fast 100.2 tokens/s，附加加速比约 1.82×。int4：106.9→160.4 tokens/s，附加加速比约 1.50×。二者不是可以无损简单相乘的固定因子，因为量化/编译先改变 Target 和 Draft Forward 的成本，进而改变 Speculative Decoding 的整体平衡。

需要区分 160.4 tokens/s 是单张 RTX 3090、LLaMA2-Chat 7B、MT-bench、T=0、gpt-fast int4 的实验结果，不是 EAGLE 单独在任何模型或显卡上的固定吞吐。

### 8. Batch Size 与服务系统：性能瓶颈转移

Table 7：LLaMA2-Chat 70B 的单 Batch 延迟加速 3.01×，bs=4 只有 2.40×。批大小增加后 GPU 计算资源逐渐占满，一次验证多个 Token 的边际收益下降；更宽草稿树消耗额外算力。

同时，EAGLE 的最大可运行 Batch Size 在实验中比 Vanilla 略低（Vicuna 7B：7 vs 8；LLaMA2-Chat 70B：4 vs 5）。因此作者为了比较最大可承载吞吐量，分别为基线与 EAGLE 选择其可达到的最高吞吐，并得到 1.97×、1.99× 的相对吞吐改善。**这不同于固定 bs=1 的延迟加速，也不能当成在任意并发、任意 KV 长度下的吞吐承诺。**

对现代 vLLM/SGLang Continuous Batching，建议另测 TTFT、TPOT、End-to-end latency、TPS、不同 Prompt/Output 长度、并发和 KV 显存占用；论文原文没有提供这类 Serving 曲线。

### 9. 与已读 DFlash 的关联

EAGLE 通过 Feature+Shifted Token 和 Tree Attention 得到较高 $\tau$，但草稿依旧要随着树深度进行多个顺序 Forward。此前精读的 [DFlash](../2602.06036-dflash/README.md) 用 Block Diffusion 并行产生草稿，结合 Target 多层 Feature 与 KV Injection 增加候选质量。两种方法都体现 $S$ 受 Draft 延迟、Verification 成本、接受长度共同约束。DFlash 原文对 EAGLE-3 的某些比较不能当作与**本篇 2024 EAGLE**的同条件横向 benchmark。

工程复现：建议固定目标模型、推理后端、硬件、Temperature、Prompt/Output Length、并发配置，在 EAGLE 上重做 Chain vs Tree、四种 Draft 输入、固定数据 vs Target 生成数据消融，并记录每步草稿时间、验证时间、Target accepted tokens、总体 TPS 与额外显存。

## 阅读断点

- 已完成 Abstract、1. Introduction、2. Preliminaries、3. EAGLE、**4. Experiments（4.1–4.4，含 4.3.1–4.3.3）**。
- Table 1–7 全部原始数据与完整图注翻译已纳入文件；Figure 7/8 已在本次聊天从原 PDF 裁图展示，Markdown 中使用原 PDF 精确页码定位。图像二进制尚未上传 GitHub。
- **下一单元：5. Related Work**，PDF 第 8–9 页。结论 Section 6 与 Appendix A/B 依原论文顺序后续阅读。
