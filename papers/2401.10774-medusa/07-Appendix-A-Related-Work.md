# Medusa Appendix A Related Work：全文中文翻译、推理加速分类与跨论文关联

**原始论文：** Tianle Cai et al., Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads, arXiv:2401.10774v3, ICML 2024。

**严格原文范围**：PDF 第 13–14 页，Appendix A.1 LLM Inference Acceleration（4 个自然段）与 Appendix A.2 Sampling Scheme（3 个自然段），**总计 7 段**，其中 A.2 第二段跨页。下一节 Appendix B. Experiment Settings 从 PDF 第 14 页中部开始，**未在此阅读单元翻译**。

- [原始 PDF 第 13 页](https://arxiv.org/pdf/2401.10774v3#page=13)
- [原始 PDF 第 14 页](https://arxiv.org/pdf/2401.10774v3#page=14)
- [arXiv v3 HTML 版本](https://arxiv.org/html/2401.10774v3)

本附录**没有 Figure、Table 或正式编号公式**。本文先保留原文顺序和全部引用，完成中文翻译；之后另列「技术讲解／外部资料」，避免将后来的 EAGLE（2024）、DFlash（2026）的工作错误地写进 Medusa 原文。

## 一、A.1 LLM Inference Acceleration（四段全文翻译）

### 第一自然段：LLM 推理加速的两类技术路径

大型语言模型（Large Language Model，LLM）推理效率低下，主要归因于自回归解码（Auto-regressive Decoding）过程中受到内存带宽限制（Memory-Bandwidth-Bound）的特点。研究者已经提出多种缓解方法，以改善推理延迟和吞吐量。传统上，批量推理（Batch Inference）是一种直接提高算术强度（Arithmetic Intensity）、摆脱内存带宽限制的手段。然而，对于 LLM 来说，模型参数和键值缓存（Key-Value Cache，KV Cache）都需要消耗大量加速器显存，妨碍了大 Batch Size 的使用。解决这些问题的方法从概念上可以分成两类：（1）减少内存消耗，降低内存传输开销、允许更大的 Batch Size；（2）减少解码步骤数量，直接降低延迟。

### 第二自然段：Reducing KV Cache

Multi-query Attention（Shazeer, 2019）和 Grouped-query Attention（Ainslie et al., 2023）等方法直接减少 KV Cache。通过使注意力模块中 Key/Value Head 的数量少于 Query Head，这些方法大幅降低 KV 内存消耗，从而支持更大的 Batch Size、提高加速器利用率（Pope et al., 2022）。此外，Zhang et al.（2023）提出选择性保留最重要的 KV Token，进一步减少 KV Cache。从系统角度，Kwon et al.（2023）提出分页式内存管理（Paged Memory Management）方案，以减少 KV Cache 内存碎片。

### 第三自然段：Quantization

量化技术被广泛用于缩减 LLM 的内存消耗。Xiao et al.（2023a）通过在激活值与参数之间重新缩放（Rescaling），消除异常值（Outliers）的影响并简化量化过程。Dettmers et al.（2022）将矩阵乘法分解为以 8-bit 为主、少量使用 16-bit 的操作。Frantar et al.（2022）迭代地将权重矩阵列舍入到 3/4-bit。Lin et al.（2023）提出激活感知量化（Activation-Aware Quantization）方案，保护重要权重，将 LLM 压缩到 3/4-bit。Kim et al.（2023）提出结合稀疏与低精度表示的模式，以处理少量重要权重，除此之外也存在其他量化方案。

### 第四自然段：Speculative Decoding

作为与前述方法正交（Orthogonal）的技术路线，推测解码（Leviathan et al., 2022; Chen et al., 2023）旨在并行执行多个解码步骤，从而减少总步骤数量。其并行性来自一个较小的 Draft Model：该模型猜测后续多个词，然后由原始 LLM 集中评估并接受合适候选。这与非自回归生成领域（Xiao et al., 2023b）的工作具有共通思想，但专门针对 LLM 的上述效率问题设计。不同于以往工作，我们提议直接使用原始模型进行预测，而不是引入额外的 Draft Model。这种方法更直接，也免去了同时管理两套模型的复杂性，因而容易集成。

独立地，Miao et al.（2023）和 Spector & Re（2023）也提出以 Tree-Structured Attention 并行处理多个候选。其中 Miao et al.（2023）建议使用多个模型组成的集成来提供候选；Spector & Re（2023）主张为草稿模型再增加一个层级。然而草稿模型需要专门预训练并与目标模型对齐。维护多个草稿模型较为繁琐，涉及并行管理复杂性；相比之下，仅依赖 Decoding Heads 是更简单的替代方案。Miao et al.（2023）使用多个 Draft Model 生成 Token，并通过 Tree Attention 合并；Spector & Re（2023）使用小 Draft Model，按批次处理树的每一层。与这些做法不同，我们直接使用每个 Medusa Head 的最高概率候选，构建**静态稀疏树**；候选生成无需自回归，也不需要在推理时反复调整树拓扑。这简化了过程并提高效率。此外，我们还通过详细的消融实验展示，树节点的选择会怎样影响解码速度。

**翻译边界提醒**：以上 Miao et al.（2023）、Spector & Re（2023）的描述来自 Medusa 作者原文，本节未将之后发表的 EAGLE、DFlash 强行插入原文。

## 二、A.2 Sampling Scheme（三段全文翻译）

### 第一自然段：采样质量问题与截断采样

从大型语言模型中采样文本的方法可以显著影响生成输出质量。近期研究发现，直接从语言模型采样可能产生不连贯甚至荒谬的结果（Pillutla et al., 2021; Holtzman et al., 2020）。为解决这一问题，研究者提出截断采样方法（Fan et al., 2018; Basu et al., 2021; Meister et al., 2022; Hewitt et al., 2022; Meister et al., 2023）。这些方法在每次解码时先确定一个允许集合（Allowed Set），再在该集合对应的截断分布中采样，从而同时追求较高质量与多样性。

### 第二自然段：Top-k / Top-p / Typical Decoding（跨第 13–14 页）

不同策略使用不同方式定义允许集合。例如，Top-k Sampling（Fan et al., 2018）保留概率最高的 k 个词；Top-p Sampling（Holtzman et al., 2020）则选择累计概率达到 p 所要求比例的最小词集合。另一种方法 Typical Decoding（Meister et al., 2023）使用预测概率分布的熵来设定允许范围的阈值。Hewitt et al.（2022）提出了一个统一框架，用于全面理解截断采样技术。

### 第三自然段：Medusa 的 Typical Acceptance

受到这些方法启发，我们提出的 Typical Acceptance 也使用「定义允许集合、排除不可能的候选」的思想。然而，我们的方法与严格的截断采样有所不同，因为我们不坚持输出严格匹配语言模型的概率分布。这种偏离让我们能够得到更具多样性、同时保持高质量的结果，提升效率而不破坏生成文本的合理性。

**Appendix A 两个小节全部七段至此译完。** 原文 A.2 在第 14 页结束，之后立即进入 Appendix B Experiment Settings。

## 三、独立讲解：内存带宽瓶颈与 KV Cache / 量化的工程含义

### 3.1 算术强度与 Roofline

一般形式：
$$
I=\frac{\mathrm{FLOPs}}{\mathrm{Bytes\ transferred}},\qquad
P_{\mathrm{achieved}}\le \min(P_{\mathrm{peak}},\,B_{\mathrm{HBM}}I).
$$
小 Batch 自回归 Decode 的矩阵/向量乘法对模型权重复用有限，加载大量权重但每次只生成少量 Token，很容易 Memory-Bound。提高 Batch、减少权重/缓存传输、或每轮计算多个候选均可能改善效率；但也可能使系统转为 Compute-Bound，实际收益需测量。

### 3.2 KV Cache 的主要项

对于相同精度的标准 Transformer，可用近似式：
$$
M_{\mathrm{KV}}\simeq 2BLSH_{\mathrm{KV}}d_{\mathrm{head}}b.
$$
其中 B=Batch，L=层数，S=实际保留的缓存序列长度，$H_{\mathrm{KV}}$=每层 Key/Value Heads 数，$d_{\mathrm{head}}$=每个 Head 的维度，b=每元素字节数。系数 2 来自 Key 与 Value。

MQA：所有 Q Heads 共享 1 个 K/V Head；GQA：若干 Q Heads 共享一个 K/V Head；MHA：Q/K/V Head 数相同。假设 32 Q Heads 与 8 KV Heads，其他条件相同，GQA 相比 32 KV Heads 的理论 KV 缓存节省比例是 $1-8/32=75\%$，**并非整个模型显存直接节省 75%**。

H2O（Zhang et al., 2023）选择性保留对 Attention 重要的 Heavy-Hitter Token，实际属于丢弃部分旧 KV 内容的方案，可能引入近似误差。PagedAttention（Kwon et al., 2023，vLLM）主要管理非连续 KV 物理内存、降低碎片和空闲浪费；它本身不意味着保留的每个 Token KV 张量元素数减少。两者降低显存使用的机制不同。

### 3.3 量化路线与 KV 压缩的不同对象

- Xiao et al.（2023a）对应 SmoothQuant，重缩放降低 Activation Outliers 对量化的难度。
- Dettmers et al.（2022）对应 LLM.int8()，大多数计算以 8-bit 路径处理，少量异常值由高精度路径处理。
- Frantar et al.（2022）对应 GPTQ，训练后量化到少比特权重。
- Lin et al.（2023）对应 AWQ，利用激活重要性保护部分关键权重。
- Kim et al.（2023）对应 SqueezeLLM，结合稠密低精度量化与重要权重的稀疏存储（该方法有 2023 预印本和 2024 会议版）。

忽略 Scale、Zero Point 与元数据时，P 个参数以 q bit 存储约为 $Pq/8$ 字节，例如 BF16→INT4 的纯权重理论大小是四分之一。量化常主要作用于模型权重；MQA/GQA 改变 KV 几何结构；H2O 改变保留的 KV Token；PagedAttention 改变 KV 内存分配。这些技术与 Medusa 在概念上可叠加，但实际 Decode Kernel 与计算瓶颈相互影响，速度增益并不简单相乘。

## 四、独立讲解：Speculative Decoding 的验证算法与概率保证

假设 Target 分布 p(x)，实际 Draft 提议分布 q(x)，在**给定完全相同已接受前缀**的条件下，从 q 抽候选 $x$，然后以
$$
a(x)=\min\left(1,\frac{p(x)}{q(x)}\right)
$$
接受。若拒绝，则必须从
$$
r(x)=\frac{[p(x)-q(x)]_+}{\sum_y[p(y)-q(y)]_+}
$$
抽样。这样每个生成位置保持目标 p，而不是仅判断“Target 有没有抽到同一 Token”。标准单候选平均接受率：
$$
\alpha=\sum_xq(x)a(x)=\sum_x\min(p(x),q(x))
=1-\frac12\sum_x|p(x)-q(x)|
=1-\operatorname{TV}(p,q).
$$
例如 $p=(0.45,0.30,0.15,0.10)$，$q=(0.40,0.35,0.15,0.10)$，则总变差为 0.05，平均接受率 0.95；若 p=q，则接受率**严格等于 1**。这也与此前 Medusa Section 2.3.1 作者对“独立采样导致即使 p=q 也可拒绝”的直觉陈述形成校核：作者的直觉不适用于正确实现的标准接受比与残差修正算法。不能在翻译中默改作者论述，独立分析则应指出数学差异。

Draft 的多个未来 Token 可由小模型**串行**产生，而 Target 借助 Causal/Tree Attention 一次评估候选路径不同前缀的概率；验证是否接受仍沿同一候选路径顺序进行。这是“草稿可串行、验证可并行”两个不同的并行概念。小模型的时间成本不可忽略，尤其在 Target 更快或 Batch 较大时。

## 五、独立讲解：树状推测的不同来源

| 路线 | 候选来源 | 草稿内部计算 | Target 验证 | 额外模型权重 |
| --- | --- | --- | --- | --- |
| 传统 Speculative Decoding (Leviathan/Chen) | 单个小 Draft Model | 多轮 AR Draft | 一次并行因果验证 | 独立小模型 |
| SpecInfer (Miao et al., 2023) | 多个不同的小 Draft Models 组成集成 | 多模型生成、合并候选 Token Tree | Token Tree 并行验证 | 多个 Draft |
| Staged Speculative Decoding (Spector & Ré, 2023) | 增加草稿级别、逐层批量生成 | 分阶段和按树层处理 | 树式验证 | 额外 Draft 层级 |
| Medusa (本论文) | Target $h_t$ 上的多个未来 Token Heads | Heads 可并行预测不同偏移 | Static Sparse Tree Attention | 无独立 Draft Transformer；额外 Heads 权重 |
| Self-Speculative / Layer Skipping（外部补充） | 使用同一大模型的部分层 | 部分层 Forward 的近似预测 | 完整 Target 层验证 | 通常无独立小模型 |
| Lookahead Decoding（外部补充） | Jacobi 迭代 / n-gram 匹配 | 迭代并行位置猜测 | 验证可接受前缀 | 无独立小 Draft LLM |

Medusa Appendix A.1 **只直接讨论前四行**，Self-Speculative / Lookahead 是为了回答“无独立草稿模型加速”的用户关注而补充，不能声称这两个名字在该附录原文出现。

**Static Sparse Tree 的限定含义**：可以先通过校准集的每 Head Rank Accuracy 搜索优化树（见 Section 2.3.3），但推理中各次生成共享预先选择的静态拓扑，而不是每步靠小 Draft Model 自回归生成并在线调整树层。静态不意味着没有拓扑优化；也不能把“无独立 Draft Transformer”误说成“零额外参数”。

Medusa Head k 基于同一 $h_t$ 预测第 $t+k+1$ 个 Token，**不知道真实中间 Token**；Tree Verification 则必须使用每条路径真实候选前缀并正确设置祖先 Mask 与 Position IDs。SpecInfer/Staged 的小 Draft 可通过串行条件生成获得更准确的未来条件，但需要付出草稿计算和模型管理成本。

## 六、独立讲解：Top-k、Top-p、Typical Decoding、Typical Acceptance 的数学区别

### 6.1 截断分布的统一形式

给定模型预测分布 $p(x)$ 和所选允许集合 $A$：
$$
p_A(x)=\frac{p(x)\mathbf{1}[x\in A]}{\sum_{y\in A}p(y)}.
$$
**Top-k**：集合 $A$ 为概率最大的 k 个 Token；**Top-p/Nucleus**：按概率排序，选择累计质量达到 p 的最小前缀集合。

例子 $p=(0.45,0.30,0.15,0.10)$：
- Top-2：$A=\{1,2\}$，归一化后为 $(0.60,0.40,0,0)$。
- Top-p=0.8：前两项总质量 0.75 不足，必须包含第三项达到 0.90；归一化后为 $(0.50,0.3333,0.1667,0)$。

### 6.2 Local Typical Decoding 不等于 Medusa 的 Typical Acceptance

普通 Local Typical Sampling 可以根据 Token 的 Self-Information $s(x)=-\log p(x)$ 与分布熵 $H(p)$ 的偏差 $|s(x)-H(p)|$ 来排序/选择允许 Token，然后在截断分布上重新归一化采样。论文 Appendix A.2 仅概括为“利用熵定义阈值”，没有给出其详细选择公式。

Medusa 在 Section 2.3.1 的实际 Typical Acceptance 是：
$$
\mathrm{accept}(x)\iff
p_{\mathrm{Target}}(x\mid\text{branch prefix})>
\min\left(\epsilon,\delta\,e^{-H(p_{\mathrm{Target}})}\right).
$$
这是对**已提出的候选 Token**进行判定，而不是直接从重新归一化的 $p_A$ 按概率采样。它还要在多条候选树路径中选择最长可接受前缀，因此最终输出分布一般不等于原始 p，也不必等于简单截断分布 $p_A$。**“典型”在两种方案中相关，但算法并不相同。**

### 6.3 何时使用哪种方案

若应用要求可复现的模型采样分布、概率一致性或与原 Target 精确可比，应优先采用正确的 Speculative Sampling 和残差修正，不能用 MT-Bench 分数相近替代严格分布保证。若主要目标是低延迟、允许生成分布变化，Typical Acceptance 可作为速度与输出质量的可调权衡，但必须在真实数据集上测试幻觉、格式、代码正确性和敏感任务风险。

## 七、与 EAGLE 和 DFlash 的跨论文联系（非 Medusa 原文）

| 关键维度 | Medusa (2401.10774) | EAGLE (2401.15077) | DFlash (2602.06036) |
| --- | --- | --- | --- |
| Draft 预测对象 | 未来 Token | 未来倒数第二层 Hidden Feature | 并行 Block Tokens |
| Draft 的核心条件 | 单个当前 $h_t$ | 自回归 Feature + Shifted Token | 多层 Target Feature + KV Injection |
| Draft 的串行深度 | 多个 Heads 可并行 | 多轮小 Draft Forward | 一次 Block Diffusion Forward |
| Draft 额外网络 | K 个浅层 Heads | 独立轻量 Feature Draft Adapter | 独立轻量 Block Diffusion Adapter |
| 验证形式 | Static Sparse Tree Attention | 动态多分支树 Target Verification | 并行 Target Block Verification |
| 原始分布保证 | Typical Acceptance 不保持严格分布；可选精确验证需正确提议修正 | 论文采用精确 Speculative Sampling | 论文采用正确 Target Verification 下的无损加速 |
| 研究关注 | 降低 Draft 生成复杂性、树预算/接受阈值 | 提高条件特征预测准确率，降低不确定性 | 并行生成全草稿块并增强目标上下文条件 |

**EAGLE 的差异（原论文核对）**：EAGLE 论文指出 Feature-level Autoregression 相比直接 Token Autoregression 更规整，并用 Shifted Token（提前一位置的采样 Token 信息）缓解未来 Feature 的条件不确定性；与 Medusa 共享当前隐藏状态分别猜未来 Token 不同。EAGLE 仍需多次草稿前向以生成足够深的树，且在正确概率修正下严格保持 Target 采样分布。[EAGLE ICML 官方页面](https://proceedings.mlr.press/v235/li24bt.html)，[项目内已完成笔记](../2401.15077-eagle/README.md)。

**DFlash 的差异（外部 2026 年研究）**：DFlash 的轻量 Block Diffusion Adapter 通过目标模型的多层特征获得条件信息，利用 KV Injection 减少目标信息在多层 Draft 内的稀释，旨在**一次 Forward 并行预测整个草稿块**。因此它既不使用 Medusa 的多个独立未来 Token Head，也不采用 EAGLE 逐步的 Feature Draft 串行链。[DFlash ICML 官方页面](https://proceedings.mlr.press/v306/chen26bv.html)，[项目内已完成笔记](../2602.06036-dflash/README.md)。

**历史关系**：Medusa v3 是 2024 年论文，而 DFlash 是 2026 年后续研究。Medusa Appendix A 不会也不可能将 DFlash 列为自己当时的 Related Work；本表是独立的跨论文技术对照。EAGLE 与 Medusa 都发表于 ICML 2024，比较它们的成果应以具体论文版本和实验条件为准。

### 统一的性能分析指标

使用一轮验证平均可推进 $A$ 个 Token、验证轮次耗时相对普通 AR 为 $O$：
$$
\mathrm{Speedup}\simeq\frac{A}{O}.
$$
在此基础上，各方法尝试分别降低：
1. Draft 时间与额外参数（Medusa 的多头）；
2. Draft Token 条件预测错误率（EAGLE 的 Feature + Shifted Token）；
3. Draft 串行深度（DFlash 的 Block Diffusion）；
4. Target Verification 冗余计算（多方法可结合 Tree Attention、合适树深/块长）。

但 A 与 O 都依赖模型、采样温度、GPU、Batch Size、KV Cache 和 Serving Kernel，不能从不同论文的最高 Speedup 数字直接判断优劣。

## 八、阅读小结与进度

- 本次 **A.1 全四段、A.2 全三段**均按 v3 原文顺序完整译出，所有原文引用和技术方法得到保留，跨页段落未省略。
- 本节没有公式、原图或表格需要提取。数学式均是独立讲解，与论文原文明确分离。
- 核心知识：带宽/算力与 KV Cache/Quantization 两类优化；SpecInfer、Staged、Medusa 三种不同树式候选来源；Top-k/Top-p/Typical Decoding 与 Medusa Typical Acceptance 的概率差异；EAGLE 与 DFlash 的 Feature/Block Draft 技术演进。
- 下一阅读单元：**Appendix B Experiment Settings（B.1–B.4）**。Medusa 主文与 Appendix A 已完成；B–G 仍待逐节精读，**整篇论文未完成**。
