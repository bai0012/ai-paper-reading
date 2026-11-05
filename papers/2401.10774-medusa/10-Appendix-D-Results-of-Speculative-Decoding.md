# Appendix D · Results of Speculative Decoding

**论文**：[Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774)  
**版本**：arXiv:2401.10774v3（2024-06-14）  
**原始 PDF**：[第 15 页：Figure 7](https://arxiv.org/pdf/2401.10774v3#page=15)  
**HTML 原文**：[Appendix D](https://arxiv.org/html/2401.10774v3)  
**阅读范围**：Appendix D 的全部两段正文、Figure 7 三个子图和完整图注；结合正文 Section 3、Table 1、Appendix B 的通用设置，以及作者脚注指向的开源 Speculative Sampling 实现进行分析。  
**来源边界**：第一部分为 v3 原文完整翻译；图中数值为读图估计而非论文表格数据；机制解释、模型匹配原因和跨论文比较均属于独立分析。

## 一、Appendix D 原文完整翻译

### D. Results of Speculative Decoding（推测解码的实验结果）

**原文第一段**

在本研究中，我们将推测解码应用于不同规模的 Vicuna 模型（Chiang et al., 2023），具体包括 7B、13B 和 33B。

初步实验框架使用了多个开源模型，包括 Llama-68M 和 Llama-160M（Miao et al., 2023），以及 Tiny-Llama（Zhang et al., 2024）和 Tiny-Vicuna（Pan, 2023）。其中，Tiny-Vicuna 是以 Tiny-Llama 为基础，采用 Vicuna 风格的指令微调策略进行微调得到的模型。

由于 Chen et al. (2023) 和 Leviathan et al. (2022) 所提出的推测解码方法具有专有实现的性质，我们采用了开源替代实现进行评估。此外，我们使用 \`torch.compile()\` 来加速 Draft Model 的推理速度。

**原文第二段**

Figure 7 展示的实验结果表明，Draft Model 的最优配置会随着 Vicuna 模型规模的变化而变化。

具体而言：

- 对于 Vicuna-7B，使用 Llama-68M 作为 Draft Model，并将 Draft Token 数量设置为 \(\gamma=4\)，获得了最佳性能。
- 对于 Vicuna-13B，同一个 Draft Model 在 \(\gamma=3\) 时效果最好。
- 对于规模更大的 Vicuna-33B，使用 Tiny-Vicuna（Vicuna-1B）作为 Draft Model，并将 \(\gamma=3\)，获得了最大的加速效果。

这些结果表明，Draft Model 的选择及其参数配置应当根据 LLM 的规模进行调整。这也为推测解码领域提供了一个值得进一步探索的研究方向。

**原文翻译结束。** Appendix D 只有以上两段正文；后面直接进入 Figure 7 和 Appendix E，没有额外的公式、表格或第三段正文。

### Figure 7 完整图注翻译

**图 7：在 MT-Bench 上使用推测解码时，不同模型的推理速度。**

图中的灰色虚线表示原始模型（Baseline Model）的推理速度，用于对比。

\(\gamma\) 表示 Draft Token 的数量。

三个子图分别为：

- (a) Vicuna-7B；
- (b) Vicuna-13B；
- (c) Vicuna-33B。

图例中的四种 Draft Model 分别为 Llama-68M、Llama-160M、Llama-1B 和 Vicuna-1B。

**命名说明**：正文称后两种模型为 Tiny-Llama 和 Tiny-Vicuna；图例则简写为 Llama-1B 和 Vicuna-1B。正文明确将 Tiny-Vicuna 标为 Vicuna-1B。对于 Tiny-Llama，本文按正文名称称呼，并在表格中保留图例标签 Llama-1B；Appendix D 没有给出具体 checkpoint 字符串，不能据此补造仓库 ID。

## 二、Figure 7 原图与读图说明

**原图位置**：[arXiv v3 PDF 第 15 页](https://arxiv.org/pdf/2401.10774v3#page=15)。本节 Figure 7 是 PDF 第 15 页下半部分的三个并列折线图；图上横轴为 Draft Token 数量 \(\gamma\)，纵轴为 Tokens per Second，灰色水平虚线是各 Target 模型的原始解码速度。

每个子图都比较四条 Draft Model 曲线：

| 图例颜色 | 图例模型 | 正文对应名称 |
| --- | --- | --- |
| 蓝色 | Llama-68M | Llama-68M |
| 橙色 | Llama-160M | Llama-160M |
| 绿色 | Llama-1B | Tiny-Llama（正文名称） |
| 红色 | Vicuna-1B | Tiny-Vicuna（正文名称） |

横轴刻度标出 0、2、4、…、14；曲线数据点对应逐渐增加的 Draft Token 数量，图中覆盖约 \(\gamma=1\) 到 15。随着 \(\gamma\) 增大，多数曲线先达到峰值，随后逐步下降。

### 2.1 图中可直接读出的实验结论

1. Vicuna-7B：Llama-68M 曲线整体领先，正文明确给出最优点 \(\gamma=4\)。
2. Vicuna-13B：Llama-68M 仍然领先，正文明确给出最优点 \(\gamma=3\)。
3. Vicuna-33B：红色 Vicuna-1B（正文 Tiny-Vicuna）曲线在低 \(\gamma\) 区域达到最高，正文明确给出 \(\gamma=3\)。
4. 三个子图中的最佳点都位于较小的 \(\gamma\) 区域；继续增加草稿长度并不会持续提高吞吐。

### 2.2 读图速度数据：近似峰值，不是原文表格

论文没有为 Figure 7 提供逐点数值表。下面是根据 PDF 原图折线位置进行的近似读数，只用于比较不同 Draft Model 的相对表现，不能当成作者报告的精确 Benchmark 数据。不同曲线的峰顶可能相差约 1–2 tokens/s，峰值所在的相邻 \(\gamma\) 也可能难以仅凭图像精确区分。

| Target Model | Draft Model（正文名 / 图例名） | 图上峰值附近的 \(\gamma\)（约） | 峰值速度（约 tokens/s） | 证据等级 |
| --- | --- | ---: | ---: | --- |
| Vicuna-7B | Llama-68M | **4** | 约 60 | 正文明确最优模型和 \(\gamma\)；速度为读图估计 |
| Vicuna-7B | Llama-160M | 2–3 | 约 53–54 | 读图估计 |
| Vicuna-7B | Tiny-Llama / Llama-1B | 1–2 | 约 49–51 | 读图估计 |
| Vicuna-7B | Tiny-Vicuna / Vicuna-1B | 2–3 | 约 51–53 | 读图估计 |
| Vicuna-13B | Llama-68M | **3** | 约 53–54 | 正文明确最优模型和 \(\gamma\)；速度为读图估计 |
| Vicuna-13B | Llama-160M | 2–3 | 约 45–46 | 读图估计 |
| Vicuna-13B | Tiny-Llama / Llama-1B | 1–2 | 约 39–40 | 读图估计 |
| Vicuna-13B | Tiny-Vicuna / Vicuna-1B | 2–3 | 约 45–46 | 读图估计 |
| Vicuna-33B | Llama-68M | 3–5 | 约 26–27 | 读图估计 |
| Vicuna-33B | Llama-160M | 3–4 | 约 24–25 | 读图估计 |
| Vicuna-33B | Tiny-Llama / Llama-1B | 2–3 | 约 24–25 | 读图估计 |
| Vicuna-33B | Tiny-Vicuna / Vicuna-1B | **3** | 约 27–28 | 正文明确最优模型和 \(\gamma\)；速度为读图估计 |

**灰色虚线基线**也只能从图中近似读取：Vicuna-7B 约 42–45 tokens/s，Vicuna-13B 约 34–35 tokens/s，Vicuna-33B 约 17–18 tokens/s。论文没有在 Appendix D 单独列出这三个基线的精确数值。

需要特别区分：
- **正文明确给出的三个最优组合**：7B / Llama-68M / \(\gamma=4\)；13B / Llama-68M / \(\gamma=3\)；33B / Tiny-Vicuna / \(\gamma=3\)。
- **其他九个组合的峰值位置和速度**：上表是图像估读，原文没有逐项给出精确数值。
- **图中扫描范围**：是对多个 \(\gamma\) 值的曲线比较；不能把曲线最高点之外的所有点解释成论文正文明确推荐的参数。

### 2.3 与主文 Table 1 的交叉核对

主文 Section 3 的 Table 1 另行报告了 Speculative Decoding 与 Medusa 的总体 Speedup：

| Target Model | Speculative Decoding Speedup | Medusa-2 Speedup |
| --- | ---: | ---: |
| Vicuna-7B | 1.47× | 2.83× |
| Vicuna-13B | 1.56× | 2.83× |
| Vicuna-33B | 1.60× | 2.35× |

这张表是主文的总体比较，不是 Figure 7 中每条 Draft 曲线的逐项数值表。尤其不能把 Figure 7 某一条曲线的估读速度与 Table 1 的总体 Speedup 直接拼接成一个精确的逐模型实验记录。

## 三、实验设置与模型配对

### 3.1 Appendix D 明确说明的设置

| 项目 | 原文信息 |
| --- | --- |
| Target Models | Vicuna-7B、Vicuna-13B、Vicuna-33B |
| Draft Models | Llama-68M、Llama-160M、Tiny-Llama、Tiny-Vicuna |
| Tiny-Vicuna 来源 | 基于 Tiny-Llama，采用 Vicuna-style instruction tuning 微调 |
| Benchmark | MT-Bench |
| 推测解码实现 | 由于相关方法的实现具有专有性质，使用开源替代实现 |
| Draft 加速 | 使用 \`torch.compile()\` 加速 Draft Model 推理 |
| 核心超参数 | \(\gamma\)：Draft Token 数量 |
| 结果展示 | 三个 Target 各一个子图；灰色虚线为 vanilla baseline |

论文主文 Introduction 说明整个评测主要关注 Batch Size = 1 的场景。Appendix D 本身没有重新列出 batch size、GPU 型号、精度、Prompt 长度、生成长度或每个 checkpoint 的完整模型 ID。对这些缺失项应保留“论文未说明”，不能用其他章节或现代默认配置填补。

### 3.2 四种 Draft Model 的定位

- **Llama-68M**：最小的 Draft，计算成本低。它在 Vicuna-7B 与 Vicuna-13B 上都获得正文明确报告的最优结果。
- **Llama-160M**：比 68M 更大，但在 Figure 7 中没有超过 Llama-68M 的总体表现；Draft 计算开销的增加没有换来足够的接受收益。
- **Tiny-Llama（图例 Llama-1B）**：约 1B 级别的 Llama 系 Draft。图中在三个 Target 上都没有成为正文所报告的最优组合。
- **Tiny-Vicuna（图例 Vicuna-1B）**：基于 Tiny-Llama 进行 Vicuna 风格指令微调。它在 Vicuna-33B 上以 \(\gamma=3\) 获得最佳结果，体现了 Draft 规模和指令分布匹配可能同时影响性能。

Appendix D 没有提供四种 Draft 的逐项接受率、Draft 单步耗时或拒绝次数，因此不能仅凭吞吐曲线定量拆解“更高接受率”与“更低 Draft 延迟”各自贡献了多少。

## 四、为什么 Draft Model 越大不一定越快？

### 4.1 推测解码的端到端成本

标准 Speculative Decoding 每轮大致包含：

1. Draft Model 自回归地产生 \(\gamma\) 个候选 Token；
2. Target Model 对候选序列进行并行验证；
3. 根据接受/拒绝规则，接受一个连续前缀，必要时修正并补采样；
4. 进入下一轮。

可用简化的每轮时间表示：

\[
T_{\mathrm{round}}(\gamma)
\approx
T_{\mathrm{draft}}(\gamma)
+
T_{\mathrm{verify}}(\gamma)
+
T_{\mathrm{sampling}}(\gamma)
\]

若 Draft 是自回归模型，则：

\[
T_{\mathrm{draft}}(\gamma)
\approx
\gamma \cdot t_{\mathrm{draft}}
\]

Target 验证通常能将 \(\gamma\) 个 Draft Token 合并到一次 Target Forward，但其计算成本仍随候选长度增长。真正决定性能的是每轮接受了多少 Token，以及为此花费了多少时间。

### 4.2 Draft 太小与太大的两种问题

**Draft 太小：**
- 单 Token 推理很快；
- 但模型表达能力可能不足；
- 如果 Draft 与 Target 分布差距较大，候选更容易被拒绝；
- Target 已经花费了验证开销，却无法接受足够长的前缀。

**Draft 太大：**
- 预测质量可能更接近 Target；
- 但每个 Draft Token 的计算成本更高；
- 当 Target 本身较小、单步验证成本不高时，Draft 的额外开销可能抵消接受率收益。

因此，最优 Draft 并不是“越接近 Target 参数量越好”，而是：

\[
\boxed{
\text{最优 Draft}
=
\arg\max_D
\frac{\mathbb{E}[\text{Accepted Tokens per Round}]}
{\mathbb{E}[\text{Round Latency}]}
}
\]

这只是对端到端目标的简化表达，论文没有将它作为 Appendix D 的正式优化公式。

### 4.3 为什么 \(\gamma\) 会有最佳点？

增加 \(\gamma\) 有两个相反效果：

- 正面：一次 Target 验证可能接受更多 Token，减少后续 Target Forward 次数；
- 负面：Draft 需要串行生成更多 Token；越靠后的 Token 往往越难被接受；验证更长的候选序列也增加开销。

当额外候选 Token 带来的接受收益低于 Draft 与验证开销时，吞吐开始下降。Figure 7 的曲线普遍在较小 \(\gamma\) 附近达到峰值，随后下降，正体现了这一权衡。

### 4.4 为什么 Vicuna-33B 更适合 Tiny-Vicuna？

**论文直接支持的事实：**

- Vicuna-7B 和 Vicuna-13B 的最佳 Draft 都是 Llama-68M；
- Vicuna-33B 的最佳 Draft 则是 Tiny-Vicuna（Vicuna-1B）；
- Tiny-Vicuna 通过 Vicuna 风格指令微调从 Tiny-Llama 得到。

**独立分析：**

Target 从 7B 扩大到 33B 后，每次 Target Forward 的成本通常显著增加。相对于更昂贵的 Target，1B Draft 的额外计算成本占比下降；与此同时，更大的 Draft 可能提供更高的候选质量。Tiny-Vicuna 与 Vicuna 的指令微调风格也可能缩小两者输出分布差异。

这使得 33B Target 有条件用更大的 Draft 换取更高接受效率。反过来，7B/13B Target 的每次验证相对便宜，68M Draft 的低延迟优势更明显。

以上是根据图表和模型结构作出的机制解释，不是论文提供的接受率分解或消融证明。论文没有公开足够数据来确认“33B 选择 Tiny-Vicuna 的主要原因一定是接受率提高”。

## 五、与 Medusa、EAGLE、DFlash 的技术关联

### 5.1 四种方法的 Draft 生成方式

| 维度 | 传统 Speculative Decoding（Appendix D） | Medusa | 原始 EAGLE | DFlash |
| --- | --- | --- | --- | --- |
| Draft 载体 | 独立小型 LLM | Target 顶部多个 Token Heads | 目标模型隐藏 Feature | 条件化 Block Diffusion |
| 预测方式 | Token-level autoregressive | 多个未来位置并行预测 | Feature-level autoregressive | 单次 Forward 并行生成一个 Block |
| Draft 是否独立 | 是 | 否 | 轻量 Draft 网络，使用 Target Feature 与 Token 条件 | 轻量 Draft 网络，使用 Target 多层 Feature 条件 |
| Draft 串行开销 | 约 \(\gamma\) 次小模型逐 Token 生成 | 多 Head 并行，不需要逐层自回归 Draft Transformer | 原论文树状 Draft 执行 5 次 Draft Forward | 一次并行 Block Draft Forward |
| Target 验证 | 并行验证 Draft Token | Tree Attention 一次验证候选树 | Tree Attention 一次验证候选树 | Target 自回归验证 Block |
| 主要取舍 | Draft 速度与接受率 | Head 预测准确度与树验证成本 | Feature 预测质量与 Draft 轮数 | Block 并行度、条件注入与 Target 验证成本 |

### 5.2 与 Medusa 的区别

传统 Speculative Decoding 需要额外加载、调度和运行一个独立 Draft Model。Medusa 则在 Target Backbone 上增加多个轻量 Head，从同一个隐藏状态并行预测不同未来位置的 Token。

Appendix D 的实验恰好展示了传统方法的模型选择难题：7B、13B、33B 的 Target 对应不同的最佳 Draft/ \(\gamma\) 组合。Medusa 通过多个 Head 和优化树结构避免维护独立小模型，但需要训练这些 Head，并承担树状验证的计算开销。

主文 Table 1 中，Medusa-2 的 Speedup 为 2.83×、2.83×、2.35×；Speculative Decoding 为 1.47×、1.56×、1.60×。这些是论文对应实验设置下的报告值，不代表所有模型、硬件和实现下 Medusa 都必然更快。

### 5.3 与 EAGLE 的区别

EAGLE 不再依赖一个独立的小型 Token LLM 直接自回归预测所有 Token，而是预测 Target 模型的中间 Feature，并使用 Shifted Token Embedding 提供额外条件，再通过共享 Target LM Head 形成 Token 分布。

原始 EAGLE Appendix A 指出，其 Draft Tree 和 Chain Draft 都需要 5 次 Draft Forward，随后 Target 通过一次 Forward 验证候选。其树结构按经验设计，高概率分支分配更宽或更深的扩展。

因此：
- Appendix D 的传统 Draft：每次串行预测一个 Token，依赖独立小模型；
- EAGLE：每轮预测 Feature，再形成 Token，并以树结构扩展候选；
- Medusa：多个 Token Heads 并行预测，离线优化候选树。

EAGLE 仍有多轮 Draft 计算，但 Draft 的特征条件化能力比简单独立小 LM 更强。

### 5.4 与 DFlash 的区别

DFlash 使用 Target 多层隐藏特征作为条件，通过 KV Injection 将目标上下文注入 Draft 网络，在一次 Block Diffusion Forward 中并行生成一个候选块，再由 Target 自回归验证。

它和 Appendix D 的核心区别在于 Draft 的计算并行度：

- 传统 SD：生成 \(\gamma\) 个 Token 需要约 \(\gamma\) 个串行 Draft Token 步骤；
- DFlash：一次并行 Draft Forward 预测一个 Block；
- EAGLE：通过 Feature Autoregression 逐步构建候选；
- Medusa：多个 Head 并行预测未来 Token，树状组合候选。

所以 DFlash 可以看作针对传统自回归 Draft 串行开销的另一条路线。它并不是把传统 SD 的 Draft Model 简单做大或做小，而是改变 Draft 的生成范式。

## 六、复现建议与论文信息缺口

若要复现 Figure 7，应至少记录：

1. 每个 Target/Draft 的完整 checkpoint、Tokenizer 和词表兼容性；
2. Draft 模型是否使用 \`torch.compile()\`，编译耗时是否排除在 Benchmark 外；
3. \(\gamma\) 的扫描范围和每个值的运行次数；
4. Target baseline、Draft-only latency、Target verification latency；
5. 每个 \(\gamma\) 的接受 Token 数、接受率、拒绝/重采样次数；
6. GPU 型号、精度、KV Cache 实现、Batch Size、Prompt 与生成长度；
7. MT-Bench 的任务组成、吞吐统计口径和 warmup 设置。

论文 Appendix D 没有公开所有这些细节，也没有逐点速度数据表。复现时应将缺失信息作为待补充实验记录，而不是根据 Figure 7 估读值反向假设硬件或 checkpoint。

作者脚注链接的 [LLMSpeculativeSampling](https://github.com/feifeibear/LLMSpeculativeSampling) README 也明确提醒：Draft/Target 配对会显著影响速度；两者都很小时，Target 与 Draft 的速度差距不足以产生收益；模型规模差距过大时，拒绝和重采样可能增多。该 README 是开源实现自身的说明，不应当作 Figure 7 的额外实验数据。

## 七、阅读小结与进度

- Appendix D 的两段正文和 Figure 7 完整图注已按 arXiv v3 原文顺序翻译。
- Figure 7 对比 Vicuna-7B/13B/33B，四条 Draft 曲线为 Llama-68M、Llama-160M、Llama-1B（正文 Tiny-Llama）和 Vicuna-1B（正文 Tiny-Vicuna）。
- 正文明确的最优组合：7B 使用 Llama-68M、\(\gamma=4\)；13B 使用 Llama-68M、\(\gamma=3\)；33B 使用 Tiny-Vicuna（Vicuna-1B）、\(\gamma=3\)。
- 其他曲线的峰值速度与峰值 \(\gamma\) 只提供近似读图值，论文没有原始逐点数据表。
- 核心结论：Draft 模型大小、任务分布匹配和 \(\gamma\) 需要联合调优；Draft 更大不等于端到端更快。
- 与后续方法的关联：Medusa 通过并行 Heads 消除独立自回归 Draft；EAGLE 通过 Feature-level Autoregression 改善 Draft 条件；DFlash 通过一次 Block Diffusion Forward 减少 Draft 串行轮次。
- **下一阅读单元：Appendix E Additional Results for All Models（Figure 8）。** 主文与 Appendices A–D 已完成，E–G 待精读；整篇论文尚未完成。

