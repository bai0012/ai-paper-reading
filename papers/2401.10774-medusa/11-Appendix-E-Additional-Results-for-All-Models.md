# Appendix E · Additional Results for All Models

**论文**：[Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774)  
**版本**：arXiv:2401.10774v3（2024-06-14）  
**原始 PDF**：[第 16 页（印刷页码 15）：Figure 8](https://arxiv.org/pdf/2401.10774v3#page=16)  
**HTML 原文**：[Appendix E](https://arxiv.org/html/2401.10774v3#appendix-e-additional-results-for-all-models)  
**阅读范围**：Appendix E 唯一正文句子与 Figure 8 完整图注；结合主文 Table 1、Section 3.1/3.2、Appendix B.3/B.4 及 Appendix D 的 Speculative Decoding 结果，分析四种模型的速度、加速、质量和自蒸馏影响。  
**来源边界**：Appendix E 原文只有一句正文和 Figure 8 图注。Table 1 是主文实验数据，引用时明确标出；Section 3.1/3.2 和 Appendix B 是对训练流程的交叉核对。Figure 8 的绝对 tokens/s 柱高没有逐项数值标注，图上读数只能给近似范围；加速比采用图中标签和 Table 1 的正式报告值。

## 一、Appendix E 原文顺序完整翻译

### E. Additional Results for All Models（所有模型的补充结果）

**原文唯一正文句子**

> We show speedup on various models in Fig. 8.

**完整翻译**

我们在 Figure 8 中展示了不同模型上的加速效果。

**原文正文到此结束。** Appendix E 没有其他正文段落、公式或独立表格；Figure 8 是本节的核心实验结果。

### Figure 8 完整图注翻译

> Figure 8. Speedup of various models with MEDUSA-2. MEDUSA-2 shows significant speed improvement over all the models, while models trained with self-distillation (Zephyr-7B, Vicuna-13/33B) have weaker speedup due to the trade-off between preserving quality and boosting speed.

**完整翻译**

**图 8：不同模型使用 MEDUSA-2 后的加速效果。**

MEDUSA-2 在所有模型上都表现出显著的速度提升。然而，使用自蒸馏（Self-Distillation）训练的模型（Zephyr-7B、Vicuna-13B/33B），由于需要在保持生成质量与提升推理速度之间进行权衡，因此获得的加速效果相对较弱。

## 二、Figure 8 原图与图表读数

**原图**：[arXiv v3 PDF 第 16 页](https://arxiv.org/pdf/2401.10774v3#page=16)（PDF 页索引 15，论文印刷页码 15）。已通过 PDF 原始页面截图核对 Figure 8 的四组柱形、坐标轴、图例和四个速度比标签。截图所在页面还包含 Appendix F 的 Table 4 和 Appendix G 开头，Figure 8 位于页面上半部。

图表属性：

- 标题：Speedup on different model sizes（不同模型规模下的加速效果）
- 横轴：Model Size，四个模型按 Vicuna-7B、Zephyr-7B、Vicuna-13B、Vicuna-33B 排列
- 纵轴：Tokens per Second
- 蓝色柱：w/o Medusa，原始模型
- 橙色柱：Medusa-2
- 每组柱顶标注的 Speedup：2.83×、2.66×、2.83×、2.35×

### 2.1 图中速度的近似读取

Figure 8 只标出纵轴刻度和 Speedup，没有在每根柱子上打印精确 tokens/s 数值。按原图柱高估读：

| 模型 | 原始模型（约 tokens/s） | Medusa-2（约 tokens/s） | 图中 Speedup |
| --- | ---: | ---: | ---: |
| Vicuna-7B | 44–46 | 126–130 | **2.83×** |
| Zephyr-7B | 39–41 | 105–108 | **2.66×** |
| Vicuna-13B | 34–36 | 96–99 | **2.83×** |
| Vicuna-33B | 16–18 | 39–41 | **2.35×** |

**数据精度说明**：前两列是根据 PDF 图像纵轴估读的区间，不是作者公布的精确测量值；最后一列是原图标签，并由主文 Table 1 交叉核对。不要把 Appendix F 的 AlpacaEval Table 4 绝对速度误当作 Figure 8 的 MT-Bench 速度。

## 三、主文 Table 1 的精确指标交叉核对

Figure 8 给出可视化结果，主文 Section 3 的 Table 1 提供四个模型的 Acceleration Rate、Overhead、Quality、Speculative Decoding Speedup 和 Medusa Speedup。Table 1 是主文数据，不是 Appendix E 的附属表格。

| 指标 | Vicuna-7B | Zephyr-7B | Vicuna-13B | Vicuna-33B |
| --- | ---: | ---: | ---: | ---: |
| Acceleration Rate | 3.47 | 3.14 | 3.51 | 3.01 |
| Overhead | 1.22 | 1.18 | 1.23 | 1.27 |
| Quality | 6.18 | 7.25 | 6.43 | 7.18 |
| Quality change | +0.01 | −0.07 | −0.14 | +0.05 |
| Speculative Decoding Speedup | 1.47× | — | 1.56× | 1.60× |
| **Medusa-2 Speedup** | **2.83×** | **2.66×** | **2.83×** | **2.35×** |

Quality 是 GPT-4 在 MT-Bench 上评估的平均分（0–10）；括号内的变化表示 Medusa 相对于原始模型的分数变化。Speculative Decoding 的 Zephyr-7B 数据用破折号表示，论文没有报告该配对结果。

### 3.1 Speedup 的数学含义

Appendix B.1 定义：

\[
S = \frac{A}{O}
\]

其中：
- \(A\)：Acceleration Rate，即每轮解码的平均推进 Token 数；
- \(O\)：Overhead，即 Medusa 解码每轮时间相对于普通解码每轮时间的比值；
- \(S\)：最终 Speedup。

用 Table 1 的已四舍五入数值估算：

| 模型 | Acceleration Rate / Overhead | Table 1 报告 Speedup |
| --- | ---: | ---: |
| Vicuna-7B | \(3.47/1.22\approx2.84\) | 2.83× |
| Zephyr-7B | \(3.14/1.18\approx2.66\) | 2.66× |
| Vicuna-13B | \(3.51/1.23\approx2.85\) | 2.83× |
| Vicuna-33B | \(3.01/1.27\approx2.37\) | 2.35× |

四舍五入后的分项数值与正式 Speedup 有轻微差异。应保留论文正式报告值；不能用已舍入的两个分项反向覆盖论文结果。

### 3.2 四个模型逐项分析

**Vicuna-7B**

- Acceleration Rate = 3.47，为四个模型第二高。
- Overhead = 1.22。
- 最终 Speedup = 2.83×。
- Quality = 6.18（+0.01）。
- 主文 Section 3.1 说明 7B 使用 ShareGPT 训练 Medusa Heads 两个 Epoch；Appendix B.3 说明先 Medusa-1，再以其为初始化进行 Medusa-2 两阶段训练。
- 它与 Vicuna-13B 并列取得最高 Medusa-2 Speedup，质量变化接近零。

**Zephyr-7B**

- Acceleration Rate = 3.14，是四者中第二低。
- Overhead = 1.18，是四者中最低。
- 最终 Speedup = 2.66×。
- Quality = 7.25（−0.07）。
- Zephyr 经过 SFT 和 Alignment/RLHF 风格训练，论文采用自蒸馏生成训练数据。
- 较低 Overhead 抵消了部分较低 Acceleration Rate，因此 Speedup 仍达到 2.66×。

**Vicuna-13B**

- Acceleration Rate = 3.51，是四者最高。
- Overhead = 1.23。
- 最终 Speedup = 2.83×，与 Vicuna-7B 并列最高。
- Quality = 6.43（−0.14），四者中质量分数下降幅度最大。
- Section 3.1 和 Appendix B.3 将其描述为使用公开 ShareGPT 数据、两阶段 Medusa-1 → Medusa-2 训练；但 Figure 8 图注又将 Vicuna-13B 列入自蒸馏模型。该处存在论文内部表述不一致，详见第四节。

**Vicuna-33B**

- Acceleration Rate = 3.01，是四者最低。
- Overhead = 1.27，是四者最高。
- 最终 Speedup = 2.35×，是四者最低。
- Quality = 7.18（+0.05），相较原模型没有质量下降。
- 作者认为，原模型隐藏的训练数据分布与自蒸馏数据集之间可能存在不匹配，导致 Medusa Heads 学到的分布偏移，影响预测准确率和接受长度。
- 因此它的主要问题是接受效率较弱，而不是评测质量下降。

## 四、自蒸馏对加速和质量的影响

### 4.1 论文明确给出的自蒸馏流程

Section 3.2 研究 Vicuna-33B 和 Zephyr-7B，使用 ShareGPT、UltraChat 作为 Seed Dataset，分别收集约 100k 条模型生成样本。

- Zephyr-7B：模型能够从单个 Prompt 继续生成多轮对话，因此数据收集相对直接。
- Vicuna-33B：从多轮 Seed Conversation 中迭代输入 Prompt，使用 temperature=0.3 的随机采样生成多轮对话。
- 两者训练 Sequence Length=2048、Batch Size=128。

Appendix B.4 进一步给出训练设置：

- 直接使用 Medusa-2，不采用 Vicuna-7B/13B 的两阶段训练流程；
- 对 \(\theta_0\) 使用 Sine Schedule，逐渐增加至训练结束时的峰值；
- Backbone LoRA Peak LR=\(10^{-4}\)；
- Warmup=20 steps；
- \(\lambda_0=0.01\)。

这里的自蒸馏并非简单地将模型自己生成的 Token 当作唯一标签。Section 2.3.2 使用原始模型的概率分布作为 Teacher，采用 KL 蒸馏约束 Backbone；通过关闭 LoRA Adapter 获得原始 Teacher，避免额外保存一份完整模型。

### 4.2 速度与质量的权衡

图注把 Zephyr-7B、Vicuna-13B/33B 归入自蒸馏模型，并指出它们由于保质量而牺牲部分速度。根据 Table 1：

- Zephyr-7B：2.66×，Quality −0.07；
- Vicuna-13B：2.83×，Quality −0.14；
- Vicuna-33B：2.35×，Quality +0.05。

但不能将这三个模型的结果简单归因于自蒸馏导致速度下降：
1. 三者 Target 规模、原始模型训练过程、训练数据和分布都不同。
2. Vicuna-13B 的 Speedup 与 Vicuna-7B 完全相同，并未体现更低的加速比。
3. Vicuna-33B 的低加速主要由较低 Acceleration Rate 和较高 Overhead 共同体现；作者进一步提出数据分布错配假设。
4. Table 1 没有提供同一模型“使用自蒸馏/不使用自蒸馏”的严格对照实验。

### 4.3 v3 论文内部存在的自蒸馏归属矛盾

这是本节需要保留的证据边界：

- **Figure 8 图注**：明确把 Zephyr-7B、Vicuna-13B/33B 归为 self-distillation models。
- **Section 3.2 标题和正文**：只把 Vicuna-33B、Zephyr-7B 作为自蒸馏案例。
- **Section 3.1 正文**：明确说 Vicuna-7B/13B 使用公开 ShareGPT 数据训练 Medusa Heads 两个 Epoch。
- **Appendix B.3**：Vicuna-7B/13B 先 Medusa-1，再用该模型初始化 Medusa-2，使用 QLoRA。
- **Appendix B.4**：自蒸馏设置仅讨论 Vicuna-33B 和 Zephyr-7B。

因此，Figure 8 图注中的“Vicuna-13B”与 Section 3.1、Appendix B.3 的训练设置不一致。本文保留原始各处说法，不擅自将 Vicuna-13B 重新分类。若要严格复现实验，应进一步核对作者当时的训练配置和模型 checkpoint。

## 五、与 Appendix D 传统 Speculative Decoding 的对照

Appendix D 使用独立 Draft Model 对 Vicuna-7B/13B/33B 做 Speculative Decoding；Figure 7 给出不同 Draft Model 和 \(\gamma\) 的速度曲线。主文 Table 1 汇总了其最终 Speedup，可与 Figure 8 的 Medusa-2 对照。

| Target Model | Appendix D 最佳 Draft | \(\gamma\) | 传统 SpecDec Speedup | Medusa-2 Speedup |
| --- | --- | ---: | ---: | ---: |
| Vicuna-7B | Llama-68M | 4 | 1.47× | **2.83×** |
| Vicuna-13B | Llama-68M | 3 | 1.56× | **2.83×** |
| Vicuna-33B | Tiny-Vicuna / Vicuna-1B | 3 | 1.60× | **2.35×** |
| Zephyr-7B | 未报告 | — | — | **2.66×** |

按相同原始模型基线计算，Medusa-2 的加速倍数相对传统 Speculative Decoding 倍数分别约为：

\[
\frac{2.83}{1.47}\approx1.93,\quad
\frac{2.83}{1.56}\approx1.81,\quad
\frac{2.35}{1.60}\approx1.47
\]

这表示在论文对应实验设置中，Medusa-2 的 Speedup 倍数约是传统 Speculative Decoding 的 1.47–1.93 倍；它不是说 Medusa-2 绝对 tokens/s 必然在所有硬件和服务场景中按同样比例领先。

### 5.1 为什么 Medusa-2 在三种 Vicuna 上都领先？

**传统 Speculative Decoding：**
- 需要独立 Draft Model；
- Draft Token 逐个自回归生成；
- Draft/Target 的规模和分布需要匹配；
- Draft 计算成本与 Target 验证成本共同决定速度。

**Medusa-2：**
- 复用 Target Backbone；
- 多个 Heads 从共享隐藏状态预测多个未来位置；
- Tree Attention 并行验证候选树；
- 联合训练 Backbone 和 Heads，提高未来 Token 预测准确率。

因此，Medusa-2 避免了传统 SD 的独立 Draft 模型和串行 Draft 推理成本。但这不是一个跨所有配置的无条件结论；Medusa 的树计算也有额外 Overhead，且 Table 1 的 Zephyr 没有传统 SD 对照值。

## 六、研究结论与复现注意事项

1. Figure 8 的四个模型 Medusa-2 Speedup 分别为 2.83×、2.66×、2.83×、2.35×，与主文 Table 1 一致。
2. Vicuna-7B 和 Vicuna-13B 加速比最高；Vicuna-33B 的 Acceleration Rate 最低、Overhead 最高，因此加速比最低。
3. 自蒸馏可以在没有原始训练集时构造匹配模型输出分布的训练数据，并通过 KL Teacher 约束保持 Backbone 能力；但 Heads 的预测分布与原训练分布之间仍可能出现偏移。
4. Figure 8 图注对 Vicuna-13B 的自蒸馏归属与 Section 3.1/Appendix B.3 不一致，复现时必须核对 checkpoint 和训练配置。
5. Appendix D 的传统 SD 在 Vicuna-7B/13B/33B 分别为 1.47×/1.56×/1.60×；Medusa-2 分别为 2.83×/2.83×/2.35×。Zephyr 没有传统 SD 对照值。
6. Figure 8 绝对 tokens/s 没有精确逐点表格；图上速度只能标为近似值。Appendix F 的 AlpacaEval Table 4 是另一个 Benchmark，不能与 Figure 8 的 MT-Bench 数据混用。

## 七、阅读小结与进度

- Appendix E 唯一正文句子和 Figure 8 完整图注已按 v3 PDF 顺序翻译。
- Figure 8 四模型的速度柱形、图例、轴标签和加速比已核对；图中明确标注 2.83×、2.66×、2.83×、2.35×。
- 主文 Table 1 的 Acceleration Rate、Overhead、Quality、Speculative Decoding 和 Medusa Speedup 全部交叉核对。
- 详细分析了 Vicuna-7B、Zephyr-7B、Vicuna-13B、Vicuna-33B 的速度、质量和自蒸馏差异。
- 发现并保留 Figure 8 caption 与 Section 3.1/Appendix B.3 对 Vicuna-13B 自蒸馏归属的内部不一致。
- 与 Appendix D 的 Draft Model、gamma 和传统 Speculative Decoding Speedup 逐项比较。
- **Figure 8 原始页面已核对并截图展示；GitHub PNG 二进制未保存**。原图准确来源和页码见 [figures/figure8-source.md](figures/figure8-source.md)。
- **下一阅读单元：Appendix F Additional Results on AlpacaEval Dataset（Table 4）。** 主文与 Appendices A–E 已完成，F–G 待精读；整篇论文尚未完成。
