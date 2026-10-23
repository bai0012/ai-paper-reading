# Appendix B — Detailed Experimental Results（完整中文翻译与 Table 8 全部数据）

> 论文：[EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/html/2401.15077v3)  
> 版本：arXiv:2401.15077v3；原始 PDF 共 13 页。  
> **原文位置：** [PDF 第 12 页 Appendix B 正文](https://arxiv.org/pdf/2401.15077v3#page=12)；[PDF 第 13 页 Table 8](https://arxiv.org/pdf/2401.15077v3#page=13)。  
> **完成日期：** 2026-10-23。  
> 本文先逐句翻译附录原文、逐项转录 Table 8 的 18 行全部数值；然后明确分隔**独立分析**、数据计算、复现建议，避免把分析推断写成作者的新实验结论。
>
> **原表原图状态：** 本次聊天已直接从用户提供的 PDF 第 13 页提取 Table 8 PNG 并完整展示；GitHub 暂无已成功上传的原图二进制。这里使用原始 PDF 精确页码链接而不是不存在的 \`figures/Table-8.png\`。

## 一、Appendix B 原文完整翻译

### B. Detailed experimental results（详细实验结果）

Table 8 展示了 EAGLE 在 HumanEval、GSM8K 和 Alpaca 数据集上的加速比（Speedup Ratio）、平均接受长度 $\tau$（Average Acceptance Length）和接受率 $\alpha$（Acceptance Rate）。

**本节正文翻译至此结束。Appendix B 仅这一段文字，余下是 Table 8。**

### Table 8 原表完整图注翻译

**原始表格：** [PDF 第 13 页 Table 8](https://arxiv.org/pdf/2401.15077v3#page=13)。已在本次聊天中展示原始裁图。

**图注/表注完整翻译：** 表 8：在 Temperature=0 下，EAGLE 在 HumanEval、GSM8K 和 Alpaca 上的加速比、平均接受长度 $\tau$ 及接受率 $\alpha$。

**表中列定义：**
- **Dataset：** 测试数据集
- **Model：** Target LLM
- **Speedup：** 相对 Vanilla Autoregressive Decoding 的实际耗时加速比
- **$\tau$：** 平均接受长度（本论文正文使用的口径）
- **$0\text{-}\alpha,\ldots,4\text{-}\alpha$：** 链式草稿中，输入包含 $0\ldots4$ 个 Draft 预测的、可能存在误差的 Feature 时的接受率。**不是**同一棵候选树中所有节点被接受的比例。

#### HumanEval（代码生成；原始 6 行）

| Model | Speedup | $\tau$ | $0\text{-}\alpha$ | $1\text{-}\alpha$ | $2\text{-}\alpha$ | $3\text{-}\alpha$ | $4\text{-}\alpha$ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Vicuna 7B | 3.33× | 4.29 | 0.82 | 0.77 | 0.72 | 0.69 | 0.71 |
| Vicuna 13B | 3.58× | 4.39 | 0.85 | 0.78 | 0.74 | 0.72 | 0.73 |
| Vicuna 33B | 3.67× | 4.28 | 0.83 | 0.77 | 0.74 | 0.70 | 0.70 |
| LLaMA2-Chat 7B | 3.17× | 4.24 | 0.81 | 0.76 | 0.73 | 0.74 | 0.72 |
| LLaMA2-Chat 13B | 3.76× | 4.52 | 0.85 | 0.80 | 0.78 | 0.76 | 0.75 |
| LLaMA2-Chat 70B | 3.52× | 4.42 | 0.84 | 0.79 | 0.75 | 0.73 | 0.74 |

#### GSM8K（数学推理；原始 6 行）

| Model | Speedup | $\tau$ | $0\text{-}\alpha$ | $1\text{-}\alpha$ | $2\text{-}\alpha$ | $3\text{-}\alpha$ | $4\text{-}\alpha$ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Vicuna 7B | 3.01× | 4.00 | 0.79 | 0.71 | 0.70 | 0.71 | 0.70 |
| Vicuna 13B | 3.08× | 3.97 | 0.79 | 0.71 | 0.67 | 0.68 | 0.64 |
| Vicuna 33B | 3.25× | 3.94 | 0.79 | 0.71 | 0.67 | 0.67 | 0.67 |
| LLaMA2-Chat 7B | 2.91× | 3.82 | 0.75 | 0.69 | 0.64 | 0.65 | 0.63 |
| LLaMA2-Chat 13B | 3.20× | 4.03 | 0.80 | 0.70 | 0.70 | 0.68 | 0.66 |
| LLaMA2-Chat 70B | 3.03× | 3.93 | 0.77 | 0.71 | 0.66 | 0.64 | 0.60 |

#### Alpaca（指令遵循；原始 6 行）

| Model | Speedup | $\tau$ | $0\text{-}\alpha$ | $1\text{-}\alpha$ | $2\text{-}\alpha$ | $3\text{-}\alpha$ | $4\text{-}\alpha$ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Vicuna 7B | 2.79× | 3.86 | 0.74 | 0.68 | 0.66 | 0.66 | 0.67 |
| Vicuna 13B | 3.03× | 3.95 | 0.72 | 0.67 | 0.64 | 0.63 | 0.64 |
| Vicuna 33B | 2.97× | 3.61 | 0.70 | 0.64 | 0.64 | 0.63 | 0.64 |
| LLaMA2-Chat 7B | 2.78× | 3.71 | 0.73 | 0.66 | 0.62 | 0.64 | 0.62 |
| LLaMA2-Chat 13B | 3.01× | 3.83 | 0.75 | 0.67 | 0.64 | 0.63 | 0.63 |
| LLaMA2-Chat 70B | 2.97× | 3.77 | 0.76 | 0.68 | 0.65 | 0.61 | 0.62 |

**18 行全部数值已经逐项与 PDF 原表核对。Table 8 是全篇最后一张表。**

---

## 二、独立数据分析：不同任务、模型与连续特征误差

以下统计是**对 Table 8 数据的后处理**，并不是作者在 Appendix B 单独报告的均值或显著性检验。每个任务均对表中的六种 Target 模型赋予同等权重做简单算术平均。模型/Token/请求的真实总体权重可能不同，不能把这些均值当作服务级别综合指标。

### 1. 三个数据集的跨模型简单平均

| Dataset | 平均 Speedup | 平均 $\tau$ | 平均 $0\text{-}\alpha$ | 平均 $1\text{-}\alpha$ | 平均 $4\text{-}\alpha$ |
| --- | ---: | ---: | ---: | ---: | ---: |
| HumanEval | **3.505×** | **4.3567** | **0.8333** | **0.7783** | **0.7250** |
| GSM8K | 3.080× | 3.9483 | 0.7817 | 0.7050 | 0.6500 |
| Alpaca | 2.925× | 3.7883 | 0.7333 | 0.6667 | 0.6367 |

在同一组模型、相同 T=0 设置中，HumanEval 的平均加速比、平均 $\tau$ 和各主要接受率均高于 GSM8K 与 Alpaca。对 6 个模型逐个比较，HumanEval 的加速比均高于 GSM8K，GSM8K 又均高于 Alpaca。**可以结合代码样板/重复片段较多的特点做机制解释，但不能直接把速度差归因于“代码能力更强”或“数学题一定更难”。** Prompt/Output 长度、模型行为与 GPU 执行开销都可能影响速度。

### 2. 特征误差累积的一个重要观察：最初几步降幅更明显

各任务跨六模型的接受率简单均值：

| Dataset | $0\text{-}\alpha$ | $1\text{-}\alpha$ | $2\text{-}\alpha$ | $3\text{-}\alpha$ | $4\text{-}\alpha$ |
| --- | ---: | ---: | ---: | ---: | ---: |
| HumanEval | 0.8333 | 0.7783 | 0.7433 | 0.7233 | 0.7250 |
| GSM8K | 0.7817 | 0.7050 | 0.6733 | 0.6717 | 0.6500 |
| Alpaca | 0.7333 | 0.6667 | 0.6417 | 0.6333 | 0.6367 |

0-α→1-α 的平均**绝对下降**分别为：
- HumanEval：0.0550，即 **5.50 个百分点**。
- GSM8K：0.0767，即 **7.67 个百分点**。
- Alpaca：0.0667，即 **6.67 个百分点**。

0-α→4-α 的平均总下降分别为 0.1083、0.1317、0.0967（绝对概率单位），说明连续使用预测特征后仍有损失，但很多情况下后续下降更缓。**这支持作者在 Section 4.1 所说的“Feature Error 有影响而 EAGLE 对误差累积有一定稳健性”。**

### 3. 为什么个别 $n\text{-}\alpha$ 反而升高？

原表并非全部严格单调下降：
- HumanEval / Vicuna 7B：$3\text{-}\alpha=0.69$，$4\text{-}\alpha=0.71$。
- GSM8K / Vicuna 7B：$2\text{-}\alpha=0.70$，$3\text{-}\alpha=0.71$。
- Alpaca / LLaMA2-Chat 7B：$2\text{-}\alpha=0.62$，$3\text{-}\alpha=0.64$。

这**不是**“预测特征越不准确反而越好”的证明。接受率是随实际生成轨迹、可用前缀、Token 候选及评测样本变化的统计结果，并不保证每一列都必须严格单调。在没有误差条、置信区间和更深入采样条件控制的情况下，不能将这些小反弹解释为新机制。

### 4. 单模型案例：LLaMA2-Chat 13B

| Dataset | Speedup | $\tau$ | $0\text{-}\alpha$ | $4\text{-}\alpha$ |
| --- | ---: | ---: | ---: | ---: |
| HumanEval | **3.76×** | **4.52** | 0.85 | 0.75 |
| GSM8K | 3.20× | 4.03 | 0.80 | 0.66 |
| Alpaca | 3.01× | 3.83 | 0.75 | 0.63 |

同一个 Target，代码任务加速 3.76×，数学任务 3.20×，指令任务 3.01×；对应 $\tau$ 从 4.52 降到 4.03 和 3.83。这个案例说明候选预测质量与实际推进长度、综合速度有相关趋势。但**Speedup 不等于 $\tau$，不能由接受长度直接推出精确加速倍数**，因为一次草稿树扩展与验证仍需计算时间。

### 5. 模型规模与草稿质量不是严格单调关系

例如 Alpaca 中 Vicuna 13B 的 Speedup=3.03×、$\tau=3.95$，高于 Vicuna 33B 的 2.97×、3.61；GSM8K 中 Vicuna 33B 的 Speedup=3.25×、$\tau=3.94$，则高于 Vicuna 13B 的 3.08×、3.97，说明**稍短的平均接受长度也可能对应更高的实际加速比**。该差异受不同模型的 Target Forward 成本、Draft 计算、候选树与硬件运行影响，不能直接将模型规模和加速收益等同。

### 6. 为什么这些数据不能说明模型“任务能力提升”？

EAGLE 的精确 Target 校正规则旨在使最终输出仍来自 Target 的有效采样分布。Table 8 展示的是生成效率（Speedup）、每次 Target Forward 的有效推进（$\tau$）与候选接受率，而非 HumanEval Pass@1、GSM8K 答题准确率或 Alpaca 人类偏好得分。

**EAGLE 改善推理效率，而不是训练出一个在相同 Target 分布保证下普遍能力更强的主模型。**

---

## 三、与正文 Table 1 的关系

[Section 4 Experiments](04-Experiments.md) Table 1 已提供这三个任务、T=0/1 的 Speedup 和 $\tau$。**Appendix B Table 8 把其中 T=0 的 Speedup、$\tau$ 重列，并补充了全部 $0\text{-}\alpha$ 到 $4\text{-}\alpha$ 指标。**

不要把 Table 1 和 Table 8 的重复速度数据误认成两次独立实验：这主要是相同实验条件下不同详细程度的报告。

完整原图浏览：
- [Table 1：PDF 第 6 页](https://arxiv.org/pdf/2401.15077v3#page=6)
- [Table 8：PDF 第 13 页](https://arxiv.org/pdf/2401.15077v3#page=13)

---

## 四、可复现实验与附加分析建议（独立建议）

1. **按条件记录接受率**：针对每一轮已接受的连续 Draft 数量，记录对应的 $n\text{-}\alpha$、Feature 误差范数与残差抽样率，核对 $n$ 的含义，不要以树节点总数代替链式接受率。
2. **逐阶段 Profiling**：记录每轮 Draft 各步耗时、树宽/深、Target 树验证时长、被接受路径的 Token 数和 KV/Mask 开销。
3. **多温度扩展**：对同一任务分别测试 Temperature=0/1，必要时再改变 top-p，但确保 Draft/Target 有效采样分布和严格校正规则一致。
4. **服务测量**：固定目标模型与硬件，测试不同 Batch/Concurrency、Prompt 长度和 Output 长度下的 TTFT、TPOT、E2E Latency、TPS 和 KV 显存占用。
5. **跨论文**：与 [DFlash](../2602.06036-dflash/README.md) 对齐硬件和模型后比较 Draft 关键路径 vs 接受长度；勿将 DFlash 对 EAGLE-3 的结果直接视为与本篇 EAGLE (2024) 的严格比较。

## 五、全文阅读完成状态

- Abstract、Section 1 Introduction、Section 2 Preliminaries、Section 3 EAGLE、Section 4 Experiments、Section 5 Related Work、Section 6 Conclusion、Appendix A（A.1/A.2 + Algorithm 1）、Appendix B（本文件 + Table 8）均已阅读。
- 全文 Figure 1–9、Table 1–8 和 Algorithm 1 的原文图注/表注、数据或原伪代码均已随相应章节处理。
- 重点核对备注：Appendix A.2 Algorithm 1 中印刷的标量残差更新无法直接作为词表概率分布，应参考 [Appendix A](Appendix-A.md) 的校正分析；原 Section 6 将 2.1–3.8× 整体标记为 MT-bench，但可核对的最高 3.76× 出自 HumanEval，见 [Conclusion](06-Conclusion.md)。
- 全文的跨论文综合总结见 [summary.md](summary.md)。  
- **图表原始 PNG 的 GitHub 状态：** 当前章节只保留原 PDF 精确页码和转录数值，不声称原始二进制已上传；原表 PNG 已在本次聊天由原 PDF 提取展示。
