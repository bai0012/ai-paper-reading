# Appendix F · Additional Results on AlpacaEval Dataset

**论文**：[Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774)  
**版本**：arXiv:2401.10774v3（2024-06-14）  
**原始 PDF**：[第 15 页（Appendix F 正文）](https://arxiv.org/pdf/2401.10774v3#page=15)；[第 16 页（Table 4 原表）](https://arxiv.org/pdf/2401.10774v3#page=16)  
**HTML 原文**：[Appendix F](https://arxiv.org/html/2401.10774v3#appendix-f-additional-results-on-alpacaeval-dataset)  
**阅读范围**：F 节完整正文、Table 4 表头/全部数据/表注；与 Section 3 Table 1、Appendix E Figure 8、Appendix D 传统 Speculative Decoding 结果对照。  
**来源边界**：Appendix F 正文只有两句；Table 4 提供 AlpacaEval 上的吞吐、Acceleration Rate 和 Speedup。Table 4 未报告 Overhead，文中反推值明确标记为计算结果。它也没有提供 AlpacaEval 上的传统 Speculative Decoding 对照。

## 一、Appendix F 原文顺序完整翻译

### F. Additional Results on AlpacalEval Dataset（AlpacaEval 数据集上的补充结果）

**原文：**

> We conduct further experiments on the AlpacaEval (Li et al., 2023) dataset. MEDUSA-2 achieves consistent speedup similar to the results on MT-Bench.

**完整翻译：**

我们在 AlpacaEval（Li et al., 2023）数据集上进行了进一步实验。MEDUSA-2 获得了与 MT-Bench 上的结果相近的加速效果。

原文正文到此结束。Appendix F 没有其他段落、公式或额外实验设置说明，结果由 Table 4 给出。

## 二、Table 4 原表与完整翻译

**原始 PDF 位置**：第 16 页（PDF 页索引 15，论文印刷页码 16）。原表页面已核对；原始页面截图和字段位置记录见 [tables/table4-source.md](tables/table4-source.md)。

### Table 4 原始数据转录

| Model | Base speed (tokens/s) | MEDUSA speed (tokens/s) | Acc. rate | Speedup |
| --- | ---: | ---: | ---: | ---: |
| Vicuna-7b | 37.07 | 106.76 | 3.23 | 2.88 |
| Vicuna-13b | 29.01 | 91.54 | 3.28 | 3.16 |
| Vicuna-33b | 17.87 | 40.43 | 2.85 | 2.26 |
| Zephyr-7b | 34.21 | 99.50 | 3.08 | 2.91 |

### 表头逐项翻译

| 原文 | 中文 | 含义 |
| --- | --- | --- |
| Model | 模型 | 被评估的基础模型 |
| Base speed (tokens/s) | 基础模型速度（Token/秒） | 不使用 Medusa 的原始模型吞吐 |
| MEDUSA speed (tokens/s) | MEDUSA 速度（Token/秒） | 使用 Medusa-2 后的吞吐 |
| Acc. rate | 加速率 | 每个解码步骤平均解码的 Token 数 |
| Speedup | 加速比 | 实际墙钟时间的加速倍数 |

### 表注完整翻译

> Table 4. Speedup results on AlpacaEval (Li et al., 2023) dataset.

**表 4：在 AlpacaEval（Li et al., 2023）数据集上的加速结果。**

## 三、逐项验证 Speedup

Appendix B.1 对指标作出定义：

- **Acceleration Rate**：每个解码步骤平均解码的 Token 数。普通自回归模型的该值为 1.0。
- **Overhead**：Medusa 模型每个解码步骤的平均延迟除以普通模型每步平均延迟。
- **Speedup**：墙钟时间加速比。

因此：

\[
S = \frac{A}{O}
\]

其中 \(A\) 是 Acceleration Rate，\(O\) 是 Overhead，\(S\) 是 Speedup。

由于 Table 4 同时提供 Base speed 和 MEDUSA speed，还可以直接计算：

\[
S = \frac{v_{\mathrm{MEDUSA}}}{v_{\mathrm{Base}}}
\]

### 3.1 用吞吐量反算

| 模型 | 吞吐量计算 | 计算值 | Table 4 |
| --- | --- | ---: | ---: |
| Vicuna-7B | \(106.76/37.07\) | 2.880 | 2.88 |
| Vicuna-13B | \(91.54/29.01\) | 3.155 | 3.16 |
| Vicuna-33B | \(40.43/17.87\) | 2.262 | 2.26 |
| Zephyr-7B | \(99.50/34.21\) | 2.909 | 2.91 |

四组数值均与原表四舍五入后的 Speedup 一致。这验证了表中的 Speedup 是实际吞吐量比值，而不是直接把 Acceleration Rate 当作最终加速比。

### 3.2 反推 Overhead

Table 4 没有单独列出 Overhead，可以用 \(O=A/S\) 反推：

| 模型 | Acceleration Rate | Speedup | 推算 Overhead |
| --- | ---: | ---: | ---: |
| Vicuna-7B | 3.23 | 2.88 | 1.122 |
| Vicuna-13B | 3.28 | 3.16 | 1.038 |
| Vicuna-33B | 2.85 | 2.26 | 1.261 |
| Zephyr-7B | 3.08 | 2.91 | 1.058 |

**注意**：这些 Overhead 是根据论文已四舍五入的 Acceleration Rate 和 Speedup 反推的近似值，不是 Table 4 直接报告的数据。

这揭示了一个重要现象：Zephyr-7B 的 Acceleration Rate（3.08）低于 Vicuna-7B（3.23），但 Speedup（2.91×）反而高于 Vicuna-7B（2.88×）。原因是 Zephyr-7B 的推算 Overhead 更低（约 1.058 vs 1.122）。因此，不能只优化每轮推进的 Token 数，还必须控制每轮候选处理和验证的时间开销。

### 3.3 四模型表现

- **Vicuna-7B**：基础速度 37.07 tokens/s，Medusa-2 达到 106.76 tokens/s，提升至 2.88×。
- **Vicuna-13B**：基础速度 29.01 tokens/s，Medusa-2 达到 91.54 tokens/s；Acceleration Rate 最高（3.28），Speedup 也最高（3.16×）。
- **Vicuna-33B**：基础速度 17.87 tokens/s，Medusa-2 达到 40.43 tokens/s；Acceleration Rate 最低（2.85），推算 Overhead 最高（约 1.261），最终 Speedup 最低（2.26×）。
- **Zephyr-7B**：基础速度 34.21 tokens/s，Medusa-2 达到 99.50 tokens/s；尽管 Acceleration Rate 只有 3.08，但由于推算 Overhead 较低，取得 2.91×。

因此，按 Speedup 排名：Vicuna-13B（3.16×）> Zephyr-7B（2.91×）> Vicuna-7B（2.88×）> Vicuna-33B（2.26×）。

## 四、与 Appendix E / MT-Bench 的对照

Appendix E 的 Figure 8 展示四种模型的 Medusa-2 加速；主文 Table 1 提供 MT-Bench 上更精确的 Acceleration Rate、Overhead 和 Speedup。按模型对齐：

| 模型 | MT-Bench Acc. Rate | MT-Bench Overhead | MT-Bench Speedup | AlpacaEval Acc. Rate | AlpacaEval Speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| Vicuna-7B | 3.47 | 1.22 | 2.83× | 3.23 | 2.88× |
| Vicuna-13B | 3.51 | 1.23 | 2.83× | 3.28 | 3.16× |
| Vicuna-33B | 3.01 | 1.27 | 2.35× | 2.85 | 2.26× |
| Zephyr-7B | 3.14 | 1.18 | 2.66× | 3.08 | 2.91× |

### 4.1 Speedup 差值

| 模型 | MT-Bench | AlpacaEval | 差值（AlpacaEval − MT-Bench） |
| --- | ---: | ---: | ---: |
| Vicuna-7B | 2.83× | 2.88× | +0.05 |
| Vicuna-13B | 2.83× | 3.16× | +0.33 |
| Vicuna-33B | 2.35× | 2.26× | −0.09 |
| Zephyr-7B | 2.66× | 2.91× | +0.25 |

这与 Appendix F 所称“与 MT-Bench 结果相近”总体一致，但两个数据集的数值并不完全相同。

### 4.2 Acceleration Rate 与 Overhead 的变化

四个模型在 AlpacaEval 上的 Acceleration Rate 均略低于 MT-Bench：

- Vicuna-7B：3.47 → 3.23；
- Vicuna-13B：3.51 → 3.28；
- Vicuna-33B：3.01 → 2.85；
- Zephyr-7B：3.14 → 3.08。

但 AlpacaEval 的 Speedup 并没有全部下降。结合 Table 4 反推的 Overhead，Vicuna-7B、Vicuna-13B 和 Zephyr-7B 的每轮额外开销估计均低于 MT-Bench；Vicuna-33B 的估计值则与 MT-Bench 接近。

这说明 Speedup 由接受/推进效率和单轮计算成本共同决定。候选 Token 数稍少，并不必然意味着整体吞吐更低。

**证据边界**：Appendix F 没有列出 Prompt 长度、生成长度、KV Cache 长度或逐算子时间。因此，不能确定两个数据集的 Overhead 差异具体由哪一个因素造成。Prompt/输出长度、Attention 工作量、候选树执行成本等只能作为待验证的工程假设。

### 4.3 绝对吞吐量不能跨数据集直接比较

Table 4 的 Base speed 和 MEDUSA speed 是 AlpacaEval 实验的实际吞吐量；Figure 8 属于前面的模型速度对比，不能把两个实验的绝对 tokens/s 混成同一个 Benchmark 结果。

跨数据集更适合对照：
- Speedup；
- Acceleration Rate；
- 在有直接报告时的 Overhead。

即便 Speedup 接近，也不能据此断言两组实验的硬件、序列长度和计时设置完全一致。

## 五、与 Appendix D 传统 Speculative Decoding 对照

Appendix D 的传统 Speculative Decoding 使用独立 Draft Model：

| Target | 最佳 Draft | \(\gamma\) |
| --- | --- | ---: |
| Vicuna-7B | Llama-68M | 4 |
| Vicuna-13B | Llama-68M | 3 |
| Vicuna-33B | Tiny-Vicuna（1B） | 3 |

主文 Table 1 报告的 MT-Bench 传统 SD Speedup：

| Target | 传统 SD（MT-Bench） | Medusa-2（MT-Bench） |
| --- | ---: | ---: |
| Vicuna-7B | 1.47× | 2.83× |
| Vicuna-13B | 1.56× | 2.83× |
| Vicuna-33B | 1.60× | 2.35× |

Table 4 没有提供 AlpacaEval 上传统 SD 的速度，因此不能将它与 Table 4 直接做同数据集的严格比较。可以确认的是：Medusa-2 在 AlpacaEval 上仍取得 2.26–3.16× Speedup；而在 MT-Bench 上，论文报告的 Medusa-2 Speedup 高于传统 SD。

技术上，传统 SD 的额外开销主要包含独立 Draft Model 的串行生成；Medusa-2 复用 Backbone，通过多个 Medusa Heads 并行预测，再用 Tree Attention 处理候选。Medusa 不需要单独部署 Draft LLM，但会引入多头计算、候选树和并行验证开销。

## 六、研究结论与复现注意事项

1. Table 4 的四组 Speedup 均能由 MEDUSA speed / Base speed 复算得到。
2. AlpacaEval 上最高 Speedup 是 Vicuna-13B 的 3.16×，最低是 Vicuna-33B 的 2.26×。
3. Zephyr-7B 说明 Acceleration Rate 不是唯一决定因素：其 Acceleration Rate 低于 Vicuna-7B，但由于单轮额外开销更低，最终 Speedup 略高。
4. Vicuna-33B 在 MT-Bench 与 AlpacaEval 上都处于四模型中较弱水平，且推算 Overhead 较高。
5. Appendix F 支持 Medusa-2 在不同数据集上保持相近加速比的结论，但没有给出足以解释数据集间 Overhead 变化的详细计时信息。
6. Appendix D 的传统 SD 结果属于 MT-Bench；Table 4 没有 AlpacaEval 传统 SD 对照。复现时应避免跨数据集得出严格方法优劣结论。
7. 复现 Table 4 时，应同时记录 Base/Medusa tokens/s、Acceleration Rate、每轮延迟、Prompt/生成长度、Batch Size、GPU、精度和计时边界；否则无法解释 Acceleration Rate 与 Speedup 的差异。

## 七、阅读小结与进度

- Appendix F 两句正文完整翻译。
- Table 4 五列表头、四行原始数据及表注完整翻译。
- 用实际吞吐量验证四个 Speedup。
- 反推四个 Overhead，并明确标注为计算值。
- 对比 Appendix E Figure 8 / 主文 Table 1 的 MT-Bench 结果。
- 对照 Appendix D 传统 Speculative Decoding，并标明数据集不一致的限制。
- Table 4 原始 PDF 页面已核对；PNG 二进制未保存，准确页码和来源记录在 [tables/table4-source.md](tables/table4-source.md)。

**下一阅读单元：Appendix G — Exploration and Modeling of Hardware Constraints and Medusa。** 主文与 Appendices A–F 已完成；Appendix G 尚待精读，整篇论文尚未完成。
