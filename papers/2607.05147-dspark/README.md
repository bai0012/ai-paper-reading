# DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation

**中文标题：** DSpark：基于置信度调度与半自回归生成的推测解码

- arXiv：[2607.05147v1](https://arxiv.org/abs/2607.05147)（2026-07-06）
- 作者：Xin Cheng、Xingkai Yu、Chenze Shao、Jiashi Li、Yunfan Xiong 等（北京大学、DeepSeek-AI）
- 论文：33 页，7 个正文章节、Appendix A
- PDF：https://arxiv.org/pdf/2607.05147v1
- [阅读进度](progress.json)

## 研究问题

如何兼顾并行 Draft 的低延迟与自回归 Draft 的 Token 依赖建模能力，同时避免高并发服务中不加区分地验证长草稿块造成吞吐下降？

## 核心贡献

1. **半自回归 Draft**：重型并行 DFlash 式 Backbone + 轻量 Markov/RNN 顺序头，以局部条件修正缓解 suffix decay。
2. **置信度调度验证**：逐位置条件接受概率估计 + Sequential Temperature Scaling（STS）校准，获得前缀存活概率。
3. **硬件感知前缀调度**：结合引擎 profiled SPS(B) 吞吐曲线和当前 batch 的请求分布，选择逐请求的目标验证长度，减少低价值验证。
4. **生产部署**：在 DeepSeek-V4-Flash/Pro preview 线上环境评估延迟/吞吐 Pareto 前沿。

## 关键实验（预览；尚未逐节精读）

- Qwen3-4B/8B/14B 的离线宏平均接受长度：对 Eagle3 +30.9%/+26.7%/+30.0%；对 DFlash +16.3%/+18.4%/+18.3%。
- DeepSeek-V4-Flash 相同吞吐下单用户生成速度 +60%～85%；V4-Pro +57%～78%，均相较 MTP-1 生产基线。
- 离线接受长度和生产 TPS 指标不同；Section 4/5 的条件需分别校核。

## 章节导航与精读进度

1. [Abstract：完整翻译与技术讲解](00-Abstract.md) ✅
2. [1. Introduction：完整译文、推测解码架构取舍与服务调度分析](01-Introduction.md) ✅
3. [2. Background：2.1–2.2 全文翻译、公式 (1)–(3) 与 KV Injection 分析](02-Background.md) ✅
4. 3. Architecture（下一单元）：3.1 Semi-Autoregressive Generation；3.2 Confidence-Scheduled Verification；3.3 Training
5. 4. Experiments：4.1 Experimental Setup；4.2 Results；4.3 Analysis
6. 5. Real-World Deployment of DSpark：5.1 Scalable Training；5.2 Scheduler in Practice；5.3 Inference；5.4 Live Traffic
7. 6. Related Work
8. 7. Conclusion
9. Appendix A：Counterexample: Selection Bias Without Early-Stopping
10. summary.md（待读完全文后生成）

## Introduction 精读结论

- [Section 1 完整中文译文、两大瓶颈、半自回归与置信度调度分析](01-Introduction.md)：全部 7 个正文段落和 2 个贡献条目；对 `T_draft / T_verify / τ` 的讨论标注为 Section 2 预读，而非 Intro 原公式。
- 论文明确区分 **数据侧**（Code vs Chat 的接受率差异）与 **系统侧**（低负载多验证近乎免费，高并发验证占用 Batch 容量），两者共同决定最优验证长度。
- 离线接受长度（Qwen3 4B/8B/14B vs Eagle3 与 DFlash）与线上用户 TPS（V4-Flash/Pro vs MTP-1）分别解释，不能直接横向视作同一加速率。
- **图表状态：** Section 1 无原图原表；下一节 Background 需要解释 Equation (1)–(3)，Figure 1 在 Section 3 / PDF 第 5 页。

## Background 精读结论

- [Section 2 完整译文、原始公式 (1)–(3)、DFlash KV Injection 维度与数据流分析](02-Background.md)。Section 2.1 给出 $L=(T_{draft}+T_{verify})/\tau$ 三个优化方向；2.2 区分自回归 Draft 的串行成本和并行 Draft 的一次前向。
- DFlash 使用 Target 多层隐藏状态特征拼接、共享线性映射和 RMSNorm 得到 $H_{ctx}$，并将其作为每个 Draft Layer 的 Key/Value 上下文；Mask Block 在 Backbone 内可双向 Attention，但**缺少已采样 Token 条件化**，为 DSpark 的顺序头留下空间。
- 严格区分第 (2) 式**特征维拼接**与第 (3) 式**序列维拼接**；$	au$ 通常包含 Bonus Token（参见论文 Section 4 注脚）；本文讨论了拒绝采样的残差修正但明确标注为独立理论扩展。
- Section 2 **没有 Figure/Table**，无图片待提交；下一节 Section 3 开始 Figure 1 及 Markov/RNN Head。

## 阅读问答

- [帕累托前沿：总吞吐量与单用户 TPS 权衡、Figure 7 解读](questions.md)

## 跨论文关联

- [DFlash](../2602.06036-dflash/README.md)：DSpark 的并行 Backbone 基于其 Target 特征条件下的并行 Draft；DSpark 进一步通过轻量自回归头补充块内依赖并以调度减少多余 Target 验证。
- [EAGLE](../2401.15077-eagle/README.md)：原版 EAGLE 强调特征级自回归和采样不确定性；论文实验比较的 **Eagle3** 是后续的 EAGLE-3，须注意版本差异。
- [Medusa](../2401.10774-medusa/README.md)：多个 Future Token Head 的并行候选与树状验证；与 DSpark 的半自回归块生成/按负载调度有结构性差别。

**当前阅读位置：Abstract、Section 1、Section 2（2.1–2.2）已完成；下一单元为 Section 3 Architecture，PDF 第 4–5 页起。**

**图表情况：** Abstract、Section 1、Section 2 均无 Figure/Table；Figure 1 位于 Section 3 PDF 第 5 页。未上传任何 PNG 二进制，不创建虚假的图片链接。
