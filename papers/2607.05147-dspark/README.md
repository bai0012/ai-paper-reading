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
2. 1. Introduction（下一单元）
3. 2. Background：2.1 Speculative Decoding；2.2 Drafter Architectures
4. 3. Architecture：3.1 Semi-Autoregressive Generation；3.2 Confidence-Scheduled Verification；3.3 Training
5. 4. Experiments：4.1 Experimental Setup；4.2 Results；4.3 Analysis
6. 5. Real-World Deployment of DSpark：5.1 Scalable Training；5.2 Scheduler in Practice；5.3 Inference；5.4 Live Traffic
7. 6. Related Work
8. 7. Conclusion
9. Appendix A：Counterexample: Selection Bias Without Early-Stopping
10. summary.md（待读完全文后生成）

## 阅读问答

- [帕累托前沿：总吞吐量与单用户 TPS 权衡、Figure 7 解读](questions.md)

## 跨论文关联

- [DFlash](../2602.06036-dflash/README.md)：DSpark 的并行 Backbone 基于其 Target 特征条件下的并行 Draft；DSpark 进一步通过轻量自回归头补充块内依赖并以调度减少多余 Target 验证。
- [EAGLE](../2401.15077-eagle/README.md)：原版 EAGLE 强调特征级自回归和采样不确定性；论文实验比较的 **Eagle3** 是后续的 EAGLE-3，须注意版本差异。
- [Medusa](../2401.10774-medusa/README.md)：多个 Future Token Head 的并行候选与树状验证；与 DSpark 的半自回归块生成/按负载调度有结构性差别。

**当前阅读位置：Abstract 已完成；下一个单元为 Section 1 Introduction。**

**图表情况：** Abstract 无图表；尚未提取或提交任何 Figure/PNG 文件，后续章节严格标记原 PDF 页码与实际保存状态。
