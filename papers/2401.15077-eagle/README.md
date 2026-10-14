# EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty

**中文标题：** EAGLE：推测采样需要重新思考特征的不确定性

- arXiv：[2401.15077v3](https://arxiv.org/html/2401.15077v3)（2025-03-04）
- 作者：Yuhui Li、Fangyun Wei、Chao Zhang、Hongyang Zhang
- 机构：北京大学、微软研究院、滑铁卢大学、Vector Institute
- PDF：<https://arxiv.org/pdf/2401.15077v3>（13 页，含附录）
- 官方代码：<https://github.com/SafeAILab/EAGLE>
- 本仓库：[阅读进度](progress.json)

## 研究问题

推测采样中的 Draft Model 既需要足够快，又要与目标 LLM 的候选分布接近。依赖独立小模型的传统方法存在合适模型难找、Draft 计算开销高的问题。

## 核心贡献（摘要和引言导读，后续逐节核对）

1. 观察：目标模型倒数第二层隐藏特征序列具有比直接 Token 序列更易外推的结构。
2. 观察：若只用历史特征预测未来特征，Token 采样引入的分支不确定性会阻碍预测。
3. 方法：用**上一阶段已采样、相对 Feature 提前一时间步的 Token**为特征预测提供条件；低成本预测下一个 Feature，再借用目标 LM Head 形成 Token 分布。
4. 目标模型采用精确推测采样验证，不需要微调主干参数，以维持目标模型生成分布。

## 摘要重要结果

LLaMA2-Chat 70B 报告约 2.7–3.5× 延迟加速和约 2× 吞吐增益。实验更具体的任务、温度、批大小以及可比较基线须在 Section 4 逐表核对。

## 章节导航与状态

1. [Abstract：完整译文、Figure 1 原图图注与技术导读](00-Abstract.md) ✅
2. [1. Introduction：完整译文、Figure 2–4 图注与机制分析](01-Introduction.md) ✅
3. [2. Preliminaries：完整译文、接受概率、残差分布与严格证明](02-Preliminaries.md) ✅
4. 3. EAGLE（3.1 Drafting、3.2 Training、3.3 Verification）— ⏳ 下一阅读单元
5. 4. Experiments（4.1–4.4）— 未读
6. 5. Related Work — 未读
7. 6. Conclusion — 未读
8. Appendix A（A.1 Tree Structure、A.2 Multi-Round Speculative Sampling）— 未读
9. Appendix B（详细实验结果）— 未读

## 原图存储状况

- [Figure 1 原图：PDF 第 1 页](https://arxiv.org/pdf/2401.15077v3#page=1)：已在本次聊天中直接从 PDF 裁图展示。
- [Figure 2 和 Figure 3 原图：PDF 第 2 页](https://arxiv.org/pdf/2401.15077v3#page=2)：已从原 PDF 提取，在对话中展示并完整翻译图注。
- [Figure 4 原图：PDF 第 3 页](https://arxiv.org/pdf/2401.15077v3#page=3)：已从原 PDF 提取，在对话中展示并完整翻译图注。
- 当前 GitHub 尚未存储 Figure 1–4 的 PNG 二进制；不使用不存在的 figures/ 相对地址冒充原图。

## Introduction 阅读小结

[Section 1 完整译文与讲解](01-Introduction.md) 包括传统小型 Draft 模型的成本、Lookahead/Medusa 对比、Figure 2 非贪心加速、Figure 3 特征分支的不确定性、Figure 4 Token/Feature/Feature+Shifted-Token 消融，以及条件方差和 Token 时间索引解释。原文报告：Feature 约 1.9× vs Token 约 1.5×；加 Shifted Token 后约 2.8×；单张 RTX 3090 + gpt-fast 的 LLaMA2-Chat 7B 为 160.4 tokens/s（原文实验，非普遍承诺）。

## Preliminaries 阅读小结

[Section 2 完整译文与数学推导](02-Preliminaries.md)：明确符号 $t,e,f,p,T_{i:j}$ 和 Target/Draft；标准 Speculative Sampling 在一次目标模型 Causal Forward 中得到候选位置分布，再依次以 $\min(1,p/q)$ 接受草稿，第一次拒绝后依据 $\mathrm{norm}([p-q]_+)$ 补足概率质量，丢弃后续草稿。独立证明 $q(x)a(x)+(1-\alpha)r(x)=p(x)$，推导 $\alpha=1-\mathrm{TV}(p,q)$，提供三 Token 样例、PyTorch 例程及多 Token 条件正确性的解释；注意 Section 3 和 Appendix A.2 的树状验证与本节基础链式算法不同。本节 PDF 第 3 页无新增 Figure/Table。

## 跨论文关联

- [DFlash（仓库已精读）](../2602.06036-dflash/README.md)：EAGLE 的特征级自回归 Draft 与 DFlash 的 Block Diffusion 并行 Draft；复用 Target 隐藏状态与减少 Draft 串行成本。
- [EAGLE-3](https://arxiv.org/abs/2503.01840)：后续版本，独立论文，不混同于本篇 EAGLE。

**最近阅读：** 2026-10-14，Abstract、Section 1 Introduction 和 Section 2 Preliminaries 已完成；下一步 Section 3 EAGLE。
