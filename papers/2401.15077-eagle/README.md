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
4. [3. EAGLE：完整译文、Figure 5/6、网络结构、双损失与树式验证](03-EAGLE.md) ✅
5. [4. Experiments：完整译文、Table 1–7、Figure 7–8、消融与吞吐](04-Experiments.md) ✅
6. [5. Related Work：完整译文及 Medusa、Lookahead、SpecInfer、Self-Speculative 与传统推测采样对比](05-Related-Work.md) ✅
7. 6. Conclusion — ⏳ 下一阅读单元
8. Appendix A（A.1 Tree Structure、A.2 Multi-Round Speculative Sampling）— 未读
9. Appendix B（详细实验结果）— 未读

## 原图存储状况

- [Figure 1 原图：PDF 第 1 页](https://arxiv.org/pdf/2401.15077v3#page=1)：已在本次聊天中直接从 PDF 裁图展示。
- [Figure 2 和 Figure 3 原图：PDF 第 2 页](https://arxiv.org/pdf/2401.15077v3#page=2)：已从原 PDF 提取，在对话中展示并完整翻译图注。
- [Figure 4 原图：PDF 第 3 页](https://arxiv.org/pdf/2401.15077v3#page=3)：已从原 PDF 提取，在对话中展示并完整翻译图注。
- [Figure 5 与 Figure 6 原图：PDF 第 4 页](https://arxiv.org/pdf/2401.15077v3#page=4)：已在本次对话裁图展示，并在 [Section 3](03-EAGLE.md) 完整翻译图注。
- [Figure 7 原图：PDF 第 7 页](https://arxiv.org/pdf/2401.15077v3#page=7)、[Figure 8 原图：PDF 第 8 页](https://arxiv.org/pdf/2401.15077v3#page=8)：已在本次对话裁图并完整翻译图注。Table 1–4 位于 PDF 第 6 页、Table 5–6 位于第 7 页、Table 7 位于第 8 页；聊天已提取原表，章节笔记保存完整数字。
- 当前 GitHub 尚未存储 Figure 1–8 与 Table 1–7 的 PNG 二进制；使用原 PDF 图页链接，不使用不存在的 figures/ 相对地址冒充原图。

## Introduction 阅读小结

[Section 1 完整译文与讲解](01-Introduction.md) 包括传统小型 Draft 模型的成本、Lookahead/Medusa 对比、Figure 2 非贪心加速、Figure 3 特征分支的不确定性、Figure 4 Token/Feature/Feature+Shifted-Token 消融，以及条件方差和 Token 时间索引解释。原文报告：Feature 约 1.9× vs Token 约 1.5×；加 Shifted Token 后约 2.8×；单张 RTX 3090 + gpt-fast 的 LLaMA2-Chat 7B 为 160.4 tokens/s（原文实验，非普遍承诺）。

## Preliminaries 阅读小结

[Section 2 完整译文与数学推导](02-Preliminaries.md)：明确符号 $t,e,f,p,T_{i:j}$ 和 Target/Draft；标准 Speculative Sampling 在一次目标模型 Causal Forward 中得到候选位置分布，再依次以 $\min(1,p/q)$ 接受草稿，第一次拒绝后依据 $\mathrm{norm}([p-q]_+)$ 补足概率质量，丢弃后续草稿。独立证明 $q(x)a(x)+(1-\alpha)r(x)=p(x)$，推导 $\alpha=1-\mathrm{TV}(p,q)$，提供三 Token 样例、PyTorch 例程及多 Token 条件正确性的解释；注意 Section 3 和 Appendix A.2 的树状验证与本节基础链式算法不同。本节 PDF 第 3 页无新增 Figure/Table。

## EAGLE 方法精读小结

[Section 3 完整译文、原图与技术分析](03-EAGLE.md)：Figure 5 比较 Speculative Sampling、Lookahead、Medusa 与 EAGLE，Figure 6 用 3 次 Draft Forward 构造 10 Token 的示例树；历史 Target Features 与提前一时间步的 Token Embedding 拼接后经 FC+单层 Decoder 预测 Feature，共享冻结的 Target Embedding/LM Head。训练损失为 Smooth L1 回归 + 0.1×软标签分类 Cross Entropy，特征噪声为 U(-0.1,0.1)；Tree Attention 使 Target 用一次 Forward 评估整棵草稿树，再递归应用精确推测采样，接受的 Token 和对应 Target Feature 用于下一轮草稿。详细树结构和多轮推测算法见待读 Appendix A.1/A.2。

## Experiments 阅读小结

[Section 4 全部译文和 Table 1–7 / Figure 7–8](04-Experiments.md) 已完成：HumanEval、GSM8K、Alpaca 上 T=0/1 加速与平均接受长度；MT-bench 的 $0\text{-}\alpha$ 至 $4\text{-}\alpha$、Mixtral 8x7B 的 1.50× 与 MoE 权重路由开销；RTX 3090 上 EAGLE+gpt-fast int4 达 160.4 tokens/s；Tree Attention 提高平均接受长度约 0.6–0.8、加速比约 0.3–0.5；Feature+Shifted Token 消融优于 Token/Feature/Unshifted Token；Target 生成数据 2.88× vs 固定 ShareGPT 2.78×；不同 Batch Size 与最大吞吐约 1.97×/1.99×。完整测试条件、原始表格和解释边界见章节文件。

## Related Work 阅读小结

[Section 5 全文译文与技术谱系](05-Related-Work.md)：按原文顺序完成三个段落（PDF 第 8–9 页），归纳知识蒸馏/量化/剪枝与 Speculative Decoding 的不同优化层级，区分严格分布保持的标准推测采样和 DistillSpec/BiLD/Medusa 某些放宽接受规则；系统比较标准 Speculative Sampling、小模型草稿、Self-Speculative 跳层、SpecInfer 树验证、Medusa 多头预测、Lookahead 的 Jacobi+n-gram、Cascade/Staged/Online、REST 与 LLMA。**EAGLE 的独特重点是 Feature-level autoregressive Draft + Shifted Token，而非首次提出 Tree Attention。** 本节没有新 Figure/Table。论文原句“do not employ a separate target LLM”已照字面翻译，同时标注与这些方法实际使用 Target 主模型的术语不一致。

## 跨论文关联

- [DFlash（仓库已精读）](../2602.06036-dflash/README.md)：EAGLE 的特征级自回归 Draft 与 DFlash 的 Block Diffusion 并行 Draft；复用 Target 隐藏状态与减少 Draft 串行成本。
- [EAGLE-3](https://arxiv.org/abs/2503.01840)：后续版本，独立论文，不混同于本篇 EAGLE。

**最近阅读：** 2026-10-21，Abstract、Section 1–5 已完成；下一步 Section 6 Conclusion。
