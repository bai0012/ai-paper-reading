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
7. [6. Conclusion：完整译文、方法贡献、实验边界与 DFlash 关联](06-Conclusion.md) ✅
8. [Appendix A：A.1 Tree Structure、Figure 9、A.2 Multi-Round Speculative Sampling 与 Algorithm 1](Appendix-A.md) ✅
9. [Appendix B：详细实验结果与 Table 8 全部 18 行数据](Appendix-B.md) ✅

**全文综合总结：** [summary.md](summary.md)（全部章节核心贡献、数学机制、实验与证据边界、复现路线、与 DFlash 的跨论文关联）。

**阅读状态：** ✅ 全文精读完成（2026-10-23），包括 Abstract、Sections 1–6、Appendix A.1/A.2、Appendix B，Figure 1–9、Table 1–8 和 Algorithm 1。

## 原图存储状况

- [Figure 1 原图：PDF 第 1 页](https://arxiv.org/pdf/2401.15077v3#page=1)：已在本次聊天中直接从 PDF 裁图展示。
- [Figure 2 和 Figure 3 原图：PDF 第 2 页](https://arxiv.org/pdf/2401.15077v3#page=2)：已从原 PDF 提取，在对话中展示并完整翻译图注。
- [Figure 4 原图：PDF 第 3 页](https://arxiv.org/pdf/2401.15077v3#page=3)：已从原 PDF 提取，在对话中展示并完整翻译图注。
- [Figure 5 与 Figure 6 原图：PDF 第 4 页](https://arxiv.org/pdf/2401.15077v3#page=4)：已在本次对话裁图展示，并在 [Section 3](03-EAGLE.md) 完整翻译图注。
- [Figure 7 原图：PDF 第 7 页](https://arxiv.org/pdf/2401.15077v3#page=7)、[Figure 8 原图：PDF 第 8 页](https://arxiv.org/pdf/2401.15077v3#page=8)：已在本次对话裁图并完整翻译图注。Table 1–4 位于 PDF 第 6 页、Table 5–6 位于第 7 页、Table 7 位于第 8 页；聊天已提取原表，章节笔记保存完整数字。
- [Figure 9（PDF 第 12 页）](https://arxiv.org/pdf/2401.15077v3#page=12) 与 [Algorithm 1（PDF 第 13 页）](https://arxiv.org/pdf/2401.15077v3#page=13)：均已在当前对话从原版 PDF 裁图展示，完整翻译；Algorithm 1 的残差更新公式印刷为标量差，笔记中保留原式并解释数学上的向量式修正。
- [Table 8 原表：PDF 第 13 页](https://arxiv.org/pdf/2401.15077v3#page=13)：本次已直接从用户原始 PDF 提取完整裁图，在对话中显示，完整 18 行数字收录于 [Appendix B](Appendix-B.md)。
- 当前 GitHub 尚未存储 Figure 1–9、Algorithm 1 和 Table 1–8 的 PNG 二进制；完整译文及表格数值已保存，图像均给出原 PDF 精确页码链接，不使用不存在的 figures/ 路径。

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

## Conclusion 阅读小结

[Section 6 全文译文与独立综合评述](06-Conclusion.md) 已完成（原始 PDF 第 9 页，Section 6 仅一个自然段）。作者总结 EAGLE 在倒数第二层 Feature 空间的自回归 Draft 和提前一个时间步的 Shifted Token 如何缓解采样不确定性，并强调正确推测采样的分布保持。Conclusion 声称 MT-bench 相比 Vanilla 2.1–3.8×、Lookahead 1.7–2.1×、Medusa 1.5–1.6×；需注意 MT-bench Figure 1/2 的标注没有直接展示 3.8×，而 HumanEval Table 1 有 3.76×，所以在章节中特别记录了**作者结论的范围与图表归属未完全一致**。本节无新图表和公式；局限性、Serving/长上下文缺口、MoE 1.5×、Draft 串行深度、Tree 额外开销、EAGLE 与 DFlash 的区别均属于**独立分析**，不是作者在 Conclusion 中明确列出的 Limitations/未来计划。

## Appendix A 阅读小结

[Appendix A 完整译文、Figure 9、Algorithm 1 与严格采样推导](Appendix-A.md) 已完成：作者在 Figure 9 采用非均匀分叉树，高概率分支更深更宽，根节点 k=4；无论 Tree/Chain，正式实验草稿阶段各执行 5 次 Draft Forward，Target 验证 1 次 Forward。树结构凭经验选择、未严格优化，Batch 较大时小树可能更佳。A.2 对同一位置多个候选进行多轮验收，每次拒绝后进一步尝试候选，全部拒绝后再抽残差。**重要核对：v3 Algorithm 1 印刷的残差更新为 `norm(max(0,p(t_i)−p_hat_i(t_i)))`，这是标量而非词表分布**；章节严格保留原式，并在额外分析中推导合法条件提议分布前提下的向量修正规则 `norm([p−q_i]_+)` 与分布守恒证明。Figure 9/Algorithm 1 原 PNG 在对话展示，未上传 GitHub 的图像二进制不能标记已存。下一节 Appendix B。

## Appendix B 阅读小结

[Appendix B 完整翻译和 Table 8 全部数据](Appendix-B.md) 与 PDF 第 12–13 页对应：原文仅一自然段和一张表，列出 HumanEval/GSM8K/Alpaca、6 个 Target 模型、T=0 的 Speedup、平均接受长度及 0-α 至 4-α。原始 18 行均已保留。对每项任务的 6 模型做不加权算术平均：HumanEval 3.505× / τ 4.3567，GSM8K 3.080× / τ 3.9483，Alpaca 2.925× / τ 3.7883（为独立后处理，不是论文另报的聚合指标）。0-α→1-α 通常降幅较大，而后续 α 并非严格单调；原表没有置信区间与显著性检验。Table 8 重列正文 Table 1 中 T=0 的速度与 τ，并补充详细接受率。完整理论、实验、局限、复现建议见 [summary.md](summary.md)。

## 跨论文关联

- [DFlash（仓库已精读）](../2602.06036-dflash/README.md)：EAGLE 的特征级自回归 Draft 与 DFlash 的 Block Diffusion 并行 Draft；复用 Target 隐藏状态与减少 Draft 串行成本。
- [EAGLE-3](https://arxiv.org/abs/2503.01840)：后续版本，独立论文，不混同于本篇 EAGLE。

**最近阅读：** 2026-10-23，Abstract、Sections 1–6、Appendix A/B 均已完成；[全文总结](summary.md) 已生成。
