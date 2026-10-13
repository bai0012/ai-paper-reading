# DFlash: Block Diffusion for Flash Speculative Decoding

**中文标题：** DFlash：用于极速推测解码的块扩散方法

- arXiv: [2602.06036v2](https://arxiv.org/html/2602.06036v2)（2026-05-28）
- 作者：Jian Chen、Yesheng Liang、Zhijian Liu（UC San Diego）
- 会议：ICML 2026（PMLR 306）
- 论文页数：13（正文、参考文献与附录）
- [PDF](https://arxiv.org/pdf/2602.06036v2) | [项目主页](https://dflash.z-lab.ai)
- 阅读进度：[progress.json](progress.json)

## 研究问题

如何克服自回归 Draft 的顺序瓶颈，以及扩散 Draft 因规模大/去噪成本高/接受长度短而难以实际部署的问题？

## 核心方法

使用**轻量级 Block Diffusion Draft Model**，在一次 Forward 内并行预测候选 Token；从 Target Model 多层隐藏状态提取并融合上下文特征，向草稿 Transformer 每层注入条件 KV；以 Target Model 进行精确验证，保持正确采样机制下的输出分布。

## 关键实验

- Qwen3-8B、Transformers、贪心解码、MATH-500：DFlash 6.08× vs EAGLE-3 2.05×（Figure 1）。
- 更广泛结果、真实服务吞吐和并发退化见第 5 节，尚待精读。
- 注意：Figure 1 是特定部署条件，不能直接推断其他 GPU/Serving 后端。

## 章节导航

1. [Abstract：完整译文与讲解](00-Abstract.md) ✅
2. [1. Introduction：完整译文、原图 Figure 1 数据与讲解](01-Introduction.md) ✅
3. [2. Related Work：三个小节完整译文与讲解](02-Related-Work.md) ✅
4. [3. Preliminaries：完整译文、式 (1)–(3)、Figure 3 分析与推导](03-Preliminaries.md) ✅
5. [4. Method：完整译文、原图 Figure 2/4、KV 注入与训练讲解](04-Method.md) ✅
6. [5. Experiments：完整译文、Table 1–9、Serving 与消融实验](05-Experiments.md) ✅
7. 6. Conclusion（下一单元）
8. Appendix A（待读）

## Related Work 阅读小结

已阅读 Section 2：2.1 推测解码（Medusa/EAGLE 系列）、2.2 扩散语言模型（LLaDA/Block Diffusion/Fast-dLLM v2/SDAR）、2.3 扩散式推测解码（TiDAR/Samragh 等、DiffuSpec/SpecDiff-2）。这些方法揭示 Draft 串行性、模型大小及验收质量之间的取舍。全文与技术讲解见 [02-Related-Work.md](02-Related-Work.md)。下一节为第 3 节 Preliminaries。

## Preliminaries 阅读小结

已阅读 Section 3 的 3.1、3.2，形式化给出平均 Token 延迟 \(L=(T_{\mathrm{draft}}+T_{\mathrm{verify}})/\tau\)，比较顺序 Draft 的 \(T_{\mathrm{draft}}=\gamma t_{\mathrm{step}}\) 与扩散式 Draft 的 \(T_{\mathrm{draft}}=t_{\mathrm{parallel}}\)，并对原始 Figure 3 进行读图和推导。见 [03-Preliminaries.md](03-Preliminaries.md)，原图在论文 PDF 第 3 页，当前仓库未上传二进制裁图。下一节为 4. Method。

## Method 阅读小结

已完成 Section 4 的 4.1 Inference 和 4.2 Training：[完整译文及原图说明见章节笔记](04-Method.md)。关键机制包括：
- Prefill 后提取目标模型多个层的隐藏状态，跨层融合。
- 通过每层的 Key/Value 注入稳定条件信息，使草稿 Token 从缓存中查询目标上下文。
- 一次 Forward 并行预测 Mask Block。
- 随机 Anchor 和 Block-local 稀疏 Attention 使训练更符合推理行为，防止跨块标签泄漏。
- 使用位置损失权重 \(w_k=\exp(-(k-1)/\gamma)\)，优先提高草稿块前部 Token 的准确性；式 (4) 的 \(\gamma\) 为衰减超参数，和 Block Size 不应混同。

原始 [Figure 2（PDF 第 4 页）](https://arxiv.org/pdf/2602.06036v2#page=4) 与 [Figure 4（PDF 第 5 页）](https://arxiv.org/pdf/2602.06036v2#page=5) 已在聊天中裁图展示，当前 GitHub 使用原 PDF 来源链接，未存储 PNG 二进制。附录 A.3 的 KV 公式已交叉引用，但正式附录精读仍在后续。下一节为 **5. Experiments**。

## Experiments 阅读小结

[05-Experiments.md](05-Experiments.md) 已完成 5.1 Instruct Models、5.2 Reasoning Models、5.3 Serving Frameworks、5.4 Long Context Adaptation、5.5.1–5.5.5 Ablation Study 的逐段译文，以及 Table 1–9 完整数字和原表注。原表定位 PDF 第 6–9 页，已在对话中提取展示原始裁图；仓库尚无二进制 PNG 文件，不提供不存在的相对链接。

主要结论：
- Qwen3-8B 在 Transformers、Temperature=0 的平均加速 4.86×、平均 τ=6.49，EAGLE-3 (16) 为 1.76×、τ=2.96（Table 1）。
- B200+SGLang+FA4+Spec-v2 下，Qwen3-8B Math500 在并发 1→32 时的加速为 5.1×→2.8×，DFlash 吞吐 1175→16076 tok/s（Table 3）。
- Qwen3.5-27B 长上下文少量微调改善 τ，但没有报告完整长上下文 TPS/加速数据（Table 4）。
- 5 层 Draft 在三项任务上取得最高平均加速，但不代表每项任务均最优（Table 6）。
- KV Injection 在块扩散和自回归 Drafting 中均优于 Input Fusion（Table 9）。
- 大 Block Size 训练向较小推理 Block Size 的泛化较好，为未来动态 Block 调度提供依据（Table 8）。

下一阅读单元 **6. Conclusion**，随后按原始顺序阅读 Appendix A。

## 相关论文（仅表示技术关联，不代表已精读）

- [EAGLE-3](https://arxiv.org/abs/2503.01840)：基于隐藏特征的自回归草稿、树式验证
- [PARD](https://arxiv.org/abs/2504.18583)：低成本并行草稿适配
- [Block Diffusion](https://arxiv.org/abs/2503.09573)：块扩散生成
- [Your LLM Knows the Future](https://arxiv.org/abs/2507.11851)：目标隐藏特征中的未来 Token 信息

> 中文译文、作者原结论和独立讲解分别标注；图表引用论文实际位置，不使用类似图片冒充原图。
