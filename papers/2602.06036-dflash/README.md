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
3. 2. Related Work（待读）
4. 3. Preliminaries（待读）
5. 4. Method（待读）
6. 5. Experiments（待读）
7. 6. Conclusion（待读）
8. Appendix A（待读）

## 相关论文（仅表示技术关联，不代表已精读）

- [EAGLE-3](https://arxiv.org/abs/2503.01840)：基于隐藏特征的自回归草稿、树式验证
- [PARD](https://arxiv.org/abs/2504.18583)：低成本并行草稿适配
- [Block Diffusion](https://arxiv.org/abs/2503.09573)：块扩散生成
- [Your LLM Knows the Future](https://arxiv.org/abs/2507.11851)：目标隐藏特征中的未来 Token 信息

> 中文译文、作者原结论和独立讲解分别标注；图表引用论文实际位置，不使用类似图片冒充原图。
