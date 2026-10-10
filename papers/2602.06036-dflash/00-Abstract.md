# Abstract — 完整中文翻译与讲解

- 论文：[DFlash: Block Diffusion for Flash Speculative Decoding](https://arxiv.org/html/2602.06036v2)
- PDF 第 1 页
- 完成日期：2026-10-10
- 说明：本节译文与独立技术讲解明确分离。

## 完整中文翻译

自回归大语言模型（Autoregressive Large Language Models, LLMs）表现出很强的性能，却需要在本质上串行执行的解码过程，导致推理延迟高、GPU 利用率低。推测解码（Speculative Decoding）使用一个快速草稿模型（Draft Model）来缓解这一瓶颈，草稿模型的输出由目标大语言模型（Target LLM）并行验证。然而，现有方法仍依赖自回归草稿生成，其过程依然串行，从而限制实际加速效果。

扩散语言模型（Diffusion LLMs）通过实现并行生成，提供了一种有前景的替代方案，但目前扩散模型的性能通常落后于自回归模型。本文提出 DFlash：一个使用轻量级块扩散模型（Lightweight Block Diffusion Model）进行并行草稿生成的推测解码框架。我们表明，推测解码为扩散模型提供了一个自然且有效的应用场景。

DFlash 通过在单次前向传播（Single Forward Pass）中生成草稿 Token，实现高效的草稿生成；同时，让 Draft Model 以从目标模型中提取的上下文特征（Context Features）为条件，生成高质量草稿并提高接受率。实验表明，DFlash 在一系列模型和任务中实现超过 6 倍的无损加速，相比先进推测解码方法 EAGLE-3，可提供最高 2.5 倍的额外加速比。

**原文附带链接：** [代码 GitHub](https://github.com/z-lab/dflash)；[模型 Hugging Face](https://huggingface.co/collections/z-lab/dflash)（项目主页：[dflash.z-lab.ai](https://dflash.z-lab.ai)）。

## 技术讲解（独立分析）

1. **自回归 Draft 的限制：** 即使 Target Model 能批量验证，Draft 阶段仍需顺序调用，形成独立的性能瓶颈。
2. **扩散 Draft 的长处：** 同一块内的多个 Mask 位置可并行预测，单次前向传播给出一组候选 Token。
3. **为何仍保持无损：** 最终以目标模型的验证与采样决策为准；在正确实现接受/拒绝采样算法时，Draft 只影响计算量，不改变目标分布。
4. **为何使用目标特征：** 小模型无需从零学习完整的下文生成能力，而是对目标模型的高维语义表示做轻量适配。
5. **评估时需要注意：** 论文的加速数字依赖 GPU、后端、上下文、温度及并发条件，不应视为普适保证。

## 下一个阅读单元

[1. Introduction](01-Introduction.md)（已完成，2026-10-10）。
