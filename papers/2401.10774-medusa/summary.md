# Medusa 论文综合总结

**论文：** Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads  
**作者：** Tianle Cai, Yuhong Li, Zhengyang Geng, Hongwu Peng, Jason D. Lee, Deming Chen, Tri Dao  
**版本：** arXiv:2401.10774v3，2024-06-14  
**会议：** ICML 2024，PMLR 235  
**精读状态：** 主文 Abstract、Introduction、Sections 2–4、Appendices A–G 全部完成。  
**原文：** [HTML](https://arxiv.org/html/2401.10774v3) · [PDF](https://arxiv.org/pdf/2401.10774v3) · [官方代码](https://github.com/FasterDecoding/Medusa)

## 1. 研究问题与核心思想

自回归 LLM 解码每轮只生成一个 Token。Batch=1 时，模型每轮需要从 HBM 读取大量参数，但只进行一次 Token 对应的计算，导致 GPU 算力未充分利用，Decode 经常受显存带宽限制。

Medusa 的核心思路是不再依赖独立 Draft Model，而是在 Target Backbone 的最后隐藏状态上增加多个轻量预测 Head，让它们并行预测多个未来 Token。随后把各 Head 的 Top-k 组合成候选树，通过 Tree Attention 一次性验证多条候选路径，接受最长有效前缀，从而减少 Target 自回归迭代次数。

## 2. 核心方法

### 2.1 Medusa Heads

原始 LM Head 预测下一个 Token，额外的第 \(k\) 个 Medusa Head 预测更远的未来位置。每个 Head 采用单层前馈网络和残差连接：

\[
p_t^{(k)}=
\mathrm{softmax}\left(
W_2^{(k)}
\left[
\mathrm{SiLU}(W_1^{(k)}h_t)+h_t
\right]\right)
\]

原 PDF 中 \(W_2\) 的矩阵维度声明与乘法记法存在行/列向量约定差异，精读章节保留原式并说明自洽实现，不将其隐藏。

### 2.2 Tree Attention

各 Head 输出多个 Top-k 候选后形成组合树。树节点共享前缀，Attention Mask 仅允许节点关注其祖先和合法上下文，使多条候选路径能够在一次 Target Forward 中并行验证。Position IDs 按树深度设置。

候选树节点数量取决于每个 Head 的 Top-k 分支数；树结构可以通过验证统计和贪心搜索优化，而不必保留完整笛卡尔积。

### 2.3 Medusa-1 与 Medusa-2

- **Medusa-1：** 冻结 Backbone，仅训练额外 Heads。优点是保持原模型参数不变，训练成本低；Heads 的预测能力受到冻结特征限制。
- **Medusa-2：** 联合训练 Backbone 和 Heads，提高未来 Token 预测准确率和加速率；需要差异化学习率、Heads Warmup、两阶段训练或自蒸馏等策略，避免破坏原模型 next-token 能力。

### 2.4 Typical Acceptance 与 Self-Distillation

标准 Rejection Sampling 能保持目标分布，但额外采样/拒绝开销和严格分布约束可能限制接受长度。Typical Acceptance 用阈值选择合理候选，以类似生成质量换取更高接受效率，但不应被描述为严格保持原模型完整采样分布。

Self-Distillation 在缺少原始训练集或模型经过 Alignment/RLHF 时，让原模型生成训练数据，并用原始模型分布作为 Teacher 约束 Backbone。这样避免单纯拟合生成 Token 而破坏模型原有输出分布。

### 2.5 Optimized Tree

使用验证数据估计各 Head 的 Top-k 排名准确率，按路径预测概率或期望接受长度构建树，并通过连通约束贪心搜索剪枝。Appendix C 的 64 节点树由 63 个 Medusa 候选节点加 LM Head 根节点组成；树深对应 Head 预测步数。

## 3. 主要实验结论

### 3.1 主文结果

论文报告：
- Medusa-1 在不损失生成质量的情况下可以超过 2.2× 加速；
- Medusa-2 的 Speedup 约 2.3–2.8×；
- 主实验重点是 Batch Size=1 的单请求场景。

主文 Table 1 中 Medusa-2 的 Speedup：
- Vicuna-7B：2.83×
- Zephyr-7B：2.66×
- Vicuna-13B：2.83×
- Vicuna-33B：2.35×

Speedup 由 Acceleration Rate 与 Overhead 共同决定：

\[
\mathrm{Speedup}=
\frac{\mathrm{Acceleration\ Rate}}{\mathrm{Overhead}}
\]

因此不能只优化每轮接受 Token 数而忽略每轮计算开销。

### 3.2 传统 Speculative Decoding 对照（Appendix D）

最佳 Draft 配置：
- Vicuna-7B：Llama-68M，\(\gamma=4\)
- Vicuna-13B：Llama-68M，\(\gamma=3\)
- Vicuna-33B：Tiny-Vicuna/Vicuna-1B，\(\gamma=3\)

论文主文 Table 1 中传统 Speculative Decoding 的 Speedup 为 1.47×、1.56×、1.60×；对应 Medusa-2 为 2.83×、2.83×、2.35×。Appendix D 未提供 Zephyr 的传统 SD 对照。

### 3.3 AlpacaEval（Appendix F）

Table 4 原始数据：

| Model | Base speed | MEDUSA speed | Acc. rate | Speedup |
|---|---:|---:|---:|---:|
| Vicuna-7B | 37.07 | 106.76 | 3.23 | 2.88 |
| Vicuna-13B | 29.01 | 91.54 | 3.28 | 3.16 |
| Vicuna-33B | 17.87 | 40.43 | 2.85 | 2.26 |
| Zephyr-7B | 34.21 | 99.50 | 3.08 | 2.91 |

速度单位 tokens/s。所有 Speedup 都通过 Medusa speed / Base speed 复算。Table 4 未直接报告 Overhead，精读中按定义反推并标记为推算值。

### 3.4 硬件 Roofline（Appendix G）

Table 5 将 Prefill、Decode、Parallel Decode 的 QKV、Attention 和 FFN 复杂度并列。核心差异：
- Decode Linear：\(O(bh^2)\)、\(O(bhi)\)；
- Medusa Parallel Linear：\(O(bqh^2)\)、\(O(bqhi)\)；
- Decode Attention：\(O(bsnd)\)；
- Medusa Attention：\(O(bsqnd)\)。

Medusa 通过同一轮处理 \(q\) 个候选，提高每次权重读取所完成的计算量。

三种 GPU 的论文 Roofline 参数：
- A100 80GB PCIe：1935 GB/s、312 TFLOP/s；
- A40：696 GB/s、149.7 TFLOP/s；
- A6000：768 GB/s、181 TFLOP/s。

A100 Ridge Point 约 161.2 FLOP/Byte。Table 6–7 显示 Attention 的 OI 最高约 56.44，仍低于 Ridge Point；Table 8 中 Linear 的 OI 可达到 2893.91，实际吞吐达到约 246.91 TFLOP/s，表现为 Compute-Bound。

### 3.5 性能预测（Appendix G）

作者使用 Llama-7B 简化 Block，依次模拟 QKV、QK/PV、Up/Gate/Down，并忽略验证/接受后处理延迟：
- Batch=1、Sequence Length=1024 时，候选数超过约 64 后 Speedup 开始下降；
- Sequence Length=1024 时，Batch Size 超过 32 后 Speedup 下降，Batch=64 甚至可能低于 1；
- Batch=4 时，Sequence Length 越长，整体 Speedup 越低，原因主要是 Attention 历史 KV 访问开销增加。

**公式边界：** Appendix G.3 打印的 \(0.477\log(\mathrm{num\ candidate})\) 没有截距且未说明对数底数，与 Figure 21 在候选数 1 时 Acceleration Rate=1 的基线不一致。精读保留该不一致，没有擅自修正式子。

## 4. 主要创新与技术价值

1. **无独立 Draft Model：** 多个 Heads 复用 Target Backbone 隐藏状态，减少部署和分布式服务中维护第二模型的复杂度。
2. **多候选并行验证：** Tree Attention 让一次 Target Forward 检查多条候选路径。
3. **训练策略分层：** Medusa-1 保护原模型能力，Medusa-2 通过联合训练进一步提高 Head 准确率。
4. **采样与树搜索扩展：** Typical Acceptance、Self-Distillation、Optimized Tree 分别改善接受效率、训练数据可得性和候选树效率。
5. **硬件解释：** Appendix G 说明 Medusa 的加速来自提高 Decode 算子的并行度和运算强度，而不是减少模型 FLOPs；但 Attention 与大 Batch 下的 Compute-Bound 仍会限制收益。

## 5. 局限性

1. 主实验以 Batch Size=1 为主，不能直接外推到高并发 Serving。
2. 候选数增加会提升计算开销；树越大不一定越快。
3. 长上下文中 Attention/KV 访问开销会抵消候选并行收益。
4. Typical Acceptance 不保证与原模型严格同分布。
5. Self-Distillation 可能存在生成数据与原始训练分布错配。
6. Figure 8 图注把 Vicuna-13B 列为自蒸馏模型，但 Section 3.1/Appendix B.3 对训练流程的描述不一致；保留原文冲突。
7. Appendix G 的单 Block 简化模型忽略验证/接受后处理，不能作为真实 Serving 性能的精确预测器。
8. G.3 对数拟合式缺少解释，打印公式与 Figure 21 基线存在数学不一致。
9. Figure 9–23 和 Table 5–8 的 PDF 原页已核对、图注和数据已保存，但 PNG 二进制未提交到 GitHub。

## 6. 复现建议

1. **先复现基线：** 固定模型、Tokenizer、精度、GPU、Prompt/生成长度，测量 Prefill、Decode、Tokens/s、TPOT 和显存。
2. **再复现算子：** 单独 Profile QKV、QK/PV、Up/Gate/Down，记录 TFLOP/s、HBM Throughput、OI、Kernel Latency。
3. **建立 Roofline：** 用实际 GPU 的峰值计算能力和 HBM 带宽计算 Ridge Point，不直接套用 A100 参数。
4. **候选数扫描：** 从 1、16、32、48、64、80、96、112 进行候选数扫描，记录真实接受长度和端到端 Speedup。
5. **Batch/Context 网格：** 至少测试 Batch=1/2/4/8/16/32，以及 Context=128/512/1024/2048/4096/8192。
6. **分离算法与硬件收益：** 同时报告 Acceleration Rate、每轮延迟、Overhead、Speedup 和 Quality。
7. **核对分布保持：** 分别测试 Rejection Sampling 与 Typical Acceptance，不能只看吞吐。
8. **服务化评估：** 在目标 serving runtime 中测并发、排队、KV Cache、调度和多租户影响；单算子 Roofline 不代表端到端吞吐。

## 7. 与 EAGLE、DFlash 的跨论文关联

| 维度 | Medusa | EAGLE | DFlash |
|---|---|---|---|
| Draft 单元 | 多个未来 Token Heads | Hidden Feature | Token Block |
| Draft 生成 | 多 Head 并行预测 | Feature-level autoregression | Block Diffusion 并行生成 |
| 条件信息 | Target 最后隐藏状态 | Target Feature + Shifted Token | Target 多层隐藏特征与 KV Injection |
| Target 验证 | Tree Attention | Tree Attention / Speculative Sampling | Target 一次验证 Draft Block |
| 主要瓶颈 | Head 准确率、树验证、Attention 和 Batch | Feature Draft 轮数、树宽与验证成本 | Block Draft 质量、特征/KV 注入及 Target 验证 |
| 分布保证 | Rejection Sampling 可严格保持；Typical Acceptance 是近似质量权衡 | 标准 Speculative Sampling | 以 Target 验证和接受机制保证输出 |

技术演进可理解为：
- Medusa：从 Token 位置上并行预测；
- EAGLE：把 Draft 自回归对象从 Token 扩展为 Feature；
- DFlash：将一个 Token 一个 Token 的 Draft 进一步改成 Block Diffusion。

三者都试图减少 Target 的串行解码轮次，但 Draft 表示、训练方式和计算开销不同，不能仅按单一 Speedup 数字横向排名。

## 8. 精读章节导航

- [Abstract](00-Abstract.md)
- [Introduction](01-Introduction.md)
- [2.1 Key Components](02-Key-Components.md)
- [2.2 Training Strategies](03-Training-Strategies.md)
- [2.3 Extensions](04-Extensions.md)
- [3. Experiments](05-Experiments.md)
- [4. Discussion](06-Discussion.md)
- [Appendix A](07-Appendix-A-Related-Work.md)
- [Appendix B](08-Appendix-B-Experiment-Settings.md)
- [Appendix C](09-Appendix-C-Visualization-of-Optimized-Tree-Attention.md)
- [Appendix D](10-Appendix-D-Results-of-Speculative-Decoding.md)
- [Appendix E](11-Appendix-E-Additional-Results-for-All-Models.md)
- [Appendix F](12-Appendix-F-Additional-Results-on-AlpacaEval-Dataset.md)
- [Appendix G](13-Appendix-G-Hardware-Constraints-and-Medusa.md)

**全文精读完成。** 日期：2026-11-05。
