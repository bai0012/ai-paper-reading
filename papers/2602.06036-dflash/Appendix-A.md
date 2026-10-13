# Appendix A — 附录完整中文翻译及技术精读

> 原论文：[DFlash: Block Diffusion for Flash Speculative Decoding — arXiv:2602.06036v2](https://arxiv.org/html/2602.06036v2) | [PDF](https://arxiv.org/pdf/2602.06036v2)  
> 原文位置：PDF 第 12–13 页。阅读完成：2026-10-13。  
> 本文保存 **A.1–A.5.2 全部原文段落的中文翻译**，原始 Table 10–13 的数值及完整图表注，A.3 原始公式，以及独立推导、性能分析、复现建议。  
> “原文翻译”“技术讲解”分开标注。论文原始裁图已在对应 ChatGPT 对话中展示；仓库的原图说明使用 [PDF 第 12 页](https://arxiv.org/pdf/2602.06036v2#page=12)、[第 13 页](https://arxiv.org/pdf/2602.06036v2#page=13) 的原文链接，暂不虚构 images/figures 本地路径。

## A.1 Training Implementation — 训练实现（完整翻译）

**第 1 段。** 草稿模型使用 AdamW 优化器训练 6 个 Epoch，学习率为 \(6\times10^{-4}\)、梯度裁剪阈值 1.0，使用 Warmup Ratio 为 0.04 的 Cosine Schedule。使用混合数据集，最大序列长度 3072 Token（Qwen3-Coder 为 4096）；每条序列随机采样 512 个 Anchor 位置。第 4 节公式 (4) 中控制 Loss Decay 的超参数 \(\gamma\) 随 Block Size 调整：Block Size=16 时为 7，Block Size=10 时为 5，Block Size=8 时为 4。

**第 2 段。** 训练既可以 Online 进行，也可以 Offline 进行。在线训练时，目标模型隐藏特征在每个训练步骤中实时计算；离线训练时，预先计算并缓存这些隐藏特征，在 Draft Model 优化时读取，以降低计算开销。

### A.1 参数原文汇总

| 配置 | 原始值 |
|---|---|
| Optimizer | AdamW |
| Epochs | 6 |
| Learning rate | \(6\times 10^{-4}\) |
| Gradient clipping | 1.0 |
| LR schedule | Cosine |
| Warmup ratio | 0.04 |
| Max sequence length | 3072；Qwen3-Coder 4096 |
| Random anchors per sequence | 512 |
| Training mode | Online / Offline |
| Block Size → Loss Decay \(\gamma\) | 16→7；10→5；8→4 |

### A.1 技术讲解（独立分析）

- 如果 Block Size \(B=16\)，每个 Anchor 配 \(B-1=15\) 个 Mask 位置；512 个 Anchor 在展开前可对应 \(512\times15=7680\) 个有监督的位置实例。这**不等于** 7680 个互不重叠的真实 Token：不同 Anchor 区间可重叠，训练还需考虑 Padding、序列结尾和 Block Packing。
- 512 个 Anchor 说明随机样本起点丰富，但不能据此推断 GPU 一次 Forward 的实际 Token 数就是 \(512\times16\)；Sparse Attention 和 Block Packing 的真实内存需求需要运行 Profiling。
- Offline 会减少目标模型的重复 Forward，但带来目标特征的磁盘存储与加载；Online 避免离线缓存，但每个 Step 需要调用目标模型。论文没有给出两种模式的精确时间/存储实测对照。
- 为避免符号冲突，在自己的复现代码中建议将式 (4) 中的衰减 \(\gamma\) 记为 \(\gamma_{\mathrm{loss}}\)，并与第 3 节的推测 Token 预算 \(\gamma_{\mathrm{draft}}\) 区分。

## A.2 Diffusion Drafter without Target Feature — 不使用目标特征的扩散 Draft

这一小节**只有 Table 10 及图注，无独立正文段落**。

### Table 10 — 原表注完整翻译及全部数据

**图表原件：** [PDF 第 12 页，Table 10](https://arxiv.org/pdf/2602.06036v2#page=12)。

> 表 10. 一个不使用目标上下文特征的五层 Block Diffusion 草稿模型。其接受长度与加速比只能达到相对有限的水平。

原表每格格式：Speedup / \(\tau\)。

| Temperature | GSM8K | Math500 | AIME24 | AIME25 |
|---:|---|---|---|---|
| 0 | 2.83× / 3.38 | 3.73× / 4.61 | 3.43× / 4.12 | 3.35× / 4.07 |
| 1 | 2.76× / 3.29 | 3.31× / 4.12 | 2.66× / 3.23 | 2.65× / 3.24 |

### A.2 技术讲解（独立分析）

无 Target Hidden Features 的五层块扩散仍能够并行生成候选，Math500 Greedy 下达到 3.73×、\(\tau=4.61\)。与正文带条件信息的 DFlash 在特定配置下较高的数字形成直观对照，支持“仅并行化并不保证最高接受长度”的方法动机。但 Table 10 没有全面列出与正文 Table 1 完全一致的所有对照变量，所以**不能**将数值差额全归因于 Target Features，更不能当作严格受控的因果消融。

## A.3 KV Injection Mechanism and Memory Overhead — KV 注入机制和内存成本（完整翻译）

**第 1 段。** DFlash 使用 KV Injection，让 Diffusion Drafter 以 Target Model Features 为条件。首先，从选定的 Target Model Layers 中提取隐藏状态并拼接，再通过一次投影映射到 Draft Hidden Dimension：

\[
\boxed{
\mathbf H_t=\operatorname{RMSNorm}\left(
W_c[\mathbf H^{(l_1)};\ldots;\mathbf H^{(l_5)}]\right)
}
\]

**第 2 段。** 所有 Draft Layers 共享已经完成投影的 Target Features。在第 \(i\) 层，Draft Tokens 生成 Query，而 Target Features 与 Draft Tokens 都被投影为 Key、Value：

\[
\boxed{\mathbf Q_i=W_i^Q\mathbf H_d}
\]

\[
\boxed{\mathbf K_i=[W_i^K\mathbf H_t;\,W_i^K\mathbf H_d]_{\mathrm{seq}}}
\]

\[
\boxed{\mathbf V_i=[W_i^V\mathbf H_t;\,W_i^V\mathbf H_d]_{\mathrm{seq}}}
\]

因而 Target Features 仅为 Masked Block 的 Draft Tokens 提供额外 KV 条目，绕过 Draft Model 的 Query Projection、Output Projection、Self-attention Update 和 FFN。

**第 3 段。** 内存开销很小。唯一额外引入参数的组件是共享投影 \(W_c\in\mathbb R^{D\times5D}\)。对于 Qwen3.5-35B-A3B，隐藏维度 \(D=2048\)、BF16 权重，对应新增开销

\[
\boxed{5\times2048\times2048\times2\approx42\ \mathrm{MB}},
\]

与约 70GB 的 Target Model 相比可以忽略。激活额外开销也相对有限：当 Batch Size=1、Sequence Length=2048 时，投影输入和输出分别需要约 40MB、8MB；解码 Block Size=16 的临时激活低于 400KB。

### A.3 技术讲解与独立推导

**跨层拼接（特征维）**：若五个 Target Layer Hidden States 的形状为 \(T\times D\)，拼接得到 \(T\times5D\)，投影矩阵的形状为 \(D\times5D\)，经过 RMSNorm 得到 \(H_t\in\mathbb{R}^{T\times D}\)。

**逐层注入（序列维）**：Draft Block \(H_d\in\mathbb{R}^{B\times D}\)；简化为单头，Query 长度为 \(B\)，K/V 长度为 \(T+B\)。对应注意力可以表示为

\[
\operatorname{Attention}(Q,K,V)=\operatorname{Softmax}
\left(\frac{QK^\top}{\sqrt{d_h}}+M\right)V.
\]

这是帮助理解矩阵形状的常见 Attention 表达式；论文 A.3 直接给出了 \(Q_i,K_i,V_i\) 的生成式，没有在此给出完整 RoPE/GQA/Mask 内核细节。目标特征只提供被读取的 KV，不作为普通 Draft Token 逐层更新，减少重复计算。

**新增投影参数内存：**

\[
5\cdot2048^2=20,971,520\ \text{parameters}.
\]

\[
20,971,520\cdot2=41,943,040\ \text{bytes}
=40\ \mathrm{MiB}\approx41.94\ \mathrm{MB}.
\]

**长度 2048 的融合激活：**

\[
\text{Input}=2048\cdot5\cdot2048\cdot2=40\ \mathrm{MiB},
\]

\[
\text{Output}=2048\cdot2048\cdot2=8\ \mathrm{MiB}.
\]

**长度 16 的临时融合输入**：

\[
16\cdot5\cdot2048\cdot2=327,680\ \mathrm{bytes}
=320\ \mathrm{KiB}<400\ \mathrm{KB}.
\]

最后一项只是与论文所述临时激活数量级一致的解释，**不是**测得的整轮推理峰值显存。

**重要区分**：42MB 仅是新增投影权重，不等于 DFlash 的完整附加显存。Draft Transformer Layers、本身的 K/V Cache、上下文相关的 KV Injection Buffer、Attention Workspace、Serving 调度器缓存等均需另外计入。论文没有提供随 Batch、并发、上下文长度变化的完整峰值显存曲线。

## A.4 Results on More Models and vLLM — 更多模型和 vLLM（完整翻译）

**第 1 段。** 我们进一步在更多 Target Models 和推理框架上评估 DFlash。Table 11 报告在单张 B200 GPU、并发 8 下的 SGLang 结果。每项给出接受长度 / 相对原生自回归解码的加速比。当原生 MTP 可用时，DFlash 始终比 MTP 更好；DFlash 也可以扩展到更大型的 Qwen3.5、Qwen3-Coder 和 GPT-OSS（Agarwal et al., 2025）模型。

### Table 11 — 原表注完整翻译及全部数据

**原图**：[PDF 第 12 页，Table 11](https://arxiv.org/pdf/2602.06036v2#page=12)。

> 表 11. 更多模型在 SGLang 上的性能。每格依次报告接受长度 / 加速比。

| 模型 | 方法 | Math500 | HumanEval | MT-Bench |
|---|---|---|---|---|
| Qwen3.5-4B | MTP | 6.5 / 1.5× | 6.3 / 1.6× | 5.3 / 1.3× |
| | DFlash | 7.1 / 3.0× | 7.3 / 2.9× | 5.6 / 2.3× |
| Qwen3.5-9B | MTP | 6.7 / 1.7× | 6.6 / 1.7× | 5.3 / 1.3× |
| | DFlash | 7.3 / 3.5× | 7.9 / 3.4× | 5.5 / 2.5× |
| Qwen3.5-35B-A3B | MTP | 6.9 / 1.7× | 7.1 / 1.6× | 5.2 / 1.2× |
| | DFlash | 7.2 / 2.4× | 7.9 / 2.3× | 5.4 / 1.7× |
| Qwen3.5-27B | DFlash | 7.7 / 3.8× | 9.1 / 3.9× | 5.5 / 2.5× |
| Qwen3-Coder-Next | DFlash | 6.0 / 1.9× | 7.2 / 1.9× | 3.9 / 1.2× |
| GPT-OSS-20B | DFlash | 5.1 / 2.2× | 4.3 / 2.2× | 4.2 / 2.0× |
| GPT-OSS-120B | DFlash | 5.4 / 1.6× | 4.4 / 1.7× | 3.7 / 1.3× |

**第 2 段。** 另外，还在 Qwen3.5-9B 上用 vLLM 评估 DFlash。Table 12 报告了不同并发水平下的吞吐量及相对自回归解码的加速比。在低、中并发时，DFlash 取得较强的加速，在高并发下仍保持吞吐优势。

### Table 12 — 原表注完整翻译及全部数据

**原图**：[PDF 第 12 页，Table 12](https://arxiv.org/pdf/2602.06036v2#page=12)。

> 表 12. Qwen3.5-9B 的 vLLM 测试。每格给出 DFlash 吞吐量（tok/s），括号中为相对自回归解码的加速比。

| 并发 | Math500 | HumanEval | MT-Bench |
|---:|---|---|---|
| 1 | 849 (4.0×) | 969 (4.6×) | 627 (3.0×) |
| 8 | 5096 (3.2×) | 5434 (3.4×) | 3536 (2.2×) |
| 16 | 7669 (2.5×) | 8131 (2.7×) | 5297 (1.7×) |
| 32 | 9836 (1.9×) | 10258 (2.1×) | 6787 (1.3×) |

### A.4 技术讲解（独立分析）

**与 MTP 的比较**：Qwen3.5-9B Math500，原生 MTP 接受长度 6.7、加速 1.7×；DFlash 接受长度 7.3、加速 3.5×。加速比之比 \(3.5/1.7\approx2.06\)，大于接受长度之比 \(7.3/6.7\approx1.09\)，说明两者 Draft/Verification 成本的差异在总效率中有重要作用。然而 Table 11 没有独立报告各步骤耗时，不能精确归因到某个 Kernel。

**较大参数量不保证更多加速**：Qwen3.5-27B Math500 达到 3.8×，GPT-OSS-120B Math500 仅 1.6×。模型结构、基本延迟和 Token 可预测性不同；表格无法仅用总参数量解释变化。

**vLLM 高并发趋势**：HumanEval 并发 1→8→16→32，加速 4.6×→3.4×→2.7×→2.1×，绝对吞吐则为 969→5434→8131→10258 tok/s。和正文 SGLang 实验共同说明，增加并发可能提高绝对总吞吐，却缩小相对自回归基线的加速幅度。

**证据边界**：论文的附录没有给出 Table 12 对应的 vLLM 版本、内核细分 Profiling 或完整 TTFT/P95 延迟。Table 11 的 MTP 与 DFlash 差异不能被简单归因为“纯块扩散”和“纯原生 MTP”之间一项机制的差别。

## A.5 Further Ablations — 更多消融

### A.5.1 Loss Decay — 完整翻译

**第 1 段（原文跨 PDF 第 12–13 页）。** 作者对第 4.2 节引入的位置相关 Loss Decay 进行消融：比较使用指数衰减 Token 权重的默认配置，以及在 Draft Block 内使用均匀 Token 权重训练的变体。这一实验单独检验训练时更注重较早 Token 准确率带来的作用。如 Figure 5 所示，应用 Loss Decay 可以使训练收敛得更快、更好。

**Figure 5 原图**：[PDF 第 13 页，Figure 5](https://arxiv.org/pdf/2602.06036v2#page=13)。

**完整图注翻译：** 图 5. 损失衰减让训练收敛得更快、更好。

**图解（独立分析）**：Figure 5 横轴 Epoch（1–9），纵轴在 Math500 上的 Acceptance Length。蓝色“with loss decay”、橙色“without loss decay”。在最初几轮 Epoch，蓝线有较明显的优势；5–9 Epoch 时两者逐渐接近，约在 6.3–6.4 范围。此图直接支持 Loss Decay 加速前期收敛；后期微小差距不能在无随机种子/误差线条件下断言统计显著性，也不能将接受长度增加等同于端到端加速提升。

原始权重来自式 (4)：

\[
w_k=\exp\left(-\frac{k-1}{\gamma_{\mathrm{loss}}}\right).
\]

例如 \(B=16,\ \gamma_{\mathrm{loss}}=7\) 时，最前预测位置权重 \(w_1=1\)，后部位置权重随 \(k\) 指数递减。这符合连续前缀验收中“前部错误使后续候选无效”的非对称机制。

### A.5.2 Random Sampling of Masked Blocks — 完整翻译

这一小节**只有 Table 13 及图注，没有独立的正文解释段落**。

**Table 13 原图**：[PDF 第 13 页，Table 13](https://arxiv.org/pdf/2602.06036v2#page=13)。

**原表注完整翻译：** 训练中随机选取 Anchor Token 构建 Masked Blocks，能有效增强训练数据，并带来更长接受长度和更高加速比。两个 Draft Model 都使用 3 层、从目标模型提取 5 个 Hidden Features，Block Size 为 16，均使用第 5.5 节介绍的 100K 数据训练。

| 训练策略 | Math500 加速 | Math500 τ | HumanEval 加速 | HumanEval τ | MT-Bench 加速 | MT-Bench τ |
|---|---:|---:|---:|---:|---:|---:|
| Standard | 4.13× | 4.94 | 3.29× | 3.86 | 2.13× | 2.80 |
| Sample | **4.69×** | **5.64** | **3.90×** | **4.61** | **2.38×** | **3.18** |

**独立分析（由表格计算的相对增幅）：**

| 任务 | 加速比相对提高 | 接受长度相对提高 |
|---|---:|---:|
| Math500 | +13.6% | +14.2% |
| HumanEval | +18.5% | +19.4% |
| MT-Bench | +11.7% | +13.6% |

Random Anchor 的贡献是增加训练起点变化，同时使训练样本形式匹配推测解码在上一轮 Target Token/Bonus Token 后开始一个新块的行为。样本数据增强以及 Train–Inference Alignment 都可能推动性能改善。这是作者提出的机制解释，Table 13 主要验证最终接受长度及加速确实提高。

## 附录综合讨论与复现建议（独立分析）

1. **训练核心配置**：AdamW、学习率 \(6\cdot10^{-4}\)、6 Epoch、Cosine + 4% Warmup、Clip=1；不要遗漏 512 Anchor/Sequence 和 BS 对应的 \(\gamma_{\mathrm{loss}}\)。
2. **控制训练成本**：可通过 Offline Target Feature Precomputation 降低重复 Target Forward，但必须估计五层特征缓存容量与 I/O；Online 更灵活但目标前向计算重。
3. **显存计量需分拆**：额外投影 \(W_c\) 约 42MB，长度 2048、BS=1 时输入/输出激活约 40/8MB；运行时 Draft Layers 与 Target Context KV Cache 仍有额外占用，不能从这些数字断言整体只有 42MB。
4. **对照条件**：无条件 Block Diffusion 仍可加速，但不同实验训练细节不完全对齐，不应从 Table 10 机械计算全部因果贡献。
5. **原生 MTP / vLLM 覆盖**：Table 11 显示在若干 Qwen3.5 模型上 DFlash 优于原生 MTP；Table 12 验证可在 vLLM 上加速，但并发提高会让相对 Speedup 衰减。
6. **两个训练消融**：Figure 5 最直接验证 Loss Decay 的早期收敛加速，Table 13 进一步证实随机 Anchor 的接受长度和端到端 Speedup 收益。
7. **复现时应记录**：Target Checkpoint、Tokenizer/LM Head 共享、抽取的具体目标层号、Draft Layers、BS、Anchor、Loss Decay、Prefill/Decode 模式、采样温度、\(\tau\)、Draft/Verify 毫秒、TPOT/TTFT/P95、GPU 显存、并发与 Serving 后端。

### 推荐继续研究的方向

- **自适应 Block Size**：借助正文 Table 8 大 BS 训练向小 BS 推理的非对称泛化，根据实际 GPU 负载动态选择 BS；作者明确留作未来工作。
- **Target KV 条件化缓存压缩**：优化额外上下文 KV 的存储、分组投影与量化，建立显存—接受率—吞吐 Pareto 曲线。
- **更严格的 MTP / DFlash 对照**：相同硬件、Tokenizer、Serving 后端、Candidate Budget、Temperature 下分析 Draft/Verify 分开耗时，避免把全部差距都归因于扩散。
- **长上下文和高并发**：在 4K–32K 以上、不同请求长度分布测量相对加速及 P95/峰值显存，弥补正文 Table 4 只报告 \(\tau\) 的局限。

## 附录完成状态

- [x] A.1 Training Implementation — 两段翻译、完整超参数
- [x] A.2 Diffusion Drafter without Target Feature — Table 10
- [x] A.3 KV Injection Mechanism and Memory Overhead — 三段翻译、所有原始公式与显存计算
- [x] A.4 Results on More Models and vLLM — 两段翻译、Table 11–12
- [x] A.5.1 Loss Decay — 原文翻译、Figure 5
- [x] A.5.2 Random Sampling of Masked Blocks — Table 13

**正文和附录已读完。** 详见 [summary.md](summary.md)，并在本论文 [README.md](README.md) 与仓库根目录 [论文知识索引.md](../../论文知识索引.md) 中维护检索导航。
