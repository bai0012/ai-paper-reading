# 3. Preliminaries（预备知识）

- 论文：**DFlash: Block Diffusion for Flash Speculative Decoding**
- 版本：[arXiv:2602.06036v2](https://arxiv.org/html/2602.06036v2)
- 来源：PDF 第 3 页下半部分及第 4 页开头
- 阅读日期：2026-10-10
- 内容：章节导言、3.1、3.2 **完整逐段翻译**，原式 (1)–(3)，Figure 3 原图定位、图注翻译、数学推导与独立分析
- 说明：“原文翻译”与“技术讲解”严格分离；推导和数值例子不是论文原文，未经过作者实验验证。

## 第一部分：论文原文完整翻译

### 3. Preliminaries — 章节导言

本节对推测解码的加速机制进行形式化描述，阐明自回归草稿生成（Autoregressive Drafting）与基于扩散的草稿生成（Diffusion-based Drafting）之间的效率权衡。我们的分析强调，为什么扩散草稿模型具有独特优势，能够同时实现较低的草稿生成延迟和较高的 Token 接受率。

### 3.1. Speculative Decoding Speedup — 推测解码加速比

**第 1 段。** 推测解码使用一个较小的草稿模型 \(\mathcal{M}_d\)，来加速目标模型 \(\mathcal{M}_t\) 的推理。在每一轮解码中，草稿模型提出 \(\gamma\) 个 Token，目标模型再并行验证这些 Token。

**第 2 段及原式 (1)。** 按照 Sadhukhan et al.（2025）的定义，平均每 Token 的延迟为：

\[
\boxed{L=\frac{T_{\mathrm{draft}}+T_{\mathrm{verify}}}{\tau}}\tag{1}
\]

其中，\(T_{\mathrm{draft}}\) 表示生成草稿 Token 所用的时间；\(T_{\mathrm{verify}}\) 表示验证草稿 Token 的成本；\(\tau\in[1,\gamma+1]\) 表示每轮预期接受的 Token 数，其中包含由目标模型产生的 Bonus Token。令 \(L_{\mathrm{target}}\) 为目标模型 \(\mathcal{M}_t\) 自回归生成单个 Token 的延迟，则加速比为 \(\eta=L_{\mathrm{target}}/L\)。

**第 3 段。** 这一表达式清楚显示了两方面的权衡：增大预期接受长度 \(\tau\)，或者减小草稿生成开销 \(T_{\mathrm{draft}}\)，都能改善加速比。

### 3.2. Autoregressive vs. Diffusion Drafting — 自回归与扩散式草稿生成

**第 1 段及原式 (2)。** 自回归草稿模型以顺序方式生成 Token，因此草稿生成成本为：

\[
\boxed{T_{\mathrm{draft}}=\gamma\cdot t_{\mathrm{step}}}\tag{2}
\]

这里的 \(t_{\mathrm{step}}\) 是单次前向传播的延迟。因此，草稿生成成本会随推测预算 \(\gamma\) 线性增长。

**第 2 段。** 为了控制延迟，自回归草稿模型受到很浅网络结构的限制（如 EAGLE-3 使用单层 Transformer）。这严重限制了草稿质量：增加 \(\gamma\) 虽然增大 Draft 成本，但由于模型容量受限，接受长度 \(\tau\) 很快饱和。实际应用中，这种不平衡限制了可获得的加速。

**第 3 段及原式 (3)。** 扩散草稿模型在一次前向传播中并行生成所有 \(\gamma\) 个 Token，因此有：

\[
\boxed{T_{\mathrm{draft}}=t_{\mathrm{parallel}}}\tag{3}
\]

其中 \(t_{\mathrm{parallel}}\) 表示块生成的延迟。现代 GPU 对此类并行操作的执行效率远高于多次顺序前向传播。因此对于规模相近的模型，\(t_{\mathrm{parallel}}\ll \gamma t_{\mathrm{step}}\)。当块长度适中时，\(T_{\mathrm{draft}}\) 因而基本不受 \(\gamma\) 增加的影响。

#### Figure 3：草稿生成延迟（论文原图）

**原图**：请查看 [arXiv 原始 PDF 第 3 页 Figure 3](https://arxiv.org/pdf/2602.06036v2#page=3)，或论文 [HTML 的 Figure 3](https://arxiv.org/html/2602.06036v2)。图像来自原 PDF，已在对应 ChatGPT 阅读会话中裁取为 dflash_Figure_3_original.png；目前尚未成功上传 PNG 到本仓库，故此处不创建不存在的 figures/ 相对路径。

**原图注完整翻译：** 图 3. 比较单层 EAGLE-3 与分别采用 1 层、3 层、5 层的 DFlash 草稿模型的草稿生成成本。

**读图说明（独立分析）：** x 轴是 Number of Draft Tokens（4 / 8 / 16），y 轴是 Latency (ms)。灰色 EAGLE-3 随候选长度从 4 增至 16，延迟显著增大；蓝色单层 DFlash、橙色 3 层 DFlash 与绿色 5 层 DFlash 的延迟整体保持在较低且相对稳定的范围。5 层 DFlash 草稿长度 16 时延迟仍低于 EAGLE-3 草稿长度 8 的延迟。注意图中**只显示延迟**，接受长度并未绘出，后者是作者在同段正文中的另一个结论。原图缺少具体数值表，不能把读图约数当成作者精确测量结果。

**第 4 段（跨到 PDF 第 4 页）。** 这种并行性从根本上改变了设计空间。因为草稿生成成本不再随生成 Token 数量线性增长，扩散草稿模型能够采用更深、更有表达能力的网络，而不必牺牲延迟。更高的模型容量显著改善草稿质量和接受长度。实验证明，使用五层 DFlash 草稿模型生成 16 个 Token，不仅比 EAGLE-3 生成 8 个 Token 时延迟更低（Figure 3），还能达到更长的接受长度，使 DFlash 在草稿质量与草稿开销的帕累托前沿（Pareto Frontier）上占据更有利位置。

至此结束 Section 3。后文进入 Section 4. Method。

---

## 第二部分：数学推导与技术讲解（独立分析）

### 1. 公式 (1) 的推导

目标模型 AR 解码每个 Token 平均用时 \(L_{\mathrm{target}}\)，生成 \(N\) 个 Token：

\[
T_{\mathrm{AR}}\approx N L_{\mathrm{target}}.
\]

每轮推测解码用时为：

\[
T_{\mathrm{cycle}}=T_{\mathrm{draft}}+T_{\mathrm{verify}}.
\]

每轮平均推进 \(\tau\) 个 Token，生成 \(N\) 个 Token 约需 \(R\approx N/\tau\) 轮：

\[
T_{\mathrm{Spec}}\approx\frac{N}{\tau}(T_{\mathrm{draft}}+T_{\mathrm{verify}}).
\]

因此：

\[
L=\frac{T_{\mathrm{Spec}}}{N}
=\frac{T_{\mathrm{draft}}+T_{\mathrm{verify}}}{\tau}.
\]

\[
\eta=\frac{T_{\mathrm{AR}}}{T_{\mathrm{Spec}}}
=\frac{L_{\mathrm{target}}\,\tau}{T_{\mathrm{draft}}+T_{\mathrm{verify}}}.
\]

上述推导是平均吞吐/平均延迟的近似分析，省略 batch、调度重叠、最后一轮不足、同步与缓存维护成本；不是对每个请求真实墙钟时间的精确等式。

**加速条件：**

\[
\eta>1
\iff
L_{\mathrm{target}}\tau>T_{\mathrm{draft}}+T_{\mathrm{verify}}
\iff
\tau>\frac{T_{\mathrm{draft}}+T_{\mathrm{verify}}}{L_{\mathrm{target}}}.
\]

在 \(T_{\mathrm{draft}}\)、\(T_{\mathrm{verify}}\) 不变时，提高 \(\tau\) 按比例提高 \(\eta\)。在 \(\tau\) 不变时，降低草稿/验证成本提高 \(\eta\)。实际调整模型时，它们常常相关，不能任意独立控制。

### 2. 为何 \(\tau\in[1,\gamma+1]\)？

对于标准单分支、正确实现的推测解码，一轮先提出 \(\gamma\) 个候选。若第一个候选被拒绝，目标模型在该位置采样/确认一个 Token，本轮至少推进 1 个 Token。若所有 \(\gamma\) 个草稿 Token 均被接受，则目标模型还能生成一个额外 Bonus Token，最多推进 \(\gamma+1\) 个 Token。EOS、截断、树状验证等细节需依照相应算法讨论。

**补充的简化概率模型（非论文原式）：** 若条件接受概率可近似视为各位置固定的 \(p\)，那么草稿被接受的连续前缀长度 \(A\) 满足

\[
\mathbb{E}[A]=\sum_{k=1}^{\gamma}p^k
=\frac{p(1-p^\gamma)}{1-p}\quad(p\ne1).
\]

标准线性 speculative sampling 若每轮仍能提交一个拒绝修正 Token 或全部接受时的 Bonus Token，则可近似写为

\[
\tau\approx1+\mathbb{E}[A].
\]

这只是帮助理解“增加草稿预算未必显著增加接受长度”的简化模型，不是 DFlash 本节推导，也不适用于无条件的任意树式验收算法。

### 3. 公式 (2)：自回归草稿的串行依赖

草稿 Token 的条件采样：

\[
x_i\sim q_\theta(x_i\mid x_{<i},c),\qquad i=1,\ldots,\gamma.
\]

第 \(i+1\) 步必须等待 \(x_i\) 得到确定。若一次生成的延迟近似 \(t_{\mathrm{step}}\)，则

\[
T_{\mathrm{draft}}\approx \sum_{i=1}^{\gamma}t_{\mathrm{step}}
=\gamma t_{\mathrm{step}}.
\]

这是论文采用的**串行长度成本近似**。EAGLE-3 使用树形候选结构，不保证物理 Forward 次数等于候选节点总数；因此公式 (2) 不应直接用于精确计算某个特定 EAGLE-3 配置的实际运行时长。

### 4. 公式 (3)：一次前向预测整块候选

令 Mask 块为

\[
X_{\mathrm{mask}}=[m_1,m_2,\ldots,m_\gamma].
\]

扩散 Draft 使用上下文 \(c\)，在一次 Forward 中并行计算 \(\gamma\) 个位置的预测：

\[
X_{\mathrm{draft}}=f_\theta(X_{\mathrm{mask}},c).
\]

这可把串行次数从 \(\gamma\) 变为 1 次，但单次 Forward 的内部计算量仍随 Block Size 增长。因此更精确的工程写法是

\[
T_{\mathrm{draft}}=t_{\mathrm{parallel}}(\gamma,D,\text{context},\text{hardware}),
\]

其中 \(D\) 是 Draft 深度。论文式 (3) 把此开销整体记为 \(t_{\mathrm{parallel}}\)，并在中等块长度的实测条件下观察到其对 \(\gamma\) 不敏感，并非声称该延迟对所有长度都是数学常数。

对于简化的普通自注意力层，当前块内注意力具有 \(O(\gamma^2 d)\) 计算项，投影与 FFN 也增加；如果有前缀 KV Cache，则还存在当前块读取历史上下文的成本。GPU 利用率提升和减少重复 Launch/串行依赖使得单次较大计算在实际延迟上仍可能更优。

### 5. 结合式 (1)–(3) 比较完整加速

对简化的自回归 Draft：

\[
\eta_{\mathrm{AR}}=
\frac{L_{\mathrm{target}}\tau_{\mathrm{AR}}}
{\gamma t_{\mathrm{step}}+T_{\mathrm{verify,AR}}}.
\]

对 DFlash 块扩散 Draft：

\[
\eta_{\mathrm{Diff}}=
\frac{L_{\mathrm{target}}\tau_{\mathrm{Diff}}}
{t_{\mathrm{parallel}}+T_{\mathrm{verify,Diff}}}.
\]

两者的 \(\tau\) 以及验证开销通常也会不同；例如候选树验证与线性块验证存在不同的 Attention/KV 工作量。只要

\[
\frac{\tau_{\mathrm{Diff}}}
{t_{\mathrm{parallel}}+T_{\mathrm{verify,Diff}}}
>
\frac{\tau_{\mathrm{AR}}}
{\gamma t_{\mathrm{step}}+T_{\mathrm{verify,AR}}},
\]

则在相同目标基线下 DFlash 取得更高加速。

**独立虚构数值示例（非论文测试）：**

设目标单 Token 延迟 20ms。AR Draft 预测 8 个候选，每步 2ms、每轮验证 20ms、平均完成 4 个 Token，则 \(L_{\mathrm{AR}}=(16+20)/4=9\) ms/token，加速 \(\eta_{\mathrm{AR}}\approx2.22\times\)。

设 DFlash 一次 Draft 5ms、每轮验证 20ms、平均完成 6 个 Token，则 \(L_{\mathrm{Diff}}=(5+20)/6\approx4.17\) ms/token，加速 \(\eta_{\mathrm{Diff}}\approx4.8\times\)。

这说明加速优势来自**同时减少 Draft 成本、提高每轮输出 Token 数**，但虚构数据不可当作作者结果。

### 6. 为什么扩散模型可以增加层数

若 Draft 深度为 \(D\)，自回归草稿在简化情况下需要 \(\gamma\) 次通过这 \(D\) 层，延迟可粗略表示为

\[
T_{\mathrm{AR\ draft}}\sim\gamma D\,t_{\mathrm{single-layer-step}},
\]

并行扩散块一次通过 \(D\) 层：

\[
T_{\mathrm{Diff\ draft}}\sim D\,t_{\mathrm{single-layer-block}}.
\]

这不是精确系统性能模型，而是表达串行深度/Token 数依赖。只要并行块的单位层延迟显著低于 \(\gamma\) 倍单 Token 层延迟，就有空间让小型块扩散 Draft 加深，而不被多步串行成本压垮。

但质量和延迟仍有 Pareto Trade-off：不是层数越多就必然越快。论文本节将五层 DFlash（16 Token）与单层 EAGLE-3（8 Token）比较，称前者既更快、接受长度又更长。对于究竟选择三层、五层或八层，以及如何注入上下文，必须结合之后 Section 5.5 消融实验判断。

### 7. 与其他研究的关联

- [MagicDec / Sadhukhan et al.](https://arxiv.org/abs/2408.11049)：本节采用的推测解码延迟分析形式。二者都关注 \(\tau\) 与运行时成本的平衡。
- [EAGLE-3](https://arxiv.org/abs/2503.01840)：使用目标隐藏特征的自回归 Draft；DFlash 把核心优化转向“目标条件特征 + 块内并行”。
- [Block Diffusion](https://arxiv.org/abs/2503.09573)：块内去噪/块间顺序生成，为 DFlash 一次处理 Mask Block 的 Draft 设计提供基础。
- 第 4 节原始 Figure 2 将说明多层 Target 隐藏状态如何融合，并通过各 Draft 层 KV 注入来获得更高接受长度。

## 完成总结及下一节

式 (1) 确立了共同评价框架：

\[
\eta=\frac{L_{\mathrm{target}}\tau}{T_{\mathrm{draft}}+T_{\mathrm{verify}}}.
\]

式 (2) 描述串行 Draft 成本随 \(\gamma\) 增长；式 (3) 描述单次块并行 Draft 的成本优势；Figure 3 提供 1/3/5 层 DFlash 与 1 层 EAGLE-3 的实测延迟趋势。核心并非“扩散一定更准”，而是它为**更深但仍然快速的 Draft 模型**提供了条件，配合来自 Target 的上下文特征有机会同时提高接受长度。

**已完成：Section 3 Preliminaries（3.1、3.2）**

**下一阅读单元：Section 4 Method**（4.1 Inference、4.2 Training，Figure 2/4，式 (4)；附录 A.3 的公式可根据阅读需要提前交叉参照）。
