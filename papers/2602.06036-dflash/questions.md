# DFlash 精读过程中的技术问答

## Q1. Target Model 是怎么并行验证 Draft Token 的？

**提问日期：2026-10-13**  
**关联章节：** [3. Preliminaries](03-Preliminaries.md)、[4. Method](04-Method.md)、[5. Experiments](05-Experiments.md)  
**论文原文：** [arXiv:2602.06036v2](https://arxiv.org/html/2602.06036v2)，Section 3.1 说明一次 Draft \(\gamma\) 个 Token 由 Target 并行验证；Section 4.1 说明 DFlash 一次并行预测候选块。论文在此**没有详细推导所有精确验证/拒绝算法**，下面相关数学和伪代码属于标准 Speculative Decoding 知识的扩展解释。

### 一、核心原理：有候选序列以后，可以用 Teacher Forcing 一次计算所有位置的下一 Token 概率

标准 AR 解码必须先得到下一个 Token \(x_{t+1}\)，才能为 \(x_{t+2}\) 提供条件，因此是串行的。DFlash 预先提出 \(D=(d_1,\ldots,d_\gamma)\) 后，Verifier 可以把整块作为已知 Token 输入 Target Transformer，并在一次 Causal Forward 中获得

\[
\begin{aligned}
p_1&=P_t(\cdot\mid C),\\
p_2&=P_t(\cdot\mid C,d_1),\\
p_3&=P_t(\cdot\mid C,d_1,d_2),\\
&\dots\\
p_{\gamma+1}&=P_t(\cdot\mid C,d_1,\dots,d_\gamma).
\end{aligned}
\]

每个位置有单独的 **Causal Attention Mask**，只看已知前缀和草稿序列中该位置以前的元素；多个位置可在单次矩阵计算中并行得到不同条件的 logits。验证阶段的 Target Model **仍然是 Causal Transformer**，不是 DFlash Draft 中的 Bidirectional Attention。

由于 LM logits 右移一个 Token，Target 在已有上下文最后位置的 logits 验证 \(d_1\)，在输入 \(d_1\) 的位置输出 logits 验证 \(d_2\)，……输入 \(d_\gamma\) 的位置预测额外 Bonus Token。使用 KV Cache 增量解码时，验证第一候选的 \(p_1\) 通常已在上一轮算好，所以只需在一次新 Forward 中计算本次候选块各位置的后续分布。

### 二、贪心验证：只提交连续一致前缀

例：Draft \([A,B,C,D]\)，Target 在对应的候选条件上得到的 argmax 为 \([A,B,X,Y]\)。本轮前两 Token 一致，第三 Token 第一次不一致：提交 \([A,B,X]\)，第四位置的 Y 不能继续使用，因为它是在错误候选前缀 \([A,B,C]\) 下预测，而真正上下文必须为 \([A,B,X]\)。

若所有 \(\gamma\) 候选都与 Target Argmax 相同，还能从最后位置获得一个 Bonus Token，因此标准无 EOS/边界情况下每轮最多提交 \(\gamma+1\) 个 Token。注意本例只是 Temp=0；EOS 或 max_new_tokens 边界可截断。

### 三、非贪心温度采样：应使用分布正确的 Speculative Sampling

若 Target 分布为 \(p_i(x)\)，Draft 在第 \(i\) 位置**实际生成该候选时的 proposal 条件分布**为 \(q_i(x)\)，对候选 \(d_i\) 使用：

\[
\alpha_i(d_i)=\min\!\left(1,\frac{p_i(d_i)}{q_i(d_i)}\right).
\]

随机决定接受或拒绝；拒绝时从正残差中采样

\[
r_i(x)=\frac{[p_i(x)-q_i(x)]_+}
{\sum_y[p_i(y)-q_i(y)]_+}.
\]

接受先行位置后才能尝试下一位置；一旦拒绝，采样校正 Token 并停止该轮；全部接受则从 \(p_{\gamma+1}\) 获得 Bonus Token。

**重要限制**：这是标准精确 Speculative Sampling 的数学规则，不是 arXiv 论文 Section 3/4 的逐字伪代码。对于 DFlash 这种一次性并行扩散 Draft，必须使用与**实际 proposal 机制**一致的 \(q_i\) 或相应正确验收算法，不能未经验证就把任意 denoising logits 当作自回归 \(q_i(x\mid C,d_{<i})\)，也不能简单地把 draft token 与一次独立随机采样的 target token 当作贪心一致性检查。具体非贪心的实现细节应检查所用 SGLang/vLLM/Transformers 版本。

一位置的无损性直觉：草稿提议 \(x\) 的概率为 \(q(x)\)，被接受并输出 \(x\) 的概率为 \(q(x)\min(1,p(x)/q(x))=\min(q(x),p(x))\)。剩余概率质量使用 \(r(x)\) 补足：

\[
\min(q(x),p(x))+[p(x)-q(x)]_+=p(x).
\]

故正确执行后单步输出分布仍为 \(p(x)\)。整条序列逐前缀归纳可保留目标采样分布。

### 四、性能原因

**一次 Target Forward 不等于一个 Token 的计算量。** 验证 \(\gamma\) 个候选需要处理 \(\gamma\) 个 Query、附带 Attention/FFN FLOPs，但有利于同时并行处理这些位置，减少重复模型权重读取/算子调度及串行等待。其成本可低于 \(\gamma\) 次单 Token AR Forward；高并发时 Verification 可能变得 Compute-bound，故收益未必继续线性增加。DFlash 与 EAGLE-3 的主要区别在 **Draft 怎么提出候选**（DFlash 单次块内并行而 EAGLE-3 往往串行展开/树式），而不是 Target 是否以并行方式验证。

### 五、KV Cache 必须正确提交和回滚

Verifier 一次 Forward 会为整个候选块生成相应的 Target KV。若在第 \(k\) 个候选拒绝，就只能保留对应已确认的前缀 KV，丢弃后续错误候选的 KV，然后以被 Target 校正的 Token 继续下一轮。若全部接受，可保留已验证草稿的 KV 并以 Bonus Token 为新的锚点。具体 Cache Offset、Anchor 位置与 Bonus 是否已有 KV 依赖推理引擎实现，不能误将“接受数量”与缓存应保留的物理长度始终等同。

### 六、工程上的最小伪代码（概念）

1. Target 在前缀 C 上得到下一 Token 概率及 Hidden Features；DFlash 通过单次 Forward 并行提议候选 \([d_1,\dots,d_\gamma]\)。
2. 将全部候选 Token 输入 Target 的一次 Causal Forward，连同已有第一个 logits 对齐得到 \(p_1,\dots,p_{\gamma+1}\)。
3. 在小规模 GPU/CPU Sampling Kernel 上按位置顺序执行 Accept/Reject，找到最早拒绝点，或确认全部接受。
4. 提交已接受前缀及一枚由正确 Target 分布生成的 Token；修正 Target/Draft 的 KV Cache，使状态对应真实已提交序列。
5. 从校正/Bonus Token 的正确 Target Hidden Features 继续下一轮 DFlash Draft。

**核心结论**：Target Model 的 Transformer **计算**是并行的；按序停止的 Accept/Reject **判定**具有前缀依赖，但只需要比较 logits、抽随机数与更新缓存，不需要为每个候选分别再次执行大模型 Forward。

### 参考来源

- [DFlash arXiv Section 3.1, 4.1](https://arxiv.org/html/2602.06036v2)
- [vLLM Speculators 的 DFlash 机制说明](https://github.com/vllm-project/speculators/blob/main/docs/user_guide/algorithms/dflash.md)
- [Speculative Decoding (Leviathan et al., 2023)](https://arxiv.org/abs/2211.17192)
