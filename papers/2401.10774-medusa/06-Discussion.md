# Section 4 Discussion — 完整译文、原 Table 3、加速分解与现代 Serving 分析

- 论文：Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads
- 版本：arXiv:2401.10774v3；ICML 2024；精读日期：2026-11-03
- 范围：[PDF 第 9 页](https://arxiv.org/pdf/2401.10774v3#page=9)；Section 4 全部两个自然段、Table 3 全部内容及表注。
- 排版说明：Table 3 浮动在该页 Section 4 标题上方，但正文明确引用它。Acknowledgements / Impact Statement 是独立部分，不包含在 Section 4 中。
- 图像状态：Table 3 PNG 已由用户提供的原始 PDF 裁图在本次聊天展示；GitHub 当前 Connector 仅能创建 UTF-8 文本，因此**没有上传 PNG 二进制**，本文只引用原始 PDF 页码，不创建虚假的 figures/ 相对路径。
- 下文先完整翻译原文，再单独记录推导、外部资料和独立分析。

## 一、Section 4 Discussion 原文完整中文翻译

**第一自然段**：总而言之，Medusa 通过为大型语言模型添加额外的预测解码头，使模型能够同时生成多个 Token，并绕过自回归解码的顺序执行限制，从而将 LLM 推理速度提高到原来的 **2.3–2.8 倍**。Medusa 的主要优势包括设计简单、参数效率高，并且易于集成到现有系统中。Medusa 不需要专门的草稿模型（Draft Model）。典型接受机制（Typical Acceptance）在提供合理输出的同时，避免了拒绝采样（Rejection Sampling）带来的复杂性。我们的方法包含两种高效的训练流程，可以在不同模型和不同类型的 Prompt 上保持较高的输出质量。我们在 Table 3 中总结了各项技术的发展以及它们对加速比的影响。

**第二自然段**：在本文中，为了简化研究，我们主要关注 Batch Size 为 1 的设置。不过，我们希望强调，本文提出的思想同样可以推广到更大的 Batch Size。文章完成之后，TensorRT 和 Hugging Face TGI 等库已经支持这样的更大 Batch Size 设置。

上述“支持”属于 2024 年原文的作者陈述，不是 2026 年所有 Serving 框架、所有版本均支持的证明。

## 二、Table 3 原始表与完整表注

原版：[PDF 第 9 页 Table 3](https://arxiv.org/pdf/2401.10774v3#page=9)，本次聊天已展示直接裁取的原始 PNG。

原文标题：**Table 3. Impact of Techniques on Speedup**。

完整表题翻译：**表 3：各项技术对加速比的影响。**

| 论文 Technique 原文 | 中文译文 | Speedup（原表） |
| --- | --- | ---: |
| Medusa-1 heads without tree attention | Medusa-1 多解码头，但不使用树状注意力 | ~1.5× |
| Adding tree attention | 加入 Tree Attention（树状注意力） | ~1.9× |
| Using optimized tree configuration | 采用优化后的候选树结构 | ~2.2× |
| Training heads with Medusa-2 | 以 Medusa-2 方式训练解码头 | ~2.8× |

**数据校核**：原表只有两个列标题和四条记录，没有单独的 Table Note 脚注，也没有误差棒、置信区间、采样规模、GPU 或各阶段详细训练配置。这里的四个加速值均是约数（~）。四行表示**按阶段逐步增加技术之后的累计加速比**，不是可乘的独立因子；也不能把 1.5+1.9+2.2+2.8 当作总收益。

## 三、独立推导：逐阶段的绝对与相对收益（非原文）

设四阶段论文近似速度比为：
$$
S_{\mathrm{heads}}\simeq1.5,\quad
S_{\mathrm{tree}}\simeq1.9,\quad
S_{\mathrm{optimized}}\simeq2.2,\quad
S_{\mathrm{medusa2}}\simeq2.8.
$$

| 过渡阶段 | 速度倍数点增量（约） | 相对上一步新增提升（约） |
| --- | ---: | ---: |
| Heads → 加入 Tree Attention | +0.4× | 1.9/1.5−1 = 26.7% |
| Tree Attention → Optimized Tree | +0.3× | 2.2/1.9−1 = 15.8% |
| Optimized Tree → Medusa-2 | +0.6× | 2.8/2.2−1 = 27.3% |
| 最初 Heads → 最终 Medusa-2 | +1.3× | 2.8/1.5−1 = 86.7% |

四行都来自原表**约数**的二次计算，不应解读为严格的独立因果估计。最后一行表示完整 Medusa-2 相对最初仅 Heads 方案大约又快 1.87 倍，而不是相对原始 AR 基线 1.87 倍；完整相对 AR 是约 2.8 倍。

### 3.1 Medusa-1 Heads：~1.5×

多 Head 共享目标模型最后隐藏状态 $h_t$，分别预测位置 $t+k+1$ 的未来 Token。避免外部小 Draft Transformer 的连续前向，但远期 Heads 没有真实中间 Token 条件，草稿准确率会受影响。没有 Tree Attention 时，验证候选范围较有限。**Table 3 没有明确这一行的 Head 数、候选长度和每轮接受统计**，不能凭约 1.5× 反推出。

### 3.2 Tree Attention：~1.9×

不同 Head 的 Top-k 候选排列成分叉路径；Tree Attention 在一次 Target Forward 中用祖先可见 Mask 和按深度设置的 Position IDs 验证多条候选，减少共享前缀的重复计算。更多候选提高连续前缀接受机会，验证成本也上升。~1.9× 是这一配置的累计速度，而不是“Tree Attention 自身有 1.9×”。

### 3.3 Optimized Tree：~2.2×

Section 2.3.3 根据校准集估计不同 Head 排名准确率 $a_k^{(i)}$，对路径 $v=(i_1,\ldots,i_k)$ 使用乘积评分 $w(v)=\prod_{j=1}^{k}a_j^{(i_j)}$，在固定预算下优先添加最有希望的树节点。Section 3 Figure 4 显示优化稀疏树的 64 个节点在 Acceleration Rate 上能优于随机稠密树的 256 个节点，说明候选预算的**布局**比一味加宽更重要。但该近似目标与真实 Token/s 仍不完全相同。

### 3.4 Medusa-2 联合训练：~2.8×

Medusa-1 只训练额外 Heads，Medusa-2 允许 Backbone 和 Heads 在联合 LM Loss 下共同适应多 Token 预测，搭配差异化学习率、Heads Warmup 或 Self-Distillation。它能够提高草稿未来 Token 预测质量，扩大有效接受前缀。然而主干权重更新可能改变原模型分布，不能从 Table 3 推断其严格保真；~2.8× 也不是“只打开某个训练开关”的逐项训练消融结果。

### 3.5 加速率与单步开销不能混同

根据 Appendix B.1 指标定义，平均每个解码步骤推进的 Token 数为 $A$，该步耗时对普通 AR 步的比例为 $O$：
$$
\mathrm{Net\ Speedup}=S=\frac{A}{O}.
$$
例如 $A=3.5,O=1.25$，净加速是 2.8×，不是 3.5×。**Table 3 仅给 Speedup 的约数**，没有四个阶段各自的 $A$ 与 $O$，因此无法单独量化某个阶段在接受长度和开销上的贡献。原 Section 3 Table 1 有 Medusa-2 四模型的这两项指标，但不能用它倒推 Table 3 的所有阶段。

## 四、独立分析：Batch Size=1 的证据边界与可扩展性

### 4.1 为什么单请求通常收益更高

普通 AR 解码在小 Batch 下每次仅为少量新 Token 读取大量权重，算术强度相对低，容易被 HBM 带宽限制。Medusa 让目标一次验证多个树节点，提高计算复用率，并减少生成相同数量最终 Token 所需的串行 Target Forward 次数，尤其针对低 QPS、延迟敏感任务有潜在优势。

### 4.2 为什么 Batch 增大时加速未必保持

对 Batch Size B 的简化分析：
$$
S(B)\approx\frac{A(B)}{T_{\mathrm{Medusa\ step}}(B)/T_{\mathrm{AR\ step}}(B)}.
$$

大 Batch 的普通 AR 已通过 GEMM 复用权重，因此未必继续受 HBM 带宽限制。Medusa 验证树增加 Target 计算量与临时 KV/Mask/Scratch 访问，容易转向 Compute-Bound；B 变大后验证的候选 Token 大致随 B×树节点数增加，未接受的分支还可能浪费更多计算。不同序列实际接受长度不相同，连续批处理调度器必须处理每条序列的 KV Commit / Rollback、不同 Token Budget 以及 CUDA Graph Shape。Tensor Parallel All-Reduce、Prefix Cache、Decode Attention Kernel、Chunked Prefill 与低/高 QPS 也会影响性能。因此，论文 Batch=1 的 2.3–2.8× **不能直接推导到生产高并发吞吐**。

### 4.3 建议的现实评测协议

保持相同 Target、GPU、量化与采样参数，对 Baseline / 多 Heads / Tree / Optimized Tree / Medusa-2 逐级测试，按 QPS 与输出长度分桶：
- TTFT（Prefill/首 Token 延迟），TPOT（稳态每输出 Token 时间），P50/P95 端到端请求延迟；
- 单请求 Token/s 与整机输出 Token/s、可持续 Requests/s，避免混用延迟加速与吞吐；
- 每轮 Accepted Length、Acceptance Rate、Tree Nodes、Target Verification 时间、Draft Heads 时间与 GPU 峰值显存；
- 多卡 TP 开销、KV Cache 内存、Kernel launch、连续批处理调度；
- 质量分数与输出分布校核分开报告，特别记录 Greedy、Random Sampling、Typical Acceptance 各自规则。

这些都是工程建议，非论文 Section 4 报告的额外实验。

## 五、独立分析：三个不同层面的“无损”

**Backbone 参数不变**：Medusa-1 冻结主干参数，是权重层面的不变。Typical Acceptance 即便固定 Backbone 仍可能改变采样的 Token 分布。

**质量评估相近**：MT-Bench/GPT-4 评分相似是质量指标，不保证原有的逐 Token 概率完全不变。论文本节的“high-quality outputs”应按这一语义理解。

**精确分布保持**：标准 Speculative Sampling 在正确条件提议 $q$ 下用 $a(x)=\min(1,p(x)/q(x))$ 接受、拒绝时使用归一化正部残差 $[p-q]_+$，可以维持给定 Target $p$ 的采样分布。Tree Attention 只保证不同分支的因果条件不串扰，本身不是分布等价证明。Medusa 的 Typical Acceptance 则用 $p(x)>\min(\epsilon,\delta e^{-H(p)})$ 判断 Token 合理性，默认不提供这种严格概率修正。

此外，Medusa-2 修改了 Backbone。因此即便对**修改后的 Target** 使用正确精确采样，也无法据此证明与**最初未训练 Target** 的分布完全一致。

## 六、2026-11-03 Serving 框架状态（外部核实，非论文原文）

### 6.1 vLLM 主项目

- [vLLM 官方推测解码指南](https://docs.vllm.ai/en/latest/features/speculative_decoding/index.html)：列出 EAGLE、MTP、Draft Model、PARD、N-gram、Suffix 等多种方案，并说明 Speculative Decoding 通常更适用于低至中 QPS、带宽受限的场景。
- **重要时间敏感事实**：[vLLM Pull Request #61701](https://github.com/vllm-project/vllm/pull/61701) 于 **2026-10-15 合并到主线**，明确删除 Medusa 和 MLP Speculator，移除 method="medusa" 与 method="mlp_speculator" 配置（后者已不再使用），理由包括 Medusa 不支持 Model Runner V2，现有 EAGLE、MTP、DFlash、DSpark 可覆盖相关应用。**因此当前主线 vLLM 不应给用户提供已失效的 medusa speculative-config 启动示例。** 特定旧版本和 Fork 需单独确认。

### 6.2 vLLM-Ascend 是独立平台

- [vLLM-Ascend 官方推测解码指南](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/speculative_decoding.html) 仍明确列出 medusa（嵌入 Target 的多头）。但这是 Ascend 项目的支持列表，不能据此推断当前 vLLM 主线 CUDA 实现也支持。

### 6.3 SGLang

- [SGLang 官方推测解码文档](https://docs.sglang.io/docs/advanced_features/speculative_decoding) 当前主要列出 EAGLE-2/EAGLE-3、MTP、UNO、DFlash、独立 Draft Model、N-gram；**文档没有列出标准 Medusa 多头作为直接开箱启动的受支持算法**。这不等于理论上不能通过 Fork、自定义 Worker/Proposer 或其他扩展集成 Medusa。
- MTP 只在模型具备相应原生多 Token Prediction 架构时适用，不应把 Medusa Head 与 EAGLE/MTP 的内部条件机制混为一谈。现成 MTP/EAGLE/DFlash 能否启动应当按模型、Checkpoint、SGLang 版本分别验证。

### 6.4 作者官方实现及其他后端

- [FasterDecoding/Medusa 官方仓库](https://github.com/FasterDecoding/Medusa) 在 README 中说明其直接推理实现初期主要支持单 GPU、Batch=1，并列出 TensorRT-LLM、Hugging Face TGI、RTP-LLM 社区集成。
- [TensorRT-LLM Legacy Speculative Decoding 文档](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/legacy/advanced/speculative-decoding.md) 介绍 Medusa 的 Head 和树状候选验证，但 Legacy 文档不保证当前特定版本、硬件和模型组合仍能直接复现论文的 2.8×。
- 结论：研究本篇可用官方 Medusa 实现作为算法复现基线；生产 Serving 推荐优先检查目标模型与当前引擎对原生 MTP、EAGLE、DFlash 等路径的版本支持，再统一基准测试。不要把 2024 年论文的“易于集成”直接等同于 2026 年零改造即可部署。

## 七、EAGLE 与 DFlash 的跨论文联系（独立分析）

| 比较 | Medusa（2401.10774） | EAGLE（2401.15077） | DFlash（2602.06036） |
| --- | --- | --- | --- |
| Draft 表示 | 共享 $h_t$ 的不同未来 Token Head | Feature 自回归 + Shifted Token | Target 多层 Hidden Feature 条件化的并行 Block Diffusion |
| Draft 内部串行深度 | 多 Head 并行 | 多轮 Draft Forward | 一次并行 Block Draft Forward |
| 多路径/多 Token 验证 | Tree Attention | Target Tree Attention | Target 并行验证块 |
| 主要质量优化 | Medusa-2、校准优化树、Typical Threshold | Feature 预测质量与 Shifted Token | KV Injection、多层特征条件与块大小 |
| 分布保证 | Typical 可偏离；正确 Spec Sampling 可保持 | 正确精确采样可保持 | 正确 Target Verification 可保持 |
| 真实性能限制 | Batch=1 结果不等于高 QPS | 多轮 Draft 成本、Tree Verify | Block Draft + Verify 成本、并发与 Serving 后端 |

Medusa 说明“树上计算预算如何分配”能带来实际速度变化；EAGLE 侧重“如何让草稿 Feature 更准确”，但需要多轮 Draft；DFlash 则侧重“减少 Draft 本身的串行深度”。这些方向构成可比较的优化坐标，但不应直接按跨论文最大 Speedup 给出绝对排名。尤其 Medusa 的 Typical Acceptance 是近似采样，不能与使用严格分布修正的 EAGLE/DFlash 直接作为同等分布约束的算法比较。

## 八、本单元完成与下一阅读位置

- 完成 Section 4 全部两段原文翻译；Table 3 原图在对话裁取展示，表题与四行数据完整转录。
- 独立解释每阶段大约 1.5×→1.9×→2.2×→2.8× 的**累计**贡献，分析 Batch=1 / 高并发适用条件和分布保持。
- 结合 2026-11-03 官方资料核验 vLLM 主线 Medusa 移除与 vLLM-Ascend 独立支持，并核对 SGLang 文档列表；关联已读 EAGLE/DFlash。
- 下一单元：**Appendix A Related Work**（按 PDF 原始顺序正式进入附录）。全文尚未完成，不应生成已完成状态或 summary.md。
