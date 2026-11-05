# Appendix G · Exploration and Modeling of Hardware Constraints and Medusa

**论文版本：** arXiv:2401.10774v3（2024-06-14）  
**原始 PDF：** https://arxiv.org/pdf/2401.10774v3  
**范围：** G 引言、G.1、G.2、G.3；Figure 9–23；Table 5–8。  
**数据边界：** 图注、公式和表格数据按 v3 PDF 转录。图中没有配套逐点 CSV，曲线估读不作为精确 Benchmark。Table 6–8 单元格格式为 TFLOP/s / Operational Intensity（FLOP/Byte）。PDF 原图页已核对，但 PNG 二进制未提交，页码见 figures/figure9-23-source.md。

## 一、Appendix G 引言完整翻译

### G. Exploration and Modeling of Hardware Constraints and MEDUSA

**G. Medusa 的硬件约束探索与性能建模**

我们通过引入一个简化的 Llama 系列模型，研究硬件约束，特别是内存带宽限制，以及这些约束对 Medusa 风格并行解码的影响。

首先，我们确定了涉及矩阵乘法的算子，例如线性层和注意力矩阵乘法，它们是主要的额外开销来源。

我们在多种 GPU 上分析 FLOP/s 与 Operational Intensity（运算强度）之间的关系，这些 GPU 包括 A100-80GB-PCIe、A40 和 A6000。

接下来，我们研究使用 Medusa 时，不同算子的 FLOP/s 与 Operational Intensity 如何变化。

最后，我们采用一个简单的解析模型计算加速率，并结合硬件 Benchmark。通过这种方式，我们分析不同模型规模、序列长度和 Batch Size 对性能的影响。

## 二、G.1 Roofline Model of Operators

### 2.1 正文完整翻译

**G.1 算子的 Roofline 模型**

我们对大型语言模型（LLM）中的各种算子进行 Roofline Model（屋顶线模型）分析，重点研究 Llama-7B、Llama-13B 和 Llama-33B（Touvron et al., 2023）。

这些模型在不同 GPU 上进行 Benchmark，包括 A100-80GB-PCIe、A40 和 A6000。

我们研究了三类矩阵乘法算子，因为它们是这些模型中计算开销的主要来源。

我们的研究参考了 Chen（2023）的报告。该报告研究 Batch Size 的有效性，而我们的工作更加关注解码和并行解码。

Table 5 详细列出了 Prefill、Decoding 和 MEDUSA Decoding 三个阶段中各个算子的计算复杂度和空间复杂度。

涉及的算子包括 Query、Key、Value 矩阵对应的线性层 \(XW_Q\)、\(XW_K\)、\(XW_V\)；Attention 矩阵乘法 \(QK^T\)、\(PV\)；以及 FFN 中的 Up、Gate、Down 线性层 \(XW_u\)、\(XW_g\)、\(XW_d\)。

其中，\(b\) 表示 Batch Size，\(s\) 表示 Sequence Length，\(h\) 表示 Hidden Dimension，\(i\) 表示 Intermediate Dimension，\(n\) 表示 Attention Head 数量，\(d\) 表示每个 Attention Head 的维度，\(q\) 表示 MEDUSA 的候选长度。关于这些算子的更多细节，请参考 Touvron et al. (2023) 和 Chen (2023)。

Figure 9–17 展示了不同模型规模（7B、13B、33B）在不同设置下三类算子的 Benchmark 结果。为了评估每个算子的性能和吞吐量，我们选择 Batch Size 从 1 到 64、以 2 的幂递增，以及 Sequence Length 从 128 到 8192、以 2 的幂递增的配置；每个算子共有 49 种设置。

从所有图中，我们观察到，在不同 GPU 和不同模型规模下，每个算子在 Prefill 和 Decoding 阶段的数据点都聚集在非常相似的位置。

在 Prefill 阶段，增加 Batch Size 会改变 Attention 矩阵乘法的 FLOP/s（参见 Figure 9 中的 qk/pv init），但不会影响 Operational Intensity（对应 Figure 9 中的垂直虚线箭头）。相比之下，增加 Sequence Length 会同时影响 Prefill 阶段的 FLOP/s 和 Operational Intensity（对应 Figure 9 中的斜向虚线箭头）。

在 Decoding 阶段，Attention 矩阵乘法明显受到内存带宽限制。尽管改变 Batch Size 和 Sequence Length 会提高 FLOP/s，但 Operational Intensity 几乎保持不变（参见 qk/pv ar）。这表明 Self-Attention 机制对硬件资源的利用并不充分。

Prefill 阶段的线性层大多受到计算能力限制（参见 qkv mlp init 和 up/gate/down init）。

在 Decoding 阶段，线性层的数据点形成一条斜率与 GPU 内存带宽相同的直线（参见 qkv mlp ar 和 up/gate/down ar）。这表明 Decode 阶段的线性层同样受到内存带宽限制。

增加 Batch Size 可以通过更好的并行性，在内存带宽约束下提高实际达到的 FLOP/s 和 Operational Intensity。需要注意，线性层只处理新 Token，其计算与 Sequence Length 无关（参见 Table 5 的 Decoding 部分）。

### 2.2 Table 5 完整翻译与转录

**表题：** 不同阶段主要算子的计算复杂度与空间复杂度。该表基于 Chen（2023）报告中的 Table 2。

#### Prefill

| 算子 | 输入形状 | 输出形状 | 计算复杂度 | 空间复杂度 |
|---|---|---|---|---|
| \(XW_Q,XW_K,XW_V\) | \((b,s,h)\) | \((b,s,h)\) | \(O(bsh^2)\) | \(O(2bsh+h^2)\) |
| \(QK^T\) | \((b,n,s,d),(b,n,s,d)\) | \((b,n,s,s)\) | \(O(bs^2nd)\) | \(O(2bsnd+bs^2n)\) |
| \(PV\) | \((b,n,s,s),(b,n,s,d)\) | \((b,n,s,d)\) | 同组复杂度 | 同组复杂度 |
| \(XW_u,XW_g\) | \((b,s,h)\) | \((b,s,i)\) | \(O(bshi)\) | \(O(bs(h+i)+hi)\) |
| \(XW_d\) | \((b,s,i)\) | \((b,s,h)\) | 同组复杂度 | 同组复杂度 |

#### Decoding

| 算子 | 输入形状 | 输出形状 | 计算复杂度 | 空间复杂度 |
|---|---|---|---|---|
| \(XW_Q,XW_K,XW_V\) | \((b,1,h)\) | \((b,1,h)\) | \(O(bh^2)\) | \(O(2bh+h^2)\) |
| \(QK^T\) | \((b,n,1,d),(b,n,s,d)\) | \((b,n,s,1)\) | \(O(bsnd)\) | \(O(bsn+bsnd+bnd)\) |
| \(PV\) | \((b,n,s,1),(b,n,1,d)\) | \((b,n,1,d)\) | 同组复杂度 | 同组复杂度 |
| \(XW_u,XW_g\) | \((b,1,h)\) | \((b,1,i)\) | \(O(bhi)\) | \(O(b(h+i)+hi)\) |
| \(XW_d\) | \((b,1,i)\) | \((b,1,h)\) | 同组复杂度 | 同组复杂度 |

#### Parallel Decoding

| 算子 | 输入形状 | 输出形状 | 计算复杂度 | 空间复杂度 |
|---|---|---|---|---|
| \(XW_Q,XW_K,XW_V\) | \((b,q,h)\) | \((b,q,h)\) | \(O(bqh^2)\) | \(O(2bqh+h^2)\) |
| \(QK^T\) | \((b,n,q,d),(b,n,s,d)\) | \((b,n,s,q)\) | \(O(bsqnd)\) | \(O(bsqn+b(s+q)nd)\) |
| \(PV\) | \((b,n,s,q),(b,n,q,d)\) | \((b,n,q,d)\) | 同组复杂度 | 同组复杂度 |
| \(XW_u,XW_g\) | \((b,q,h)\) | \((b,q,i)\) | \(O(bqhi)\) | \(O(bq(h+i)+hi)\) |
| \(XW_d\) | \((b,q,i)\) | \((b,q,h)\) | 同组复杂度 | 同组复杂度 |

**维度说明：**
- 空间复杂度是渐进元素规模表达，不等于实际峰值显存字节数；具体实现还取决于数据类型、临时 Buffer、Kernel Fusion 和缓存策略。
- 原表 Parallel Decoding 的 Attention 张量形状按 PDF 原样保留。若直接按常规矩阵乘法维度解释，其转置布局不够直观；论文未在 Table 5 单独解释该表示方式，工程实现应以实际 Attention Kernel 的张量布局和 Tree Attention Mask 为准，不应擅自将原表改写。
- 同组的 QKV、Up/Gate、Down 算子共享复杂度表达式。

### 2.3 Table 5 的复杂度推导

QKV 投影：
\[
\begin{aligned}
\text{Prefill: }&O(bsh^2)\\
\text{Decode: }&O(bh^2)\\
\text{Medusa: }&O(bqh^2)
\end{aligned}
\]

FFN：
\[
\begin{aligned}
\text{Prefill: }&O(bshi)\\
\text{Decode: }&O(bhi)\\
\text{Medusa: }&O(bqhi)
\end{aligned}
\]

Attention：
\[
\begin{aligned}
\text{Prefill: }&O(bs^2nd)\\
\text{Decode: }&O(bsnd)\\
\text{Medusa: }&O(bsqnd)
\end{aligned}
\]

Medusa 将一次只处理一个新 Token 的线性层计算扩展为同时处理 \(q\) 个候选 Token。对于 \(X\in\mathbb R^{b\times q\times h}\)、\(W\in\mathbb R^{h\times i}\)，计算 \(Y=XW\) 的复杂度为 \(O(bqhi)\)，但权重矩阵仍只有 \(hi\) 个元素，并不随候选数复制 \(q\) 份。这是并行候选能够提高权重复用率的基础。

### 2.4 Roofline 模型

Operational Intensity：
\[
OI=\frac{\mathrm{FLOPs}}{\mathrm{Bytes\ Transferred}}
\quad[\mathrm{FLOP/Byte}]
\]

Roofline 理论性能上界：
\[
P_{\max}=\min(P_{\mathrm{peak}},BW\times OI)
\]

Ridge Point：
\[
OI_{\mathrm{ridge}}=\frac{P_{\mathrm{peak}}}{BW}
\]

| GPU | HBM 带宽 | 峰值计算性能 | Ridge Point（推算） |
|---|---:|---:|---:|
| A100 80GB PCIe | 1935 GB/s | 312 TFLOP/s | 161.2 FLOP/Byte |
| A40 | 696 GB/s | 149.7 TFLOP/s | 215.1 FLOP/Byte |
| A6000 | 768 GB/s | 181 TFLOP/s | 235.7 FLOP/Byte |

Ridge Point 是根据论文给出的带宽与峰值计算性能计算，不是论文直接列出的数值。OI 低于 Ridge Point 时更可能是 Memory-Bound；高于 Ridge Point 时更可能是 Compute-Bound。

### 2.5 Figure 9–17 图注完整翻译

- **Figure 9：** Llama-7B 在 A100-80GB-PCIe 上所有被测试算子的 FLOP/s 与 Operational Intensity 关系。虚线分别表示 HBM 带宽上限 1935 GB/s 和峰值计算性能 312 TFLOP/s。qkv mlp 表示将隐藏特征投影到 Q/K/V 特征的线性层；up/gate/down 表示 Attention Block 后的线性层；qk/pv 表示 Attention 矩阵乘法的两个步骤；ar 表示自回归解码，init 表示 Prefill。
- **Figure 10：** A100-80GB-PCIe 上 Llama-13B 各算子的性能图。
- **Figure 11：** A100-80GB-PCIe 上 Llama-33B 各算子的性能图。
- **Figure 12：** A40 上 Llama-7B 各算子的性能图。
- **Figure 13：** A40 上 Llama-13B 各算子的性能图。
- **Figure 14：** A40 上 Llama-33B 各算子的性能图。
- **Figure 15：** A6000 上 Llama-7B 各算子的性能图。
- **Figure 16：** A6000 上 Llama-13B 各算子的性能图。
- **Figure 17：** A6000 上 Llama-33B 各算子的性能图。

原图页码：Figure 9–10：PDF 第 18 页；Figure 11–12：第 19 页；Figure 13–14：第 20 页；Figure 15–16：第 21 页；Figure 17：第 22 页。来源记录见 figures/figure9-23-source.md。

### 2.6 Figure 9–17 的技术分析

- Prefill QKV/MLP 和 Up/Gate/Down 线性层具有较高矩阵并行度，主要落在 Compute-Bound 一侧。
- Decode 线性层每轮只处理新 Token，Batch Size 小时权重读取量相对计算量大，因而呈现 Memory-Bound 特征。
- Decode Attention 每轮用新 Query 访问历史 KV，计算量相对 KV 数据访问较少，OI 低，受内存带宽限制。
- Prefill 增加 Batch Size 会改变 Attention FLOP/s，但 OI 基本不变；增加 Sequence Length 会同时改变 FLOP/s 和 OI。
- Decode 阶段增加 Batch Size 能提高线性层并行度、FLOP/s 和 OI；线性层只处理新 Token，其计算复杂度与历史 Sequence Length 无关。
- 作者观察到同类算子在不同模型规模和 GPU 上的点位呈相似聚类，但不同 GPU 的 Roofline 上限不同。

## 三、G.2 FLOP/s vs. Operational Intensity Variations in MEDUSA

### 3.1 正文完整翻译

**G.2 Medusa 中 FLOP/s 与运算强度的变化**

我们研究 Medusa 如何改变 Operational Intensity 并提高 FLOP/s。实验选择 Llama-33B，并使用 A100-80GB-PCIe。

首先，我们研究 Attention 矩阵乘法。Figure 18 和 Table 6 展示了在 Batch Size 固定为 16 时，MEDUSA 带来的影响。

我们观察到，随着候选 Token 数量增加，FLOP/s 和 Operational Intensity 都有所提高（普通解码的结果用灰色点表示）。这表明 MEDUSA 能够利用额外的候选 Token 提高计算吞吐量。

与普通解码相比，在 Batch Size 为 16、Sequence Length 为 1024、候选 Token 数量为 64 的设置下，MEDUSA 实现了 FLOP/s 提高 44 倍、Operational Intensity 提高 41 倍。

Figure 19 和 Table 7 展示了在 Sequence Length 固定为 1024 时，MEDUSA 解码产生的影响。在这种情况下，增加 Batch Size 并不会提高 Operational Intensity。

接下来，我们研究线性层，重点关注 Up/Gate/Down 线性层，结果见 Figure 20 和 Table 8。

由于 Decode 阶段的线性层只处理未来 Token，而过去的 Token 已经缓存在 KV Cache 中，因此这些线性层的计算与 Sequence Length 无关。

我们改变 Batch Size 来观察其影响。随着 MEDUSA 在 Batch Size 增加时处理越来越多的候选 Token，我们观察到算子从 Memory-Bandwidth-Bound 区域转向 Compute-Bound 区域。

这种变化说明，MEDUSA 能够将线性层的性能特征从受显存带宽限制转变为受计算能力限制。

### 3.2 Figure 18–20 图注完整翻译

- **Figure 18：** Batch Size 为 16 时，Attention 矩阵乘法的 FLOP/s 与 Operational Intensity 之间的关系。
- **Figure 19：** Sequence Length 为 1024 时，Attention 矩阵乘法的 FLOP/s 与 Operational Intensity 之间的关系。
- **Figure 20：** 线性层的 FLOP/s 与 Operational Intensity 之间的关系。

三图实验环境：Llama-33B、A100-80GB-PCIe；硬件上限为 312 TFLOP/s 和 1935 GB/s；候选数量为 1、16、32、48、64、80、96、112。

### 3.3 Table 6 完整转录

**表题：** Llama-33B 在 A100-80GB-PCIe 上，Batch Size 为 16 时，Attention 矩阵乘法的 TFLOP/s 与 Operational Intensity。

| Sequence Length | 1 | 16 | 32 | 48 | 64 | 80 | 96 | 112 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 0.54 / 0.98 | 7.87 / 12.8 | 14.73 / 21.33 | 19.78 / 27.43 | 25.25 / 32.0 | 28.63 / 35.56 | 32.58 / 38.4 | 36.57 / 40.73 |
| 256 | 0.75 / 0.99 | 11.2 / 13.47 | 21.29 / 23.27 | 28.69 / 30.72 | 36.59 / 36.57 | 41.2 / 41.29 | 45.99 / 45.18 | 52.33 / 48.43 |
| 512 | 1.02 / 0.99 | 14.69 / 13.84 | 27.47 / 24.38 | 37.35 / 32.68 | 47.09 / 39.38 | 52.24 / 44.91 | 59.55 / 49.55 | 66.35 / 53.49 |
| 1024 | 1.24 / 0.99 | 17.42 / 14.03 | 32.15 / 24.98 | 43.89 / 33.76 | 54.8 / 40.96 | 60.19 / 46.97 | 68.28 / 52.07 | 75.45 / 56.44 |
| 2048 | 1.39 / 0.99 | 19.03 / 14.12 | 35.05 / 25.28 | 48.03 / 34.32 | 59.66 / 41.8 | 63.91 / 48.08 | 72.83 / 53.43 | 80.05 / 58.04 |
| 4096 | 1.48 / 0.99 | 19.8 / 14.17 | 36.59 / 25.44 | 50.4 / 34.61 | 62.29 / 42.23 | 65.84 / 48.65 | 74.86 / 54.13 | 82.06 / 58.87 |
| 8192 | 1.53 / 0.99 | 20.08 / 14.2 | 36.89 / 25.52 | 50.44 / 34.76 | 62.11 / 42.45 | 67.5 / 48.94 | 76.97 / 54.49 | 84.5 / 59.3 |

### 3.4 Table 7 完整转录

**表题：** Llama-33B 在 A100-80GB-PCIe 上，Sequence Length 为 1024 时，Attention 矩阵乘法的 TFLOP/s 与 Operational Intensity。

| Batch Size | 1 | 16 | 32 | 48 | 64 | 80 | 96 | 112 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.37 / 0.99 | 5.22 / 14.03 | 10.15 / 24.98 | 15.02 / 33.76 | 19.79 / 40.96 | 21.52 / 46.97 | 25.65 / 52.07 | 29.4 / 56.44 |
| 2 | 0.54 / 0.99 | 8.25 / 14.03 | 16.0 / 24.98 | 21.62 / 33.76 | 28.24 / 40.96 | 31.84 / 46.97 | 37.49 / 52.07 | 43.04 / 56.44 |
| 4 | 0.75 / 0.99 | 11.41 / 14.03 | 21.97 / 24.98 | 30.02 / 33.76 | 38.71 / 40.96 | 43.41 / 46.97 | 50.06 / 52.07 | 56.77 / 56.44 |
| 8 | 1.02 / 0.99 | 14.78 / 14.03 | 27.78 / 24.98 | 38.09 / 33.76 | 47.99 / 40.96 | 53.32 / 46.97 | 61.0 / 52.07 | 68.11 / 56.44 |
| 16 | 1.24 / 0.99 | 17.42 / 14.03 | 32.15 / 24.98 | 43.89 / 33.76 | 54.8 / 40.96 | 60.19 / 46.97 | 68.28 / 52.07 | 75.45 / 56.44 |
| 32 | 1.39 / 0.99 | 18.89 / 14.03 | 34.67 / 24.98 | 47.57 / 33.76 | 58.89 / 40.96 | 63.61 / 46.97 | 72.17 / 52.07 | 79.21 / 56.44 |
| 64 | 1.48 / 0.99 | 19.58 / 14.03 | 35.87 / 24.98 | 49.45 / 33.76 | 61.13 / 40.96 | 64.84 / 46.97 | 73.73 / 52.07 | 81.02 / 56.44 |

### 3.5 Table 8 完整转录

**表题：** Llama-33B 在 A100-80GB-PCIe 上，线性层（Up/Gate/Down）的 TFLOP/s 与 Operational Intensity。

| Batch Size | 1 | 16 | 32 | 48 | 64 | 80 | 96 | 112 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.26 / 1.0 | 19.95 / 15.95 | 39.69 / 31.79 | 58.4 / 47.53 | 76.57 / 63.17 | 94.4 / 78.7 | 111.91 / 94.14 | 128.64 / 109.47 |
| 2 | 2.51 / 2.0 | 39.66 / 31.79 | 76.53 / 63.17 | 112.05 / 94.14 | 145.73 / 124.71 | 130.67 / 154.89 | 129.1 / 184.69 | 148.56 / 214.12 |
| 4 | 5.03 / 4.0 | 76.44 / 63.17 | 145.8 / 124.71 | 128.85 / 184.69 | 167.85 / 243.17 | 201.19 / 300.21 | 236.93 / 355.85 | 195.91 / 410.14 |
| 8 | 10.06 / 7.99 | 145.72 / 124.71 | 168.26 / 243.17 | 236.83 / 355.85 | 221.11 / 463.14 | 207.79 / 565.44 | 236.95 / 663.07 | 227.8 / 756.36 |
| 16 | 19.96 / 15.95 | 168.35 / 243.17 | 221.41 / 463.14 | 237.5 / 663.07 | 224.71 / 845.59 | 232.49 / 1012.87 | 241.12 / 1166.74 | 229.25 / 1308.76 |
| 32 | 39.69 / 31.79 | 221.74 / 463.14 | 224.88 / 845.59 | 241.33 / 1166.74 | 239.02 / 1440.25 | 245.83 / 1675.97 | 243.55 / 1881.24 | 240.33 / 2061.59 |
| 64 | 76.57 / 63.17 | 225.19 / 845.59 | 239.2 / 1440.25 | 243.26 / 1881.24 | 246.16 / 2221.31 | 246.91 / 2491.55 | 244.52 / 2711.46 | 246.14 / 2893.91 |

### 3.6 Table 6–8 的数据分析

**Attention：候选增加提高 FLOP/s 和 OI。**

在 \(b=16,s=1024\) 时，普通 Decode（1 个候选）为 1.24 TFLOP/s、OI=0.99；64 个候选为 54.8 TFLOP/s、OI=40.96：

\[
\frac{54.8}{1.24}\approx44.19,\qquad
\frac{40.96}{0.99}\approx41.37
\]

与作者报告的 44× FLOP/s、41× OI 一致。这是算子吞吐与运算强度的提升，不是 GPU 峰值算力变成 44 倍。

**Attention：固定 Sequence Length 时，Batch Size 增加不改变 OI。**

Table 7 中，同一候选数量列的 OI 在不同 Batch Size 下保持不变。例如 64 个候选时，OI 始终为 40.96 FLOP/Byte；但 FLOP/s 从 Batch=1 的 19.79 提高到 Batch=64 的 61.13。原因是计算量和数据访问量随 Batch Size 近似同比增长，OI 比值保持不变，而并行任务数量提高了 GPU 利用率。

**Linear：Batch Size 与候选数量共同提高 OI，最终进入 Compute-Bound。**

A100 Ridge Point 约为 161.2 FLOP/Byte。Table 8 中，Batch=4、候选=16 时 OI=63.17，仍是 Memory-Bound；Batch=4、候选=48 时 OI=184.69，超过 Ridge Point；Batch=8、候选=32 时 OI=243.17；Batch=16、候选=16 时 OI=243.17；Batch=64、候选=112 时 OI=2893.91、FLOP/s=246.14 TFLOP/s。

最后一组的运算强度远高于 Ridge Point，实际吞吐也接近 A100 标称 312 TFLOP/s 峰值，表明线性层已从显存带宽受限转为计算吞吐受限。

**核心差异：** G.2 中 Attention 的 OI 最高约 56.44，仍低于 A100 Ridge Point；Linear 的 OI 则能达到数千 FLOP/Byte。因此，候选并行虽然显著改善 Attention，但线性层更容易真正跨越 Roofline 的 Ridge Point。

## 四、G.3 Predicting MEDUSA Performance

### 4.1 正文完整翻译

**G.3 预测 MEDUSA 的性能**

我们进一步使用一个简单的解析模型来估算加速率。

Section 3.3.1 中的消融实验结果表明，加速率可以近似表示为一个简单的对数函数。根据 Figure 4a 的结果，我们将该曲线建模为：

\[
\mathrm{acc\ rate}=0.477\log(\mathrm{num\ candidate})
\]

我们模拟 Llama-7B 模型中一个简化 Block 的延迟。该 Block 按顺序处理以下算子：

\[
XW_Q,\ XW_K,\ XW_V,\ QK^T,\ PV,\ XW_u,\ XW_g,\ XW_d
\]

首先将 Batch Size 固定为 1，Sequence Length 固定为 1024。通过构建 Section 2.1.2 中描述的 Tree Attention，对候选 Token 进行并行处理。

对于 MEDUSA，我们省略验证和接受等后处理步骤的延迟，因为这些步骤引入的额外开销很小。

Figure 21 展示了在上述设置下，不同候选 Token 数量对应的模拟加速率和 Speedup。

随着候选 Token 数量增加，加速率和 Speedup 最初都会提高。然而，当候选 Token 数量超过 64 后，Speedup 开始下降，这表明继续增加候选长度会产生递减收益。

这一结果与 Figure 4b 中的实验结果一致，并说明 MEDUSA 存在一个最优候选 Token 数量范围，在这个范围内能够获得最显著的性能提升。

我们在 Figure 22 中绘制了不同 Batch Size 设置下的模拟 Speedup，同时将 Sequence Length 固定为 1024。结果表明，当 Batch Size 超过 32 时，Speedup 会下降，甚至可能产生负面效果。这是因为线性层会从 Memory-Bandwidth-Bound 状态转变为 Compute-Bound 状态。

我们还使用 Batch Size=4、不同 Sequence Length 进行了另一组实验。如 Figure 23 所示，不同 Sequence Length 下的最优候选 Token 数量基本保持一致。然而，随着 Sequence Length 增加，整体性能会下降。

性能下降主要来自 Attention 矩阵乘法的额外开销，而线性层的计算保持不变，因为线性层的计算与 Sequence Length 无关。

### 4.2 原文最后总结段翻译

我们的模拟结果表明，对于 MEDUSA 的模型扩展而言，最优候选 Token 数量至关重要，因为超过某个范围后，继续增加候选数量所带来的收益会逐渐下降。

最初，增加 Batch Size 可以通过并行性提升性能，但过大的 Batch Size 会使线性层从 Memory-Bandwidth-Bound 转变为 Compute-Bound，从而降低加速比。

更长的序列会增加 Attention 矩阵乘法的开销，降低整体性能，这也凸显了优化 Attention 机制的必要性。

要有效地扩展模型，需要在候选 Token 数量之间进行权衡，调整 Batch Size 以避免进入计算受限状态，并针对更长序列改进 Attention 机制。

这些策略能够确保更好的硬件资源利用率和更高的性能，也体现了通过模拟预测性能、指导加速策略设计的价值。

### 4.3 Figure 21–23 图注完整翻译

- **Figure 21：** 在 Batch Size=1、Sequence Length=1024 的设置下，对 Llama-7B 在 A100 80GB PCIe 上使用不同候选 Token 数量进行消融，展示模拟加速率、Speedup 和归一化延迟。
- **Figure 22：** Llama-7B 在 Sequence Length=1024 时，不同候选 Token 数量对应的模拟 Speedup。
- **Figure 23：** Llama-7B 在 Batch Size=4 时，不同候选 Token 数量对应的模拟 Speedup。

原图页码：Figure 21 位于 PDF 第 26 页；Figure 22 和 Figure 23 位于 PDF 第 27 页。

### 4.4 Figure 21：候选数量与加速效果

实验设置：Llama-7B、A100 80GB PCIe、Batch=1、Sequence Length=1024。图中包含 Simulated Acceleration Rate、Simulated Speedup，以及 QK/PV、QKV Linear、Up/Gate/Down 的归一化自回归延迟分量。

作者观察到候选数量增加时 Acceleration Rate 持续上升，但 Speedup 在约 64 个候选之后开始下降。原因是并行验证延迟增长逐渐抵消了每轮推进 Token 数的收益。

图中没有提供逐点原始数值表，因此具体折线只能作近似读图，不应视为精确 Benchmark。

**公式边界问题：** PDF 打印的拟合式为 \(0.477\log(\mathrm{num\ candidate})\)，没有明确给出对数底数，也没有截距。但 Figure 21 在候选数为 1 时显示基准 Acceleration Rate=1，而打印公式在 \(\log(1)=0\) 时给出 0。原文没有解释这一差异。本文保留公式原貌并标注不一致，不擅自添加截距或指定对数底数。

### 4.5 Figure 22：Batch Size 对加速的影响

实验固定 Sequence Length=1024，比较 Batch Size=1、2、4、8、16、32、64。小 Batch 下增加候选数量能够获得较明显加速；Batch 增大后，Speedup 逐步下降；Batch=64 的曲线在候选数大于 1 后低于 1，表示 Medusa 可能比普通解码更慢。

图中纵轴印有 “Speedup (%)”，但刻度以 1.0 为基准、曲线按加速比方式展示。本文保留作者轴标签，并按其正文和图中基线解释为 Speedup 比率，不将 2.0 解读为 2%。

### 4.6 Figure 23：Sequence Length 对加速的影响

实验固定 Batch Size=4，比较 Sequence Length=128、256、512、1024、2048、4096、8192。不同序列长度下的最优候选数量相对接近，曲线峰值大致位于 32 个候选附近；Sequence Length 越长，整体 Speedup 越低。8192 长度的最佳曲线值约 1.4，短序列的峰值约 2.5，均为图像估读而非精确数据。

原因是 Attention 需要访问更长的历史 KV，Attention 矩阵乘法成本随序列长度增长；而 Decode 线性层只处理新候选 Token，其计算不随历史 Sequence Length 增长。

### 4.7 性能模型的适用范围与局限

1. 使用 Llama-7B 的简化单个 Block，而不是完整模型端到端 Serving。
2. Figure 21 固定 Batch=1、Sequence Length=1024。
3. Figure 22 固定 Sequence Length=1024，改变 Batch Size。
4. Figure 23 固定 Batch=4，改变 Sequence Length。
5. 将候选 Token 通过 Tree Attention 并行处理。
6. 忽略验证和接受后处理延迟，作者认为其开销较小。
7. Acceleration Rate 采用对数近似，打印公式的截距/对数底数存在未解释之处。

因此，G.3 更适合用作候选规模、Batch 和上下文长度的趋势分析工具，不能直接当作任意 GPU、模型和 Serving 框架的精确性能预测器。

## 五、Appendix G 综合技术分析

### 5.1 Medusa 为什么能够缓解 Memory-Bound？

普通 Decode 每轮只处理一个 Token，模型权重读取与计算之间的比例不利于 GPU 利用率。Medusa 在同一轮处理 \(q\) 个候选 Token：

\[
X\in\mathbb R^{b\times q\times h},\quad
W\in\mathbb R^{h\times i}
\]

权重仍被共享，计算量随候选数增长，减少了每个 Token 对权重读取的摊销成本。这提高线性层的 OI，使其在一定 Batch 和候选规模下从 Memory-Bound 转向 Compute-Bound。

### 5.2 为什么 Attention 仍是瓶颈？

Attention 必须访问历史 KV。候选数量提高并行计算量，但历史 KV 访问仍然存在。Table 6–7 中 Attention 的 OI 仍低于 A100 Ridge Point。随着 Sequence Length 增大，Attention 读取和计算开销增加，最终限制 Medusa 的整体加速。

### 5.3 候选数量为什么有最优值？

可以将 Speedup 理解为：
\[
S(q)=
\frac{A(q)}
{T_{\mathrm{Medusa}}(q)/T_{\mathrm{base}}}
\]
其中 \(A(q)\) 是平均推进 Token 数，分母表示每轮相对延迟。增加 \(q\) 初期提高平均推进 Token 数，同时也增加 QKV、FFN 和 Attention 的计算；超过硬件和接受率的有效区间后，延迟增长会抵消收益。

### 5.4 对实际推理系统的工程启示

1. Batch=1 本地推理是 Medusa 最容易发挥优势的场景，但仍要测量真实 Kernel 和端到端延迟。
2. 高 Batch 服务不能照搬单请求下的候选树；应按 Batch Size 重新扫描候选数量。
3. 长上下文需要单独测量 Attention 和 KV Cache 访存成本。
4. 需要分别 Profile QKV、QK/PV、Up/Gate/Down，避免只看整体 Tokens/s。
5. 最优候选数应通过真实端到端 Benchmark 确定，不能仅由 G.3 的简化模型推算。

## 六、Figure 9–23 和 Table 5–8 原图来源

- Figure 9–10：PDF 第 18 页。
- Figure 11–12：PDF 第 19 页。
- Figure 13–14：PDF 第 20 页。
- Figure 15–16：PDF 第 21 页。
- Figure 17：PDF 第 22 页。
- Figure 18：PDF 第 23 页。
- Figure 19–20：PDF 第 24 页。
- Table 5：PDF 第 17 页。
- Table 6–8：PDF 第 25 页。
- Figure 21：PDF 第 26 页。
- Figure 22–23：PDF 第 27 页。

完整 PDF：https://arxiv.org/pdf/2401.10774v3

原始 PDF 页已核对，图注与表格数据已保存到本章节；PNG 二进制没有上传到 GitHub。没有使用重新绘制的图片冒充原始论文图片。来源记录见 figures/figure9-23-source.md 和 tables/table5-8-source.md。

## 七、Appendix G 完成小结

- G 引言、G.1、G.2、G.3 全部正文按原文顺序翻译。
- Figure 9–23 的全部图注翻译并完成逐图技术分析。
- Table 5–8 的表题、表头、公式、形状和全部数值转录。
- 推导 Roofline 上界、Ridge Point、Attention 与 Linear 的复杂度及 OI 变化。
- 解释 Medusa 对 Memory-Bound 的改善、Attention 瓶颈、候选数量最优区间、Batch Size 和 Sequence Length 的影响。
- 标注 G.3 对数拟合公式与 Figure 21 基线不一致，以及 Figure 22 纵轴单位标签的歧义。
- 区分论文直接结论、表格计算和独立工程分析。

**Appendix G 完成。**
