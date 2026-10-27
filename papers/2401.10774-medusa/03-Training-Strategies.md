# Section 2.2 Training Strategies — 完整中文翻译、公式推导和 PyTorch 复现

- 论文：Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads
- 原文版本：arXiv:2401.10774v3，2024-06-14
- 位置：[PDF 第 4 页](https://arxiv.org/pdf/2401.10774v3#page=4)、[PDF 第 5 页](https://arxiv.org/pdf/2401.10774v3#page=5)
- 本单元阅读：2.2 Training Strategies、2.2.1 Medusa-1: Frozen Backbone、2.2.2 Medusa-2: Joint Training、2.2.3 How to Select the Number of Heads。
- **本节没有新增 Figure 或 Table**，正式编号公式为 (1)、(2)。先完整翻译后独立讲解，不将 Appendix B 的超参数混入本节原文。
- [可运行的 PyTorch 教学示例](examples/medusa_training_example.py)（单元测试验证冻结主干/联合梯度；非论文官方实现）。

## 一、Section 2.2 全文翻译

### 2.2 Training Strategies — 训练策略

在最基本的情况下，我们可以冻结主干模型（Backbone Model），只对 Medusa Heads 进行微调。然而，将主干模型与 Medusa Heads 一起训练，可以显著提高 Medusa Heads 的预测准确率。根据可用的计算资源以及具体应用场景的要求，我们提出了两种不同层级的 Medusa Heads 训练策略。

在本节中，我们假设存在一个与目标模型输出分布相匹配的训练数据集。这个数据集可以是原本用于目标模型监督微调（Supervised Fine-Tuning，SFT）的数据集。我们将在 Section 2.3.2 中讨论如何通过自蒸馏（Self-Distillation）方法消除对这种训练数据集的依赖。

### 2.2.1 Medusa-1: Frozen Backbone — 冻结主干模型

**自然段 1（原文顺序）**：为了在冻结主干模型的情况下训练 Medusa Heads，我们可以使用 Medusa Heads 的预测结果与真实标签（Ground Truth）之间的交叉熵损失（Cross-Entropy Loss）。具体而言，给定位置 $t+k+1$ 的真实 Token $y_{t+k+1}$，第 $k$ 个解码头的损失为 $\mathcal L_k=-\log p_t^{(k)}(y_{t+k+1})$，其中 $p_t^{(k)}(y)$ 表示第 $k$ 个解码头预测 Token $y$ 的概率。我们还观察到，当 $k$ 增大时，$\mathcal L_k$ 通常也会增大。这是合理的，因为第 $k$ 个解码头需要预测更加遥远的未来 Token，其预测的不确定性也更高。因此，我们可以为 $\mathcal L_k$ 添加权重 $\lambda_k$，用于平衡不同解码头之间的损失。Medusa 的总损失为：

$$
\boxed{\mathcal L_{\mathrm{MEDUSA-1}}=\sum_{k=1}^{K}-\lambda_k\log p_t^{(k)}(y_{t+k+1})}\tag{1}
$$

**自然段 2（原文顺序）**：在实际应用中，我们将 $\lambda_k$ 设为某个常数（例如 $0.8$）的第 $k$ 次幂。由于主干模型只需要提供隐藏状态，因此我们可以使用量化后的主干模型来减少显存占用。这使得 LLM 推理加速技术能够被更广泛地使用，因为通过量化，可以像 QLoRA（Dettmers et al., 2023）一样，在单张消费级 GPU 上为大型模型训练 Medusa。训练过程只需要数小时。例如，在一张 NVIDIA A100 PCIe GPU 上，使用 60,000 条 ShareGPT 样本训练 Vicuna-7B 的 Medusa-1，大约需要 5 小时。

### 2.2.2 Medusa-2: Joint Training — 联合训练

**引入自然段**：为了进一步提高 Medusa Heads 的预测准确率，我们可以将 Medusa Heads 与主干模型一起训练。然而，这需要特殊的训练方案，以保留主干模型原有的下一 Token 预测能力和输出质量。为此，我们提出三项策略。

**策略一：Combined Loss（联合损失）**。为了保持主干模型预测下一 Token 的能力，我们需要在 Medusa Loss 之外，加入主干模型的交叉熵损失 $\mathcal L_{\mathrm{LM}}=-\log p_t^{(0)}(y_{t+1})$。我们还添加权重 $\lambda_0$，用于平衡主干模型与 Medusa Heads 的损失。因此，总损失为：

$$
\boxed{\mathcal L_{\mathrm{MEDUSA-2}}=\mathcal L_{\mathrm{LM}}+\lambda_0\mathcal L_{\mathrm{MEDUSA-1}}}\tag{2}
$$

**策略二：Differential Learning Rates（差异化学习率）**。由于主干模型已经得到了充分训练，而 Medusa Heads 仍然需要更多训练，因此，我们可以为两者设置不同的学习率。这样既能使 Medusa Heads 更快收敛，也能保留主干模型原有的能力。

**策略三：Heads Warmup（解码头预热）**。我们注意到，在训练开始时，Medusa Heads 的损失通常很大，这会产生较大的梯度，并且可能使主干模型的参数发生不利变化。借鉴 Kumar et al.（2022）的思想，我们可以采用两阶段训练过程。在第一个阶段，我们按照 Medusa-1 的方式，只训练 Medusa Heads。在第二个阶段，我们利用 Warmup 策略，将主干模型与 Medusa Heads 一起训练。具体而言，我们首先对主干模型进行若干个 Epoch 的训练，然后再将 Medusa Heads 与主干模型联合训练。除了这一简单策略，我们也可以采用更加复杂的 Warmup 方法，即逐渐增大主干模型损失的权重 $\lambda_0$。我们发现，这两种策略在实际应用中都表现良好。

**总结自然段**：将这些策略结合起来，我们就能够在不损害主干模型原有能力的情况下，对 Medusa Heads 与主干模型进行联合训练。此外，这一训练方案也可以与监督微调（SFT）结合使用，使我们能够得到原生支持 Medusa 的模型。

### 2.2.3 How to Select the Number of Heads — 如何选择解码头数量

**全文一段**：根据经验，我们发现最多使用五个解码头通常就已经足够。因此，我们建议训练五个解码头，并参考 Section 2.3.3 中介绍的方法，确定最优的 Tree Attention 配置。在优化 Tree Attention 后，有时只需要三个或四个解码头就足以完成推理。在这种情况下，我们可以忽略多余的解码头，而不会产生额外的推理开销。

## 二、重要原文校核：式 (2) 和 Warmup 对 λ0 的归属不一致

式 (2) 明确写 $\mathcal L_{\mathrm{LM}}+\lambda_0\mathcal L_{\mathrm{MEDUSA-1}}$，即 **λ0 乘在 Medusa 多头辅助损失上**；然而 Heads Warmup 段落的原文又写“gradually increasing the weight λ0 of the backbone model’s loss”，即“逐渐增大主干模型损失的权重 λ0”。按公式该措辞不成立。这是原文内部用词/记号不一致，**已忠实保留译文**，不能静默更正为某个作者没有明确写出的 Warmup 算法。

若遵循式 (2)，把 λ0 从 0 逐步调高，相当于逐渐增大辅助多头 Loss 的权重；若目的是相对增加 LM Loss 权重，则需重新定义目标函数或相应调低 λ0。这两种不相同，不应混同。正文没有给定完整 Warmup 时间表，所附 PyTorch 代码只是一个符合公式的工程示例。

## 三、独立数学推导与数值核验（非原文）

### 3.1 多头前向与标签偏移

给定主干在位置 $t$ 的隐藏状态 $h_t$，原 LM Head 预测 $y_{t+1}$，第 $k$ 个额外 Medusa Head 预测 $y_{t+k+1}$（$k=1,\dots,K$）。在 Teacher Forcing 中，真实整条序列可以一次输入 Causal Transformer 得到所有 $h_t$，但 Head $k$ 训练标签必须向后偏移 $k+1$ 个位置。

例如序列 $[x_0,\dots,x_5]$：原 Head 用 $h_{0:4}$ 对齐 $x_{1:5}$；Head 1 用 $h_{0:3}$ 对齐 $x_{2:5}$；Head 2 用 $h_{0:2}$ 对齐 $x_{3:5}$；Head 5 因序列太短而没有有效标签。伪代码：

~~~python
for k, head in enumerate(heads, start=1):
    shift = k + 1
    logits_k = head(hidden_states[:, :-shift, :])
    targets_k = labels[:, shift:]
~~~

由于输入 Attention 必须是 Causal 的，$h_t$ 无法看到位置 $t$ 之后的真实 Token；向后 Shift 只是构造监督信号，不意味着把未来真值作为 Head 输入。训练集的 Pad/Prompt Token 可以在 Labels 中设为 -100，损失仅计算有效目标位置。

### 3.2 多头 Cross-Entropy 与 0.8^k

对一个位置，交叉熵是 $\mathcal L_k=-\log p_t^{(k)}(y_{t+k+1})$。对实际 Batch 的有效训练位置可使用 Mean Reduction：
$$
\overline{\mathcal L}_k=-\frac{1}{|\mathcal I_k|}
\sum_{(b,t)\in\mathcal I_k}\log p_{b,t}^{(k)}(y_{b,t+k+1}).
$$
这种 Batch/Token 平均是一般工程实践，原论文式 (1) 只写了单个位置的损失，**未在 2.2.1 指定每个 Head 的 Batch 聚合归一化方式**。

设 $K=5$，$\lambda_k=0.8^k$，各 Head 权重为：

| k | 预测位置 | λk | 若 λ0=0.2，联合损失实际 Head 系数 |
| --- | --- | --- | --- |
| 1 | t+2 | 0.80000 | 0.160000 |
| 2 | t+3 | 0.64000 | 0.128000 |
| 3 | t+4 | 0.51200 | 0.102400 |
| 4 | t+5 | 0.40960 | 0.081920 |
| 5 | t+6 | 0.32768 | 0.065536 |

采用递减权重是为了缓解远端位置更高不确定性带来的损失不平衡，也吻合推测解码中“越靠前的草稿预测越重要”的工程直觉。并不能据此推导各层真实梯度范数严格按相同系数下降，亦不能由此证明条件熵随位置严格单调。

### 3.3 数值示例

假设 $\mathcal L_{\mathrm{LM}}=2.0$，五 Head 的损失依次为 $[2.2,2.4,2.6,2.8,3.0]$：
$$
\mathcal L_{\mathrm{MEDUSA-1}}=0.8(2.2)+0.64(2.4)+0.512(2.6)+0.4096(2.8)+0.32768(3.0)=6.75712.
$$
若 $\lambda_0=0.2$，则：
$$\mathcal L_{\mathrm{MEDUSA-2}}=2+0.2(6.75712)=3.351424.$$
以上为独立演示，不是论文训练结果。

### 3.4 为什么联合 Loss 保留下一 Token 能力

令 Backbone 的共享参数为 $\theta$，各 Head 参数为 $\phi_k$。Medusa-1 仅更新 $\phi_k$，冻结 $\theta$。Medusa-2 有：
$$
\nabla_\theta\mathcal L_{\mathrm{MEDUSA-2}}
=\nabla_\theta\mathcal L_{\mathrm{LM}}
+\lambda_0\sum_{k=1}^K\lambda_k\nabla_\theta\mathcal L_k.
$$
这表明多个未来预测任务会向 Backbone 反向传播梯度，有可能与原 LM 的下一 Token 梯度冲突。因此 LM Loss 是保留原任务的直接训练约束，不是输出分布的严格数学保证；原论文的“保持质量”由实际实验支持。$\lambda_0$、各 $\lambda_k$ 控制显式损失系数，不严格决定最终梯度范数。

### 3.5 差异化学习率与论文 Appendix B 的补充训练参数

本节只描述独立学习率的思想；具体数字出现在 **Appendix B**：
- Appendix B.2：五个 1-layer Medusa Heads；$\lambda_k=0.8^k$；M-2 的 Medusa Head LR 是 Backbone LR 的 4 倍；LoRA/QLoRA；LoRA rank=32、alpha=16、dropout=0.05，作用于包括 LM Head 的线性层；AdamW 8-bit，余弦学习率调度。
- Appendix B.3（Vicuna-7B/13B）：global batch 64；主干峰值 LR $5\times10^{-4}$，Heads 为 $2\times10^{-3}$；40 warmup steps；4-bit Backbone；先完成 M-1 训练，再以此初始化 M-2；M-2 使用 QLoRA，式 (2) $\lambda_0=0.2$。
- Appendix B.4（自蒸馏实验）：主干 LoRA 峰值 LR $10^{-4}$，20 warmup steps，$\lambda_0=0.01$。这些场景的不同设置不能混为一个通用配置。

### 3.6 Two-Stage Heads Warmup 的实现语义

原文同时有如下表述：（a）第一阶段只训练 Heads 如 M-1；（b）第二阶段 Backbone 与 Heads 联合训练；（c）具体说先训练 Backbone 几个 Epoch，之后再一起训练；（d）另一策略逐渐增大“Backbone Loss 的权重 λ0”。（c）可能指第二阶段内部额外先训练 Backbone、然后再联合；但具体学习率和 Epoch 调度不充分，不能自作主张将原文变成一个确定的唯一流程。更重要的是（d）与式 (2) 的 λ0 归属不符。

**工程示例**：先冻结 Backbone 训练 Heads，再解冻 Backbone 的 LoRA/可训练参数，设置 Heads 学习率更高，联合优化式 (2)，可选逐步引入 $\lambda_0$（注意这会强化 Medusa Loss）。这是一种易实现且与式 (2) 一致的近似方案，不代表作者正式复现实验配置完整公开。

### 3.7 为什么最多训练 5 个 Head、推理未必用满 5 个

Head 数 $K$ 决定最远预测的 Token 距离，但增加 Head 需要更多投影计算、Top-k 候选构造、Tree Verification。较远 Head 的接受贡献通常受前面 Token 是否成功接受所限制。训练 5 个、推理仅选 3–4 个是一种灵活方式；不用的 Head 若不进行 Forward，自然没有它们的输出计算开销。是否最优仍取决于 2.3.3 的 Tree Search 和实际 GPU/Batch 设置。Section 2.2.3 的“5”是作者经验结论，不是严格的理论上界。

## 四、PyTorch 教学实现：可运行的训练目标与梯度检查

代码文件：**[examples/medusa_training_example.py](examples/medusa_training_example.py)**。在本次会话已使用 PyTorch CPU 运行过其等价源码：Medusa-1 Heads 有梯度、Backbone 无梯度；Medusa-2 Backbone 与 Heads 均有梯度。它包含 DummyLM 仅用于形状与反向传播验证，**不是 Causal Transformer，不可将演示损失解读为实验性能**。

核心部分如下：

~~~python
import torch
from torch import nn
from torch.nn import functional as F

class MedusaHead(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int, lm_head: nn.Linear):
        super().__init__()
        self.fc = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out = nn.Linear(hidden_size, vocab_size, bias=False)
        nn.init.zeros_(self.fc.weight)
        with torch.no_grad():
            self.out.weight.copy_(lm_head.weight)

    def forward(self, hidden):
        return self.out(F.silu(self.fc(hidden)) + hidden)

def medusa_loss(hidden, heads, labels, decay=0.8):
    total = hidden.new_zeros(())
    for k, head in enumerate(heads, start=1):
        shift = k + 1
        if hidden.size(1) <= shift:
            continue
        target = labels[:, shift:].contiguous()
        valid = (target != -100).sum()
        if not valid.item():
            continue
        logits = head(hidden[:, :-shift, :])
        ce = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), target.reshape(-1),
            ignore_index=-100, reduction="sum"
        ) / valid
        total = total + (decay ** k) * ce
    return total
~~~

完整示例文件还包含 training_loss：在 Medusa-1 下用 no_grad() 前向 Backbone 并训练 Heads；在 Medusa-2 下从 Backbone Logits 构造 $t+1$ 的 LM Loss，最后返回 $\mathcal L_{\mathrm{LM}}+\lambda_0\mathcal L_{\mathrm{MEDUSA-1}}$。

**示意 Optimizer 设置**（真实模型请以 LoRA/QLoRA 可训练参数替代全量 Backbone）：

~~~python
optimizer_stage1 = torch.optim.AdamW(heads.parameters(), lr=2e-3)

backbone_trainable = [p for p in backbone.parameters() if p.requires_grad]
optimizer_stage2 = torch.optim.AdamW([
    {"params": backbone_trainable, "lr": 5e-4},
    {"params": heads.parameters(), "lr": 2e-3},
])
~~~

训练循环应显式检查 Backbone 中训练参数列表非空；在两阶段切换时，重新构造 Optimizer/参数组。如果实际使用 QLoRA，需把底层量化权重保持冻结，只训练 LoRA Adapter 与额外 Heads，并考虑不同张量类型/设备、梯度累积、有效 Labels Mask、Position IDs、模型是否返回最后一层 Hidden States。

## 五、与 EAGLE、DFlash 的训练方法比较（独立分析）

| 对比 | Medusa | EAGLE（2401.15077） | DFlash（2602.06036） |
| --- | --- | --- | --- |
| 草稿监督对象 | 多个未来 Token 的 CE | 未来 Feature 回归 + Soft Token Distribution | 目标特征条件下的并行块 Token 预测 |
| 关键损失 | $\sum_k\lambda_k\mathcal L_k$；联合训练时加 LM Loss | Smooth L1 + $\lambda$×Soft-label CE | Block Diffusion CE，靠前位置损失可衰减 |
| Target Backbone | M-1 冻结；M-2 联合优化 | Target 维持冻结（论文方案） | 主体 Target 冻结、训练轻量 Draft Adapter |
| 不确定性应对 | 较远 Head 降权；可联合调整 Backbone | Shifted Token + 特征级建模 | 多层 Feature 和 KV Injection 条件化块预测 |
| 工程取舍 | 可获得更准多头但 M-2 有原 LM 损伤风险 | Feature Draft 仍含自回归序列成本 | 块内并行、但需扩散适配器与验证 |

这里重点在 **Head/Feature/Block 的训练目标差异**，并不意味着不同论文速度可直接横比。

## 六、结论与后续章节

- Medusa-1 冻结原模型，训练多步 Token Heads；式 (1) 配合 $\lambda_k=0.8^k$ 平衡不同距离的预测任务。
- Medusa-2 采用式 (2) 把原 LM Loss 和多个未来预测 Loss 联合起来，配合学习率分组和 Warmup，减少对原模型能力的负面影响。
- 原文 Warmup 关于 $\lambda_0$ 归属的叙述与公式有内部不一致；记录原式、不作静默更改。
- 训练时五个 Heads，推理时可根据 Tree Search 选择其中三到四个；不执行的 Head 不产生该 Head 的 Forward 开销。
- **下一阅读单元：Section 2.3 Extensions（Typical Acceptance、Self-Distillation、Searching for the Optimized Tree Construction）**。
