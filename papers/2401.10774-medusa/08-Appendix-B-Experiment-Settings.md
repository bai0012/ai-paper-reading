# Appendix B Experiment Settings：B.1–B.4 完整中文翻译、原始超参数、代码校核与复现建议

**论文**：Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads（arXiv:2401.10774v3，ICML 2024）。**原文范围**：[PDF 第 14 页](https://arxiv.org/pdf/2401.10774v3#page=14) Appendix B.1–B.4 全部四个小节，各一个完整自然段；B.4 结束后进入 Appendix C。**本节无 Figure/Table，唯一给出的公式是 Speedup = Acceleration rate / Overhead（非单独编号）**。

**按来源明确区分**：第一部分是用户提供原始 v3 PDF 的逐段完整译文。之后的公式 (1)(2) 出自已读主文；Section 3 的数据由其正文补充；Axolotl 配置参数与 Sine 调度的代码校核来自作者指定的 Fork，而非 Appendix B 原句；PyTorch 片段与复现注意事项属于独立工程分析。

## 一、Appendix B 原文完整逐段翻译

### B. Experiment Settings（实验设置）

### B.1 Common Terms（常用术语）

我们明确三个常用术语：（a）**Acceleration rate（加速率）**：每个解码步骤平均解码的 Token 数量。在标准自回归模型中，这个值为 1.0。（b）**Overhead（额外开销）**：用于刻画相对于经典解码每一步的额外开销，其计算方式为 Medusa 模型每步平均延迟除以普通模型每步平均延迟。（c）**Speedup（加速比）**：指实际运行时间（Wall-time）的加速率。根据这些定义，有以下关系：

$$
\boxed{\mathrm{Speedup}=\frac{\mathrm{Acceleration\ rate}}{\mathrm{Overhead}}}.
$$

### B.2 Shared Settings（通用设置）

在所有实验中，我们使用 Axolotl（Axolotl, 2023）框架训练。学习率使用带 Warmup 的余弦调度（Cosine Learning Rate Scheduler），优化器使用 8-bit AdamW（Dettmers et al., 2021）。我们训练 **5 个 Medusa Heads，每个 Head 只有 1 层**，公式 (1) 中的 $\lambda_k$ 设置成 $0.8^k$。对于 Medusa-2，使用 LoRA（Hu et al., 2021）或 QLoRA（Dettmers et al., 2023）进行微调，并设置 Medusa Heads 的学习率为 Backbone 学习率的 **4 倍**。LoRA 作用于 Backbone 模型**全部线性层，包括语言模型输出头**（LM Head）。LoRA Adapter 的 Rank 为 **32**，LoRA $\alpha$ 为 **16**，Dropout 为 **0.05**。

### B.3 Medusa-1 v.s. Medusa-2 on Vicuna 7B and 13B（Vicuna-7B/13B 上的 Medusa-1 与 Medusa-2）

使用全局 Batch Size **64**。Backbone 峰值学习率为 $5\times10^{-4}$，Medusa Heads 峰值学习率为 $2\times10^{-3}$，学习率预热 **40 Steps**。对于两个模型都使用 **4-bit 量化 Backbone**。先用 Medusa-1 训练，再以训练完成的模型初始化 Medusa-2。Medusa-2 使用 **QLoRA**，且公式 (2) 的 $\lambda_0$ 设置为 **0.2**。

### B.4 Training with Self-Distillation on Vicuna-33B and Zephyr-7B（Vicuna-33B 与 Zephyr-7B 的自蒸馏训练）

对于这两个模型，我们直接使用 Medusa-2，**不采用两阶段训练流程**。我们对 **$\theta_0$** 使用 **Sine Schedule（正弦调度）**，使它逐步增大，并在训练结束时达到峰值。我们发现这种方法同样有效。由于自蒸馏损失相对较小，将 Backbone **LoRA Adapter** 的峰值学习率设为 $1\times10^{-4}$，学习率预热 **20 Steps**。把公式 (2) 中的 **$\lambda_0$ 设置为 0.01**。

**原文校核要点**：原文在 Sine Schedule 处明确印刷为 **$\theta_0$**，不是 $\lambda_0$。同段又明确说式 (2) 的 $\lambda_0=0.01$。不可在“译文”中静默改写原始符号。具体代码采用 Sine 调度辅助损失系数，详见下面代码核实。本节没有介绍该符号 $\theta_0$ 的独立数学定义。

## 二、原文超参数矩阵：勿混淆来源

| 设定 | B.3 Vicuna-7B/13B | B.4 Vicuna-33B/Zephyr-7B |
| --- | --- | --- |
| 训练方式 | Medusa-1→Medusa-2 两阶段 | Medusa-2 一阶段，自蒸馏 |
| Heads | 5 个、每个 1 层 | 5 个、每个 1 层 |
| Head k CE 权重 | $0.8^k$ | $0.8^k$ |
| 主干微调 | 4-bit + QLoRA | LoRA（作者配置明确没有 4-bit） |
| 主干峰值 LR | $5e^{-4}$ | $1e^{-4}$ |
| Medusa Heads 峰值 LR | $2e^{-3}$ | 一般 Head/Backbone=4；若同样适用则推得 $4e^{-4}$，**B.4 原文没有单独给出 Head LR 的数值** |
| 学习率 Warmup | 40 Steps | 20 Steps |
| 优化器/调度 | AdamW 8-bit、Cosine LR | AdamW 8-bit、Cosine LR；额外 Sine Medusa Loss 调度 |
| LoRA r / alpha / dropout | 32 / 16 / 0.05 | 32 / 16 / 0.05 |
| 式 (2) lambda0 | 0.2 | 0.01 |
| Global batch | 64（B.3） | 128（Section 3.2；**不是 B.4 本段**） |
| 训练序列长度 | 4096（Section 3.1） | 2048（Section 3.2） |
| 数据 | ShareGPT（Section 3.1） | 自蒸馏数据（Section 3.2） |
| Epochs | 2（Section 3.1） | 作者 Fork 配置为 2；**B.4 未给** |

**LoRA 缩放独立计算**：经典 LoRA $\Delta W=(\alpha/r)BA=(16/32)BA=0.5BA$。这里的 $\alpha$ 是 LoRA 缩放超参数，和此前 Typical Acceptance 的阈值超参数 $\alpha$ 不相同，必须区别。

## 三、三个性能指标的严格区分与例子

令普通 AR 的每轮平均耗时为 $t_0$，Medusa 每轮平均耗时为 $t_M$，Medusa 每轮实际解码的平均 Token 数量为 $A$：
$$
O=t_M/t_0,\quad S=\frac{A}{O}=\frac{A\,t_0}{t_M}.
$$
例如普通 AR 每轮花 20ms 且只生成 1 Token；Medusa 每轮花 25ms 但平均推进 3.5 Token，则 Overhead=$25/20=1.25$、Acceleration Rate=3.5、净 Speedup=$3.5/1.25=2.8×$。

**论文数据核验**：Section 3 Table 1 Vicuna-7B $A=3.47,O=1.22,S_{\mathrm{report}}=2.83$；由表中四舍五入后数值计算是 $3.47/1.22\approx2.844$，与论文 2.83 不完全一致。其他模型存在约 0.001–0.024 的差距，可能涉及不同聚合或精度，原文没有提供逐样本计时，不能擅自修改正式加速比。

**另一个复现要点**：每次验证成功的输出 Token 数包括经过 Target Head 直接产生、保底输出的 Token，因此需统一 Acceleration Rate 是“每轮最终输出的 Token”还是“额外被接受的 Heads Token”。注意 Term 一致并使 Baseline=1。

## 四、5 个 Heads、Loss 权重和联合训练的数学含义

主文 Section 2.2 式 (1)：
$$
\mathcal L_{\mathrm{M1}}=\sum_{k=1}^5\lambda_k\mathcal L_k,\qquad
\mathcal L_k=-\log p_t^{(k)}(y_{t+k+1}),\quad \lambda_k=0.8^k.
$$

| Head k | Label 相对偏移 | $0.8^k$ | Medusa-2 B.3 有效系数（$\lambda_0=0.2$） | 自蒸馏 B.4 峰值系数（$\lambda_0=0.01$） |
| --- | --- | ---: | ---: | ---: |
| 1 | $t+2$ | 0.80000 | 0.160000 | 0.008000 |
| 2 | $t+3$ | 0.64000 | 0.128000 | 0.006400 |
| 3 | $t+4$ | 0.51200 | 0.102400 | 0.005120 |
| 4 | $t+5$ | 0.40960 | 0.081920 | 0.004096 |
| 5 | $t+6$ | 0.32768 | 0.065536 | 0.0032768 |

单步所有 Head 权重总和 $2.68928$；乘上 B.3 的 0.2 是 0.537856；乘上 B.4 的 0.01 是 0.0268928。这只是系数加和，实际不同 Head 的 CE 与梯度范数不同，不应把它解读为实际 Loss、梯度或速度提升的比例。

主文 Section 2.2 式 (2)：
$$
\mathcal L_{\mathrm{M2}}=
\mathcal L_{\mathrm{LM}}+\lambda_0\sum_{k=1}^5(0.8^k)\mathcal L_k.
$$
在 Medusa-1 下 Backbone 冻结，只更新 Heads；Medusa-2 可训练 Backbone 的 LoRA/QLoRA Adapter 和各 Heads。Head k 的监督标签应当向右偏移 $k+1$ 个位置，Head 1 是预测**第二个未来 Token**，而不是 $t+1$。有 Prompt/Padding Mask 时，需要只计算具有有效未来 Target 的位置；序列长度不足的 Head 不能参与该 Batch 的均值损失。

**实际 Fork 与抽象公式的一个差异**：作者 Fork 的 Stage 1 config 也写了 medusa_heads_coefficient=0.2，而 its Medusa loss implementation even in medusa_only_heads mode applies this global factor to each extra-head loss；因此源码的 Stage1 目标是 $0.2\sum_k 0.8^k\mathcal L_k$，而论文式 (1) 没有单独的 0.2 全局项。这是**实际配置与论文表达的差异**，不能默改原文或不加说明地将其等价为完全相同的数值训练目标。该因子未必简单等效于乘 learning rate（AdamW 具有自适应更新与 epsilon 项）。

## 五、Cosine LR + Warmup 与 Sine Medusa Schedule（最重要的复现校核）

### 5.1 作者原始 Medusa Fork 的证据

论文作者的 [FasterDecoding/Medusa README](https://github.com/FasterDecoding/Medusa) 明确指向其专用 [ctlllll/axolotl Fork 的 examples/medusa](https://github.com/ctlllll/axolotl/tree/main/examples/medusa)，并说明主要修改在 src/axolotl/utils/models.py。我们已读取真实代码 [src/axolotl/monkeypatch/medusa_utils.py](https://github.com/ctlllll/axolotl/blob/main/src/axolotl/monkeypatch/medusa_utils.py) 及以下四份原 YAML：
- [Vicuna-7B QLoRA Stage 1](https://github.com/ctlllll/axolotl/blob/main/examples/medusa/vicuna_7b_qlora_stage1.yml)
- [Vicuna-7B QLoRA Stage 2](https://github.com/ctlllll/axolotl/blob/main/examples/medusa/vicuna_7b_qlora_stage2.yml)
- [Vicuna-33B Together](https://github.com/ctlllll/axolotl/blob/main/examples/medusa/vicuna_33b_together.yml)
- [Zephyr Together](https://github.com/ctlllll/axolotl/blob/main/examples/medusa/zephyr_together.yml)

### 5.2 精确 Sine 调度函数（来自作者 Fork 代码，不是论文 B.4 的正式编号公式）

在源码的 replace_compute_loss 内，medusa_scheduler="sine" 对应：

~~~python
medusa_scheduler_coefficient = math.sin(
    self.state.global_step / self.state.max_steps * math.pi / 2
)
~~~

因此可以设 $u=s/S$，已实现的系数是：
$$
\boxed{g(u)=\sin\left(\frac{\pi}{2}u\right),\quad u\in[0,1].}
$$

实际训练 Loss 中**只对 Medusa Heads Auxiliary CE**乘上：
$$
\lambda_0(s)=\lambda_0^{\max}g(s/S)
$$
而 LM/自蒸馏 KL Loss 不乘这个系数。B.4 作者配置 lambda0 最大值 0.01，所以自蒸馏时的实现可表示为：
$$
\boxed{
\mathcal L_{\mathrm{B4}}(s)
=
D_{\mathrm{KL}}(p_{\mathrm{teacher}}\Vert p_{\mathrm{student}})
+0.01\,\sin\!\left(\frac{\pi s}{2S}\right)
\sum_{k=1}^5 0.8^k\mathcal L_k.
}
$$
这就是**对 B.4 的原始“$\theta_0$ Sine”印刷问题作出代码层面澄清**：论文印刷记号 $\theta_0$ 没有定义，而源码明确将 Sine multiplier 用于 Medusa 辅助损失，不是 Backbone LR，也不是 LM KL 权重。仍保留原文 $\theta_0$，不把它说成作者正式定义的符号。

| 训练进度 $u$ | Sine 系数 $g(u)$ | B.4 额外 Heads 总系数前因子 $0.01g(u)$ |
| --- | ---: | ---: |
| 0% | 0.000000 | 0.000000 |
| 25% | 0.382683 | 0.003827 |
| 50% | 0.707107 | 0.007071 |
| 75% | 0.923880 | 0.009239 |
| 100% | 1.000000 | 0.010000 |

Sine 从头到尾逐渐增加辅助损失影响，初期优先保证 KL/LM、后期渐进引入 Heads 目标。这是作者代码实际行为的解释，不能与 Cosine LR 混同。

**Cosine Learning Rate Schedule** 另行控制优化器，例如在 Warmup 后可按（仅给一种常见工程实例，不代表原始调度器所有边界条件）：
$$
\eta(s)=\eta_{\max}\cdot \frac12\left(1+\cos\left(\pi\frac{s-s_w}{S-s_w}\right)\right).
$$
这里 $s_w$ 是 Warmup Steps（B.3 40，B.4 20），实际完整实现需考虑 Warmup 初始 LR、End LR 和各优化器 Group 的倍数。

## 六、两个训练流程与真实 YAML 核验（补充作者配置，非 B.1–B.4 原文）

### 6.1 Vicuna-7B/13B：两阶段 QLoRA

原始 Stage 1 核心配置：

~~~yaml
base_model: lmsys/vicuna-7b-v1.5
adapter: qlora
load_in_4bit: true
lora_r: 32
lora_alpha: 16
lora_dropout: 0.05
lora_target_modules:
  - gate_proj
  - down_proj
  - up_proj
  - q_proj
  - v_proj
  - k_proj
  - o_proj
  - lm_head
sequence_len: 4096
micro_batch_size: 2
gradient_accumulation_steps: 4
num_epochs: 2
optimizer: adamw_bnb_8bit
lr_scheduler: cosine
learning_rate: 0.0005
warmup_steps: 40
medusa_num_heads: 5
medusa_num_layers: 1
medusa_heads_coefficient: 0.2
medusa_decay_coefficient: 0.8
medusa_scheduler: constant
medusa_lr_multiplier: 4.0
medusa_only_heads: true
~~~

这些是作者旧 Fork 的专用 Medusa 扩展字段，不是当前 Axolotl 标准配置必然接受的选项。保留原样便于复现，不能只复制到最新普通 Axolotl 就声称可用。 Stage 2 在上述基础上：从 Stage 1 的输出加载 LoRA/Medusa Checkpoint（原 YAML 的 lora_model_dir 指向 ./vicuna_7b_qlora_stage1），不启用 medusa_only_heads，使 Backbone 的 QLoRA Adapter 与 Heads 联合更新。两阶段均在原 YAML 设置 2 Epoch、40 Warmup Steps、Cosine Scheduler 和 Peak LR 5e-4，并使用 Head LR Multiplier=4，所以 Heads Peak LR 为 2e-3。

**Stage 1 YAML 的微妙点**：虽然 config 声明 adapter=qlora，但训练开关 medusa_only_heads=true 让模型只优化额外 Heads。加载量化 Backbone 不等于 Stage 1 正在微调所有量化后的 LoRA Adapter；是否有梯度应通过 Trainable Parameter List 验证。

### 6.2 Vicuna-33B / Zephyr-7B：直接联合 LoRA + KL 蒸馏

作者 Fork 两个原始 together.yml 都包含：

~~~yaml
adapter: lora
load_in_4bit: false
lora_r: 32
lora_alpha: 16
lora_dropout: 0.05
sequence_len: 2048
num_epochs: 2
optimizer: adamw_bnb_8bit
lr_scheduler: cosine
learning_rate: 0.0001
warmup_steps: 20
medusa_num_heads: 5
medusa_num_layers: 1
medusa_heads_coefficient: 0.01
medusa_decay_coefficient: 0.8
medusa_lr_multiplier: 4.0
medusa_scheduler: sine
medusa_self_distillation: true
~~~

这里的 learning_rate=1e-4 用于 Backbone LoRA 参数，Head LR multiplier 为 4，按 Fork 配置对应 Heads 4e-4。**推导 $4e^{-4}$ 有实际 YAML 证据，但 Appendix B.4 自身没有印刷这个数值**。两种模型使用未量化 LoRA，符合 Section 2.3.2 的 Teacher 建议。作者的代码关闭 LoRA Adapter 进行无梯度 Teacher Forward，随后开启 Adapter 进行 Student/Heads Forward；对 Student 的下一 Token logits 与 Teacher soft labels 计算 KL，对多头未来 Token 计算衰减 CE。

作者 YAML 具体 Batch：
- Vicuna-33B：micro_batch_size=1，gradient_accumulation_steps=16；
- Zephyr-7B：micro_batch_size=4，gradient_accumulation_steps=4。
Section 3.2 报告 global batch 128。按单机标准 DDP $B_{\mathrm{global}}=B_{\mathrm{micro}}\times\mathrm{grad\ accum}\times N_{\mathrm{GPU}}$，这两份 config 要达到 128 均对应 $N_{\mathrm{GPU}}=8$；这**是对数值的反算**，不是论文 B.4 声明必须用 8 张卡。更多分布式拓扑/梯度累积定义应按具体训练框架验证。

**数据**：Section 3.2 明确对两模型生成约 100k 条自蒸馏样本，Vicuna-33B 随机生成温度 0.3。官方 [data_generation/README.md](https://github.com/FasterDecoding/Medusa/blob/main/data_generation/README.md) 提供 Prompt→Response 的 vLLM 批量生成脚本说明；带 --chat 模式会逐轮喂入原始 Prompt，适合不支持 Self-Talk 的模型。Zephyr 可使用只给首轮 Prompt 的 Self-Talk 模式。真实运行前应确定 Chat Template、上下文窗口、EOS、Data Leakage、防止把原种子 Assistant Reply 当 Teacher 目标。

## 七、PyTorch 教学实现与关键检查点（非官方完整训练器）

### 7.1 一个 loss 函数同时覆盖两种策略

~~~python
import math
import torch
from torch.nn import functional as F

def medusa_aux_loss(hidden, heads, labels, decay=0.8):
    # hidden: [B,T,d], labels: [B,T]; -100 are ignored
    total = hidden.new_zeros(())
    for k, head in enumerate(heads, start=1):
        shift = k + 1                   # head 1 predicts token t+2
        if hidden.shape[1] <= shift:
            continue
        targets = labels[:, shift:].contiguous()
        valid = targets.ne(-100).sum()
        if valid.item() == 0:
            continue
        logits = head(hidden[:, :-shift, :])
        per_head = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        ) / valid
        total = total + (decay ** k) * per_head
    return total

def medusa_sine(step: int, total_steps: int) -> float:
    u = min(1.0, max(0.0, step / total_steps))
    return math.sin(0.5 * math.pi * u)

# Medusa-1: backbone with no_grad, optimize heads-only
# m1_loss = medusa_aux_loss(hidden.detach(), heads, labels)

# B.3 Medusa-2: QLoRA on backbone, aux coefficient 0.2
# joint_loss = lm_next_token_ce + 0.2 * medusa_aux_loss(...)

# B.4 single-stage self-distillation: LoRA adapter OFF for
# teacher, ON for student; KL(teacher || student) + sine Heads
# loss = kl_teacher_student + 0.01 * medusa_sine(step, max_steps) * aux
~~~

注意 PyTorch 切换 Adapter 需严格防止 Teacher Logits 被 Student 的 LoRA 影响，并确保在相同上下文和词表上比较概率。实际部署代码还应校验 LoRA 作用模块包含 LM Head，处理 Tied Embeddings、Mixed Precision、4-bit 权重梯度、梯度检查点、分布式同步和生成数据 Mask。用户之前阅读 Section 2.2 保存的 [Medusa 双模式完整梯度校验教学代码](examples/medusa_training_example.py) 已实现了公式 (1)(2) 的独立计算。

### 7.2 Teacher KL 的准确方向

原文 Section 2.3.2：
$$
\mathcal L_{\mathrm{distill}}=D_{\mathrm{KL}}(p_{\mathrm{original}}||p_{\mathrm{student}}).
$$
在 PyTorch 中：
~~~python
teacher_logp = teacher_logits.float().log_softmax(-1).detach()
student_logp = student_logits.float().log_softmax(-1)
per_token_kl = F.kl_div(
    student_logp, teacher_logp.exp(), reduction="none"
).sum(dim=-1)
valid = labels[:, 1:].ne(-100)
loss_kl = (per_token_kl * valid).sum() / valid.sum()
~~~
这是更标准的**按有效 Token 数归一化**示例。源码 Fork 自身用 reduction="sum" 并除以 token_positions 的总数，Mask 处理和精确尺度可能有差别；若目标是数值一致地复现作者代码，应保留源码归一化，而不能静默替换为此处改良版本。两种归一化都实现 KL 的同方向梯度，但权重尺度和最终优化轨迹不同。

### 7.3 Optimizer 参数分组与学习率同步

~~~python
from bitsandbytes.optim import AdamW8bit
from transformers import get_cosine_schedule_with_warmup

optimizer = AdamW8bit([
    {"params": backbone_lora_params, "lr": 5e-4},
    {"params": medusa_head_params, "lr": 2e-3},
], weight_decay=0.0)
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=40,
    num_training_steps=total_optimizer_steps,
)
~~~
这一示例对应 B.3 两组**峰值学习率**，实际参数必须过滤 requires_grad=True，并确保每个 Parameter 只属于一个 Optimizer Group。调用 scheduler.step() 要与 Optimizer Step 保持同步，不能误按 Microbatch Step 调度。B.4 应将 Backbone LR 改为 1e-4，Heads LR 改为作者 YAML 所示倍率的 4e-4、Warmup 20，但仍额外对 Auxiliary Loss 应用 Sine Multiplier；这与 Cosine LR 并行存在。

### 7.4 全局 Batch 的检验与调整

$$
B_{\mathrm{global}}=B_{\mathrm{micro}}\times\mathrm{grad\ accum}\times N_{\mathrm{GPU}}.
$$
对官方 YAML：
- Vicuna-7B Stage 1 micro=2、accum=4；若 8 GPUs，则 global=64，符合 B.3；
- Vicuna-33B micro=1、accum=16；若 8 GPUs，则 global=128，符合 Section 3.2；
- Zephyr micro=4、accum=4；若 8 GPUs，则 global=128，符合 Section 3.2。

若仅有单卡，可通过相应增大 Gradient Accumulation（例如 B.3 micro=2、accum=32 达到 global=64）保持**名义样本数**一致，但梯度累积策略、Sample Packing、序列长度、BN/Dropout 随机数、Gradient Checkpointing、Optimizer Step 与数据顺序仍会影响复现一致性。不要把梯度累积增加视作必然完整等价于多卡 DDP，也要重新核实 Warmup Step 在新的 optimizer_steps 体系中的长度。

### 7.5 开源 Axolotl 普通版本与作者专用 Fork 的边界

作者使用的专用字段包括 medusa_num_heads、medusa_num_layers、medusa_heads_coefficient、medusa_decay_coefficient、medusa_scheduler、medusa_lr_multiplier、medusa_only_heads、medusa_self_distillation。这些字段由其 Fork 的 [monkeypatch/medusa_utils.py](https://github.com/ctlllll/axolotl/blob/main/src/axolotl/monkeypatch/medusa_utils.py) 注入和计算。**只安装最新主线 Axolotl 并把这些字段写进 YAML，不保证有 Medusa 模型插入和自定义 Loss 支持。** 需要实际安装、pin 相应作者 Fork commit 并检查 Axolotl、Transformers、PEFT、bitsandbytes、Accelerate、FlashAttention 之间的兼容性；否则建议自己给当前 HF Trainer/SFTTrainer 写 Custom Model + Custom compute_loss + Custom Optimizer Groups。

近期标准 [Axolotl 官方配置参考](https://docs.axolotl.ai/docs/config-reference.html) 支持 adapter、load_in_4bit、lora_r、lora_alpha、lora_target_modules、learning_rate、warmup_steps、optimizer、cosine 等基本键，但不等于标准 Trainer 自动包含论文特定的 Medusa Heads、未来 Token Shift 与 Teacher KL。

**官方仓库的旧 medusa/train/train_legacy.py 命令属于初始版本的最小 Medusa-1 训练示例**，它在 README 中使用 3 Heads、seq_len=2048、其他 LR/Batch；不能误拿其超参数声称完全对应本文 Appendix B 的 5 Heads/Qlora 配置。用户应优先参考上文作者 Fork 的 examples/medusa 四个 YAML。

## 八、需要保留的原文/代码差异与复现实验限制

1. **B.4 $\theta_0$ 与式 (2) $\lambda_0$**：原 PDF 使用未定义的 $\theta_0$，源码实际对 Medusa 辅助 CE 乘 $g(s)=\sin(\pi s/2S)$；原句保留、代码含义另列。
2. **Medusa-1 代码也含 medusa_heads_coefficient**：Stage 1 YAML 为 0.2，源码在 medusa_only_heads=True 时依旧对 Heads Loss 乘它；论文式 (1) 未列额外整体系数，训练数值并非完全逐字符相同。
3. **自蒸馏 KL 归一化与软标签数值**：源码 Teacher 关闭 Adapter，无梯度；对训练位置计算 KL，但按原日志/源码分母归一化。更标准的有效 Mask Token Mean 会改变 Loss 尺度；想复现论文就需要复现源码处理。
4. **正文只报告训练设置，不包含完整随机种子、全部运行环境、模型权重精确 Revision、最终分布式拓扑和 Trainer Patch 版本**。作者提供 Fork/YAML/生成脚本有助于复现，但现代依赖升级可能无法一键运行。
5. **不能混用 B.3/B.4 的量化设置**：B.3 使用 QLoRA + 4-bit；B.4 官方 YAML 使用未量化 LoRA，以避免 Self-Distillation Teacher 被量化而与原输出分布不同。
6. **速度评价与质量评价须分别完成**：训练完成后仍应按 Appendix B.1 标准测 Acceleration Rate、Overhead、Wall-Time Speedup，并核对正常解码质量及 Typical Acceptance 的分布偏差；不能由 Train Loss 直接推出推理速度。

## 九、下一阅读断点

**Appendix B.1–B.4 全部完成**（原 PDF 第 14 页，正文、所有原始参数和公式均保留）；独立保存了作者实际 Axolotl Fork 配置、Sine Loss Scheduler 的精确公式、两阶段与自蒸馏训练实现差异、对真实训练可复现性的校核。

**下一节：Appendix C. Visualization of Optimized Tree Attention，包含 Figure 6 原始树形示意图及图注。** 此后尚有 D/E/F/G，整篇论文尚未完成。
