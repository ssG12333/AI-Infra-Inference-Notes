# 深度学习算子全解：从单一操作到组合架构

> 循序渐进地理解深度学习中的每一个计算单元——从最简单的加法到最前沿的 Mamba 块。

---

## 阅读路线

```
第一章 基石          第二章 构建          第三章 组合          第四章 进化
┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│ 数值变换  │     │ 空间处理  │     │ 经典架构  │     │ 前沿架构  │
│ · 数学运算│ --> │ · 卷积   │ --> │ · CNN块  │ --> │ · 注意力 │
│ · 激活函数│     │ · 池化   │     │ · RNN单元│     │ · SSM   │
│ · 归一化  │     │ · 注意力 │     │ · 编解码 │     │ · MoE   │
│ · 张量操作│     │ · 循环   │     │ · 检测   │     │ · 多模态│
│ · Loss    │     │ · 嵌入   │     │ · 拆解   │     │ · 量化  │
│ · 硬件    │     │ · 算法   │     │          │     │ · 部署  │
│ · 优化器  │     │ · 内存   │     │          │     │          │
│ · 调度    │     │          │     │          │     │          │
└──────────┘     └──────────┘     └──────────┘     └──────────┘
   逐元素            单层            多层组合          系统级

第五章 部署实战                          第六章 模型压缩
┌──────────────────┐                 ┌──────────────────┐
│ · 内存管理       │                 │ · 剪枝          │
│ · 图优化         │                 │ · 知识蒸馏      │
│ · 平台适配       │                 │ · 量化进阶      │
│ · 性能调优       │                 │ · INT4/INT2     │
│ · 量化调试       │                 │ · KV Cache量化  │
└──────────────────┘                 └──────────────────┘
   推理工程                            压缩技术
```

---

## 目录

- [第一章 基石算子——逐元素的世界](#第一章-基石算子逐元素的世界)
  - [1.1 数学运算：加减乘除的魔法](#11-数学运算加减乘除的魔法)
  - [1.2 激活函数：注入非线性](#12-激活函数注入非线性)
  - [1.3 归一化：稳定数值范围](#13-归一化稳定数值范围)
  - [1.4 张量操作：形状与维度的魔法](#14-张量操作形状与维度的魔法)
  - [1.5 Loss 函数](#15-loss-函数损失函数)
  - [1.6 硬件执行模型](#16-硬件执行模型)
  - [1.7 优化器（训练视角）](#17-优化器训练视角)
  - [1.8 学习率调度](#18-学习率调度)
- [第二章 构建算子——从单元素到空间](#第二章-构建算子从单元素到空间)
  - [2.1 线性变换：仿射世界的基石](#21-线性变换仿射世界的基石)
  - [2.2 卷积：局部特征的捕获器](#22-卷积局部特征的捕获器)
    - [2.2.4 卷积算法对比](#224-卷积算法对比)
  - [2.3 池化：空间信息的压缩](#23-池化空间信息的压缩)
    - [2.3.1 内存布局](#231-内存布局memory-layout)
  - [2.4 注意力：全局依赖的建立](#24-注意力全局依赖的建立)
  - [2.5 循环：时序记忆的载体](#25-循环时序记忆的载体)
  - [2.6 嵌入：离散到连续的桥梁](#26-嵌入离散到连续的桥梁)
- [第三章 组合结构——从算子到架构块](#第三章-组合结构从算子到架构块)
  - [3.1 CNN 基础块](#31-cnn-基础块)
  - [3.2 RNN 单元](#32-rnn-单元)
  - [3.3 Transformer Block](#33-transformer-block)
  - [3.4 编解码架构](#34-编解码架构)
  - [3.5 检测头与后处理](#35-检测头与后处理)
  - [3.6 ResNet 完整拆解](#36-resnet-完整拆解)
  - [3.7 ViT 完整拆解](#37-vit-完整拆解)
  - [3.8 LLaMA/Qwen 完整拆解](#38-llamaqwen-完整拆解)
- [第四章 进化架构——前沿模型的算子组合](#第四章-进化架构前沿模型的算子组合)
  - [4.1 注意力变体](#41-注意力变体)
  - [4.2 状态空间模型](#42-状态空间模型)
  - [4.3 混合专家](#43-混合专家)
  - [4.4 量化与部署](#44-量化与部署)
    - [4.4.1 量化基础概念](#441-量化基础概念)
    - [4.4.2 Quantize/Dequantize/Requantize](#442-quantize--dequantize--requantizencnn-三大量化层)
    - [4.4.3 Convolution INT8](#443-convolution-int8-推理ncnn-实现详解)
    - [4.4.4 SDPA INT8](#444-sdpa-int8-推理llm-专用)
    - [4.4.5 静态量化校准](#445-静态量化校准流程)
    - [4.4.6 AWQ/GPTQ](#446-awq--gptq权重量化校准)
    - [4.4.7 FP8 量化](#447-fp8-量化)
    - [4.4.8 PagedAttention](#448-pagedattentionkv-cache-分页管理)
    - [4.4.9 Tensor Parallelism](#449-tensor-parallelism张量并行)
    - [4.4.10 INT8 与 Vulkan 互斥](#4410-int8-与-vulkan-互斥)
  - [4.5 多模态](#45-多模态)
- [第五章 部署实战——从模型到产品](#第五章-部署实战从模型到产品)
  - [5.1 ncnn 内存管理](#51-ncnn-内存管理)
  - [5.2 ncnn 图优化](#52-ncnn-图优化)
  - [5.3 ncnn 平台适配](#53-ncnn-平台适配)
  - [5.4 性能调优](#54-性能调优)
  - [5.5 量化调试](#55-量化调试)
- [第六章 模型压缩——更小更快的模型](#第六章-模型压缩更小更快的模型)
  - [6.1 剪枝](#61-剪枝pruning)
  - [6.2 知识蒸馏](#62-知识蒸馏knowledge-distillation)
  - [6.3 量化进阶](#63-量化进阶)
- [附录 算子速查表](#附录-算子速查表)
  - [A. 算子分类总表](#a-算子分类总表)
  - [B. 组合结构速查表](#b-组合结构速查表)
  - [C. 框架参数命名对照](#c-框架参数命名对照)
  - [D. ncnn 量化层参数详解](#d-ncnn-量化层参数详解)
  - [E. 算子 FLOPs 公式速查](#e-算子-flops-公式速查)
  - [F. 模型参数量与 FLOPs 对照表](#f-常见模型参数量与-flops-对照表)
  - [G. ncnn Param 文件格式规范](#g-ncnn-param-文件格式完整规范)
  - [H. 梯度推导手册](#h-梯度推导手册)
  - [实战案例：Qwen3 0.6B](#实战案例qwen3-06b-模型详解)

---

# 第一章 基石算子——逐元素的世界

> 深度学习的所有复杂操作，都可以追溯到对单个数值的简单变换。本章从最基础的逐元素操作开始。

## 1.1 数学运算：加减乘除的魔法

### 1.1.1 二元运算（BinaryOp）

两个张量之间的逐元素运算，是所有复杂运算的原子操作。

| 操作 | 公式 | 典型用途 |
|------|------|---------|
| **Add** | a + b | 残差连接、偏置加法 |
| **Sub** | a − b | 残差计算 |
| **Mul** | a × b | 门控（SwiGLU）、注意力权重×Value |
| **Div** | a / b | 归一化、缩放 |
| **Pow** | a^b | 幂运算、RoPE 频率 |
| **Max** | max(a, b) | 最大池化的原子操作 |
| **Min** | min(a, b) | 截断 |

**广播（Broadcasting）**

> 当两个张量形状不完全一致时，框架会自动"复制"较小的张量，使其能与较大的张量进行逐元素运算。

**核心规则**：从**最后一个维度**开始向前比较，每个维度要么**相等**，要么**其中一个为 1**。

```
张量 A: [B, C, 1, 1]
张量 B: [B, 1, H, W]
结果:   [B, C, H, W]    ← 所有维度都扩展

逐维度比较:
  维度 0: B vs B   → 相等，保留 B
  维度 1: C vs 1   → 1 扩展为 C
  维度 2: 1 vs H   → 1 扩展为 H
  维度 3: 1 vs W   → 1 扩展为 W
```

**直观理解**：`[B,C,1,1]` 相当于 C 个标量，`[B,1,H,W]` 相当于一张 H×W 的图。广播后，每个通道的标量被"涂满"整张图。

**不能广播的情况**：`[3,4] + [2,4]` → ❌ 维度 0 不匹配 (3 vs 2，都不为1)

> 在深度学习中，BN 的 μ/σ/γ/β（形状为 `[C]`）广播到 `[B,C,H,W]` 做逐通道操作；注意力掩码（`[seq,seq]`）广播到 `[batch,heads,seq,seq]`。

> 💡 **Mul 是门控的基石**：从 LSTM 的遗忘门到 SwiGLU，所有"选择性传递信息"的机制都建立在逐元素乘法之上——一个操作数作为"门控信号"（0~1 范围），另一个作为"被控信息"。

### 1.1.2 一元运算（UnaryOp）

单个张量的逐元素变换：

| 操作 | 公式 | 典型用途 |
|------|------|---------|
| **Abs** | \|x\| | L1 正则化 |
| **Neg** | −x | 梯度反转 |
| **Sqrt** | √x | RMSNorm、L2 范数 |
| **Exp** | eˣ | Softmax、Pos 编码 |
| **Log** | ln(x) | 损失函数、信息熵 |
| **Floor / Ceil / Round** | ⌊x⌋ / ⌈x⌉ / round(x) | 量化取整 |
| **Sign** | sign(x) | 梯度截断 |

### 1.1.3 归约运算（Reduction）

沿指定维度将多个值聚合为一个：

| 操作 | 公式 | 输出形状（沿 dim=1 归约 [B,C,H,W]） |
|------|------|--------------------------------------|
| **Sum** | Σxᵢ | [B, H, W] |
| **Mean** | (Σxᵢ)/n | [B, H, W] |
| **Max** | max(xᵢ) | [B, H, W] |
| **Prod** | Πxᵢ | [B, H, W] |
| **L2Norm** | √(Σxᵢ²) | [B, H, W] |

> 归约是"从多到少"的核心操作。GlobalAvgPool 本质就是 Mean 归约，Softmax 中的分母是 Sum 归约。

### 1.1.4 矩阵乘法（MatMul / Gemm / InnerProduct）

从逐元素运算升级到矩阵运算——这是神经网络参数化的核心。

```
MatMul:  C = A × B            [M,K] × [K,N] → [M,N]

Gemm:    Y = α·op(A)·op(B) + β·C   含转置选项和缩放

InnerProduct:  y = W·x + b     本质是 Gemm 的特例（全连接层）
```

> 💡 **三者的关系**：InnerProduct ⊂ Gemm ⊂ MatMul。InnerProduct 封装了权重和偏置；Gemm 支持转置和缩放；MatMul 是最通用的矩阵乘法。

**为什么矩阵乘法如此重要？** 几乎所有带参数的操作都是矩阵乘法：
- 卷积 → im2col + MatMul
- 全连接层 → MatMul + Bias
- Q/K/V 投影 → MatMul
- Embedding → Gather（特殊的 MatMul）

### 1.1.5 EinSum（爱因斯坦求和约定）

EinSum 用简洁的字符串表达任意张量缩并，编译器可自动优化计算顺序。

```python
# 矩阵乘法: [M,K] × [K,N] → [M,N]
torch.einsum('ij,jk->ik', A, B)

# 注意力 Q·Kᵀ: [B,H,Q,D] × [B,H,K,D] → [B,H,Q,K]
torch.einsum('bhqd,bhkd->bhqk', Q, K)

# 注意力 ×V: [B,H,Q,K] × [B,H,K,D] → [B,H,Q,D]
torch.einsum('bhqk,bhkd->bhqd', Attn, V)

# 逐通道乘 (SE Block): [B,C,1,1] × [B,C,H,W] → [B,C,H,W]
torch.einsum('bc,bchw->bchw', scale, x)

# Batch Matrix Multiply: [B,M,K] × [B,K,N] → [B,M,N]
torch.einsum('bik,bkj->bij', A, B)
```

**规则速查**：

| 模式 | 含义 |
|------|------|
| `'ij,jk->ik'` | 消去重复字母 j，输出 ik |
| `'i->'` | 全归约（Sum） |
| `'ij->ji'` | 转置 |
| `'i,i->i'` | 逐元素乘 |
| `'...ij,...jk->...ik'` | 广播矩阵乘法（任意前置维度） |

> 💡 EinSum 是 MatMul 的泛化：任何线性代数运算都可以写成 EinSum。现代框架会自动将其分解为最优的底层算子（BatchMatMul、逐元素乘、归约等）。

---

## 1.2 激活函数：注入非线性

没有激活函数，无论堆叠多少层线性变换，等价的仍然是一个线性变换。激活函数让网络有能力拟合非线性关系。

**梯度和反向传播基础**

> 神经网络通过**梯度下降**优化参数。梯度是损失函数对每个参数的偏导数（`∂Loss/∂param`），告诉参数该往哪个方向调整。
>
> **反向传播**：从输出层往回，用链式法则逐层计算梯度。每一层的梯度 = 后一层的梯度 × 本层算子的导数。
>
> **为什么激活函数必须有非零梯度？** 如果某个算子的梯度恒为 0（如 ReLU 的负值区），反向传播到此为止，前面的参数无法更新。这就是"死亡 ReLU"问题的根源。

### 1.2.1 第一代：硬门控

#### ReLU（修正线性单元）

**公式**：`ReLU(x) = max(0, x)`

**输入/输出**：任意形状，输入输出同形 `[B,C,...] → [B,C,...]`

**梯度**：`x>0 时为1，x≤0 时为0`

```
    输出
     ^
     |    /
     |   /  斜率=1
     |  /
     | /
─────┼─────> 输入
     |
     |  斜率=0 (死亡区)
```

- ✅ 计算极快，只需一个 max 操作
- ❌ **死亡 ReLU**：负值梯度恒为 0，神经元永久失活
- 参数量：0
- 代表模型：ResNet, VGG, 大多数 CNN

#### LeakyReLU（带泄漏 ReLU）

**公式**：`LeakyReLU(x) = x > 0 ? x : α·x`（α 固定，通常 0.01）

**输入/输出**：任意形状，输入输出同形

**关键参数**：`alpha`（negative_slope），典型值 0.01~0.2

```
    输出
     ^    /
     |   /  斜率=1
     |  /
     | /  斜率=α (微小但不为零)
─────┼─────> 输入
```

- 解决死亡 ReLU：负值区域保留小梯度
- 当 α=0 时退化为 ReLU
- 代表模型：YOLO, Darknet, 部分 GAN

#### PReLU（参数化 ReLU）

**公式**：`PReLU(x) = x > 0 ? x : α·x`（α 可学习）

**关键参数**：`alpha`（可学习，初始 0.25），支持逐通道独立

- α 通过反向传播自动学习最优值
- 每通道仅增加 1 个参数，额外开销极小
- 代表模型：PReLU-ResNet, 人脸识别模型

#### ELU（指数线性单元）

**公式**：`ELU(x) = x > 0 ? x : α·(eˣ − 1)`

**关键参数**：`alpha`，典型值 1.0

- 输出均值接近 0，加速收敛
- 负值区域饱和到 -α，对噪声更鲁棒
- 计算包含 exp 运算，比 ReLU 稍慢
- 代表模型：部分生成模型, ELU-Net

#### SELU（缩放 ELU）

**公式**：`SELU(x) = scale · ELU(x)`（scale ≈ 1.0507, α ≈ 1.6733）

- **自归一化**：输出自动收敛到均值 0 方差 1，无需 BatchNorm
- 必须配合 Alpha Dropout 使用
- 权重初始化推荐 LeCun Normal：W ~ N(0, 1/fan_in)
- 代表模型：自归一化神经网络（SNN）

#### Sigmoid（S 型激活）

**公式**：`σ(x) = 1 / (1 + e⁻ˣ)`

**输出范围**：(0, 1)

**梯度**：`σ'(x) = σ(x) · (1 − σ(x))`，最大值为 0.25（x=0 处）

- 双重身份：输出层时是"概率映射"，中间层时是"软门控信号"
- 共同缺点：两端梯度趋近 0（梯度消失）
- 代表用途：二分类输出、LSTM/GRU 门控、注意力权重

#### Tanh（双曲正切）

**公式**：`tanh(x) = (eˣ − e⁻ˣ) / (eˣ + e⁻ˣ) = 2σ(2x) − 1`

**输出范围**：(−1, 1)，零中心化

- 比 Sigmoid 更适合隐藏层（零中心化）
- 代表用途：LSTM 隐藏状态、注意力分数缩放

#### GELU（高斯误差线性单元）

**公式**：`GELU(x) = x · Φ(x)`（Φ 是标准正态 CDF）

**近似**（推理常用）：`≈ 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))`

```
    输出
     ^       ___/
     |     _/
     |   _/     平滑过渡
     | _/
     |/       (负值不完全归零)
─────┼─────> 输入
```

- ReLU 的平滑近似：0 附近有非零梯度，负值不严格归零
- 与 Swish 行为高度相似
- 推理常用 tanh 近似加速
- 代表模型：**BERT, GPT-2/3/4, ViT**——所有原始 Transformer

#### Swish / SiLU（自门控激活）

**公式**：`Swish(x) = x · σ(x) = x / (1 + e⁻ˣ)`

- **核心洞察**：x 同时充当"信息"和"门控信号"——大幅值时 σ≈1 原样通过，接近 0 时 σ≈0.5 部分抑制
- 与 GELU 形状几乎一致，计算更简单
- 带参数版本：`Swish_β(x) = x · σ(βx)`，β→∞ 时趋近 ReLU
- 代表模型：**LLaMA, Qwen, Mistral**——所有现代 LLM

> 💡 **GELU vs Swish**：β=1 时 Swish ≈ GELU。BERT 系用 GELU，LLaMA 系用 Swish。

#### Mish

**公式**：`Mish(x) = x · tanh(softplus(x)) = x · tanh(ln(1 + eˣ))`

- 比 Swish 更平滑，负值区域保留更多信号
- 计算开销高：需要 exp、log、tanh 三个运算
- 代表模型：YOLOv4, EfficientNet 变体

#### HardSwish（硬 Swish）

**公式**：`HardSwish(x) = x · clip(x/6 + 0.5, 0, 1) = x · clip(x+3, 0, 6) / 6`

- 用 clip 替代 sigmoid/exp，移动端友好
- 在 |x| ≥ 3 时与 ReLU 行为一致
- 推理速度比 Swish 快数倍，精度损失 < 0.02
- 代表模型：**MobileNetV3**, EfficientDet

#### HardSigmoid（硬 Sigmoid）

**公式**：`HardSigmoid(x) = clip(x/6 + 0.5, 0, 1)`

- Sigmoid 的分段线性近似，3 次加减乘 + 1 次 clip
- 与 HardSwish 的关系：`HardSwish(x) = x · HardSigmoid(x)`
- 代表模型：MobileNetV3（SE 模块中替代 Sigmoid）

#### Softplus

**公式**：`Softplus(x) = ln(1 + eˣ)`

- ReLU 的平滑近似，处处可导
- 梯度恰好等于 Sigmoid
- 数值稳定：当 x > threshold 时直接返回 x
- 代表用途：Mish 的组成部分、变分推断

#### Softsign

**公式**：`Softsign(x) = x / (1 + |x|)`

- Tanh 的有理函数替代，梯度衰减更慢
- 无 exp 运算，计算比 Tanh 更快
- 代表模型：部分语音识别模型

#### BNLL

**公式**：`BNLL(x) = ln(1 + eˣ)`（与 Softplus 数学等价）

- 来自概率图模型中的二项对数似然损失
- 在 Caffe 框架中作为激活层出现

#### CELU（连续 ELU）

**公式**：`CELU(x) = x ≥ 0 ? x : α·(e^(x/α) − 1)`

- ELU 的连续可微版本：在 x=0 处梯度连续（ELU 在零点不连续）
- 梯度在零点恰好为 1
- 代表用途：需要梯度连续性的优化场景

#### Threshold（阈值激活）

**公式**：`Threshold(x) = x > θ ? x : v`（通常 v=0）

- 比 ReLU 更极端的硬截断
- 当 threshold=0, value=0 时退化为 ReLU
- 代表用途：稀疏编码网络

#### GLU（门控线性单元）

**公式**：`GLU(x) = a · σ(b)`（x 沿特征维度一分为二：a, b）

**输入/输出**：`[B, 2C, ...] → [B, C, ...]`（通道减半）

- 门控机制允许网络选择性传递信息
- 是 SwiGLU 的基础（将 Sigmoid 替换为 Swish）

#### Shrink（软阈值）

**公式**：`Shrink(x) = sign(x) · max(|x| − λ, 0)`

- 将绝对值小于阈值的输入置零
- 产生精确零值，天然具有稀疏性
- L1 正则化的近端算子
- 代表用途：稀疏自编码器、信号去噪

### 激活函数进化路线图

```
ReLU (2012)          → 解决梯度消失
  ↓
LeakyReLU / PReLU    → 解决死亡 ReLU
  ↓
ELU / SELU           → 输出零均值
  ↓
GELU / Swish (2017-) → 平滑门控，负值非零
  ↓
HardSwish (2019)     → 移动端优化
  ↓
SwiGLU (2022-)       → 显式门控，现代 LLM 标准
```

### 激活函数选择指南

| 场景 | 推荐激活 | 原因 |
|------|---------|------|
| CNN 分类网络 | ReLU / GELU | 简单有效 |
| Transformer NLP | GELU (BERT) / Swish (LLaMA) | 平滑门控 |
| LLM MLP | SwiGLU | 门控+平滑，当前最优 |
| 移动端部署 | HardSwish | 计算快 |
| GAN / 检测 | LeakyReLU | 负值保留梯度 |
| 门控机制 (LSTM/GRU) | Sigmoid | 输出(0,1)天然适合门控 |
| RNN 隐藏状态 | Tanh | 输出(-1,1)零中心化 |

#### Softmax：从 logits 到概率

**公式**：`softmax(xᵢ) = exp(xᵢ) / Σⱼ exp(xⱼ)`

**输出范围**：每个值 ∈ (0, 1)，所有值之和 = 1

**梯度**：`∂softmax(i)/∂xⱼ = softmax(i) · (δᵢⱼ − softmax(j))`

```python
# 标准实现 (数值稳定版)
x_max = max(x)                  # 减最大值防止溢出
exp_x = exp(x - x_max)
output = exp_x / sum(exp_x)
```

- 分类任务输出层的标准选择：将任意实数 logits 转为概率分布
- 注意力机制中：将 Q·Kᵀ 的相似度分数转为注意力权重
- 与 CrossEntropy Loss 结合时，梯度简化为 `softmax − label`（无需手动求导）
- **温度参数（Temperature）**：`softmax(x / T)`，T > 1 使分布更平滑（知识蒸馏），T < 1 使分布更尖锐

**与 Sigmoid 的关系**：

| 对比 | Sigmoid | Softmax |
|------|---------|---------|
| 适用范围 | 逐元素，独立 | 整个向量，联合 |
| 多类别 | 各类别独立（可能多个激活） | 概率和为1（互斥选择） |
| 用途 | 二分类/门控 | 多分类/注意力 |

---

## 1.3 归一化：稳定数值范围

训练深层网络时，每层的输出分布会不断漂移（Internal Covariate Shift），导致梯度不稳定。归一化将数值拉回稳定范围。

**Internal Covariate Shift 是什么？**

> 当网络前面的层更新参数后，后面层接收到的输入分布就变了。就像射击训练中靶子在不断移动，射手（后面层）需要不断适应新的靶子位置。这导致：
>
> - 训练需要更小的学习率
> - 需要更谨慎的参数初始化
> - 深层网络容易梯度消失/爆炸
>
> 归一化相当于"固定靶子位置"——每层的输入都被拉到均值 0 方差 1 附近，后面层可以稳定学习。

### 1.3.1 BatchNorm（批归一化）—— CNN 时代的标配

**公式**：
```
训练: μ = mean(x, dim=batch), σ² = var(x, dim=batch)
     y = (x − μ) / √(σ² + ε) · γ + β
推理: 用训练累积的 running_mean / running_var 替代 batch 统计
```

**输入/输出**：`[B, C, H, W] → [B, C, H, W]`（对每个 C 通道，在 B,H,W 维度统计）

**可学习参数**：`gamma` [C]（初始1.0）、`beta` [C]（初始0.0）

**运行统计**：`running_mean` [C]、`running_var` [C]、`momentum`（通常 0.1）

```
           N (batch)  C (channel)  H (height)  W (width)
BatchNorm: ─────────  ✗ 归一化     ✗ 归一化     ✗ 归一化     ← 沿 N,H,W
```

- 依赖 batch size，小 batch 效果差
- 有效的正则化效果（batch 统计引入随机噪声）
- **推理时可与前一层卷积融合**：融合后减少推理计算量
- 代表模型：ResNet, VGG, YOLO

### 1.3.2 LayerNorm（层归一化）—— Transformer 的标配

**公式**：`μ = mean(x, dim=feature), y = (x − μ) / √(σ² + ε) · γ + β`

**输入/输出**：`[B, C, H, W] → [B, C, H, W]`（每个样本独立，在 C,H,W 维度统计）

对于序列 `[B, L, D]`：对每个样本的每个位置，在 D 维度统计

**可学习参数**：`gamma` [D]、`beta` [D]

```
LayerNorm: ✗ 独立     ─────────    ─────────    ─────────    ← 沿 C,H,W
```

- 不依赖 batch size，训练推理行为一致
- 无需维护 running_mean/running_var
- 代表模型：BERT, GPT-2, T5, ViT

### 1.3.3 RMSNorm（均方根归一化）—— 现代 LLM 的标配

**公式**：`rms = √(mean(x²) + ε), y = x / rms · γ`（无均值减法，无 β）

**输入/输出**：同 LayerNorm

**可学习参数**：仅 `gamma` [D]（无 beta）

```
RMSNorm:   ✗ 独立     ─────────    ─────────    ─────────    ← 同 LayerNorm，简化版
```

- LayerNorm 简化版：去掉均值中心化和偏置项
- 计算更快（省减法和偏置加法），参数更少（只有 γ 没有 β）
- 28 层模型中累计省约 30% 归一化计算量和 50% 归一化参数量
- 代表模型：**LLaMA, Qwen, Mistral, Gemma**

### 1.3.4 GroupNorm（组归一化）—— 小 batch 的救星

**公式**：将 C 个通道分 G 组（每组 C/G 个），组内统计 μ 和 σ²

**输入/输出**：`[B, C, H, W] → [B, C, H, W]`

**关键参数**：`num_groups`（通常 32）、`eps`（通常 1e-5）

```
GroupNorm: ✗ 独立     分组内        ─────────    ─────────    ← 分组内 H,W
```

- 不依赖 batch size，训练推理一致
- G=1 等价于 LayerNorm，G=C 等价于 InstanceNorm
- 代表模型：**Stable Diffusion UNet**, 小 batch 检测模型

### 1.3.5 InstanceNorm（实例归一化）—— 风格迁移专用

**公式**：每个样本的每个通道独立归一化（GroupNorm 中 G=C 的特例）

**输入/输出**：`[B, C, H, W] → [B, C, H, W]`（对每个样本每个通道，在 H,W 维度统计）

```
InstanceN: ✗ 独立     每通道独立    ─────────    ─────────    ← 每通道 H,W
```

- 消除实例级对比度差异
- 通常 `affine=False`（无 gamma/beta）
- 空间维度较小时统计量不稳定（如 3×3 特征图）
- 代表模型：**StyleGAN, 风格迁移, CycleGAN**

### 1.3.6 SyncBatchNorm（同步批归一化）

**公式**：同 BatchNorm，但统计量跨所有 GPU 同步

- 分布式训练中跨 GPU 同步 μ 和 σ²，等价于大 batch 上的普通 BatchNorm
- 通信开销：每次前向传播 1 次 AllReduce（约 2C 个 float）
- 推理时退化为普通 BatchNorm

### 1.3.7 LRN（局部响应归一化）—— 已过时

**公式**：沿通道窗口做局部归一化，模拟生物侧抑制

**关键参数**：`size`（窗口大小）、`alpha`、`beta`、`k`（AlexNet: 5, 0.0001, 0.75, 2）

- 增加 1-2% 计算量，现代网络中效果微弱
- 已被 BatchNorm 完全取代
- 代表模型：AlexNet（已过时）

### 1.3.8 MVN（均值方差归一化）

**公式**：`y = (x − μ) / √(σ² + ε)`（无 gamma/beta 参数）

- 等价于不带 affine 的 LayerNorm 或 InstanceNorm
- 代表用途：特征预处理

### 1.3.9 LayerScale（层缩放）

**公式**：`y = x + γ ⊙ F(x)`（γ 是 [hidden_dim] 的可学习缩放向量，初始为小值）

**输入/输出**：逐通道缩放残差分支的输出 `[hidden_dim] → [hidden_dim]`

**可学习参数**：`gamma` [hidden_dim]，初始值为 `[1e-5, 1e-6, ...]`

- 在残差连接处对变换分支的输出做逐通道缩放
- 初始值设为极小值，让网络从"近似恒等映射"开始，逐步学习
- 稳定深层网络训练（200+ 层 ViT/ConvNeXt 训练关键技巧）
- 代表模型：**ConvNeXt, ViT-Deep, CAIT**

### 归一化选择指南

| 场景 | 推荐 | 原因 |
|------|------|------|
| CNN (大 batch) | BatchNorm | 批统计稳定 |
| Transformer (NLP) | LayerNorm / RMSNorm | 不依赖 batch |
| 现代 LLM | RMSNorm | 更快更省 |
| 扩散模型 / 小 batch | GroupNorm | 不依赖 batch |
| 风格迁移 | InstanceNorm | 逐样本逐通道 |

### 1.3.10 Dropout（随机丢弃）

**训练**：以概率 p 随机将神经元置零，其余值缩放 1/(1-p)

**推理**：不做任何操作

- 防止过拟合的正则化手段：每次随机丢弃相当于训练一个"子网络"集合
- LLM 推理时通常关闭
- Alpha Dropout：配合 SELU 使用，保持自归一化特性

---

## 1.4 张量操作：形状与维度的魔法

张量操作不涉及计算，只改变数据的组织方式——它们是连接各个算子的"管道"。

### 1.4.1 形状变换

#### Reshape（形状变换）

**功能**：改变张量形状但不改变数据。输入输出元素数必须一致。

**示例**：`[B, C, H, W] → [B, C·H·W]`

```python
x.reshape(B, -1)     # -1 自动推导
x.view(B, C*H*W)     # view 要求内存连续
```

> ⚠️ `reshape` vs `view`：`view` 要求张量内存连续，`reshape` 不要求（必要时会拷贝）。

#### Flatten（展平）

**功能**：将多维张量展平为一维（保留 batch 维度）

**示例**：`[B, C, H, W] → [B, C·H·W]`

- 使用场景：卷积特征图送入全连接层前的维度转换

#### Squeeze（压缩维度）

**功能**：删除大小为 1 的维度，ExpandDims 的逆操作

**示例**：`[B, C, 1, W] → [B, C, W]`

#### ExpandDims（扩展维度）

**功能**：在指定位置插入大小为 1 的维度，不复制数据

**示例**：`[B, C, H] → [B, C, H, 1]`

- 使用场景：GQA 中 KV head 扩展前的维度准备；广播运算前的维度对齐

### 1.4.2 维度操作

#### Permute（维度置换）

**功能**：重新排列维度顺序，数据在内存中可能需要重排

**示例**：`Permute(0,2,3,1): [B,C,H,W] → [B,H,W,C]`

```python
x.permute(0, 2, 3, 1)   # 通用维度置换
x.transpose(1, 2)        # 只交换两个维度
x.T                       # 2D 矩阵转置
```

#### Tile / Repeat（平铺重复）

**功能**：沿指定维度复制张量数据

**示例**：`[B, C, 1] --Tile(1,1,3)--> [B, C, 3]`

- 使用场景：GQA 中将 8 个 KV head 复制为 16 个

### 1.4.3 切分与拼接

#### Slice（切片）

**功能**：沿指定维度取子张量

**示例**：`x[:, 0:10]` 取前 10 个；`x[:, ::2]` 隔一个取一个

#### Split（分割）

**功能**：沿指定维度将张量分成多份

**示例**：`Split(dim=1, sizes=[2,4]): [B,6,H,W] → [B,2,H,W] + [B,4,H,W]`

- Concat 的逆操作
- 使用场景：Q/K/V 投影后分割 embedding；MLP 中 gate/up 分支

#### Concat（拼接）

**功能**：沿指定维度拼接多个张量，除拼接维度外其他维度必须一致

**示例**：`Concat(dim=1): [B,2,H,W] + [B,4,H,W] → [B,6,H,W]`

- 使用场景：残差连接 / 特征融合 / KV Cache 拼接 / 多分支合并

> 💡 **Split 和 Concat 互为逆操作**。在 Transformer Block 中，Split 将特征分为 Q/K/V 三路；Concat 将多个 head 的输出合并。

### 1.4.4 索引与条件

#### Gather（收集）

**功能**：按索引从指定维度取元素

**示例**：`Gather(axis=0): embedding[indices] → [N, hidden_dim]`

- Embedding 本质就是 Gather 操作
- 使用场景：Token Embedding 查找、Top-K 采样

#### Scatter（散布）

**功能**：Gather 的逆操作，将值写入指定位置

#### Where（条件选择）

**功能**：`condition ? x : y`，逐元素条件选择

```python
torch.where(mask > 0, x, torch.tensor(-1e38))   # 掩码应用
```

- 使用场景：注意力掩码应用、条件填充

#### ArgMax / ArgMin（取最大值/最小值索引）

**功能**：沿指定维度返回最大（小）值的索引

```python
idx = torch.argmax(x, dim=-1)    # [B, L, V] → [B, L]
```

- 贪心解码：`next_token = ArgMax(logits)`
- 分类任务：`predicted_class = ArgMax(logits, dim=-1)`

#### TopK（取前 K 个最大值及其索引）

**功能**：返回指定维度最大的 K 个值和对应索引

```python
values, indices = torch.topk(x, k=5, dim=-1)   # [B,L,V] → [B,L,5] + [B,L,5]
```

- Top-K 采样生成：只从前 K 个概率最高的 token 中采样
- MoE 路由：取 Top-K 个专家（通常 K=2）

### 1.4.5 排序与选择

#### OneHot（独热编码）

**功能**：将类别索引转为 one-hot 向量

```python
torch.nn.functional.one_hot(idx, num_classes=1000)
# idx=3, num_classes=5 → [0, 0, 0, 1, 0]
```

- 分类标签编码、BCE Loss 的输入格式
- CrossEntropy Loss 内部自动处理 one-hot 编码

### 1.4.6 累积操作

#### CumSum（累积求和）

**功能**：沿指定维度计算前缀和

```python
torch.cumsum(x, dim=0)    # [1, 2, 3] → [1, 3, 6]
```

- 并行扫描（Parallel Scan）的基础：SSM/RNN 的并行化训练
- 位置编码、序列长度掩码生成

### 1.4.7 空间操作

#### Padding（填充）

**功能**：在张量边界填充 0 或指定值

**示例**：`Padding(top=1,bottom=1,left=1,right=1): [B,C,H,W] → [B,C,H+2,W+2]`

- 使用场景：卷积 padding、序列对齐

#### Crop（裁剪）

**功能**：裁剪张量的某些维度

#### Flip（翻转）

**功能**：沿指定维度翻转张量元素顺序

- 使用场景：数据增强、双向 RNN 的反向处理

#### Clamp / Clip（截断）

**功能**：将值限制在 `[min, max]` 范围内

```
y = min(max(x, min_val), max_val)
```

- 使用场景：梯度裁剪、HardSwish 中的 clip

#### PixelShuffle（像素重组）

**功能**：将通道维度重排到空间维度

```
[B, C·r², H, W] → [B, C, H·r, W·r]
```

- 高效上采样，比转置卷积更少伪影
- 代表模型：超分辨率 (ESPCN, Real-ESRGAN), Stable Diffusion

#### Interp（插值缩放）

**功能**：双线性/最近邻/双三次插值，改变特征图的空间大小

```
[B, C, H, W] → [B, C, H*scale, W*scale]    scale=2: 放大2倍
```

- 上采样/下采样通用
- 比 TransposedConv 更轻量（无参数）

#### GridSample（网格采样）

**功能**：按照坐标网格从输入特征图采样

```
输入: 特征图 [B,C,H_in,W_in] + 网格 [B,H_out,W_out,2]
输出: [B,C,H_out,W_out]
```

- 支持双线性/最近邻/双三次插值
- 代表模型：空间变换网络 (STN), 光流估计, DETR

#### Reorg（YOLO 重组）

**功能**：将空间维度重排到通道维度（PixelShuffle 的逆操作）

```
[B, C, H, W] → [B, C·4, H/2, W/2]
```

- YOLOv2 的 passthrough 层，将细粒度特征传递到检测头

#### ShuffleChannel（通道混洗）

**功能**：将通道分组后交错排列

```
输入: [B, g·n, ...] → 按 g 分组 → 交错排列 → [B, g·n, ...]
```

- 分组卷积后恢复跨组信息流
- 代表模型：ShuffleNet

---

## 1.5 Loss 函数（损失函数）

> 损失函数衡量模型输出与真实标签的差距。反向传播时，梯度从 Loss 开始往回传播。

### 1.5.1 CrossEntropy（交叉熵损失）

```
CE = −Σ y_true · log(y_pred)
简化: 当 y_true 是 one-hot 时 → −log(y_pred[class])
```

- **多分类任务的标准选择**：配合 Softmax 输出层
- 与 Softmax 结合时梯度简化为 `y_pred − y_true`（形式简洁）
- 代表用途：图像分类、语言模型（预测下一个 token）

### 1.5.2 BCE（二元交叉熵）

```
BCE = −[y·log(p) + (1−y)·log(1−p)]
```

- 二分类任务：配合 Sigmoid 输出层
- 多标签分类：每个类别独立做 BCE（非互斥）
- 代表用途：目标检测的置信度分支、多标签分类

### 1.5.3 MSE（均方误差）

```
MSE = (1/n) · Σ(y_pred − y_true)²
```

- 回归任务的标准损失
- 对异常值敏感（平方放大误差）
- 代表用途：关键点回归、深度估计

### 1.5.4 Focal Loss

```
Focal = −α·(1−pₜ)ᵞ · log(pₜ)
         ↑             ↑
      类别权重      聚焦参数(通常γ=2)
```

- 在 BCE 基础上加入 (1−pₜ)ᵞ 因子：对容易样本降权，困难样本加权
- 解决**正负样本极度不平衡**问题
- 代表模型：RetinaNet

### 1.5.5 CTC Loss

```
CTC: 对输入输出长度不一致的序列建模
输入: [T, vocab]    输出: [L, vocab]   (T >> L)
```

- 处理变长输入输出对齐问题
- 引入空白符（blank），允许多对一映射
- 代表模型：CRNN 文字识别, 传统 ASR

---

## 1.6 硬件执行模型

理解算子如何在硬件上执行——这是优化的起点。

### 1.6.1 FLOPs vs MACs

**FLOPs**（Floating Point Operations）：浮点运算次数

**MACs**（Multiply-Accumulate）：乘加运算次数

```
一个 MAC = 1次乘法 + 1次加法 = 2 FLOPs

Conv2D FLOPs:  2 × C_out × C_in × k² × H_out × W_out
Conv2D MACs:      C_out × C_in × k² × H_out × W_out

MatMul FLOPs:  2 × M × N × K
MatMul MACs:      M × N × K
```

> 💡 **业界常见混淆**：论文中的"FLOPs"有时指 MACs。PyTorch 的 `thop` 和 `ptflops` 库报告的是 MACs×2。ncnn 的 `get_memory_footprint()` 也基于 MACs×2。

### 1.6.2 Roofline Model（屋顶线模型）

硬件的性能受两个瓶颈限制：算力和内存带宽。

```
                算力上限（GFLOPs/s）
                ┃
     计算密集区  ┃
        ╲       ┃
         ╲      ┃
          ╲     ┃
           ╲    ┃
            ╲   ┃
     ───────╲───┃──────  ← 内存带宽上限
               ╲ ┃
   内存密集区    ╲┃
                 ╲
              ────────→ 算子的计算强度 (FLOPs/Byte)
```

**计算强度（Arithmetic Intensity）** = FLOPs / 访问字节数

| 算子 | 计算强度 | 瓶颈 | 原因 |
|------|---------|------|------|
| MatMul (大矩阵) | 高 | 计算密集 | 每个数据复用多次 |
| Conv2D (3×3) | 中 | 视情况 | 权重复用但激活不 |
| DepthwiseConv | 低 | 内存密集 | 每个权重只用一次 |
| Add/ReLU/Clip | 极低 | 内存密集 | 每个数据只读写一次 |
| SDPA Decode | 极低 | 内存密集 | KV Cache 大，每 token 只算一次 |

**优化直觉**：

- 内存密集算子 → 优化数据布局、缓存命中率、内存带宽
- 计算密集算子 → 优化计算指令吞吐（SIMD 宽度、TensorCore 利用）

### 1.6.3 内存层级与带宽

```
CPU 寄存器     ~1 cycle,     ~100 GB/s      几 KB
L1 Cache       ~4 cycles,    ~100 GB/s      几十 KB
L2 Cache       ~12 cycles,   ~50 GB/s       几百 KB-几 MB
L3 Cache       ~30 cycles,   ~20 GB/s       几 MB-几十 MB
主存 (DDR)     ~100 cycles,  ~10-50 GB/s    几 GB-几十 GB
设备间 (PCIe)  ~μs,          ~1-16 GB/s     —
```

**ncnn 的内存优化策略**：

```
1. 算子内：利用 L1/L2 cache tiling（Conv 分块计算）
2. 算子间：blob 复用（计算完的中间结果立即释放）
3. 权重量化：INT8 权重使带宽需求减半
4. Vulkan：利用 GPU 的 shared memory（FlashAttention）
```

### 1.6.4 SIMD 并行

| 架构 | 指令集 | SIMD 宽度（FP32） | FP16 支持 |
|------|--------|-----------------|-----------|
| x86 | SSE | 4 | 无 |
| x86 | AVX2 | 8 | 无 |
| x86 | AVX-512 | 16 | FP16 转换 |
| ARM | NEON | 4 | FP16 (ARMv8.2+) |
| ARM | SVE | 可变 (128-2048 bit) | FP16 |
| RISC-V | V (Vector) | 可变 | FP16 |
| LoongArch | LSX | 4 | FP16 |
| GPU (CUDA) | TensorCore | 256+ (矩阵) | 原生 FP16/BF16 |

**ncnn 的实现**：每个算子有 CPU 基准版本 + 架构专属优化版本：

```
src/layer/convolution.cpp        ← 通用 C++ 基准
src/layer/arm/convolution_arm.cpp ← ARM NEON 优化
src/layer/x86/convolution_x86.cpp ← x86 AVX 优化
src/layer/vulkan/convolution_vulkan.cpp ← GPU 计算着色器
```

---

## 1.7 优化器（训练视角）

> 优化器是参数更新的规则。从算子角度看，它们是对梯度的变换。

### 1.7.1 SGD（随机梯度下降）

```
w ← w − lr · g          # g = ∂L/∂w
```

- 最简单：直接沿梯度反方向更新
- 容易震荡，需要较小的学习率

### 1.7.2 SGD with Momentum（动量）

```
vₜ = μ·vₜ₋₁ + g          # 累积历史梯度（指数移动平均）
w ← w − lr · vₜ
```

```
    梯度方向
      ↑
     ╱ ╲
    ╱   ╲  ← 有动量：累积方向一致的分量
   ╱     ╲   垂直方向的分量被抵消
  ───────────→ 参数空间
```

- μ 通常 0.9：相当于看最近 ~10 步的平均梯度
- 减少震荡，加速收敛

### 1.7.3 Adam（自适应矩估计）

```
mₜ = β₁·mₜ₋₁ + (1−β₁)·g     # 一阶矩（梯度均值）
vₜ = β₂·vₜ₋₁ + (1−β₂)·g²    # 二阶矩（梯度方差）
m̂ₜ = mₜ / (1−β₁ᵗ)           # 偏差修正（防止初期低估）
v̂ₜ = vₜ / (1−β₂ᵗ)
w ← w − lr · m̂ₜ / (√v̂ₜ + ε)
```

**默认参数**：β₁=0.9, β₂=0.999, ε=1e-8

- 自适应学习率：梯度大的参数更新步小，梯度小的更新步大
- 不需要手动调学习率（相对 SGD）

### 1.7.4 AdamW（解耦权重衰减）

```
# AdamW 的改进：将权重衰减从梯度中解耦
w ← w − lr · λ · w            # 权重衰减（独立于梯度）
w ← w − lr · m̂ₜ / (√v̂ₜ + ε)   # Adam 更新
```

- Adam 的 L2 正则化 ≠ 权重衰减（在自适应学习率下不等价）
- AdamW 修复了这个问题
- 现代 LLM 训练的标准选择

### 优化器对比

| 优化器 | 收敛速度 | 泛化性 | 调参难度 | 推荐场景 |
|--------|---------|--------|---------|---------|
| SGD | 慢 | 最好 | 高 | CNN 分类、精细调参 |
| SGD+Momentum | 中 | 好 | 中 | 传统视觉任务 |
| Adam | 快 | 中 | 低 | NLP、快速原型 |
| AdamW | 快 | 好 | 低 | **LLM 训练（推荐）** |

---

## 1.8 学习率调度

学习率决定了每次更新的步长。好的调度策略比优化器选择更重要。

### 1.8.1 Step Decay

```
lr(t) = lr₀ × γ^⌊t/step⌋
```

- 每隔固定步数乘以衰减因子 γ（通常 0.1）
- 简单但不平滑

### 1.8.2 Cosine Annealing

```
lr(t) = lr_min + ½(lr₀ − lr_min)·(1 + cos(π·t/T))
```

```
lr
 ^
 │   ╱╲
 │  ╱  ╲
 │ ╱    ╲      ← 余弦曲线
 │╱      ╲
 └──────────→ t
 0        T
```

- 从 lr₀ 平滑降到 lr_min
- 现代训练的标准选择

### 1.8.3 Warmup + Cosine

```
warmup 阶段:  lr(t) = lr₀ · (t / warmup_steps)
cosine 阶段:  lr(t) = lr_min + ½(lr₀ − lr_min)·(1 + cos(π·(t−warmup)/T_remaining))
```

```
lr
 ^
 │     ╱╲
 │    ╱  ╲
 │   ╱    ╲
 │  ╱      ╲
 │ ╱        ╲
 └───────────→ t
   warmup   训练结束
```

- 先用小学习率热身（防止初期梯度爆炸）
- 然后用余弦衰减精细收敛
- **LLM 训练标配**：warmup 通常 2000-5000 steps

### 学习率调度选择

| 场景 | 推荐 | 原因 |
|------|------|------|
| CNN 分类 | Step Decay / Cosine | 简单有效 |
| Transformer | Warmup + Cosine | 稳定训练 |
| LLM 预训练 | Warmup(5000 steps) + Cosine | 大规模标配 |
| 微调 | 线性衰减 | 小步精调 |

---

# 第二章 构建算子——从单元素到空间

> 第一章的算子处理逐元素或全连接关系。本章引入空间维度的处理——卷积捕获局部模式，注意力建立全局依赖。

## 2.1 线性变换：仿射世界的基石

### 2.1.1 全连接层（InnerProduct / Linear）

```
y = W·x + b
输入: [N, in_features]   输出: [N, out_features]
权重: [out_features, in_features]   偏置: [out_features]
```

- 最简单的参数化变换：每个输出是所有输入的加权和
- 参数量 = out_features × in_features + out_features
- 代表用途：分类头、Q/K/V 投影、MLP

### 2.1.2 Bias（偏置加法）

```
y = x + b    # b 是 [C] 的可学习偏置
```

> 💡 **Conv+BN 时 bias=False**：BN 的 β 参数已起到偏置作用，额外 bias 是冗余的。现代网络中紧跟 BN 的卷积几乎总是关闭 bias。

---

## 2.2 卷积：局部特征的捕获器

卷积是视觉模型的基石——通过共享参数的局部滤波器提取空间特征。

### 2.2.1 Conv2D 核心概念

```
输入: [B, C_in, H, W]  →  输出: [B, C_out, H', W']
权重: [C_out, C_in/groups, kH, kW]
```

**卷积 vs 全连接**：

```
全连接: 每个 output 看到 所有 input → 参数量 C_out × C_in × H × W
卷积:   每个 output 看到 局部 k×k   → 参数量 C_out × C_in × k × k

关键差异:
  1. 局部连接: 只看 k×k 邻域 (先验: 近邻更相关)
  2. 权重共享: 同一个滤波器滑过整个空间 (先验: 平移等变性)
  3. 参数量: 从 O(C²HW) 降到 O(C²k²)
```

**参数量 vs 计算量（FLOPs）**

> 参数量和 FLOPs 是两个不同的概念：
>
> - **参数量**：模型有多少可学习的权重数字。决定**内存占用**和**存储大小**。
> - **FLOPs**（Floating Point Operations）：推理时做了多少次浮点运算。决定**推理速度**。
>
> 一个模型可以参数量很小但 FLOPs 很大（如大卷积核小输出），也可以参数量很大但 FLOPs 很小（如全连接层 batch=1）。
>
> **估算公式**：Conv2D 的 FLOPs ≈ 2 × C_out × C_in × k² × H' × W'（乘法和加法各算一次）

**计算密集型 vs 内存密集型**

> 硬件执行算子时，瓶颈通常有两种：
>
> - **计算密集型（Compute-bound）**：算力是瓶颈，如大规模矩阵乘法。Prefill 阶段、大 batch 推理属于此类。GPU 利用率高。
> - **内存密集型（Memory-bound）**：内存带宽是瓶颈，如 Decode 阶段每步只算 1 个 token 但需加载全部权重和 KV Cache。CPU 缓存友好时反而更快。
>
> **直观判断**：计算量/数据量比值高 → 计算密集；比值低 → 内存密集。INT8 量化主要优化内存密集型场景（数据量减半）。

### 2.2.2 卷积参数完全解析

输出尺寸公式：

```
H_out = ⌊(H_in + 2×padding − dilation×(kernel−1) − 1) / stride + 1⌋
```

#### 参数①：kernel_size（卷积核大小）

```
k=1 (1×1):  只做通道变换，不看邻域
k=3 (3×3):  最常用，感受野 3×3
k=5 (5×5):  较大感受野，现多用两个 3×3 替代
k=7 (7×7):  早期网络 stem，现少见
```

> 💡 **为什么 3×3 最常用？** 两个 3×3 叠加的感受野 = 一个 5×5，但参数量 2×3²=18 < 5²=25，且多一次非线性。VGGNet 验证了这一设计。

#### 参数②：stride（步幅）

```
stride=1:  逐像素滑动，输出尺寸不变 (配合 pad=(k−1)/2)
stride=2:  每次跳过 1 像素，输出尺寸减半 (替代 MaxPool 下采样)
stride=patch_size:  极大下采样 (ViT PatchEmbed)
```

```
stride=1:  ┌───┐┌───┐┌───┐     stride=2:  ┌───┐  ┌───┐  ┌───┐
           │3×3││3×3││3×3│...              │3×3│  │3×3│  │3×3│...
           └───┘└───┘└───┘                 └───┘  └───┘  └───┘
           移动1像素                         移动2像素
```

| stride | 效果 | 替代方案 | 使用场景 |
|--------|------|---------|---------|
| 1 | 保持分辨率 | — | 常规特征提取 |
| 2 | 分辨率减半 | MaxPool2D | 可学习的下采样 (ResNet) |

#### 参数③：padding（填充）

```
padding=0 (valid):  不填充，输出缩小    k=3: H_out = H−2
padding=(k−1)/2 (same):  填充后输出不变  k=3,pad=1: H_out = H
```

```
padding=0:              padding=1:
· · · · ·               0 0 0 0 0 0 0
· 1 2 3 4 ·             0 0 1 2 3 4 0 0
· 5 6 7 8 ·             0 0 5 6 7 8 0 0
· 9 A B C ·             0 0 9 A B C 0 0
· · · · ·               0 0 0 0 0 0 0
输出: 2×2 (缩小)        输出: 4×4 (不变)
```

| 填充类型 | 填充值 | 使用场景 |
|---------|--------|---------|
| zeros | 0 | 绝大多数卷积 |
| reflect | 镜像反射 | 边缘保持 |
| replicate | 复制边界值 | 边缘保持 |
| circular | 循环填充 | 周期性信号 |

#### 参数④：dilation（空洞率）

在卷积核元素间插入空洞，扩大感受野但不增加参数。

```
dilation=1 (标准):    dilation=2 (空洞):
┌─┬─┬─┐              ┌─┬─┬─┬─┬─┐
│●│●│●│              │●│○│●│○│●│
├─┼─┼─┤              ├─┼─┼─┼─┼─┤
│●│●│●│              │○│○│○│○│○│
├─┼─┼─┤              ├─┼─┼─┼─┼─┤
│●│●│●│              │●│○│●│○│●│
└─┴─┴─┘              ├─┼─┼─┼─┼─┤
                      │○│○│○│○│○│
感受野: 3×3           ├─┼─┼─┼─┼─┤
                      │●│○│●│○│●│
                      └─┴─┴─┴─┴─┘
                      感受野: 5×5
●=有权重  ○=空洞(跳过)
```

```
有效核大小: k_eff = dilation × (k − 1) + 1
dilation=1, k=3: k_eff=3   感受野 3×3
dilation=2, k=3: k_eff=5   感受野 5×5  ← 参数量不变！
dilation=4, k=3: k_eff=9   感受野 9×9
```

代表模型：**DeepLab V2/V3**（ASPP 模块用 dilation=1,6,12,18 并行提取多尺度上下文）

#### 参数⑤：groups（分组数）

```
groups=1 (标准):       groups=2:            groups=C_in (深度可分离):
┌─────┐┌─────┐        ┌───┐┌───┐          ┌─┐┌─┐
│C_in ││C_out│        │C/2││C/2│          │1││1│
│全连接││     │        │   ││   │          │ ││ │
└─────┘└─────┘        └───┘└───┘          └─┘└─┘
参数: C_out·C_in·k²   参数: C_out·C_in/2·k²  参数: C_in·1·k²
```

| groups 值 | 名称 | 参数量比 | 代表模型 |
|-----------|------|---------|---------|
| 1 | 标准卷积 | 1× | ResNet |
| G (1<G<C) | 分组卷积 | 1/G | ResNeXt, ShuffleNet |
| C_in | 深度卷积 | 1/C_in | MobileNet |

#### 参数⑥：bias（偏置）

| 场景 | bias | 原因 |
|------|------|------|
| Conv 后接 BN | False | BN 的 β 已等效于偏置 |
| Conv 不接 BN | True | 需要偏置拟合 |

### 常见卷积配置速查表

| kernel | stride | pad | dilation | 输出尺寸 | 感受野 | 用途 |
|--------|--------|-----|----------|---------|--------|------|
| 3 | 1 | 1 | 1 | N | 3×3 | 标准特征提取 |
| 3 | 2 | 1 | 1 | N/2 | 3×3 | 下采样 |
| 1 | 1 | 0 | 1 | N | 1×1 | 通道变换 |
| 3 | 1 | 2 | 2 | N | 5×5 | 空洞卷积 (DeepLab) |
| 3 | 1 | 4 | 4 | N | 7×7 | 空洞卷积 (DeepLab) |
| 7 | 2 | 3 | 1 | N/2 | 7×7 | ResNet stem |

### 感受野计算

```
单层:  RF = dilation × (kernel − 1) + 1

多层累积:
  RF₁ = k₁
  RFₗ = RFₗ₋₁ + (kₗ − 1) × ∏ᵢ₌₁ˡ⁻¹ strideᵢ

示例: 3 个 stride=1 的 3×3 卷积
  RF₁ = 3,  RF₂ = 3+2×1 = 5,  RF₃ = 5+2×1 = 7
  等效于 1 个 7×7 卷积，但参数量 3×9=27 < 49
```

### 2.2.3 卷积变体

| 变体 | 核心思想 | 参数量 | 代表模型 |
|------|---------|--------|---------|
| **DepthwiseConv** | 每通道独立卷积 | C·k² | MobileNet |
| **PointwiseConv (1×1)** | 只做通道变换 | C_out·C_in | ResNet 瓶颈 |
| **SeparableConv** | Depthwise + Pointwise | C·k² + C_out·C_in | MobileNet |
| **DilatedConv** | 空洞扩大感受野 | 同标准卷积 | DeepLab |
| **TransposedConv** | 上采样（反卷积） | C_in·C_out·k² | U-Net, GAN |
| **DeformableConv** | 学习采样偏移量 | 额外 2K 偏移参数 | DCN, DETR |
| **Conv3D** | 时空气卷积 | C_out·C_in·k³ | 视频理解 |
| **Conv1D** | 序列卷积 | C_out·C_in·k | Mamba, WaveNet |

### 2.2.4 卷积算法对比

同一卷积运算，可以用不同算法实现——每种算法的适用场景不同。

#### im2col + GEMM（ncnn 默认策略）

```
Step 1: im2col — 将输入每个滑动窗口展开为一列
        输入 [B, C_in, H, W]  →  矩阵 [C_in·k², H_out·W_out]

Step 2: GEMM — 权重组 × 展开矩阵
        [C_out, C_in·k²] × [C_in·k², H_out·W_out] → [C_out, H_out·W_out]

Step 3: reshape → 输出 [B, C_out, H_out, W_out]
```

```
输入 5×5, kernel 3×3:
┌─────────────┐         ┌───┬───┬───┬───┬───┬───┬───┬───┬───┐
│ 1  2  3  4  5│         │ 1 │ 2 │ 3 │ 4 │ 6 │ 7 │ 8 │ . │ . │
│ 6  7  8  9 10 │         │ 6 │ 7 │ 8 │ 9 │11 │12 │13 │ . │ . │
│11 12 13 14 15 │  im2col │11 │12 │13 │14 │16 │17 │18 │ . │ . │
│16 17 18 19 20 │  ───→   │ . │ . │ . │ . │ . │ . │ . │ . │ . │
│21 22 23 24 25 │         │ . │ . │ . │ . │ . │ . │ . │ . │ . │
└─────────────┘         └───┴───┴───┴───┴───┴───┴───┴───┴───┘
  5×5 = 25 元素            9×9 = 81 元素 (重复存储)
```

- **优点**：复用高度优化的 GEMM kernel
- **缺点**：im2col 阶段内存膨胀（滑动窗口重叠区域重复存储）
- **适用**：通用卷积，k≥3

#### Winograd（小核专用加速）

```
核心思想: 用更少的乘法做同样的卷积
        3×3 卷积: Direct 需要 9 次乘法 → Winograd 只需要 4 次乘法

公式: Y = Aᵀ · [(G·W·Gᵀ) ⊙ (Bᵀ·x·B)] · A
      ↑ 变换域的点乘 ⊙ 替代空域的卷积

变换矩阵 A, B, G 是常数矩阵（预先计算）
```

- **适用**：k=3×3 的卷积（ResNet 中最常见）
- **速度提升**：比 im2col 快 1.5-2×（小核时）
- **缺点**：数值精度略有损失，大核时不适用
- **ncnn 实现**：`src/layer/arm/convolution_3x3_winograd_fp32.cpp`

#### FFT（大核卷积）

```
核心思想: 卷积定理 — 空域卷积 = 频域点乘
        y = IFFT(FFT(x) · FFT(W))
```

- **适用**：k≥7×7 的大核卷积
- **实际使用**：现代网络极少用大核，FFT 卷积很少出现

#### Direct（直接计算）

```
逐位置计算：每个输出位置独立做 k²·C_in 次乘加
```

- **适用**：DepthwiseConv、stride>1、dilation>1 等非标准情况
- **优点**：无额外内存开销
- **缺点**：不如 GEMM 高效

### 算法选择策略

| kernel | stride | dilation | 推荐算法 | ncnn 实现 |
|--------|--------|----------|---------|----------|
| 3×3 | 1 | 1 | Winograd | 3×3_winograd |
| 1×1 | 1 | 1 | Direct → GEMM | Conv1×1 特殊路径 |
| k×k (k≥3) | 任意 | 1 | im2col+GEMM | 通用 Convolution |
| 任意 | 任意 | >1 | Direct | dilation 路径 |
| groups=C | 任意 | 1 | Direct (Depthwise) | ConvolutionDepthwise |

---

## 2.3 池化：空间信息的压缩

池化用局部统计量替代局部区域，实现降采样和空间不变性。

| 池化类型 | 操作 | 输入 → 输出 | 代表用途 |
|---------|------|------------|---------|
| **MaxPool** | 取窗口最大值 | [B,C,H,W] → [B,C,H/2,W/2] | VGG, ResNet |
| **AvgPool** | 取窗口平均值 | [B,C,H,W] → [B,C,H/2,W/2] | 更平滑的下采样 |
| **GlobalAvgPool** | 全局平均 | [B,C,H,W] → [B,C,1,1] | SENet, ResNet 末端 |
| **AdaptiveAvgPool** | 指定输出大小 | 任意 → [B,C,target_H,target_W] | 分类头 |
| **SPP** | 多尺度池化+拼接 | [B,C,H,W] → [B,C·(1+4+16)] | YOLOv3 |
| **RoIAlign** | 双线性插值 RoI | 特征图+RoI → [K,C,7,7] | Mask R-CNN |

> 💡 **GlobalAvgPool 替代了全连接层**：早期 CNN（如 VGG）末端用 4096 维 FC 层，参数量巨大。ResNet 用 GAP 将 [B,C,7,7] 压缩为 [B,C,1,1] 再接 1×1 分类头，大幅减少参数。

---

## 2.3.1 内存布局（Memory Layout）

张量在内存中的排列方式直接影响硬件执行效率。

### NCHW vs NHWC

```
NCHW（Channel-Major，PyTorch/ncnn 默认）:
  数据按通道连续存储: [N0C0H0W0, N0C0H0W1, ..., N0C0H1W0, ..., N0C1...]

  内存: N N N N N N N N C C C C C C C C H H H H H H H H W W W W W W W W
        ╰─── 通道 0 全部 ───╯ ╰─── 通道 1 全部 ───╯

  优点: + BatchNorm 统计量连续（每个通道数据连续）
        + ncnn 的 Conv/Depthwise 对此优化最好
  缺点: - Per-pixel 操作需要跨步访问（cache miss）

NHWC（Spatial-Major，TensorFlow/TFLite 默认）:
  数据按空间位置连续: [N0H0W0C0, N0H0W0C1, ..., N0H0W0Cn, N0H0W1...]

  内存: N H W [C0, C1, C2, ..., Cn] N H W [C0, C1, C2, ..., Cn] ...
        ╰── 一个空间位置的所有通道 ───╯

  优点: + Per-pixel 操作友好（所有通道连续）
        + Mobile/嵌入式设备通常更快
  缺点: - BatchNorm 需要跨步统计
```

### ncnn 的 3D 内存布局

```
ncnn 的 Mat 采用 interleaved 布局:

对于 [C, H, W] 3D blob（去掉 batch 维度）:

  内存: [C0H0W0, C0H0W1, ..., C0H0Wn,   ← 通道0行0
         C0H1W0, C0H1W1, ..., C0H1Wn,   ← 通道0行1
         ...                             ← 通道0全部
         C1H0W0, C1H0W1, ..., C1H0Wn,   ← 通道1行0
         ...                             ← 通道1全部
         ...
         CnH0W0, ...]                    ← 通道n全部

  每个通道是一个连续的 "row-major 2D 平面"
  通道之间也是连续排列
```

### 硬件对布局的偏好

| 硬件 | 偏好布局 | 原因 |
|------|---------|------|
| x86 CPU (AVX) | NCHW | 通道连续，SIMD 按通道向量化 |
| ARM CPU (NEON) | NCHW 或 NHWC | 取决于算子实现 |
| Mobile GPU | NHWC | texture 采样按像素 |
| NVIDIA GPU (CUDA) | NCHW 或 NHWC | TensorCore 偏好 NCHW |
| Vulkan | NHWC | compute shader 按 tile 分组 |

**ncnn 的做法**：内部统一用 NCHW，Vulkan 路径自动转换。

---

## 2.4 注意力：全局依赖的建立

卷积只看局部邻域，注意力让每个位置直接"看到"所有位置。

### 2.4.1 SDPA（缩放点积注意力）

```
Attention(Q, K, V) = softmax(Q·Kᵀ / √dₖ) · V

Q: [batch, heads, seq_q, dₖ]
K: [batch, heads, seq_k, dₖ]
V: [batch, heads, seq_k, dᵥ]
Output: [batch, heads, seq_q, dᵥ]
```

**逐步拆解**：

```
Step 1: Q·Kᵀ          → 点积衡量 Q 与每个 K 的相似度
Step 2: / √dₖ         → 缩放防止数值过大（dₖ=128 时 scale≈0.088）
Step 3: + mask         → 因果掩码：未来位置设为 −∞
Step 4: softmax        → 归一化为概率分布（注意力权重）
Step 5: · V            → 加权求和得到输出
```

### 2.4.2 多头注意力（MHA / GQA / MQA）

```
MHA (原始): 16个Q头 × 16个KV头    每个Q头有独立KV    KV Cache: 大
GQA (主流): 16个Q头 × 8个KV头    每2个Q头共享KV     KV Cache: -50%
MQA (极端): 16个Q头 × 1个KV头    所有Q头共享1个KV   KV Cache: -93.75%
```

**MHA（Multi-Head Attention）**：Q、K、V 分别投影到 num_heads 组，每组独立做 SDPA，Concat 所有 head 的输出，通过输出投影层。代表模型：BERT, GPT-2, 原始 Transformer

**GQA（Grouped-Query Attention）**：KV head 通过 ExpandDims + Tile 复制扩展到 Q head 数量

> **为什么是复制而不是学习？** GQA 在推理时直接将 8 个 KV head 复制为 16 个（每 2 个 Q head 共享 1 个 KV head）。这看似"浪费"——如果 KV head 可以学习，质量会不会更好？
>
> 实验表明：复制和学习的注意力质量差距极小（<0.5%），但复制省下了 50% 的 KV Cache 内存。在长序列推理中，KV Cache 是内存瓶颈，省内存比微小的质量提升更重要。，`num_heads_per_group = num_q_heads / num_kv_heads`。代表模型：LLaMA-2, Qwen2/3, Mistral

**MQA（Multi-Query Attention）**：所有 Q head 共享同一组 KV。代表模型：PaLM, StarCoder

### 注意力类型对比

| 类型 | Q heads | KV heads | KV Cache | 代表模型 |
|------|---------|----------|---------|---------|
| MHA | 16 | 16 | 标准 | BERT, GPT-2 |
| GQA | 16 | 8 | -50% | LLaMA-2, Qwen2/3 |
| MQA | 16 | 1 | -93.75% | PaLM |

### 2.4.3 RoPE（旋转位置编码）

```
对 Q/K 的每对相邻维度旋转:
(x₂ᵢ, x₂ᵢ₊₁) → (x₂ᵢcosθ − x₂ᵢ₊₁sinθ, x₂ᵢsinθ + x₂ᵢ₊₁cosθ)
θ = pos / base^(2i/d)

实现: inv_freq[k] = 1 / theta^(2k/d)
     cos[i][k] = cos(pos_i * inv_freq[k])
     sin[i][k] = sin(pos_i * inv_freq[k])
```

- 旋转角度差天然编码相对位置，低维度旋转快，高维度旋转慢
- 支持多种变体：NTK-Aware、YaRN、LongRoPE（详见推理 README Phase 3）
- 代表模型：LLaMA, Qwen, Mistral, Falcon

### 2.4.4 FlashAttention

```
传统: Q·Kᵀ → [seq,seq] 矩阵 → softmax → ·V    内存 O(n²)
Flash: 分块处理 Q·Kᵀ → online softmax → ·V    内存 O(n)

分块参数: M=4, N=32, K=32
Online Softmax: smem_row_max, smem_row_sum, smem_correction
```

- 将 QK+softmax+QKV 融合为单个 GPU kernel，使用 shared memory 分块累积
- 当新块产生更大最大值时，用 exp(old_max − new_max) 校正旧结果
- S=1000 序列：传统 4MB → Flash 512B
- 代表模型：所有现代 LLM 训练/推理框架

---

## 2.5 循环：时序记忆的载体

### LSTM（长短期记忆）

```
遗忘门: fₜ = σ(Wf·[hₜ₋₁, xₜ] + bf)    # 丢弃多少旧记忆
输入门: iₜ = σ(Wi·[hₜ₋₁, xₜ] + bi)    # 接收多少新信息
候选值: c̃ₜ = tanh(Wc·[hₜ₋₁, xₜ] + bc)  # 新信息内容
细胞态: cₜ = fₜ⊙cₜ₋₁ + iₜ⊙c̃ₜ          # 更新长期记忆
输出门: oₜ = σ(Wo·[hₜ₋₁, xₜ] + bo)    # 输出多少
隐藏态: hₜ = oₜ⊙tanh(cₜ)               # 输出
```

### GRU（门控循环单元）

```
重置门: rₜ = σ(Wr·[hₜ₋₁, xₜ])
更新门: zₜ = σ(Wz·[hₜ₋₁, xₜ])
候选值: h̃ₜ = tanh(Wh·[rₜ⊙hₜ₋₁, xₜ])
输出:   hₜ = (1−zₜ)⊙hₜ₋₁ + zₜ⊙h̃ₜ
```

> 💡 **LSTM vs GRU**：GRU 合并了遗忘门和输入门为更新门，合并了细胞状态和隐藏状态。参数量约为 LSTM 的 2/3，训练更快但表达能力略弱。

---

## 2.6 嵌入：离散到连续的桥梁

### Embed（嵌入查找）

```
token_id → embedding_matrix[token_id] → [hidden_dim]
本质: Gather 操作，从 [vocab_size, hidden_dim] 的矩阵中取一行
```

- 所有 NLP/LLM 模型的第一步：将离散的 token ID 映射为连续向量

### BPE Tokenization（详见推理 README Phase 2）

```
文本 → 预分词 → 字节级编码 → BPE 合并 → Token IDs
```

### Position Embedding（位置编码）

Transformer 的注意力机制本身不感知位置——需要额外注入位置信息。

**绝对位置编码（可学习 / 正弦）**

```
可学习:  Embed([seq_len, hidden_dim])         每位置一个可学习向量
正弦:    PE(pos, 2i)   = sin(pos / 10000^(2i/d))
         PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

- 原始 Transformer 用正弦，BERT 用可学习
- 缺点：推理时无法外推到比训练更长的序列

**RoPE（旋转位置编码，见 2.4.3）**

- 通过旋转向量来编码位置，支持外推，现代 LLM 标配

**ALiBi（见 4.1）**

- 在注意力分数上直接加偏置，无需位置编码

---

# 第三章 组合结构——从算子到架构块

> 单个算子能力有限，组合起来才构成强大的模型。

## 3.1 CNN 基础块

### 3.1.1 Conv-BN-ReLU (CBR)

```
x_in → Conv2D → BatchNorm → ReLU → x_out
```

- 几乎所有 CNN 的基本构建单元
- 变体：CBR-Swish, CBR-LeakyReLU, CBR-HardSwish

### 3.1.2 ResNet 残差块

```
x_in ────────────────────────────┐
  |                               |
  v Conv-BN-ReLU (C→C, k=3)     |
  v Conv-BN (C→C, k=3)          |
  |                               |
  v Add + ReLU ←─────────────────┘

BasicBlock (ResNet-18/34): 2 层 3×3
Bottleneck (ResNet-50+):   1×1 → 3×3 → 1×1 (先降维再升维)
```

- **残差连接的数学**

> 标准层：`x_out = F(x_in)`，梯度 `dL/dx_in = dL/dx_out × dF/dx_in`
> 残差层：`x_out = F(x_in) + x_in`，梯度 `dL/dx_in = dL/dx_out × (dF/dx_in + 1)`
>
> 关键差异在于那个 **+1**：即使 `dF/dx_in` 很小（甚至接近 0），梯度至少为 `dL/dx_out × 1`，不会消失。这使 100+ 层网络可训练。
- Pre-activation 变体：BN-ReLU-Conv 顺序（更稳定）

### 3.1.3 Inception 模块

```
x_in
  ├── Conv1×1 ──────────────────────┐
  ├── Conv1×1 → Conv3×3 ───────────┤
  ├── Conv1×1 → Conv5×5 ───────────┼── Concat → x_out
  └── MaxPool3×3 → Conv1×1 ────────┘
```

- 多尺度并行提取特征，无需手动选择核大小

### 3.1.4 SE Block（通道注意力）

```
x_in → GlobalAvgPool → FC-ReLU → FC-Sigmoid → Scale → x_out
         [B,C,H,W]→[B,C]       压缩C/r    恢复C    逐通道乘
```

- 自适应重校准通道权重，让网络关注重要通道
- 代表模型：SE-ResNet, EfficientNet

---

## 3.2 RNN 单元

### LSTM Cell（见 2.5 节完整公式）

4 个门控协同工作：遗忘门决定遗忘什么，输入门决定接收什么，输出门决定输出什么。细胞状态是"长期记忆"，隐藏状态是"短期输出"。

---

## 3.3 Transformer Block

### 现代 LLM 版本（Pre-Norm + RMSNorm + SwiGLU + GQA）

```
x_in [seq_len, hidden_dim]
  │
  ├─ RMSNorm ──→ Q/K/V 投影 ──→ QK-Norm ──→ RoPE
  │                                │
  │                    GQA Expand (KV头复制)
  │                                │
  │                    SDPA (scale=1/√dₖ, KV Cache)
  │                                │
  │                    输出投影 (Gemm)
  │                                │
  ├─── Add (残差) ←───────────────┘
  │
  ├─ RMSNorm ──→ Split(gate, up)
  │                │         │
  │           Swish(gate)   up
  │                │         │
  │              Mul → Down (Gemm)
  │                │
  ├─── Add (残差) ←┘
  │
  x_out [seq_len, hidden_dim]
```

### Transformer 进化路线

```
**Pre-Norm vs Post-Norm 的区别**

> Pre-Norm 先归一化再做变换（`y = x + F(Norm(x))`），Post-Norm 先变换再归一化（`y = Norm(x + F(x))`）。
>
> - **Pre-Norm**：残差路径上没有任何非线性变换，梯度可以无损流过，训练极稳定。但输出分布可能不如 Post-Norm 好。
> - **Post-Norm**：输出归一化后分布更稳定，但深层时训练困难，需要 warm-up。
>
> 现代 LLM 几乎全用 Pre-Norm，并在最后加一层最终归一化弥补输出分布问题。
  ↓
BERT/GPT-2 (2018-9):    Post-LN + MHA + FFN(GELU)
  ↓
GPT-3 (2020):           Post-LN + MHA + FFN(GELU) + Sparse Attention
  ↓
PaLM (2022):            Pre-LN + MQA + SwiGLU
  ↓
LLaMA (2023):           Pre-Norm(RMSNorm) + GQA + SwiGLU + RoPE
  ↓
Qwen3 (2024):           Pre-Norm(RMSNorm) + GQA + SwiGLU + RoPE + QK-Norm
```

---

## 3.4 编解码架构

### U-Net

```
编码器 (下采样)           解码器 (上采样)
Conv-BN-ReLU ×2           Conv-BN-ReLU ×2
MaxPool2D ────────────→ TransposedConv
Conv-BN-ReLU ×2           Conv-BN-ReLU ×2
MaxPool2D ────────────→ TransposedConv
       │                         ↑
       └──── Skip (Concat) ─────┘
```

- 编码器提取语义，解码器恢复细节，skip connection 保留精确位置
- 代表模型：医学分割, Stable Diffusion 去噪网络

### FPN（特征金字塔）

```
P5 ──→ 1×1 Conv ──→ Upsample ──┐
                                │ Add
P4 ──→ 1×1 Conv ───────────────┤──→ P4'
                                │
P3 ──→ 1×1 Conv ───────────────┘──→ P3'
```

- 多尺度特征融合，不同层级检测不同大小目标
- 代表模型：Faster R-CNN, Mask R-CNN

---

## 3.5 检测头与后处理

### IoU（交并比）

```
IoU = Area(预测框 ∩ 真实框) / Area(预测框 ∪ 真实框)
    = |A ∩ B| / |A ∪ B|    值域 [0, 1]

变体:
  GIoU  = IoU − (外接框 − 并集) / 外接框     # 处理不相交情况
  DIoU  = IoU − 中心点距离² / 对角线距离²    # 更快收敛
  CIoU  = IoU − DIoU − 宽高比一致性惩罚       # 最完整
```

- NMS 的核心度量，也是边界框损失的组成单元
- GIoU/DIoU/CIoU 逐步增加几何约束，提升回归精度

### Detection Head（检测头）

**Anchor-based（Faster R-CNN / SSD）**

```
特征图 [B, C, H, W]
  → Conv: 每个位置生成 K 个 anchor 的 box 偏移 + 置信度
  → 输出: [B, K×4, H, W] + [B, K×num_classes, H, W]
```

- 预定义不同比例/大小的 anchor box，回归偏移量

**Anchor-Free（FCOS / YOLOX / CenterNet）**

```
特征图 [B, C, H, W]
  → Conv: 每个位置直接预测 box 中心偏移 + 宽高 + 置信度
  → 输出: [B, 4, H, W] + [B, num_classes, H, W]
```

- 无 anchor 先验，直接回归目标框，更简洁
- YOLOv8+ 默认采用

### PriorBox（SSD 专用）

```
在每个 (i, j) 位置生成不同比例/大小的默认框
输出: [B, num_priors×4, H, W]
```

- 预生成 anchor box 坐标，后续接 DetectionOutput 层解码

### NMS（非极大值抑制）

```
1. 按置信度从高到低排序所有候选框
2. 取最高分框作为保留框
3. 计算该框与所有剩余框的 IoU
4. 删除 IoU > 阈值（通常 0.5）的框
5. 重复步骤 2-4 直到无剩余框
```

**变体**：

| 变体 | 核心思想 | 使用场景 |
|------|---------|---------|
| **Soft-NMS** | 不删除而是降分：score *= (1 − IoU) | 密集目标 |
| **DIoU-NMS** | 用 DIoU 替代 IoU 作为抑制度量 | 更精确的位置筛选 |
| **Class-Agnostic** | 跨类别统一抑制 | 通用检测 |
| **Class-Specific** | 按类别独立抑制 | 多类别检测 |

### CTC Decode（CTC 后处理）

```
1. 对每步输出取 ArgMax 得到 token 序列
2. 合并连续重复 token
3. 删除空白符（blank）
```

- 语音识别经典后处理，处理变长输入输出对齐
- 代表模型：DeepSpeech, 传统 ASR

---

## 3.6 ResNet 完整拆解

### 基本块组成

| 模型 | 层数 | BasicBlock | Bottleneck | 参数量 | ImageNet Top-1 |
|------|------|-----------|-----------|--------|---------------|
| ResNet-18 | 18 | 4×[2层] | — | 11.7M | 69.8% |
| ResNet-34 | 34 | 6×[2层] | — | 21.8M | 73.3% |
| ResNet-50 | 50 | — | 3+4+6+3×[3层] | 25.6M | 76.1% |
| ResNet-101 | 101 | — | 3+4+23+3×[3层] | 44.5M | 77.4% |

### ResNet-50 逐阶段拆解

```
Input: [B, 3, 224, 224]

Stage 0 (Stem):
  Conv 7×7, stride=2, 64 → [B,64,112,112]    FLOPs: 3.7G
  MaxPool 3×3, stride=2   → [B,64,56,56]      FLOPs: 0.02G

Stage 1 (conv2):  3× Bottleneck(64→256)
  1×1(64→64) → 3×3(64→64) → 1×1(64→256) + Add
  输出: [B,256,56,56]                         FLOPs: 3.9G

Stage 2 (conv3):  4× Bottleneck(256→512)
  第一个 block stride=2: 1×1(256→128) → 3×3(128→128) → 1×1(128→512)
  输出: [B,512,28,28]                         FLOPs: 3.9G

Stage 3 (conv4):  6× Bottleneck(512→1024)
  第一个 block stride=2
  输出: [B,1024,14,14]                        FLOPs: 3.9G

Stage 4 (conv5):  3× Bottleneck(1024→2048)
  第一个 block stride=2
  输出: [B,2048,7,7]                          FLOPs: 3.9G

Head:
  GlobalAvgPool    → [B,2048,1,1]
  FC(2048→1000)    → [B,1000]                  FLOPs: 2.0M

Total FLOPs: ~4.1G (ImageNet 224×224)
```

### 为什么 Bottleneck 是 1×1→3×3→1×1？

```
直接 3×3:    C_in × C_out × 3² = 256 × 256 × 9  = 589,824
Bottleneck:  C×C/4×1 + C/4×C/4×9 + C/4×C×1
           = 256×64×1 + 64×64×9 + 256×64×1     = 69,632
           = 8.5× 减少！
```

- 1×1 先降维（256→64），3×3 在低维做卷积，1×1 再升维（64→256）
- 保持表达能力同时大幅减少参数和计算

---

## 3.7 ViT 完整拆解

### ViT-B/16 架构（BERT-base 等价参数量）

```
Input: 图像 [1, 3, 224, 224]

Patch Embedding:
  Conv2D(k=16, stride=16, dim=768)
  [1,3,224,224] → [1,768,14,14] → reshape → [1,196,768]
  参数量: 3×16²×768 = 590K

Position Embedding:
  可学习: [1, 197, 768]  ← 196 patches + 1 CLS token
  参数量: 197×768 = 151K

Transformer Encoder (12 层):
  每层:
    Pre-LayerNorm → MultiHeadAttention(12 heads, d_k=64)
    + Add (残差)
    Pre-LayerNorm → MLP(768→3072→768, GELU)
    + Add (残差)

  每层参数量:
    MHA: 4 × (768×768) + 768 = 2.36M    ← Q,K,V,O 投影
    MLP: 768×3072 + 3072×768 = 4.72M    ← 两层 FC
    Norm: 2 × 768 × 2 = 3.07K           ← γ, β
    合计: ~7.08M × 12 = 85.0M

Classifier Head:
  LayerNorm → FC(768→1000)              ← CLS token 输出
  参数量: 768×1000 = 768K

Total: ~86.6M 参数    FLOPs: ~17G (224×224)
```

### ViT 关键设计

| 设计 | 说明 | 影响 |
|------|------|------|
| Patch Size=16 | 16×16 像素一个 token | 序列长度 196，计算量适中 |
| CLS Token | 特殊 [CLS] token 聚合全局信息 | 替代 GAP |
| Position Embed | 可学习 1D 位置编码 | 限制外推能力（不如 RoPE） |
| Pre-LN | LayerNorm 在残差内部 | 稳定训练（Post-LN 深层发散） |
| MLP 扩宽比=4 | 隐藏维度 3072 = 768×4 | 标准 Transformer 设置 |

### ViT vs CNN 的 FLOPs 对比

| 模型 | 参数量 | FLOPs | Top-1 |
|------|--------|-------|-------|
| ResNet-50 | 25.6M | 4.1G | 76.1% |
| ViT-B/16 | 86.6M | 17G | 77.9% |
| Swin-B | 88M | 15G | 83.5% |

> ViT 在大数据集上超越 CNN，但小数据不如 CNN（缺少归纳偏置）。Swin 通过滑动窗口恢复了局部归纳偏置。

---

## 3.8 LLaMA/Qwen 完整拆解

### LLaMA-7B 架构

```
Input: token_ids [1, seq_len]

Embedding:
  Gather from [32000, 4096]     ← vocab_size × hidden_dim
  参数量: 32000 × 4096 = 131M   ← 占总参数 ~16%

Transformer Block (32 层，每层相同):
  输入: [seq_len, 4096]

  Attention:
    RMSNorm(4096)
    Q: Linear(4096→4096, 32 heads)      4096² = 16.8M
    K: Linear(4096→4096, 32 heads)      4096² = 16.8M
    V: Linear(4096→4096, 32 heads)      4096² = 16.8M
    RoPE: 旋转向量（无参数）
    SDPA: scale=1/√128, causal mask     O(n²·d)
    O: Linear(4096→4096)               4096² = 16.8M

  SwiGLU MLP:
    RMSNorm(4096)
    gate: Linear(4096→11008)            4096×11008 = 45.1M
    up:   Linear(4096→11008)            4096×11008 = 45.1M
    Swish(gate) × up
    down: Linear(11008→4096)            11008×4096 = 45.1M

  每层参数量:
    Attention: 4 × 16.8M = 67.1M
    MLP: 3 × 45.1M = 135.3M
    Norm: 2 × 4096 × 1 = 8.2K
    合计: ~202.4M × 32 = 6.48B

Final Norm:
  RMSNorm(4096)

LM Head:
  Linear(4096→32000)                    ← 与 Embedding Weight Tying
  参数量: 4096 × 32000 = 131M           ← 不 Tying 时

Total (with tying): ~6.74B
```

### Qwen3-0.6B 架构

```
Embedding: [151936, 1024]              ← 594M (FP32)

Transformer Block (28 层):
  hidden_dim: 1024
  num_q_heads: 16
  num_kv_heads: 8                       ← GQA
  head_dim: 128
  intermediate_dim: 3072                ← SwiGLU
  scale: 0.0884 = 1/√128
  eps: 1e-6

  Attention (GQA):
    RMSNorm(1024)
    Q: Linear(1024→2048)               1024×2048 = 2.1M
    K: Linear(1024→1024)               1024×1024 = 1.0M   ← 8 KV heads
    V: Linear(1024→1024)               1024×1024 = 1.0M
    QK-Norm: RMSNorm(128, per-head)    16×128 = 2K (Q) + 8×128 = 1K (K)
    RoPE + GQA Expand(8→16)
    SDPA (KV Cache)
    O: Linear(2048→1024)               2048×1024 = 2.1M

  SwiGLU MLP:
    RMSNorm(1024)
    gate: Linear(1024→3072)            3.1M
    up:   Linear(1024→3072)            3.1M
    down: Linear(3072→1024)            3.1M

  每层: ~15.4M × 28 = 431M

Projection:
  Linear(1024→151936)                  ← Weight Tying → 复用 Embedding

Total: ~594M + 431M = ~1.0B (含 Embedding)
       Decoder-only: ~841M (FP16)
```

### KV Cache 大小计算

```
每层 KV Cache:
  K: [batch, seq_len, kv_heads, head_dim] = 1 × n × 8 × 128
  V: 同上
  每层: 2 × n × 8 × 128 × 4 bytes (FP32) = 8192 × n bytes

32 层总 KV Cache:
  FP32: 32 × 8192 × n = 256 KB × n
  FP16: 128 KB × n
  INT8: 64 KB × n

示例: seq_len=2048
  FP32: 512 MB
  FP16: 256 MB
  INT8: 128 MB
```

### LLaMA vs Qwen3 架构对比

| 特性 | LLaMA-7B | Qwen3-0.6B |
|------|---------|-----------|
| hidden_dim | 4096 | 1024 |
| layers | 32 | 28 |
| heads | 32 MHA | 16Q / 8KV (GQA) |
| head_dim | 128 | 128 |
| MLP | SwiGLU (11008) | SwiGLU (3072) |
| Norm | RMSNorm | RMSNorm |
| Position | RoPE | RoPE |
| QK-Norm | ✗ | ✓ |
| 总参数量 | 7B | 0.6B |

---

# 第四章 进化架构——前沿模型的算子组合

> 第三章的经典架构仍在进化。本章展示最新的算子组合方式。

## 4.1 注意力变体

### GQA Block（Qwen3 风格）

Qwen3 在标准 Transformer Block 基础上增加了 **QK-Norm** 和 **GQA** 两个关键设计。

```
Q: Gemm → Reshape(128,16,-1) → RMSNorm(128) → RoPE
K: Gemm → Reshape(128,8,-1)  → RMSNorm(128) → RoPE → ExpandDims+Tile(8→16)
V: Gemm → Reshape(128,8,-1)  → ExpandDims+Tile(8→16)
→ SDPA → 输出投影
```

- **QK-Norm**：防止长序列注意力熵坍缩
- **GQA**：KV Cache 节省 50%

### QK-Norm

```
Q → RMSNorm → RoPE     # 每头独立归一化，head_dim=128
K → RMSNorm → RoPE     # 同上
```

- 对 Q/K 向量做逐头归一化，防止长序列注意力熵坍缩
- 配合 RoPE 使用，使注意力分布不随序列长度增长而极端化
- 代表模型：**Qwen3**, Gemini

### FlashAttention

> 原理详见 [2.4.4](#244-flashattention)。4.1 节补充部署要点：

- **FlashAttention-2**：优化 kernel launch 和 work partitioning，比 v1 快 2×
- **FlashAttention-3**：利用 Hopper 的 FP8 + TMA 异步拷贝，进一步加速
- 部署要点：需定制 CUDA kernel，ncnn 等通用框架通常回退到标准 SDPA

### Sliding Window Attention（滑动窗口注意力）

```
标准注意力: 每个 token 关注 所有 token         复杂度 O(n²)
滑动窗口:   每个 token 只关注 前后 W 个邻居   复杂度 O(n·W)

示例 (W=4):
token 5:  关注 [1,2,3,4,5,6,7,8,9]  ← 全部
          ↓
token 5:  关注 [3,4,5,6,7]          ← 只关注窗口内
```

- 限制每个 token 只关注局部窗口，大幅降低内存
- **全局 token**：通常保留 [CLS] 或每 N 个 token 设一个全局头
- 代表模型：**Mistral**（4096 窗口），**Longformer**，**LongChat**

### ALiBi（Attention with Linear Biases）

```
标准 RoPE:    Attention = softmax(Q·Kᵀ / √d + RoPE)
ALiBi:       Attention = softmax(Q·Kᵀ / √d + m·(i − j))
                                       ↑
                                    斜率 × 相对距离
```

- 不用位置编码，而是在注意力分数上直接加与相对距离成比例的偏置
- 不同 head 用不同斜率 m（head 越近斜率越大）
- **关键优势**：训练时最大序列长度 = 512，推理可外推到任意长度
- 代表模型：**MPT**（MosaicML），部分 LongFormer 变体

### Cross-Attention（交叉注意力）

```
Decoder Q: 来自解码器隐藏态 [B, heads, target_len, dₖ]
Encoder K/V: 来自编码器输出   [B, heads, source_len, dₖ]
Output: softmax(Q·Kᵀ / √d) · V    → [B, heads, target_len, dᵥ]
```

- Q 来自解码器，K/V 来自编码器——让解码器"看"编码器输出
- 编码器-解码器架构（Transformer、T5、Whisper）的核心
- 与 Self-Attention 的区别：Self-Attention 的 Q/K/V 同源

### Linear Attention

```
标准: softmax(Q·Kᵀ)·V          O(n²·d)
线性: φ(Q)·(φ(K)ᵀ·V)           O(n·d²)
      先算 Kᵀ·V (d×d)，再乘 Q (n×d)

代价: 去掉 softmax 后注意力质量下降
代表: Performer, RWKV
```

---

## 4.2 状态空间模型

### Mamba Block

```
x_in → Linear → Split(x₁, x₂)
  x₁ → CausalConv1D → SiLU → SelectiveSSM(Δ, B, C)
  x₂ → SiLU (门控)
  → Mul → Linear → x_out

SSM 核心方程:
  hₜ = Ā·hₜ₋₁ + B̄·xₜ       # 状态更新
  yₜ = C·hₜ                  # 输出

选择性: Δ, B, C 由输入 x 动态生成 (而非固定参数)
复杂度: O(n) — 比 Transformer 的 O(n²) 更适合长序列
```

### S4 (Structured State Space)

Mamba 的前身：固定 SSM 参数 + HiPPO 初始化 + 结构化矩阵 (diagonal + low-rank) 实现高效计算。

---

## 4.3 混合专家

### MoE Block

```
x_in → Router(Linear+Softmax) → Top-K 选择
  ├── Expert_0: FFN(x) ──┐
  ├── Expert_1: FFN(x) ──┤ → 加权求和 → x_out
  └── Expert_K: FFN(x) ──┘

关键: 每个 token 只激活 K 个专家 (通常 K=2)
效果: 参数量大但计算量可控
代表: Mixtral 8×7B (8个专家, Top-2), GPT-4 (据报告)
```

---

## 4.4 量化与部署

### 4.4.1 量化基础概念

**对称量化 vs 非对称量化**：

```
对称量化:   int8 = round(float × scale)            scale = 127 / max(|float|)
            float ≈ int8 / scale
            零点对齐：float=0 ↔ int8=0

非对称量化: int8 = round((float - zero_point) / scale)
            float ≈ int8 × scale + zero_point
            零点不对齐：float=0 ↔ int8=zero_point
```

ncnn 采用**对称量化**，所有 scale 统一为 `int8_val = round(float_val × scale)`，clamp 到 [-127, 127]。

**逐张量量化 vs 逐通道量化**：

```
逐张量 (per-tensor):  整个张量共享 1 个 scale
                       优点: 存储省，推理快
                       缺点: 通道间分布差异大时精度损失

逐通道 (per-channel): 每个通道有独立 scale
                       优点: 精度高，通道差异自适应
                       缺点: scale 存储开销大
```

### 4.4.2 Quantize / Dequantize / Requantize（ncnn 三大量化层）

#### Quantize（FP32 → INT8）

**公式**：`int8_val = round(float_val × scale)`，clamp 到 [-127, 127]

```cpp
// ncnn 实现
static inline signed char float2int8(float v) {
    int int32 = static_cast<int>(round(v));
    if (int32 > 127) return 127;
    if (int32 < -127) return -127;
    return (signed char)int32;
}
```

**参数**：

| Param | 含义 | 示例值 |
|-------|------|--------|
| `0=scale_data_size` | scale 数量 | 1=逐张量，C=逐通道 |

**存储**：`scale_data` [scale_data_size]（从 bin 文件加载）

**维度支持**：

| 维度 | scale 选择 | 典型场景 |
|------|-----------|---------|
| 1D `[W]` | 必须 scale_data_size=1 | 向量量化 |
| 2D `[H,W]` | scale_data_size=1 或 H | 矩阵行级量化 |
| 3D `[C,H,W]` | scale_data_size=1 或 C | 逐通道量化 |

#### Dequantize（INT32 → FP32）

**公式**：`float_val = int_val × scale + bias`

**参数**：

| Param | 含义 |
|-------|------|
| `0=scale_data_size` | scale 数量（1 或 C） |
| `1=bias_data_size` | bias 数量（0、1 或 C） |

**存储**：`scale_data` + `bias_data`（可选）

- Dequantize 输入是 **INT32**（来自 INT8×INT8 乘积累加结果），不是 INT8
- bias 用于补偿量化过程中的系统性偏移

#### Requantize（INT32 → INT8，含激活融合）

**公式**：`int8_out = round(activation(int_in × scale_in + bias) × scale_out)`

**参数**：

| Param | 含义 |
|-------|------|
| `0=scale_in_data_size` | 输入 scale 数量（1 或 C） |
| `1=scale_out_data_size` | 输出 scale 数量（1 或 C） |
| `2=bias_data_size` | bias 数量（0、1 或 C） |
| `3=activation_type` | 激活类型 |
| `4=activation_params` | 激活参数 |

**支持的激活函数（activation_type）**：

| Type | 名称 | 公式 | params |
|------|------|------|--------|
| 0 | 无 | `v` | — |
| 1 | ReLU | `max(v, 0)` | — |
| 2 | LeakyReLU | `v > 0 ? v : v × slope` | `[slope]` |
| 3 | Clip | `clip(v, min, max)` | `[min, max]` |
| 4 | Sigmoid | `1/(1+exp(-v))` | — |
| 5 | Mish | `v × tanh(ln(1+eᵛ))` | — |
| 6 | HardSwish | 分段线性近似 | `[alpha, beta]` |

**存储**：`scale_in_data` + `scale_out_data` + `bias_data`（可选）

> 💡 **Requantize 的核心价值**：在 INT32→INT8 转换过程中同时完成激活函数和第二次量化，省去中间的 FP32 暂存，减少内存读写。这是卷积/全连接层 INT8 推理的关键。

### 4.4.3 Convolution INT8 推理（ncnn 实现详解）

**param 8 = int8_scale_term**：

| 值 | 含义 | 输出类型 |
|----|------|---------|
| 0 | 不使用 INT8 | FP32 |
| 1 | INT8 权重，FP32 输出 | FP32 |
| >100 | INT8 权重 + Requantize 输出 | INT8 |

**加载的量化参数**：

```
weight_data_int8_scales: [num_output]     ← 每输出通道 1 个 scale
bottom_blob_int8_scales: [1]              ← 输入 blob 的 1 个 scale
top_blob_int8_scales: [1]                 ← 当 int8_scale_term > 100
```

**推理流程**：

```
Step 1: 输入 FP32 → INT8（如果输入不是 INT8）
          scale = bottom_blob_int8_scales[0]

Step 2: INT8 权重 × INT8 输入 → INT32 累加
          sum = Σ(weight_int8[q] × input_int8[q])

Step 3: INT32 → FP32 反量化（含 bias）
          scale_in = 1.0 / (bottom_blob_scale × weight_scale[p])
          sum_fp32 = sum × scale_in + bias[p]

Step 4: 激活函数
          sum_fp32 = activation_ss(sum_fp32, activation_type, params)

Step 5a: FP32 输出 → 直接使用 sum_fp32（int8_scale_term ≤ 100）
Step 5b: INT8 输出 → Requantize（int8_scale_term > 100）
          scale_out = top_blob_int8_scales[0]
          output_int8 = float2int8(sum_fp32 × scale_out)
```

**为什么 scale_in = 1/(bottom_scale × weight_scale)？**

```
FP32 卷积:  Σ(W_fp32 × X_fp32)
INT8 卷积:  Σ(W_int8/scale_w × X_int8/scale_x)
          = Σ(W_int8 × X_int8) / (scale_w × scale_x)
          = Σ(W_int8 × X_int8) × 1/(scale_w × scale_x)
          = sum_int32 × scale_in
其中 scale_in = 1/(scale_w × scale_x)
```

### 4.4.4 SDPA INT8 推理（LLM 专用）

SDPA 的 INT8 推理策略与卷积不同——采用**动态逐行量化**：

```
Q: 逐行动态量化（per-h）     ← 每个 token 的分布不同
K: 逐张量动态量化            ← 整个 KV 共享 scale
V: 逐张量动态量化
Softmax: 保持 FP32            ← 精度敏感，不做量化
```

**推理流程**：

```
Step 1: 动态量化 Q（per-h）
          max_per_row = max(|Q[h,:]|)
          scale_q[h] = 127.0 / max_per_row     ← 每行独立 scale
          Q_int8[h,:] = round(Q[h,:] × scale_q[h])

Step 2: 动态量化 K（per-tensor）
          max_all = max(|K|)
          scale_k = 127.0 / max_all            ← 全张量 1 个 scale
          K_int8 = round(K × scale_k)

Step 3: INT8 QK 乘法 + 反量化
          sum_int32 = Σ(Q_int8[h,:] × K_int8[h,:])
          descale = 1.0 / (scale_q[h] × scale_k)
          QK_fp32 = sum_int32 × descale × (1/√dₖ)

Step 4: Softmax（FP32）
          attn = softmax(QK_fp32)              ← 保持 FP32

Step 5: 动态量化 attn（per-h）→ INT8
Step 6: 动态量化 V（per-tensor）→ INT8
Step 7: INT8 × INT8 乘法 → FP32 反量化 → 输出
```

**为什么 Softmax 不做量化？**

> Softmax 对输入值极其敏感：一个小的量化误差经过 exp 函数会被指数放大。实验表明，Softmax 做 INT8 量化会导致注意力分布严重失真，精度损失 5-10%。

### 4.4.5 静态量化：校准流程

静态量化需要预先确定 scale 值，流程如下：

```
1. 准备校准数据集（通常 100-1000 张样本）
2. 用 FP32 模型跑一遍校准集
3. 收集每层的激活统计信息：
   - 最大值/最小值
   - KL 散度分布（KL divergence calibration）
4. 计算每层的 scale：
   - max 校准: scale = 127 / max(|activation|)
   - KL 校准: 找到最优截断阈值，使量化前后分布的 KL 散度最小
5. 将 scale 写入 param/bin 文件
```

**KL 校准（KL Divergence Calibration）**：

```
不是简单地用 max 值做截断，而是：
1. 收集激活值的直方图（通常 2048 个 bin）
2. 对每个候选截断阈值，计算量化前后分布的 KL 散度
3. 选择使 KL 散度最小的阈值
4. 这个阈值通常比 max 小很多，保留"大部分"数据的高精度
```

> ncnn 使用 `ncnnoptimize` 工具完成此流程。对于 Convolution/InnerProduct 层，`int8_scale_term` 参数决定了量化策略。

### 4.4.6 AWQ / GPTQ（权重量化校准）

**AWQ（Activation-aware Weight Quantization）**：

```
1. 用校准集跑一遍推理，记录每层的激活分布
2. 对激活值大的权重（对输出影响大），减小量化缩放因子
3. 对激活值小的权重（敏感度低），正常量化
4. 仅约 0.1% 的权重需要特殊保护
```

- 感知激活分布的权重量化，避免"一刀切"量化损失
- 4bit AWQ 精度损失 < 1%，适合端侧部署

**GPTQ（Generative Pre-trained Quantization）**：

```
1. 逐层量化，每层独立优化
2. 对每个权重 w，找到最优 int4 量化值使该层输出误差最小
3. 贪心迭代：每次更新一个权重，更新残差
```

- 逐层后处理，无需重新训练
- GPTQ 4bit 是当前端侧 LLM 的主流选择

### 4.4.7 FP8 量化

```
FP8 E4M3:  4位指数 + 3位尾数  范围 [-120, 120]，适合前向传播
FP8 E5M2:  5位指数 + 2位尾数  范围 [-57344, 57344]，适合梯度
```

- H100/B200 等 GPU 原生支持 FP8 矩阵乘法
- 比 INT8 更灵活（动态范围更大），比 FP16 省一半内存

### 4.4.8 PagedAttention（KV Cache 分页管理）

```
传统 KV Cache:  连续分配，序列长度变化时需重新分配
PagedAttention:  将 KV Cache 分成固定大小 page（如 16 tokens/page）
                  按需分配，支持跨序列共享 page
```

- vLLM 的核心创新：像 OS 虚拟内存一样管理 KV Cache
- 解决碎片化问题，GPU 利用率提升 2-4×
- 支持 Continuous Batching（连续批处理），吞吐量提升 10-24×

### 4.4.9 Tensor Parallelism（张量并行）

```
将大模型权重切分到多张 GPU：
  - Column Parallel: 按输出维度切分（Gemm 的 out_features 维度）
  - Row Parallel:    按输入维度切分（Gemm 的 in_features 维度）

示例 (hidden_dim=4096, 2张GPU):
  GPU0 持有权重 W[:2048]  → 输出 [:2048]
  GPU1 持有权重 W[2048:]  → 输出 [2048:]
  AllReduce 合并
```

- Megatron-LM 范式：Attention 的 Q/K/V 投影用 Column Parallel，输出投影用 Row Parallel
- 单卡放不下的大模型必须用多卡并行

### 4.4.10 INT8 与 Vulkan 互斥

```
ncnn 限制: if (int8_scale_term) support_vulkan = false
```

- INT8 量化需要精确控制数值范围，Vulkan GPU 计算着色器的精度模型与 CPU INT8 不完全兼容
- 启用 INT8 时自动回退到 CPU 推理

---

## 4.5 多模态

### Vision Transformer (ViT)

```
图像 [B,3,224,224]
  → PatchEmbed: Conv2D(k=16, stride=16) → [B,196,dim]
  → + CLS Token + Position Embed
  → N × Transformer Block
  → CLS Token → 分类头
```

### CLIP（对比学习）

```
图像 → ViT → image_features ──→ L2 Norm ──→ Cosine Similarity
文本 → Transformer → text_features → L2 Norm ──↗
                                                   ↓
                                            对比损失: 匹配对相似度最大化

**余弦相似度 vs 点积**

> 点积 `a·b` 同时受方向和模长影响：`a·b = |a| × |b| × cos(θ)`。
> 余弦相似度只看方向：`cos(θ) = a·b / (|a| × |b|)`。
>
> 当向量先做了 L2 归一化（|a|=|b|=1），点积就等价于余弦相似度。CLIP 中对 image_features 和 text_features 都做了 L2 归一化，所以余弦相似度 = 点积。
```

### mRoPE（多维位置编码）

```
文本 token: 一维 temporal 位置编码
图像 patch: 三维 (temporal, height, width) 位置编码
交错 mRoPE (Qwen3.5-VL): 维度按 modulo-3 交替分配
```

### Spectrogram / InverseSpectrogram

**Spectrogram**：时域信号 → STFT → 频率幅度谱 `[B, T] → [B, freq, time]`

**InverseSpectrogram**：频谱图 → ISTFT → 时域信号

- 音频处理的第一步：将一维波形转为二维时频图
- 代表模型：**Whisper**, 语音识别, 语音合成

### Diffusion UNet

```
噪声图 x_t + 时间步 t
  → TimeEmbed: SiLU + Linear
  → 编码器(ResBlock + Attn + Downsample)
  → 解码器(ResBlock + Attn + Upsample)
  → Skip Connections
  → 每层注入 TimeEmb (SiLU+Linear+Add)
```

- 时间步条件注入：每个 ResBlock 通过 SiLU+Linear 将时间嵌入加到特征上
- 使网络根据噪声水平调整去噪策略

### RetNet

```
x_in [seq_len, dim]
  → GroupNorm → Split → Q, K, V
  → XPOS (带指数衰减的位置编码)
  +-- 并行模式 (训练): Retention(Q, K, V)  # 类似注意力，可并行
  +-- 循环模式 (推理): RNN-like 递推       # O(1) 每步
  → GroupNorm + Swish gating + Linear
  → x_out
```

- 同一个块既可并行训练（像 Transformer），又可递推推理（像 RNN）

### RWKV Block

```
TimeMix (时间混合): WKV 注意力 (线性复杂度)
  → 无 softmax 的注意力: φ(Q)·(φ(K)ᵀ·V)
ChannelMix (通道混合): SiLU 门控 FFN
  → SiLU(gate) · up → down
```

- Transformer 质量 + RNN 推理效率，每步 O(1)

---

# 第五章 部署实战——从模型到产品

> 训练好的模型如何变成高效推理产品？本章从 ncnn 视角讲部署全链路。

## 5.1 ncnn 内存管理

### Blob 与 Mat

```
Blob (blob):       网络中的数据节点（有名字、有生产者消费者关系）
Mat (matrix):      Blob 的实际数据容器（有维度、有类型）

Param 文件: 定义 blob 拓扑（谁连谁）
Bin 文件:    定义权重数据
运行时:      blob 引用 mat，mat 指向实际内存
```

### Allocator 机制

ncnn 有两层内存分配器：

```
1. PoolAllocator (ncnn::PoolAllocator):
   - 预分配一块大内存，按需分配子块
   - 避免频繁 malloc/free
   - 用途: 中间 blob 的分配

2. UnlockedPoolAllocator (ncnn::UnlockedPoolAllocator):
   - PoolAllocator 的线程安全版本
   - 多线程推理时使用

3. Workspace (工作空间):
   - Conv/SDPA 等算子的临时缓冲区
   - 推理结束后立即释放
   - 可通过 Option::workspace_allocator 自定义
```

### 中间结果复用策略

```
ncnn 的图执行器会分析 blob 的生命周期:
  blob A 只在 layer 1-3 使用 → layer 3 结束后内存可释放
  blob B 在 layer 5 才需要  → layer 4 输出时复用 A 的内存

效果: 1000+ 层的网络可能只需要几个 blob 同时存活
```

## 5.2 ncnn 图优化

### 5.2.1 算子融合

将多个连续算子合并为一个，减少内存读写：

```
Conv + BatchNorm → Conv (融合 BN 参数到 Conv 权重)
  公式: W' = W × (slope / sqrt(var + eps))
        b' = bias - slope × mean / sqrt(var + eps)

Conv + ReLU → Conv (融合激活到卷积内部)
  推理时: output = max(conv_out, 0)  不需要额外 blob

InnerProduct + Dropout → InnerProduct (推理时丢弃不做)
```

**ncnn 的 ncnnoptimize 工具**：

```bash
ncnnoptimize model.param model.bin model-opt.param model-opt.bin [options]
  0=FP32, 1=FP16  → 权重转 FP16
  -n  → 跳过 ncnnoptimize 的图优化
```

### 5.2.2 融合清单（ncnnoptimize 全部融合）

| 融合模式 | 效果 |
|---------|------|
| Conv + BatchNorm | BN 参数吸收，消除 BN 层 |
| ConvolutionDepthwise + BatchNorm | 同上 |
| InnerProduct + BatchNorm | 同上 |
| Conv + Activation (ReLU/Sigmoid/Mish/HardSwish) | 激活融合到 Conv 内部 |
| InnerProduct + Dropout | 消除 Dropout（推理时无效） |
| BatchNorm + Scale | 合并 BN 和 Scale |
| Convolution + Mul/Add | 融合 Elementwise 操作 |
| MemoryData + BinaryOp | 常量折叠 |
| Flatten after GlobalPooling | 消除多余 Flatten |
| Reshape after GlobalPooling | 消除多余 Reshape |

### 5.2.3 Dead Code Elimination

```
ncnnoptimize 会消除:
  - 无消费者的 blob（孤立数据）
  - 不需要的 Noop 层
  - 可以被替代的 Pooling1×1
  - 重复的 Split（多路复用）
```

## 5.3 ncnn 平台适配

### 5.3.1 CPU 后端

| 平台 | 优化路径 | 关键指令集 |
|------|---------|-----------|
| ARM 32-bit | arm 目录 | NEON (32-bit) |
| ARM 64-bit | arm 目录 | NEON (64-bit), FP16 (ARMv8.2+) |
| x86 64-bit | x86 目录 | SSE → AVX2 → AVX-512 |
| RISC-V | riscv 目录 | V (Vector Extension) |
| LoongArch | loongarch 目录 | LSX (256-bit SIMD) |
| MIPS | mips 目录 | MSA (MIPS SIMD) |

**运行时检测**：ncnn 在 `Extractor::extract()` 时自动选择最优实现。

### 5.3.2 Vulkan GPU 后端

```
启用: opt.use_vulkan_compute = true

优势:
  - 并行度大幅提升（GPU 数千核心 vs CPU 几十核心）
  - 大矩阵乘法加速 5-10×
  - FP16 原生支持

限制:
  - 小模型可能不如 CPU（kernel launch 开销）
  - INT8 量化不兼容（int8_scale_term → 回退 CPU）
  - Vulkan 驱动依赖（老旧设备不支持）
```

### 5.3.3 ARM 专属优化

```
ncnn 在 ARM 上的关键优化:

1. Winograd FP16 (conv 3×3):
   src/layer/arm/convolution_3x3_winograd_fp32.cpp
   → 比 im2col 快 1.5-2×

2. Depthwise Conv ARM NEON:
   按行向量化，8-16 像素并行

3. ARM SVE (Scalable Vector Extension):
   支持可变向量长度（128-2048 bit）
   → 适配不同 ARM 芯片（Cortex-A710 vs A715）
```

## 5.4 性能调优

### 5.4.1 Profiling 方法

```cpp
// ncnn 内建 profiling
ncnn::Option opt;
opt.num_threads = 4;
opt.lightmode = true;

ncnn::Net net;
net.opt = opt;
net.load_param("model.param");
net.load_model("model.bin");

// 逐层计时
net.opt.openmp = true;  // 启用多线程
// 需要修改 ncnn 源码开启每层计时:
// #define NCNN_STRING 1
// net.print_layer_info();
```

### 5.4.2 瓶颈诊断流程

```
Step 1: 看总耗时
  ncnn::get_current_time() 包装推理
  → 确定是慢还是正常

Step 2: 看各层耗时占比
  最耗时的通常是:
  - 大矩阵乘法 (Attention QKV, FC)
  - 大卷积 (stem 7×7, 第一层)
  - SDPA (长序列时 O(n²))

Step 3: 看内存占用
  net.get_memory_footprint()
  → 确认是否在内存预算内

Step 4: 看线程利用率
  top -H -p <pid>  (Linux)
  → 如果线程数远小于 opt.num_threads，说明并行度不够
```

### 5.4.3 常见优化技巧

| 问题 | 解法 |
|------|------|
| 总耗时慢 | INT8 量化、FP16 权重 |
| 大模型内存不足 | 分批推理、量化到 INT8 |
| SDPA 长序列慢 | FlashAttention、滑动窗口、KV Cache 压缩 |
| CPU 利用率低 | 增加 num_threads、检查算子是否支持多线程 |
| GPU 不如 CPU | 模型太小（Vulkan 开销 > 加速收益） |
| Vulkan crash | 检查驱动版本、关闭 INT8 |

## 5.5 量化调试

### 5.5.1 精度退化诊断

```
当 INT8 模型精度低于 FP32 时:

1. 逐层对比:
   FP32 和 INT8 分别跑同一个输入
   逐层对比输出差异 (L2 norm, cosine similarity)
   → 找到精度退化最严重的层

2. 常见退化原因:
   - 某层激活值分布极端（长尾分布，max 校准截断太多信息）
   - Softmax 前做了量化（exp 放大误差）
   - LayerNorm 内部做了量化（方差计算对精度敏感）

3. 解决策略:
   - 退化层保持 FP32（混合精度量化）
   - 用 KL 校准替代 max 校准
   - 增加校准集数量和多样性
```

### 5.5.2 Scale 校准技巧

```
Max 校准（简单但有缺陷）:
  scale = 127 / max(|activation|)
  问题: 一个极端值会拉低整体精度

KL 校准（推荐）:
  1. 收集激活直方图（2048 bins）
  2. 对每个候选阈值，计算量化前后的 KL 散度
  3. 选择最小 KL 散度的阈值
  效果: 通常保留 99.9% 的数据精度

Percentile 校准（折中方案）:
  scale = 127 / percentile(|activation|, 99.9%)
  比 max 好，比 KL 简单
```

---

# 第六章 模型压缩——更小更快的模型

> 压缩让大模型能在小设备上跑。本章覆盖剪枝、蒸馏和量化进阶。

## 6.1 剪枝（Pruning）

### 6.1.1 非结构化剪枝

```
做法: 逐个权重剪，绝对值小的先剪
      if |w| < threshold → w = 0

效果: 稀疏度可达 90%
问题: 稀疏矩阵需要专用硬件加速
      通用 CPU/GPU 上稀疏矩阵不一定比密集快
```

**幅度剪枝（Magnitude Pruning）**：

```
Iterative Magnitude Pruning (IMP):
  1. 训练模型到收敛
  2. 剪掉 k% 最小权重
  3. 重新训练剩余权重（学习率恢复）
  4. 重复 2-3

"彩票假说" (Lottery Ticket Hypothesis):
  大网络中存在一个小子网络，单独训练能达到同样效果
  这个子网络就是"中奖彩票"
```

### 6.1.2 结构化剪枝

```
做法: 按通道/滤波器维度剪
      if channel 的 L1 norm < threshold → 整通道剪掉

效果: 直接减少计算量和参数量
      不需要专用加速硬件

代表:
  - L1 Norm Channel Pruning: 按通道 L1 范数排序
  - Slimming: 训练时加 L1 正则到 BN 的 γ，自动稀疏通道
```

### 剪枝 vs 量化对比

| 方法 | 压缩比 | 加速比 | 精度影响 | 硬件需求 |
|------|--------|--------|---------|---------|
| 非结构化剪枝 | 极高 (90%+) | 低 | 小 (迭代恢复) | 稀疏加速硬件 |
| 结构化剪枝 | 中 (30-70%) | 高 | 小-中 | 无特殊需求 |
| INT8 量化 | 高 (4×) | 高 | 小 | INT8 推理支持 |
| INT4 量化 | 极高 (8×) | 极高 | 中-大 | 专用硬件/AWQ |

## 6.2 知识蒸馏（Knowledge Distillation）

### 6.2.1 Logits 蒸馏

```
Teacher (大模型) → logits_T [batch, vocab]
Student (小模型) → logits_S [batch, vocab]

蒸馏损失:
  KD_Loss = KL_Divergence(softmax(logits_T/T), softmax(logits_S/T))
  ↑ T 是温度参数

温度 T 的作用:
  T=1: 正常 softmax，概率分布尖锐
  T>1: 分布变平滑，"暗知识"（dark knowledge）暴露
       比如 "猫" vs "狗" 的相似度 > "猫" vs "汽车"

总损失:
  Loss = α · CE(logits_S, labels) + (1−α) · T² · KD_Loss
```

### 6.2.2 特征蒸馏

```
Teacher 中间层特征 → 学生对应层特征

做法:
  1. 对齐维度: 学生加投影层 (Linear) 匹配 Teacher 维度
  2. 对齐空间: 上采样/下采样匹配分辨率
  3. 计算损失: MSE/Attention 匹配中间特征

FitNet: 学生学 Teacher 中间层的回归
AT (Attention Transfer): 学生学 Teacher 的注意力图
```

### 6.2.3 蒸馏流程

```
1. 训练 Teacher 模型（大模型，高精度）
2. 冻结 Teacher，用同一训练集提取 logits 和特征
3. 训练 Student 模型:
   学生损失 = 交叉熵(学生输出, 标签) + KD 权重 × KL 散度
4. 评估 Student 精度

典型效果:
  Teacher: ResNet-50 (76.1%)
  Student: ResNet-18 (69.8%)
  + 蒸馏: ResNet-18 → 72.5% (+2.7%)
```

## 6.3 量化进阶

### 6.3.1 PTQ vs QAT

| 方法 | 全称 | 流程 | 精度 | 成本 |
|------|------|------|------|------|
| **PTQ** | Post-Training Quantization | 校准 → 量化 | 中 | 低 |
| **QAT** | Quantization-Aware Training | 伪量化训练 → 微调 → 导出 | 高 | 高 |

**PTQ**（训练后量化）：

```
优点: 不需要重新训练，只需少量校准数据（100-1000 样本）
缺点: 精度损失 1-3%
适用: 快速部署、模型不可重训
```

**QAT**（量化感知训练）：

```
优点: 精度几乎无损（< 0.5% 损失）
缺点: 需要完整微调，计算成本高
流程:
  1. 在前向传播中插入"伪量化"操作
     fake_quant(x) = round(x/scale) * scale
  2. scale 作为可学习参数或固定校准值
  3. 正常训练，反向传播通过直通估计器（STE）
  4. 导出时用真实量化替代伪量化
```

### 6.3.2 INT4 / INT2 极限量化

```
INT4 范围: [-8, 7]    比 INT8 再省 50%
INT2 范围: [-2, 1]    比 INT8 省 75%

挑战:
  - 动态范围极小，大量信息丢失
  - 必须用 AWQ/GPTQ 感知权重量化
  - 某些层（Attention 输出投影）必须保持 INT8/FP16

实践:
  - LLM.int8() (LLaMA): 99% INT8 + 1% 异常值 FP16
  - QLoRA: 4bit 基础模型 + LoRA 适配器
  - BitNet: 1.58bit 训练（{-1, 0, 1} 三值量化）
```

### 6.3.3 KV Cache 量化

```
KV Cache 是 LLM 推理的主要内存瓶颈。

量化策略:
  Per-head 量化: 每个 attention head 独立 scale
  Per-layer 量化: 整层共享 scale
  Mixed precision: K 量化 + V 保持 FP16

效果 (8 heads, head_dim=128, seq=2048):
  FP16: 2 × 8 × 128 × 2048 × 2 bytes = 8 MB
  INT8: 4 MB
  INT4: 2 MB
```

---

# 附录 算子速查表

## A. 算子分类总表

| 类别 | 算子 | 核心功能 |
|------|------|---------|
| **数学** | Add, Sub, Mul, Div, Pow, Sqrt, Exp, Log, Abs | 数值运算 |
| **矩阵** | MatMul, Gemm, InnerProduct, EinSum | 线性变换 |
| **归约** | Sum, Mean, Max, Min, Prod, Norm | 聚合 |
| **激活** | ReLU, GELU, Swish, Sigmoid, Tanh, SwiGLU, Softmax | 非线性 |
| **归一化** | BatchNorm, LayerNorm, RMSNorm, GroupNorm, LayerScale | 稳定训练 |
| **卷积** | Conv1D/2D/3D, Depthwise, Dilated, Deconv | 局部特征 |
| **池化** | MaxPool, AvgPool, GAP, SPP, RoIAlign | 空间压缩 |
| **注意力** | SDPA, MHA, GQA, MQA, RoPE, Cross-Attn, QK-Norm | 全局依赖 |
| **注意力优化** | FlashAttention, SlidingWindow, ALiBi | 高效推理 |
| **循环** | RNN, LSTM, GRU | 时序记忆 |
| **张量** | Reshape, Permute, Slice, Split, Concat, Gather, ShuffleChannel | 形状操作 |
| **选择/排序** | ArgMax, TopK, Where | 索引与条件 |
| **累积** | CumSum | 前缀和 |
| **嵌入** | Embed, OneHot, Position Embedding | 离散→连续 |
| **Loss** | CrossEntropy, BCE, MSE, Focal, CTC | 训练优化 |
| **量化** | Quantize, Dequantize, Requantize | 对称量化 per-tensor/channel |
| **Conv-INT8** | INT8×INT8→INT32, Requantize, 激活融合 | 精度压缩 |
| **SDPA-INT8** | 动态量化 per-h, Softmax-FP32 | LLM 加速 |
| **部署** | PagedAttention, TensorParallel | 推理加速 |
| **空间** | PixelShuffle, GridSample, Interp, Reorg, Spectrogram | 部署辅助 |

## B. 组合结构速查表

| 结构 | 核心算子 | 复杂度 | 代表模型 |
|------|---------|--------|---------|
| CBR | Conv+BN+ReLU | O(n²k²C²) | ResNet |
| ResBlock | CBR+CBR+Add | O(n²k²C²) | ResNet |
| SE Block | GAP+FC+ReLU+FC+Sig+Scale | O(C²/r) | SENet |
| LSTM Cell | 4×Sigmoid+Tanh+Mul+Add | O(nd²) | 语音/翻译 |
| Transformer | RMSNorm+SDPA+SwiGLU+Add | O(n²d+nd²) | LLaMA, Qwen |
| GQA Block | RMSNorm+QKNorm+RoPE+SDPA | O(n²d·g/h) | Qwen3 |
| Enc-Dec | Cross-Attn+SDPA+FFN | O(n²d+nd²) | T5, Whisper |
| U-Net | Conv+Down+Up+Skip | O(n²C) | Stable Diffusion |
| ViT | PatchEmbed+Attn+MLP | O(n²d+nd²) | ViT, CLIP |
| MoE | Router+TopK+FFN | O(nd²·K/E) | Mixtral |
| Mamba | SSM+Conv+Swish | O(nd) | Mamba |
| Detection Head | Conv+IoU+NMS | O(m²) m=框数 | YOLO, Faster R-CNN |

## C. 框架参数命名对照

| 参数 | PyTorch | TensorFlow | ncnn param |
|------|---------|------------|------------|
| kernel | kernel_size | filter_size | 0=kernel_w 1=kernel_h |
| stride | stride | strides | 2=stride_w 3=stride_h |
| padding | padding | padding | 4=pad_left 5=pad_right |
| dilation | dilation | dilation_rate | 6=dilation_w 7=dilation_h |
| groups | groups | groups | 8=group |
| bias | bias | use_bias | 单独 Bias 层 |
| **量化参数** | — | — | Conv: 8=int8_scale_term; Quantize: 0=scale_data_size; Dequantize: 0=scale_data_size 1=bias_data_size; Requantize: 0=scale_in_size 1=scale_out_size 2=bias_size 3=activation_type 4=activation_params |

## D. ncnn 量化层参数详解

| 层 | 输入 | 输出 | bin 存储内容 |
|----|------|------|-------------|
| **Quantize** | FP32 blob | INT8 blob | scale_data [N] |
| **Dequantize** | INT32 blob | FP32 blob | scale_data [N] + bias_data [M] |
| **Requantize** | INT32 blob | INT8 blob | scale_in [N] + scale_out [M] + bias_data [K] |
| **Conv-INT8** | FP32/INT8 | FP32/INT8 | weight_int8_scales [num_output] + bottom_int8_scales [1] + top_int8_scales [1] |
| **SDPA-INT8** | FP32 blobs | FP32 blob | param 18=int8_scale_term，动态量化无需 bin 存储 |

> ncnn 量化公式统一为对称量化：`int8 = round(float × scale)`，范围 [-127, 127]。
> 反量化：`float = int × scale + bias`。Requantize 融合激活函数，省去中间 FP32 暂存。


# 实战案例：Qwen3 0.6B 模型详解

> 以 `LLM-model/` 目录中的 Qwen3 模型为例，将前文所有算子串联，展示一个真实大模型是如何组合使用的。

## A.1 模型概况

### 基本信息

```
模型类型:    Qwen3-0.6B
Token 类型: BBPE (Byte-level BPE)
词表大小:    151,642
注意力层数:  28
位置编码:    RoPE (theta = 100,000, head_dim = 128)
视觉支持:    关闭
工具调用:    启用
```

### 架构参数

| 参数 | 值 | 说明 |
|------|-----|------|
| hidden_dim | 1024 | 隐藏层维度 |
| num_q_heads | 16 | 查询头数 |
| num_kv_heads | 8 | 键值头数 (GQA) |
| head_dim | 128 | 每个头的维度 |
| intermediate_dim | 3072 | MLP 中间维度 (3x) |
| scale | 0.0883883 | 1/sqrt(128) |
| eps | 1e-6 | RMSNorm epsilon |

### 模型文件列表

| 文件 | 大小 | 说明 |
|------|------|------|
| qwen3_embed_token.ncnn.bin | 594M | Embedding 权重 (FP32) |
| qwen3_embed_token.ncnn.fp16.bin | 297M | Embedding (FP16) |
| qwen3_embed_token.ncnn.int8.bin | 149M | Embedding (INT8) |
| qwen3_decoder.ncnn.bin | 1.7G | Decoder 权重 (FP32) |
| qwen3_decoder.ncnn.fp16.bin | 841M | Decoder (FP16) |
| qwen3_decoder.ncnn.int8.bin | 421M | Decoder (INT8) |
| qwen3_embed_token.ncnn.param | - | Embedding 网络结构 |
| qwen3_decoder.ncnn.param | - | Decoder 网络结构 (1017 层, 1405 blob) |
| qwen3_proj_out.ncnn.param | - | Projection 网络结构 |
| vocab.txt | 151,642 行 | BPE 词表 |
| merges.txt | - | BPE 合并规则 |
| model.json | - | 模型配置 |

## A.2 算子到架构块的映射

Qwen3 的每个 Transformer Block (28层完全相同) 包含以下算子：

### Attention 子块

```
输入 [seq_len, 1024]
  |
  v RMSNorm (affine_size=1024, eps=1e-6)  <- 归一化算子
  v Split -> 3路 (Q, K, V)                <- 张量操作
  |
  +-- Q: Gemm(1024->2048)                 <- 矩阵乘法
  |     Reshape(128,16,-1)                <- 头拆分
  |     RMSNorm(128, 1e-6)                <- QK-Norm: 每头独立归一化
  |     Permute -> RotaryEmbed(RoPE)      <- 位置编码
  |
  +-- K: Gemm(1024->1024)                 <- 8 KV heads
  |     Reshape(128,8,-1)
  |     RMSNorm(128, 1e-6)                <- QK-Norm
  |     Permute -> RotaryEmbed(RoPE)
  |     ExpandDims + Tile (8->16 heads)   <- GQA: 头复制
  |     Reshape(128,-1,16)
  |
  +-- V: Gemm(1024->1024)
  |     Reshape(128,8,-1)
  |     Permute
  |     ExpandDims + Tile (8->16 heads)
  |     Reshape(128,-1,16)
  |
  v SDPA (scale=0.0883883, kv_cache=1)   <- 缩放点积注意力
     输入: Q_rope, K_exp, V_exp, mask, past_kv
     输出: attn_out, out_cache_k, out_cache_v
  |
  v Permute + Reshape(2048,-1)
  v Gemm(2048->1024)                     <- 输出投影
  v Add (残差)                            <- 残差连接
```

### MLP 子块 (SwiGLU)

```
  v RMSNorm (affine_size=1024, eps=1e-6)
  v Split -> 2路 (gate, up)
  |
  +-- gate: Gemm(1024->3072)              <- 门控投影
  |     Swish/SiLU                         <- 激活函数
  |
  +-- up:   Gemm(1024->3072)              <- 上投影
  |
  v Mul (逐元素乘)                          <- 门控: Swish(gate) * up
  v Gemm(3072->1024)                      <- 下投影
  v Add (残差)
  |
  输出 [seq_len, 1024]
```

## A.3 从 param 文件读取每个算子

以下是 decoder 第 0 层的完整算子序列（对应 param 文件第 12-46 行）：

| 行号 | 算子类型 | 输入 -> 输出 | 关键参数 | 对应章节 |
|------|---------|-------------|---------|---------|
| 12 | RMSNorm | 2 -> 146 | affine=1024, eps=1e-6 | 1.3.1 |
| 13 | Split | 146 -> 147,148,149 | 3路 (Q,K,V) | 1.4.3 |
| 14 | Gemm (Q) | 149 -> 150 | out=2048, in=1024 | 2.1.1 |
| 15 | Reshape | 150 -> 151 | (128, 16, -1) | 1.4.1 |
| 16 | RMSNorm | 151 -> 152 | affine=128 (QK-Norm) | 1.3.3 |
| 17 | Permute | 152 -> 153 | transpose dim=2 | 1.4.2 |
| 18 | Gemm (K) | 148 -> 154 | out=1024, in=1024 | 2.1.1 |
| 19 | Reshape | 154 -> 155 | (128, 8, -1) | 1.4.1 |
| 20 | RMSNorm | 155 -> 156 | affine=128 (QK-Norm) | 1.3.3 |
| 21 | Permute | 156 -> 157 | transpose dim=2 | 1.4.2 |
| 22 | Gemm (V) | 147 -> 158 | out=1024, in=1024 | 2.1.1 |
| 23 | Reshape | 158 -> 159 | (128, 8, -1) | 1.4.1 |
| 24 | Permute | 159 -> 160 | transpose dim=2 | 1.4.2 |
| 25 | RotaryEmbed | 153,88,145 -> 161 | Q RoPE | 2.4.3 |
| 26 | RotaryEmbed | 157,87,144 -> 162 | K RoPE | 2.4.3 |
| 27 | ExpandDims | 162 -> 163 | K GQA expand | 1.4.1 |
| 28 | Tile | 163 -> 164 | rep=2 (8->16) | 1.4.2 |
| 29 | Reshape | 164 -> 165 | (128, -1, 16) | 1.4.1 |
| 30-32 | ExpandDims+Tile+Reshape | V GQA | V 8->16 heads | GQA |
| 33 | SDPA | 6入3出 | scale=0.088, kv_cache | 2.4.1 |
| 34-35 | Permute+Reshape | -> (2048, -1) | 合并头 | 1.4 |
| 36 | Gemm (O) | 171 -> 172 | out=1024, in=2048 | 2.1.1 |
| 37 | Add (残差) | 1,172 -> 173 | + x_in | 3.1.2 |
| 38-39 | Split + RMSNorm | MLP 归一化 | affine=1024 | 1.3.3 |
| 40 | Split | -> gate, up | 2路 | 1.4.3 |
| 41 | Gemm (gate) | 178 -> 179 | out=3072 | 2.1.1 |
| 42 | Swish (SiLU) | 179 -> 180 | 激活 | 1.2 |
| 43 | Gemm (up) | 177 -> 181 | out=3072 | 2.1.1 |
| 44 | Mul | 180,181 -> 182 | gate * up | 1.1 |
| 45 | Gemm (down) | 182 -> 183 | out=1024 | 2.1.1 |
| 46 | Add (残差) | 174,183 -> 184 | + attn_out | 3.1.2 |

此模式在 28 层中完全重复，仅权重参数不同。

## A.4 Embedding 和 Projection 层

### Embedding (qwen3_embed_token.ncnn.param)

```
Input   in0 -> in0
Embed   in0 -> out0   0=1024 1=151936 2=0 3=155582464
```

- **hidden_dim=1024**, **vocab_size=151936**
- **权重参数量**：155,582,464 个 float = 151936 x 1024 = 594 MB (FP32)
- **Weight Tying**：proj_out 复用 embed_token 的二进制文件 (W_proj = E_embed^T)
- **本质**：Gather 操作 —— token_id -> embedding_matrix[token_id]

**为什么 Weight Tying 有效？**

> Embedding 把离散 token ID 映射为连续向量（输入端），Projection 把连续向量映射回 token 概率（输出端）。两者互为逆操作。
>
> 直觉：如果模型学会用某个向量表示"猫"的含义（embedding），那用同一个向量的转置来识别"这个向量是否表示猫"（projection）是合理的。两者共享权重 = 让输入和输出使用同一套语义空间。
>
> 大模型（如 GPT-3 175B）中这省下了约 50% 的参数（embedding + projection 通常占总参数 1/3~1/2）。

### Projection (qwen3_proj_out.ncnn.param)

```
Input   in0 -> in0
Gemm    in0 -> out0   10=-1 2=0 3=1 4=0 5=1 6=1 7=1 8=151936 9=1024
```

- **Gemm**：输出维度 151936，输入维度 1024
- **7=1**：无 bias（因为 Weight Tying，偏置在 Embedding 侧）
- **bin 文件复用**：proj_out_bin 指向 qwen3_embed_token.ncnn.bin

## A.5 精度版本对比

| 版本 | Decoder 权重 | Embedding 权重 | 总大小 | 速度 | 精度 |
|------|-------------|---------------|--------|------|------|
| FP32 | 1.7 GB | 594 MB | ~2.9 GB | 基准 | 100% |
| FP16 | 841 MB | 297 MB | ~1.4 GB | 2x | 99.9% |
| INT8 | 421 MB | 149 MB | ~0.7 GB | 4x | 97-99% |

- FP32：全精度，适用于对精度要求极高的场景
- FP16：使用 BF16 存储，推理速度翻倍，精度几乎无损
- INT8：动态量化，SDPA 层使用 INT8 matmul，适合边缘部署
- INT8 与 Vulkan 互斥：`if (int8_scale_term) support_vulkan = false`

## A.6 BPE Tokenizer 配置

### 词表 (vocab.txt)

- **151,642 行**，每行一个 token，行号即 token ID
- 前几个 token：!, ", #, $, %, &, ...
- 特殊 token：<|im_start|>, <|im_end>, <|system|>, <|user|>, <|assistant|>, <|tool_response|>, <|vision_start|>, <|vision_end|> 等 26 个

### 合并规则 (merges.txt)

- 每行格式：token_A token_B
- 行号 = rank（越小优先级越高）
- 前几行：`ā ā`, `āā āā`, `i n`, `ā t`...

## A.7 完整推理流程

```
1. 加载 model.json -> 读取所有路径和参数
2. 加载 vocab.txt + merges.txt -> 构建 BPE Tokenizer
3. 加载 embed_token -> Embedding 网络
4. 加载 decoder -> 28层 Transformer
5. 加载 proj_out -> Projection 网络 (复用 embed_token 权重)

推理:
  输入文本 -> BPE Tokenize -> token_ids [N]
    -> Embedding (Gather) -> token_embed [N, 1024]
    -> RoPE cos/sin Cache -> [64, N]
    -> Decoder (28层) -> hidden_state [1, 1024] + KV Cache
    -> Projection (Gemm) -> logits [151936]
    -> Sampling (Softmax + Top-P/Top-K) -> next_token_id
    -> BPE Decode -> 输出文本
    -> 循环直到 EOS 或 max_new_tokens
```

---

## E. 算子 FLOPs 公式速查

> 所有 FLOPs 计数 = 乘法次数 + 加法次数 = 2 × MACs

### 基础算子

| 算子 | FLOPs | 说明 |
|------|-------|------|
| **MatMul** [M,K]×[K,N] | 2·M·N·K | 每个输出元素 K 次乘加 |
| **Add** [N] + [N] | N | 逐元素加 |
| **Mul** [N] × [N] | N | 逐元素乘 |
| **Sqrt** [N] | N | 逐元素 |
| **Exp** [N] | N | 逐元素 |
| **Softmax** [N] | 4N | max + exp + sum + div |
| **ReLU** [N] | N | 逐元素比较 |

### 卷积类

| 算子 | FLOPs | 说明 |
|------|-------|------|
| **Conv2D** | 2·C_out·C_in·k²·H_out·W_out | 标准卷积 |
| **DepthwiseConv** | 2·C_in·k²·H_out·W_out | 每通道独立 |
| **ConvTranspose2D** | 2·C_in·C_out·k²·H_out·W_out | 上采样卷积 |
| **1×1 Conv** | 2·C_out·C_in·H·W | 通道变换 |

### 池化类

| 算子 | FLOPs | 说明 |
|------|-------|------|
| **MaxPool** k×k | k²·H_out·W_out·C | 比较次数 |
| **AvgPool** k×k | k²·H_out·W_out·C | 加+除 |
| **GlobalAvgPool** | H·W·C | 加+除 |

### 注意力类

| 算子 | FLOPs | 说明 |
|------|-------|------|
| **SDPA** | 2·n²·d + 2·n·d² | QKᵀ + Softmax + ×V |
| **RoPE** | 4·n·d | 每对维度 4 次运算 |
| **QK-Norm** | 4·n·d | RMSNorm per-head |

### Transformer 类

| 算子 | FLOPs | 说明 |
|------|-------|------|
| **LayerNorm** | 5·n·d | mean + var + norm + γ + β |
| **RMSNorm** | 4·n·d | mean(x²) + rms + γ |
| **FFN** (d→fd→d) | 4·n·d²·f | 两次线性 |
| **SwiGLU FFN** | 6·n·d²·f | gate + up + mul + down |

---

## F. 常见模型参数量与 FLOPs 对照表

### CNN 模型

| 模型 | 参数量 | FLOPs (224²) | Top-1 | 年代 |
|------|--------|-------------|-------|------|
| AlexNet | 60M | 0.7G | 57.1% | 2012 |
| VGG-16 | 138M | 15G | 71.6% | 2014 |
| ResNet-18 | 12M | 1.8G | 69.8% | 2015 |
| ResNet-34 | 22M | 3.7G | 73.3% | 2015 |
| ResNet-50 | 26M | 4.1G | 76.1% | 2015 |
| ResNet-101 | 45M | 7.8G | 77.4% | 2015 |
| MobileNetV2 | 3.4M | 0.3G | 71.8% | 2018 |
| MobileNetV3-S | 2.5M | 0.06G | 67.4% | 2019 |
| EfficientNet-B0 | 5.3M | 0.4G | 77.1% | 2019 |

### Vision Transformer

| 模型 | 参数量 | FLOPs (224²) | Top-1 |
|------|--------|-------------|-------|
| ViT-Tiny | 5M | 1.3G | — |
| ViT-Small | 22M | 4.6G | 81.4% |
| ViT-Base/16 | 86M | 17G | 77.9% |
| ViT-Large/16 | 307M | 61G | 82.5% |
| Swin-T | 29M | 4.5G | 81.3% |
| Swin-B | 88M | 15G | 83.5% |

### LLM

| 模型 | 参数量 | FLOPs/token (prefill) | FLOPs/token (decode) |
|------|--------|----------------------|---------------------|
| LLaMA-7B | 7B | ~14G | ~14G |
| LLaMA-13B | 13B | ~26G | ~26G |
| LLaMA-70B | 70B | ~140G | ~140G |
| Qwen3-0.6B | 0.6B | ~1.7G | ~1.7G |
| Qwen3-8B | 8B | ~16G | ~16G |

> **Prefill vs Decode**：
> - Prefill（首 token）：所有 token 并行计算，FLOPs 主导
> - Decode（后续 token）：每次只算 1 个 token，内存带宽主导
> - LLM 推理中 decode 阶段通常占总时间的 70-90%

---

## G. ncnn Param 文件格式完整规范

### 文件头

```
7767517                    ← 魔数（固定）
<layer_count> <blob_count>  ← 网络层数和数据节点数
```

### 每行格式

```
<层类型> <in_count> <out_count>
  <输入blob1> <输入blob2> ... → <输出blob1> <输出blob2> ...
  <param_idx>=<param_value> <param_idx>=<param_value> ...
```

### 常用层 param 索引

#### Convolution (Conv2D)

| 索引 | 含义 | 示例 |
|------|------|------|
| 0 | kernel_w | 3 |
| 1 | kernel_h | 3 |
| 2 | stride_w | 1 |
| 3 | stride_h | 1 |
| 4 | pad_left | 1 |
| 5 | pad_right | 1 |
| 6 | dilation_w | 1 |
| 7 | dilation_h | 1 |
| 8 | int8_scale_term | 0 (否) / 101 (是) |
| 9 | num_output | 64 |
| 10 | bias_term | 0 (否) / 1 (是) |
| 11 | weight_data_size | C_out×C_in×k×k |
| 18 | activation_type | 0=无 1=ReLU 2=LeakyReLU 6=HardSwish |
| 20 | pad_value | 0 (默认) |

#### InnerProduct (全连接)

| 索引 | 含义 | 示例 |
|------|------|------|
| 0 | num_output | 1024 |
| 1 | bias_term | 0/1 |
| 2 | weight_data_size | out×in |
| 8 | int8_scale_term | 0/1 |

#### BatchNorm

| 索引 | 含义 | 示例 |
|------|------|------|
| 0 | channels | 64 |
| bins 文件: | mean[C], var[C], slope[C], bias[C] |

#### RMSNorm

| 索引 | 含义 | 示例 |
|------|------|------|
| 0 | affine_size | 1024 |
| 1 | eps | 1e-6 |

#### SDPA

| 索引 | 含义 | 示例 |
|------|------|------|
| 0 | embed_dim | 1024 |
| 1 | num_heads | 16 |
| 2 | key_dim | 128 |
| 3 | value_dim | 128 |
| 4 | mask | 0=无 1=causal 2=自定义 |
| 5 | scale | 1/√dₖ |
| 18 | int8_scale_term | 0/1 |

### Bin 文件权重顺序

```
Convolution:
  weight_data [C_out×C_in×k×k]    ← 如果有 bias
  bias_data [C_out]

BatchNorm:
  mean_data [C]
  var_data [C]
  slope_data [C]
  bias_data [C]

RMSNorm:
  weight_data [affine_size]
  bias_data 不存在

Quantize:
  scale_data [scale_data_size]

Dequantize:
  scale_data [scale_data_size]
  bias_data [bias_data_size]       ← 可选

Requantize:
  scale_in_data [scale_in_size]
  scale_out_data [scale_out_size]
  bias_data [bias_data_size]       ← 可选
```

---

## H. 梯度推导手册

> 反向传播时，每个算子的梯度 = 后一层的梯度 × 本算子的导数。

### 激活函数梯度

| 激活函数 | 前向 | 梯度 (∂output/∂input) |
|---------|------|---------------------|
| **ReLU** | max(0,x) | 1 if x>0 else 0 |
| **LeakyReLU** | x if x>0 else αx | 1 if x>0 else α |
| **Sigmoid** | 1/(1+e⁻ˣ) | σ(x) · (1 − σ(x)) |
| **Tanh** | (eˣ−e⁻ˣ)/(eˣ+e⁻ˣ) | 1 − tanh²(x) |
| **GELU** | x·Φ(x) | Φ(x) + x·φ(x) |
| **Swish** | x·σ(x) | σ(x) + x·σ(x)·(1−σ(x)) |
| **Softplus** | ln(1+eˣ) | σ(x) |
| **Mish** | x·tanh(ln(1+eˣ)) | tanh(sp) + x·sech²(sp)·σ(x) |
| **HardSwish** | x·clip(x+3,0,6)/6 | clip(x+3,0,6)/6 + x·1/6 (if −3<x<3) |
| **Softmax** | eˣⁱ/Σeˣʲ | Sᵢ·(δᵢⱼ − Sⱼ) |

### 矩阵运算梯度

| 运算 | 前向 | ∂L/∂A | ∂L/∂B |
|------|------|-------|-------|
| **MatMul** C=A×B | [M,N]=[M,K]×[K,N] | ∂L/∂C × Bᵀ | Aᵀ × ∂L/∂C |
| **Add** C=A+B | [N]=[N]+[N] | ∂L/∂C | ∂L/∂C |
| **Mul** C=A⊙B | [N]=[N]⊙[N] | ∂L/∂C ⊙ B | A ⊙ ∂L/∂C |
| **Transpose** B=Aᵀ | [N,M]=[M,N] | (∂L/∂B)ᵀ | — |

### 归一化层梯度

| 归一化 | ∂L/∂γ | ∂L/∂β | ∂L/∂x |
|--------|-------|-------|-------|
| **BatchNorm** | Σ ∂L/∂y ⊙ x̂ | Σ ∂L/∂y | ... (见下文) |
| **LayerNorm** | Σ ∂L/∂y ⊙ x̂ | Σ ∂L/∂y | ... (类似 BN) |
| **RMSNorm** | Σ ∂L/∂y ⊙ x/rms | 不存在 | γ·(∂L/∂y/rms − x·Σ(x·∂L/∂y)/d·rms³) |

> **BatchNorm 梯度直觉**：γ 的梯度 = 归一化后值的加权梯度之和。β 的梯度 = 梯度的直接求和。x 的梯度需要同时考虑均值和方差的影响。

### CrossEntropy + Softmax 的简化梯度

```
Softmax:  Sᵢ = eˣⁱ / Σⱼ eˣʲ
CE Loss:  L = −Σ yᵢ · ln(Sᵢ)

链式法则: ∂L/∂xᵢ = Σⱼ (∂L/∂Sⱼ · ∂Sⱼ/∂xᵢ)

神奇的结果: ∂L/∂xᵢ = Sᵢ − yᵢ
                    ↑         ↑
                预测概率   真实标签

→ Softmax + CE 的梯度 = 预测概率 − 真实标签
→ 形式极其简洁，无需中间变量！
```

### RoPE 梯度

```
前向: (x₂ᵢ, x₂ᵢ₊₁) 旋转 θ 角
梯度: 旋转矩阵的逆 = 旋转 −θ 角
     ∂L/∂x = RoPE⁻¹(∂L/∂y, −θ)

→ RoPE 梯度 = 用相反角度旋转梯度
→ 正交变换，梯度不会放大或缩小
```