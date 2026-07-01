# Module 1: 基石算子——逐元素的世界

> 深度学习的所有复杂操作，都可以追溯到对单个数值的简单变换。
> 本模块从最基础的逐元素操作开始，建立算子执行代价的直觉。

---

## 📋 学习目标

完成本模块后，你应该能：

- [ ] 说出 BinaryOp 的 7 种基本操作及其在 LLM 中的应用
- [ ] 理解广播 (Broadcasting) 的规则，能判断两个张量能否广播
- [ ] 画出 ReLU/GELU/Swish 的函数图像，解释它们的差异
- [ ] 对比 Sigmoid 和 Softmax 的适用场景
- [ ] 解释 BatchNorm / LayerNorm / RMSNorm 的归一化维度差异
- [ ] 用手算一个 BatchNorm 融合到 Conv 的数学推导
- [ ] 判断一个算子是 memory-bound 还是 compute-bound
- [ ] 解释 SIMD 为什么能加速逐元素算子

---

## 1. 数学运算：加减乘除的魔法

### 1.1 二元运算 (BinaryOp)

两个张量之间的逐元素运算，是所有复杂运算的**原子操作**。

| 操作 | 公式 | 典型用途 |
|------|------|---------|
| **Add** | a + b | 残差连接、偏置加法 |
| **Sub** | a − b | 残差计算 |
| **Mul** | a × b | 门控（SwiGLU）、注意力权重×Value |
| **Div** | a / b | 归一化、缩放 |
| **Pow** | a^b | 幂运算、RoPE 频率 |
| **Max** | max(a, b) | 最大池化的原子操作 |
| **Min** | min(a, b) | 截断 |

#### 广播 (Broadcasting) 📖

> 当两个张量形状不完全一致时，框架会自动"扩展"较小的张量。

**核心规则**：从**最后一个维度**开始向前比较，每个维度要么**相等**，要么**其中一个为 1**。

```
张量 A: [B, C, 1, 1]    ← C 个标量
张量 B: [B, 1, H, W]    ← 一张 H×W 的图
结果:   [B, C, H, W]    ← 所有维度都扩展

逐维度比较:
  维度 0: B vs B   → 相等，保留
  维度 1: C vs 1   → 1 扩展为 C  ← A 的每个标量被"涂满"B 的整个空间
  维度 2: 1 vs H   → 1 扩展为 H
  维度 3: 1 vs W   → 1 扩展为 W
```

**不能广播的情况**：`[3, 4] + [2, 4]` → ❌ 维度 0 不匹配 (3 vs 2，都不为 1)

> 💡 **Mul 是门控的基石**：从 LSTM 的遗忘门到 SwiGLU，所有"选择性传递信息"的机制都建立在逐元素乘法之上——一个操作数作为"门控信号"（0~1 范围），另一个作为"被控信息"。这是理解 GLU 家族的前提。

### 1.2 一元运算 (UnaryOp)

| 操作 | 公式 | 典型用途 |
|------|------|---------|
| **Abs** | \|x\| | L1 正则化 |
| **Neg** | −x | 梯度反转 |
| **Sqrt** | √x | RMSNorm、L2 范数 |
| **Exp** | eˣ | Softmax、位置编码 |
| **Log** | ln(x) | 损失函数、信息熵 |
| **Floor/Ceil/Round** | ⌊x⌋ / ⌈x⌉ / round(x) | 量化取整 |
| **Sign** | sign(x) | 梯度截断 |

### 1.3 归约运算 (Reduction)

沿指定维度将多个值聚合为一个：

| 操作 | 公式 | 输出形状（归约 dim=1, 输入 [B,C,H,W]） |
|------|------|--------------------------------------|
| **Sum** | Σxᵢ | [B, H, W] |
| **Mean** | (Σxᵢ)/n | [B, H, W] |
| **Max** | max(xᵢ) | [B, H, W] |
| **L2Norm** | √(Σxᵢ²) | [B, H, W] |

> 💡 GlobalAvgPool 本质就是 Mean 归约。Softmax 的分母是 Sum 归约。归约是"从多到少"的核心操作。

### 1.4 矩阵乘法 (MatMul / GEMM / InnerProduct)

这是神经网络**参数化的核心**。

```
MatMul:  C = A × B           [M,K] × [K,N] → [M,N]
GEMM:    Y = α·op(A)·op(B) + β·C    (含转置选项和缩放)
InnerProduct: y = W·x + b           (GEMM 的特例，全连接层)
```

> 💡 **三者关系**：InnerProduct ⊂ GEMM ⊂ MatMul。几乎所有带参数的操作最终都落到矩阵乘法上。

### 1.5 EinSum (爱因斯坦求和约定) 🔬

EinSum 用简洁字符串表达任意张量缩并：

```python
# 矩阵乘法
torch.einsum('ij,jk->ik', A, B)

# 注意力 Q·K^T: [B,H,Q,D] × [B,H,K,D] → [B,H,Q,K]
torch.einsum('bhqd,bhkd->bhqk', Q, K)

# 逐通道乘 (SE Block)
torch.einsum('bc,bchw->bchw', scale, x)
```

**规则**：消去重复字母 → 保留未出现字母

| 模式 | 含义 |
|------|------|
| `'ij,jk->ik'` | 消去 j (矩阵乘法) |
| `'i->'` | 全归约 (Sum) |
| `'ij->ji'` | 转置 |
| `'i,i->i'` | 逐元素乘 |

---

## 2. 激活函数：注入非线性

> 没有激活函数，100 层线性变换 = 1 层线性变换。激活函数让网络有能力拟合非线性关系。

### 2.1 激活函数进化路线 📖

```
ReLU (2012)          → 解决梯度消失
  ↓
LeakyReLU / PReLU    → 解决死亡 ReLU
  ↓
ELU / SELU           → 输出零均值，自归一化
  ↓
GELU / Swish (2017-) → 平滑门控，负值非零
  ↓
HardSwish (2019)     → 移动端优化
  ↓
SwiGLU (2022-)       → 显式门控，现代 LLM 标准
```

### 2.2 核心激活函数对比

#### ReLU — 最经典

```
公式:    ReLU(x) = max(0, x)
梯度:    x>0 → 1;  x≤0 → 0
优点:    计算极快，只需一次 max 操作
缺点:    死亡 ReLU——负值梯度恒为 0，神经元永久失活
参数量:  0
代表:    ResNet, VGG
```

#### GELU — BERT 的选择

```
公式:    GELU(x) = x · Φ(x)   (Φ 是标准正态 CDF)
近似:    ≈ 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))
特点:    ReLU 的平滑版——0 附近有非零梯度，负值不严格归零
代表:    BERT, GPT-2/3/4, ViT
```

#### Swish/SiLU — 现代 LLM 标配

```
公式:    Swish(x) = x · σ(x) = x / (1 + e⁻ˣ)
特点:    x 同时充当"信息"和"门控信号"
         x→+∞: Swish→x (原样通过)
         x≈0:  Swish≈0.5x (部分抑制)
         x→-∞: Swish→0 (渐进归零)
代表:    LLaMA, Qwen, Mistral
```

#### SwiGLU — 门控 MLP 的标准 🔬

```
公式:    SwiGLU(x) = Swish(xW_gate) ⊙ (xW_up) · W_down
         = SiLU(xW₁) ⊙ (xW₂) · W₃

为什么中间维度是 3×hidden 而非 4×?
  标准 FFN: 2 个权重矩阵, 中间维度 4×
  SwiGLU:   3 个权重矩阵, 中间维度 3×  (保持总参数量不变)
  参数量: 2×4 = 3×3 = 各约 12×hidden²

直觉: gate 分支通过 Swish 产生 0~1 的软门控信号，
      与 up 分支逐元素相乘，决定每个维度"放行多少"
```

### 2.3 Softmax：从 logits 到概率

```
公式:    softmax(xᵢ) = exp(xᵢ) / Σⱼ exp(xⱼ)
输出:    每个值 ∈ (0,1)，所有值之和 = 1
梯度:    ∂softmax(i)/∂xⱼ = softmax(i) · (δᵢⱼ − softmax(j))
```

**数值稳定实现**：
```python
x_max = max(x)              # 减最大值防止 exp 溢出
exp_x = exp(x - x_max)
output = exp_x / sum(exp_x)
```

**温度参数 Temperature**：`softmax(x / T)`
- T > 1：分布更平滑（知识蒸馏）
- T < 1：分布更尖锐（更确定）

**Softmax vs Sigmoid**：

| 对比 | Sigmoid | Softmax |
|------|---------|---------|
| 适用 | 逐元素独立 | 整个向量联合 |
| 多类 | 各类独立（可多个激活） | 概率和为 1（互斥选择） |
| 用途 | 二分类/门控 | 多分类/注意力权重 |

### 2.4 激活函数选择指南

| 场景 | 推荐 | 原因 |
|------|------|------|
| CNN 分类 | ReLU / GELU | 简单有效 |
| Transformer NLP | GELU (BERT 系) | 平滑门控 |
| 现代 LLM MLP | SwiGLU | 门控+平滑，当前最优 |
| 移动端部署 | HardSwish | 计算快，无 exp |
| 门控 (LSTM/GRU) | Sigmoid | 输出 (0,1) 天然适合 |

---

## 3. 归一化：稳定数值范围

> 训练深层网络时，每层输出分布会不断漂移。归一化把数值拉回稳定范围。

### 3.1 为什么需要归一化？ 📖

```
Internal Covariate Shift:
  → 前面层参数更新后，后面层的输入分布就变了
  → 就像射击训练中靶子不断移动
  → 需要更小的学习率 + 更谨慎的初始化

归一化 ≈ 固定靶子位置
  → 每层输入被拉到均值 0 方差 1 附近
  → 深层网络可以稳定学习
```

### 3.2 五种归一化对比

```
           N (batch)  C (channel)  H/W (spatial)

BatchNorm: ─────────  ✗ 归一化    ✗ 归一化     ← 跨 batch 统计
LayerNorm: ✗ 独立     ─────────   ─────────     ← 跨 feature 统计
RMSNorm:   ✗ 独立     ─────────   ─────────     ← LayerNorm 简化版
GroupNorm: ✗ 独立     分组内      ─────────     ← BN 和 LN 的折中
InstanceN: ✗ 独立     每通道独立   ─────────     ← 每样本每通道独立
```

### 3.3 BatchNorm — CNN 时代标配

```
公式:
  训练: y = (x − μ_batch) / √(σ²_batch + ε) · γ + β
  推理: y = (x − running_mean) / √(running_var + ε) · γ + β

关键特性:
  - 依赖 batch size，小 batch 效果差
  - 推理时可与前一 Conv 融合 → 零额外开销
  - 融合: W' = γ/σ · W,  b' = γ/σ · (b − μ) + β
```

### 3.4 LayerNorm — Transformer 标配

```
公式:    y = (x − μ_feature) / √(σ²_feature + ε) · γ + β
特点:    不依赖 batch size，训练推理行为一致
        每个样本独立归一化
代表:    BERT, GPT-2, ViT
```

### 3.5 RMSNorm — 现代 LLM 标配 💡

```
公式:    y = x / √(mean(x²) + ε) · γ
特点:    省略均值减法 + 偏置项 → 只有 γ 参数

对比 LayerNorm:
  参数量:   减半 (无 β)
  计算量:   减少 ~30% (省减法 + 偏置加法)
  精度影响: <0.1%

为什么现代 LLM 都选 RMSNorm？
  28 层模型 → 省 30% 归一化计算量 + 50% 归一化参数量
  → 累计效果显著
```

### 3.6 归一化选择指南

| 场景 | 推荐 | 原因 |
|------|------|------|
| CNN (大 batch) | BatchNorm | 批统计稳定 |
| Transformer NLP | LayerNorm / RMSNorm | 不依赖 batch |
| 现代 LLM | **RMSNorm** | 更快更省 |
| 扩散模型/小 batch | GroupNorm | 不依赖 batch |
| 风格迁移 | InstanceNorm | 逐样本逐通道 |

---

## 4. 张量操作：形状与维度的魔法

### 4.1 常用张量操作速查

| 操作 | 功能 | 示例 |
|------|------|------|
| **Reshape** | 改变形状，元素数不变 | `[B,C,H,W] → [B, C·H·W]` |
| **Permute** | 重排维度顺序 | `Permute(0,2,3,1): [B,C,H,W] → [B,H,W,C]` |
| **Concat** | 沿指定维度拼接 | `Concat(dim=1): [B,2,H,W] + [B,4,H,W] → [B,6,H,W]` |
| **Split** | 沿指定维度分割 | `Split(dim=1): [B,6,H,W] → [B,2,H,W] + [B,4,H,W]` |
| **Tile/Repeat** | 沿指定维度复制 | GQA 中 8 KV heads → 16 heads |
| **Gather** | 按索引取元素 | Embedding 查表 |
| **Slice** | 取子张量 | `x[:, 0:10]` |
| **Squeeze** | 删除大小为 1 的维度 | `[B,1,H,W] → [B,H,W]` |
| **Padding** | 边界填充 | 卷积 padding |

> 💡 **Split + Concat 互为逆操作**。在 Transformer Block 中，Split 将特征分为 Q/K/V 三路；Concat 将多个 head 的输出合并。

---

## 5. Loss 函数

| Loss | 公式 | 使用场景 |
|------|------|---------|
| **CrossEntropy** | −Σ y·log(ŷ) | 多分类 → LLM 预测下一个 token |
| **BCE** | −[y·log(p) + (1−y)·log(1−p)] | 二分类/多标签 |
| **MSE** | (1/n)·Σ(ŷ−y)² | 回归任务 |
| **Focal Loss** | −α·(1−pₜ)ᵞ·log(pₜ) | 正负样本不平衡 (目标检测) |
| **CTC Loss** | 变长序列对齐 | 文字识别/语音 |

> 💡 CrossEntropy + Softmax 结合时梯度简化为 `ŷ − y`，不需要手动求导。

---

## 6. 硬件执行模型 🔬

### 6.1 FLOPs vs MACs

```
一个 MAC (Multiply-Accumulate) = 1 次乘法 + 1 次加法 = 2 FLOPs

Conv2D FLOPs:  2 × C_out × C_in × k² × H_out × W_out
Conv2D MACs:      C_out × C_in × k² × H_out × W_out

MatMul FLOPs:  2 × M × N × K
MatMul MACs:      M × N × K
```

> ⚠️ 论文中"FLOPs"有时实际指 MACs。PyTorch 的 thop 库报告的是 MACs×2。

### 6.2 Roofline Model（屋顶线模型）

```
                算力上限 (GFLOPs/s)
                ┃
     计算密集区  ┃          ← MatMul (大矩阵), Prefill
        ╲       ┃
         ╲      ┃
    ─────╲─────┃──────  ← 内存带宽上限
           ╲    ┃
  内存密集区 ╲   ┃          ← Add/ReLU/LayerNorm, Decode
              ╲ ┃
              ────────→ 计算强度 (FLOPs/Byte)
```

**计算强度 = FLOPs / 字节访问量**

| 算子 | 计算强度 | 瓶颈 |
|------|---------|------|
| MatMul (大矩阵) | 高 (100+) | 计算密集 |
| Conv2D (3×3) | 中 (10-100) | 视情况 |
| DepthwiseConv | 低 (1-10) | 内存密集 |
| Add/ReLU/Clip | 极低 (<1) | 内存密集 |
| SDPA Decode | 极低 (<1) | 内存密集 |

### 6.3 SIMD 并行

| 架构 | 指令集 | FP32 宽度 |
|------|--------|----------|
| x86 | AVX2 | 8 |
| x86 | AVX-512 | 16 |
| ARM | NEON | 4 |
| GPU CUDA | TensorCore | 256+ (矩阵) |

> ncnn 的做法：每个算子有 CPU 基准版 + ARM NEON 优化版 + x86 AVX 优化版 + Vulkan 版。

---

## 7. 优化器与学习率 (训练视角)

### 优化器对比

| 优化器 | 收敛 | 泛化 | 调参 | 推荐 |
|--------|:----:|:----:|:----:|------|
| SGD | 慢 | 最好 | 高 | CNN 精细调参 |
| SGD+Momentum | 中 | 好 | 中 | 传统视觉 |
| Adam | 快 | 中 | 低 | 快速原型 |
| **AdamW** | 快 | 好 | 低 | **LLM 训练推荐** |

> AdamW 的关键改进：权重衰减从梯度中解耦——`w = w − lr·λ·w` 独立于自适应学习率。

### 学习率调度

```
现代 LLM 训练标配: Warmup + Cosine

lr
 ^        ╱╲
 │       ╱  ╲
 │      ╱    ╲
 │     ╱      ╲
 │    ╱        ╲
 └───╱──────────╲───→ steps
    warmup  cosine decay

Warmup: 前 2000-5000 steps 从小 lr 线性增长 (防梯度爆炸)
Cosine: 之后按余弦曲线平滑降到 lr_min (精细收敛)
```

---

## 🛠️ 动手练习

### 练习 1：广播判断
判断以下能否广播，如果能，输出 shape 是什么？
```
a) [3, 1, 4] + [1, 5, 4]  → ?
b) [2, 3, 4] + [3, 4]     → ?
c) [2, 3] + [2, 4]         → ?
```

### 练习 2：激活函数手绘
在一张纸上画出 ReLU, LeakyReLU(α=0.1), GELU, Swish 的曲线。标注关键特征。

### 练习 3：Softmax 手算
对 `[2.0, 1.0, 0.1]` 计算：
- 标准 Softmax
- T=0.5 的 Softmax
- T=2.0 的 Softmax
观察温度对分布的影响。

### 练习 4：BN 融合推导
给定 Conv 的 W=[[1,2],[3,4]], b=[0.5, -0.3]，BN 的 γ=[2,1], β=[0,0], μ=[1,2], σ²=[4,1]。
计算融合后的 W' 和 b'。

---

## 📚 延伸阅读

- [deep_learning_operators_original.md](../deep_learning_operators_original.md) §1.1-1.8
- [02_compute_operator_layer.md](../../docs/02_compute_operator_layer.md)
- 激活函数可视化: [desmos.com](https://www.desmos.com/calculator)

---

*下一模块: [Module 2: 构建算子——从单元素到空间](../module-02-building-blocks/README.md)*
