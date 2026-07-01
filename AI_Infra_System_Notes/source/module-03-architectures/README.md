# Module 3: 组合结构——从算子到架构块

> 单个算子能力有限，组合起来才构成强大的模型。本模块拆解 CNN Block、RNN 单元、Transformer Block 和编解码架构。

---

## 📋 学习目标

- [ ] 能画出 ResNet BasicBlock 和 Bottleneck 的完整算子序列
- [ ] 能解释残差连接为什么能训练 100+ 层网络（梯度的 "+1"）
- [ ] 能画出完整 Transformer Block 的算子图（含每步的 shape）
- [ ] 能写出 Pre-Norm vs Post-Norm 的区别
- [ ] 能分析 SwiGLU MLP 的参数量和计算量
- [ ] 能理解 Qwen3-0.6B 的 28 层 decoder 中 1017 个算子的组成

---

## 1. CNN 基础块

### 1.1 Conv-BN-ReLU (CBR)

```
x → Conv2D → BatchNorm → ReLU → x_out

几乎所有 CNN 的基本构建单元。
变体: CBR-Swish, CBR-LeakyReLU, CBR-HardSwish
```

### 1.2 ResNet 残差块 📖

```
BasicBlock (ResNet-18/34):
  x ──────────────────────┐
    │                      │
    v Conv3×3-BN-ReLU      │
    v Conv3×3-BN           │
    │                      │
    v Add + ReLU ←─────────┘

Bottleneck (ResNet-50+):
  x ──────────────────────────┐
    │                          │
    v Conv1×1-BN-ReLU (降维)   │
    v Conv3×3-BN-ReLU           │
    v Conv1×1-BN (升维)        │
    │                          │
    v Add + ReLU ←─────────────┘
```

#### 残差连接的数学本质 💡

```
标准层:  x_out = F(x_in)
         梯度: dL/dx_in = dL/dx_out × dF/dx_in

残差层:  x_out = F(x_in) + x_in
         梯度: dL/dx_in = dL/dx_out × (dF/dx_in + 1)
                                            ↑
                                    即使 dF/dx_in ≈ 0，
                                    梯度至少为 dL/dx_out × 1，
                                    不会消失！

这就是 100+ 层网络可训练的原因。
```

### 1.3 SE Block (通道注意力)

```
x → GlobalAvgPool → FC-ReLU → FC-Sigmoid → Scale → x_out
     [B,C,H,W]→[B,C]   C→C/r      C/r→C   逐通道乘
```

### 1.4 Inception 模块

```
x
 ├── Conv1×1 ──────────────────────┐
 ├── Conv1×1 → Conv3×3 ────────────┤
 ├── Conv1×1 → Conv5×5 ────────────┼── Concat → out
 └── MaxPool3×3 → Conv1×1 ─────────┘

思想: 多尺度并行提取，让网络自己选择最佳核大小
```

---

## 2. RNN 单元

### LSTM — 四门控的经典

```
遗忘门: f = σ(Wf·[h,x] + bf)    丢弃什么
输入门: i = σ(Wi·[h,x] + bi)    接收什么
候选:   c̃ = tanh(Wc·[h,x] + bc)  新内容
细胞态: c = f⊙c_old + i⊙c̃        长期记忆更新
输出门: o = σ(Wo·[h,x] + bo)    输出什么
隐藏态: h = o⊙tanh(c)            短期输出

核心设计: 细胞态 c 作为"高速公路"——
         只有逐元素乘加，梯度可以无损流过
```

### GRU — 简化但有效

```
重置门: r = σ(Wr·[h,x])          控制历史信息的利用率
更新门: z = σ(Wz·[h,x])          合并遗忘+输入门
候选:   h̃ = tanh(Wh·[r⊙h, x])    新内容
输出:   h = (1−z)⊙h + z⊙h̃        线性插值更新

vs LSTM: 2 门 vs 4 门, 参数量 2/3, 无独立细胞态
```

---

## 3. Transformer Block 🔬 — 绝对重点

### 3.1 完整算子序列 (Qwen3-0.6B Layer 0)

```
x_in [seq_len, 1024]
  │
  ├─ RMSNorm (affine_size=1024) → x_norm
  │   └─ Split → Q/K/V 三路
  │
  ├── Attention 分支:
  │   Q: Gemm(1024→2048) → Reshape(128,16,seq) → QK-Norm → Permute → RoPE
  │   K: Gemm(1024→1024) → Reshape(128,8,seq)  → QK-Norm → Permute → RoPE
  │   V: Gemm(1024→1024) → Reshape(128,8,seq)  → Permute
  │   K: ExpandDims + Tile(8→16 heads, GQA)
  │   V: ExpandDims + Tile(8→16 heads, GQA)
  │   SDPA(Q, K_exp, V_exp, mask, past_K, past_V)
  │   → Permute + Reshape(2048,seq) → Gemm(O:2048→1024)
  │
  ├─── Add (残差 1) ←───────────────┘
  │
  ├─ RMSNorm (pre-MLP)
  │   └─ Split → gate, up 两路
  │
  ├── SwiGLU MLP:
  │   gate: Gemm(1024→3072) → Swish
  │   up:   Gemm(1024→3072)
  │   gate ⊙ up → Gemm(down:3072→1024)
  │
  └─── Add (残差 2) → x_out

单层算子数: ~35
28 层总算子数: ~1017
```

### 3.2 Pre-Norm vs Post-Norm

```
Pre-Norm (Qwen3, LLaMA, 现代 LLM):
  x_out = x + Attention(RMSNorm(x))    ← 归一化在子层前
  → 残差路径无归一化：梯度可以无损流过
  → 训练极稳定，不需要 warm-up

Post-Norm (原始 Transformer, BERT):
  x_out = RMSNorm(x + Attention(x))    ← 归一化在残差后
  → 输出分布更稳定，但深层训练困难
```

### 3.3 SwiGLU MLP 的参数量

```
Qwen3-0.6B: hidden=1024, intermediate=3072 (3×)

gate: 1024 × 3072 = 3.15M
up:   1024 × 3072 = 3.15M
down: 3072 × 1024 = 3.15M
────────────────────────
MLP 总计: ~9.45M / 层

对比标准 FFN (4×, 2 矩阵):
  W1: 1024 × 4096 = 4.19M
  W2: 4096 × 1024 = 4.19M
  ──────────────────────
  FFN 总计: ~8.39M / 层

SwiGLU 的 9.45M vs FFN 的 8.39M → 相近，但精度好得多
```

### 3.4 Transformer 进化路线

```
BERT/GPT-2 (2018-19):  Post-LN + MHA + GELU-FFN
GPT-3 (2020):          Post-LN + MHA + Sparse Attention
PaLM (2022):           Pre-LN + MQA + SwiGLU
LLaMA (2023):          Pre-RMSNorm + GQA + SwiGLU + RoPE
Qwen3 (2024):          Pre-RMSNorm + GQA + SwiGLU + RoPE + QK-Norm
```

---

## 4. 编解码架构

### 4.1 U-Net

```
编码器 (下采样)            解码器 (上采样)
Conv-BN-ReLU ×2           Conv-BN-ReLU ×2
MaxPool ──────────────→ TransposedConv / Interp
Conv-BN-ReLU ×2           Conv-BN-ReLU ×2
MaxPool ──────────────→ TransposedConv / Interp
      │                        ↑
      └── Skip (Concat) ──────┘

Skip connection: 保留精确空间位置信息
```

### 4.2 FPN (特征金字塔)

```
P5 → 1×1 Conv → Upsample ──┐
                            ├ Add → P4'
P4 → 1×1 Conv ──────────────┤
                            │
P3 → 1×1 Conv ──────────────┘

多尺度特征融合，不同层级检测不同大小的目标
```

---

## 5. 模型完整拆解

### 5.1 Qwen3-0.6B 架构参数

```
embed_dim:      1024
num_layers:     28
num_q_heads:    16
num_kv_heads:   8        (GQA)
head_dim:       128
intermediate:   3072      (SwiGLU, 3× hidden)
vocab_size:     151936
max_seq_len:    32768

总参数: ~600M
  每层 Attention:  ~6.2M  (Q+K+V+O proj)
  每层 MLP:        ~9.5M  (gate+up+down)
  每层 Norm:       ~3K    (RMSNorm gamma)
  Embedding:       ~156M  (151936×1024, 与 Projection 共享)
```

### 5.2 参数量分布

```
Embedding (共享):  156M  (26%)
Attention:         174M  (29%)  → 28层 × 6.2M
MLP:               266M  (44%)  → 28层 × 9.5M
Norms:             ~0.1M (<1%)
─────────────────────────────
总计:              ~600M
```

---

## 🛠️ 动手练习

1. **画 Transformer Block**: 在纸上画出 Qwen3 一个 Block 的完整算子序列，标注每个算子的输入输出 shape。

2. **算参数量**: 计算 Qwen3-1.7B (hidden=2048, layers=28, intermediate=6144, heads=16, kv_heads=8) 的总参数量。

3. **残差梯度验证**: 对 `y = x + F(x)`，假设 `dL/dy = 1`，`dF/dx = 0.01`，计算 `dL/dx`。对比 `y = F(x)` 的情况。

4. **Pre-Norm 证明**: 画图说明为什么 Pre-Norm 的梯度可以直达输入层。

---

*下一模块: [Module 4: 进化架构](../module-04-advanced/README.md)*
