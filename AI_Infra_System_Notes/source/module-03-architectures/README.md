# Module 3: 组合结构——单打独斗不如团队协作

> 单个算子就像单个工人，能力有限。但当他们组成流水线——CNN Block 用残差连接让 100+ 层梯度顺畅流淌，Transformer Block 用注意力让每个词都能看到整段上下文——就产生了令人震撼的能力。

---

## 学习目标

- [ ] 能手写残差连接的梯度公式，解释"为什么 +1 让 100 层网络可训练"
- [ ] 能画出 Transformer Block 的完整算子序列（含每步 shape）
- [ ] 能对比 Pre-Norm vs Post-Norm，用梯度流分析为什么现代 LLM 全选 Pre-Norm
- [ ] 能计算 SwiGLU 的参数量，解释"3×hidden"的设计逻辑
- [ ] 能说出 Qwen3-0.6B 的 1017 个算子中，哪些类型占比最高
- [ ] 理解 QK-Norm 为什么能防止长上下文注意力崩塌

---

## 1. 残差连接——深度学习史上最重要的 "+1"

### 1.1 在 ResNet 之前：深层网络的"退化"之谜

2015 年之前，人们发现一个诡异现象：56 层网络在**训练集**上的误差比 20 层还高。

```
这不能用"过拟合"解释——过拟合是训练误差低、测试误差高。
"训练误差也更高"意味着——网络根本学不动。
```

原因：梯度在反向传播时每经过一层就要乘一个雅可比矩阵。如果大多数雅可比的特征值 < 1，连乘 56 次后梯度指数衰减到几乎为 0。浅层参数收不到任何学习信号。

### 1.2 残差的解法——不是"学整个变换"，而是"学修正量"

```
标准层:  x_out = F(x_in)         "从头学映射"
残差层:  x_out = F(x_in) + x_in   "学修正量 + 直接通路"

梯度对比:
  标准: dL/dx_in = dL/dx_out × dF/dx_in
        └─ 如果 dF/dx_in ≈ 0 → 梯度消失！
        
  残差: dL/dx_in = dL/dx_out × (dF/dx_in + 1)
        └─ 即使 dF/dx_in ≈ 0，也有个 +1 保底！
        └─ 梯度至少是 dL/dx_out × 1 → 不会完全消失
```

**为什么这个 "+1" 如此关键？**

```
假设 100 层残差网络，每层的 dF/dx ≈ 0.1（很小，接近梯度消失）:

标准网络: dL/dx₀ = dL/dx₁₀₀ × 0.1^100 ≈ dL/dx₁₀₀ × 10^(-100) → 彻底归零！
残差网络: dL/dx₀ = dL/dx₁₀₀ × (0.1+1)^100 ≈ dL/dx₁₀₀ × 1.1^100
         → 不仅没消失，还有放大效应！（实际上会被 BN/LN 控制住）

直觉: 残差连接为梯度修了一条"高速公路"——
      不论主路的梯度多弱，高速公路永远畅通。
```

### 1.3 Bottleneck——"先压缩再解压"的降维艺术

```
BasicBlock (ResNet-18):     Conv3×3 → Conv3×3 → +x
Bottleneck (ResNet-50+):    Conv1×1(降维) → Conv3×3 → Conv1×1(升维) → +x

为什么先降维？
  Conv3×3 在所有维度都做 → 维度越高越贵 (O(C²k²))
  先用 1×1 把 256 维压到 64 维 → 在 64 维做昂贵的 3×3 → 再用 1×1 升回 256 维

参数对比:
  两个 3×3 在 256 维: 2 × 256² × 9 = 1,179,648
  Bottleneck:         256×64 + 64²×9 + 64×256 = 16,384 + 36,864 + 16,384 = 69,632
  → 省了 94% 的参数！
```

---

## 2. Transformer Block——现代 LLM 的核心构建块

### 2.1 完整解剖 (Qwen3-0.6B Layer 0，含 shape)

```
x_in [seq_len, 1024]
  │
  ├─ RMSNorm(1024) → 归一化到稳定范围
  │   └─ Split → {Q分支, K分支, V分支}
  │
  ├── Q: Gemm(1024→2048) → Reshape→[128,16,seq] → RMSNorm(128) → Permute → RoPE
  ├── K: Gemm(1024→1024) → Reshape→[128,8,seq]  → RMSNorm(128) → Permute → RoPE
  │                                                → ExpandDims→Tile→[128,seq,16] (GQA!)
  ├── V: Gemm(1024→1024) → Reshape→[128,8,seq]  → Permute
  │                                                → ExpandDims→Tile→[128,seq,16] (GQA!)
  │
  ├── SDPA(Q[16,seq,128], K[16,seq,128], V[16,seq,128], mask, past_KV)
  │     scale=1/√128≈0.088, 读写 KV Cache
  │
  ├── O Proj: Permute→Reshape→[2048,seq] → Gemm(2048→1024)
  │
  ├─── + x_in (残差 1) → x_attn
  │
  ├─ RMSNorm(1024) → Split → {gate, up}
  │
  ├── SwiGLU:
  │     gate: Gemm(1024→3072) → Swish → ┐
  │     up:   Gemm(1024→3072) ─────────→ Mul(gate⊙up) → Gemm(3072→1024)
  │
  └─── + x_attn (残差 2) → x_out

单层算子数: ~35
28层总计:   ~1017 个 ncnn 算子

算力分布 (单层):
  Q/K/V/O 投影: 4 × 1024×1024 = 4.2M 参数, ~8.4M FLOPs
  gate/up/down:  3 × 1024×3072 = 9.4M 参数, ~18.9M FLOPs
  SDPA:          ~5M FLOPs (decode) / ~N²×128 FLOPs (prefill)
  RMSNorm ×3:    ~6K 参数, ~6K FLOPs
```

### 2.2 Pre-Norm vs Post-Norm——为什么现代 LLM 全选 Pre-Norm？

```
Pre-Norm:  x_out = x + F(Norm(x))       ← 先归一化输入
Post-Norm: x_out = Norm(x + F(x))       ← 先做变换再归一化

梯度流分析:

Pre-Norm:
  d(x_out)/d(x) = I + dF/dx × d(Norm)/dx
  → 残差路径上恒有 I（单位矩阵）→ 梯度直达输入层

Post-Norm:
  d(x_out)/d(x) = d(Norm)/d(x+F(x)) × (I + dF/dx)
  → 梯度要通过 Norm 的导数 → 有额外衰减
  → 在深层时，多个 Norm 的导数连乘 → 梯度消失风险

实验证据: Post-Norm 在深层（>12层）时需要精细的 warmup；
          Pre-Norm 不需要 warmup，训练天然稳定。

结论: Pre-Norm 让残差路径上干干净净——没有归一化挡路，梯度想怎么流就怎么流。
```

### 2.3 QK-Norm——Qwen3 的独特贡献

```
位置: Q 和 K 投影后、RoPE 前
内容: 对每个 head 独立做 RMSNorm(affine_size=128)

为什么需要？
  长上下文中，Q 和 K 都没有归一化 → 点积方差随 d_k 增大
  → softmax 趋向 one-hot（所有注意力集中在一个 token）
  → 模型"盯死"一个位置，忽视 99% 的上下文

有了 QK-Norm:
  Q 和 K 被约束在稳定范围 → 点积方差可控
  → softmax 保持合理分布 → 长上下文仍能关注多个关键位置

哪些模型有？
  Qwen3: 有 ✅
  Qwen2.5/LLaMA/Mistral: 无 ❌
  Gemma 2: 有 ✅
```

### 2.4 SwiGLU 参数量推导

```
标准 ReLU-FFN:
  W1: hidden × 4×hidden = 1024 × 4096
  W2: 4×hidden × hidden = 4096 × 1024
  总计: 2 × 4 × hidden² = 8 × hidden²

SwiGLU:
  gate: hidden × 3×hidden = 1024 × 3072
  up:   hidden × 3×hidden = 1024 × 3072
  down: 3×hidden × hidden = 3072 × 1024
  总计: 3 × 3 × hidden² = 9 × hidden²

对比: 8 vs 9，参数量接近（多 ~12.5%），但多了一层门控
→ 用极小的参数代价换取显著的质量提升
→ 现代 LLM 的共识选择
```

### 2.5 Transformer 进化编年史

```
2017: 原始 Transformer  Post-LN + MHA + ReLU-FFN + 绝对位置编码
2018: BERT/GPT-2        Post-LN + MHA + GELU-FFN + 可学习PE
2020: GPT-3             Post-LN + MHA + Sparse Attention
2022: PaLM              Pre-LN + MQA + SwiGLU + RoPE      ← 范式转折点
2023: LLaMA 1/2         Pre-RMSNorm + GQA + SwiGLU + RoPE  ← 现代范式确立
2024: Qwen3             + QK-Norm + 长上下文优化

每一步在解决什么?
  Post→Pre:      深层训练稳定性
  MHA→GQA:       KV Cache 减半
  ReLU→SwiGLU:   MLP 表达能力
  可学习PE→RoPE:  外推能力
  +QK-Norm:      长上下文注意力质量
```

---

## 3. CNN Block、RNN 单元、编解码架构（速查）

| 结构 | 核心算子序列 | 关键创新 | 代表模型 |
|------|------------|---------|---------|
| Conv-BN-ReLU | Conv→BN→ReLU | 标准 CNN 构建块 | VGG, ResNet |
| ResNet Bottleneck | 1×1Conv→3×3Conv→1×1Conv | 降维→卷积→升维 | ResNet-50+ |
| SE Block | GAP→FC→ReLU→FC→Sigmoid→Scale | 通道注意力 | SENet |
| U-Net | 下采样→...→上采样 + Skip | 保留空间细节 | 分割/Diffusion |
| FPN | 多尺度特征融合 | 金字塔检测 | Faster R-CNN |

---

## 动手练习

1. **残差梯度实验**：对有残差 `y = x + F(x)` 和无残差 `y = F(x)`，设 `dL/dy=1, dF/dx=0.01`，分别计算 `dL/dx`。

2. **Transformer Block 画图**：在纸上画出完整算子序列，标注每个环节的 shape（参照 §2.1）。

3. **参数量手算**：Qwen3-1.7B (hidden=2048, intermediate=6144, layers=28, heads=16, kv_heads=8, vocab=151936)。计算 Attention 参数量、MLP 参数量、Embedding 参数量、总参数量。

4. **Pre-Norm 证明**：写出 Pre-Norm 和 Post-Norm 的 `∂x_out/∂x_in` 表达式，对比残差路径上的梯度衰减。

---

*下一模块: [Module 4: 进化架构](../module-04-advanced/README.md)*
