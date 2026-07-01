# Module 3: 组合结构——单打独斗不如团队协作

> 单个算子就像单个工人，能力有限。但当他们组成流水线——CNN Block 用残差连接让 100+ 层的梯度顺畅流淌，Transformer Block 用注意力让每个词都能"看到"整段上下文——就产生了令人震撼的能力。本章拆解这些经典"团队组合"。

---

## 📋 学习目标

- [ ] 能画出 ResNet 残差块，解释"为什么 +1 能让 100 层网络可训练"
- [ ] 能画出完整 Transformer Block 的算子序列（含每一步的 shape 变化）
- [ ] 能对比 Pre-Norm vs Post-Norm，说出为什么现代 LLM 全选 Pre-Norm
- [ ] 能计算 SwiGLU MLP 的参数量，解释"3× hidden"的设计逻辑
- [ ] 能说出 Qwen3-0.6B 的 1017 个算子中，哪些类型占比最高

---

## 1. 残差连接——深度学习史上最重要的"+1"

### 1.1 问题：为什么深层网络反而不如浅层？

2015 年之前，人们发现一个反直觉的现象：56 层的网络在训练集上的误差**比 20 层的还高**。不是过拟合（过拟合应该是训练误差低、测试误差高），而是**网络太深了，梯度传不回去**。

### 1.2 残差的魔法——一个"+x"改变了一切

```
标准层:  x_out = F(x_in)
         梯度: dL/dx_in = dL/dx_out × dF/dx_in
         如果 dF/dx_in 很小（梯度消失），dL/dx_in 就几乎为 0

残差层:  x_out = F(x_in) + x_in
         梯度: dL/dx_in = dL/dx_out × (dF/dx_in + 1)
                                          ↑
                                  有了这个 +1！
                                  即使 dF/dx_in ≈ 0，
                                  梯度至少是 dL/dx_out × 1
```

> 💡 **一句话记住**：残差连接在反向传播时，为梯度开了一条 **"高速公路"**——不管 F 的梯度多小，总有一个 +1 保证梯度不会完全消失。这就是为什么 152 层的 ResNet 不仅比 20 层的 VGG 深，还更好训练。

### 1.3 Bottleneck——"先压后扩"的降维艺术

```
BasicBlock (ResNet-18/34):   Conv3×3 → Conv3×3 → +x
Bottleneck (ResNet-50+):     Conv1×1(降维) → Conv3×3 → Conv1×1(升维) → +x

为什么中间要"压"一下？
  Conv1×1(256→64):  256×64=16K 参数
  Conv3×3(64→64):   64×64×9=37K 参数  ← 在低维空间做昂贵操作
  Conv1×1(64→256):  64×256=16K 参数
  
  如果都在 256 维: Conv3×3 需要 256×256×9=590K 参数
  → Bottleneck 省了 10 倍参数！
```

---

## 2. Transformer Block——现代 LLM 的核心构建块 ⭐

### 2.1 完整解剖（Qwen3-0.6B Layer 0）

```
x_in [seq_len, 1024]
  │
  ├─ RMSNorm  → "稳定一下数值"
  │   └─ Split → 分成 Q, K, V 三路
  │
  ├── Attention 分支:
  │   Q: Gemm(1024→2048) → 拆16个头 → QK-Norm → RoPE
  │   K: Gemm(1024→1024) → 拆8个头  → QK-Norm → RoPE → 复制到16个(GQA)
  │   V: Gemm(1024→1024) → 拆8个头  → 复制到16个(GQA)
  │   └─ SDPA(Q, K_exp, V_exp, mask, past_KV) ← 读写 KV Cache
  │   └─ O Proj: Gemm(2048→1024) → 合并多头
  │
  ├─── + x_in (残差1: "跳过 attention，保留原始信息")
  │
  ├─ RMSNorm
  │   └─ Split → gate, up
  │
  ├── SwiGLU MLP:
  │   gate: Gemm(1024→3072) → Swish → ┐
  │   up:   Gemm(1024→3072) ─────────→ Mul(gate⊙up) → Gemm(3072→1024)
  │
  └─── + (残差2: "跳过 MLP，保留 attention 后的状态")

× 28 层 = 1017 个 ncnn 算子
```

### 2.2 Pre-Norm——"先洗澡再出门"

```
Pre-Norm:  x_out = x + F(Norm(x))    ← 先归一化输入，再做变换
Post-Norm: x_out = Norm(x + F(x))    ← 先做变换，再归一化输出

Pre-Norm 为什么是现代 LLM 的共同选择？
  → 残差路径上没有任何非线性变换 → 梯度可以"无损"流过
  → 训练极其稳定，不需要 warm-up
  → Post-Norm 在深层时训练困难，需要精细的学习率预热
```

### 2.3 为什么 Qwen3 有 QK-Norm？

QK-Norm 是 Qwen3 的独特设计——在 Q 和 K 投影后、RoPE 前，额外加一层 RMSNorm。目的是**防止长上下文中注意力崩塌**：

```
没有 QK-Norm: Q·K 的点积方差随序列长度急剧增大
            → softmax 趋向 one-hot（只关注一个 token）
            → 其余 99% 的信息被忽略

有 QK-Norm: Q 和 K 被约束在稳定范围
          → softmax 保持合理分布
          → 长上下文时仍然能关注到多个关键位置
```

---

## 3. 进化史——从 BERT 到 Qwen3

```
2018: BERT/GPT-2    Post-LN + MHA + GELU-FFN
2020: GPT-3         Post-LN + MHA + Sparse Attention
2022: PaLM          Pre-LN + MQA + SwiGLU           ← Pre-Norm 登场
2023: LLaMA         Pre-RMSNorm + GQA + SwiGLU + RoPE  ← 现代范式的确立
2024: Qwen3         + QK-Norm                       ← 长上下文优化

每一次进化都在解决一个具体问题:
  Post→Pre:   训练稳定性
  MHA→GQA:    KV Cache 省一半
  FFN→SwiGLU: MLP 表达能力
  +QK-Norm:   长上下文注意力质量
```

---

## 🛠️ 动手练习

1. **画出 Transformer Block**：在纸上画出完整算子序列，标注每个环节的 shape。

2. **残差梯度实验**：对 `y = x + F(x)` 和 `y = F(x)`，设 `dL/dy=1, dF/dx=0.01`，分别计算 `dL/dx`。

3. **参数量计算**：Qwen3-1.7B (hidden=2048, layers=28, intermediate=6144, heads=16, kv_heads=8)。计算总参数量。

---

*下一模块: [Module 4: 进化架构](../module-04-advanced/README.md)*
