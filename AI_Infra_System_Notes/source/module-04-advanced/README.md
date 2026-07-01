# Module 4: 进化架构——突破 Transformer 的边界

> 经典 Transformer 是基础，但工程师们从未停止改进。本模块覆盖注意力变体、状态空间模型 Mamba（"线性复杂度的 Transformer 替代者"）、混合专家 MoE（"让 8 个专家各司其职"），以及量化与多模态的基础概念。

---

## 📋 学习目标

- [ ] 理解 GQA/MQA 的实现和 KV Cache 收益
- [ ] 理解 RoPE 长上下文外推的三种方法
- [ ] 理解 Mamba 的 selective scan——"选择性记忆"为什么比 Attention 更快
- [ ] 理解 MoE 的 Router + Top-K 路由机制
- [ ] 了解量化基础和多模态架构

---

## 1. GQA 与 MQA——省钱的艺术

```
MHA: 16 Q heads × 16 KV heads → KV Cache: 16 份
GQA: 16 Q heads × 8 KV heads  → KV Cache: 8 份 (省 50%)
MQA: 16 Q heads × 1 KV head   → KV Cache: 1 份 (省 93.75%)

为什么多个 Q 能共享一组 KV？
  → Q head 之间的冗余度很高——它们关注的内容大量重叠
  → "让它们拼车"几乎不损失质量

ncnn 实现: ExpandDims → Tile → Reshape
  8 个 KV head → 插入维度 → 复制 2 倍 → 重组成 16 个
```

## 2. Mamba/SSM——线性复杂度的挑战者

Transformer 的致命弱点：Attention 是 O(L²) 的。序列每翻一倍，计算量翻四倍。

Mamba 的答案：**选择性状态空间模型**——O(L) 复杂度。

```
SSM 基础:
  h'(t) = A·h(t) + B·x(t)    状态更新
  y(t)  = C·h(t) + D·x(t)    输出

Mamba 的关键创新:
  让 B, C, Δ 由输入 x 动态决定（不再是固定参数）
  → "选择性"地记住或遗忘信息
  → 不需要 Attention 也能建模长程依赖

直觉: Attention 是"看所有历史"→ O(L²)
      Mamba 是"逐步更新一个压缩状态"→ O(L)
      就像读一本书——Attention 是每读一句翻回第一页重新看，
      Mamba 是记住一个不断更新的摘要
```

## 3. MoE——让 8 个专家各司其职

```
MoE 层:
  x → Router(线性层) → softmax → Top-2 选择
        ↓
  Expert₀  Expert₁  Expert₂  ...  Expert₇
        ↓
  输出 = g₁·Expertᵢ(x) + g₂·Expertⱼ(x)

参数量大了 8 倍（8 个 Expert），
但每次只激活 2 个 → 计算量只增加 ~2 倍
→ "大参数量，小计算量"的巧妙设计

挑战: 负载均衡。某些 Expert 可能被"所有人找"（热点）
解决: Load Balancing Loss + Expert Capacity 限制
```

## 4. 量化概念——"整形"比"浮点"省空间

```
对称量化:  q = round(x / s)     s = max(|x|) / (2^(bits-1) − 1)
INT8:      s = max(|x|) / 127   → 值域 [−127, 127]

量化粒度:
  per-tensor: 全张量一个 scale → 但 outlier 会毁掉精度
  per-channel: 每通道独立 scale → 精度好（INT8 卷积常用）
  per-group: 每128个值一组 → GPTQ/AWQ 的选择

PTQ (训练后量化): 不需要重训，校准数据跑一遍 → AWQ/GPTQ
QAT (量化感知训练): 训练时模拟量化 → 精度最高但成本高
```

---

*下一模块: [Module 5: 推理流水线](../module-05-inference-pipeline/README.md)*
