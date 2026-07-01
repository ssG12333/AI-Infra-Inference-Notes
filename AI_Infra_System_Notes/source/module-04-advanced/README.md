# Module 4: 进化架构——前沿模型的算子组合

> 从经典 Transformer 到现代 LLM，模型架构在持续进化。本模块覆盖注意力变体、状态空间模型、混合专家和多模态架构。

---

## 📋 学习目标

- [ ] 理解 GQA/MQA 的实现机制和 KV Cache 收益
- [ ] 能解释 RoPE 长上下文外推的三种方法
- [ ] 理解 Mamba/SSM 的 selective scan 原理
- [ ] 理解 MoE 的 Router + Top-K 路由机制
- [ ] 了解量化基础概念和部署考量

---

## 1. 注意力变体

### 1.1 GQA — 分组的艺术

```
MHA: Q_heads = KV_heads = 16 → KV Cache 存 16 份
GQA: Q_heads = 16, KV_heads = 8  → 每 2 个 Q 共享 1 组 KV
MQA: Q_heads = 16, KV_heads = 1  → 所有 Q 共享 1 组 KV

GQA 的 ncnn 实现:
  K 投影: [1024] × [1024, 1024] → [1024]
  头拆分: Reshape → [128, 8, seq]  (8 个 KV head)
  GQA 扩展: ExpandDims → [128, 8, seq, 1]
            Tile(2) → [128, 8, seq, 2]
            Reshape → [128, seq, 16]  (16 个 head)

内存节省: 28 层 × seq×8×128×2 = 57,344×seq bytes
         vs 28 层 × seq×16×128×2 = 114,688×seq bytes
         → 省 50%
```

### 1.2 QK-Norm — Qwen3 的独特设计 💡

```
位置: Q/K 投影后、RoPE 前
内容: 对每个 head 独立做 RMSNorm(affine_size=128)

为什么需要？
  长上下文中, Q 和 K 的点积方差可能急剧增大
  → softmax 趋向 one-hot (注意力熵坍塌)
  → QK-Norm 约束 Q/K 数值范围, 防止集中

哪些模型有?
  Qwen3: 有
  Qwen2.5/LLaMA 2/3: 无
  Gemma 2: 有
```

### 1.3 RoPE 长上下文外推

| 方法 | 技巧 | 扩展倍数 |
|------|------|:------:|
| NTK-Aware | 调大 theta 而非位置 | 1.5-2× |
| YaRN | NTK + ramp 平滑过渡 + mscale | 2-8× |
| LongRoPE | 逐维度独立缩放因子 | 8×+ |

---

## 2. 状态空间模型 (SSM / Mamba)

### 2.1 核心公式

```
SSM:  h'(t) = A·h(t) + B·x(t)
      y(t)  = C·h(t) + D·x(t)

离散化 (Zero-Order Hold):
  Ā = exp(Δt·A)
  B̄ = (Δt·A)⁻¹(exp(Δt·A) − I)·Δt·B

Mamba 的改进 (Selective SSM):
  B, C, Δt 不再是固定的 → 由输入 x 动态生成
  → "选择性"记住或遗忘信息

复杂度: O(L) vs Attention 的 O(L²)
→ 长序列更有优势
```

### 2.2 SSM vs Attention

| 维度 | Attention | SSM/Mamba |
|------|-----------|-----------|
| 复杂度 | O(L²) | O(L) |
| 长序列 | 显存瓶颈 | 天然友好 |
| 并行训练 | 天然并行 | 需要 Parallel Scan |
| 上下文利用 | 全局 | 偏向近期（可控） |
| 代表 | LLaMA, Qwen | Mamba, Jamba (混合) |

---

## 3. 混合专家 (MoE)

### 3.1 工作原理

```
MoE Layer:
  x → Router (线性层) → softmax → Top-K 选择
       ↓
  Expert₀  Expert₁  Expert₂  ...  Expert₇
       ↓
  加权聚合: y = Σ(g_i · Expert_i(x))

关键参数:
  num_experts: 8 (Mixtral) ~ 256 (DeepSeek-V3)
  top_k:       2 (活跃专家数)
  Router:      一个小型线性层，输出每个专家的权重
```

### 3.2 计算量分析

```
标准 FFN (Qwen3-0.6B):  1 次 MLP forward → ~9.5M 参数计算
MoE (8 experts, top-2): 2 次 MLP forward → ~19M 参数计算

总参数量大了 8×，但实际计算量只增加 2× (top-2)
→ "大参数量，小计算量" 的巧妙设计

负载均衡:
  问题: 某些专家可能被过度使用 (热点)
  解决: Load Balancing Loss + Expert Capacity 限制
```

---

## 4. 量化基础概念

### 4.1 量化的数学

```
对称量化:   q = round(x / s)
反量化:     x' = q × s
           s = max(|x|) / (2^(bits-1) − 1)

FP32 → INT8:  s = max(|x|) / 127
  [−0.5, 0.8] → s = 0.8/127 = 0.0063
  −0.5 → round(−79.4) = −79
   0.8 → round(127.0) = 127

量化粒度:
  per-tensor:  全张量一个 s → 简单但 outlier 影响大
  per-channel: 每个通道独立 s → 精度好
  per-group:   每 128 个值一组 s → GPTQ/AWQ 常用
```

### 4.2 PTQ vs QAT

| 方法 | 说明 | 成本 | 精度 |
|------|------|:--:|:--:|
| PTQ | 训练后量化 (AWQ/GPTQ) | 低 | 可能退化 |
| QAT | 量化感知训练 | 高 | 更稳 |

---

## 5. 多模态架构

```
Vision-Language Model 推理:
  图像 → Vision Encoder (ViT) → image_tokens [N, hidden]
  文本 → Tokenizer → text_tokens [M, hidden]
  ────────────────────────────────────────────────
  拼接: [image_tokens, text_tokens] → LLM Decoder
  → 生成文本 (可能包含视觉理解)

常见模型:
  Qwen-VL: ViT + Qwen LLM
  LLaVA:   CLIP ViT + LLaMA
  GPT-4V:  闭源

关键算子: mRoPE (多维 RoPE), Cross-Attention, 空间合并
```

---

## 🛠️ 动手练习

1. 计算 Qwen3-0.6B 改为 MHA (16 KV heads) vs GQA (8 heads) 在 seq=4096 时的 KV Cache 大小。

2. 对 4 个 expert、top-2 的 MoE，画出 Router 的选择逻辑。如果 Router 输出 [0.6, 0.01, 0.3, 0.09]，最终哪两个 expert 被激活？

3. 将一个值域 [−2.5, 3.0] 的权重张量做 INT8 对称量化，计算 scale 和量化后的值。

---

*下一模块: [Module 5: 推理流水线](../module-05-inference-pipeline/README.md)*
