# Module 8: 优化技术——量化、FlashAttention、图优化与 Profiling

> 推理优化的本质：在精度可接受的前提下，最大化地减少显存占用和计算时间。

---

## 📋 学习目标

- [ ] 能手写对称量化公式并算一个具体例子
- [ ] 能画出 ncnn 的 Quantize→Conv INT8→Requantize 数据流
- [ ] 能解释 AWQ 和 GPTQ 的核心差异
- [ ] 能画出 FlashAttention 的分块计算流程图
- [ ] 能推导 Conv-BN 融合的数学公式
- [ ] 能根据 profiling 数据诊断瓶颈

---

## 1. 量化：从浮点到整数

### 1.1 对称量化公式

```
量化:   q = round(x / s)         s = max(|x|) / (2^(bits-1) − 1)
反量化:  x' = q × s

INT8 例子: 值域 [−0.8, 1.2]
  scale = 1.2 / 127 ≈ 0.00945
  −0.8 → round(−84.7) = −85
   1.2 → round(127.0) = 127
   0.3 → round(31.7)  =  32
  反量化误差 ≈ 0.004
```

### 1.2 量化粒度

```
per-tensor:  全张量 1 个 scale → 简单但 outlier 破坏精度
per-channel: 每个输出通道 1 个 scale → 精度好 (INT8 卷积常用)
per-token:   每个 token 1 个 scale → 激活值量化 (动态)
per-group:   每 128 个值 1 个 scale → GPTQ/AWQ 常用

精度: per-group > per-channel > per-tensor
开销: per-group > per-channel > per-tensor
```

### 1.3 ncnn INT8 三大量化层 🔬

```
Quantize (FP32 → INT8):
  output_int8[i] = round(clip(fp32_input[i] × scale, −127, 127))

Requantize (INT32 → INT8) — 最关键的层:
  // ncnn requantize.cpp 核心
  float v = int32_input[i] × scale_in + bias;  // INT32 → FP32
  v = activation(v);                            // 在 FP32 精度下激活
  output_int8[i] = round(clip(v × scale_out, −127, −127));

  为什么 scale_in + scale_out？
    scale_in: dequantize (INT32 累加结果 → FP32)
    scale_out: requantize (FP32 → INT8, 下一层期望的 scale)

Dequantize (INT8 → FP32):
  output_fp32[i] = int8_input[i] / scale
```

### 1.4 AWQ vs GPTQ

```
AWQ (Activation-aware):
  策略: 分析激活分布 → 找到 salient 通道 → 缩放保护 → 量化
  速度: 快 (几分钟)
  精度: 好 (INT4 损失 <1%)

GPTQ (误差补偿):
  策略: 逐列量化 → 量化误差补偿到未量化列 → 最小化输出误差
  速度: 慢 (几十分钟, 需要 Hessian)
  精度: 略优于 AWQ (极端低 bit 下优势明显)

选择: 快速部署 → AWQ; 极致精度 → GPTQ
```

### 1.5 FP8 — 新一代 GPU 格式

```
FP8 的两种格式:
  E4M3: 4-bit exponent, 3-bit mantissa → 范围 ±448, 适合前向
  E5M2: 5-bit exponent, 2-bit mantissa → 范围 ±57344, 适合梯度

vs INT8: FP8 有指数位 → 天然动态范围大 → 无需 scale 校准!
→ H100 等新 GPU 硬件加速
```

---

## 2. FlashAttention 🔬

### 2.1 问题的本质

```
标准 Attention:
  S = Q × K^T  → 存储 [seq, seq]  → 写入 HBM
  P = softmax(S) → 存储 [seq, seq] → 写入 HBM
  O = P × V     → 存储 [seq, d]   → 写入 HBM

当 seq=8192: S 和 P 各占 8192² × 4B = 268 MB
28 层 × 16 heads → 120 GB 中间结果!

更致命的问题: S/P 写入 HBM 又读回来 → 带宽浪费严重
```

### 2.2 三个核心技巧

```
1. Tiling (分块):
   切 Q 为 [Br, d], K/V 为 [Bc, d] 的小块
   每块在 GPU 的 SRAM (shared memory) 中完成全部计算
   块间不存储 S 和 P!

2. Online Softmax:
   for each block:
       m_new = max(m_old, max(block))
       sum_new = sum_old × exp(m_old − m_new) + Σexp(block − m_new)
       修正旧结果: O_block *= exp(m_old − m_new)
       O_block += new_contribution
       m_old = m_new; sum_old = sum_new

3. Recomputation (仅训练):
   前向不存 S/P → 反向时用 Q/K/V 重新计算

效果: 显存 O(seq²) → O(seq),  带宽节省 3×
```

### 2.3 分块参数

```
ncnn Vulkan FlashAttention:
  Br = 4   (query block: 4 行)
  Bc = 32  (key/value block: 32 列)
  Bk = 32  (head_dim 分块)

Shader 中的循环:
  for each Q_block:
      load Q_block to shared memory
      O_accum = 0, m = −∞, l = 0
      for each K_block, V_block:
          load K_block, V_block to shared memory
          S = Q_block × K_block^T (在 shared memory 中)
          S *= scale
          (m_new, l_new, P) = online_softmax(S, m, l)
          O_accum *= exp(m − m_new)   ← correction
          O_accum += P × V_block
          (m, l) = (m_new, l_new)
      O_block = O_accum / l  ← 最终归一化
      write O_block to global memory
```

---

## 3. 图优化

### 3.1 六大优化 Pass

| Pass | 说明 | 示例 |
|------|------|------|
| **Constant Folding** | 编译时计算常量 | `3×4` → `12` |
| **Dead Code Elimination** | 删除无用节点 | 不被使用的输出分支 |
| **Operator Fusion** | 合并连续算子 | Conv+BN+ReLU → 1 个 kernel |
| **CSE** | 消除公共子表达式 | `a×b` 只算一次 |
| **Layout Optimization** | 选择最优内存布局 | 消除 NCHW↔NHWC 转换 |
| **Memory Planning** | 复用 buffer | 不重叠的 blob 共享内存 |

### 3.2 Conv-BN 融合推导

```
BatchNorm 推理公式 (running mean/var 固定):
  BN(x) = γ × (x − μ) / √(σ² + ε) + β
        = α × x + b'     ← 线性变换!

  其中: α = γ / √(σ² + ε)
        b' = β − γ × μ / √(σ² + ε)

融合到卷积:
  Conv(x) = Wx + b
  ConvBN(x) = α(Wx + b) + b' = (αW)x + (αb + b')
  → W' = αW,  b' = αb + β'
  → 模型加载时一次性计算, 推理零开销!
```

---

## 4. Profiling：用数据驱动优化

### 4.1 瓶颈诊断速查

| 现象 | 可能原因 | 优化方向 |
|------|---------|---------|
| 首 token 很慢 | Prompt 太长 | Prefix cache, Chunked prefill |
| 后续 token 慢 | KV Cache 读取瓶颈 | FlashAttention, KV 量化 |
| GPU 利用率 <30% | Decode batch 太小 | Continuous Batching |
| 显存爆 | KV Cache 累积 | PagedAttention, KV 量化 |
| INT8 精度差 | Per-tensor scale 粗 | Per-channel, AWQ/GPTQ |
| Vulkan 不加速 | Shader 覆盖不全 | 确认算子支持, 考虑 fallback |

### 4.2 关键指标分层

```
硬件层: GPU 利用率, 显存带宽, Cache Miss
算子层: 每层耗时, FLOPs, memory footprint
模型层: TTFT, TPOT, Throughput, KV Cache usage
服务层: QPS, P99 latency, Error rate
```

---

## 🛠️ 动手练习

1. **量化手算**: 对权重 `[0.5, −1.2, 3.0, −0.8]` 做 INT8 对称量化。

2. **BN 融合计算**: 给定 Conv: W=[[2,1],[0,3]], b=[0.5,−0.2]; BN: γ=[2,1], β=[0,0], μ=[1,2], σ²=[4,1]。计算融合后的 W' 和 b'。

3. **FlashAttention trace**: 在纸上模拟 seq=8, Br=2, Bc=3 的分块过程。

---

*下一模块: [Module 9: 部署实战](../module-09-deployment/README.md)*
