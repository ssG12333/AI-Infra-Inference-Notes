# 06｜Optimization：量化、FlashAttention、图优化与性能调优

> 在 AI Infra 中，"优化"不是玄学——它是量化公式、内存布局、kernel 替换和 profiler 驱动决策的系统工程。

---

## 1. 优化的全景图

### 1.1 四维优化空间

```
                    ┌──────────────────────────────┐
                    │        推理优化全景            │
                    │                              │
        Latency ───┤  首 token 快不快？             │
      Throughput ──┤  同时能服务多少人？             │
         Memory ───┤  显存/内存够不够？              │
       Stability ──┤  精度会不会崩？                │
                    └──────────────────────────────┘

四条路径互相制约:
  降低精度 → 省显存、提升速度 → 可能损失精度
  加大 batch → 提升吞吐 → 增加延迟和显存
  Kernel fusion → 减少访存 → 实现复杂，维护成本高
```

### 1.2 优化手段速览

| 优化手段 | 收益维度 | 实现成本 | 影响范围 |
|---------|---------|---------|---------|
| **INT8 权重量化** | Memory ↓50% | 中（需要校准） | 模型权重 |
| **INT8 KV Cache 量化** | Memory ↓50% | 低（动态量化） | KV Cache |
| **INT4 权重量化** | Memory ↓75% | 高（精度风险） | 模型权重 |
| **FlashAttention** | Memory ↓, Speed ↑ | 高（需专用 kernel） | Attention |
| **算子融合** | Speed ↑, Memory ↓ | 中（图优化 pass） | 计算图 |
| **PagedAttention** | Memory ↑利用率 | 高（需重构 cache） | KV Cache |
| **Continuous Batching** | Throughput ↑2-4x | 高（需调度器） | 服务层 |
| **Prefix Cache** | Latency ↓ | 中 | Prefill |

---

## 2. 量化基础：从浮点到整数的数学

### 2.1 量化的核心公式

```
量化本质上是一个仿射变换：

  量化:  q = round(x / s) + z
  反量化: x' = s × (q − z)

  其中:
    x     = 原始浮点值
    q     = 量化后的整数值
    s     = scale（缩放因子，浮点数）
    z     = zero_point（零点偏移，整数）
    x'    = 反量化后的近似值

对称量化 (z=0):
  量化:  q = round(x / s) = round(x × scale)
  反量化: x' = q / scale = q × (1/scale)

  scale 通常取值: s = max(|x|) / (2^(bits-1) − 1)
  对于 INT8: s = max(|x|) / 127
```

### 2.2 一个具体例子

```
假设某层的权重值范围是 [-0.8, 1.2]:

INT8 对称量化（值域 [-127, 127]）:
  max_abs = max(|-0.8|, |1.2|) = 1.2
  scale = 1.2 / 127 ≈ 0.009449

  原始值 → 量化:
    -0.8 → round(-0.8 / 0.009449) = round(-84.7) = -85
     1.2 → round( 1.2 / 0.009449) = round(127.0) =  127
     0.3 → round( 0.3 / 0.009449) = round( 31.7) =   32

  反量化:
    -85 → -85 × 0.009449 = -0.803  (误差 0.003)
    127 → 127 × 0.009449 =  1.200  (误差 0)
     32 →  32 × 0.009449 =  0.302  (误差 0.002)

量化误差 ≈ 0.004 → 精度影响很小
```

### 2.3 量化的粒度选择

```
Per-Tensor（整层一个 scale）:
  所有值共享一个 scale
  问题: outlier 会拉大 scale，导致大部分值精度差

  scale = max_abs(整个 tensor) / 127

Per-Channel（每个通道独立 scale）:
  卷积中: 每个输出通道一个 scale
  全连接中: 每个输出神经元一个 scale
  精度显著优于 per-tensor

  scale[i] = max_abs(第 i 个通道) / 127

Per-Token（每个 token 独立 scale）:
  用于激活值量化（运行时动态计算）
  因为激活值随输入变化，不能预先校准

  scale[t] = max_abs(第 t 个 token 的激活) / 127

Per-Group（分组量化，如每 128 个值一组）:
  GPTQ/AWQ 的常见选择
  平衡精度和 scale 存储开销
```

---

## 3. ncnn 量化系统源码解析

ncnn 的 INT8 推理使用**三级流水线**：

```
FP32 输入 ──→ Quantize ──→ INT8 计算 ──→ Requantize ──→ 下一层
                  │              │                │
               FP32→INT8      INT8 MatMul      INT32→INT8
                              (INT32 累加)
```

### 3.1 Quantize（FP32 → INT8）

```cpp
// ncnn/src/layer/quantize.cpp 核心逻辑
// 功能: 将 FP32 张量量化为 INT8

// scale = max(|x|) / 127  或  预先校准的 scale
// q[i] = round(clip(x[i] * scale, -127, 127))

static inline signed char float2int8(float v) {
    int int32 = static_cast<int>(round(v));
    if (int32 > 127) return 127;    // 饱和截断
    if (int32 < -127) return -127;  // 饱和截断
    return (signed char)int32;
}

// 量化过程:
// 对于输入张量的每个元素:
//   int8_value = float2int8(fp32_value * scale)
```

### 3.2 Requantize（INT32 → INT8）

这是 ncnn INT8 推理的**关键环节**——卷积/矩阵乘法的 INT32 累加结果需要重新量化为 INT8 传给下一层。

```cpp
// ncnn/src/layer/requantize.cpp 核心逻辑
// 功能: INT32 累加结果 → INT8（用于下一层输入）

// 公式: output_int8 = float2int8((input_int32 * scale_in + bias) * scale_out)
//                                                    ↑          ↑
//                                                INT32→FP32   FP32→INT8

static void requantize(
    const int* intptr,       // INT32 输入（卷积/矩阵乘法累加结果）
    signed char* ptr,        // INT8 输出
    float scale_in,          // 输入的 dequantize scale（INT32 → FP32）
    float bias,              // 偏置（FP32）
    float scale_out,         // 输出的 quantize scale（FP32 → INT8）
    int activation_type,     // 激活函数类型（ReLU/Clip/...）
    const Mat& activation_params,
    int size
) {
    for (int i = 0; i < size; i++) {
        // Step 1: INT32 → FP32 (dequantize)
        float v = *intptr * scale_in + bias;

        // Step 2: 激活函数（在 FP32 精度下计算！）
        v = activation_ss(v, activation_type, activation_params);
        //    ↑ 这一步很关键：激活函数在较高精度下计算，
        //    避免量化误差在非线性变换中被放大

        // Step 3: FP32 → INT8 (quantize)
        *ptr = float2int8(v * scale_out);

        intptr++;
        ptr++;
    }
}
```

### 3.3 完整的 INT8 卷积流程

```
假设输入 [B, C_in, H, W]，输出 [B, C_out, H', W']

Step 1: 输入量化（FP32 → INT8）
  input_int8[b,c,h,w] = float2int8(input_fp32[b,c,h,w] * input_scale)
  注意: input_scale 可能 per-channel

Step 2: INT8 卷积计算（INT8 × INT8 → INT32 累加）
  权重已离线量化为 INT8: weight_int8[oc, ic, kh, kw]
  权重有自己的 scale: weight_scale[oc]（per-channel）

  output_int32[b,oc,oh,ow] = Σ(input_int8[b,ic,oh+kh,ow+kw]
                              × weight_int8[oc,ic,kh, kw])
  ↑ INT8 × INT8，累加到 INT32（不会溢出）

Step 3: Requantize（INT32 → INT8 输出）
  scale_in  = 1 / (input_scale × weight_scale)
           = 1 / requantize_scale
  scale_out = output_scale（下一层期望的输入 scale）

  output_int8 = requantize(output_int32, scale_in, bias, scale_out, act)

关键公式:
  fp32_result = Σ(input_fp32 × weight_fp32)
              = Σ((input_int8 / input_scale) × (weight_int8 / weight_scale))
              = (Σ input_int8 × weight_int8) / (input_scale × weight_scale)
              = output_int32 × scale_in
```

### 3.4 ncnn 中的 scale 存储

```python
# ncnn param 文件中量化层的参数示例:
# Convolution 层:
#   8=1          ← int8_scale_term（是否使用 INT8）
#   9=256        ← weight_scale 的元素数
#   weight_scale 数据在 .bin 文件中

# Requantize 层:
#   0=256        ← scale_in 的元素数（per-channel=256，per-tensor=1）
#   1=256        ← scale_out 的元素数
#   2=256        ← bias 的元素数（0 表示无 bias）
#   3=0          ← 激活函数类型（0=无，1=ReLU，2=LeakyReLU）
```

---

## 4. AWQ 与 GPTQ：LLM 权重量化的两种路线

### 4.1 为什么普通量化对 LLM 不够？

```
问题：LLM 的权重中存在 "salient channels"（显著通道）

某些通道的权重数值范围是其他通道的 10-100 倍。
如果用 per-tensor 量化，scale 会被这些 outlier 拉大，
导致正常通道的量化精度极差。

例子：
  Channel 0:  [0.1, -0.2, 0.15, -0.1]    ← max_abs = 0.2
  Channel 1:  [8.5, -3.2, 7.1, -9.8]     ← max_abs = 9.8  SALIENT!
  Channel 2:  [0.05, -0.08, 0.12, -0.03] ← max_abs = 0.12

  Per-tensor scale = 9.8 / 127 ≈ 0.0772
  Channel 0 量化: 0.1 / 0.0772 = 1.29 → round → 1
  Channel 2 量化: 0.05 / 0.0772 = 0.65 → round → 1
  → Channel 0 的小差异和 Channel 2 的小差异都被量化为同一个值！

  Per-channel scale:
  Channel 0: scale = 0.2/127 ≈ 0.00157 → 0.1/0.00157 = 64
  Channel 2: scale = 0.12/127 ≈ 0.00094 → 0.05/0.00094 = 53
  → 保留了细粒度差异
```

### 4.2 AWQ (Activation-aware Weight Quantization)

**核心思想**：不只看权重的分布，还要看**激活值**的分布——因为推理时权重是静态的，但激活值是动态的。通过分析激活值的统计特性，找到那些对输出影响大的"重要权重通道"，保护它们。

```
AWQ 的工作流程:

Step 1: 跑一批校准数据，收集每层的激活值分布
  → 发现某些输入通道的激活值特别大
  → 这些通道对应的权重列就是 "salient"

Step 2: 对 salient 通道做 "scaling"
  → 把权重的 scale 缩小（让量化更精细）
  → 在激活值侧做等价的 scale 放大
  → 数学等价！但量化的精度提高了

  原始: y = W × x
  Salient ch 处理: W' = W × diag(s), x' = x × diag(1/s)
  结果: W' × x' = W × diag(s) × diag(1/s) × x = W × x  ✓ 数学等价

Step 3: 用调整后的权重做 per-channel INT4/INT8 量化

优势:
  - 不需要重新训练（PTQ）
  - 在校准数据上找到最优的 s
  - 对大多数 LLM 有效
```

### 4.3 GPTQ (Post-Training Quantization with GPT)

**核心思想**：逐列量化权重，每量化一列，把量化误差"补偿"到未量化的列上。

```
GPTQ 的工作流程:

Step 1: 收集校准数据的 Hessian 矩阵的逆
  H = (X^T × X)^(-1)   ← 输入激活的二阶统计信息

Step 2: 逐列量化 + 误差补偿
  for col in range(num_columns):
      # 量化第 col 列
      W_quant[:, col] = quantize(W[:, col])

      # 计算量化误差
      error = W[:, col] - dequantize(W_quant[:, col])

      # 将误差补偿到剩余未量化的列上
      for remaining_col in range(col+1, num_columns):
          W[:, remaining_col] -= error × H[col, remaining_col] / H[col, col]

  这一步保证: 量化引入的误差在后续列中得到补偿
  → 整体输出误差最小化

优势:
  - 数学上更优雅（最小化输出误差而非权重误差）
  - 可以一次性把所有层都量化
  - 支持 INT4/INT3/INT2

对比 AWQ:
  - GPTQ: 误差补偿策略，更强但更慢
  - AWQ: 缩放策略，更简单但依赖于找到好的 s
  - 两者都是 PTQ，不需要重新训练
```

### 4.4 AWQ vs GPTQ 选择指南

| 维度 | AWQ | GPTQ |
|------|-----|------|
| 核心策略 | 缩放 salient 通道 | 逐列量化 + 误差补偿 |
| 量化速度 | 较快（几分钟） | 较慢（几十分钟） |
| 精度 (INT4) | 好 | 更好一点 |
| 实现复杂度 | 较低 | 较高（需要 Hessian） |
| 代表框架 | llama.cpp, vLLM | AutoGPTQ, vLLM |
| 显存需求 | 低（只需激活统计） | 高（需要 Hessian 矩阵） |

---

## 5. FP8 量化：新一代 GPU 的秘密武器

### 5.1 FP8 的两种格式

FP8 不是 INT8——它保留了浮点的指数位，数值范围远超 INT8。

```
E4M3 (4-bit exponent, 3-bit mantissa):
  格式: [sign:1][exponent:4][mantissa:3]
  范围: ±448（远超 INT8 的 ±127）
  精度: 指数位多 → 动态范围大
  用途: 前向传播的权重和激活

E5M2 (5-bit exponent, 2-bit mantissa):
  格式: [sign:1][exponent:5][mantissa:2]
  范围: ±57344（更大了！）
  精度: 更粗（只有 2-bit mantissa）
  用途: 反向传播的梯度（需要大范围防溢出）

对比 INT8:
  INT8:  范围 ±127, 均匀间隔, 精度 = scale/127
  FP8:   范围 ±448, 非均匀间隔, 小值精度高, 大值范围大
  → FP8 不需要 scale！或者说 scale 天然内建在指数中
```

### 5.2 FP8 为什么适合 Transformer？

```
Transformer 的数值模式:
  - Attention score: softmax 前可能在 [-50, 50]，softmax 后在 [0, 1]
  - LayerNorm/RMSNorm: 输入输出在 [-3, 3] 附近
  - MLP 中间激活: SwiGLU 后可能达到 [-20, 20]
  - 权重: 通常 [-0.5, 0.5]

INT8 的问题: 需要为不同类型的值选不同的 scale
  → attention score 的 scale ≠ MLP 激活的 scale ≠ 权重的 scale
  → 每层甚至每个算子都需要独立的 scale 校准

FP8 的优势: 指数位提供天然的动态范围
  → 小值（如 attention prob）和大值（如 MLP 中间结果）
    都用同一个 E4M3 格式表示，无需校准 scale
  → 简化了量化流程
```

---

## 6. FlashAttention：让 Attention 不再成为瓶颈

### 6.1 标准 Attention 的显存问题

```
标准 Attention 的计算步骤:

Step 1: S = Q × K^T         [seq, d] × [d, seq] → [seq, seq]
        ↑ 这个 seq × seq 矩阵是关键！

Step 2: P = softmax(S)      [seq, seq]  仍在显存中

Step 3: O = P × V           [seq, seq] × [seq, d] → [seq, d]

问题:
  S 和 P 都是 [seq, seq] 的矩阵
  当 seq=4096:  S 占 4096² × 4 bytes = 67 MB
  当 seq=8192:  S 占 8192² × 4 bytes = 268 MB
  当 seq=32768: S 占 32768² × 4 bytes = 4.3 GB

这还只是单层、单 head 的 S！
完整模型: num_layers × num_heads × seq² × 4 bytes
  → 代价惊人

更致命的是: S 和 P 先写入 HBM (GPU 显存)，
            又从 HBM 读回来做下一步计算
  → 带宽浪费严重
```

### 6.2 FlashAttention 的核心技巧

```
三个关键技巧:

1. Tiling（分块计算）
   不一次性算 [seq, seq] 的 S
   而是切成小块: Q_block [Br, d] × K_block [d, Bc] → S_block [Br, Bc]
   每个 block 只在 GPU 的 SRAM (shared memory) 中存在

2. Online Softmax（在线 Softmax）
   传统: 先算完所有 S，找 max，exp，sum，div
   Flash: 每个 block 边算边更新
   当新 block 有更大的 max 时，用公式修正旧结果:
     correct_factor = exp(old_max - new_max)
     old_sum *= correct_factor

3. Recomputation（反向传播时重算）
   前向不存 S/P，只存 softmax 的归一化统计量
   反向时用 Q、K、V 重新计算 attention
   → 省掉了最大的显存开销！

显存对比:
  标准 Attention: O(seq²)  显存
  FlashAttention: O(seq)   显存（只存统计量）

  当 seq=8192:  268 MB → ~1 MB  (per head per layer)
```

### 6.3 Online Softmax 的数学

```
传统 Softmax:
  m = max(x)
  y = exp(x - m)
  sum = Σy
  softmax = y / sum

  → 需要先扫描一遍找 max，再扫描一遍算 exp/sum

Online Softmax (一个 block 一个 block 地处理):
  for each block B:
      m_new = max(m_old, max(B))
      sum_new = sum_old × exp(m_old - m_new) + Σexp(B - m_new)
      m_old = m_new
      sum_old = sum_new

  最终: softmax = exp(x_i - m_final) / sum_final

关键公式: sum_old × exp(m_old - m_new)
  当发现更大的 m_new 时，旧 sum 中所有值都多除了 exp(m_old-m_new)
  → 需要乘回来！这就是 "correction"
```

### 6.4 ncnn 中的 FlashAttention（Vulkan 实现）

ncnn 的 Vulkan backend 实现了基于 compute shader 的 FlashAttention。

```cpp
// ncnn/src/layer/vulkan/sdpa_vulkan.cpp 核心逻辑
// FlashAttention in Vulkan compute shader

// 分块参数（通过 specialization constant 配置）:
//   Br = 4, Bc = 32, Bk = 32
//   ← 每个 workgroup 处理 4 行 query × 32 列 key

// Shader 伪代码:
// layout: Q[heads, seq_q, head_dim], K[heads, seq_k, head_dim]
//
// for each query block (size Br):
//     shared float Q_block[Br][head_dim];  // 从 global 读到 shared
//     shared float O_block[Br][head_dim];  // 输出累积器
//     float m_prev[Br] = -INF;             // online softmax 的 max
//     float l_prev[Br] = 0;                // online softmax 的 sum
//
//     for each key/value block (size Bc):
//         shared float K_block[Bc][head_dim];  // 从 global 读到 shared
//         shared float V_block[Bc][head_dim];
//
//         // 1. QK^T (在 shared memory 中)
//         S_block = matmul(Q_block, K_block^T)  // [Br, Bc]
//         S_block *= scale
//
//         // 2. Online Softmax
//         m_curr = max(m_prev, rowmax(S_block))
//         P_block = exp(S_block - m_curr)
//         l_curr = l_prev × exp(m_prev - m_curr) + rowsum(P_block)
//
//         // 3. Correction (修正旧的 O)
//         O_block *= exp(m_prev - m_curr)
//
//         // 4. PV
//         O_block += matmul(P_block, V_block)
//
//         m_prev = m_curr
//         l_prev = l_curr
//
//     // 5. 最终归一化
//     O_block /= l_prev
//     写回 global memory
```

**ncnn Vulkan FlashAttention 的内存布局**：

```
标准 SDPA 的显存访问:
  Global → Shared → QK^T → Global (write S)
  Global ← S (read back) → Shared → Softmax → Global (write P)
  Global ← P (read back) → Shared → PV → Global (write O)
  共 3 次 round-trip 到 global memory!

FlashAttention (Vulkan):
  Global → Shared (Q, K, V blocks)
  All computation in Shared (QK^T + Softmax + PV)
  Shared → Global (write O)
  只需 1 次 round-trip！(读 Q/K/V，写 O)
```

---

## 7. 图优化：编译器的魔法

### 7.1 六大优化 Pass

```
Pass 1: Constant Folding（常量折叠）
  在编译时计算静态表达式

  前:  y = x + (3 × 4)      ← 3×4 是常量
  后:  y = x + 12           ← 折叠为一个常数

  ncnn 例子: Conv+BatchNorm 的融合参数在加载时计算


Pass 2: Dead Code Elimination（死代码消除）
  删除输出不被使用的节点

  前:  A → B → C → unused_output
            → D → used_output
  后:  A → B → D → used_output   ← C 和 unused_output 被删除


Pass 3: Operator Fusion（算子融合）
  将多个连续算子合并为一个

  前: Conv → BatchNorm → ReLU   (3 个 kernel launch)
  后: ConvBNReLU                (1 个 kernel launch)

  融合条件:
  1. 数据流是一对一的（没有分支）
  2. 融合后的算子有对应实现
  3. 不破坏量化精度


Pass 4: Common Subexpression Elimination（公共子表达式消除）
  前:  y = a × b + c
       z = a × b + d            ← a×b 算了两次
  后:  t = a × b
       y = t + c
       z = t + d


Pass 5: Layout Optimization（布局优化）
  改变张量的内存布局以减少转置

  前:  NCHW 输入 → Transpose → NHWC 计算 → Transpose → NCHW 输出
  后:  全链路 NHWC，取消 Transpose

  ncnn 做法: 由后端自动选择最优 layout


Pass 6: Memory Planning（内存规划）
  复用中间 buffer，减少 peak memory

  前: 每层分配独立的输出 buffer
  后: 分析张量生命周期，复用已释放的 buffer

  ncnn 的 blob 复用: 两个算子的输出 blob 如果生命周期不重叠，
  可以使用同一块内存
```

### 7.2 最重要的融合：Conv-BN

```
融合 Conv 和 BatchNorm 是推理框架最基础的优化。

原理: BatchNorm 是一个线性变换（推理时 running_mean/var 固定）
  BN(x) = γ × (x − μ) / √(σ² + ε) + β
        = (γ / √(σ² + ε)) × x + (β − γ × μ / √(σ² + ε))
        = α × x + b'

融合到 Conv:
  Conv(x) = W × x + b
  ConvBN(x) = α × (W × x + b) + b'
              = (α × W) × x + (α × b + b')
              = W' × x + b'

效果:
  - 减少 1 次逐元素乘和 1 次逐元素加（每个通道！）
  - 减少 1 次 kernel launch
  - W' 和 b' 在模型加载时一次性计算
  - 推理时完全无开销
```

---

## 8. Profiling：用数据驱动优化

### 8.1 指标分层

```
层级        关注指标               工具
─────────────────────────────────────────────
硬件层      GPU 利用率、显存带宽    nvidia-smi, nsys
           CPU 利用率、Cache Miss  perf, VTune

算子层      每层耗时、FLOPs、       ncnn benchmark,
           memory footprint       torch.profiler

模型层      TTFT, TPOT, Throughput vLLM metrics,
           KV Cache usage         Prometheus

服务层      QPS, P99 latency,     Grafana, ELK
           Error rate
```

### 8.2 ncnn 性能分析

```cpp
// ncnn 提供的 profiling API
ncnn::Net net;
net.opt.use_vulkan_compute = true;

// 开启 profiling
net.opt.use_packing_layout = true;

// 运行推理
ncnn::Extractor ex = net.create_extractor();
ex.set_light_mode(true);  // 启用 blob 复用

// 获取每层性能
// ncnn 提供了 benchmark 工具: ./benchncnn
// 输出每层的:
//   - 耗时 (ms)
//   - 内存占用 (KB)
//   - gemm flops / total flops 比例
```

### 8.3 瓶颈诊断速查表

| 现象 | 可能原因 | 验证方法 | 优化方向 |
|------|---------|---------|---------|
| **首 token 很慢** | Prompt 太长 | TTFT vs prompt_len 曲线 | Prefix cache, Chunked prefill |
| **后续 token 慢** | KV Cache 读取瓶颈 | 内存带宽使用率 | FlashAttention, KV 量化 |
| **GPU 利用率 <30%** | Decode batch 太小 | batch_size vs utilization | Continuous Batching |
| **显存爆** | KV Cache 累积 | KV cache usage 曲线 | PagedAttention, KV 量化, swap |
| **INT8 精度差** | Per-tensor scale 不够细 | Layer-wise error 分析 | Per-channel, AWQ/GPTQ |
| **Vulkan 不加速** | Shader 覆盖不全 | 运行 vulkan vs cpu 对比 | Fallback 策略, 补齐 shader |
| **吞吐瓶颈** | Scheduler 不够 aggressive | Queue length, token budget | 调整 max_num_batched_tokens |

---

## 9. INT8 与 Vulkan 的工程取舍

### 9.1 为什么它们可能互斥？

```
ncnn 的 INT8 路径需要:
  - Quantize layer (FP32 → INT8)
  - INT8 Convolution/GEMM kernel
  - Requantize layer (INT32 → INT8)
  - Dequantize layer (INT8 → FP32)

ncnn 的 Vulkan 路径需要:
  - 对应的 compute shader
  - 上传权重到 GPU buffer
  - 管理 GPU memory / descriptor set

互斥的原因:
  1. Vulkan shader 可能没有实现 INT8 版本的 attention
     (FP16 的 shader 写好了，INT8 的还没写)
  2. INT8→FP32→INT8 频繁转换破坏了融合机会
  3. GPU 上的 INT8 计算不一定比 FP16 快
     (现代 GPU 的 FP16 throughput 可能 > INT8)

工程现实:
  不能假设"理论最优 = 实际最优"
  必须实测: CPU INT8 vs Vulkan FP16 vs Vulkan INT8
  不同模型、不同硬件上结论可能完全不同
```

### 9.2 ncnn 的选择策略

```cpp
// ncnn_llm 中的 backend 选择逻辑

if (use_vulkan) {
    // 使用 Vulkan FP16/BF16 路径
    // - Attention: FlashAttention shader
    // - GEMM: Vulkan cooperative matrix
    // - 不需要 Quantize/Requantize 层
    net.opt.use_vulkan_compute = true;
} else {
    // 使用 CPU INT8 路径
    // - Conv: INT8 GEMM kernel
    // - 需要量化层
    // - ARM NEON / x86 AVX2 优化
    net.opt.use_vulkan_compute = false;
    // 加载 INT8 量化模型
}
```

---

## 10. 面试回答模板

### Q1: FlashAttention 为什么能省显存，同时还能加速？

```
回答框架：

FlashAttention 通过两个核心技巧同时解决显存和速度问题：

1. Tiling（分块）：
   不一次性计算和存储完整的 [seq, seq] attention matrix，
   而是在 GPU shared memory 中以 block 为单位逐步计算。
   block 之间不存储中间的 S 和 P 矩阵。
   → 显存从 O(seq²) 降到 O(seq)

2. Kernel Fusion：
   传统实现中 QK^T、softmax、PV 是三个独立的 kernel，
   中间结果（S, P）需要写回 HBM 再读回来。
   FlashAttention 在一个 kernel 中完成全部计算。
   → 省掉了最耗时的 HBM 读写往返

加速的来源：
  不是"计算更少了"（计算量相同），
  而是"数据搬运更少了"（HBM→SRAM 的往返从 3 次 → 1 次）。
  在长序列下（seq>2048），内存带宽是主要瓶颈，
  减少 HBM 访问直接转化为速度提升。

额外收益：
  FlashAttention 的 online softmax 算法在数学上等价于标准 softmax，
  不会引入精度损失。
```

### Q2: AWQ 和 GPTQ 的核心区别？

```
AWQ (Activation-aware Weight Quantization):
  - 核心：保护 salient 通道
  - 方法：对重要通道做 scale 变换，让量化更精细
  - 实现：按输入通道找到最佳缩放因子 s
  - 速度：快（只需校准数据的激活统计）
  - 精度：好（对大多数 LLM，INT4 精度损失 <1%）

GPTQ (Post-Training Quantization):
  - 核心：误差补偿
  - 方法：逐列量化，未量化列吸收已量化列的误差
  - 实现：需要 Hessian 矩阵的逆
  - 速度：慢（需要计算 Hessian + 逐列优化）
  - 精度：略优于 AWQ（在极端低 bit 下优势更明显）

选择建议：
  - 快速部署 → AWQ
  - 追求极致精度 → GPTQ
  - 现代框架（vLLM, llama.cpp）两者都支持
```

---

## 11. 本章学习清单

- [ ] 能写出对称量化的公式 `q = round(x / s)`，并手算一个例子
- [ ] 能区分 per-tensor / per-channel / per-token / per-group 量化
- [ ] 能画出 ncnn 的 Quantize → INT8 Conv → Requantize 数据流
- [ ] 能解释 Requantize 中 `scale_in` 和 `scale_out` 的含义
- [ ] 能对比 AWQ 和 GPTQ 的核心思想差异
- [ ] 能画出 FlashAttention 的分块计算流程图
- [ ] 能解释 Online Softmax 的 correction 公式
- [ ] 能列出至少 4 种图优化 Pass 并举例
- [ ] 能解释 Conv-BN 融合的数学推导
- [ ] 能根据 profiling 数据诊断瓶颈并提出优化方向
- [ ] 能说明为什么 INT8 和 Vulkan 可能互斥

---

*上一篇: [05_execution_scheduler.md](05_execution_scheduler.md) — Scheduler 与 Continuous Batching*
*下一篇: [07_ncnn_to_vllm_comparison.md](07_ncnn_to_vllm_comparison.md) — 端侧 vs 服务端推理系统*
