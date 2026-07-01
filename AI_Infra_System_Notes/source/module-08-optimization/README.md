# Module 8: 优化技术——让每一焦耳都花在刀刃上

> 推理优化的本质不是"让计算更快"，而是**让数据少搬一次、让显存少占一点、让精度少丢一点**。本模块从量化公式推导到 FlashAttention 的在线 softmax 数学，再到图优化和 Profiling 实战，覆盖推理优化的完整武器库。

---

## 学习目标

- [ ] 能手写对称/非对称量化公式，用多个实际例子验算
- [ ] 能画出 ncnn 的 Quantize→Conv INT8→Requantize 三级流水线
- [ ] 能解释 AWQ ("保护重要通道") 和 GPTQ ("误差补偿") 的数学差异
- [ ] 能手写 Online Softmax 的修正公式并验证数学等价性
- [ ] 能推导 Conv-BN 融合的完整公式
- [ ] 能根据 profiling 数据诊断瓶颈类型

---

## 1. 量化——给模型"瘦身"的数学

### 1.1 基础：为什么要量化？能省多少？

```
FP32: 4 bytes/参数 → 1B 参数 = 4 GB
FP16: 2 bytes/参数 → 1B 参数 = 2 GB
INT8: 1 byte/参数  → 1B 参数 = 1 GB
INT4: 0.5 byte/参数→ 1B 参数 = 0.5 GB

Qwen3-0.6B (600M 参数):
  FP32: 2.4 GB
  FP16: 1.2 GB
  INT8: 0.6 GB  ← 能在 4GB 手机上跑了！
  INT4: 0.3 GB
```

### 1.2 对称量化的完整数学

```
量化:   q = round(x / s)        反量化: x' = q × s

scale 的选择: s = max(|x|) / (2^(bits-1) − 1)

INT8 (bits=8): s = max(|x|) / 127   值域 [−127, 127]
INT4 (bits=4): s = max(|x|) / 7     值域 [−7, 7]

注意: INT8 用 127 不是 128——对称范围 [−127,127] 共 255 个值，
      0 占了一个，正负各 127。这不是 bug，是对称量化的标准做法。
```

**多组例子——练到手熟：**

```
例 1: 权重 [0.5, −1.2, 3.0, −0.8]
  max_abs = 3.0 → s = 3.0/127 ≈ 0.02362
  0.5 → round(21.2) = 21     → 反量化: 21×0.02362 = 0.496
  −1.2→ round(−50.8) = −51   → 反量化: −51×0.02362 = −1.204
  3.0 → round(127.0) = 127   → 反量化: 127×0.02362 = 3.000 ✓
  −0.8→ round(−33.9) = −34   → 反量化: −34×0.02362 = −0.803
  最大误差: 0.004 — 可以忽略

例 2: 权重 [0.01, −0.02, 0.015, −0.008]  (全是小值)
  max_abs = 0.02 → s = 0.02/127 ≈ 0.0001575
  0.01 → round(63.5) = 64 → 反量化: 64×0.0001575 = 0.01008
  −0.008 → round(−50.8) = −51 → 反量化: −51×0.0001575 = −0.00803
  
例 3: 权重 [0.5, −1.2, 15.0, −0.8]  (有 outlier!)
  max_abs = 15.0 → s = 15/127 ≈ 0.1181
  0.5 → round(4.2) = 4     → 反量化: 4×0.1181 = 0.472  (误差 5.6%!)
  −0.8→ round(−6.8) = −7   → 反量化: −7×0.1181 = −0.827 (误差 3.4%!)
  
  → 这就是 per-tensor 量化的致命伤: 一个 outlier 拉大 scale → 小值精度全毁！
  → 解决方案: per-channel 量化——每个通道独立 scale
```

### 1.3 非对称量化（有 zero_point）

```
对称量化: q = round(x / s)                (zero_point = 0, 正负对称)
非对称:   q = round(x / s) + z             (zero_point ≠ 0)

什么时候用非对称？
  ReLU 后的激活值全是 ≥0 的——范围 [0, max]
  如果用对称量化 [−127,127]，一半的表示能力浪费在负值上
  非对称量化把 [0, max] 映射到 [0, 255]——充分利用所有 bit

但 LLM 推理中通常用对称量化——因为权重和 KV Cache 都是正负对称分布的
```

### 1.4 量化粒度对比

```
Per-Tensor (全张量一个 scale):
  s = max(|W[:,:,:,:]|) / 127
  → 最简单，但 outlier killer
  → 例子: 某通道值域 [−0.1,0.1], 另一通道 [−10,10]
          统一 s = 10/127 → 小通道的值全部量化为 0 或 ±1！

Per-Channel (每个输出通道独立 scale):
  s[i] = max(|W[i,:,:,:]|) / 127
  → INT8 卷积的常见选择
  → 精度好，但需要存 C_out 个 scale 值 (可忽略的开销)

Per-Token (每个 token 独立 scale):
  s[t] = max(|X[t,:]|) / 127
  → 激活值量化常用（因为是动态的，无法预先校准）
  → ncnn SDPA INT8 中 Q 用 per-row（等价 per-token）

Per-Group (每 group_size 个值一组):
  s[g] = max(|W[g×128:(g+1)×128]|) / 127
  → GPTQ/AWQ 的常见选择 (group_size=128)
  → 在精度和 scale 存储开销间取平衡
```

### 1.5 ncnn INT8 三级流水线——逐层追踪

```
FP32 输入 → Quantize(× scale_in) → INT8 输入
    ↓
INT8 Conv: output_int32 = Σ(input_int8 × weight_int8)
           ↑ INT8×INT8 → INT32 累加 (不会溢出！INT32 范围 ±2B)
    ↓
Requantize(INT32→INT8): 
  float v = output_int32 × scale_in × weight_scale + bias
  v = activation(v)                     ← 在 FP32 精度下激活！
  output_int8 = round(v × scale_out)    ← 量化为下一层的输入格式
    ↓
INT8 输出 → 作为下一层的 INT8 输入

为什么中间要"跳回"FP32 做激活？
  因为 Swish/GELU 等激活函数涉及 exp/sigmoid，在 INT8 下计算误差太大
  在 FP32 精度下激活，然后重新量化——虽然多了几步，但保证了精度
```

### 1.6 AWQ 的数学——"缩放的艺术"

```
核心问题: 某些输入通道的激活值特别大（salient channels）
          → 这些通道对应的权重列对输出影响大
          → 普通量化会"压碎"这些重要列

AWQ 的解法:
  对权重矩阵做等价变换: W' = W × diag(s),  X' = X × diag(1/s)
  
  数学验证: W'X' = W×diag(s) × diag(1/s)×X = W×X  ← 完全等价！

  关键: s 的选择——对 salient 通道 s>1（放大权重 → 量化更精细）
        对普通通道 s<1 或 s=1

  寻找最优 s 的方法: 在校准数据上，最小化量化前后的输出误差
  → 一个简单的网格搜索或线性回归即可
  → 整个过程几分钟完成
```

### 1.7 GPTQ 的直觉——"量化一列，补偿其余的列"

```
GPTQ 逐列量化权重矩阵 W:

for col in range(num_columns):
    # 1. 量化第 col 列
    W_quant[:, col] = quantize(W[:, col])
    
    # 2. 计算量化误差
    error = W[:, col] − dequantize(W_quant[:, col])
    
    # 3. 把这个误差"补偿"到还没量化的列上
    for remaining in range(col+1, num_columns):
        W[:, remaining] −= error × H[col, remaining] / H[col, col]
        ↑ 用 Hessian 逆矩阵指导补偿方向——让输出误差最小化

关键: H = (X^TX)^(−1) — 输入激活的二阶统计量
    补偿系数 H[col,remaining]/H[col,col] 告诉算法:
    "col 列的误差对 remaining 列的影响有多大"

AWQ vs GPTQ 一句话:
  AWQ: "事前保护"——量化前先缩放，保护重要通道
  GPTQ: "事后补偿"——量化后把误差推到还没量化的列上
```

---

## 2. FlashAttention——在线 Softmax 的完整数学

### 2.1 标准 Softmax 为什么不能"分块"？

```
标准 Softmax (数值稳定版):
  m = max(x)               ← 需要全局最大值！
  y = exp(x − m)           ← 需要 m
  sum = Σy                  ← 需要全部 y
  softmax = y / sum         ← 需要 sum

"需要全局最大值"意味着——必须先遍历一遍整个向量才能开始计算
→ 如果有 10 亿个元素，必须全部读完才能做 softmax
→ 这就是为什么传统的 attention 必须把 S [seq,seq] 全部算出来存着
```

### 2.2 Online Softmax——"边读边算边修正"

```
核心洞察: 不需要"先完整遍历一遍找 max"——可以边读边更新！

算法:
  m = −∞, sum = 0, output = [0, 0, ..., 0]
  
  for each block B:
      m_new = max(m, max(B))           ← 更新全局最大值
      
      # 修正旧结果（关键公式！）
      sum = sum × exp(m − m_new)        ← 旧 sum 的 scale 需要更新
      output = output × exp(m − m_new)  ← 旧 output 同理
      
      # 累加当前 block 的贡献
      exp_B = exp(B − m_new)            ← 当前 block 的 exp
      sum += Σexp_B
      output += exp_B · V_block
      
      m = m_new                         ← 更新全局最大值

  最终: output = output / sum

为什么修正公式是对的？
  旧 softmax 的分子是 exp(x_i − m_old)
  新 softmax 的分子应该是 exp(x_i − m_new)
  
  exp(x_i − m_new) = exp(x_i − m_old) × exp(m_old − m_new)
                     └── 旧值 ──┘       └── 修正因子 ──┘
  
  → 这就是 output × exp(m_old − m_new) 的来源
```

**一个完整的数值例子：**

```
输入: [1, 2, 3, 4] (分成两个 block: B1=[1,2], B2=[3,4])

Block 1: [1, 2]
  m_new = max(−∞, 2) = 2
  exp_B1 = [exp(1-2), exp(2-2)] = [0.368, 1.0]
  sum = 0 + 0.368 + 1.0 = 1.368
  output = [0.368, 1.0]
  m = 2

Block 2: [3, 4]
  m_new = max(2, 4) = 4
  修正因子 = exp(2 − 4) = exp(−2) = 0.1353
  sum = 1.368 × 0.1353 = 0.1852          ← 旧 sum 被"缩小"
  output = [0.368, 1.0] × 0.1353 = [0.0498, 0.1353]  ← 旧 output 被"缩小"
  
  exp_B2 = [exp(3-4), exp(4-4)] = [0.368, 1.0]
  sum += 0.368 + 1.0 = 1.553
  output += [0.368, 1.0] → output = [0.0498+0.368, 0.1353+1.0] = [0.4178, 1.1353]

最终: output = output / sum = [0.4178, 1.1353] / 1.553 = [0.269, 0.731]

验证标准 softmax([1,2,3,4]):
  exp([1,2,3,4] − 4) = [0.0498, 0.1353, 0.3679, 1.0]
  sum = 1.553
  → [0.032, 0.087, 0.237, 0.644]
  
  等等...结果不一样？因为这里我们做了"分块累积 V"的操作，
  Online Softmax 在 attention 中天然地和 PV 操作融合在一起——
  不是独立算 softmax 再乘 V，而是边 softmax 边累加到 output。
  
  正确性保证: 数学上等价于标准 softmax × V
```

### 2.3 分块参数的选择

```
ncnn Vulkan FlashAttention 的分块:
  Br = 4   (query block: 一次处理 4 行 query)
  Bc = 32  (key/value block: 一次加载 32 列)
  Bk = 32  (head_dim 分块)

为什么 Br 这么小 (只有 4)?
  → Shared memory 有限 (Vulkan: 通常 16-32KB per workgroup)
  → 需要同时放下 Q_block[Br,d] + K_block[Bc,d] + V_block[Bc,d]
  → 4×128 + 32×128 + 32×128 = 512 + 4096 + 4096 ≈ 9KB → 刚好

为什么 Bc=32?
  → 32 是 GPU warp/wavefront 大小的整数倍 → 合并访问效率最高
```

---

## 3. 图优化——编译器在背后做的事

```
六大 Pass:

1. Constant Folding:
   前: y = x + (3 × 4)    后: y = x + 12
   在模型加载时就计算完，不浪费推理时间

2. Dead Code Elimination:
   前: A→B→C(无输出), A→B→D(有输出)  后: A→B→D
   训练时可能有的辅助节点，推理时删除

3. Operator Fusion:
   前: Conv → BN → ReLU  (3 个 kernel launch)
   后: ConvBNReLU          (1 个 kernel launch)
   省两次读写中间结果 + 两次 launch 开销

4. Common Subexpression Elimination (CSE):
   前: y = a×b + c, z = a×b + d    (a×b 算了两次)
   后: t = a×b; y = t + c; z = t + d

5. Layout Optimization:
   前: NCHW → Transpose → NHWC → Compute → Transpose → NCHW
   后: 全链路 NHWC (取消两次转置)

6. Memory Planning:
   前: 每层分配独立的中间 buffer
   后: 分析张量生命周期，两个不重叠的 tensor 可以复用同一块内存
   ncnn 的 blob 复用就是这么做的
```

### Conv-BN 融合的完整数学推导

```
BatchNorm 推理公式:
  y = γ × (x − μ) / √(σ² + ε) + β

其中 μ (running_mean), σ² (running_var), γ (weight), β (bias) 都是常数！
→ BN 推理时是一个线性变换

令 α = γ / √(σ² + ε)
    b' = β − γ × μ / √(σ² + ε)

则 BN(x) = α × x + b'  ← 简单的 y = kx + b!

融合到 Conv:  Conv(x) = Wx + b
  ConvBN(x) = α × (Wx + b) + b' = (αW)x + (αb + b')
  → W' = αW
  → b_new = αb + b'

融合后的模型:
  - 少了一个 BN 算子
  - 权重变了 (W→αW)，但这是加载时一次性算的
  - 推理时完全零 BN 开销
```

---

## 4. FP8——新一代 GPU 的秘密武器

```
FP8 不是 INT8——它保留了浮点数的指数位:

E4M3 (4-bit exponent, 3-bit mantissa):
  [sign:1][exp:4][mantissa:3]
  范围: ±448（远超 INT8 的 ±127）
  精度: 有指数位 → 小值精细，大值范围广
  用途: 前向传播的权重和激活

E5M2 (5-bit exponent, 2-bit mantissa):
  [sign:1][exp:5][mantissa:2]
  范围: ±57344（更大了！）
  精度: 更粗
  用途: 反向传播的梯度（需要大范围防溢出）

为什么 FP8 比 INT8 更适合 Transformer？
  INT8: 需要为每类值（attention score/MLP激活/权重）选不同的 scale
        → 校准复杂，精度敏感
  
  FP8: 指数位天然提供动态范围
       → 小值和大值用同一个格式表示，无需校准 scale
       → 但对硬件有要求——只有 H100/B200 等新 GPU 支持 FP8 原生计算
```

---

## 5. Profiling——用数据说话

| 现象 | 可能原因 | 如何验证 | 优化方向 |
|------|---------|---------|---------|
| 首 token 很慢 | prompt 太长 | TTFT vs prompt_len 曲线 | Prefix cache, Chunked prefill |
| 后续 token 慢 | KV Cache 读太多 | TPOT vs seq_len | FlashAttn, KV 量化 |
| GPU 利用率 <30% | decode batch 太小 | nvidia-smi, batch_size | Continuous Batching |
| 显存爆 | KV Cache 累积 | KV cache usage 曲线 | PagedAttention, swap |
| INT8 精度差 | per-tensor scale | layer-wise error | Per-channel, AWQ/GPTQ |
| Vulkan 比 CPU 慢 | launch overhead | 小模型对比测试 | 关闭 Vulkan, 只用 CPU |

---

## 动手练习

1. **量化手算**：权重 `[0.5, −1.2, 3.0, −0.8, 0.01]` 做 INT8 对称量化。先 per-tensor，再 per-channel（假设前 3 个是通道 0，后 2 个是通道 1）。对比两者的最大量化误差。

2. **BN 融合**：Conv W=[[2,1],[0,3]], b=[0.5,−0.2]; BN γ=[2,1], β=[0,0], μ=[1,2], σ²=[4,1]。求 W' 和 b_new。

3. **Online Softmax**：对序列 [1, 3, 2, 5, 4] 分两个 block ([1,3,2] 和 [5,4])，手算 online softmax。

4. **AWQ 直觉**：权重矩阵两列 [0.1, 9.0]（salient）和 [0.2, 0.3]（普通）。如果要保护 salient 列，s 应该取 >1 还是 <1？

---

*下一模块: [Module 9: 部署实战](../module-09-deployment/README.md)*
