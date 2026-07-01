# 02｜计算与算子层：从"认识算子"到"理解执行代价"

> 模型最终不是以"Transformer""ResNet"的形式在硬件上运行，而是被拆成数百到数千个底层算子。AI Infra 的第一步，就是把模型结构还原成**可执行的算子图**，并理解每个算子的**真实执行代价**。

---

## 1. 从模型到算子：一个 Transformer Block 的解剖

### 1.1 结构图 vs 算子图

```
你看到的模型结构（论文中的图）:

    ┌─────────────────────────────────────────┐
    │           Transformer Block              │
    │                                         │
    │   x → RMSNorm → Attention → + →         │
    │                  ↓            ↓          │
    │               RMSNorm → SwiGLU → + → out │
    └─────────────────────────────────────────┘

实际上在 ncnn param 文件中（每个方框 = 一个算子）:

    输入 [seq, 1024]
      │
      ├─ RMSNorm(rmsn_196) ────→ Split ──→ Gemm(Q) ──→ Reshape ──→ RMSNorm(QK-Norm)
      │                                     └─→ Gemm(K) ──→ Reshape ──→ RMSNorm(QK-Norm)
      │                                     └─→ Gemm(V) ──→ Reshape
      │                                         ↓
      │                                    Permute → RotaryEmbed(RoPE)
      │                                    ExpandDims → Tile (GQA expand)
      │                                         ↓
      │                                    SDPA (with KV Cache)
      │                                         ↓
      │                                    Permute → Reshape → Gemm(O)
      │                                         ↓
      ├─── Add (残差 1) ←──────────────────────┘
      │
      ├─ RMSNorm ──→ Split ──→ Gemm(gate) ──→ Swish
      │                      └─→ Gemm(up)
      │                              ↓
      │                         Mul(gate × up) → Gemm(down)
      │                              ↓
      ├─── Add (残差 2) ←───────────┘
      │
     输出

数一下: 1个 Transformer Block = 约 35 个 ncnn 算子
       Qwen3-0.6B 28 层 = 28 × 35 ≈ 980 + 辅助 = 1017 个算子!
```

### 1.2 算子分类全景图

```
                         深度学习算子
                              │
        ┌──────────┬──────────┼──────────┬──────────┐
        │          │          │          │          │
    逐元素运算   归约运算    矩阵运算    空间运算    序列运算
        │          │          │          │          │
   Add,Mul,Div  Sum,Mean   MatMul     Conv2D    Attention
   Exp,Log      Max,Min    GEMM       Pooling   RoPE
   ReLU,Swish   Softmax    InnerProduct Interp  Embedding
   Sigmoid      L2Norm     EinSum     Padding   Sampling
                              │
                    ┌─────────┼─────────┐
                    │         │         │
                 归一化    激活函数   张量操作
                    │         │         │
               BatchNorm  ReLU     Reshape
               LayerNorm  GELU     Permute
               RMSNorm    SwiGLU   Concat/Split
               GroupNorm  Sigmoid  Tile/Gather
```

---

## 2. 每类算子的计算特征与优化重点

### 2.1 逐元素算子（Element-wise）

```
特点:
  - 输入输出同形
  - 每个元素独立计算，天然并行
  - FLOPs 很低（每个元素 1~10 次运算）
  - 但内存访问 = 读整个输入 + 写整个输出

瓶颈来源:
  ┌──────────────────────────────────────┐
  │ 为什么逐元素算子可能比 GEMM 还慢？   │
  │                                      │
  │ Add [1024, 1024]:                    │
  │   计算: 1024² = 1M 次加法 ≈ 0.001ms │
  │   内存: 读 4MB + 写 4MB ≈ 0.1ms     │
  │   → 99% 的时间在等内存！             │
  │                                      │
  │ MatMul [1024, 4096] × [4096, 1024]: │
  │   计算: 2 × 1024² × 4096 ≈ 8.6G ops │
  │   内存: 读 16MB + 写 4MB ≈ 0.3ms    │
  │   → 大部分时间在计算                 │
  └──────────────────────────────────────┘

优化策略:
  1. 算子融合: Add → ReLU → Add 3个kernel → 1个kernel
     减少中间结果的读写
  2. 向量化: SIMD 一次处理 4/8/16 个元素
  3. 内存布局: 确保连续访问（coalesced access）
```

### 2.2 归约算子（Reduction）

```
特点:
  - 沿指定维度将多个值合并为一个
  - Reduce dimension 上的元素有依赖关系
  - 需要并行归约算法（tree reduction）

Softmax: 归约算子中最复杂的
  Step 1: max(x) — 沿维度归约找最大值（数值稳定）
  Step 2: exp(x − max) — 逐元素（无依赖）
  Step 3: sum(exp(x − max)) — 归约求和（有依赖）
  Step 4: exp(x − max) / sum — 逐元素除（无依赖）

  归约步骤通常串行，是瓶颈所在。
```

### 2.3 矩阵算子（GEMM / MatMul）

```
这是神经网络的最核心计算——占了 80%+ 的计算量。

MatMul: C[M,N] = A[M,K] × B[K,N]
  计算量: 2 × M × N × K FLOPs
  访存量: (M×K + K×N + M×N) × sizeof(dtype) bytes

Tiling 的重要性:
  大矩阵无法一次性放入 cache
  → 切成小块，每个块完整计算完再换下一个
  → 最大化数据复用，减少 cache miss

GPU 上的优化层次:
  Level 1: 全局内存 → 共享内存 (tiling by warp)
  Level 2: 共享内存 → 寄存器 (tile by thread)
  Level 3: Tensor Core 指令 (每个 cycle = 一个 16×16 块)

为什么 1×1 卷积比 3×3 卷积更适合 GPU？
  1×1: 等价于每个空间位置的 MatMul
       → 可以直接用 GEMM kernel
  3×3: im2col 膨胀 9 倍内存
       → 或者用直接卷积（访存不规则）
```

### 2.4 归一化算子

```
LayerNorm / RMSNorm 在现代 Transformer 中每层至少出现 2 次:

计算步骤:
  RMSNorm:  mean(x²) → rsqrt → scale
  LayerNorm: mean(x) → var(x) → normalize → scale + shift

每个元素的 FLOPs 极少（5~10 FLOPs）
但需要:
  - 读整个输入（归约求 mean）
  - 写整个输出

→ memory-bound 算子

融合机会:
  RMSNorm + Linear: 可以融合！
  先归一化再做矩阵乘法 → 在读取数据的同时完成归一化
  ncnn 和 vLLM 都有这类融合优化
```

---

## 3. Roofline Model 实战：判断瓶颈

### 3.1 计算强度的计算

```
计算强度 = FLOPs / (字节访问量)

FLOPs: 浮点运算次数
字节访问量: 读输入 + 写输出 + 读权重的总字节数

例子——Qwen3 的 Q 投影 (GEMM):
  输入: [1, 1024]  (decode 阶段)
  权重: [1024, 2048]
  输出: [1, 2048]

  FLOPs = 2 × 1 × 2048 × 1024 = 4,194,304 ≈ 4.2M
  字节访问 = (1024 + 2048 + 1024×2048) × 4 bytes
           = (1024 + 2048 + 2,097,152) × 4
           ≈ 8.4 MB
  计算强度 = 4.2M / 8.4M ≈ 0.5 FLOPs/Byte

对比 A100 的 roofline:
  峰值算力: 312 TFLOPS (FP16 Tensor Core)
  内存带宽: 2039 GB/s
  Roofline 拐点: 312T / 2039G ≈ 153 FLOPs/Byte

  0.5 << 153 → 严重 memory-bound!
  (decode 阶段，batch=1 的 GEMM)

例子——Prefill 阶段的 Q 投影:
  输入: [512, 1024]  (更大的 batch)
  权重: [1024, 2048]
  输出: [512, 2048]

  FLOPs = 2 × 512 × 2048 × 1024 ≈ 2.1G
  字节访问 ≈ (512×1024 + 2048×1024 + 512×2048) × 4 ≈ 14 MB
  计算强度 = 2100M / 14M ≈ 150 FLOPs/Byte

  150 ≈ 153 拐点 → compute-bound!
```

### 3.2 各算子的瓶颈类型速查

```python
# 基于实际 profiling 的分类

OPERATOR_PROFILE = {
    # Memory-bound (< 10 FLOPs/Byte)
    "memory_bound": {
        "Add/Mul/Div":         "逐元素 → 每个元素只做 1 次运算，读+写",
        "ReLU/GELU/Swish":     "逐元素激活 → 同逐元素",
        "LayerNorm/RMSNorm":   "需要归约统计，但仍是 memory 主导",
        "Dropout":             "仅 mask 操作",
        "Reshape/Permute":     "零 FLOPs，纯内存操作",
        "Concat/Split":        "零 FLOPs，纯内存操作",
        "RoPE":                "逐元素旋转，memory-bound",
    },

    # Intermediate (10-100 FLOPs/Byte)
    "intermediate": {
        "Softmax":             "归约+exp，内存和计算相当",
        "Conv3×3 (small batch)": "权重复用不够，偏向 memory",
        "SwiGLU":              "Mul+Swish 都是逐元素，但跟在 GEMM 后可融合",
    },

    # Compute-bound (> 100 FLOPs/Byte)
    "compute_bound": {
        "MatMul (大矩阵)":     "每个权重元素复用多次",
        "Conv1×1 (large batch)": "等价于 MatMul",
        "Prefill Attention":   "QK^T 是大矩阵乘法",
        "Conv3×3 (large batch)": "batch 够大时权重复用率上升",
    },
}
```

---

## 4. 逐元素算子的"隐藏成本"：Kernel Launch

### 4.1 为什么小算子多反而慢？

```
GPU 上每个 kernel launch 有固定开销：
  - CUDA: ~5-10 μs per launch
  - Vulkan: ~10-50 μs per launch
  - CPU: <1 μs (线程同步开销)

Qwen3-0.6B decoder 有 1017 个算子。
如果 GPU 每个 launch 10 μs：
  1017 × 10 μs ≈ 10 ms 纯 launch 开销

对比：Prefill 总共可能只需 50 ms
  → 20% 的时间浪费在 launch 上！

这就是为什么需要算子融合：
  融合后算子数: 1017 → ~500 → 5 ms launch 开销
```

### 4.2 Kernel Launch vs Kernel Execution

```
单个小算子（如 Add）的时间分解:

  CPU 端:                             GPU 端:
  ┌─ 准备参数       0.5 μs ─┐
  │─ 设置 descriptor 0.3 μs │
  │─ 提交到 queue    0.2 μs │        ┌─ GPU 接收  1 μs ─┐
  │─ CPU 继续执行    0.1 μs │        │─ 读输入    5 μs   │
  │                          │   →    │─ 计算      0.5 μs │
  │   Total: ~10 μs          │        │─ 写输出    4 μs   │
  └──────────────────────────┘        └─ Total: ~10.5 μs ─┘

  Kernel 执行本身只需 0.5 μs！
  但 launch 开销 + 内存读写 = 20 μs
  → 97.5% 的时间花在"非计算"上！
```

---

## 5. 算子图分析实战：Qwen3 Decoder 的 1017 个算子

### 5.1 算子分布统计

从 ncnn param 文件分析 Qwen3-0.6B 的 1017 个算子：

```
算子类型          数量    占比     主要用途
────────────────────────────────────────────
Gemm              168     16.5%    Q/K/V/O 投影 + gate/up/down
RMSNorm            86      8.5%    注意力和 MLP 前的归一化 (含 QK-Norm)
BinaryOp (Add)     84      8.3%    残差连接 + 其他二元运算
Permute            84      8.3%    维度重排
Reshape           112     11.0%    头拆分/合并
RotaryEmbed        56      5.5%    RoPE 位置编码
SDPA               28      2.8%    缩放点积注意力
Split              84      8.3%    特征分支
ExpandDims         56      5.5%    GQA 头扩展
Tile               56      5.5%    GQA 头复制
Swish              28      2.8%    SwiGLU 激活
Mul (BinaryOp)     28      2.8%    SwiGLU gate×up
Input/Output       60+     5.9%    输入输出/缓存节点
其他               47      4.6%    Clip, Concat, ...
────────────────────────────────────────────
总计              ~1017    100%
```

### 5.2 前 5 个占比最高的算子

```
1. Gemm (168 个, 16.5%)
   所有带权重的操作
   每层: Q(2048→1024) + K(1024→1024) + V(1024→1024) + O(2048→1024)
        + gate(1024→3072) + up(1024→3072) + down(3072→1024)
   = 7 个 per layer × 24 层 (有效 attention 层)

2. Reshape (112 个, 11.0%)
   几乎全是零 FLOPs 的维度重排
   主要用于头拆分: Gemm 输出是展平的 [heads×dim, seq]
                 Reshape 变成 [dim, heads, seq]

3. RMSNorm (86 个, 8.5%)
   每层 3 个（pre-attention, QK-Norm×2, pre-MLP）
   + 最终输出归一化

4. Split (84 个, 8.3%)
   每层至少 2 个 Split：
   - 将 RMSNorm 输出分给 Q/K/V 三条路径
   - 将残差分支分给主路径和 skip connection

5. Permute (84 个, 8.3%)
   NCHW ↔ NHWC 之间的维度重排
   为 SDPA 准备正确的输入格式
```

### 5.3 一个 Block 的完整执行时间估算

```
Qwen3-0.6B 单层 Decoder, decode 阶段, batch=1:

算子            FLOPs        内存访问      估算时间(CPU)  估算时间(GPU)
────────────────────────────────────────────────────────────────────
RMSNorm×3       ~50K         ~24 KB        ~15 μs         ~5 μs
Gemm×7          ~60M         ~30 MB        ~800 μs        ~50 μs
Reshape×4       ~0           ~0 (view)     ~1 μs          ~1 μs
Permute×3       ~0           ~0 (reorder)  ~2 μs          ~2 μs
RotaryEmbed×2   ~10K         ~8 KB         ~10 μs         ~5 μs
ExpandDims×2    ~0           ~0            ~1 μs          ~1 μs
Tile×2          ~0           ~8 KB         ~5 μs          ~3 μs
SDPA×1          ~5M          ~500 KB       ~200 μs        ~30 μs
Mul×1           ~3K          ~12 KB        ~5 μs          ~3 μs
Swish×1         ~3K          ~12 KB        ~5 μs          ~3 μs
Add×3           ~3K          ~12 KB        ~10 μs         ~5 μs
────────────────────────────────────────────────────────────────────
Total/层        ~65M         ~31 MB        ~1.05 ms       ~110 μs

28 层总计       ~1.8G        ~870 MB       ~29 ms         ~3.1 ms

关键观察:
  - CPU 上每层 ~1ms，28 层 ~29ms → ~34 tokens/s
  - GPU 上每层 ~110μs，28 层 ~3.1ms → ~320 tokens/s (理论)
  - 实际 GPU 更慢：因为多次 kernel launch 和 HBM 访问开销
```

---

## 6. 算子选择：同一个算子，不同硬件不同实现

### 6.1 MatMul 的多种实现

```
同一个数学运算 y = W × x，不同场景选不同实现：

场景                    实现                 硬件
───────────────────────────────────────────────────────────
CPU, 大矩阵             OpenBLAS/MKL         x86 AVX512
CPU, 小矩阵             Direct loop          ARM NEON
GPU, batch>1            cuBLAS               CUDA
GPU, batch=1            自定义 kernel         CUDA (小矩阵优化)
Vulkan                  compute shader       Mobile GPU
端侧 CPU                量化 GEMM INT8        ARM NEON
Apple Silicon           Accelerate/MPS       Apple GPU

"选哪个实现"这个决策 = Kernel Dispatch
```

### 6.2 ncnn 的 Dispatch 链

```cpp
// ncnn 中的算子 dispatch 逻辑（以 Convolution 为例）
// src/layer/convolution.cpp → 根据设备和参数选择实现

int Convolution::create_pipeline(const Option& opt) {
    if (opt.use_vulkan_compute) {
        return new Convolution_vulkan(...);     // Vulkan GPU
    }
    // CPU 路径
#if NCNN_X86
    if (use_winograd && kernel == 3 && stride == 1)
        return new Convolution_3x3_winograd(...); // x86 Winograd
    if (kernel == 1)
        return new Convolution_1x1(...);          // 1×1 特殊路径
#endif
#if NCNN_ARM
    if (use_winograd && kernel == 3)
        return new Convolution_3x3_winograd_arm(...);
#endif
    // 通用回退
    return new Convolution_im2col_gemm(...);    // im2col + GEMM 通用
}
```

---

## 7. 学习任务（动手实践）

### 任务 1：手工分析一个 Transformer Block 的算子序列

从 ncnn param 文件中选出 Layer 0 的算子序列（约 35 个算子），为每个算子标注：

```
算子名称 | 输入 Shape | 输出 Shape | FLOPs | 内存访问 | Compute/Memory/Layout
```

### 任务 2：计算不同 batch size 下的计算强度

对 Q 投影 GEMM:
```
batch=1:      FLOPs / Bytes = ?
batch=32:     FLOPs / Bytes = ?
batch=256:    FLOPs / Bytes = ?

从 memory-bound 变为 compute-bound 的拐点 batch 是多大？
```

### 任务 3：走读 ncnn SDPA 源码

```
文件: src/layer/sdpa.cpp
任务:
  1. 画出 SDPA 的 6 个输入的数据流（Q, K_cur, V_cur, mask, past_K, past_V）
  2. 标出 KV Cache concat 发生的位置
  3. 标出 Softmax 计算的位置
  4. 说明 GQA 的 head 分组如何影响循环
```

---

*上一篇: [01_system_architecture.md](01_system_architecture.md) — 六层架构总览*
*下一篇: [03_llm_inference_pipeline.md](03_llm_inference_pipeline.md) — LLM 推理流水线*
