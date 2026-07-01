# 01｜AI Infra 系统架构：六层模型详解

> 每一层都是对下层能力的封装，同时也是对上层需求的支撑。理解这六层的关系，就能理解整个 AI Infra 推理系统。

---

## 1. 六层架构全景图

```
┌──────────────────────────────────────────────────────────────────┐
│  第 6 层: 服务与部署层 (Serving & Deployment)                      │
│  API / Streaming / Tool Calling / Monitoring / Edge / Cloud       │
│  "把模型变成用户可以请求的服务"                                    │
├──────────────────────────────────────────────────────────────────┤
│  第 5 层: 运行时与系统层 (Runtime & System)                        │
│  Scheduler / Batching / Memory Management / PagedAttention        │
│  Graph Optimization / Kernel Dispatch / Tensor Parallel           │
│  "决定哪些请求一起执行，KV Cache 如何管理"                         │
├──────────────────────────────────────────────────────────────────┤
│  第 4 层: 模型与推理流水线层 (Inference Pipeline)                  │
│  Prompt Template → Tokenizer → Embed → Decoder → KV Cache         │
│  → Projection → Sampling → Token to Text                          │
│  "LLM 推理不是一次 forward，而是一个自回归循环"                    │
├──────────────────────────────────────────────────────────────────┤
│  第 3 层: 组合结构层 (Architecture Blocks)                         │
│  Transformer Block / CNN Block / Mamba SSM / MoE / Multi-Modal    │
│  "算子的组合形成模型结构"                                          │
├──────────────────────────────────────────────────────────────────┤
│  第 2 层: 计算与算子层 (Compute & Operators)                       │
│  MatMul / Conv / Attention / RoPE / Norm / Activation / Sampling  │
│  "模型被编译成数千个基础算子的执行序列"                            │
├──────────────────────────────────────────────────────────────────┤
│  第 1 层: 硬件基础层 (Hardware)                                    │
│  CPU (SIMD/Cache) / GPU (CUDA/TensorCore) / Vulkan / NPU / Memory │
│  "一切计算的物理基础"                                              │
└──────────────────────────────────────────────────────────────────┘

依赖方向:  第 6 层 → 第 5 层 → 第 4 层 → 第 3 层 → 第 2 层 → 第 1 层
影响方向:  第 1 层 → 第 2 层 → ... → 第 6 层
           ↑ 下层的特性决定上层的优化策略
```

---

## 2. 第 1 层：硬件基础层

### 2.1 硬件决定了推理系统的上限

```
不同硬件的核心差异:

CPU (x86/ARM):
  - 大缓存 (L3 可达 64MB+)
  - SIMD 向量化 (AVX512 一次处理 16 个 FP32)
  - 低延迟 (无 kernel launch 开销)
  - 适合: 小 batch、端侧推理、latency-critical

GPU (NVIDIA):
  - 高并行度 (A100: 6912 CUDA cores + 432 Tensor Cores)
  - 高带宽 HBM (A100: 2TB/s)
  - kernel launch 有固定开销 (~5-10μs)
  - 适合: 大 batch、高吞吐、prefill 密集计算

Vulkan GPU (Mobile):
  - 跨平台 (Android/iOS/Desktop)
  - 无 CUDA 依赖
  - Compute shader 性能不如 CUDA kernel
  - 适合: 端侧 GPU 加速

NPU (Edge):
  - 固定算子支持 (不支持所有算子)
  - 极致能效比
  - 模型转换需要工具链支持
  - 适合: 专用推理加速
```

### 2.2 内存层级决定优化策略

```
CPU 内存层级 (从快到慢):
  L1 Cache  → ~1ns,  ~32KB/core
  L2 Cache  → ~4ns,  ~256KB/core
  L3 Cache  → ~12ns, ~16MB/shared
  DDR RAM   → ~100ns, ~16GB+
  SSD       → ~100μs, ~512GB+

GPU 内存层级:
  Register  → ~0 cycles, ~256KB/SM
  SharedMem → ~20 cycles, ~164KB/SM (A100)
  L2 Cache  → ~200 cycles, ~40MB (A100)
  HBM       → ~400 cycles, ~80GB (A100)

关键认知:
  - 从 HBM 读一个数的时间 ≈ 做 200 次 FP32 乘法的时间
  - "减少显存访问" 比 "减少计算" 更重要 (对于 memory-bound 算子)
  - Tiling 的本质: 把数据留在更快的层级中反复使用
```

---

## 3. 第 2 层：计算与算子层

详见 [02_compute_operator_layer.md](02_compute_operator_layer.md)。作为架构文档，这里强调算子层的**系统视角**：

```
同一个算子在 6 个维度上的差异:

  MatMul [M,K] × [K,N]:
    FLOPs:      2MNK
    参数量:      0 (纯计算)
    Compute/Memory: 取决于 M,N,K 的大小
    Precision:   FP32/FP16/BF16/INT8/FP8
    Hardware:    CPU GEMM / cuBLAS / Vulkan shader
    Fusion:      MatMul + Bias + Activation 融合

  不同 batch size 下，同一个 MatMul 可能是 compute-bound 或 memory-bound
  → 这就是为什么要 profiling 而不是猜测
```

---

## 4. 第 3 层：组合结构层

### 4.1 Transformer Block 是当前绝对主流

```
一个现代 LLM Block (Pre-Norm, RMSNorm, GQA, SwiGLU, RoPE):

  x ─────────────────────────────────────┐
    │                                     │
    ├─ RMSNorm                            │
    │   └─ Split ──→ Q Proj ──→ QK-Norm ──→ RoPE ──┐
    │             ──→ K Proj ──→ QK-Norm ──→ RoPE ──┤
    │             ──→ V Proj ───────────────────────┤
    │                                               │
    │   SDPA(Q, K, V, mask, past_KV)                │
    │   └─ KV Cache 读/写                           │
    │                                               │
    │   O Proj                                      │
    │                                               │
    ├─── Add (残差 1) ←─────────────────────────────┘
    │
    ├─ RMSNorm
    │   └─ Split ──→ Gate Proj ──→ Swish ──→ Mul
    │             ──→ Up Proj ───────────────────┘
    │   └─ Down Proj
    │
    └─── Add (残差 2) → out

参数量 (Qwen3-0.6B 单层):
  Q Proj:  1024 × 2048 = 2.1M
  K Proj:  1024 × 1024 = 1.0M
  V Proj:  1024 × 1024 = 1.0M
  O Proj:  2048 × 1024 = 2.1M
  Gate:    1024 × 3072 = 3.1M
  Up:      1024 × 3072 = 3.1M
  Down:    3072 × 1024 = 3.1M
  Norms:   ~6K (RMSNorm 只有 gamma)
  ─────────────────────────
  总计:    ~15.5M / 层 × 28 层 ≈ 434M

加上 Embedding (151936 × 1024 = 155M，共享) = ~600M 参数
```

### 4.2 架构差异对推理系统的影响

| 架构特征 | 推理系统影响 |
|---------|------------|
| Pre-Norm vs Post-Norm | Pre-Norm 训练更稳定，推理无差异 |
| RMSNorm vs LayerNorm | RMSNorm 参数量减半，计算更快 |
| GQA vs MHA | GQA KV Cache 减半 |
| SwiGLU vs ReLU-FFN | SwiGLU 多一个权重矩阵,中间维度 3x (非 4x) |
| RoPE vs 可学习 PE | RoPE 支持外推,需预计算 cos/sin cache |
| QK-Norm (Qwen3) | 额外 RMSNorm 开销,但提升长上下文注意力质量 |

---

## 5. 第 4 层：模型与推理流水线层

详见 [03_llm_inference_pipeline.md](03_llm_inference_pipeline.md)。架构视角的关键认知：

```
LLM 推理 = 一个状态机

状态: KV Cache, cur_token, position_id, history
循环: embed → decoder → project → sample → output
终止: EOS 或 max_tokens

为什么不是一次 forward？
  因为每个 token 的生成依赖前面的 token → 必须串行
  KV Cache 让串行中的重复计算被消除

为什么 KV Cache 是核心？
  因为它既是 加速机制 (避免重算), 也是 瓶颈 (显存占用),
  也是 调度单元 (PagedAttention 中的 block),
  也是 复用单元 (prefix cache)
```

---

## 6. 第 5 层：运行时与系统层

详见 [04_memory_kv_cache.md](04_memory_kv_cache.md) 和 [05_execution_scheduler.md](05_execution_scheduler.md)。这里是 AI Infra 的核心：

```
这一层回答了三个核心问题:

1. 内存问题: KV Cache 如何存？如何分配？如何释放？
   → PagedAttention (04 章)

2. 调度问题: 哪些请求一起执行？新请求何时加入？
   → Continuous Batching (05 章)

3. 执行问题: 算子如何映射到 kernel？图如何优化？
   → Graph Optimization, Kernel Dispatch (06 章)
```

---

## 7. 第 6 层：服务与部署层

### 7.1 部署矩阵

```
              端侧 (Edge)                  服务端 (Cloud)
硬件        手机/树莓派/IoT               A100/H100 集群
内存        <4GB                          80GB × 8
功耗        受限                          不限
框架        ncnn / llama.cpp              vLLM / TensorRT-LLM
量化        INT8/INT4 必须                FP16/BF16 基线, FP8 可选
并发        1 用户                        1000+ 用户
延迟要求    <100ms/token (感知流畅)       <50ms TTFT
优化重点    内存占用、跨平台              吞吐、调度、GPU 利用率
```

### 7.2 服务层关键指标

```
TTFT (Time To First Token):
  = queue_time + prefill_time + first_sample_time
  用户感知: "第一句话出来的快不快？"
  目标: <200ms (端侧), <50ms (服务端)

TPOT (Time Per Output Token):
  = 平均每个 decode step 的时间
  用户感知: "后面的话出来的快不快？"
  目标: <50ms (端侧), <10ms (服务端)

Throughput (tokens/s):
  = 系统整体每秒能处理多少 token
  服务端核心: 单 GPU ~2000-5000 tokens/s (取决于模型和优化)

GPU Utilization:
  Prefill: 应该 >80%
  Decode: 通常 20-50% (受限于 KV Cache 内存带宽)
  太低 → batching 不足 / kernel 太小 / 同步开销大
```

---

## 8. 文档关系图

```
                    00_overview (总览)
                          │
                    01_system_architecture (架构图)  ← 你在这里
                          │
         ┌────────────────┼────────────────┐
         ▼                ▼                ▼
    02_operators     03_pipeline      04_kv_cache
    (算子→执行)       (推理流水线)      (内存系统)
         │                │                │
         └────────────────┼────────────────┘
                          ▼
                  05_scheduler (调度系统)
                          │
                          ▼
                  06_optimization (量化/FlashAttn/调优)
                          │
                          ▼
                  07_ncnn_to_vllm (端侧→服务端)
                          │
                          ▼
                  08_project (简历/GitHub)
                           │
                           ▼
                  09_learning_plan (学习计划)
                           │
                           ▼
                  10_glossary (术语速查)
```

---

## 9. 学习建议

初次阅读时，建议按这个顺序理解六层：

```
先看懂 1→2→3→4 四层 (模型怎么在硬件上跑起来的)
  再深入 5 层 (多请求怎么高效服务的)
    再理解 6 层 (怎么把服务交付给用户的)

重点投入时间:
  第 2 层: 理解每个算子的代价 (20%)
  第 4 层: 理解推理的完整链路 (20%)
  第 5 层: 理解调度和内存管理 (50%) ← AI Infra 的核心
  第 6 层: 理解部署选项 (10%)
```

---

*下一篇: [02_compute_operator_layer.md](02_compute_operator_layer.md) — 计算与算子层深入*
