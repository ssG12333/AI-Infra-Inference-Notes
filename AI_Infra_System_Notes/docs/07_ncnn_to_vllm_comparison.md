# 07｜从 ncnn_llm 到 vLLM：端侧推理与服务端推理的系统差异

> 你已经分析了 ncnn_llm 的完整推理流程。从 ncnn_llm 到 vLLM，不是"换一个框架"，而是一次**系统范式转换**——从"让一个模型跑起来"到"让大量请求高效运行"。

---

## 1. 一句话概括两种范式

```
ncnn_llm 的思路:
  "我有一部手机，怎么在手机上流畅地跑一个 LLM？"
  → 轻量、量化、跨平台、单用户、内存受限

vLLM 的思路:
  "我有一台 8×A100 服务器，怎么同时服务 1000 个用户？"
  → 调度、分页、batching、高并发、高吞吐
```

### 1.1 全景对比图

```
┌── ncnn_llm (端侧单用户) ──┐     ┌── vLLM (服务端多用户) ──┐
│                            │     │                            │
│  Text                      │     │  100+ Requests             │
│    ↓                       │     │    ↓                       │
│  Prompt Template           │     │  Queue + Scheduler         │
│    ↓                       │     │    ↓                       │
│  BPE Tokenizer             │     │  Batch Tokenizer           │
│    ↓                       │     │    ↓                       │
│  embed_net (查表)           │     │  Model Executor            │
│    ↓                       │     │    ├─ Prefill Worker       │
│  decoder_net (28层,单请求)  │     │    ├─ Decode Worker        │
│    ├─ KV Cache (连续Mat)    │     │    └─ Paged KV Cache       │
│    ├─ SDPA (CPU/GPU kernel) │     │       ├─ Block Allocator  │
│    └─ INT8 量化路径         │     │       ├─ Block Table      │
│    ↓                       │     │       └─ COW Fork         │
│  proj_out_net (投影)        │     │    ↓                       │
│    ↓                       │     │  Sampling                  │
│  Sampling (单token)         │     │    ↓                       │
│    ↓                       │     │  Streaming Response        │
│  Stream to user            │     │    ↓                       │
│                            │     │  API Server (/v1/chat)     │
└────────────────────────────┘     └────────────────────────────┘
```

---

## 2. 七大维度的深度对比

### 2.1 模型组织

| 维度 | ncnn_llm | vLLM |
|------|----------|------|
| **模型拆分** | 3 个独立 ncnn::Net（embed/decoder/proj） | HuggingFace 模型 + 自定义 weight loader |
| **权重格式** | ncnn .param/.bin | HF safetensors → GPU memory |
| **按需加载** | 3 个 net 按需创建 extractor | 全部加载到 GPU 显存 |
| **复用** | Embedding 和 Projection 共享 .bin 文件 | 独立权重，支持 Tensor Parallel 拆分 |
| **设计意图** | 内存峰值分段控制（端侧显存 <4GB） | 全部常驻 GPU 显存（服务器有 80GB HBM） |

### 2.2 执行模型

```
ncnn_llm:
  ┌──────────────────────────────────────────┐
  │  generate(ctx, cfg, callback)             │
  │    for step in range(max_new_tokens):    │
  │      embed_net.extract()                 │  ← 同步调用
  │      decoder_net.extract()               │  ← 包含 KV Cache I/O
  │      proj_out_net.extract()              │  ← 同步调用
  │      sample(logits)                      │  ← CPU 上采样
  │      callback(token_text)                │  ← 流式输出
  └──────────────────────────────────────────┘
  一个 while 循环 = 整个推理系统

vLLM:
  ┌──────────────────────────────────────────┐
  │  Scheduler.step()                        │
  │    → _schedule_new_requests()            │  ← 从队列选请求
  │    → _select_batch()                     │  ← 组 batch
  │    → ModelRunner.execute(batch)          │  ← 批量执行
  │        ├─ prepare_input()                │
  │        ├─ attention_with_paged_cache()   │  ← PagedAttention
  │        └─ sample_all()                   │  ← 批量采样
  │    → _update_states()                    │  ← 更新每个请求
  │    → _free_finished()                    │  ← 释放资源
  └──────────────────────────────────────────┘
  一个分布式系统 = 调度器 + 执行引擎 + 状态管理
```

### 2.3 KV Cache 管理

这已经在 [04_memory_kv_cache.md](04_memory_kv_cache.md) 中详细展开，这里做对照小结：

```
ncnn_llm:
  KVCache = vector<pair<Mat, Mat>>
  每层一对，连续存储 [head_dim, seq_len, num_kv_heads]
  Decode 时: memcpy(old_KV) + memcpy(new_token)
  → 简单但每个 step 都做全量拷贝

llama.cpp (中间态):
  Cell 池 + find_slot() 分配
  支持多序列、序列操作、cache shift
  → 灵活但仍要求连续 cell

vLLM:
  Block 池 + Block Table 映射
  每个 block 16 tokens，按需分配
  COW fork 支持 beam search 和 prefix sharing
  → 消除碎片，最大化显存利用率
```

### 2.4 调度系统

```
ncnn_llm:  无调度器
  generate() 是一个同步循环
  一次只能处理一个请求
  → 适合本地 CLI 工具

vLLM:  完整的调度器
  Scheduler 管理 waiting/running/swapped 队列
  Continuous Batching: 每 step 动态重组 batch
  Token budget: 按 token 数（而非请求数）分配资源
  Preemption: 显存不够时换出低优先级请求
  → 适合高并发服务

新增的概念（ncnn_llm 中没有）:
  - Request Lifecycle (WAITING→PREFILL→DECODING→FINISHED)
  - Chunked Prefill (长 prompt 分批处理)
  - Priority-based scheduling (优先级调度)
  - Prefix caching (自动检测和共享前缀 KV)
```

### 2.5 量化策略

```
ncnn_llm:
  静态 INT8 量化
  - 模型文件中的权重已量化为 INT8
  - Quantize/Requantize 层在 param 文件中显式存在
  - 激活值在线动态量化（per-tensor 或 per-channel）
  - INT8 路径和 Vulkan 路径可能互斥

vLLM:
  多种量化策略并存
  - AWQ/GPTQ: 模型加载时应用（权重 INT4/INT8）
  - FP8: 新一代 GPU 的硬件加速
  - KV Cache 量化: 独立配置，可不同于权重量化
  - 量化不影响 PagedAttention 的实现
```

### 2.6 服务能力

```
ncnn_llm:
  API 形式: C++ 函数调用
  Streaming: 通过 callback 函数
  并发: 单线程为主
  协议: 无标准 API 接口

vLLM:
  API 形式: OpenAI-compatible REST API
           POST /v1/chat/completions
           POST /v1/completions
  Streaming: Server-Sent Events (SSE)
  并发: 异步处理，多 worker
  协议: 标准 HTTP + SSE
  额外功能: 多模型 serving, LoRA adapter, 监控 metrics
```

### 2.7 硬件适配

```
ncnn_llm:
  CPU:  x86 AVX2/AVX512, ARM NEON, RISC-V, LoongArch
  GPU:  Vulkan (跨平台), 无 CUDA 依赖
  NPU:  无
  特点: 一个代码库支持所有平台

vLLM:
  CPU:  有限的 CPU 后端支持
  GPU:  CUDA (NVIDIA), ROCm (AMD), 部分 Intel GPU
  NPU:  部分支持
  特点: 深度依赖 CUDA 生态
```

---

## 3. 七个"概念跃迁"

从 ncnn_llm 到 vLLM，你需要完成这七个认知跃迁：

```
跃迁 1: 单请求 → 请求流
─────────────────────────
ncnn:  "这个请求怎么推理？"
vLLM:  "这些请求怎么组织、调度、服务？"

跃迁 2: 连续 KV → 分页 KV
─────────────────────────
ncnn:  "KV Cache = 一段连续内存，追加写入"
vLLM:  "KV Cache = block 池，按需分配，逻辑映射"

跃迁 3: 同步循环 → 异步调度
─────────────────────────
ncnn:  "for step in range(...): forward() + sample()"
vLLM:  "while True: schedule() + execute() + update()"

跃迁 4: 无 batch → Continuous Batching
─────────────────────────
ncnn:  "一次处理一个"
vLLM:  "每个 step 重新组 batch，完成即移出，新来即加入"

跃迁 5: 静态量化 → 多策略量化
─────────────────────────
ncnn:  "模型文件 INT8 → 直接推理"
vLLM:  "AWQ/GPTQ/FP8/KV-quant 按需组合"

跃迁 6: 函数调用 → API 服务
─────────────────────────
ncnn:  "generate(ctx, cfg, callback)"
vLLM:  "POST /v1/chat/completions → SSE streaming → JSON response"

跃迁 7: 单设备 → 分布式
─────────────────────────
ncnn:  "一个 CPU 核/一个 GPU"
vLLM:  "Tensor Parallel 多 GPU, Pipeline Parallel 多节点"
```

---

## 4. 对照学习路径

### 4.1 从 ncnn_llm 出发，映射到 vLLM 概念

```
你在 ncnn_llm 中分析的:          vLLM 中的对应概念:
─────────────────────────────────────────────────────
model.json 配置加载           →  HF config + Engine config
embed_net / decoder_net       →  Model Runner / Worker
  / proj_out_net 拆分
ncnn::Extractor::extract()    →  Model Runner 的 execute_model()
BPE Tokenizer                 →  Tokenizer (同源，但支持 batching)
Prompt Template               →  Chat Template (HuggingFace 标准)
Single Decoder Forward        →  Batch Decode (多请求拼接)
KV Cache (KVCache 结构)       →  Paged KV Cache (Block Table)
generate() 循环               →  Scheduler.step()
Sampling                      →  Sampler (批量采样)
INT8 / Vulkan 后端            →  CUDA kernel / Tensor Parallel
```

### 4.2 推荐学习顺序

```
Phase 1 (当前): ncnn_llm 源码分析 ✅
  → 目标: 理解单请求推理的完整链路

Phase 2 (下一步): 阅读 vLLM 核心概念
  1. vLLM 论文 (PagedAttention, SOSP'23)
  2. vLLM 架构博客 (官方 docs)
  3. Sequence / Block Table 数据结构

Phase 3 (深入): vLLM 源码阅读
  1. vllm/v1/engine/  → 引擎入口
  2. vllm/v1/core/scheduler.py  → 调度器
  3. vllm/v1/core/kv_cache_manager.py  → KV Cache 管理
  4. vllm/v1/worker/gpu/model_runner.py  → 模型执行
  5. vllm/attention/  → PagedAttention kernel

Phase 4 (进阶): TensorRT-LLM
  1. Graph optimization
  2. Plugin system
  3. FP8 quantization
  4. In-flight batching
```

---

## 5. 何时选择哪个？

```
选择 ncnn/ncnn_llm:
  ✅ 移动端 / 嵌入式设备
  ✅ 无 CUDA GPU（或需要跨平台 Vulkan）
  ✅ 单用户本地推理
  ✅ 内存极度受限 (<4GB)
  ✅ 需要离线部署
  ✅ 轻量依赖（无 Python/PyTorch）

选择 llama.cpp:
  ✅ 本地桌面/服务器 CPU 推理
  ✅ 需要 GGUF 格式的量化模型
  ✅ 适中的并发需求
  ✅ 需要 mmap 快速加载

选择 vLLM:
  ✅ 在线服务（API serving）
  ✅ 高并发（100+ 请求同时）
  ✅ 有 NVIDIA GPU（A100/H100）
  ✅ 需要 OpenAI-compatible API
  ✅ 需要 Continuous Batching

选择 TensorRT-LLM:
  ✅ 追求极致性能
  ✅ NVIDIA 生态深度绑定
  ✅ 需要 FP8/INT4 硬件加速
  ✅ 可以接受较长的 build 时间
```

---

## 6. ncnn_llm + llama.cpp + vLLM 协同学习法

这三者不是竞争关系，而是**互补的学习材料**：

```
难度递增:

ncnn_llm  ──→ llama.cpp ──→ vLLM ──→ TensorRT-LLM
(最简单)      (中等)        (较高)     (最高)

ncnn_llm:
  代码量: ~3000 行 C++
  适合: 入门理解 LLM 推理全链路
  学到的: Tokenizer, Embed, Decoder, KV Cache, Sampling

llama.cpp:
  代码量: ~10 万行 C/C++
  适合: 深入理解 KV Cache 管理、量化、CPU 优化
  学到的: GGUF 格式, cell-based KV cache, mmap, SIMD

vLLM:
  代码量: ~10 万行 Python/C++
  适合: 理解服务端调度、PagedAttention、Continuous Batching
  学到的: Scheduler, Block Manager, 分布式推理

TensorRT-LLM:
  代码量: ~20 万行 C++/Python
  适合: 极致性能优化、kernel 开发
  学到的: Graph optimization, plugin, FP8, Tensor Parallel
```

---

## 7. 最重要的认知

```
┌──────────────────────────────────────────────────────┐
│                                                      │
│  从 ncnn_llm 到 vLLM，你要完成这个转变：             │
│                                                      │
│  "一个请求如何完成推理"                              │
│       →                                              │
│  "很多请求如何共享 GPU、显存和 KV Cache，            │
│   并持续输出 token"                                  │
│                                                      │
│  这就是 AI Infra 和普通模型部署的分界线。             │
│  也是面试中最能体现你系统深度的分界线。               │
│                                                      │
└──────────────────────────────────────────────────────┘
```

---

## 8. ncnn、llama.cpp、vLLM、TensorRT-LLM、TGI 全景对比

| 项目 | 最佳场景 | 核心创新 | 学习价值 | 代码可读性 |
|------|---------|---------|---------|-----------|
| **ncnn_llm** | 端侧推理入门 | 三 net 拆分、INT8 | ⭐⭐⭐⭐⭐ 入门最佳 | ⭐⭐⭐⭐⭐ |
| **llama.cpp** | CPU/本地推理 | GGUF、量化、mmap | ⭐⭐⭐⭐⭐ 内存管理 | ⭐⭐⭐ |
| **vLLM** | 服务端高并发 | PagedAttention、Scheduler | ⭐⭐⭐⭐⭐ 系统设计 | ⭐⭐⭐ |
| **TensorRT-LLM** | NVIDIA 极致性能 | FP8、plugin、graph opt | ⭐⭐⭐⭐ Kernel 优化 | ⭐⭐ |
| **TGI** | HuggingFace 生态 | 易用性、streaming | ⭐⭐⭐ Serving 入门 | ⭐⭐⭐ |

---

*上一篇: [06_quantization_optimization.md](06_quantization_optimization.md) — 量化与性能优化*
*下一篇: [08_project_and_resume.md](08_project_and_resume.md) — 项目包装与简历*
