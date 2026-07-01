# 08｜项目包装：把学习资料变成 GitHub 项目、简历项目和面试素材

## 1. GitHub 项目名称建议

可以选一个正式一点的名称：

```text
AI-Infra-Inference-Notes
LLM-Inference-System-Study
DeepLearning-Operator-to-Inference-System
AI-Infra-From-Operators-to-Serving
```

推荐：

```text
AI-Infra-Inference-Notes
```

简洁、方向明确，适合长期维护。

---

## 2. GitHub README 项目介绍模板

```markdown
# AI Infra Inference Notes

本项目系统整理 AI Infra / LLM 推理系统核心知识，从深度学习算子、计算图、ncnn 端侧推理，到 KV Cache、PagedAttention、Scheduler、Continuous Batching 和服务端推理框架。

项目目标：
- 建立从算子到推理系统的完整知识图谱；
- 拆解 ncnn_llm 的端侧大模型推理链路；
- 对照 vLLM / TGI / TensorRT-LLM，理解服务端高吞吐推理系统；
- 形成可用于面试、简历和工程实践的 AI Infra 学习项目。
```

---

## 3. 简历项目写法

### 版本 A：学习项目型

```text
AI Infra 推理系统学习项目
系统整理深度学习算子、Transformer 推理链路、ncnn 端侧大模型推理、KV Cache、量化、FlashAttention、PagedAttention 和连续批处理等 AI Infra 核心内容。基于 ncnn_llm 源码梳理 Prompt Template、BPE Tokenizer、RoPE、Embedding、Decoder、Projection、Sampling、KV Cache 与多轮对话流程，并进一步对照 vLLM 服务端推理框架，总结端侧推理与高吞吐 LLM Serving 在内存管理、调度策略和运行时执行方面的差异。
```

### 版本 B：偏工程实践型

```text
大模型推理系统与端侧部署分析项目
围绕 ncnn_llm 推理框架完成从 model.json 配置加载、BPE 分词、RoPE 位置编码、Decoder 前向、KV Cache 维护到 Sampling 输出的完整链路分析；总结 INT8 量化、Vulkan 加速、FlashAttention、Prefill/Decode 性能差异和 Tool Calling 生命周期。进一步构建 AI Infra 系统架构笔记，补充 PagedAttention、Continuous Batching、Scheduler、KV Cache 量化与服务端推理系统设计，为后续 vLLM / TensorRT-LLM 源码学习打下基础。
```

### 版本 C：面试突出型

```text
AI Infra / LLM Inference 系统化学习项目
从算子执行、模型结构、推理流水线、内存管理、请求调度和服务部署六个层面搭建 AI Infra 知识体系。重点分析 Transformer 推理中的 KV Cache 机制、Prefill/Decode 差异、长上下文显存瓶颈、INT8/FP8/INT4 量化、FlashAttention、PagedAttention 与 Continuous Batching，并对比 ncnn 端侧推理和 vLLM 服务端推理的系统设计差异。
```

---

## 4. 面试高频问题与回答框架

### Q1：LLM 推理为什么要分 Prefill 和 Decode？

回答框架：

```text
Prefill 处理完整 prompt，一次性计算所有输入 token 的 hidden states 和初始 KV Cache，属于计算密集型阶段，主要影响首 token 延迟。
Decode 阶段每次只输入一个新 token，复用历史 KV Cache，循环生成后续 token，属于小步循环和内存带宽敏感阶段，主要影响生成速度和吞吐。
区分两者有利于分别设计调度策略和优化策略，例如 prefill 控制 token budget，decode 使用 continuous batching。
```

### Q2：KV Cache 解决了什么问题？带来了什么问题？

回答框架：

```text
KV Cache 通过缓存历史 token 在每层 attention 中的 K/V，避免 decode 阶段重复计算历史序列的 K/V，从而显著降低自回归生成计算量。
但它会随着层数、序列长度、KV head 数和并发请求数线性增长，成为长上下文和高并发推理下的主要显存瓶颈。因此需要 PagedAttention、KV Cache 量化、prefix cache 和内存池管理等优化。
```

### Q3：PagedAttention 的核心思想是什么？

回答框架：

```text
PagedAttention 借鉴操作系统分页思想，把 KV Cache 切成固定大小 block。逻辑上每个请求的 token 序列是连续的，但物理上 KV block 可以不连续，通过 block table 映射。
这样可以减少不同请求长度差异造成的显存碎片，提高 KV Cache 利用率，从而在固定显存下支持更多并发请求，提高 LLM serving 吞吐。
```

### Q4：ncnn_llm 和 vLLM 有什么区别？

回答框架：

```text
ncnn_llm 更偏端侧和轻量推理，重点是模型拆分、低内存、CPU/Vulkan 后端、INT8 量化和单请求上下文管理。
vLLM 更偏服务端高吞吐推理，重点是多请求调度、continuous batching、paged KV cache、GPU 利用率和 OpenAI-compatible serving。
两者都涉及 LLM 推理流水线和 KV Cache，但系统目标完全不同。
```

### Q5：为什么 decode 阶段 GPU 利用率容易低？

回答框架：

```text
Decode 阶段每个请求每轮通常只处理 1 个 token，单请求矩阵尺寸很小，难以充分利用 GPU。同时每轮都要读取历史 KV Cache，长上下文下访存压力大。不同请求生成长度不同，还会导致 batch 动态变化。因此需要 continuous batching 将多个请求的 decode step 合并执行，并用高效 KV Cache 管理减少显存浪费。
```

---

## 5. 项目后续可做实验

### 实验 1：KV Cache 显存计算器

输入：

```text
num_layers, seq_len, num_kv_heads, head_dim, dtype, batch_size
```

输出：

```text
单请求 KV Cache 大小
多请求并发 KV Cache 大小
FP16 / INT8 / INT4 对比
```

### 实验 2：最小 Scheduler 模拟器

模拟：

```text
waiting queue
prefill queue
running decode requests
token budget
KV block allocation
```

指标：

```text
平均等待时间
平均完成时间
tokens/s
KV block 使用率
```

### 实验 3：ncnn 与 vLLM 对照表

把相同概念对应起来：

```text
ncnn::Net -> Model Executor
ncnn::Mat -> Tensor / Buffer
decoder_net -> LLM Block Executor
KV Cache Mat -> KV Block
generate loop -> Scheduler decode step
```

---

## 6. GitHub 维护建议

每次学习新项目时，按固定格式记录：

```markdown
## 模块名称

### 1. 它解决什么问题？
### 2. 输入输出是什么？
### 3. 核心数据结构是什么？
### 4. 执行流程是什么？
### 5. 性能瓶颈在哪里？
### 6. 和已有系统如何对照？
### 7. 面试中怎么讲？
```

这样你的笔记会越来越像工程文档，而不是普通学习笔记。
