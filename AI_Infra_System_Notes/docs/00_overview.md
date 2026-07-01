# 00｜总览：AI Infra 推理系统学习地图

> 本项目系统整理 AI Infra / LLM 推理系统核心知识，从深度学习算子到推理服务部署，形成可学习、可面试、可实战的完整体系。

---

## 1. 学习地图

```
                         ┌──────────────────────┐
                         │   00_overview (你在这)  │
                         │   项目全景 + 学习地图    │
                         └──────────┬───────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
        ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
        │ 01_架构 (骨架) │  │ 02_算子 (原子) │  │ 03_流水线 (血液)│
        │ 六层架构模型   │  │ 算子→执行代价  │  │ 推理完整链路  │
        └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
               │                 │                 │
               └─────────┬───────┴─────────┬───────┘
                         ▼                 ▼
               ┌──────────────┐   ┌──────────────┐
               │ 04_KV Cache  │   │ 05_Scheduler │
               │ 内存系统 (心脏)│   │ 调度系统 (大脑)│
               │ 含PagedAttn  │   │ 含Cont. Batch│
               └──────┬───────┘   └──────┬───────┘
                      │                  │
                      └────────┬─────────┘
                               ▼
                      ┌──────────────┐
                      │ 06_优化 (肌肉) │
                      │ 量化+FlashAttn│
                      │ +图优化+调优  │
                      └──────┬───────┘
                             │
                             ▼
                      ┌──────────────┐
                      │ 07_对比 (视角) │
                      │ ncnn→vLLM   │
                      │ 端侧→服务端   │
                      └──────┬───────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
      ┌──────────┐  ┌──────────┐  ┌──────────┐
      │08_包装    │  │09_计划    │  │10_术语    │
      │简历/GitHub│  │6周学习    │  │速查表     │
      └──────────┘  └──────────┘  └──────────┘
```

---

## 2. 各文档定位与学习目标

| 文档 | 定位 | 核心学习目标 | 预估阅读 |
|------|------|-------------|---------|
| **01_system_architecture** | 骨架 | 理解六层架构，建立全局视图 | 30 min |
| **02_compute_operator_layer** | 原子 | 算子分类 + 执行代价分析 + roofline 判断 | 45 min |
| **03_llm_inference_pipeline** | 血液 | LLM 推理 8 个 Phase 的完整数据流 | 45 min |
| **04_memory_kv_cache** | 心脏 | 三种 KV Cache 架构 + PagedAttention 源码级 | 90 min |
| **05_execution_scheduler** | 大脑 | Continuous Batching + Scheduler + Request Lifecycle | 60 min |
| **06_quantization_optimization** | 肌肉 | INT8/FP8/INT4 + FlashAttention + 图优化 + Profiling | 60 min |
| **07_ncnn_to_vllm_comparison** | 视角 | 端侧→服务端的 7 个概念跃迁 + 框架选型 | 30 min |
| **08_project_and_resume** | 输出 | GitHub README + 简历包装 + 面试回答框架 | 20 min |
| **09_learning_plan** | 节奏 | 6 周推进计划 + 每周任务 | 15 min |
| **10_glossary** | 字典 | 100+ 术语速查 | 随时查阅 |

---

## 3. 三条学习主线

### 主线一：从算子到计算系统

```
01 (架构) → 02 (算子) → 03 (流水线) → 06 (优化)

回答:
- 模型如何被拆成数千个算子执行？
- 如何判断一个算子是 compute-bound 还是 memory-bound？
- INT8/FP8 量化如何工作？FlashAttention 如何加速？
```

### 主线二：从单请求到多请求调度

```
03 (流水线) → 04 (KV Cache) → 05 (Scheduler)

回答:
- KV Cache 为什么是显存瓶颈？三种实现范式的差异？
- PagedAttention 如何消除碎片、支持 prefix sharing？
- Continuous Batching 为什么能提升 2-4x 吞吐？
```

### 主线三：从端侧到服务端

```
03 (ncnn_llm 分析) → 07 (ncnn→vLLM) → 05 (Scheduler) → 07 (框架对比)

回答:
- ncnn_llm 和 vLLM 的系统目标有何本质不同？
- 从端侧到服务端，需要完成哪些认知跃迁？
- 什么场景选择什么框架？
```

---

## 4. 你的现有基础和本项目的关系

```
你的现有资料:

  deep_learning_operators (3734行)
    ↓ 覆盖了第 2 层的每个算子

  ncnn_llm 源码分析 (2683行)
    ↓ 覆盖了第 3、4 层的完整推理链路

本项目补充的 AI Infra 系统内容:

  第 4 层: KV Cache 三种实现 + PagedAttention + 显存公式
  第 5 层: Scheduler + Continuous Batching + Token Budget
  第 6 层: 量化/FlashAttention/图优化/Profiling
  跨层:   ncnn→llama.cpp→vLLM 的对照学习
  实战:   简历/GitHub/面试/学习计划
```

---

## 5. 推荐学习路径

### 路线 A：快速入门（1-2 周）

```
01 (30min) → 03 (45min) → 04 §1-3 (30min) → 10 (随时查)
目标: 能解释 LLM 推理全链路 + KV Cache 基本原理
```

### 路线 B：系统学习（4-6 周）

```
Week 1: 01 + 02    (建立架构和算子认知)
Week 2: 03         (完整推理流水线)
Week 3: 04         (KV Cache 深度 + PagedAttention)
Week 4: 05         (调度系统 + Continuous Batching)
Week 5: 06         (量化 + FlashAttention + 优化)
Week 6: 07 + 08    (框架对比 + 简历/GitHub 输出)
```

### 路线 C：面试突击（1 周）

```
精读: 04 (§2.3 PagedAttention) + 05 (§4 Continuous Batching) + CHECKLIST
重点: 3 道高频面试题的完整回答框架 (各文档末尾)
辅助: 10 (术语速查) + 08 (面试模板)
```

---

## 6. 配套资源

### 源码
- [ncnn_llm-main](../../源码/ncnn_llm-main/) — 端侧 LLM 推理实现（~3000 行 C++）
- [ncnn-master](../../源码/ncnn-master/) — 通用端侧推理框架（含 SDPA/Vulkan/量化）
- [llama.cpp-master](../../源码/llama.cpp-master/) — CPU LLM 推理标杆（含 KV Cache cell 管理）

### 原始资料
- [deep_learning_operators(1).md](../../doc/deep_learning_operators(1).md) — 200+ 算子的百科全书（3734 行）
- [doc/README.md](../../doc/README.md) — ncnn_llm 逐 Phase 源码分析（2683 行）

### 图表
- [ai_infra_architecture.mmd](../diagrams/ai_infra_architecture.mmd) — Mermaid 架构图源码
- [ai_infra_architecture.svg](../assets/ai_infra_architecture.svg) — SVG 架构图

---

## 7. AI Infra 能力模型

```
Level 1: 认识算子
  能说出常见算子的名称和用途。
  → 标准深度学习课程的内容。

Level 2: 理解执行
  能判断算子的瓶颈类型、理解 roofline model。
  → 对应本项目第 02 章。

Level 3: 理解推理系统
  能画出 LLM 推理的完整数据流、理解 KV Cache 的三种实现。
  → 对应本项目第 03-04 章。

Level 4: 理解调度
  能解释 Continuous Batching、PagedAttention、Token Budget。
  → 对应本项目第 05 章。这是 AI Infra 面试的核心。

Level 5: 能优化和选型
  能根据 profiling 数据选择优化策略、能为场景选择合适框架。
  → 对应本项目第 06-07 章。
```

---

*开始学习: [01_system_architecture.md](01_system_architecture.md) — 六层架构全景*
