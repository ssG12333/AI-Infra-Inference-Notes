# AI Infra 推理系统 · 完整学习资料

> 从深度学习算子的加减乘除，到 vLLM 服务端高并发推理——系统化掌握 AI Infra 全栈。
> **🔥 持续更新：从学习资料到部署实战，全流程跟进。**

[![Modules](https://img.shields.io/badge/modules-9-blue)](AI_Infra_System_Notes/source/)
[![Docs](https://img.shields.io/badge/docs-10篇-green)](AI_Infra_System_Notes/docs/)
[![Lines](https://img.shields.io/badge/总行数-24000+-orange)](#)
[![Status](https://img.shields.io/badge/更新-持续进行中-brightgreen)]()
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

---

## 这是什么？

一套面向 **AI Infra / LLM 推理系统** 的自学资料，适合：

- 想从深度学习算法转向 **推理工程 / 大模型部署**
- 已经了解 Transformer 基本结构，想理解推理系统**为什么快、为什么省显存**
- 想把 ncnn_llm、llama.cpp、vLLM 串成一条学习主线
- 想形成可写进简历、可用于面试的**系统级知识体系**

---

## 快速导航

| 你想... | 从这里开始 |
|---------|-----------|
| 快速了解全貌 | [系统架构笔记](AI_Infra_System_Notes/docs/00_overview.md) |
| 系统学习 (9模块) | [分模块学习资料](AI_Infra_System_Notes/source/README.md) |
| 自检测试 | [45条 CHECKLIST](AI_Infra_System_Notes/CHECKLIST.md) |
| 跟着计划走 | [8周学习计划](AI_Infra_System_Notes/docs/09_learning_plan.md) |
| 准备面试 | [简历模板 + 面试问答](AI_Infra_System_Notes/docs/08_project_and_resume.md) |
| 查术语 | [100+ 术语速查](AI_Infra_System_Notes/docs/10_glossary.md) |

---

## 内容结构

```
├── README.md                          ← 你在这里
│
├── doc/                               ← 原始学习资料
│   ├── deep_learning_operators(1).md   ← 200+ 算子百科全书 (3734行)
│   └── README.md                       ← ncnn_llm 推理全流程分析 (2683行)
│
└── AI_Infra_System_Notes/             ← 系统化学习体系
    ├── README.md                       ← 项目总说明
    ├── CHECKLIST.md                    ← 45条自检清单
    │
    ├── docs/                           ← 10篇系统架构笔记
    │   ├── 00_overview.md              ← 学习地图
    │   ├── 01_system_architecture.md   ← 六层架构
    │   ├── 02_compute_operator_layer.md← 算子→执行代价
    │   ├── 03_llm_inference_pipeline.md← 推理流水线
    │   ├── 04_memory_kv_cache.md       ← KV Cache + PagedAttention
    │   ├── 05_execution_scheduler.md   ← Scheduler + Continuous Batching
    │   ├── 06_quantization_optimization.md ← 量化/FlashAttn/图优化
    │   ├── 07_ncnn_to_vllm_comparison.md   ← 端侧→服务端
    │   ├── 08_project_and_resume.md    ← 简历/GitHub/面试
    │   ├── 09_learning_plan.md         ← 8周学习计划
    │   └── 10_glossary.md              ← 术语速查
    │
    └── source/                         ← 9模块完整学习资料
        ├── README.md                   ← 学习地图 + 三条路径
        ├── module-01-foundations/      ← 基石算子
        ├── module-02-building-blocks/  ← 构建算子
        ├── module-03-architectures/    ← 组合架构
        ├── module-04-advanced/         ← 进化架构
        ├── module-05-inference-pipeline/← 推理流水线
        ├── module-06-kv-cache-system/  ← KV Cache 系统 ⭐
        ├── module-07-scheduling/       ← 调度系统
        ├── module-08-optimization/     ← 优化技术
        └── module-09-deployment/       ← 部署实战
```

---

## 9 个学习模块

| # | 模块 | 核心内容 | 时间 |
|---|------|---------|:--:|
| 1 | **基石算子** | 数学运算、激活函数、归一化、张量操作、硬件模型 | 8h |
| 2 | **构建算子** | 卷积(6参数4算法)、注意力(SDPA/RoPE/Flash)、RNN、内存布局 | 8h |
| 3 | **组合架构** | CNN Block、Transformer Block(完整算子序列)、Qwen3拆解 | 6h |
| 4 | **进化架构** | GQA/QK-Norm、Mamba/SSM、MoE、量化基础、多模态 | 6h |
| 5 | **推理流水线** | 8 Phase完整数据流、Prefill vs Decode、generate()源码 | 10h |
| 6 | **KV Cache系统** ⭐ | 三种架构源码分析、PagedAttention完整实现、COW Fork | 12h |
| 7 | **调度系统** | Continuous Batching 5 Phase、Token Budget、性能指标 | 8h |
| 8 | **优化技术** | INT8/FP8/INT4、AWQ/GPTQ、FlashAttention、图优化 | 10h |
| 9 | **部署实战** | ncnn_llm/llama.cpp/vLLM部署 + 四框架性能对比 | 15h |

---

## 三条学习路径

| 路径 | 时间 | 适合 |
|------|:--:|------|
| 🟢 **快速入门** | 2周 | 想快速了解 LLM 推理全链路 |
| 🟡 **系统学习** | 6周 | 想面试 AI Infra 岗位 |
| 🔴 **全栈深入** | 8周 | 想动手部署 + 性能调优 |

> 每天一下午 (3-4h) 的话，快速入门 1 周，系统学习 4 周，全栈 6 周可完成。

---

## 配套源码

本项目分析基于以下开源项目（需自行 clone）：

| 项目 | 用途 | 仓库地址 |
|------|------|---------|
| **ncnn** | 腾讯端侧推理框架 | [github.com/Tencent/ncnn](https://github.com/Tencent/ncnn) |
| **ncnn_llm** | 基于 ncnn 的 LLM 推理 | [github.com/nihui/ncnn_llm](https://github.com/nihui/ncnn_llm) |
| **llama.cpp** | CPU LLM 推理标杆 | [github.com/ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) |

---

## 开始学习

→ **[进入学习地图](AI_Infra_System_Notes/source/README.md)**

---

## 📜 版权 & 协议

<p align="center">
  <b>© 2025 <a href="https://github.com/ssG12333">ssG12333</a> — AI Infra 推理系统学习资料</b><br>
  🔥 持续更新中：从资料整理到部署全流程<br><br>
  本项目遵循 <a href="LICENSE">MIT License</a> 开源<br>
  欢迎自由学习、分享、二次创作，引用时请注明出处<br><br>
  <b>如果这份资料帮到了你，请给个 ⭐ Star</b><br>
  你的 Star 是我持续更新的动力 🚀<br><br>
  <a href="https://github.com/ssG12333/AI-Infra-Inference-Notes">
    <img src="https://img.shields.io/github/stars/ssG12333/AI-Infra-Inference-Notes?style=social" alt="GitHub stars">
  </a>
</p>
