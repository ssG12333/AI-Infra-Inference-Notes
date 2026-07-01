# AI Infra 推理系统 · 完整学习资料

> 从深度学习算子的加减乘除，到 vLLM 服务端高并发推理——9 个模块，系统化掌握 AI Infra 全栈。

---

## 学习路线图

```
模块 1: 基石算子                       模块 2: 构建算子
┌──────────────────────┐          ┌──────────────────────┐
│ 数学运算 · 激活函数    │          │ 线性变换 · 卷积       │
│ 归一化 · 张量操作     │   ──→    │ 池化 · 注意力 · 循环  │
│ Loss · 硬件执行模型   │          │ 嵌入 · 内存布局       │
└──────────────────────┘          └──────────────────────┘
         │                                    │
         └──────────────┬─────────────────────┘
                        ▼
模块 3: 组合架构                        模块 4: 进化架构
┌──────────────────────┐          ┌──────────────────────┐
│ CNN Block · RNN 单元  │          │ 注意力变体 · SSM      │
│ Transformer Block    │   ──→    │ MoE · 量化部署       │
│ 编解码 · 模型拆解     │          │ 多模态 · 分布式       │
└──────────────────────┘          └──────────────────────┘
         │                                    │
         └──────────────┬─────────────────────┘
                        ▼
模块 5: 推理流水线                      模块 6: KV Cache 系统
┌──────────────────────┐          ┌──────────────────────┐
│ Prompt → Tokenize    │          │ ncnn 连续缓存         │
│ → Embed → Decoder    │   ──→    │ llama.cpp cell 管理   │
│ → Project → Sample   │          │ vLLM PagedAttention  │
└──────────────────────┘          └──────────────────────┘
         │                                    │
         └──────────────┬─────────────────────┘
                        ▼
模块 7: 调度系统                        模块 8: 优化技术
┌──────────────────────┐          ┌──────────────────────┐
│ Request Lifecycle    │          │ INT8/FP8/INT4 量化    │
│ Continuous Batching  │   ──→    │ FlashAttention       │
│ Token Budget · 指标   │          │ 图优化 · Profiling   │
└──────────────────────┘          └──────────────────────┘
         │                                    │
         └──────────────┬─────────────────────┘
                        ▼
                模块 9: 部署实战
┌──────────────────────────────────────────────┐
│ ncnn_llm 端侧部署 · llama.cpp 本地推理        │
│ vLLM 服务端部署 · 四框架性能对比 · 排错指南    │
└──────────────────────────────────────────────┘
```

---

## 9 个模块速览

| 模块 | 内容 | 对应 8 周计划 | 预估时间 |
|------|------|:----------:|:------:|
| **[Module 1](module-01-foundations/)** | 数学运算、激活函数、归一化、张量操作、Loss、硬件模型 | Week 1 | 8-10h |
| **[Module 2](module-02-building-blocks/)** | 线性变换、卷积、池化、注意力、RNN、嵌入 | Week 1-2 | 8-10h |
| **[Module 3](module-03-architectures/)** | CNN/RNN/Transformer Block、编解码、ResNet/ViT/LLaMA 拆解 | Week 2 | 6-8h |
| **[Module 4](module-04-advanced/)** | 注意力变体、SSM/Mamba、MoE、量化概念、多模态 | Week 3 | 6-8h |
| **[Module 5](module-05-inference-pipeline/)** | LLM 推理 8 Phase 完整链路、Prefill vs Decode | Week 3-4 | 10-12h |
| **[Module 6](module-06-kv-cache-system/)** | KV Cache 三种架构、PagedAttention 源码级分析 | Week 4-5 | 12-15h |
| **[Module 7](module-07-scheduling/)** | Scheduler、Continuous Batching、Token Budget | Week 5 | 8-10h |
| **[Module 8](module-08-optimization/)** | 量化全解、FlashAttention、图优化、Profiling | Week 6 | 10-12h |
| **[Module 9](module-09-deployment/)** | ncnn/llama.cpp/vLLM 部署实战 + 对比 | Week 7-8 | 15-20h |

---

## 配套资源索引

### 系统架构笔记（高层概览）
- [00_overview.md](../docs/00_overview.md) — 学习地图 + 三条主线
- [01_system_architecture.md](../docs/01_system_architecture.md) — 六层架构详解
- [10_glossary.md](../docs/10_glossary.md) — 100+ 术语速查

### 原始资料（完整参考）
- [deep_learning_operators_original.md](deep_learning_operators_original.md) — 200+ 算子百科全书 (3734 行)
- [ncnn_llm_original_README.md](ncnn_llm_original_README.md) — ncnn_llm 逐 Phase 源码分析 (2683 行)

### 源码仓库（动手实战）
- [llama.cpp-master](../../../源码/llama.cpp-master/) — CPU LLM 推理标杆
- [ncnn-master](../../../源码/ncnn-master/) — 端侧推理框架
- [ncnn_llm-main](../../../源码/ncnn_llm-main/) — 端侧 LLM 推理实现

### 学习管理
- [CHECKLIST.md](../CHECKLIST.md) — 45 条自检清单
- [09_learning_plan.md](../docs/09_learning_plan.md) — 6-8 周计划
- [08_project_and_resume.md](../docs/08_project_and_resume.md) — GitHub + 简历 + 面试

---

## 三条学习路径

### 🟢 路径 A：快速入门 (2 周, ~30h)
```
Module 1 §1-3 (数学+激活) → Module 2 §4 (注意力)
→ Module 3 §3 (Transformer) → Module 5 §1-8 (推理流水线)
→ Module 6 §1-2 (KV Cache 基础)
目标: 能讲清楚 LLM 推理全链路
```

### 🟡 路径 B：系统学习 (6 周, ~90h)
```
Module 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8
按照模块顺序完整推进，每个模块做输出
目标: 能面试 + 能写核心伪代码
```

### 🔴 路径 C：全栈深入 (8 周, ~160h)
```
全部 9 个模块 + Module 9 部署实战 + 性能对比实验
目标: 能部署 + 能选型 + 能调优
```

---

## 每模块的结构

每个模块目录包含：

```
module-XX-*/
├── README.md          ← 学习指南 (本文档)
├── 01-*.md            ← 子章节 (逐步深入)
├── 02-*.md
├── ...
└── exercises.md       ← 动手练习
```

每章统一使用以下标注：

| 标注 | 含义 |
|------|------|
| 📖 | 概念解释——必读 |
| 🔬 | 源码分析——对照代码阅读 |
| 💡 | 关键洞察——理解设计意图 |
| ⚠️ | 常见误区——面试/实战易错点 |
| 🛠️ | 动手练习——必须自己写/跑一遍 |

---

## 开始学习

→ [Module 1: 基石算子——逐元素的世界](module-01-foundations/README.md)
