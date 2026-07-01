# 09｜6 周学习计划：从当前基础到 AI Infra 推理系统入门

## 总目标

6 周后达到：

```text
能完整解释 LLM 推理链路；
能解释 KV Cache / PagedAttention / Continuous Batching；
能对比 ncnn_llm、vLLM、TensorRT-LLM、TGI；
能把项目写进简历，并回答 AI Infra 推理相关面试问题。
```

---

## Week 1：整理算子与 Transformer 执行图

目标：把已有算子笔记和 Transformer 结构联系起来。

任务：

- [ ] 整理 MatMul / GEMM / Softmax / RMSNorm / RoPE / SwiGLU；
- [ ] 画出一个 Transformer Block 的算子图；
- [ ] 标注每个算子的输入输出 shape；
- [ ] 判断哪些算子 compute-bound，哪些 memory-bound；
- [ ] 整理 FLOPs 和显存访问关系。

输出：

```text
Transformer Block 算子拆解表
Transformer Block 数据流图
```

---

## Week 2：精读 ncnn_llm 推理流水线

目标：把 ncnn_llm 从“流程知道”变成“能讲清数据流”。

任务：

- [ ] Prompt Template；
- [ ] BPE Tokenizer；
- [ ] RoPE cache；
- [ ] embed_net / decoder_net / proj_out_net；
- [ ] Sampling；
- [ ] 多轮对话；
- [ ] Tool Calling。

输出：

```text
ncnn_llm 推理时序图
ncnn_llm 三段网络拆分说明
```

---

## Week 3：KV Cache 与 Memory System

目标：掌握 KV Cache 的公式、生命周期和优化方向。

任务：

- [ ] 推导 KV Cache 显存公式；
- [ ] 计算不同 seq_len / batch / dtype 下的 KV 大小；
- [ ] 理解 GQA / MQA 如何减少 KV；
- [ ] 理解 KV Cache 量化；
- [ ] 理解 PagedAttention 的 block table 思想。

输出：

```text
KV Cache 显存计算器
PagedAttention 一页纸解释
```

---

## Week 4：Scheduler 与 Continuous Batching

目标：进入真正 AI Infra 系统层。

任务：

- [ ] 理解 Request Lifecycle；
- [ ] 区分 static batching / dynamic batching / continuous batching；
- [ ] 理解 prefill 和 decode 调度；
- [ ] 写一个最小 scheduler 伪代码；
- [ ] 阅读 vLLM scheduler 相关资料或源码目录。

输出：

```text
最小 LLM Scheduler 设计文档
Continuous Batching 图解
```

---

## Week 5：量化、FlashAttention 与性能分析

目标：理解常见推理优化手段。

任务：

- [ ] 整理 INT8 / FP8 / INT4 区别；
- [ ] 理解 AWQ / GPTQ；
- [ ] 理解 KV Cache 量化；
- [ ] 理解 FlashAttention 为什么省显存；
- [ ] 整理 TTFT / TPOT / tokens/s 指标；
- [ ] 做一张瓶颈诊断表。

输出：

```text
LLM 推理优化速查表
性能指标解释文档
```

---

## Week 6：ncnn → vLLM 对照与项目包装

目标：完成 GitHub 项目和简历表述。

任务：

- [ ] 做 ncnn_llm 和 vLLM 对照表；
- [ ] 整理 AI Infra 六层架构图；
- [ ] 写项目 README；
- [ ] 写简历项目描述；
- [ ] 准备 10 个面试问答；
- [ ] 规划下一阶段源码阅读。

输出：

```text
GitHub README
简历项目描述
面试问答文档
下一阶段路线图
```

---

## 每周复盘模板

```markdown
# Week X 复盘

## 本周学了什么？

## 最重要的 3 个概念

## 能讲清楚的问题

## 还没理解的问题

## 下周要补什么？

## 可以写进项目 README 的更新
```

---

## 判断自己是否学会的标准

不要只看是否“看完了”，而要看能否完成下面任务：

- [ ] 不看资料讲清 LLM 推理完整流程；
- [ ] 不看资料推导 KV Cache 大小；
- [ ] 不看资料解释 Prefill vs Decode；
- [ ] 不看资料解释 PagedAttention；
- [ ] 能写出一个 scheduler 伪代码；
- [ ] 能说明 ncnn 和 vLLM 的系统差异；
- [ ] 能把这个项目讲成一个 2 分钟面试介绍。
