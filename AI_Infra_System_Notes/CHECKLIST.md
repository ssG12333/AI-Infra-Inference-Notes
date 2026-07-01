# AI Infra 学习检查清单

## 1. 算子与计算图

- [ ] 能解释 Binary / Unary / Reduction 的区别；
- [ ] 能解释 MatMul / GEMM / InnerProduct 的关系；
- [ ] 能拆解 Transformer Block 的算子序列；
- [ ] 能判断一个算子是 compute-bound 还是 memory-bound；
- [ ] 能说明算子融合为什么能加速。

## 2. LLM 推理流水线

- [ ] 能画出 Text -> Token -> Embedding -> Decoder -> Logits -> Sampling -> Text；
- [ ] 能解释 Prompt Template；
- [ ] 能解释 BPE 的基本合并过程；
- [ ] 能解释 RoPE 的作用；
- [ ] 能解释 Sampling 中 top-k、top-p、temperature。

## 3. KV Cache 与内存系统

### 基础
- [ ] 能解释 KV Cache 存什么，画数据流图；
- [ ] 能推导 KV Cache 显存公式，手算 3 个模型的显存占用；
- [ ] 能解释 Prefill 和 Decode 对 KV Cache 的不同访问模式；
- [ ] 能解释 GQA / MQA 为什么省显存，ncnn 中 ExpandDims+Tile 的实现；

### 三种架构对比（源码级）
- [ ] 能写出 ncnn_llm 的 KVCache 数据结构（`KVCache = vector<pair<Mat,Mat>>`）；
- [ ] 能解释 ncnn SDPA 层中 memcpy concat 的实现逻辑；
- [ ] 能解释 llama.cpp 的 cell 池 + find_slot() 机制；
- [ ] 能对比 ncnn_llm / llama.cpp / vLLM 三种 cache 架构的适用场景；

### PagedAttention 深入
- [ ] 能画出 block table 映射图（逻辑 token → 物理 block）；
- [ ] 能解释 BlockAllocator 的 allocate/free/fork 三个核心操作；
- [ ] 能写出 PagedAttention kernel 中 block_table 寻址的伪代码；
- [ ] 能解释 COW fork 在 beam search 和 prefix cache 中的工作原理；
- [ ] 能解释 PagedAttention 如何消除外部碎片、支持动态增长；

### 量化
- [ ] 能解释 KV Cache 量化和权重量化的五个关键差异；
- [ ] 能解释 ncnn SDPA INT8 中 per-row 和 per-tensor 量化的区别；
- [ ] 能说明为什么 decode 阶段是 memory-bound 而 prefill 是 compute-bound；

## 4. 调度与服务系统

- [ ] 能解释 Request Lifecycle；
- [ ] 能区分 static batching、dynamic batching、continuous batching；
- [ ] 能解释 token budget；
- [ ] 能写出最小 scheduler 伪代码；
- [ ] 能解释为什么 decode 阶段 GPU 利用率低；
- [ ] 能解释 TTFT、TPOT、tokens/s。

## 5. 优化与部署

- [ ] 能解释 INT8、FP8、INT4 的区别；
- [ ] 能解释 AWQ / GPTQ；
- [ ] 能解释 FlashAttention 为什么省显存；
- [ ] 能列出常见图优化方法；
- [ ] 能说明 ncnn、vLLM、TensorRT-LLM、TGI 的差异。

## 6. 面试表达

- [ ] 能用 2 分钟介绍这个项目；
- [ ] 能回答 KV Cache / PagedAttention / Continuous Batching；
- [ ] 能解释 ncnn_llm 和 vLLM 的系统差异；
- [ ] 能把学习内容写成简历项目；
- [ ] 能规划下一步源码阅读路线。

## 7. 部署实战 (Module 9)

- [ ] 能编译运行 ncnn_llm，完成 Qwen3-0.6B 端侧推理；
- [ ] 能给 generate() 加计时，测量 embed/decoder/project/sample 耗时；
- [ ] 能部署 vLLM，并发压测 10/50/100 并发；
- [ ] 能制作 ncnn_llm / llama.cpp / vLLM 性能对比矩阵；
- [ ] 能解释 INT8 与 Vulkan 互斥的工程原因；
- [ ] 能解决常见部署错误 (OOM / 加载失败 / 精度退化)；
