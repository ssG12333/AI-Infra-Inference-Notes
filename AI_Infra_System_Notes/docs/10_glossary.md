# 10｜AI Infra 推理系统术语速查

> 按类别组织的 100+ 术语速查表。每个术语控制在 2-5 行——说清楚"是什么"和"为什么重要"。

---

## 算子与计算

### MatMul / GEMM / InnerProduct
**是什么**：矩阵乘法及其变体。MatMul 最通用，GEMM 支持转置和缩放（`C = αAB + βC`），InnerProduct 是全连接层的封装（`y = Wx + b`）。
**为什么重要**：占了神经网络 80%+ 的计算量，是优化的首要目标。

### Convolution (Conv2D)
**是什么**：通过共享参数的局部滤波器提取空间特征。`output = input ⊛ kernel + bias`。
**为什么重要**：CNN 的核心算子。不同 kernel size/stride/dilation 对应不同的感受野和下采样策略。有 im2col+GEMM / Winograd / FFT / Direct 四种实现算法。

### Depthwise Convolution
**是什么**：每个输入通道独立做卷积（groups=C_in）。参数量 = C_in × k²，远小于标准卷积的 C_out × C_in × k²。
**为什么重要**：MobileNet 的核心设计。极致降低参数量，但计算强度低（memory-bound）。

### SDPA (Scaled Dot-Product Attention)
**是什么**：`Attention(Q,K,V) = softmax(QK^T/√d_k) × V`。Transformer 的核心计算单元。
**为什么重要**：QK^T 矩阵是 O(seq²) 的显存瓶颈，FlashAttention 通过分块计算解决这个问题。

### RoPE (Rotary Position Embedding)
**是什么**：通过旋转 Q/K 向量的二维子空间来编码位置信息。旋转角度差天然编码相对位置。
**为什么重要**：现代 LLM 标配（LLaMA/Qwen/Mistral），支持上下文外推。

### GQA / MQA (Grouped/Multi-Query Attention)
**是什么**：GQA 中多个 Q head 共享一组 KV head（如 16 Q heads 共享 8 组 KV），MQA 中所有 Q head 共享 1 组 KV。
**为什么重要**：减少 KV Cache 50%（GQA）或 93.75%（MQA），是长上下文推理的关键优化。

### RMSNorm
**是什么**：LayerNorm 的简化版：只做缩放（γ），不做均值中心化和偏置（β）。`y = x / rms(x) × γ`。
**为什么重要**：比 LayerNorm 快 ~30%，参数量减半，现代 LLM 标配。

### SwiGLU
**是什么**：`SwiGLU(x) = Swish(x·W_gate) ⊙ (x·W_up) · W_down`。用 Swish 做门控的 GLU 变体。
**为什么重要**：现代 LLM 标配 MLP 结构（替代 ReLU-FFN），中间维度 3×hidden（非传统 4×）。

### Softmax
**是什么**：`softmax(x_i) = exp(x_i) / Σexp(x_j)`，将 logits 转为概率分布。
**为什么重要**：分类任务和注意力机制的标配。数值稳定实现需要减 max。Temperature 参数控制分布的尖锐度。

### FlashAttention
**是什么**：分块计算 attention，避免物化完整的 [seq, seq] attention matrix。使用 online softmax 算法保持数学等价。
**为什么重要**：显存从 O(seq²) 降到 O(seq)，是长序列推理的关键技术。

---

## 量化

### Quantization (量化)
**是什么**：将 FP32/FP16 高精度数值映射到 INT8/INT4 低精度表示的压缩技术。`q = round(x/scale) + zero_point`。
**为什么重要**：权重从 2 bytes → 1 byte (INT8) 或 0.5 byte (INT4)。移动端和显存受限场景的关键技术。

### Scale (量化缩放因子)
**是什么**：`s = max(|x|) / (2^(bits-1) − 1)`。FP→INT 映射的比例因子。
**为什么重要**：量化精度取决于 scale 的选择。Per-channel scale 精度远优于 per-tensor。

### Per-Tensor vs Per-Channel Quantization
**是什么**：Per-tensor = 整个张量一个 scale。Per-channel = 每个输出通道独立 scale。
**为什么重要**：Per-channel 精度高但 scale 存储开销大。INT8 卷积通常用 per-channel 量化权重。

### PTQ (Post-Training Quantization)
**是什么**：训练后量化——不需要重新训练，只需少量校准数据统计激活范围。
**为什么重要**：成本低，适合快速部署。AWQ 和 GPTQ 是最先进的 LLM PTQ 方法。

### QAT (Quantization-Aware Training)
**是什么**：在训练过程中模拟量化误差，让模型适应低精度。
**为什么重要**：精度最高但成本高，通常用于 INT4 以下的极端量化。

### AWQ (Activation-aware Weight Quantization)
**是什么**：根据激活分布识别"重要权重通道"，做 scale 变换保护关键通道后量化。
**为什么重要**：PTQ 中最简单有效的 LLM INT4 量化方法，几分钟完成。

### GPTQ
**是什么**：逐列量化权重，量化误差补偿到未量化列。使用 Hessian 矩阵指导补偿。
**为什么重要**：精度通常优于 AWQ（在极低 bit 下），但计算成本更高。

### FP8
**是什么**：8-bit 浮点格式（E4M3/E5M2），有指数位 → 动态范围远超 INT8。
**为什么重要**：H100 等新 GPU 的硬件加速格式，无需 scale 校准，transformer 友好。

### KV Cache 量化
**是什么**：将运行时产生的 K/V 缓存以 INT8 存储，attention 计算时动态反量化。
**为什么重要**：KV Cache 是显存主要瓶颈，量化可压缩 50%；但 attention 对精度敏感，需要细粒度量化。

### Dynamic Quantization (动态量化)
**是什么**：激活值的 scale 在推理时动态计算（而非预先校准）。
**为什么重要**：激活值随输入变化，无法预先校准。用于 ncnn INT8 推理中的输入量化。

### Requantize
**是什么**：INT32 累加结果 → INT8 的转换操作。`output_int8 = round((input_int32 × scale_in + bias) × scale_out)`。
**为什么重要**：INT8 推理中每层输出的必经之路，ncnn 中有专门的 Requantize 层。

---

## KV Cache 与内存

### KV Cache
**是什么**：缓存每层 attention 的历史 K/V 向量，避免 decode 阶段重复计算。
**为什么重要**：自回归推理的必需机制。是显存的主要瓶颈（长序列下甚至超过模型权重）。

### PagedAttention
**是什么**：将 KV Cache 切成固定大小 block，通过 block table 映射逻辑位置到物理 block。借鉴操作系统虚拟内存思想。
**为什么重要**：消除外部碎片（利用率 60%→95%），支持 prefix sharing 和 COW fork。

### Block Table
**是什么**：从逻辑 token 位置到物理 KV block 的映射表。`physical_block = block_table[logical_position / BLOCK_SIZE]`。
**为什么重要**：PagedAttention 的核心数据结构。类比操作系统的页表。

### COW Fork (Copy-on-Write Fork)
**是什么**：多个序列共享 KV blocks，只在写入时才复制。ref_count 引用计数管理。
**为什么重要**：Beam search 和 prefix sharing 的零拷贝实现。

### Prefix Cache
**是什么**：缓存共享前缀（如 system prompt）的 KV Cache，多个请求复用。
**为什么重要**：长 system prompt 场景下可省 50%+ KV Cache 空间。

### KV Cache Eviction (KV Cache 淘汰)
**是什么**：当显存不足时，选择性地丢弃部分 KV Cache（如最远 token 或低注意力权重 token）。
**为什么重要**：极端长上下文的兜底策略。配合 SWA 使用。

### Memory-bound vs Compute-bound
**是什么**：Memory-bound = 瓶颈在内存带宽（如 decode 阶段）；Compute-bound = 瓶颈在算力（如 prefill 阶段）。
**为什么重要**：决定优化方向——前者优化访存，后者优化计算。

### External Fragmentation (外部碎片)
**是什么**：总空闲空间足够但无连续大块 → 分配失败。传统连续 KV 分配的主要问题。
**为什么重要**：PagedAttention 的核心动机——页式管理消除外部碎片。

---

## 调度与服务

### Prefill
**是什么**：处理用户 prompt 的阶段。一次性计算所有 prompt token，生成初始 KV Cache。
**为什么重要**：影响首 token 延迟（TTFT），是计算密集阶段，适合大 batch。

### Decode
**是什么**：自回归生成阶段。每步输入 1 个 token，复用历史 KV Cache，循环直到 EOS。
**为什么重要**：影响生成速度和吞吐（TPOT），是内存密集阶段，需要 continuous batching。

### Continuous Batching
**是什么**：每个 decode step 动态重组 batch。完成请求移出，新请求加入。
**为什么重要**：LLM serving 吞吐提升 2-4x 的关键技术。消除"木桶效应"。

### Static Batching
**是什么**：等一批请求凑齐后一起处理，pad 到相同长度。
**为什么重要**：概念简单，但短请求等长请求 + padding 浪费严重。已被 Continuous Batching 取代。

### Chunked Prefill
**是什么**：将长 prompt 切成多个 chunk 逐步 prefill，中间允许 decode 请求插入。
**为什么重要**：避免单长 prompt 长时间阻塞所有短请求。

### Token Budget
**是什么**：每轮 step 允许处理的最大 token 数（如 8192）。Prefill 请求的 prompt token + decode 请求的 1 token。
**为什么重要**：调度的"货币"——按 token 数（而非请求数）分配资源，保证公平性。

### Request Lifecycle
**是什么**：请求的状态流转：WAITING → PREFILL → DECODING → FINISHED / SWAPPED。
**为什么重要**：Scheduler 的状态管理基础。

### Preemption (抢占)
**是什么**：当显存不足时，swap out 低优先级请求的 KV Cache 到 CPU，给高优先级请求让路。
**为什么重要**：服务端资源管理的兜底机制。

### TTFT (Time To First Token)
**是什么**：从请求到达到第一个生成 token 输出之间的延迟。
**为什么重要**：用户感知的"首 token 响应速度"。Prefill 优化的核心指标。

### TPOT (Time Per Output Token)
**是什么**：Decode 阶段平均每个 token 的生成时间。
**为什么重要**：用户感知的"生成速度"。Decode 优化的核心指标。

### Throughput (吞吐)
**是什么**：系统整体每秒处理的 token 数 (tokens/s) 或每秒请求数 (requests/s)。
**为什么重要**：服务端核心指标。与延迟通常存在 trade-off。

---

## 图优化与编译

### Operator Fusion (算子融合)
**是什么**：将连续多个算子合并为单个 kernel。如 Conv + BN + ReLU → ConvBNReLU。
**为什么重要**：减少显存读写和 kernel launch 开销。是最基础的图优化。

### Constant Folding (常量折叠)
**是什么**：在编译时计算静态表达式。如 `ConvBN` 融合时 W' = α × W 在加载时计算。
**为什么重要**：消除运行时的不必要计算。

### Graph Optimization (图优化)
**是什么**：对计算图进行等价变换以提升性能。包括融合、折叠、死代码消除、布局变换。
**为什么重要**：编译器级别的自动优化，对上层透明。

### Kernel Dispatch
**是什么**：运行时根据硬件（CPU/GPU/Vulkan）、dtype（FP32/FP16/INT8）、shape 选择最优 kernel 实现。
**为什么重要**：同一算子不同场景性能差异可达 10x，正确的 dispatch 至关重要。

### Tiling (分块)
**是什么**：将大矩阵运算切分成适合 cache/SRAM 的小块，逐块计算。
**为什么重要**：提高数据复用率，减少 cache miss。GEMM 和 FlashAttention 的核心技巧。

### Im2Col + GEMM
**是什么**：将卷积的滑动窗口展开为矩阵列（im2col），然后用 GEMM 计算。
**为什么重要**：ncnn 的默认卷积实现。内存膨胀 k² 倍，但可复用高度优化的 GEMM。

### Winograd
**是什么**：用变换域的逐元素乘替代空域的卷积。3×3 卷积只需 4 次乘法（原始需 9 次）。
**为什么重要**：ncnn 中 3×3 stride=1 卷积的加速实现。比 im2col 快 1.5-2x。

---

## 并行与分布式

### Tensor Parallelism (张量并行)
**是什么**：将单层权重的列/行切分到多 GPU，每 GPU 计算一部分然后聚合。
**为什么重要**：单 GPU 放不下大模型时的扩展方案。NVLink 互联是关键。

### Pipeline Parallelism (流水线并行)
**是什么**：将模型按层分段，不同 GPU 负责不同层。micro-batch 流水线执行。
**为什么重要**：适合层数较多的模型。与 tensor parallel 正交，可以组合使用。

### Data Parallelism (数据并行)
**是什么**：每 GPU 持有完整模型副本，不同数据在不同 GPU 上并行。
**为什么重要**：训练中最基础的并行方式。推理中较少使用（除非 batch 极大）。

### Expert Parallelism (专家并行)
**是什么**：MoE 模型中不同专家分布在不同 GPU。Router 决定 token 路由到哪个 GPU 的专家。
**为什么重要**：大 MoE 模型（如 Mixtral 8×7B, DeepSeek-V3）的必需技术。

### AllReduce
**是什么**：分布式训练/推理中的集合通信原语——将所有 GPU 的数据求和并广播。
**为什么重要**：Tensor parallel 中聚合部分结果需要 AllReduce。

---

## 部署与框架

### ncnn
**是什么**：腾讯开源的轻量级端侧推理框架。支持 ARM/x86/Vulkan，无第三方依赖。
**为什么重要**：移动端推理标杆。你的项目以 ncnn_llm 为基础分析 LLM 推理链路。

### llama.cpp
**是什么**：C/C++ 实现的 LLM 推理框架，专注 CPU 和边缘设备。使用 GGUF 格式。
**为什么重要**：CPU 推理性能标杆。cell-based KV cache 是实现参考。

### vLLM
**是什么**：高吞吐 LLM serving 框架。核心创新：PagedAttention + Continuous Batching。
**为什么重要**：服务端 LLM 推理的事实标准。AI Infra 必学项目。

### TensorRT-LLM
**是什么**：NVIDIA 的 LLM 推理优化框架。深度利用 Tensor Core、FP8、graph optimization。
**为什么重要**：NVIDIA GPU 上最高性能的 LLM 推理方案。

### TGI (Text Generation Inference)
**是什么**：HuggingFace 的 LLM serving 框架。易用性高，与 HF 生态深度集成。
**为什么重要**：快速搭建 LLM API 服务的常用选择。

### GGUF
**是什么**：llama.cpp 的模型文件格式。支持 INT4/INT8/FP16 量化，mmap 快速加载。
**为什么重要**：CPU 推理的事实标准格式。

### ONNX Runtime
**是什么**：跨平台的推理运行时，支持多种 execution provider（CPU/CUDA/TensorRT/OpenVINO）。
**为什么重要**：模型互操作性的标准。从 PyTorch/TF 导出到部署的桥梁。

---

## Tokenizer

### BPE (Byte Pair Encoding)
**是什么**：通过反复合并高频相邻符号来构建子词词表的分词算法。
**为什么重要**：GPT/LLaMA/Qwen 系列的标准 tokenizer。词表大小和 token 效率直接影响推理成本。

### BBPE (Byte-level BPE)
**是什么**：在字节级别（而非字符级别）操作 BPE。任何 UTF-8 文本都能编码。
**为什么重要**：消除 UNK token 问题。Qwen3 等模型使用 BBPE。

### SentencePiece
**是什么**：语言无关的分词工具库。支持 BPE 和 Unigram 两种算法。
**为什么重要**：训练 tokenizer 的标准工具。处理空格的方式（▁前缀）影响 token 效率。

### Special Token
**是什么**：ChatML 等对话模板使用的特殊标记。如 `<|im_start|>`, `<|im_end|>`, `<think>`, `</think>`。
**为什么重要**：Tokenizer 需要"最长匹配"来正确识别，不能错误拆分。

### Vocabulary Size (词表大小)
**是什么**：tokenizer 能识别的 token 总数。Qwen3 约 151K，LLaMA 约 32K。
**为什么重要**：大词表 = 更少 token（中文优势），但也 = 更大的 embedding 矩阵。

---

## 采样

### Temperature
**是什么**：`logits / T`。T < 1 使分布更尖锐（更确定），T > 1 使分布更平滑（更随机）。
**为什么重要**：控制生成多样性的最常用参数。

### Top-K Sampling
**是什么**：只保留概率最大的 K 个 token，其余置 0 后重新归一化再采样。
**为什么重要**：过滤长尾低概率 token，避免生成质量下降。

### Top-P (Nucleus) Sampling
**是什么**：累积概率达到 P 时截断，保留最小 token 集合。
**为什么重要**：比 Top-K 更灵活（动态调整候选数），当前最常用的采样策略。

### Repetition Penalty
**是什么**：对已生成的 token 的 logit 进行惩罚（正值除以 penalty，负值乘以 penalty）。
**为什么重要**：减少 LLM 的重复生成问题。应用在 softmax 之前。

### Greedy Decoding (贪心解码)
**是什么**：每步选择概率最大的 token（argmax）。temperature 无效。
**为什么重要**：确定性输出，适合需要一致性的场景（如翻译、代码生成）。

---

## 注意力机制扩展

### MLA (Multi-head Latent Attention)
**是什么**：DeepSeek-V2/V3 的注意力设计。通过低秩压缩减少 KV Cache。
**为什么重要**：极致省 KV Cache，但需要额外的压缩/解压缩计算。

### Sliding Window Attention (SWA)
**是什么**：每个 token 只注意窗口大小 W 内的最近 token，而非全部历史。
**为什么重要**：长上下文模型（如 Mistral）的常用技巧。减少 attention 计算量。

### Cross Attention (交叉注意力)
**是什么**：Q 来自 decoder，K/V 来自 encoder 的注意力。Encoder-decoder 架构的核心。
**为什么重要**：Seq2Seq 模型（T5, BART）的核心。多模态模型中图文交互。

### Causal Mask (因果掩码)
**是什么**：上三角为 -∞ 的注意力掩码。确保 token i 只能看到 token 0..i。
**为什么重要**：Decoder-only LLM 的必需机制。是自回归生成的数学基础。

### ALiBi (Attention with Linear Biases)
**是什么**：在 attention score 上直接加一个与距离成正比的偏置，无需位置编码。
**为什么重要**：某些模型（如 BLOOM）使用。比 RoPE 更简单但不能外推。

---

*这是术语速查表的精简版。完整版参见 [10_glossary.md](10_glossary.md) 中持续更新的术语。
更多详细概念请阅读对应章节的文档。*
