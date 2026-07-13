# Module 5: LLM 推理流水线——一个 Token 的奇幻漂流

> 想象你是一个 token。你诞生于用户的键盘，穿过模板的包装、分词的拆解、嵌入的映射、28 层 Transformer 的层层审视，最终化身为一个整数 ID，再被解码回人类可读的文字。这不是一次简单的 forward——这是一场精心编排的接力赛。

---

## 📋 学习目标

- [ ] 能画出 LLM 推理 8 个 Phase 的完整数据流图
- [ ] 能解释 ncnn_llm 为什么拆成 3 个 Net
- [ ] 能写出 BPE Tokenizer 的贪心合并算法
- [ ] 能解释 RoPE cache 的生成和复用机制
- [ ] 能说出 Prefill 和 Decode 在计算模式上的本质差异
- [ ] 能写出 generate() 循环的 7 个步骤
- [ ] 能解释 Weight Tying 省 600MB 的原理

---

## 1. 开场：为什么 LLM 推理和你想的不一样？

如果你之前只接触过图像分类模型，你可能会以为所有深度学习推理都是这样的：

```
输入 → model.forward() → 输出
搞定，收工。
```

但 LLM 推理完全不是这么回事。它更像一个**永不停歇的流水线**——每生成一个 token，就要跑一次完整的 forward，然后把中间结果（KV Cache）存下来供下一次使用，再根据概率分布掷一次骰子决定下一个 token，如此循环往复，直到模型自己喊停（输出 EOS）或者你设的上限到了。

这听起来很浪费——为什么不一次输出所有 token？因为**每个 token 的生成都依赖前面所有 token**。第 100 个 token 是什么，取决于第 1~99 个 token 是什么。这不是矩阵运算的并行世界，而是序列生成的串行命运。

> 💡 **核心认知**：LLM 推理 = 一个**带状态的循环**。状态 = KV Cache + 当前位置 + 历史 token。循环 = 每次迭代吐出一个新 token。

---

## 2. 总体架构：三个 Net，各司其职

说到"推理引擎"，你脑子里可能是"一个巨大的神经网络"。但 ncnn_llm 的做法很聪明——它把大模型**拆成三个独立的子网络**，就像把一条完整的流水线拆成三个工位：

```
┌──────────────────┐     ┌───────────────────┐     ┌──────────────────┐
│                  │     │                   │     │                  │
│    embed_net     │     │    decoder_net    │     │   proj_out_net  │
│                  │     │                   │     │                  │
│  "把数字变成向量" │────→│ "28层深度思考"     │────→│ "把向量变回文字"  │
│   (Gather查表)   │     │  (含KV Cache读写)  │     │   (Linear投影)   │
│                  │     │                   │     │                  │
└──────────────────┘     └────────┬──────────┘     └──────────────────┘
                                  │
                           KV Cache 在此出入
                     (这是整个系统的"记忆中枢")
```

### 2.1 为什么偏偏拆成 3 个？

这里面有三个精妙的考量：

**第一，KV Cache 的物理位置。** 注意力机制的 K 和 V 向量产生于 decoder 的中间层。如果整个模型是一个大网络，从外部根本没法"伸手进去"把 KV Cache 拿出来存、下次再塞回去。拆出独立的 decoder_net，KV Cache 就是它的输入输出，天然可管理——就像把发动机单独拆出来，换机油才方便。

**第二，权重的复用（Weight Tying）。** Embedding 和 Projection（输入嵌入层和输出投影层）在数学上是转置关系：`Projection(h) = h · W_embed^T`。Qwen3-0.6B 的词表有 151936 个 token，每个 1024 维：

```
一份 Embedding 矩阵的大小 = 151936 × 1024 × 2 bytes (FP16) ≈ 311 MB
如果各存一份 → 622 MB
Weight Tying 共享 → 只占 311 MB
节省 311 MB ≈ 端侧设备总内存的 15%~25%
```

ncnn_llm 让它们**共享同一份权重文件**——直接省下 311 MB。对于端侧设备来说，这可能就是"能跑"和"跑不了"的天壤之别。

**第三，内存的按需调度。** embed_net 只在 prefill 和每次 decode 的第一步跑一次；proj_out_net 只在需要输出 token 时才跑；decoder_net 才是那个每步都要跑的"重体力活"。拆开后，三者的内存可以独立管理——跑完就释放，不要让不需要的网络占着茅坑。

> 🔬 **源码印证**：ncnn_llm_gpt.cpp 构造函数中，`embed_net->load_model(bin)` 和 `proj_out_net->load_model(bin)` 加载的是**同一个 bin 文件**——这就是 Weight Tying 的直接证据。

### 2.2 为什么不拆成 2 个或 4 个？

这是一个很好的架构设计问题。对比各选型的取舍：

| 拆分方案 | 优点 | 缺点 | 适用场景 |
|---------|------|------|---------|
| **1 个大网** | 实现最简单，forward 一步到位 | KV Cache 被封闭在内部无法管理；无法做 Weight Tying | 实验原型 |
| **2 个网** (embed+decoder 合并 / decoder+proj 合并) | 比 1 个略好 | 要么牺牲 Weight Tying，要么牺牲 KV Cache 管理 | 折中方案 |
| **✅ 3 个网 (ncnn_llm 方案)** | KV Cache 自由出入 + Weight Tying + 按需调度 | 多一个网络的加载开销（但只一次） | **端侧推理部署** |
| **4 个网** (每层独立) | 极致的按需调度 | 管理复杂度爆炸；层间依赖折叠 | 分布式推理 / 流水线并行 |

3 Net 在"灵活性"和"工程复杂度"之间找到了最佳平衡点——不多不少，刚刚好。

### 2.3 各 Net 的活跃状态矩阵

不同推理阶段，三个 Net 的参与度截然不同：

| 推理阶段 | embed_net | decoder_net | proj_out_net |
|---------|:---------:|:-----------:|:------------:|
| Prefill（处理 Prompt） | ✅ 一次 | ✅ N 层 × 1 次 | ✅ 最后一步 |
| Decode（逐 token 生成） | ✅ 每步一次 | ✅ 每步 N 层 | ✅ 每步一次 |
| KV Cache 管理 | ❌ | ✅ 核心出入口 | ❌ |
| 多轮对话追加 | ❌（历史已缓存） | ✅ 只算新 token | ❌ |

这张表揭示了性能优化的方向：**decoder_net 是唯一在所有阶段都满载运行的组件**，它就是推理系统的"瓶颈所在"。

---

## 3. Phase 全景：一个 token 的 8 步生命旅程

让我们跟踪一个具体的请求——用户对 Qwen3 说"你好"。从用户按下回车，到模型吐出第一个字，再到整段回复完成，数据经历了什么？

### Phase 速查总表

在深入每个 Phase 之前，先用一张表看清全貌。**重点关注"耗时占比"这一列**——它揭示了优化的方向。

| Phase | 名称 | 输入 → 输出 | 关键算子 | 耗时占比 | 执行频率 |
|:-----:|------|-------------|----------|:-------:|:--------:|
| 0 | 配置加载 | model.json → 3个Net对象 | 文件 I/O | <1% | 启动时 1 次 |
| 1 | Prompt 模板 | 用户文本 → 格式化字符串 | 字符串拼接 | <1% | 每轮对话 |
| 2 | Tokenization | 文本 → token ID 数组 | BPE 贪心合并 | 1~3% | 每轮对话 |
| 3 | 位置编码 | position_id → cos/sin 向量 | RoPE 旋转 | 1~2% | Prefill 1次+每次 Decode |
| 4 | Embedding | token_id → 向量 [N, 1024] | Gather 查表 | 2~5% | Prefill 1次+每次 Decode |
| **5** | **Decoder** | 向量[N,1024] → 向量[N,1024] | **RMSNorm + SDPA + SwiGLU** | **~85%** | **每次 Decode 都跑** |
| 6 | Projection | 向量[1024] → logits[151936] | Linear(Gemm) | 5~8% | 每次 Decode |
| 7 | Sampling | logits → next_token_id | Softmax+TopK+TopP | 1~2% | 每次 Decode |
| 8 | Token to Text | token_id → UTF-8 文本 | Vocab 查表 | <1% | 每次 Decode |

> 💡 **优化铁律**：Phase 5（Decoder）占 85%+ 的时间。任何不触及 Phase 5 的优化，都是锦上添花，而非雪中送炭。KV Cache、FlashAttention、Continuous Batching——所有这些技术的终极目标，都是让 Phase 5 跑得更快。

### Phase 0: 启程——配置加载（启动时执行一次）

```
model.json → 3个 Net + Tokenizer + RoPE参数

这就像开餐厅前的准备工作：
你把菜谱(model.json)看一遍，确认今天有几层灶台(attn_cnt=28)，
烤箱温度多高(rope_theta=100000)，用多大碗装菜(head_dim=128)。
```

别看这一步简单——整个推理系统的全部行为都靠这个 JSON 文件定义。换一个模型？改 model.json 就行，代码一行不用动。这是**数据驱动**的设计哲学。

**一个真实的 model.json 长什么样？** 下面是 Qwen3-0.6B 的配置：

```json
{
  "embed_param": 157,          // Embedding 参数数量
  "attn_cnt": 28,              // Decoder 层数 = 28
  "head_dim": 128,             // 每头维度 = 128
  "q_n_head": 16,              // Q 的头数 = 16
  "kv_n_head": 8,              // K/V 的头数 = 8（GQA，K/V 只有 Q 的一半）
  "dim_model": 1024,           // 隐藏层维度 = 1024
  "dim_ffn_hidden": 3072,      // FFN 中间维度 = 3072
  "max_position": 131072,      // 最大上下文长度 = 128K
  "rope_theta": 1000000,       // RoPE 基频 = 1,000,000
  "norm_eps": 1e-6,            // RMSNorm epsilon
  "do_lm_head_weight_bias": false  // LM Head 不加 bias
}
```

> 💡 **注意 `rope_theta` 的变化**：Qwen2 用的是 10000，Qwen3 升级到 1000000。更大的 theta 意味着 RoPE 旋转频率更慢，相同位置下向量旋转的角度更小——这是为了支持 128K 超长上下文而做的调整。如果 theta 太小，长距离位置的旋转角度会"转太多圈"，导致位置信息混淆。

### Phase 1: 包装——Prompt Template

```
用户输入 "你好"
        ↓  ChatML 模板包装
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
你好<|im_end|>
<|im_start|>assistant
                              ← 注意这里！模型从这里开始写
```

你可能会想：为什么要把"你好"搞得这么复杂？答案很简单——**模型是在这种格式上训练出来的**。训练时喂给它的每一条数据都是 `<|im_start|>角色\n内容<|im_end|>` 的格式，推理时如果不保持完全一致，模型就会"懵"——它看到的是陌生的格式，输出的质量会断崖式下降。

最下面那个 `<|im_start|>assistant\n` 是 `add_generation_prompt` 自动拼接的。它相当于告诉模型："好了，轮到你说话了，从这开始写吧。"

> 💡 **生活类比**：你给朋友写信，开头要写"亲爱的某某"，结尾要写"此致敬礼"。如果你突然收到一封没有称呼没有落款的信，你也会觉得奇怪——模型对格式的敏感度，比你对书信格式的敏感度还要高。

### Phase 2: 拆解——BPE Tokenization

```
"<|im_start|>assistant\n" → [151644, 220, ..., 151645, ...]

文本 → Byte Encode(字节级转义) → Pretokenize(预分词) → BPE Merge(贪心合并) → Vocab查找 → Token IDs
```

大模型不认识文字——它只认识整数。Tokenizer 就是文字和整数之间的"翻译官"。但它不是简单的查字典——它用了一种叫 BPE（Byte Pair Encoding）的**学习到的压缩算法**。

**BPE 的核心思想非常好懂**：把一篇文章里最常一起出现的两个字符合并成一个新符号，反复合并，直到词表够大为止。就像小时候学汉字——先学"日"和"月"，发现它们老是一起出现，就合并成"明"。

推理时把训练时的合并顺序倒过来执行——这就是那个"贪心合并"算法：

```
symbols = ["h", "e", "l", "l", "o"]
第1轮: (h,e) rank=0 最小 → 合并为 "he"  → ["he", "l", "l", "o"]
第2轮: (he,l) rank=1 最小 → 合并为 "hel" → ["hel", "l", "o"]
第3轮: (l,o) rank=3 vs (hel,l) rank=2 → 合并 (hel,l) → ["hell", "o"]
第4轮: (hell,o) rank=4 → 合并为 "hello" → ["hello"]
```

> 🔬 **源码位置**：`bpe_tokenizer.cpp` 中的 `BpeForPiece()` 函数。每次找 rank 最小的相邻对，合并、删除、继续——直到找不到任何可合并的对为止。

**这对推理系统意味着什么？** 同样一句话，中文需要的 token 数是英文的 1.5~2 倍（因为中文字符多、合并效率低）。而每个 token 都要在 KV Cache 里占一个位置——**中文 LLM 的显存压力天然比英文大**。

### Phase 3: 定位——RoPE 位置编码

Transformer 的注意力机制本身是个"脸盲"——它看不出"我 爱 你"和"你 爱 我"有什么区别，因为注意力只看向量之间的相似度，不关心位置。RoPE 就是给每个 token 贴上"位置标签"的方法。

但 RoPE 的做法非常优雅——它**不是加一个位置向量**，而是把每个 token 的 Query 和 Key 向量**旋转一个角度**：

```
位置 0: 不旋转
位置 1: 旋转 1 步
位置 i: 旋转 i 步

精妙之处：
Q_pos_i · K_pos_j = (旋转了i步的q) · (旋转了j步的k)
                   = q · (旋转了(j-i)步的k)
                   ↑ 只依赖相对位置差！
```

这意味着模型不需要记忆"位置 537 的特征是什么"——它只需要知道"两个 token 之间隔了多远"。换个角度说：**RoPE 编码的是相对位置，不是绝对位置**。这就是为什么 RoPE 支持上下文外推——即使推理时序列比训练时更长，只要相对距离的规律不变，模型就能适应。

> 💡 **生活类比**：RoPE 不是在每本书上写"这是第 537 本书"，而是在每个书架上贴"我左边第 3 本"。当你把书架延长时，不需要重新编号。

**具体数值示例**——用 2 维向量感受 RoPE 的旋转：

```
设 q = (1, 0), 位置 pos = 1, theta = 10000

θ = pos × 10000^(-0/1) = 1 × 1 = 1 (弧度)  ← 简化 2 维情况

RoPE 旋转公式:
┌ q0' ┐   ┌ cos(θ)  -sin(θ) ┐ ┌ q0 ┐
└ q1' ┘ = └ sin(θ)   cos(θ) ┘ └ q1 ┘

q0' = 1 × cos(1) - 0 × sin(1) = 0.540
q1' = 1 × sin(1) + 0 × cos(1) = 0.841

所以 q 在位置 1 被旋转了 1 弧度 ≈ 57.3°
验证: q0'² + q1'² = 0.540² + 0.841² = 1.0 (旋转保持模长不变！)
```

在 Qwen3 的 128 维 head 中，每个维度对的旋转频率不同——第 0/1 维转得最快（捕获 token 级别的位置），第 126/127 维转得最慢（捕获句子级别的长期依赖）。

**多轮对话时特别要注意一点**：position_id 必须连续累加。第 1 轮用了位置 0~99，第 2 轮就要从 100 开始，而不是又从 0 开始——否则模型会觉得"第二轮的第一个 token 和第一轮的第一个 token 在同一个位置"，这会严重混淆位置信息。

### Phase 4: 具象——Embedding 词嵌入

```
token_ids [N] → embed_net (查表) → token_embed [N, 1024]

本质: E[token_ids[i], :]  ← 从 151936×1024 的大矩阵里取 N 行
```

每个 token ID 对应一个 1024 维的向量。你可以把这个向量理解为这个 token 的"语义画像"——"猫"和"狗"的向量在空间中很接近，"猫"和"汽车"的向量则相距很远。这不是人手工定义的，而是模型在训练时自己学会的。

> 💡 **有趣的事实**：Embedding 矩阵的参数量 = 151936 × 1024 ≈ 1.56 亿。Qwen3-0.6B 总共才 6 亿参数——光"查字典"这一步就占了 26%！

### Phase 5: 思考——Decoder 前向（全流程最重的部分）🔬

这是整个推理系统的**核心战场**——28 层 Transformer，每层 35 个算子，总共约 1017 个算子。这里分两个截然不同的场景：

#### 场景 A：Prefill（首次处理 prompt）

```
你刚输入 "你好，请介绍一下深度学习"（假设 12 个 token）

Prefill 做的事:
  一口气把这 12 个 token 全部送进 decoder
  → 12 个 Q 向量, 12 个 K 向量, 12 个 V 向量
  → Q·K^T: 每个 Q 和所有 12 个 K 算相似度 → 12×12 的注意力矩阵
  → 写 KV Cache: 12 个 token 的 K 和 V 全部存下来

特点是"大力出奇迹":
  大矩阵乘法，GPU 干得热火朝天
  12×12 的 attention 矩阵虽然不大，但架不住 28 层 × 16 个头
  GPU 利用率 >80%——这才叫"物尽其用"
```

#### 场景 B：Decode（后续逐 token 生成）

```
上一个 token 是 "深度"（token_id = 3728）

Decode 做的事:
  只把这 1 个 token 送进 decoder
  → 只有 1 个 Q 向量
  → 但 K 和 V 呢？从 KV Cache 里读之前存的所有历史！
    （假设现在已经有 128 个历史 token）
  → Q·K^T: 1 个 Q × 128 个 K → 1×128 的注意力向量
  → 这个新 token 的 K 和 V 追加到 KV Cache → 现在有 129 个

特点是"举轻若重":
  计算量很小（只有 1 个 token 的矩阵乘法）
  但内存读取量很大（要把 128 个 token 的 28 层 K/V 全读一遍）
  → 绝大多数时间在等内存，GPU 利用率 <30%
  → bottleneck 不是算力，是带宽！
```

> 💡 **一句话记住**：Prefill 是"一辆大卡车一次装满"，Decode 是"一个人一趟趟搬砖"。

**Prefill vs Decode 的 FLOPs 定量对比（Qwen3-0.6B，12 token prompt → 100 token 生成）**：

| 指标 | Prefill | Decode（单步） | Decode（100步累计） |
|------|---------|:-------------:|:------------------:|
| 每步输入 token 数 | 12 | 1 | 1×100 |
| 单层 Q 投影 FLOPs | 2×12×1024×2048 ≈ **50M** | 2×1×1024×2048 ≈ **4.2M** | 420M |
| 单层 SDPA FLOPs | 2×16×12×12 ≈ **4.6K** | 2×16×1×S_avg≈16K (S 逐渐增大) | ~1600K |
| 单层所有算子 FLOPs | ~110M | ~9M | ~900M |
| 28 层总计 FLOPs | ~**3.1 GFLOPS** | ~0.25 GFLOPS | ~**25 GFLOPS** |
| 瓶颈 | **算力（Compute-bound）** | **带宽（Memory-bound）** | 带宽 |
| 理想 GPU 利用率 | 80%+ | <30% | <30% |

> 注：SDPA 在 Decode 阶段的计算量随序列长度线性增长，但在端侧场景（小 batch）下，整体瓶颈仍是 KV Cache 的内存读取而非 SDPA 计算。这就是为什么大 batch + FlashAttention 才能显著提升 Decode 速度。

#### Decoder 内部到底发生了什么？（以 Qwen3 Layer 0 为例）

```
x_in [seq, 1024]  ← 输入的 embedding
  │
  ├─ RMSNorm → 标准化到稳定范围
  │   └─ Split → 分成三路
  │
  ├── Q 路: Gemm(1024→2048) → 拆成16个头 → QK-Norm → RoPE旋转
  ├── K 路: Gemm(1024→1024) → 拆成8个头  → QK-Norm → RoPE旋转 → 复制到16个(GQA)
  ├── V 路: Gemm(1024→1024) → 拆成8个头  → 复制到16个(GQA)
  │
  ├── SDPA: Q·K^T → /√128 → +mask → softmax → ·V
  │         ↑ 此处读/写 KV Cache
  │
  ├── O Proj: Gemm(2048→1024) → 合并多头输出
  ├── + 残差连接 (x_in 直接跳过 attention 加过来)
  │
  ├─ RMSNorm → Split → gate/up 两路
  │   gate: Gemm(1024→3072) → Swish激活 → ┐
  │   up:   Gemm(1024→3072) ──────────────→ Mul → Gemm(3072→1024)
  │                                          ↑ gate ⊙ up
  └── + 残差连接 → x_out

以上 × 28 层 = 一次完整的 decoder forward
```

> 🔬 **关键源码**：SDPA 层是 KV Cache 拼接的发生地。`ncnn sdpa.cpp` 中 `forward()` 函数的 69-87 行——如果 `past_seqlen > 0`，就分配新内存，把旧的 `past_key` 和新的 `cur_key` 用 `memcpy` 拼在一起。

### Phase 6-7: 抉择——Projection → Sampling

```
Phase 6: Projection
  hidden [1, 1024] × W^T [1024, 151936] → logits [151936]
  "把 1024 维的思想压缩成 151936 个候选词的得分"

Phase 7: Sampling
  logits 是原始分数，不能直接用。需要一套"烹饪流程":

  logits [151936 个浮点数]
    │
    ├─ Repetition Penalty  ← "说过的话不要说第二遍"
    │   已出现的 token: logit>0 则除以 penalty, logit<0 则乘以 penalty
    │
    ├─ Temperature Scaling  ← "控制创意程度"
    │   T=0.7: 分布更尖锐 → 更确定，适合翻译
    │   T=1.2: 分布更平滑 → 更随机，适合写诗
    │
    ├─ Softmax → 变成概率分布 (所有值在 0~1 之间, 总和=1)
    │
    ├─ Top-K  ← "只考虑最好的 K 个选择"
    │   K=50: 从 15 万个候选缩到 50 个
    │
    ├─ Top-P  ← "累积概率达到 P 就截断"
    │   P=0.9: 保留最可能的几个 token，使累积概率刚好 ≥90%
    │
    └─ 最终选择:
        do_sample=1 → 按概率随机选 (多样性高)
        do_sample=0 → 选概率最大的 (确定性输出)
```

> 💡 **为什么要有 Temperature？** 想象你在玩一个文字冒险游戏。T=0 时，你每次面对同样的场景都做同样的选择——可靠但无聊。T=1 时，你根据概率随机选——每次体验都不同。T→∞ 时，你完全随机选——言语混乱，毫无逻辑。好的 Temperature 在"合理"和"有趣"之间取平衡。

> 💡 **Repetition Penalty 为什么要在 softmax 之前做？** 因为如果在 softmax 之后做，概率分布的总和就不是 1 了，需要重新归一化——麻烦且容易引入数值问题。在 logits 阶段直接惩罚，softmax 会自动处理归一化。

### Phase 8: 还原——Token to Text

```
next_token_id → BPE Decode → ByteDecode → UTF-8 → 流式输出给用户

BPE Decode 的反向过程:
  token_id → 查 id_to_token_ 表 → token 字符串
  → 跳过特殊 token (如 <|im_end|> 不输出给用户)
  → 如果是字节级编码的 token，ByteDecode 还原为原始 UTF-8 字节
```

**具体例子——一个 token 如何变回汉字？**

假设模型生成了 token_id = 364，查表得到 `token_str = "世界"`。在 GPT 系列 tokenizer 中，这个过程是：

1. **查 id_to_token**：`id_to_token[364]` → 输出 token 字符串（可能是多字节 UTF-8 编码）
2. **处理字节级 token**：BPE 词表中包含很多形如 `<0xE4>`、`<0xB8>`、`<0x96>` 的字节级 token。这些 token 不直接拼接，而是先收集所有字节，再用 UTF-8 解码器一次性还原为字符。例如 `[<0xE4>, <0xB8>, <0x96>]` → `"世"`。
3. **拼接与输出**：将解码后的字符串片段拼接到已有输出后，如果是流式场景，通过 SSE（Server-Sent Events）逐个推送字符。

```
Token IDs 流:  [151645, 364, 597, 151644]   ← 模型依次生成的 token
        │
        ▼ 逐 token decode
字符串片段:  ["", "世界", "你好", ""]
        │  ↑ 跳过特殊 token <|im_start|>(151645) 和 <|im_end|>(151644)
        ▼ 拼接
最终输出:  "世界你好"
```

> 💡 **为什么有些 token 解码为空字符串？** 因为特殊 token（如 `<|im_start|>`, `<|im_end|>`, `<|endoftext|>`）在训练时被赋予了语义，但它们不应该出现在给用户的文本中。decode 时检测到这些 token ID 就跳过——这就是 `skip_special_tokens=True` 的默认行为。

---

## 4. 完整自回归生成循环——所有 Phase 的协作舞蹈

把 Phase 4~8 串起来，就是推理系统的**心跳循环**：

```cpp
// ncnn_llm_gpt.cpp generate() — 每一行背后都是血与火的工程决策

for (int step = 0; step < cfg.max_new_tokens; ++step) {

    // 第 0 步: "该闭嘴了吗？"
    if (ctx->cur_token == eos) break;  // 模型说完了
    // EOS = End Of Sequence — 模型自己学会什么时候结束

    // Step 1: Token → 向量 (Phase 4)
    ncnn::Mat cur_embed;
    embed_net->extract("in0", token_mat, "out0", cur_embed);
    // 这一步很快——就是一次查表。但因为词表有 15 万行，
    // 在 CPU 上做 Gather 还是有一定开销

    // Step 2: 生成位置标签 (Phase 3)
    generate_rope_embed_cache(1, head_dim, ctx->position_id++, cos, sin, theta);
    // position_id 每次 +1——这是多轮对话中位置连续性的保证

    // Step 3: 构建注意力掩码 (Phase 5 的准备)
    ncnn::Mat mask(ctx->kv_cache[0].first.h + 1, 1);
    mask.fill(0.f);  // 全零 = 允许看到所有历史 token
    // 为什么全零就够了？因为 causal mask 已经在 SDPA 内部通过
    // "未来 token= -∞" 的方式实现了，这里只需要控制可见范围

    // Step 4: 深度思考 (Phase 5) — 整个系统最重的部分
    ncnn::Extractor ex = decoder_net->create_extractor();
    ex.input("in0", cur_embed);
    ex.input("in1", mask);
    ex.input("in2", cos);
    ex.input("in3", sin);

    // 把 28 层的旧 KV Cache 全部喂进去
    for (int i = 0; i < attn_cnt; i++) {
        ex.input("cache_k" + str(i), kv_cache[i].first);
        ex.input("cache_v" + str(i), kv_cache[i].second);
    }

    // 取出更新后的 28 层 KV Cache
    for (int i = 0; i < attn_cnt; i++) {
        ex.extract("out_cache_k" + str(i), kv_cache[i].first);
        ex.extract("out_cache_v" + str(i), kv_cache[i].second);
    }

    ex.extract("out0", decode_out);
    // 这一步在 CPU 上可能耗时几十毫秒——
    // 28 层 × 每次读全部 KV Cache × memcpy 拼接

    // Step 5: 向量 → 分数 (Phase 6)
    ncnn::Mat logits;
    proj_out_net->extract("in0", decode_out, "out0", logits);
    // 1024维 → 151936维的线性变换
    // 如果 projection 和 embedding 共享权重，这一步用的是 embedding 矩阵的转置

    // Step 6: 惩罚重复 (Phase 7 的第一步)
    for (int t : history) {
        if (logits[t] < 0) logits[t] *= cfg.repetition_penalty;
        else               logits[t] /= cfg.repetition_penalty;
    }

    // Step 7: 掷骰子选下一个 token (Phase 7 的核心)
    softmax(logits, cfg.temperature);
    apply_top_k(logits, cfg.top_k);
    apply_top_p(logits, cfg.top_p);
    int next = cfg.do_sample ? sample(logits) : argmax(logits);

    // 更新状态，准备下一轮
    ctx->cur_token = next;
    history.insert(next);
}
// 循环结束——一段完整的回复产生了
```

> 💡 **为什么 prefill 后要"单独处理最后一个 token"？** Prefill 的 decoder 输出是 `[N, 1024]`——所有 N 个位置的 hidden state。但我们只关心**最后一个位置**的输出（它"看过了"整个 prompt），用它来预测第一个生成 token。所以 prefill 结束后，取最后一个 token 的 hidden state 单独做一次 decode forward——这次 forward 产生的 hidden state 才用来做第一个 token 的采样。之后进入正常的 decode 循环。

---

## 5. Prefill vs Decode——两种截然不同的人生

这两个阶段的核心矛盾，是所有 LLM 推理优化的出发点：

| 维度 | Prefill | Decode |
|------|---------|--------|
| **输入规模** | Prompt 全部 N 个 token | 每次只有 1 个新 token |
| **矩阵大小** | Q: [16, N, 128]，大！ | Q: [16, 1, 128]，小！ |
| **注意力矩阵** | [16, N, N] — N²！ | [16, 1, S] — 1×S |
| **谁在忙** | GPU 的计算单元 | GPU 的显存带宽 |
| **瓶颈在哪** | 算力 (Compute-bound) | **带宽 (Memory-bound)** |
| **GPU 利用率** | >80% — 干劲十足 | <30% — 大部分时间在等数据 |
| **影响什么** | TTFT — 第一个字出来的速度 | TPOT — 后续字出来的速度 |

> 💡 **直觉理解**：Prefill 像一个壮汉在搬一车砖——一次全部卸完，力量是瓶颈。Decode 像一个老人在一盏一盏地数灯——每数一盏都要走回仓库看一眼，腿脚是瓶颈。

这就是为什么：
- **Prefill 优化方向**：更大的 batch、更好的 GEMM kernel、Tensor Core 利用
- **Decode 优化方向**：KV Cache 压缩、FlashAttention、Continuous Batching 增加有效 batch

---

## 6. 多轮对话——KV Cache 的"记忆复用"

多轮对话是 LLM 应用最常见的场景，而 ncnn_llm 处理它的方式非常优雅：

```
Round 1: 用户说 "你好"
  prefill("system+user1") → ctx.kv_cache 存了 100 个 token 的 KV
  generate(ctx) → 模型回复了 50 个 token → 现在有 150 个

Round 2: 用户说 "你能做什么？"
  ctx2 = ctx.clone()  ← 浅拷贝！共享内存！
  prefill("user2", ctx2) → 只计算 "你能做什么？" 这 5 个 token 的 KV
  → 追加到已有 150 个后面 → 共 155 个
  → system + user1 + assistant1 的 KV 完全不用重算！

关键原理:
  clone() 浅拷贝了 Mat 的 data 指针
  → 多轮对话中，历史 KV Cache 零拷贝共享
  → 新用户消息只需要计算自己那部分的 KV
```

> 💡 **给面试官的解释**：ncnn_llm 的 clone 是浅拷贝——新旧上下文共享 KVCache 的底层 data 指针。这意味着多轮对话中历史消息的注意力计算**完全不需要重新执行**，只需追加新消息的 KV 即可。但注意，这要求新的 prefill 在旧 cache 的基础上追加，而不是覆盖——`position_id` 必须连续递增。

---

## 7. 一个你可能没想过的问题

**中文为什么比英文"贵"？**

同样的意思，中文通常比英文多用 1.5~2 倍的 token。这不是模型的问题，而是 BPE tokenizer 的固有特性——英文单词天然由字母组成，BPE 可以高效合并；中文字符本身就是一个"原子"，合并空间更小。

```
"Hello, how are you?"     → ~6 tokens
"你好，你最近怎么样？"      → ~15 tokens
```

每个 token 都在 KV Cache 里占一个位置，都在 prefill 时产生一次计算。所以说——**中文 LLM 的推理成本天生比英文高**。这不是偏见，是 tokenization 效率和信息熵的物理规律。

---

## 🛠️ 动手练习

1. **Trace 完整链路**：对 "你好" 做一次完整的 tokenize，然后手动写出每个 Phase 的输入输出 shape。

2. **BPE 手算**：对 `"hello"` 做 BPE。初始拆成 `["h","e","l","l","o"]`，已知 merges 为 `(h,e,rank=0)`, `(he,l,rank=1)`, `(hel,lo,rank=2)`，写出每一轮的合并结果。

3. **RoPE 魔法**：对向量 `(1, 0)` 在 `pos=1, theta=10000` 时做 2D RoPE 旋转，验证旋转矩阵的正交性。

4. **打点测时**：修改 ncnn_llm 的 generate() 函数，在 Phase 4/5/6/7 前后各插入一个计时点，测量真实耗时分布。你会惊讶地发现——**Decoder 占了 85%+ 的时间**。

---

## 📚 延伸阅读

- [ncnn_llm_original_README.md](../ncnn_llm_original_README.md) — 逐行源码分析，每个 Phase 的完整代码走读 (2683 行)
- [03_llm_inference_pipeline.md](../../docs/03_llm_inference_pipeline.md) — 系统笔记版，架构视角的补充
- [Module 5 Notebook](./module-05-notebook.ipynb) — 配套 Jupyter Notebook，用代码和可视化验证每个 Phase

---

*下一模块: [Module 6: KV Cache 系统——记忆的艺术](../module-06-kv-cache-system/README.md)*
