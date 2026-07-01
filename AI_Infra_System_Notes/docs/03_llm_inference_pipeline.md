# 03｜LLM 推理流水线：从文本到 token 的完整旅程

> LLM 推理不是一次性 forward——它是以 **KV Cache 为状态**、**自回归循环为骨架**、**8 个 Phase 为流水线**的持续运行系统。

---

## 1. 全局视角：一个 token 的诞生

### 1.1 时间线

```
用户输入 "你好，请介绍一下深度学习"
        │
        ▼
┌─ Phase 0: 配置加载 ─────────────────────── 启动时执行一次 ─┐
│  model.json → 3 个 ncnn::Net + Tokenizer + RoPE 参数       │
└───────────────────────────────────────────────────────────┘
        │
        ▼
┌─ Phase 1: Prompt Template ────────────────────────────────┐
│  "你好，请介绍一下深度学习" → ChatML 格式包装              │
│  "<|im_start|>user\n你好，请介绍一下深度学习<|im_end|>\n   │
│   <|im_start|>assistant\n"                                 │
└───────────────────────────────────────────────────────────┘
        │
        ▼
┌─ Phase 2: Tokenization (BPE) ─────────────────────────────┐
│  文本 → Byte Encode → BPE Merge → vocab 查找 → Token IDs  │
│  "...assistant\n" → [151644, 220, ..., 151645, ...]       │
│  Prompt 被编码为 N 个 token IDs                            │
└───────────────────────────────────────────────────────────┘
        │
        ▼
┌─ Phase 3: RoPE Cache ────────────────────────────────────┐
│  position 0..N-1 → inv_freq → cos/sin cache               │
│  cos_cache: [N, head_dim/2]    sin_cache: [N, head_dim/2] │
└───────────────────────────────────────────────────────────┘
        │
        ▼
┌─ Phase 4: Embedding ─────────────────────────────────────┐
│  token_ids [N] → embed_net (Gather) → [N, hidden_dim]    │
│  每个 token ID 查表得到 hidden_dim 维向量                  │
└───────────────────────────────────────────────────────────┘
        │
        ▼
┌─ Phase 5: Decoder (Prefill) ─────────── 最重的阶段 ──────┐
│  输入: token_embed [N, hidden_dim]                         │
│  通过 28 层 Transformer                                    │
│   每层: RMSNorm → QKV Proj → RoPE → SDPA → O Proj → +   │
│         → RMSNorm → SwiGLU → +                            │
│  输出: hidden_state [1, hidden_dim] (最后一个 token)       │
│         + 28 层 KV Cache (prefill 写入的初始 KV)          │
└───────────────────────────────────────────────────────────┘
        │
        ▼
┌─ Phase 6: Projection ────────────────────────────────────┐
│  hidden_state [1, hidden_dim] → proj_out_net → logits    │
│  [1, hidden_dim] × [hidden_dim, vocab_size]               │
│  → logits [vocab_size] = [151936]                        │
└───────────────────────────────────────────────────────────┘
        │
        ▼
┌─ Phase 7: Sampling ─────── 唯一有随机性的地方 ───────────┐
│  logits → Repetition Penalty → Temperature → Softmax      │
│        → Top-K → Top-P → Sample/Argmax → next_token_id   │
└───────────────────────────────────────────────────────────┘
        │
        ▼
┌─ Phase 8: Token to Text ─────────────────────────────────┐
│  next_token_id → BPE Decode → UTF-8 → 流式输出给用户      │
└───────────────────────────────────────────────────────────┘
        │
        ▼
   ┌──────────────────────────────────────────┐
   │  Decode Loop (重复直到 EOS/max_tokens)    │
   │  Step 1-token → Embed → RoPE → Decoder   │
   │  → KV Cache 更新 → Projection → Sample   │
   └──────────────────────────────────────────┘
```

### 1.2 数据形状变化全链路

```
阶段              输入 Shape              输出 Shape             变化
─────────────────────────────────────────────────────────────────────
0. Config         model.json 文件         3 个 Net + params       startup
1. Template       raw text                formatted text          text→text
2. Tokenize       formatted text          token_ids [N]           text→ints
3. RoPE           无                      cos[N,hd/2] sin[N,hd/2] 预计算
4. Embed          token_ids [N]           token_embed [N, 1024]   ints→float
5. Decoder        token_embed [N, 1024]   hidden_state [1, 1024]  N→1
                  + KV Cache (写入)       + KV Cache (28层)
6. Projection     hidden_state [1, 1024]  logits [151936]         1024→vocab
7. Sampling       logits [151936]         next_token (int)        151936→1
8. Decode         next_token (int)        text                    int→text
─────────────────────────────────────────────────────────────────────
N = prompt token 数
[1, 1024] 中的 1 = 仅取最后一个位置的 hidden state
```

---

## 2. 每个 Phase 的深度解析

### 2.1 Phase 0: 模型配置加载

这是**一切推理的基础**——决定了模型的全部参数。

```cpp
// ncnn_llm_gpt.cpp 构造函数: 加载 model.json
json config;
ifstream ifs(model_path + "/model.json"); ifs >> config;

// 从中提取的关键信息:
int attn_cnt = config["setting"]["attn_cnt"];          // 28 层
int rope_head_dim = config["setting"]["rope"]["rope_head_dim"];  // 128
float rope_theta = config["setting"]["rope"]["rope_theta"];      // 100000.0
int eos = bpe->token_to_id().at(config["tokenizer"]["eos"]);     // EOS ID

// 创建 3 个网络
embed_net->load_param("qwen3_embed_token.ncnn.param");
embed_net->load_model("qwen3_embed_token.ncnn.bin");
decoder_net->load_param("qwen3_decoder.ncnn.param");
decoder_net->load_model("qwen3_decoder.ncnn.bin");
proj_out_net->load_param("qwen3_proj_out.ncnn.param");
proj_out_net->load_model("qwen3_embed_token.ncnn.bin");  // 共享权重!
```

**关键设计决策：为什么拆成 3 个 Net？**

```
1 个 Net 方案:
  输入 token_ids → 大网 → logits
  问题: KV Cache 难以提取（它在 decoder 中间层）

3 个 Net 方案:
  embed_net:   token → embedding（只在 prefill 和每次 decode 运行）
  decoder_net: embedding → hidden_state + KV Cache I/O（每步都要跑）
  proj_out_net: hidden_state → logits（只在需要采样时运行）

  优势:
  - KV Cache 作为 decoder 的输入输出，天然可管理
  - embedding 和 projection 可共享权重（省 ~600MB）
  - 三段网络可分别选择 CPU/Vulkan
```

### 2.2 Phase 1: Prompt Template

```
为什么大模型不能"裸"接收用户输入？

因为模型是在特定格式上训练的。如果你训练时用的是:
  "<|im_start|>user\n{问题}<|im_end|>\n<|im_start|>assistant\n{回答}<|im_end|>"
那你推理时也必须用同样的格式。

Qwen3 使用 ChatML 格式:

  <|im_start|>system
  You are a helpful assistant.<|im_end|>
  <|im_start|>user
  你好<|im_end|>
  <|im_start|>assistant
  [模型从这里开始生成]

add_generation_prompt = true → 自动加上 "<|im_start|>assistant\n"
相当于告诉模型 "该你说话了"
```

### 2.3 Phase 2: Tokenization (BPE)

BPE 分词已在 `doc/README.md` 中做了 2683 行的详尽分析。这里从推理系统视角总结关键点：

```
BPE 对推理系统的影响:

1. Token 数决定计算量
   "你好" → 2 个 token
   "Hello" → 1 个 token
   "𬱖" (生僻字) → 4+ 个 token
   → 同样的"意思"，中文通常比英文多 1.5-2× token

2. Token 数决定 KV Cache 长度
   每个 token 的 K/V 都要存储
   → 中文 LLM 的 KV Cache 压力比英文大

3. Special Token 的处理需要"最长匹配"
   "<|im_start|>" 和 "<|im_end|>" 不能错误拆分为 "<|im" + "_start|>"
   ncnn_llm 的 encode() 在遍历文本时优先匹配 special tokens

4. Tokenizer 的缓存
   同一个 piece（子词片段）的 BPE 结果会被缓存
   → 多轮对话中，历史消息不变，Tokenizer 结果可复用
```

### 2.4 Phase 3: RoPE 位置编码

```
RoPE 的核心洞察: 用旋转角度差编码相对位置

位置 0: 不旋转
位置 1: 旋转 1 步
位置 i: 旋转 i 步

Q_i · K_j = (R_i · q_i) · (R_j · k_j)
          = q_i · (R_i^T · R_j) · k_j
          = q_i · R_{j-i} · k_j
          ↑ 只依赖位置差 (j-i)，不依赖绝对位置！

多轮对话时的位置连续性:
  Round 1: position 0..99
  Round 2: position 100..199 (不是 0..99！)
  → 通过 ctx->position_id 跟踪绝对位置
  → RoPE cache 生成时使用累计的 position_id
```

### 2.5 Phase 4: Embedding

```
Embedding = 查表操作

E = embedding_matrix[vocab_size, hidden_dim]
embed[i] = E[token_ids[i], :]  ← Gather 操作

与 Projection 的权重共享 (Weight Tying):
  Embedding: token → E[token_id]          ← 从 E 中取一行
  Projection: hidden → E^T @ hidden       ← 用 E^T 做矩阵乘法

  共享后: vocab × hidden × 4 bytes = 151936 × 1024 × 4 ≈ 592 MB
  不共享: 2 × 592 MB ≈ 1.2 GB
  → 端侧设备上，省 600MB 意义重大
```

### 2.6 Phase 5: Decoder — Prefill vs Decode

```
这 28 层 Transformer 是计算的核心。

Prefill:
  输入: [N, 1024]  ← 所有 prompt token 一次送入
  每层:
    QKV Proj: [N, 1024] × [1024, 2048] = [N, 2048]  (大矩阵!)
    SDPA: [16, N, 128] × [16, 128, N] → [16, N, N]  (N×N 注意力矩阵)
    MLP: GEMM × 3 (gate/up/down)
  KV Cache: 写入 N 个 token 的 KV
  输出: hidden_state [N, 1024]，但只取最后一个位置 [1, 1024]
  特点: 计算密集，GPU 利用率高

Decode:
  输入: [1, 1024]  ← 只有 1 个 token
  每层:
    QKV Proj: [1, 1024] × [1024, 2048] = [1, 2048]  (小矩阵!)
    SDPA: [16, 1, 128] × [16, 128, S] → [16, 1, S]  (S = 历史 token 数)
    KV Cache: 读取全部 S 个 token 的 KV，追加新 token
  输出: hidden_state [1, 1024]
  特点: 内存密集，瓶颈在 KV Cache 读取
```

### 2.7 Phase 6-7: Projection → Sampling

```python
# 从 hidden_state 到 next_token 的完整过程

# Projection
hidden = decoder_output         # [1, 1024]
logits = hidden @ W_proj.T      # [1, vocab_size] = [1, 151936]

# Repetition Penalty (在 softmax 之前！)
for token_id in history:
    if token_id >= vocab_size:
        continue
    if logits[token_id] < 0:
        logits[token_id] *= repetition_penalty  # penalty > 1 让负值更负
    else:
        logits[token_id] /= repetition_penalty  # penalty > 1 让正值变小

# Temperature Scaling
logits = logits / temperature   # T<1: 更确定; T>1: 更随机

# Softmax
logits = logits - max(logits)   # 数值稳定
probs = exp(logits) / sum(exp(logits))

# Top-K Filtering (只保留概率最大的 K 个)
threshold = nth_largest(probs, K)
probs[probs < threshold] = 0
probs = probs / sum(probs)    # 重新归一化

# Top-P Filtering (累积概率截断)
sorted_probs = sort(probs, descending=True)
cumsum = cumsum(sorted_probs)
cutoff = find_first(cumsum >= P)
probs[after(cutoff)] = 0
probs = probs / sum(probs)

# 采样
if do_sample:
    next_token = random_choice(probs)  # 按概率分布随机选
else:
    next_token = argmax(probs)         # 贪心：取概率最大的
```

### 2.8 Phase 8: Decode (Token → Text)

```
BPE Decode: token_id → token_string → ByteDecode → UTF-8

ncnn_llm 的实现:
  callback(bpe->decode({ctx->cur_token}, false));

BpeTokenizer::decode() 内部:
  1. ID → token 字符串 (id_to_token_ 查表)
  2. 跳过特殊 token (可选)
  3. 如果 use_byte_encoder_ → ByteDecode (还原 UTF-8 字节)
  4. SentencePiece 模式 → ▁ → 空格转换
```

---

## 3. 自回归生成循环的完整代码

这是将 Phase 4-8 串联起来的核心循环：

```cpp
// 简化自 ncnn_llm_gpt.cpp generate()
for (int step = 0; step < cfg.max_new_tokens; ++step) {
    // ---- 停止条件 ----
    if (ctx->cur_token == eos) break;

    // ---- Step 1: 当前 token → embedding ----
    ncnn::Mat cur_token_mat = ncnn::Mat(1, (void*)&ctx->cur_token).clone();
    ncnn::Mat cur_embed;
    embed_net->create_extractor()->input("in0", cur_token_mat);
    embed_net->create_extractor()->extract("out0", cur_embed);

    // ---- Step 2: 生成当前 token 的 RoPE ----
    generate_rope_embed_cache(1, rope_head_dim, ctx->position_id,
                              cos_cache, sin_cache, rope_theta);
    ctx->position_id++;  // 位置递增！

    // ---- Step 3: 构建 mask (全 0，因为只看过去) ----
    ncnn::Mat mask(ctx->kv_cache[0].first.h + 1, 1);  // cache长度+1
    mask.fill(0.f);

    // ---- Step 4: Decoder 前向 (复用 KV Cache) ----
    ncnn::Extractor ex = decoder_net->create_extractor();
    ex.input("in0", cur_embed);
    ex.input("in1", mask);
    ex.input("in2", cos_cache);
    ex.input("in3", sin_cache);
    // 输入旧 KV Cache (28 层)
    for (int i = 0; i < attn_cnt; ++i) {
        ex.input("cache_k" + to_string(i), ctx->kv_cache[i].first);
        ex.input("cache_v" + to_string(i), ctx->kv_cache[i].second);
    }
    // 提取更新后的 KV Cache
    for (int i = 0; i < attn_cnt; ++i) {
        ex.extract("out_cache_k" + to_string(i), ctx->kv_cache[i].first);
        ex.extract("out_cache_v" + to_string(i), ctx->kv_cache[i].second);
    }
    ex.extract("out0", decode_out);

    // ---- Step 5: Projection ----
    ncnn::Mat logits_mat;
    proj_out_net->create_extractor()->input("in0", decode_out);
    proj_out_net->create_extractor()->extract("out0", logits_mat);

    // ---- Step 6: Repetition Penalty ----
    for (int t : history) {
        if (logits[t] < 0) logits[t] *= cfg.repetition_penalty;
        else               logits[t] /= cfg.repetition_penalty;
    }

    // ---- Step 7: Sampling ----
    softmax_vec(logits, cfg.temperature);
    if (cfg.top_k > 0) apply_top_k(logits, cfg.top_k);
    if (cfg.top_p < 1.0f) apply_top_p(logits, cfg.top_p);
    int next_id = cfg.do_sample ? sample_from_probs(logits)
                                : argmax(logits);

    ctx->cur_token = next_id;
    history.insert(next_id);
}
```

---

## 4. Prefill 的"特殊"处理：最后一个 token 单独跑

```cpp
// 为什么 prefill 后要单独处理最后一个 token？

Prefill 阶段:
  输入: 全部 prompt token [N, 1024]
  输出: KV Cache 包含了 N 个 token 的 K/V
  但 hidden state 也是 [N, 1024]
  我们只需要最后一个位置 → hidden_state[-1, :]

第一个预测 token 的生成:
  // 取最后一个 token
  int last_token_id = token_ids.back();

  // 单独 embedding
  ncnn::Mat last_embed;
  embed_net->extract(last_token_id, last_embed);

  // 单独 decoder forward (使用 prefill 产生的 KV Cache)
  // mask 是 [N+1, 1]，全 0（允许看到全部历史）
  decoder_net->forward(last_embed, mask, ..., kv_cache_from_prefill);

  // 得到第一个生成 token 的 hidden_state → sample → 进入 decode loop

为什么这样设计？
  - Prefill 的 decoder 输出是 [N, 1024]，其中 N-1 个位置的输出没用
  - 只需要最后一个位置的 hidden_state 做 projection
  - 单独跑一次 decode (1 token) 拿第一个预测
  - 这样代码统一：后续 decode loop 全部是 1-token forward
```

---

## 5. 工程关注点分布

```
每个 Phase 的工程优化点:

Phase 0 (Config):    模型文件大小、加载速度、参数验证
Phase 1 (Template):  多轮对话拼接、System prompt 复用
Phase 2 (Tokenizer): Token 数预估、BPE 缓存、Special token 匹配
Phase 3 (RoPE):      Cache 预计算、长上下文外推、多模态 mRoPE
Phase 4 (Embed):     Weight tying 省内存、CPU/GPU 放置
Phase 5 (Decoder):   **最重** → KV Cache 管理、Attention kernel、量化
Phase 6 (Projection): Vocab 大小影响计算量、Weight tying
Phase 7 (Sampling):  CPU/GPU 同步延迟、低延迟要求
Phase 8 (Decode):    Streaming 输出、特殊 token 跳过
```

---

## 6. 本章学习清单

- [ ] 能画出从用户输入到第一个 token 输出的完整数据流图
- [ ] 能标注每个 Phase 的输入输出 shape（用 Qwen3-0.6B 的实际数字）
- [ ] 能写出 decode loop 的 7 个步骤
- [ ] 能解释为什么 prefill 后要单独处理最后一个 token
- [ ] 能说明 3 个 Net 拆分的设计原理（KV Cache 提取、权重共享、内存控制）
- [ ] 能写出 Sampling 的完整流程（Repetition Penalty → Temperature → Top-K → Top-P → Sample）

---

*上一篇: [02_compute_operator_layer.md](02_compute_operator_layer.md) — 计算与算子层*
*下一篇: [04_memory_kv_cache.md](04_memory_kv_cache.md) — KV Cache 与 PagedAttention*
