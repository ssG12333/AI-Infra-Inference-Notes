# Module 5: LLM 推理流水线——从文本到 Token

> LLM 推理不是一次 forward——它是一个以 KV Cache 为状态、自回归循环为骨架的持续运行系统。

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

## 1. 总体架构

ncnn_llm 把大模型拆成 **3 个独立的 ncnn::Net**：

```
┌──────────────┐     ┌───────────────┐     ┌──────────────┐
│  embed_net   │     │  decoder_net  │     │ proj_out_net │
│              │     │               │     │              │
│ Token → Emb  │────→│ Emb → Hidden  │────→│ Hidden→Logits│
│ (Gather)     │     │ (28层 Transf) │     │ (Linear)     │
└──────────────┘     └───────┬───────┘     └──────────────┘
                             │
                      KV Cache 提取/注入
```

**为什么拆成 3 个？**
1. KV Cache 需要从 decoder 中间层提取 K/V
2. Embedding 和 Projection 可共享权重 (省 ~600MB)
3. 内存峰值更低，可按需加载

---

## 2. Phase 完整数据流

### 数据形状全链路

```
Phase 0: Config     model.json → 3个Net + params
Phase 1: Template   raw text → formatted text
Phase 2: Tokenize   text → token_ids [N]
Phase 3: RoPE       位置 → cos[N,hd/2] + sin[N,hd/2]
Phase 4: Embed      ids[N] → embed [N, 1024]
Phase 5: Decoder    embed[N,1024] → hidden[-1,1024] + KV Cache
Phase 6: Project    hidden[1,1024] → logits [151936]
Phase 7: Sample     logits → next_token_id
Phase 8: Decode     token_id → text (streaming)

Decode Loop: Phase 4→5→6→7 重复直到 EOS/max_tokens
```

### Phase 1: Prompt Template 📖

```
用户输入 "你好"
  ↓ ChatML 模板
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
你好<|im_end|>
<|im_start|>assistant
                            ← add_generation_prompt 自动添加
```

### Phase 2: BPE Tokenization 🔬

```
文本 → Byte Encode → Pretokenize → BPE Merge → Vocab 查找 → Token IDs

BPE 贪心合并算法:
  symbols = list(UTF-8 chars of piece)
  while True:
    找 rank 最小的相邻符号对 (symbols[i], symbols[i+1])
    如果不存在 → break
    合并: symbols[i] += symbols[i+1]
    删除 symbols[i+1]
  → 最终 symbols 就是 token 列表

Rank 系统:
  merges.txt 的行号 = rank
  行号越小 → 训练时越早被合并 → 推理时优先级越高
```

### Phase 3: RoPE Cache 💡

```cpp
// 预计算 cos/sin cache (所有位置共享同一份 base 频率)
inv_freq[k] = 1 / (theta ^ (2k / dim))

for pos in range(seq_len):
    cos[pos][k] = cos(pos * inv_freq[k])
    sin[pos][k] = sin(pos * inv_freq[k])

// 精妙之处: Q_i · K_j 只依赖 (j - i)
// 低维度旋转快，高维度旋转慢 → 天然的多尺度位置编码
```

### Phase 4: Embedding

```
Embedding 本质: Gather 操作
  E = embedding_matrix[vocab_size, hidden_dim]
  embed[i] = E[token_ids[i], :]

Weight Tying (权重共享):
  Embedding: 从 E 取一行
  Projection: 用 E^T 做矩阵乘法
  → 共享同一份权重，省 vocab_size × hidden_dim × 4 bytes
  Qwen3-0.6B: 151936 × 1024 × 4 ≈ 592 MB 节省
```

### Phase 5: Decoder 前向 🔬

```
Prefill (首次，输入 N 个 token):
  Q: [16, N, 128], K: [8, N, 128], V: [8, N, 128]
  Q·K^T: [16, N, N]  ← 计算密集!
  KV Cache: 写入 N 个 token

Decode (后续，输入 1 个 token):
  Q: [16, 1, 128], K_cache: [8, S, 128], V_cache: [8, S, 128]
  Q·K^T: [16, 1, S]  ← 内存密集! 需读全部 KV Cache
  KV Cache: 追加 1 个 token → S = S + 1

关键差异:
  Prefill: 计算密集 (大矩阵乘), GPU 利用率高
  Decode:  内存密集 (读全部 KV Cache), 瓶颈在带宽
```

### Phase 5b: SDPA 层内部 🔬

```cpp
// ncnn sdpa.cpp — KV Cache concat 核心
int past_seqlen = kv_cache ? past_key.h : 0;
int dst_seqlen = past_seqlen + cur_seqlen;

// 拼接 past + current K
Mat key(dst_seqlen);  // 分配新内存
for (int q = 0; q < num_group; q++) {
    memcpy(key.row(0),           past_key_head, past_seqlen * sizeof(float));
    memcpy(key.row(past_seqlen), cur_key_head,  cur_seqlen  * sizeof(float));
}
// 然后用拼接后的 key/value 做完整 attention
```

### Phase 6-7: Projection → Sampling

```
Projection: hidden [1, 1024] × W^T [1024, 151936] → logits [151936]

Sampling 流程:
  logits
    → Repetition Penalty (已出现 token 的 logit 被惩罚)
    → Temperature Scaling (logits / T)
    → Softmax → probs
    → Top-K (保留 K 个最大的)
    → Top-P (累积概率达 P 时截断)
    → Sample (按概率随机选) or Argmax (贪心)
    → next_token_id
```

### Phase 8: Decode → Streaming

```
BPE Decode: token_id → token_string → ByteDecode → UTF-8 → 输出

ncnn_llm 通过 callback 实现流式输出:
  callback(bpe->decode({ctx->cur_token}, false));
```

---

## 3. 完整自回归生成循环 (完整代码)

```cpp
// 简化自 ncnn_llm_gpt.cpp generate()
for (int step = 0; step < cfg.max_new_tokens; ++step) {

    // 停止判断
    if (ctx->cur_token == eos) break;

    // === Step 1: Embedding ===
    ncnn::Mat cur_embed;
    embed_net->extract("in0", token_mat, "out0", cur_embed);

    // === Step 2: RoPE Cache ===
    generate_rope_embed_cache(1, head_dim, ctx->position_id++, cos, sin, theta);

    // === Step 3: Mask (全 0, 因为 cache 已有历史) ===
    ncnn::Mat mask(ctx->kv_cache[0].first.h + 1, 1); mask.fill(0.f);

    // === Step 4: Decoder (复用 KV Cache) ===
    ncnn::Extractor ex = decoder_net->create_extractor();
    ex.input("in0", cur_embed); ex.input("in1", mask);
    ex.input("in2", cos); ex.input("in3", sin);
    for (int i = 0; i < attn_cnt; i++) {          // 输入旧 KV Cache
        ex.input("cache_k" + str(i), kv_cache[i].first);
        ex.input("cache_v" + str(i), kv_cache[i].second);
    }
    for (int i = 0; i < attn_cnt; i++) {          // 提取更新后 KV Cache
        ex.extract("out_cache_k" + str(i), kv_cache[i].first);
        ex.extract("out_cache_v" + str(i), kv_cache[i].second);
    }
    ex.extract("out0", decode_out);

    // === Step 5: Projection ===
    ncnn::Mat logits;
    proj_out_net->extract("in0", decode_out, "out0", logits);

    // === Step 6: Repetition Penalty ===
    for (int t : history) { /* penalty logic */ }

    // === Step 7: Sampling ===
    softmax(logits, cfg.temperature);
    apply_top_k(logits, cfg.top_k); apply_top_p(logits, cfg.top_p);
    int next = cfg.do_sample ? sample(logits) : argmax(logits);

    ctx->cur_token = next; history.insert(next);
}
```

---

## 4. Prefill vs Decode — 两大阶段的性能特征

| 维度 | Prefill | Decode |
|------|---------|--------|
| **输入** | Prompt 全部 N 个 token | 1 个新 token |
| **Q 形状** | [16, N, 128] | [16, 1, 128] |
| **Attention Shape** | [16, N, N] | [16, 1, S] |
| **计算模式** | 大矩阵乘法，GPU 友好 | 小向量 × 大矩阵，内存瓶颈 |
| **KV Cache** | 一次性写入 N 个 K/V | 每次追加 1 个 K/V |
| **瓶颈** | Compute-bound | **Memory-bound** |
| **GPU 利用率** | >80% | <30% |
| **影响延迟** | TTFT (首 token) | TPOT (后续 token) |

---

## 5. 多轮对话与 KV Cache 复用

```
Round 1:
  prefill("system+user1") → ctx.kv_cache 有了 100 个 token
  generate(ctx) → 生成 assistant1 50 个 token → ctx.kv_cache 有 150 个

Round 2:
  ctx2 = ctx.clone()  ← 共享 KV Cache (浅拷贝 Mat 的 data 指针)
  prefill("user2", ctx2) → 只计算 user2 的 KV，追加到已有 cache
  → 不需要重新计算 system+user1+assistant1 的 KV!

关键: clone() 浅拷贝 Mat → 内存共享 → 多轮对话极低 overhead
```

---

## 🛠️ 动手练习

1. **Trace 一次完整推理**: 对 "你好" 做 tokenize，手动跟踪每个 Phase 的输入输出 shape。

2. **BPE 手算**: 将字符串 "hello" 的字母逐个 split，若 merges 中有 (h,e,rank=0)、(he,l,rank=1)、(hel,lo,rank=2)，求最终 token 列表。

3. **RoPE 验证**: 对 2D 向量 v = [1, 0]，在位置 pos=1, theta=10000 下计算 RoPE 旋转后的结果。

4. **Phase 时间分析**: 在 ncnn_llm 中打点，测量 Phase 4/5/6/7 的耗时分布。

---

## 📚 延伸阅读

- [ncnn_llm_original_README.md](../ncnn_llm_original_README.md) — 逐行源码分析 (2683 行)
- [03_llm_inference_pipeline.md](../../docs/03_llm_inference_pipeline.md) — 系统笔记版

---

*下一模块: [Module 6: KV Cache 系统](../module-06-kv-cache-system/README.md)*
