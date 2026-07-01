# Module 7: 调度系统——让 GPU 不再空转

> 如果说 KV Cache 是 LLM serving 的"内存系统"，Scheduler 就是"操作系统"——它决定谁先用、谁后用、怎么一起用。

---

## 📋 学习目标

- [ ] 能画出 Request 的完整生命周期状态图
- [ ] 能区分 Static / Dynamic / Continuous Batching
- [ ] 能手写 Continuous Batching 的 5 个 Phase 伪代码
- [ ] 能解释 Token Budget 为什么是调度核心
- [ ] 能解释为什么 decode 阶段 GPU 利用率低 (三个层次)
- [ ] 能定义 TTFT、TPOT、Throughput

---

## 1. Request Lifecycle

```
                 ┌──────────┐
   请求到达  →   │ WAITING  │   在队列中等待
                 └────┬─────┘
                      │ Scheduler 决定调度
                      ▼
                 ┌──────────┐
                 │ PREFILL  │   处理 prompt, 分配 KV Cache
                 └────┬─────┘
                      │ Prefill 完成, 拿到第一个 token
                      ▼
                 ┌──────────┐
                 │ DECODING │   逐 token 生成
                 └────┬─────┘
                      │
           ┌──────────┼──────────┐
           ▼          ▼          ▼
      ┌────────┐ ┌────────┐ ┌────────┐
      │FINISHED│ │ABORTED │ │SWAPPED │
      └────────┘ └────────┘ └────────┘
```

---

## 2. Batching 三代进化

### Static Batching
```
等一批凑齐 → pad 到相同长度 → 一起跑完 → 等下一批
问题: 短等长 + padding 浪费 + 新请求无法中途加入
```

### Dynamic Batching
```
时间窗口内动态组 batch
比 Static 灵活, 但 decode 阶段仍不解决根本问题
```

### Continuous Batching ⭐
```
每个 decode step 都重新组 batch:
  Step 1: [A(prefill), B(dec), C(dec)]
  Step 2: [A(dec), B(dec), C(dec)]
  Step 3: [A(dec), B(dec), D(prefill)]  ← C 结束移出, D 加入!
  Step 4: [B(dec), D(dec), E(prefill)]  ← A 结束移出, E 加入!

关键: 完成即移出, 新来即加入
→ GPU 始终在处理有效 token
→ 吞吐提升 2-4x
```

---

## 3. Scheduler 核心代码

```python
class LLMScheduler:
    def step(self):
        # Phase 1: 清理已完成的请求
        for req in self.running:
            if req.is_finished:
                self.cache_manager.release(req.req_id)
        self.running = [r for r in self.running if not r.is_finished]

        # Phase 2: 从 waiting 队列取新请求做 prefill
        #   限制: token_budget, max_seqs, free_blocks
        scheduled = []
        budget = self.max_num_batched_tokens - len(self.running)  # decode 预留
        while self.waiting and budget > 0:
            req = self.waiting[0]
            if len(req.prompt_tokens) > budget:
                # Chunked Prefill: 切一部分先处理
                if req.prompt_tokens <= self.max_num_batched_tokens:
                    scheduled.append(self._chunk_prefill(req, budget))
                break
            if not self.cache_manager.can_allocate(req): break
            self.waiting.popleft()
            self.cache_manager.schedule_prefill(req)
            scheduled.append(req)
            budget -= len(req.prompt_tokens)

        # Phase 3: 组 batch → running + scheduled
        batch = self.running + scheduled

        # Phase 4: 模型前向
        if batch: self._execute_batch(batch)

        # Phase 5: 采样 & 更新状态
        for req in batch:
            next_token = self._sample(req, req.pending_logits)
            req.output_tokens.append(next_token)
            if req.status == PREFILL:
                req.status = DECODING
                self.running.append(req)
            if next_token == EOS or len(req.output_tokens) >= req.max_tokens:
                req.status = FINISHED
```

---

## 4. Token Budget：调度的"货币"

```
max_num_batched_tokens = 8192

Decode 请求: 每个占 1 token
Prefill 请求: 每个占 prompt_length tokens

→ 1 个 8000-token prompt ≈ 8000 个 decode 请求!

预算分配策略:
  30% 留给 prefill (保证新请求首 token 延迟)
  100% 用于 decode (尽可能提高吞吐)
  → 动态平衡
```

---

## 5. 为什么 Decode 阶段 GPU 利用率低？

```
三个层次的原因:

层次 1: 算术强度低
  Decode: Q[1,d] × K[S,d]^T → O(d) compute, O(S×d) memory
  → 计算量/访存量 ≈ 0.02 FLOPs/Byte << roofline 拐点
  → 绝对 memory-bound

层次 2: Batch 太小
  1 token 的矩阵乘法 → 连一个 warp 都填不满
  Continuous Batching → N 个请求的 1 token 拼成 [N,d] 矩阵
  → 增大有效 batch size

层次 3: KV Cache 读取量随序列增长
  每层都要读全部历史 KV → 长序列带宽需求暴增
  → 必须有 KV Cache 量化 / FlashAttention
```

---

## 6. 性能指标

| 指标 | 含义 | 目标 (服务端) |
|------|------|:----------:|
| **TTFT** | Time To First Token | <50ms |
| **TPOT** | Time Per Output Token | <10ms |
| **Throughput** | tokens/s (系统) | 2000-5000 |

```
TTFT = queue_wait + prefill_compute + first_sample
TPOT = avg(decode_step_time)

延迟 vs 吞吐的矛盾:
  低延迟 → 来即处理 → batch 小 → 吞吐低
  高吞吐 → 多凑 batch → 等待 → 延迟高

Continuous Batching 的优雅: decode 阶段自然形成 batch → 兼得!
```

---

## 7. ncnn_llm vs vLLM Scheduler

```
ncnn_llm:  无调度器 (generate() 同步循环, 单请求)
vLLM:      完整调度器 (waiting/running/swapped 状态机)

从 ncnn_llm 到 vLLM 需要加的:
  1. Request 数据结构 (状态 + block_table + metrics)
  2. Waiting Queue + Running Queue
  3. Token Budget 控制
  4. Chunked Prefill
  5. Preemption 机制
```

---

## 🛠️ 动手练习

1. **写一个最小 Scheduler**: 用 Python 实现 Scheduler.step() 的 5 个 Phase。

2. **模拟 Continuous Batching**: 创建 10 个不同长度的请求，手动 trace 每个 step 的 batch 成员变化。

3. **计算 Token Budget**: 已知 max_num_batched_tokens=8192，running=200 个 decode 请求，waiting 有 [500, 2000, 6000, 300] tokens 的 prompt。求本轮能 prefill 几个？

---

*下一模块: [Module 8: 优化技术](../module-08-optimization/README.md)*
