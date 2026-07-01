# Module 7: 调度系统——让 GPU 每一毫秒都在干活

> 如果 KV Cache 是 LLM serving 的"记忆系统"，Scheduler 就是它的"操作系统大脑"。它时刻在做决策：谁先进来？谁先出去？谁和谁凑一拨？——每一个决策都直接影响着延迟、吞吐和用户体感。

---

## 📋 学习目标

- [ ] 能画出 Request 的完整生命周期状态图
- [ ] 能用自己的话讲清楚 Static / Dynamic / Continuous Batching 的区别
- [ ] 能手写 Continuous Batching 的 5 个 Phase 伪代码
- [ ] 能解释 Token Budget 为什么是调度的"货币"
- [ ] 能从三个层次解释"为什么 decode 阶段 GPU 利用率低"
- [ ] 能定义 TTFT、TPOT、Throughput 并说出它们之间的制约关系

---

## 1. 问题的本质——从"一个请求"到"一群请求"

### 1.1 为什么单请求推理可以很简单，但多请求服务就很难？

```
单请求:
  model.generate("你好") → 等 2 秒 → 返回结果
  简单、直接、没有竞争

多请求服务:
  100 个用户同时发请求
  → 用户 A 的 prompt 只有 5 个 token，用户 B 的有 5000 个
  → 用户 A 期望 0.5 秒出第一个字，用户 B 可以接受 2 秒
  → 有的请求在生成中，有的刚进来，有的已经结束了
  → 显存里塞满了各种长度的 KV Cache
  → 如何组织？如何排队？如何最大化 GPU 利用率？
```

**调度系统要回答的核心问题**：每一轮 decode，到底让哪些请求一起跑？

### 1.2 一个生活类比——餐厅后厨

```
Static Batching  = 等 4 桌客人全点完菜，一起做，一起上
  → 第 1 桌只点了一碗面（3 分钟），但要等第 4 桌的佛跳墙（2 小时）

Dynamic Batching = 在 5 分钟窗口内收集订单，凑一批做一批
  → 比 Static 灵活，但"一批做完才能做下一批"的限制还在

Continuous Batching = 每完成一道菜就端走，同时新来的订单可以随时插进来
  → 后厨的每个灶台每分每秒都在做有效产出
  → 面先好先上，佛跳墙慢慢炖
```

---

## 2. Request Lifecycle——一个请求的"一生"

```
                    ┌──────────┐
      请求到达  →   │ WAITING  │   在队列里等着
                    └────┬─────┘
                         │ Scheduler 说"轮到你了"
                         ▼
                    ┌──────────┐
                    │ PREFILL  │   处理你的 prompt
                    │          │   分配 KV Cache blocks
                    └────┬─────┘
                         │ Prompt 处理完，拿到第一个 token
                         ▼
                    ┌──────────┐
                    │ DECODING │   一个字一个字往外蹦
                    │          │   每蹦一个字跑一次 forward
                    └────┬─────┘
                         │
              ┌──────────┼──────────┐
              ▼          ▼          ▼
         ┌────────┐ ┌────────┐ ┌────────┐
         │FINISHED│ │ABORTED │ │SWAPPED │
         │ 正常结束│ │用户取消│ │ 被换出 │
         └────────┘ └────────┘ └────────┘
```

每个状态对应不同的资源占用和调度优先级：

| 状态 | KV Cache | 计算 | 调度优先级 |
|------|----------|------|:--------:|
| WAITING | 无 | 无 | 等待中 |
| PREFILL | 正在写入 | 大矩阵乘 | **最高** (影响首 token 延迟) |
| DECODING | 每步追加 | 小向量乘 | 中 |
| SWAPPED | 在 CPU 侧 | 无 | 低 (等待换回) |
| FINISHED | 待释放 | 无 | — |

---

## 3. Batching 三代进化——从"等齐再走"到"即来即走"

### 3.1 Static Batching：等齐出发的旅游团

```
做法: 等 4 个请求凑齐 → pad 到相同长度 → 一起跑 → 等下一批

问题惨不忍睹:
  - 短请求要等长请求 ("最慢的人决定全团速度")
  - padding 浪费大量计算 (把所有人的输入 pad 到最长那个)
  - 新请求无法半路加入
  - decode 阶段 GPU 利用率 < 10% (每个请求每步只算 1 个 token)
```

### 3.2 Dynamic Batching：有 5 分钟就不等了的公交

```
做法: 在时间窗口内收集请求，够一车就发车

比 Static 好，但对 decode 阶段仍然不够——
decode 不是"一次算完"的，是"持续循环"的。
```

### 3.3 Continuous Batching：永远在运转的流水线 ⭐

```
做法: 每个 decode step 都重新组织 batch。
      完成的请求立刻移出，新来的请求立刻加入。

时间线:
  Step 1: [A(prefill), B(dec), C(dec)]
  Step 2: [A(dec), B(dec), C(dec)]
  Step 3: [A(dec), B(dec), D(prefill)]  ← C 完成移出！D 加入！
  Step 4: [B(dec), D(dec), E(prefill)]  ← A 完成移出！E 加入！
  Step 5: [B(dec), D(dec), E(dec), F(prefill)]
  ...

关键效果:
  → 没有"等 batch 凑齐"的空白期
  → 长请求不阻塞短请求
  → GPU 始终在处理有效 token
  → 吞吐比 Static Batching 提升 2-4x
```

---

## 4. Scheduler 内部——5 个 Phase 的精密编排

```python
class LLMScheduler:
    def step(self):
        # =========================================
        # Phase 1: 打扫卫生——回收已完成的资源
        # =========================================
        for req in self.running:
            if req.output_tokens[-1] == EOS or len(req.output_tokens) >= req.max_tokens:
                req.status = FINISHED
                self.cache_manager.release(req.block_table)  # 归还 KV blocks

        self.running = [r for r in self.running if not r.is_finished]

        # =========================================
        # Phase 2: 迎接新人——从 waiting 队列选请求做 prefill
        # =========================================
        scheduled = []
        budget = self.max_num_batched_tokens - len(self.running)
        # 为什么减去 len(running)？
        # → 每个 decode 请求在下一步各占 1 个 token budget

        while self.waiting and budget > 0:
            req = self.waiting[0]
            prompt_len = len(req.prompt_tokens)

            # 检查 1: token budget 够吗？
            if prompt_len > budget:
                if prompt_len <= self.max_num_batched_tokens:
                    # Chunked Prefill: 切一块先处理
                    # 比如 8000 token 的 prompt，budget 只有 3000
                    # → 先 prefill 前 3000 个，下轮再 3000，再 2000
                    chunk_req = self._chunk_prefill(req, budget)
                    scheduled.append(chunk_req)
                break  # 长 prompt 占满了本轮 budget

            # 检查 2: 序列数超限了吗？
            if len(self.running) + len(scheduled) >= self.max_num_seqs:
                break

            # 检查 3: 还有空闲 KV blocks 吗？
            blocks_needed = ceil(prompt_len / BLOCK_SIZE)
            if not self.cache_manager.can_allocate(blocks_needed):
                if not self._try_preempt(blocks_needed):
                    break  # 实在没地方了

            # 一切就绪——分配资源，开始 prefill
            self.waiting.popleft()
            self.cache_manager.allocate(req)
            req.status = PREFILL
            scheduled.append(req)
            budget -= prompt_len

        # =========================================
        # Phase 3: 组队——确定本轮 batch 成员
        # =========================================
        batch = self.running + scheduled  # decode + 新 prefill

        # =========================================
        # Phase 4: 执行——模型前向
        # =========================================
        if batch:
            logits = self.model_runner.execute(batch)
            # 这里内部调用 PagedAttention kernel
            # 每个请求有自己的 block_table

        # =========================================
        # Phase 5: 收尾——采样 & 状态更新
        # =========================================
        for req in batch:
            next_token = self._sample(req, req.pending_logits)
            req.output_tokens.append(next_token)

            if req.status == PREFILL:
                req.first_token_time = now()  # 记录 TTFT！
                req.status = DECODING
                self.running.append(req)

            # 新 token 可能需要新 KV block
            if req.num_tokens % BLOCK_SIZE == 0:
                new_block = self.cache_manager.allocate_block(req)
                req.block_table.append(new_block)

            if next_token == EOS:
                req.status = FINISHED
```

---

## 5. Token Budget——调度的"货币"

为什么按 token 数调度，而不是请求数？

```
max_num_batched_tokens = 8192

Decode 请求: 每步 1 个 token
Prefill 请求: 每步 prompt_length 个 token

→ 1 个 8000-token 的 prefill ≈ 8000 个 decode 请求！

如果不按 token 数控制:
  一个 8000-token 的 prompt prefill 可以把 GPU 独占几百毫秒
  → 所有 decode 请求在此期间全部卡住
  → 用户体验: "回复突然停了" 

Token Budget 的本质:
  把 GPU 时间按 token 做细粒度分片
  → 长 prompt 不会饿死短请求
  → 这是"公平性"在 LLM 推理系统中的具体实现
```

---

## 6. 为什么 Decode 阶段 GPU 利用率低？——三层深度解析

这是面试高频题。你要能说出三个层次：

**层次 1：算术强度低**

```
Prefill: Q[16,N,128] × K^T[16,128,N] → 一个元素要乘 N 次 → 计算密集
Decode:  Q[16,1,128] × K_cache[16,128,S] → S 次乘法但 S 次都是从显存读的

计算强度 (FLOPs/Byte):
  Prefill: ~150 → 接近 roofline 拐点 → compute-bound ✓
  Decode:  ~0.02 → 远低于拐点 → memory-bound ✗

这不是 GPU 的错——它有能力算得更快，但数据供不上。
```

**层次 2：Batch 天然小**

```
1 个 token 的 GEMM: [1, 1024] × [1024, 2048]
  → 连 GPU 的一个 warp (32 线程) 都填不满
  → Tensor Core 需要较大的矩阵维度才能发挥威力

Continuous Batching 的补救:
  把 N 个 decode 请求的 1-token GEMM 拼成 [N, 1024] × [1024, 2048]
  → N 越大，越接近 GPU 的设计目标
```

**层次 3：KV Cache 读取量随序列暴涨**

```
每层 Attention 都要读全部历史 KV:
  seq=1024:  读 ~4MB / 层
  seq=8192:  读 ~32MB / 层
  seq=32768: 读 ~128MB / 层
  
但每个 decode step 的计算量几乎不变（只多乘几次）
→ 序列越长，"读显存/算力" 的比值越离谱
```

---

## 7. 性能指标——TTFT、TPOT、Throughput

```
用户感知:  "第一句话出来快不快？"  → TTFT
          "后面出来的快不快？"      → TPOT
          "整体能服务多少人？"      → Throughput

TTFT  = queue_wait + prefill_compute + first_sample
        ↑ 等前面的人    ↑ 处理 prompt   ↑ 选第一个 token

TPOT  = avg(每个 decode step 的时间)
        ↑ 随 KV Cache 增长而缓慢变大 (因为要读更多历史)

Throughput = 系统整体 tokens/s
              ↑ 不等于 1/TPOT × 并发数 (因为有排队和资源竞争)

延迟 vs 吞吐的矛盾:
  要低延迟 → 来即处理 → batch 小 → GPU 利用率低 → 吞吐低
  要高吞吐 → 多凑点一起跑 → 等待 → 个别延迟高

Continuous Batching 的优雅:
  decode 阶段自然形成 batch → 不需要"凑"
  → 在保证低延迟的前提下，获得了高吞吐
  → 这是它被广泛采用的根本原因
```

---

## 🛠️ 动手练习

1. **写一个最小 Scheduler**：实现 step() 的 5 个 Phase，测试 10 个不同长度的请求。

2. **模拟 Continuous Batching**：创建 A(短)/B(中)/C(长) 三个请求，手动 trace 每个 step 的 batch 成员变化。

3. **Token Budget 计算**：max_num_batched_tokens=8192, running=200 个 decode, waiting 队列有 [500, 2000, 6000, 300] token 的 prompt。本轮能 prefill 几个？

---

*下一模块: [Module 8: 优化技术](../module-08-optimization/README.md)*
