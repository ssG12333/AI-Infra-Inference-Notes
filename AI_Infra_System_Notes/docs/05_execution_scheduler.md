# 05｜Execution System：Runtime、Scheduler 与 Continuous Batching

> 如果说 KV Cache（04 章）是 LLM serving 的"内存系统"，本章就是"操作系统"——它决定了谁先用、谁后用、怎么一起用。

---

## 1. 问题的本质：为什么需要调度？

### 1.1 从单请求到多请求的范式转换

单请求推理简单到只有一行：

```python
output = model.generate(prompt)
```

但真实服务面对的是**并发的、异步的、长度不一的、随时到达和离开的**请求流：

```
客户端 A: "写一首诗"           → 需要 ~200 tokens，3 秒
客户端 B: "翻译这段 3000 字文章"  → 需要 ~4000 tokens，60 秒
客户端 C: "1+1=?"              → 需要 ~5 tokens，0.1 秒
客户端 D: (刚到达) "总结一下"     → prompt 长，还没开始
客户端 E: (刚完成)               → 释放资源
```

### 1.2 核心矛盾：长请求"霸占"了 GPU

```
最笨的方法（串行）：
  GPU 做完 A → 做 B → 做 C → 做 D
  C 只需 0.1 秒但要等 A 和 B 的 63 秒

稍微好点（静态 batch）：
  把 A+B 凑一起 → 等到两个都完成 → 再凑 C+D
  C 还是要等 A 和 B 都完成（因为 A 很长）

更好的方法（Continuous Batching）：
  [A, B, C, D] 一起跑 → C 先结束就移出，E 新来就加入
  每步都在动态重组！
```

整个调度系统的本质问题就是：**每一轮 decode 到底让哪些请求一起跑？**

---

## 2. Request Lifecycle：一个请求的一生

### 2.1 状态机

```
                  ┌──────────┐
    请求到达  →   │ WAITING  │   ← 在队列中等待
                  └────┬─────┘
                       │ Scheduler 决定让它进入
                       ▼
                  ┌──────────┐
                  │ PREFILL  │   ← 处理 prompt，分配 KV Cache
                  └────┬─────┘
                       │ Prefill 完成，拿到第一个 token
                       ▼
                  ┌──────────┐
                  │ DECODING │   ← 逐 token 生成
                  └────┬─────┘
                       │
            ┌──────────┼──────────┐
            ▼          ▼          ▼
       ┌────────┐ ┌────────┐ ┌────────┐
       │FINISHED│ │ABORTED │ │TIMEOUT │
       └────────┘ └────────┘ └────────┘
        正常结束    用户取消    超时未完成
```

### 2.2 状态详解

| 状态 | 含义 | KV Cache 状态 | 计算特征 |
|------|------|--------------|---------|
| **WAITING** | 请求入队，还没分配资源 | 无 | 无计算 |
| **PREFILL** | 正在处理 prompt token | 正在写入 | 大矩阵乘法，计算密集 |
| **DECODING** | 正在逐 token 自回归生成 | 每步追加新 token 的 K/V | 小矩阵 × N 次，内存密集 |
| **SWAPPED** | KV Cache 被换出到 CPU 内存 | 在 CPU 侧 | 无计算（等待换回） |
| **FINISHED** | 生成结束（EOS 或 max_tokens） | 待释放 | 无 |
| **ABORTED** | 用户取消 | 待释放 | 无 |

### 2.3 Request 数据结构（代码视角）

```python
# ============================================
# 一个请求的完整运行时状态
# ============================================
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

class ReqStatus(Enum):
    WAITING   = "waiting"
    PREFILL   = "prefill"
    DECODING  = "decoding"
    SWAPPED   = "swapped"
    FINISHED  = "finished"

@dataclass
class Request:
    """一个推理请求的完整状态"""
    req_id: str
    prompt_tokens: List[int]
    max_tokens: int = 256

    # --- 生成状态 ---
    output_tokens: List[int] = field(default_factory=list)
    status: ReqStatus = ReqStatus.WAITING

    # --- KV Cache ---
    block_table: List[int] = field(default_factory=list)  # 逻辑→物理映射

    # --- 性能指标 ---
    arrival_time: float = 0.0
    first_token_time: Optional[float] = None
    finish_time: Optional[float] = None

    # --- 采样参数 ---
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1

    @property
    def num_tokens(self) -> int:
        """当前序列总长度"""
        return len(self.prompt_tokens) + len(self.output_tokens)

    @property
    def is_finished(self) -> bool:
        return self.status == ReqStatus.FINISHED

    @property
    def ttft(self) -> Optional[float]:
        """Time To First Token (秒)"""
        if self.first_token_time and self.arrival_time:
            return self.first_token_time - self.arrival_time
        return None

    @property
    def tpot(self) -> Optional[float]:
        """Time Per Output Token (秒)"""
        if (self.finish_time and self.first_token_time
            and len(self.output_tokens) > 1):
            return (self.finish_time - self.first_token_time) / (len(self.output_tokens) - 1)
        return None
```

---

## 3. Batching 的三代进化

### 3.1 第一代：Static Batching（静态批处理）

```
工作原理:
  1. 等一批请求凑齐（比如 4 个）
  2. Pad 到相同长度
  3. 一起跑完
  4. 等下一批

示意:
  Batch 1: [A(100 tokens), B(200 tokens), C(50 tokens)]
           → 全部 pad 到 200 (浪费计算!)
           → 等所有完成（C 只需 50 但要等 A 和 B）
           → 释放，开始下一批

  Batch 2: [D(80 tokens), E(150 tokens)]
           → 全部 pad 到 150
           → ...
```

**问题**：
- ❌ 短请求要等长请求（"木桶效应"）
- ❌ Padding 浪费大量计算
- ❌ 新请求无法中途加入
- ❌ Decode 阶段 GPU 利用率极低（每个请求每步只生成 1 token）

```
GPU 利用率时间线（Static Batching）:
  Prefill: ████████████████████░░  ~80%
  Decode:  █░░░░░░░░░░░░░░░░░░░░  ~5-10%
            ↑ 大量时间在等内存而不是计算
```

### 3.2 第二代：Dynamic Batching（动态批处理）

```
工作原理:
  1. 在时间窗口内（如 1ms）收集到达的请求
  2. 组成一个 batch，动态 pad
  3. 跑完后再组下一批

比 Static Batching 更灵活，但对 decode 阶段的持续性浪费仍然无能为力。
```

### 3.3 第三代：Continuous Batching（连续批处理）

**核心思想：不在 "batch 级别" 等待——在每个 decode step 都重新组织 batch。**

```
示意（每行 = 一个 decode step）:

Step 1:  [A(prefill), B(dec), C(dec)]
Step 2:  [A(dec), B(dec), C(dec)]
Step 3:  [A(dec), B(dec), C(dec)]     ← C 在这一步生成了 EOS
Step 4:  [A(dec), B(dec), D(prefill)]  ← C 移出，D 加入!
Step 5:  [A(dec), B(dec), D(dec)]      ← A 在这一步生成了 EOS
Step 6:  [B(dec), D(dec), E(prefill)]  ← A 移出，E 加入!
...

关键观察:
  - 每个 step 都动态决定 batch 成员
  - 完成的请求立即移出，空间释放给新请求
  - 新请求可以随时以 prefill 模式加入
  - GPU 始终在处理有效 token，而非等待或 pad
```

**效果对比**：

```
                         Static Batching    Continuous Batching
单请求平均等待时间:         5-30s               <1s
GPU 利用率 (decode):       5-10%               30-50%
吞吐 (tokens/s):           baseline            2-4x
长请求阻塞短请求:           严重                 基本消除
```

---

## 4. Continuous Batching 的完整实现

### 4.1 Scheduler 主循环

```python
# ============================================
# LLM Scheduler 的核心调度循环
# 参考 vLLM Scheduler 的核心逻辑
# ============================================

from collections import deque
from typing import List, Deque

class LLMScheduler:
    """
    LLM 推理调度器

    核心职责：
    1. 管理 waiting 队列和 running 请求
    2. 每个 step 决定 batch 组成
    3. 控制 prefill/decode 的比例（token budget）
    4. 分配/释放 KV Cache blocks
    """
    def __init__(
        self,
        max_num_batched_tokens: int = 8192,   # 每轮最多处理的总 token 数
        max_num_seqs: int = 256,               # 每轮最多处理的序列数
        block_size: int = 16,                  # 每个 KV block 的 token 数
    ):
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_num_seqs = max_num_seqs
        self.block_size = block_size

        # 请求队列
        self.waiting: Deque[Request] = deque()   # 等待队列
        self.running: List[Request] = []          # 正在运行
        self.swapped: List[Request] = []          # 被换出

        # KV Cache block 管理器
        self.cache_manager = PagedAttentionCacheManager(
            num_blocks=4096,
            block_size=block_size
        )

        # 统计
        self.step_counter = 0

    # ==========================================
    # 主循环
    # ==========================================
    def step(self) -> List[Request]:
        """
        执行一个调度 + 推理 step

        返回本轮参与计算的请求列表
        """
        self.step_counter += 1

        # ---- Phase 1: 清理已完成的请求 ----
        self._cleanup_finished()

        # ---- Phase 2: 从 waiting 队列取请求做 prefill ----
        scheduled_reqs = self._schedule_new_requests()

        # ---- Phase 3: 组织本轮的 batch ----
        # scheduled_reqs 包含本轮要做 prefill 的新请求
        # self.running 包含正在 decode 的请求
        # 但总数受 max_num_seqs 和 token budget 限制
        batch = self._select_decode_batch(scheduled_reqs)

        # ---- Phase 4: 执行推理（模型前向） ----
        if batch:
            self._execute_batch(batch)

        # ---- Phase 5: 采样 & 状态更新 ----
        self._sample_and_update(batch)

        return batch

    # ==========================================
    # Phase 1: 清理完成的请求
    # ==========================================
    def _cleanup_finished(self):
        """释放已完成请求的资源"""
        still_running = []
        for req in self.running:
            if req.is_finished:
                # 释放该请求占用的所有 KV Cache blocks
                self.cache_manager.release_sequence(req.req_id)
                req.finish_time = time.time()
            else:
                still_running.append(req)
        self.running = still_running

    # ==========================================
    # Phase 2: 新请求调度（Prefill）
    # ==========================================
    def _schedule_new_requests(self) -> List[Request]:
        """
        从 waiting 队列中取请求做 prefill

        受两个限制：
        1. Token budget: 本轮所有 prefill token 总和不能超
        2. 序列数上限: 不能超 max_num_seqs
        """
        scheduled = []
        token_budget = self.max_num_batched_tokens

        # 预留一些 budget 给 decode 请求
        # 每个 decode 请求 1 个 token
        decode_budget = len(self.running)
        available_budget = token_budget - decode_budget

        while self.waiting and available_budget > 0:
            req = self.waiting[0]
            num_prompt_tokens = len(req.prompt_tokens)

            # 检查：token budget 是否够？
            if num_prompt_tokens > available_budget:
                # 这个请求的 prompt 太长，本轮放不下
                # 选项 1: 等待下一轮
                # 选项 2: Chunked Prefill — 把长 prompt 切成多块
                if num_prompt_tokens <= token_budget:
                    # Chunked Prefill: 切出 available_budget 个 token
                    # 先处理前 available_budget 个 token
                    chunk_size = available_budget
                    req_chunk = self._chunk_prefill(req, chunk_size)
                    scheduled.append(req_chunk)
                break

            # 检查：序列数是否超限？
            if len(self.running) + len(scheduled) >= self.max_num_seqs:
                break

            # 检查：KV Cache 空间是否够？
            blocks_needed = (num_prompt_tokens + self.block_size - 1) // self.block_size
            if blocks_needed > self.cache_manager.allocator.num_free_blocks:
                # 空间不够：尝试抢占低优先级请求
                if not self._try_preempt(blocks_needed):
                    break

            # 一切就绪：分配资源
            self.waiting.popleft()
            success = self.cache_manager.schedule_prefill(req)
            if success:
                req.status = ReqStatus.PREFILL
                scheduled.append(req)

            available_budget -= num_prompt_tokens

        return scheduled

    # ==========================================
    # Chunked Prefill: 长 prompt 的分块处理
    # ==========================================
    def _chunk_prefill(self, req: Request, chunk_size: int) -> Request:
        """
        将长 prompt 切成多个 chunk 逐步处理

        为什么需要？
        - 避免单个长 prompt 长时间阻塞其他请求
        - 让 decode 请求可以和 prefill 交替进行

        例如: 8000 token 的 prompt
          Chunk 1: token 0~2047    (prefill 2048 tokens)
          Chunk 2: token 2048~4095  (prefill 2048 tokens)
          Chunk 3: token 4096~6143  (prefill 2048 tokens)
          Chunk 4: token 6144~7999  (prefill 1856 tokens)
          → 然后进入正常 decode

        每做完一个 chunk，释放一次 token budget，
        让其他请求有机会插入。
        """
        # 创建临时请求对象，只处理 chunk_size 个 token
        chunk_req = Request(
            req_id=req.req_id,
            prompt_tokens=req.prompt_tokens[:chunk_size],
            max_tokens=req.max_tokens,
        )
        chunk_req.status = ReqStatus.PREFILL
        # 把未处理的 token 放回 waiting 队列头部
        remaining = req.prompt_tokens[chunk_size:]
        if remaining:
            req.prompt_tokens = remaining
            self.waiting.appendleft(req)
        return chunk_req

    # ==========================================
    # Phase 3: 组织 Decode Batch
    # ==========================================
    def _select_decode_batch(self, new_prefill_reqs: List[Request]) -> List[Request]:
        """
        本轮参与的请求 = running requests + new prefill requests

        受 max_num_seqs 限制：
        - 如果 running 已满，新的 prefill 请求需要等待
        """
        # running 中每个请求贡献 1 个 token（decode 阶段）
        batch = list(self.running)

        # 新 prefill 的请求也加入（它们贡献 prompt_length 个 token）
        for req in new_prefill_reqs:
            batch.append(req)
            if req.status == ReqStatus.PREFILL:
                # prefill 完成后，标记为 DECODING
                # （实际在 _sample_and_update 中处理）
                pass

        return batch[:self.max_num_seqs]

    # ==========================================
    # Phase 4: 执行推理
    # ==========================================
    def _execute_batch(self, batch: List[Request]):
        """
        执行模型前向

        真实实现会：
        1. 拼接 batch 中所有请求的输入 token
        2. 调用 model.forward()
        3. 返回每个请求的 logits
        4. 使用优化的 attention kernel（PagedAttention/FlashAttention）
        """
        # 伪代码：
        # batch_tokens = [req.get_next_input_token() for req in batch]
        # batch_block_tables = [req.block_table for req in batch]
        # logits = model.forward(batch_tokens, batch_block_tables)
        # for req, logit in zip(batch, logits):
        #     req.pending_logits = logit
        pass

    # ==========================================
    # Phase 5: 采样 & 状态更新
    # ==========================================
    def _sample_and_update(self, batch: List[Request]):
        """对每个请求进行采样，更新状态"""
        for req in batch:
            if req.status == ReqStatus.PREFILL:
                # Prefill 完成 → 取最后一个 token 的 logits 做采样
                # （prefill 时计算了所有 prompt token，但只需要最后一个的输出）
                next_token = self._sample(req, req.pending_logits[-1])
                req.output_tokens.append(next_token)

                # 记下首个 token 的时间
                if req.first_token_time is None:
                    req.first_token_time = time.time()

                req.status = ReqStatus.DECODING
                # 把请求加入 running 列表
                if req not in self.running:
                    self.running.append(req)

            elif req.status == ReqStatus.DECODING:
                # Decode 阶段 → 采样下一个 token
                next_token = self._sample(req, req.pending_logits[0])

                # 判断是否需要新 KV block
                self.cache_manager.schedule_decode(req)

                req.output_tokens.append(next_token)

                # 结束条件检查
                if next_token == EOS_TOKEN_ID or len(req.output_tokens) >= req.max_tokens:
                    req.status = ReqStatus.FINISHED

    def _sample(self, req: Request, logits: List[float]) -> int:
        """采样逻辑：temperature + top_k + top_p"""
        # 应用 temperature
        logits = [l / req.temperature for l in logits]
        # softmax
        probs = softmax(logits)
        # top-k
        if req.top_k > 0:
            probs = apply_top_k(probs, req.top_k)
        # top-p
        if req.top_p < 1.0:
            probs = apply_top_p(probs, req.top_p)
        # 采样
        return sample_from(probs)

    # ==========================================
    # 抢占机制
    # ==========================================
    def _try_preempt(self, blocks_needed: int) -> bool:
        """
        当 KV Cache 空间不够时，抢占低优先级请求

        策略：swap out 优先级最低的运行中请求
        """
        # 简单实现：按已生成的 token 数排序
        # 生成最多的请求最"接近完成"，不应该被抢占
        # 生成最少的请求最"浪费"（prefill 的成本还没摊销），优先被抢占
        victims = sorted(self.running, key=lambda r: len(r.output_tokens))
        freed = 0
        for victim in victims:
            if freed >= blocks_needed:
                break
            # 把这个请求的 KV blocks swap 到 CPU
            # self._swap_out(victim)
            freed += len(victim.block_table)
            victim.status = ReqStatus.SWAPPED
            self.swapped.append(victim)
            self.running.remove(victim)

        return freed >= blocks_needed

    # ==========================================
    # 统计接口
    # ==========================================
    def stats(self) -> dict:
        return {
            "step": self.step_counter,
            "waiting": len(self.waiting),
            "running": len(self.running),
            "swapped": len(self.swapped),
            "free_blocks": self.cache_manager.allocator.num_free_blocks,
        }
```

### 4.2 调度决策树

```
每个 step 开始时：

1. 有没有请求结束了？
   → 是：释放 KV Cache，回收 blocks
   → 否：继续

2. waiting 队列有没有新请求？
   → 是：检查 token budget 够不够
       → 够：检查 KV space 够不够
           → 够：prefill 它！
           → 不够：能抢占吗？
               → 能：swap out 受害者，prefill 新请求
               → 不能：等下一轮
       → 不够：等下一轮（或 chunked prefill）
   → 否：只用 running requests 做 decode

3. 组 batch：running + 新 prefill 的请求

4. 执行 forward → 采样 → 更新状态 → 下一轮
```

---

## 5. Token Budget：调度的"货币"

### 5.1 为什么按 token 数而不是请求数？

```
假设 max_num_batched_tokens = 8192:

场景 A：全是 decode 请求
  每请求 1 token → 最多 8192 个请求同时跑
  但 max_num_seqs = 256 → 实际最多 256

场景 B：1 个长 prompt 来了
  请求 X: 6000 tokens prefill
  剩余 budget: 8192 - 6000 = 2192 tokens
  → 还可以放入 2192 个 decode 请求 或 一个 2000 token 的 prompt

场景 C：一个 10K token 的超长 prompt
  一轮放不下 → Chunked Prefill
  Chunk 1: 分 5000 tokens → 剩余 3192
  Chunk 2: 下一轮再分 5000 tokens
```

**Token budget 是一种"公平性"机制**：它确保一个长 prompt 不会无限期地阻塞所有短请求。通过控制每轮的总 token 数，短请求有机会在长 prompt 的 chunk 之间插入执行。

### 5.2 Prefill 的"加权"问题

```
Decode 请求：每个只占 1 个 token budget
Prefill 请求：每个占 prompt_length 个 token budget

这就意味着：
  - 1 个 2000-token 的 prefill ≈ 2000 个 decode 请求
  - 如果 budget=8192，同时有 200+ 个 decode 请求在跑
  - 调度器必须平衡：不能为了 decode 吞吐一直推迟 prefill
    （否则新用户首 token 延迟过高）
```

典型策略：**预算分配比例**

```python
# vLLM 等框架通常的做法
class BudgetPolicy:
    def __init__(self, total_budget: int = 8192):
        self.total = total_budget
        self.prefill_reserved = int(total * 0.3)  # 30% 留给 prefill
        self.decode_max = total_budget             # decode 可以用全部

    def can_schedule_prefill(self, prompt_len: int, current_decode: int) -> bool:
        """判断是否可以调度一个 prefill"""
        # 如果 decode 请求太多，限制 prefill
        if current_decode > 200:
            # 只允许短 prompt
            return prompt_len <= self.prefill_reserved
        # decode 请求不多，prefill 可以用更多 budget
        available = self.total - current_decode
        return prompt_len <= available
```

---

## 6. 为什么 Decode 阶段 GPU 利用率低？

这是一个经典的 AI Infra 面试题。答案是分层的：

### 6.1 第一层：算术强度低

```
Prefill 时:
  QK^T: [16, seq, 128] × [16, 128, seq] → [16, seq, seq]
  计算量: O(seq² × d) FLOPs
  每个 Q 元素要乘 seq 次 → 算术强度高

Decode 时:
  QK^T: [16, 1, 128] × [16, 128, seq] → [16, 1, seq]
  计算量: O(seq × d) FLOPs → 比 prefill 少 seq 倍！
  但 KV Cache 的大小不变（还是 seq × d 的内存访问）
  → 算术强度 = 计算量 / 访存量 → 极低 → memory-bound
```

### 6.2 第二层：batch 太小

```
单请求 Decode: 每个 step 只有 1 个 token 的矩阵乘法
  矩阵形状：[1, hidden] × [hidden, hidden]
  → GPU 的 Tensor Core 设计为处理大矩阵
  → 这个形状连一个 warp 都填不满

多请求 Decode (Continuous Batching):
  把 N 个请求的 1-token 矩阵 "vertical concatenation"
  变成 [N, hidden] × [hidden, hidden]
  → N 越大，越接近 GPU 设计目标
  → 这就是 Continuous Batching 提升 GPU 利用率的关键！
```

### 6.3 第三层：KV Cache 读取

```
每层 Attention 都需要读取全部历史 KV Cache:
  KV Cache 大小 = 2 × num_layers × seq_len × num_kv_heads × head_dim × dtype

对于 Qwen3-0.6B, seq=4096:
  每层读 = 2 × 4096 × 8 × 128 × 2 = 16 MB
  28 层总读 = 28 × 16 = 448 MB

每次 decode（1 token!）要读 448 MB → 内存带宽成为绝对瓶颈
```

### 6.4 可视化对比

```
Prefill (计算密集):
  GPU SM ┌─────────────────────────────┐
         │ ████████████████████████████ │  活跃
         │ ████████████████████████████ │
  Memory │ ████░░░░░░░░░░░░░░░░░░░░░░░ │  等待少
         └─────────────────────────────┘
  Utilization: ~80%

Decode (内存密集):
  GPU SM ┌─────────────────────────────┐
         │ ██░░░░██░░░░██░░░░██░░░░██░░│  大量气泡
         │ ██░░░░██░░░░██░░░░██░░░░██░░│
  Memory │ ████████████████████████████ │  持续忙碌
         └─────────────────────────────┘
  Utilization: ~10-30%

Continuous Batching 后:
  GPU SM ┌─────────────────────────────┐
         │ ██████░░██████░░██████░░████│  气泡减少
         │ ██████░░██████░░██████░░████│
  Memory │ ████████████████████████████ │  仍在饱和
         └─────────────────────────────┘
  Utilization: ~30-50%  (仍然受限于内存，改善有限但明显)
```

---

## 7. 性能指标：TTFT、TPOT、Throughput

### 7.1 三大指标

```
用户视角                 系统视角

"第一句话出来快不快？"    TTFT (Time To First Token)
         ↓
"后面快不快？"            TPOT (Time Per Output Token)
         ↓
"整体能服务多少人？"       Throughput (tokens/s)
```

### 7.2 精确计算

```
TTFT = prefill_compute + first_token_sample + queue_wait

  prefill_compute 受：
    - prompt 长度（线性）
    - 模型大小
    - 是否有 prefix cache 命中
    - batch 中其他 prefill 的影响

TPOT ≈ decode_step_time 的平均值

  decode_step_time 受：
    - KV Cache 大小（随生成进度递增！）
    - 当前 batch 大小
    - attention kernel 质量

Throughput = total_output_tokens / total_wall_time

  系统级吞吐 ≠ 单请求速度
  好的调度器能在相同时间内服务更多请求
  即使每个请求的延迟稍高一点
```

### 7.3 延迟 vs 吞吐的权衡

```
Latency (延迟):    "我的请求多久完成？"
Throughput (吞吐): "系统总共处理了多少？"

冲突场景:
  要低延迟 → 来一个请求立即处理
           → batch 小，GPU 利用率低
           → 吞吐低

  要高吞吐 → 等更多请求凑齐再处理
           → batch 大，GPU 利用率高
           → 但单个请求的等待时间增加

Continuous Batching 的优雅之处:
  不等待！立即开始处理。
  decode 阶段自然地"凑"出了 batch
  → 既保证了低延迟，又获得了高吞吐
  → 但这只在 decode 阶段成立；prefill 的新请求仍需调度决策
```

---

## 8. ncnn_llm 的 Runtime vs vLLM 的 Scheduler

| 维度 | ncnn_llm Runtime | vLLM Scheduler |
|------|-----------------|---------------|
| **请求模型** | 单请求，同步调用 | 多请求，异步队列 |
| **Batch** | 无 batching | Continuous Batching |
| **状态管理** | `ncnn_llm_gpt_ctx` 持有单个上下文 | `Request` 实例，scheduler 统一管理 |
| **内存管理** | 连续 KVCache Mat，每步 memcpy | Paged KV blocks，按需 allocate/free |
| **并发** | 单线程为主 | 高度并发，多 worker |
| **调度策略** | 无（来一个跑一个） | Token budget + priority + preemption |
| **优化重点** | 轻量、跨平台、低内存 | 吞吐、延迟、GPU 利用率 |

**ncnn_llm 的 `generate()` 循环**本质上是一个**单请求的 continuous loop**：

```cpp
// ncnn_llm 的 generate() 就是一个 decode loop
// 但没有多请求调度，没有 batching
for (int step = 0; step < cfg.max_new_tokens; ++step) {
    // 当前 token → embedding
    // 生成 RoPE cache
    // Decoder 前向 (使用已有 KV Cache)
    // Projection → logits
    // Repetition Penalty → Sampling
    // 流式输出
}
```

如果要把 ncnn_llm 扩展成简单的 serving 系统，需要在外面包一层：

```cpp
// 伪代码：把 ncnn_llm 包装成简单的 serving
class SimpleLLMServer {
    deque<shared_ptr<Request>> waiting_queue;
    vector<shared_ptr<Request>> running;

    void step() {
        // 1. 清理完成的请求
        // 2. 从 waiting 取一个做 prefill
        // 3. 对 running 做一轮 decode
        // 4. 采样 & 更新

        // 没有 Continuous Batching 时，
        // 只能处理一个请求的 decode
        // → 吞吐严重受限
    }
};
```

---

## 9. 面试回答模板

### Q1: 什么是 Continuous Batching？为什么它能提升吞吐？

```
回答框架：

Continuous Batching 的核心思想是在每个 decode step 动态重组 batch，
而不是等整个 batch 全部完成。

传统 static batching 的问题：
- 短请求要等长请求（batch 级别的"木桶效应"）
- decode 阶段 GPU 利用率极低（每请求每步只生成 1 token）

Continuous Batching 的解决方案：
- 每个 decode step 结束时，检查哪些请求已完成（EOS/max_tokens）
- 立即将它们移出 batch，释放 KV Cache
- 从 waiting 队列中取出新请求，以 prefill 模式加入
- 下一轮 batch = 剩余 decode 请求 + 新 prefill 请求

效果：
- GPU 绝大部分时间在处理有效 token（而非 padding 或等待）
- 长请求不再阻塞短请求
- 吞吐提升 2-4x
- 要求配合 PagedAttention 实现灵活的 KV Cache 管理
```

### Q2: 为什么 decode 阶段 GPU 利用率低？有哪些优化手段？

```
回答框架：

Decode 阶段 GPU 利用率低的原因有三个层次：

1. 算术强度低：每 token 的计算量与 KV Cache 读取量的比值极低
   → 瓶颈在内存带宽，而非算力

2. Batch 太小：单请求 1 token 的矩阵乘法无法充分利用 Tensor Core
   → 需要 batching 来增大矩阵维度

3. KV Cache 随序列增长：越长的生成，内存读取越多

优化手段：
- Continuous Batching：增加有效 batch 大小
- FlashAttention：减少 attention 中间结果的显存读写
- KV Cache 量化 (INT8/FP8)：减少内存带宽需求
- Prefill/Decode disaggregation：把两种负载分开部署
- Speculative Decoding：用 draft model 一次预测多个 token
```

---

## 10. 本章学习清单

- [ ] 能画出 Request 的完整生命周期状态图
- [ ] 能写出 Continuous Batching 的伪代码（至少 5 个 Phase）
- [ ] 能解释 Token Budget 为什么用"token 数"而非"请求数"
- [ ] 能区分 Static / Dynamic / Continuous Batching 的差异
- [ ] 能从三个层次解释"为什么 decode 阶段 GPU 利用率低"
- [ ] 能解释 TTFT、TPOT、Throughput 的含义和相互制约关系
- [ ] 能说明 Scheduler 在 prefill 和 decode 阶段的调度差异
- [ ] 能解释 Chunked Prefill 解决什么问题
- [ ] 能用 ncnn_llm 的 `generate()` 循环类比单请求调度

---

*上一篇: [04_memory_kv_cache.md](04_memory_kv_cache.md) — KV Cache 与 PagedAttention*
*下一篇: [06_quantization_optimization.md](06_quantization_optimization.md) — 量化、FlashAttention 与性能优化*
