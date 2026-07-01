# 04｜Memory System：KV Cache、PagedAttention 与显存瓶颈

> 本章从源码级深度拆解 KV Cache 的三种实现范式：ncnn_llm 的连续缓存、llama.cpp 的 cell 管理、vLLM 的 PagedAttention 分页系统。

---

## 1. 为什么 KV Cache 是 LLM 推理的核心瓶颈？

### 1.1 自回归生成的本质矛盾

LLM 自回归生成时，每步只产出 1 个新 token。如果不缓存历史 K/V：

```
第 1 步：计算 token_0 的 K/V
第 2 步：重新计算 token_0 + token_1 的 K/V
第 3 步：重新计算 token_0 + token_1 + token_2 的 K/V
...
第 n 步：重新计算所有 n 个 token 的 K/V  ← O(n²) 计算量！
```

**KV Cache 的核心权衡**：用内存换计算——多占 O(n×d) 的内存，省下 O(n²) 的重复计算。

### 1.2 从源码看 KV Cache 存什么

以 Qwen3-0.6B（28 层，GQA: 16 Q heads / 8 KV heads，head_dim=128）为例：

**ncnn_llm 的数据结构**（[ncnn_llm_base.h:14](源码/ncnn_llm-main/src/ncnn_llm_base.h#L14)）：

```cpp
// 一个 layer 的 KV Cache = 一对 ncnn::Mat
using KVCache = std::vector<std::pair<ncnn::Mat, ncnn::Mat>>;
// KVCache.size() == attn_cnt == 28

// 每层的 Mat 形状（GQA，8 个 KV head）：
// K_cache[i]: Mat(w=128, h=seq_len, c=8)  → [head_dim, seq_len, num_kv_heads]
// V_cache[i]: Mat(w=128, h=seq_len, c=8)  → [head_dim, seq_len, num_kv_heads]
```

**上下文结构**（[ncnn_llm_gpt.h:45-54](源码/ncnn_llm-main/src/ncnn_llm_gpt.h#L45-L54)）：

```cpp
class ncnn_llm_gpt_ctx {
public:
    KVCache kv_cache;       // 28 层的 (K, V)，每层一对 Mat
    int cur_token = 0;      // 当前 token ID（下一步的输入）
    int position_id = 0;    // RoPE 位置计数器（单调递增）

    // 深拷贝——多轮对话时复制上下文
    virtual std::shared_ptr<ncnn_llm_gpt_ctx> clone() const = 0;
};
```

三个字段的协同关系：

```
cur_token ──→ embed_net ──→ decoder_net(使用 position_id 生成 RoPE, 使用 kv_cache 做 Attention)
                                    │
                    position_id++  ←┘
                    kv_cache 被更新（追加新 token 的 K/V）
                    cur_token ← 采样结果
```

---

## 2. 三种 KV Cache 架构的源码级对比

从简单到复杂，业界有三种 KV Cache 管理范式：

```
ncnn_llm 连续缓存  ──→  llama.cpp cell 管理  ──→  vLLM PagedAttention
  (最简单，端侧)         (序列管理，灵活分配)        (分页，服务端高并发)
```

### 2.1 范式一：ncnn_llm 的连续缓存——最简单直接的实现

**核心思路**：每层的 K/V 存在一个连续的大 Mat 中，新 token 追加时用 `memcpy` 拼接。

**SDPA 层中的 concat 逻辑**（[ncnn sdpa.cpp:69-87](源码/ncnn-master/src/layer/sdpa.cpp#L69-L87)）：

```cpp
// 关键变量
const int past_seqlen = kv_cache ? past_key.h : 0;  // 历史缓存长度
const int cur_seqlen  = cur_key.h;                   // 当前新 token 数
const int dst_seqlen  = past_seqlen + cur_seqlen;    // 拼接后总长度

Mat key = cur_key;  // 默认直接用当前 key
if (past_seqlen > 0) {
    // 有历史缓存：分配新内存，拼接 past + current
    key.create(embed_dim, dst_seqlen, num_group, 4u, opt.blob_allocator);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < num_group; q++) {
        const Mat past_key_head = past_key.channel(q);  // 历史 K
        const Mat cur_key_head  = cur_key.channel(q);   // 新 K
        Mat key_head = key.channel(q);

        // 先拷贝历史 → 再拷贝新 token
        memcpy(key_head.row(0),           past_key_head, embed_dim * past_seqlen * sizeof(float));
        memcpy(key_head.row(past_seqlen), cur_key_head,  embed_dim * cur_seqlen  * sizeof(float));
    }
}
// Value 同理...
```

**内存布局可视化**：

```
decode step t=0 (prefill 4 tokens):
  K_cache[0]: [d₀t₀ d₀t₁ d₀t₂ d₀t₃] [d₁t₀ d₁t₁ d₁t₂ d₁t₃] ... (8 channels)
              ╰─── token 0~3 连续存放 ───╯

decode step t=1 (生成 1 token):
  复制旧 4 + 追加 1 → 新 Mat 5 行
  K_cache[0]: [d₀t₀ d₀t₁ d₀t₂ d₀t₃ d₀t₄] ...

decode step t=2:
  复制旧 5 + 追加 1 → 新 Mat 6 行
  ...
```

**ncnn_llm generate 循环中的 cache 更新**（[ncnn_llm_gpt.cpp:842-1007](源码/ncnn_llm-main/src/ncnn_llm_gpt.cpp)）：

```cpp
for (int step = 0; step < cfg.max_new_tokens; ++step) {
    // Step 1: 当前 token → embedding
    // Step 2: 生成 RoPE cache（position_id++）
    // Step 3: 构建 mask（全 0，因为 cache 已有完整历史）
    ncnn::Mat mask(ctx->kv_cache[0].first.h + 1, 1);  // cache 长度 + 1
    mask.fill(0.f);

    // Step 4: Decoder 前向
    {
        ncnn::Extractor ex = decoder_net->create_extractor();
        ex.input("in0", cur_embed);

        // 输入旧 KV Cache（每层的 past K/V）
        for (int i = 0; i < attn_cnt; ++i) {
            ex.input("cache_k" + std::to_string(i), ctx->kv_cache[i].first);
            ex.input("cache_v" + std::to_string(i), ctx->kv_cache[i].second);
        }

        // 提取更新后的 KV Cache（旧 + 新 token 拼接后的完整版本）
        for (int i = 0; i < attn_cnt; ++i) {
            ncnn::Mat k_cache, v_cache;
            ex.extract("out_cache_k" + std::to_string(i), k_cache);
            ex.extract("out_cache_v" + std::to_string(i), v_cache);
            ctx->kv_cache[i] = { move(k_cache), move(v_cache) };
        }

        ex.extract("out0", decode_out);
    }
    // Step 5-7: Projection → 采样 → 下一轮
}
```

**ncnn_llm 方案的优缺点**：

| 优点 | 缺点 |
|------|------|
| 实现极简，50 行 C++ | 每步 decode 都要 memcpy 整个 cache |
| 内存连续，cache 友好 | 单请求场景专用，无并发支持 |
| 适合端侧单用户 | 无法复用 prefix/共享 cache |

---

### 2.2 范式二：llama.cpp 的 Cell 管理——序列感知的灵活分配

llama.cpp 将 KV Cache 抽象为 **cell 池**，每个 cell 代表一个 token 位置。多个序列（请求）共享这个 cell 池，通过 `find_slot()` 按需分配。

**核心数据结构**（[llama-kv-cache.h:20-321](源码/llama.cpp-master/src/llama-kv-cache.h)）：

```cpp
class llama_kv_cache : public llama_memory_i {
private:
    // 物理存储：每层一个大的 K/V tensor
    struct kv_layer {
        uint32_t il;            // 模型层索引
        ggml_tensor * k;        // K 存储 [n_embd_head_k, n_head_k, kv_size]
        ggml_tensor * v;        // V 存储 [n_embd_head_v, n_head_v, kv_size]
        // 多 stream 视图（stream = 独立序列的逻辑分组）
        std::vector<ggml_tensor *> k_stream;
        std::vector<ggml_tensor *> v_stream;
    };

    // KV cells：每个 cell 描述一个 token 位置的元数据
    std::shared_ptr<llama_kv_cells_vec> v_cells_impl;
    llama_kv_cells_vec & v_cells;  // [n_stream][kv_size] 的 cell 数组

    // 每个 stream 的写入头（类似操作系统的 brk 指针）
    std::vector<uint32_t> v_heads;

    // 序列 ID → stream ID 的映射
    std::vector<uint32_t> seq_to_stream;

    const uint32_t n_seq_max;  // 最大并发序列数
    const uint32_t n_stream;   // stream 数量（unified 模式下 = 1，否则 = n_seq_max）
};
```

**Cell 的元数据**（每个 cell 记录）：

```
cell[i]:
  - pos:         该 cell 存储的 token 在序列中的位置
  - seq_id[]:    哪些序列引用了这个 cell（支持多序列共享）
  - n_seq_id:    引用计数
  - is_empty():  是否空闲
```

**find_slot() 算法**——在 cell 池中找到连续空间（[llama-kv-cache.cpp:910](源码/llama.cpp-master/src/llama-kv-cache.cpp#L910)）：

```cpp
llama_kv_cache::slot_info llama_kv_cache::find_slot(
    const llama_ubatch & ubatch, bool cont) const {

    const int n_tokens = ubatch.n_tokens;

    for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
        const auto seq_id = ubatch.seq_id_unq[s];
        const auto stream_id = seq_to_stream[seq_id];
        auto & cells = v_cells[stream_id];
        const uint32_t head_cur = v_heads[stream_id];

        // 从 head_cur 开始扫描，找 n_tokens 个连续空闲 cell
        // head_cur 是一个启发式指针：记录上次分配的位置附近

        // 如果要求连续（cont=true），则必须找到连续的 n_tokens 个 cell
        // 否则（cont=false），允许使用非连续的 cell

        // 找不到 → 返回空 slot_info → 上层决定是否做 cache shift
    }
}
```

**llama.cpp cache 的 cell 状态可视化**（debug 模式输出）：

```
stream[0], n = 4096, used = 1823, head = 1850, size = 4096

0123456789ABCDEF...0123..................MMM..............5566..............
╰─ seq 0  ─╯        ╰─ seq 1 ─╯          ╰共享╯         ╰ seq 5 ─╯╰ seq 6 ─╯

. = 空闲 cell
数字 = 该 cell 被某个 seq 独占
M = 多个 seq 共享该 cell
```

**cache shift 机制**——当 cell 池不够时，将旧 token 移出（SWA 模式）：

```
shift 前:  [t₀ t₁ t₂ t₃ t₄ t₅ t₆ t₇]  ← head 在末尾，无空间
shift 后:  [t₂ t₃ t₄ t₅ t₆ t₇ .  .]   ← t₀ t₁ 被丢弃（SWA 外），head 前移
```

**llama.cpp 相比 ncnn_llm 的进步**：

| 特性 | ncnn_llm | llama.cpp |
|------|----------|-----------|
| 存储模型 | 每层独立 Mat，每步整体复制 | 固定大小 cell 池，按需分配 |
| 多序列支持 | ❌ 单请求 | ✅ 多 stream 管理 |
| 序列操作 | ❌ | ✅ seq_rm / seq_cp / seq_keep / seq_add |
| Cache 共享 | ❌ | ✅ 多 seq 可引用同一 cell |
| 内存效率 | 低（每次重新分配） | 中（cell 复用但需要连续分配） |
| Code 复杂度 | ~50 行 | ~2300 行 |

---

### 2.3 范式三：vLLM PagedAttention——虚拟内存思想在 GPU 上的应用

llama.cpp 解决了"多序列共享 cell 池"的问题，但仍要求每个序列的 cell **连续分配**。这导致：

```
问题场景：cell 池总空闲 1000 个，但最大连续空闲块只有 100 个
         → 一个需要 200 cell 的请求无法分配
         → 外部碎片！
```

**PagedAttention 的核心洞察**：

```
操作系统的解决方案：虚拟内存 → 物理页框的映射
PagedAttention 的方案：逻辑 token 位置 → 物理 KV block 的映射
```

#### 2.3.1 核心数据结构设计

```python
# ============================================
# vLLM PagedAttention 核心数据结构（简化但忠于原设计）
# 参考: vLLM v1 cache_manager, block_table, KVBlock
# ============================================

from dataclasses import dataclass
from typing import List, Optional, Dict

# ---- 常量 ----
BLOCK_SIZE = 16          # 每个 block 存 16 个 token
NUM_LAYERS = 28          # Transformer 层数
NUM_KV_HEADS = 8         # GQA 的 KV head 数
HEAD_DIM = 128           # 每个 head 的维度
BYTES_PER_ELEM = 2       # FP16

# 每个 block 的字节数
BLOCK_BYTES = 2 * NUM_LAYERS * BLOCK_SIZE * NUM_KV_HEADS * HEAD_DIM * BYTES_PER_ELEM
# = 2 × 28 × 16 × 8 × 128 × 2 = 1,835,008 bytes ≈ 1.75 MB


@dataclass
class KVCacheBlock:
    """一个物理 KV block —— 存储固定数量 token 的 K/V"""
    block_id: int
    # 物理存储：每层一对 K/V tensor
    # k_layers[il]: [BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM]
    # v_layers[il]: [BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM]
    ref_count: int = 0     # 引用计数（多序列共享时 >1）

    @property
    def is_free(self) -> bool:
        return self.ref_count == 0


class BlockTable:
    """
    逻辑 token 序列 → 物理 block 的映射表

    逻辑序列:  [tok₀ ... tok₁₅] [tok₁₆ ... tok₃₁] [tok₃₂ ... tok₄₇]
                   ↓                ↓                 ↓
    物理 block:  block_7         block_2          block_9

    这就是 "Block Table" — 类比操作系统的页表
    """
    def __init__(self, max_blocks_per_seq: int):
        self.block_ids: List[int] = []          # [7, 2, 9, ...]
        self.max_blocks_per_seq = max_blocks_per_seq

    def num_blocks(self) -> int:
        return len(self.block_ids)

    def num_tokens(self) -> int:
        return len(self.block_ids) * BLOCK_SIZE

    def append_block(self, block_id: int):
        self.block_ids.append(block_id)

    def __repr__(self):
        return f"BlockTable({self.block_ids})"


class BlockAllocator:
    """
    全局 block 池管理器

    核心操作：
    1. allocate()  — 分配一个空闲 block
    2. free()      — 释放 block（ref_count 减到 0 时回收）
    3. fork()      — 复制 block table（COW 语义：只增 ref_count）
    """
    def __init__(self, num_blocks: int):
        # 预分配所有物理 block
        self.blocks: List[KVCacheBlock] = [
            KVCacheBlock(block_id=i) for i in range(num_blocks)
        ]
        self.free_blocks: List[int] = list(range(num_blocks))

    def allocate(self) -> Optional[int]:
        """从空闲池取一个 block"""
        if not self.free_blocks:
            return None  # OOM!
        block_id = self.free_blocks.pop()
        self.blocks[block_id].ref_count = 1
        return block_id

    def free(self, block_id: int):
        """减少引用计数，计数归零时回收到空闲池"""
        block = self.blocks[block_id]
        block.ref_count -= 1
        if block.ref_count == 0:
            self.free_blocks.append(block_id)

    def fork(self, block_ids: List[int]) -> List[int]:
        """COW fork：所有 block ref_count +1，返回相同的 block_ids"""
        for bid in block_ids:
            self.blocks[bid].ref_count += 1
        return list(block_ids)  # 返回副本

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_blocks)

    @property
    def num_used_blocks(self) -> int:
        return len(self.blocks) - len(self.free_blocks)
```

#### 2.3.2 Scheduler 与 Cache Manager 的协作流程

```python
class Sequence:
    """一个请求的完整状态"""
    def __init__(self, seq_id: int, prompt_tokens: List[int], max_tokens: int):
        self.seq_id = seq_id
        self.prompt_tokens = prompt_tokens
        self.max_tokens = max_tokens

        # 逻辑 → 物理映射
        self.block_table = BlockTable(max_blocks_per_seq=2048 // BLOCK_SIZE)

        # 状态机
        self.status = "WAITING"    # WAITING → PREFILL → DECODING → FINISHED
        self.output_tokens: List[int] = []

    @property
    def num_tokens(self) -> int:
        return len(self.prompt_tokens) + len(self.output_tokens)


class PagedAttentionCacheManager:
    """
    协调 block 分配和序列管理的核心组件

    这是 vLLM scheduler 与 KV cache 之间的桥梁
    """
    def __init__(self, num_blocks: int, block_size: int = BLOCK_SIZE):
        self.block_size = block_size
        self.allocator = BlockAllocator(num_blocks)
        # 所有活跃序列
        self.sequences: Dict[int, Sequence] = {}

    # ---------- 核心 API ----------

    def schedule_prefill(self, seq: Sequence) -> bool:
        """
        为 prefill 分配 block

        逻辑：
        1. 计算需要多少 block：ceil(prompt_tokens / BLOCK_SIZE)
        2. 检查空闲 block 是否足够
        3. 逐个分配并填入 block_table
        4. 如果不够，触发 preemption（抢占/换出）
        """
        num_blocks_needed = (
            len(seq.prompt_tokens) + self.block_size - 1
        ) // self.block_size

        if num_blocks_needed > self.allocator.num_free_blocks:
            # 空间不够：尝试抢占低优先级序列
            # 简化版：直接拒绝
            return False

        seq.status = "PREFILL"
        self.sequences[seq.seq_id] = seq

        for _ in range(num_blocks_needed):
            block_id = self.allocator.allocate()
            seq.block_table.append_block(block_id)

        return True

    def schedule_decode(self, seq: Sequence) -> bool:
        """
        Decode 阶段：每生成一个 token 可能需要新 block

        判断条件：当前序列的 token 数刚好是 BLOCK_SIZE 的整数倍
        即：最后一个 block 已满，需要新 block
        """
        if seq.num_tokens % self.block_size == 0:
            # 最后一个 block 满了
            if self.allocator.num_free_blocks == 0:
                return False  # 无法继续生成
            block_id = self.allocator.allocate()
            seq.block_table.append_block(block_id)
        return True

    def release_sequence(self, seq_id: int):
        """释放一个序列的全部 block"""
        seq = self.sequences.pop(seq_id, None)
        if seq is None:
            return
        seq.status = "FINISHED"
        for block_id in seq.block_table.block_ids:
            self.allocator.free(block_id)

    def fork_sequence(self, parent_seq_id: int, child_seq_id: int) -> Optional[Sequence]:
        """COW fork：beam search / parallel decoding 的关键"""
        parent = self.sequences.get(parent_seq_id)
        if parent is None:
            return None

        child = Sequence(
            seq_id=child_seq_id,
            prompt_tokens=parent.prompt_tokens.copy(),
            max_tokens=parent.max_tokens,
        )
        # 关键：共享 block_table（只增加 ref_count）
        child.block_table.block_ids = self.allocator.fork(
            parent.block_table.block_ids
        )
        child.output_tokens = parent.output_tokens.copy()
        child.status = "DECODING"
        self.sequences[child_seq_id] = child
        return child

    # ---------- 内存统计 ----------

    def memory_usage_report(self) -> dict:
        total = len(self.allocator.blocks)
        used = self.allocator.num_used_blocks
        free = self.allocator.num_free_blocks

        # 碎片率 = (总空闲 - 最大连续空闲) / 总空闲
        # PagedAttention 的关键优势：碎片率接近 0
        # （因为不需要连续分配）
        return {
            "total_blocks": total,
            "used_blocks": used,
            "free_blocks": free,
            "utilization": f"{used / total * 100:.1f}%",
            "fragmentation": "≈ 0% (页式管理天然消除外部碎片)",
        }
```

#### 2.3.3 Attention Kernel 中的 Block Table 寻址

这是 PagedAttention **最核心的计算**——如何从逻辑 token 位置找到物理 KV 数据：

```python
# ============================================
# PagedAttention Kernel 伪代码
# 说明如何在 attention 计算中通过 block_table 寻址
# ============================================

def paged_attention_kernel(
    Q: torch.Tensor,              # [num_q_heads, head_dim]  单个 token 的 query
    block_table: List[int],       # [num_blocks]  逻辑→物理映射
    kv_cache_k: torch.Tensor,     # [num_blocks, BLOCK_SIZE, num_kv_heads, head_dim]
    kv_cache_v: torch.Tensor,     # [num_blocks, BLOCK_SIZE, num_kv_heads, head_dim]
    context_len: int,             # 当前序列已缓存的 token 数
    scale: float,
) -> torch.Tensor:
    """
    单个 query token 对所有 cached token 的 attention

    关键：通过 block_table 将逻辑位置映射到物理 block
    """
    num_kv_heads = kv_cache_k.shape[2]
    head_dim = Q.shape[1]
    output = torch.zeros(num_kv_heads, head_dim)

    # ---- Step 1: 遍历所有逻辑位置 ----
    for logical_pos in range(context_len):
        # 逻辑位置 → (物理 block_id, block 内偏移)
        block_idx = logical_pos // BLOCK_SIZE       # 第几个 block
        offset    = logical_pos %  BLOCK_SIZE       # block 内的位置

        # !!! 关键寻址 !!!
        physical_block_id = block_table[block_idx]

        # 取出对应的 K/V
        k = kv_cache_k[physical_block_id, offset, :, :]  # [num_kv_heads, head_dim]
        v = kv_cache_v[physical_block_id, offset, :, :]  # [num_kv_heads, head_dim]

        # ---- Step 2: 计算 attention score ----
        # Q·K^T（GQA: 多个 Q head 共享一组 KV head）
        for q_head in range(num_q_heads):
            kv_head = q_head // num_heads_per_group
            score = torch.dot(Q[q_head], k[kv_head]) * scale
            # ... softmax, weighted sum ...

    return output


# ============================================
# GPU Kernel 中的高效实现（简化版伪代码）
# 真正的 vLLM kernel 在 CUDA 中实现，
# 核心优化：每个 thread block 处理一个 query head
# ============================================

# vLLM 实际的 kernel 签名（简化）：
# void paged_attention_kernel_launcher(
#     scalar_t* out,           // [num_seqs, num_heads, head_dim]
#     scalar_t* query,         // [num_seqs, num_heads, head_dim]
#     scalar_t* key_cache,     // [num_blocks, num_kv_heads, head_dim, BLOCK_SIZE]
#     scalar_t* value_cache,   // [num_blocks, num_kv_heads, head_dim, BLOCK_SIZE]
#     int* block_tables,       // [num_seqs, max_blocks_per_seq]
#     int* context_lens,       // [num_seqs]
#     int max_context_len,
#     float scale,
#     ...
# )

# 关键性能优化：
# 1. 每个 warp 处理一个 query head → 并行度 = num_seqs × num_heads
# 2. 使用 shared memory 缓存 block_table 查找
# 3. KV cache 在 HBM 中按 block 组织，支持高效的合并访问
# 4. 在线 softmax（类似 FlashAttention 的分块方法）
```

#### 2.3.4 PagedAttention 解决的四大问题

```
问题 1：外部碎片
──────────────
传统方式（连续分配）:
  [seqA: 200 slots][空闲100][seqB: 500 slots][空闲50]
  新请求需要 300 slots → 空闲合计 150，但最大连续只有 100 → 分配失败！

PagedAttention:
  空闲 block 池: [b1 b5 b8 b12 b17 ...] 共 20 个 block = 320 slots
  任意请求只需 pool 中有足够 block 数，无需连续！

  → 显存利用率提升：从 ~60% → ~95%


问题 2：动态增长
──────────────
传统方式:
  预分配 max_tokens 长度 → 大部分被浪费（实际生成往往远小于 max_tokens）
  或：预分配较短，不够时 realloc → 需要拷贝整个 cache（昂贵！）

PagedAttention:
  按需逐 block 分配 → 生成的 token 多则多分，少则少分
  新 block 追加到 block_table 末尾 → 零拷贝


问题 3：序列共享（Prefix Cache / COW Fork）
────────────────────────────────────
场景 1：多个请求共享同一个 system prompt
  请求 A: system(1000 tokens) + query_A
  请求 B: system(1000 tokens) + query_B  ← system prompt 相同！

PagedAttention 解决方案:
  system prompt 的 block table: [b1, b2, b3, ..., b63]  (1000/16=63 blocks)
  seq_A.block_table = [b1..b63] + [b_new_1]   → b1..b63 ref_count=2
  seq_B.block_table = [b1..b63] + [b_new_2]   → b1..b63 ref_count=2

  节省: 63 blocks × 1.75 MB = 110 MB per request

场景 2：Beam search
  每个 beam 分支共享父序列的 KV Cache
  COW fork: 只增加 ref_count，不复制数据
  当某个 beam 写入新 token 时，才 copy-on-write 分配新 block

场景 3：多轮对话
  每一轮的 history 可以复用上一轮的 block_table


问题 4：Preemption（抢占/换出）
────────────────────────
高优先级请求到达，但 block 池已满：
  传统：等待 → 延迟增加
  PagedAttention：swap out 低优先级请求的 block 到 CPU 内存
                 → swap in 时按 block 粒度量力而为
                 → 不需要整个序列一次性换入换出
```

---

## 3. KV Cache 显存计算器（实战工具）

### 3.1 精确公式

```
KV Cache 总量 = 2 × num_layers × max_seq_len × num_kv_heads × head_dim × bytes_per_elem

其中:
  2:          K + V 两份
  num_layers: Transformer 层数
  max_seq_len: 最大上下文长度（prefill prompt + 生成 token 数）
  num_kv_heads: GQA/MQA 中的 KV head 数
  head_dim:    每个 head 的维度
  bytes_per_elem: FP16=2, BF16=2, INT8=1, INT4=0.5, FP8=1
```

### 3.2 典型模型计算

```python
# ============================================
# KV Cache Calculator — 可运行的 Python 工具
# ============================================

def calc_kv_cache_bytes(
    num_layers: int,
    seq_len: int,
    num_kv_heads: int,
    head_dim: int,
    bytes_per_elem: int = 2,  # FP16/BF16
) -> dict:
    """计算单个请求的 KV Cache 大小"""
    per_layer = 2 * seq_len * num_kv_heads * head_dim * bytes_per_elem
    total = per_layer * num_layers

    return {
        "per_layer": per_layer,
        "per_layer_mb": per_layer / (1024 * 1024),
        "total_bytes": total,
        "total_mb": total / (1024 * 1024),
        "total_gb": total / (1024 * 1024 * 1024),
    }


# ---- 典型模型参数 ----
models = {
    "Qwen3-0.6B":    {"layers": 28, "kv_heads": 8,  "head_dim": 128},
    "Qwen3-1.7B":    {"layers": 28, "kv_heads": 8,  "head_dim": 128},
    "LLaMA-3-8B":    {"layers": 32, "kv_heads": 8,  "head_dim": 128},
    "LLaMA-3-70B":   {"layers": 80, "kv_heads": 8,  "head_dim": 128},
    "Qwen3-235B":    {"layers": 94, "kv_heads": 8,  "head_dim": 128},
    "DeepSeek-V3":   {"layers": 61, "kv_heads": 16, "head_dim": 128},  # MLA 实际更省
}

# ---- 不同序列长度下的计算 ----
for name, p in models.items():
    for seq_len in [512, 2048, 8192, 32768, 131072]:
        result = calc_kv_cache_bytes(p["layers"], seq_len, p["kv_heads"], p["head_dim"])
        print(f"{name:16s} | seq={seq_len:6d} | KV={result['total_mb']:8.1f} MB")

# 输出示例:
# Qwen3-0.6B       | seq=   512 | KV=    28.0 MB
# Qwen3-0.6B       | seq=  2048 | KV=   112.0 MB
# Qwen3-0.6B       | seq=  8192 | KV=   448.0 MB
# Qwen3-0.6B       | seq= 32768 | KV=  1792.0 MB
# Qwen3-0.6B       | seq=131072 | KV=  7168.0 MB  ← 注意！7 GB 仅 KV Cache
# LLaMA-3-8B       | seq=131072 | KV=  8192.0 MB
# DeepSeek-V3      | seq=131072 | KV= 32768.0 MB  ← 32 GB 仅 KV Cache！
```

### 3.3 并发场景下的显存压力

```
服务端场景: 同时服务 N 个请求，每个平均 4096 tokens

Qwen3-0.6B × 32 并发 × 4096 tokens:
  = 32 × 224 MB = 7,168 MB ≈ 7 GB 仅 KV Cache

LLaMA-3-8B × 32 并发 × 8192 tokens:
  = 32 × 512 MB = 16,384 MB ≈ 16 GB

还不包括:
  - 模型权重（LLaMA-3-8B ≈ 16 GB FP16）
  - 激活值（中间 tensor）
  - 显存碎片（传统方式 30-40%；PagedAttention <5%）

结论: KV Cache 是服务端推理的主要显存瓶颈。
```

### 3.4 GQA / MQA 的省显存效果

```
Qwen3-0.6B, seq_len=8192:

MHA (16 KV heads):  2 × 28 × 8192 × 16 × 128 × 2 = 896 MB
GQA (8 KV heads):   2 × 28 × 8192 ×  8 × 128 × 2 = 448 MB  ← 节省 50%
MQA (1 KV head):    2 × 28 × 8192 ×  1 × 128 × 2 =  56 MB  ← 节省 93.75%

但 MQA 可能影响模型质量，GQA 是当前主流平衡点。
```

---

## 4. KV Cache 量化：另一个维度的压缩

### 4.1 与权重量化的区别

```
权重量化（AWQ/GPTQ）:
  - 离线完成，模型文件变小
  - 推理时 dequantize 到 FP16 计算
  - 静态：量化参数不变

KV Cache 量化:
  - 在线完成，每步 decode 都要处理
  - K/V 以低精度存储，attention 计算时 dequantize
  - 动态：每步新 token 的 K/V 需要量化写入
  - 风险更高：attention 对精度敏感
```

### 4.2 ncnn 的 SDPA INT8 实现分析

ncnn 的 SDPA 层支持 INT8 KV Cache（[sdpa.cpp 中的 forward_int8](源码/ncnn-master/src/layer/sdpa.cpp#L275-L496)）：

```cpp
// 核心思想：Q、K、V、QK^T 在 INT8 下计算
// 每个张量保存一个 float scale，dequantize 时使用

// Step 1: Q 量化（per-head, per-row 动态量化）
Mat query_head_int8, query_head_int8_scales;
dynamic_quantize_2d_per_h(query_head, query_head_int8, query_head_int8_scales);
// query_head_int8[i] = round(query_head[i] * scale_q[i])
// 每行一个 scale（因为不同 token 的分布不同）

// Step 2: K 量化（全矩阵统一 scale）
Mat key_head_int8;
float key_head_int8_scale;
dynamic_quantize_2d(key_head, key_head_int8, key_head_int8_scale);

// Step 3: Q·K^T 在 INT8 域计算
int sum = 0;
for (int k = 0; k < embed_dim; k++) {
    sum += qptr[k] * kptr[k];  // INT8 × INT8 → INT32 累加
}
float sum_fp32 = sum * (1.0f / (scale_q * scale_k));  // dequantize
outptr[j] = sum_fp32 * _scale;

// Step 4: QK^T 重新量化（为下一步 QK^T·V 准备）
dynamic_quantize_2d_per_h(qk_cross_head, qk_cross_head_int8, qk_cross_head_int8_scales);

// Step 5: QK^T·V 在 INT8 域计算
// 类似的 INT8 矩阵乘法 + dequantize
```

**INT8 KV Cache 的关键权衡**：

| 方案 | 精度 | 显存 | 速度 | 适用场景 |
|------|------|------|------|---------|
| K INT8 + V FP16 | 高 | 降 ~25% | 中 | 保守选择 |
| K INT8 + V INT8 | 中 | 降 50% | 快 | 长上下文 |
| K INT4 + V INT8 | 低 | 降 62.5% | 快 | 极端显存受限 |

**为什么要 per-head / per-row 量化？**

```
统一 scale (per-tensor):      所有 token 共享一个 scale
                              问题: outlier token 会拉高 scale
                              → 大部分 token 量化精度差

per-token 量化 (per-row):    每个 token 独立 scale
                              问题: 同一 token 不同 head 分布差异大

per-head + per-token:         每个 (head, token) 独立 scale
                              → 最优精度，但 scale 存储开销大
                              → ncnn 的选择：Q 用 per-row，K/V 用 per-tensor
```

---

## 5. Prefill 与 Decode 中的内存访问模式

### 5.1 两种阶段的计算特征

```
                        Prefill                          Decode
                    ┌──────────────────┐          ┌──────────────┐
  输入              │ prompt: N tokens  │          │ 1 new token   │
  计算模式           │ 并行处理 N tokens │          │ 逐个串行生成   │
  Attention 形状    │ Q[N,d] × K[N,d]^T │          │ Q[1,d] × K[S,d]^T │
  KV Cache 操作     │ 写入（新分配）      │          │ 读取 + 追加     │
  瓶颈              │ 计算密集           │          │ 内存密集        │
  GPU 利用率        │ 高 (>80%)         │          │ 低 (<30%)      │
  Batch 效率        │ 大 batch 友好      │          │ 依赖 continuous batching │
```

### 5.2 Decode 为什么内存密集？

```
第 t 步 decode:
  计算量: 1 个 token × 28 层 × (Gemm(QKV) + SDPA + FFN)
         ≈ 28 × (3×1024×hidden + hidden×seq×head_dim + 3×hidden×3×hidden)
         ≈ 28 × (6M + 128×t + 18M) FLOPs

  内存访问量:
    读取模型权重:  模型大小 ≈ 600M × 2 bytes = 1.2 GB
    读取 KV Cache: 2 × 28 × t × 8 × 128 × 2 = 114,688 × t bytes
                  当 t=4096 → 约 449 MB

  计算强度: FLOPs / Bytes
    当 t=1024:  ~50M FLOPs / (1.2GB + 112MB) ≈ 0.04 FLOPs/Byte
    当 t=8192:  ~50M FLOPs / (1.2GB + 896MB) ≈ 0.02 FLOPs/Byte

  对比: A100 的 roofline 拐点 ≈ 10-20 FLOPs/Byte
  → Decode 阶段严重内存密集！GPU 大部分时间在等数据。
```

---

## 6. Cache Reuse 策略深入

### 6.1 Prefix Cache —— 最高收益的优化

```
场景: 多个请求共享 system prompt

传统方式:
  Request A: system(2000 tokens) + user_A(100 tokens) → 2100 tokens KV
  Request B: system(2000 tokens) + user_B(100 tokens) → 2100 tokens KV
  总 KV: 4200 tokens 的存储

Prefix Cache:
  system prompt → block_table_system = [b1, b2, ..., b125]  (2000/16=125 blocks)
  seq_A.block_table = block_table_system + [b_new_1]  → b1..b125 ref_count=2
  seq_B.block_table = block_table_system + [b_new_2]  → b1..b125 ref_count=2
  总 KV: (125 + 1 + 1) = 127 blocks × 16 tokens = 2032 tokens 的等价存储

  节省: (4200 - 2032) / 4200 ≈ 51.6% KV Cache 空间

RadixAttention（vLLM 的实际实现）:
  使用 Radix Tree（压缩前缀树）自动检测和共享前缀
  不需要用户手动指定哪些是"共享前缀"
```

### 6.2 COW Fork 在 Beam Search 中的应用

```
Beam width = 4, 每步 fork 出 4 个候选:

Step 0: 父序列 [block_a, block_b] → 生成 4 个候选 token

Step 1: fork 4 个子序列
  child_0.block_table = fork([block_a, block_b])  → ref_count: a=5, b=5
  child_1.block_table = fork([block_a, block_b])
  child_2.block_table = fork([block_a, block_b])
  child_3.block_table = fork([block_a, block_b])

  此时: 物理存储仍然是 2 个 block
        （4 个孩子共享父序列的 KV Cache）

Step 1 decode 完成:
  child_0 需要 block_c → allocate → ref_count: c=1
  child_1 需要 block_d → allocate → ref_count: d=1
  ...

  此时: 物理存储增长到 2 + 4 = 6 个 block
        如果不共享: 需要 4 × 3 = 12 个 block

Step 2: 选 top-4，释放其余的 → ref_count 递减 → 自动回收
```

### 6.3 多轮对话的 KV 复用（ncnn_llm 实现）

```cpp
// ncnn_llm_gpt.h - clone() 实现
class ncnn_llm_gpt_base_ctx : public ncnn_llm_gpt_ctx {
    std::shared_ptr<ncnn_llm_gpt_ctx> clone() const override {
        auto dst = std::make_shared<ncnn_llm_gpt_base_ctx>();
        dst->kv_cache.resize(kv_cache.size());
        for (size_t i = 0; i < kv_cache.size(); ++i) {
            dst->kv_cache[i].first = kv_cache[i].first;   // 浅拷贝 Mat
            dst->kv_cache[i].second = kv_cache[i].second;
        }
        dst->cur_token = cur_token;
        dst->position_id = position_id;
        return dst;
    }
};

// 多轮对话流程：
// Round 1: prefill("system + user1") → ctx_1
//          generate(ctx_1) → output "assistant1", ctx_1 保留完整 KV
//
// Round 2: prefill("user2", ctx_1) → ctx_2  (复用 ctx_1 的 KV!)
//          └─ ctx_2 = ctx_1.clone() → 共享已有 KV
//          └─ 只计算 "user2" 的 K/V 并追加
//          └─ 不需要重新计算 system + user1 + assistant1
//
// ncnn_llm 用的是浅拷贝（共享 ncnn::Mat 的 data 指针）
// 注意：写时不会自动 COW，prefill 时创建新的 extractor 使用旧 cache
```

---

## 7. 三种架构的最终对比

| 维度 | ncnn_llm | llama.cpp | vLLM (PagedAttention) |
|------|----------|-----------|----------------------|
| **存储模型** | 连续 Mat | Cell 池（需连续） | Block 池（无需连续） |
| **分配粒度** | 整个序列 | 单个 token | 固定 block（16 tokens） |
| **外部碎片** | 无（单请求） | 有（要求连续 cell） | 无（任意 block 可分配） |
| **序列共享** | ❌ | ✅ 多 seq 引用同 cell | ✅ COW block table |
| **Prefix Cache** | ❌ | 手动实现 | ✅ RadixAttention 自动 |
| **Beam Search** | ❌ | ✅ seq_cp | ✅ COW fork |
| **Swap/Preempt** | ❌ | ❌ | ✅ block 级换入换出 |
| **适用场景** | 端侧单用户 | 本地多序列 | 服务端高并发 |
| **Code 复杂度** | ~50 行 | ~2300 行 | ~5000+ 行 |
| **显存利用率** | 低（每步 realloc） | 中（~70-80%） | 高（~95%） |

---

## 8. 面试高频问题精讲

### Q1: 为什么 vLLM 的 PagedAttention 能提升吞吐？

```
核心逻辑链：

1. LLM serving 中 KV Cache 是显存瓶颈
   → 更长上下文、更多并发 → 显存先于算力耗尽

2. 传统连续分配导致严重外部碎片
   → 不同请求长度差异大，预分配的连续空间利用率很低
   → 实际可用空间远小于总空闲空间

3. PagedAttention 借鉴操作系统虚拟内存：
   a. 固定大小 block → 消除外部碎片
   b. Block table 映射 → 逻辑连续 ≠ 物理连续
   c. COW fork → 零拷贝共享 prefix
   d. Block 级 swap → 细粒度显存管理

4. 效果：显存利用率从 ~60% → ~95%
   → 同样显存能容纳更多并发请求
   → 更多并发 → GPU 调度更充分 → 整体吞吐提升 2-4x
```

### Q2: KV Cache 量化为什么比权重量化更难？

```
1. 动态性: KV Cache 每步 decode 都在变化，不能离线校准
2. 累积误差: 长序列中，量化误差随 attention 逐层累积
3. 分布漂移: 不同 token 的 K/V 数值分布差异大
4. 敏感度: attention softmax 对异常值敏感，量化 outlier 影响大

解决方案:
  - per-token + per-head 细粒度量化
  - 只量化 K，V 保持 FP16（折中方案）
  - 滑动窗口内保留 FP16（近邻 token 更重要）
  - SmoothQuant 式的迁移缩放因子
```

### Q3: GQA 为什么能减少 KV Cache？

```
MHA:  Q_heads=16, KV_heads=16 → Cache 存 16 份 K/V
GQA:  Q_heads=16, KV_heads=8  → Cache 存 8 份 K/V
      每 2 个 Q head 共享 1 组 KV head
      → KV Cache 减半，质量损失 <1%

原理: Q head 之间的冗余度很高，
      多个 Q head 可以用同一组 K/V 来计算 attention。
      实验表明 16→8 的分组几乎不损失质量，
      16→1 (MQA) 则有明显退化。

ncnn 的实现: ExpandDims + Tile 将 8 个 KV head 复制为 16 个
llama.cpp: 直接在 attention 计算中按 num_heads_per_group 索引
```

---

## 9. 本章学习清单

- [ ] 能画出 KV Cache 的数据流图（从 prefill 写入到 decode 追加）
- [ ] 能写出 KV Cache 显存公式并手算至少 3 个模型的显存占用
- [ ] 能解释三种 KV Cache 架构（ncnn_llm / llama.cpp / vLLM）的设计差异
- [ ] 能画 PagedAttention 的 block table 映射图
- [ ] 能解释 COW fork 在 beam search 和 prefix cache 中的工作原理
- [ ] 能说明 KV Cache 量化与权重量化的五个关键差异
- [ ] 能解释为什么 decode 阶段是 memory-bound 而 prefill 是 compute-bound
- [ ] 能用自己的话写一段 PagedAttention kernel 伪代码

---

## 10. 进一步阅读

- [vLLM Paper: Efficient Memory Management for Large Language Model Serving with PagedAttention (SOSP'23)](https://arxiv.org/abs/2309.06180)
- [Splitwise: Efficient Generative LLM Inference Using Phase Splitting (ISCA'24)](https://arxiv.org/abs/2311.18677)
- llama.cpp KV cache 实现：[llama-kv-cache.h](源码/llama.cpp-master/src/llama-kv-cache.h) / [llama-kv-cache.cpp](源码/llama.cpp-master/src/llama-kv-cache.cpp)
- ncnn SDPA 实现：[sdpa.cpp](源码/ncnn-master/src/layer/sdpa.cpp)
- ncnn_llm generate 循环：[ncnn_llm_gpt.cpp](源码/ncnn_llm-main/src/ncnn_llm_gpt.cpp)

---

*下一篇: [05_execution_scheduler.md](05_execution_scheduler.md) — Scheduler、Continuous Batching 与请求调度*
