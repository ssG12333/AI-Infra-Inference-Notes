# Module 6: KV Cache 系统——三种架构与 PagedAttention

> KV Cache 是 LLM 推理的"心脏"——它既让自回归推理成为可能，也是最大的显存瓶颈。本模块从源码级拆解三种实现范式。

---

## 📋 学习目标

- [ ] 能写出 KV Cache 的显存公式，手算任意模型的 KV 占用
- [ ] 能解释 ncnn_llm 的连续缓存实现（memcpy concat）
- [ ] 能解释 llama.cpp 的 cell 池 + find_slot() 机制
- [ ] 能手写 PagedAttention 的 Block Table 寻址伪代码
- [ ] 能说明 COW fork 在 beam search 和 prefix cache 中的应用
- [ ] 能对比三种架构在 10 个维度上的差异

---

## 1. KV Cache 基础

### 1.1 为什么必须要有 KV Cache？

```
没有 KV Cache:  第 t 步要重算全部 t 个 token 的 K/V → O(t²)
有 KV Cache:    第 t 步只算 1 个 token 的 K/V → O(t)

KV Cache 用内存换计算——省 O(n²) 计算，多占 O(n×d) 内存。
```

### 1.2 存什么？

```
每层 Attention 保存一对张量:
  K: [num_kv_heads, seq_len, head_dim]
  V: [num_kv_heads, seq_len, head_dim]

Qwen3-0.6B: 28 层 × 2 (K+V) × 8 heads × seq × 128 dim × 2 bytes
  seq=1024:  28 × 2 × 8 × 1024 × 128 × 2 = 112 MB
  seq=4096:  448 MB
  seq=32768: 3.6 GB  ← 超过模型权重大小！
```

### 1.3 显存公式

```
KV_bytes = 2 × num_layers × seq_len × num_kv_heads × head_dim × bytes_per_elem

关键变量:
  num_layers:   层数越多，KV 越大 (线性)
  seq_len:      上下文越长，KV 越大 (线性)
  num_kv_heads: GQA/MQA 减少 KV head (省 50%-93.75%)
  bytes_per_elem: FP16=2, INT8=1, INT4=0.5

⚠️ 并发场景: 上述 × num_requests!
  32 个并发 × 224 MB = 7 GB 仅 KV Cache
```

---

## 2. 范式一：ncnn_llm 连续缓存 🔬

### 数据结构

```cpp
// ncnn_llm_base.h:14
using KVCache = std::vector<std::pair<ncnn::Mat, ncnn::Mat>>;
// KVCache.size() == attn_cnt == 28
// 每层: K [128, seq, 8]  +  V [128, seq, 8]
```

### SDPA 层中的 concat 实现

```cpp
// ncnn sdpa.cpp:69-87 — 核心逻辑
const int past_seqlen = past_key.h;   // 已有缓存长度
const int dst_seqlen = past_seqlen + cur_seqlen;  // 拼接后总长

Mat key = cur_key;
if (past_seqlen > 0) {
    key.create(embed_dim, dst_seqlen, num_group, 4u, allocator);

    #pragma omp parallel  // OpenMP 并行加速
    for (int q = 0; q < num_group; q++) {
        // 先拷贝旧 KV → 再追加新 token
        memcpy(key.row(0),           past_key_head,
               embed_dim * past_seqlen * sizeof(float));
        memcpy(key.row(past_seqlen), cur_key_head,
               embed_dim * cur_seqlen  * sizeof(float));
    }
}
// Value 同理
```

### 特点

```
优点: 实现极简 (~50 行), 连续内存 cache 友好
缺点: 每步 decode 全量 memcpy + 每次重新分配
      单请求专用, 无法共享 prefix
适合: 端侧单用户场景
```

---

## 3. 范式二：llama.cpp Cell 管理 🔬

### 核心思想

将 KV Cache 抽象为 **cell 池**，每个 cell = 1 个 token 位置。通过 `find_slot()` 在池中查找可用空间。

```cpp
// llama-kv-cache.h — 核心数据结构
class llama_kv_cache {
    // 物理存储: 每层一个大的 K/V tensor
    struct kv_layer {
        ggml_tensor * k;  // [n_embd_head_k, n_head_k, kv_size]
        ggml_tensor * v;
    };

    // Cell 元数据: 每个 cell 记录位置、所属序列、引用计数
    llama_kv_cells_vec v_cells;  // [n_stream][kv_size]

    // 写入头指针 (类比 OS brk)
    std::vector<uint32_t> v_heads;

    // 序列→Stream 映射
    std::vector<uint32_t> seq_to_stream;
};
```

### find_slot() 算法

```
从 v_heads[stream] 开始扫描 cell 数组:
  找 n_tokens 个连续空闲 cell
  找不到 → 尝试 cache shift (丢弃 SWA 外的旧 token)
  还找不到 → 分配失败

Cache Shift:
  shift 前: [t₀ t₁ t₂ t₃ t₄ t₅ t₆ t₇] head=8 (满了!)
  shift 后: [t₂ t₃ t₄ t₅ t₆ t₇ ·  ·]  head=6 (释放 t₀ t₁)
```

### 相比 ncnn_llm 的进步

| 特性 | ncnn_llm | llama.cpp |
|------|----------|-----------|
| 多序列 | ❌ | ✅ 多 stream 管理 |
| 序列操作 | ❌ | ✅ rm/cp/keep/add |
| Cell 共享 | ❌ | ✅ 多 seq 同 cell |
| Code | ~50 行 | ~2300 行 |
| 碎片 | 无 (单请求) | 外部碎片 (要求连续) |

---

## 4. 范式三：vLLM PagedAttention 🔬— 核心重点

### 4.1 核心数据结构

```python
BLOCK_SIZE = 16  # 每个 block 存 16 个 token

# 一个物理 KV block
class KVCacheBlock:
    block_id: int
    ref_count: int = 0        # 引用计数 (>1 = 多序列共享)
    # k_layers[il]: [BLOCK_SIZE, num_kv_heads, head_dim]
    # v_layers[il]: [BLOCK_SIZE, num_kv_heads, head_dim]

# 逻辑→物理映射表
class BlockTable:
    block_ids: List[int]    # [7, 2, 9, ...]
    # 逻辑 token 15 → block 0 offset 15; token 16 → block 1 offset 0

# 全局 block 池
class BlockAllocator:
    blocks: List[KVCacheBlock]
    free_blocks: List[int]
    def allocate() → int     # 取一个空闲 block
    def free(block_id)        # ref_count--, 归零则回收
    def fork(block_ids)       # COW: ref_count++ 全部
```

### 4.2 Attention Kernel 中的寻址

```python
# PagedAttention 的核心寻址逻辑
def paged_attention(Q, block_table, kv_cache_k, kv_cache_v, context_len):
    """
    Q: [num_q_heads, head_dim]  ← 单个 query token
    block_table: [num_blocks]   ← 逻辑→物理映射
    """
    for logical_pos in range(context_len):
        # 关键寻址: 逻辑位置 → (物理 block_id, block 内偏移)
        block_idx = logical_pos // BLOCK_SIZE   # 第几个 block
        offset    = logical_pos %  BLOCK_SIZE   # block 内位置
        physical_block = block_table[block_idx]  # ← 页表查找!

        k = kv_cache_k[physical_block, offset, :, :]  # 取出 K
        v = kv_cache_v[physical_block, offset, :, :]  # 取出 V
        # 计算 attention score, weighted sum...
```

### 4.3 PagedAttention 解决的四大问题

```
问题 1: 外部碎片
  传统: [A:200][空闲100][B:500][空闲50] → 空闲 150 但最大连续 100
  Page: 空闲 block 池有 20 blocks → 任意请求任意分配

问题 2: 动态增长
  传统: 预分配 max_tokens → 大量浪费
  Page: 按需逐 block 分配 → 零浪费

问题 3: 序列共享 (Prefix Cache + Beam Search)
  seq_A: [sys blocks] + [new_A]  → sys blocks ref_count=2
  seq_B: [sys blocks] + [new_B]  → sys blocks ref_count=2
  → 物理上只有 1 份! COW fork 零拷贝

问题 4: Preemption (抢占)
  Block 级别 swap out → 细粒度换入换出
  → 不需要整个序列一次性换出
```

### 4.4 COW Fork 详解

```
Beam Search beam_width=4:

Step 0: 父序列 [b1, b2]  → ref_count: b1=1, b2=1

Step 1: fork 4 个子序列
  child_0.block_table = fork([b1, b2])  → ref_count: b1=5, b2=5
  child_1.block_table = fork([b1, b2])
  child_2.block_table = fork([b1, b2])
  child_3.block_table = fork([b1, b2])
  物理存储: 仍是 2 个 block! (4 个子序列共享)

Step 1 Decode 完成:
  child_0 追加 block_c → ref_count: c=1
  child_1 追加 block_d → ref_count: d=1
  物理存储: 2 + 4 = 6 blocks
  不共享需要: 4 × 3 = 12 blocks → 省 50%
```

---

## 5. 三架构终极对比

| 维度 | ncnn_llm | llama.cpp | vLLM PagedAttention |
|------|----------|-----------|---------------------|
| **存储模型** | 连续 Mat | Cell 池 (需连续) | Block 池 (无需连续) |
| **分配粒度** | 整个序列 | 单个 token | 固定 16-token block |
| **外部碎片** | 无 (单请求) | 有 | **无** ← 关键优势 |
| **序列共享** | ❌ | ✅ | ✅ COW |
| **Prefix Cache** | ❌ | 手动 | ✅ RadixAttention 自动 |
| **Beam Search** | ❌ | ✅ seq_cp | ✅ COW fork |
| **Swap/Preempt** | ❌ | ❌ | ✅ Block 级 |
| **适用场景** | 端侧单用户 | 本地多序列 | 服务端高并发 |
| **复杂度** | ~50 行 | ~2300 行 | ~5000+ 行 |
| **显存利用率** | 低 | 中 (~70-80%) | **高 (~95%)** |

---

## 6. KV Cache 量化

```
策略:
  K INT8 + V FP16:  降 25%, 保守安全
  K INT8 + V INT8:  降 50%, 主流方案
  K INT4 + V INT8:  降 62.5%, 极端场景

与权重量化的关键差异:
  1. 动态性: 每步 decode 都在变化
  2. 累积误差: 随序列增长逐渐放大
  3. 分布漂移: 不同 token 数值分布差异大
  4. 敏感度: attention softmax 对 outlier 敏感

ncnn SDPA INT8 策略:
  Q: per-row 量化 (不同 query token 独立 scale)
  K/V: per-tensor 量化 (平衡精度和开销)
  QK^T 和 PV 均在 INT8 域计算, 在线 dequantize
```

---

## 🛠️ 动手练习

1. **显存计算器**: 写一个 Python 函数 `calc_kv_cache(layers, seq_len, kv_heads, head_dim, dtype)`，计算 5 个模型的 KV Cache。

2. **PagedAttention 最小实现**: 用 Python 实现 `KVCacheBlock`, `BlockTable`, `BlockAllocator`, 并通过 block table 寻址取出 K/V。

3. **Prefix Cache 收益计算**: 10 个请求共享 2000-token system prompt，计算 PagedAttention 节省的 block 数。

4. **源码走读**: 在 ncnn sdpa.cpp 中找到 memcpy concat 的位置，标注关键行号。

---

## 📚 延伸阅读

- [04_memory_kv_cache.md](../../docs/04_memory_kv_cache.md) — 系统笔记完整版 (1059 行)
- [ncnn sdpa.cpp](../../../../源码/ncnn-master/src/layer/sdpa.cpp) — ncnn SDPA 层
- [llama-kv-cache.h](../../../../源码/llama.cpp-master/src/llama-kv-cache.h) — llama.cpp KV Cache
- [vLLM SOSP'23 Paper](https://arxiv.org/abs/2309.06180)

---

*下一模块: [Module 7: 调度系统](../module-07-scheduling/README.md)*
