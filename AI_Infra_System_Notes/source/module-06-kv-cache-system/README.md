# Module 6: KV Cache 系统——大模型的"记忆中枢"

> 如果说 GPU 是 LLM 的"大脑"，KV Cache 就是它的"海马体"——负责记住每一个说过的话、每一个思考过的上下文。没有它，每生成一个新 token 都要把整段对话从头到尾重新读一遍。有了它，模型只需关注"最新的一行字"和"历史的模糊印象"。但它也是整个推理系统最大的显存黑洞。

---

## 📋 学习目标

- [ ] 能写出 KV Cache 的显存公式，手算任意模型的 KV 占用
- [ ] 能解释 ncnn_llm 的连续缓存实现（memcpy concat）
- [ ] 能解释 llama.cpp 的 cell 池 + find_slot() 机制
- [ ] 能手写 PagedAttention 的 Block Table 寻址伪代码
- [ ] 能说明 COW fork 在 beam search 和 prefix cache 中的应用
- [ ] 能对比三种架构在 10 个维度上的差异

---

## 1. 开篇：为什么 KV Cache 是 LLM 推理的"心脏"？

先讲一个故事。

假设有一个特别喜欢重复的教授，他每个学期都教同一门课。但他有个毛病——每次在课堂上讲到一句话之前，他必须把**从学期第一天到现在的所有话**在心里默念一遍，才敢说出下一句。第一堂课还很轻松，第十堂课时他要先回忆整整九堂课的内容。到期末，他每说一句话都要先花半小时回忆。

这就是**没有 KV Cache 的 LLM 推理**。

现在换一种做法：教授每天下课后，把当天讲的要点记在一本笔记本上。第二天上课前翻一翻笔记本——只需要几秒钟。笔记本就是 KV Cache。

```
没有 KV Cache: 第 t 步要重算全部 t 个 token 的 Key 和 Value → O(t²) 计算
有 KV Cache:    第 t 步只算新 token，历史的 Key/Value 从缓存里读 → O(t) 读内存

KV Cache 做了一件什么事？用空间换时间。
用额外的显存，避免了 t² 级别的重复计算。
```

但这个策略有一个致命弱点——**显存不是无限的**。当上下文越来越长，KV Cache 会膨胀到占满整个显存。在服务端，它甚至比模型权重本身还占地方。

---

## 2. KV Cache 到底存了什么？——用真实的数字说话

### 2.1 物理上它是啥？

每一层 Transformer 的注意力机制都需要两组矩阵：Key 和 Value。

```
Key:   模型拿 Query 去和 Key 做点积——"我的问题和这段历史相关吗？"
Value: 根据注意力分数，从 Value 里提取相关信息——"这段历史说了什么？"

Qwen3-0.6B 为例:
  28 层 × 每层 2 个矩阵 (K + V)
  K 的形状: [8 个 KV head, seq_len, 128 维]
  V 的形状: [8 个 KV head, seq_len, 128 维]
  每个元素: FP16 = 2 bytes

当 seq_len = 1024 时:
  KV = 28 × 2 × 8 × 1024 × 128 × 2 = 112 MB
当 seq_len = 4096 时:
  KV = 28 × 2 × 8 × 4096 × 128 × 2 = 448 MB  ← 已经超过模型权重(600MB)的 75%！
当 seq_len = 32768 时:
  KV = 28 × 2 × 8 × 32768 × 128 × 2 = 3.58 GB ← 比模型大 6 倍！
```

> ⚠️ **深刻警告**：KV Cache 的大小随上下文长度**线性增长**。在长上下文推理中，它才是真正的显存杀手——不是模型参数，不是激活值，就是这些缓存的 K 和 V。

### 2.2 通用显存公式

```
KV_bytes = 2 × L × S × H_kv × D × B

L:    num_layers      层数（Qwen3: 28）
S:    seq_len         上下文长度（你写的 prompt + 模型生成的所有 token）
H_kv: num_kv_heads    KV 头数（GQA 可以减少这个数）
D:    head_dim        每个头的维度（通常 128）
B:    bytes_per_elem  FP16=2, INT8=1, INT4=0.5

2:    K 和 V 各一份
```

**⚠️ 并发场景下乘以请求数！**

```
32 个并发用户 × 平均 2048 token × 每请求 224 MB = 7 GB 仅 KV Cache
再加上模型权重、激活值 → 一张 80GB 的 A100 瞬间被吃掉一半以上
```

### 2.3 GQA 是怎么省 KV Cache 的？

GQA（Grouped-Query Attention）是现代 LLM 最重要的显存优化之一。原理非常简单：

```
MHA (全多头):  16 个 Q head × 16 个 KV head → KV 存 16 份
GQA (分组):    16 个 Q head × 8 个 KV head  → KV 存 8 份，省 50%
MQA (极致):    16 个 Q head × 1 个 KV head  → KV 存 1 份，省 93.75%

直觉: 多个 Q head 之间本来就在关注差不多的东西。
      与其给每个 Q 配一个独立的 KV，不如让它们"拼车"。
      研究发现: 16→8 几乎不损失质量，16→1 有明显退化。
      GQA 是当前最佳的"省钱/保质"平衡点。
```

---

## 3. 三大架构——KV Cache 管理的三种哲学

业界有三种管理 KV Cache 的方式。它们不是"谁比谁好"的关系，而是**在不同约束下的不同答案**。

### 3.1 范式一：ncnn_llm 的"老实人"方案——完整拷贝 🔬

**一句话概括**：每次来新 token，把整个旧 cache 复制一份，拼上新 token，算完了用新的换掉旧的。

```cpp
// ncnn sdpa.cpp:69-87 — 核心逻辑
// 这就是一个朴实无华的 memcpy:
int past_seqlen = kv_cache ? past_key.h : 0;   // 旧的有多少
int dst_seqlen = past_seqlen + cur_seqlen;      // 新的有多少

Mat key(dst_seqlen);  // 新分配一块更大的内存
for (int q = 0; q < num_group; q++) {
    memcpy(key.row(0),           past_key_head, past_seqlen * sizeof(float));
    memcpy(key.row(past_seqlen), cur_key_head,  cur_seqlen  * sizeof(float));
}
// 现在是旧 + 新的完整 key，拿去算 attention
// Value 同理处理
```

**设计哲学**：极简主义。50 行代码搞定。内存连续，cache 友好。代价也很明显——每步 decode 都要 allocate + memcpy + free，生成 1000 个 token 就要做 1000 次全量拷贝。

> 💡 **适合谁**：端侧单用户。手机上一次只有一个对话，cache 再大也大不到哪去。简单可靠比省内存更重要。

### 3.2 范式二：llama.cpp 的"泊车员"方案——Cell 池管理 🔬

**一句话概括**：把 KV Cache 想象成一个停车场，每个车位 (cell) 停一个 token。多辆车 (多序列) 共享停车场，来一辆停一辆，走了就腾出车位。

```cpp
// llama-kv-cache.h — 核心数据结构
class llama_kv_cache {
    // 物理存储: 每层一个巨大的 K/V tensor
    // 逻辑管理: 一个 cell 数组，每个 cell 记录:
    //   - 这个位置存的是哪个 token？
    //   - 属于哪个序列？
    //   - 有几个序列在共享它？

    llama_kv_cells_vec v_cells;  // 停车场的车位表
    vector<uint32_t> v_heads;    // 每个序列的"写入指针"

    // 核心操作: 找车位！
    slot_info find_slot(ubatch, cont) {
        // 从 v_heads 开始扫描
        // 找 n_tokens 个连续空闲 cell
        // 找不到？触发 cache shift（丢弃最旧的 token）
    }
};
```

**设计哲学**：比 ncnn_llm 多了一层抽象。不要求"整个序列占一段连续内存"，但要求"每次分配必须是连续的 cell"。这带来几个好处：

- **多序列共存**：可以同时服务多个对话，各自占有不同段的 cell
- **序列操作**：可以删除一个序列（用户取消了）、复制一个序列（beam search）、合并共享的 cell
- **Cache Shift**：当 cell 快用完时，可以自动丢弃最旧的 token（配合 Sliding Window Attention 使用）

但它有一个致命缺陷——**外部碎片**。不同序列长度不一，释放后留下不连续的空隙。就像停车场里，虽然总共空了 50 个车位，但分散在 10 个角落，每个角落只有 5 个——一辆 8 座商务车（需要 8 个连续 cell）就是停不进来。

> 💡 **适合谁**：本地多序列推理。需要比 ncnn_llm 更强的并发管理，但又不至于像 vLLM 那么重。

### 3.3 范式三：vLLM 的"操作系统"方案——PagedAttention 🔬⭐

**一句话概括**：借鉴操作系统虚拟内存的思想。逻辑上连续的 token 序列，在物理上可以存放在任意位置的 block 中。通过一张映射表（Block Table）来翻译。

这是本模块的重头戏——PagedAttention 是整个 AI Infra 领域近年来最重要的系统创新之一。

#### 你家的书架 vs 图书馆的索书号

想象你有 100 本书。你家的做法是——买一个 100 本书容量的大书架，按顺序排列。这很简单，但当你想插入一本新书在第 50 本的位置时，需要把后面 50 本全部挪一遍。而且如果你只有一个小书架，最大容量 80 本，那 100 本根本放不下——哪怕你实际大部分空间都空着。

图书馆怎么做？每本书有一个索书号。你在电脑上查到索书号，去对应的书架拿书。书在物理上不需要连续排列——它们可以散落在图书馆的各个角落。你只需要一张"索书号→物理位置"的映射表。

PagedAttention 就是**图书馆的方案**：

```
逻辑序列: [tok₀...tok₁₅][tok₁₆...tok₃₁][tok₃₂...tok₄₇]
              ↓              ↓              ↓
物理 Block:  Block 7       Block 2       Block 9

Block Table = [7, 2, 9]
翻译规则: token i → block_table[i / 16] 的第 (i % 16) 个位置
```

#### 核心数据结构——完整可运行的实现

```python
BLOCK_SIZE = 16  # 每个物理 block 存 16 个 token

# 一个 KV block——物理存储的最小单元
class KVCacheBlock:
    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0  # 多少序列在引用这个 block


# 映射表——逻辑到物理的"翻译官"
class BlockTable:
    def __init__(self, max_blocks):
        self.block_ids = []  # [7, 2, 9, 14, ...]

    def append(self, block_id):
        self.block_ids.append(block_id)

    def translate(self, logical_pos):
        """逻辑 token 位置 → 物理 (block_id, offset)"""
        block_idx = logical_pos // BLOCK_SIZE
        offset    = logical_pos % BLOCK_SIZE
        return self.block_ids[block_idx], offset


# 全局管理器——整个系统的"后勤部长"
class BlockAllocator:
    def __init__(self, num_blocks):
        self.blocks = [KVCacheBlock(i) for i in range(num_blocks)]
        self.free_list = list(range(num_blocks))

    def allocate(self):
        """借一个 block。没有了？返回 None——OOM！"""
        if not self.free_list: return None
        bid = self.free_list.pop()
        self.blocks[bid].ref_count = 1
        return bid

    def free(self, block_id):
        """还回一个 block。ref_count 减到 0 才能真正回收"""
        self.blocks[block_id].ref_count -= 1
        if self.blocks[block_id].ref_count == 0:
            self.free_list.append(block_id)  # 回归空闲池

    def fork(self, block_ids):
        """写时复制: 所有 ref_count +1，但不复制实际数据"""
        for bid in block_ids:
            self.blocks[bid].ref_count += 1
        return list(block_ids)  # 返回同样的列表——零拷贝！
```

#### 寻址魔法——Attention Kernel 中的关键计算

```python
def paged_attention(Q, block_table, kv_cache_k, kv_cache_v, context_len):
    """
    这是 PagedAttention 最核心的循环
    每一行代码都在回答同一个问题:
    "逻辑位置 i 的 K/V 在哪里？"
    """
    output = 0
    for logical_pos in range(context_len):
        # 翻译！翻译！翻译！
        block_idx = logical_pos // BLOCK_SIZE   # 第几个 block
        offset    = logical_pos %  BLOCK_SIZE   # block 内的第几个位置
        physical_block = block_table[block_idx]  # 查表！

        # 从物理 block 中取出对应的 K 和 V
        k = kv_cache_k[physical_block, offset, :, :]
        v = kv_cache_v[physical_block, offset, :, :]

        # 计算 attention...
        score = dot(Q, k) * scale
        # softmax, weighted sum...

    return output
```

> 💡 **为什么这个循环不能直接写成 `for i in range(S): k = cache[i]`？** 因为 PagedAttention 中 `cache[i]` 不是连续的。第 16 个 token 可能在第 7 个 block，第 17 个 token 可能在第 2 个 block。必须通过 block_table 做翻译。这多了一层间接寻址，但换来的是物理内存的彻底解放。

#### PagedAttention 解决的四大痛点

**痛点 1：外部碎片——"总空闲够，但没有连续空间"**

```
传统连续分配:
  [A:200 slots][空闲:100][B:500 slots][空闲:50]
  空闲共计 150，但最大连续只有 100
  新请求 C 需要 120 slots → 放不下！尽管总空间够！

PagedAttention:
  空闲 block 池: [b3, b8, b11, b15, ..., b27] 共 20 个 = 320 slots
  任意请求只需空闲 block 数够就行，不需要连续
  C 需要 120/16 = 8 个 block → 池里有 20 个 → 直接分配！
  
  → 显存利用率: 传统 ~60% → PagedAttention ~95%
```

**痛点 2：动态增长——"我不知道用户会说多少话"**

```
用户的 prompt 可能 50 字，也可能 5000 字。
模型要生成多少字？完全无法预测。

传统: 预分配 max_tokens 长度 → 大部分被浪费
      或者预分配短一些 → 满了 realloc → 拷贝整个 KV Cache（昂贵！）

PagedAttention: 按需逐 block 分配
  生成第 17 个 token: "哦，上一个 block 满了，再分配一个"
  → 零预判，零浪费，零拷贝
```

**痛点 3：序列共享——"为什么同样的 system prompt 要存 100 份？"**

```
100 个用户请求同一个 AI 助手，system prompt 完全相同。

传统: 100 × system_prompt_KV = 100 份
PagedAttention: 
  system prompt 的 blocks → ref_count = 100
  100 个序列共享同一份物理 KV blocks！
  
  省了多少？如果 system prompt = 1000 tokens:
    blocks = 1000/16 = 63 blocks
    每 block ≈ 1.75 MB → 63 × 1.75 = 110 MB
    100 个请求省了 99 × 110 MB ≈ 10.9 GB！
```

**痛点 4：Beam Search——"四个候选方向，但前三步是一样的"**

```
Beam Search (beam_width=4):

Step 0: 父序列 block_table = [b1, b2]
Step 1: fork 出 4 个分支
  child_0: fork([b1, b2]) → b1,b2 ref_count = 5
  child_1: fork([b1, b2]) → 共享！零拷贝！
  child_2: fork([b1, b2])
  child_3: fork([b1, b2])
  物理上仍然只有 2 个 block

Step 2: 每个分支自己生成新 token
  child_0 追加 b_new_0 → 独立分配
  child_1 追加 b_new_1 → 独立分配
  物理: 2 + 4 = 6 blocks
  无共享: 4 × 3 = 12 blocks → 省 50%

当某个分支被淘汰时:
  free 它的独占 blocks → ref_count 归零 → 自动回收
  公共 blocks ref_count 减少 → 继续保持
```

#### 三架构终极对比

| 维度 | ncnn_llm | llama.cpp | vLLM PagedAttention |
|------|----------|-----------|---------------------|
| **核心比喻** | 每次重新誊写笔记本 | 停车场泊车 | 图书馆索书号 |
| **存储模型** | 连续 Mat | Cell 池 (需连续) | Block 池 (无需连续) |
| **分配粒度** | 整个序列 | 单个 token | 16-token block |
| **外部碎片** | 无 (单请求) | 有 | **无** ← 杀手锏 |
| **序列共享** | ❌ | ✅ | ✅ COW 零拷贝 |
| **Prefix Cache** | ❌ | 手动实现 | ✅ 自动检测共享 |
| **Beam Search** | ❌ | ✅ seq_cp | ✅ COW fork |
| **换入换出** | ❌ | ❌ | ✅ Block 级 swap |
| **适用场景** | 手机端单用户 | 本地多序列 | **云端高并发** |
| **代码量** | ~50 行 | ~2300 行 | ~5000+ 行 |
| **显存利用率** | 低 (每步重分配) | 中 (~70-80%) | **高 (~95%)** |

---

## 4. KV Cache 量化——给"记忆"减减肥

KV Cache 量化是解决长上下文显存问题的第二道防线。它的做法是：把 K 和 V 用 INT8（甚至 INT4）存储，attention 计算时动态反量化回 FP16。

**但它比权重量化难得多——为什么？**

1. **KV Cache 是活的**：每步 decode 都有新 token 的 K/V 进来，你必须"在线"量化——不能像权重那样离线校准
2. **误差会累积**：量化误差随着 KV Cache 的增长一层一层叠加。第 28 层的 attention 输入是前面 27 层量化误差的累积
3. **Attention 对精度敏感**：softmax 对异常值极度敏感。一个被量化搞坏了的 outlier 可能让整个注意力分布崩溃
4. **不同层的分布截然不同**：浅层 KV 值域窄，深层宽——一个 scale 管所有层是不可能的

**ncnn 的 INT8 SDPA 策略**：
- Q 用 per-row 量化（不同 token 独立 scale，保护 outlier）
- K/V 用 per-tensor 量化（balance 精度和开销）
- QK^T 和 QK^T×V 都在 INT8 域计算，在线 dequantize

---

## 🛠️ 动手练习

1. **显存计算器**：写一个 Python 函数，输入模型参数（L, S, H_kv, D, dtype），输出 KV Cache 显存占用。测试 5 种模型 × 4 种序列长度。

2. **PagedAttention 最小实现**：用 Python 实现 KVCacheBlock + BlockTable + BlockAllocator 三个类，写一个简单的 attention kernel 模拟，通过 block_table 寻址取出 K/V。

3. **Prefix Cache 收益**：10 个请求共享 2000 token 的 system prompt，BLOCK_SIZE=16。计算 PagedAttention 比传统方案省了多少 blocks。

4. **源码走读**：打开 ncnn sdpa.cpp，定位 KV Cache concat 的 memcpy 位置（第 69-87 行前后）。用注释标注新旧 K/V 的拼接逻辑。

---

*下一模块: [Module 7: 调度系统——让 GPU 不再空转](../module-07-scheduling/README.md)*
