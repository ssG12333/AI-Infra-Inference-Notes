# NCNN 大模型推理全流程详解

> 基于本项目实际代码（`ncnn_llm-main`），从预处理到输出的每一个阶段、每一行关键代码、每一个数据流转。

---

## 目录

1. [总体架构](#1-总体架构)
2. [Phase 0: model.json 配置加载](#2-phase-0-modeljson-配置加载)
3. [Phase 1: Prompt Template（对话模板）](#3-phase-1-prompt-template对话模板)
4. [Phase 2: Tokenization（BPE 分词）](#4-phase-2-tokenizationbpe-分词)
5. [Phase 3: RoPE 位置编码](#5-phase-3-rope-位置编码)
6. [Phase 4: Embedding（词嵌入）](#6-phase-4-embedding词嵌入)
7. [Phase 5: Decoder 解码器前向](#7-phase-5-decoder-解码器前向)
8. [Phase 6: Projection（投影输出）](#8-phase-6-projection投影输出)
9. [Phase 7: Sampling（采样策略）](#9-phase-7-sampling采样策略)
10. [Phase 8: Token to Text（解码回文本）](#10-phase-8-token-to-text解码回文本)
11. [完整自回归生成循环](#11-完整自回归生成循环)
12. [KV Cache 深入解析](#12-kv-cache-深入解析)
13. [多轮对话续聊机制](#13-多轮对话续聊机制)
14. [附录：数据形状速查表](#14-附录数据形状速查表)
15. [Prefill vs Decode 性能分析](#15-prefill-vs-decode-性能分析)
16. [INT8 量化推理](#16-int8-量化推理)
17. [Vulkan GPU 加速与 FlashAttention](#17-vulkan-gpu-加速与-flashattention)
18. [Tool Calling 完整生命周期](#18-tool-calling-完整生命周期)
19. [Vision 多模态推理](#19-vision-多模态推理)

---

## 1. 总体架构

ncnn_llm 把大模型拆成 **3 个独立的 ncnn::Net**，不是一张大图：

```
+---------------------+     +----------------------+     +---------------------+
|   embed_net         |     |    decoder_net       |     |   proj_out_net      |
|                     |     |                      |     |                     |
| Token IDs -> Embed  |---->| Embed -> Hidden State|---->| Hidden -> Logits    |
| (查表: Gather)      |     | (N 层 Transformer)   |     | (Linear: dim->vocab)|
+---------------------+     +----------------------+     +---------------------+
                                    |
                             KV Cache 提取/注入
```

**对应文件**（以 Qwen3 为例）：

| 网络 | param 文件 | bin 文件 | 作用 |
|------|-----------|---------|------|
| embed_net | qwen3_embed_token.ncnn.param | qwen3_embed_token.ncnn.bin | token ID -> embedding 向量 |
| decoder_net | qwen3_decoder.ncnn.param | qwen3_decoder.ncnn.bin | N 层 Transformer 推理 |
| proj_out_net | qwen3_proj_out.ncnn.param | qwen3_embed_token.ncnn.bin | hidden state -> 词表 logits |

**为什么要拆开？**

- KV Cache 需要从 decoder 中间层提取 K/V，单网络不好操作
- embedding 和 projection 参数可以复用（某些架构中两者共享权重）
- 内存峰值更低，可以按需加载

---

## 2. Phase 0: model.json 配置加载

**文件**: `ncnn_llm_gpt.cpp` 第 17-245 行（构造函数）

模型不是"硬编码"的，所有关键参数都从 model.json 动态读取。

### model.json 结构

你的 Qwen3 模型配置文件位于 `LLM-model/model.json`，包含以下四大块：

**1. 模型参数路径 (params)**
- 指定 embed_token、decoder、proj_out 三个网络的 .param 和 .bin 文件路径

**2. Tokenizer 配置 (tokenizer)**
- `type`: "bbpe" (Byte-level BPE)
- `vocab_file`: "vocab.txt" - 词表文件，每行一个 token，行号即 token ID
- `merges_file`: "merges.txt" - BPE 合并规则
- `eos`: 结束 token，用于判断生成结束
- `additional_special_tokens`: 特殊 token 列表（如 system/user/assistant 分隔符、工具调用标记、thinking 标记等）

**3. 模型参数 (setting)**
- `attn_cnt`: 28 - Attention 层数
- `rope`: 位置编码配置
  - `type`: "RoPE"（旋转位置编码）
  - `rope_head_dim`: 128 - 每个 head 的 RoPE 维度
  - `rope_theta`: 100000.0 - RoPE 基频

**4. 功能配置 (functions)**
- `type`: "tool_call" - 启用工具调用
- `tool_call_id`: 工具调用开始标记的 token ID
- `tool_call_end_id`: 工具调用结束标记的 token ID

### 构造函数加载流程

```cpp
// 1. 读取 model.json
json config; std::ifstream ifs(model_path + "/model.json"); ifs >> config;

// 2. 创建 3 个 ncnn 网络
decoder_net = make_shared<ncnn::Net>();
embed_net = make_shared<ncnn::Net>();
proj_out_net = make_shared<ncnn::Net>();

// 3. 设置线程数（可选）
decoder_net->opt.num_threads = num_threads;
// ... embed_net, proj_out_net 同理

// 4. 加载模型权重
decoder_net->load_param(decoder_param);
decoder_net->load_model(decoder_bin);
// ... embed_net, proj_out_net 同理

// 5. 加载 tokenizer (BPE)
bpe = BpeTokenizer::LoadFromFiles(vocab_file, merges_file, ...);

// 6. 注册特殊 token 到 tokenizer
for (const auto& token : config["tokenizer"]["additional_special_tokens"]) {
    bpe->AddAdditionalSpecialToken(token);
}

// 7. 读取 EOS/BOS token ID
eos = bpe->token_to_id().at(config["tokenizer"]["eos"]);
bos = bpe->token_to_id().at(config["tokenizer"]["bos"]);

// 8. 读取模型层数、RoPE 配置等
attn_cnt = config["setting"]["attn_cnt"];  // 28 层
rope_head_dim = config["setting"]["rope"]["rope_head_dim"];  // 128
rope_theta = config["setting"]["rope"]["rope_theta"];  // 100000.0

// 9. 读取工具调用相关 token ID
tool_call_id = bpe->token_to_id().at(config["setting"]["functions"]["tool_call_id"]);
tool_call_end_id = bpe->token_to_id().at(config["setting"]["functions"]["tool_call_end_id"]);

// 10. 读取 thinking 相关 token ID
think_id = bpe->token_to_id()["惱怒"];
think_end_id = bpe->token_to_id()["惱怓"];
```

---

## 3. Phase 1: Prompt Template（对话模板）

**文件**: `src/utils/prompt.cpp`

大模型不能直接"裸"接收用户文本，需要包装成模型训练时见过的格式。

### ChatML 模板（Qwen3 / MiniCPM4）

用户输入 "你好" 经过模板包装后变成：

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
你好<|im_end|>
<|im_start|>assistant
```

注意最后那个 `<|im_start|>assistant\n` 就是 `add_generation_prompt` 的作用，告诉模型"轮到你了"。

### 核心数据结构

```cpp
// prompt.h
struct Message {
    std::string role;      // "system" / "user" / "assistant" / "tool"
    std::string content;   // 对话内容
    std::string reasoning_content;  // 思考内容（Qwen3 特有）
    std::vector<json> tool_calls;   // 工具调用
};

enum class TemplateType { CHATML, YOUTU };
```

### 多轮对话的模板拼接

假设三轮对话：

```
Message 0: system -> "You are a helpful assistant."
Message 1: user   -> "你好"
Message 2: assistant -> "你好！有什么可以帮你的？"
Message 3: user   -> "你能做什么？"
```

经过 ChatML 模板后：

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
你好<|im_end|>
<|im_start|>assistant
你好！有什么可以帮你的？<|im_end|>
<|im_start|>user
你能做什么？<|im_end|>
<|im_start|>assistant
```

### 关键代码逻辑

```cpp
// prompt.cpp - apply_chatml_template 核心逻辑
for (const auto& msg : messages) {
    if (msg.role == "system" && i == 0) continue;  // system 已单独处理

    if (msg.role == "user" || msg.role == "system") {
        prompt << "<|im_start|>" << msg.role << "\n"
               << msg.content << "<|im_end|>\n";
    }
    else if (msg.role == "assistant") {
        prompt << "<|im_start|>assistant\n";
        // 思考内容处理
        if (!reasoning_content.empty()) {
            prompt << "<think>\n" << reasoning_content << "\n</think>\n\n";
        }
        prompt << final_content << "<|im_end|>\n";
    }
    else if (msg.role == "tool") {
        // 工具响应：连续多个 tool 消息合并到一个 user 块中
        prompt << "<tool_response>\n" << content << "\n</tool_response>";
    }
}

if (add_generation_prompt) {
    prompt << "<|im_start|>assistant\n";  // 触发模型生成
}
```

---

## 4. Phase 2: Tokenization（BPE 分词）

**文件**: `src/utils/tokenizer/bpe_tokenizer.cpp`, `bpe_tokenizer.h`, `tokenizer_types.h`

把文本变成整数序列（token IDs），这是模型能理解的唯一输入格式。

### 为什么需要 Tokenization？

大模型不能直接处理字符串——它只认识整数。Tokenization 就是文本和整数之间的桥梁：

```
"你好世界" -> BPE -> [151644, 220, 151645, 7326, ...] -> 模型输入
```

但 Tokenization 不是简单的"查字典"，它涉及一个**学习到的压缩算法**——BPE（Byte Pair Encoding），这个算法决定了：
- 多少个字符组成一个 token（"hello" 是 1 个 token 还是 5 个？）
- 中文、emoji、代码怎么处理
- 词表大小和推理效率的关系

### BPE 的核心思想

BPE 最初是一种数据压缩算法，后来被 [Sennrich et al. 2016] 引入 NLP 作为子词分词方法。核心思想很简单：

> **反复合并最频繁的相邻符号对，直到达到预设的词表大小。**

训练阶段（离线完成，生成 merges.txt）：
```
初始词表：所有单个字符（或字节）
统计所有相邻符号对的频率
while 词表大小 < 目标大小:
    找到频率最高的相邻对 (A, B)
    合并为新符号 AB
    更新所有文本中的出现
    将 AB 加入词表
```

推理阶段（在线执行，就是我们的 `BpeForPiece()`）：
```
按 merges.txt 中记录的合并顺序，贪心地合并相邻符号
rank 越小 = 训练时越早被合并 = 优先级越高
```

### BPE 分词的完整 6 个步骤

```
输入: "你好世界"
```

#### Step 2.1: 字节级编码（Byte-Level Encoding）

**为什么需要字节级？** 传统的字符级 BPE 遇到问题：Unicode 有 15 万+ 字符，不可能把每个字符都放进词表。GPT-2 的解决方案是：**在字节级别操作**，而不是字符级别。

UTF-8 文本先转成字节序列，每个字节映射到一个"可见"的 Unicode 字符：

```cpp
// bpe_tokenizer.cpp - InitByteMaps()
// 可打印字节（0x21~0x7E, 0xA1~0xAC, 0xAE~0xFF）保持不变
// 不可打印字节（控制字符、空白、0x7F DEL、0xAD soft hyphen 等）映射到 256+n
for (int b = 0; b < 256; ++b) {
    if (is_printable(b)) {
        byte_encoder_[b] = static_cast<uint32_t>(b);  // 188 个可打印字节：原样映射
    } else {
        byte_encoder_[b] = static_cast<uint32_t>(256 + n++);  // 68 个不可打印字节：映射到 256~323
    }
}
```

**映射表详解**：

| 字节范围 | 是否可打印 | 映射目标 | 示例 |
|---------|----------|---------|------|
| 0x00~0x20 | 否 (控制字符+空格) | 256+ | 空格 0x20 -> 256, 换行 0x0A -> 258 |
| 0x21~0x7E | 是 (ASCII 可见字符) | 自身 | 'A'=65, 'z'=122, '!'=33 |
| 0x7F | 否 (DEL) | 256+ | -> 323 |
| 0x80~0xA0 | 否 (扩展控制字符) | 256+ | -> 256+... |
| 0xA1~0xAC | 是 (U+00A1~U+00AC) | 自身 | 0xA1=161, 0xAC=172 |
| 0xAD | 否 (soft hyphen) | 256+ | -> ... |
| 0xAE~0xFF | 是 (U+00AE~U+00FF) | 自身 | 0xAE=174, 0xFF=255 |

**为什么这样设计？** 这样所有 256 个字节值都能用可见的 Unicode 字符表示，BPE 合并时不会遇到不可见字符的问题。这是 GPT-2 的经典设计，被 BBPE 等广泛采用。

**ByteEncode 示例**：

```cpp
// bpe_tokenizer.cpp - ByteEncode()
// 输入: "你" 的 UTF-8 编码 = [0xE4, 0xBD, 0xA0]
// 0xE4 (可打印) -> Unicode 0xE4 = 'ä'
// 0xBD (可打印) -> Unicode 0xBD = '½'
// 0xA0 (不可打印) -> 256 + n = 'Ġ' (映射后的字符)
// 输出: "ä½Ġ"
```

**ByteDecode（逆操作）**：

```cpp
// bpe_tokenizer.cpp - ByteDecode()
// 遍历解码后的字符串，每个 Unicode 码点查 byte_decoder_ 还原为原始字节
// 未在 byte_decoder_ 中的码点被静默跳过
```

#### Step 2.2: 预分词（Pretokenize）

本项目支持**两种预分词模式**，由 `use_byte_encoder_` 标志控制：

**模式 A：SentencePiece 风格**（`use_byte_encoder_ = false`，默认模式）

```cpp
// bpe_tokenizer.cpp - PretokenizeSentencePiece()
// 按 Unicode 空白字符切分，每段加 ▁ (U+2581) 前缀
// 空格本身被消费，不作为独立 token

"hello world" -> ["▁hello", "▁world"]
"你好 世界"   -> ["▁你好", "▁世界"]
```

`IsUnicodeSpace()` 覆盖的空白字符：
- ASCII: 空格(0x20)、制表符(0x09)、换行(0x0A)等
- Unicode: NBSP(U+00A0)、U+1680、U+2000~U+200A、U+2028、U+2029、U+202F、U+205F、U+3000

**模式 B：字节级编码**（`use_byte_encoder_ = true`，BBPE 模式）

```cpp
// 不做预分词，整个文本先 ByteEncode()，然后作为一个整体走 BPE
// 这是 GPT-2 风格，但注意：原版 GPT-2 还会用正则预分词
// 本实现省略了正则预分词，直接对整个字符串做 BPE
```

> ⚠️ **与原版 GPT-2 的差异**：GPT-2 使用正则 `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+` 做预分词，将文本切分为单词片段后再分别做 BPE。本实现没有这个正则，在 `use_byte_encoder_ = true` 时对整个字符串做 BPE，对长文本可能较慢且结果略有不同。

#### Step 2.3: BPE 合并（Greedy Merge）—— 核心算法

这是 BPE 分词的核心。算法对每个预分词后的片段（piece），反复找 rank 最小（优先级最高）的相邻 token 对，合并它们。

```cpp
// bpe_tokenizer.cpp - BpeForPiece()
std::vector<std::string> BpeForPiece(const std::string& piece) {
    // 1. 先拆成 UTF-8 字符
    std::vector<std::string> symbols = Utf8Chars(piece);
    if (symbols.size() <= 1) return symbols;  // 单字符无需合并

    // 2. 贪心合并循环
    while (symbols.size() >= 2) {
        int best_rank = INT_MAX;
        int best_i = -1;

        // 遍历所有相邻对，找 rank 最小的
        for (int i = 0; i + 1 < (int)symbols.size(); ++i) {
            // PairKey: 用 tab 分隔，避免与 token 内的空格冲突
            std::string key = symbols[i] + "\t" + symbols[i + 1];
            auto it = merges_rank_.find(key);
            if (it != merges_rank_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_i = i;
            }
        }
        if (best_i < 0) break;  // 没有可合并的对了

        // 合并：拼接两个符号，删除后一个
        symbols[best_i] += symbols[best_i + 1];
        symbols.erase(symbols.begin() + best_i + 1);
    }
    return symbols;
}
```

**算法复杂度**：O(n^2 × m)，其中 n 是片段长度，m 是合并次数。每轮扫描所有相邻对（O(n)），合并后重新扫描，直到无法合并。这不是最优实现（可以用优先队列优化到 O(n log n)），但对于推理时的短片段足够快。

**为什么用 tab 分隔 PairKey？** 因为 token 本身可能包含空格（如 `" hello"`），用空格分隔会导致歧义。Tab 字符不会出现在 token 中，所以 `a + "\t" + b` 是唯一的。

**完整示例**：假设 merges.txt 前几行为：

```
h e          # rank=0, 最高优先级
t h          # rank=1
th e         # rank=2
▁ t          # rank=3
▁t h         # rank=4
▁th e        # rank=5
```

输入 `"▁the"` 的合并过程：

```
初始: [▁, t, h, e]

第1轮: 扫描相邻对
  (▁,t) -> rank=3
  (t,h) -> rank=1
  (h,e) -> rank=0  <- 最小！
  -> 合并 (h,e)，rank=0 优先
  结果: [▁, t, he]

第2轮: 扫描相邻对
  (▁,t) -> rank=3
  (t,he) -> 未在 merges 中
  -> 合并 (▁,t)，rank=3
  结果: [▁t, he]

第3轮: 扫描相邻对
  (▁t,he) -> 未在 merges 中
  -> 无可合并对，结束

最终: [▁t, he] -> 查 vocab 得到 token IDs
```

> 💡 **注意**：虽然 `(h,e)` 的 rank=0 比 `(t,h)` 的 rank=1 更小，但 BPE 是贪心的——每轮只合并一个最优对。如果先合并了 `(h,e)`，后续 `(t,he)` 可能不在 merges 中，导致 `▁t` 和 `he` 无法进一步合并。这体现了 BPE 的**贪心性质**：局部最优不保证全局最优，但实践中效果很好。

**BPE 结果缓存**：

```cpp
// bpe_tokenizer.cpp - BpeForPieceCached()
// 使用 unordered_map<string, vector<string>> 缓存 BPE 结果
// 同一个 piece 只需计算一次，后续直接查缓存
// 线程安全：用 mutex 保护缓存读写
static std::unordered_map<std::string, std::vector<std::string>> cache;
static std::mutex cache_mutex;
```

#### Step 2.4: merges.txt 加载与 Rank 系统

```cpp
// bpe_tokenizer.cpp - LoadMergesRank()
// 每行格式：token_A token_B（空格分隔）
// 行号 = rank（从 0 开始，越小优先级越高）
// 以 # 开头的行被跳过（通常是 #version: 0.2 头部）
// 空行被跳过

void LoadMergesRank(const std::string& merges_path) {
    std::ifstream ifs(merges_path);
    int rank = 0;
    std::string line;
    while (std::getline(ifs, line)) {
        if (line.empty() || line[0] == '#') continue;
        // 解析两个 token
        size_t space = line.find(' ');
        std::string a = line.substr(0, space);
        std::string b = line.substr(space + 1);
        // 构建 PairKey 并记录 rank
        merges_rank_[PairKey(a, b)] = rank++;
    }
}
```

**Rank 的语义**：rank 就是训练时合并的顺序。rank=0 是训练语料中最频繁的相邻对，最先被合并。推理时按 rank 从小到大贪心合并，等价于"重放"训练时的合并过程。

#### Step 2.5: vocab.txt 加载

```cpp
// bpe_tokenizer.cpp - LoadVocab()
// 每行一个 token，行号 = token ID
// 自动去除 Windows 换行符 \r
// 预分配 50000 个条目空间

void LoadVocab(const std::string& vocab_path) {
    std::ifstream ifs(vocab_path);
    int id = 0;
    std::string line;
    while (std::getline(ifs, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        id_to_token_[id] = line;
        id++;
    }
}

// 反向映射：token 字符串 -> ID
void BuildTokenToId() {
    for (auto& [id, token] : id_to_token_) {
        token_to_id_.emplace(token, id);  // 重复 token 只保留第一个
    }
}
```

**vocab.txt 示例**：

```
行号 0: <unk>           # 未知 token
行号 1: \n              # 换行（字面量 \\n）
行号 2: !               # 普通字符
...
行号 256: Ġ              # 字节级编码中空格的映射
...
行号 151644: <|im_start|>  # ChatML 特殊 token
行号 151645: <|im_end|>    # ChatML 特殊 token
...
```

#### Step 2.6: Token -> ID 查找与 Fallback

BPE 合并后的每个符号需要查找对应的 token ID：

```cpp
// bpe_tokenizer.cpp - TokensToIds()
// 正常情况：直接查 token_to_id_
// Fallback 模式（fallback_to_chars_ = true）：
//   如果合并后的 token 不在词表中，拆成单个 UTF-8 字符逐个查找
//   单个字符也不在词表中，返回 UNK ID
// 非 Fallback 模式：未知 token 直接返回 UNK ID
```

### 特殊 Token 处理

#### 标准 Special Tokens

```cpp
// tokenizer_types.h
struct SpecialTokensConfig {
    std::string bos_token;   // 句首标记
    std::string eos_token;   // 句尾标记（生成停止信号）
    std::string unk_token;   // 未知 token
    std::string sep_token;   // 分隔标记
    std::string pad_token;   // 填充标记
    std::string cls_token;   // 分类标记
    std::string mask_token;  // 掩码标记
};

struct SpecialTokenIds {
    int bos_id = -1;   // -1 表示不存在
    int eos_id = -1;
    int unk_id = -1;
    // ...
};
```

#### Additional Special Tokens（ChatML 标记等）

Qwen3 等模型使用大量额外特殊 token（如 `<|im_start|>`, `<|im_end|>`, `<think>`, `</think>`, 工具调用标记等）：

```cpp
// bpe_tokenizer.h
std::vector<std::string> additional_special_tokens_;        // token 字符串列表
std::vector<int> additional_special_token_ids_;              // 对应的 ID 列表
std::unordered_map<std::string, int> additional_special_token_to_id_;  // 快速查找
std::unordered_set<int> additional_special_id_set_;          // ID 集合（用于 decode 时判断）
```

**添加流程**（构造函数中）：

```cpp
// ncnn_llm_gpt.cpp - 构造函数
// 从 model.json 读取 additional_special_tokens 列表
for (const auto& token : config["tokenizer"]["additional_special_tokens"]) {
    bpe->AddAdditionalSpecialToken(token);
}

// AddAdditionalSpecialToken 逻辑：
// 1. 如果 token 已存在 -> 跳过（去重）
// 2. 如果 token 在 vocab 中 -> 使用现有 ID
// 3. 如果 token 不在 vocab 中且 add_if_missing=true -> 追加到 vocab 末尾
// 4. 如果 token 不在 vocab 中且 add_if_missing=false -> 静默忽略
```

### 完整的 encode 流程

```cpp
// bpe_tokenizer.cpp - encode()
std::vector<int> BpeTokenizer::encode(const string& text, bool add_bos, bool add_eos) {
    vector<int> ids;

    // 1. 添加 CLS（如果配置了）
    if (add_cls && cls_id >= 0) ids.push_back(cls_id);
    // 2. 添加 BOS（如果配置了）
    if (add_bos && bos_id >= 0) ids.push_back(bos_id);

    // 3. 扫描文本，遇到 special token 直接输出 ID，普通文本走 BPE
    string buffer;
    for (size_t i = 0; i < text.size(); ) {
        // 检查是否命中 additional_special_tokens（最长匹配）
        int matched_len = 0;
        int matched_id = -1;
        for (size_t k = 0; k < additional_special_tokens_.size(); k++) {
            const string& tok = additional_special_tokens_[k];
            if (text.compare(i, tok.size(), tok) == 0 && (int)tok.size() > matched_len) {
                matched_len = (int)tok.size();
                matched_id = additional_special_token_ids_[k];
            }
        }

        if (matched_id >= 0) {
            // 先 flush buffer 中的普通文本
            if (!buffer.empty()) {
                auto piece_ids = EncodeBuffer(buffer);
                ids.insert(ids.end(), piece_ids.begin(), piece_ids.end());
                buffer.clear();
            }
            ids.push_back(matched_id);
            i += matched_len;
            continue;
        }
        buffer.push_back(text[i++]);
    }
    // flush 剩余的普通文本
    if (!buffer.empty()) {
        auto piece_ids = EncodeBuffer(buffer);
        ids.insert(ids.end(), piece_ids.begin(), piece_ids.end());
    }

    // 4. 添加 SEP（如果配置了）
    if (add_sep && sep_id >= 0) ids.push_back(sep_id);
    // 5. 添加 EOS（如果配置了）
    if (add_eos && eos_id >= 0) ids.push_back(eos_id);

    return ids;
}
```

**EncodeBuffer 内部流程**：

```
buffer (普通文本)
  |
  +-- use_byte_encoder_ = true:
  |    ByteEncode(buffer) -> 整体 BPE -> TokensToIds
  |
  +-- use_byte_encoder_ = false:
       PretokenizeSentencePiece(buffer) -> 逐片段 BPE -> TokensToIds
```

### decode 流程（反向）

```cpp
// bpe_tokenizer.cpp - decode()
string BpeTokenizer::decode(const vector<int>& ids, bool skip_special) {
    string s;
    for (int id : ids) {
        // 跳过越界 ID
        if (id < 0 || id >= (int)id_to_token_.size()) continue;
        // 跳过特殊 token
        if (skip_special && is_special(id)) continue;
        s += id_to_token_[id];  // ID -> token 字符串
    }

    // 转义序列处理：\\t -> \t, \\n -> \n, \\r -> \r
    // ...

    // 字节级解码的逆操作
    if (use_byte_encoder_) {
        return ByteDecode(s);  // 还原原始字节
    } else {
        // SentencePiece 模式：▁ -> 空格，去除首空格
        // ...
    }
}
```

### BPE 关键设计要点总结

| 要点 | 说明 |
|------|------|
| **字节级编码** | 解决 Unicode 字符过多问题，所有文本统一在字节级别处理 |
| **贪心合并** | 每轮只合并 rank 最小的对，不保证全局最优但实践效果好 |
| **Rank 系统** | merges.txt 的行号即 rank，越小 = 训练时越早合并 = 推理时优先级越高 |
| **Tab 分隔 PairKey** | 避免 token 内空格导致的键冲突 |
| **最长匹配特殊 token** | 多个特殊 token 共享前缀时，选最长的 |
| **Fallback 到字符** | 合并后的 token 不在词表中时，拆成单字符逐个查找 |
| **结果缓存** | 同一片段只计算一次 BPE，用 mutex 保证线程安全 |
| **两种预分词模式** | SentencePiece（▁前缀）vs 字节级（无预分词） |

### BPE vs 其他分词方法对比

| 方法 | 代表模型 | 词表大小 | 特点 |
|------|---------|---------|------|
| **BPE** | GPT-2, Qwen, LLaMA | 32k~150k | 贪心合并，确定性输出 |
| **BBPE** | Qwen3, ChatGLM | 150k+ | 字节级 BPE，无需 UNK |
| **Unigram** | T5, ALBERT | 32k | 概率模型，可多种分词，Viterbi 解码 |
| **WordPiece** | BERT | 30k | 类似 BPE 但用似然选合并对 |
| **SentencePiece** | 工具库 | - | 支持 BPE/Unigram，语言无关 |

> 💡 **BBPE vs BPE**：BBPE（Byte-level BPE）是 BPE 的变体，初始词表是 256 个字节而不是字符。本项目 Qwen3 使用的就是 BBPE（model.json 中 `type: "bbpe"`）。BBPE 的优势：任何文本都能编码，不需要 UNK token；劣势：中文等语言可能需要更多 token 才能表示同一个词。

### 本项目中的 Unigram Tokenizer

除了 BPE，本项目还实现了 `UnigramTokenizer`（`unigram_tokenizer.cpp`），使用完全不同的算法：

- 加载模型文件，每行包含 token 字符串和对数概率
- 构建 **Trie 树**做前缀匹配
- 使用 **Viterbi 动态规划**（`SegmentPiece`）找到概率最高的分词路径
- 未匹配的字符使用 `unk_penalty_`（默认 -10.0）作为惩罚
- 不支持字节级编码和额外特殊 token

目前 Qwen3 使用 BPE，Unigram 作为备选实现保留。

---

## 5. Phase 3: RoPE 位置编码

**文件**: `src/utils/rope_embed.cpp`

Transformer 的注意力机制本身没有位置概念，需要注入位置信息。Qwen3 用 **RoPE（Rotary Position Embedding）**。

### 核心原理

对每个 token 的 embedding，按位置生成一对 cos/sin 旋转矩阵：

```
位置 i 的向量: 旋转 i 次
位置 j 的向量: 旋转 j 次

注意力计算时，旋转角度的差值 (i - j) 天然编码了相对位置
```

### RoPE 计算公式

```
inv_freq[k] = 1 / (theta ^ (2k / dim))
cos[i][k] = cos(pos_i * inv_freq[k])
sin[i][k] = sin(pos_i * inv_freq[k])

其中:
- pos_i: 第 i 个 token 的位置
- k: 维度索引 (0 ~ dim/2-1)
- theta: 基频 (Qwen3 为 100000)
```

### 代码实现

```cpp
// rope_embed.cpp - generate_rope_embed_cache()
void generate_rope_embed_cache(int seqlen, int embed_dim, int position_id,
                               ncnn::Mat& cos_cache, ncnn::Mat& sin_cache,
                               float rope_theta) {
    // 1. 计算逆频率
    vector<float> inv_freq(embed_dim / 2);
    for (int i = 0; i < embed_dim / 2; i++) {
        inv_freq[i] = 1.0f / powf(rope_theta, (float)(i * 2) / embed_dim);
    }

    // 2. 生成 cos/sin cache
    cos_cache.create(embed_dim / 2, seqlen);  // [dim/2, seqlen]
    sin_cache.create(embed_dim / 2, seqlen);

    for (int i = 0; i < seqlen; i++) {
        float* cos_ptr = cos_cache.row(i);
        float* sin_ptr = sin_cache.row(i);

        for (int j = 0; j < embed_dim / 2; j++) {
            const int pos = position_id + i;  // 绝对位置
            const float t = pos * inv_freq[j];
            *cos_ptr++ = cosf(t);
            *sin_ptr++ = sinf(t);
        }
    }
}
```

### 生成的 Cache 示例

假设 `embed_dim=8, seqlen=3, position_id=0, theta=10000`:

```
inv_freq = [1.0, 0.01, 0.0001, 0.000001]  // dim/2=4

位置 0: cos=[1.000, 1.000, 1.000, 1.000], sin=[0.000, 0.000, 0.000, 0.000]
位置 1: cos=[0.540, 0.999, 1.000, 1.000], sin=[0.841, 0.010, 0.000, 0.000]
位置 2: cos=[-0.416, 0.999, 1.000, 1.000], sin=[0.909, 0.020, 0.000, 0.000]
```

**注意**：低维度旋转快，高维度旋转慢。这是 RoPE 的精妙之处。

### 支持多种 RoPE 变体

你的项目支持 4 种 RoPE：

| 变体 | 适用场景 | 关键区别 |
|------|---------|---------|
| RoPE | 标准模型 | 基础旋转 |
| NTK_RoPE | 长上下文外推 | 调整 theta 值 |
| YARN_RoPE | 超长上下文 | 带 mscale 缩放 |
| LongRoPE | 超长序列 | short/long factor 双模式 |

### RoPE 变体深入解析

**为什么需要 RoPE 变体？** 标准 RoPE 在训练上下文长度之外表现不佳——超出训练长度的位置会引发注意力分数崩塌，相对位置信息丢失。长上下文外推需要调整频率分配策略。

#### NTK-Aware RoPE

核心思想：**调整 theta 而非位置**，让低频维度变化更大。

```cpp
// rope_embed.cpp - generate_ntk_rope_embed_cache()
float ntk_theta = rope_theta * powf(rope_scaling_factor, (float)embed_dim / (embed_dim - 2));
// 然后用 ntk_theta 替代 rope_theta 计算 inv_freq
for (int i = 0; i < embed_dim / 2; i++) {
    inv_freq[i] = 1.0f / powf(ntk_theta, (float)(i * 2) / embed_dim);
}
```

**直觉**：RoPE 的高频维度（小维度索引）本身就能区分近处位置，而低频维度（大维度索引）旋转很慢，在长距离上难以区分。NTK-Aware RoPE 通过增大 theta 使低频维度旋转更慢，从而"拉伸"它们能覆盖的位置范围。

与简单的线性位置插值（所有位置除以缩放因子）相比，NTK 只调整低频维度，保留高频维度的精细位置分辨能力。

#### YaRN RoPE

YaRN = NTK base + 幅度缩放（mscale）+ 平滑过渡（ramp）。

```cpp
// rope_embed.cpp - generate_yarn_rope_embed_cache()
// 1. 先计算 NTK 修正后的 theta
float ntk_theta = rope_theta * powf(rope_scaling_factor, (float)embed_dim / (embed_dim - 2));

// 2. 对每个维度计算 inv_freq，加入 ramp 平滑过渡
for (int i = 0; i < embed_dim / 2; i++) {
    float freq = 1.0f / powf(ntk_theta, (float)(i * 2) / embed_dim);
    // ramp 函数：在 beta_fast 和 beta_slow 之间平滑插值
    if (freq < beta_fast) {
        // 高频维度：不缩放
        inv_freq[i] = freq;
    } else if (freq > beta_slow) {
        // 低频维度：按缩放因子调整
        inv_freq[i] = freq / rope_scaling_factor;
    } else {
        // 过渡区：线性插值
        float smooth = (freq - beta_fast) / (beta_slow - beta_fast);
        inv_freq[i] = (1 - smooth) * freq + smooth * freq / rope_scaling_factor;
    }
}

// 3. mscale 幅度缩放：补偿长距离注意力的衰减
float mscale = sqrtf(1 + logf(rope_scaling_factor) / logf(rope_theta));
// cos/sin 乘以 mscale，放大远处 token 的注意力贡献
```

**为什么需要 mscale？** 在扩展上下文中，远处 token 的注意力分数天然较弱（旋转角差异虽大但信号衰减），mscale 通过放大 cos/sin 值来补偿这种衰减。

#### LongRoPE

LongRoPE 使用**逐维度的缩放因子**，而非全局缩放：

```cpp
// rope_embed.cpp - generate_rope_embed_cache_LongRoPE()
// 选择 short_factor 还是 long_factor
const vector<float>& ext_factor = (seqlen > original_max_position_embeddings)
    ? long_factor : short_factor;

for (int i = 0; i < embed_dim / 2; i++) {
    inv_freq[i] = 1.0f / powf(rope_theta, (float)(i * 2) / embed_dim) / ext_factor[i];
}

// 额外的全局缩放因子
float compute_scaling_factor = sqrtf(1 + logf(rope_scaling_factor)
    / logf((float)original_max_position_embeddings));
```

**为什么两套因子？** 短上下文（在训练范围内）需要最小扰动以保持原始性能，长上下文需要激进缩放以支持外推。`ext_factor` 数组为每个维度提供独立的缩放，比全局缩放更精细。

#### Full RoPE

与标准 RoPE（只旋转前半维度）不同，Full RoPE 将 cos/sin 值复制到后半维度：

```cpp
// rope_embed.cpp - generate_rope_embed_cache_full()
// 生成前半维度的 cos/sin
for (int i = 0; i < seqlen; i++) {
    for (int j = 0; j < embed_dim / 2; j++) {
        const float t = (position_id + i) * inv_freq[j];
        cos_ptr[j] = cosf(t);
        sin_ptr[j] = sinf(t);
    }
    // 复制到后半维度（用于某些 RoPE 作用在完整 head_dim 上的模型）
    for (int j = 0; j < embed_dim / 2; j++) {
        cos_ptr[j + embed_dim / 2] = cos_ptr[j];
        sin_ptr[j + embed_dim / 2] = sin_ptr[j];
    }
}
```

#### RoPE 变体对比总表

| 变体 | inv_freq 公式 | 位置缩放 | 幅度缩放 | 因子切换 | 选择建议 |
|------|--------------|----------|----------|----------|---------|
| 标准 | 1/θ^(2j/d) | 无 | 无 | 无 | 训练长度内 |
| NTK | 1/(θ·α^(d/(d-2)))^(2j/d) | 无 | 无 | 无 | 1.5~2x 扩展 |
| YaRN | NTK + ramp 过渡 | 无 | mscale | 无 | 2~8x 扩展 |
| LongRoPE | 1/θ^(2j/d) / ext_factor[j] | 无 | scaling_factor | short/long | 8x+ 扩展 |
| Full | 同标准，cos 复制到全维度 | 无 | 无 | 无 | 特定模型 |

### 多轮对话的 RoPE

多轮续聊时，RoPE 起始位置不是 0，而是 `position_id`（上一轮结束时的位置）：

```cpp
int current_pos = ctx->position_id;  // 从上次结束位置开始
generate_rope_embed_cache(token_ids.size(), rope_head_dim, current_pos, ...);
ctx->position_id += token_ids.size();
```

---

## 6. Phase 4: Embedding（词嵌入）

**文件**: `ncnn_llm_gpt.cpp` 第 267-273 行

将 token IDs 转换成连续向量表示。

### 基本流程

```
输入: [151644, 220, 151645, 7326]    // token IDs (seq_len=4)
      |
      v embed_net (ncnn network)
      |
输出: [seq_len, hidden_dim] Mat       // 例如 [4, 1024] 的浮点矩阵
```

在 ncnn 中，这本质上是一个**查表操作**（Gather）：
- 每个 token ID 对应 embedding 矩阵中的一行
- ncnn 用 Embed 层实现这个查表

```cpp
// 代码：ncnn_llm_gpt.cpp
ncnn::Mat input_ids_mat = ncnn::Mat((int)token_ids.size(), 1, (void*)token_ids.data()).clone();
ncnn::Mat token_embed;
{
    ncnn::Extractor ex = embed_net->create_extractor();
    ex.input("in0", input_ids_mat);
    ex.extract("out0", token_embed);
}
```

### ncnn::Mat 的数据格式

`ncnn::Mat` 是 ncnn 的核心数据结构：

```cpp
ncnn::Mat mat(w, h, c);  // width, height, channels
// w = 特征维度（如 embedding dim）
// h = 序列长度（token 数量）
// c = 通道数（通常为 1）

// 一维向量：
ncnn::Mat mat(w);

// 二维矩阵：
ncnn::Mat mat(w, h);

// 三维张量：
ncnn::Mat mat(w, h, c);
```

---

## 7. Phase 5: Decoder 解码器前向

**文件**: `ncnn_llm_gpt.cpp` 第 275-400 行，`ncnn-master/src/layer/sdpa.cpp`

这是整个推理最复杂、计算量最大的部分。Qwen3-0.6B 的 decoder 共 **1017 层** ncnn 算子，实现 **28 层 Transformer 解码器**。

### RMSNorm vs LayerNorm

在深入 Decoder 之前，必须理解 RMSNorm——它是 Decoder 中出现频率最高的操作。

RMSNorm（Root Mean Square Normalization）和 LayerNorm 都是对隐藏维度做归一化，但 RMSNorm 做了关键简化：

```
LayerNorm:   y = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta
RMSNorm:     y = x / sqrt(mean(x²) + eps) * gamma
```

**核心差异**：RMSNorm 省略了均值减法（`- mean`）和偏置项（`+ beta`），只保留缩放因子 `gamma`。

ncnn 中的实现（`ncnn/src/layer/rmsnorm.cpp`）：

```cpp
// Step 1: 计算平方和
float sqsum = 0.f;
for (int i = 0; i < size; i++)
    sqsum += ptr[i] * ptr[i];

// Step 2: 计算均方根
float rms = sqsum / size;

// Step 3: 计算归一化系数
float a = 1.f / sqrtf(rms + eps);

// Step 4: 归一化 + 缩放
ptr[i] = (ptr[i] * a) * gamma_ptr[i];
```

本项目（Qwen3-0.6B）中 RMSNorm 的两种使用场景：

| 场景 | affine_size | eps | 说明 |
|------|------------|-----|------|
| Pre-Attention / Pre-MLP 归一化 | 1024 | 1e-6 | 对整个 hidden_dim 归一化 |
| QK-Norm（Q/K 归一化） | 128 | 1e-6 | 对单个 head_dim 归一化 |

**RMSNorm vs LayerNorm 对比**：

| 特性 | RMSNorm | LayerNorm |
|------|---------|-----------|
| 均值减法 | 无 | 有 (`- mean`) |
| 偏置参数 beta | 无 | 有 |
| 缩放参数 gamma | 有 | 有 |
| 参数量/层 | affine_size | 2 x affine_size |
| 计算量 | 较少（省减法+偏置加法） | 较多 |
| 典型使用 | Qwen、LLaMA、Mistral | BERT、GPT-2 |

> 💡 RMSNorm 被现代 LLM 普遍采用，因为省略均值中心化和偏置项后，计算更快且对模型精度几乎无影响，而减少的参数和运算在 28 层叠加后效果显著。

### Decoder 网络 I/O

```
Decoder 输入:
  in0           token embedding       [seq_len, 1024]
  in1           causal mask           [seq_len+cache_len, seq_len]
  in2           RoPE cos              [seq_len, 64]
  in3           RoPE sin              [seq_len, 64]
  cache_k0..27  28层 Key KV Cache     (可选, 首次 prefill 无)
  cache_v0..27  28层 Value KV Cache   (可选, 首次 prefill 无)

Decoder 输出:
  out0                hidden_state    [1, 1024]
  out_cache_k0..27    28层更新后 Key Cache
  out_cache_v0..27    28层更新后 Value Cache
```

### 单个 Transformer Block 数据流

28 层 Decoder 的每一层都是完全相同的结构。以下是**单个 Transformer Block** 的完整数据流：

```
x_in [seq_len, 1024]
  |
  v RMSNorm (affine_size=1024, eps=1e-6) -> x_norm
  | Split -> 3 branches (Q, K, V)
  |
  +-- Q branch:
  |   Gemm(Q): 1024 -> 2048 (16 heads x 128 dim)
  |   Reshape: (128, 16, seq_len)
  |   RMSNorm(128, eps=1e-6) -> QK-Norm
  |   Permute -> (128, seq_len, 16)
  |   RotaryEmbed(Q, cos, sin)
  |
  +-- K branch:
  |   Gemm(K): 1024 -> 1024 (8 kv_heads x 128 dim)
  |   Reshape: (128, 8, seq_len)
  |   RMSNorm(128, eps=1e-6) -> QK-Norm
  |   Permute -> (128, seq_len, 8)
  |   RotaryEmbed(K, cos, sin)
  |   ExpandDims + Tile: 8 KV heads -> 16 heads (GQA)
  |   Reshape: (128, seq_len, 16)
  |
  +-- V branch:
  |   Gemm(V): 1024 -> 1024 (8 kv_heads x 128 dim)
  |   Reshape: (128, 8, seq_len)
  |   Permute -> (128, seq_len, 8)
  |   ExpandDims + Tile: 8 KV heads -> 16 heads (GQA)
  |   Reshape: (128, seq_len, 16)
  |
  v SDPA(Q_rot, K_exp, V_exp, mask, past_K, past_V)
  |   scale=0.0883883 = 1/sqrt(128)
  |   outputs: attn_out, out_cache_k, out_cache_v
  v Permute + Reshape: attn_out -> (2048, seq_len)
  v Gemm(O): 2048 -> 1024 (output projection)
  v Residual Add: attn_out_proj + x_in -> x_attn_res
  |
  v RMSNorm (affine_size=1024, eps=1e-6) -> x_mlp_norm
  | Split -> 2 branches (gate, up)
  |
  +-- Gate branch:
  |   Gemm(gate): 1024 -> 3072
  |   Swish(x) = x * sigmoid(x)
  |
  +-- Up branch:
  |   Gemm(up): 1024 -> 3072
  |
  v Element-wise Multiply: Swish(gate) * up -> (3072, seq_len)
  v Gemm(down): 3072 -> 1024
  v Residual Add: mlp_out + x_attn_res -> x_out [seq_len, 1024]
```

对应 ncnn param 文件中的实际算子序列（第 0 层）：

```
RMSNorm   rmsn_196     2 -> 146       0=1024 1=1e-6 2=1
Split     splitncnn_4  146 -> 147,148,149
Gemm      gemm_0       149 -> 150     8=2048 9=1024    # Q projection
Reshape   reshape_393  150 -> 151     0=128 1=16 2=-1  # (head_dim, num_heads, seq_len)
RMSNorm   rmsn_197     151 -> 152     0=128 1=1e-6     # QK-Norm
Permute   transpose_561 152 -> 153    0=2              # (head_dim, seq_len, num_heads)
Gemm      gemm_1       148 -> 154     8=1024 9=1024    # K projection
Reshape   reshape_394  154 -> 155     0=128 1=8 2=-1   # (head_dim, kv_heads, seq_len)
RMSNorm   rmsn_198     155 -> 156     0=128 1=1e-6     # QK-Norm
Permute   transpose_562 156 -> 157    0=2
Gemm      gemm_2       147 -> 158     8=1024 9=1024    # V projection
Reshape   reshape_395  158 -> 159     0=128 1=8 2=-1
Permute   transpose_563 159 -> 160    0=2
RotaryEmbed rope_0     153,88,145 -> 161               # Q RoPE
RotaryEmbed rope_1     157,87,144 -> 162               # K RoPE
ExpandDims unsqueeze_673 162 -> 163   -23303=1,1       # K GQA expand
Tile       expand_337  163 -> 164     -23302=4,1,2,1,1 # K GQA tile
Reshape    reshape_396 164 -> 165     0=128 1=-1 2=16  # K -> 16 heads
ExpandDims unsqueeze_674 160 -> 166   -23303=1,1       # V GQA expand
Tile       expand_338  166 -> 167     -23302=4,1,2,1,1 # V GQA tile
Reshape    reshape_397 167 -> 168     0=128 1=-1 2=16  # V -> 16 heads
SDPA       sdpa_729    161,165,168,31,cache_k0,cache_v0 -> 169,out_cache_k0,out_cache_v0
                                       5=1 6=0.0883883 7=1
Permute    transpose_564 169 -> 170   0=2
Reshape    reshape_398 170 -> 171     0=2048 1=-1      # concat all heads
Gemm       gemm_3      171 -> 172     8=1024 9=2048    # O projection
BinaryOp   add_0       1,172 -> 173   0=0              # residual add
Split      splitncnn_5 173 -> 174,175
RMSNorm    rmsn_199    175 -> 176     0=1024 1=1e-6    # pre-MLP norm
Split      splitncnn_6 176 -> 177,178
Gemm       gemm_4      178 -> 179     8=3072 9=1024    # gate projection
Swish      silu_309    179 -> 180                       # Swish activation
Gemm       gemm_5      177 -> 181     8=3072 9=1024    # up projection
BinaryOp   mul_1       180,181 -> 182 0=2              # gate * up
Gemm       gemm_6      182 -> 183     8=1024 9=3072    # down projection
BinaryOp   add_2       174,183 -> 184 0=0              # residual add
```

> 💡 以上模式在 28 层中完全重复，只是每层使用不同的 RMSNorm gamma 权重、Gemm 投影权重和 SDPA cache blob。

### Q/K/V 投影与头拆分

输入 `x_norm` 经过 `Split` 算子复制为三份，分别送入 Q、K、V 三个 Gemm 投影层：

```
x_norm [seq_len, 1024]
  |
  +-- Gemm(Q): [seq_len, 1024] x [1024, 2048] -> [seq_len, 2048]
  |     2048 = num_q_heads(16) x head_dim(128)
  |
  +-- Gemm(K): [seq_len, 1024] x [1024, 1024] -> [seq_len, 1024]
  |     1024 = num_kv_heads(8) x head_dim(128)
  |
  +-- Gemm(V): [seq_len, 1024] x [1024, 1024] -> [seq_len, 1024]
        1024 = num_kv_heads(8) x head_dim(128)
```

**Reshape 头拆分**：Gemm 输出是展平的，需要 `Reshape` 为三维张量以分离头维度：

```
Q: [2048, seq_len] --Reshape(128, 16, -1)--> [128, 16, seq_len]
K: [1024, seq_len] --Reshape(128,  8, -1)--> [128,  8, seq_len]
V: [1024, seq_len] --Reshape(128,  8, -1)--> [128,  8, seq_len]
     ^      ^                    ^    ^     ^
     |      |                    |    |     seq_len (自动推导)
     |      |                    |    num_heads / kv_heads
     |      |                    head_dim
     total_dim = num_heads x head_dim
```

**Permute 维度重排**：SDPA 层期望 `[head_dim, seq_len, num_heads]` 格式，需要 `Permute(0=2)` 将 c 维移到 h 位置。

### QK-Norm（Q/K 归一化）

QK-Norm 是 Qwen3 的独特设计，在 Q 和 K 投影后、RoPE 旋转编码前，对每个注意力头独立做 RMSNorm。

**位置**：
```
Gemm(Q) -> Reshape(128,16,seq_len) -> RMSNorm(128) -> Permute -> RotaryEmbed
Gemm(K) -> Reshape(128,8,seq_len)  -> RMSNorm(128) -> Permute -> RotaryEmbed
```

**为什么需要 QK-Norm？** 在标准 Transformer 中，随着序列长度增加，注意力分数（Q 和 K 的点积）的方差可能急剧增大，导致 softmax 输出趋向 one-hot 分布（**注意力熵坍塌**）。QK-Norm 通过对 Q 和 K 归一化，直接约束了 QK 点积的数值范围，防止长上下文中注意力完全集中在少数 token 上。

**affine_size=128 的含义**：归一化在每个头的维度（128）上独立进行，而非整个 hidden_dim（1024）。

| 模型 | QK-Norm | 归一化位置 |
|------|---------|-----------|
| Qwen3 | 有，affine_size=128 | Q/K 投影后、RoPE 前 |
| Qwen2.5 / LLaMA 2/3 | 无 | - |
| Gemma 2 | 有 | 类似位置 |

### GQA (Grouped-Query Attention)

GQA 是一种在计算效率和模型质量之间折中的注意力机制。先对比三种注意力类型：

```
MHA (Multi-Head Attention) — 所有头独立:
  num_q_heads = 16, num_kv_heads = 16
  每个 Q 头有独立的 K、V 头
  KV Cache: 存储 16 个头的 K/V

GQA (Grouped-Query Attention) — 头分组共享 KV:
  num_q_heads = 16, num_kv_heads = 8
  每 2 个 Q 头共享 1 组 K、V 头 (16/8 = 2)
  KV Cache: 存储 8 个头的 K/V (节省 50%)

MQA (Multi-Query Attention) — 所有头共享 1 组 KV:
  num_q_heads = 16, num_kv_heads = 1
  所有 16 个 Q 头共享同一组 K、V
  KV Cache: 存储 1 个头的 K/V (节省 93.75%)
```

**Qwen3-0.6B 的 GQA 参数**：

| 参数 | 值 | 说明 |
|------|-----|------|
| num_q_heads | 16 | 查询头数量 |
| num_kv_heads | 8 | 键值头数量 |
| num_heads_per_group | 2 | 每组内 Q 头数 = 16/8 |
| head_dim | 128 | 每个头的维度 |

**ExpandDims + Tile 复制机制**：

GQA 的核心：8 个 KV 头如何为 16 个 Q 头服务？答案是在注意力计算前将 KV 头**复制扩展**到与 Q 头数量一致：

```
K 经过 RoPE 后: [128, 8, seq_len]    — 8 个 KV 头
  |
  v ExpandDims(axis=-1)
  [128, 8, seq_len, 1]               — 在末尾增加一个维度
  |
  v Tile(rep=(2,1))                   — 沿新增维度复制 2 份
  [128, 8, seq_len, 2]               — 每个头变成 2 份
  |
  v Reshape(128, -1, 16)
  [128, seq_len, 16]                  — 8*2 = 16 个头，与 Q 数量一致
```

**内存节省**：KV Cache 只需存储 8 个头的 K/V 而非 16 个，节省 50%。对于 28 层模型，这显著降低了长序列推理的内存占用。

**SDPA 中的 GQA 实现**：

```cpp
// sdpa.cpp
const int num_heads_per_group = num_heads / num_group;  // 16 / 8 = 2

for (int q = 0; q < num_heads; q++) {
    const Mat key_head = key.channel(q / num_heads_per_group);
    // Q头0,1 -> KV头0; Q头2,3 -> KV头1; ...
}
```

### SDPA (Scaled Dot-Product Attention)

**计算步骤**：

```
Step 1: KV Cache 拼接
  key = concat(past_key, cur_key)   [128, past_seqlen+cur_seqlen, 8]
  value = concat(past_value, cur_value)

Step 2: QK 交叉注意力（点积 + 缩放）
  scale = 0.0883883 = 1/sqrt(128) = 1/sqrt(head_dim)
  qk_cross[i][j] = sum(Q[q][i] * K[kv_group][j]) * scale

Step 3: 掩码加法
  qk_cross[i][j] += mask[i][j]  // 未来位置 = -1e38

Step 4: Softmax（数值稳定版本）
  max_val = max(qk_cross[i][:])
  qk_cross[i][j] = exp(qk_cross[i][j] - max_val)
  qk_cross[i][:] /= sum(qk_cross[i][:])

Step 5: QKV 交叉（加权求和）
  attention_out[q][i][d] = sum(qk_cross[i][j] * V[kv_group][j][d])

Step 6: 输出 KV Cache
  out_cache_k = key, out_cache_v = value
```

**scale=0.0883883 的由来**：`1 / sqrt(head_dim) = 1 / sqrt(128) ≈ 0.088388`。缩放的目的是防止点积值过大导致 softmax 梯度消失。

### SwiGLU / MLP（前馈网络）

Qwen3 使用 SwiGLU 激活函数，而非传统 ReLU：

```
SwiGLU 公式: output = (Swish(x @ W_gate) ⊙ (x @ W_up)) @ W_down

其中 Swish/SiLU: Swish(x) = x * sigmoid(x) = x / (1 + exp(-x))
```

**Swish vs ReLU**：

```
ReLU(x)  = max(0, x)         -- 硬门控：负值直接归零
Swish(x) = x * sigmoid(x)    -- 软门控：负值被压缩但非零
  x=-1: Swish = -0.269 (非零！)
  x->+inf: Swish -> x (线性通过)
```

Swish 的"软门控"特性：值接近 0 的特征被软性抑制（但保留微弱信号），大幅值特征几乎原样通过。相比 ReLU 的硬截断，Swish 提供了更平滑的梯度流。

**为什么中间维度是 3072 = 3×1024？**

| 模型 | hidden_dim | intermediate_dim | 比率 | 权重矩阵数 |
|------|-----------|-----------------|------|-----------|
| 标准 FFN (ReLU) | 1024 | 4096 | 4x | 2 (W1, W2) |
| SwiGLU (LLaMA/Qwen) | 1024 | 3072 | 3x | 3 (W_gate, W_up, W_down) |

标准 FFN 只有 2 个权重矩阵，中间维度通常取 4x。SwiGLU 需要 3 个权重矩阵，为保持总参数量大致不变，中间维度降低为 3x。

**"门控"的直觉**：gate 分支通过 Swish 激活后产生 0~1 范围的软门控信号，与 up 分支逐元素相乘，相当于对 up 的每个维度决定"放行多少"。

### 残差连接（Residual Connection）

每个 Transformer Block 包含**两条残差连接**：

```
残差 1（注意力残差）:
  x_attn = Attention_output_projection(x_norm) + x_in

残差 2（MLP 残差）:
  x_out = MLP(x_mlp_norm) + x_attn
```

**Pre-Norm vs Post-Norm**：

Qwen3 使用 **Pre-Norm**（先归一化，再做变换）：

```
Pre-Norm (Qwen3, LLaMA, 现代 LLM):
  x_attn = Attention(RMSNorm(x_in)) + x_in     ← 归一化在 Attention 前
  x_out  = MLP(RMSNorm(x_attn)) + x_attn        ← 归一化在 MLP 前

Post-Norm (原始 Transformer, BERT):
  x_attn = RMSNorm(Attention(x_in) + x_in)      ← 归一化在残差加法后
```

Pre-Norm 优势：训练更稳定，不需要 warm-up，梯度在深层网络中流动更顺畅。现代 LLM 几乎全部采用 Pre-Norm。

### 因果掩码（Causal Mask）

保证每个 token 只能看到它之前的内容：

```cpp
ncnn::Mat mask((int)token_ids.size(), (int)token_ids.size());
mask.fill(0.0f);
for (int i = 0; i < (int)token_ids.size(); i++) {
    float* row = mask.row(i);
    for (int j = i + 1; j < (int)token_ids.size(); j++) {
        row[j] = -1e38f;  // 负无穷，softmax 后变成 0
    }
}
```

```
掩码矩阵示意 (seq_len=4):

     t0     t1     t2     t3
t0   0    -1e38  -1e38  -1e38   <-- t0 只能看到自己
t1   0     0    -1e38  -1e38   <-- t1 能看到 t0,t1
t2   0     0     0     -1e38   <-- t2 能看到 t0,t1,t2
t3   0     0     0      0      <-- t3 能看到全部
```

### KV Cache 提取

Prefill 阶段，从 decoder 中提取每一层的 K 和 V：

```cpp
std::vector<pair<ncnn::Mat, ncnn::Mat>> kv_cache;

ncnn::Extractor ex = decoder_net->create_extractor();
ex.input("in0", token_embed);
ex.input("in1", mask);
ex.input("in2", cos_cache);
ex.input("in3", sin_cache);

for (int i = 0; i < attn_cnt; i++) {  // attn_cnt = 28
    char name_k_out[32], name_v_out[32];
    snprintf(name_k_out, sizeof(name_k_out), "out_cache_k%d", i);
    snprintf(name_v_out, sizeof(name_v_out), "out_cache_v%d", i);

    ncnn::Mat k_cache, v_cache;
    ex.extract(name_k_out, k_cache);
    ex.extract(name_v_out, v_cache);
    kv_cache.emplace_back(move(k_cache), move(v_cache));
}
```

### 最后一个 token 单独处理

Prefill 结束后，最后一个 token 需要单独再跑一次 decoder 来得到第一个预测：

```cpp
// 单独处理最后一个 token
ncnn::Mat last_token_mat = ncnn::Mat(1, 1, (void*)&last_token_id).clone();
ncnn::Mat last_token_embed;
{
    ncnn::Extractor ex = embed_net->create_extractor();
    ex.input("in0", last_token_mat);
    ex.extract("out0", last_token_embed);
}

generate_rope_embed_cache(1, rope_head_dim, token_ids.size(), last_cos_cache, last_sin_cache, rope_theta);

ncnn::Mat last_mask(token_ids.size() + 1, 1);
last_mask.fill(0.0f);

{
    ncnn::Extractor ex = decoder_net->create_extractor();
    ex.input("in0", last_token_embed);
    ex.input("in1", last_mask);
    ex.input("in2", last_cos_cache);
    ex.input("in3", last_sin_cache);

    for (int i = 0; i < attn_cnt; i++) {
        char name_k_in[16], name_v_in[16];
        snprintf(name_k_in, sizeof(name_k_in), "cache_k%d", i);
        snprintf(name_v_in, sizeof(name_v_in), "cache_v%d", i);
        ex.input(name_k_in, kv_cache[i].first);
        ex.input(name_v_in, kv_cache[i].second);
    }

    for (int i = 0; i < attn_cnt; i++) {
        char name_k_out[32], name_v_out[32];
        snprintf(name_k_out, sizeof(name_k_out), "out_cache_k%d", i);
        snprintf(name_v_out, sizeof(name_v_out), "out_cache_v%d", i);
        ncnn::Mat k_cache, v_cache;
        ex.extract(name_k_out, k_cache);
        ex.extract(name_v_out, v_cache);
        kv_cache[i] = make_pair(move(k_cache), move(v_cache));
    }

    ex.extract("out0", decode_out);
}
```

---

## 8. Phase 6: Projection（投影输出）

**文件**: `ncnn_llm_gpt.cpp` 第 402-419 行

Decoder 输出的 hidden_states 需要映射回词表空间。

```
输入: hidden_states  [1, hidden_dim]     // 例如 [1, 1024]
      |
      v proj_out_net (线性层: hidden_dim -> vocab_size)
      |
输出: logits         [vocab_size]        // 例如 [151936]
```

```cpp
ncnn::Mat logits;
{
    ncnn::Extractor ex = proj_out_net->create_extractor();
    ex.input("in0", decode_out);
    ex.extract("out0", logits);
}
```

proj_out_net 通常包含：
1. LayerNorm：归一化 hidden state
2. Linear 层：hidden_dim -> vocab_size 的线性变换

### Weight Tying（权重共享）

在 Qwen3-0.6B 中，Token Embedding 层与 Projection 层共享同一份权重。证据来自 `model.json`：

```json
"embed_token_bin": "qwen3_embed_token.ncnn.bin",
"proj_out_bin":    "qwen3_embed_token.ncnn.bin"   // 同一个文件！
```

数学上：Embedding 矩阵 E 维度为 `[vocab_size, hidden_dim]`，投影矩阵 W 维度为 `[hidden_dim, vocab_size]`。当 W = E^T 时：
- Token Embedding：`token_embed = E[token_id]`，从 E 中取第 token_id 行
- Logits 投影：`logits = E^T @ hidden_state`，用 E 的转置做矩阵乘法

**内存节省**：不共享需要 2 × (151936 × 1024 × 4 bytes) ≈ **1.2 GB**，共享后只需 ≈ **600 MB**。

> ⚠️ 并非所有模型都使用 Weight Tying。Qwen3-0.6B 使用了共享，但更大的模型可能有独立的投影权重。

### Logits -> 下一个 Token（贪心模式）

Prefill 阶段用 argmax 快速选出最可能的下一个 token：

```cpp
int next_token_id = 0;
{
    const float* p = logits;
    float max_val = p[0];
    for (int i = 1; i < logits.w; ++i) {
        if (p[i] > max_val) {
            max_val = p[i];
            next_token_id = i;
        }
    }
}
```

---

## 9. Phase 7: Sampling（采样策略）

**文件**: `src/sampling.cpp`

从 logits 中选出下一个 token。这是**唯一有随机性**的地方。

### 完整采样流程

```
logits [vocab_size]
  |
  v [1] Repetition Penalty (重复惩罚)
  |
  v [2] Temperature Scaling (温度调节)
  |
  v [3] Softmax (归一化为概率分布)
  |
  v [4] Top-K Filtering (只保留概率最大的 K 个)
  |
  v [5] Top-P / Nucleus (累积概率截断)
  |
  v [6] Sample / Argmax (随机采样或贪心)
  |
  v next_token_id (int)
```

### 1. Repetition Penalty（重复惩罚）

降低已经出现过的 token 的概率：

```cpp
// ncnn_llm_gpt.cpp
for (int t : history) {
    if (t >= vocab_size) continue;
    if (logits[t] < 0) logits[t] *= cfg.repetition_penalty;  // 负值更负
    else               logits[t] /= cfg.repetition_penalty;  // 正值变小
}
```

- `repetition_penalty > 1.0`：惩罚重复（如 1.1）
- `repetition_penalty = 1.0`：不惩罚

### 2. Temperature Scaling

```cpp
// sampling.cpp
void softmax_vec(vector<float>& logits, float temperature) {
    float max_logit = *max_element(logits.begin(), logits.end());  // 数值稳定
    float sum = 0.f;
    for (float& x : logits) {
        x = exp((x - max_logit) / temperature);
        sum += x;
    }
    for (float& x : logits) x /= sum;  // 归一化
}
```

- `temperature < 1.0`：更确定性，倾向于选概率大的
- `temperature = 1.0`：正常
- `temperature > 1.0`：更随机，分布更均匀

### 3. Top-K Filtering

只保留概率最大的 K 个 token：

```cpp
// sampling.cpp
void apply_top_k(vector<float>& probs, int k) {
    if (k <= 0 || k >= probs.size()) return;

    // 找到第 k 大的概率值
    vector<float> tmp = probs;
    nth_element(tmp.begin(), tmp.end() - k, tmp.end());
    float threshold = tmp[tmp.size() - k];

    // 低于阈值的设为 0
    for (float& p : probs) if (p < threshold) p = 0.f;
}
```

### 4. Top-P / Nucleus Sampling

累积概率达到 P 就截断：

```cpp
// sampling.cpp
void apply_top_p(vector<float>& probs, float p) {
    if (p >= 1.0f) return;

    // 按概率从大到小排序
    vector<pair<float,int>> v;
    for (int i = 0; i < probs.size(); ++i)
        v.emplace_back(probs[i], i);
    sort(v.begin(), v.end(), greater<>());

    // 累积求和，找到截断点
    float cum = 0.f;
    size_t cutoff = v.size();
    for (size_t i = 0; i < v.size(); i++) {
        cum += v[i].first;
        if (cum >= p) { cutoff = i + 1; break; }
    }

    // 截断点外的设为 0
    vector<char> keep(probs.size(), 0);
    for (size_t i = 0; i < cutoff; ++i) keep[v[i].second] = 1;
    for (int i = 0; i < probs.size(); i++)
        if (!keep[i]) probs[i] = 0.f;
}
```

### 5. 最终选择

```cpp
int next_id;
if (cfg.do_sample == 1) {
    // 随机采样：按概率分布随机选
    next_id = sample_from_probs(logits);
} else {
    // 贪心：取概率最大的
    next_id = max_element(logits.begin(), logits.end()) - logits.begin();
}
```

---

## 10. Phase 8: Token to Text（解码回文本）

拿到 `next_token_id` 后，通过 BPE tokenizer 的反向查找表转回字符串：

```cpp
// ncnn_llm_gpt.cpp - generate() 循环中
callback(bpe->decode({ctx->cur_token}, false));
```

```cpp
// bpe_tokenizer.cpp - decode()
string BpeTokenizer::decode(const vector<int>& ids, bool skip_special) {
    string s;
    for (int id : ids) {
        if (skip_special && is_special_token(id)) continue;
        s += id_to_token_[id];  // ID -> token 字符串
    }

    // 字节级解码的逆操作
    if (use_byte_encoder_) return ByteDecode(s);
    // 否则：▁ -> 空格，去除首空格
    ...
}
```

---

## 11. 完整自回归生成循环

**文件**: `ncnn_llm_gpt.cpp` 第 842-1007 行

把上面所有阶段串起来，形成完整的自回归生成循环：

```cpp
std::shared_ptr<ncnn_llm_gpt_ctx> ncnn_llm_gpt::generate(
    const shared_ptr<ncnn_llm_gpt_ctx>& ctx_in,
    const GenerateConfig& cfg,
    function<void(const string&)> callback  // 流式输出回调
) const {
    const int vocab_size = bpe->vocab_size();
    auto ctx = clone_ctx(ctx_in);  // 拷贝上下文
    unordered_set<int> history;    // 已生成的 token 集合（用于重复惩罚）
    history.insert(ctx->cur_token);

    for (int step = 0; step < cfg.max_new_tokens; ++step) {

        // === 停止条件 ===
        if (ctx->cur_token == eos) break;

        // === 工具调用检测 ===
        if (ctx->cur_token == tool_call_id) {
            flag_in_tool_call = true;
        } else if (ctx->cur_token == tool_call_end_id) {
            flag_in_tool_call = false;
            handle_tool(tool_call_content, ctx);  // 执行工具
            tool_call_content.clear();
            history.clear();
            continue;
        } else if (flag_in_tool_call) {
            tool_call_content += bpe->decode({ctx->cur_token}, false);
        } else {
            callback(bpe->decode({ctx->cur_token}, false));  // 流式输出
        }

        // === Step 1: 当前 token -> embedding ===
        ncnn::Mat cur_token_mat = ncnn::Mat(1, 1, (void*)&ctx->cur_token).clone();
        ncnn::Mat cur_embed;
        {
            ncnn::Extractor ex = embed_net->create_extractor();
            ex.input("in0", cur_token_mat);
            ex.extract("out0", cur_embed);
        }

        // === Step 2: 生成当前 token 的 RoPE ===
        ncnn::Mat cos_cache, sin_cache;
        generate_rope_embed_cache(1, rope_head_dim, ctx->position_id,
                                  cos_cache, sin_cache, rope_theta);
        ctx->position_id++;

        // === Step 3: 构建 mask（全 0，因为只看前面所有）===
        ncnn::Mat mask(ctx->kv_cache[0].first.h + 1, 1);
        mask.fill(0.f);

        // === Step 4: Decoder 前向（单 token，复用 KV Cache）===
        ncnn::Mat decode_out;
        {
            ncnn::Extractor ex = decoder_net->create_extractor();
            ex.input("in0", cur_embed);
            ex.input("in1", mask);
            ex.input("in2", cos_cache);
            ex.input("in3", sin_cache);

            // 输入旧 KV Cache
            for (int i = 0; i < attn_cnt; ++i) {
                char kname[16], vname[16];
                snprintf(kname, sizeof(kname), "cache_k%d", i);
                snprintf(vname, sizeof(vname), "cache_v%d", i);
                ex.input(kname, ctx->kv_cache[i].first);
                ex.input(vname, ctx->kv_cache[i].second);
            }

            // 提取更新后的 KV Cache
            for (int i = 0; i < attn_cnt; ++i) {
                char kname[32], vname[32];
                snprintf(kname, sizeof(kname), "out_cache_k%d", i);
                snprintf(vname, sizeof(vname), "out_cache_v%d", i);
                ncnn::Mat k_cache, v_cache;
                ex.extract(kname, k_cache);
                ex.extract(vname, v_cache);
                ctx->kv_cache[i] = { move(k_cache), move(v_cache) };
            }

            ex.extract("out0", decode_out);
        }

        // === Step 5: Projection -> logits ===
        ncnn::Mat logits_mat;
        {
            ncnn::Extractor ex = proj_out_net->create_extractor();
            ex.input("in0", decode_out);
            ex.extract("out0", logits_mat);
        }

        // 拷贝到 std::vector
        vector<float> logits(vocab_size);
        memcpy(logits.data(), logits_mat.data, sizeof(float) * vocab_size);

        // === Step 6: Repetition Penalty ===
        for (int t : history) {
            if (t >= vocab_size) continue;
            if (logits[t] < 0) logits[t] *= cfg.repetition_penalty;
            else               logits[t] /= cfg.repetition_penalty;
        }

        // === Step 7: 采样 ===
        softmax_vec(logits, cfg.temperature);
        if (cfg.top_k > 0) apply_top_k(logits, cfg.top_k);
        if (cfg.top_p < 1.0f) apply_top_p(logits, cfg.top_p);

        int next_id;
        if (cfg.do_sample == 1) {
            next_id = sample_from_probs(logits);
        } else {
            next_id = max_element(logits.begin(), logits.end()) - logits.begin();
        }

        ctx->cur_token = next_id;
        history.insert(next_id);
    }
    return ctx;
}
```

---

## 12. KV Cache 深入解析

### 什么是 KV Cache？

自回归生成时，每次只生成 1 个新 token。如果每次都重新计算前面所有 token 的 K/V，计算量是 O(n²) 的浪费。

KV Cache 缓存了之前所有 token 的 Key 和 Value 向量，使每次生成只需要 O(1) 的注意力计算（1 个 query 对 n 个 key），而不是 O(n)。

**没有 KV Cache vs 有 KV Cache**：

| | 无 KV Cache | 有 KV Cache |
|---|---|---|
| 第 t 步计算量 | O(t × d) 每层 QKV 投影 + O(t × t) 注意力 | O(d) 每层 QKV 投影 + O(t) 注意力 |
| 总计算量（n 步） | O(n³) | O(n²) |
| 内存 | O(d) | O(n × d) |
| 典型 1000 token 生成 | 约 1000³ = 10⁹ ops/层 | 约 10⁶ ops/层 |

**核心权衡**：KV Cache 用内存换计算——多占 O(n×d) 的内存，但省下 O(n²) 的重复计算。

### 数据结构

```cpp
// ncnn_llm_base.h
using KVCache = std::vector<std::pair<ncnn::Mat, ncnn::Mat>>;
// 每对 (ncnn::Mat, ncnn::Mat) = 一层的 (K_cache, V_cache)
// vector 的大小 = 层数 (attn_cnt = 28)
```

### 上下文结构（完整定义）

```cpp
// ncnn_llm_gpt.h
class ncnn_llm_gpt_ctx {
public:
    virtual ~ncnn_llm_gpt_ctx() = default;
    virtual std::shared_ptr<ncnn_llm_gpt_ctx> clone() const = 0;

    KVCache kv_cache;    // 28 层的 (K, V) cache
    int cur_token = 0;   // 当前 token ID（下一步的输入）
    int position_id = 0; // RoPE 位置计数器（单调递增）
};
```

> 💡 **三个字段的关系**：`cur_token` 是上一步生成的 token，它将被送入下一步的 embedding + decoder；`position_id` 是 `cur_token` 在序列中的绝对位置，用于生成正确的 RoPE 编码；`kv_cache` 存储了 `cur_token` 之前所有 token 的 K/V 向量。

### KV Cache 的精确数据形状

以 Qwen3 0.6B 为例（hidden_dim=1024, num_heads=16, num_kv_heads=8, head_dim=128, 28 层）：

**每层 KV Cache 的 ncnn::Mat 形状**：

```
K_cache[i]: ncnn::Mat(w=128, h=S, c=8)   // [head_dim, seq_len, num_kv_heads]
V_cache[i]: ncnn::Mat(w=128, h=S, c=8)   // [head_dim, seq_len, num_kv_heads]
```

其中 S 是当前已缓存的序列长度。

**为什么 num_kv_heads=8 而不是 num_heads=16？** 这是因为 Qwen3 使用了 **GQA（Grouped-Query Attention）**：8 个 KV head 被 16 个 Q head 共享（每 2 个 Q head 共享 1 个 KV head），所以 cache 只需存储 8 份而不是 16 份，节省了一半的 KV Cache 内存。

**GQA 的 KV 复制过程**（在 decoder param 中）：

```
K 投影输出: (128, 8, seq_len)  -- 只有 8 个 KV head
  -> ExpandDims + Tile: 复制 2 倍
  -> Reshape: (128, seq_len, 16)  -- 扩展为 16 个 head 供 Attention 使用
  -> 但 cache 存储的是扩展前的 (128, seq_len, 8)
```

### KV Cache 在 ncnn 网络中的流转

从 decoder 的 `.param` 文件可以看到 KV Cache 的完整流转路径：

```
# decoder param 中声明 56 个 cache 输入/输出 blob
Input  kv_cache  0 56 cache_k0 cache_v0 cache_k1 cache_v1 ... cache_k27 cache_v27

# 每层 SDPA 层的连接关系
SDPA  sdpa_729  6 3
    # 6 个输入 (bottom blobs):
    161 165 168 31 cache_k0 cache_v0
    # Q=161, K_cur=165, V_cur=168, mask=31, past_K=cache_k0, past_V=cache_v0
    # 3 个输出 (top blobs):
    169 out_cache_k0 out_cache_v0
    # attn_output=169, updated_K=out_cache_k0, updated_V=out_cache_v0
    # 参数:
    5=1          # attn_mask 启用
    6=0.0883883  # scale = 1/sqrt(128) = 1/sqrt(head_dim)
    7=1          # kv_cache 启用
```

**数据流向图**：

```
                 +-----------+
                 |   SDPA    |
                 |  Layer i  |
cache_k{i} ----> |           | ----> out_cache_k{i}
cache_v{i} ----> |           | ----> out_cache_v{i}
Q(新token) ---> |           | ----> attn_output
K_cur(新token) ->|          |
V_cur(新token) ->|          |
mask ----------> |          |
                 +-----------+
```

### SDPA 层内部的 KV Cache 拼接机制

**文件**: `ncnn-master/src/layer/sdpa.cpp`

SDPA（Scaled Dot-Product Attention）层是 KV Cache 拼接的核心实现：

```cpp
// sdpa.cpp - forward() 精简版
int SDPA::forward(const std::vector<Mat>& bottom_blobs,
                   std::vector<Mat>& top_blobs,
                   const Option& opt) const {

    const Mat& query = bottom_blobs[0];       // Q: (embed_dim, src_seqlen, num_heads)
    const Mat& cur_key = bottom_blobs[1];     // K_new: (embed_dim, cur_seqlen, num_group)
    const Mat& cur_value = bottom_blobs[2];   // V_new: (out_embed_dim, cur_seqlen, num_group)
    const Mat& attn_mask = bottom_blobs[3];   // mask
    const Mat& past_key = bottom_blobs[4];    // past_K cache
    const Mat& past_value = bottom_blobs[5];  // past_V cache

    const int past_seqlen = kv_cache ? past_key.h : 0;
    const int dst_seqlen = past_seqlen + cur_seqlen;  // 拼接后的总长度

    // === KV Cache 拼接 ===
    Mat key = cur_key;
    if (past_seqlen > 0) {
        key.create(embed_dim, dst_seqlen, num_group, 4u, opt.blob_allocator);
        for (int q = 0; q < num_group; q++) {
            // 先拷贝 past cache 行
            memcpy(key_channel_row(0), past_key_data, embed_dim * past_seqlen * sizeof(float));
            // 再拷贝当前新 token 的 K
            memcpy(key_channel_row(past_seqlen), cur_key_data, embed_dim * cur_seqlen * sizeof(float));
        }
    }
    // Value 同理...

    // === Attention 计算 ===
    // Q @ K^T / sqrt(d) + mask -> softmax -> @ V
    // ...

    // === 输出更新后的 KV Cache ===
    if (kv_cache) {
        top_blobs[1] = key;     // out_cache_k = [past_K; cur_K]
        top_blobs[2] = value;  // out_cache_v = [past_V; cur_V]
    }

    return 0;
}
```

**拼接示意**：

```
past_K:  [token_0, token_1, ..., token_{t-1}]  // 形状: (128, t, 8)
cur_K:   [token_t]                               // 形状: (128, 1, 8)
                                      拼接
out_K:   [token_0, token_1, ..., token_{t-1}, token_t]  // 形状: (128, t+1, 8)
```

> 💡 **关键理解**：每次 decoder 前向时，SDPA 层会创建一个**新的** ncnn::Mat 来存储拼接后的 KV，而不是在原来的 Mat 上原地扩展。旧的 cache Mat 由 ncnn::Mat 的引用计数/析构函数自动释放。

### KV Cache 的完整生命周期

#### Phase 1: Prefill（首轮，无历史 cache）

```
Step 1a: 批量处理 N-1 个 token（不设 KV cache 输入）

  token_embed [N-1, hidden] + mask [N-1, N-1] -> decoder
  不传入任何 cache_k{i} / cache_v{i}
  提取 out_cache_k{i}, out_cache_v{i}
  -> kv_cache[i] = (K_cache: [128, N-1, 8], V_cache: [128, N-1, 8])

Step 1b: 处理最后一个 token（传入 Step 1a 的 KV cache）

  last_token_embed [1, hidden] + mask [N, 1] + kv_cache -> decoder
  传入 cache_k{i}, cache_v{i}
  提取 out_cache_k{i}, out_cache_v{i}
  -> kv_cache[i] = (K_cache: [128, N, 8], V_cache: [128, N, 8])

  创建 ctx:
  ctx->kv_cache = kv_cache       // 完整的 N 个 token 的 KV
  ctx->cur_token = next_token_id // 第一个预测的 token
  ctx->position_id = N            // 下一个位置
```

#### Phase 2: Generate（自回归循环，每步 1 个 token）

```
每一步:
  cur_token -> embed -> decoder (传入 kv_cache, mask [past_len+1, 1])
  -> 更新 kv_cache (h 维度 +1)
  -> projection -> logits -> sample -> next_token

KV Cache 增长:
  Step 0: kv_cache[i].h = N      (prefill 后)
  Step 1: kv_cache[i].h = N + 1
  Step 2: kv_cache[i].h = N + 2
  ...
  Step t: kv_cache[i].h = N + t
```

#### Phase 3: Multi-Turn（多轮对话）

```
新一轮用户输入 M 个 token:

  clone_ctx(old_ctx) -> new_ctx  // 深拷贝 KV cache
  new M tokens -> embed -> decoder (传入 new_ctx->kv_cache)
  -> 更新 new_ctx->kv_cache (h 维度 += M)
  -> new_ctx->position_id += M
```

### Mask 在不同阶段的构造

KV Cache 的长度直接决定了 mask 的形状。mask 的 width = past_seqlen + cur_seqlen（总 KV 长度），height = cur_seqlen（新 query 数量）。

| 场景 | Mask 形状 (w × h) | 内容 |
|------|-------------------|------|
| **Prefill 批量** (N-1 tokens, 无 cache) | (N-1, N-1) | 下三角因果：`mask[i][j] = -1e38 if j > i else 0` |
| **Prefill 最后 token** (1 token, cache 有 N-1) | (N, 1) | 全零（单个 token 可见所有历史） |
| **Generate 单步** (1 token, cache 有 past_len) | (past_len+1, 1) | 全零 |
| **Multi-turn 批量** (M tokens, cache 有 past_len) | (past_len+M, M) | 历史部分全零 + 新 token 间因果 |
| **Multi-turn 最后 token** (1 token, cache 有 past_len+M-1) | (past_len+M, 1) | 全零 |

**Multi-turn 批量 mask 详解**：

```cpp
// 最复杂的情况：新输入 M 个 token，之前有 past_len 个 cached token
ncnn::Mat mask((int)token_ids.size() + new_ctx->kv_cache[0].first.h,
               (int)token_ids.size());
mask.fill(0.0f);
for (int i = 0; i < (int)token_ids.size(); i++) {
    float* row = mask.row(i);
    // 新 token i 能看到所有历史 (0 ~ past_len-1) 和自己之前的新 token
    // 但不能看到未来的新 token
    for (int j = new_ctx->kv_cache[0].first.h + i + 1;
         j < (int)token_ids.size() + new_ctx->kv_cache[0].first.h; j++) {
        row[j] = -1e38f;  // 屏蔽未来的新 token
    }
}
```

```
示例: past_len=5, M=3 (新 token a,b,c)

mask (8×3):
      past0  past1  past2  past3  past4  a      b      c
a  [   0,     0,     0,     0,     0,    0,  -1e38, -1e38]
b  [   0,     0,     0,     0,     0,    0,    0,   -1e38]
c  [   0,     0,     0,     0,     0,    0,    0,     0  ]

历史部分：全零（所有新 token 都能看到所有历史）
新 token 之间：下三角因果（a 不能看 b,c；b 不能看 c）
```

### KV Cache 的内存占用（精确计算）

以 Qwen3 0.6B 为例（num_kv_heads=8, head_dim=128, attn_cnt=28）：

```
每层 K cache: 128 × S × 8 × 4 bytes = 4096 × S bytes
每层 V cache: 128 × S × 8 × 4 bytes = 4096 × S bytes
每层合计: 8192 × S bytes

28 层合计: 28 × 8192 × S = 229376 × S bytes ≈ 224 KB × S

序列长度 S 的内存占用:
  S=100:   22 MB
  S=500:   112 MB
  S=1000:  224 MB
  S=2000:  448 MB
  S=4000:  896 MB
  S=8192:  1.8 GB
```

**为什么 GQA 如此重要？** 如果没有 GQA（num_kv_heads = num_heads = 16），内存会翻倍：

```
无 GQA: 28 × 128 × S × 16 × 4 × 2 = 458752 × S ≈ 448 KB × S
有 GQA: 28 × 128 × S × 8  × 4 × 2 = 229376 × S ≈ 224 KB × S
节省: 50%
```

**与业界方案对比**：

| 优化技术 | 原理 | 内存节省 | 本项目是否实现 |
|---------|------|---------|--------------|
| **GQA** | 多个 Q head 共享 KV head | ~50% (8 vs 16 heads) | ✅ |
| **MQA** | 所有 Q head 共享 1 个 KV head | ~93% | ❌ |
| **KV Cache 量化** | 将 FP32 cache 降为 INT8/FP8 | ~75% | ❌ |
| **PagedAttention** | 虚拟内存式非连续存储 | 减少碎片 | ❌ |
| **Sliding Window** | 只缓存最近 W 个 token | 固定上限 | ❌ |
| **Token Eviction** | 驱逐不重要的 token | 可变 | ❌ |

### Context Clone（深拷贝机制）

多轮对话时，`prefill()` 和 `generate()` 都会先 `clone_ctx(ctx)`：

```cpp
// ncnn_llm_gpt.h - ncnn_llm_gpt_base_ctx::clone()
std::shared_ptr<ncnn_llm_gpt_ctx> clone() const override {
    auto dst = std::make_shared<ncnn_llm_gpt_base_ctx>();
    dst->kv_cache.resize(kv_cache.size());
    for (size_t i = 0; i < kv_cache.size(); ++i) {
        dst->kv_cache[i].first = kv_cache[i].first;    // ncnn::Mat 深拷贝
        dst->kv_cache[i].second = kv_cache[i].second;  // ncnn::Mat 深拷贝
    }
    dst->cur_token = cur_token;
    dst->position_id = position_id;
    return dst;
}
```

**为什么需要深拷贝？**

1. **保护原 context 不被修改**：`generate()` 会逐步增长 KV Cache，如果不拷贝，原始 context 会被污染
2. **支持对话分支**：用户可以在同一 context 上分叉出多条对话路径，每条路径独立演化
3. **ncnn::Mat 赋值 = 深拷贝**：ncnn::Mat 的 `operator=` 内部调用 `Mat::clone()`，分配新内存并复制数据

**深拷贝的代价**：对于一个序列长度 1000 的 context：

```
224 MB × 2 (K+V) 的 KV Cache 被完整复制
-> 需要 224 MB 额外内存
-> 拷贝耗时约 10~50ms（取决于内存带宽）
```

### 典型多轮对话交互

```
用户: "你好"
  -> prefill("你好") -> ctx1 (KV Cache 包含 system + user prompt)
  -> generate(ctx1) -> 回复 "你好！有什么可以帮你的？" (KV Cache 持续增长)
  -> 返回 ctx1' (KV Cache 包含完整对话历史)

用户: "你能做什么？"
  -> clone_ctx(ctx1') -> new_ctx1  // 深拷贝保护原 context
  -> prefill("你能做什么？", new_ctx1) -> ctx2
  -> generate(ctx2) -> 回复 "我可以..."

用户: "详细说说"
  -> clone_ctx(ctx2') -> new_ctx2
  -> prefill("详细说说", new_ctx2) -> ctx3
  -> generate(ctx3) -> ...
```

### KV Cache 的局限与改进方向

| 问题 | 说明 | 业界解决方案 |
|------|------|------------|
| **内存线性增长** | 每多一个 token，多占 224 KB | PagedAttention (vLLM), Token Eviction |
| **无上限** | 生成越长越占内存，可能 OOM | Sliding Window (Mistral), 长度上限 |
| **深拷贝开销** | 多轮对话每轮拷贝整个 cache | Copy-on-Write, 引用计数 |
| **FP32 存储** | 每个 float 4 字节 | INT8/FP8 量化 (llm.int8(), FlashAttention-3) |
| **连续内存** | 每步新分配 + 拷贝，产生碎片 | PagedAttention 虚拟内存映射 |
| **不支持 beam search** | 只保留一条路径的 cache | 扩展为 beam cache 管理 |

---

## 13. 多轮对话续聊机制

**文件**: `ncnn_llm_gpt.cpp` 第 638-840 行（prefill 带 ctx 的重载）

多轮对话的核心思想：**复用上一轮的 KV Cache**，而不是从头开始。

### 续聊流程

```cpp
std::shared_ptr<ncnn_llm_gpt_ctx> ncnn_llm_gpt::prefill(
    const string& input_text,          // 新输入的文本
    const shared_ptr<ncnn_llm_gpt_ctx> ctx  // 上一轮的上下文
) const {
    // 1. 克隆上下文
    shared_ptr<ncnn_llm_gpt_ctx> new_ctx = clone_ctx(ctx);

    // 2. 对新输入分词
    auto token_ids = bpe->encode(input_text, false, false);
    int last_token_id = token_ids.back();
    token_ids.pop_back();

    // 3. 从 position_id 开始生成 RoPE
    int current_pos = new_ctx->position_id;
    generate_rope_embed_cache(token_ids.size(), rope_head_dim, current_pos, ...);
    new_ctx->position_id += token_ids.size();

    // 4. 新输入 -> embedding
    ncnn::Mat input_ids_mat = ncnn::Mat((int)token_ids.size(), 1, (void*)token_ids.data()).clone();
    ncnn::Mat token_embed;
    ncnn::Extractor ex = embed_net->create_extractor();
    ex.input("in0", input_ids_mat);
    ex.extract("out0", token_embed);

    // 5. 构建因果掩码（能看到旧 cache + 新 token）
    ncnn::Mat mask(token_ids.size() + new_ctx->kv_cache[0].first.h,
                   token_ids.size());
    mask.fill(0.0f);
    for (int i = 0; i < token_ids.size(); i++) {
        float* row = mask.row(i);
        for (int j = new_ctx->kv_cache[0].first.h + i + 1;
             j < token_ids.size() + new_ctx->kv_cache[0].first.h; j++) {
            row[j] = -1e38f;
        }
    }

    // 6. Decoder 前向（传入旧 KV Cache）
    ncnn::Extractor ex = decoder_net->create_extractor();
    ex.input("in0", token_embed);
    ex.input("in1", mask);
    ex.input("in2", cos_cache);
    ex.input("in3", sin_cache);

    // 传入旧的 KV Cache
    for (int i = 0; i < attn_cnt; i++) {
        snprintf(kname, sizeof(kname), "cache_k%d", i);
        snprintf(vname, sizeof(vname), "cache_v%d", i);
        ex.input(kname, new_ctx->kv_cache[i].first);
        ex.input(vname, new_ctx->kv_cache[i].second);
    }

    // 提取更新后的 KV Cache
    for (int i = 0; i < attn_cnt; i++) {
        snprintf(kname, sizeof(kname), "out_cache_k%d", i);
        snprintf(vname, sizeof(vname), "out_cache_v%d", i);
        ex.extract(kname, k_cache);
        ex.extract(vname, v_cache);
        new_ctx->kv_cache[i] = { move(k_cache), move(v_cache) };
    }

    // 7. 最后一个 token 单独处理
    // 8. Projection -> logits -> argmax
    // 9. 更新 cur_token 和 position_id

    return new_ctx;
}
```

### 典型多轮对话交互

```
用户: "你好"
  -> prefill("你好") -> ctx1
  -> generate(ctx1) -> 回复"你好！有什么可以帮你的？"

用户: "你能做什么？"
  -> prefill("你能做什么？", ctx1) -> ctx2  // 复用 KV Cache!
  -> generate(ctx2) -> 回复"我可以..."
```

---

## 14. 附录：数据形状速查表

### 各阶段数据形状

```
Phase          | 输入形状                              | 输出形状
---------------|--------------------------------------|------------------------
Tokenization   | "你好" (string)                      | [seq_len] (int[])
RoPE           | seqlen, head_dim=128                  | cos: [head_dim/2, seqlen]
               |                                      | sin: [head_dim/2, seqlen]
Embedding      | [seq_len] (token IDs)                | [seq_len, hidden_dim]
Attention Mask | seq_len x seq_len                    | [seq_len, seq_len] (float)
Decoder        | in0: [seq_len, hidden_dim]           | out0: [seq_len, hidden_dim]
               | in1: [seq_len, seq_len] (mask)       | cache_k/v: [head_dim, seq_len, num_kv_heads]
               | in2: [head_dim/2, seqlen] (cos)      |
               | in3: [head_dim/2, seqlen] (sin)      |
Projection     | [seq_len, hidden_dim]                | [vocab_size]
Sampling       | logits: [vocab_size]                 | next_token_id (int)
Decode         | [token_id] (int)                     | "token" (string)
```

### KV Cache 形状速查

```
阶段              | K/V Cache 形状 (每层)                   | 说明
------------------|----------------------------------------|------
Prefill 后 (N tok)| (128, N, 8)                             | 8 = num_kv_heads
Generate t 步后   | (128, N+t, 8)                           | h 维度持续增长
Multi-turn (累计) | (128, total_tokens, 8)                  | 包含所有历史
```

### 关键参数总结（Qwen3 0.6B）

| 参数 | 值 | 说明 |
|------|-----|------|
| vocab_size | ~151936 | 词表大小 |
| hidden_dim | 1024 | 隐藏层维度 |
| attn_cnt | 28 | Transformer 层数 |
| rope_head_dim | 128 | RoPE 每个 head 的维度 |
| rope_theta | 100000.0 | RoPE 基频 |
| num_heads | 16 | Q 注意力头数 |
| num_kv_heads | 8 | KV 注意力头数（GQA） |
| head_dim | 64 | 每个头的维度 (hidden_dim/num_heads) |

### 关键文件对照

| 文件 | 作用 |
|------|------|
| `model.json` | 模型配置入口 |
| `vocab.txt` | BPE 词表文件 |
| `merges.txt` | BPE 合并规则 |
| `qwen3_embed_token.ncnn.param` | Embedding 网络结构 |
| `qwen3_embed_token.ncnn.bin` | Embedding 权重 |
| `qwen3_decoder.ncnn.param` | Decoder 网络结构 |
| `qwen3_decoder.ncnn.bin` | Decoder 权重 |
| `qwen3_proj_out.ncnn.param` | Projection 网络结构 |
| `sdpa.cpp` | SDPA 层实现（KV Cache 拼接核心） |

### 推理时序总结

```
用户输入 -> Prompt Template -> BPE Tokenize
                                     |
                                     v
                              RoPE cos/sin Cache
                                     |
                                     v
                              Embedding (embed_net)
                                     |
                                     v
                              Causal Mask
                                     |
                                     v
                              Decoder (decoder_net) -> KV Cache
                                     |
                                     v
                              Projection (proj_out_net) -> logits
                                     |
                                     v
                              Sampling -> next_token_id
                                     |
                                     v
                              Token -> Text (BPE decode)
                                     |
                          +----------+
                          v          |
                    流式输出    是否 EOS/超限？
                          |          |
                          v          v
                          YES -> 结束
                          NO  -> 回到 Embedding（自回归）
```

---

## 15. Prefill vs Decode 性能分析

LLM 推理分为两个截然不同的阶段，计算特征和性能瓶颈完全不同。

### Prefill 阶段：计算密集型

Prefill 阶段一次性处理输入序列的所有 S 个 token，总计算量为 O(S² × d × L)。

- 所有 S 个 token 可以批量处理——矩阵乘法规模大且密集
- GPU 并行度、SIMD 指令、FlashAttention 等技术可以充分利用
- **瓶颈在计算**（Compute-bound），GPU 算力是限制因素

### Decode 阶段：内存带宽密集型

Decode 阶段每步仅生成 1 个新 token，但需要访问所有历史 token 的 KV Cache。每步计算量为 O(S × d × L)。

- 仅 1 个新 token 参与计算，矩阵乘法退化为向量-矩阵乘
- 主要瓶颈是加载 KV Cache 和模型权重——计算量很小
- 随着序列增长，KV Cache 越来越大，内存带宽成为瓶颈
- 典型现象：decode 速度随序列长度线性下降

### "最后一个 token 单独处理"模式

当前 prefill 实现采用一种特殊模式：先将 N-1 个 token 批量处理，再将最后一个 token 单独送入 decoder：

1. **批量处理 N-1 个 token**：构建 KV Cache，充分利用密集矩阵运算
2. **单独处理第 N 个 token**：利用已构建的 KV Cache，获得预测下一个 token 的 logits

分开处理是因为：批量处理 N 个 token 时，只有最后一个位置的输出对预测有用，但该 token 需要看到所有前 N-1 个 token 的 KV Cache。分开意味着批量路径是密集计算（Compute-bound），单 token 路径是轻量级操作。

### 性能对比

| 阶段 | 计算量 | 受限于 | 可批处理 | 典型耗时 |
|------|--------|--------|----------|---------|
| Prefill (S tokens) | O(S²×d×L) | Compute | Yes | ~50ms (S=100) |
| Decode (1 token) | O(S×d×L) | Memory | No | ~5ms/token, 逐步变慢 |

### ncnn 平台特性

在 ncnn 实现中，`decoder_net` 启用 Vulkan BF16 加速，对 Prefill 阶段非常有利——大规模矩阵乘法和 FlashAttention 可以充分利用 GPU 并行性。然而 Decode 阶段由于是 Memory-bound，GPU 加速收益有限。`embed_net` 和 `proj_out_net` 始终运行在 CPU 上——它们的计算量不足以抵消 GPU 调度开销。

---

## 16. INT8 量化推理

ncnn 提供两套 decoder 参数文件：

- `qwen3_decoder.ncnn.param`：Gemm 层无 `18=` 参数，FP32 推理
- `qwen3_decoder.ncnn.int8.param`：Gemm 层带 `18=2`，SDPA 层带 `18=2`，启用 INT8 推理

### Gemm INT8 量化

在 INT8 param 文件中，Gemm 层标记 `18=2` 表示权重以 INT8 格式存储，推理时激活值动态量化为 INT8 执行矩阵乘法，结果再反量化回 FP32。

### SDPA INT8 动态量化路径

SDPA 的 INT8 路径在 `sdpa.cpp` 的 `forward_int8()` 中实现，采用**动态量化**策略——每次推理时实时计算量化参数：

**Q（Query）：逐行动态量化**
```cpp
dynamic_quantize_2d_per_h(query_head, query_head_int8, query_head_int8_scales);
```
每个 query 行独立计算缩放因子。Decode 阶段 Q 只有 1 个 token，不同 head 的数值范围可能差异很大。

**K（Key）和 V（Value）：逐张量动态量化**
```cpp
float key_head_int8_scale;
dynamic_quantize_2d(key_head, key_head_int8, key_head_int8_scale);
```
整个 K/V 张量共享一个缩放因子。K/V 跨越整个序列长度，逐张量量化更高效。

**Q×K^T 矩阵乘法：INT8 点积 + 反量化**
```cpp
int sum = 0;
for (int k = 0; k < embed_dim; k++) {
    sum += qptr[k] * kptr[k];           // INT8 点积
}
float qk_descale = 1.f / (query_head_int8_scales[i] * key_head_int8_scale);
float sum_fp32 = sum * qk_descale;      // 反量化回 FP32
```

**Softmax：保持 FP32 精度**

INT8 的精度不足以支撑 softmax 运算（exp 函数对微小差异极为敏感），因此 softmax 必须在 FP32 下执行。这是一个关键的精度保障。

### 为什么 Q 逐行而 K/V 逐张量？

- **Q 逐行**：Decode 时 Q 只有 1 个 token（1 行），各 head 可能呈现不同的数值范围，逐行量化精度更高
- **K/V 逐张量**：K/V 覆盖整个序列长度（可能数百至数千行），逐行量化会引入大量缩放因子增加开销；逐张量量化精度损失可接受

### INT8 与 Vulkan 的互斥

```cpp
// sdpa_vulkan.cpp
if (int8_scale_term) {
    support_vulkan = false;    // INT8 SDPA 禁用 Vulkan 加速
}
```

当前 Vulkan shader 尚未实现 INT8 量化路径——FlashAttention shader 仅支持 FP16/BF16。因此 INT8 推理只能在 CPU 上执行，这是速度和内存之间的权衡。

### 内存节省

| 精度 | 权重大小 | 说明 |
|------|---------|------|
| FP32 | ~1.2 GB | 28 层 decoder 权重 |
| INT8 | ~300 MB | 权重缩小为 1/4 |

对于内存受限的嵌入式设备（如 ARM 开发板），INT8 量化是部署 LLM 的必要手段。

---

## 17. Vulkan GPU 加速与 FlashAttention

### Vulkan 配置策略

```cpp
// ncnn_llm_gpt.cpp
if (use_vulkan) {
    // 仅 decoder_net 启用 Vulkan
    decoder_net->opt.use_bf16_storage = true;     // BF16 存储
    decoder_net->opt.use_fp16_arithmetic = false;  // 算术仍为 FP32
    decoder_net->opt.use_vulkan_compute = true;    // 启用 Vulkan 计算
}
// embed_net 和 proj_out_net 始终运行在 CPU
```

关键设计决策：
- **只有 decoder_net 使用 Vulkan**：Embedding 查找和最终投影的计算量不足以抵消 GPU 调度开销
- **BF16 存储而非 FP16**：保留与 FP32 相同的指数范围（8 位），避免溢出/下溢
- **算术保持 FP32**：存储用 BF16 节省显存/带宽，运算精度仍为 FP32

### FlashAttention Shader

FlashAttention 将 Q×K + Softmax + Attn×V 融合为单个 GPU kernel，避免在显存中物化完整的 S×S Attention 矩阵。

**共享内存分块计算**：

```
分块参数: M=4, N=32, K=32
每次处理 4 行 Query、32 行 Key/Value、32 维 Embedding

Online Softmax 技术:
  smem_row_max[M]     -- 每行最大值
  smem_row_sum[M]     -- 每行指数和
  smem_correction[M]  -- 校正因子（处理新最大值时的修正）

当处理新的 K 分块时，如果发现新的最大值:
  correction = exp(old_max - new_max)
  O = O * correction + new_partial_sum
```

**避免物化完整 Attention 矩阵**：对于 S=1000 的序列，传统方法需要 1000×1000 的 FP32 矩阵（4MB），FlashAttention 只需要 4×32 的分块（512B）。

### Cooperative Matrix 变体

利用 GPU 硬件加速的 Cooperative Matrix 运算，提供更高的矩阵乘法吞吐：

- 支持 KHR 标准（`GL_KHR_cooperative_matrix`）和 NVIDIA 专有扩展（`GL_NV_cooperative_matrix`）
- 激活条件：GPU 支持 `support_cooperative_matrix()` 且启用 FP16/BF16 存储
- 适用于现代 GPU（NVIDIA Ampere+、AMD RDNA3+、Intel Arc）

### BF16 vs FP16

| 格式 | 指数位 | 尾数位 | 数值范围 | 精度 |
|------|--------|--------|----------|------|
| FP32 | 8 | 23 | ±3.4e38 | 高 |
| BF16 | 8 | 7 | ±3.4e38 | 低 |
| FP16 | 5 | 10 | ±65504 | 中 |

**选择 BF16 的原因**：LLM 权重和激活值的数值范围较宽，FP16 的 5 位指数容易溢出/下溢。BF16 保留与 FP32 相同的指数范围，牺牲尾数精度但对推理影响有限。

### x86 CPU 优化路径

在不使用 Vulkan 的 x86 平台上：
- Q×K^T 和 Attn×V：使用 ncnn Gemm 层（可链接 BLAS 库如 OpenBLAS、MKL）
- 并行策略：OpenMP 按 head 并行
- 编译变体：AVX、FMA、AVX512 等 SIMD 指令集自动选择

---

## 18. Tool Calling 完整生命周期

Tool Calling（工具调用）允许 LLM 在生成过程中调用外部函数获取信息，然后基于函数返回结果继续生成。这是 Agent 应用和 RAG 系统的核心能力。

### 完整生命周期

```
  ┌──────────────────────────────────────────────────────────┐
  │  1. define_tools(ctx, tools, system_prompt)              │
  │     -> 工具定义注入 system prompt                         │
  │                                                            │
  │  2. 用户提问 -> prefill(input_text)                       │
  │     -> 构建 KV Cache，获得首个预测 token                   │
  │                                                            │
  │  3. 模型生成 -> 遇到 tool_call_id token                   │
  │     -> 进入工具调用模式                                     │
  │                                                            │
  │  4. 累积 tool_call_content 直到 tool_call_end_id          │
  │     -> 解析 JSON 获取函数名和参数                          │
  │                                                            │
  │  5. 调用 tool_callback 函数                                │
  │     -> 获取工具返回结果                                     │
  │                                                            │
  │  6. 构造 tool_response 字符串                              │
  │     -> 格式化为 ChatML 的 user 消息                       │
  │                                                            │
  │  7. prefill(tool_response, ctx)                            │
  │     -> 模型在新上下文上继续生成                             │
  │                                                            │
  │  8. 最终响应输出                                           │
  └──────────────────────────────────────────────────────────┘
```

### 工具定义注入

工具定义通过 XML 风格的 `<tools>...</tools>` 块注入到 system prompt 中：

```
<|im_start|>system
{system_prompt}

# Tools

You may call one or more functions to assist with the user query.

<tools>
{"name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}
</tools>

For each function call, return a json object within [TOOL_CALL] tags:
[TOOL_CALL]{"name": <function-name>, "arguments": <args-json-object>}[/TOOL_CALL]
<|im_end|>
```

### generate() 中的工具调用检测

```cpp
if (ctx->cur_token == tool_call_id) {
    flag_in_tool_call = true;               // 进入工具调用模式
} else if (ctx->cur_token == tool_call_end_id) {
    flag_in_tool_call = false;
    handle_tool(tool_call_content, ctx);    // 处理工具调用
    tool_call_content.clear();
    history.clear();                        // 清除重复惩罚历史
    continue;                               // 跳过正常 token 输出
} else if (flag_in_tool_call) {
    tool_call_content += bpe->decode({ctx->cur_token}, false);  // 累积工具调用内容
} else {
    callback(bpe->decode({ctx->cur_token}, false));             // 正常流式输出
}
```

### tool_response 格式构造

工具返回结果后，格式化为新的上下文送入模型：

```cpp
std::string tool_response_pre  = "<|im_end|>\n<|im_start|>user\n[TOOL_RESULT]\n\n";
std::string tool_response_post = "\n\n[/TOOL_RESULT]\n<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n\n";
```

注意 `tool_response_post` 中的 `<think>\n</think>`：这强制模型在回答前再次进入思考模式，确保模型基于工具结果进行推理后再生成最终回复。

### history.clear() 的原因

工具调用后清除重复惩罚历史，因为上下文已经发生根本变化——工具结果引入了全新的 token，不应基于调用前的统计进行惩罚。

### Thinking 模式与工具调用的交互

Qwen3 的 `<think>` / `</think>` token 与工具调用交互：
- 模型可以**先思考再决定**调用哪个工具
- 收到工具结果后，`<think>\n</think>` 触发又一次思考阶段
- 这种多步骤 think -> call tool -> think about result -> final answer 模式支持复杂推理链

---

## 19. Vision 多模态推理

### 图像嵌入注入

Vision 推理的核心挑战是将图像特征融入文本 token 序列。`inject_image_embeds()` 函数（`rope_embed.cpp`）实现：

1. 在 token_ids 序列中找到 `<|image_pad|>` 占位符
2. 用 `image_embeds.h` 个 patch embedding 向量替换该占位符
3. 调整周围的 token_ids 和 token_embed 维度

序列长度变化：N 个原始 token → N-1（去掉占位符）+ num_patches（图像 patch 数量）。

### mRoPE（多维旋转位置编码）

传统文本 RoPE 将所有维度视为单一时间序列，但图像 token 需要空间位置信息。mRoPE 将位置编码分为三个维度：

- **Temporal（时间）**：文本 token 按顺序排列的位置
- **Height（高度）**：图像 patch 在图像中的行坐标
- **Width（宽度）**：图像 patch 在图像中的列坐标

`mrope_section` 参数定义维度边界，例如 `[11, 11, 10]` 表示：
- 第 0~10 维：Temporal
- 第 11~21 维：Height
- 第 22~31 维：Width

### 交错 mRoPE（Qwen3.5-VL）

Qwen3.5-VL 使用**交错 mRoPE**（Interleaved mRoPE），维度分配不是连续块，而是按 modulo-3 模式交替分配：

```cpp
// generate_rope_embed_cache_vision_mrope_interleaved()
int which_pos = 0;
if (j < mrope[1] * 3 && (j % 3 == 1)) {
    which_pos = 1;   // Height
} else if (j < mrope[2] * 3 && (j % 3 == 2)) {
    which_pos = 2;   // Width
}
// which_pos == 0: Temporal
```

这种交错分配使得每个维度在 RoPE 的频率空间中均匀分布，可能有助于模型更好地学习跨维度的位置关系。

### Position ID 增长

对于包含图像的序列，position_id 的增长不再是简单的 +1：

```cpp
if (image_embeds.empty()) {
    new_ctx->position_id += token_ids.size();
} else {
    new_ctx->position_id += token_ids.size() - image_embeds_size + (num_patches_w / spatial_merge_size);
}
```

图像 token 消耗的 position slot 少于其实际数量，因为 spatial merge 将多个 patch 合并。这确保后续文本 token 的位置编码正确衔接图像区域。

### 两种视觉模型类型

| 类型 | model.json 配置 | 位置编码 | 特点 |
|------|----------------|----------|------|
| VISION_VIT | `"type": "vit"` | 标准 mRoPE | 标准 ViT 编码器，支持窗口注意力 |
| VISION_QWEN3_5_VL | `"type": "qwen3.5_vl"` | 交错 mRoPE | 专有视觉编码器，更复杂的空间合并策略 |
