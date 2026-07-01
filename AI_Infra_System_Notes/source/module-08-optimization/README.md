# Module 8: 优化技术——让每一焦耳都花在刀刃上

> 推理优化的本质不是"让计算更快"，而是**让数据少搬一次、让显存少占一点、让精度少丢一点**。本模块覆盖量化的完整链路、FlashAttention 的显存魔法、图优化和 Profiling 驱动的性能调优。

---

## 📋 学习目标

- [ ] 能手写对称量化公式，算一个具体例子
- [ ] 能画出 ncnn 的 Quantize→Conv INT8→Requantize 三级流水线
- [ ] 能解释 AWQ ("保护重要通道") 和 GPTQ ("误差补偿") 的核心差异
- [ ] 能画出 FlashAttention 的分块计算图
- [ ] 能推导 Conv-BN 融合公式
- [ ] 能根据 profiling 数据诊断瓶颈

---

## 1. 量化——给模型"瘦身"

### 1.1 对称量化的数学

```
量化:   q = round(x / s)         s = max(|x|) / 127
反量化:  x' = q × s

例子: 权重 [−0.8, 1.2, 0.3, −0.5]
  max_abs = 1.2 → s = 1.2/127 ≈ 0.00945
  −0.8 → round(−84.7) = −85
   1.2 → round(127.0) = 127
   0.3 → round( 31.7) =  32
  反量化误差 < 0.005 — 几乎无感
```

### 1.2 ncnn INT8 三级流水线

```
FP32 → Quantize(×scale_in) → INT8 Conv(INT8×INT8→INT32) → Requantize(×scale_in×scale_out) → INT8

Requantize 是核心:
  float v = int32_accum[i] × scale_in + bias;   // INT32→FP32
  v = activation(v);                              // 在FP32精度下激活!
  output_int8[i] = round(v × scale_out);          // FP32→INT8

为什么 scale_in + scale_out？
  scale_in:  INT32 累加结果的 dequantize → 还原为 FP32
  scale_out: 下一层期望的输入 scale → 重新量化为 INT8
```

### 1.3 AWQ vs GPTQ——两种哲学

```
AWQ: "找到重要通道，保护它"
  分析激活分布 → 识别 salient channels → 缩放变换 → 量化
  灵感: 有些通道的激活值特别大，这些通道对应的权重对输出影响大
  速度: 快（几分钟）

GPTQ: "量化一列，补偿一列"
  逐列量化权重 → 量化误差实时补偿到未量化列
  需要 Hessian 矩阵 → 更慢但更精确
  速度: 慢（几十分钟）

选 AWQ 还是 GPTQ？
  → 快速部署 → AWQ
  → 极致精度 → GPTQ
  → vLLM 和 llama.cpp 都支持两者
```

---

## 2. FlashAttention——"不写回显存"的魔法

### 2.1 问题的残酷真相

```
标准 Attention 的内存占用:
  S = Q·K^T:    [seq, seq] × 4 bytes
  P = softmax:  [seq, seq] × 4 bytes

seq=2048:  2048² × 4 = 16 MB — 还好
seq=8192:  8192² × 4 = 268 MB — 开始吃力
seq=32768: 32768² × 4 = 4.3 GB！— 单层单头就这么大

28 层 × 16 heads = 28 × 16 × 4.3 GB = 1.9 TB!
这还没考虑 P 矩阵。显存根本不够。

更致命: S 和 P 先写入 HBM，再读回来做下一步计算
→ HBM 带宽被大量浪费在"存了读、读了存"上
```

### 2.2 FlashAttention 的三个绝招

```
绝招 1: Tiling (分块)
  不一次性生成 [seq,seq] 的 S
  切成小块在 SRAM 中算完即丢
  → S 和 P 从不整体出现在 HBM 中！

绝招 2: Online Softmax
  传统: 先找全局 max → 再 exp → 再 sum → 再 div (需全量数据)
  Flash: 每个 block 边算边更新:
    m_new = max(m_old, block_max)
    修正旧结果: O_old *= exp(m_old - m_new)
    累加新结果: O += P_block · V_block
  → 数学等价于标准 softmax，但不需要全局同步

绝招 3: Recomputation (仅训练)
  前向不存 S 和 P → 反向时重算
  → 省掉最大的显存开销

效果: 显存 O(seq²) → O(seq)
      seq=8192: 268 MB → ~1 MB (per head per layer)
```

---

## 3. 图优化——编译器的自动魔法

```
六大优化 Pass:

Constant Folding:   3×4 → 12 (加载时算好)
Dead Code:          删除输出不用的分支
Operator Fusion:    Conv+BN+ReLU → 一个 kernel (省读写+launch)
CSE:                a×b 算两次 → 算一次复用
Layout:             取消不必要的 NCHW↔NHWC 转换
Memory Planning:    生命周期不重叠的 buffer 共享内存
```

### Conv-BN 融合——最经典的融合

```
BN 推理时是线性函数 (running mean/var 固定):
  BN(x) = γ(x−μ)/σ + β = αx + b'

融合: W' = αW,  b' = αb + β'
→ 加载模型时一次性算好
→ 推理时零 BN 开销
```

---

## 4. Profiling——用数据说话

| 现象 | 看什么 | 方向 |
|------|--------|------|
| 首 token 慢 | TTFT vs prompt_len | Prefix cache, Chunked prefill |
| 后续 token 慢 | TPOT vs seq_len | FlashAttn, KV 量化 |
| GPU 利用率 <30% | batch size | Continuous Batching |
| 显存爆 | KV usage | PagedAttention, KV 量化 |
| INT8 精度差 | Layer-wise error | Per-channel, AWQ/GPTQ |

---

*下一模块: [Module 9: 部署实战](../module-09-deployment/README.md)*
