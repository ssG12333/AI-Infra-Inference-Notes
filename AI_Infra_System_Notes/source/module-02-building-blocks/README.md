# Module 2: 构建算子——从单元素到空间

> 第一章的算子处理逐元素或全连接关系。本章引入**空间维度**的处理——卷积捕获局部模式，注意力建立全局依赖。

---

## 📋 学习目标

- [ ] 理解卷积的 6 个核心参数 (kernel/stride/padding/dilation/groups/bias)
- [ ] 能计算任意配置的卷积输出尺寸和感受野
- [ ] 理解 im2col+GEMM / Winograd / FFT / Direct 四种算法的选择策略
- [ ] 能解释 NCHW vs NHWC 的内存布局差异
- [ ] 理解 SDPA 的 5 步计算流程
- [ ] 能画 GQA 的 ExpandDims+Tile 扩展机制
- [ ] 理解 RoPE 的核心原理：旋转角度差 = 相对位置
- [ ] 能对比 LSTM 和 GRU 的门控差异

---

## 1. 线性变换

### 1.1 全连接层 (InnerProduct / Linear)

```
y = W·x + b
输入: [N, in_features]   输出: [N, out_features]
权重: [out_features, in_features]   偏置: [out_features]

参数量 = out_features × in_features + out_features
```

### 1.2 Bias (偏置加法) 💡

```
y = x + b    (b 是 [C] 的可学习偏置)

⚠️ Conv+BN 时 bias=False —— BN 的 β 已经起到偏置作用。
   现代网络中紧跟 BN 的卷积几乎总是关闭 bias。
```

---

## 2. 卷积 (Conv2D) 🔬

### 2.1 核心概念

```
输入: [B, C_in, H, W]  →  输出: [B, C_out, H', W']
权重: [C_out, C_in/groups, kH, kW]

卷积 vs 全连接的关键差异:
  1. 局部连接: 只看 k×k 邻域 (先验: 近邻更相关)
  2. 权重共享: 同一个滤波器滑过整个空间 (先验: 平移等变性)
  3. 参数量: 从 O(C²HW) 降到 O(C²k²)
```

### 2.2 六个核心参数

#### ① kernel_size (卷积核大小)

```
k=1 (1×1):  只做通道变换
k=3 (3×3):  最常用 ← 两个 3×3 叠加 = 一个 5×5 的感受野
k=5 (5×5):  较大感受野，现多用两个 3×3 替代
k=7 (7×7):  早期网络 stem
```

#### ② stride (步幅)

```
stride=1:  输出尺寸不变 (配合 pad=(k-1)/2)
stride=2:  输出尺寸减半 (替代 MaxPool 下采样)
stride=patch_size:  极大下采样 (ViT PatchEmbed, stride=14/16)
```

#### ③ padding (填充)

```
padding=0 (valid):   不填充，输出缩小
padding=(k-1)/2 (same): 填充后输出不变
```

#### ④ dilation (空洞率)

```
dilation=1 (标准):   ┌─┬─┬─┐       dilation=2 (空洞): ┌─┬─┬─┬─┬─┐
                     │●│●│●│                           │●│○│●│○│●│
                     ├─┼─┼─┤                           ├─┼─┼─┼─┼─┤
                     │●│●│●│                           │○│○│○│○│○│
                     ├─┼─┼─┤                           ├─┼─┼─┼─┼─┤
                     │●│●│●│                           │●│○│●│○│●│
                     └─┴─┴─┘                           └─┴─┴─┴─┴─┘
                     感受野: 3×3                        感受野: 5×5

有效核大小: k_eff = dilation × (k−1) + 1
dilation=1, k=3 → k_eff=3;  dilation=2, k=3 → k_eff=5
代表: DeepLab ASPP (dilation=1,6,12,18 并行提取多尺度)
```

#### ⑤ groups (分组数)

```
groups=1:    标准卷积           参数 C_out·C_in·k²
groups=G:    分组卷积           参数 C_out·C_in/G·k²   (1/G 参数)
groups=C_in: 深度可分离卷积      参数 C_in·1·k²         (最小)
```

#### ⑥ bias

```
Conv 后接 BN  → bias=False (BN 的 β 已等效)
Conv 不接 BN  → bias=True
```

### 2.3 输出尺寸公式

```
H_out = ⌊(H_in + 2×pad − dilation×(kernel−1) − 1) / stride + 1⌋
```

### 2.4 卷积算法对比 🔬

| 算法 | 核心思想 | 适用场景 | 优点 | 缺点 |
|------|---------|---------|------|------|
| **im2col+GEMM** | 滑动窗口展开为矩阵列 | 通用卷积 k≥3 | 复用 GEMM kernel | im2col 内存膨胀 k² 倍 |
| **Winograd** | 变换域点乘替代卷积 | 3×3, stride=1 | 比 im2col 快 1.5-2× | 数值精度略损 |
| **FFT** | 频域点乘 = 空域卷积 | k≥7 大核 | O(n log n) | 现代网络极少大核 |
| **Direct** | 逐位置直接计算 | DWConv, dilation | 无额外内存 | 不如 GEMM 高效 |

**ncnn 的选择**：
- 3×3 stride=1 → Winograd
- 1×1 → Direct (特殊路径)
- k≥3 通用 → im2col+GEMM
- groups=C → Direct (Depthwise)
- dilation>1 → Direct

### 2.5 卷积变体

| 变体 | 思想 | 参数量 | 代表 |
|------|------|--------|------|
| DepthwiseConv | 每通道独立卷积 | C·k² | MobileNet |
| PointwiseConv (1×1) | 只做通道变换 | C_out·C_in | ResNet 瓶颈 |
| SeparableConv | Depthwise + Pointwise | C·k² + C_out·C_in | MobileNet |
| DilatedConv | 空洞扩大感受野 | 同标准卷积 | DeepLab |
| TransposedConv | 上采样（反卷积） | C_in·C_out·k² | U-Net/GAN |
| DeformableConv | 学习采样偏移 | 额外 2K 偏移参数 | DCN/DETR |

### 2.6 感受野计算

```
单层:  RF = dilation × (kernel − 1) + 1

多层累积:
  RF₁ = k₁
  RFₗ = RFₗ₋₁ + (kₗ − 1) × ∏ᵢ₌₁ˡ⁻¹ strideᵢ

示例: 3 个 stride=1 的 3×3 卷积
  RF₁ = 3,  RF₂ = 3+2×1=5,  RF₃ = 5+2×1=7
  → 等效于 1 个 7×7 卷积，但参数量 3×9=27 < 49
```

---

## 3. 内存布局 (Memory Layout) 📖

### 3.1 NCHW vs NHWC

```
NCHW (Channel-Major, PyTorch/ncnn 默认):
  数据按通道连续: [N₀C₀H₀W₀, N₀C₀H₀W₁, ..., N₀C₀H₁W₀, ..., N₀C₁...]

  内存: ╰── 通道 0 全部 ──╯╰── 通道 1 全部 ──╯
  优点: BN 统计量连续, ncnn Conv/DWConv 优化最好
  缺点: Per-pixel 操作需要跨步访问 (cache miss)

NHWC (Spatial-Major, TF/TFLite 默认):
  数据按空间位置连续: [N₀H₀W₀C₀, N₀H₀W₀C₁, ..., N₀H₀W₁C₀...]

  内存: ╰── 一个位置的所有通道 ──╯╰── 下一个位置 ──╯
  优点: Per-pixel 操作友好, Mobile/嵌入式通常更快
  缺点: BN 需要跨步统计
```

### 3.2 硬件偏好

| 硬件 | 偏好 | 原因 |
|------|------|------|
| x86 CPU (AVX) | NCHW | 通道连续, SIMD 按通道向量化 |
| ARM CPU (NEON) | NCHW 或 NHWC | 取决于实现 |
| Mobile GPU | NHWC | texture 采样按像素 |
| NVIDIA GPU | NCHW 或 NHWC | TensorCore 偏好 NCHW |
| Vulkan | NHWC | compute shader 按 tile 分组 |

---

## 4. 池化：空间压缩

| 池化 | 操作 | 输入→输出 | 用途 |
|------|------|-----------|------|
| MaxPool | 取窗口最大值 | [B,C,H,W]→[B,C,H/2,W/2] | VGG/ResNet |
| AvgPool | 取窗口平均值 | 同上 | 更平滑的下采样 |
| GlobalAvgPool | 全局平均 | [B,C,H,W]→[B,C,1,1] | SENet, ResNet 末端 |
| AdaptiveAvgPool | 指定输出大小 | 任意→[B,C,H',W'] | 分类头 |
| RoIAlign | 双线性插值 RoI | 特征图+RoI→[K,C,7,7] | Mask R-CNN |

> 💡 GAP 替代 FC：ResNet 用 GlobalAvgPool 将 [B,C,7,7] 压缩为 [B,C,1,1] 再分类，参数从 4096×C 降到 C×num_classes。

---

## 5. 注意力：全局依赖的建立 🔬

### 5.1 SDPA (缩放点积注意力)

```
Attention(Q, K, V) = softmax(Q·Kᵀ / √dₖ) · V

逐步拆解:
  Step 1: Q·Kᵀ        → 点积衡量 Q 与每个 K 的相似度
  Step 2: / √dₖ       → 缩放防数值过大 (dₖ=128 → scale≈0.088)
  Step 3: + mask       → 因果掩码: 未来位置设为 −∞
  Step 4: softmax      → 归一化为概率分布 (注意力权重)
  Step 5: · V          → 加权求和得到输出

Q: [batch, heads, seq_q, dₖ]
K: [batch, heads, seq_k, dₖ]
V: [batch, heads, seq_k, dᵥ]
```

### 5.2 多头注意力: MHA vs GQA vs MQA

```
MHA: 16 Q heads × 16 KV heads    → KV Cache: 16 份
GQA: 16 Q heads × 8 KV heads     → KV Cache: 8 份 (省 50%)
MQA: 16 Q heads × 1 KV head      → KV Cache: 1 份 (省 93.75%)

Qwen3-0.6B: GQA, num_q_heads=16, num_kv_heads=8
→ ExpandDims + Tile 将 8 KV heads 复制为 16
```

### 5.3 RoPE (旋转位置编码) 💡

```
核心原理: 通过旋转 Q/K 向量编码位置

(x₂ᵢ, x₂ᵢ₊₁) → (x₂ᵢcosθ − x₂ᵢ₊₁sinθ, x₂ᵢsinθ + x₂ᵢ₊₁cosθ)
θ = pos / base^(2i/d)

精妙之处:
  Q_i · K_j = (R_i·q_i) · (R_j·k_j)
            = q_i · (R_iᵀ · R_j) · k_j
            = q_i · R_{j-i} · k_j
            ↑ 只依赖相对位置 (j-i)！
```

### 5.4 FlashAttention

```
传统: Q·Kᵀ → [seq,seq] → softmax → ·V    显存 O(seq²)
Flash: 分块处理 → online softmax → 累积    显存 O(seq)

三个技巧:
  1. Tiling:  切小块在 SRAM 中计算
  2. Online Softmax: 每块边算边更新 max/sum
  3. Recomputation: 反向时重算 attention (不存 S/P)

效果: seq=8192 → 268 MB → ~1 MB (per head per layer)
```

---

## 6. LSTM 与 GRU 📖

### LSTM
```
fₜ = σ(Wf·[hₜ₋₁, xₜ] + bf)    # 遗忘门: 丢弃多少旧记忆
iₜ = σ(Wi·[hₜ₋₁, xₜ] + bi)    # 输入门: 接收多少新信息
c̃ₜ = tanh(Wc·[hₜ₋₁, xₜ] + bc) # 候选记忆
cₜ = fₜ⊙cₜ₋₁ + iₜ⊙c̃ₜ          # 更新长期记忆
oₜ = σ(Wo·[hₜ₋₁, xₜ] + bo)    # 输出门
hₜ = oₜ⊙tanh(cₜ)               # 输出

共 4 个门控, 长期记忆(cₜ) + 短期输出(hₜ)
```

### GRU
```
rₜ = σ(Wr·[hₜ₋₁, xₜ])          # 重置门
zₜ = σ(Wz·[hₜ₋₁, xₜ])          # 更新门 (合并遗忘+输入门)
h̃ₜ = tanh(Wh·[rₜ⊙hₜ₋₁, xₜ])    # 候选
hₜ = (1−zₜ)⊙hₜ₋₁ + zₜ⊙h̃ₜ       # 输出

共 2 个门控, 无独立细胞态, 参数量 ≈ LSTM 的 2/3
```

---

## 7. 嵌入：离散→连续

```
Embedding = Gather 操作
  token_id → embedding_matrix[token_id] → [hidden_dim]
  本质: 从 [vocab_size, hidden_dim] 矩阵中取一行

位置编码:
  可学习:  Embed([seq_len, hidden_dim]) — BERT 用
  正弦:    PE(pos,2i)=sin(pos/10000^(2i/d)) — 原始 Transformer
  RoPE:    旋转编码 — 现代 LLM 标配
  ALiBi:   加偏置，无需位置编码 — BLOOM 用
```

---

## 🛠️ 动手练习

1. **卷积参数计算**: 输入 [1,3,224,224], Conv2D(k=3,s=2,p=1), C_out=64。求输出 shape 和参数量。

2. **感受野计算**: 三个 Conv3×3(s=1) 堆叠 vs 一个 Conv7×7，感受野各是多少？参数量呢？

3. **SDPA 手算**: 给定简化的 Q=[1,0], K=[[1,0],[0,1]], V=[[1,2],[3,4]]，手算 attention 输出。

4. **RoPE 验证**: 对 2D 向量 (x,y)，验证旋转 θ₁ 再旋转 θ₂ = 直接旋转 (θ₁+θ₂)。

---

*下一模块: [Module 3: 组合结构](../module-03-architectures/README.md)*
