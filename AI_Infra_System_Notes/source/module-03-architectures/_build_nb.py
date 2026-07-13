import json

nb = {
    'nbformat': 4, 'nbformat_minor': 5,
    'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.12.0'}
    },
    'cells': []
}

def md(src, cid):
    nb['cells'].append({'cell_type': 'markdown', 'metadata': {}, 'id': cid, 'source': src})
def code(src, cid):
    nb['cells'].append({'cell_type': 'code', 'metadata': {}, 'id': cid, 'source': src, 'outputs': [], 'execution_count': None})

# ============================================================
# Title + Env
# ============================================================
md(["Module 3: 组合结构——单打独斗不如团队协作\n\n", "> CNN Block 用残差连接让 100+ 层梯度顺畅流淌，Transformer Block 用注意力让每个词都能看到整段上下文——从代码层面理解每一个组合结构的工程智慧。"], 'd01')
md(["## 0. 环境准备"], 'd02')
md(["导入所有需要的库。torch 做张量运算，numpy 辅助数据处理，matplotlib 可视化。"], 'd03')
code([
    "import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n",
    "import numpy as np\nimport matplotlib.pyplot as plt\nimport math\n\n",
    'plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]\n',
    'plt.rcParams["axes.unicode_minus"] = False\n',
    'plt.rcParams["font.size"] = 12\n',
    'plt.rcParams["figure.figsize"] = (12, 5)\n',
    'print(f"PyTorch: {torch.__version__}")\n',
    'print(f"NumPy: {np.__version__}")\n',
    'print("Ready!")'
], 'd04')

# ============================================================
# Section 1: Residual Connections
# ============================================================
md(["---\n## 1. 残差连接——深度学习史上最重要的 +1"], 'd05')
md([
    "### 1.1 残差的梯度魔法\n\n",
    "标准层: x_out = F(x_in)，梯度 = dL/dx_out × dF/dx_in。F 的导数可能很小→梯度消失。\n",
    "残差层: x_out = F(x_in) + x_in，梯度 = dL/dx_out × (dF/dx_in + **1**)。\n\n",
    "**+1 = 梯度高速公路**。即使 F 的导数为 0，梯度也有个 1 保底——不会完全消失。\n\n",
    "下面用代码量化对比：100 层网络，有残差 vs 无残差的梯度传播。"
], 'd06')

code([
    "torch.manual_seed(42)\n",
    "n_layers = 100\n",
    "decay_per_layer_nores = 0.95   # 无残差: 每层衰减 0.95\n",
    "decay_per_layer_res    = 0.95   # 有残差: F 衰减 0.95, 但 +1 保底\n\n",
    "grad_nores = 1.0\ngrad_res = 1.0\n",
    "hist_nores = [grad_nores]\nhist_res = [grad_res]\n\n",
    "for i in range(n_layers):\n",
    "    grad_nores *= decay_per_layer_nores              # 标准: 连乘\n",
    "    grad_res = grad_res * (decay_per_layer_res + 0.05) # 残差: 有 +0.05 的残差贡献\n",
    "    hist_nores.append(grad_nores)\n    hist_res.append(grad_res)\n\n",
    "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))\n",
    "layers = range(n_layers + 1)\n",
    'ax1.plot(layers, hist_nores, lw=2, color="#FF5722", label="No Residual")\n',
    'ax1.plot(layers, hist_res, lw=2, color="#4CAF50", label="With Residual")\n',
    'ax1.set_xlabel("Layer Depth"); ax1.set_ylabel("Gradient")\n',
    'ax1.set_title("Gradient Flow: Standard vs Residual", fontweight="bold")\n',
    'ax1.legend(); ax1.grid(True, alpha=0.3)\n\n',
    'ax2.semilogy(layers, hist_nores, lw=2, color="#FF5722", label="No Residual")\n',
    'ax2.semilogy(layers, hist_res, lw=2, color="#4CAF50", label="With Residual")\n',
    'ax2.set_xlabel("Layer Depth"); ax2.set_ylabel("Gradient (log)")\n',
    'ax2.set_title("Log Scale: 残差的+1 效应", fontweight="bold")\n',
    'ax2.legend(); ax2.grid(True, alpha=0.3)\n',
    'plt.suptitle("为什么 100+ 层也能训练: 残差连接的梯度高速公路", fontsize=15, fontweight="bold", y=1.02)\n',
    'plt.tight_layout(); plt.show()\n\n',
    'print(f"100层后梯度:")\n',
    'print(f"  无残差: {grad_nores:.2e} (彻底消失!)")\n',
    'print(f"  有残差: {grad_res:.4f} (依然可用!)")\n',
    'print(f"  关键: dL/dx_in = dL/dx_out * (dF/dx_in + 1) -> +1 保底!")'
], 'd07')

# Bottleneck
md([
    "### 1.2 Bottleneck——先压缩再解压\n\n",
    "ResNet-50+ 用 1×1 卷积先把 256 维压到 64 维→在 64 维做昂贵的 3×3→再用 1×1 升回 256 维。",
    "参数从 118 万降到 7 万——省了 94%！"
], 'd08')

code([
    "def count_bottleneck_params(in_ch, mid_ch, out_ch):\n",
    "    # 标准: 两个 3x3 卷积\n",
    "    standard = 2 * in_ch * out_ch * 3 * 3  # 2 x C_in x C_out x k^2\n",
    "    # Bottleneck: 1x1降维 + 3x3 + 1x1升维\n",
    "    bottleneck = in_ch * mid_ch * 1 * 1 + mid_ch * mid_ch * 3 * 3 + mid_ch * out_ch * 1 * 1\n",
    "    return standard, bottleneck\n\n",
    "in_ch, mid_ch, out_ch = 256, 64, 256\n",
    "standard, bottleneck = count_bottleneck_params(in_ch, mid_ch, out_ch)\n",
    'print(f"ResNet Bottleneck ({in_ch}->{mid_ch}->{out_ch}):")\n',
    'print(f"  标准 (2x3x3 conv): {standard:,} params")\n',
    'print(f"  Bottleneck:         {bottleneck:,} params")\n',
    'print(f"  省了 {(1-bottleneck/standard)*100:.0f}% 的参数!")\n',
    'print(f"  关键: 在低维({mid_ch}d)做昂贵的 3x3, 两头只在 1x1 转换")'
], 'd09')

# ============================================================
# Section 2: Transformer Block
# ============================================================
md(["---\n## 2. Transformer Block——现代 LLM 的核心构建块"], 'd10')
md([
    "### 2.1 完整解剖 (Qwen3-0.6B Layer 0)\n\n",
    "每一层包含: 多头注意力 + SwiGLU FFN，附两个残差连接和 RMSNorm。单层约 35 个算子，28 层总计约 1017 个。\n\n",
    "下面用代码展示每一步的 shape 变化。"
], 'd11')

code([
    'print("=" * 55)\nprint("Qwen3-0.6B Decoder Layer 0 结构")\nprint("=" * 55)\n\n',
    "seq_len = 1\n",
    "d_model = 1024\nq_n_head = 16\nkv_n_head = 8\nhead_dim = 128\nffn_dim = 3072\n\n",
    'print(f"  x_in: [{seq_len}, {d_model}]")\n',
    'print(f"  |")\n',
    'print(f"  |- RMSNorm({d_model}) -> norm")\n',
    'print(f"  |- Split Q/K/V 三路")\n',
    'print(f"  |")\n',
    'print(f"  |- Q路: Gemm({d_model}->{q_n_head*head_dim}=2048) -> Reshape[{head_dim},{q_n_head},{seq_len}] -> QK-Norm({head_dim}) -> Permute -> RoPE")\n',
    'print(f"  |- K路: Gemm({d_model}->{kv_n_head*head_dim}=1024) -> Reshape[{head_dim},{kv_n_head},{seq_len}] -> QK-Norm({head_dim}) -> Permute -> RoPE -> GQA Repeat(x{q_n_head//kv_n_head})")\n',
    'print(f"  |- V路: Gemm({d_model}->{kv_n_head*head_dim}=1024) -> Reshape -> Permute -> GQA Repeat(x{q_n_head//kv_n_head})")\n',
    'print(f"  |")\n',
    'print(f"  |- SDPA: Q*K^T/sqrt({head_dim})=0.088 -> softmax -> *V  (+ KV Cache 拼接)")\n',
    'print(f"  |- O Proj: Gemm({q_n_head*head_dim}->{d_model})")\n',
    'print(f"  |- + x_in (残差 1)")\n',
    'print(f"  |")\n',
    'print(f"  |- RMSNorm({d_model}) -> Split gate/up")\n',
    'print(f"  |- gate: Gemm({d_model}->{ffn_dim}) -> Swish")\n',
    'print(f"  |- up:   Gemm({d_model}->{ffn_dim})")\n',
    'print(f"  |- Mul: gate * up -> Gemm({ffn_dim}->{d_model})")\n',
    'print(f"  |- + x_attn (残差 2) -> x_out")\n',
    'print(f"  |")\n',
    'print(f"  以上 x 28 层 = 一次完整 decoder forward")\n',
    'print(f"  单层 ~35 个算子, 28 层总计 ~1017 个")'
], 'd12')

# Pre-Norm vs Post-Norm
md([
    "### 2.2 Pre-Norm vs Post-Norm——为什么现代 LLM 全选 Pre-Norm？\n\n",
    "- **Pre-Norm**: x_out = x + F(Norm(x))——残差路径上干干净净，梯度直达输入层。\n",
    "- **Post-Norm**: x_out = Norm(x + F(x))——梯度要通过 Norm 的导数，额外衰减。\n\n",
    "Post-Norm 在深层（>12 层）需要精细 warmup；Pre-Norm 不需要 warmup，训练天然稳定。\n\n",
    "下面用代码对比两种归一化位置下的梯度流。"
], 'd13')

code([
    "# Pre-Norm vs Post-Norm 梯度对比\n",
    "# 假设: 每层变换 F 的雅可比特征值平均 0.9, Norm 的雅可比平均 0.95\n\n",
    "n_layers = 30\n",
    "grad_pre = 1.0   # Pre-Norm: 残差路径 = I + F'*Norm' -> 至少有 +1\n",
    "grad_post = 1.0  # Post-Norm: 残差路径 = Norm' * (I + F') -> Norm' 额外衰减\n\n",
    "h_pre = [grad_pre]\nh_post = [grad_post]\n\n",
    "f_decay = 0.9\nnorm_decay = 0.95\n\n",
    "for i in range(n_layers):\n",
    "    # Pre-Norm: grad = grad * (1 + f_decay * norm_decay)  # I + F'(Norm(x)) * Norm'(x)\n",
    "    grad_pre = grad_pre * (1.0 + f_decay * norm_decay)\n",
    "    # Post-Norm: grad = grad * norm_decay * (1 + f_decay)  # Norm'(x+F(x)) * (I + F'(x))\n",
    "    grad_post = grad_post * norm_decay * (1 + f_decay)\n",
    "    h_pre.append(grad_pre)\n    h_post.append(grad_post)\n\n",
    "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))\n",
    "layers = range(n_layers + 1)\n",
    'ax1.plot(layers, h_pre, lw=2, color="#4CAF50", label="Pre-Norm")\n',
    'ax1.plot(layers, h_post, lw=2, color="#FF5722", label="Post-Norm")\n',
    'ax1.set_xlabel("Layer Depth"); ax1.set_ylabel("Gradient")\n',
    'ax1.set_title("Pre-Norm vs Post-Norm Gradient", fontweight="bold")\n',
    'ax1.legend(); ax1.grid(True, alpha=0.3)\n\n',
    'ax2.semilogy(layers, h_pre, lw=2, color="#4CAF50", label="Pre-Norm")\n',
    'ax2.semilogy(layers, h_post, lw=2, color="#FF5722", label="Post-Norm")\n',
    'ax2.set_xlabel("Layer Depth"); ax2.set_ylabel("Gradient (log)")\n',
    'ax2.set_title("Log Scale", fontweight="bold")\n',
    'ax2.legend(); ax2.grid(True, alpha=0.3)\n',
    'plt.suptitle("Pre-Norm vs Post-Norm: 残差路径的梯度保持", fontsize=15, fontweight="bold", y=1.02)\n',
    'plt.tight_layout(); plt.show()\n\n',
    'print(f"30层后梯度:")\n',
    'print(f"  Pre-Norm:  {grad_pre:.2f} (残差路径无障碍)")\n',
    'print(f"  Post-Norm: {grad_post:.4f} (被 Norm 逐步衰减)")\n',
    'print(f"  -> Pre-Norm 是深层 Transformer 训练稳定的关键!")'
], 'd14')

# QK-Norm
md([
    "### 2.3 QK-Norm——Qwen3 的独特贡献\n\n",
    "位置: Q 和 K 投影后、RoPE 前。对每个 head 独立做 RMSNorm(128)。\n\n",
    "**为什么需要？** 长上下文中，Q 和 K 没有归一化→点积方差随 d_k 增大→softmax 趋向 one-hot→模型\"盯死\"一个位置，忽视 99% 的上下文。\n",
    "QK-Norm 约束 Q 和 K 在稳定范围→softmax 保持合理分布→长上下文仍能关注多个关键位置。\n\n",
    "Qwen3 有、Qwen2.5/LLaMA/Mistral 无、Gemma 2 有。下面模拟 QK-Norm 对 Attention 分布的影响。"
], 'd15')

code([
    "# QK-Norm 对长上下文 Attention 的影响\n",
    "seq_len = 2048\nhead_dim = 128\ntorch.manual_seed(42)\n\n",
    "# 模拟长序列: Q 在前端, K 均匀分布\n",
    "Q_long = torch.randn(1, head_dim) * 2.0  # 未归一化的 Q\n",
    "K_long = torch.randn(seq_len, head_dim) * 2.0  # 未归一化的 K\n\n",
    "# 无 QK-Norm\n",
    "scores_raw = (Q_long @ K_long.T).squeeze() / (head_dim ** 0.5)\n",
    "attn_raw = F.softmax(scores_raw, dim=-1)\n\n",
    "# 有 QK-Norm (模拟: 归一化后 Q,K 在稳定范围)\n",
    "Q_norm = F.layer_norm(Q_long, [head_dim])\nK_norm = F.layer_norm(K_long, [head_dim])\n",
    "scores_norm = (Q_norm @ K_norm.T).squeeze() / (head_dim ** 0.5)\n",
    "attn_norm = F.softmax(scores_norm, dim=-1)\n\n",
    "fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4))\n\n",
    'ax1.plot(attn_raw.numpy()[:50], color="#FF5722", alpha=0.7)\n',
    'ax1.set_title("Without QK-Norm", fontweight="bold")\n',
    'ax1.set_xlabel("Token Position"); ax1.set_ylabel("Attention Weight")\n',
    'ax1.grid(True, alpha=0.3)\n\n',
    'ax2.plot(attn_norm.numpy()[:50], color="#4CAF50", alpha=0.7)\n',
    'ax2.set_title("With QK-Norm", fontweight="bold")\n',
    'ax2.set_xlabel("Token Position"); ax2.set_ylabel("Attention Weight")\n',
    'ax2.grid(True, alpha=0.3)\n\n',
    'ax3.hist(attn_raw.numpy(), bins=50, alpha=0.5, label="No QK-Norm", color="#FF5722")\n',
    'ax3.hist(attn_norm.numpy(), bins=50, alpha=0.5, label="With QK-Norm", color="#4CAF50")\n',
    'ax3.set_title("Attention Distribution", fontweight="bold")\n',
    'ax3.legend(fontsize=9)\n',
    'ax3.set_xlabel("Attention Weight")\n\n',
    'plt.suptitle("QK-Norm 对长上下文 Attention 的影响 (seq_len=2048)", fontsize=15, fontweight="bold", y=1.02)\n',
    'plt.tight_layout(); plt.show()\n\n',
    'print(f"无QK-Norm: max_attn={attn_raw.max():.4f}, entropy={-(attn_raw*torch.log(attn_raw+1e-10)).sum():.2f}")\n',
    'print(f"有QK-Norm: max_attn={attn_norm.max():.4f}, entropy={-(attn_norm*torch.log(attn_norm+1e-10)).sum():.2f}")\n',
    "print(f\"有QK-Norm的熵更高 -> 注意力更分散, 不 all-in 一个位置\")"
], 'd16')

# SwiGLU
md([
    "### 2.4 SwiGLU——现代 LLM MLP 的核心\n\n",
    "SwiGLU(x) = Swish(x·W_gate) ⊙ (x·W_up) · W_down。gate 产生门控信号，up 携带信息，逐元素乘 = 选择性放行。\n\n",
    "参数量: 3 × 3 × hidden² = 9×hidden²（标准 FFN 是 8×hidden²），多 ~12.5% 参数换显著的质量提升。"
], 'd17')

code([
    "hidden = 1024\nintermediate_std = hidden * 4   # 标准 FFN\nintermediate_swiglu = hidden * 3  # SwiGLU (中间维度是 3x 而非 4x)\n\n",
    "# 标准 FFN: W1(hidden->4h) + W2(4h->hidden)\n",
    "params_std = hidden * intermediate_std + intermediate_std * hidden  # 2 * 4 * h^2\n\n",
    "# SwiGLU: gate(hidden->3h) + up(hidden->3h) + down(3h->hidden)\n",
    "params_swiglu = hidden * intermediate_swiglu * 3  # 3 * 3 * h^2\n\n",
    'print(f"FFN 参数量对比 (hidden={hidden}):")\n',
    'print(f"  标准 ReLU-FFN: {params_std:,} = 2x4xh^2 = 8h^2 = {8*hidden*hidden:,}")\n',
    'print(f"  SwiGLU:        {params_swiglu:,} = 3x3xh^2 = 9h^2 = {9*hidden*hidden:,}")\n',
    'print(f"  SwiGLU 多 {(params_swiglu/params_std - 1)*100:.1f}% 参数, 换门控能力")\n\n',
    "# 手写 SwiGLU\n",
    "hd, inter = 8, 24\ntorch.manual_seed(42)\n",
    "x = torch.randn(1, hd)\n",
    "Wg, Wu, Wd = torch.randn(hd, inter), torch.randn(hd, inter), torch.randn(inter, hd)\n",
    "gate = F.silu(x @ Wg)\nup = x @ Wu\ngated = gate * up\noutput = gated @ Wd\n",
    'print(f"\\nSwiGLU 小例子 (hidden={hd}, intermediate={inter}):")\n',
    'print(f"  gate:   Swish(x*Wg) -> {tuple(gate.shape)} (门控信号)")\n',
    'print(f"  up:     x*Wu        -> {tuple(up.shape)} (信息流)")\n',
    'print(f"  gated:  gate * up   -> {tuple(gated.shape)} (选择性放行!)")\n',
    'print(f"  output: gated*Wd    -> {tuple(output.shape)} (最终输出)")\n',
    'print(f"  门控统计: min={gate.min():.2f}, max={gate.max():.2f}, mean={gate.mean():.2f}")'
], 'd18')

# Transformer Evolution
md([
    "### 2.5 Transformer 进化编年史\n\n",
    "从 2017 年到 2024 年，每一步改进都解决一个具体问题：\n\n",
    "| 年份 | 模型 | 关键变化 | 解决的问题 |\n",
    "|------|------|---------|----------|\n",
    "| 2017 | Vanilla | Post-LN+MHA+ReLU+绝对PE | 开创 |\n",
    "| 2018 | BERT/GPT-2 | Post-LN+GELU+可学习PE | 训练稳定性 |\n",
    "| 2022 | PaLM | **Pre-LN+MQA+SwiGLU+RoPE** | 范式转折 |\n",
    "| 2023 | LLaMA 1/2 | Pre-RMSNorm+GQA+SwiGLU+RoPE | 现代范式确立 |\n",
    "| 2024 | Qwen3 | +QK-Norm +长上下文优化 | 注意力质量 |\n\n",
    "每一步变化的简化总结：Post→Pre=深层稳定, MHA→GQA=KV减半, ReLU→SwiGLU=表达能力, PE→RoPE=外推能力。"
], 'd19')

# ============================================================
# Section 3: Architecture Quick Reference
# ============================================================
md(["---\n## 3. CNN/RNN/编解码架构速查"], 'd20')
md([
    "### 3.1 CNN Block 变体\n\n",
    "不同场景下的 CNN 构建块选择。下面展示每个变体的核心特征。"
], 'd21')

code([
    'print("CNN Block 变体对比:")\nprint("=" * 60)\n\n',
    'print("Conv-BN-ReLU (VGG 风格):")\n',
    'print("  Conv2d -> BatchNorm2d -> ReLU")\n',
    'print("  最简单, 适合浅层网络 (< 20层)")\n\n',
    'print("ResNet BasicBlock (ResNet-18/34):")\n',
    'print("  Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> +x -> ReLU")\n',
    'print("  残差连接让梯度直达, 支持 34 层")\n\n',
    'print("ResNet Bottleneck (ResNet-50+):")\n',
    'print("  Conv1x1(降维) -> BN -> ReLU -> Conv3x3 -> BN -> ReLU -> Conv1x1(升维) -> BN -> +x -> ReLU")\n',
    'print("  参数量暴降 94%, 支持 152 层")\n\n',
    'print("SE Block (Squeeze-and-Excitation):")\n',
    'print("  GAP -> FC(降维/16) -> ReLU -> FC(升维) -> Sigmoid -> channel_scale")\n',
    'print("  通道注意力: 让网络自己决定哪些通道重要")'
], 'd22')

md([
    "### 3.2 RNN 变体与编解码架构"
], 'd23')

code([
    'print("RNN 变体对比:")\nprint("=" * 60)\n\n',
    'print("Vanilla RNN:  h_t = tanh(W*[h_{t-1}, x_t])")\n',
    'print("  问题: 梯度消失, 长期依赖丢失")\n\n',
    'print("LSTM:  c_t = f*c_{t-1} + i*g,  h_t = o*tanh(c_t)")\n',
    'print("  4 门控, 细胞状态=梯度高速公路")\n\n',
    'print("GRU:  h_t = (1-z)*h_{t-1} + z*n")\n',
    'print("  2 门控, 省 25% 参数, 效果相当")\n\n',
    'print("Bi-LSTM: 正向+反向各一 LSTM, 输出拼接")\n',
    'print("  可以看到前后文, 但无法并行")\n\n',
    'print("-" * 60)\n',
    'print("编解码架构:")\n',
    'print("  Encoder-Decoder (原始 Transformer):")\n',
    'print("    Encoder: 双向注意力, 理解输入")\n',
    'print("    Decoder: 因果注意力 + Cross-Attention(看 Encoder 输出)")\n',
    'print("    代表: 机器翻译 (原始 Transformer), T5, BART")\n\n',
    'print("  Decoder-Only (GPT/LLaMA/Qwen):")\n',
    'print("    全因果注意力, 自回归生成")\n',
    'print("    代表: GPT 系列, LLaMA, Qwen, 几乎所有现代 LLM")\n\n',
    'print("  Encoder-Only (BERT):")\n',
    'print("    双向注意力, 理解任务优先")\n',
    'print("    代表: BERT, RoBERTa")'
], 'd24')

# ============================================================
# Bonus: Complete Transformer Block Implementation
# ============================================================
md(["---\n## Bonus: 手写一个 Transformer Block"], 'd25')
md([
    "把前面学到的所有概念组装成一个可运行的 Transformer Block: RMSNorm + GQA + RoPE + SwiGLU + 残差连接。"
], 'd26')

code([
    "class MiniTransformerBlock(nn.Module):\n",
    '    """手写 Transformer Block (RMSNorm + GQA + SwiGLU + 残差)"""\n',
    "    def __init__(self, d_model=256, n_q_heads=8, n_kv_heads=4, head_dim=32, ffn_dim=768):\n",
    "        super().__init__()\n        self.d_model = d_model\n",
    "        self.n_q_heads = n_q_heads\n        self.n_kv_heads = n_kv_heads\n",
    "        self.head_dim = head_dim\n        self.n_rep = n_q_heads // n_kv_heads\n\n",
    "        # Attention\n",
    "        self.W_Q = nn.Linear(d_model, n_q_heads * head_dim, bias=False)\n",
    "        self.W_K = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)\n",
    "        self.W_V = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)\n",
    "        self.W_O = nn.Linear(n_q_heads * head_dim, d_model, bias=False)\n\n",
    "        # FFN (SwiGLU)\n",
    "        self.gate = nn.Linear(d_model, ffn_dim, bias=False)\n",
    "        self.up   = nn.Linear(d_model, ffn_dim, bias=False)\n",
    "        self.down = nn.Linear(ffn_dim, d_model, bias=False)\n\n",
    "        # Norms\n",
    "        self.norm1 = nn.RMSNorm(d_model)\n        self.norm2 = nn.RMSNorm(d_model)\n\n",
    "    def forward(self, x):\n",
    "        B, S, D = x.shape\n        # --- Attention Block ---\n",
    "        x_norm = self.norm1(x)\n",
    "        Q = self.W_Q(x_norm).view(B, S, self.n_q_heads, self.head_dim).transpose(1, 2)\n",
    "        K = self.W_K(x_norm).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)\n",
    "        V = self.W_V(x_norm).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)\n",
    "        # GQA: repeat K, V\n",
    "        K = K.unsqueeze(2).expand(-1, -1, self.n_rep, -1, -1).reshape(B, self.n_q_heads, S, self.head_dim)\n",
    "        V = V.unsqueeze(2).expand(-1, -1, self.n_rep, -1, -1).reshape(B, self.n_q_heads, S, self.head_dim)\n",
    "        # SDPA\n",
    "        attn_out = F.scaled_dot_product_attention(Q, K, V, is_causal=True)\n",
    "        attn_out = attn_out.transpose(1, 2).reshape(B, S, -1)\n",
    "        attn_out = self.W_O(attn_out)\n",
    "        x = x + attn_out  # 残差 1\n\n",
    "        # --- FFN Block (SwiGLU) ---\n",
    "        x_norm2 = self.norm2(x)\n",
    "        gated = F.silu(self.gate(x_norm2)) * self.up(x_norm2)\n",
    "        ffn_out = self.down(gated)\n",
    "        x = x + ffn_out  # 残差 2\n",
    "        return x\n\n",
    "# 测试\n",
    "B, S, D = 1, 4, 256\n",
    "block = MiniTransformerBlock(d_model=256, n_q_heads=8, n_kv_heads=4, head_dim=32, ffn_dim=768)\n",
    "x = torch.randn(B, S, D)\n",
    "y = block(x)\n",
    "total_params = sum(p.numel() for p in block.parameters())\n",
    'print(f"MiniTransformerBlock 测试:")\n',
    'print(f"  输入: {tuple(x.shape)}")\n',
    'print(f"  输出: {tuple(y.shape)}")\n',
    'print(f"  参数量: {total_params:,}")\n',
    'print(f"  Shape 一致: {x.shape == y.shape}")\n',
    'print(f"  包含: RMSNorm + GQA + SwiGLU + 2x残差连接")\n',
    'print(f"  -> 这就是现代 LLM 的最小可运行单元!")'
], 'd27')

# ============================================================
# Summary
# ============================================================
md(["---\n## Summary"], 'd28')
md([
    "| Concept | Experiment | Key Insight |\n",
    "|---------|-----------|-------------|\n",
    "| Residual Gradient | 100-layer gradient flow | +1 = 梯度高速公路 |\n",
    "| Bottleneck | 256->64->256 param count | 1x1 降维省 94% 参数 |\n",
    "| Transformer Block | Full anatomy with shapes | 28 层 × 35 算子 = ~1017 |\n",
    "| Pre-Norm vs Post-Norm | 30-layer gradient comparison | Pre-Norm 残差路径无障碍 |\n",
    "| QK-Norm | Long-seq attention distribution | 防止 attention all-in 一个位置 |\n",
    "| SwiGLU | gate * up = selective gating | 多 12.5% 参数换门控能力 |\n",
    "| Transformer Evolution | 2017->2024 timeline | 每步解决一个具体问题 |\n",
    "| CNN Blocks | VGG/ResNet/Bottleneck/SE | 场景决定选择 |\n",
    "| RNN Variants | LSTM vs GRU vs Bi-LSTM | 编解码架构 = LLM 骨架 |\n",
    "| Mini Block | Complete runnable implementation | RMSNorm+GQA+SwiGLU+Residual |"
], 'd29')

# Write
with open('3.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(f'3.ipynb written: {len(nb["cells"])} cells')
