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
# Cell 1-4: Title + Env
# ============================================================
md(["Module 2: 构建算子——从单元素到空间\n\n", "> 卷积用滑动窗口捕获局部纹理，注意力让每个 token 看到所有其他 token，LSTM 用门控对抗梯度消失——从代码层面理解每一个算子的数学本质。"], 'c01')
md(["## 0. 环境准备"], 'c02')
md(["导入所有需要的库。torch 做张量运算，numpy 辅助数据处理，matplotlib 可视化。"], 'c03')
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
], 'c04')

# ============================================================
# Section 1: Convolution
# ============================================================
md(["---\n## 1. 卷积——计算机视觉的放大镜"], 'c05')
md([
    "### 1.1 六个核心参数\n\n",
    "卷积用两个先验知识替代全连接：**局部性**（一个像素只和周围像素有关）和**平移等变性**（特征在哪里都该被同样识别）。\n\n",
    "H' = floor((H + 2*pad - dilation*(k-1) - 1) / stride + 1)\n\n",
    "下面用代码验证每个参数的效果。"
], 'c06')

code([
    "torch.manual_seed(42)\nB, C_in, H, W = 1, 3, 8, 8\nx = torch.randn(B, C_in, H, W)\n\n",
    'print("=" * 55)\nprint("卷积六个参数效果演示")\nprint("=" * 55)\n\n',
    "conv_k3 = nn.Conv2d(3, 16, 3, stride=1, padding=1)\n",
    "conv_k5 = nn.Conv2d(3, 16, 5, stride=1, padding=2)\n",
    'print(f"kernel=3, pad=1 -> {tuple(conv_k3(x).shape)} (same)")\n',
    'print(f"kernel=5, pad=2 -> {tuple(conv_k5(x).shape)} (same, 感受野更大)")\n',
    'print(f"  两个3x3 = 一个5x5感受野, 但参数 2*9=18 < 25")\n\n',
    "conv_s1 = nn.Conv2d(3, 16, 3, stride=1, padding=1)\n",
    "conv_s2 = nn.Conv2d(3, 16, 3, stride=2, padding=1)\n",
    'print(f"\\nstride=1 -> {tuple(conv_s1(x).shape)}")\n',
    'print(f"stride=2 -> {tuple(conv_s2(x).shape)} (H/W减半)")\n\n',
    "conv_p0 = nn.Conv2d(3, 16, 3, stride=1, padding=0)\n",
    "conv_p1 = nn.Conv2d(3, 16, 3, stride=1, padding=1)\n",
    'print(f"\\npadding=0 -> {tuple(conv_p0(x).shape)} (H从8缩到6)")\n',
    'print(f"padding=1 -> {tuple(conv_p1(x).shape)} (same)")\n\n',
    "conv_d1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, dilation=1)\n",
    "conv_d2 = nn.Conv2d(3, 16, 3, stride=1, padding=2, dilation=2)\n",
    'print(f"\\ndilation=1 -> {tuple(conv_d1(x).shape)} (标准)")\n',
    'print(f"dilation=2 -> {tuple(conv_d2(x).shape)} (空洞, 隔点采样)")\n\n',
    "conv_std = nn.Conv2d(8, 8, 3, padding=1)\nconv_dw  = nn.Conv2d(8, 8, 3, padding=1, groups=8)\n",
    "std_p = sum(p.numel() for p in conv_std.parameters())\n",
    "dw_p  = sum(p.numel() for p in conv_dw.parameters())\n",
    'print(f"\\n标准Conv: {std_p} params")\n',
    'print(f"Depthwise(groups=8): {dw_p} params (省了 {100*dw_p//std_p}%)")\n\n',
    "conv_b = nn.Conv2d(3, 16, 3, padding=1, bias=True)\n",
    "conv_nb = nn.Conv2d(3, 16, 3, padding=1, bias=False)\n",
    'print(f"\\nbias=True: {sum(p.numel() for p in conv_b.parameters())} params")\n',
    'print(f"bias=False: {sum(p.numel() for p in conv_nb.parameters())} params (接BN时关bias)")'
], 'c07')

# im2col
md([
    "### 1.2 im2col——卷积变矩阵乘法\n\n",
    "**im2col = image to column**：把滑动窗口的每一次停留展开为矩阵的一列。",
    "虽然内存膨胀了 k² 倍，但换来了高度优化的 GEMM kernel。\n\n",
    "下面用 5×5 图像 + 3×3 核演示，并与 PyTorch 卷积对比验证。"
], 'c08')

code([
    "def im2col(x, kernel_size, stride=1, padding=0):\n",
    "    if padding > 0:\n        x = F.pad(x, [padding] * 4)\n",
    "    N, C, H, W = x.shape\n    kH = kW = kernel_size\n",
    "    out_h = (H - kH) // stride + 1\n    out_w = (W - kW) // stride + 1\n",
    "    cols = []\n",
    "    for i in range(0, H - kH + 1, stride):\n",
    "        for j in range(0, W - kW + 1, stride):\n",
    "            patch = x[:, :, i:i+kH, j:j+kW]\n",
    "            cols.append(patch.reshape(-1))\n",
    "    return torch.stack(cols, dim=1)\n\n",
    'x = torch.arange(1, 26, dtype=torch.float32).reshape(1, 1, 5, 5)\n',
    'print("输入图像 5x5:")\nprint(x[0, 0].numpy().astype(int))\n\n',
    'X_col = im2col(x, kernel_size=3, stride=1, padding=0)\n',
    'print(f"\\nim2col 后形状: {tuple(X_col.shape)}")\n',
    'print(f"  9 = 3x3(kernel展平), 9 = 3x3(滑动窗口数)")\n',
    'print(f"  内存膨胀: {X_col.numel()}/{x.numel()} = {X_col.numel()/x.numel():.2f}x")\n\n',
    "conv = nn.Conv2d(1, 1, 3, bias=False)\n",
    "with torch.no_grad():\n",
    "    pt_out = conv(x)\n",
    "    W = conv.weight.data.reshape(1, -1)\n",
    "    my_out = (W @ X_col).reshape(1, 1, 3, 3)\n",
    '    print(f"\\nPyTorch Conv vs im2col+GEMM max error: {(pt_out - my_out).abs().max():.2e}")\n',
    '    print("im2col 正确地将卷积转化为了矩阵乘法!")'
], 'c09')

# Winograd
md([
    "### 1.3 Winograd F(2×2, 3×3)——用加法换乘法\n\n",
    "Winograd 用变换矩阵把卷积拆成 变换→逐元素乘→逆变换：正常 4个输出×9次乘法=36次，Winograd 仅 16 次逐元素乘。",
    "ncnn 的 3x3_winograd 实现是 ResNet 类模型推理的加速秘诀。"
], 'c10')

code([
    "B_T = torch.tensor([[1,0,-1,0],[0,1,1,0],[0,-1,1,0],[0,1,0,-1]], dtype=torch.float32)\n",
    "G   = torch.tensor([[1,0,0],[1/2,1/2,1/2],[1/2,-1/2,1/2],[0,0,1]], dtype=torch.float32)\n",
    "A_T = torch.tensor([[1,1,1,0],[0,1,-1,-1]], dtype=torch.float32)\n\n",
    'print("Winograd F(2x2,3x3) 变换矩阵:")\n',
    'print(f"  B^T [{B_T.shape}]: 输入变换")\n',
    'print(f"  G   [{G.shape}]:  权重变换")\n',
    'print(f"  A^T [{A_T.shape}]:  输出逆变换")\n\n',
    "torch.manual_seed(42)\nx_tile = torch.randn(4, 4)\nw = torch.randn(3, 3)\n",
    "normal_out = F.conv2d(x_tile.view(1,1,4,4), w.view(1,1,3,3)).squeeze()\n",
    "U = G @ w @ G.T\nV = B_T @ x_tile @ B_T.T\nY = A_T @ (U * V) @ A_T.T\n\n",
    'print(f"\\n标准卷积输出:\\n{normal_out}")\n',
    'print(f"\\nWinograd 输出:\\n{Y}")\n',
    'print(f"\\nMax error: {(normal_out - Y).abs().max():.2e}")\n',
    'print("乘法从 36 -> 16, 用加法换, 约 1.5~2x 加速")'
], 'c11')

# Receptive Field
md([
    "### 1.4 感受野——这一层能看到多远？\n\n",
    "单层: RF = dilation×(kernel-1)+1。多层累积时，后面的层继承前面所有 stride 的扩张效应。"
], 'c12')

code([
    "def compute_rf(layers):\n",
    "    rf, stride_cum = 1, 1\n",
    "    for i, (k, s) in enumerate(layers, 1):\n",
    "        rf = rf + (k - 1) * stride_cum\n",
    "        stride_cum *= s\n",
    '        print(f"  Layer {i}: k={k}, s={s} -> RF={rf:>3}, stride_cum={stride_cum}")\n',
    "    return rf\n\n",
    'print("VGG风格: 3层3x3(stride=1)")\n',
    "rf = compute_rf([(3, 1), (3, 1), (3, 1)])\n",
    'print(f"等效感受野: {rf}x{rf} (等价一个7x7卷积)")\n',
    'print(f"参数: 3x9=27 vs 7x7=49")\n\n',
    'print("\\n带下采样: k3s1 -> k3s2 -> k3s1 -> k3s2")\n',
    "compute_rf([(3, 1), (3, 2), (3, 1), (3, 2)])"
], 'c13')

# ============================================================
# Section 2: Memory Layout
# ============================================================
md([
    "---\n## 2. 内存布局——NCHW 还是 NHWC？\n\n",
    "同一个模型，换排列方式，手机上可能快 2 倍。NCHW 通道连续→Conv 向量化友好（ncnn 默认）。",
    "NHWC 像素连续→逐像素操作友好（Vulkan 内部转换）。"
], 'c14')

code([
    "C, H, W = 3, 4, 4\n",
    "data = torch.arange(C * H * W, dtype=torch.float32).reshape(1, C, H, W)\n",
    "nchw = data.contiguous().flatten().numpy()\n",
    "nhwc = data.permute(0, 2, 3, 1).contiguous().flatten().numpy()\n\n",
    'print(f"原图 {C}x{H}x{W}, 共 {C*H*W} 个元素")\n',
    'print(f"\\nNCHW (PyTorch默认, Channel-major):")\n',
    'print(f"  R通道: {nchw[0:16].astype(int).tolist()}")\n',
    'print(f"  G通道: {nchw[16:32].astype(int).tolist()}")\n',
    'print(f"  B通道: {nchw[32:48].astype(int).tolist()}")\n',
    'print(f"  -> 同一通道的数据连续存放")\n\n',
    'print(f"\\nNHWC (TensorFlow默认, Pixel-major):")\n',
    'print(f"  Pixel(0,0) RGB: {nhwc[0:3].astype(int).tolist()}")\n',
    'print(f"  Pixel(0,1) RGB: {nhwc[3:6].astype(int).tolist()}")\n',
    'print(f"  Pixel(0,2) RGB: {nhwc[6:9].astype(int).tolist()}")\n',
    'print(f"  -> 同一像素的 RGB 连续存放")\n\n',
    'print(f"\\n逐通道操作 (BatchNorm): NCHW Cache友好")\n',
    'print(f"逐像素操作 (ReLU):     NHWC Cache友好")'
], 'c15')

# ============================================================
# Section 3: Attention
# ============================================================
md(["---\n## 3. 注意力机制——全局依赖的建立者"], 'c16')
md([
    "### 3.1 SDPA：五步精密舞蹈\n\n",
    "Q=我要找什么, K=我有什么标签, V=我有什么内容。",
    "下面逐步执行 SDPA，观察每步的 shape 和数值变化。"
], 'c17')

code([
    "seq_len, head_dim = 4, 8\ntorch.manual_seed(42)\n",
    "Q = torch.randn(seq_len, head_dim) * 0.5\n",
    "K = torch.randn(seq_len, head_dim) * 0.5\n",
    "V = torch.randn(seq_len, head_dim) * 0.5\n\n",
    'print("SDPA 五步手算")\nprint("=" * 40)\n\n',
    "scores = Q @ K.T\n",
    'print(f"Step 1 Q*K^T: scores [{scores.shape}]")\n\n',
    "scale = head_dim ** 0.5\nscores_scaled = scores / scale\n",
    'print(f"Step 2 /sqrt(d_k)={1/scale:.3f}: 防止点积过大")\n\n',
    'mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()\n',
    'scores_scaled[mask] = float("-inf")\n',
    'print(f"Step 3 +mask: 未来token->-inf")\n\n',
    "attn_weights = F.softmax(scores_scaled, dim=-1)\n",
    'print(f"Step 4 softmax: 每行和 = {attn_weights.sum(dim=-1).tolist()}")\n\n',
    "output = attn_weights @ V\n",
    'print(f"Step 5 weights@V: output [{output.shape}]")'
], 'c18')

# Scale factor
md([
    "### 3.2 为什么要除以 √d_k？\n\n",
    "当 d_k 很大时 Q·K 的方差≈d_k → 点积值可能很大 → softmax 趋向 one-hot → 梯度≈0。",
    "下面可视化不同 d_k 下 scale 与否的分布差异。"
], 'c19')

code([
    "dims = [16, 64, 128, 512]\ntorch.manual_seed(42)\n",
    "fig, axes = plt.subplots(1, 4, figsize=(16, 3))\n",
    "for ax, d_k in zip(axes, dims):\n",
    "    Q = torch.randn(1, d_k) * 0.3\n    K = torch.randn(100, d_k) * 0.3\n",
    "    raw = (Q @ K.T).squeeze()\n    scaled = raw / (d_k ** 0.5)\n",
    "    p_raw = F.softmax(raw, dim=-1)\n    p_scaled = F.softmax(scaled, dim=-1)\n",
    '    ax.hist(p_raw.numpy(), bins=30, alpha=0.5, label="No Scale", color="#FF5722")\n',
    '    ax.hist(p_scaled.numpy(), bins=30, alpha=0.5, label=f"Scaled (/{d_k**0.5:.0f})", color="#2196F3")\n',
    '    ax.set_title(f"d_k = {d_k}", fontweight="bold", fontsize=13)\n',
    '    ax.legend(fontsize=9)\n    ax.set_xlabel("Attention prob", fontsize=10)\n',
    'plt.suptitle("Scale Factor 对 Softmax 分布的影响", fontsize=15, fontweight="bold", y=1.02)\n',
    'plt.tight_layout()\nplt.show()\n',
    'print("d_k=512: 不scale分布高度尖峰 -> one-hot -> 梯度消失")\n',
    'print("这就是 Scaled Dot-Product Attention 中 Scaled 的含义")'
], 'c20')

# QKV & GQA
md([
    "### 3.3 QKV 投影与 GQA\n\n",
    "Qwen3-0.6B: Q=16头, KV=8头, 每 2 个 Q 共享 1 组 KV。KV Cache 直接减半！"
], 'c21')

code([
    "seq_len, hidden = 4, 1024\nn_q_heads, n_kv_heads, head_dim = 16, 8, 128\n",
    "x = torch.randn(1, seq_len, hidden)\n",
    "W_Q = torch.randn(hidden, n_q_heads * head_dim)\n",
    "W_K = torch.randn(hidden, n_kv_heads * head_dim)\n",
    "W_V = torch.randn(hidden, n_kv_heads * head_dim)\n\n",
    "Q = (x @ W_Q).view(1, seq_len, n_q_heads, head_dim).transpose(1, 2)\n",
    "K = (x @ W_K).view(1, seq_len, n_kv_heads, head_dim).transpose(1, 2)\n",
    "V = (x @ W_V).view(1, seq_len, n_kv_heads, head_dim).transpose(1, 2)\n\n",
    'print("Qwen3-0.6B QKV 投影 (GQA):")\n',
    'print(f"  Q: {tuple(Q.shape)}  -- {n_q_heads} heads")\n',
    'print(f"  K: {tuple(K.shape)}   -- {n_kv_heads} heads (GQA)")\n',
    'print(f"  V: {tuple(V.shape)}   -- {n_kv_heads} heads")\n\n',
    "n_rep = n_q_heads // n_kv_heads\n",
    "K_exp = K.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).reshape(1, n_q_heads, seq_len, head_dim)\n",
    "V_exp = V.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).reshape(1, n_q_heads, seq_len, head_dim)\n",
    'print(f"\\nGQA repeat_kv (x{n_rep}): {tuple(K.shape)} -> {tuple(K_exp.shape)}")\n',
    'print(f"  配对: Q0,Q1->KV0 | Q2,Q3->KV1 | ...")\n',
    'print(f"  KV Cache 只有 {n_kv_heads/n_q_heads*100:.0f}% 的大小!")'
], 'c22')

# RoPE
md([
    "### 3.4 RoPE——旋转编码的数学之美\n\n",
    "RoPE 把 Q/K 的每对相邻维度视为 2D 向量，按位置旋转。",
    "精妙之处：**Q_i·K_j 只依赖相对位置差 (i-j)**——这就是外推能力的来源。"
], 'c23')

code([
    "def rope_rotate(x, cos, sin):\n",
    "    x_rot = torch.empty_like(x)\n",
    "    x_rot[..., 0::2] = x[..., 0::2] * cos - x[..., 1::2] * sin\n",
    "    x_rot[..., 1::2] = x[..., 0::2] * sin + x[..., 1::2] * cos\n",
    "    return x_rot\n\n",
    "# 验证 1: 模长不变\n",
    "q = torch.tensor([3.0, 4.0])\n",
    "c1, s1 = math.cos(1.0), math.sin(1.0)\n",
    "q_rot = rope_rotate(q, torch.tensor([c1]), torch.tensor([s1]))\n",
    'print(f"验证1 模长不变: |q|={q.norm():.4f}, |q_rot|={q_rot.norm():.4f}")\n',
    'print(f"  旋转角={(math.degrees(math.acos((q@q_rot)/(q.norm()*q_rot.norm())))):.1f}度 = 1.0 rad")\n\n',
    "# 验证 2: 相对位置 R_i^T R_j = R_{j-i}\n",
    "ca,sa=math.cos(0.5),math.sin(0.5); cb,sb=math.cos(0.3),math.sin(0.3)\n",
    "R1=torch.tensor([[ca,-sa],[sa,ca]]); R2=torch.tensor([[cb,-sb],[sb,cb]])\n",
    "R_diff=R1.T@R2; cd,sd=math.cos(0.3-0.5),math.sin(0.3-0.5)\n",
    "R_exp=torch.tensor([[cd,-sd],[sd,cd]])\n",
    'print(f"\\n验证2 相对位置: max err={(R_diff-R_exp).abs().max():.2e}")\n',
    'print(f"  R_i^T*R_j = R_{{j-i}}  得证!")\n\n',
    "# 验证 3: Qwen3 128维频率\n",
    'print(f"\\n验证3 Qwen3-0.6B theta=1e6, 128维频率:")\n',
    "for d in [0, 32, 64, 126]:\n",
    '    period = 2*math.pi/(1.0/(1000000.0**(d/128)))\n',
    '    print(f"  dim {d:>3}: period={period:.0f} tokens")\n',
    'print("低维旋转快(tok级) 高维旋转慢(句级)")'
], 'c24')

# ============================================================
# Section 4: LSTM & GRU
# ============================================================
md(["---\n## 4. LSTM 与 GRU——循环神经网络的梯度博弈"], 'c25')
md([
    "### 4.1 RNN 梯度消失——数学本质\n\n",
    "反向传播时梯度要穿过时间：∂L/∂h₀ = ∂L/∂h_T × Π W·tanh'。",
    "每步乘子<1 时，连乘 T 次后梯度指数衰减。下面模拟 100 步传播。"
], 'c26')

code([
    "T = 100\nsteps = np.arange(T + 1)\n",
    "rnn_09 = 0.9 ** steps\nrnn_095 = 0.95 ** steps\nlstm_099 = 0.99 ** steps\n\n",
    "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))\n",
    'ax1.plot(steps, rnn_09, lw=2, color="#FF5722", label="RNN decay=0.9")\n',
    'ax1.plot(steps, rnn_095, lw=2, color="#FF9800", label="RNN decay=0.95")\n',
    'ax1.plot(steps, lstm_099, lw=2, color="#4CAF50", label="LSTM decay=0.99")\n',
    'ax1.set_xlabel("Time Steps"); ax1.set_ylabel("Gradient")\n',
    'ax1.set_title("Gradient Propagation", fontweight="bold")\n',
    'ax1.legend(); ax1.grid(True, alpha=0.3)\n',
    'ax2.semilogy(steps, rnn_09, lw=2, color="#FF5722", label="RNN decay=0.9")\n',
    'ax2.semilogy(steps, rnn_095, lw=2, color="#FF9800", label="RNN decay=0.95")\n',
    'ax2.semilogy(steps, lstm_099, lw=2, color="#4CAF50", label="LSTM decay=0.99")\n',
    'ax2.set_xlabel("Time Steps"); ax2.set_ylabel("Gradient (log)")\n',
    'ax2.set_title("Log Scale", fontweight="bold")\n',
    'ax2.legend(); ax2.grid(True, alpha=0.3)\n',
    'plt.suptitle("为什么 RNN 学不到长期依赖", fontsize=16, fontweight="bold", y=1.02)\n',
    'plt.tight_layout(); plt.show()\n\n',
    'print(f"T=100 梯度衰减:")\nprint(f"  RNN decay=0.9:  0.9^100  = {0.9**100:.2e}")\n',
    'print(f"  RNN decay=0.95: 0.95^100 = {0.95**100:.4f}")\n',
    'print(f"  LSTM decay=0.99: 0.99^100 = {0.99**100:.4f}")\n',
    'print(f"  关键: LSTM 用逐元素乘替代矩阵乘!")'
], 'c27')

# LSTM
md([
    "### 4.2 LSTM——梯度高速公路\n\n",
    "c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t。∂c_t/∂c_{t-1} = f_t——逐元素乘，没有矩阵乘法！",
    "f_t≈1 时梯度几乎无损穿过时间。下面手写 LSTM 四个门控。"
], 'c28')

code([
    "def lstm_step(x_t, h_prev, c_prev, W_ih, W_hh, b_ih, b_hh):\n",
    "    hidden = h_prev.shape[-1]\n",
    "    gates = x_t @ W_ih.T + b_ih + h_prev @ W_hh.T + b_hh\n",
    "    f = torch.sigmoid(gates[:, :hidden])\n",
    "    i = torch.sigmoid(gates[:, hidden:2*hidden])\n",
    "    g = torch.tanh(gates[:, 2*hidden:3*hidden])\n",
    "    o = torch.sigmoid(gates[:, 3*hidden:])\n",
    "    c_t = f * c_prev + i * g\n",
    "    h_t = o * torch.tanh(c_t)\n    return h_t, c_t\n\n",
    "hidden, input_dim = 64, 32\ntorch.manual_seed(42)\n",
    "W_ih = torch.randn(4 * hidden, input_dim) * 0.1\n",
    "W_hh = torch.randn(4 * hidden, hidden) * 0.1\n",
    "b_ih = torch.zeros(4 * hidden)\nb_hh = torch.zeros(4 * hidden)\n",
    "h, c = torch.zeros(1, hidden), torch.zeros(1, hidden)\n",
    "xs = torch.randn(5, 1, input_dim)\n",
    'print("LSTM 序列前向 (5步):")\n',
    "for t in range(5):\n",
    "    h, c = lstm_step(xs[t], h, c, W_ih, W_hh, b_ih, b_hh)\n",
    '    print(f"  t={t}: h mean={h.mean():.4f}")\n',
    'print(f"\\n关键: d(c_t)/d(c_{{t-1}}) = f_t (逐元素乘, 无矩阵乘!)")\n',
    'print(f"对比 RNN: d(h_t)/d(h_{{t-1}}) approx W (矩阵乘, 易衰减)")'
], 'c29')

# GRU
md([
    "### 4.3 GRU vs LSTM——简化的智慧\n\n",
    "GRU 合并遗忘门+输入门→更新门 z，保留重置门 r。参数少 ~25%，多数任务效果相当。"
], 'c30')

code([
    "def gru_step(x_t, h_prev, W_ih, W_hh, b_ih, b_hh):\n",
    "    hidden = h_prev.shape[-1]\n",
    "    gates = x_t @ W_ih.T + b_ih + h_prev @ W_hh.T + b_hh\n",
    "    r = torch.sigmoid(gates[:, :hidden])\n",
    "    z = torch.sigmoid(gates[:, hidden:2*hidden])\n",
    "    n = torch.tanh(x_t @ W_ih[2*hidden:3*hidden].T + b_ih[2*hidden:3*hidden]\n",
    "        + r * (h_prev @ W_hh[2*hidden:3*hidden].T + b_hh[2*hidden:3*hidden]))\n",
    "    h_t = (1 - z) * h_prev + z * n\n    return h_t\n\n",
    "hd, inp = 128, 64\n",
    "lstm_p = 4 * hd * (hd + inp)\ngru_p  = 3 * hd * (hd + inp)\n",
    'print(f"参数量 (hidden={hd}, input={inp}):")\n',
    'print(f"  LSTM: {lstm_p:,} = 4 x {hd} x ({hd}+{inp})")\n',
    'print(f"  GRU:  {gru_p:,} = 3 x {hd} x ({hd}+{inp})")\n',
    'print(f"  GRU 省 {(1-gru_p/lstm_p)*100:.0f}%")\n\n',
    "torch.manual_seed(42)\n",
    "W_ih_g = torch.randn(3 * hd, inp) * 0.1\n",
    "W_hh_g = torch.randn(3 * hd, hd) * 0.1\n",
    "b_ih_g = torch.zeros(3 * hd)\nb_hh_g = torch.zeros(3 * hd)\n",
    "h = torch.zeros(1, hd)\n",
    "for t in range(5):\n",
    "    h = gru_step(torch.randn(1, inp), h, W_ih_g, W_hh_g, b_ih_g, b_hh_g)\n",
    'print(f"GRU 验证: output shape={tuple(h.shape)}, mean={h.mean():.4f}")'
], 'c31')

# ============================================================
# Section 5: Embedding
# ============================================================
md(["---\n## 5. 嵌入——离散世界的连续化翻译"], 'c32')
md([
    "### 5.1 Embedding 的本质——查表\n\n",
    "一个离散整数→1024维连续向量。语义相近的词在空间中距离近。",
    "Qwen3-0.6B 的 Embedding 有 1.56 亿参数，占总参 26%！"
], 'c33')

code([
    "vocab_size, emb_dim = 1000, 64\ntorch.manual_seed(42)\n",
    "embed = nn.Embedding(vocab_size, emb_dim)\n\n",
    'word_ids = {"猫":42, "狗":43, "老虎":44, "汽车":128, "火车":129, "飞机":130, "苹果":256, "香蕉":257}\n',
    "vecs = {w: embed(torch.tensor(tid)).detach() for w, tid in word_ids.items()}\n\n",
    'print("词向量余弦相似度矩阵:")\n',
    'words = list(word_ids.keys())\n',
    'print("      " + " ".join(f"{w:>6}" for w in words))\n',
    "for w1 in words:\n",
    '    row = f"{w1:>6} "\n',
    "    for w2 in words:\n",
    "        sim = F.cosine_similarity(vecs[w1].unsqueeze(0), vecs[w2].unsqueeze(0)).item()\n",
    '        row += f"{sim:>6.2f} "\n',
    "    print(row)\n\n",
    "real_vocab, real_dim = 151936, 1024\nparams = real_vocab * real_dim\n",
    'print(f"\\nQwen3-0.6B: {real_vocab}x{real_dim} = {params/1e8:.2f}亿参数 = {params/6e8*100:.0f}%总参")'
], 'c34')

# Position Encoding
md([
    "### 5.2 三种位置编码全家福\n\n",
    "可学习PE(BERT): 简单但无法外推。正弦PE(原版Transformer): 可外推但绝对位置。",
    "RoPE(Qwen/LLaMA): 编码相对位置，支持长上下文——现代 LLM 标配。"
], 'c35')

code([
    "seq_len, d_model = 64, 32\n",
    "position = np.arange(seq_len)[:, np.newaxis]\n",
    "div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))\n",
    "pe_sin = np.zeros((seq_len, d_model))\n",
    "pe_sin[:, 0::2] = np.sin(position * div_term)\n",
    "pe_sin[:, 1::2] = np.cos(position * div_term)\n\n",
    "learnable_pe = nn.Embedding(seq_len, d_model)\n",
    "pe_learn = learnable_pe(torch.arange(seq_len)).detach().numpy()\n\n",
    "freqs = 1.0 / (10000.0 ** (torch.arange(0, d_model, 2).float() / d_model))\n",
    "cos_rope = torch.cos(torch.arange(seq_len).float().unsqueeze(1) * freqs.unsqueeze(0))\n",
    "pe_rope = cos_rope.numpy()\n\n",
    "fig, axes = plt.subplots(1, 3, figsize=(16, 3.5))\n",
    'titles = ["Learnable PE (BERT)", "Sinusoidal (Transformer)", "RoPE cos (Qwen/LLaMA)"]\n',
    "for ax, title, d in zip(axes, titles, [pe_learn, pe_sin, pe_rope]):\n",
    '    im = ax.imshow(d.T, aspect="auto", cmap="RdBu_r", origin="lower")\n',
    '    ax.set_title(title, fontweight="bold", fontsize=12)\n',
    '    ax.set_xlabel("Position"); ax.set_ylabel("Dimension")\n',
    '    plt.colorbar(im, ax=ax, shrink=0.8)\n',
    'plt.suptitle("Position Encoding 对比", fontsize=15, fontweight="bold", y=1.02)\n',
    'plt.tight_layout(); plt.show()\n',
    'print("可学习PE: 简单, 无法外推")\n',
    'print("正弦PE: 可外推, 无参数, 但绝对位置")\n',
    'print("RoPE: 编码相对位置, 支持长上下文, 现代LLM标配")'
], 'c36')

# ============================================================
# Summary
# ============================================================
md(["---\n## Summary"], 'c37')
md([
    "| Concept | Experiment | Key Insight |\n",
    "|---------|-----------|-------------|\n",
    "| Conv 6 Params | kernel/stride/pad/dil/groups/bias | 两个 3x3 = 一个 5x5 感受野 |\n",
    "| im2col | 5x5 -> 9x9 matrix demo | 膨胀 k^2 倍内存换 GEMM 算力 |\n",
    "| Winograd | F(2x2,3x3) transform | 36->16 次乘法, 用加法换 |\n",
    "| Receptive Field | 3-layer VGG | RF=7, 参数 27 < 49 |\n",
    "| NCHW vs NHWC | Memory layout | 选对布局 = 2x 速度提升 |\n",
    "| SDPA 5 Steps | QKT->scale->mask->softmax->V | 每步有明确数学含义 |\n",
    "| Scale Factor | Distribution histogram | d_k 大时 scale 防 one-hot |\n",
    "| GQA | repeat_kv (8->16 heads) | KV Cache 减半 |\n",
    "| RoPE | Rotation + relative proof | 相对位置 = 外推能力 |\n",
    "| RNN Gradient | 0.9^100 ≈ 2.7e-5 | 指数衰减->长期依赖丢失 |\n",
    "| LSTM Highway | f_t * c_{t-1} (element-wise) | 无矩阵乘 = 梯度高速路 |\n",
    "| GRU vs LSTM | 3 gates vs 4 gates | 省 25% 参数, 效果相当 |\n",
    "| Embedding | Cosine similarity | 语义向量空间中靠近 |\n",
    "| Position Encoding | 3 types viz | RoPE = 现代 LLM 标配 |"
], 'c38')

# Write
with open('2.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(f'2.ipynb written: {len(nb["cells"])} cells')
