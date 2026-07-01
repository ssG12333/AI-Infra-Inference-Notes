# Module 9: 部署实战——从跑通到跑好

> 前面 8 个模块是"纸上谈兵"，本模块是"真枪实弹"。你会亲手编译 ncnn_llm、部署 vLLM 服务、做并发压测、制作性能对比矩阵——并在这个过程中学到所有书本上没有的工程教训。

---

## 📋 学习目标

- [ ] 能编译运行 ncnn_llm，完成 Qwen3-0.6B 端侧推理
- [ ] 能部署 vLLM 服务，进行并发压测
- [ ] 能制作 ncnn_llm / llama.cpp / vLLM 性能对比矩阵
- [ ] 能解释 INT8 与 Vulkan 互斥的工程原因
- [ ] 能解决常见部署错误 (OOM / 加载失败 / 精度退化)

---

## 1. 端侧部署——在手机上跑 LLM

### 1.1 编译 ncnn_llm

```bash
# 你的源码就在 源码/ncnn-master 和 源码/ncnn_llm-main

# 1. 编译 ncnn
cd ncnn-master && mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DNCNN_VULKAN=ON ..
make -j$(nproc)

# 2. 编译 ncnn_llm (记得改 CMakeLists.txt 里的 ncnn 路径)
cd ncnn_llm-main && mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

### 1.2 跑起来

```bash
./ncnn_llm_qwen3 \
    --model /path/to/Qwen3-0.6B \
    --prompt "你好，介绍一下你自己" \
    --max_tokens 256
```

### 1.3 性能打点——知道时间花在哪

```cpp
// 在 generate() 中加入计时:
auto t1 = chrono::now();
// ... embed ...
auto t2 = chrono::now();
// ... decoder ...
auto t3 = chrono::now();
// ... project + sample ...
auto t4 = chrono::now();

// 你会发现: Decoder 通常占 85%+ 的耗时
```

### 1.4 INT8 vs Vulkan：一个工程上的残酷事实

```
理论上 INT8 能让模型变小、变快。
理论上 Vulkan 能用 GPU 加速。

但现实中——INT8 和 Vulkan 可能互斥！

原因:
  - Vulkan shader 可能没实现 INT8 版本的 attention
  - INT8↔FP32 频繁转换破坏融合机会
  - GPU 上的 INT8 compute 吞吐可能不如 FP16

工程经验: 必须实测！不能用"理论"替代"benchmark"
  跑一遍 INT8 CPU vs Vulkan FP16 vs CPU FP32
  → 数据说话，不为理论站台
```

---

## 2. vLLM 服务端部署——高并发的艺术

### 2.1 启动服务

```bash
pip install vllm
vllm serve Qwen/Qwen3-0.6B \
    --host 0.0.0.0 --port 8000 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90
```

### 2.2 发请求

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="x")

response = client.chat.completions.create(
    model="Qwen3-0.6B",
    messages=[{"role": "user", "content": "你好"}],
    max_tokens=256, stream=True
)
for chunk in response:
    print(chunk.choices[0].delta.content or "", end="")
```

### 2.3 压测——让数字说话

```
并发数   TTFT(ms)   TPOT(ms)   Throughput(tok/s)
    1      120         8             30
   10      180        12            200     ← Continuous Batching 开始发力
   50      500        25            500     ← KV Cache 成瓶颈
  100     1200        50            600     ← 边际收益递减

观察:
  - 并发从 1→10，吞吐提升 6.7× (不是 10×)
  - 并发再大时 TPOT 明显恶化 → 显存带宽成为瓶颈
  - 最优并发数需要根据模型大小和 GPU 显存来调
```

### 2.4 排错速查

```
OOM:               降低 max-model-len / gpu-memory-utilization / 开 KV 量化
加载失败:          检查 model path / HF token / 磁盘空间
吞吐不如预期:      增大 max_num_batched_tokens / 检查 GPU 利用率
INT8 精度退化:     检查具体退化层 / 尝试 per-channel / 换 AWQ
Vulkan 比 CPU 慢:  小模型 GPU launch overhead > 收益 → 关闭 Vulkan
```

---

## 3. 四框架性能对比矩阵

```
              ncnn_llm    llama.cpp    vLLM      TensorRT-LLM
硬件           CPU         CPU          GPU        GPU
最佳精度       FP32        Q4_K_M       FP16       FP8
单请求速度     ★★          ★★★          ★★★        ★★★★
并发吞吐       ★           ★★           ★★★★       ★★★★
内存占用       ★★★★        ★★★★★        ★★         ★★
学习难度       ★★          ★★★★         ★★★        ★★★★★
场景           端侧         本地          云端        云端极致
```

---

## 🛠️ 实验清单

```
□ 1.  编译 ncnn_llm，跑通 Qwen3-0.6B 推理
□ 2.  加计时，测 embed/decoder/project/sample 耗时
□ 3.  ncnn_llm INT8 vs FP32 精度/速度对比
□ 4.  编译 llama.cpp，同模型同硬件对比
□ 5.  部署 vLLM，跑 API
□ 6.  vLLM 并发压测: 10/50/100，记录 TTFT/TPOT/Throughput
□ 7.  做四框架对比矩阵
□ 8.  写部署总结 (性能数据 + 踩坑记录)
```

---

*🏁 恭喜完成全部 9 个模块！回到 → [主 README](../README.md)*
