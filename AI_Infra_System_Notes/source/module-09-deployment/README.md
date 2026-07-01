# Module 9: 部署实战——从模型到服务

> 本模块带你完成端侧 (ncnn_llm)、本地 (llama.cpp) 和服务端 (vLLM) 三种部署，并做性能对比。

---

## 📋 学习目标

- [ ] 能编译运行 ncnn_llm，完成 Qwen3-0.6B 端侧推理
- [ ] 能编译运行 llama.cpp，对比端侧框架性能
- [ ] 能部署 vLLM 服务，进行并发压测
- [ ] 能解释 INT8 与 Vulkan 互斥的工程原因
- [ ] 能制作四框架性能对比矩阵

---

## 1. ncnn_llm 端侧部署

### 1.1 编译步骤

```bash
# 1. 编译 ncnn
cd ncnn-master
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DNCNN_VULKAN=ON ..
make -j$(nproc)

# 2. 编译 ncnn_llm
cd ncnn_llm-main
# 修改 CMakeLists.txt 中的 ncnn 路径
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

### 1.2 运行推理

```bash
# 命令行推理
./ncnn_llm_qwen3 \
    --model /path/to/Qwen3-0.6B \
    --prompt "你好，请介绍深度学习" \
    --max_tokens 256 \
    --temperature 0.7

# 关键参数:
# --vulkan         启用 Vulkan GPU 加速 (可能有互斥问题)
# --num_threads 4  CPU 线程数
# --debug          打印每层耗时
```

### 1.3 性能打点

在代码中加入计时：

```cpp
#include <chrono>
auto t1 = chrono::high_resolution_clock::now();
// ... embed ...
auto t2 = chrono::high_resolution_clock::now();
// ... decoder ...
auto t3 = chrono::high_resolution_clock::now();
// ... project + sample ...
auto t4 = chrono::high_resolution_clock::now();

printf("Embed: %.2f ms\n", chrono::duration_ms(t2-t1));
printf("Decoder: %.2f ms\n", chrono::duration_ms(t3-t2));
printf("Project+Sample: %.2f ms\n", chrono::duration_ms(t4-t3));
```

### 1.4 INT8 vs Vulkan 互斥问题 ⚠️

```
ncnn 的 INT8 需要:
  - Quantize layer (FP32→INT8)
  - INT8 Convolution kernel
  - Requantize layer (INT32→INT8)

ncnn 的 Vulkan 需要:
  - 对应的 compute shader

互斥原因:
  - Vulkan shader 可能没实现 INT8 版本的 attention
  - INT8↔FP32 频繁转换破坏了融合机会
  - GPU 上 INT8 compute 吞吐可能不如 FP16

工程决策:
  → 必须实测 CPU INT8 vs Vulkan FP16 的真实性能
  → 不同模型、不同硬件上结论不同
```

---

## 2. llama.cpp 本地推理

### 2.1 编译

```bash
cd llama.cpp-master
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF ..
make -j$(nproc)
```

### 2.2 下载模型并推理

```bash
# 下载 Qwen3-0.6B GGUF 格式
# (从 HuggingFace 下载或自行转换)

# 命令行推理
./llama-cli \
    -m qwen3-0.6b-Q4_K_M.gguf \
    -p "你好，请介绍深度学习" \
    -n 256 \
    --temp 0.7 \
    -t 4

# Server 模式
./llama-server \
    -m qwen3-0.6b-Q4_K_M.gguf \
    --port 8080
```

### 2.3 ncnn_llm vs llama.cpp 对比

| 维度 | ncnn_llm | llama.cpp |
|------|----------|-----------|
| 模型格式 | ncnn .param/.bin | GGUF |
| 量化支持 | INT8 静态量化 | Q4_0/Q4_K_M/Q8_0/IQ4 等 |
| GPU 后端 | Vulkan | CUDA/Metal/Vulkan |
| 代码量 | ~3K 行 | ~10W 行 |
| 学习难度 | ★★☆ | ★★★★ |
| 社区活跃度 | 低 | 极高 |

### 2.4 性能对比实验

```
同模型 (Qwen3-0.6B) 同硬件 (CPU 4 线程):

实验设计:
  1. 相同的 prompt (200 tokens)
  2. 相同的生成参数 (max 256, temp 0.7)
  3. 测量: TTFT, TPOT, 总耗时, 峰值内存

预期:
  - llama.cpp Q4_K_M:  最快 (极致 CPU 优化 + 4-bit 量化)
  - llama.cpp FP16:    较慢 (但比 ncnn FP32 快)
  - ncnn_llm INT8:     中间
  - ncnn_llm FP32:     最慢 (基线)
```

---

## 3. vLLM 服务端部署

### 3.1 安装与启动

```bash
# 安装
pip install vllm

# 启动 OpenAI-compatible API
vllm serve Qwen/Qwen3-0.6B \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90
```

### 3.2 测试 API

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

# 单请求测试
response = client.chat.completions.create(
    model="Qwen3-0.6B",
    messages=[{"role": "user", "content": "你好"}],
    max_tokens=256,
    temperature=0.7,
    stream=True,
)
for chunk in response:
    print(chunk.choices[0].delta.content or "", end="")
```

### 3.3 并发压测

```python
import asyncio, time
from concurrent.futures import ThreadPoolExecutor

async def benchmark(concurrency=10, requests=100):
    """并发压测 vLLM"""
    start = time.time()
    results = []

    async def send_one(i):
        t0 = time.time()
        # ... 发送请求 ...
        ttft = time.time() - t0  # 首 token 时间
        return {"ttft": ttft, "tokens": n_tokens}

    tasks = [send_one(i) for i in range(requests)]
    # 控制并发数
    for batch in chunks(tasks, concurrency):
        batch_results = await asyncio.gather(*batch)
        results.extend(batch_results)

    total_time = time.time() - start
    total_tokens = sum(r["tokens"] for r in results)

    print(f"Concurrency: {concurrency}")
    print(f"Throughput: {total_tokens / total_time:.1f} tokens/s")
    print(f"Avg TTFT: {np.mean([r['ttft'] for r in results])*1000:.0f} ms")
    return results

# 测试不同并发级别
for conc in [1, 5, 10, 20, 50]:
    benchmark(concurrency=conc, requests=100)
```

### 3.4 预期结果与观察

```
单请求 (conc=1):
  TTFT: ~100ms,  Throughput: ~30 tokens/s

中等并发 (conc=10):
  TTFT: ~150ms,  Throughput: ~200 tokens/s
  → Continuous Batching 开始起作用

高并发 (conc=50):
  TTFT: ~500ms,  Throughput: ~500 tokens/s
  → KV Cache 开始成为瓶颈, 部分请求排队

观察点:
  1. 吞吐不随并发线性增长 (受限于显存和带宽)
  2. TTFT 随并发增加而增加 (prefill 排队)
  3. KV Cache block 利用率 (vLLM metrics 可查)
```

---

## 4. 四框架性能对比矩阵

```
┌──────────────┬─────────┬──────────┬──────────┬──────────┐
│              │ ncnn_llm│llama.cpp │  vLLM    │ TensorRT │
├──────────────┼─────────┼──────────┼──────────┼──────────┤
│ 硬件         │  CPU    │  CPU     │  GPU     │  GPU     │
│ 最佳精度     │  FP32   │  Q4_K_M  │  FP16    │  FP8     │
│ 单请求速度    │  ★★    │  ★★★    │  ★★★    │  ★★★★   │
│ 并发吞吐      │  ★     │  ★★     │  ★★★★   │  ★★★★   │
│ 内存占用      │  ★★★★ │  ★★★★★  │  ★★     │  ★★     │
│ 学习难度      │  ★★    │  ★★★★   │  ★★★    │  ★★★★★  │
│ 部署场景      │  端侧   │  本地    │  云端    │  云端    │
│ 跨平台        │  YES   │  YES     │  GPU only│  NVIDIA  │
└──────────────┴─────────┴──────────┴──────────┴──────────┘
```

---

## 5. 部署排错指南

### ncnn_llm 常见问题

```
Q: 模型加载失败 (return code -1)
A: 检查 .param 和 .bin 文件路径 → 确认 model.json 配置 → 确认 ncnn 版本匹配

Q: Vulkan 比 CPU 还慢
A: 小模型 GPU launch overhead > 计算收益 → 尝试更大的 batch → 或关闭 Vulkan

Q: INT8 精度显著差
A: 量化校准数据与推理数据分布不匹配 → 确认 requantize 的 scale 是否正确
```

### vLLM 常见问题

```
Q: CUDA out of memory
A: 降低 max-model-len → 降低 gpu-memory-utilization → 开启 KV Cache 量化

Q: 吞吐不如预期
A: 增大 max_num_batched_tokens → 确认 GPU 利用率 (nvidia-smi) → 检查是否有请求 queue 堆积
```

---

## 🛠️ 实验清单

```
部署实验 (按优先级):

□ 1. 编译运行 ncnn_llm, 测试 Qwen3-0.6B 推理
□ 2. 给 generate() 加计时, 测量 embed/decoder/project/sample 各段时间
□ 3. 测试 ncnn_llm INT8 vs FP32 的精度和速度
□ 4. 编译运行 llama.cpp, 测试同一模型
□ 5. 做 ncnn_llm vs llama.cpp 同硬件性能对比
□ 6. 部署 vLLM, 跑通 API 调用
□ 7. vLLM 并发压测: 10/50/100 并发, 记录 TTFT/TPOT/Throughput
□ 8. vLLM 不同量化策略对比: FP16 vs INT8 (AWQ)
□ 9. 制作四框架性能对比矩阵
□ 10. 写部署总结报告
```

---

## 📚 延伸阅读

- [ncnn_llm_original_README.md](../ncnn_llm_original_README.md) — 源码级分析
- [07_ncnn_to_vllm_comparison.md](../../docs/07_ncnn_to_vllm_comparison.md) — 系统对比
- [06_quantization_optimization.md](../../docs/06_quantization_optimization.md) — 量化深入
- [05_execution_scheduler.md](../../docs/05_execution_scheduler.md) — 调度系统

---

*← 回到 [主 README](../README.md)*
