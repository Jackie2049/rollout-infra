# GPU 性能分析工具与方法

> 从 nvidia-smi 到 Nsight — 定位 GPU 瓶颈的系统方法

## 1. 性能分析层次

```
Level 1: 系统级 — GPU 利用率是否满了?
  工具: nvidia-smi, dcgm
  关注: GPU 利用率, 显存占用, 温度, 功耗

Level 2: 应用级 — 时间花在哪里?
  工具: Nsight Systems (nsys), PyTorch Profiler
  关注: kernel 执行时间, 通信时间, CPU 瓶颈

Level 3: Kernel 级 — 为什么这个 kernel 慢?
  工具: Nsight Compute (ncu)
  关注: 带宽利用率, occupancy, 指令吞吐

Level 4: 指令级 — 哪些指令是热点?
  工具: Nsight Compute (详细模式)
  关注: pipeline stall 原因, bank conflict, 分支
```

## 2. nvidia-smi — 系统监控

### 2.1 基础命令

```bash
# GPU 状态一览
nvidia-smi

# 持续监控
watch -n 1 nvidia-smi

# 只看关键指标
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total --format=csv -l 1

# 进程级监控
nvidia-smi pmon -c 10  # 采样 10 次

# 算力利用率 (更精确)
nvidia-smi dmon -s u -d 1
# sm: SM 利用率, memory: 显存带宽利用率, encoder/decoder
```

### 2.2 关键指标解读

```
GPU-Util:     GPU 计算 (SM) 利用率
              0-100%, 看的是 "过去采样周期内是否有 kernel 在执行"
              高 (90%+): GPU 在持续计算
              低 (<50%): 可能有 CPU 瓶颈/数据瓶颈/通信瓶颈

Memory-Util:  显存带宽利用率
              高: kernel 在密集读写 HBM
              低: 计算密集型 kernel

GPU-Util 高 + Memory-Util 低 → 计算密集 (好, 大部分矩阵乘法)
GPU-Util 低 + Memory-Util 高 → 内存密集 (可能需要优化访存)
GPU-Util 低 + Memory-Util 低 → 空闲 (有优化空间)
```

## 3. Nsight Systems — 应用级分析

### 3.1 基本用法

```bash
# CPU + GPU 时间线
nsys profile -o trace python train.py

# 只 profile 一部分
nsys profile -o trace -c cudaProfilerApi python train.py
# 在代码中:
#   torch.cuda.profiler.start()
#   ...  # 要 profile 的代码
#   torch.cuda.profiler.stop()

# 常用参数
nsys profile \
  -o trace \              # 输出文件名
  -t cuda,nvtx,osrt \    # 跟踪 CUDA, NVTX, OS runtime
  -s none \               # 不自动分析 (节省时间)
  --force-overwrite=true  # 覆盖已有文件
  python train.py

# 分析结果
nsys stats trace.nsys-rep  # 生成文本报告
# 或用 Nsight Systems GUI 打开 .nsys-rep 文件
```

### 3.2 关键分析点

```
看时间线 (Timeline):
  1. CPU 和 GPU 是否重叠?
     → CPU dispatch kernel 时 GPU 应该在执行上一个

  2. 是否有大量 Gap?
     → GPU 空闲时间 = 浪费
     → 常见原因: CPU 瓶颈, 数据加载慢, 同步点

  3. 通信与计算是否重叠?
     → NCCL kernel 应该和计算 kernel 同时执行
     → 如果串行, 说明需要开启 overlap

  4. Kernel 执行时间分布?
     → 哪些 kernel 占时间最多?
     → 通常是 gemm (矩阵乘法) 和 attention

  5. Memory 操作?
     → 大量 memcpy 说明在 CPU↔GPU 搬数据
     → 应该用 pinned memory 或 prefetch
```

## 4. PyTorch Profiler — Python 集成

### 4.1 基本用法

```python
import torch.profiler as profiler

with profiler.profile(
    activities=[
        profiler.ProfilerActivity.CPU,
        profiler.ProfilerActivity.CUDA,
    ],
    schedule=profiler.schedule(
        wait=1,      # 跳过第一个 step (warmup)
        warmup=1,    # 1 个 step warmup
        active=3,    # 记录 3 个 step
        repeat=1,    # 只做一轮
    ),
    on_trace_ready=profiler.tensorboard_trace_handler('./profiler_logs'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    for step, batch in enumerate(dataloader):
        train_step(batch)
        prof.step()  # 通知 profiler 进入下一个 step

# 查看 CPU 时间排序
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))

# 查看 CUDA 时间排序
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
```

### 4.2 TensorBoard 可视化

```bash
# 安装
pip install torch-tb-profiler

# 启动
tensorboard --logdir=./profiler_logs

# 浏览器打开 http://localhost:6006
# 查看:
#   - Overview: 时间分布
#   - Operator View: kernel 级别耗时
#   - GPU Kernel View: CUDA kernel 统计
#   - Trace View: 时间线
#   - Memory View: 显存使用
```

## 5. Nsight Compute — Kernel 级分析

### 5.1 基本用法

```bash
# 分析特定 kernel
ncu --set full -o profile python train.py

# 只分析特定的 kernel (通过名字过滤)
ncu --kernel-name "regex:.*gemv.*" -o profile python train.py

# 快速分析 (只看关键指标)
ncu --set quick -o profile python train.py

# 查看 roofline 图
ncu --set roofline -o profile python train.py
```

### 5.2 关键指标

```
1. Roofline 分析:
   - 横轴: 算术强度 (FLOP/Byte)
   - 纵轴: 性能 (GFLOP/s)
   - 上限: GPU 峰值算力
   - 下限: 带宽上限
   → 看 kernel 是计算密集还是访存密集

2. Occupancy:
   - 每 SM 活跃 warp 数
   - 受 register 用量, shared memory 用量, block size 影响
   - 高 occupancy ≠ 高性能 (但通常有帮助)

3. 带宽利用率:
   - Achieved vs Peak 带宽比
   - DRAM 带宽: 数据从 HBM 到 L2 的速度
   - L2 带宽: 数据从 L2 到 SM 的速度

4. Stall 原因:
   - Memory dependency: 等待数据
   - Execution dependency: 等待前一条指令
   - Not selected: warp 调度未选中
   - Other: 同步、分支等
```

## 6. 分析工作流

```
Step 1: 确认 GPU 利用率
  nvidia-smi → GPU-Util < 70%?
  → 是: 有优化空间

Step 2: 找时间花在哪里
  nsys profile → 看 Timeline
  → 大量 Gap? → CPU 瓶颈 / 数据加载
  → 通信时间长? → 通信/计算重叠
  → 特定 kernel 慢? → 继续分析

Step 3: 分析热点 kernel
  PyTorch Profiler → 找出 top-5 耗时 kernel
  → 这些 kernel 占总时间的比例?

Step 4: Kernel 级优化
  ncu → Roofline + Stall 分析
  → 访存密集: 优化内存访问模式, 增加 tile
  → 计算密集: 优化指令, 利用 Tensor Core

Step 5: 验证优化效果
  对比优化前后的 nsys trace
  → GPU-Util 提升
  → 总时间缩短
```

## 7. NVTX 标注 — 让 Trace 更可读

NVTX (NVIDIA Tools Extension) 允许在代码中标记区域, 在 Nsight Systems 时间线中显示。

```python
import torch
from torch.cuda.nvtx import range_push, range_pop, range_start, range_end

# 方式 1: Push/Pop (嵌套)
range_push("attention_layer_0")
# ... attention 计算 ...
range_pop()

# 方式 2: Start/End (非嵌套)
handle = range_start("mlp_layer_0")
# ... MLP 计算 ...
range_end(handle)

# 方式 3: 装饰器
def nvtx_range(name):
    def decorator(func):
        def wrapper(*args, **kwargs):
            range_push(name)
            result = func(*args, **kwargs)
            range_pop()
            return result
        return wrapper
    return decorator

@nvtx_range("forward_pass")
def forward(x):
    return model(x)

# 方式 4: 上下文管理器
from contextlib import contextmanager

@contextmanager
def nvtx(name):
    range_push(name)
    try:
        yield
    finally:
        range_pop()

# 使用
with nvtx("prefill_phase"):
    model.prefill(tokens)
```

## 8. vLLM Profiling 实战

### 8.1 使用 Nsight Systems 分析 vLLM

```bash
# Profile vLLM 推理
nsys profile \
  -o vllm_trace \
  -t cuda,nvtx,osrt,nw \
  -s none \
  --force-overwrite=true \
  --duration=30 \
  python -c "
from vllm import LLM, SamplingParams

llm = LLM(model='meta-llama/Llama-3.1-8B', enforce_eager=True)
params = SamplingParams(max_tokens=256)

# Warmup
for i in range(3):
    llm.generate(['Hello'] * 8, params)

# Profile 这部分
import torch.cuda as cuda
cuda.profiler.start()
for i in range(10):
    llm.generate(['Explain quantum computing'] * 16, params)
cuda.profiler.stop()
"
```

### 8.2 vLLM 内置 NVTX 标注

vLLM 源码中已经包含 NVTX 标注, 关键区域:

```
Timeline 上的标注:
  ┌─ ModelRunner.execute_model ─────────────────────────────┐
  │  ┌─ prepare_inputs ─┐  ┌─ forward ────────────────┐     │
  │  │  prepare ATTN     │  │  Layer 0                 │     │
  │  │  metadata         │  │  ├── Attention           │     │
  │  │                   │  │  └── MLP                 │     │
  │  │                   │  │  Layer 1                 │     │
  │  │                   │  │  ├── Attention           │     │
  │  │                   │  │  └── MLP                 │     │
  │  │                   │  │  ...                     │     │
  │  │                   │  │  Layer N                 │     │
  │  │                   │  │  └── Sampling            │     │
  │  └───────────────────┘  └─────────────────────────┘     │
  └──────────────────────────────────────────────────────────┘

GPU Stream:
  Stream 0 (compute): [Kernel][Kernel][Kernel]...[Sampling]
  Stream 1 (copy):         [D2H async]    [D2H async]
                          ↑ copy_stream   ↑ copy_stream
```

### 8.3 vLLM 关键性能指标

```python
# 使用 vLLM 内置 metrics
from vllm import LLM

# 开启 Prometheus metrics
llm = LLM(model="...", enable_prefix_caching=True)

# 通过 /metrics 端点获取:
# - vllm:num_requests_running  (并发数)
# - vllm:gpu_cache_usage_perc  (KV Cache 使用率)
# - vllm:e2e_request_latency   (端到端延迟)
# - vllm:time_to_first_token   (TTFT)
# - vllm:time_per_output_token  (TPOT)
```

### 8.4 常见 vLLM 性能瓶颈

```
症状 1: GPU-Util < 50%
  原因: 请求不足 / CPU 调度瓶颈
  定位: nsys 看 CPU 线程是否有阻塞
  解决: 增加并发请求 / 检查 tokenizer 预处理

症状 2: KV Cache 使用率 > 90%
  原因: GPU 内存不足, 频繁抢占
  定位: vllm:num_preemptions_total 持续增长
  解决: 减少 max_model_len / 增加 gpu_memory_utilization

症状 3: TTFT 高但 TPOT 正常
  原因: Prefill 瓶颈 (长 prompt)
  定位: nsys 看 prefill kernel 时间
  解决: Chunked prefill / 增加前缀缓存

症状 4: TPOT 高但 GPU-Util 正常
  原因: KV Cache 大导致 memory-bound 加重
  定位: nsys 看 attention kernel 时间占比
  解决: 减少 batch size / 使用 FP8 KV cache

症状 5: 请求排队 > 50
  原因: 吞吐量不足
  定位: vllm:num_requests_waiting 高
  解决: 增加 replicas / 减小 max_model_len
```

## 9. 分布式训练 Profiling

### 9.1 NCCL 通信分析

```bash
# Profile NCCL 通信
nsys profile \
  -t cuda,nvtx,nccl \
  -s none \
  torchrun --nproc_per_node=4 train.py

# NCCL 环境变量 (启用详细日志)
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# 查看 NCCL 拓扑
nccl topo
nvidia-smi topo -m  # 查看 GPU 拓扑

# 常见拓扑:
#   GPU0-GPU1: NVLink (600 GB/s)
#   GPU0-GPU4: PCIe + NVSwitch (同节点)
#   GPU0-GPU8: Ethernet/InfiniBand (跨节点)
```

### 9.2 通信与计算重叠验证

```
理想的时间线 (通信被计算隐藏):
  GPU 0: [Compute]→[AllReduce ║ Compute]→[AllReduce ║ Compute]
         ←────────── 通信被计算覆盖 ──────────→

未重叠的时间线 (通信串行):
  GPU 0: [Compute]→[AllReduce]→[Compute]→[AllReduce]
                   ↑ GPU 空闲等待 ↑

检查方法:
  1. nsys 时间线中搜索 "nccl" kernel
  2. 检查 nccl kernel 是否与 gemm/attention kernel 时间重叠
  3. 如果不重叠, 检查是否启用了 overlap (e.g., ZeRO overlap)
```

### 9.3 ZeRO Stage 分析

```python
# Profiling ZeRO 各阶段
import torch.profiler as profiler

with profiler.profile(
    activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
) as prof:
    # 一个训练步
    optimizer.zero_grad()
    loss = model(batch)
    loss.backward()
    optimizer.step()

# 查看 AllReduce/ReduceScatter 时间
print(prof.key_averages().table(
    sort_by="cuda_time_total",
    row_limit=30,
    filter_name="nccl|all_reduce|reduce_scatter"
))
```

## 10. Memory Profiling

### 10.1 CUDA 内存分析

```python
import torch

# PyTorch 内存统计
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Reserved:  {torch.cuda.memory_reserved() / 1e9:.2f} GB")
print(f"Max allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

# 内存快照 (详细分析)
torch.cuda.memory._record_memory_history()
# ... 运行代码 ...
snapshot = torch.cuda.memory._snapshot()
torch.cuda.memory._save_memory_snapshot("snapshot.pickle")

# 用可视化工具打开
# python -m torch.cuda.memory._snapshot_viz snapshot.pickle
```

### 10.2 Nsight Systems 内存追踪

```bash
# 跟踪内存分配
nsys profile \
  -t cuda,nvtx,osrt \
  --cuda-memory-usage=true \
  -o mem_trace \
  python train.py

# 在 Nsight Systems GUI 中:
# View → CUDA Memory → 查看分配/释放事件
# 可以看到:
#   - 每次分配的大小和地址
#   - 峰值内存使用
#   - 内存碎片 (分配/释放的间隙)
```

## 11. 常用 Profile 命令速查

```bash
# === 快速检查 ===
nvidia-smi                                    # GPU 状态
nvidia-smi dmon -s pucvmet -i 0 -d 1         # 持续监控

# === Nsight Systems ===
nsys profile -t cuda,nvtx -o trace python app.py       # 基本
nsys profile -t cuda,nvtx,nccl -o trace torchrun ...    # 含 NCCL
nsys profile --duration=10 -o trace python app.py       # 限时
nsys profile -c cudaProfilerApi -o trace python app.py  # 手动控制
nsys stats trace.nsys-rep                               # 文本报告

# === Nsight Compute ===
ncu --set full -o profile python app.py                 # 完整分析
ncu --set roofline -o profile python app.py             # Roofline
ncu --kernel-name "regex:.*cutlass.*" -o profile ...    # 过滤 kernel
ncu --launch-skip 100 --launch-count 10 -o profile ...  # 跳过+限制

# === PyTorch Profiler ===
# 见第 4 节

# === 内存分析 ===
torch.cuda.memory._snapshot()                           # PyTorch 内存快照
```

## 参考

- [NVIDIA Nsight Systems Documentation](https://docs.nvidia.com/nsight-systems/)
- [NVIDIA Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
- [PyTorch Profiler Tutorial](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- [CUDA Performance Analysis](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#performance-analysis)
