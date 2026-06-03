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

## 参考

- [NVIDIA Nsight Systems Documentation](https://docs.nvidia.com/nsight-systems/)
- [NVIDIA Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
- [PyTorch Profiler Tutorial](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- [CUDA Performance Analysis](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#performance-analysis)
