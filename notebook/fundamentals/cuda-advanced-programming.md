# CUDA Advanced Programming for AI Infra

> 2026-06-07 | 基于CUDA C++ Programming Guide + 实际kernel开发经验 + RTX 4090实测

## 一、Warp-Level Primitives

### Warp Shuffle Instructions

Warp = 32 threads, 同步执行 → 无需显式同步, 可直接在warp内交换数据.

```cuda
// Warp shuffle: 直接在warp内传递数据, 不经过shared memory!
float val = __shfl_sync(0xFFFFFFFF, my_val, src_lane);  // 从src_lane获取
float sum = __shfl_down_sync(0xFFFFFFFF, my_val, delta); // 向下传递delta步
float sum2 = __shfl_up_sync(0xFFFFFFFF, my_val, delta);   // 向上传递
float xor_val = __shfl_xor_sync(0xFFFFFFFF, my_val, lane_mask); // XOR交换

// Butterfly reduction (warp内sum):
// 我的RMSNorm backward kernel用了这个!
float sum = val;
for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset);
}
// 结果: 所有32个lane都有相同的sum → O(log2(32)) = 5步
```

**优点 vs Shared Memory**:
- Shuffle: 不占用shared memory, 更快(寄存器级交换)
- Shared Memory: 更灵活(跨warp通信), 但有bank conflict风险

**我的实测**: RMSNorm backward kernel用butterfly reduction → CUDA 2.2x over PyTorch (比Triton 1.7x更快) → warp shuffle是关键优势!

### Cooperative Groups (CG)

CUDA 9+引入的统一线程协作API, 替代传统`__shfl_*`:

```cuda
// Warp-level group
cooperative_groups::thread_block_tile<32> warp = cooperative_groups::tiled_partition<32>(block);
float sum = cooperative_groups::reduce(warp, val, cooperative_groups::plus<float>());

// Block-level group
cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
block.sync();  // 替代 __syncthreads()

// Cluster-level (Hopper SM90+)
cooperative_groups::cluster cluster = cooperative_groups::this_cluster();
cluster.sync();  // 跨block同步!

// Grid-level (需要launch参数配置)
// cooperative_groups::this_grid() — 整个grid同步
```

**CG层级**:
| 层级 | 线程数 | 用途 | GPU要求 |
|------|--------|------|---------|
| Warp | 32 | reduction, shuffle | 所有GPU |
| Block | ≤1024 | block内同步 | 所有GPU |
| Cluster | 多block | 跨block同步+共享 | SM90+(Hopper) |
| Grid | 整个grid | 全kernel同步 | SM90+ |

## 二、Shared Memory Bank Conflicts

### 基础知识

- Shared memory有32个bank (每个bank 4字节带宽)
- 同一warp内多个线程访问同一bank的不同地址 → **bank conflict** → 序列化访问
- 同一warp内多个线程访问同一bank的同一地址 → **broadcast** → 无冲突

### Bank映射规则

```
对于4字节访问:
  bank = (address / 4) % 32

示例: float data[32][33];  // 33 = 32 + 1 padding
  data[0][0] → bank 0
  data[0][1] → bank 1
  ...
  data[0][31] → bank 31
  data[0][32] → bank 0  ← 但data[1][0]也→bank 0? 不是!
  data[1][0] → bank 0 + 32 = (1*33*4/4) % 32 = 33 % 32 = 1 ← 不同bank!
```

### Bank Conflict优化策略

```cuda
// 策略1: Padding (最常用)
// 原始: float data[N][M]; → M=32时每行起始bank相同 → 冲突!
// 修复: float data[N][M+1]; → 加1列padding → 每行起始bank偏移1 → 无冲突!

// 策略2: 访问模式重组
// 原始: thread i访问 data[i][j] → 行优先 → 可能冲突
// 修复: thread i访问 data[j][i] → 列优先 → 相邻线程访问不同bank

// 策略3: 使用shuffle替代shared memory
// reduction不需要shared memory → 用__shfl_xor → 零bank conflict

// 策略4: 8字节访问 (双bank合并)
// CUDA允许8字节访问占用2个连续bank → float2/double访问
// double: 8字节 → bank = (address/8) % 16 (只有16个有效bank对)
```

### 我的RMSNorm kernel中的bank conflict

```cuda
// RMSNorm backward: 3-pass (inv_rms → dot/dw/dr → dx)
// Pass 1: 每行1个warp → butterfly reduction → shuffle, 无bank conflict
// Pass 2: dw跨行累加 → atomicAdd → 无bank conflict (atomic是硬件级)
// Pass 3: dx = inv_rms * (dy*w - x_norm*coeff)
//         coeff = warp内broadcast → shuffle → 无bank conflict

// 关键: 1 warp/row设计 → 所有reduce都用shuffle → 自然避免bank conflict!
```

## 三、Occupancy 优化

### 什么是Occupancy?

Occupancy = 活跃warp数 / 最大warp数(SM支持)

高occupancy → 更多warp → 更好隐藏延迟(内存/计算切换)
低occupancy → SM资源浪费 → 延迟未隐藏

### Occupancy影响因素

| 因素 | 影响 | RTX 4090 (SM89) |
|------|------|----------------|
| Block size | 太小→warp少, 太大→block少 | 128-512 threads推荐 |
| Registers/thread | 每warp越多→同SM并发warp少 | 限制≤64 registers |
| Shared memory/block | 太多→同SM并发block少 | 动态分配更灵活 |
| Max threads/SM | 硬件限制 | 1536 threads (48 warps) |
| Max blocks/SM | 硬件限制 | 32 blocks |

### Occupancy计算

```
假设: block_size=256 (8 warps), registers=32/thread, smem=4KB/block

每SM寄存器: 256 threads × 32 registers = 8192 registers
SM总寄存器: 65536 → 可容纳 65536/8192 = 8 blocks × 256 = 2048 threads

但max threads/SM=1536 → 1536/256 = 6 blocks
→ Occupancy = 6×8 warps / 48 max warps = 48/48 = 100%!

如果registers=64: 256×64=16384 → 65536/16384=4 blocks
→ 4×256=1024 threads → 1024/1536=66.7% occupancy
→ 寄存器是occupancy瓶颈!
```

### 优化策略

```
1. --ptxas-options=-v: 编译时显示register/smem使用
2. __launch_bounds__(maxThreadsPerBlock, minBlocksPerSM):
   告诉编译器预期并发 → 优化register分配
3. 动态shared memory: extern __shared__ → 运行时决定大小
4. Register spill: 如果register太多 → 编译器spill到local memory(L1→慢)
   → 用__launch_bounds__强制编译器减少register使用
```

### 我的RMSNorm kernel occupancy

```
1 warp/row设计:
  block = 1 warp = 32 threads → 太小! → occupancy低
  修复: 多行/block → 32 threads × multiple rows

实际: 我用了32 threads/block(1 warp) → occupancy低
但RMSNorm每行独立 → 不需要多warp → 1 warp够了
而且RTX 4090 128 MPs → 即使每个SM只有1 warp也能饱和(行数足够)
```

## 四、CUDA Streams 优先级

```cuda
// Stream优先级: 高优先级stream可抢占低优先级
cudaStream_t high_priority_stream, low_priority_stream;
cudaStreamCreateWithPriority(&high_priority_stream, cudaStreamNonBlocking, 0);   // 最高
cudaStreamCreateWithPriority(&low_priority_stream, cudaStreamNonBlocking, -1);   // 最低

// 用途:
// 1. 推理: decode stream(高优先级) + prefill stream(低优先级)
//    → decode不被prefill阻塞 → 更低TPOT
// 2. 训练: gradient stream(高) + communication stream(低)
//    → 关键计算优先完成
```

**我的实测**: RTX 4090双stream重叠无收益(0.93x) → 大GPU(A100/H100)才有效.
原因: RTX 4090单stream已接近饱和 → 重叠无额外GPU资源可用.

## 五、Dynamic Parallelism

```cuda
// Parent kernel内启动child kernel → 无需CPU介入
__global__ void parent_kernel(...) {
    // 在GPU上直接启动子kernel
    child_kernel<<<1, 32>>>(...);
    cudaDeviceSynchronize();  // 等child完成(在GPU上!)
}

// 用途:
// 1. 递归算法(树搜索, 自适应网格)
// 2. 不确定并行度的任务(动态分配)
// 3. 多级嵌套并行(如batch→row→element)

// 注意:
// - 深度嵌套有开销(child launch + synchronization)
// - 需要CUDA 5.0+
// - 不是所有场景都比单kernel快 → 需要实测
```

## 六、TMA (Tensor Memory Accelerator) — Hopper新特性

```cuda
// TMA: 异步bulk copy从global→shared memory → 释放线程做计算
// SM90+ (H100/H200) only!

// 传统: 线程手动加载global到shared
for (int i = threadIdx.x; i < tile_size; i += blockDim.x) {
    smem[i] = gmem[offset + i];
}
__syncthreads();

// TMA: 硬件自动搬运 → 线程做计算!
__shared__ float smem[TILE_SIZE];
cuda::memcpy_async(block, smem, gmem + offset, TILE_SIZE);
// 线程可以在等待TMA完成时做其他计算!
cuda::pipeline::commit();
cuda::pipeline::wait();  // 等TMA完成

// CUTLASS 3.x 大量使用TMA → GEMM吞吐大幅提升
```

**RTX 4090不支持TMA**(SM89, 需SM90) → 我的kernel不能用.

## 七、Warp Group (128 threads) — Hopper MMA

```cuda
// SM90+: warp group = 4 warps = 128 threads
// WGMMA指令: warp group级别的矩阵乘 → 比传统warp MMA更快

// CUTLASS 3.x WGMMA:
// 128 threads协作 → 更大tile → 更高compute密度
// TMA提供数据 → warp group做MMA → 完美重叠
```

## 八、实际kernel开发经验总结

### 我的RMSNorm kernel开发教训

```
1. Warp shuffle >> Shared memory reduction:
   Butterfly reduction: 5步完成32-element sum → 比shared memory快
   实测: CUDA 9x vs Triton 1.8x → shuffle是关键

2. 1 warp/row 简化设计:
   每行独立 → 无跨行依赖 → 无bank conflict → 代码简单
   缺点: occupancy低 → 但行数足够时128 SM饱和

3. FP32 accumulation → dtype convert:
   FP16/BF16 input → FP32计算 → dtype output
   精度: FP32/BF16 backward PASS, FP16 rel diff 3.9e-4 OK

4. atomicAdd for dw:
   dw跨行累加 → atomicAdd最简单
   每行1个dw贡献 → atomics少 → 开销可忽略

5. 2-pass backward (v6 optimization):
   forward保存inv_rms → backward省1-pass → 22%加速!
   关键: 仔细分析数据流, 看哪些中间结果可以保存

6. API变更: c10::cuda::getCurrentCUDAStream() (不是at::cuda!)
   PyTorch 2.9改变了API → 编译错误 → 需要查最新文档
```

### Kernel开发决策树

```
需要自定义kernel吗?
│
├── 先benchmark PyTorch/Triton → 如果够快就不写!
│   (教训: Triton softmax自定义反而慢3x)
│
├── 确定并行模式:
│   ├── 每元素1线程 → 简单但低occupancy
│   ├── 每行1warp → 中等, 适合reduction
│   ├── 每tile多warp → 高occupancy, 适合GEMM
│   └── 每block多行 → 高occupancy + bank conflict需管理
│
├── 数据流分析:
│   ├── 哪些中间结果可保存? → 减少pass数
│   ├── 哪些reduce可用shuffle? → 避免shared memory
│   ├── 哪些累加可用atomicAdd? → 简化跨行操作
│
├── 精度选择:
│   ├── FP32 accumulation → 高精度, 但需要类型转换
│   ├── FP16/BF16直接操作 → 快但精度低
│   ├── Mixed: 输入FP16 → 计算FP32 → 输出FP16 → 最优权衡
│
└── 编译验证:
    ├── --ptxas-options=-v → 检查register/smem使用
    ├── cuda-memcheck → 检查内存错误
    ├── 正确性: cos_sim≈1.0, diff<FP16精度范围
    ├── 性能: vs PyTorch/Triton → 需要实际加速才值得
```

## 九、RTX 4090 vs Hopper(H100) 特性对比

| 特性 | RTX 4090 (SM89) | H100 (SM90) |
|------|-----------------|-------------|
| Warp shuffle | 支持 | 支持 |
| Cooperative Groups | warp/block | warp/block/cluster/grid |
| Shared Memory | 100KB/SM | 228KB/SM |
| TMA | **不支持** | **支持**(异步bulk copy) |
| WGMMA | **不支持** | **支持**(128线程MMA) |
| FP8 Tensor Core | 支持 | 支持(更强) |
| Dynamic Parallelism | 支持 | 支持(更高效) |
| Cluster | **不支持** | **支持**(跨block共享) |

**教训**: SM89(RTX 4090)没有TMA和WGMMA → kernel优化上限受限. 大部分优化靠warp shuffle + FP32 accumulation + 精心数据流.

Sources:
- [CUDA C++ Programming Guide v12.x](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [NVIDIA Cooperative Groups Documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cooperative-groups)
- [FlashAttention-3 (Dao et al., 2024)](https://arxiv.org/abs/2407.0868)
- [CUTLASS 3.x Documentation](https://github.com/NVIDIA/cutlass)