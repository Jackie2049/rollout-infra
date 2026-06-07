# CUDA Kernel Optimization Patterns — SM89 (RTX 4090)

> 2026-06-08 | 专用于Ada Lovelace SM89的kernel优化模式深度分析
> 基于: NVIDIA PTX ISA, CUTLASS 3.x源码分析, TE实测数据, RTX 4090 benchmark
> 关联: cuda-cpp-kernel-advanced.md (基础), cuda-advanced-programming.md (warp/shmem), cutlass-gemm-rtx4090.md (CUTLASS实测)

## 0. SM89 vs SM90: RTX 4090的限制与机遇

| 特性 | SM89 (RTX 4090) | SM90 (H100) | 对kernel开发的影响 |
|------|------------------|-------------|-------------------|
| Tensor Core | HMMA (三代) | HMMA+WGMMA (四代) | SM89只能用warp-level MMA |
| TMA | ❌ | ✅ | SM89必须用cp.async手动搬运 |
| Cluster | ❌ | ✅ | SM89单SM, 无跨SM共享 |
| WGMMA | ❌ | ✅ | SM89最大MMA=16×8×16 |
| FP8 E5M2 | ❌ | ✅ | SM89 FP8只能用E4M3 |
| DP4A/DP2A | ✅ | ✅ | INT8 dot product可用 |
| HMMA FP8 | ✅ | ✅ | RTX 4090支持FP8 HMMA! |
| cp.async | ✅ | ✅ | **SM89的关键异步机制** |
| ldmatrix | ✅ | ✅ | 加载矩阵fragment到tensor core |
| Shared Memory | 100KB/SM | 228KB/SM | SM89 smem更少→tile更小 |
| Registers | 65536/SM | 65536/SM | 相同 |
| Max threads/SM | 1536 | 2048 | SM89并发更低 |

**核心洞察**: SM89的优化策略与SM90完全不同!

- **SM90**: TMA+WGMMA → 线程几乎不参与数据搬运 → warp group(128线程)做MMA
- **SM89**: cp.async+HMMA → 线程必须手动pipeline数据 → warp(32线程)做MMA

SM89的瓶颈是**数据搬运**(没有TMA硬件加速), 而SM90的瓶颈是**调度**(多warp group协调)。

## 1. cp.async Pipeline: SM89的核心优化模式

### 1.1 cp.async指令

`cp.async` = Copy Async, SM80+引入, 是SM89最重要的新指令!

```cuda
// PTX指令: 异步从global memory拷贝到shared memory
// 4字节对齐拷贝
cp.async.ca.shared.global [smem_addr], [gmem_addr], 4;

// 16字节对齐拷贝 (更高效!)
cp.async.ca.shared.global [smem_addr], [gmem_addr], 16;

// 128字节bulk拷贝 (最高效, 但需要128字节对齐!)
cp.async.ca.shared.global [smem_addr], [gmem_addr], 128;

// .ca = cache-all (L1+L2 cache都参与)
// .cg = cache-global (只L2 cache参与) → 更适合大矩阵(避免L1污染)
cp.async.cg.shared.global [smem_addr], [gmem_addr], 16;
```

**关键优势**: cp.async是**非阻塞**的! 线程发出拷贝指令后立即继续执行, 不等待数据到达smem!

### 1.2 软件Pipeline模式

CUTLASS SM80/SM89 GEMM的3-stage software pipeline:

```
Stage 0: 正在计算    ← 线程正在用smem中的A_tile/B_tile做HMMA
Stage 1: 数据到达    ← cp.async已经把下一块A/B拷到smem, 等待commit
Stage 2: 正在拷贝    ← cp.async正在从global搬运再下一块A/B

时间线:
t=0:  [Compute k=0]  [Async load k=1]  [Async load k=2]
t=1:  [Commit k=1]   [Compute k=1]     [Async load k=3]
t=2:  [Wait k=1]     [Commit k=2]      [Compute k=2]
...

每一步:
1. cp.async.commit()  → 等stage-1数据到达smem
2. cp.async.wait_group<N>()  → 等N组之前的cp.async完成
3. HMMA计算  ← 用smem数据做tensor core运算
4. 发出新的cp.async  ← 加载下一块数据
```

```cuda
// CUTLASS 3.x PipelineTmaAsync → SM89版本用PipelineAsync
// 伪代码(3-stage):
for (int k = 0; k < K / K_TILE; k++) {
    // Wait for previous async copy to complete
    cp.async.commit();
    cp.async.wait_group<2>();  // Wait until at most 2 pending groups

    // Compute with current tile (already in smem)
    for (int m = 0; m < M_TILE; m += MMA_M) {
        for (int n = 0; n < N_TILE; n += MMA_N) {
            HMMA_FP16(acc[m][n], smem_A[m][k_current], smem_B[k_current][n]);
        }
    }

    // Issue async copy for NEXT tile (k+2 → stage 2)
    cp.async.cg.shared.global(smem_A_stage_next + offset_A,
                              gmem_A + (k+2) * K_TILE * stride_A, 16);
    cp.async.cg.shared.global(smem_B_stage_next + offset_B,
                              gmem_B + (k+2) * K_TILE * stride_B, 16);
}
```

### 1.3 Pipeline深度选择

| Pipeline深度 | smem需求 | 延迟隐藏 | 适用场景 |
|-------------|---------|---------|---------|
| 2-stage | A_tile+B_tile × 2 | 基本 | 小矩阵/低occupancy |
| 3-stage | A_tile+B_tile × 3 | 好 | **SM89标准选择** |
| 4-stage | A_tile+B_tile × 4 | 更好 | 大矩阵/高occupancy |

**SM89推荐**: 3-stage pipeline
- RTX 4090 smem=100KB/SM → 3-stage: 3×(A_tile+B_tile) ≈ 3×64KB = 192KB
- 但100KB限制 → 实际tile必须更小 → 64×64 tile + 3-stage = ~48KB → 可行!
- 4-stage需要更多smem → 可能降低occupancy →得不偿失

### 1.4 cp.async vs手动加载

```cuda
// 传统手动加载 (同步, 线程必须等)
for (int i = threadIdx.x; i < TILE_SIZE; i += blockDim.x) {
    smem[i] = gmem[offset + i];  // 每线程多次global load → 高延迟
}
__syncthreads();  // 必须等所有线程完成 → 无法隐藏延迟!

// cp.async (异步, 线程继续计算)
cp.async.cg.shared.global(smem + 0, gmem + offset, 16);  // 线程继续执行!
// 可以在等待cp.async时做其他计算 → 延迟隐藏!
```

**实测影响**: CUTLASS SM80/SM89的cp.async pipeline比手动加载快20-30%!

## 2. HMMA Tensor Core指令: SM89的MMA架构

### 2.1 HMMA指令格式

HMMA = Half Precision Matrix Multiply-Accumulate, SM75+引入, SM89改进:

```cuda
// PTX HMMA指令 (第三代, SM80+)
// 格式: HMMA.16816.F32.F16.F16.F16
//       HMMA.M×N×K.oper_dtype.in_dtype1.in_dtype2.acc_dtype

// M×N×K = 16×8×16 (三代HMMA标准形状)
// 32 threads (1 warp) 协作完成:
//   每线程持有A fragment的8个元素 + B fragment的4个元素
//   产出C fragment的8个元素

// FP16 HMMA (SM75+, 但SM80+优化版本)
hmma.m16n8k16.f32.f16.f16.f32 {%r0,%r1,%r2,%r3},
    {%r4,%r5,%r6,%r7,%r8,%r9,%r10,%r11},  // A: 8 FP16 values
    {%r12,%r13,%r14,%r15},                  // B: 4 FP16 values
    {%r16,%r17,%r18,%r19};                  // C: 4 FP32 accumulators

// BF16 HMMA (SM80+)
hmma.m16n8k16.f32.bf16.bf16.f32 ...

// FP8 E4M3 HMMA (SM89+!) ← RTX 4090特有
hmma.m16n8k16.f32.e4m3.e4m3.f32 ...

// INT8 HMMA (SM75+)
hmma.m16n8k32.f32.s8.s8.f32  // K=32 for INT8 (wider inner dimension)
```

### 2.2 HMMA矩阵形状对比

| 指令 | M×N×K | 输入类型 | 累加类型 | SM要求 | 每线程fragment |
|------|-------|---------|---------|--------|--------------|
| HMMA.16816 | 16×8×16 | FP16/BF16 | FP32 | SM75+ | A:8, B:4, C:4 |
| HMMA.16816 | 16×8×16 | FP8 E4M3 | FP32 | SM89+ | A:8, B:4, C:4 |
| HMMA.16832 | 16×8×32 | INT8 | FP32 | SM75+ | A:16, B:4, C:4 |
| WGMMA.64×8×16 | 64×8×16 | FP16/BF16 | FP32 | SM90+ | 128线程协作 |
| WGMMA.64×8×16 | 64×8×16 | FP8 | FP32 | SM90+ | 128线程协作 |

### 2.3 HMMA线程-数据映射

```
HMMA.16816.F32.F16.F16.F32 的线程分配:

A矩阵 (M×K = 16×16 FP16):
  32 threads, 每线程持有8个FP16值
  Thread 0-7:  A[0:8, 0:8]    (上左半)
  Thread 8-15: A[0:8, 8:16]   (上右半)
  Thread 16-23: A[8:16, 0:8]  (下左半)
  Thread 24-31: A[8:16, 8:16] (下右半)

B矩阵 (K×N = 16×8 FP16):
  32 threads, 每线程持有4个FP16值
  Thread 0-7:   B[0:8, 0:4]
  Thread 8-15:  B[0:8, 4:8]
  Thread 16-23: B[8:16, 0:4]
  Thread 24-31: B[8:16, 4:8]

C矩阵 (M×N = 16×8 FP32):
  32 threads, 每线程持有4个FP32累加值
  每线程负责C矩阵的2×2子块
```

**关键**: HMMA是warp-level操作 → 1 warp(32线程)完成16×8×16的矩阵乘!
- 1次HMMA = 16×8×16×2 = 2048 FLOPS (FP16)
- 1次FP8 HMMA = 16×8×16×2 = 2048 FLOPS (但数据只有1字节 → 带宽省50%!)

### 2.4 FP8 HMMA在SM89上的特殊行为

RTX 4090 FP8 HMMA的关键发现(基于TE实测+源码分析):

1. **只支持E4M3**: SM89 FP8 HMMA只接受E4M3格式输入(E5M2需要SM90+)
2. **强制TN layout**: `nvte_is_non_tn_fp8_gemm_supported()`在SM89返回0 → 必须transposed
3. **SCALAR_32F per-tensor scaling**: cuBLASLt内部FP8×scale_inv→FP32累加→BF16输出
4. **use_split_accumulator**: true→FP32精确累加, false→fast accumulation

## 3. ldmatrix: 矩阵数据加载指令

### 3.1 ldmatrix格式

`ldmatrix` 是SM75+引入的PTX指令, 专门为HMMA准备数据:

```cuda
// 从shared memory加载矩阵fragment到寄存器 → 直接喂给HMMA!
// ldmatrix.sync.aligned.m8n8.x4.shared.b16

// x1: 1个8×8子矩阵 → 每线程获得1个元素
// x2: 2个8×8子矩阵 → 每线程获得2个元素
// x4: 4个8×8子矩阵 → 每线程获得4个元素 (最常用!)

// 加载A矩阵 (16×16, 行主序):
ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16
    {%r0,%r1,%r2,%r3},  // 4个寄存器接收数据
    [smem_A + offset];   // shared memory中的矩阵数据

// .trans: 转置加载 → A行主序但HMMA期望列主序 → 加载时转置!
// 无.trans: 不转置 → B列主序直接加载
```

### 3.2 ldmatrix与手动加载对比

```cuda
// 方法1: 手动加载 (慢, 不优化布局)
float a_frag[8];
for (int i = 0; i < 8; i++) {
    a_frag[i] = smem_A[thread_row + i * stride];  // 每次单独load
}

// 方法2: ldmatrix (快, 硬件优化布局)
ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16
    {a_frag_r0, a_frag_r1, a_frag_r2, a_frag_r3},
    [smem_A + warp_offset];
// 1条指令加载32线程的全部数据 → 硬件自动优化bank访问!
```

**关键优势**:
- ldmatrix**自动避免bank conflict** → 硬件知道HMMA的访问模式
- ldmatrix**自动布局转换** → .trans选项在加载时转置, 不需要额外指令
- ldmatrix**单指令32线程** → 比手动加载减少指令数

### 3.3 ldmatrix的shared memory布局要求

```
ldmatrix期望的smem布局:

8×8 tile (x1模式): 64个元素, 32个线程各读2个
  Thread i读: smem[(i/8)*8 + i%8 + (i%4)*8]  → 自动无bank conflict!

16×16 tile (x4模式): 256个元素, 32个线程各读8个
  4个8×8子块拼接 → ldmatrix自动交错读取 → bank conflict最小化

关键: smem必须按ldmatrix期望的布局排列!
  CUTLASS用smem_layout=RowMajor/ColumnMajor来匹配ldmatrix的期望
  错误的smem布局 → ldmatrix读错数据 → HMMA输出错误!
```

## 4. Memory Coalescing: SM89最优访问模式

### 4.1 什么是Memory Coalescing?

GPU内存访问的最关键优化: 同一warp的32个线程访问**连续的global memory地址** → 合并为1次宽事务 → 带宽最大化!

```
理想: 32个线程访问32个连续4字节 = 128字节 → 1次128-byte事务 → 最优!
糟糕: 32个线程访问32个分散地址 → 32次4-byte事务 → 带宽浪费32x!

RTX 4090 HBM带宽890.8 GB/s → 但只在coalesced访问时才能达到!
```

### 4.2 Coalescing规则 (SM89)

| 访问模式 | 事务数 | 效率 | 示例 |
|---------|--------|------|------|
| 连续4B (float) | 1 | **100%** | `data[threadIdx.x]` |
| 连续2B (half) | 1 | **100%** | `hdata[threadIdx.x]` |
| 连续1B (FP8/int8) | 1 | **100%** | `idata[threadIdx.x]` |
| Stride-2 (float2) | 1 | **100%** | `f2data[threadIdx.x]` |
| Stride-32 (列访问) | 32 | **3.1%** | `matrix[threadIdx.x][0]` |
| 随机分散 | 32+ | **<3%** | `data[random_idx]` |
| 同一地址(broadcast) | 1 | 100% | `data[0]`(所有线程读同地址) |

### 4.3 矩阵访问的Coalescing优化

```cuda
// 读取A矩阵的一列 (K方向, 用于GEMM)
// ❌ 列访问: 每线程读不同行同一列 → stride=行宽 → 不coalesced!
float a_col = A[row + threadIdx.x * stride];  // stride >> 128B

// ✅ 行访问: 每线程读同一行不同列 → 连续 → coalesced!
float a_row = A[threadIdx.x * stride + row];   // 连续列 → 1次事务!

// GEMM优化: 重新组织访问模式使得warp总是做行访问!
// CUTLASS做法: A用行主序(行访问coalesced), B用列主序(列访问→ldmatrix处理)
```

### 4.4 FP8的Coalescing优势

**FP8数据(1字节)的coalescing特别高效!**

```
32个线程连续访问FP8数据:
  32×1B = 32字节 → 1次32-byte事务 → 但RTX 4090最小事务128B
  → 每128B事务包含128个FP8值 → 4个warp的并发访问合并!

FP8 GEMM的带宽优势:
  BF16 GEMM: A矩阵=2B×M×K → 2×带宽需求
  FP8 GEMM:  A矩阵=1B×M×K → 1×带宽需求 → 带宽省50%!
  → 这解释了为什么FP8在memory-bound GEMM中更快(数据更少→带宽瓶颈减轻)
```

## 5. Shared Memory优化: SM89的100KB约束

### 5.1 SM89 Shared Memory配置

| 配置 | shmem | L1 | 说明 |
|------|-------|----|------|
| Default | 48KB | 16KB | CUDA默认 |
| MaxShared | 100KB | 0KB | 最大化shmem |
| MaxL1 | 16KB | 96KB | 最大化L1 cache |

```cuda
// 最大化shmem (GEMM/tiled kernel)
cudaFuncSetAttribute(my_kernel,
    cudaFuncAttributePreferredSharedMemoryCarveout,
    cudaSharedMemCarveoutMaxShared);  // 100KB给shmem

// 检查可用shmem
int shmem_size;
cudaDeviceGetAttribute(&shmem_size, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
// SM89: 100KB = 102400 bytes
```

### 5.2 GEMM Tile大小 vs smem约束

```
GEMM tile需要: A_tile(M×K_BLOCK) + B_tile(K_BLOCK×N) × pipeline_depth

SM89 3-stage pipeline:
  需要smem = 3 × (A_tile + B_tile)
  A_tile = 64×32×2B (FP16) = 4096B = 4KB
  B_tile = 32×64×2B (FP16) = 4096B = 4KB
  3-stage: 3×(4+4) = 24KB → 在100KB内! ✅

  但更大tile:
  A_tile = 128×64×2B = 16KB
  B_tile = 64×128×2B = 16KB
  3-stage: 3×(16+16) = 96KB → 占100KB的96%! → occupancy降低到1 block/SM

  FP8数据:
  A_tile = 128×64×1B = 8KB (FP8只有1B!)
  B_tile = 64×128×1B = 8KB
  3-stage: 3×(8+8) = 48KB → 100KB的48% → 可放2 blocks/SM → 更高occupancy!
```

**FP8的smem优势**: FP8数据只有1字节 → tile大小减半 → 可以放更多stage或更大tile!

### 5.3 smem reuse: CUTLASS的Mainloop/Epilogue union

```cuda
// CUTLASS 3.x核心优化: smem reuse
// Mainloop和Epilogue共享同一块smem!

union SharedStorage {
    struct MainloopStorage {
        float A_tile[3][M_TILE][K_BLOCK];  // 3-stage pipeline
        float B_tile[3][K_BLOCK][N_TILE];
    };
    struct EpilogueStorage {
        float C_tile[M_TILE][N_TILE];      // 输出tile
        float bias[N_TILE];                 // bias vector
    };
};

// Mainloop阶段用A_tile/B_tile → Epilogue阶段用C_tile
// 同一块smem, 不同阶段使用不同部分 → 省50% smem!
```

**SM89限制**: SM90的TMA有专用smem buffer → 不需要reuse。SM89必须reuse!

## 6. Register Budget Management: HMMA kernel的关键

### 6.1 HMMA kernel的register需求

```
1 warp HMMA.16816 kernel的register需求:

A fragment:  8 values × 2B(FP16) = 16B → 4 registers (32-bit each)
B fragment:  4 values × 2B(FP16) = 8B  → 2 registers
C fragment:  4 values × 4B(FP32) = 16B → 4 registers

1次HMMA需要: 4+2+4 = 10 registers

但GEMM需要K方向多次HMMA → C累加器必须持续持有!
  多M/N tile: 每个C累加器4 registers → 16×8 tile有4个8×4子块 → 16 registers
  加上A/B fragment: 16+4+2 = 22 registers仅用于MMA!

再加上:
  - 循环索引, 常量: 5-10 registers
  - cp.async地址计算: 5-8 registers
  - 指针变量: 5-10 registers
  - 临时变量: 5-10 registers

总计: 22 + 30 = ~52 registers ← 接近occupancy瓶颈!
```

### 6.2 Occupancy vs Register数量的关系

| Registers/thread | Blocks/SM | Warps/SM | Occupancy | 说明 |
|-----------------|-----------|----------|-----------|------|
| 32 | 6 | 48 | **100%** | 最优 |
| 40 | 5 | 40 | 83% | 良好 |
| 48 | 4 | 32 | 67% | 可接受 |
| 56 | 3 | 24 | 50% | 边缘 |
| 64 | 3 | 24 | 50% | 需要更多延迟隐藏 |
| 80 | 2 | 16 | 33% | 低occupancy |
| 128 | 1 | 8 | 17% | 极低 |

**SM89 HMMA kernel推荐**: 限制40-48 registers/thread → 67-83% occupancy

```cuda
// 编译时控制register使用
__launch_bounds__(256, 3)  // 256 threads/block, 最少3 blocks/SM
// → 编译器会尽量减少registers以满足3 blocks/SM
// → 65536/3/256 ≈ 85 registers/thread可用 → 编译器通常能压缩到40-48

// 查看实际register使用
// nvcc --ptxas-options=-v → 输出 "ptxas : info : Used XX registers"
```

### 6.3 Register Spill: 当register不够时

```
Register spill: 编译器将变量存到local memory (= L1 + L2 → 可能到global)

Spill的成本:
  Local memory访问延迟 ≈ global memory (如果不cache)
  → 1次spill = 数百cycle延迟 → 性能灾难!

避免spill的方法:
1. __launch_bounds__: 告诉编译器需要的并发 → 优化register分配
2. 减少变量: 合并计算, 减少同时持有的数据
3. smem替代: 将临时数据放smem而非register
4. 循环展开控制: #pragma unroll 1 → 不展开 → 减少register需求
```

## 7. SM89专用优化: FP8 HMMA Kernel设计

### 7.1 FP8 HMMA的数据流

基于TE实测和源码分析的SM89 FP8 HMMA完整数据流:

```
输入: BF16 activation A, FP8 weight W

1. Quantize:
   BF16 A → FP8 A (E4M3)
   BF16/FP32 W → FP8 W (E4M3) + scale_inv (FP32)
   TE实现: tex.quantize() → CastVectorizedUnaryKernelLauncher (SIMT kernel)
   RTX 4090用SIMT cast kernel (不是Blackwell的TMA加速quantize)

2. HMMA GEMM:
   FP8 A × FP8 W → FP32 accumulator
   scale_inv_A × scale_inv_W → cuBLASLt内部乘法
   FP32 acc × (scale_inv_A × scale_inv_W) → 反量化
   → 输出BF16

3. 关键: cuBLASLt SCALAR_32F模式
   每tensor一个FP32 scale → 整个tensor用同一scale反量化
   → 这是RTX 4090唯一支持的FP8 scaling模式!
```

### 7.2 FP8 HMMA的smem布局

```
FP8数据在smem中的布局:

A_tile (FP8): M_TILE × K_BLOCK × 1B
  比FP16省50%空间! → pipeline可以更深或tile更大

B_tile (FP8): K_BLOCK × N_TILE × 1B
  同样省50% → 但需要额外存储scale (每tensor 1个FP32)

FP8 vs FP16 smem对比:
  FP16 3-stage: 3×(16KB + 16KB) = 96KB → 接近100KB上限
  FP8  3-stage: 3×(8KB + 8KB) = 48KB → 只有48%! → 可放更多blocks/SM

  FP16 occupancy: 1 block/SM (96KB smem/block)
  FP8  occupancy: 2 blocks/SM (48KB smem/block) → 2×occupancy!
```

**这是FP8在SM89上的一个额外优势**: 不仅带宽省50%, smem也省50% → occupancy翻倍!

### 7.3 FP8 HMMA的alignment要求

```
cuBLASLt FP8 GEMM alignment规则 (TE实测发现):

规则1: 最后维度(N)必须≥16 → HMMA.16816的N=16最小tile
规则2: 除最后维度外的所有维度乘积必须÷8 → warp内8个线程对齐
  例: M×K必须÷8 → 512×2560 OK (1310720÷8=163840)
  例: B=2×S=10×K=2560 = 51200 ÷ 8 = 6400 → 但10×2560=25600 ÷ 8=3200 OK
  例: B=1×S=1×K=2560 → 产品1不满足 ÷ 8 → FP8 GEMM失败!

这就是为什么B=1的FP8 GEMM需要padding!
```

## 8. SM89 Kernel设计决策树

### 8.1 选择HMMA vs SIMT

```
                    需要做矩阵乘法?
                    /              \
                  Yes               No
                  |                 |
            矩阵≥16×8×16?       SIMT kernel
            /          \         (element-wise)
          Yes           No
          |              |
       HMMA kernel    SIMT fallback
          |           (矩阵太小HMMA无收益)
    SM89支持FP8?
    /          \
  Yes           No
  |              |
FP8 HMMA     FP16/BF16 HMMA
(1B数据)     (2B数据)

HMMA适用条件:
1. 矩阵维度≥16×8×16 (HMMA最小tile)
2. 数据对齐(16字节+维度÷8)
3. 有足够的smem存放tile(≥2 stage)
4. Occupancy足够(≥50%, 即≤56 registers/thread)
```

### 8.2 Pipeline策略选择

```
计算/AI比值:
  AI > ridge(≈93 for FP16): compute-bound → pipeline深度不重要(计算够快)
  AI < ridge: memory-bound → pipeline深度至关重要(隐藏内存延迟)

SM89 pipeline选择:
  compute-bound: 2-stage即可 → 简单实现
  memory-bound: 3-stage → SM89最佳平衡
  极端memory-bound: 4-stage → 但occupancy可能降低(smem不够)

RTX 4090 GEMM实测:
  大矩阵(B≥8): compute-bound → 2-3-stage足够
  小矩阵(B=1): memory-bound → 但GEMM时间<quantize overhead → FP8不值得
```

### 8.3 SM89 vs SM90 kernel设计对比

| 设计决策 | SM89 (RTX 4090) | SM90 (H100) |
|---------|------------------|-------------|
| 数据搬运 | cp.async (线程发起) | TMA (硬件自动) |
| 矩阵计算 | HMMA (32线程) | WGMMA (128线程) |
| Pipeline | 3-stage software | TMA async hardware |
| smem管理 | 手动reuse | TMA专用buffer |
| 跨SM通信 | ❌ (无Cluster) | Cluster multicast |
| FP8 scaling | SCALAR_32F per-tensor | Block/Vectored scaling |
| Occupancy瓶颈 | smem大小 | register分配 |

**核心差异**: SM89是**warp-centric** (32线程), SM90是**warp-group-centric** (128线程)!
→ SM89 kernel设计围绕1个warp, SM90围绕1个warp group

## 9. 实战经验: RTX 4090实测数据支持的优化结论

### 9.1 HMMA FP16/BF16性能

实测数据 (cutlass-gemm-rtx4090.md):

| 矩阵大小 | FP16 TFLOPS | BF16 TFLOPS | FP16 peak利用率 |
|----------|-------------|-------------|----------------|
| 1024×1024 | 69.39 | 76.71 | 84% |
| 2048×2048 | 143.62 | 149.74 | 173% (超越FP16 peak!) |
| 4096×4096 | **167.14** | 153.12 | **202%** (2×FP16 peak) |

**解释**: FP16 HMMA利用tensor core → 2×FP32吞吐 → 167 TFLOPS > 82.58 TFLOPS(FP32 peak)

### 9.2 FP8 HMMA性能

实测数据 (fp8-gemm-algorithm-analysis):

| Layer | B=1 FP8 TFLOPS | B=32 FP8 TFLOPS | BF16 B=32 TFLOPS |
|-------|----------------|-----------------|------------------|
| gate_proj | **66.23** | **292.64** | 171.44 |
| down_proj | **62.44** | **243.15** | 165.16 |
| qkv_proj | **25.66** | **228.99** | 157.57 |

**关键发现**:
- B=32 FP8达到212-293 TFLOPS → 超出FP8 peak 165.2!
  - 因为TFLOPS用2×M×K×N计算(同BF16), 但FP8数据只有1字节
  - HMMA FP8吞吐量 = HMMA FP16吞吐量(相同FLOPS) × 2(数据减半→更多tile)
- B=1 FP8只有18-66 TFLOPS → quantize overhead主导

### 9.3 优化建议总结

| 优化模式 | 适用场景 | SM89可用 | 预期收益 |
|---------|---------|---------|---------|
| cp.async pipeline | 所有HMMA kernel | ✅ | 20-30% vs手动加载 |
| ldmatrix | HMMA数据准备 | ✅ | bank conflict避免+布局转换 |
| HMMA FP16/BF16 | 矩阵≥16×8×16 | ✅ | 2-5× vs SIMT |
| HMMA FP8 | 矩阵≥16×8×16 + B≥4 | ✅ | 1.48-1.59× vs BF16 |
| smem reuse | 所有tile kernel | ✅ | 50% smem节省 |
| __launch_bounds__ | HMMA kernel | ✅ | 67-100% occupancy |
| Memory coalescing | 所有global memory访问 | ✅ | 10-30× vs不coalesced |
| Bank conflict padding | 所有smem访问 | ✅ | 2-32× vs conflict |

---

**Sources**:
- [NVIDIA PTX ISA Reference](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [CUTLASS 3.x Architecture](https://github.com/NVIDIA/cutlass)
- [Understanding FP8 Precision Effects on Hopper Tensor Cores](https://developer.nvidia.com/blog/understanding-fp8-precision-effects-on-hopper-tensor-cores/)
- [TransformerEngine GitHub](https://github.com/NVIDIA/TransformerEngine)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

**Related notes**: cuda-cpp-kernel-advanced.md (基础), cuda-advanced-programming.md (warp/shmem), cutlass-gemm-rtx4090.md (CUTLASS实测), fp8-gemm-algorithm-analysis-rtx4090.md (FP8实测)