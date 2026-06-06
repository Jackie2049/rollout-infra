# CUTLASS 3.x GEMM Architecture Deep Dive

> 2026-06-07 | 源码深度阅读: NVIDIA CUTLASS 3.x (Hopper SM90+) GEMM Pipeline

## 一、CUTLASS 3.x 目录结构

```
cutlass/
├── include/cutlass/
│   ├── gemm/                    # GEMM 核心实现
│   │   ├── kernel/              # Kernel 入口 (GemmUniversal)
│   │   │   ├── sm90_gemm_tma_warpspecialized.hpp        # Warp-Specialized TMA kernel (522行)
│   │   │   ├── sm90_gemm_tma_warpspecialized_cooperative.hpp  # Cooperative 多WG版本
│   │   │   ├── sm90_gemm_tma_warpspecialized_pingpong.hpp     # Pingpong 双缓冲版本
│   │   │   ├── sm90_gemm_tma.hpp                            # 非WarpsSpecialized TMA版本
│   │   │   ├── sm100_gemm_tma_warpspecialized.hpp           # Blackwell SM100
│   │   │   └── sm70/sm80/...                                # 旧架构
│   │   ├── collective/          # Collective 操作 (Mainloop + Epilogue)
│   │   │   ├── sm90_mma_tma_gmma_ss_warpspecialized.hpp    # TMA+WGMMA Mainloop (584行)
│   │   │   ├── sm90_mma_tma_gmma_ss.hpp                    # 非WS Mainloop
│   │   │   ├── sm90_mma_tma_gmma_rs_warpspecialized.hpp    # A在寄存器版本
│   │   │   ├── sm90_mma_tma_gmma_ss_warpspecialized_fp8.hpp  # FP8版本
│   │   │   └── sm100_mma_*                                   # Blackwell
│   │   ├── device/              # Device-level API
│   │   ├── dispatch_policy.hpp  # 调度策略定义 (9+种)
│   │   ├── thread/warp/threadblock/  # 各层级MMA原子
│   │   └── kernel/sm90_tile_scheduler.hpp  # Tile调度器
│   ├── epilogue/                # Epilogue 实现
│   │   ├── collective/
│   │   │   ├── sm90_epilogue_tma_warpspecialized.hpp       # TMA Epilogue (963行)
│   │   │   ├── sm90_epilogue_tma_warpspecialized_bias_elementwise.hpp  # 融合bias+激活
│   │   │   └── sm100_epilogue_*                            # Blackwell
│   │   └── thread/              # Epilogue 原子操作
│   ├── pipeline/                # Pipeline 同步基础设施
│   │   ├── sm90_pipeline.hpp    # SM90 Pipeline (1388行) — 核心!
│   │   ├── sm100_pipeline.hpp   # Blackwell Pipeline
│   │   └── pipeline.hpp         # 通用Pipeline
│   ├── arch/                    # 硶件架构抽象
│   │   ├── mma_sm90.h           # WGMMA 原子定义
│   │   ├── barrier.h            # Barrier 实现
│   │   ├── reg_reconfig.h       # 寄存器重配置
│   │   └── grid_dependency_control.h  # GDC (Grid Dependency Control)
│   └── layout/                  # 内存布局 (swizzle, padding等)
├── examples/                    # 30+ 示例
├── python/                      # Python 绑定 (cutlass library)
└── docs/                        # 文档
```

## 二、SM90 GEMM 三种调度模式

CUTLASS 3.x 为 Hopper (SM90) 提供了3种Kernel调度模式:

| 模式 | Kernel Schedule | Warp Group | 特点 | 适用场景 |
|------|----------------|------------|------|---------|
| **Warp-Specialized** | KernelTmaWarpSpecialized | 1 Load WG + 1 MMA WG | 最简单, 1个CTA处理1个tile | 标准GEMM |
| **Pingpong** | KernelTmaWarpSpecializedPingpong | 1 Load WG + 2 MMA WG | 双MMA WG乒乓执行 | 大tile, 高计算密度 |
| **Cooperative** | KernelTmaWarpSpecializedCooperative | 1 Load WG + 2 MMA WG + TileScheduler | 多CTA协作, Persistent kernel | 小问题/大GPU利用率 |

### 核心设计: Warp Group Specialization

```
CTA (Thread Block) 内部角色分工:

Producer Warp Group (128 threads):
  ├── Warp 0: TMA Load (A+B) + Epilogue Load (C/bias)
  ├── Warp 1-3: 空闲(可用于其他任务)

Consumer Warp Group (128 threads):
  ├── 全128 threads: WGMMA (矩阵乘)
  └── 之后: Epilogue Store (D → gmem via TMA)

角色分配: WarpGroupRole::Producer vs WarpGroupRole::Consumer
```

**关键**: Producer(1个warp)做TMA异步加载 → Consumer(1个warp group=4 warp)做WGMMA → **计算和加载完全重叠**!

## 三、GEMM Pipeline 数据流

### 3.1 Kernel入口: GemmUniversal::operator()

```cpp
// sm90_gemm_tma_warpspecialized.hpp 核心流程
CUTLASS_DEVICE void operator()(Params const& params, char* smem_buf) {
    // 1. TMA Descriptor Prefetch (warp 0)
    CollectiveMainloop::prefetch_tma_descriptors(params.mainloop);
    CollectiveEpilogue::prefetch_tma_descriptors(params.epilogue);

    // 2. 初始化Pipeline (TMA async barrier)
    MainloopPipeline mainloop_pipeline(shared_storage.pipelines.mainloop, ...);
    EpiLoadPipeline epi_load_pipeline(shared_storage.pipelines.epi_load, ...);
    EpiStorePipeline epi_store_pipeline(...);

    // 3. Cluster同步
    cluster_wait_fn();  // Cluster>1: cluster_arrive+cluster_wait

    // 4. 分支: Producer vs Consumer
    if (warp_group_role == WarpGroupRole::Producer) {
        // TMA Load A+B → smem (逐K-tile)
        collective_mainloop.load(params.mainloop, mainloop_pipeline, ...);
        collective_mainloop.load_tail(mainloop_pipeline, ...);
        // Epilogue Load C/bias → smem (如果需要)
        collective_epilogue.load(epi_load_pipeline, ...);
        collective_epilogue.load_tail(epi_load_pipeline, ...);
    }
    else if (warp_group_role == WarpGroupRole::Consumer) {
        // WGMMA: smem A+B → accum (逐K-tile)
        Tensor accumulators = partition_fragment_C(tiled_mma, ...);
        collective_mainloop.mma(mainloop_pipeline, ..., accumulators, ...);
        collective_mainloop.mma_tail(mainloop_pipeline, ...);
        // Epilogue Store: accum → smem → gmem
        collective_epilogue.store(epi_load_pipeline, ..., accumulators, ...);
        collective_epilogue.store_tail(...);
    }
}
```

### 3.2 TMA Load: Producer视角

```cpp
// sm90_mma_tma_gmma_ss_warpspecialized.hpp — load() 方法
CUTLASS_DEVICE void load(...) {
    // 仅 elect_one_sync() 的线程执行TMA指令
    if (lane_predicate) {
        // 创建smem tensor views
        Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.data()), SmemLayoutA{});
        Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.data()), SmemLayoutB{});

        // Partition全局tensor到TMA slice
        auto block_tma_a = mainloop_params.tma_load_a.get_slice(cluster_local_block_id.y);
        auto block_tma_b = mainloop_params.tma_load_b.get_slice(cluster_local_block_id.x);

        // K-tile 循环
        for ( ; k_tile_count > 0; --k_tile_count) {
            pipeline.producer_acquire(smem_pipe_write);  // 等smem buffer空闲

            // TMA异步拷贝: gmem → smem (一个K-tile)
            // Barrier自动跟踪transaction完成
            BarrierType* tma_barrier = pipeline.producer_get_barrier(smem_pipe_write);
            copy(mainloop_params.tma_load_a.with(*tma_barrier, mcast_mask_a),
                 tAgA(_,_,_,*k_tile_iter), tAsA(_,_,_,write_stage));
            copy(mainloop_params.tma_load_b.with(*tma_barrier, mcast_mask_b),
                 tBgB(_,_,_,*k_tile_iter), tBsB(_,_,_,write_stage));
            ++smem_pipe_write;
        }
    }
}
```

**关键**: TMA是硬件级异步操作 → 1条指令拷贝整个tile → 不需要线程手动搬运 → 释放线程做其他计算!

### 3.3 WGMMA: Consumer视角

```cpp
// sm90_mma_tma_gmma_ss_warpspecialized.hpp — mma() 方法
CUTLASS_DEVICE void mma(MainloopPipeline pipeline, PipelineState smem_pipe_read,
                         FrgTensorC& accum, int k_tile_count, ...) {
    // Partition smem A/B到WGMMA fragment
    Tensor tCsA = thread_mma.partition_A(sA);  // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCsB = thread_mma.partition_B(sB);
    Tensor tCrA = thread_mma.make_fragment_A(tCsA);  // descriptor → register
    Tensor tCrB = thread_mma.make_fragment_B(tCsB);

    // Prologue MMA (前K_PIPE_MMAS个K-tile)
    warpgroup_fence_operand(accum);
    {
        pipeline.consumer_wait(smem_pipe_read, ...);  // 等TMA数据到达
        warpgroup_arrive();  // 128 threads同步
        // 第一个K-tile: ScaleOut=Zero(清零accumulator)
        cute::gemm(tiled_mma, tCrA(_,_,k_block,read_stage), tCrB(_,_,k_block,read_stage), accum);
        warpgroup_commit_batch();
        ++smem_pipe_read;
    }

    // Mainloop: 循环K-tile
    tiled_mma.accumulate_ = GMMA::ScaleOut::One;  // 累加模式
    for ( ; k_tile_count > 0; --k_tile_count) {
        pipeline.consumer_wait(smem_pipe_read, ...);
        warpgroup_fence_operand(accum);
        warpgroup_arrive();
        cute::gemm(tiled_mma, tCrA(_,_,_,read_stage), tCrB(_,_,_,read_stage), accum);
        warpgroup_commit_batch();

        warpgroup_wait<K_PIPE_MMAS>();  // 等MMA完成(释放smem buffer)
        pipeline.consumer_release(smem_pipe_release);  // 通知Producer可以写下一个K-tile
    }
}
```

### 3.4 Pipeline同步机制

```
TMA Load Pipeline (N-stage buffering):

Producer (TMA Load):                   Consumer (WGMMA):
  producer_acquire(stage_i)              consumer_wait(stage_i)
  │ 等empty_barrier                     │ 等full_barrier
  │ (smem buffer空闲)                    │ (TMA数据已到达)
  ↓                                      ↓
  TMA copy A+B → smem[stage_i]           WGMMA: smem[stage_i] → accum
  │ barrier自动追踪                       │ warpgroup_arrive()
  ↓                                      ↓
  ++smem_pipe_write                       warpgroup_commit_batch()
                                          warpgroup_wait<K_PIPE_MMAS>()
                                          consumer_release(stage_i)
                                          │ 通知empty_barrier
                                          │ (smem buffer释放)
```

**N-stage buffering**: smem有N个buffer → Producer写stage i时Consumer读stage i-K_PIPE_MMAS → 完全重叠!

## 四、Dispatch Policy 分类 (9+种SM90)

| Policy | A来源 | B来源 | 特点 |
|--------|-------|-------|------|
| MainloopSm90TmaGmma | smem | smem | 基础TMA+WGMMA |
| MainloopSm90TmaGmmaWarpSpecialized | smem | smem | WS动态调度 |
| MainloopSm90TmaGmmaRmemAWarpSpecialized | register | smem | A在寄存器→更灵活布局 |
| MainloopSm90TmaGmmaRmemAWarpSpecializedMixedInput | register | smem | 混合精度(FP16→FP8等) |
| MainloopSm90TmaGmmaWarpSpecializedFP8 | smem | smem | FP8专用 |
| MainloopSm90TmaGmmaWarpSpecializedBlockwiseFP8 | smem | smem | FP8+Blockwise scaling |
| MainloopSm90TmaGmmaWarpSpecializedSparse | smem | smem | 2:4稀疏 |
| MainloopSm90TmaGmmaWarpSpecializedSparseFP8 | smem | smem | 稀疏+FP8 |

**命名规则**: Tma=TMA加载, Gmma=WGMMA计算, Ss=双smem源, Rs=A在寄存器, WarpSpecialized=WS调度

## 五、Shared Memory 管理

### SmemLayout设计

```cpp
// SmemLayoutA: tile_to_shape(SmemLayoutAtomA{}, make_shape(BLK_M, BLK_K, Stages))
// SmemLayoutB: tile_to_shape(SmemLayoutAtomB{}, make_shape(BLK_N, BLK_K, Stages))

// 关键: SmemLayoutAtom是基本单元, 通过tile_to_shape扩展到完整tile+N-stage
// Step<_1,_2,_3>: 沿K维staging(标准GEMM tiling)
// Step<_2,_1,_3>: 沿M维staging(行列交换, 用于row-major A)
```

### Smem Reuse

```cpp
// Kernel层: union TensorStorage — Mainloop和Epilogue共享smem!
union TensorStorage {
    MainloopTensorStorage mainloop;
    EpilogueTensorStorage epilogue;  // Epilogue用完后, smem还给Mainloop
};
```

**设计**: Mainloop完成 → smem释放 → Epilogue用同一个smem空间 → 减少总smem需求!

### Cluster内Multicast

```cpp
// TMA_LOAD_MULTICAST: 一个TMA操作将数据广播到Cluster内的多个CTA
// mcast_mask_a: bitmask指定哪些CTA接收A数据
// 优势: 1次TMA传输 → 多个CTA同时获得数据 → 节省HBM带宽
```

## 六、Epilogue架构

SM90 Epilogue也使用TMA异步存储:

```
Epilogue Pipeline:
  Load (Producer): TMA Load C/bias → smem_C
  │ (Consumer完成MMA后)
  ↓
  Compute: accum + smem_C → smem_D (elementwise fusion)
  │ (bias + activation + scaling)
  ↓
  Store (Producer): TMA Store smem_D → gmem_D
  │ (异步, TMA硬件执行)
```

**Epilogue Fusion支持**:
- `sm90_epilogue_tma_warpspecialized.hpp`: 标准Epilogue (C→D)
- `sm90_epilogue_tma_warpspecialized_bias_elementwise.hpp`: 融合bias+elementwise激活

## 七、RTX 4090 (SM89) vs Hopper (SM90) 关键差异

| 特性 | RTX 4090 (SM89) | H100 (SM90) | 影响 |
|------|-----------------|-------------|------|
| TMA | **不支持** | **支持** | RTX 4090必须用cp.async代替TMA → 多线程搬运→效率低 |
| WGMMA | **不支持** | **支持**(128线程) | RTX 4090只能用warp-level MMA(32线程) → tile更小→compute密度低 |
| Cluster | **不支持** | **支持** | RTX 4090无法跨block同步+共享smem |
| Barrier | 简单 | ClusterTransactionBarrier | TMA barrier追踪transaction字节 |
| FP8 Tensor Core | 支持(基础) | 支持+Blockwise scaling | SM90有per-block FP8 scaling |

**RTX 4090不能用CUTLASS 3.x SM90 kernel!** SM89没有TMA/WGMMA → 必须回退到SM80(SM70)版本的CUTLASS kernel, 使用cp.async+HMMA。

CUTLASS SM80版本 (`sm80_mma_multistage.hpp`):
- 使用cp.async (而非TMA) 加载A+B → 多线程搬运
- 使用HMMA (warp-level MMA, 32 threads) → 每次计算更小tile
- 2-stage或3-stage pipeline → 比SM90的N-stage更少缓冲

## 八、CuTe (C++ Tensor Expression) 框架

CUTLASS 3.x基于CuTe —— 嵌入式tensor DSL:

```cpp
// CuTe核心概念:
make_tensor(ptr, layout)    // 创建tensor (指针+布局)
make_layout(shape, stride)  // 创建布局 (形状+步幅)
make_shape(M, K, L)         // 创建形状
local_tile(tensor, tile, coord)  // 切tile
partition_S/partition_D     // Partition源/目标
make_fragment_A/B           // 创建寄存器fragment (WGMMA descriptor)
cute::gemm(tiled_mma, A, B, C)  // 执行WGMMA
```

**CuTe设计哲学**: Tensor = Pointer + Layout → Layout = Shape + Stride → 所有操作都是layout变换 → 零开销编译时计算

## 九、Blackwell (SM100/SM120) 新特性

CUTLASS 3.x已支持Blackwell架构:

- `sm100_gemm_tma_warpspecialized.hpp`: Blackwell TMA+WGMMA
- `sm100_mma_warpspecialized.hpp`: SM100 MMA
- `sm100_blockscaled_mma_warpspecialized.hpp`: Blockwise scaling (NVFP4/FP8)
- `sm120_mma_tma.hpp`: SM120 (下一代)
- `sm100_tile_scheduler_*`: Blackwell tile调度器

Blackwell新增特性(推测):
- 更大WGMMA tile
- 更强FP8/FP4支持
- 更高效TMA

## 十、实际应用: vLLM/PyTorch 如何使用CUTLASS

PyTorch 2.x + vLLM 都使用cuBLAS作为GEMM backend → cuBLAS内部使用CUTLASS:
- PyTorch: `torch.matmul()` → cuBLAS → CUTLASS SM80/SM90 kernel
- vLLM: `x @ w` → cuBLAS → CUTLASS
- Triton: 自定义kernel → 不依赖CUTLASS → 但性能上限不如CUTLASS SM90

**我的CUDA kernel vs CUTLASS**:
- 我的RMSNorm kernel: 9x over PyTorch (warp shuffle)
- CUTLASS SM90 GEMM: 使用TMA+WGMMA → 比 SM80 HMMA 快2-3x
- 但RTX 4090 (SM89) 只能用SM80路径 → 我的kernel已接近SM80上限

## 十一、关键文件索引

| 文件 | 行数 | 内容 |
|------|------|------|
| `gemm/kernel/sm90_gemm_tma_warpspecialized.hpp` | 522 | WS kernel入口 |
| `gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp` | ~600 | Cooperative版本 |
| `gemm/kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp` | ~500 | Pingpong版本 |
| `gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp` | 584 | TMA+WGMMA Mainloop |
| `gemm/collective/sm90_mma_tma_gmma_ss.hpp` | 538 | 非WS Mainloop |
| `gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized_fp8.hpp` | ~600 | FP8版本 |
| `epilogue/collective/sm90_epilogue_tma_warpspecialized.hpp` | 963 | TMA Epilogue |
| `pipeline/sm90_pipeline.hpp` | 1388 | Pipeline同步 |
| `gemm/dispatch_policy.hpp` | ~400 | 调度策略定义 |
| `gemm/kernel/sm90_tile_scheduler.hpp` | ~300 | Tile调度器 |
| `arch/mma_sm90.h` | ~200 | WGMMA原子 |
| `arch/barrier.h` | ~100 | Barrier实现 |

Sources:
- [CUTLASS 3.x GitHub Repository](https://github.com/NVIDIA/cutlass)
- [CUTLASS Documentation](https://github.com/NVIDIA/cutlass/tree/main/docs)
- [CuTe Layout and Tensor DSL](https://github.com/NVIDIA/cutlass/tree/main/include/cute)
- [Hopper WGMMA Instructions (NVIDIA Docs)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)