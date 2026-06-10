# GPU Memory Management & Virtualization Deep Dive

> 2026-06-10 | GPU内存=推理的瓶颈! 从CUDA内存层级到PagedAttention, 从BlockPool到prefix caching, 内存管理=推理核心!
> 关联: kv-cache-management-deep-dive.md, cuda-memory-allocator-rtx4090.md, scheduler-architecture-deep-dive.md

## 0. 核心定律: 内存 = 推理的最大约束

GPU推理不是compute-bound → 是memory-bound → 内存决定吞吐!

关键事实:
- 7B decode B=1: 仅利用0.6% peak TFLOPS → 98.4%计算能力浪费 → memory-bound!
- HBM带宽=890.8 GB/s (RTX 4090) → 权重读取=瓶颈 → 每次decode读14GB权重!
- KV cache内存=并发上限 → KV越多→并发越多→吞吐越高 → 但内存有限!
- 量化=内存优化的核心 → INT4→75%省 → INT8 KV→50%省 → 更多空间→更多并发!

## 1. CUDA Memory Hierarchy — GPU内存层级

```
CUDA内存层级 (RTX 4090 Ada):

1. Registers (per SM):
   - 65536 registers per SM → 每warp最多256 → 每thread最多255
   - 最快 → 0 cycle latency → 但容量小 → kernel设计关键!
   - occupancy = registers_per_thread → 太多register→低occupancy→低性能!

2. Shared Memory (per SM):
   - 100KB configurable per SM → 32 banks → 4 bytes per bank
   - ~20 cycle latency → SRAM → kernel间通信+数据复用!
   - bank conflict → padding避免 → 32 bank并行 → 128 bytes×8 rows=1024!
   - vLLM PagedAttention: smem存储partial softmax结果 → online softmax!

3. L2 Cache (global):
   - 72MB (RTX 4090) → 比A100大4.5x → 低occupancyOK → cache替代延迟隐藏!
   - ~200 cycle latency → 自动管理 → 程序员不直接控制!
   - cache line=128 bytes → 工作集fit L2 → 高吞吐(798 GB/s实测)!
   - L2 hit rate → weight reuse → decode B=32时权重常驻L2 → 省HBM带宽!

4. HBM (global memory):
   - 24GB → 890.8 GB/s实测 → 主要存储 → 模型+KV cache!
   - ~400 cycle latency → 最慢 → 但容量最大 → 瓶颈所在!
   - 权重14GB(BF16)/3.5GB(INT4) → KV cache剩余 → 并发上限!

5. Host Memory (CPU):
   - 系统RAM → pinned memory → PCIe传输 → 12 GB/s → swap路径!
   - swap: KV cache→CPU pinned→PCIe→回GPU → 比recompute快4-44x!

内存层级决策树:
  数据常访问 → Registers (最快, 程序员手动管理)
  数据kernel间复用 → Shared Memory (SM级, 程序员手动管理)
  数据偶尔访问 → L2 Cache (自动管理, hit率高)
  数据长期存储 → HBM (最大容量, 最慢, 程序员分配)
  数据暂存备用 → Host Memory (CPU, PCIe传输)
```

## 2. PagedAttention — 虚拟内存化管理

```
PagedAttention = OS虚拟内存分页 → GPU KV cache!

类比OS虚拟内存:
  OS: page(4KB) → page_table → virtual→physical → 按需分配!
  PA: block(16 tokens) → block_table → logical→physical → 按需分配!

vLLM V1数据结构:
  KVCacheBlock: block_id(int) + ref_cnt(int) + block_hash(Optional)
    → block_id → physical block在KV cache tensor中的位置!
    → ref_cnt → 引用计数 → 多请求共享prefix → ref_cnt>1!
    → block_hash → prefix caching → hash→查找→命中→省计算!

  BlockPool: 管理所有block → alloc+free+cache+evict!
    → blocks: list[KVCacheBlock] → 全部block对象 → 预分配!
    → free_block_queue: FreeKVCacheBlockQueue → 双向链表 → LRU驱逐!
    → cached_block_hash_to_block: BlockHashToBlockMap → hash→block查找!

  FreeKVCacheBlockQueue:
    → 双向链表 → head/tail → O(1) alloc/popleft!
    → O(1) free/append → 驱逐顺序=FIFO(LRU近似)!
    → num_free_blocks → 实时计数 → scheduler用!

  BlockHashToBlockMap: hash→block映射 → prefix caching核心!
    → 1:N mapping → 一个hash可对应多个block → 相同内容多副本!
    → dict内部 → {block_hash: KVCacheBlock | dict[int, KVCacheBlock]}
    → 单block → 直接KVCacheBlock → 避免inner dict → 减GC开销!
    → 多block → dict[block_id→KVCacheBlock] → 多请求同一prefix!

  Block分配流程:
    1. request→hash→查找cached_block_hash_to_block → 命中?
    2. 命中 → touch(block) → ref_cnt++ → 从free_queue移除 → 共享!
    3. 未命中 → get_new_blocks(n) → free_block_queue.popleft_n(n)!
    4. block满 → cache_full_blocks() → hash→insert→cached_map!

  驱逐流程:
    1. free不足 → _maybe_evict_cached_block → 从hash_map移除!
    2. reset_hash → block_hash=None → 不再可命中!
    3. 引用计数 → ref_cnt>0 → 不驱逐 → 安全!
    4. ref_cnt=0 → free_queue → 可分配 → FIFO顺序!

PagedAttention优势(验证):
  ✓ 快分配: pool.slice 3.9x faster (tensor view vs malloc!)
  ✓ 零碎片: block_size=16 → 按需分配 → 实测碎片率0-1%! → vs OS heap不同!
  ✓ Block管理: 双向链表+hash → O(1)操作 → 高效!
  ✓ Prefix caching: hash→hit→省prefill计算 → RAG场景84%KV省!
```

## 3. Memory Budgeting — RTX 4090内存预算

```
RTX 4090 24GB内存分配:

模型权重:
  7B BF16: 14GB → 剩余10GB → KV可用 → 有限!
  7B INT4 AWQ: 3.5GB → 剩余20.5GB → KV充裕 → 推荐!
  7B INT4+INT8KV: 3.5GB模型 + KV减半 → 最多并发!

KV Cache预算:
  可用=24-model-overhead → 24-3.5-2=18.5GB (INT4)!
  KV/tok(BF16): 81.92KB (7B GQA-5) → 18.5GB/81.92KB=225K tok!
  KV/tok(INT8): 40.96KB → 18.5GB/40.96KB=450K tok → 2x!

  并发=总KV/单请求KV → S=4096:
    BF16: 225K/4096×81.92KB×32=225K/2.62MB=86 req → B=86!
    INT8: 450K/4096×40.96KB×32=450K/1.31MB=172 req → B=172!

  实测: INT4+INT8KV → B=118 (FlashInfer GQA-8) → 4,791 tok/s!

Memory Budget决策树:
  BF16全精度 → model=14GB → KV=10GB → B≈22 → 低吞吐!
  INT8 KV → model=14GB → KV=10GB×2→等效20GB → B≈44 → 中吞吐!
  INT4+INT8KV → model=3.5GB → KV=18.5GB×2→等效37GB → B≈118 → 高吞吐!
  INT4+INT8KV+Eagle → model=3.5GB → KV+draft → B≈52 → 4.2x!

  → → → INT4+INT8KV=内存最优组合 → BF16→3x模型省+2x KV省=6x并发!
```

## 4. Eviction & Preemption — 内存不足时的策略

```
当free_block_queue耗尽 → 必须驱逐或抢占!

vLLM V1驱逐策略:
  1. 驱逐cached block → hash_map移除 → 不可命中 → 但block可重用!
  2. → → → FIFO顺序 → free_queue尾部 → 最老block先驱逐!
  3. → → → → → 驱逐=从prefix cache移除 → 新请求不能命中 → 精度↓!

  抢占(preemption):
  1. 整请求抢占 → 释放全部block → 其他请求用 → 粗粒度!
  2. → → → 抢占后 → request回waiting queue → 重新调度 → 重新prefill!
  3. → → → → → 短请求recompute快 → 长请求recompute慢!

  Recompute vs Swap:
    Recompute: 重新prefill → 计算成本∝权重(15.6GB) → S=16→17ms!
    Swap: KV→CPU pinned→PCIe→回GPU → 传输成本∝KV(1.25MB/block)!
    → → → swap快4-44x! → 但需要CPU pinned memory → 受限!
    → → → → → vLLM V1默认recompute → RTX 4090 swap可能更优!

  ITL影响:
    1 block swap → ITL +2.9% → 几乎无影响!
    8 blocks swap → ITL +23.4% → 可接受!
    → → → swap=recompute更好 → 但vLLM默认recompute → 可能需修正!

  StreamingLLM替代:
    → 不需要swap/recompute → sink+window → 固定KV → 无增长!
    → → → 4+4K=168MB → 固定 → 无驱逐 → 最简单!
    → → → → → RTX 4090最优=StreamingLLM+INT8 KV → 无限对话!
```

## 5. vLLM V1 KVCacheManager — 源码解析

```
KVCacheManager (vllm/v1/core/kv_cache_manager.py):

核心数据结构:
  - block_pool: BlockPool → alloc+free+cache+evict!
  - coordinator: KVCacheCoordinator → 多KV cache group协调!
  - empty_kv_cache_blocks: KVCacheBlocks(tuple()) → GC优化!

核心方法:
  1. get_computed_blocks(request):
     → prefix cache hit → find_longest_cache_hit!
     → → → hash查找 → 最长prefix → 省prefill计算!
     → → → → → max_cache_hit = num_tokens - 1 (最后token需重算!)
     → → → → → → → 返回: (cached_blocks, num_computed_tokens)

  2. allocate_slots(request, num_new_tokens, ...):
     → 为请求分配新KV slots → 关键方法!
     → → → 计算需要的block数 → num_new_tokens/block_size!
     → → → → → 查free_blocks → 够? → alloc → 不够? → 驱逐!
     → → → → → → → cache_full_blocks → hash→insert → prefix缓存!
     → → → → → → → → → delay_cache_blocks=True → PD分离时不立即缓存!

  3. free(request):
     → 请求完成 → 释放全部block → free_blocks!
     → → → ordered_blocks → eviction优先级排序!
     → → → → → prepend=False → append到free尾部 → FIFO!

BlockPool (vllm/v1/core/block_pool.py):

  核心方法:
    get_new_blocks(n): 从free_queue取 → popleft_n → O(1)!
    touch(blocks): ref_cnt++ → prefix共享 → 从free_queue移除!
    free_blocks(ordered_blocks): ref_cnt-- → ref_cnt=0 → append free_queue!
    cache_full_blocks: hash→insert → cached_block_hash_to_block!
    _maybe_evict_cached_block: 从hash_map移除 → 驱逐!

  引用计数(ref_cnt):
    ref_cnt=0 → 在free_queue → 可分配!
    ref_cnt=1 → 单请求用 → 可驱逐(ref_cnt降0)!
    ref_cnt>1 → 多请求共享(prefix) → 不能驱逐 → 安全!

BlockHashToBlockMap:
    1:N → 一个hash→多个block → 同prefix多副本!
    → → → insert: None→直接KVCacheBlock; 已有→合并dict!
    → → → → → pop: 从dict移除 → dict空→删除key!
    → → → → → → → 避免inner dict → 减GC → 单block直接KVCacheBlock!
```

## 6. Memory Pooling & Allocation Patterns

```
CUDA内存分配模式:

1. 预分配(Pre-allocation) — vLLM模式:
   → 模型加载时 → 计算num_gpu_blocks → 预分配全部KV cache!
   → → → 一次malloc → 之后只view(tensor slice) → 不再malloc!
   → → → → → 3.9x faster (slice vs dynamic malloc)!
   → → → → → → → 零碎片(0-1%) → alloc/free cycle不累积!

2. 动态分配(PyTorch模式):
   → 每次创建tensor → malloc → cache allocator → 非预分配!
   → → → caching_allocator → pool → 但碎片化风险!
   → → → → → vLLM不用动态 → 预分配+PagedAttention → 更高效!

3. Memory Pool(PyTorch caching allocator):
   → 预留大块 → 分slice → 用完→回pool → 不free→OS!
   → → → 冷启动231ms(10GB) → 之后近零!
   → → → → → 但碎片化存在(不是0-1% → OS heap不同!)!

4. CUDA Graph模式:
   → 固定input shape → 预捕获graph → 固定内存 → 无动态!
   → → → vLLM V1用CUDA Graph → 固定input_ids → 静态!

分配性能实测:
  pool slice: 3us → 3.9x faster than dynamic(12us)!
  pool cold start: 231ms(10GB) → 之后近零 → 启动时一次!
  7B BF16 13GB → malloc一次 → 之后全slice → 零碎片!
  20GB请求 → OOM → 24GB不够 → 量化是必需品!
```

## 7. Multi-Group KV Cache — MLA/MoE内存管理

```
vLLM V1 KV Cache Group架构:
  → kv_cache_config.kv_cache_groups → 不同block_size → 不同内存需求!
  → → → 例如: full attention vs sliding window → 不同block策略!

MLA内存管理:
  → DeepSeek-V3 MLA → W^UK absorption → KV压缩56.9x!
  → → → 但RTX 4090不支持FlashMLA(SM89→MLA慢2-8x) → 不推荐!
  → → → → → GQA-5最优(RTX 4090) → FlashInfer 15.72x加速!

MoE内存管理:
  → DeepSeek-V3 671B → shared+routered experts → 不同KV策略!
  → → → shared expert → 全KV → 不驱逐!
  → → → → → routered expert → 按需加载 → 可能需要offload!
  → → → → → → → EP All-to-All → expert→GPU → 内存管理复杂!

Sliding Window:
  → window_size=W → KV只保留最近W tokens → block管理:
  → → → 旧block → 驱逐或null → 新block → alloc!
  → → → → → block_mask → 哪些block可prefix cache → 哪些不可!
  → → → → → → → StreamingLLM=sink+window → 固定KV → 无增长!
```

## 8. Core Laws — GPU内存管理核心定律

1. **Memory-Bound Law**: decode memory-bound → 内存决定吞吐 → 量化是核心杠杆!
   → INT4→75%省 → B↑5.4x → throughput↑6.7x → 内存=瓶颈!

2. **PagedAttention Law**: block_size=16 → 按需分配 → 零碎片 → O(1)操作!
   → vs 连续分配 → 碎片化严重 → malloc/free慢 → PA远优!

3. **Prefix-Caching Law**: BlockHash→1:N→hash命中→省prefill → RAG场景84%KV省!
   → ref_cnt>1 → 共享 → 不能驱逐 → 安全 → prefix=巨大优化!

4. **Recompute-vs-Swap Law**: swap快4-44x → 但vLLM默认recompute → RTX 4090可能需修正!
   → swap成本∝KV(1.25MB) → recompute∝权重(15.6GB) → swap胜!

5. **Pre-allocation Law**: 预分配→pool slice→3.9x faster → 零碎片 → vs 动态malloc!
   → 启动一次malloc → 之后全slice → 生产最优!

6. **Budget-Law**: 可用=HBM-模型-overhead → INT4→剩余20.5GB → 更多并发!
   → 内存预算 → 模型+KV+overhead → 量化→省模型→省KV→双赢!

7. **StreamingLLM-Fixed-Law**: sink+window → 固定168MB → 无增长 → 无驱逐 → 无OOM!
   → → → 最简单内存策略 → 无限对话 → RTX 4090最优!

## 关键参考

- vLLM V1 BlockPool: vllm/v1/core/block_pool.py → alloc+free+cache+evict
- vLLM V1 KVCacheManager: vllm/v1/core/kv_cache_manager.py → allocate_slots+get_computed_blocks
- vLLM V1 FreeKVCacheBlockQueue: vllm/v1/core/kv_cache_utils.py → 双向链表+O(1)
- vLLM V1 BlockHashToBlockMap: vllm/v1/core/kv_cache_manager.py → 1:N hash→block
- CUDA Memory Allocator 实测: results/cuda_memory_allocator.json
- StreamingLLM: sink(4)+window(W) → 固定KV → 无限对话
- PagedAttention: block_size=16 → 零碎片 → 按需分配
- Swap vs Recompute: results/kv_cache_offloading_benchmark.json → swap快4-44x