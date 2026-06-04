# GPU 实验: Paged Attention — vLLM 核心机制

> 实验日期: 2026-06-04
> GPU: NVIDIA A16 (15GB), CUDA 11.7, PyTorch 2.5.1
> 脚本: `tools/gpu_paged_attention.py`

## 核心概念

vLLM 的 Paged Attention 借鉴了 OS 的虚拟内存分页机制:
- **Paged KV Cache**: KV cache 不需要连续物理内存，按固定大小 block 分配
- **Block Table**: 逻辑 block → 物理 block 的间接寻址表
- **Copy-on-Write**: 多个 sequence 可共享同一物理 block（prefix sharing）

## 实验 1: Paged vs Contiguous KV Cache

| SeqLen | Paged (ms) | Contiguous (ms) | 开销 |
|--------|-----------|----------------|------|
| 256    | 0.976     | 0.398          | 145% |
| 512    | 1.025     | 0.407          | 152% |
| 1024   | 1.137     | 0.408          | 178% |
| 2048   | 1.490     | 0.525          | 184% |
| 4096   | 2.523     | 0.584          | 332% |

**关键洞察**: Python 实现的间接寻址开销很大，但 vLLM 用 **自定义 CUDA kernel**（Triton/C++）将 paged 访问融合到 attention 计算中，实际开销接近 0。

## 实验 2: Prefix Sharing (Copy-on-Write)

| Prefix Len | Requests | Shared Blocks | Saved Blocks | Saved% |
|-----------|----------|---------------|-------------|--------|
| 256       | 8        | 16            | 112         | 29%    |
| 256       | 32       | 16            | 496         | 32%    |
| 512       | 16       | 32            | 480         | 47%    |
| 512       | 32       | 32            | 992         | 48%    |

**关键洞察**: 32 个请求共享 512 token 前缀 → 节省 48% KV 内存。这是 RLHF/Chat 场景的重要优化。

## 实验 3: Block Size 分析

| Block Size | Blocks/seq | Internal Frag | Waste% | Mem/Block |
|-----------|-----------|---------------|--------|-----------|
| 8         | 64        | 0             | 0.0%   | 16 KB     |
| 16        | 32        | 0             | 0.0%   | 32 KB     |
| 32        | 16        | 0             | 0.0%   | 64 KB     |
| 64        | 8         | 0             | 0.0%   | 128 KB    |
| 128       | 4         | 0             | 0.0%   | 256 KB    |
| 256       | 2         | 0             | 0.0%   | 512 KB    |

vLLM 默认 block_size=16，是碎片和管理开销的平衡点。

## 实验 4: Preemption (Swap vs Recompute)

| Tokens Evicted | Swap (ms) | Recompute (ms) | Winner |
|---------------|-----------|----------------|--------|
| 64            | 0.022     | 9.130          | Swap   |
| 128           | 0.044     | 18.261         | Swap   |
| 256           | 0.087     | 36.522         | Swap   |
| 512           | 0.175     | 73.044         | Swap   |
| 1024          | 0.350     | 146.087        | Swap   |

**关键洞察**: A16 算力有限 (14.7 TFLOPS)，Swap 始终更快。但在高端 GPU (A100/H100) 上，Recompute 可能更优（因为 GPU 算力充裕，PCIe 带宽不变）。

## 实验 5: Continuous Batching + Paged KV

| Avg SeqLen | Max Concurrent | Step Time (ms) | Throughput (tok/s) |
|-----------|---------------|---------------|-------------------|
| 512       | 14171         | 87.4          | 162K              |
| 1024      | 7085          | 87.4          | 81K               |
| 2048      | 3542          | 87.4          | 40K               |
| 4096      | 1771          | 87.4          | 20K               |

## vLLM V1 BlockPool 关键数据结构

```python
# 简化的 vLLM V1 Paged KV Cache 管理
class BlockPool:
    block_size: int = 16  # tokens per block
    num_blocks: int       # total physical blocks

    # Free list: O(1) allocate/free
    free_block_queue: FreeKVCacheBlockQueue  # doubly-linked list

    # Hash-based prefix caching
    block_hash_to_block: BlockHashToBlockMap  # hash → [block] (1:N)

    # Reference counting for sharing
    ref_cnt: List[int]     # per-block ref count

    # Eviction: LRU
    policy: str = "lru"
```

## 总结

1. Paged KV Cache 是 vLLM 的核心创新，消除了 KV cache 碎片化
2. 实际性能依赖自定义 CUDA kernel（非 Python 间接寻址）
3. Prefix Sharing 对 Chat/RL 场景意义重大（40-50% 节省）
4. Block Size 16 是 vLLM 的最优平衡点
5. Preemption 策略取决于 GPU 算力与带宽的比值
