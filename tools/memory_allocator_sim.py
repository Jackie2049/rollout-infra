"""GPU 内存分配器模拟器 — PyTorch Caching Allocator + vLLM Memory Pool

模拟 GPU 内存管理策略:
1. Naive 分配器: 直接 malloc/free，展示碎片化问题
2. PyTorch Caching Allocator: Block pool + split/merge 策略
3. vLLM Memory Pool: 预分配 + PagedAttention block 管理
4. 碎片化分析: 不同分配模式下的碎片化程度
5. KV Cache 内存管理: block 分配与回收模拟

使用方法:
    python memory_allocator_sim.py   # CPU 可运行
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from collections import defaultdict


# ============================================================
# Block 数据结构
# ============================================================

@dataclass
class Block:
    """内存块。"""
    addr: int          # 起始地址
    size: int          # 大小 (bytes)
    in_use: bool = False
    req_id: str = ""   # 分配请求 ID

    def __repr__(self):
        status = "USED" if self.in_use else "FREE"
        return f"Block({status} {self.size/1e6:.1f}MB @{self.addr})"


# ============================================================
# Naive 分配器 (直接 malloc/free)
# ============================================================

class NaiveAllocator:
    """朴素分配器: 直接分配和释放，不做缓存。"""

    def __init__(self, total_memory: int):
        self.total_memory = total_memory
        self.used = 0
        self.peak = 0
        self.alloc_count = 0
        self.free_count = 0
        self.fragments = 0  # 碎片化事件

        # 空闲列表: 所有空闲块（可能有碎片）
        self.free_blocks: List[Block] = [Block(0, total_memory)]
        # 使用中块
        self.used_blocks: Dict[str, Block] = {}

    def allocate(self, size: int, req_id: str) -> Optional[Block]:
        """分配一块内存 (First Fit)。"""
        # 找到第一个够大的空闲块
        for i, block in enumerate(self.free_blocks):
            if block.size >= size:
                # 从空闲块中切出所需大小
                allocated = Block(block.addr, size, True, req_id)

                # 剩余部分
                remaining = block.size - size
                if remaining > 0:
                    new_free = Block(block.addr + size, remaining)
                    self.free_blocks[i] = new_free
                else:
                    self.free_blocks.pop(i)

                self.used_blocks[req_id] = allocated
                self.used += size
                self.peak = max(self.peak, self.used)
                self.alloc_count += 1
                return allocated

        # 无法分配 — 检查是否是碎片化问题
        total_free = sum(b.size for b in self.free_blocks)
        if total_free >= size:
            self.fragments += 1
        return None  # OOM

    def free(self, req_id: str):
        """释放一块内存。"""
        if req_id not in self.used_blocks:
            return
        block = self.used_blocks.pop(req_id)
        block.in_use = False
        block.req_id = ""
        self.used -= block.size
        self.free_blocks.append(block)
        self.free_count += 1

    def fragmentation_ratio(self) -> float:
        """计算碎片化比率 (空闲块数 / 理论最少块数)。"""
        if not self.free_blocks:
            return 0.0
        total_free = sum(b.size for b in self.free_blocks)
        if total_free == 0:
            return 0.0
        # 理想情况: 所有空闲内存是 1 个连续块
        return len(self.free_blocks)  # 空闲块数越多，碎片化越严重

    def largest_free_block(self) -> int:
        """最大连续空闲块大小。"""
        if not self.free_blocks:
            return 0
        return max(b.size for b in self.free_blocks)


# ============================================================
# PyTorch Caching Allocator
# ============================================================

class CachingAllocator:
    """PyTorch 风格的 Caching Allocator:

    核心策略:
    1. 分配时先从 pool 找合适大小的块
    2. 释放时不真正 free，而是放回 pool
    3. 大块可以 split 成小块
    4. 相邻空闲块可以 merge
    5. 只在 pool 不够时才真正从 GPU 申请
    """

    def __init__(self, total_memory: int, max_split_size: int = 10*1024*1024):
        self.total_memory = total_memory
        self.max_split_size = max_split_size  # 超过此大小不 split
        self.pool_used = 0      # 实际使用
        self.pool_reserved = 0  # 从系统预留
        self.peak_used = 0
        self.alloc_count = 0
        self.free_count = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.splits = 0
        self.merges = 0

        # Pool: 按 size 分桶的空闲块
        self.free_pool: Dict[int, List[Block]] = defaultdict(list)
        # 按地址索引的空闲块 (用于 merge)
        self.addr_map: Dict[int, Block] = {}
        # 使用中块
        self.used_blocks: Dict[str, Block] = {}

        # 初始分配: 从系统获取全部内存
        initial = Block(0, total_memory)
        self.free_pool[total_memory].append(initial)
        self.addr_map[0] = initial
        self.pool_reserved = total_memory

    def allocate(self, size: int, req_id: str) -> Optional[Block]:
        """分配 (优先从 pool 找)。"""
        self.alloc_count += 1

        # 1. 尝试从 pool 找 exact match
        if size in self.free_pool and self.free_pool[size]:
            block = self.free_pool[size].pop(0)
            block.in_use = True
            block.req_id = req_id
            self.used_blocks[req_id] = block
            self.pool_used += size
            self.peak_used = max(self.peak_used, self.pool_used)
            self.cache_hits += 1
            return block

        # 2. 找到比 size 大的块，split
        best_size = None
        for pool_size in sorted(self.free_pool.keys()):
            if pool_size >= size and self.free_pool[pool_size]:
                best_size = pool_size
                break

        if best_size is not None:
            block = self.free_pool[best_size].pop(0)
            del self.addr_map[block.addr]

            # Split 如果差距足够大
            if best_size - size > 0 and best_size <= self.max_split_size:
                allocated = Block(block.addr, size, True, req_id)
                remaining = Block(block.addr + size, best_size - size)
                self.free_pool[best_size - size].append(remaining)
                self.addr_map[remaining.addr] = remaining
                self.splits += 1
                self.used_blocks[req_id] = allocated
                self.pool_used += size
                self.peak_used = max(self.peak_used, self.pool_used)
                self.cache_hits += 1
                return allocated
            else:
                block.in_use = True
                block.req_id = req_id
                self.used_blocks[req_id] = block
                self.pool_used += best_size
                self.peak_used = max(self.peak_used, self.pool_used)
                self.cache_hits += 1
                return block

        # 3. Pool 不够 — cache miss
        self.cache_misses += 1
        return None  # OOM

    def free(self, req_id: str):
        """释放 (放回 pool，不真正 free)。"""
        if req_id not in self.used_blocks:
            return
        block = self.used_blocks.pop(req_id)
        block.in_use = False
        block.req_id = ""
        self.pool_used -= block.size
        self.free_pool[block.size].append(block)
        self.addr_map[block.addr] = block
        self.free_count += 1

        # 尝试 merge 相邻空闲块
        self._try_merge(block)

    def _try_merge(self, block: Block):
        """尝试合并相邻空闲块。"""
        # 检查后面是否空闲
        next_addr = block.addr + block.size
        if next_addr in self.addr_map:
            neighbor = self.addr_map[next_addr]
            if not neighbor.in_use:
                # Merge
                self.free_pool[neighbor.size].remove(neighbor)
                self.free_pool[block.size].remove(block)
                del self.addr_map[block.addr]
                del self.addr_map[neighbor.addr]

                merged = Block(block.addr, block.size + neighbor.size)
                self.free_pool[merged.size].append(merged)
                self.addr_map[merged.addr] = merged
                self.merges += 1
                # 递归尝试继续 merge
                self._try_merge(merged)

    def cache_hit_rate(self) -> float:
        if self.alloc_count == 0:
            return 0.0
        return self.cache_hits / self.alloc_count

    def fragmentation_ratio(self) -> float:
        """碎片化比率 = pool 中的空闲块数。"""
        total_free_blocks = sum(len(blocks) for blocks in self.free_pool.values())
        return total_free_blocks


# ============================================================
# vLLM Paged Memory Pool
# ============================================================

class PagedMemoryPool:
    """vLLM 风格的分页内存池:

    核心策略:
    1. 预分配固定大小的 block (如 16 tokens 的 KV Cache)
    2. block 按 reference count 管理
    3. prefix caching: 相同前缀共享 block
    4. swap: 当 GPU 内存不够时 swap 到 CPU
    """

    def __init__(self, total_blocks: int, block_size_bytes: int,
                 swap_space_blocks: int = 0):
        self.total_blocks = total_blocks
        self.block_size = block_size_bytes
        self.swap_space = swap_space_blocks

        # Block 状态
        self.free_blocks = list(range(total_blocks))  # 空闲 block ID 列表
        self.used_blocks: Dict[int, str] = {}          # block_id → req_id
        self.ref_counts: Dict[int, int] = defaultdict(int)  # block_id → ref count
        self.block_tables: Dict[str, List[int]] = {}   # req_id → block list

        # Swap space
        self.swapped_blocks: Dict[str, List[int]] = {}  # req_id → swapped blocks
        self.swap_free = list(range(swap_space_blocks))

        # Stats
        self.alloc_count = 0
        self.free_count = 0
        self.swap_out_count = 0
        self.swap_in_count = 0
        self.prefix_hits = 0

        # Prefix cache: hash → block_id
        self.prefix_cache: Dict[int, int] = {}

    def allocate_block(self, req_id: str, prefix_hash: Optional[int] = None) -> Optional[int]:
        """为一个请求分配一个 block。"""
        # 检查 prefix cache
        if prefix_hash is not None and prefix_hash in self.prefix_cache:
            cached_block = self.prefix_cache[prefix_hash]
            self.ref_counts[cached_block] += 1
            if req_id not in self.block_tables:
                self.block_tables[req_id] = []
            self.block_tables[req_id].append(cached_block)
            self.prefix_hits += 1
            return cached_block

        # 从 free pool 分配
        if not self.free_blocks:
            # 尝试 swap out
            if self.swap_free:
                self._swap_out_oldest()
            else:
                return None  # OOM

        block_id = self.free_blocks.pop(0)
        self.used_blocks[block_id] = req_id
        self.ref_counts[block_id] = 1

        if req_id not in self.block_tables:
            self.block_tables[req_id] = []
        self.block_tables[req_id].append(block_id)

        self.alloc_count += 1
        return block_id

    def free_request(self, req_id: str):
        """释放一个请求的所有 block。"""
        if req_id not in self.block_tables:
            return

        for block_id in self.block_tables[req_id]:
            self.ref_counts[block_id] -= 1
            if self.ref_counts[block_id] == 0:
                # 真正释放
                if block_id in self.used_blocks:
                    del self.used_blocks[block_id]
                self.free_blocks.append(block_id)
                self.free_count += 1

        del self.block_tables[req_id]

    def _swap_out_oldest(self):
        """Swap out 最老的请求。"""
        if not self.block_tables:
            return
        oldest_req = next(iter(self.block_tables))
        blocks = self.block_tables.pop(oldest_req)
        self.swapped_blocks[oldest_req] = blocks
        self.swap_out_count += 1

    def cache_prefix(self, prefix_hash: int, block_id: int):
        """缓存一个 prefix block。"""
        self.prefix_cache[prefix_hash] = block_id
        self.ref_counts[block_id] += 1

    def utilization(self) -> float:
        return (self.total_blocks - len(self.free_blocks)) / self.total_blocks


# ============================================================
# 实验函数
# ============================================================

def experiment_fragmentation():
    """实验 1: Naive vs Caching 碎片化对比。"""
    print("=" * 60)
    print("实验 1: Naive vs Caching Allocator 碎片化对比")
    print("=" * 60)

    total_mem = 1000 * 1024 * 1024  # 1000 MB
    np.random.seed(42)

    # 模拟 100 次分配/释放
    sizes = np.random.choice([10, 20, 50, 100, 200], size=100) * 1024 * 1024

    naive = NaiveAllocator(total_mem)
    caching = CachingAllocator(total_mem)

    # 运行分配/释放序列
    active_ids = []
    oom_naive = 0
    oom_caching = 0

    for i, size in enumerate(sizes):
        req_id = f"req_{i}"

        # Naive
        result = naive.allocate(int(size), req_id)
        if result is None:
            oom_naive += 1
        else:
            active_ids.append(req_id)

        # Caching
        result = caching.allocate(int(size), req_id)
        if result is None:
            oom_caching += 1

        # 随机释放一些旧的
        if len(active_ids) > 5 and np.random.random() < 0.3:
            free_id = active_ids.pop(np.random.randint(len(active_ids)))
            naive.free(free_id)
            caching.free(free_id)

    print(f"\n  100 次分配请求:")
    print(f"    Naive Allocator:")
    print(f"      OOM 次数:       {oom_naive}")
    print(f"      碎片化事件:     {naive.fragments}")
    print(f"      空闲块数:       {naive.fragmentation_ratio()}")
    print(f"      最大连续空闲:   {naive.largest_free_block()/1e6:.1f} MB")
    print(f"      已用/总量:      {naive.used/1e6:.1f}/{total_mem/1e6:.0f} MB")

    print(f"\n    Caching Allocator:")
    print(f"      OOM 次数:       {oom_caching}")
    print(f"      Cache 命中率:   {caching.cache_hit_rate()*100:.1f}%")
    print(f"      Split 次数:     {caching.splits}")
    print(f"      Merge 次数:     {caching.merges}")
    print(f"      空闲块数:       {caching.fragmentation_ratio()}")
    print(f"      Pool used:      {caching.pool_used/1e6:.1f} MB")
    print(f"      Pool reserved:  {caching.pool_reserved/1e6:.0f} MB")

    print(f"""
关键洞察:
  - Naive: 碎片化导致即使有足够总空闲内存也无法分配 (OOM)
  - Caching: 通过 split/merge 维持较大的连续空闲块
  - Caching 命中率高 = 少向系统申请新内存 = 更快
  - PyTorch: pool_reserved >= pool_used (多出来的就是 cached blocks)
    """)


def experiment_training_memory():
    """实验 2: 训练场景的内存分配模式。"""
    print("=" * 60)
    print("实验 2: 训练场景内存分配模拟")
    print("=" * 60)

    total_mem = 80 * 1024 * 1024 * 1024  # 80 GB (A100)
    alloc = CachingAllocator(total_mem)

    # 训练步骤的内存分配序列
    def model_params_mb(size_b):
        return size_b * 2 / 1e6  # FP16

    # LLaMA-7B 训练步骤
    model_size = 13_482_752_000 * 2  # FP16 params bytes
    grad_size = model_size
    optimizer_size = model_size * 2 * 2  # AdamW m+v, FP32
    activation_size = 6 * 1024 * 1024 * 1024  # ~6 GB
    misc = 500 * 1024 * 1024  # 500 MB misc

    print(f"\n  模型: LLaMA-7B, A100 80GB")
    print(f"  分配序列:")

    steps = [
        ("model_params", model_size),
        ("gradients", grad_size),
        ("optimizer_m", optimizer_size // 2),
        ("optimizer_v", optimizer_size // 2),
        ("activations_L1", activation_size // 32),
        ("activations_L2", activation_size // 32),
        ("activations_L3", activation_size // 32),
        ("misc", misc),
    ]

    for name, size in steps:
        result = alloc.allocate(size, name)
        status = "OK" if result else "OOM"
        print(f"    {name:<20} {size/1e9:>6.2f} GB — {status}")

    print(f"\n  Pool 状态:")
    print(f"    Used:     {alloc.pool_used/1e9:>6.2f} GB")
    print(f"    Reserved: {alloc.pool_reserved/1e9:>6.2f} GB")
    print(f"    Free:     {(alloc.pool_reserved - alloc.pool_used)/1e9:>6.2f} GB")
    print(f"    Peak:     {alloc.peak_used/1e9:>6.2f} GB")

    # 模拟 backward: 释放 activation，分配 gradient
    print(f"\n  Backward (释放 activation, 分配梯度):")
    for i in range(1, 4):
        alloc.free(f"activations_L{i}")
        print(f"    free activations_L{i} → used: {alloc.pool_used/1e9:.2f} GB")

    print(f"\n  最终:")
    print(f"    Used:     {alloc.pool_used/1e9:>6.2f} GB")
    print(f"    Reserved: {alloc.pool_reserved/1e9:>6.2f} GB")
    print(f"    Cache命中率: {alloc.cache_hit_rate()*100:.1f}%")


def experiment_paged_memory():
    """实验 3: vLLM 分页内存池模拟。"""
    print("\n" + "=" * 60)
    print("实验 3: vLLM 分页内存池模拟")
    print("=" * 60)

    # 模拟参数
    total_blocks = 1000
    block_size = 16 * 4096 * 2 * 32 * 2  # 16 tokens * hidden * n_kv_heads * fp16 (approx)
    pool = PagedMemoryPool(total_blocks, block_size, swap_space_blocks=200)

    print(f"\n  配置: {total_blocks} blocks, block_size={block_size/1e6:.1f}MB")
    print(f"  总 KV Cache: {total_blocks * block_size / 1e9:.2f} GB")

    # 模拟请求序列
    np.random.seed(42)
    requests = []
    req_id_counter = 0

    # 模拟 100 个请求
    for step in range(200):
        # 新请求到达 (poisson, 平均每步 0.5 个)
        if np.random.random() < 0.5:
            seq_len = np.random.choice([128, 256, 512, 1024, 2048])
            n_blocks_needed = seq_len // 16
            req_id = f"req_{req_id_counter}"
            req_id_counter += 1

            # 分配 blocks
            allocated = 0
            for b in range(n_blocks_needed):
                # Prefix cache: 前 4 个 block 可能命中
                prefix_hash = hash(f"system_prompt_{b}") if b < 4 else None
                block = pool.allocate_block(req_id, prefix_hash)
                if block is None:
                    break
                allocated += 1

            if allocated > 0:
                requests.append(req_id)
            if step % 50 == 0:
                print(f"\n  Step {step}: {len(requests)} active requests, "
                      f"util={pool.utilization()*100:.1f}%, "
                      f"free={len(pool.free_blocks)}, "
                      f"prefix_hits={pool.prefix_hits}")

        # 完成旧请求
        completed = [r for r in requests if np.random.random() < 0.1]
        for req_id in completed:
            pool.free_request(req_id)
            requests.remove(req_id)

    print(f"\n  最终统计:")
    print(f"    总请求数:       {req_id_counter}")
    print(f"    活跃请求:       {len(requests)}")
    print(f"    利用率:         {pool.utilization()*100:.1f}%")
    print(f"    Prefix 命中:    {pool.prefix_hits}")
    print(f"    Swap out:       {pool.swap_out_count}")
    print(f"    Block 分配:     {pool.alloc_count}")
    print(f"    Block 释放:     {pool.free_count}")

    print(f"""
关键洞察:
  - 分页管理: 每个 block 固定大小，消除碎片化
  - Prefix Caching: 共享公共前缀的 block，节省内存
  - Swap: 当 GPU 满时 swap 到 CPU，避免 OOM
  - Reference Counting: 多个请求共享同一 block 时，引用计数管理生命周期
    """)


def experiment_memory_comparison():
    """实验 4: 不同分配器的 OOM 对比。"""
    print("=" * 60)
    print("实验 4: OOM 抵抗力对比 (锯齿形分配模式)")
    print("=" * 60)

    total_mem = 1000 * 1024 * 1024  # 1000 MB

    # 锯齿形分配: 大→小→大→小，最容易碎片化的模式
    pattern = []
    for _ in range(10):
        pattern.extend([200, 100, 50, 20, 200, 100, 50, 20])
    sizes = [s * 1024 * 1024 for s in pattern]

    naive = NaiveAllocator(total_mem)
    caching = CachingAllocator(total_mem)

    active = []
    oom_naive = 0
    oom_caching = 0

    for i, size in enumerate(sizes):
        req_id = f"r{i}"

        r1 = naive.allocate(size, req_id)
        if r1 is None:
            oom_naive += 1
        else:
            active.append(req_id)

        r2 = caching.allocate(size, req_id)
        if r2 is None:
            oom_caching += 1

        # 每隔几个释放一个
        if len(active) > 3 and i % 3 == 0:
            fid = active.pop(0)
            naive.free(fid)
            caching.free(fid)

    print(f"\n  锯齿形分配模式 (80 次分配, 交替大小: 200/100/50/20 MB)")
    print(f"    总内存: 1000 MB")
    print(f"")
    print(f"    {'分配器':<20} {'OOM':>6} {'碎片事件':>10} {'最大连续空闲':>14}")
    print(f"    {'-'*20} {'-'*6} {'-'*10} {'-'*14}")
    print(f"    {'Naive':<20} {oom_naive:>6} {naive.fragments:>10} "
          f"{naive.largest_free_block()/1e6:>12.1f} MB")
    print(f"    {'Caching':<20} {oom_caching:>6} {0:>10} "
          f"{'N/A':>14}")

    print(f"\n  Caching Allocator 详情:")
    print(f"    Cache 命中率: {caching.cache_hit_rate()*100:.1f}%")
    print(f"    Split 次数:   {caching.splits}")
    print(f"    Merge 次数:   {caching.merges}")

    print("""
关键洞察:
  - 锯齿形模式下 Naive 分配器容易碎片化 OOM
  - Caching Allocator 通过 split/merge 维持内存连续性
  - Cache 命中率高 = 减少实际 GPU 内存分配/释放次数
  - 实际训练中: 每步的 activation 大小相似，cache 命中率很高
    """)


def main():
    print("=" * 60)
    print("GPU 内存分配器模拟器")
    print("=" * 60)

    experiment_fragmentation()
    experiment_training_memory()
    experiment_paged_memory()
    experiment_memory_comparison()

    # 总结
    print("=" * 60)
    print("总结")
    print("=" * 60)
    print("""
GPU 内存管理核心知识:

1. Naive 分配器:
   - 直接 malloc/free
   - 碎片化严重，容易 OOM
   - 实际不用于深度学习

2. PyTorch Caching Allocator:
   - 释放 → 放回 pool (不真正 free)
   - 分配 → 优先从 pool 取
   - Split: 大块切成小块
   - Merge: 相邻空闲块合并
   - pool_reserved >= pool_used (差值 = cached)
   - 配置: PYTORCH_CUDA_ALLOC_CONF

3. vLLM Paged Memory:
   - 预分配固定大小 block
   - Block = KV Cache for N tokens (通常 16)
   - Reference counting 管理共享
   - Prefix caching: 相同前缀共享 block
   - Swap to CPU: GPU 满时的 fallback
   - 消除碎片化 (固定大小 block)

4. 关键指标:
   - torch.cuda.memory_allocated() = 实际使用
   - torch.cuda.memory_reserved() = 从系统预留
   - torch.cuda.max_memory_allocated() = 峰值
   - 碎片化 = reserved - allocated

5. 实际应用:
   - 训练: Caching Allocator 自动管理
   - 推理: vLLM Paged Memory 手动管理
   - OOM 排查: 先看 reserved vs allocated
    """)


if __name__ == "__main__":
    main()
