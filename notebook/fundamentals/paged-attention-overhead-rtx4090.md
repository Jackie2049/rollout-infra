# Paged Attention Overhead: RTX 4090 实测分析

> 2026-06-07 | RTX 4090 (SM 8.9, 128 MPs, 24GB HBM)

## 核心问题

vLLM 使用 block_table 将虚拟 KV 位置映射到物理 block, 这增加了一个间接层:
```
virtual_position → block_table[virtual_block_idx] → physical_block → KV data
```

**为什么 vLLM 不直接用 torch.sdpa 而需要自定义 Triton kernel?**

## 实测结果

### 1. Contiguous Attention (PyTorch SDPA baseline)

| Config | Type | Time (ms) |
|--------|------|-----------|
| B=1 H=32 S=512 | MHA | 0.095 |
| B=8 H=32 S=512 | MHA | 0.199 |
| B=32 H=32 S=512 | MHA | 0.761 |
| B=128 H=32 S=512 | MHA | 2.855 |
| B=1 Hq=32 Hkv=8 S=2048 | GQA | 0.335 |
| B=8 GQA S=2048 | GQA | 0.692 |
| B=32 GQA S=2048 | GQA | 2.730 |
| B=128 GQA S=2048 | GQA | 10.896 |

### 2. Block_table Gather Overhead (只看内存访问)

| Config | Gather (ms) | Contiguous Read (ms) | Overhead |
|--------|------------|---------------------|----------|
| B=1 S=512 | 0.016 | 0.010 | **57.7%** |
| B=8 S=512 | 0.016 | 0.009 | **70.0%** |
| B=32 S=512 | 0.148 | 0.147 | **1.3%** |
| B=128 S=512 | 0.593 | 0.583 | **1.8%** |
| B=1 S=2048 | 0.014 | 0.009 | **52.4%** |
| B=32 S=2048 | 0.591 | 0.583 | **1.4%** |

**关键发现**: Gather 开销随 batch 增大急剧下降 — 小 batch 时 50-70% overhead, 大 batch 仅 1-2%.

原因: 小 batch 时操作本身很快 (launch overhead 主导), gather 的额外内存访问相对占大头. 大 batch 时操作本身慢 (HBM-bound), gather 只是多读一次 block_table (很小), 相比 KV 数据本身占比极低.

### 3. Full Paged vs Contiguous Attention (gather + SDPA)

| Config | Contiguous (ms) | Paged (ms) | Overhead |
|--------|----------------|------------|----------|
| B=4 GQA S=512 | 0.088 | 0.181 | **106.1%** |
| B=16 GQA S=512 | 0.354 | 0.849 | **139.6%** |
| B=32 GQA S=512 | 0.697 | 1.711 | **145.5%** |
| B=128 GQA S=512 | 2.765 | 6.883 | **149.0%** |
| B=4 GQA S=2048 | 0.353 | 0.836 | **137.0%** |
| B=32 GQA S=2048 | 2.737 | 6.863 | **150.7%** |

**平均 overhead: 138%** — Paged attention 几乎是 contiguous 的 2.4 倍慢!

## 为什么 Python-level Gather 开销这么大?

对比 Section 2 (仅gather) 和 Section 3 (gather+compute):

| | B=32 S=512 | B=128 S=512 |
|---|-----------|------------|
| Gather-only overhead | 1.3% | 1.8% |
| Full paged overhead | 145.5% | 149.0% |

Gather 本身仅增加 1-2% 内存开销, 但 **full attention overhead 是 145-149%!** 原因:

1. **Python-level gather 创建临时 tensor**: `physical_kv[block_table]` 产生新的 `[B, H_kv, n_blocks, BLOCK_SIZE, D]` tensor → 需要 GPU 内存分配+数据拷贝
2. **reshape 不是零开销**: `reshape(B, H_kv, seq_len, D)` 触发 view 操作, 但底层数据仍然是非连续的
3. **KV head expand (GQA)**: `repeat_interleave(n_groups)` 再产生一次内存拷贝
4. **SDPA 无法利用原地 paged KV**: 需要 contiguous K/V 作为输入 → gather 是必须的中间步骤

总开销 = gather(内存分配+拷贝) + reshape(view但可能不连续) + expand(GQA额外拷贝) ≈ **数据被拷贝 3 次**, 而 contiguous 只 1 次

## 为什么 vLLM 的 Triton kernel 可以消除这个开销?

vLLM 的 Paged Attention Triton kernel (vLLM v1 用 FlashInfer) 直接在 CUDA level 处理 block_table:

```
# Python approach (2.4x慢):
gathered_kv = physical_kv[block_table]      # Step 1: 拷贝到临时buffer
gathered_kv = gathered_kv.reshape(...)      # Step 2: view操作
gathered_kv = gathered_kv.repeat_interleave(...) # Step 3: GQA expand
output = sdpa(q, gathered_kv, ...)          # Step 4: 计算

# Triton kernel approach (native):
for each query_head:
    for each virtual_position:
        physical_block = block_table[virtual_block]  # CUDA-level indirection
        kv_data = physical_kv[physical_block, pos_in_block]  # 直接读
        # 累加到 attention score — 无中间拷贝!
```

Triton kernel 的优势:
1. **零额外内存分配**: 直接读 physical KV, 不创建 gathered tensor
2. **CUDA-level indirection**: block_table lookup 在 kernel 内完成, 无 Python 中间步骤
3. **GQA handled inline**: 4个 Q head 共享同一个 KV head, 只读一次 KV
4. **FlashAttention tiling**: 与 block_table 天然对齐 (block_size = tile_size)

## 对比: A16 上的数据

之前在 A16 (10 SM, 15GB) 上测试 Paged Attention:
- **A16 overhead: 145-332%** (更大, 因为 A16 SM少、BW低)
- **RTX 4090 overhead: 106-151%** (更一致, 大GPU开销稳定)

两张卡 overhead 都很大 → 证明问题本质是 Python-level indirection, 不是硬件特定

## 实用结论

1. **Paged Attention 必须用自定义 kernel**: Python-level gather overhead 138% → 不可接受
2. **vLLM 的 Triton/FlashInfer kernel 是必要的**: 不是"过度工程", 而是性能必需
3. **为什么不用 torch.sdpa + gather**: 因为 overhead 是 2.4x → 比自定义 kernel 慢太多
4. **GQA 使 overhead 更大**: repeat_interleave 额外拷贝 → 生产kernel直接处理
5. **block_size 的影响**: 小 block (16) → 更多 block_table 查找 → kernel 需要高效间接寻址

## vLLM V1 Attention Backend

vLLM V1 支持多种 attention backend, 都实现 paged KV indirection:

| Backend | 实现 | 适用GPU |
|---------|------|---------|
| FlashInfer | CUDA-level paged KV | SM 80+ (A100/H100/RTX 4090) |
| Triton (fallback) | Triton-level paged KV | 所有 GPU |
| FlashMLA | 专为 MLA 设计 | SM 90 (H100) |
| xFormers | 旧版 paged attention | 已弃用 |

FlashInfer 是 V1 默认 backend — 专门为 decode + paged KV 优化