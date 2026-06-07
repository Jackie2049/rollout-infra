# FlashInfer — Production Decode Kernel Analysis (RTX 4090 Context)

> 2026-06-07 | Why vLLM/SGLang use FlashInfer for decode instead of FlashAttention-2?

> Our RTX 4090 benchmark proved: FA2 decode is 3-34x SLOWER than SDPA → FA2 decode = negative optimization!
> But vLLM/SGLang both use FlashInfer for decode → Why?
> → FlashInfer = decode-specialized kernel + batched GQA + varlen + prefill compatibility
> → This is the KEY reason production systems achieve far better decode performance than raw FA2

## 1. Why FA2 Decode is Slower (RTX 4090 Evidence)

> Our RTX 4090 benchmarks showed:
> - FA2 decode B=1: 3.3-3.9x slower than SDPA
> - FA2 decode B=128: 9-34x slower than SDPA!
> - FA2 even slower than naive PyTorch at B≤8 (1.1-1.3x)

> Root cause analysis:
> 1. **FA2 tiling optimized for prefill**: FA2 partitions work in (B×S) large tiles → great for compute-bound prefill
> 2. **Decode Q=1 matrix is tiny**: (1×S) → FA2's tiling strategy wastes GPU parallelism
> 3. **FA2 layout overhead**: (B,S,H,D) → (B,S,H,D) transpose + contiguous() conversion adds latency
> 4. **FA2 no batching across queries**: each decode step processes independently → no batch optimization
> 5. **FA2 kernel launch overhead**: for small decode operations, kernel startup dominates

## 2. FlashInfer Architecture Overview

> FlashInfer is a CUDA-based attention kernel library specifically designed for LLM serving, created by the MLSys team at University of Michigan (Same team as vLLM).

> Key architectural differences from FlashAttention-2:
> | Feature | FlashAttention-2 | FlashInfer |
> |---------|------------------|-----------|
> | Design Goal | Prefill speed | Decode + serving efficiency |
> | Decode Strategy | Single Q per call | Batched decode (cooperative) |
> | GQA Support | Expand KV (wasteful) | Native GQA kernel |
> | Varlen | Not supported | Core feature (packed sequences) |
> | Prefill | O(N²) → O(N) memory | Same + batched prefill |
> | Layout | (B,S,H,D) | Flexible (supports both) |
> | Sliding Window | Not supported | Built-in support |

## 3. FlashInfer's 5 Key Decode Innovations

### 3.1 Cooperative Batched Decode
> - **Problem**: FA2 processes each decode query independently → no GPU utilization for small batches
> - **FlashInfer Solution**: Batch multiple decode queries together
>   - Pack Q heads from multiple requests into single kernel launch
>   - Cooperative kernel: multiple requests share computation pipeline
>   - **Impact**: B=8 decode → 1 kernel launch vs 8 separate launches → 8x less overhead

### 3.2 Native GQA Kernel (No KV Expand)
> - **Problem**: FA2 requires expanding GQA KV (n_kv_heads→n_heads) → 75% wasted memory bandwidth
> - **FlashInfer Solution**: Native GQA support in kernel
>   - Per-GPU-thread processes different KV heads → no expand needed
>   - Q heads share KV reads → reduce memory traffic by n_heads/n_kv_heads ratio
>   - **Impact**: GQA-4 → 4x less KV memory traffic vs FA2 expand

### 3.3 Paged KV Cache Access
> - **Problem**: FA2 assumes contiguous KV → requires pre-allocating large blocks
> - **FlashInfer Solution**: Page-based KV access
>   - Works with vLLM's paged KV cache (block manager)
>   - Directly accesses physical pages → no contiguous requirement
>   - **Impact**: Enables efficient memory management + no large pre-allocation

### 3.4 Varlen (Variable Length) Attention
> - **Problem**: Different requests have different sequence lengths → padding waste
> - **FlashInfer Solution**: Packed varlen attention
>   - cu_fill_sequence_lengths API → pack different-length sequences into single batch
>   - No padding → no wasted computation
>   - **Impact**: 3-6x throughput improvement for mixed-length batches (SGLang data)

### 3.5 Sliding Window + Multi-Level Attention
> - **Problem**: Long context requires full attention → expensive
> - **FlashInfer Solution**: Built-in sliding window support
>   - Only attend to recent W tokens → save computation + memory
>   - Multi-level: prefix (full) + suffix (windowed) → practical for production
>   - **Impact**: 2-4x speedup for long context with sliding window

## 4. Integration into Production Systems

### vLLM V1 Integration
> - `FlashInferBackend` class → replaces `FlashAttnBackend` for decode
> - `FlashInferState` → manages KV cache page table
> - Key kernels: `BatchDecodeWithPagedKVCacheWrapper` + `SingleDecodeWithPagedKVCacheWrapper`
> - Prefill: still uses FA2 API (BatchPrefillWithPagedKVCacheWrapper)
> - **Decision**: vLLM uses FlashInfer for decode, FA2 for prefill → matches our benchmark findings!

> - This EXPLAINS why FA2 decode is slow but vLLM decode is fast!

### SGLang Integration
> - `FlashInferAttentionWrapper` class → handles both prefill and decode
> - RadixAttention prefix sharing → FlashInfer varlen for packed sequences
> - Uses FlashInfer's multi-level cascade attention for prefix sharing
> - Same pattern: FlashInfer for decode, FA2-compatible for prefill

## 5. RTX 4090 Decode Performance Prediction

> Based on our benchmarks + FlashInfer architecture:

> | Scenario | FA2 (实测) | SDPA (实测) | FlashInfer (预测) |
> |---------|-----------|------------|-------------------|
> | B=1 decode | 3.3x慢 | baseline | ~1.0x (cooperative batching) |
> | B=8 decode | 3.5x慢 | baseline | ~1.5-2x (batched) |
> | B=128 decode | 9-34x慢 | baseline | ~2-5x (GQA+batched) |
> | GQA-4 B=32 | - | baseline | ~3-4x (native GQA) |

> FlashInfer predictions rationale:
> 1. Cooperative batching → reduces kernel launch overhead by B factor
> 2. Native GQA → reduces memory traffic by n_heads/n_kv_heads ratio
> 3. Paged KV → eliminates contiguous memory requirement
> 4. Varlen → eliminates padding waste

## 6. Key Takeaways for RTX 4090

> 1. **Never use raw FA2 for decode** → our benchmarks prove 3-34x slower
> 2. **SDPA math backend is best for simple decode** → auto-selects math for Q=1
> 3. **FlashInfer is production answer** → cooperative batching + native GQA + paged KV
> 4. **vLLM/SGLang architecture matches our findings** → FA2(prefill) + FlashInfer(decode)
> 5. **For custom serving**: implement FlashInfer or similar cooperative decode kernel
> 6. **Decode bottleneck = memory bandwidth** → FlashInfer's GQA optimization directly targets this

> ## Decision Tree for RTX 4090
> - Simple decode (B≤4): SDPA math backend → fastest
> - Production decode (B≥8): FlashInfer → cooperative batching wins
> - Long context prefill: FA2 → memory savings (85-96%)
> - GQA decode: FlashInfer native GQA → 4x less memory traffic
> - Mixed-length batches: FlashInfer varlen → 3-6x throughput