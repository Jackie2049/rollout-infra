# RTX 4090 Attention Variant Benchmark 实测

> GPU: NVIDIA GeForce RTX 4090, 24GB
> 日期: 2026-06-05

## 1. Decode 性能 (B=8)

| Config | S=512 | S=1024 | S=2048 | S=4096 | KV/tok (bytes) |
|--------|-------|--------|--------|--------|----------------|
| MHA (32 KV) | 0.06ms | 0.15ms | 0.29ms | 0.57ms | 16,384 |
| GQA-8 | 0.17ms | 0.34ms | 0.69ms | 1.33ms | 4,096 |
| GQA-4 | 0.14ms | 0.33ms | 0.63ms | 1.24ms | 2,048 |
| MQA (1 KV) | 0.06ms | 0.11ms | 0.21ms | 0.43ms | 512 |

**Surprising**: GQA-8/4 is SLOWER than MHA at decode! The KV head expansion (unsqueeze+expand+reshape) has overhead that outweighs the memory savings for small sequences.

## 2. Prefill 性能 (TFLOPS)

| Config | S=256 | S=512 | S=1024 | S=2048 |
|--------|-------|-------|--------|--------|
| MHA-32 | 71.8T | 72.8T | 80.5T | 82.5T |
| GQA-8 | 42.5T | 51.5T | 66.0T | 74.2T |
| GQA-4 | 49.1T | 53.4T | 66.9T | 75.3T |
| MQA | 71.1T | 79.3T | 80.7T | 82.4T |

**Key**: MHA and MQA have similar peak TFLOPS (~82T). GQA-8 is slowest at short sequences due to expansion overhead, but catches up at S=2048.

## 3. MLA Compression

| Latent Dim | MLA KB/tok | Compression | Decode ms (upsample) |
|-----------|-----------|-------------|---------------------|
| 256 | 0.5KB | **32.0x** | 0.448ms |
| 512 | 1.0KB | **16.0x** | 0.825ms |
| 1024 | 2.0KB | 8.0x | 1.643ms |
| 2048 | 4.0KB | 4.0x | 3.189ms |

**MLA-256**: 32x KV cache compression at only 0.448ms upsample cost. This is the DeepSeek-V2 approach.

## 4. KV Cache 容量分析 (7B on 24GB, 5.2GB KV budget)

| Config | KV/tok | Seq=512 | Seq=2K | Seq=8K |
|--------|--------|---------|--------|--------|
| MHA-32 | 524KB | 19 | 4 | 1 |
| GQA-8 | 131KB | 77 | 19 | 4 |
| GQA-4 | 66KB | 154 | 38 | 9 |
| MQA | 16KB | **619** | **154** | **38** |
| MLA-256 | 33KB | 309 | 77 | 19 |
| MLA-512 | 66KB | 154 | 38 | 9 |

**MQA wins on capacity**: 619 concurrent requests @ seq=512. MLA-256 achieves 32x compression but still uses more memory per token than MQA due to the latent dimension.

## 5. FlashAttention vs Manual Attention

| Seq | SDPA | Manual | Speedup |
|-----|------|--------|---------|
| 512 | 0.238ms | 1.034ms | 4.3x |
| 1024 | 0.855ms | 3.835ms | 4.5x |
| 2048 | 3.332ms | 14.764ms | 4.4x |
| 4096 | 13.293ms | OOM | - |

Consistent **4.3-4.5x** speedup. Manual attention OOMs at S=4096 (O(N²) memory).

## Key Takeaways

1. **GQA decode slower than expected**: The KV expansion operation has real overhead. vLLM/FlashInfer handle this with dedicated GQA kernels (shared KV load) that avoid explicit expansion.
2. **MLA compression is impressive**: 16-32x KV reduction with moderate upsample cost. DeepSeek-V2's choice of latent_dim=512 is a good balance.
3. **MQA is the capacity king**: 4x more concurrent requests than MLA-256 for short sequences. But MLA offers better quality retention.
4. **FlashAttention saves both time AND memory**: 4.4x faster + enables S=4096+ sequences that would otherwise OOM.
