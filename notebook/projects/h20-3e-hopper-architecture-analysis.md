# NVIDIA H20-3e GPU — Hopper Architecture for China Market

## Key Specs (from A100 server observations)

| Feature | H20-3e | H100 | A100-SXM4-80GB |
|---------|--------|------|----------------|
| Architecture | Hopper | Hopper | Ampere |
| Memory | 143 GB HBM3 | 80-100 GB HBM3 | 80 GB HBM2e |
| Memory BW | ~4 TB/s | ~3.35 TB/s | ~2 TB/s |
| FP16 TFLOPS | ~148 | ~1979 (with FP8) | ~312 |
| FP8 support | YES | YES | NO |
| CUDA Compute | 9.0 | 9.0 | 8.0 |
| TMA (Tensor Mem Accelerator) | YES | YES | NO |
| Thread Block Cluster | YES | YES | NO |
| DPX (Dynamic Programming X) | YES | YES | NO |

## H20-3e Specifics

- **3e variant**: Reduced FP32/FP64 throughput compared to H100 (export compliance)
- **HBM3**: Faster memory bandwidth than A100's HBM2e
- **143 GB VRAM**: Significantly more than A100 (80GB) — enables larger models
- **FP8 native**: Can use FP8 for training (E5M2/E4M3 formats) via CUTLASS
- **TMA unit**: Async tensor memory access — important for attention kernels
- **No FP64 restriction for ML**: FP64 not needed for training/inference

## Relevance to Our Work

### GRPO Training on H20-3e
- 143GB VRAM → can fit Qwen3.5-32B (4×143 = 572GB total, ~40GB model + KV cache)
- FP8 training support → P9's fp8-bf16 fusion guard applies (SM90 compatible)
- TMA → vLLM V1's Triton attention kernels use TMA on SM90

### DeepSpeed AutoEP on H20-3e
- 4 GPUs × 143GB = enough for MoE models with EP
- Cross-lane EP (from #8064) would work well here
- FP8 expert weights → further memory savings for MoE

### vLLM Serving on H20-3e
- FP8 KV cache → 2x cache capacity vs BF16
- 143GB → can serve larger models or more concurrent requests
- FlashAttention-3 (Hopper-specific) should be available

### RTX 4090 vs H20-3e Comparison
| Aspect | RTX 4090 | H20-3e |
|--------|----------|---------|
| VRAM | 24 GB | 143 GB |
| Architecture | Ada Lovelace | Hopper |
| FP8 | YES | YES |
| NVLink | NO | YES (4-GPU) |
| Multi-GPU | PCIe only | NVLink + NVSwitch |

## FP8 Training Considerations on Hopper

- **SM90 FP8 templates**: CUTLASS has `sm90_fp8_e5m2_gemm` and `sm90_fp8_e4m3_gemm`
- **Hybrid FP8 training**: Keep master weights in BF16, compute in FP8
- **Delayed scaling**: Most common FP8 scaling strategy (collect max values from previous step)
- **Calibration**: May need offline calibration for stable FP8 training
- **P9 fusion guard**: jansel's SM89 guard (PR #184119) extends to SM90

## Key Differences from A100

1. **Memory**: 143GB vs 80GB — nearly 2x, critical for MoE and long-context
2. **FP8 native**: A100 has no FP8 hardware — must emulate in software
3. **TMA**: Async memory access enables better overlap of compute and memory
4. **Thread Block Cluster**: Groups of thread blocks share memory — MoE-friendly
5. **NVLink**: 4-GPU NVLink topology vs A100's 4-GPU NVLink (same topology but faster)

## Impact on Patch Testing

Our TRL patches (P7-2 top_n_sigma, P9-1 bypass_mode) are GPU-independent —
they modify algorithm logic not hardware paths. However:
- P9-1 bypass_mode=True saves more on H20-3e because each forward pass is more expensive
- #6274 VLM fix will benefit from 143GB VRAM (can load Qwen3-VL-72B)
- FP8 KV cache on H20-3e would complement SGLang's RolloutKV pinning

## Reference
- Server: 10.26.6.88:31954 (root, password JDF+H200@sribd)
- Observed: 2026-07-14, nvidia-smi reports NVIDIA H20-3e, 143771 MiB total
