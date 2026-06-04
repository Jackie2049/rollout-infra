# RTX 4090 Mini LLM Training Pipeline 实测

> GPU: NVIDIA GeForce RTX 4090, 24GB
> 模型: 838K 参数 GPT, 4层, 4头, d_model=128
> 日期: 2026-06-05

## 1. Pretraining (Next-Token Prediction)

| Epoch | Loss | LR | Throughput |
|-------|------|----|------------|
| 0 | 4.923 | 3.0e-4 | 556K tok/s |
| 9 | 2.993 | 2.3e-4 | 567K tok/s |
| 18 | 2.598 | 8.9e-5 | 567K tok/s |
| 29 | **2.462** | 0.0 | 571K tok/s |

- Loss: 4.92 → 2.46 (50% reduction)
- CosineAnnealing LR schedule
- AdamW optimizer, weight_decay=0.1, grad_clip=1.0

## 2. SFT (Supervised Fine-Tuning)

| Epoch | Loss | LR | Throughput |
|-------|------|----|------------|
| 0 | 1.684 | 1.0e-4 | 287K tok/s |
| 10 | 1.136 | 4.2e-5 | 291K tok/s |
| 19 | **1.114** | 0.0 | 290K tok/s |

- 500 instruction-response pairs (reverse/double patterns)
- SFT throughput: 290K tok/s (lower due to smaller batch=16)

## 3. Parameter Breakdown

| Component | Params | % |
|-----------|--------|---|
| Embedding | 49K | 5.9% |
| Attention | 262K | 31.3% |
| MLP | 524K | **62.6%** |
| LayerNorm | 2.3K | 0.3% |

**Key insight**: MLP is the largest component (62.6%), confirming that SwiGLU FFN dominates transformer parameter count.

## 4. Scaling Projections

| Model Size | FLOPS Needed | Hardware-Days (A100) |
|-----------|-------------|---------------------|
| 838K | 0.05 PFLOPS | <1s |
| 125M | 1.0 EFLOPS | ~0.01 days |
| 1.3B | 108 EFLOPS | ~1 day |
| 7B | 3144 EFLOPS | ~36 days (A100×256) |

**Chinchilla scaling**: FLOPS ≈ 6 × N² × D/N = 6ND, where training tokens D ∝ N.

## 5. GPU Utilization

- Peak memory: 268.5 MB / 24564 MB (**1.1%** of VRAM)
- Model is tiny — real training would use 90%+ of VRAM
- Throughput ~570K tok/s for 838K model
- RTX 4090 peak: ~529K tok/s decode (inference), comparable training throughput
