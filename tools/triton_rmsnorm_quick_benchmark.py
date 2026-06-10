#!/usr/bin/env python3
"""Quick Triton RMSNorm Benchmark — RTX 4090 (Triton 3.1.0 compatible)"""

import torch
import triton
import triton.language as tl
import time
import json

device = 'cuda:0'
n_gpu = torch.cuda.device_count()
print(f'GPUs: {n_gpu}, Triton: {triton.__version__}')

# Triton 3.1.0 kernel definition
@triton.jit
def rms_norm_kernel(x_ptr, eps_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    col = tl.program_id(0)
    row_start = col * BLOCK_SIZE
    row_end = min((row_start + BLOCK_SIZE), N)
    row = row_start + tl.arange(0, BLOCK_SIZE)
    mask = row < N
    x_row = tl.load(x_ptr + row, mask=mask, other=0.0).to(tl.float32)
    x_row_sq = x_row * x_row
    mean = tl.sum(x_row_sq, axis=0) / N
    variance = tl.sum(x_row_sq, axis=0) / N - mean * mean
    eps_val = tl.load(eps_ptr).to(tl.float32)
    rms = tl.sqrt(variance + eps_val)
    inv_rms = 1.0 / rms
    tl.store(out_ptr + row, x_row * inv_rms, mask=mask)

def rms_norm_triton(x, eps, out, N, block_size=256):
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    rms_norm_kernel[grid](x, eps, out, N, BLOCK_SIZE=block_size)

results = {}
for N in [256, 512, 1024, 2048, 4096, 8192]:
    block_size = min(256, N)
    x = torch.randn(N, device=device, dtype=torch.float32)
    eps_val = torch.tensor([1e-6], device=device, dtype=torch.float32)
    out = torch.zeros_like(x)

    warmup = 10
    runs = 200

    # Triton RMSNorm
    for _ in range(warmup):
        rms_norm_triton(x, eps_val, out, N, block_size)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(runs):
        rms_norm_triton(x, eps_val, out, N, block_size)
    torch.cuda.synchronize()
    dt_triton = (time.perf_counter() - t0) / runs

    # PyTorch RMSNorm
    def pytorch_rms_norm(x, eps=1e-6):
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + eps)

    for _ in range(warmup):
        pytorch_rms_norm(x)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(runs):
        pytorch_rms_norm(x)
    torch.cuda.synchronize()
    dt_py = (time.perf_counter() - t0) / runs

    speedup = dt_py / dt_triton
    results[f'N={N}'] = {
        'triton_us': round(dt_triton*1e6, 2),
        'pytorch_us': round(dt_py*1e6, 2),
        'speedup': round(speedup, 2),
    }
    print(f'N={N}: Triton={dt_triton*1e6:.1f}us, PyTorch={dt_py*1e6:.1f}us, speedup={speedup:.2f}x')

# Save
output = 'triton_rmsnorm_quick_benchmark.json'
with open(output, 'w') as f:
    json.dump(results, f, indent=2)
print(f'Saved to {output}')