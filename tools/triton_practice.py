#!/usr/bin/env python3
"""Triton Kernel 实战练习

在 CPU 上模拟 Triton kernel 的核心概念，为 GPU 实战做准备。
覆盖 5 个练习:
  1. Vector Add: 基础 kernel 结构、block 分配、边界处理
  2. Fused Multiply-Add: 融合 kernel vs 分离操作的性能对比
  3. Softmax: 数值稳定性 (online softmax)、reduction 操作
  4. Matrix Multiply (GEMM): 分块 (tiled) 算法、L2 cache 优化
  5. Flash Attention Simplified: tiling + online softmax 模拟

Triton 的核心思想:
  - 每个 Program Instance 处理一个 block 的数据
  - PID (program ID) → block index → global offset 计算
  - 内存层次: HBM → SRAM (block 级缓存) → register

Usage:
    conda run -n ai-infra python tools/triton_practice.py
"""

import argparse
import math
import time
import numpy as np
from typing import Optional


# ============================================================
# Triton 概念模拟
# ============================================================

def triton_kernel_sim(name: str, grid_size: int, block_size: int,
                      kernel_fn, *args, **kwargs):
    """模拟 Triton 的执行模型

    Triton 中:
      - grid = (num_blocks,) → 每个 program_id 处理一个 block
      - pid = tl.program_id(0) → 当前 block 的 index
      - offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
      - mask = offsets < n_elements  (边界处理)
    """
    total_elements = grid_size * block_size
    results = []

    # Simulate launch overhead
    launch_overhead_us = 5  # ~5μs kernel launch

    # Execute each "program" (block)
    for pid in range(grid_size):
        block_result = kernel_fn(pid, block_size, total_elements, *args, **kwargs)
        results.append(block_result)

    return results


def experiment_1_vector_add():
    """练习 1: Vector Add — Triton kernel 基础结构"""
    print("\n" + "=" * 70)
    print("练习 1: Vector Add — Triton kernel 基础结构")
    print("=" * 70)

    n = 1024 * 1024  # 1M elements
    BLOCK_SIZE = 1024
    num_blocks = math.ceil(n / BLOCK_SIZE)

    # Input data
    x = np.random.randn(n).astype(np.float32)
    y = np.random.randn(n).astype(np.float32)

    # --- NumPy baseline ---
    t0 = time.perf_counter()
    z_np = x + y
    t_numpy = (time.perf_counter() - t0) * 1000

    # --- Simulated Triton kernel ---
    def vector_add_kernel(pid, block_size, n_elements, x_arr, y_arr):
        """模拟 Triton vector_add kernel

        真实 Triton 代码:
          @triton.jit
          def vector_add_kernel(x_ptr, y_ptr, z_ptr, n, BLOCK_SIZE: tl.constexpr):
              pid = tl.program_id(0)
              offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
              mask = offsets < n
              x = tl.load(x_ptr + offsets, mask=mask)
              y = tl.load(y_ptr + offsets, mask=mask)
              z = x + y
              tl.store(z_ptr + offsets, z, mask=mask)
        """
        start = pid * block_size
        end = min(start + block_size, n_elements)
        z_block = x_arr[start:end] + y_arr[start:end]
        return (start, end, z_block)

    t0 = time.perf_counter()
    results = triton_kernel_sim("vector_add", num_blocks, BLOCK_SIZE,
                                vector_add_kernel, x, y)
    z_triton = np.empty_like(x)
    for start, end, z_block in results:
        z_triton[start:end] = z_block
    t_triton = (time.perf_counter() - t0) * 1000

    # Verify
    assert np.allclose(z_np, z_triton), "Mismatch!"

    print(f"\n📊 Vector Add ({n/1e6:.1f}M elements, BLOCK_SIZE={BLOCK_SIZE}):")
    print(f"  NumPy:         {t_numpy:.2f} ms")
    print(f"  Sim Triton:    {t_triton:.2f} ms  ({num_blocks} blocks)")
    print(f"  结果验证:      ✅ 一致")

    # Triton 关键概念
    print(f"\n💡 Triton 核心概念:")
    print(f"  1. Grid = ({num_blocks},) → 启动 {num_blocks} 个 program instances")
    print(f"  2. pid = tl.program_id(0) → 每个 instance 获得唯一 ID")
    print(f"  3. offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)")
    print(f"  4. mask = offsets < n → 处理最后一个 block 的边界")
    print(f"  5. tl.load/tl.store → 从/向 HBM 读写数据")

    # Block size impact
    print(f"\n📊 BLOCK_SIZE 对 kernel launch 开销的影响:")
    for bs in [128, 256, 512, 1024, 2048, 4096]:
        nb = math.ceil(n / bs)
        # Real GPU: kernel launch ~5μs per kernel, not per block
        launch_us = 5
        compute_us = n / (bs * nb) * 1  # Simulated
        total_us = launch_us + compute_us
        print(f"  BS={bs:<6} → {nb:<6} blocks, launch={launch_us}μs + compute")


def experiment_2_fused_multiply_add():
    """练习 2: Fused Multiply-Add — 融合操作 vs 分离操作"""
    print("\n" + "=" * 70)
    print("练习 2: Fused Multiply-Add — 融合 vs 分离")
    print("=" * 70)

    n = 1024 * 512  # 512K elements
    x = np.random.randn(n).astype(np.float32)
    a = 2.5
    b = 1.0

    # --- 方法 1: 分离操作 (2 次 HBM 读写) ---
    # temp = x * a      → read x, write temp
    # out = temp + b    → read temp, write out
    t0 = time.perf_counter()
    temp = x * a
    out_separate = temp + b
    t_separate = (time.perf_counter() - t0) * 1000

    # --- 方法 2: 融合操作 (1 次 HBM 读写) ---
    # out = x * a + b   → read x, write out (temp in register/SRAM)
    t0 = time.perf_counter()
    out_fused = x * a + b
    t_fused = (time.perf_counter() - t0) * 1000

    # --- 方法 3: NumPy fused ---
    t0 = time.perf_counter()
    out_np_fused = np.multiply(x, a, out=np.empty_like(x))
    np.add(out_np_fused, b, out=out_np_fused)
    t_np = (time.perf_counter() - t0) * 1000

    assert np.allclose(out_separate, out_fused), "Mismatch!"

    print(f"\n📊 Fused Multiply-Add ({n/1e3:.0f}K elements):")
    print(f"  分离 (x*a → temp, temp+b → out): {t_separate:.3f} ms")
    print(f"  融合 (x*a+b → out):              {t_fused:.3f} ms")
    print(f"  NumPy in-place:                   {t_np:.3f} ms")

    # Memory traffic analysis
    bytes_per_elem = 4  # float32
    mem_read_separate = 3 * n * bytes_per_elem  # x, temp (read x, write temp, read temp, write out)
    mem_read_fused = 2 * n * bytes_per_elem     # x (read x, write out)

    print(f"\n📊 HBM 访问分析:")
    print(f"  分离: 4×N 字节 (x→temp→out) = {mem_read_separate/1e6:.1f} MB")
    print(f"  融合: 2×N 字节 (x→out)      = {mem_read_fused/1e6:.1f} MB")
    print(f"  融合节省: {(1 - mem_read_fused/mem_read_separate)*100:.0f}% HBM 带宽")

    print(f"\n💡 Triton 融合的优势:")
    print(f"  - temp = x * a 的中间结果只存于 register/SRAM, 不写回 HBM")
    print(f"  - 对 memory-bound 操作, 节省 50% HBM 访问 ≈ 2x 加速")
    print(f"  - 这就是 PyTorch 的 torch.compile 和 Triton 的核心价值")


def experiment_3_softmax():
    """练习 3: Softmax — 数值稳定性 + reduction"""
    print("\n" + "=" * 70)
    print("练习 3: Softmax — 数值稳定性 + Online Softmax")
    print("=" * 70)

    n = 4096
    x = np.random.randn(n).astype(np.float32) * 10  # Scale up for instability

    # --- 方法 1: Naive softmax (数值不稳定) ---
    def naive_softmax(x):
        """softmax(x) = exp(x) / sum(exp(x))"""
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x)

    # --- 方法 2: Safe softmax (减去 max) ---
    def safe_softmax(x):
        """softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))"""
        x_max = np.max(x)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x)

    # --- 方法 3: Online softmax (FlashAttention 风格) ---
    def online_softmax(x, block_size=256):
        """Online softmax: 分块计算 max 和 sum

        FlashAttention 的核心技巧:
          m_new = max(m_old, m_block)
          l_new = l_old * exp(m_old - m_new) + sum(exp(x_block - m_new))
        """
        n = len(x)
        m = -np.inf  # running max
        l = 0.0      # running sum of exp(x - m)
        o = np.zeros_like(x)  # running output accumulator

        for start in range(0, n, block_size):
            end = min(start + block_size, n)
            x_block = x[start:end]

            # New max for this block
            m_block = np.max(x_block)
            m_new = max(m, m_block)

            # Update running sum with correction
            l_new = l * np.exp(m - m_new) + np.sum(np.exp(x_block - m_new))

            # Update output accumulator
            if m != -np.inf:
                o[:start] *= np.exp(m - m_new)
            o[start:end] = np.exp(x_block - m_new)

            m = m_new
            l = l_new

        o /= l
        return o

    # --- 方法 4: Triton softmax 模拟 (per-block reduction) ---
    def triton_softmax_sim(x, BLOCK_SIZE=256):
        """模拟 Triton softmax kernel (row-wise: 整行作为一个 kernel)

        真实 Triton 中 softmax 通常是 2D: 每个 program 处理一行
          @triton.jit
          def softmax_kernel(input_ptr, output_ptr, n_cols, stride,
                             BLOCK_SIZE: tl.constexpr):
              row_idx = tl.program_id(0)
              offsets = row_idx * stride + tl.arange(0, BLOCK_SIZE)
              mask = offsets < n_cols  # actually mask by column index
              x = tl.load(input_ptr + offsets, mask=mask, other=-float('inf'))
              x_max = tl.max(x, axis=0)
              exp_x = tl.exp(x - x_max)
              sum_exp = tl.sum(exp_x, axis=0)
              out = exp_x / sum_exp
              tl.store(output_ptr + offsets, out, mask=mask)

        注意: 真正的 Triton softmax 通常按行处理 (2D grid)。
        这里模拟单行的情况: 一个 program 实例处理整行。
        如果行很长 > BLOCK_SIZE, 需要用 online softmax 模式。
        """
        n = len(x)
        output = np.empty_like(x)

        # Triton 中如果 n <= BLOCK_SIZE, 一个 program 实例即可
        # 这里模拟 n <= BLOCK_SIZE 的情况
        x_padded = np.full(max(BLOCK_SIZE, n), -np.inf, dtype=np.float32)
        x_padded[:n] = x

        # Triton reduction operations (单 program instance)
        x_max = np.max(x_padded)
        exp_x = np.exp(x_padded - x_max)
        sum_exp = np.sum(exp_x)
        out = exp_x / sum_exp

        output[:] = out[:n]

        return output

    # Compare
    out_naive = naive_softmax(x)
    out_safe = safe_softmax(x)
    out_online = online_softmax(x)
    out_triton = triton_softmax_sim(x)

    print(f"\n📊 Softmax 对比 (n={n}, x scaled by 10):")
    print(f"  Naive sum:     {np.sum(out_naive):.10f}  (应=1.0)")
    print(f"  Safe sum:      {np.sum(out_safe):.10f}")
    print(f"  Online sum:    {np.sum(out_online):.10f}")
    print(f"  Triton sum:    {np.sum(out_triton):.10f}")
    print(f"  Safe vs Online max diff: {np.max(np.abs(out_safe - out_online)):.2e}")
    print(f"  Safe vs Triton max diff: {np.max(np.abs(out_safe - out_triton)):.2e}")

    # Block size impact on online softmax
    print(f"\n📊 Online Softmax BLOCK_SIZE 影响:")
    for bs in [64, 128, 256, 512, 1024, 2048]:
        t0 = time.perf_counter()
        for _ in range(100):
            online_softmax(x, block_size=bs)
        t_ms = (time.perf_counter() - t0)
        out = online_softmax(x, block_size=bs)
        diff = np.max(np.abs(out_safe - out))
        print(f"  BS={bs:<5} → 100次 {t_ms:.3f}s, max diff={diff:.2e}")

    print(f"\n💡 Triton Softmax 关键:")
    print(f"  - tl.max() / tl.sum(): block 内 reduction (在 SRAM 中完成)")
    print(f"  - mask + other=-inf: 边界元素不影响 max/sum")
    print(f"  - Online softmax 是 FlashAttention 的核心组件")


def experiment_4_tiled_gemm():
    """练习 4: Tiled GEMM — 分块矩阵乘法"""
    print("\n" + "=" * 70)
    print("练习 4: Tiled GEMM — 分块矩阵乘法")
    print("=" * 70)

    M, N, K = 1024, 1024, 1024
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32

    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)

    # --- NumPy baseline ---
    t0 = time.perf_counter()
    C_np = A @ B
    t_numpy = (time.perf_counter() - t0) * 1000

    # --- Tiled GEMM simulation ---
    def tiled_gemm(A, B, BLOCK_M, BLOCK_N, BLOCK_K):
        """模拟 Triton tiled GEMM

        真实 Triton:
          @triton.jit
          def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                           BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                           BLOCK_K: tl.constexpr):
              pid_m = tl.program_id(0)
              pid_n = tl.program_id(1)
              offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
              offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
              offs_k = tl.arange(0, BLOCK_K)

              accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
              for k in range(0, K, BLOCK_K):
                  a = tl.load(a_ptr + (offs_m[:, None] * K + (k + offs_k[None, :])))
                  b = tl.load(b_ptr + ((k + offs_k[:, None]) * N + offs_n[None, :]))
                  accumulator += tl.dot(a, b)
              tl.store(c_ptr + (offs_m[:, None] * N + offs_n[None, :]), accumulator)
        """
        M, K = A.shape
        _, N = B.shape
        C = np.zeros((M, N), dtype=np.float32)

        for pid_m in range(M // BLOCK_M):
            for pid_n in range(N // BLOCK_N):
                accumulator = np.zeros((BLOCK_M, BLOCK_N), dtype=np.float32)
                for k_start in range(0, K, BLOCK_K):
                    k_end = min(k_start + BLOCK_K, K)

                    a_block = A[pid_m * BLOCK_M:(pid_m + 1) * BLOCK_M,
                               k_start:k_end]
                    b_block = B[k_start:k_end,
                               pid_n * BLOCK_N:(pid_n + 1) * BLOCK_N]

                    accumulator += a_block @ b_block

                C[pid_m * BLOCK_M:(pid_m + 1) * BLOCK_M,
                  pid_n * BLOCK_N:(pid_n + 1) * BLOCK_N] = accumulator

        return C

    t0 = time.perf_counter()
    C_tiled = tiled_gemm(A, B, BLOCK_M, BLOCK_N, BLOCK_K)
    t_tiled = (time.perf_counter() - t0) * 1000

    # Analysis
    num_blocks_m = M // BLOCK_M
    num_blocks_n = N // BLOCK_N
    total_blocks = num_blocks_m * num_blocks_n
    num_k_steps = K // BLOCK_K

    # Memory traffic analysis
    # Each output block: BLOCK_M * BLOCK_N * (2 * BLOCK_K * (BLOCK_M + BLOCK_N)) bytes
    # SRAM usage per block
    sram_per_block = (BLOCK_M * BLOCK_K + BLOCK_K * BLOCK_N + BLOCK_M * BLOCK_N) * 4
    hbm_per_block = 2 * BLOCK_K * (BLOCK_M + BLOCK_N) * 4  # A block + B block reads

    total_ops = 2 * M * N * K  # FLOPs
    total_hbm = M * K * 4 + K * N * 4 + M * N * 4  # Minimum: read A, read B, write C

    print(f"\n📊 Tiled GEMM ({M}x{K} × {K}x{N}):")
    print(f"  NumPy:        {t_numpy:.2f} ms  ({total_ops/1e9:.2f} GFLOP)")
    print(f"  Tiled sim:    {t_tiled:.2f} ms  ({total_blocks} blocks × {num_k_steps} k-steps)")
    print(f"  结果验证:     {'✅' if np.allclose(C_np, C_tiled, atol=1e-3) else '❌'}")

    print(f"\n📊 分块分析:")
    print(f"  Block sizes:       {BLOCK_M}x{BLOCK_N}x{BLOCK_K}")
    print(f"  Output blocks:     {total_blocks}")
    print(f"  K-steps per block: {num_k_steps}")
    print(f"  SRAM/block:        {sram_per_block/1024:.1f} KB")
    print(f"  HBM reads total:   {total_hbm/1e6:.1f} MB")
    print(f"  Arithmetic Intensity: {total_ops / total_hbm:.1f} ops/byte")

    # L2 cache optimization: group blocks by pid_m for locality
    print(f"\n💡 L2 Cache 优化 (Triton 2D grid):")
    print(f"  - Triton grid = ({num_blocks_m}, {num_blocks_n})")
    print(f"  - 默认遍历: row-major → A 行数据在 L2 中复用")
    print(f"  - 可以用 pid_m = pid // GROUP_SIZE_M 进一步优化 L2 命中")
    print(f"  - 每个 SM 的 SRAM ≈ 192KB (A100), block 需 fit")


def experiment_5_flash_attention_simplified():
    """练习 5: Flash Attention 简化 — Tiling + Online Softmax"""
    print("\n" + "=" * 70)
    print("练习 5: Flash Attention 简化 — Tiling + Online Softmax")
    print("=" * 70)

    # Small dimensions for CPU simulation
    batch = 1
    n_heads = 1
    seq_len = 256
    head_dim = 64
    BLOCK_SIZE = 64

    Q = np.random.randn(seq_len, head_dim).astype(np.float32) / np.sqrt(head_dim)
    K = np.random.randn(seq_len, head_dim).astype(np.float32)
    V = np.random.randn(seq_len, head_dim).astype(np.float32)

    # --- 方法 1: Naive attention (O(N²) memory) ---
    def naive_attention(Q, K, V):
        """S = QK^T / sqrt(d), O = softmax(S) @ V"""
        scale = 1.0 / np.sqrt(head_dim)
        S = (Q @ K.T) * scale
        # Safe softmax
        S_max = np.max(S, axis=-1, keepdims=True)
        S_exp = np.exp(S - S_max)
        S_sum = np.sum(S_exp, axis=-1, keepdims=True)
        P = S_exp / S_sum
        O = P @ V
        return O

    # --- 方法 2: Flash Attention (tiling, O(N) memory) ---
    def flash_attention_sim(Q, K, V, block_size=64):
        """模拟 Flash Attention 算法

        核心思想:
          1. 分块遍历 K,V (外循环)
          2. 对每个 Q block, 增量更新:
             m_new = max(m_old, m_block)
             l_new = l_old * exp(m_old - m_new) + sum(exp(S_block - m_new))
             O_new = (O_old * l_old * exp(m_old - m_new) + P_block @ V_block) / l_new

        内存: 只需 O(N × d) for O, m, l
        不需: N × N attention matrix S
        """
        N = Q.shape[0]
        d = Q.shape[1]
        scale = 1.0 / np.sqrt(d)

        # Output accumulators
        O = np.zeros((N, d), dtype=np.float32)
        m = np.full(N, -np.inf, dtype=np.float32)  # running max per row
        l = np.zeros(N, dtype=np.float32)            # running sum per row

        # Outer loop: iterate over K, V blocks
        for j_start in range(0, N, block_size):
            j_end = min(j_start + block_size, N)
            K_block = K[j_start:j_end]  # (block_size, d)
            V_block = V[j_start:j_end]

            # Inner loop: all Q rows (can also be blocked)
            # S_block = Q @ K_block^T * scale  → (N, block_size)
            S_block = (Q @ K_block.T) * scale

            # Block max
            m_block = np.max(S_block, axis=-1)  # (N,)

            # New running max
            m_new = np.maximum(m, m_block)

            # Correction factor
            alpha = np.exp(m - m_new)
            beta = np.exp(S_block - m_new[:, None])

            # Update running sum
            l_new = l * alpha + np.sum(beta, axis=-1)

            # Update output accumulator
            O = O * alpha[:, None] + beta @ V_block

            m = m_new
            l = l_new

        O = O / l[:, None]
        return O

    t0 = time.perf_counter()
    O_naive = naive_attention(Q, K, V)
    t_naive = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    O_flash = flash_attention_sim(Q, K, V, block_size=BLOCK_SIZE)
    t_flash = (time.perf_counter() - t0) * 1000

    diff = np.max(np.abs(O_naive - O_flash))
    rel_diff = diff / np.max(np.abs(O_naive))

    print(f"\n📊 Flash Attention 对比 (seq={seq_len}, dim={head_dim}):")
    print(f"  Naive:        {t_naive:.3f} ms")
    print(f"  Flash sim:    {t_flash:.3f} ms  (block_size={BLOCK_SIZE})")
    print(f"  Max abs diff: {diff:.2e}")
    print(f"  Max rel diff: {rel_diff:.2e}")

    # Memory analysis
    naive_mem = seq_len * seq_len * 4  # N×N attention matrix
    flash_mem = seq_len * head_dim * 4 + seq_len * 4 * 2  # O + m + l

    print(f"\n📊 内存分析:")
    print(f"  Naive (S 矩阵):     {naive_mem/1024:.1f} KB")
    print(f"  Flash (O+m+l):      {flash_mem/1024:.1f} KB")
    print(f"  内存节省:           {(1 - flash_mem/naive_mem)*100:.1f}%")

    # Scale analysis
    print(f"\n📊 不同 seq_len 的内存对比:")
    for sl in [256, 512, 1024, 4096, 8192, 16384, 32768]:
        naive_m = sl * sl * 4 / 1024  # KB
        flash_m = (sl * head_dim * 4 + sl * 8) / 1024  # KB
        ratio = naive_m / flash_m
        print(f"  seq={sl:<6} → Naive: {naive_m:>10.1f} KB, Flash: {flash_m:>8.1f} KB "
              f"({ratio:>6.1f}x 节省)")

    print(f"\n💡 Flash Attention 在 Triton 中的实现:")
    print(f"  - Q block 在 SRAM, 遍历 K/V blocks")
    print(f"  - Online softmax 避免 materialize N×N 矩阵")
    print(f"  - 实际 Triton kernel: 2D grid, Q blocks × K/V blocks")
    print(f"  - causal mask: j_start <= i_end 时才计算")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Triton Kernel 实战练习")
    parser.add_argument("--experiment", "-e", type=int, default=0,
                        help="运行特定练习 (1-5), 0=全部")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║          Triton Kernel 实战练习                            ║")
    print("║   在 CPU 上模拟 Triton kernel 核心概念                     ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    experiments = {
        1: experiment_1_vector_add,
        2: experiment_2_fused_multiply_add,
        3: experiment_3_softmax,
        4: experiment_4_tiled_gemm,
        5: experiment_5_flash_attention_simplified,
    }

    if args.experiment == 0:
        for exp_fn in experiments.values():
            exp_fn()
    elif args.experiment in experiments:
        experiments[args.experiment]()
    else:
        print(f"Unknown experiment: {args.experiment}")
        return

    print("\n" + "=" * 70)
    print("✅ 练习完成 — 理解这些概念后可在 GPU 上编写真实 Triton kernel")
    print("=" * 70)


if __name__ == "__main__":
    main()
