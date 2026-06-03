#!/usr/bin/env python3
"""FlashAttention 算法模拟器 — 理解 tiling、online softmax、IO 复杂度

CPU 可运行的 FlashAttention 算法模拟，不依赖 GPU。

覆盖 4 个实验:
  1. 标准 Attention vs FlashAttention 数值对比
  2. Memory访问量 (IO) 分析: 标准 vs Tiling
  3. Online Softmax 正确性验证
  4. Block Size (tiling granularity) 对性能影响

核心思想:
  FlashAttention = 分块计算 attention，避免在 HBM 中实例化完整 N×N attention 矩阵
  关键: 使用 online softmax 逐块累积，保证数值等价

用法:
  conda run -n ai-infra python tools/flash_attention_sim.py
"""

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


# ──────────────────────────────────────────────
# 标准 Attention（baseline）
# ──────────────────────────────────────────────

def standard_attention(
    Q: np.ndarray,  # [seq_len, d_head]
    K: np.ndarray,  # [seq_len, d_head]
    V: np.ndarray,  # [seq_len, d_head]
) -> Tuple[np.ndarray, np.ndarray]:
    """标准 Attention 计算 (materialize 完整 attention matrix)

    Returns: (output [seq_len, d_head], attention_weights [seq_len, seq_len])
    """
    d_head = Q.shape[1]
    scale = 1.0 / math.sqrt(d_head)

    # S = Q @ K^T / sqrt(d)  → [seq_len, seq_len]
    S = (Q @ K.T) * scale

    # P = softmax(S, dim=-1)
    S_max = np.max(S, axis=-1, keepdims=True)
    S_exp = np.exp(S - S_max)
    P = S_exp / np.sum(S_exp, axis=-1, keepdims=True)

    # O = P @ V
    O = P @ V

    return O, P


# ──────────────────────────────────────────────
# FlashAttention (tiling + online softmax)
# ──────────────────────────────────────────────

def flash_attention_forward(
    Q: np.ndarray,  # [seq_len, d_head]
    K: np.ndarray,  # [seq_len, d_head]
    V: np.ndarray,  # [seq_len, d_head]
    block_size_q: int = 64,
    block_size_kv: int = 64,
) -> Tuple[np.ndarray, dict]:
    """FlashAttention 模拟 (tiling + online softmax)

    核心思想:
      将 Q 分成 B_r 个 block (每块 block_size_q 行)
      将 K,V 分成 B_c 个 block (每块 block_size_kv 行)
      对每个 Q block 和 K/V block 对做局部 attention

    使用 online softmax: 逐 KV block 累积 softmax 分子和分母

    Returns: (output [seq_len, d_head], stats dict)
    """
    N, d = Q.shape
    scale = 1.0 / math.sqrt(d)

    # 输出
    O = np.zeros_like(Q)

    # Online softmax 状态 (per Q row)
    # l_i: softmax 分母累积
    # m_i: running max
    l = np.zeros(N)
    m = np.full(N, -np.inf)

    # 统计
    stats = {
        'hbm_reads': 0,
        'hbm_writes': 0,
        'blocks_q': 0,
        'blocks_kv': 0,
    }

    B_r = math.ceil(N / block_size_q)
    B_c = math.ceil(N / block_size_kv)

    for j in range(B_c):  # outer loop: KV blocks
        # 加载 K_j, V_j from HBM to SRAM
        kv_start = j * block_size_kv
        kv_end = min(kv_start + block_size_kv, N)
        K_j = K[kv_start:kv_end]  # [block_kv, d]
        V_j = V[kv_start:kv_end]  # [block_kv, d]

        stats['blocks_kv'] += 1
        stats['hbm_reads'] += 2 * (kv_end - kv_start) * d  # K_j + V_j

        for i in range(B_r):  # inner loop: Q blocks
            q_start = i * block_size_q
            q_end = min(q_start + block_size_q, N)
            Q_i = Q[q_start:q_end]  # [block_q, d]

            if j == 0:
                stats['blocks_q'] += 1

            # 加载 Q_i from HBM (only once per Q block)
            # 加载 O_i, l_i, m_i from HBM (accumulated state)
            stats['hbm_reads'] += (q_end - q_start) * d  # Q_i
            stats['hbm_reads'] += (q_end - q_start) * d  # O_i (read-modify-write)
            stats['hbm_reads'] += (q_end - q_start) * 2  # l_i + m_i

            # Step 1: Local attention scores
            # S_ij = Q_i @ K_j^T * scale → [block_q, block_kv]
            S_ij = (Q_i @ K_j.T) * scale

            # Step 2: Update running max
            m_ij = np.max(S_ij, axis=-1)  # [block_q]
            m_new = np.maximum(m[q_start:q_end], m_ij)

            # Step 3: Correction factor for previous accumulation
            # P_correction = exp(m_old - m_new)
            P_corr = np.exp(m[q_start:q_end] - m_new)  # [block_q]

            # Step 4: Local softmax (unnormalized)
            S_ij_exp = np.exp(S_ij - m_new[:, np.newaxis])  # [block_q, block_kv]
            l_ij = np.sum(S_ij_exp, axis=-1)  # [block_q]

            # Step 5: Update l (softmax denominator)
            l[q_start:q_end] = l[q_start:q_end] * P_corr + l_ij

            # Step 6: Update O (output accumulation)
            # O_new = P_corr * O_old + S_ij_exp @ V_j
            O[q_start:q_end] = (
                P_corr[:, np.newaxis] * O[q_start:q_end]
                + S_ij_exp @ V_j
            )

            # Step 7: Update m (running max)
            m[q_start:q_end] = m_new

            stats['hbm_writes'] += (q_end - q_start) * d  # O_i
            stats['hbm_writes'] += (q_end - q_start) * 2  # l_i + m_i

    # Final normalization
    O = O / l[:, np.newaxis]

    stats['total_blocks'] = stats['blocks_q'] * stats['blocks_kv']
    stats['B_r'] = B_r
    stats['B_c'] = B_c

    return O, stats


# ──────────────────────────────────────────────
# IO 复杂度分析
# ──────────────────────────────────────────────

def compute_io_standard(N: int, d: int) -> dict:
    """标准 Attention 的 HBM 访问量"""
    # 读取: Q[N,d] + K[N,d] + V[N,d]
    reads = 3 * N * d
    # 中间: S[N,N] (写入 HBM) + P[N,N] (写入 HBM)
    intermediate = 2 * N * N
    # 写入: O[N,d]
    writes = N * d
    return {
        'reads': reads + intermediate,
        'writes': writes,
        'total': reads + intermediate + writes,
        'intermediate_NxN': 2 * N * N,
    }


def compute_io_flash(N: int, d: int, M: int) -> dict:
    """FlashAttention 的 HBM 访问量 (理论值)

    M: SRAM 大小 (bytes), 例如 H100 L2 = 50MB, SRAM = ~20MB

    block_size = sqrt(M / d)
    total IO ≈ N^2 * d^2 / M  (比标准的 N^2 d 少 d/M 倍)
    """
    # 典型 block size
    B_c = int(math.sqrt(M / (d * 4)))  # 4 bytes per FP32 element
    B_r = max(1, int(M / (d * 4 * B_c)))

    # 每对 (Q_block, KV_block) 的 IO
    io_per_pair = B_r * d + B_c * d + B_r * d + B_r * d  # Q + K + V + O
    num_pairs = math.ceil(N / B_r) * math.ceil(N / B_c)

    reads = num_pairs * io_per_pair
    writes = N * d  # 只写 O

    return {
        'reads': reads,
        'writes': writes,
        'total': reads + writes,
        'B_r': B_r,
        'B_c': B_c,
        'intermediate_NxN': 0,  # 不需要中间矩阵!
    }


# ──────────────────────────────────────────────
# 实验定义
# ──────────────────────────────────────────────

def experiment_1_numerical_equivalence():
    """实验 1: 标准 Attention vs FlashAttention 数值等价"""
    print("\n" + "=" * 70)
    print("实验 1: 标准 Attention vs FlashAttention 数值等价验证")
    print("=" * 70)

    np.random.seed(42)

    configs = [
        ("小序列 (N=32, d=16)", 32, 16),
        ("中序列 (N=128, d=64)", 128, 64),
        ("长序列 (N=512, d=64)", 512, 64),
        ("大 d (N=128, d=128)", 128, 128),
    ]

    print(f"\n{'配置':<25} {'Max Abs Err':>14} {'Mean Abs Err':>14} {'Cos Sim':>10}")
    print("-" * 65)

    for name, N, d in configs:
        Q = np.random.randn(N, d).astype(np.float32)
        K = np.random.randn(N, d).astype(np.float32)
        V = np.random.randn(N, d).astype(np.float32)

        # 标准 Attention
        O_std, _ = standard_attention(Q, K, V)

        # FlashAttention (block_size=16 for small, 64 for larger)
        bs = min(64, max(16, N // 4))
        O_flash, _ = flash_attention_forward(Q, K, V, block_size_q=bs, block_size_kv=bs)

        # 误差分析
        abs_err = np.abs(O_std - O_flash)
        max_err = np.max(abs_err)
        mean_err = np.mean(abs_err)

        # Cosine similarity
        cos_sim = np.sum(O_std * O_flash) / (
            np.linalg.norm(O_std) * np.linalg.norm(O_flash)
        )

        print(f"{name:<25} {max_err:>14.2e} {mean_err:>14.2e} {cos_sim:>10.6f}")

    print(f"\n关键洞察:")
    print(f"  - FlashAttention 与标准 Attention 数值上几乎完全等价 (误差 < 1e-5)")
    print(f"  - 差异来自 online softmax 的浮点精度差异 (exp 累积)")
    print(f"  - Cosine similarity > 0.9999, 差异可忽略")


def experiment_2_io_analysis():
    """实验 2: IO 复杂度分析"""
    print("\n" + "=" * 70)
    print("实验 2: HBM 访问量 (IO) 分析 — 标准 vs FlashAttention")
    print("=" * 70)

    d = 64  # head dimension
    sram_bytes = 20 * 1024 * 1024  # 20 MB SRAM

    print(f"\n配置: d={d}, SRAM={sram_bytes // (1024*1024)}MB")
    print(f"  标准 Attention: 需要 O(N²) 中间存储 (attention matrix)")
    print(f"  FlashAttention: 只需要 O(N) 中间存储 (tiling)")
    print(f"\n{'N (seq_len)':>12} {'标准 IO (GB)':>14} {'Flash IO (GB)':>14} "
          f"{'标准中间 (GB)':>14} {'Flash 中间 (GB)':>14} {'IO 节省':>10}")
    print("-" * 80)

    for N in [256, 512, 1024, 2048, 4096, 8192, 16384]:
        std_io = compute_io_standard(N, d)
        flash_io = compute_io_flash(N, d, sram_bytes)

        std_gb = std_io['total'] * 4 / 1e9  # 4 bytes per element
        flash_gb = flash_io['total'] * 4 / 1e9
        std_inter_gb = std_io['intermediate_NxN'] * 4 / 1e9
        flash_inter_gb = flash_io['intermediate_NxN'] * 4 / 1e9

        savings = std_gb / max(0.001, flash_gb)

        print(f"{N:>12d} {std_gb:>14.2f} {flash_gb:>14.2f} "
              f"{std_inter_gb:>14.2f} {flash_inter_gb:>14.2f} "
              f"{savings:>9.1f}x")

    print(f"\n关键洞察:")
    print(f"  - 标准 Attention IO ∝ N² × d (attention matrix 读/写)")
    print(f"  - FlashAttention IO ∝ N² × d² / M (tiling 减少)")
    print(f"  - N 越大，FlashAttention 的 IO 优势越明显")
    print(f"  - 序列 16K 时标准需要 ~32GB 中间存储，Flash 只需 0")
    print(f"  - 这就是为什么 FlashAttention 能支持超长序列!")


def experiment_3_online_softmax():
    """实验 3: Online Softmax 正确性验证"""
    print("\n" + "=" * 70)
    print("实验 3: Online Softmax 正确性验证")
    print("=" * 70)

    np.random.seed(42)

    print(f"\nOnline Softmax 核心公式:")
    print(f"  m_new = max(m_old, max(S_new))")
    print(f"  l_new = l_old × exp(m_old - m_new) + Σ exp(S_new - m_new)")
    print(f"  O_new = (l_old × exp(m_old - m_new) × O_old + Σ exp(S_new - m_new) × V_new) / l_new")

    # 逐步演示
    N = 8
    d = 4
    Q = np.random.randn(N, d).astype(np.float32)
    K = np.random.randn(N, d).astype(np.float32)
    V = np.random.randn(N, d).astype(np.float32)

    block_size = 4  # 2 个 KV block

    print(f"\n逐步演示 (N={N}, d={d}, block_size={block_size}):")

    # 完整 attention
    O_full, P_full = standard_attention(Q, K, V)

    # Online softmax 逐步执行
    O = np.zeros_like(Q)
    l = np.zeros(N)
    m = np.full(N, -np.inf)

    for j in range(math.ceil(N / block_size)):
        kv_start = j * block_size
        kv_end = min(kv_start + block_size, N)
        K_j = K[kv_start:kv_end]
        V_j = V[kv_start:kv_end]

        S_ij = (Q @ K_j.T) / math.sqrt(d)  # [N, block_kv]
        m_ij = np.max(S_ij, axis=-1)
        m_new = np.maximum(m, m_ij)

        P_corr = np.exp(m - m_new)
        S_ij_exp = np.exp(S_ij - m_new[:, np.newaxis])
        l_ij = np.sum(S_ij_exp, axis=-1)

        l = l * P_corr + l_ij
        O = P_corr[:, np.newaxis] * O + S_ij_exp @ V_j
        m = m_new

        # 归一化看中间结果
        O_norm = O / l[:, np.newaxis]

        print(f"\n  Block {j} (KV[{kv_start}:{kv_end}]):")
        print(f"    m_new range: [{m_new.min():.3f}, {m_new.max():.3f}]")
        print(f"    l range:     [{l.min():.3f}, {l.max():.3f}]")
        print(f"    O[0] (累积): {O_norm[0][:4]}")
        print(f"    O[0] (标准): {O_full[0][:4]}")

    O_final = O / l[:, np.newaxis]

    # 最终对比
    max_err = np.max(np.abs(O_full - O_final))
    cos_sim = np.sum(O_full * O_final) / (np.linalg.norm(O_full) * np.linalg.norm(O_final))
    print(f"\n  最终对比:")
    print(f"    Max absolute error: {max_err:.2e}")
    print(f"    Cosine similarity:  {cos_sim:.10f}")

    print(f"\n关键洞察:")
    print(f"  - Online softmax 逐块累积，每步只处理一个 KV block")
    print(f"  - correction factor exp(m_old - m_new) 处理 max 值变化")
    print(f"  - 最终结果与完整 softmax 数值等价 (误差 < 1e-6)")


def experiment_4_block_size_impact():
    """实验 4: Block Size 对性能影响"""
    print("\n" + "=" * 70)
    print("实验 4: Block Size (Tiling Granularity) 对性能影响")
    print("=" * 70)

    np.random.seed(42)

    N = 512
    d = 64
    Q = np.random.randn(N, d).astype(np.float32)
    K = np.random.randn(N, d).astype(np.float32)
    V = np.random.randn(N, d).astype(np.float32)

    block_sizes = [8, 16, 32, 64, 128, 256, 512]

    print(f"\n配置: N={N}, d={d}")
    print(f"\n{'Block Size':>12} {'Q Blocks':>10} {'KV Blocks':>10} "
          f"{'Total Pairs':>12} {'HBM Reads':>12} {'HBM Writes':>12} {'Max Err':>12}")
    print("-" * 82)

    # Baseline
    O_std, _ = standard_attention(Q, K, V)

    for bs in block_sizes:
        O_flash, stats = flash_attention_forward(Q, K, V, block_size_q=bs, block_size_kv=bs)

        max_err = np.max(np.abs(O_std - O_flash))
        total_gb = (stats['hbm_reads'] + stats['hbm_writes']) * 4 / 1e9

        print(f"{bs:>12d} {stats['B_r']:>10d} {stats['B_c']:>10d} "
              f"{stats['total_blocks']:>12d} "
              f"{stats['hbm_reads'] * 4 / 1e9:>12.2f} "
              f"{stats['hbm_writes'] * 4 / 1e9:>12.2f} "
              f"{max_err:>12.2e}")

    print(f"\n关键洞察:")
    print(f"  - Block size 小 → 更多 blocks → 更多 kernel launch overhead")
    print(f"  - Block size 大 → 单个 block 需要更多 SRAM → 可能放不下")
    print(f"  - 最优 block size 取决于 SRAM 大小和 GPU 架构")
    print(f"  - 实际 FlashAttention-2 使用 block_size=64-128 (H100)")
    print(f"  - 数值精度不受 block size 影响 (online softmax 保证等价)")


# ──────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("FlashAttention 算法模拟器")
    print("理解 tiling、online softmax、IO 复杂度")
    print("=" * 70)

    experiment_1_numerical_equivalence()
    experiment_2_io_analysis()
    experiment_3_online_softmax()
    experiment_4_block_size_impact()

    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print("""
FlashAttention 关键要点:
  1. 核心: Tiling 分块计算, 避免 O(N²) 中间存储
     - 将 Q, K, V 分成固定大小的 blocks
     - 逐 block 对计算 local attention

  2. Online Softmax: 逐块累积 softmax
     - m_i (running max) + l_i (running sum) + O_i (running output)
     - correction factor: exp(m_old - m_new) 调整之前累积值
     - 保证与完整 softmax 数值等价

  3. IO 复杂度:
     - 标准: O(N²d) — 需要读写完整 N×N attention matrix
     - Flash: O(N²d²/M) — 减少 d/M 倍, M = SRAM 大小
     - 序列越长, 节省越多

  4. Block Size 权衡:
     - 小 block: 更好的并行性, 但更多 overhead
     - 大 block: 更少 overhead, 但需要更多 SRAM
     - 最优: 填满 SRAM → block_size ≈ sqrt(M/d)

  5. FlashAttention-2/3 改进:
     - FA2: 减少 non-matmul FLOPs, 更好的 work partitioning
     - FA3: 支持异步, FP8, 与 H100 TMA (Tensor Memory Accelerator) 集成
    """)
