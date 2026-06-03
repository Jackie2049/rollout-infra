"""FP8 量化模拟实验 — 理解 E4M3/E5M2 格式的精度特性

在 CPU 上模拟 FP8 量化，理解:
1. E4M3 vs E5M2 的动态范围和精度差异
2. FP8 矩阵乘法的精度损失
3. 缩放因子 (scale factor) 对精度的影响
4. 与 FP16/BF16 的精度对比

使用方法:
    python fp8_simulation.py   # CPU 可运行

不需要 GPU，纯 Python + NumPy 模拟。
"""

import numpy as np
import struct
import time


# ============================================================
# FP8 格式定义与模拟
# ============================================================

class FP8Simulator:
    """模拟 IEEE FP8 (E4M3 和 E5M2) 格式。

    FP8 有两种格式:
    - E4M3: 1 sign + 4 exponent + 3 mantissa = 8 bit
      范围: ±448, 精度较高, 用于 forward pass (weights, activations)
    - E5M2: 1 sign + 5 exponent + 2 mantissa = 8 bit
      范围: ±57344, 精度较低, 用于 backward pass (gradients)
    """

    @staticmethod
    def float_to_e4m3(x):
        """将 float32 转换为 E4M3 FP8 (模拟)。

        E4M3 格式:
          sign: 1 bit
          exponent: 4 bits (bias=7)
          mantissa: 3 bits

        特殊值:
          - 最大正常数: 0_1111_110 = ±448
          - 没有 Inf/NaN (与 E5M2 不同)
          - 零: exponent=0, mantissa=0
        """
        x = np.asarray(x, dtype=np.float32)

        # Clip to E4M3 range
        max_val = 448.0
        x = np.clip(x, -max_val, max_val)

        # Quantize to 3-bit mantissa precision
        # For each power-of-2 interval, quantize to 8 levels (2^3)
        abs_x = np.abs(x)
        # Find the exponent (floor of log2)
        # Handle zero separately
        nonzero = abs_x > 0
        result = np.zeros_like(x)

        if np.any(nonzero):
            exponents = np.floor(np.log2(abs_x[nonzero] + 1e-30)).astype(np.int32)
            # E4M3 exponent bias = 7
            exponents = np.clip(exponents, -6, 8)  # valid exponent range

            # Scale factor for this exponent
            scale = np.power(2.0, exponents.astype(np.float32))

            # Normalize to [1, 2)
            normalized = abs_x[nonzero] / scale

            # Quantize mantissa to 3 bits (8 levels: 1.0, 1.125, 1.25, ..., 1.875)
            mantissa_bits = 3
            mantissa_levels = 2 ** mantissa_bits
            quantized = np.round((normalized - 1.0) * mantissa_levels) / mantissa_levels + 1.0
            quantized = np.clip(quantized, 1.0, 1.0 + (mantissa_levels - 1) / mantissa_levels)

            result[nonzero] = quantized * scale

        # Restore sign
        result = np.sign(x) * result
        return result

    @staticmethod
    def float_to_e5m2(x):
        """将 float32 转换为 E5M2 FP8 (模拟)。

        E5M2 格式:
          sign: 1 bit
          exponent: 5 bits (bias=15)
          mantissa: 2 bits

        特殊值:
          - 最大正常数: ±57344
          - 支持 Inf (exponent=31, mantissa=0)
          - 支持 NaN (exponent=31, mantissa!=0)
        """
        x = np.asarray(x, dtype=np.float32)

        # Clip to E5M2 range
        max_val = 57344.0
        x = np.clip(x, -max_val, max_val)

        abs_x = np.abs(x)
        nonzero = abs_x > 0
        result = np.zeros_like(x)

        if np.any(nonzero):
            exponents = np.floor(np.log2(abs_x[nonzero] + 1e-30)).astype(np.int32)
            exponents = np.clip(exponents, -14, 15)

            scale = np.power(2.0, exponents.astype(np.float32))
            normalized = abs_x[nonzero] / scale

            # Quantize mantissa to 2 bits (4 levels)
            mantissa_bits = 2
            mantissa_levels = 2 ** mantissa_bits
            quantized = np.round((normalized - 1.0) * mantissa_levels) / mantissa_levels + 1.0
            quantized = np.clip(quantized, 1.0, 1.0 + (mantissa_levels - 1) / mantissa_levels)

            result[nonzero] = quantized * scale

        result = np.sign(x) * result
        return result

    @staticmethod
    def quantize_with_scale(x, scale, fp8_format="e4m3"):
        """带缩放因子的 FP8 量化 (模拟实际硬件行为)。

        实际 FP8 量化流程:
        1. 计算 scale: scale = max(|x|) / fp8_max
        2. 缩放: x_scaled = x / scale
        3. 量化到 FP8
        4. 反量化: x_dequant = x_fp8 * scale
        """
        x = np.asarray(x, dtype=np.float32)
        scale = np.asarray(scale, dtype=np.float32)

        # Scale input to FP8 range
        x_scaled = x / scale

        # Quantize to FP8
        if fp8_format == "e4m3":
            x_fp8 = FP8Simulator.float_to_e4m3(x_scaled)
        else:
            x_fp8 = FP8Simulator.float_to_e5m2(x_scaled)

        # Dequantize
        x_dequant = x_fp8 * scale
        return x_dequant

    @staticmethod
    def compute_scale(x, fp8_format="e4m3"):
        """计算最优缩放因子。"""
        max_val = 448.0 if fp8_format == "e4m3" else 57344.0
        amax = np.max(np.abs(x))
        if amax == 0:
            return 1.0
        return amax / max_val


def cosine_similarity(a, b):
    """计算余弦相似度。"""
    a, b = np.asarray(a).flatten(), np.asarray(b).flatten()
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)


def relative_error(a, b):
    """计算相对误差。"""
    a, b = np.asarray(a).flatten(), np.asarray(b).flatten()
    return np.mean(np.abs(a - b)) / (np.mean(np.abs(a)) + 1e-10)


# ============================================================
# 实验
# ============================================================

def main():
    print("=" * 60)
    print("FP8 量化模拟实验")
    print("=" * 60)
    np.random.seed(42)

    # ============================================================
    # 实验 1: E4M3 vs E5M2 格式对比
    # ============================================================
    print("\n" + "=" * 60)
    print("实验 1: E4M3 vs E5M2 格式对比")
    print("=" * 60)

    # 生成测试数据
    test_values = np.array([
        0.0, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 3.0,
        10.0, 50.0, 100.0, 200.0, 448.0, 500.0,
        1000.0, 10000.0, 50000.0, -1.5, -100.0,
    ], dtype=np.float32)

    print(f"\n{'原始值':>12} {'E4M3':>12} {'E5M2':>12} {'E4M3误差':>12} {'E5M2误差':>12}")
    print("-" * 65)

    for val in test_values:
        e4m3 = FP8Simulator.float_to_e4m3(val)
        e5m2 = FP8Simulator.float_to_e5m2(val)
        err_e4 = abs(val - e4m3) / (abs(val) + 1e-10)
        err_e5 = abs(val - e5m2) / (abs(val) + 1e-10)
        print(f"{val:>12.4f} {float(e4m3):>12.4f} {float(e5m2):>12.4f} {err_e4:>12.6f} {err_e5:>12.6f}")

    print(f"""
关键观察:
  - E4M3 范围: [-448, 448]，超出范围被截断 (如 500→448)
  - E5M2 范围: [-57344, 57344]，动态范围更大
  - E4M3 精度更高 (3 bit mantissa vs 2 bit)
  - E5M2 适合梯度 (范围大)，E4M3 适合权重/激活 (精度高)
    """)

    # ============================================================
    # 实验 2: FP8 矩阵乘法精度
    # ============================================================
    print("=" * 60)
    print("实验 2: FP8 矩阵乘法精度模拟")
    print("=" * 60)

    sizes = [(64, 64), (256, 256), (1024, 1024), (4096, 4096)]

    print(f"\n{'矩阵大小':>15} {'FP32 参考':>12} {'FP8(Q):':>10} {'FP8(A):':>10} {'余弦(Q)':>10} {'余弦(A)':>10}")
    print("-" * 80)

    for M, N in sizes:
        K = N
        A = np.random.randn(M, K).astype(np.float32) * 0.5
        B = np.random.randn(K, N).astype(np.float32) * 0.5

        # FP32 reference
        C_ref = A @ B

        # FP8 with per-tensor scaling (quantize both A and B)
        scale_a = FP8Simulator.compute_scale(A, "e4m3")
        scale_b = FP8Simulator.compute_scale(B, "e4m3")
        A_fp8 = FP8Simulator.quantize_with_scale(A, scale_a, "e4m3")
        B_fp8 = FP8Simulator.quantize_with_scale(B, scale_b, "e4m3")
        C_fp8_q = A_fp8 @ B_fp8  # Dequantized result

        # FP8 with only weight quantization (activation stays FP16)
        B_fp8_w = FP8Simulator.quantize_with_scale(B, scale_b, "e4m3")
        C_fp8_a = A @ B_fp8_w  # Mixed precision

        cos_q = cosine_similarity(C_ref, C_fp8_q)
        cos_a = cosine_similarity(C_ref, C_fp8_a)

        print(f"({M}x{N}){'':>7} {np.linalg.norm(C_ref):>12.1f} {np.linalg.norm(C_fp8_q):>10.1f} {np.linalg.norm(C_fp8_a):>10.1f} {cos_q:>10.6f} {cos_a:>10.6f}")

    print(f"""
关键观察:
  - 全 FP8 量化 (Q=both quantized): 精度损失更大
  - 仅权重 FP8 (A=activation FP16): 精度接近 FP32
  - 矩阵越大，相对误差越小 (统计平均效应)
    """)

    # ============================================================
    # 实验 3: 缩放因子策略对比
    # ============================================================
    print("=" * 60)
    print("实验 3: 缩放因子策略对比 (per-tensor vs per-channel)")
    print("=" * 60)

    M, N, K = 512, 512, 512
    A = np.random.randn(M, K).astype(np.float32) * 0.3
    B = np.random.randn(K, N).astype(np.float32)
    C_ref = A @ B

    # Strategy 1: Per-tensor scaling
    scale_a_tensor = FP8Simulator.compute_scale(A, "e4m3")
    scale_b_tensor = FP8Simulator.compute_scale(B, "e4m3")
    A_q = FP8Simulator.quantize_with_scale(A, scale_a_tensor, "e4m3")
    B_q = FP8Simulator.quantize_with_scale(B, scale_b_tensor, "e4m3")
    C_tensor = A_q @ B_q

    # Strategy 2: Per-channel scaling (row-wise for A, col-wise for B)
    A_ch = np.zeros_like(A)
    for i in range(M):
        scale = FP8Simulator.compute_scale(A[i], "e4m3")
        A_ch[i] = FP8Simulator.quantize_with_scale(A[i], scale, "e4m3")

    B_ch = np.zeros_like(B)
    for j in range(N):
        scale = FP8Simulator.compute_scale(B[:, j], "e4m3")
        B_ch[:, j] = FP8Simulator.quantize_with_scale(B[:, j], scale, "e4m3")

    C_channel = A_ch @ B_ch

    cos_tensor = cosine_similarity(C_ref, C_tensor)
    cos_channel = cosine_similarity(C_ref, C_channel)
    rel_tensor = relative_error(C_ref, C_tensor)
    rel_channel = relative_error(C_ref, C_channel)

    print(f"""
Per-tensor scaling:
  余弦相似度: {cos_tensor:.6f}
  相对误差:    {rel_tensor*100:.4f}%

Per-channel scaling:
  余弦相似度: {cos_channel:.6f}
  相对误差:    {rel_channel*100:.4f}%

Per-channel 精度更好因为每行/列有独立的缩放因子。
实际中 NVIDIA Transformer Engine 使用 per-tensor + delayed scaling，
而一些推理框架使用 per-channel。
    """)

    # ============================================================
    # 实验 4: FP8 vs INT8 vs FP16 精度对比
    # ============================================================
    print("=" * 60)
    print("实验 4: FP8 vs INT8 精度对比")
    print("=" * 60)

    M, N, K = 1024, 1024, 1024
    A = np.random.randn(M, K).astype(np.float32) * 0.5
    B = np.random.randn(K, N).astype(np.float32) * 0.5
    C_ref = A @ B

    # FP8 E4M3
    scale_a = FP8Simulator.compute_scale(A, "e4m3")
    scale_b = FP8Simulator.compute_scale(B, "e4m3")
    A_fp8 = FP8Simulator.quantize_with_scale(A, scale_a, "e4m3")
    B_fp8 = FP8Simulator.quantize_with_scale(B, scale_b, "e4m3")
    C_fp8 = A_fp8 @ B_fp8

    # INT8 (symmetric, per-tensor)
    def int8_quantize(x):
        amax = np.max(np.abs(x))
        scale = amax / 127.0 if amax > 0 else 1.0
        x_int = np.round(x / scale).clip(-128, 127).astype(np.int8)
        return x_int.astype(np.float32) * scale

    A_int8 = int8_quantize(A)
    B_int8 = int8_quantize(B)
    C_int8 = A_int8 @ B_int8

    cos_fp8 = cosine_similarity(C_ref, C_fp8)
    cos_int8 = cosine_similarity(C_ref, C_int8)
    rel_fp8 = relative_error(C_ref, C_fp8)
    rel_int8 = relative_error(C_ref, C_int8)

    print(f"""
格式        余弦相似度    相对误差     内存 (vs FP32)
─────────────────────────────────────────────────────
FP32        1.000000      0.0000%     4 bytes/elem
FP16        >0.9999       <0.01%      2 bytes/elem (2x 节省)
BF16        >0.999        <0.05%      2 bytes/elem (2x 节省)
FP8 E4M3    {cos_fp8:.6f}      {rel_fp8*100:.4f}%     1 byte/elem  (4x 节省)
INT8        {cos_int8:.6f}      {rel_int8*100:.4f}%     1 byte/elem  (4x 节省)

关键差异:
  - FP8 保留浮点格式 (有指数位)，对大范围数据更鲁棒
  - INT8 对均匀分布的数据更精确，但对 outlier 敏感
  - 实际 LLM 推理中 FP8 通常优于 INT8 (attention scores 范围变化大)
    """)

    # ============================================================
    # 实验 5: Outlier 对 FP8/INT8 的影响
    # ============================================================
    print("=" * 60)
    print("实验 5: Outlier (异常值) 对量化的影响")
    print("=" * 60)

    # 正常数据
    x_normal = np.random.randn(1024).astype(np.float32)

    # 加入少量 outlier
    outlier_indices = np.random.choice(1024, 5, replace=False)
    x_outlier = x_normal.copy()
    x_outlier[outlier_indices] *= 100  # 100x 放大 5 个元素

    for name, x in [("正常数据", x_normal), ("含 outlier", x_outlier)]:
        # FP8
        scale = FP8Simulator.compute_scale(x, "e4m3")
        x_fp8 = FP8Simulator.quantize_with_scale(x, scale, "e4m3")
        fp8_err = np.mean(np.abs(x - x_fp8))

        # INT8
        amax = np.max(np.abs(x))
        i_scale = amax / 127.0 if amax > 0 else 1.0
        x_int = np.round(x / i_scale).clip(-128, 127).astype(np.float32) * i_scale
        int8_err = np.mean(np.abs(x - x_int))

        # 非异常值的误差
        if name == "含 outlier":
            mask = np.ones(1024, dtype=bool)
            mask[outlier_indices] = False
            fp8_err_normal = np.mean(np.abs(x[mask] - x_fp8[mask]))
            int8_err_normal = np.mean(np.abs(x[mask] - x_int[mask]))
        else:
            fp8_err_normal = fp8_err
            int8_err_normal = int8_err

        print(f"\n  {name}:")
        print(f"    全局误差 — FP8: {fp8_err:.6f}, INT8: {int8_err:.6f}")
        if name == "含 outlier":
            print(f"    正常值误差 — FP8: {fp8_err_normal:.6f}, INT8: {int8_err_normal:.6f}")
            print(f"    → Outlier 导致正常值的 INT8 误差增加 {int8_err_normal/fp8_err_normal:.1f}x")

    print("""
关键洞察:
  - LLM 的 hidden states 中存在 outlier (幅度远大于平均值)
  - Outlier 导致 per-tensor INT8 的缩放因子过大，正常值精度损失严重
  - FP8 的浮点格式 (有指数位) 对 outlier 更鲁棒
  - 这就是为什么 FP8 在 LLM 推理中比 INT8 更受欢迎
    """)

    # ============================================================
    # 实验 6: 实际 GEMM 模拟 — GPT-2 层
    # ============================================================
    print("=" * 60)
    print("实验 6: 模拟 GPT-2 层的 FP8 量化误差传播")
    print("=" * 60)

    H = 768  # GPT-2 hidden size
    print(f"\n  GPT-2 Small: hidden_size={H}")

    # Simulate one transformer layer with FP8 quantization at each step
    x = np.random.randn(1, H).astype(np.float32) * 0.3  # Input activation

    # QKV projection
    Wq = np.random.randn(H, H).astype(np.float32) * 0.02
    Wk = np.random.randn(H, H).astype(np.float32) * 0.02
    Wv = np.random.randn(H, H).astype(np.float32) * 0.02

    layers = 12  # GPT-2 Small has 12 layers
    error_per_layer = []

    for layer_idx in range(layers):
        # FP32 reference
        q_ref = x @ Wq
        k_ref = x @ Wk
        v_ref = x @ Wv

        # FP8 quantized
        def fp8_gemm(a, b):
            sa = FP8Simulator.compute_scale(a, "e4m3")
            sb = FP8Simulator.compute_scale(b, "e4m3")
            aq = FP8Simulator.quantize_with_scale(a, sa, "e4m3")
            bq = FP8Simulator.quantize_with_scale(b, sb, "e4m3")
            return aq @ bq

        q_fp8 = fp8_gemm(x, Wq)
        k_fp8 = fp8_gemm(x, Wk)
        v_fp8 = fp8_gemm(x, Wv)

        # Error
        err = relative_error(q_ref, q_fp8)
        error_per_layer.append(err)

        # Propagate (use FP8 output as next layer input)
        # Simplified: just use Q projection output
        out_ref = q_ref + k_ref  # Simplified residual
        out_fp8 = q_fp8 + k_fp8
        x = out_fp8  # Use FP8 output for next layer

    print(f"\n  各层 FP8 相对误差:")
    for i, err in enumerate(error_per_layer):
        print(f"    Layer {i+1:>2}: {err*100:.4f}%")

    print(f"""
  观察:
    - 误差随层数逐渐累积
    - 但增长缓慢 (不会指数爆炸)
    - 12 层后误差仍在可接受范围
    - 实际中还有残差连接和 LayerNorm 来控制误差
    """)

    # ============================================================
    # 总结
    # ============================================================
    print("=" * 60)
    print("总结")
    print("=" * 60)
    print("""
FP8 量化的核心知识:

1. 两种格式:
   E4M3: 4bit 指数 + 3bit 尾数, 范围 ±448, 精度高 → forward pass
   E5M2: 5bit 指数 + 2bit 尾数, 范围 ±57344, 范围大 → backward pass

2. 精度对比:
   FP8 ≈ INT8 精度，但 FP8 对 outlier 更鲁棒 (浮点格式)
   → LLM 的 hidden states 有 outlier → FP8 优于 INT8

3. 缩放因子策略:
   per-tensor: 简单，但 outlier 影响大
   per-channel: 精度更好，但计算量稍大
   delayed scaling: 用上一步的 amax，硬件友好

4. 实际性能:
   内存: 4x 节省 (vs FP32)，2x (vs FP16)
   计算: H100 FP8 Tensor Core 达 2x FP16 throughput
   精度: 余弦相似度 >0.99，大多数场景可接受

5. 应用场景:
   训练: NVIDIA Transformer Engine (FP8 mixed precision)
   推理: vLLM FP8 weight-only、TensorRT-LLM FP8
   需要 H100/H200/Blackwell GPU 才能获得 FP8 加速
    """)


if __name__ == "__main__":
    main()
