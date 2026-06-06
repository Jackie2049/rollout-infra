/*
 * Fused RMSNorm + Residual Add — CUDA Kernel Implementation (v4)
 *
 * Supports: FP32, FP16 (__half), and BF16 (via separate kernels)
 * All computation in FP32 internally for accuracy.
 *
 * Kernel 1: Fused RMSNorm + Residual Add
 *   y = (x / sqrt(mean(x^2) + eps)) * weight + residual
 * Kernel 2: Fused RMSNorm (no residual)
 *   y = (x / sqrt(mean(x^2) + eps)) * weight
 *
 * Design:
 *   - 1 warp (32 threads) per row
 *   - Warp XOR butterfly reduction (no shmem, no bank conflict)
 *   - Fused single pass
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cmath>

// ============================================================================
// Warp-level reduction (5-step butterfly XOR)
// ============================================================================

__device__ __forceinline__ float warp_reduce_sum(float val) {
    val += __shfl_xor_sync(0xffffffff, val, 16);
    val += __shfl_xor_sync(0xffffffff, val, 8);
    val += __shfl_xor_sync(0xffffffff, val, 4);
    val += __shfl_xor_sync(0xffffffff, val, 2);
    val += __shfl_xor_sync(0xffffffff, val, 1);
    return val;
}

// ============================================================================
// FP32 Kernels (unchanged from v3)
// ============================================================================

__global__ void fused_rms_norm_add_fwd_kernel_fp32(
    float* __restrict__ out,
    const float* __restrict__ input,
    const float* __restrict__ residual,
    const float* __restrict__ weight,
    const int batch_size,
    const int hidden_size,
    const float epsilon) {

    int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    int lane = threadIdx.x % 32;
    if (row >= batch_size) return;

    const float* x_row = input + row * hidden_size;
    const float* r_row = residual + row * hidden_size;
    const float* w_row = weight;
    float* o_row = out + row * hidden_size;

    float sum_sq = 0.0f;
    for (int col = lane; col < hidden_size; col += 32) {
        float x_val = x_row[col];
        sum_sq += x_val * x_val;
    }

    float variance = warp_reduce_sum(sum_sq);
    float inv_rms = 1.0f / sqrtf(variance / (float)hidden_size + epsilon);

    for (int col = lane; col < hidden_size; col += 32) {
        float norm_x = x_row[col] * inv_rms;
        o_row[col] = norm_x * w_row[col] + r_row[col];
    }
}

__global__ void fused_rms_norm_fwd_kernel_fp32(
    float* __restrict__ out,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const int batch_size,
    const int hidden_size,
    const float epsilon) {

    int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    int lane = threadIdx.x % 32;
    if (row >= batch_size) return;

    const float* x_row = input + row * hidden_size;
    float* o_row = out + row * hidden_size;

    float sum_sq = 0.0f;
    for (int col = lane; col < hidden_size; col += 32) {
        sum_sq += x_row[col] * x_row[col];
    }

    float variance = warp_reduce_sum(sum_sq);
    float inv_rms = 1.0f / sqrtf(variance / (float)hidden_size + epsilon);

    for (int col = lane; col < hidden_size; col += 32) {
        o_row[col] = x_row[col] * inv_rms * weight[col];
    }
}

// ============================================================================
// FP16 Kernels (__half input, FP32 compute)
// ============================================================================

__global__ void fused_rms_norm_add_fwd_kernel_fp16(
    __half* __restrict__ out,
    const __half* __restrict__ input,
    const __half* __restrict__ residual,
    const __half* __restrict__ weight,
    const int batch_size,
    const int hidden_size,
    const float epsilon) {

    int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    int lane = threadIdx.x % 32;
    if (row >= batch_size) return;

    const __half* x_row = input + row * hidden_size;
    const __half* r_row = residual + row * hidden_size;
    const __half* w_row = weight;
    __half* o_row = out + row * hidden_size;

    // Sum of squares (compute in FP32)
    float sum_sq = 0.0f;
    for (int col = lane; col < hidden_size; col += 32) {
        float x_val = __half2float(x_row[col]);
        sum_sq += x_val * x_val;
    }

    float variance = warp_reduce_sum(sum_sq);
    float inv_rms = 1.0f / sqrtf(variance / (float)hidden_size + epsilon);

    // Fused: norm * weight + residual
    for (int col = lane; col < hidden_size; col += 32) {
        float x_val = __half2float(x_row[col]);
        float w_val = __half2float(w_row[col]);
        float r_val = __half2float(r_row[col]);
        float norm_x = x_val * inv_rms;
        o_row[col] = __float2half(norm_x * w_val + r_val);
    }
}

__global__ void fused_rms_norm_fwd_kernel_fp16(
    __half* __restrict__ out,
    const __half* __restrict__ input,
    const __half* __restrict__ weight,
    const int batch_size,
    const int hidden_size,
    const float epsilon) {

    int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    int lane = threadIdx.x % 32;
    if (row >= batch_size) return;

    const __half* x_row = input + row * hidden_size;
    __half* o_row = out + row * hidden_size;

    float sum_sq = 0.0f;
    for (int col = lane; col < hidden_size; col += 32) {
        float x_val = __half2float(x_row[col]);
        sum_sq += x_val * x_val;
    }

    float variance = warp_reduce_sum(sum_sq);
    float inv_rms = 1.0f / sqrtf(variance / (float)hidden_size + epsilon);

    for (int col = lane; col < hidden_size; col += 32) {
        float x_val = __half2float(x_row[col]);
        float w_val = __half2float(weight[col]);
        o_row[col] = __float2half(x_val * inv_rms * w_val);
    }
}

// ============================================================================
// BF16 Kernels (__nv_bfloat16 input, FP32 compute)
// ============================================================================

__global__ void fused_rms_norm_add_fwd_kernel_bf16(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ weight,
    const int batch_size,
    const int hidden_size,
    const float epsilon) {

    int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    int lane = threadIdx.x % 32;
    if (row >= batch_size) return;

    const __nv_bfloat16* x_row = input + row * hidden_size;
    const __nv_bfloat16* r_row = residual + row * hidden_size;
    const __nv_bfloat16* w_row = weight;
    __nv_bfloat16* o_row = out + row * hidden_size;

    float sum_sq = 0.0f;
    for (int col = lane; col < hidden_size; col += 32) {
        float x_val = __bfloat162float(x_row[col]);
        sum_sq += x_val * x_val;
    }

    float variance = warp_reduce_sum(sum_sq);
    float inv_rms = 1.0f / sqrtf(variance / (float)hidden_size + epsilon);

    for (int col = lane; col < hidden_size; col += 32) {
        float x_val = __bfloat162float(x_row[col]);
        float w_val = __bfloat162float(w_row[col]);
        float r_val = __bfloat162float(r_row[col]);
        float norm_x = x_val * inv_rms;
        o_row[col] = __float2bfloat16(norm_x * w_val + r_val);
    }
}

__global__ void fused_rms_norm_fwd_kernel_bf16(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    const int batch_size,
    const int hidden_size,
    const float epsilon) {

    int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    int lane = threadIdx.x % 32;
    if (row >= batch_size) return;

    const __nv_bfloat16* x_row = input + row * hidden_size;
    __nv_bfloat16* o_row = out + row * hidden_size;

    float sum_sq = 0.0f;
    for (int col = lane; col < hidden_size; col += 32) {
        float x_val = __bfloat162float(x_row[col]);
        sum_sq += x_val * x_val;
    }

    float variance = warp_reduce_sum(sum_sq);
    float inv_rms = 1.0f / sqrtf(variance / (float)hidden_size + epsilon);

    for (int col = lane; col < hidden_size; col += 32) {
        float x_val = __bfloat162float(x_row[col]);
        float w_val = __bfloat162float(weight[col]);
        o_row[col] = __float2bfloat16(x_val * inv_rms * w_val);
    }
}

// ============================================================================
// Host Functions — Kernel Launch with dtype dispatch
// ============================================================================

torch::Tensor fused_rms_norm_add_forward(
    torch::Tensor input,
    torch::Tensor residual,
    torch::Tensor weight,
    float epsilon) {

    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(residual.is_cuda(), "residual must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "input must be 2D [batch, hidden]");
    TORCH_CHECK(residual.sizes() == input.sizes(), "shape mismatch");
    TORCH_CHECK(input.size(1) == weight.size(0), "hidden size mismatch");

    auto batch_size = input.size(0);
    auto hidden_size = input.size(1);
    auto output = torch::empty_like(input);

    int threads_per_block = 128;
    int blocks = (batch_size * 32 + threads_per_block - 1) / threads_per_block;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    auto dtype = input.scalar_type();

    if (dtype == at::ScalarType::Float) {
        fused_rms_norm_add_fwd_kernel_fp32<<<blocks, threads_per_block, 0, stream>>>(
            output.data_ptr<float>(),
            input.data_ptr<float>(),
            residual.data_ptr<float>(),
            weight.data_ptr<float>(),
            batch_size, hidden_size, epsilon);
    } else if (dtype == at::ScalarType::Half) {
        fused_rms_norm_add_fwd_kernel_fp16<<<blocks, threads_per_block, 0, stream>>>(
            reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(residual.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(weight.data_ptr<at::Half>()),
            batch_size, hidden_size, epsilon);
    } else if (dtype == at::ScalarType::BFloat16) {
        fused_rms_norm_add_fwd_kernel_bf16<<<blocks, threads_per_block, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr<at::BFloat16>()),
            batch_size, hidden_size, epsilon);
    } else {
        TORCH_CHECK(false, "Unsupported dtype: only FP32/FP16/BF16 supported");
    }

    return output;
}

torch::Tensor fused_rms_norm_forward(
    torch::Tensor input,
    torch::Tensor weight,
    float epsilon) {

    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "input must be 2D [batch, hidden]");
    TORCH_CHECK(input.size(1) == weight.size(0), "hidden size mismatch");

    auto batch_size = input.size(0);
    auto hidden_size = input.size(1);
    auto output = torch::empty_like(input);

    int threads_per_block = 128;
    int blocks = (batch_size * 32 + threads_per_block - 1) / threads_per_block;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    auto dtype = input.scalar_type();

    if (dtype == at::ScalarType::Float) {
        fused_rms_norm_fwd_kernel_fp32<<<blocks, threads_per_block, 0, stream>>>(
            output.data_ptr<float>(),
            input.data_ptr<float>(),
            weight.data_ptr<float>(),
            batch_size, hidden_size, epsilon);
    } else if (dtype == at::ScalarType::Half) {
        fused_rms_norm_fwd_kernel_fp16<<<blocks, threads_per_block, 0, stream>>>(
            reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(weight.data_ptr<at::Half>()),
            batch_size, hidden_size, epsilon);
    } else if (dtype == at::ScalarType::BFloat16) {
        fused_rms_norm_fwd_kernel_bf16<<<blocks, threads_per_block, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr<at::BFloat16>()),
            batch_size, hidden_size, epsilon);
    } else {
        TORCH_CHECK(false, "Unsupported dtype: only FP32/FP16/BF16 supported");
    }

    return output;
}