/*
 * Fused RMSNorm + Residual Add — CUDA Kernel Implementation (v5)
 *
 * Supports: FP32, FP16 (__half), and BF16 (__nv_bfloat16)
 * All computation in FP32 internally for accuracy.
 *
 * Forward Kernels:
 *   Fused RMSNorm + Residual Add: y = x_norm * weight + residual
 *   Fused RMSNorm (no residual):  y = x_norm * weight
 *   where x_norm = x * inv_rms(x), inv_rms = 1/sqrt(mean(x^2) + eps)
 *
 * Backward Kernels (v1):
 *   d_residual = dy (direct pass-through)
 *   d_weight   = sum_batch(dy * x_norm)
 *   d_input    = inv_rms * (dy * w - x_norm * mean(dy * w * x_norm))
 *
 * Design:
 *   - 1 warp (32 threads) per row
 *   - Warp XOR butterfly reduction (no shmem, no bank conflict)
 *   - Backward: 3-pass per row + atomicAdd for d_weight cross-row accumulation
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
// Forward: FP32 Kernels
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
        o_row[col] = norm_x * weight[col] + r_row[col];
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
// Forward: FP16 Kernels (__half input, FP32 compute)
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
// Forward: BF16 Kernels (__nv_bfloat16 input, FP32 compute)
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
// Backward: FP32 Kernels
// ============================================================================

__global__ void fused_rms_norm_add_bwd_kernel_fp32(
    float* __restrict__ grad_input,
    float* __restrict__ grad_residual,
    float* __restrict__ grad_weight,
    const float* __restrict__ grad_output,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const int batch_size,
    const int hidden_size,
    const float epsilon) {

    int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    int lane = threadIdx.x % 32;
    if (row >= batch_size) return;

    const float* dy_row = grad_output + row * hidden_size;
    const float* x_row = input + row * hidden_size;
    float* dx_row = grad_input + row * hidden_size;
    float* dr_row = grad_residual + row * hidden_size;

    // Pass 1: Compute inv_rms
    float sum_sq = 0.0f;
    for (int col = lane; col < hidden_size; col += 32) {
        float x_val = x_row[col];
        sum_sq += x_val * x_val;
    }
    float variance = warp_reduce_sum(sum_sq);
    float inv_rms = 1.0f / sqrtf(variance / (float)hidden_size + epsilon);

    // Pass 2: Accumulate dot = sum(dy*w*x_norm), dr=dy, partial dw
    float dot = 0.0f;
    for (int col = lane; col < hidden_size; col += 32) {
        float dy_val = dy_row[col];
        float x_val = x_row[col];
        float w_val = weight[col];
        float x_norm_val = x_val * inv_rms;

        dot += dy_val * w_val * x_norm_val;
        dr_row[col] = dy_val;
        atomicAdd(&grad_weight[col], dy_val * x_norm_val);
    }
    float row_dot = warp_reduce_sum(dot);
    float coeff = row_dot / (float)hidden_size;

    // Pass 3: Compute dx = inv_rms * (dy*w - x_norm * coeff)
    for (int col = lane; col < hidden_size; col += 32) {
        float dy_val = dy_row[col];
        float x_val = x_row[col];
        float w_val = weight[col];
        float x_norm_val = x_val * inv_rms;
        dx_row[col] = inv_rms * (dy_val * w_val - x_norm_val * coeff);
    }
}

__global__ void fused_rms_norm_bwd_kernel_fp32(
    float* __restrict__ grad_input,
    float* __restrict__ grad_weight,
    const float* __restrict__ grad_output,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const int batch_size,
    const int hidden_size,
    const float epsilon) {

    int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    int lane = threadIdx.x % 32;
    if (row >= batch_size) return;

    const float* dy_row = grad_output + row * hidden_size;
    const float* x_row = input + row * hidden_size;
    float* dx_row = grad_input + row * hidden_size;

    float sum_sq = 0.0f;
    for (int col = lane; col < hidden_size; col += 32) {
        sum_sq += x_row[col] * x_row[col];
    }
    float variance = warp_reduce_sum(sum_sq);
    float inv_rms = 1.0f / sqrtf(variance / (float)hidden_size + epsilon);

    float dot = 0.0f;
    for (int col = lane; col < hidden_size; col += 32) {
        float dy_val = dy_row[col];
        float x_val = x_row[col];
        float w_val = weight[col];
        float x_norm_val = x_val * inv_rms;

        dot += dy_val * w_val * x_norm_val;
        atomicAdd(&grad_weight[col], dy_val * x_norm_val);
    }
    float row_dot = warp_reduce_sum(dot);
    float coeff = row_dot / (float)hidden_size;

    for (int col = lane; col < hidden_size; col += 32) {
        float dy_val = dy_row[col];
        float x_val = x_row[col];
        float w_val = weight[col];
        float x_norm_val = x_val * inv_rms;
        dx_row[col] = inv_rms * (dy_val * w_val - x_norm_val * coeff);
    }
}

// ============================================================================
// Backward: FP16 Kernels (__half input, FP32 compute)
// ============================================================================

__global__ void fused_rms_norm_add_bwd_kernel_fp16(
    __half* __restrict__ grad_input,
    __half* __restrict__ grad_residual,
    float* __restrict__ grad_weight,
    const __half* __restrict__ grad_output,
    const __half* __restrict__ input,
    const __half* __restrict__ weight,
    const int batch_size,
    const int hidden_size,
    const float epsilon) {

    int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    int lane = threadIdx.x % 32;
    if (row >= batch_size) return;

    const __half* dy_row = grad_output + row * hidden_size;
    const __half* x_row = input + row * hidden_size;
    __half* dx_row = grad_input + row * hidden_size;
    __half* dr_row = grad_residual + row * hidden_size;

    float sum_sq = 0.0f;
    for (int col = lane; col < hidden_size; col += 32) {
        float x_val = __half2float(x_row[col]);
        sum_sq += x_val * x_val;
    }
    float variance = warp_reduce_sum(sum_sq);
    float inv_rms = 1.0f / sqrtf(variance / (float)hidden_size + epsilon);

    float dot = 0.0f;
    for (int col = lane; col < hidden_size; col += 32) {
        float dy_val = __half2float(dy_row[col]);
        float x_val = __half2float(x_row[col]);
        float w_val = __half2float(weight[col]);
        float x_norm_val = x_val * inv_rms;

        dot += dy_val * w_val * x_norm_val;
        dr_row[col] = __float2half(dy_val);
        atomicAdd(&grad_weight[col], dy_val * x_norm_val);
    }
    float row_dot = warp_reduce_sum(dot);
    float coeff = row_dot / (float)hidden_size;

    for (int col = lane; col < hidden_size; col += 32) {
        float dy_val = __half2float(dy_row[col]);
        float x_val = __half2float(x_row[col]);
        float w_val = __half2float(weight[col]);
        float x_norm_val = x_val * inv_rms;
        dx_row[col] = __float2half(inv_rms * (dy_val * w_val - x_norm_val * coeff));
    }
}

__global__ void fused_rms_norm_bwd_kernel_fp16(
    __half* __restrict__ grad_input,
    float* __restrict__ grad_weight,
    const __half* __restrict__ grad_output,
    const __half* __restrict__ input,
    const __half* __restrict__ weight,
    const int batch_size,
    const int hidden_size,
    const float epsilon) {

    int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    int lane = threadIdx.x % 32;
    if (row >= batch_size) return;

    const __half* dy_row = grad_output + row * hidden_size;
    const __half* x_row = input + row * hidden_size;
    __half* dx_row = grad_input + row * hidden_size;

    float sum_sq = 0.0f;
    for (int col = lane; col < hidden_size; col += 32) {
        float x_val = __half2float(x_row[col]);
        sum_sq += x_val * x_val;
    }
    float variance = warp_reduce_sum(sum_sq);
    float inv_rms = 1.0f / sqrtf(variance / (float)hidden_size + epsilon);

    float dot = 0.0f;
    for (int col = lane; col < hidden_size; col += 32) {
        float dy_val = __half2float(dy_row[col]);
        float x_val = __half2float(x_row[col]);
        float w_val = __half2float(weight[col]);
        float x_norm_val = x_val * inv_rms;

        dot += dy_val * w_val * x_norm_val;
        atomicAdd(&grad_weight[col], dy_val * x_norm_val);
    }
    float row_dot = warp_reduce_sum(dot);
    float coeff = row_dot / (float)hidden_size;

    for (int col = lane; col < hidden_size; col += 32) {
        float dy_val = __half2float(dy_row[col]);
        float x_val = __half2float(x_row[col]);
        float w_val = __half2float(weight[col]);
        float x_norm_val = x_val * inv_rms;
        dx_row[col] = __float2half(inv_rms * (dy_val * w_val - x_norm_val * coeff));
    }
}

// ============================================================================
// Backward: BF16 Kernels (__nv_bfloat16 input, FP32 compute)
// ============================================================================

__global__ void fused_rms_norm_add_bwd_kernel_bf16(
    __nv_bfloat16* __restrict__ grad_input,
    __nv_bfloat16* __restrict__ grad_residual,
    float* __restrict__ grad_weight,
    const __nv_bfloat16* __restrict__ grad_output,
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    const int batch_size,
    const int hidden_size,
    const float epsilon) {

    int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    int lane = threadIdx.x % 32;
    if (row >= batch_size) return;

    const __nv_bfloat16* dy_row = grad_output + row * hidden_size;
    const __nv_bfloat16* x_row = input + row * hidden_size;
    __nv_bfloat16* dx_row = grad_input + row * hidden_size;
    __nv_bfloat16* dr_row = grad_residual + row * hidden_size;

    float sum_sq = 0.0f;
    for (int col = lane; col < hidden_size; col += 32) {
        float x_val = __bfloat162float(x_row[col]);
        sum_sq += x_val * x_val;
    }
    float variance = warp_reduce_sum(sum_sq);
    float inv_rms = 1.0f / sqrtf(variance / (float)hidden_size + epsilon);

    float dot = 0.0f;
    for (int col = lane; col < hidden_size; col += 32) {
        float dy_val = __bfloat162float(dy_row[col]);
        float x_val = __bfloat162float(x_row[col]);
        float w_val = __bfloat162float(weight[col]);
        float x_norm_val = x_val * inv_rms;

        dot += dy_val * w_val * x_norm_val;
        dr_row[col] = __float2bfloat16(dy_val);
        atomicAdd(&grad_weight[col], dy_val * x_norm_val);
    }
    float row_dot = warp_reduce_sum(dot);
    float coeff = row_dot / (float)hidden_size;

    for (int col = lane; col < hidden_size; col += 32) {
        float dy_val = __bfloat162float(dy_row[col]);
        float x_val = __bfloat162float(x_row[col]);
        float w_val = __bfloat162float(weight[col]);
        float x_norm_val = x_val * inv_rms;
        dx_row[col] = __float2bfloat16(inv_rms * (dy_val * w_val - x_norm_val * coeff));
    }
}

__global__ void fused_rms_norm_bwd_kernel_bf16(
    __nv_bfloat16* __restrict__ grad_input,
    float* __restrict__ grad_weight,
    const __nv_bfloat16* __restrict__ grad_output,
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    const int batch_size,
    const int hidden_size,
    const float epsilon) {

    int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    int lane = threadIdx.x % 32;
    if (row >= batch_size) return;

    const __nv_bfloat16* dy_row = grad_output + row * hidden_size;
    const __nv_bfloat16* x_row = input + row * hidden_size;
    __nv_bfloat16* dx_row = grad_input + row * hidden_size;

    float sum_sq = 0.0f;
    for (int col = lane; col < hidden_size; col += 32) {
        float x_val = __bfloat162float(x_row[col]);
        sum_sq += x_val * x_val;
    }
    float variance = warp_reduce_sum(sum_sq);
    float inv_rms = 1.0f / sqrtf(variance / (float)hidden_size + epsilon);

    float dot = 0.0f;
    for (int col = lane; col < hidden_size; col += 32) {
        float dy_val = __bfloat162float(dy_row[col]);
        float x_val = __bfloat162float(x_row[col]);
        float w_val = __bfloat162float(weight[col]);
        float x_norm_val = x_val * inv_rms;

        dot += dy_val * w_val * x_norm_val;
        atomicAdd(&grad_weight[col], dy_val * x_norm_val);
    }
    float row_dot = warp_reduce_sum(dot);
    float coeff = row_dot / (float)hidden_size;

    for (int col = lane; col < hidden_size; col += 32) {
        float dy_val = __bfloat162float(dy_row[col]);
        float x_val = __bfloat162float(x_row[col]);
        float w_val = __bfloat162float(weight[col]);
        float x_norm_val = x_val * inv_rms;
        dx_row[col] = __float2bfloat16(inv_rms * (dy_val * w_val - x_norm_val * coeff));
    }
}

// ============================================================================
// Host Functions — Forward
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

// ============================================================================
// Host Functions — Backward
// ============================================================================

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> fused_rms_norm_add_backward(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor residual,
    torch::Tensor weight,
    float epsilon) {

    TORCH_CHECK(grad_output.is_cuda(), "grad_output must be a CUDA tensor");
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(residual.is_cuda(), "residual must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "input must be 2D [batch, hidden]");
    TORCH_CHECK(grad_output.sizes() == input.sizes(), "grad_output shape mismatch");
    TORCH_CHECK(residual.sizes() == input.sizes(), "residual shape mismatch");
    TORCH_CHECK(input.size(1) == weight.size(0), "hidden size mismatch");

    auto batch_size = input.size(0);
    auto hidden_size = input.size(1);

    auto grad_input = torch::empty_like(input);
    auto grad_residual = torch::empty_like(residual);
    // grad_weight: FP32 accumulation across batch, then convert to input dtype
    auto grad_weight = torch::zeros({hidden_size}, weight.options().dtype(at::ScalarType::Float));

    int threads_per_block = 128;
    int blocks = (batch_size * 32 + threads_per_block - 1) / threads_per_block;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    auto dtype = input.scalar_type();

    if (dtype == at::ScalarType::Float) {
        fused_rms_norm_add_bwd_kernel_fp32<<<blocks, threads_per_block, 0, stream>>>(
            grad_input.data_ptr<float>(),
            grad_residual.data_ptr<float>(),
            grad_weight.data_ptr<float>(),
            grad_output.data_ptr<float>(),
            input.data_ptr<float>(),
            weight.data_ptr<float>(),
            batch_size, hidden_size, epsilon);
    } else if (dtype == at::ScalarType::Half) {
        fused_rms_norm_add_bwd_kernel_fp16<<<blocks, threads_per_block, 0, stream>>>(
            reinterpret_cast<__half*>(grad_input.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(grad_residual.data_ptr<at::Half>()),
            grad_weight.data_ptr<float>(),
            reinterpret_cast<const __half*>(grad_output.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(weight.data_ptr<at::Half>()),
            batch_size, hidden_size, epsilon);
    } else if (dtype == at::ScalarType::BFloat16) {
        fused_rms_norm_add_bwd_kernel_bf16<<<blocks, threads_per_block, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(grad_input.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(grad_residual.data_ptr<at::BFloat16>()),
            grad_weight.data_ptr<float>(),
            reinterpret_cast<const __nv_bfloat16*>(grad_output.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr<at::BFloat16>()),
            batch_size, hidden_size, epsilon);
    } else {
        TORCH_CHECK(false, "Unsupported dtype: only FP32/FP16/BF16 supported");
    }

    // Convert grad_weight back to input dtype for autograd compatibility
    if (dtype != at::ScalarType::Float) {
        grad_weight = grad_weight.to(dtype);
    }

    return std::make_tuple(grad_input, grad_residual, grad_weight);
}

std::tuple<torch::Tensor, torch::Tensor> fused_rms_norm_backward(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor weight,
    float epsilon) {

    TORCH_CHECK(grad_output.is_cuda(), "grad_output must be a CUDA tensor");
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "input must be 2D [batch, hidden]");
    TORCH_CHECK(grad_output.sizes() == input.sizes(), "grad_output shape mismatch");
    TORCH_CHECK(input.size(1) == weight.size(0), "hidden size mismatch");

    auto batch_size = input.size(0);
    auto hidden_size = input.size(1);

    auto grad_input = torch::empty_like(input);
    auto grad_weight = torch::zeros({hidden_size}, weight.options().dtype(at::ScalarType::Float));

    int threads_per_block = 128;
    int blocks = (batch_size * 32 + threads_per_block - 1) / threads_per_block;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    auto dtype = input.scalar_type();

    if (dtype == at::ScalarType::Float) {
        fused_rms_norm_bwd_kernel_fp32<<<blocks, threads_per_block, 0, stream>>>(
            grad_input.data_ptr<float>(),
            grad_weight.data_ptr<float>(),
            grad_output.data_ptr<float>(),
            input.data_ptr<float>(),
            weight.data_ptr<float>(),
            batch_size, hidden_size, epsilon);
    } else if (dtype == at::ScalarType::Half) {
        fused_rms_norm_bwd_kernel_fp16<<<blocks, threads_per_block, 0, stream>>>(
            reinterpret_cast<__half*>(grad_input.data_ptr<at::Half>()),
            grad_weight.data_ptr<float>(),
            reinterpret_cast<const __half*>(grad_output.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(weight.data_ptr<at::Half>()),
            batch_size, hidden_size, epsilon);
    } else if (dtype == at::ScalarType::BFloat16) {
        fused_rms_norm_bwd_kernel_bf16<<<blocks, threads_per_block, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(grad_input.data_ptr<at::BFloat16>()),
            grad_weight.data_ptr<float>(),
            reinterpret_cast<const __nv_bfloat16*>(grad_output.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr<at::BFloat16>()),
            batch_size, hidden_size, epsilon);
    } else {
        TORCH_CHECK(false, "Unsupported dtype: only FP32/FP16/BF16 supported");
    }

    if (dtype != at::ScalarType::Float) {
        grad_weight = grad_weight.to(dtype);
    }

    return std::make_tuple(grad_input, grad_weight);
}
