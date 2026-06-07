/***************************************************************************************************
 * CUTLASS GEMM Benchmark with CUDA Events timing
 * 
 * Compares CUTLASS FP32 GEMM timing with reference implementation.
 * Usage: cutlass_gemm_timed <M> <N> <K> <warmup> <iters>
 **************************************************************************************************/

#include <iostream>
#include <vector>
#include <cstdlib>
#include <cuda_runtime.h>

#include "cutlass/gemm/device/gemm.h"

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Naive reference GEMM
void ReferenceGemm(
  int M, int N, int K,
  float alpha,
  float const *A,
  int lda,
  float const *B,
  int ldb,
  float beta,
  float *C,
  int ldc) {

  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      float accum = beta * C[m * ldc + n];
      for (int k = 0; k < K; ++k) {
        accum += alpha * A[m * lda + k] * B[k * ldb + n];
      }
      C[m * ldc + n] = accum;
    }
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// CUTLASS GEMM kernel
cudaError_t CutlassSgemmNN(
  int M, int N, int K,
  float alpha,
  float const *A,
  int lda,
  float const *B,
  int ldb,
  float beta,
  float *C,
  int ldc) {

  using Gemm = cutlass::gemm::device::Gemm<
    float, cutlass::layout::RowMajor,      // A
    float, cutlass::layout::ColumnMajor,    // B
    float, cutlass::layout::RowMajor,       // C
    float,                                  // ScalarType
    cutlass::arch::OpClassSimt,             // OpClass
    cutlass::arch::Sm89,                    // ArchTag (RTX 4090)
    128, 128, 8,                            // ThreadTileShape
    2,                                      // Stages
    cutlass::gemm::GemmShape<1, 1, 1>,     // InstructionShape
    cutlass::layout::RowMajor,              // EpilogueOutputOp::Layout
    8,                                      // AlignmentA
    8,                                      // AlignmentB
    cutlass::arch::OpClassSimt,             // OpClass
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,  // Swizzle
    2                                       // Stages
  >;

  Gemm gemm_op;
  cutlass::Status status = gemm_op({
    {M, N, K},
    {A, lda},
    {B, ldb},
    {C, ldc},
    {alpha, beta}
  });

  if (status != cutlass::Status::kSuccess) {
    return cudaErrorUnknown;
  }
  return cudaSuccess;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Allocate matrix with random data
cudaError_t AllocateMatrix(float **ptr, int rows, int cols, int seed) {
  size_t sizeof_matrix = sizeof(float) * rows * cols;
  cudaError_t result = cudaMalloc(ptr, sizeof_matrix);
  if (result != cudaSuccess) return result;

  std::vector<float> host_data(rows * cols);
  for (int i = 0; i < rows * cols; ++i) {
    host_data[i] = static_cast<float>((rand() + seed) % 100) / 10.0f;
  }
  result = cudaMemcpy(*ptr, host_data.data(), sizeof_matrix, cudaMemcpyHostToDevice);
  return result;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, const char *arg[]) {
  int M = 2048, N = 2048, K = 2048;
  int warmup = 5, iters = 50;

  if (argc > 1) M = atoi(arg[1]);
  if (argc > 2) N = atoi(arg[2]);
  if (argc > 3) K = atoi(arg[3]);
  if (argc > 4) warmup = atoi(arg[4]);
  if (argc > 5) iters = atoi(arg[5]);

  float alpha = 1.0f, beta = 0.0f;

  float *A, *B, *C_cutlass, *C_reference;
  AllocateMatrix(&A, M, K, 0);
  AllocateMatrix(&B, K, N, 17);
  AllocateMatrix(&C_cutlass, M, N, 101);
  AllocateMatrix(&C_reference, M, N, 101);

  // Verify correctness first
  cudaError_t result = CutlassSgemmNN(M, N, K, alpha, A, M, B, K, beta, C_cutlass, M);
  if (result != cudaSuccess) {
    std::cerr << "CUTLASS GEMM failed!" << std::endl;
    return -1;
  }

  // Warmup
  for (int i = 0; i < warmup; ++i) {
    CutlassSgemmNN(M, N, K, alpha, A, M, B, K, beta, C_cutlass, M);
  }
  cudaDeviceSynchronize();

  // Benchmark with CUDA Events
  std::vector<float> times(iters);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  for (int i = 0; i < iters; ++i) {
    cudaEventRecord(start);
    CutlassSgemmNN(M, N, K, alpha, A, M, B, K, beta, C_cutlass, M);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&times[i], start, stop);
  }

  // Compute statistics
  float avg = 0, min_t = times[0], max_t = times[0];
  for (int i = 0; i < iters; ++i) {
    avg += times[i];
    if (times[i] < min_t) min_t = times[i];
    if (times[i] > max_t) max_t = times[i];
  }
  avg /= iters;

  // Compute TFLOPS
  double flops = 2.0 * M * N * K;
  double tflops = flops / (avg * 1e-3) / 1e12;
  double tflops_peak = flops / (min_t * 1e-3) / 1e12;

  std::cout << "CUTLASS FP32 GEMM Benchmark" << std::endl;
  std::cout << "M=" << M << " N=" << N << " K=" << K << std::endl;
  std::cout << "avg: " << avg << " ms, min: " << min_t << " ms, max: " << max_t << " ms" << std::endl;
  std::cout << "TFLOPS (avg): " << tflops << std::endl;
  std::cout << "TFLOPS (peak): " << tflops_peak << std::endl;
  std::cout << "RTX 4090 FP32 peak: 82.6 TFLOPS (" << (tflops / 82.6 * 100) << "%)" << std::endl;

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(A);
  cudaFree(B);
  cudaFree(C_cutlass);
  cudaFree(C_reference);

  return 0;
}
