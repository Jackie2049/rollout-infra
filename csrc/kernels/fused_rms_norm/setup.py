"""
Fused RMSNorm + Residual Add CUDA C++ Extension — Build Configuration

This is the first real CUDA C++ kernel (not Triton) in the rollout-infra project.
It implements a fused kernel that combines:
  1. RMSNorm: x_norm = x / sqrt(mean(x^2) + eps) * weight
  2. Residual Add: y = x_norm + residual

Fusing saves 1 global memory read + 1 global memory write (~50% memory traffic).

Build on RTX 4090:
  python setup.py build_ext --inplace

Benchmark:
  python tools/benchmark_fused_rms_norm.py

Requires: CUDA 12+, PyTorch 2.x+, nvcc with -arch=sm_89 (RTX 4090)
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='fused_rms_norm',
    ext_modules=[
        CUDAExtension(
            name='fused_rms_norm._C',
            sources=[
                'fused_rms_norm.cpp',
                'fused_rms_norm_cuda.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '--use_fast_math', '-arch=sm_89'],
            },
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)