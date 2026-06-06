/*
 * Fused RMSNorm + Residual Add — C++ / pybind11 Binding (v2)
 *
 * Forward-only for now; backward uses Python autograd.
 */

#include <torch/extension.h>

// Forward declarations of CUDA functions
torch::Tensor fused_rms_norm_add_forward(
    torch::Tensor input,
    torch::Tensor residual,
    torch::Tensor weight,
    float epsilon);

torch::Tensor fused_rms_norm_forward(
    torch::Tensor input,
    torch::Tensor weight,
    float epsilon);

// pybind11 module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_rms_norm_add_forward", &fused_rms_norm_add_forward,
          "Fused RMSNorm + Residual Add forward (CUDA)");
    m.def("fused_rms_norm_forward", &fused_rms_norm_forward,
          "Fused RMSNorm forward (CUDA)");
}