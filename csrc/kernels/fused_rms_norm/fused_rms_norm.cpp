/*
 * Fused RMSNorm + Residual Add — C++ / pybind11 Binding (v3)
 *
 * Forward + backward functions bindings.
 */

#include <torch/extension.h>

#include <tuple>

// Forward declarations
torch::Tensor fused_rms_norm_add_forward(
    torch::Tensor input,
    torch::Tensor residual,
    torch::Tensor weight,
    float epsilon);

torch::Tensor fused_rms_norm_forward(
    torch::Tensor input,
    torch::Tensor weight,
    float epsilon);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> fused_rms_norm_add_backward(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor residual,
    torch::Tensor weight,
    float epsilon);

std::tuple<torch::Tensor, torch::Tensor> fused_rms_norm_backward(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor weight,
    float epsilon);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_rms_norm_add_forward", &fused_rms_norm_add_forward,
          "Fused RMSNorm + Residual Add forward (CUDA)");
    m.def("fused_rms_norm_forward", &fused_rms_norm_forward,
          "Fused RMSNorm forward (CUDA)");
    m.def("fused_rms_norm_add_backward", &fused_rms_norm_add_backward,
          "Fused RMSNorm + Residual Add backward (CUDA)");
    m.def("fused_rms_norm_backward", &fused_rms_norm_backward,
          "Fused RMSNorm backward (CUDA)");
}