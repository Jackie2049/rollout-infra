/*
 * Fused RMSNorm + Residual Add — C++ / pybind11 Binding (v6)
 *
 * Forward returns tuple (output, inv_rms), backward takes inv_rms as input.
 */

#include <torch/extension.h>
#include <tuple>

torch::Tensor fused_rms_norm_add_forward(torch::Tensor input, torch::Tensor residual, torch::Tensor weight, float epsilon);
torch::Tensor fused_rms_norm_forward(torch::Tensor input, torch::Tensor weight, float epsilon);

std::tuple<torch::Tensor, torch::Tensor> fused_rms_norm_add_forward_v6(torch::Tensor input, torch::Tensor residual, torch::Tensor weight, float epsilon);
std::tuple<torch::Tensor, torch::Tensor> fused_rms_norm_forward_v6(torch::Tensor input, torch::Tensor weight, float epsilon);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> fused_rms_norm_add_backward(torch::Tensor grad_output, torch::Tensor input, torch::Tensor residual, torch::Tensor weight, torch::Tensor inv_rms, float epsilon);
std::tuple<torch::Tensor, torch::Tensor> fused_rms_norm_backward(torch::Tensor grad_output, torch::Tensor input, torch::Tensor weight, torch::Tensor inv_rms, float epsilon);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_rms_norm_add_forward", &fused_rms_norm_add_forward_v6,
          "Fused RMSNorm + Residual Add forward (CUDA) → (output, inv_rms)");
    m.def("fused_rms_norm_forward", &fused_rms_norm_forward_v6,
          "Fused RMSNorm forward (CUDA) → (output, inv_rms)");
    m.def("fused_rms_norm_add_backward", &fused_rms_norm_add_backward,
          "Fused RMSNorm + Residual Add backward (CUDA, uses inv_rms)");
    m.def("fused_rms_norm_backward", &fused_rms_norm_backward,
          "Fused RMSNorm backward (CUDA, uses inv_rms)");
}