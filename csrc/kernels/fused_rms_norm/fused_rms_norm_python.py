"""
Fused RMSNorm CUDA C++ Extension — Python Interface (v6)

v6: Forward returns (output, inv_rms), backward uses saved inv_rms
→ backward goes from 3-pass to 2-pass, ~30% backward speedup.
"""

import torch

try:
    from fused_rms_norm._C import (
        fused_rms_norm_add_forward as _fused_add_fwd,
        fused_rms_norm_forward as _fused_fwd,
        fused_rms_norm_add_backward as _fused_add_bwd,
        fused_rms_norm_backward as _fused_bwd,
    )
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False


class FusedRMSNormAddFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, residual, weight, epsilon):
        if CUDA_AVAILABLE:
            output, inv_rms = _fused_add_fwd(input, residual, weight, epsilon)
        else:
            variance = input.pow(2).mean(dim=-1, keepdim=True)
            inv_rms = torch.rsqrt(variance + epsilon).squeeze(-1)  # [B]
            x_norm = input * inv_rms.unsqueeze(-1)
            output = x_norm * weight + residual

        ctx.save_for_backward(input, residual, weight, inv_rms)
        ctx.epsilon = epsilon
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, residual, weight, inv_rms = ctx.saved_tensors
        epsilon = ctx.epsilon

        if CUDA_AVAILABLE:
            grads = _fused_add_bwd(grad_output, input, residual, weight, inv_rms, epsilon)
            return grads[0], grads[1], grads[2], None
        else:
            # Python fallback backward (uses saved inv_rms, no recomputation)
            x_norm = input * inv_rms.unsqueeze(-1)

            grad_residual = grad_output
            grad_weight = (grad_output * x_norm).sum(dim=0)

            hidden_size = input.size(-1)
            dx_norm = grad_output * weight
            dot = (dx_norm * x_norm).sum(dim=-1, keepdim=True)
            grad_input = inv_rms.unsqueeze(-1) * (dx_norm - x_norm * dot / hidden_size)

            return grad_input, grad_residual, grad_weight, None


class FusedRMSNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weight, epsilon):
        if CUDA_AVAILABLE:
            output, inv_rms = _fused_fwd(input, weight, epsilon)
        else:
            variance = input.pow(2).mean(dim=-1, keepdim=True)
            inv_rms = torch.rsqrt(variance + epsilon).squeeze(-1)
            x_norm = input * inv_rms.unsqueeze(-1)
            output = x_norm * weight

        ctx.save_for_backward(input, weight, inv_rms)
        ctx.epsilon = epsilon
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, inv_rms = ctx.saved_tensors
        epsilon = ctx.epsilon

        if CUDA_AVAILABLE:
            grads = _fused_bwd(grad_output, input, weight, inv_rms, epsilon)
            return grads[0], grads[1], None
        else:
            x_norm = input * inv_rms.unsqueeze(-1)
            grad_weight = (grad_output * x_norm).sum(dim=0)

            hidden_size = input.size(-1)
            dx_norm = grad_output * weight
            dot = (dx_norm * x_norm).sum(dim=-1, keepdim=True)
            grad_input = inv_rms.unsqueeze(-1) * (dx_norm - x_norm * dot / hidden_size)

            return grad_input, grad_weight, None


def fused_rms_norm(input, weight, epsilon=1e-6):
    return FusedRMSNormFunction.apply(input, weight, epsilon)


def fused_rms_norm_add(input, residual, weight, epsilon=1e-6):
    return FusedRMSNormAddFunction.apply(input, residual, weight, epsilon)