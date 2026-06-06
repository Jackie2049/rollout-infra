"""
Fused RMSNorm CUDA C++ Extension — Python Interface

Provides torch.autograd.Function wrapper for seamless integration.
"""

import torch

# Try to import the CUDA extension; fall back to Python implementation
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
    """Fused RMSNorm + Residual Add with autograd support."""

    @staticmethod
    def forward(ctx, input, residual, weight, epsilon):
        if CUDA_AVAILABLE:
            output = _fused_add_fwd(input, residual, weight, epsilon)
        else:
            # Python fallback
            variance = input.pow(2).mean(dim=-1, keepdim=True)
            inv_rms = torch.rsqrt(variance + epsilon)
            x_norm = input * inv_rms
            output = x_norm * weight + residual

        ctx.save_for_backward(input, residual, weight)
        ctx.epsilon = epsilon
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, residual, weight = ctx.saved_tensors
        epsilon = ctx.epsilon

        if CUDA_AVAILABLE:
            grads = _fused_add_bwd(grad_output, input, residual, weight, epsilon)
            return grads[0], grads[1], grads[2], None
        else:
            # Python fallback backward
            variance = input.pow(2).mean(dim=-1, keepdim=True)
            inv_rms = torch.rsqrt(variance + epsilon)
            x_norm = input * inv_rms

            # grad_residual = grad_output (direct pass-through)
            grad_residual = grad_output

            # grad_weight = sum over batch of (grad_output * x_norm)
            grad_weight = (grad_output * x_norm).sum(dim=0)

            # grad_input = grad_output * weight * inv_rms
            #             - x_norm * (grad_output * weight * x_norm).sum(-1) / hidden_size * inv_rms
            #             + grad_output  (from residual)
            hidden_size = input.size(-1)
            dot = (grad_output * weight * x_norm).sum(dim=-1, keepdim=True)
            coeff = dot / hidden_size * inv_rms

            grad_input = grad_output * weight * inv_rms + grad_output - x_norm * coeff

            return grad_input, grad_residual, grad_weight, None


class FusedRMSNormFunction(torch.autograd.Function):
    """Fused RMSNorm (without residual) with autograd support."""

    @staticmethod
    def forward(ctx, input, weight, epsilon):
        if CUDA_AVAILABLE:
            output = _fused_fwd(input, weight, epsilon)
        else:
            variance = input.pow(2).mean(dim=-1, keepdim=True)
            inv_rms = torch.rsqrt(variance + epsilon)
            output = input * inv_rms * weight

        ctx.save_for_backward(input, weight)
        ctx.epsilon = epsilon
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        epsilon = ctx.epsilon

        if CUDA_AVAILABLE:
            grads = _fused_bwd(grad_output, input, weight, epsilon)
            return grads[0], grads[1], None
        else:
            variance = input.pow(2).mean(dim=-1, keepdim=True)
            inv_rms = torch.rsqrt(variance + epsilon)
            x_norm = input * inv_rms

            grad_weight = (grad_output * x_norm).sum(dim=0)

            hidden_size = input.size(-1)
            dot = (grad_output * weight * x_norm).sum(dim=-1, keepdim=True)
            coeff = dot / hidden_size * inv_rms

            grad_input = grad_output * weight * inv_rms - x_norm * coeff

            return grad_input, grad_weight, None


def fused_rms_norm(input, weight, epsilon=1e-6):
    """Fused RMSNorm (CUDA kernel when available, Python fallback)."""
    return FusedRMSNormFunction.apply(input, weight, epsilon)


def fused_rms_norm_add(input, residual, weight, epsilon=1e-6):
    """Fused RMSNorm + Residual Add (CUDA kernel when available, Python fallback)."""
    return FusedRMSNormAddFunction.apply(input, residual, weight, epsilon)