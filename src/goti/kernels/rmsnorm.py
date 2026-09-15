from __future__ import annotations

import torch


def rmsnorm_pytorch(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Idiomatic RMSNorm reference (PyTorch)."""
    orig_dtype = x.dtype
    x_f = x.float()
    var = x_f.pow(2).mean(dim=-1, keepdim=True)
    x_norm = x_f * torch.rsqrt(var + eps)
    return (x_norm * weight.float()).to(orig_dtype)


def rmsnorm_residual_pytorch(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference for fused pattern: y = RMSNorm(x + residual)."""
    h = x + residual
    return rmsnorm_pytorch(h, weight, eps=eps), h


def try_rmsnorm_triton(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    try:
        from goti.kernels.triton.rmsnorm import rmsnorm_triton

        return rmsnorm_triton(x, weight, eps=eps)
    except Exception:
        return rmsnorm_pytorch(x, weight, eps=eps)
