from __future__ import annotations

import torch

from goti.kernels.rmsnorm import rmsnorm_pytorch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # pragma: no cover
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _rmsnorm_kernel(
        X_ptr,
        W_ptr,
        Y_ptr,
        stride_x_row,
        stride_y_row,
        n_cols,
        eps,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < n_cols
        x = tl.load(X_ptr + row * stride_x_row + cols, mask=mask, other=0.0).to(tl.float32)
        var = tl.sum(x * x, axis=0) / n_cols
        rstd = tl.rsqrt(var + eps)
        w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = x * rstd * w
        tl.store(Y_ptr + row * stride_y_row + cols, y, mask=mask)


def rmsnorm_triton(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if not _HAS_TRITON or x.device.type != "cuda":
        return rmsnorm_pytorch(x, weight, eps=eps)
    assert x.is_contiguous()
    assert weight.is_contiguous()
    y = torch.empty_like(x)
    n_cols = x.shape[-1]
    n_rows = x.numel() // n_cols
    x2 = x.view(n_rows, n_cols)
    y2 = y.view(n_rows, n_cols)
    BLOCK = triton.next_power_of_2(n_cols)
    _rmsnorm_kernel[(n_rows,)](
        x2,
        weight,
        y2,
        x2.stride(0),
        y2.stride(0),
        n_cols,
        eps,
        BLOCK=BLOCK,
    )
    return y
