from __future__ import annotations

import torch

from goti.kernels.swiglu import swiglu_pytorch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # pragma: no cover
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _swiglu_kernel(G_ptr, U_ptr, Y_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        g = tl.load(G_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(U_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        sig = 1.0 / (1.0 + tl.exp(-g))
        y = g * sig * u
        tl.store(Y_ptr + offs, y, mask=mask)


def swiglu_triton(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    if not _HAS_TRITON or gate.device.type != "cuda":
        return swiglu_pytorch(gate, up)
    assert gate.shape == up.shape
    y = torch.empty_like(gate)
    flat_g = gate.reshape(-1)
    flat_u = up.reshape(-1)
    flat_y = y.reshape(-1)
    n = flat_g.numel()
    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)
    _swiglu_kernel[grid](flat_g, flat_u, flat_y, n, BLOCK=BLOCK)
    return y
