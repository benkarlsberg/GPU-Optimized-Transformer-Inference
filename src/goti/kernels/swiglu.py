from __future__ import annotations

import torch
import torch.nn.functional as F


def swiglu_pytorch(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """SiLU(gate) * up reference fragment used in SwiGLU MLPs."""
    return F.silu(gate) * up
