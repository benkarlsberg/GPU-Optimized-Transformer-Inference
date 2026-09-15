import torch
import torch.nn.functional as F

from goti.kernels.swiglu import swiglu_pytorch


def test_swiglu_matches_silu_mul():
    torch.manual_seed(0)
    g = torch.randn(8, 32)
    u = torch.randn(8, 32)
    assert torch.allclose(swiglu_pytorch(g, u), F.silu(g) * u)
