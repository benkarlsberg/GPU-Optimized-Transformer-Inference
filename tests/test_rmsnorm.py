import torch

from goti.kernels.rmsnorm import rmsnorm_pytorch, rmsnorm_residual_pytorch


def test_rmsnorm_unit_variance_ish():
    torch.manual_seed(0)
    x = torch.randn(4, 64)
    w = torch.ones(64)
    y = rmsnorm_pytorch(x, w)
    rms = y.float().pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-5, rtol=1e-4)


def test_rmsnorm_residual_matches_reference():
    torch.manual_seed(0)
    x = torch.randn(2, 128)
    r = torch.randn(2, 128)
    w = torch.randn(128)
    y, h = rmsnorm_residual_pytorch(x, r, w)
    assert torch.equal(h, x + r)
    y2 = rmsnorm_pytorch(x + r, w)
    assert torch.allclose(y, y2, atol=1e-5, rtol=1e-5)
