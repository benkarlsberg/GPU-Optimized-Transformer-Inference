import pytest
import torch

from goti.kernels.rmsnorm import rmsnorm_pytorch
from goti.kernels.triton.rmsnorm import rmsnorm_triton

pytestmark = pytest.mark.gpu


def test_triton_rmsnorm_close_to_pytorch():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(0)
    x = torch.randn(16, 1024, device="cuda", dtype=torch.float16)
    w = torch.ones(1024, device="cuda", dtype=torch.float16)
    y_ref = rmsnorm_pytorch(x, w)
    y = rmsnorm_triton(x, w)
    assert torch.allclose(y.float(), y_ref.float(), atol=2e-2, rtol=2e-2)
