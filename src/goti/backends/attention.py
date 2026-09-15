from __future__ import annotations

import os
from typing import Literal

AttentionImpl = Literal["sdpa", "eager", "flash_sdp", "mem_efficient_sdp"]


def configure_attention(impl: AttentionImpl) -> dict[str, bool]:
    """Best-effort SDPA backend selection via PyTorch SDP flags."""
    import torch.backends.cuda as cuda_backends

    flags = {"flash_sdp": False, "mem_efficient_sdp": False, "math_sdp": True}
    if impl == "sdpa":
        flags = {"flash_sdp": True, "mem_efficient_sdp": True, "math_sdp": True}
    elif impl == "flash_sdp":
        flags = {"flash_sdp": True, "mem_efficient_sdp": False, "math_sdp": False}
    elif impl == "mem_efficient_sdp":
        flags = {"flash_sdp": False, "mem_efficient_sdp": True, "math_sdp": False}
    elif impl == "eager":
        os.environ["PYTORCH_ENABLE_SDPA"] = "0"
        return flags
    else:
        raise ValueError(f"Unknown attention impl {impl}")

    if hasattr(cuda_backends, "enable_flash_sdp"):
        cuda_backends.enable_flash_sdp(flags["flash_sdp"])
        cuda_backends.enable_mem_efficient_sdp(flags["mem_efficient_sdp"])
        cuda_backends.enable_math_sdp(flags["math_sdp"])
    return flags
