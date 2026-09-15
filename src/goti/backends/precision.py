from __future__ import annotations

import torch


def autocast_context(device: torch.device, dtype: torch.dtype):
    if device.type != "cuda":
        from contextlib import nullcontext

        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)
