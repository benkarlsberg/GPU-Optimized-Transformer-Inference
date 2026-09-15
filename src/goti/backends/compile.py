from __future__ import annotations

from typing import Any

import torch


def maybe_compile(model: Any, enabled: bool, **kwargs: Any) -> Any:
    if not enabled:
        return model
    return torch.compile(model, **kwargs)
