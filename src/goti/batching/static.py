from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StaticBatch:
    """Fixed-size batch of requests started together."""

    request_ids: list[str]
    input_ids: list[list[int]]
    max_new_tokens: int
