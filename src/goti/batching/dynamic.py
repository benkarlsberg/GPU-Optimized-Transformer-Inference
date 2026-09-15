from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque


@dataclass
class Request:
    request_id: str
    input_ids: list[int]
    max_new_tokens: int
    arrival_step: int = 0


@dataclass
class DynamicScheduler:
    """Small continuous-style scheduler for research comparisons vs static batching."""

    max_batch: int = 8
    waiting: Deque[Request] = field(default_factory=deque)
    active: list[Request] = field(default_factory=list)
    step: int = 0

    def enqueue(self, req: Request) -> None:
        req.arrival_step = self.step
        self.waiting.append(req)

    def admit(self) -> list[Request]:
        admitted: list[Request] = []
        while self.waiting and len(self.active) + len(admitted) < self.max_batch:
            admitted.append(self.waiting.popleft())
        self.active.extend(admitted)
        return admitted

    def tick(self) -> None:
        self.step += 1
