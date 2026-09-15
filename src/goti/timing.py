from __future__ import annotations

import time
from collections.abc import Callable
from contextlib import contextmanager
from typing import Iterator

import torch

from goti.types import TimingStats


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@contextmanager
def cuda_timer() -> Iterator[Callable[[], float]]:
    synchronize()
    t0 = time.perf_counter()

    def elapsed_ms() -> float:
        synchronize()
        return (time.perf_counter() - t0) * 1000.0

    yield elapsed_ms


def benchmark_callable(
    fn: Callable[[], None],
    *,
    warmup: int = 5,
    iters: int = 20,
) -> TimingStats:
    for _ in range(warmup):
        fn()
    synchronize()
    samples: list[float] = []
    for _ in range(iters):
        with cuda_timer() as elapsed:
            fn()
            ms = elapsed()
        samples.append(ms)
    return TimingStats(samples_ms=samples, warmup=warmup, measured=iters)
