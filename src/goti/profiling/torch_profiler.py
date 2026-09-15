from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import torch


@contextmanager
def profile_run(trace_dir: Path, *, wait: int = 1, warmup: int = 1, active: int = 3) -> Iterator[None]:
    trace_dir.mkdir(parents=True, exist_ok=True)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(trace_dir)),
        record_shapes=True,
        with_stack=False,
    ) as prof:
        yield
        prof.step()
