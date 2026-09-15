from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


def _percentile(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    k = (len(ys) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(ys) - 1)
    if f == c:
        return ys[f]
    return ys[f] + (ys[c] - ys[f]) * (k - f)


@dataclass
class TimingStats:
    samples_ms: list[float]
    warmup: int
    measured: int

    @property
    def mean_ms(self) -> float:
        return sum(self.samples_ms) / len(self.samples_ms) if self.samples_ms else float("nan")

    @property
    def p50_ms(self) -> float:
        return _percentile(self.samples_ms, 50)

    @property
    def p90_ms(self) -> float:
        return _percentile(self.samples_ms, 90)

    @property
    def p99_ms(self) -> float:
        return _percentile(self.samples_ms, 99)

    def to_dict(self) -> dict[str, Any]:
        return {
            "warmup": self.warmup,
            "measured": self.measured,
            "mean_ms": self.mean_ms,
            "p50_ms": self.p50_ms,
            "p90_ms": self.p90_ms,
            "p99_ms": self.p99_ms,
            "samples_ms": self.samples_ms,
        }


@dataclass
class RunMetrics:
    ttft_ms: float | None = None
    itl_ms: TimingStats | None = None
    e2e_ms: float | None = None
    prefill_ms: float | None = None
    decode_ms: float | None = None
    tokens_per_second: float | None = None
    prompt_tokens: int | None = None
    generated_tokens: int | None = None
    peak_memory_bytes: int | None = None
    oom: bool = False
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.itl_ms is not None:
            d["itl_ms"] = self.itl_ms.to_dict()
        return d
