from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch

from goti.backends.attention import configure_attention
from goti.backends.compile import maybe_compile
from goti.benchmarks.workloads import SyntheticPrompt, make_input_ids
from goti.env import collect_environment
from goti.models.loader import load_causal_lm
from goti.timing import synchronize
from goti.types import RunMetrics, TimingStats


def _peak_memory_bytes(device: torch.device) -> int | None:
    if device.type != "cuda":
        return None
    return int(torch.cuda.max_memory_allocated(device))


@torch.inference_mode()
def run_generation_benchmark(cfg: dict[str, Any]) -> dict[str, Any]:
    """Controlled generate() benchmark; discards text; no logging in timed regions."""
    seed = int(cfg.get("seed", 0))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    loaded = load_causal_lm(cfg["model"], dtype=cfg.get("dtype", "fp16"), device=cfg.get("device"))
    configure_attention(cfg.get("attention", "sdpa"))
    model = maybe_compile(loaded.model, bool(cfg.get("compile", False)))

    prompt = SyntheticPrompt(
        batch_size=int(cfg.get("batch_size", 1)),
        prompt_length=int(cfg.get("prompt_length", 64)),
    )
    max_new = int(cfg.get("max_new_tokens", 32))
    warmup = int(cfg.get("warmup", 2))
    iters = int(cfg.get("iters", 5))

    input_ids = make_input_ids(loaded.tokenizer, prompt, loaded.device)
    attn = torch.ones_like(input_ids)

    def one_call() -> RunMetrics:
        if loaded.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(loaded.device)
        synchronize()
        t0 = time.perf_counter()
        try:
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attn,
                max_new_tokens=max_new,
                do_sample=False,
                use_cache=True,
                pad_token_id=loaded.tokenizer.pad_token_id,
            )
        except torch.cuda.OutOfMemoryError:
            synchronize()
            return RunMetrics(oom=True)
        synchronize()
        e2e = (time.perf_counter() - t0) * 1000.0
        gen_tokens = int(out.shape[-1] - input_ids.shape[-1])
        tps = (gen_tokens * prompt.batch_size) / (e2e / 1000.0) if e2e > 0 else None
        return RunMetrics(
            e2e_ms=e2e,
            prompt_tokens=int(input_ids.numel()),
            generated_tokens=gen_tokens * prompt.batch_size,
            tokens_per_second=tps,
            peak_memory_bytes=_peak_memory_bytes(loaded.device),
            extras={"approx_ttft": True},
        )

    for _ in range(warmup):
        one_call()

    runs: list[dict[str, Any]] = []
    e2es: list[float] = []
    for _ in range(iters):
        m = one_call()
        runs.append(m.to_dict())
        if m.e2e_ms is not None and not m.oom:
            e2es.append(m.e2e_ms)

    return {
        "config": cfg,
        "environment": collect_environment(),
        "warmup": warmup,
        "iters": iters,
        "runs": runs,
        "e2e_summary_ms": TimingStats(e2es, warmup=warmup, measured=len(e2es)).to_dict() if e2es else None,
    }


def save_run(result: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
