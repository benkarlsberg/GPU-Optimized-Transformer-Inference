# Results

## Measured

### Hardware / software

| Field | Value |
|-------|-------|
| Host | BensPC |
| GPU | NVIDIA GeForce RTX 5070 (~12 GB) |
| Driver | 610.62 |
| CUDA (PyTorch) | 12.8 |
| PyTorch | 2.11.0+cu128 |
| OS | Windows 10 (build 26200) |
| Python | 3.11.15 |
| Model | HuggingFaceTB/SmolLM-135M |

### Experiments executed

| Run ID | Config | Notes |
|--------|--------|-------|
| smoke_gpu | `configs/smoke_gpu.yaml` | Baseline smoke; see `results/smoke_gpu_summary.json` |
| kernels_smoke | `configs/kernels_smoke.yaml` | RMSNorm / residual / SwiGLU microbench on CUDA |
| matrix_small | `configs/matrix_small.yaml` | 32 cells; dtype × batch × prompt × gen × compile; **0 OOM** |

### Matrix findings (`matrix_small`)

All **32/32** cells completed successfully.

| Observation | Result |
|-------------|--------|
| Fastest E2E mean | **586.2 ms** — bf16, batch 4, prompt 256, gen 32, compile=True (~218 tok/s, ~325 MB peak) |
| Slowest E2E mean | **2855.3 ms** — fp16, batch 4, prompt 64, gen 128, eager (~180 tok/s, ~289 MB peak) |
| BF16 vs FP16 | BF16 lower latency in **all 16** pairs; median BF16/FP16 = **0.895** |
| `torch.compile` | Median eager/compile latency ratio **1.015** (mean 1.027; range 0.92–1.22) |
| Batching | Batch 4 raises tok/s from ~40–50 to ~180–210 (fp16 eager) with similar wall-clock scale |
| Gen length | g=32 → ~0.6–0.8 s; g=128 → ~2.3–2.9 s; tok/s stays relatively flat (decode-dominated) |

Aggregated per-cell rows: `results/matrix_small_analysis.json`.
Paper write-up with figures: `paper/main.pdf`.

### Smoke baseline (earlier)

SmolLM-135M fp16, batch 1, prompt 64, gen 32: E2E mean **624.5 ms**, ~52 tok/s, ~268 MiB peak.

## Hypotheses

- Long-prompt prefill tends toward compute / SDPA bound; small-batch decode toward memory / KV bound.
- `torch.compile` may help some shapes but is not guaranteed on short measured windows for small models.
- Fused RMSNorm (with residual) remains a candidate for microbench follow-ups.

## Planned

- Streaming TTFT / inter-token latency instrumentation
- Quantization and dynamic-batching comparisons
- Kernel microbench bandwidth analysis and optional E2E kernel swap-ins
