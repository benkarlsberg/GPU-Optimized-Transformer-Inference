# Results

## Measured

### Hardware / software

| Field | Value |
|-------|-------|
| Host | BensPC |
| GPU | NVIDIA GeForce RTX 5070 (12 GB) |
| Driver | 610.62 |
| CUDA (PyTorch) | 12.8 |
| PyTorch | 2.11.0+cu128 |
| OS | Windows 10 (build 26200) |
| Python | 3.11.15 |

### Experiments executed

| Run ID | Config | Notes |
|--------|--------|-------|
| smoke_gpu | `configs/smoke_gpu.yaml` | SmolLM-135M, fp16, batch 1, prompt 64, gen 32, eager, SDPA; warmup 2 / measured 5 |
| kernels_smoke | `configs/kernels_smoke.yaml` | RMSNorm, RMSNorm+residual, SwiGLU microbench shape sweep on CUDA |

### Findings

Smoke end-to-end `generate()` latency on RTX 5070 (synchronized, discarded outputs):

| Stat | Value |
|------|-------|
| mean | 624.5 ms |
| p50 | 620.3 ms |
| p90 | 634.8 ms |
| p99 | 636.6 ms |

Representative measured iteration: ~51.7 tok/s, peak CUDA memory ~268 MiB allocated, no OOM.

CPU unit tests: 6 passed. GPU kernel numerical check: 1 passed (Triton path falls back to PyTorch when Triton is unavailable).

These numbers establish a working baseline harness on this GPU; they are not yet a comparison across compile / dtype / batching axes.

## Hypotheses

- Long-prompt prefill tends toward compute / SDPA bound; small-batch decode toward memory / KV bound.
- `torch.compile` may help steady decode but can increase cold-start time to first token.
- Fused RMSNorm (with residual) is likely memory-bandwidth sensitive.

## Planned

- Full `configs/matrix_small.yaml` sweeps on this GPU
- Kernel shape sweeps with bandwidth/roofline discussion and end-to-end ablations
- Streaming TTFT / ITL instrumentation (current smoke marks E2E from `generate()`)
- Local profiler traces (not committed as large binaries)
