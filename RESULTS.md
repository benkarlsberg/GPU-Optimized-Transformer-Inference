# Results

## Measured

_No measurement campaigns committed yet. Fill only from `results/runs/` artifacts._

### Hardware / software

| Field | Value |
|-------|-------|
| GPU | _(from results/env.json)_ |
| Driver | |
| CUDA | |
| PyTorch | |
| OS | |

### Experiments executed

| Run ID | Config | Notes |
|--------|--------|-------|
| | | |

### Findings

_(After real runs only.)_

## Hypotheses

- Long-prompt prefill tends toward compute/SDPA bound; small-batch decode toward memory/KV bound.
- `torch.compile` may help steady decode but can inflate cold TTFT.
- Fused RMSNorm (+ residual) is likely memory-bandwidth sensitive.

## Planned

- `configs/matrix_small.yaml` on a single RTX-class GPU
- Kernel shape sweeps + E2E ablation
- Profiler traces (do not commit large binaries)
