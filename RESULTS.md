# Results

## Measured

No measurement campaigns are recorded in-tree yet. Entries below should cite artifacts under `results/runs/`.

### Hardware / software

| Field | Value |
|-------|-------|
| GPU | |
| Driver | |
| CUDA | |
| PyTorch | |
| OS | |

### Experiments executed

| Run ID | Config | Notes |
|--------|--------|-------|
| | | |

### Findings

|

## Hypotheses

- Long-prompt prefill tends toward compute / SDPA bound; small-batch decode toward memory / KV bound.
- `torch.compile` may help steady decode but can increase cold-start time to first token.
- Fused RMSNorm (with residual) is likely memory-bandwidth sensitive.

## Planned

- Full `configs/matrix_small.yaml` on a single RTX-class GPU
- Kernel shape sweeps and end-to-end ablations
- Profiler traces retained locally (not committed as large binaries)
