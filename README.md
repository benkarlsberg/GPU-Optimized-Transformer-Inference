# GPU-Optimized Transformer Inference

Reproducible research codebase for characterizing **GPU-optimized decoder-only transformer inference**: precision, compilation, attention, batching, memory, and small custom kernels (Triton / optional CUDA).

Experiments are configuration-driven. Timings are GPU-synchronized and distribution-aware. Results export as machine-readable JSON.

## Research questions

1. How do eager PyTorch, `torch.compile`, mixed precision, weight quantization, dynamic batching, and optimized attention affect TTFT, inter-token latency, end-to-end latency, prefill/decode throughput, tokens/s, peak/steady GPU memory, and bandwidth/utilization (when counters are available)?
2. How do results change with prompt length, generation length, batch size, model size, dtype, quantization, static vs dynamic arrivals, and prefill vs decode?
3. Which ops are limited by launch overhead, memory traffic, arithmetic throughput, synchronization, occupancy, temporary allocations, or recompilation?
4. Can focused custom kernels improve selected ops via fusion, coalescing, reduced materialization, tiling, and fewer launches — and do gains appear end-to-end?

## Layout

```
src/goti/
  backends/ batching/ benchmarks/ kernels/ models/ profiling/ reporting/
configs/ experiments/ scripts/ tests/ docs/ results/
```

## Quick start (CPU / macOS)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q -m "not gpu"
```

## GPU runs (Windows measurement machine)

Reuse the existing conda env **`nlp`** (CUDA PyTorch already installed). Do not install a second PyTorch into `base` or `ml` by default.

```bat
conda activate nlp
pip install -e ".[dev]"
python scripts/record_environment.py --out results/env.json
python scripts/run_experiment.py --config configs/smoke_gpu.yaml
```

See [docs/gpu-testing.md](docs/gpu-testing.md).

## Results policy

- Never invent measurements.
- `RESULTS.md` separates Measured / Hypotheses / Planned.
- Raw timings under `results/runs/`; figures are generated from those files.

## License

MIT
