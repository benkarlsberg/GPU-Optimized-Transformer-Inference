# GPU-Optimized Transformer Inference

Systems study of decoder-only transformer inference on a single GPU: how implementation choices and workload shape affect latency, throughput, and memory behavior.

## Research questions

1. How do eager PyTorch, `torch.compile`, mixed precision, weight quantization, dynamic batching, and optimized attention affect time to first token, inter-token latency, end-to-end latency, prefill and decode throughput, tokens per second, and peak / steady-state GPU memory (and bandwidth or utilization when counters are available)?

2. How do those metrics change with prompt length, generation length, batch size, model size, data type, quantization settings, static versus dynamic request arrival, and prefill versus autoregressive decode?

3. Which inference operations are limited by kernel-launch overhead, memory traffic, arithmetic throughput, synchronization, occupancy, temporary tensor allocation, or shape-dependent recompilation?

4. Can focused custom kernels (Triton / optional CUDA) improve selected ops through fusion, coalesced access, less intermediate materialization, better tiling, or fewer launches — and do those gains appear in an end-to-end path?

## Repository layout

```
src/goti/
  backends/      # eager, compile, precision, attention
  batching/      # static and dynamic/continuous-style scheduling
  benchmarks/    # timing harness and workloads
  kernels/       # PyTorch references, Triton, optional CUDA
  models/        # Hugging Face causal LM loading
  profiling/     # PyTorch profiler helpers
  reporting/     # plots and tables from saved JSON
configs/         # experiment YAML
experiments/     # longer campaign notes
scripts/         # CLI entrypoints
tests/
docs/
results/         # raw run JSON and figures (local artifacts)
```

## Setup

### CPU development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q -m "not gpu"
```

### GPU measurement machine

Requires a CUDA-capable PyTorch install. Then:

```bash
pip install -e ".[dev]"
python scripts/record_environment.py --out results/env.json
python scripts/run_experiment.py --config configs/smoke_gpu.yaml
```

More detail: [docs/gpu-testing.md](docs/gpu-testing.md).

## Running experiments

```bash
python scripts/record_environment.py --out results/env.json
python scripts/run_experiment.py --config configs/smoke_cpu.yaml
python scripts/run_experiment.py --config configs/smoke_gpu.yaml
python scripts/run_experiment.py --config configs/matrix_small.yaml  # via scripts/run_matrix.py
python scripts/bench_kernels.py --config configs/kernels_smoke.yaml
python scripts/plot_results.py --input results/runs --out results/figures
```

Supported knobs (where hardware and dependencies allow): eager vs `torch.compile`, FP32 / FP16 / BF16, SDPA and alternate SDPA backends, optional weight quantization, static batching, and a small dynamic batching scheduler.

## Results

Raw timings are written as JSON under `results/runs/`. Figures are generated from those files. Summary write-ups live in [`RESULTS.md`](RESULTS.md), with separate sections for measured outcomes, open hypotheses, and planned follow-ups.

## Paper

A short technical report summarizing the single-GPU matrix study is in [`paper/main.pdf`](paper/main.pdf) (source: [`paper/main.tex`](paper/main.tex)).

## License

MIT
