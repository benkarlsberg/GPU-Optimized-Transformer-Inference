# GPU testing

## Prerequisites

- NVIDIA GPU with a working driver
- PyTorch build with CUDA enabled
- This repository installed editable: `pip install -e ".[dev]"`

Verify:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

## Smoke tests

```bash
python scripts/record_environment.py --out results/env.json
pytest -m gpu
python scripts/run_experiment.py --config configs/smoke_gpu.yaml
python scripts/bench_kernels.py --config configs/kernels_smoke.yaml
```

## Profiling

### PyTorch Profiler

Wrap a short generate loop with `goti.profiling.torch_profiler.profile_run`. Keep large trace files out of git.

### Nsight Systems (optional)

```bash
nsys profile -o results/nsys_smoke python scripts/run_experiment.py --config configs/smoke_gpu.yaml
```

`.nsys-rep` files are gitignored.
