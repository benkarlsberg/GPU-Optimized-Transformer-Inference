# GPU testing (measurement machine)

## Environment

On the Windows CUDA PC, reuse conda env **`nlp`**:

```bat
conda activate nlp
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
```

Expected on this project's measurement machine: torch `2.11.0+cu128` seeing an RTX 5070.

Do **not** install another PyTorch into `ml` (CPU build) or `base` unless isolating on purpose.

## Install this package into `nlp`

```bat
cd <repo>
pip install -e ".[dev]"
```

## Run

```bat
python scripts/record_environment.py --out results\env.json
pytest -m gpu
python scripts/run_experiment.py --config configs\smoke_gpu.yaml
python scripts/bench_kernels.py --config configs\kernels_smoke.yaml
```

## Profiling

### PyTorch Profiler

Use `goti.profiling.torch_profiler.profile_run` around a short generate loop.
Export Chrome/TensorBoard traces locally; **do not commit** large trace files.

### Nsight (optional, if installed)

```bat
nsys profile -o results\nsys_smoke python scripts\run_experiment.py --config configs\smoke_gpu.yaml
```

Keep `.nsys-rep` artifacts out of git.
