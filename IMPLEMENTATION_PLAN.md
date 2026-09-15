# Implementation plan

## Phases

1. Scaffold — package, harness, kernel refs, CPU tests, CI, docs (this tree).
2. Eager baseline metrics on CUDA (TTFT, ITL, prefill/decode).
3. Variants — compile, dtypes, attention, optional quant, batching modes.
4. Profiling — torch.profiler recipes; Nsight command docs.
5. Kernels — validate + microbench + E2E swap-in.
6. Matrix runs — JSON + plots; fill Measured in RESULTS.md only with real data.

## Initial matrix

See `configs/matrix_small.yaml`. Start with SmolLM-135M-class models.

## Kernel targets

RMSNorm; RMSNorm+residual; SiLU(gate)*up fragment. PyTorch + Triton; optional CUDA extension where NVCC exists.

## Non-goals

Full serving-engine replacement; multi-GPU; fabricated performance claims.
