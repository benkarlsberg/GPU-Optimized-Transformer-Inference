# Results

## Measured

### Hardware / software

| Field | Value |
|-------|-------|
| Host | Single desktop workstation |
| GPU | NVIDIA GeForce RTX 5070 (~12 GB) |
| Driver | 610.62 (earlier session, recorded in `smoke_gpu_summary.json`); 610.88 (model-size sweep session) |
| CUDA (PyTorch) | 12.8 |
| PyTorch | 2.11.0+cu128 |
| OS | Windows 11 (build 26200) |
| Python | 3.11.15 |
| Models | HuggingFaceTB/SmolLM-135M (matrix, batch sweep); SmolLM-360M and SmolLM-1.7B added in the model-size sweep |

### Experiments executed

| Run ID | Config | Notes |
|--------|--------|-------|
| smoke_gpu | `configs/smoke_gpu.yaml` | Baseline smoke; see `results/smoke_gpu_summary.json` |
| kernels_smoke | `configs/kernels_smoke.yaml` | RMSNorm / residual / SwiGLU microbench on CUDA |
| matrix_small | `configs/matrix_small.yaml` | 32 cells; dtype × batch × prompt × gen × compile; **0 OOM** |
| batch_sweep | `configs/batch_sweep.yaml`, `configs/batch_sweep_rev.yaml` | Batch 1–256, bf16 eager, prompt 256, gen 128; 3 passes; **0 OOM** |
| model_size_sweep | `configs/batch_sweep_360m*.yaml`, `configs/batch_sweep_1p7b*.yaml` | SmolLM-135M/360M/1.7B, dynamic and static KV cache; 2 passes each |
| profile_decode | `scripts/profile_decode.py`, `scripts/probe_sdpa_gqa.py` | Per-decode-step profile (torch.profiler + CUPTI) of 11 model/cache/batch cells, plus a GQA/SDPA backend probe; 2 passes each |

### Matrix findings (`matrix_small`)

All **32/32** cells completed successfully.

| Observation | Result |
|-------------|--------|
| Fastest E2E mean | **586.2 ms** — bf16, batch 4, prompt 256, gen 32, compile=True (~218 tok/s, ~325 MB peak) |
| Slowest E2E mean | **2855.3 ms** — fp16, batch 4, prompt 64, gen 128, eager (~180 tok/s, ~289 MB peak) |
| BF16 vs FP16 | BF16 lower latency in **all 16** pairs; median BF16/FP16 = **0.895** |
| `torch.compile` | Median eager/compile latency ratio **1.015** (mean 1.027; range 0.92–1.22) |
| Batching | Batch 4 raises tok/s from ~41–49 to ~180–189 (fp16 eager) with similar wall-clock scale |
| Gen length | g=32 → ~0.6–0.8 s; g=128 → ~2.3–2.9 s; tok/s stays relatively flat (decode-dominated) |

Aggregated per-cell rows: `results/matrix_small_analysis.json`.
Paper write-up with figures: `paper/main.pdf`.

### Batch-size sweep (`batch_sweep`)

BF16, eager, prompt 256, 128 new tokens, 2 warmup + 5 measured iterations per cell. Mean of two passes (descending and ascending order), which agreed within 2% at every batch size. A first pass was inconsistent with the matrix (batch 1 at 3.4 s vs 2.37 s; batch 256 faster than 128) and is excluded from the aggregate but kept under `excluded_passes` in the analysis file.

| Batch | E2E mean (ms) | tok/s | Peak MB |
|------:|--------------:|------:|--------:|
| 1 | 2303 | 56 | 283 |
| 2 | 2314 | 111 | 299 |
| 4 | 2354 | 218 | 328 |
| 8 | 2335 | 438 | 386 |
| 16 | 2317 | 884 | 502 |
| 32 | 2336 | 1754 | 733 |
| 64 | 3687 | 2222 | 1197 |
| 128 | 6827 | 2401 | 2126 |
| 256 | 13286 | 2467 | 3980 |

- Batch 1–32: latency flat at ~2.3 s (~18 ms per decode step) while throughput rises ~31×.
- Beyond 32: latency grows roughly linearly with batch size; throughput rises only from ~2,220 tok/s (batch 64) to ~2,470 tok/s (batch 256).
- Knee near batch 32 on this GPU for this model. Batch 64 gives ~27% more throughput for ~58% more latency.

Aggregate: `results/batch_sweep_analysis.json` (`scripts/analyze_batch_sweep.py`). Figure: `paper/figs/latency_throughput_tradeoff.pdf` (`scripts/plot_tradeoff.py`).

### Model-size sweep (`model_size_sweep`)

Same settings as the batch sweep (BF16, eager, prompt 256, 128 new tokens, 2 warmup + 5 measured). Two passes per configuration (descending, ascending), agreeing to within about 1.5% (max 1.52%) at every valid cell. 135M was re-run in the same session and matched the original sweep within 1.1% up to batch 128 (4.5% faster at 256).

| Batch | 135M ms | 135M tok/s | 360M ms | 360M tok/s | 1.7B dyn ms | 1.7B dyn tok/s | 1.7B static ms | 1.7B static tok/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2308 | 55 | 2457 | 52 | 1379 | 93 | 1706 | 75 |
| 2 | 2332 | 110 | 2437 | 105 | 1514 | 169 | 1732 | 148 |
| 4 | 2365 | 216 | 2430 | 211 | 1614 | 317 | 1748 | 293 |
| 8 | 2330 | 440 | 2451 | 418 | 1801 | 569 | 1848 | 554 |
| 16 | 2341 | 875 | 2481 | 826 | 2233 | 917 | 2154 | 951 |
| 32 | 2333 | 1756 | 3504 | 1169 | 3232 | 1268 | 2751 | 1489 |
| 64 | 3680 | 2226 | 6301 | 1300 | spilled | - | 4008 | 2044 |
| 128 | 6754 | 2426 | 11544 | 1419 | - | - | - | - |
| 256 | 12684 | 2583 | 22282 | 1471 | - | - | - | - |

Memory (driver-reported in use / PyTorch peak allocated / peak reserved, largest batch run). Reserved, driver-reported and Windows shared-GPU-memory counters were logged only for the model-size runs; the matrix and first batch-sweep runs record peak allocated memory only.

| Series | Largest batch | In use | Allocated | Reserved |
|---|---:|---:|---:|---:|
| 135M | 256 | 4.7 GB | 4.0 GB | 4.2 GB |
| 360M | 256 | 7.8 GB | 7.1 GB | 7.2 GB |
| 1.7B dynamic | 32 | 10.3 GB | 5.6 GB | 9.7 GB |
| 1.7B dynamic (spilled) | 64 | 11.8 GB + ~13 GB system RAM | 8.0 GB | 24.4 GB |
| 1.7B static | 64 | 11.2 GB | 8.9 GB | 10.7 GB |

- Knee: ~32 for 135M, ~16 for 360M, no flat region for 1.7B.
- Batch-1 step time: 18.0 / 19.2 / 10.8 ms (135M / 360M / 1.7B). Not proportional to parameters; see the decode-step profile below for the measured breakdown.
- 1.7B with the growing cache (`DynamicCache`) fragments the allocator; at batch 64 reserved memory (24.4 GB) exceeds the 12 GB card, and NVIDIA's documented CUDA system-memory fallback (Windows driver, default policy; https://nvidia.custhelp.com/app/answers/detail/a_id/5490) places ~13 GB in system memory (35 s) instead of raising an OOM. Allocated memory (8.0 GB) does not show this. Static cache avoids it. The driver's "Prefer No Sysmem Fallback" policy was not tested.

Aggregate: `results/model_size_sweep_analysis.json`. Figure: `paper/figs/model_size_sweep.pdf` (`scripts/plot_model_size.py`). Configs: `configs/batch_sweep_360m*.yaml`, `configs/batch_sweep_1p7b*.yaml`.

### Decode-step profile (`profile_decode`)

**Method.** `scripts/profile_decode.py`: BF16, eager, SDPA attention, synthetic prompt of 256 tokens, manual greedy decode loop over `DynamicCache` (or `StaticCache` for the static cells) that mirrors what `generate()` does per step. Per cell: 8 warmup steps, then 32 timed steps synchronized after every step (wall), 32 timed steps without per-step synchronization (host dispatch), a count of synchronizing CUDA calls in one step, and 10 steps (after 2 skipped) under `torch.profiler` with CPU + CUDA (CUPTI) activity. GPU busy = union of kernel/memcpy/memset intervals per step; idle fraction = 1 − busy / unprofiled wall; weight bandwidth = BF16 weight bytes / GPU busy time (rated 672 GB/s). `generate()` is also timed end to end (1 warmup + 2 iterations, 128 new tokens). Two passes, the second in reverse cell order; values below are the mean of the two passes. `scripts/probe_sdpa_gqa.py` checks which SDPA backends accept the decode shapes and reruns the GQA models with Transformers' GQA fast path disabled.

| Model | Cache | Batch | Wall ms | Dispatch ms | GPU busy ms | GPU idle | Kernels/step | Weight BW (b1) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 135M | dynamic | 1 | 21.0* | 24.5* | 2.73 | 87% | 1680 | 98 GB/s (15%) |
| 135M | dynamic | 32 | 19.9 | 19.9 | 8.50 | 57% | 1680 | |
| 135M | dynamic | 128 | 44.1 | 43.3 | 36.6 | 17% | 1680 | |
| 360M | dynamic | 1 | 24.5* | 23.8* | 4.17 | 83% | 1790 | 173 GB/s (26%) |
| 360M | dynamic | 16 | 21.0 | 21.0 | 7.92 | 62% | 1790 | |
| 360M | dynamic | 64 | 41.1 | 40.2 | 33.2 | 19% | 1822 | |
| 1.7B | dynamic | 1 | 11.9 | 11.8 | 7.24 | 39% | 990 | 473 GB/s (70%) |
| 1.7B | dynamic | 16 | 14.9 | 14.7 | 11.9 | 20% | 990 | |
| 1.7B | dynamic | 32 | 19.9 | 19.5 | 16.5 | 17% | 1014 | |
| 1.7B | static | 1 | 13.8 | 13.4 | 7.52 | 45% | 1144 | 455 GB/s (68%) |
| 1.7B | static | 32 | 17.2 | 16.5 | 13.2 | 23% | 1170 | |

\* The two passes differed for these cells: wall 22.1 / 19.9 ms (135M) and 27.8 / 21.3 ms (360M); dispatch 29.2 / 19.8 and 26.2 / 21.5 ms. The second pass matches the probe's default-path runs (19.9–20.0 and 21.4–21.6 ms). All other cells agree within 3% across passes; kernel counts and GPU busy times are identical across passes (busy within 1%).

GQA counterfactual (`probe_sdpa_gqa.py`, mean of two passes, wall ms per step):

| Model | Batch | Default (GQA via `enable_gqa`) | repeat_kv + multi-head SDPA | Kernels/step |
|---|---:|---:|---:|---|
| 135M | 1 | 20.0 | 15.9 | 1680 → 1290 |
| 135M | 32 | 19.7 | 15.9 | 1680 → 1290 |
| 360M | 1 | 21.5 | 17.1 | 1790 → 1374 |
| 360M | 16 | 21.1 | 16.7 | 1790 → 1374 |

Findings:

- **Kernel counts are structural.** Kernels per step = layers × 55 + 30 for the GQA models (135M: 30 layers, 360M: 32) and layers × 40 + 30 for 1.7B (24 layers), identical in both passes and across batch sizes below the knee (one extra kernel per layer appears at 360M b64 and 1.7B b32).
- **Below the knee, steps are host-bound.** At batch 1 the GPU is idle 87% / 83% / 39% of the step. Host dispatch time / kernels = 11.7–12.0 µs in every below-knee cell (second pass for the two starred cells), and host time outside sync waits (under the profiler) varies by < 4% across batch sizes for each model with the dynamic cache.
- **Sync caveat.** With `DynamicCache`, Transformers forces one GPU→CPU sync per step (`transformers/masking_utils.py:269`, attention-mask construction), so dispatch ≈ wall by construction. The host-bound conclusion rests on the GPU idle fraction and batch-independent host time. The static cache has no per-step sync; at batch 1 its dispatch (13.4 ms) still equals its unsynchronized wall time.
- **GQA falls back to the SDPA math backend on this build.** The extra 15 kernels per layer in the GQA models are all inside `aten::scaled_dot_product_attention` (16 kernels per call vs. 1 fused memory-efficient kernel for 1.7B). Transformers passes `enable_gqa=True` when no mask is needed; this Windows wheel (torch 2.11.0+cu128) is not compiled with FlashAttention and the memory-efficient kernel needs equal Q/KV head counts, so default dispatch picks the math backend. In isolation (KV 300, batch 1) one call costs 156–188 µs of host time vs. 32 µs (3 kernels) for repeat_kv + multi-head SDPA. Disabling the GQA fast path cuts the step by about 20%. This is a property of this PyTorch build/dispatch, not of GQA in general (PyTorch documents `enable_gqa` support in its FlashAttention backend; not tested here). cuDNN attention accepted the shapes when forced but is not chosen by default dispatch; it was not timed.
- **Why 1.7B is faster per step.** After the counterfactual, step time / layers = 0.53 / 0.53 / 0.50 ms (135M / 360M / 1.7B). 1.7B vs 135M at batch 1 (11.9 vs 20.0 ms) splits roughly equally between the GQA fallback (4.1 ms) and six fewer layers (4.0 ms); vs 360M, 4.4 and 5.2 ms.
- **Weight bytes matter for 1.7B's GPU time.** At batch 1, GEMV kernels are 81% of 1.7B GPU kernel time and weights stream at 473 GB/s over busy time (70% of rated). The step is still host-bound (~11.8 ms of host time for 990 kernels vs 7.2 ms GPU busy). 135M / 360M reach 15% / 26% of rated at batch 1.
- **Above the knee** idle falls to 17–19% (23% for 1.7B static b32; GPU-bound) and the profiler lengthens steps by at most 1.15×. For the GQA models weights reach only 1.1% (135M b128) / 3.2% (360M b64) of rated bandwidth (weights + KV: 4.5% / 6.5%); about two-thirds of kernel time is element-wise copies and FP32 element-wise / GEMV kernels that appear only inside the math-backend SDPA call; BF16 GEMMs of the linear layers are under 8%. No hardware counters were collected.
- **Static vs dynamic cache (1.7B).** At batch 32, 29% of dynamic-cache GPU kernel time is `DynamicCache` `torch.cat` copies; the static cache avoids them (busy 13.2 vs 16.5 ms; step 17.2 vs 19.9 ms), consistent with its 15% advantage at batch 32 in the model-size sweep. At batch 1 the static cache is slower (13.8 vs 11.9 ms) mainly from 154 more kernels per step; attention over the full preallocated length adds only ~0.2 ms of GPU time.
- **135M crossover.** Linear interpolation of GPU busy time between b32 (8.5 ms) and b128 (36.6 ms) reaches the ~20 ms host time near b ≈ 70 at KV length 280; only three points, and the sweep's latency already rises by batch 64.
- **Profiler overhead.** Profiled step time / unprofiled wall = 1.34–1.40× below the knee, ≤ 1.15× above it; idle fractions use the unprofiled wall.

Session offset: this profile ran in a later session than the model-size sweep. `generate()` (same prompt, 128 new tokens) was 11.6–13.2% slower than the model-size sweep in every host-bound cell but only 0.7–1.2% slower in the five GPU-bound cells (135M b128, 360M b64, 1.7B dynamic b16/b32, 1.7B static b32), i.e. the session difference is host-side. Batch 1 `generate()` end to end: 2603 / 2782 / 1561 ms (20.3 / 21.7 / 12.2 ms per token) vs 2308 / 2457 / 1379 ms (18.0 / 19.2 / 10.8) in the sweep; model ratios are unchanged (360M/135M 1.07, 1.7B/135M 0.60). Compare absolute profile times with each other, not with the sweep. Windows only: no Linux comparison, so WDDM effects on host cost per kernel are not isolated.

Data: `results/profile_decode/summary.json` (per-cell aggregates), `results/profile_decode/raw.json` (per-step data for every pass), `results/profile_decode/sdpa_gqa_probe.json` (backend probe and GQA counterfactual). Chrome traces of the batch-1 cells are not committed.

### Smoke baseline (earlier)

SmolLM-135M fp16, batch 1, prompt 64, gen 32: E2E mean **624.5 ms**, ~52 tok/s, ~268 MiB peak.

## Hypotheses

- Long-prompt prefill tends toward compute / SDPA bound. (Small-batch decode was hypothesized to be memory / KV bound; the decode-step profile shows it is host-bound on this platform.)
- `torch.compile` may help some shapes but is not guaranteed on short measured windows for small models.
- Fused RMSNorm (with residual) remains a candidate for microbench follow-ups.

## Planned

- Streaming TTFT / inter-token latency instrumentation
- Quantization and dynamic-batching comparisons
- Kernel microbench bandwidth analysis and optional E2E kernel swap-ins
