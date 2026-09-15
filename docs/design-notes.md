# Design notes

## Why a harness, not scripts

Latency distributions, OOM handling, environment capture, and config matrices
need shared infrastructure so comparisons stay fair across eager/compile/dtype/batching.

## Metric caveats

The initial `model.generate()` path records synchronized E2E latency and peak
memory. Fine-grained TTFT/ITL via streaming hooks is a follow-on; early results
mark approximate TTFT when hooks are absent.

## Kernel selection

RMSNorm and SwiGLU fragments are common decode-loop residents, independently
testable, and amenable to fusion / bandwidth analysis without replacing an
entire attention stack.
