# Optional CUDA extensions

CUDA extensions are optional and intended to be built where NVCC is available
(e.g. the Windows RTX measurement PC). macOS does not require them.

Planned ops mirroring Triton/PyTorch refs: fused RMSNorm, RMSNorm+residual, SiLU(gate)*up.

Until sources land, use PyTorch + Triton paths.
