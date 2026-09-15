#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from goti.kernels.rmsnorm import rmsnorm_pytorch, rmsnorm_residual_pytorch
from goti.kernels.swiglu import swiglu_pytorch
from goti.timing import benchmark_callable


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("results/runs/kernels_smoke.json"))
    args = p.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    results = []
    for dtype_name in cfg["dtypes"]:
        dtype = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}[dtype_name]
        if device.type == "cpu" and dtype != torch.float32:
            dtype = torch.float32
        for rows in cfg["rows"]:
            for hidden in cfg["hidden"]:
                x = torch.randn(rows, hidden, device=device, dtype=dtype)
                w = torch.ones(hidden, device=device, dtype=dtype)
                if "rmsnorm" in cfg["ops"]:
                    stats = benchmark_callable(
                        lambda: rmsnorm_pytorch(x, w),
                        warmup=cfg["warmup"],
                        iters=cfg["iters"],
                    )
                    results.append(
                        {"op": "rmsnorm", "dtype": dtype_name, "rows": rows, "hidden": hidden, **stats.to_dict()}
                    )
                if "rmsnorm_residual" in cfg["ops"]:
                    r = torch.randn_like(x)
                    stats = benchmark_callable(
                        lambda: rmsnorm_residual_pytorch(x, r, w),
                        warmup=cfg["warmup"],
                        iters=cfg["iters"],
                    )
                    results.append(
                        {
                            "op": "rmsnorm_residual",
                            "dtype": dtype_name,
                            "rows": rows,
                            "hidden": hidden,
                            **stats.to_dict(),
                        }
                    )
                if "swiglu" in cfg["ops"]:
                    g = torch.randn_like(x)
                    u = torch.randn_like(x)
                    stats = benchmark_callable(
                        lambda: swiglu_pytorch(g, u),
                        warmup=cfg["warmup"],
                        iters=cfg["iters"],
                    )
                    results.append(
                        {"op": "swiglu", "dtype": dtype_name, "rows": rows, "hidden": hidden, **stats.to_dict()}
                    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"config": cfg, "device": str(device), "results": results}, indent=2),
        encoding="utf-8",
    )
    print("wrote", args.out)


if __name__ == "__main__":
    main()
