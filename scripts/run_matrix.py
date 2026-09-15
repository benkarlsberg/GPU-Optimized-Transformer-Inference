#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import yaml

from goti.benchmarks.harness import run_generation_benchmark, save_run


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("results/runs"))
    args = p.parse_args()
    doc = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    base = dict(doc["base"])
    axes = doc["axes"]
    keys = list(axes.keys())
    for values in itertools.product(*[axes[k] for k in keys]):
        cfg = dict(base)
        cfg.update(dict(zip(keys, values)))
        name = "_".join(f"{k}-{v}" for k, v in zip(keys, values))
        result = run_generation_benchmark(cfg)
        save_run(result, args.out_dir / f"matrix_{name}.json")
        print("wrote", name, "oom=" + str(any(r.get("oom") for r in result["runs"])))


if __name__ == "__main__":
    main()
