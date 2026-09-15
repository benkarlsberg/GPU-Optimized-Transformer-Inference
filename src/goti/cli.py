from __future__ import annotations

import argparse
import json
from pathlib import Path

from goti.env import collect_environment


def record_env_main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Record software/GPU environment")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(collect_environment(), indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


def run_experiment_main(argv: list[str] | None = None) -> None:
    import yaml

    from goti.benchmarks.harness import run_generation_benchmark, save_run

    p = argparse.ArgumentParser(description="Run a config-driven experiment")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    result = run_generation_benchmark(cfg)
    out = args.out or Path("results/runs") / f"{args.config.stem}.json"
    save_run(result, out)
    print(f"wrote {out}")
