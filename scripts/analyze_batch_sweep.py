#!/usr/bin/env python3
"""Aggregate repeated batch-size sweep passes into one analysis file."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def load_pass(d: Path) -> dict[int, dict]:
    out = {}
    for f in sorted(d.glob("*.json")):
        j = json.loads(f.read_text(encoding="utf-8"))
        runs = [r for r in j["runs"] if not r.get("oom")]
        b = int(j["config"]["batch_size"])
        out[b] = {
            "mean_ms": j["e2e_summary_ms"]["mean_ms"],
            "p50_ms": j["e2e_summary_ms"]["p50_ms"],
            "p90_ms": j["e2e_summary_ms"]["p90_ms"],
            "tps": statistics.mean(r["tokens_per_second"] for r in runs),
            "peak_mb": max(r["peak_memory_bytes"] for r in runs) / 2**20,
            "oom": len(runs) != len(j["runs"]),
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--passes", nargs="+", type=Path, required=True, help="pass dirs used in the aggregate")
    p.add_argument("--excluded", nargs="*", type=Path, default=[], help="pass dirs kept for reference only")
    p.add_argument("--out", type=Path, default=Path("results/batch_sweep_analysis.json"))
    args = p.parse_args()

    used = {d.name: load_pass(d) for d in args.passes}
    batches = sorted(set.intersection(*(set(v) for v in used.values())))
    agg = []
    for b in batches:
        cells = [used[k][b] for k in used]
        agg.append({
            "batch": b,
            "mean_ms": statistics.mean(c["mean_ms"] for c in cells),
            "mean_ms_spread_pct": 100 * (max(c["mean_ms"] for c in cells) / min(c["mean_ms"] for c in cells) - 1),
            "tps": statistics.mean(c["tps"] for c in cells),
            "peak_mb": max(c["peak_mb"] for c in cells),
            "ms_per_decode_step": statistics.mean(c["mean_ms"] for c in cells) / 128,
        })
    doc = {
        "config": {"model": "HuggingFaceTB/SmolLM-135M", "dtype": "bf16", "compile": False,
                   "prompt_length": 256, "max_new_tokens": 128, "warmup": 2, "iters": 5},
        "aggregate_of": list(used),
        "aggregate": agg,
        "passes": used,
        "excluded_passes": {d.name: load_pass(d) for d in args.excluded},
    }
    args.out.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    for r in agg:
        print(f"{r['batch']:>4}  {r['mean_ms']:8.0f} ms  {r['tps']:7.0f} tok/s  spread {r['mean_ms_spread_pct']:.1f}%  {r['peak_mb']:.0f} MB")


if __name__ == "__main__":
    main()