#!/usr/bin/env python3
"""Latency vs. throughput tradeoff curve from the batch-size sweep."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


OFFSETS = {32: (-6, -12), 64: (-34, 2), 128: (-40, 0), 256: (-40, 0)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=Path("results/batch_sweep_analysis.json"))
    p.add_argument("--out", type=Path, default=Path("paper/figs/latency_throughput_tradeoff.pdf"))
    args = p.parse_args()

    agg = json.loads(args.input.read_text())["aggregate"]
    tps = [r["tps"] for r in agg]
    lat = [r["mean_ms"] / 1000 for r in agg]

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.plot(tps, lat, "-o", color="tab:blue", ms=5)
    for r, x, y in zip(agg, tps, lat):
        ax.annotate(f"b={r['batch']}", (x, y), textcoords="offset points",
                    xytext=OFFSETS.get(r["batch"], (-6, 7)), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Throughput (generated tokens/s, log scale)")
    ax.set_ylabel("End-to-end latency (s)")
    ax.set_title("Batch-size sweep: BF16, eager, prompt 256, 128 new tokens")
    ax.set_ylim(0, max(lat) * 1.12)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
