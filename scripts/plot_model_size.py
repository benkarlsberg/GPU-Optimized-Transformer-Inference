#!/usr/bin/env python3
"""Latency and throughput versus batch size for the model-size sweep."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

STYLE = {
    "SmolLM-135M (dynamic cache)": dict(color="tab:blue", marker="o", ls="-"),
    "SmolLM-360M (dynamic cache)": dict(color="tab:orange", marker="s", ls="-"),
    "SmolLM-1.7B (dynamic cache)": dict(color="tab:green", marker="^", ls="-"),
    "SmolLM-1.7B (static cache)": dict(color="tab:green", marker="v", ls="--"),
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=Path("results/model_size_sweep_analysis.json"))
    p.add_argument("--out", type=Path, default=Path("paper/figs/model_size_sweep.pdf"))
    args = p.parse_args()
    series = json.loads(args.input.read_text())["series"]

    fig, (ax_l, ax_t) = plt.subplots(1, 2, figsize=(9.0, 3.6))
    for name, cells in series.items():
        ok = [c for c in cells if not c["spilled_to_system_ram"]]
        b = [c["batch"] for c in ok]
        st = STYLE[name]
        label = name.replace("SmolLM-", "")
        ax_l.plot(b, [c["e2e_mean_ms"] / 1000 for c in ok], label=label, ms=4, **st)
        ax_t.plot(b, [c["tokens_per_second"] for c in ok], label=label, ms=4, **st)
    for ax in (ax_l, ax_t):
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8, 16, 32, 64, 128, 256])
        ax.set_xticklabels(["1", "2", "4", "8", "16", "32", "64", "128", "256"])
        ax.set_xlabel("Batch size")
        ax.grid(alpha=0.3, which="both")
    ax_l.set_ylabel("End-to-end latency (s)")
    ax_l.set_yscale("log")
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    ax_l.yaxis.set_major_locator(FixedLocator([1.5, 2, 3, 5, 10, 20]))
    ax_l.yaxis.set_minor_locator(NullLocator())
    ax_l.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax_l.set_title("Latency (log scale)")
    ax_t.set_ylabel("Generated tokens/s")
    ax_t.set_title("Throughput")
    ax_t.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
