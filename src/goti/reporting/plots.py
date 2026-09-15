from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def load_runs(input_dir: Path) -> list[dict[str, Any]]:
    runs = []
    for p in sorted(input_dir.glob("*.json")):
        runs.append(json.loads(p.read_text(encoding="utf-8")))
    return runs


def plot_e2e_bars(runs: list[dict[str, Any]], out: Path) -> None:
    labels = []
    means = []
    for r in runs:
        summary = r.get("e2e_summary_ms") or {}
        if "mean_ms" not in summary:
            continue
        cfg = r.get("config", {})
        labels.append(f"{cfg.get('dtype', '?')}/b{cfg.get('batch_size', '?')}")
        means.append(summary["mean_ms"])
    if not means:
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(means)), means)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("E2E latency mean (ms)")
    ax.set_title("E2E latency by config")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
