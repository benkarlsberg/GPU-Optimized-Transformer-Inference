#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from goti.reporting.plots import load_runs, plot_e2e_bars


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    runs = load_runs(args.input)
    plot_e2e_bars(runs, args.out / "e2e_bars.png")
    print("wrote plots under", args.out)


if __name__ == "__main__":
    main()
