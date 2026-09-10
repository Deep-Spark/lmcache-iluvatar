#!/usr/bin/env python3
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Print a TTFT summary table from bench_serving JSONL result files."""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize bench_serving TTFT JSONL files.")
    parser.add_argument(
        "--results-dir",
        default=os.environ.get("RESULTS_DIR", ""),
        help="Directory containing *.jsonl result files (default: $RESULTS_DIR).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=60,
        help="Tag column width (default: 60).",
    )
    args = parser.parse_args()

    if not args.results_dir:
        return 0

    files = sorted(glob.glob(os.path.join(args.results_dir, "*.jsonl")))
    if not files:
        print("No result files found.", file=sys.stderr)
        return 0

    hdr = f"{'Config':<{args.width}} {'mean':>8} {'p50':>8} {'p99':>8} {'concur':>8}"
    print(hdr)
    print("-" * len(hdr))
    for path in files:
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            tag = os.path.basename(path).replace(".jsonl", "")
            mean = data.get("mean_ttft_ms", float("nan"))
            p50 = data.get("median_ttft_ms", float("nan"))
            p99 = data.get("p99_ttft_ms", float("nan"))
            conc = data.get("concurrency", float("nan"))
            print(f"{tag:<{args.width}} {mean:>8.1f} {p50:>8.1f} {p99:>8.1f} {conc:>8.1f}")
        except Exception as exc:
            print(f"  (could not parse {path}: {exc})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
