#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from crp_benchmark.stats import percentile_interval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize FSC Binary/Multi-C/Multi-E metrics and paired deltas over "
            "model-attack units."
        )
    )
    parser.add_argument("--comparison-json", nargs="+", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    unit_rows: list[dict] = []
    for filename in args.comparison_json:
        with Path(filename).open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("protocol", {}).get("setting") != "fsc":
            raise ValueError(f"{filename} is not an FSC comparison output")
        row: dict[str, str | float] = {"unit": filename}
        for scoring in ("binary", "multi_c", "multi_e"):
            for metric in ("balanced_accuracy", "f1_attack", "auprc"):
                row[f"{scoring}_{metric}"] = float(payload[scoring][metric])
        unit_rows.append(row)

    units = pd.DataFrame(unit_rows)
    if units.empty:
        raise ValueError("At least one comparison JSON is required")
    if len(units) != 15:
        raise ValueError(
            "Paper FSC summaries require 15 model-attack units (5 models x 3 attacks); "
            f"received {len(units)}"
        )
    rng = np.random.default_rng(args.seed)
    output_rows: list[dict] = []
    for metric in ("balanced_accuracy", "f1_attack", "auprc"):
        binary = units[f"binary_{metric}"].to_numpy(dtype=float)
        output_rows.append(
            {
                "scoring": "binary",
                "metric": metric,
                "n_model_attack_units": len(binary),
                "mean": float(np.mean(binary)),
                "delta_vs_binary": 0.0,
                "delta_ci_low": np.nan,
                "delta_ci_high": np.nan,
                "bootstrap_replicates": 0,
                "bootstrap_sampling": "not_applicable",
            }
        )
        for scoring in ("multi_c", "multi_e"):
            values = units[f"{scoring}_{metric}"].to_numpy(dtype=float)
            deltas = values - binary
            boot = np.empty(args.bootstrap_replicates, dtype=float)
            for replicate in range(args.bootstrap_replicates):
                indices = rng.choice(len(deltas), size=len(deltas), replace=True)
                boot[replicate] = np.mean(deltas[indices])
            low, high = percentile_interval(boot)
            output_rows.append(
                {
                    "scoring": scoring,
                    "metric": metric,
                    "n_model_attack_units": len(values),
                    "mean": float(np.mean(values)),
                    "delta_vs_binary": float(np.mean(deltas)),
                    "delta_ci_low": low,
                    "delta_ci_high": high,
                    "bootstrap_replicates": args.bootstrap_replicates,
                    "bootstrap_sampling": "paired_model_attack_units",
                }
            )

    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(output_rows)
    result.to_csv(output, index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
