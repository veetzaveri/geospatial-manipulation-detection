#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from crp_benchmark.stats import bootstrap_triage, zero_event_upper_false_alarms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute the paper's low-prevalence collapsed manipulation triage metrics."
    )
    parser.add_argument("--predictions", required=True, help="Four-way predictions CSV")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--prevalence", type=float, nargs="+", default=(0.01, 0.05, 0.10))
    parser.add_argument("--target-recall", type=float, default=0.90)
    parser.add_argument("--bootstrap-replicates", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.predictions)
    required = {"true_label_name", "prob_REAL"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Predictions must contain {sorted(required)}")
    labels = (frame.true_label_name != "REAL").to_numpy(dtype=int)
    scores = 1.0 - frame.prob_REAL.to_numpy(dtype=float)
    results = {}
    for prevalence in args.prevalence:
        key = f"prevalence_{prevalence:g}"
        result = bootstrap_triage(
            labels,
            scores,
            prevalence=prevalence,
            target_recall=args.target_recall,
            replicates=args.bootstrap_replicates,
            seed=args.seed,
        )
        if result["observed_fpr"] == 0.0:
            negatives = int(np.sum(labels == 0))
            result["exact_one_sided_95pct_upper_false_alarms_per_10000"] = (
                zero_event_upper_false_alarms(negatives, prevalence=prevalence, confidence=0.95)
            )
            result["zero_false_positive_negative_count"] = negatives
        results[key] = result
    results["score_definition"] = "p(PIXEL)+p(PATCH)+p(FAKE)=1-p(REAL)"
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
