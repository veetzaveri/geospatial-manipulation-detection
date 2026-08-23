#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score

from crp_benchmark.stats import paired_binary_bootstrap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Binary, Multi-C (collapsed), and Multi-E (exact-family) scoring."
    )
    parser.add_argument(
        "--setting",
        required=True,
        choices=("in-domain", "fsc"),
        help="Select the paper's in-domain or FSC comparison protocol",
    )
    parser.add_argument("--attack", required=True, choices=("FAKE", "PIXEL", "PATCH"))
    parser.add_argument(
        "--binary-threshold-json",
        required=True,
        help="validation_threshold.json produced by the matching binary training run",
    )
    parser.add_argument("--binary-test", required=True)
    parser.add_argument("--multi-test", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--key-mode", choices=("path", "basename"), default="path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold-bootstrap", type=int, default=10_000)
    parser.add_argument("--auprc-bootstrap", type=int, default=1_000)
    return parser.parse_args()


def read_frame(path: str, key_mode: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"path", "true_label_name", "pred_label_name"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{path} lacks required columns {sorted(required)}")
    if key_mode == "basename":
        frame["sample_key"] = frame.apply(
            lambda row: f"{row.true_label_name}::{Path(row.path).name}", axis=1
        )
    else:
        frame["sample_key"] = frame.apply(
            lambda row: f"{row.true_label_name}::{Path(row.path).as_posix()}", axis=1
        )
    if frame.sample_key.duplicated().any():
        raise ValueError(
            f"Duplicate sample keys in {path}; use --key-mode path or make names unique"
        )
    return frame


def aligned(binary_path: str, multi_path: str, attack: str, key_mode: str) -> pd.DataFrame:
    binary = read_frame(binary_path, key_mode)
    multi = read_frame(multi_path, key_mode)
    binary = binary[binary.true_label_name.isin(["REAL", attack])].copy()
    multi = multi[multi.true_label_name.isin(["REAL", attack])].copy()
    binary_score_column = f"prob_{attack}"
    if binary_score_column not in binary.columns:
        raise ValueError(f"Binary predictions lack {binary_score_column}")
    for column in ("prob_REAL", binary_score_column):
        if column not in multi.columns:
            raise ValueError(f"Multi predictions lack {column}")
    merged = binary[["sample_key", "true_label_name", binary_score_column]].merge(
        multi[
            [
                "sample_key",
                "true_label_name",
                "pred_label_name",
                "prob_REAL",
                binary_score_column,
            ]
        ],
        on="sample_key",
        suffixes=("_binary", "_multi"),
        validate="one_to_one",
    )
    if len(merged) != len(binary) or len(merged) != len(multi):
        raise ValueError("Binary and multi files do not contain identical matched samples")
    if not (merged.true_label_name_binary == merged.true_label_name_multi).all():
        raise ValueError("True labels disagree after alignment")
    merged["label"] = (merged.true_label_name_binary == attack).astype(int)
    merged["score_binary"] = merged[f"{binary_score_column}_binary"]
    merged["score_multi_c"] = 1.0 - merged.prob_REAL
    merged["score_multi_e"] = merged[f"{binary_score_column}_multi"]
    merged["pred_multi_c"] = (merged.pred_label_name != "REAL").astype(int)
    merged["pred_multi_e"] = (merged.pred_label_name == attack).astype(int)
    return merged


def load_binary_threshold(path: str) -> float:
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    threshold = float(payload["threshold"])
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Binary threshold must be between 0 and 1")
    return threshold


def metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    scores: np.ndarray,
    decision_rule: str,
    *,
    threshold: float | None = None,
) -> dict:
    output = {
        "decision_rule": decision_rule,
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "f1_attack": float(f1_score(labels, predictions, pos_label=1, zero_division=0)),
        "auprc": float(average_precision_score(labels, scores)),
    }
    if threshold is not None:
        output["threshold"] = float(threshold)
    return output


def main() -> None:
    args = parse_args()
    test = aligned(args.binary_test, args.multi_test, args.attack, args.key_mode)
    binary_threshold = load_binary_threshold(args.binary_threshold_json)
    labels = test.label.to_numpy(dtype=int)
    pred_binary = (test.score_binary.to_numpy() >= binary_threshold).astype(int)
    pred_multi_c = test.pred_multi_c.to_numpy(dtype=int)
    pred_multi_e = test.pred_multi_e.to_numpy(dtype=int)
    results = {
        "binary": metrics(
            labels,
            pred_binary,
            test.score_binary.to_numpy(),
            "validation-selected attack-F1 threshold",
            threshold=binary_threshold,
        ),
        "multi_c": metrics(
            labels,
            pred_multi_c,
            test.score_multi_c.to_numpy(),
            "four-way argmax collapsed as predicted class != REAL",
        ),
        "multi_e": metrics(
            labels,
            pred_multi_e,
            test.score_multi_e.to_numpy(),
            f"four-way argmax exactly equals {args.attack}",
        ),
    }
    stratified = args.setting == "fsc"
    results["paired_binary_vs_multi_c"] = paired_binary_bootstrap(
        labels,
        pred_binary,
        test.score_binary.to_numpy(),
        pred_multi_c,
        test.score_multi_c.to_numpy(),
        replicates_thresholded=args.threshold_bootstrap,
        replicates_auprc=args.auprc_bootstrap,
        seed=args.seed,
        stratified=stratified,
    )
    results["paired_binary_vs_multi_e"] = paired_binary_bootstrap(
        labels,
        pred_binary,
        test.score_binary.to_numpy(),
        pred_multi_e,
        test.score_multi_e.to_numpy(),
        replicates_thresholded=args.threshold_bootstrap,
        replicates_auprc=args.auprc_bootstrap,
        seed=args.seed,
        stratified=stratified,
    )
    results["protocol"] = {
        "setting": args.setting,
        "attack": args.attack,
        "primary_multi_rule": "multi_c" if args.setting == "in-domain" else "multi_e",
        "multi_c_hard_decision": "four-way argmax != REAL",
        "multi_c_auprc_score": "1 - p(REAL)",
        "multi_e_hard_decision": f"four-way argmax == {args.attack}",
        "multi_e_auprc_score": f"p({args.attack})",
        "binary_threshold_selection": "validation attack-class F1",
        "binary_threshold_file": str(Path(args.binary_threshold_json)),
        "bootstrap_sampling": "class_stratified" if stratified else "ordinary_image_level",
        "mcnemar": "paired hard-decision correctness, exact two-sided binomial",
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    test.to_csv(output_dir / "aligned_test_scores.csv", index=False)
    with (output_dir / "comparison.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
