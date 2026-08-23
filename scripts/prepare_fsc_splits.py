#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from crp_benchmark.constants import CANONICAL_CLASSES, FSC_TEST_PER_CLASS, FSC_VAL_PER_CLASS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create source-image-ID-disjoint FSC adaptation/validation/test splits."
    )
    parser.add_argument("--inventory", required=True, help="CSV: path,label,source_id")
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--val-source-ids", type=int, default=FSC_VAL_PER_CLASS)
    parser.add_argument("--test-source-ids", type=int, default=FSC_TEST_PER_CLASS)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.inventory)
    required = {"path", "label", "source_id"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Inventory must contain {sorted(required)}")
    frame["label"] = frame["label"].str.upper()
    bad = sorted(set(frame.label) - set(CANONICAL_CLASSES))
    if bad:
        raise ValueError(f"Unknown FSC labels: {bad}")

    cross = frame.groupby(["source_id", "label"]).size().unstack(fill_value=0)
    for label in CANONICAL_CLASSES:
        if label not in cross.columns:
            raise ValueError(f"No rows for label {label}")
    complete_ids = cross.index[(cross[list(CANONICAL_CLASSES)] > 0).all(axis=1)].to_numpy()
    if len(complete_ids) < args.val_source_ids + args.test_source_ids:
        raise ValueError("Not enough complete source IDs for requested FSC split")
    rng = np.random.default_rng(args.seed)
    rng.shuffle(complete_ids)
    val_ids = set(complete_ids[: args.val_source_ids])
    test_ids = set(complete_ids[args.val_source_ids : args.val_source_ids + args.test_source_ids])
    frame["split"] = "adaptation_pool"
    frame.loc[frame.source_id.isin(val_ids), "split"] = "validation"
    frame.loc[frame.source_id.isin(test_ids), "split"] = "test"

    leakage = frame.groupby("source_id").split.nunique().max()
    if leakage != 1:
        raise AssertionError("A source ID was assigned to multiple FSC splits")
    counts = frame.groupby(["split", "label"]).size().unstack(fill_value=0)
    for split_name, expected in (
        ("validation", args.val_source_ids),
        ("test", args.test_source_ids),
    ):
        for label in CANONICAL_CLASSES:
            actual = int(counts.loc[split_name, label])
            if actual != expected:
                raise ValueError(
                    f"{split_name}/{label} has {actual} rows, expected {expected}. "
                    "The paper protocol assumes one variant per class per source ID."
                )
    output = Path(args.output_manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(counts)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
