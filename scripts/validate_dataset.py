#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from crp_benchmark.constants import FAKE_SUBTYPES, IMAGE_EXTENSIONS

EXPECTED_CLASS_COUNTS = {
    "train": {"REAL": 21680, "PIXEL": 21680, "PATCH": 21680, "FAKE": 21680},
    "validation": {"REAL": 2710, "PIXEL": 2710, "PATCH": 2710, "FAKE": 2710},
    "test": {"REAL": 2710, "PIXEL": 2710, "PATCH": 2710, "FAKE": 2710},
}
EXPECTED_FAKE_COUNTS = {
    "train": {"FAKE-inpaint": 10840, "FAKE-StyleGAN2": 5420, "FAKE-GeoSynth": 5420},
    "validation": {"FAKE-inpaint": 1355, "FAKE-StyleGAN2": 677, "FAKE-GeoSynth": 678},
    "test": {"FAKE-inpaint": 1355, "FAKE-StyleGAN2": 677, "FAKE-GeoSynth": 678},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the published benchmark layout/counts.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-json")
    parser.add_argument("--skip-exact-count-check", action="store_true")
    return parser.parse_args()


def paths(root: Path) -> list[Path]:
    return [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def relative_keys(root: Path) -> set[str]:
    return {path.relative_to(root).as_posix() for path in paths(root)}


def main() -> None:
    args = parse_args()
    root = Path(args.dataset_root).resolve()
    report: dict = {"dataset_root": str(root), "splits": {}, "errors": []}
    split_source_ids: dict[str, set[str]] = {}
    for split in ("train", "validation", "test"):
        split_root = root / split
        split_report = {"class_counts": {}, "fake_subtype_counts": {}}
        for label in ("REAL", "PIXEL", "PATCH", "FAKE"):
            count = len(paths(split_root / label))
            split_report["class_counts"][label] = count
            if not args.skip_exact_count_check and count != EXPECTED_CLASS_COUNTS[split][label]:
                report["errors"].append(
                    f"{split}/{label}: expected {EXPECTED_CLASS_COUNTS[split][label]}, "
                    f"found {count}"
                )
        for subtype in FAKE_SUBTYPES:
            count = len(paths(split_root / "FAKE" / subtype))
            split_report["fake_subtype_counts"][subtype] = count
            if not args.skip_exact_count_check and count != EXPECTED_FAKE_COUNTS[split][subtype]:
                report["errors"].append(
                    f"{split}/{subtype}: expected {EXPECTED_FAKE_COUNTS[split][subtype]}, "
                    f"found {count}"
                )

        real_keys = relative_keys(split_root / "REAL")
        for label in ("PIXEL", "PATCH"):
            label_keys = relative_keys(split_root / label)
            if label_keys != real_keys:
                report["errors"].append(
                    f"{split}: REAL and {label} relative filenames are not matched "
                    f"({len(real_keys - label_keys)} missing, {len(label_keys - real_keys)} extra)"
                )
        split_source_ids[split] = real_keys
        report["splits"][split] = split_report

    for first, second in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = split_source_ids[first] & split_source_ids[second]
        if overlap:
            report["errors"].append(
                f"REAL relative identifiers overlap between {first} and {second}: {len(overlap)}"
            )
    report["valid"] = not report["errors"]
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
