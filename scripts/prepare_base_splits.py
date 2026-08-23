#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from crp_benchmark.constants import IMAGE_EXTENSIONS
from crp_benchmark.fileops import place_file

SPLIT_COUNTS_DM_AER = {"train": 20_000, "validation": 2_500, "test": 2_500}
SPLIT_COUNTS_UC_PER_CATEGORY = {"train": 80, "validation": 10, "test": 10}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the split-specific REAL base pools reported in the paper."
    )
    parser.add_argument("--dm-aer-real-root", required=True)
    parser.add_argument("--uc-merced-root", required=True, help="Root with 21 category folders")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--method", choices=("hardlink", "symlink", "copy"), default="hardlink")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def images_below(root: Path) -> list[Path]:
    return [
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def allocate(paths: list[Path], counts: dict[str, int], rng: random.Random):
    required = sum(counts.values())
    if len(paths) < required:
        raise ValueError(f"Need {required} images but found {len(paths)}")
    shuffled = paths.copy()
    rng.shuffle(shuffled)
    offset = 0
    for split, count in counts.items():
        yield split, shuffled[offset : offset + count]
        offset += count


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    dm_root = Path(args.dm_aer_real_root).resolve()
    uc_root = Path(args.uc_merced_root).resolve()
    output_root = Path(args.output_root).resolve()
    rows: list[dict] = []

    for split, selected in allocate(images_below(dm_root), SPLIT_COUNTS_DM_AER, rng):
        for source in selected:
            destination = output_root / split / "REAL" / "DM-AER" / source.relative_to(dm_root)
            place_file(source, destination, args.method)
            rows.append(
                {
                    "path": str(destination),
                    "label": "REAL",
                    "split": split,
                    "source_id": f"DM-AER:{source.relative_to(dm_root).as_posix()}",
                    "source_dataset": "DM-AER",
                    "category": "",
                    "original_path": str(source),
                }
            )

    categories = sorted(path for path in uc_root.iterdir() if path.is_dir())
    if len(categories) != 21:
        raise ValueError(f"Expected 21 UC Merced categories, found {len(categories)}")
    for category in categories:
        for split, selected in allocate(images_below(category), SPLIT_COUNTS_UC_PER_CATEGORY, rng):
            for source in selected:
                destination = (
                    output_root / split / "REAL" / "UC-Merced" / category.name / source.name
                )
                place_file(source, destination, args.method)
                rows.append(
                    {
                        "path": str(destination),
                        "label": "REAL",
                        "split": split,
                        "source_id": f"UC-Merced:{category.name}:{source.name}",
                        "source_dataset": "UC-Merced",
                        "category": category.name,
                        "original_path": str(source),
                    }
                )
    manifest = output_root / "base_split_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} REAL samples and {manifest}")


if __name__ == "__main__":
    main()
