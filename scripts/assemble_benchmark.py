#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from crp_benchmark.constants import IMAGE_EXTENSIONS
from crp_benchmark.fileops import place_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assemble pre-generated class trees into the canonical benchmark layout."
    )
    parser.add_argument("--real-root", required=True)
    parser.add_argument("--pixel-root", required=True)
    parser.add_argument("--patch-root", required=True)
    parser.add_argument("--inpaint-root", required=True)
    parser.add_argument("--stylegan2-root", required=True)
    parser.add_argument("--geosynth-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--method", choices=("hardlink", "symlink", "copy"), default="hardlink")
    return parser.parse_args()


def iter_images(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def resolve_split_source(source_root: Path, split: str, label: str, source_name: str) -> Path:
    """Accept either class-aware trees or roots whose split folders contain images directly."""

    split_root = source_root / split
    candidates = (
        (split_root / "FAKE" / source_name, split_root / source_name, split_root)
        if label == "FAKE"
        else (split_root / label, split_root)
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"No usable {split}/{source_name} tree below {source_root}; tried "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    sources = {
        "REAL": Path(args.real_root).resolve(),
        "PIXEL": Path(args.pixel_root).resolve(),
        "PATCH": Path(args.patch_root).resolve(),
        "FAKE-inpaint": Path(args.inpaint_root).resolve(),
        "FAKE-StyleGAN2": Path(args.stylegan2_root).resolve(),
        "FAKE-GeoSynth": Path(args.geosynth_root).resolve(),
    }
    rows: list[dict] = []
    for source_name, source_root in sources.items():
        label = "FAKE" if source_name.startswith("FAKE-") else source_name
        for split in ("train", "validation", "test"):
            split_source = resolve_split_source(source_root, split, label, source_name)
            for path in iter_images(split_source):
                relative = path.relative_to(split_source)
                if label == "FAKE":
                    destination = output_root / split / label / source_name / relative
                else:
                    destination = output_root / split / label / relative
                place_file(path, destination, args.method)
                rows.append(
                    {
                        "path": str(destination),
                        "label": label,
                        "split": split,
                        "subtype": source_name if label == "FAKE" else "",
                        "source_id": relative.as_posix()
                        if label in {"REAL", "PIXEL", "PATCH"}
                        else "",
                        "source_path": str(path),
                    }
                )
    manifest = output_root / "benchmark_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"assembled {len(rows)} files at {output_root}")


if __name__ == "__main__":
    main()
