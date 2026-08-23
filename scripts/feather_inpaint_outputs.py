#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from crp_benchmark.constants import IMAGE_EXTENSIONS
from crp_benchmark.generation import feathered_hard_composite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply the 8-pixel feathered hard composite.")
    parser.add_argument("--original-root", required=True)
    parser.add_argument("--raw-inpaint-root", required=True)
    parser.add_argument("--mask-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--feather-radius", type=float, default=8.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    original_root = Path(args.original_root).resolve()
    raw_root = Path(args.raw_inpaint_root).resolve()
    mask_root = Path(args.mask_root).resolve()
    output_root = Path(args.output_root).resolve()
    originals = [
        path
        for path in sorted(original_root.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    for source_path in originals:
        relative = source_path.relative_to(original_root)
        raw_path = raw_root / relative
        mask_path = (mask_root / relative).with_suffix(".png")
        output_path = output_root / relative
        if output_path.exists() and not args.overwrite:
            continue
        if not raw_path.exists() or not mask_path.exists():
            raise FileNotFoundError(f"Missing raw/mask pair for {relative}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with (
            Image.open(source_path) as original,
            Image.open(raw_path) as raw,
            Image.open(mask_path) as mask,
        ):
            final = feathered_hard_composite(
                original, raw, mask, feather_radius=args.feather_radius
            )
        final.save(output_path)
        print(relative)


if __name__ == "__main__":
    main()
