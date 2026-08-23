#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image
from torchvision.transforms.functional import pil_to_tensor, to_pil_image

from crp_benchmark.attacks import (
    apply_pixel_perturbation,
    gradient_saliency_coordinates,
    load_faster_rcnn,
    random_coordinates,
)
from crp_benchmark.constants import IMAGE_EXTENSIONS
from crp_benchmark.runtime import choose_device
from crp_benchmark.seeding import stable_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate PIXEL manipulation images.")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--placement", choices=("oracle", "random"), default="oracle")
    parser.add_argument("--budget", type=int, default=200)
    parser.add_argument("--patch-size", type=int, default=5)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--dummy-weight", type=float, default=0.05)
    parser.add_argument("--box-score-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    detector = load_faster_rcnn(device) if args.placement == "oracle" else None
    rows: list[dict] = []

    images = [
        path
        for path in sorted(input_root.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    for index, source_path in enumerate(images, start=1):
        relative = source_path.relative_to(input_root)
        output_path = output_root / relative
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists() and not args.overwrite:
            continue
        with Image.open(source_path) as source:
            tensor = pil_to_tensor(source.convert("RGB")).float().div(255.0).to(device)
        image_seed = stable_seed(relative.as_posix(), args.seed)
        if detector is not None:
            coordinates = gradient_saliency_coordinates(
                detector,
                tensor,
                budget=args.budget,
                dummy_weight=args.dummy_weight,
                score_threshold=args.box_score_threshold,
                fallback_seed=image_seed,
            )
        else:
            coordinates = random_coordinates(tensor, budget=args.budget, seed=image_seed)
        result = apply_pixel_perturbation(
            tensor,
            coordinates,
            patch_size=args.patch_size,
            epsilon=args.epsilon,
        )
        to_pil_image(result.cpu()).save(output_path)
        rows.append(
            {
                "source": str(source_path),
                "output": str(output_path),
                "placement": args.placement,
                "budget": args.budget,
                "patch_size": args.patch_size,
                "epsilon": args.epsilon,
                "dummy_weight": args.dummy_weight,
                "seed": image_seed,
            }
        )
        print(f"[{index}/{len(images)}] {relative}")

    with (output_root / "generation_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fieldnames = list(rows[0]) if rows else ["source", "output"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
