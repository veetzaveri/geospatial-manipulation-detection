#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image
from torchvision.transforms.functional import pil_to_tensor, to_pil_image

from crp_benchmark.attacks import load_faster_rcnn, optimize_adversarial_patch
from crp_benchmark.constants import IMAGE_EXTENSIONS
from crp_benchmark.runtime import choose_device
from crp_benchmark.seeding import stable_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate learnable PATCH manipulations.")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--geometry", choices=("circle", "square"), default="circle")
    parser.add_argument("--radius", type=int, default=5)
    parser.add_argument("--num-patches", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--step-size", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.75)
    parser.add_argument("--tv-weight", type=float, default=1e-4)
    parser.add_argument("--stop-confidence", type=float, default=0.05)
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
    detector = load_faster_rcnn(device)
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
        result = optimize_adversarial_patch(
            detector,
            tensor,
            geometry=args.geometry,
            radius=args.radius,
            num_patches=args.num_patches,
            iterations=args.iterations,
            step_size=args.step_size,
            momentum=args.momentum,
            tv_weight=args.tv_weight,
            stop_confidence=args.stop_confidence,
            box_score_threshold=args.box_score_threshold,
            seed=image_seed,
        )
        to_pil_image(result.image.cpu()).save(output_path)
        rows.append(
            {
                "source": str(source_path),
                "output": str(output_path),
                "geometry": args.geometry,
                "radius": args.radius,
                "num_patches": args.num_patches,
                "iterations_run": result.iterations,
                "final_confidence": result.final_confidence,
                "centers": repr(result.centers),
                "step_size": args.step_size,
                "momentum": args.momentum,
                "tv_weight": args.tv_weight,
                "stop_confidence": args.stop_confidence,
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
