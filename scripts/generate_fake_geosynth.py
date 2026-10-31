#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image

from crp_benchmark.constants import IMAGE_EXTENSIONS
from crp_benchmark.generation import (
    GEOSYNTH_PROMPTS,
    deterministic_prompt,
    generate_geosynth_image,
    load_geosynth_pipeline,
)
from crp_benchmark.runtime import choose_device
from crp_benchmark.seeding import stable_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate global FAKE-GeoSynth samples.")
    parser.add_argument(
        "--reference-root", required=True, help="Controls filenames/count/output sizes"
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--model-id", default="MVRL/GeoSynth")
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--guidance-scale", type=float, default=5.5)
    parser.add_argument(
        "--bottom-crop-pixels",
        type=int,
        default=8,
        help="Pixels cropped from the bottom before final resizing (default: 8).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference_root = Path(args.reference_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    device = str(choose_device(args.device))
    pipeline = load_geosynth_pipeline(args.model_id, device)
    rows: list[dict] = []
    references = [
        path
        for path in sorted(reference_root.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    for index, reference_path in enumerate(references, start=1):
        relative = reference_path.relative_to(reference_root)
        output_path = output_root / relative
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists() and not args.overwrite:
            continue
        with Image.open(reference_path) as handle:
            output_size = handle.size
        prompt = deterministic_prompt(relative.as_posix(), GEOSYNTH_PROMPTS, args.seed)
        image_seed = stable_seed(relative.as_posix(), args.seed)
        image = generate_geosynth_image(
            pipeline,
            prompt=prompt,
            seed=image_seed,
            device=device,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            bottom_crop_pixels=args.bottom_crop_pixels,
            output_size=output_size,
        )
        image.save(output_path)
        rows.append(
            {
                "reference": str(reference_path),
                "output": str(output_path),
                "prompt": prompt,
                "seed": image_seed,
                "steps": args.steps,
                "guidance_scale": args.guidance_scale,
                "bottom_crop_pixels": args.bottom_crop_pixels,
                "model_id": args.model_id,
            }
        )
        print(f"[{index}/{len(references)}] {relative}")
    with (output_root / "generation_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fieldnames = list(rows[0]) if rows else ["reference", "output"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
