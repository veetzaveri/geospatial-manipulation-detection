#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image

from crp_benchmark.constants import IMAGE_EXTENSIONS
from crp_benchmark.generation import (
    INPAINT_PROMPTS,
    create_local_mask,
    deterministic_prompt,
    feathered_hard_composite,
    generate_inpainted_image,
    load_inpainting_pipeline,
)
from crp_benchmark.runtime import choose_device
from crp_benchmark.seeding import stable_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate localized FAKE-inpaint samples.")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True, help="Final feather-composited images")
    parser.add_argument("--raw-root", required=True, help="Raw diffusion outputs")
    parser.add_argument("--mask-root", required=True, help="Saved binary masks")
    parser.add_argument(
        "--model-id",
        default="runwayml/stable-diffusion-inpainting",
        help="Reconstructed default; confirm against the unavailable original script.",
    )
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--strength", type=float, default=0.95)
    parser.add_argument("--feather-radius", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    raw_root = Path(args.raw_root).resolve()
    mask_root = Path(args.mask_root).resolve()
    for directory in (output_root, raw_root, mask_root):
        directory.mkdir(parents=True, exist_ok=True)
    device = str(choose_device(args.device))
    pipeline = load_inpainting_pipeline(args.model_id, device)
    rows: list[dict] = []
    images = [
        path
        for path in sorted(input_root.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    for index, source_path in enumerate(images, start=1):
        relative = source_path.relative_to(input_root)
        output_path = output_root / relative
        raw_path = raw_root / relative
        mask_path = (mask_root / relative).with_suffix(".png")
        for path in (output_path, raw_path, mask_path):
            path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists() and not args.overwrite:
            continue
        with Image.open(source_path) as handle:
            original = handle.convert("RGB").copy()
        mask = create_local_mask(relative.as_posix(), seed=args.seed)
        prompt = deterministic_prompt(relative.as_posix(), INPAINT_PROMPTS, args.seed)
        image_seed = stable_seed(relative.as_posix(), args.seed)
        raw = generate_inpainted_image(
            pipeline,
            original,
            mask,
            prompt=prompt,
            seed=image_seed,
            device=device,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            strength=args.strength,
        )
        mask.save(mask_path)
        raw.save(raw_path)
        final = feathered_hard_composite(original, raw, mask, feather_radius=args.feather_radius)
        final.save(output_path)
        rows.append(
            {
                "source": str(source_path),
                "raw": str(raw_path),
                "mask": str(mask_path),
                "output": str(output_path),
                "prompt": prompt,
                "seed": image_seed,
                "steps": args.steps,
                "guidance_scale": args.guidance_scale,
                "strength": args.strength,
                "feather_radius": args.feather_radius,
                "model_id": args.model_id,
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
