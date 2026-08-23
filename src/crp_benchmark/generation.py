from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .seeding import stable_seed

# The original prompt-bank file was not recoverable. These prompts implement
# the paper's stated fixed bank of overhead satellite-scene descriptions and
# are identified as reconstructed in docs/PROVENANCE.md.
INPAINT_PROMPTS = (
    "high-resolution overhead satellite image of buildings, roads, and vegetation",
    "realistic aerial imagery of an urban area viewed directly from above",
    "overhead remote-sensing image with natural terrain and human-made structures",
    "realistic satellite photograph of a mixed residential and commercial landscape",
    "high-detail aerial view of fields, roads, trees, and small structures",
)

GEOSYNTH_PROMPTS = (
    "a high-resolution satellite image of an urban landscape",
    "overhead satellite imagery of agricultural fields and roads",
    "a realistic remote-sensing image of a coastal landscape",
    "overhead aerial image of forest, buildings, and transportation infrastructure",
    "a realistic satellite image of an industrial and residential area",
)


def deterministic_prompt(path: str | Path, prompts: tuple[str, ...], seed: int = 42) -> str:
    rng = random.Random(stable_seed(path, seed))
    return prompts[rng.randrange(len(prompts))]


def create_local_mask(
    key: str | Path,
    *,
    size: tuple[int, int] = (512, 512),
    seed: int = 42,
    min_fraction: float = 0.18,
    max_fraction: float = 0.35,
    max_area_fraction: float = 0.35,
) -> Image.Image:
    rng = random.Random(stable_seed(key, seed))
    width, height = size
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    for _ in range(rng.randint(2, 3)):
        region_width = max(1, round(width * rng.uniform(min_fraction, max_fraction)))
        region_height = max(1, round(height * rng.uniform(min_fraction, max_fraction)))
        x1 = rng.randint(0, max(0, width - region_width))
        y1 = rng.randint(0, max(0, height - region_height))
        coordinates = (x1, y1, x1 + region_width, y1 + region_height)
        if rng.random() < 0.5:
            draw.rectangle(coordinates, fill=255)
        else:
            draw.ellipse(coordinates, fill=255)

    area_fraction = float(np.asarray(mask, dtype=np.float32).mean() / 255.0)
    if area_fraction > max_area_fraction:
        mask = Image.new("L", size, 0)
        draw = ImageDraw.Draw(mask)
        ellipse_width = round(width * 0.30)
        ellipse_height = round(height * 0.24)
        center_x, center_y = width // 2, height // 2
        draw.ellipse(
            (
                center_x - ellipse_width // 2,
                center_y - ellipse_height // 2,
                center_x + ellipse_width // 2,
                center_y + ellipse_height // 2,
            ),
            fill=255,
        )
    return mask


def feathered_hard_composite(
    original: Image.Image,
    inpainted: Image.Image,
    binary_mask: Image.Image,
    *,
    feather_radius: float = 8.0,
) -> Image.Image:
    """Reconstruct the final FAKE-inpaint image described in the paper."""

    original = original.convert("RGB")
    inpainted = inpainted.convert("RGB").resize(original.size, Image.Resampling.BICUBIC)
    binary_mask = binary_mask.convert("L").resize(original.size, Image.Resampling.NEAREST)
    alpha = binary_mask.filter(ImageFilter.GaussianBlur(radius=feather_radius))
    return Image.composite(inpainted, original, alpha)


def load_inpainting_pipeline(
    model_id: str,
    device: str,
):
    import torch
    from diffusers import StableDiffusionInpaintPipeline

    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    pipeline = StableDiffusionInpaintPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,
    )
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)
    return pipeline


def generate_inpainted_image(
    pipeline,
    source: Image.Image,
    mask: Image.Image,
    *,
    prompt: str,
    seed: int,
    device: str,
    num_inference_steps: int = 15,
    guidance_scale: float = 7.5,
    strength: float = 0.95,
) -> Image.Image:
    import torch

    canvas = source.convert("RGB").resize((512, 512), Image.Resampling.BICUBIC)
    mask = mask.convert("L").resize((512, 512), Image.Resampling.NEAREST)
    generator_device = "cuda" if device.startswith("cuda") else "cpu"
    generator = torch.Generator(device=generator_device).manual_seed(seed)
    return pipeline(
        prompt=prompt,
        image=canvas,
        mask_image=mask,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        strength=strength,
        generator=generator,
    ).images[0]


def load_geosynth_pipeline(model_id: str, device: str):
    import torch
    from diffusers import DiffusionPipeline

    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    pipeline = DiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)
    return pipeline


def generate_geosynth_image(
    pipeline,
    *,
    prompt: str,
    seed: int,
    device: str,
    num_inference_steps: int = 15,
    guidance_scale: float = 5.5,
    bottom_crop_pixels: int = 8,
    output_size: tuple[int, int] = (512, 512),
) -> Image.Image:
    import torch

    generator_device = "cuda" if device.startswith("cuda") else "cpu"
    generator = torch.Generator(device=generator_device).manual_seed(seed)
    generated = (
        pipeline(
            prompt=prompt,
            height=512,
            width=512,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        .images[0]
        .convert("RGB")
    )
    if bottom_crop_pixels:
        generated = generated.crop((0, 0, generated.width, generated.height - bottom_crop_pixels))
    return generated.resize(output_size, Image.Resampling.BICUBIC)
