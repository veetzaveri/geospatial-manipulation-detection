from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import BenchmarkDataset, ManifestDataset
from .models import create_model, load_experiment_checkpoint
from .reproducibility import dataloader_generator, seed_worker


def choose_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def make_loader(
    dataset,
    *,
    batch_size: int,
    workers: int,
    shuffle: bool,
    seed: int = 42,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        worker_init_fn=seed_worker,
        generator=dataloader_generator(seed),
    )


def dataset_from_args(
    *,
    root: str | None,
    manifest: str | None,
    split: str | None,
    transform,
    binary_attack: str | None = None,
    include_fake_subtypes=None,
):
    if bool(root) == bool(manifest):
        raise ValueError("Provide exactly one of a dataset root or manifest")
    if root:
        return BenchmarkDataset(
            root,
            transform,
            binary_attack=binary_attack,
            include_fake_subtypes=include_fake_subtypes,
        )
    return ManifestDataset(
        manifest,
        transform,
        split=split,
        binary_attack=binary_attack,
    )


def model_from_checkpoint(
    model_key: str,
    checkpoint: str,
    *,
    num_classes: int,
    device: torch.device,
    pretrained_checkpoint: str | None = None,
):
    model, spec = create_model(
        model_key,
        num_classes,
        pretrained=False,
        pretrained_checkpoint=pretrained_checkpoint,
    )
    metadata = load_experiment_checkpoint(model, checkpoint, strict=True)
    model.to(device).eval()
    return model, spec, metadata


def write_json(path: str | Path, payload: dict | list) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
