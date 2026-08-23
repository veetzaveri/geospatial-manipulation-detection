from __future__ import annotations

import csv
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from .constants import (
    CANONICAL_CLASSES,
    CLASS_TO_INDEX,
    FAKE_SUBTYPES,
    IMAGE_EXTENSIONS,
    IMAGENET_MEAN,
    IMAGENET_STD,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True


class AddGaussianNoise:
    def __init__(self, std: float = 0.01) -> None:
        self.std = float(std)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor + torch.randn_like(tensor) * self.std).clamp(0.0, 1.0)


def build_transform(
    regime: str = "none",
    image_size: int = 224,
    *,
    normalize: bool = True,
    interpolation: InterpolationMode = InterpolationMode.BICUBIC,
) -> Callable:
    """Construct preprocessing and the three augmentation regimes in the paper."""

    regime = regime.lower().replace("_", "").replace("-", "")
    ops: list[Callable] = [transforms.Resize((image_size, image_size), interpolation=interpolation)]

    if regime in {"light", "lightaug"}:
        ops.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05),
            ]
        )
    elif regime in {"medium", "med", "medaug"}:
        ops.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.10),
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.8))],
                    p=0.10,
                ),
            ]
        )
    elif regime in {"heavy", "heavyaug"}:
        ops.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.20,
                    contrast=0.20,
                    saturation=0.20,
                    hue=0.05,
                ),
                transforms.RandomApply(
                    [
                        transforms.RandomAffine(
                            degrees=10,
                            translate=(0.05, 0.05),
                            scale=(0.95, 1.05),
                            shear=5,
                        )
                    ],
                    p=0.30,
                ),
                transforms.RandomPerspective(distortion_scale=0.10, p=0.20),
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=5, sigma=(0.3, 1.5))],
                    p=0.25,
                ),
            ]
        )
    elif regime not in {"none", "eval", "test", "validation", "val"}:
        raise ValueError(f"Unknown augmentation regime: {regime}")

    ops.append(transforms.ToTensor())
    if regime in {"heavy", "heavyaug"}:
        ops.append(AddGaussianNoise(std=0.01))
    if normalize:
        ops.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    return transforms.Compose(ops)


def fake_subtype_from_path(path: str | Path) -> str:
    parts = Path(path).parts
    for subtype in FAKE_SUBTYPES:
        if subtype in parts:
            return subtype
    return "Unknown" if "FAKE" in parts else "N/A"


@dataclass(frozen=True)
class Sample:
    path: Path
    label_name: str
    label_index: int
    source_id: str = ""


def _iter_images(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


class BenchmarkDataset(Dataset):
    """Recursive four-class dataset with a fixed, explicit class order.

    Expected layout is ``split/CLASS/.../image``. FAKE subtype directories may
    be nested below ``split/FAKE``.
    """

    def __init__(
        self,
        split_root: str | Path,
        transform: Callable | None = None,
        *,
        include_classes: Sequence[str] = CANONICAL_CLASSES,
        include_fake_subtypes: Sequence[str] | None = None,
        binary_attack: str | None = None,
    ) -> None:
        self.root = Path(split_root).expanduser().resolve()
        self.transform = transform
        self.include_classes = tuple(include_classes)
        self.include_fake_subtypes = (
            None if include_fake_subtypes is None else set(include_fake_subtypes)
        )
        self.binary_attack = binary_attack.upper() if binary_attack else None
        self.samples: list[Sample] = []

        if not self.root.is_dir():
            raise FileNotFoundError(f"Dataset split directory not found: {self.root}")

        for class_name in self.include_classes:
            if class_name not in CLASS_TO_INDEX:
                raise ValueError(f"Unknown class: {class_name}")
            class_root = self.root / class_name
            if not class_root.is_dir():
                continue
            for path in _iter_images(class_root):
                if (
                    class_name == "FAKE"
                    and self.include_fake_subtypes is not None
                    and fake_subtype_from_path(path) not in self.include_fake_subtypes
                ):
                    continue
                label_index = CLASS_TO_INDEX[class_name]
                if self.binary_attack:
                    if class_name not in {"REAL", self.binary_attack}:
                        continue
                    label_index = int(class_name == self.binary_attack)
                self.samples.append(Sample(path, class_name, label_index))

        if not self.samples:
            raise RuntimeError(f"No supported images found below {self.root}")

        self.targets = [sample.label_index for sample in self.samples]
        self.classes = (
            ["REAL", self.binary_attack] if self.binary_attack else list(CANONICAL_CLASSES)
        )
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        with Image.open(sample.path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        return image, sample.label_index, str(sample.path)


class ManifestDataset(Dataset):
    """Dataset backed by a CSV with path, label, split, and optional source_id."""

    def __init__(
        self,
        manifest: str | Path,
        transform: Callable | None = None,
        *,
        split: str | None = None,
        binary_attack: str | None = None,
    ) -> None:
        self.manifest_path = Path(manifest).expanduser().resolve()
        self.transform = transform
        self.binary_attack = binary_attack.upper() if binary_attack else None
        self.samples: list[Sample] = []

        with self.manifest_path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            required = {"path", "label"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"Manifest must contain columns {sorted(required)}")
            for row in reader:
                if split and row.get("split") != split:
                    continue
                label = row["label"].upper()
                if label not in CLASS_TO_INDEX:
                    raise ValueError(f"Unknown label {label!r} in manifest")
                if self.binary_attack and label not in {"REAL", self.binary_attack}:
                    continue
                path = Path(row["path"]).expanduser()
                if not path.is_absolute():
                    path = (self.manifest_path.parent / path).resolve()
                label_index = (
                    int(label == self.binary_attack)
                    if self.binary_attack
                    else CLASS_TO_INDEX[label]
                )
                self.samples.append(Sample(path, label, label_index, row.get("source_id", "")))

        if not self.samples:
            raise RuntimeError("Manifest selection produced no samples")
        self.targets = [sample.label_index for sample in self.samples]
        self.classes = (
            ["REAL", self.binary_attack] if self.binary_attack else list(CANONICAL_CLASSES)
        )
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        with Image.open(sample.path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        return image, sample.label_index, str(sample.path)


class SingleLabelImageDataset(Dataset):
    """Recursively load one image tree and assign a fixed canonical label."""

    def __init__(
        self,
        root: str | Path,
        label_name: str,
        transform: Callable | None = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.label_name = label_name.upper()
        if self.label_name not in CLASS_TO_INDEX:
            raise ValueError(f"Unknown label: {label_name}")
        self.transform = transform
        self.paths = list(_iter_images(self.root))
        if not self.paths:
            raise RuntimeError(f"No supported images found below {self.root}")
        self.targets = [CLASS_TO_INDEX[self.label_name]] * len(self.paths)
        self.classes = list(CANONICAL_CLASSES)
        self.class_to_idx = CLASS_TO_INDEX.copy()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        return image, CLASS_TO_INDEX[self.label_name], str(path)


def class_counts(dataset: BenchmarkDataset | ManifestDataset) -> dict[str, int]:
    counts = {name: 0 for name in dataset.classes}
    for sample in dataset.samples:
        name = sample.label_name
        if dataset.binary_attack:
            name = dataset.binary_attack if sample.label_index == 1 else "REAL"
        counts[name] = counts.get(name, 0) + 1
    return counts
