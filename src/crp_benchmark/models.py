from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class ModelSpec:
    key: str
    paper_name: str
    backend: str
    model_id: str
    recovered: bool = False
    notes: str = ""


# The Swin-B and EfficientNetV2 identifiers are recovered from prior project
# artifacts. Standard-model identifiers otherwise implement the architecture
# named in the paper, but the unavailable original per-model scripts should be
# consulted before claiming bit-for-bit checkpoint reproduction.
MODEL_SPECS: dict[str, ModelSpec] = {
    "swin_b": ModelSpec(
        "swin_b",
        "Swin Transformer Base (patch4-window7-224)",
        "timm",
        "swin_base_patch4_window7_224.ms_in22k",
        True,
        "Exact identifier recovered from the original Swin-B script.",
    ),
    "swin_t": ModelSpec(
        "swin_t",
        "Swin Transformer Tiny (patch4-window7-224)",
        "timm",
        "swin_tiny_patch4_window7_224.ms_in1k",
    ),
    "vit_l16": ModelSpec(
        "vit_l16",
        "Vision Transformer Large (ViT-L/16, 224x224)",
        "timm",
        "vit_large_patch16_224.augreg_in21k_ft_in1k",
    ),
    "vit_b16": ModelSpec(
        "vit_b16",
        "Vision Transformer Base (ViT-B/16, 224x224)",
        "timm",
        "vit_base_patch16_224.augreg_in21k_ft_in1k",
    ),
    "resnet152": ModelSpec("resnet152", "ResNet-152", "timm", "resnet152.a1_in1k"),
    "resnet101": ModelSpec("resnet101", "ResNet-101", "timm", "resnet101.a1_in1k"),
    "resnet50": ModelSpec("resnet50", "ResNet-50", "timm", "resnet50.a1_in1k"),
    "resnet18": ModelSpec("resnet18", "ResNet-18", "timm", "resnet18.a1_in1k"),
    "effnetv2_s": ModelSpec(
        "effnetv2_s",
        "EfficientNetV2-S (ImageNet-21k pretraining, ImageNet-1k fine-tuned)",
        "timm",
        "tf_efficientnetv2_s.in21k_ft_in1k",
        True,
        "Prior code used the legacy timm alias tf_efficientnetv2_s_in21ft1k.",
    ),
    "effnet_b5": ModelSpec("effnet_b5", "EfficientNet-B5", "timm", "tf_efficientnet_b5.in1k"),
    "effnet_b4": ModelSpec("effnet_b4", "EfficientNet-B4", "timm", "tf_efficientnet_b4.in1k"),
    "effnet_b0": ModelSpec("effnet_b0", "EfficientNet-B0", "timm", "tf_efficientnet_b0.in1k"),
    "densenet201": ModelSpec("densenet201", "DenseNet-201", "timm", "densenet201.tv_in1k"),
    "densenet169": ModelSpec("densenet169", "DenseNet-169", "timm", "densenet169.tv_in1k"),
    "densenet121": ModelSpec("densenet121", "DenseNet-121", "timm", "densenet121.tv_in1k"),
    "coatnet_0": ModelSpec("coatnet_0", "CoAtNet-0", "timm", "coatnet_0_rw_224.sw_in1k"),
    "georsclip_b32": ModelSpec(
        "georsclip_b32",
        "GeoRSCLIP (ViT-B/32)",
        "open_clip",
        "ViT-B-32",
        notes="Pass the GeoRSCLIP checkpoint with --pretrained-checkpoint.",
    ),
    "remoteclip_b32": ModelSpec(
        "remoteclip_b32",
        "RemoteCLIP (ViT-B/32)",
        "open_clip",
        "ViT-B-32",
        notes="Pass the RemoteCLIP checkpoint with --pretrained-checkpoint.",
    ),
    "satdino_b16": ModelSpec(
        "satdino_b16",
        "SatDINO (ViT-B/16)",
        "satdino",
        "vit_base_patch16_224",
        notes="Pass the SatDINO checkpoint with --pretrained-checkpoint.",
    ),
    "satlas_swin_b": ModelSpec(
        "satlas_swin_b",
        "SatlasPretrain Aerial Swin-B (Single-Image)",
        "satlas",
        "Aerial_SwinB_SI",
        True,
        "The Satlas model identifier and fpn=True setting were recovered.",
    ),
}

ALIASES = {
    "swin-b": "swin_b",
    "swin-t": "swin_t",
    "effnet-b5": "effnet_b5",
    "effnet-b4": "effnet_b4",
    "effnet-b0": "effnet_b0",
    "effnetv2-s": "effnetv2_s",
    "vit-l16": "vit_l16",
    "vit-b16": "vit_b16",
    "georsclip": "georsclip_b32",
    "remoteclip": "remoteclip_b32",
    "satdino": "satdino_b16",
    "satlas": "satlas_swin_b",
}

# Table 2 reports a fixed architecture-dependent main-benchmark batch size in
# the 64--72 range. The recovered Swin-B script supplies the exact value 72;
# the remaining clean adapters use the fixed lower endpoint, 64.
PAPER_MAIN_BATCH_SIZES = {key: 64 for key in MODEL_SPECS}
PAPER_MAIN_BATCH_SIZES["swin_b"] = 72


def resolve_model_key(key: str) -> str:
    normalized = key.strip().lower()
    normalized = ALIASES.get(normalized, normalized)
    if normalized not in MODEL_SPECS:
        raise KeyError(f"Unknown model {key!r}. Choose from: {', '.join(sorted(MODEL_SPECS))}")
    return normalized


def paper_main_batch_size(key: str) -> int:
    return PAPER_MAIN_BATCH_SIZES[resolve_model_key(key)]


class FeatureClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, feature_dim: int, num_classes: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        if isinstance(features, (tuple, list)):
            features = features[0]
        if features.ndim > 2:
            features = features.mean(dim=tuple(range(2, features.ndim)))
        return self.head(features)


class OpenCLIPClassifier(nn.Module):
    def __init__(self, visual_model: nn.Module, output_dim: int, num_classes: int) -> None:
        super().__init__()
        self.visual_model = visual_model
        self.head = nn.Linear(output_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.visual_model.encode_image(x)
        return self.head(features.float())


class SatlasClassifierAdapter(nn.Module):
    """Convert Satlas classification probabilities into trainable logits."""

    def __init__(self, model: nn.Module, *, binary: bool) -> None:
        super().__init__()
        self.model = model
        self.binary = binary

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        probabilities = self.model(x).clamp(1e-7, 1.0 - 1e-7)
        if self.binary:
            return torch.log(probabilities[:, 1]) - torch.log(probabilities[:, 0])
        return torch.log(probabilities)


def _extract_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        for key in (
            "model_state_dict",
            "state_dict",
            "model",
            "teacher",
            "student",
            "backbone",
        ):
            candidate = payload.get(key)
            if isinstance(candidate, dict):
                payload = candidate
                break
    if not isinstance(payload, dict):
        raise TypeError("Checkpoint does not contain a state dictionary")

    cleaned: dict[str, torch.Tensor] = {}
    prefixes = ("module.", "model.", "backbone.", "teacher.", "student.")
    for key, value in payload.items():
        if not torch.is_tensor(value):
            continue
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if key.startswith(prefix):
                    key = key[len(prefix) :]
                    changed = True
        cleaned[key] = value
    return cleaned


def _load_weights(
    model: nn.Module,
    checkpoint: str | Path,
    *,
    strict: bool = False,
) -> tuple[list[str], list[str]]:
    payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
    state = _extract_state_dict(payload)
    result = model.load_state_dict(state, strict=strict)
    return list(result.missing_keys), list(result.unexpected_keys)


def create_model(
    key: str,
    num_classes: int,
    *,
    pretrained: bool = True,
    pretrained_checkpoint: str | Path | None = None,
) -> tuple[nn.Module, ModelSpec]:
    key = resolve_model_key(key)
    spec = MODEL_SPECS[key]

    if spec.backend == "timm":
        import timm

        candidates = [spec.model_id]
        if key == "effnetv2_s":
            candidates.append("tf_efficientnetv2_s_in21ft1k")
        last_error: Exception | None = None
        for model_id in candidates:
            try:
                model = timm.create_model(
                    model_id,
                    pretrained=pretrained,
                    num_classes=num_classes,
                )
                break
            except Exception as error:  # noqa: BLE001  # pragma: no cover
                last_error = error
        else:
            raise RuntimeError(f"Could not create timm model {candidates}") from last_error
        if pretrained_checkpoint:
            _load_weights(model, pretrained_checkpoint, strict=False)
        return model, spec

    if spec.backend == "open_clip":
        if not pretrained_checkpoint:
            raise ValueError(
                f"{key} requires --pretrained-checkpoint; no paper checkpoint is bundled."
            )
        import open_clip

        base_pretrained = "openai" if key == "georsclip_b32" else None
        visual_model = open_clip.create_model(spec.model_id, pretrained=base_pretrained)
        _load_weights(visual_model, pretrained_checkpoint, strict=False)
        output_dim = int(visual_model.visual.output_dim)
        return OpenCLIPClassifier(visual_model, output_dim, num_classes), spec

    if spec.backend == "satdino":
        if not pretrained_checkpoint:
            raise ValueError("satdino_b16 requires --pretrained-checkpoint")
        import timm

        encoder = timm.create_model(spec.model_id, pretrained=False, num_classes=0)
        _load_weights(encoder, pretrained_checkpoint, strict=False)
        feature_dim = int(encoder.num_features)
        return FeatureClassifier(encoder, feature_dim, num_classes), spec

    if spec.backend == "satlas":
        try:
            from satlaspretrain_models import Head, Weights
        except ImportError as error:  # pragma: no cover - optional dependency
            raise ImportError("Install satlaspretrain-models to use satlas_swin_b") from error
        binary = num_classes == 1
        model = Weights().get_pretrained_model(
            model_identifier=spec.model_id,
            fpn=True,
            head=Head.CLASSIFY,
            num_categories=2 if binary else num_classes,
        )
        if pretrained_checkpoint:
            _load_weights(model, pretrained_checkpoint, strict=False)
        return SatlasClassifierAdapter(model, binary=binary), spec

    raise AssertionError(f"Unhandled model backend: {spec.backend}")


def load_experiment_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    *,
    strict: bool = True,
) -> dict[str, Any]:
    payload = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
    model.load_state_dict(state, strict=strict)
    return payload if isinstance(payload, dict) else {}


def registry_as_dict() -> list[dict[str, Any]]:
    return [asdict(MODEL_SPECS[key]) for key in sorted(MODEL_SPECS)]
