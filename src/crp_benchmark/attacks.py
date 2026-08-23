from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch


def load_faster_rcnn(device: torch.device) -> torch.nn.Module:
    from torchvision.models.detection import (
        FasterRCNN_ResNet50_FPN_Weights,
        fasterrcnn_resnet50_fpn,
    )

    detector = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT).to(device)
    detector.eval()
    for parameter in detector.parameters():
        parameter.requires_grad_(False)
    return detector


def detector_boxes(
    detector: torch.nn.Module,
    image: torch.Tensor,
    *,
    score_threshold: float = 0.5,
) -> torch.Tensor:
    with torch.no_grad():
        output = detector([image])[0]
    scores = output.get("scores", torch.empty(0, device=image.device))
    boxes = output.get("boxes", torch.empty((0, 4), device=image.device))
    return boxes[scores >= score_threshold]


def gradient_saliency_coordinates(
    detector: torch.nn.Module,
    image: torch.Tensor,
    *,
    budget: int = 200,
    dummy_weight: float = 0.05,
    score_threshold: float = 0.5,
    fallback_seed: int = 42,
) -> torch.Tensor:
    """Paper Algorithms 1-2 coordinate selector.

    ``image`` must be RGB CHW in [0,1]. The detector remains frozen; gradients
    are enabled only for the image.
    """

    x = image.detach().clone().requires_grad_(True)
    output = detector([x])[0]
    scores = output.get("scores", torch.empty(0, device=x.device))
    boxes = output.get("boxes", torch.empty((0, 4), device=x.device))
    if scores.numel():
        loss = scores.mean() + dummy_weight * x.mean()
    else:
        loss = dummy_weight * x.mean()
    detector.zero_grad(set_to_none=True)
    loss.backward()
    if x.grad is None:
        raise RuntimeError("Faster R-CNN did not produce an input gradient")
    sensitivity = x.grad.detach().abs().sum(dim=0)

    selected_boxes = boxes[scores >= score_threshold]
    if selected_boxes.numel():
        height, width = sensitivity.shape
        mask = torch.zeros_like(sensitivity, dtype=torch.bool)
        for box in selected_boxes:
            x1, y1, x2, y2 = box.round().long().tolist()
            x1, x2 = max(0, x1), min(width, x2)
            y1, y2 = max(0, y1), min(height, y2)
            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = True
        restricted = sensitivity * mask
        if torch.any(restricted > 0):
            sensitivity = restricted

    flat = sensitivity.flatten()
    budget = min(int(budget), flat.numel())
    if float(flat.abs().sum()) == 0.0:
        generator = torch.Generator(device="cpu").manual_seed(fallback_seed)
        indices = torch.randperm(flat.numel(), generator=generator)[:budget]
        indices = indices.to(flat.device)
    else:
        indices = torch.topk(flat, k=budget, largest=True, sorted=False).indices
    width = sensitivity.shape[1]
    return torch.stack((indices // width, indices % width), dim=1)


def random_coordinates(
    image: torch.Tensor,
    *,
    budget: int,
    seed: int,
) -> torch.Tensor:
    height, width = image.shape[-2:]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    flat = torch.randperm(height * width, generator=generator)[:budget]
    return torch.stack((flat // width, flat % width), dim=1).to(image.device)


def apply_pixel_perturbation(
    image: torch.Tensor,
    coordinates: torch.Tensor,
    *,
    patch_size: int = 5,
    epsilon: float = 0.05,
) -> torch.Tensor:
    if patch_size <= 0 or patch_size % 2 == 0:
        raise ValueError("patch_size must be a positive odd integer")
    result = image.detach().clone()
    half = patch_size // 2
    height, width = result.shape[-2:]
    for y_value, x_value in coordinates.detach().cpu().tolist():
        y1, y2 = max(0, y_value - half), min(height, y_value + half + 1)
        x1, x2 = max(0, x_value - half), min(width, x_value + half + 1)
        result[:, y1:y2, x1:x2].add_(epsilon)
    return result.clamp_(0.0, 1.0)


def total_variation(patch: torch.Tensor) -> torch.Tensor:
    vertical = torch.abs(patch[:, 1:, :] - patch[:, :-1, :]).mean()
    horizontal = torch.abs(patch[:, :, 1:] - patch[:, :, :-1]).mean()
    return vertical + horizontal


def _sample_centers(
    image: torch.Tensor,
    boxes: torch.Tensor,
    *,
    count: int,
    radius: int,
    seed: int,
) -> list[tuple[int, int]]:
    rng = random.Random(seed)
    height, width = image.shape[-2:]
    centers: list[tuple[int, int]] = []
    usable = boxes.detach().cpu().numpy().tolist()
    for _ in range(count):
        if usable:
            x1, y1, x2, y2 = usable[rng.randrange(len(usable))]
            low_x = max(radius, int(np.ceil(x1)) + radius)
            high_x = min(width - radius - 1, int(np.floor(x2)) - radius)
            low_y = max(radius, int(np.ceil(y1)) + radius)
            high_y = min(height - radius - 1, int(np.floor(y2)) - radius)
            if high_x >= low_x and high_y >= low_y:
                centers.append((rng.randint(low_y, high_y), rng.randint(low_x, high_x)))
                continue
        centers.append(
            (
                rng.randint(radius, max(radius, height - radius - 1)),
                rng.randint(radius, max(radius, width - radius - 1)),
            )
        )
    return centers


def _shape_mask(radius: int, geometry: str, device: torch.device) -> torch.Tensor:
    side = 2 * radius + 1
    if geometry == "square":
        return torch.ones((1, side, side), device=device)
    if geometry != "circle":
        raise ValueError("geometry must be circle or square")
    coordinates = torch.arange(-radius, radius + 1, device=device)
    yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
    return ((yy**2 + xx**2) <= radius**2).float().unsqueeze(0)


def stamp_patches(
    image: torch.Tensor,
    patches: list[torch.Tensor],
    centers: list[tuple[int, int]],
    *,
    geometry: str,
) -> torch.Tensor:
    result = image.clone()
    radius = (patches[0].shape[-1] - 1) // 2
    mask = _shape_mask(radius, geometry, image.device)
    for patch, (center_y, center_x) in zip(patches, centers):
        y1, y2 = center_y - radius, center_y + radius + 1
        x1, x2 = center_x - radius, center_x + radius + 1
        current = result[:, y1:y2, x1:x2]
        result[:, y1:y2, x1:x2] = current * (1.0 - mask) + patch * mask
    return result


@dataclass
class PatchResult:
    image: torch.Tensor
    centers: list[tuple[int, int]]
    iterations: int
    final_confidence: float


def optimize_adversarial_patch(
    detector: torch.nn.Module,
    image: torch.Tensor,
    *,
    geometry: str = "circle",
    radius: int = 5,
    num_patches: int = 1,
    iterations: int = 50,
    step_size: float = 0.01,
    momentum: float = 0.75,
    tv_weight: float = 1e-4,
    stop_confidence: float = 0.05,
    box_score_threshold: float = 0.5,
    seed: int = 42,
) -> PatchResult:
    """Optimize the learnable PATCH described in Equations 5-10.

    Radius=5, momentum=0.75, and 50 steps are stated in the paper. The original
    scripts containing P, alpha, lambda_TV, and the stopping threshold were not
    recoverable; defaults for those values are transparent reconstruction choices
    and remain CLI-configurable.
    """

    boxes = detector_boxes(detector, image, score_threshold=box_score_threshold)
    centers = _sample_centers(
        image,
        boxes,
        count=num_patches,
        radius=radius,
        seed=seed,
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    side = 2 * radius + 1
    patches = [torch.rand((3, side, side), generator=generator).to(image.device) for _ in centers]
    velocities = [torch.zeros_like(patch) for patch in patches]
    final_confidence = float("nan")

    for iteration in range(1, iterations + 1):
        patches = [patch.detach().requires_grad_(True) for patch in patches]
        stamped = stamp_patches(image, patches, centers, geometry=geometry)
        output = detector([stamped])[0]
        scores = output.get("scores", torch.empty(0, device=image.device))
        confidence = scores.mean() if scores.numel() else stamped.sum() * 0.0
        smoothness = torch.stack([total_variation(patch) for patch in patches]).sum()
        loss = confidence + tv_weight * smoothness
        detector.zero_grad(set_to_none=True)
        loss.backward()
        final_confidence = float(confidence.detach().cpu())

        updated: list[torch.Tensor] = []
        for index, patch in enumerate(patches):
            if patch.grad is None:
                gradient_sign = torch.zeros_like(patch)
            else:
                gradient_sign = patch.grad.sign()
            velocities[index] = momentum * velocities[index] + gradient_sign
            updated.append((patch - step_size * velocities[index]).clamp(0.0, 1.0))
        patches = updated
        if final_confidence < stop_confidence:
            break

    final_image = stamp_patches(
        image, [patch.detach() for patch in patches], centers, geometry=geometry
    ).clamp(0.0, 1.0)
    return PatchResult(final_image, centers, iteration, final_confidence)
