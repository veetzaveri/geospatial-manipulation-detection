#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from torch.utils.data import Subset

from crp_benchmark.data import ManifestDataset, build_transform, class_counts
from crp_benchmark.evaluation import predict_loader, save_evaluation
from crp_benchmark.models import MODEL_SPECS
from crp_benchmark.reproducibility import seed_everything
from crp_benchmark.runtime import choose_device, make_loader, model_from_checkpoint, write_json
from crp_benchmark.stats import select_attack_f1_threshold
from crp_benchmark.training import FSC_PROTOCOL, fit_model, save_fit_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fully fine-tune an in-domain checkpoint on FSC.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--mode", choices=("multi", "binary"), required=True)
    parser.add_argument("--attack", choices=("FAKE", "PIXEL", "PATCH"))
    parser.add_argument("--shots-per-class", type=int, required=True, choices=(50, 200, 1000))
    parser.add_argument("--model", required=True, choices=sorted(MODEL_SPECS))
    parser.add_argument("--in-domain-checkpoint", required=True)
    parser.add_argument("--pretrained-checkpoint")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def stratified_subset(dataset: ManifestDataset, shots: int, seed: int) -> Subset:
    targets = np.asarray(dataset.targets)
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for label in sorted(np.unique(targets)):
        candidates = np.flatnonzero(targets == label)
        if len(candidates) < shots:
            raise ValueError(f"Label {label} has {len(candidates)} samples, fewer than {shots}")
        selected.extend(rng.choice(candidates, size=shots, replace=False).tolist())
    rng.shuffle(selected)
    return Subset(dataset, selected)


def main() -> None:
    args = parse_args()
    if args.mode == "binary" and not args.attack:
        raise ValueError("--attack is required for binary mode")
    if args.mode == "multi" and args.attack:
        raise ValueError("--attack is only valid in binary mode")
    seed_everything(args.seed)
    device = choose_device(args.device)
    binary_attack = args.attack if args.mode == "binary" else None
    train_full = ManifestDataset(
        args.manifest,
        build_transform("lightaug"),
        split="adaptation_pool",
        binary_attack=binary_attack,
    )
    train_dataset = stratified_subset(train_full, args.shots_per_class, args.seed)
    val_dataset = ManifestDataset(
        args.manifest,
        build_transform("eval"),
        split="validation",
        binary_attack=binary_attack,
    )
    test_dataset = ManifestDataset(
        args.manifest,
        build_transform("eval"),
        split="test",
        binary_attack=binary_attack,
    )
    expected_names = ("REAL", args.attack) if args.mode == "binary" else (
        "REAL",
        "PIXEL",
        "PATCH",
        "FAKE",
    )
    expected_held_out = {name: 542 for name in expected_names}
    validation_counts = class_counts(val_dataset)
    test_counts = class_counts(test_dataset)
    if validation_counts != expected_held_out or test_counts != expected_held_out:
        raise ValueError(
            "Paper FSC adaptation requires fixed 542/class validation and test splits; "
            f"found validation={validation_counts}, test={test_counts}"
        )
    train_loader = make_loader(
        train_dataset,
        batch_size=args.batch_size,
        workers=args.workers,
        shuffle=True,
        seed=args.seed,
    )
    val_loader = make_loader(
        val_dataset, batch_size=args.batch_size, workers=args.workers, shuffle=False, seed=args.seed
    )
    test_loader = make_loader(
        test_dataset,
        batch_size=args.batch_size,
        workers=args.workers,
        shuffle=False,
        seed=args.seed,
    )
    binary = args.mode == "binary"
    model, spec, source_metadata = model_from_checkpoint(
        args.model,
        args.in_domain_checkpoint,
        num_classes=1 if binary else 4,
        device=device,
        pretrained_checkpoint=args.pretrained_checkpoint,
    )
    result = fit_model(
        model,
        train_loader,
        val_loader,
        device,
        protocol=FSC_PROTOCOL,
        binary=binary,
    )
    output_dir = Path(args.output_dir)
    class_to_idx = {"REAL": 0, args.attack: 1} if binary else train_full.class_to_idx
    save_fit_result(
        result,
        output_dir,
        model_key=args.model,
        class_to_idx=class_to_idx,
        protocol=FSC_PROTOCOL,
        extra_metadata={
            "model_spec": spec.__dict__,
            "mode": args.mode,
            "attack": args.attack,
            "shots_per_class": args.shots_per_class,
            "source_checkpoint": str(Path(args.in_domain_checkpoint).resolve()),
            "source_checkpoint_metadata_keys": sorted(
                key for key in source_metadata if key != "model_state_dict"
            ),
            "full_backbone_finetuning": True,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "normalization": "imagenet",
            "validation_class_counts": validation_counts,
            "test_class_counts": test_counts,
        },
    )
    threshold = 0.5
    if binary:
        validation = predict_loader(model, val_loader, device, binary=True)
        threshold = select_attack_f1_threshold(validation.labels, validation.probabilities[:, 1])
    test = predict_loader(model, test_loader, device, binary=binary, threshold=threshold)
    class_names = ("REAL", args.attack) if binary else ("REAL", "PIXEL", "PATCH", "FAKE")
    save_evaluation(
        test,
        output_dir / "test",
        class_names=class_names,
        title=f"{spec.paper_name}: FSC {args.shots_per_class}/class",
    )
    write_json(
        output_dir / "validation_threshold.json",
        {
            "threshold": threshold,
            "selection": "validation_attack_f1" if binary else "argmax",
        },
    )


if __name__ == "__main__":
    main()
