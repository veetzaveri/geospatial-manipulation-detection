#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from crp_benchmark.data import ManifestDataset, build_transform, class_counts
from crp_benchmark.evaluation import predict_loader, save_evaluation
from crp_benchmark.models import MODEL_SPECS
from crp_benchmark.reproducibility import seed_everything
from crp_benchmark.runtime import choose_device, make_loader, model_from_checkpoint, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Direct, zero-adaptation FSC evaluation.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", default="all", help="Use 'all' for the 2710/class FSC set")
    parser.add_argument("--mode", choices=("multi", "binary"), required=True)
    parser.add_argument("--attack", choices=("FAKE", "PIXEL", "PATCH"))
    parser.add_argument("--model", required=True, choices=sorted(MODEL_SPECS))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--pretrained-checkpoint")
    threshold_source = parser.add_mutually_exclusive_group()
    threshold_source.add_argument(
        "--threshold-json",
        help="In-domain validation_threshold.json produced by train_binary.py",
    )
    threshold_source.add_argument("--threshold", type=float, help="In-domain validation threshold")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "binary" and not args.attack:
        raise ValueError("--attack is required for binary mode")
    if args.mode == "multi" and args.attack:
        raise ValueError("--attack is only valid in binary mode")
    if args.mode == "multi" and (args.threshold_json is not None or args.threshold is not None):
        raise ValueError("Binary threshold options are not valid in multi mode")
    seed_everything(args.seed)
    device = choose_device(args.device)
    transform = build_transform("eval")
    binary_attack = args.attack if args.mode == "binary" else None
    split = None if args.split == "all" else args.split
    dataset = ManifestDataset(args.manifest, transform, split=split, binary_attack=binary_attack)
    counts = class_counts(dataset)
    expected_names = ("REAL", args.attack) if args.mode == "binary" else (
        "REAL",
        "PIXEL",
        "PATCH",
        "FAKE",
    )
    expected = {name: 2_710 for name in expected_names}
    if args.split == "all" and counts != expected:
        raise ValueError(
            f"Paper FSC zero-adaptation evaluation requires 2,710 images/class; found {counts}"
        )
    loader = make_loader(
        dataset, batch_size=args.batch_size, workers=args.workers, shuffle=False, seed=args.seed
    )
    num_classes = 1 if args.mode == "binary" else 4
    model, spec, _ = model_from_checkpoint(
        args.model,
        args.checkpoint,
        num_classes=num_classes,
        device=device,
        pretrained_checkpoint=args.pretrained_checkpoint,
    )
    threshold = 0.5
    threshold_source = "argmax"
    if args.mode == "binary":
        if args.threshold_json is None and args.threshold is None:
            raise ValueError(
                "Binary zero-adaptation evaluation requires --threshold-json or --threshold "
                "from the in-domain validation set"
            )
        if args.threshold_json is not None:
            with Path(args.threshold_json).open(encoding="utf-8") as handle:
                payload = json.load(handle)
            threshold = float(payload["threshold"])
            threshold_source = f"in_domain_validation_file:{Path(args.threshold_json)}"
        else:
            threshold = float(args.threshold)
            threshold_source = "in_domain_validation_cli"
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
    bundle = predict_loader(
        model, loader, device, binary=args.mode == "binary", threshold=threshold
    )
    output_dir = Path(args.output_dir)
    class_names = (
        ("REAL", args.attack) if args.mode == "binary" else ("REAL", "PIXEL", "PATCH", "FAKE")
    )
    save_evaluation(
        bundle,
        output_dir,
        class_names=class_names,
        title=f"{spec.paper_name}: FSC zero adaptation",
    )
    write_json(
        output_dir / "evaluation_protocol.json",
        {
            "mode": args.mode,
            "attack": args.attack,
            "threshold": threshold,
            "threshold_source": threshold_source,
            "threshold_selection_metric": (
                "in_domain_validation_attack_class_f1" if args.mode == "binary" else None
            ),
            "parameter_updates": False,
            "class_counts": counts,
        },
    )


if __name__ == "__main__":
    main()
