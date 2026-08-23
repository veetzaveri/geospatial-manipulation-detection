#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from crp_benchmark.data import BenchmarkDataset, ManifestDataset, build_transform, class_counts
from crp_benchmark.evaluation import predict_loader, save_evaluation
from crp_benchmark.models import MODEL_SPECS, create_model, paper_main_batch_size
from crp_benchmark.reproducibility import seed_everything
from crp_benchmark.runtime import choose_device, make_loader, write_json
from crp_benchmark.stats import four_class_bootstrap
from crp_benchmark.training import MAIN_PROTOCOL, fit_model, save_fit_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one of the 20 four-way CRP benchmark architectures."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset-root", help="Root containing train/validation/test")
    source.add_argument("--manifest", help="CSV containing path,label,split[,source_id]")
    parser.add_argument("--model", required=True, choices=sorted(MODEL_SPECS))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--augmentation",
        choices=("lightaug", "medaug", "heavyaug"),
        default="lightaug",
    )
    parser.add_argument("--pretrained-checkpoint")
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Optional override; Table 2 architecture-fixed default is used when omitted",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    return parser.parse_args()


def split_dataset(args, split: str, transform):
    if args.dataset_root:
        split_name = "validation" if split == "validation" else split
        return BenchmarkDataset(Path(args.dataset_root) / split_name, transform)
    return ManifestDataset(args.manifest, transform, split=split)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = choose_device(args.device)
    batch_size = args.batch_size or paper_main_batch_size(args.model)
    train_transform = build_transform(args.augmentation)
    eval_transform = build_transform("eval")

    train_dataset = split_dataset(args, "train", train_transform)
    val_dataset = split_dataset(args, "validation", eval_transform)
    test_dataset = split_dataset(args, "test", eval_transform)
    counts = {
        "train": class_counts(train_dataset),
        "validation": class_counts(val_dataset),
        "test": class_counts(test_dataset),
    }
    print(f"class counts: {counts}")

    train_loader = make_loader(
        train_dataset,
        batch_size=batch_size,
        workers=args.workers,
        shuffle=True,
        seed=args.seed,
    )
    val_loader = make_loader(
        val_dataset,
        batch_size=batch_size,
        workers=args.workers,
        shuffle=False,
        seed=args.seed,
    )
    test_loader = make_loader(
        test_dataset,
        batch_size=batch_size,
        workers=args.workers,
        shuffle=False,
        seed=args.seed,
    )

    model, spec = create_model(
        args.model,
        num_classes=4,
        pretrained=True,
        pretrained_checkpoint=args.pretrained_checkpoint,
    )
    result = fit_model(model, train_loader, val_loader, device, protocol=MAIN_PROTOCOL)
    output_dir = Path(args.output_dir)
    checkpoint = save_fit_result(
        result,
        output_dir,
        model_key=args.model,
        class_to_idx=train_dataset.class_to_idx,
        protocol=MAIN_PROTOCOL,
        extra_metadata={
            "model_spec": spec.__dict__,
            "augmentation": args.augmentation,
            "seed": args.seed,
            "batch_size": batch_size,
            "class_counts": counts,
            "normalization": "imagenet",
        },
    )
    bundle = predict_loader(model, test_loader, device)
    metrics = save_evaluation(
        bundle,
        output_dir / "test",
        title=f"{spec.paper_name} four-way test",
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    bootstrap = four_class_bootstrap(
        bundle.labels,
        bundle.predictions,
        replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    write_json(output_dir / "test" / "bootstrap_ci.json", bootstrap)
    print(f"checkpoint: {checkpoint}")
    print(f"test metrics: {metrics}")


if __name__ == "__main__":
    main()
