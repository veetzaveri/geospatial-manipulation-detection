#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score

from crp_benchmark.data import BenchmarkDataset, ManifestDataset, build_transform, class_counts
from crp_benchmark.evaluation import predict_loader, save_evaluation
from crp_benchmark.models import MODEL_SPECS, create_model, paper_main_batch_size
from crp_benchmark.reproducibility import seed_everything
from crp_benchmark.runtime import choose_device, make_loader, write_json
from crp_benchmark.stats import bootstrap_interval, select_attack_f1_threshold
from crp_benchmark.training import MAIN_PROTOCOL, fit_model, save_fit_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a specialized REAL/attack detector.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset-root")
    source.add_argument("--manifest")
    parser.add_argument("--attack", required=True, choices=("FAKE", "PIXEL", "PATCH"))
    parser.add_argument("--model", required=True, choices=sorted(MODEL_SPECS))
    parser.add_argument("--pretrained-checkpoint")
    parser.add_argument("--output-dir", required=True)
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


def dataset_for(args, split: str, transform):
    if args.dataset_root:
        folder = "validation" if split == "validation" else split
        return BenchmarkDataset(
            Path(args.dataset_root) / folder,
            transform,
            binary_attack=args.attack,
        )
    return ManifestDataset(
        args.manifest,
        transform,
        split=split,
        binary_attack=args.attack,
    )


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = choose_device(args.device)
    batch_size = args.batch_size or paper_main_batch_size(args.model)
    train_transform = build_transform("lightaug")
    eval_transform = build_transform("eval")
    train_dataset = dataset_for(args, "train", train_transform)
    val_dataset = dataset_for(args, "validation", eval_transform)
    test_dataset = dataset_for(args, "test", eval_transform)
    counts = {
        "train": class_counts(train_dataset),
        "validation": class_counts(val_dataset),
        "test": class_counts(test_dataset),
    }
    train_loader = make_loader(
        train_dataset,
        batch_size=batch_size,
        workers=args.workers,
        shuffle=True,
        seed=args.seed,
    )
    val_loader = make_loader(
        val_dataset, batch_size=batch_size, workers=args.workers, shuffle=False, seed=args.seed
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
        num_classes=1,
        pretrained=True,
        pretrained_checkpoint=args.pretrained_checkpoint,
    )
    result = fit_model(
        model, train_loader, val_loader, device, protocol=MAIN_PROTOCOL, binary=True
    )
    output_dir = Path(args.output_dir)
    save_fit_result(
        result,
        output_dir,
        model_key=args.model,
        class_to_idx={"REAL": 0, args.attack: 1},
        protocol=MAIN_PROTOCOL,
        extra_metadata={
            "model_spec": spec.__dict__,
            "binary_attack": args.attack,
            "seed": args.seed,
            "batch_size": batch_size,
            "class_counts": counts,
            "normalization": "imagenet",
        },
    )

    validation = predict_loader(model, val_loader, device, binary=True)
    threshold = select_attack_f1_threshold(validation.labels, validation.probabilities[:, 1])
    write_json(
        output_dir / "validation_threshold.json",
        {"threshold": threshold, "selection_metric": "attack_class_f1"},
    )
    test = predict_loader(model, test_loader, device, binary=True, threshold=threshold)
    save_evaluation(
        test,
        output_dir / "test",
        class_names=("REAL", args.attack),
        title=f"{spec.paper_name}: REAL vs {args.attack}",
    )
    labels, predictions, scores = test.labels, test.predictions, test.probabilities[:, 1]
    metric_functions = {
        "balanced_accuracy": lambda idx: balanced_accuracy_score(labels[idx], predictions[idx]),
        "f1_attack": lambda idx: f1_score(
            labels[idx], predictions[idx], pos_label=1, zero_division=0
        ),
        "auprc": lambda idx: average_precision_score(labels[idx], scores[idx]),
    }
    ci = {
        name: dict(
            zip(
                ("estimate", "ci_low", "ci_high"),
                bootstrap_interval(
                    labels,
                    function,
                    replicates=(1_000 if name == "auprc" else args.bootstrap_replicates),
                    seed=args.seed,
                ),
            )
        )
        for name, function in metric_functions.items()
    }
    write_json(output_dir / "test" / "bootstrap_ci.json", ci)
    print(f"validation-selected threshold: {threshold:.8f}")


if __name__ == "__main__":
    main()
