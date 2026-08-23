#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from crp_benchmark.constants import FAKE_SUBTYPES
from crp_benchmark.data import BenchmarkDataset, build_transform, class_counts
from crp_benchmark.evaluation import predict_loader, predictions_frame, save_evaluation
from crp_benchmark.models import MODEL_SPECS, create_model, paper_main_batch_size
from crp_benchmark.reproducibility import seed_everything
from crp_benchmark.runtime import choose_device, make_loader, write_json
from crp_benchmark.stats import percentile_interval
from crp_benchmark.training import MAIN_PROTOCOL, fit_model, save_fit_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one FAKE leave-one-method-out run.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--held-out", required=True, choices=FAKE_SUBTYPES)
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


def _bootstrap_mean(values: np.ndarray, replicates: int, rng: np.random.Generator):
    samples = np.empty(replicates)
    for index in range(replicates):
        samples[index] = np.mean(rng.choice(values, size=len(values), replace=True))
    return float(np.mean(values)), percentile_interval(samples)


def lomo_statistics(mixed_frame, held_frame, replicates: int, seed: int) -> dict:
    mixed_fake = mixed_frame[mixed_frame.true_label_name == "FAKE"]
    mixed_success = (mixed_fake.pred_label_name == "FAKE").to_numpy(dtype=float)
    held_success = (held_frame.pred_label_name == "FAKE").to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    mixed_boot = np.empty(replicates)
    held_boot = np.empty(replicates)
    for index in range(replicates):
        mixed_boot[index] = np.mean(
            rng.choice(mixed_success, size=len(mixed_success), replace=True)
        )
        held_boot[index] = np.mean(rng.choice(held_success, size=len(held_success), replace=True))
    delta_boot = mixed_boot - held_boot
    output = {
        "mixed_recall": {
            "estimate": float(np.mean(mixed_success)),
            "ci": percentile_interval(mixed_boot),
        },
        "held_out_recall": {
            "estimate": float(np.mean(held_success)),
            "ci": percentile_interval(held_boot),
        },
        "delta_mixed_minus_held": {
            "estimate": float(np.mean(mixed_success) - np.mean(held_success)),
            "ci": percentile_interval(delta_boot),
        },
    }
    rng = np.random.default_rng(seed)
    diagnostics = {
        "rate_predicted_as_REAL": (held_frame.pred_label_name == "REAL").to_numpy(float),
        "rate_predicted_as_PIXEL": (held_frame.pred_label_name == "PIXEL").to_numpy(float),
        "rate_predicted_as_PATCH": (held_frame.pred_label_name == "PATCH").to_numpy(float),
        "mean_prob_FAKE": held_frame.prob_FAKE.to_numpy(float),
    }
    output["held_out_diagnostics"] = {}
    for name, values in diagnostics.items():
        estimate, ci = _bootstrap_mean(values, replicates, rng)
        output["held_out_diagnostics"][name] = {"estimate": estimate, "ci": ci}
    output["bootstrap_replicates"] = replicates
    output["bootstrap_seed"] = seed
    return output


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = choose_device(args.device)
    batch_size = args.batch_size or paper_main_batch_size(args.model)
    train_transform = build_transform("lightaug")
    eval_transform = build_transform("eval")
    seen = tuple(subtype for subtype in FAKE_SUBTYPES if subtype != args.held_out)
    root = Path(args.dataset_root)
    train_dataset = BenchmarkDataset(root / "train", train_transform, include_fake_subtypes=seen)
    val_dataset = BenchmarkDataset(root / "validation", eval_transform, include_fake_subtypes=seen)
    mixed_test_dataset = BenchmarkDataset(root / "test", eval_transform)
    held_test_dataset = BenchmarkDataset(
        root / "test",
        eval_transform,
        include_classes=("FAKE",),
        include_fake_subtypes=(args.held_out,),
    )
    counts = {
        "train": class_counts(train_dataset),
        "validation": class_counts(val_dataset),
        "mixed_test": class_counts(mixed_test_dataset),
        "held_out_test": class_counts(held_test_dataset),
    }
    loaders = {
        "train": make_loader(
            train_dataset,
            batch_size=batch_size,
            workers=args.workers,
            shuffle=True,
            seed=args.seed,
        ),
        "validation": make_loader(
            val_dataset,
            batch_size=batch_size,
            workers=args.workers,
            shuffle=False,
            seed=args.seed,
        ),
        "mixed": make_loader(
            mixed_test_dataset,
            batch_size=batch_size,
            workers=args.workers,
            shuffle=False,
            seed=args.seed,
        ),
        "held": make_loader(
            held_test_dataset,
            batch_size=batch_size,
            workers=args.workers,
            shuffle=False,
            seed=args.seed,
        ),
    }
    model, spec = create_model(
        args.model,
        num_classes=4,
        pretrained=True,
        pretrained_checkpoint=args.pretrained_checkpoint,
    )
    result = fit_model(
        model, loaders["train"], loaders["validation"], device, protocol=MAIN_PROTOCOL
    )
    output_dir = Path(args.output_dir)
    save_fit_result(
        result,
        output_dir,
        model_key=args.model,
        class_to_idx=train_dataset.class_to_idx,
        protocol=MAIN_PROTOCOL,
        extra_metadata={
            "model_spec": spec.__dict__,
            "held_out_fake_subtype": args.held_out,
            "seen_fake_subtypes": seen,
            "class_counts": counts,
            "seed": args.seed,
            "batch_size": batch_size,
            "normalization": "imagenet",
        },
    )
    mixed = predict_loader(model, loaders["mixed"], device)
    held = predict_loader(model, loaders["held"], device)
    save_evaluation(
        mixed,
        output_dir / "mixed_test",
        title="LOMO mixed-FAKE test",
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    save_evaluation(
        held,
        output_dir / "held_out_test",
        title="LOMO held-out-FAKE test",
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    mixed_frame = predictions_frame(mixed, ("REAL", "PIXEL", "PATCH", "FAKE"))
    held_frame = predictions_frame(held, ("REAL", "PIXEL", "PATCH", "FAKE"))
    statistics = lomo_statistics(mixed_frame, held_frame, args.bootstrap_replicates, args.seed)
    write_json(output_dir / "lomo_statistics.json", statistics)
    print(statistics)


if __name__ == "__main__":
    main()
