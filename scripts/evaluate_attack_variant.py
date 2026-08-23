#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from crp_benchmark.data import BenchmarkDataset, build_transform, class_counts
from crp_benchmark.diagnostics import target_class_diagnostics
from crp_benchmark.evaluation import predict_loader, save_evaluation
from crp_benchmark.models import MODEL_SPECS
from crp_benchmark.reproducibility import seed_everything
from crp_benchmark.runtime import choose_device, make_loader, model_from_checkpoint, write_json
from crp_benchmark.stats import four_class_bootstrap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate PIXEL B=50/B=400 or unseen square-PATCH stress tests."
    )
    parser.add_argument("--test-root", required=True, help="Balanced four-class variant test tree")
    parser.add_argument("--target-class", required=True, choices=("PIXEL", "PATCH"))
    parser.add_argument("--variant-name", required=True)
    parser.add_argument("--model", required=True, choices=sorted(MODEL_SPECS))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--pretrained-checkpoint")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = choose_device(args.device)
    transform = build_transform("eval")
    dataset = BenchmarkDataset(args.test_root, transform)
    counts = class_counts(dataset)
    expected = {name: 2_710 for name in ("REAL", "PIXEL", "PATCH", "FAKE")}
    if counts != expected:
        raise ValueError(
            f"Paper stress tests require 10,840 balanced images (2,710/class); found {counts}"
        )
    loader = make_loader(
        dataset, batch_size=args.batch_size, workers=args.workers, shuffle=False, seed=args.seed
    )
    model, spec, _ = model_from_checkpoint(
        args.model,
        args.checkpoint,
        num_classes=4,
        device=device,
        pretrained_checkpoint=args.pretrained_checkpoint,
    )
    bundle = predict_loader(model, loader, device)
    output_dir = Path(args.output_dir)
    save_evaluation(
        bundle,
        output_dir,
        title=f"{spec.paper_name}: {args.variant_name}",
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    overall_ci = four_class_bootstrap(
        bundle.labels,
        bundle.predictions,
        replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    diagnostics = target_class_diagnostics(
        bundle.labels,
        bundle.predictions,
        bundle.probabilities,
        args.target_class,
        replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    write_json(output_dir / "overall_bootstrap_ci.json", overall_ci)
    write_json(output_dir / "target_diagnostics.json", diagnostics)
    write_json(output_dir / "evaluation_dataset.json", {"class_counts": counts})
    print(diagnostics)


if __name__ == "__main__":
    main()
