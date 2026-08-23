#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from crp_benchmark.data import BenchmarkDataset, ManifestDataset, build_transform
from crp_benchmark.evaluation import predict_loader, save_evaluation
from crp_benchmark.models import MODEL_SPECS
from crp_benchmark.reproducibility import seed_everything
from crp_benchmark.runtime import choose_device, make_loader, model_from_checkpoint, write_json
from crp_benchmark.stats import four_class_bootstrap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a four-way CRP checkpoint.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--test-root")
    source.add_argument("--manifest")
    parser.add_argument("--split", default="test")
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
    dataset = (
        BenchmarkDataset(args.test_root, transform)
        if args.test_root
        else ManifestDataset(args.manifest, transform, split=args.split)
    )
    loader = make_loader(
        dataset,
        batch_size=args.batch_size,
        workers=args.workers,
        shuffle=False,
        seed=args.seed,
    )
    model, spec, _metadata = model_from_checkpoint(
        args.model,
        args.checkpoint,
        num_classes=4,
        device=device,
        pretrained_checkpoint=args.pretrained_checkpoint,
    )
    bundle = predict_loader(model, loader, device)
    output_dir = Path(args.output_dir)
    metrics = save_evaluation(
        bundle,
        output_dir,
        title=f"{spec.paper_name} four-way evaluation",
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    ci = four_class_bootstrap(
        bundle.labels,
        bundle.predictions,
        replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    write_json(output_dir / "bootstrap_ci.json", ci)
    print(metrics)


if __name__ == "__main__":
    main()
