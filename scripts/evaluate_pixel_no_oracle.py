#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from crp_benchmark.data import SingleLabelImageDataset, build_transform
from crp_benchmark.diagnostics import target_class_diagnostics
from crp_benchmark.evaluation import predict_loader, predictions_frame
from crp_benchmark.models import MODEL_SPECS
from crp_benchmark.reproducibility import seed_everything
from crp_benchmark.runtime import choose_device, make_loader, model_from_checkpoint, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained four-way checkpoint on no-oracle/random-placement PIXEL images."
        )
    )
    parser.add_argument("--pixel-root", required=True)
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
    dataset = SingleLabelImageDataset(args.pixel_root, "PIXEL", transform)
    if len(dataset) != 2_710:
        raise ValueError(
            "Paper no-oracle control requires 2,710 PIXEL images, matching the main test set; "
            f"found {len(dataset)}"
        )
    loader = make_loader(
        dataset, batch_size=args.batch_size, workers=args.workers, shuffle=False, seed=args.seed
    )
    model, _spec, _ = model_from_checkpoint(
        args.model,
        args.checkpoint,
        num_classes=4,
        device=device,
        pretrained_checkpoint=args.pretrained_checkpoint,
    )
    bundle = predict_loader(model, loader, device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_frame(bundle, ("REAL", "PIXEL", "PATCH", "FAKE")).to_csv(
        output_dir / "predictions.csv", index=False
    )
    diagnostics = target_class_diagnostics(
        bundle.labels,
        bundle.predictions,
        bundle.probabilities,
        "PIXEL",
        replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    write_json(output_dir / "pixel_no_oracle_metrics.json", diagnostics)
    print(diagnostics)


if __name__ == "__main__":
    main()
