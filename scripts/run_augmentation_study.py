#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

TOP_FIVE = ("swin_b", "effnet_b5", "swin_t", "satlas_swin_b", "georsclip_b32")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LightAug/MedAug/HeavyAug for the five representative architectures."
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--models", nargs="+", default=TOP_FIVE)
    parser.add_argument("--regimes", nargs="+", default=("lightaug", "medaug", "heavyaug"))
    parser.add_argument(
        "--foundation-checkpoint", action="append", default=[], metavar="MODEL=PATH"
    )
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoints = dict(item.split("=", 1) for item in args.foundation_checkpoint)
    train_script = Path(__file__).with_name("train_main.py")
    for model in args.models:
        for regime in args.regimes:
            output_dir = Path(args.output_root) / model / regime
            command = [
                sys.executable,
                str(train_script),
                "--dataset-root",
                args.dataset_root,
                "--model",
                model,
                "--output-dir",
                str(output_dir),
                "--augmentation",
                regime,
                "--workers",
                str(args.workers),
                "--seed",
                str(args.seed),
                "--device",
                args.device,
                "--bootstrap-replicates",
                str(args.bootstrap_replicates),
            ]
            if args.batch_size is not None:
                command.extend(["--batch-size", str(args.batch_size)])
            if model in checkpoints:
                command.extend(["--pretrained-checkpoint", checkpoints[model]])
            print("running:", " ".join(command))
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
