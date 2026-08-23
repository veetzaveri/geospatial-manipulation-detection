from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader

from .constants import CANONICAL_CLASSES, FAKE_SUBTYPES
from .data import fake_subtype_from_path
from .stats import percentile_interval


@dataclass
class PredictionBundle:
    labels: np.ndarray
    predictions: np.ndarray
    probabilities: np.ndarray
    paths: list[str]


def predict_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    binary: bool = False,
    threshold: float = 0.5,
) -> PredictionBundle:
    model.eval()
    labels: list[int] = []
    predictions: list[int] = []
    probabilities: list[np.ndarray] = []
    paths: list[str] = []

    with torch.no_grad():
        for inputs, batch_labels, batch_paths in loader:
            logits = model(inputs.to(device, non_blocking=True))
            if binary:
                attack_probability = torch.sigmoid(logits.reshape(-1))
                batch_probabilities = torch.stack(
                    (1.0 - attack_probability, attack_probability), dim=1
                )
                batch_predictions = (attack_probability >= threshold).long()
            else:
                batch_probabilities = torch.softmax(logits, dim=1)
                batch_predictions = torch.argmax(logits, dim=1)
            labels.extend(batch_labels.numpy().astype(int).tolist())
            predictions.extend(batch_predictions.cpu().numpy().astype(int).tolist())
            probabilities.extend(batch_probabilities.cpu().numpy())
            paths.extend(list(batch_paths))

    return PredictionBundle(
        labels=np.asarray(labels, dtype=np.int64),
        predictions=np.asarray(predictions, dtype=np.int64),
        probabilities=np.asarray(probabilities, dtype=np.float64),
        paths=paths,
    )


def four_class_metrics(bundle: PredictionBundle) -> dict[str, float | dict]:
    labels = list(range(len(CANONICAL_CLASSES)))
    report = classification_report(
        bundle.labels,
        bundle.predictions,
        labels=labels,
        target_names=CANONICAL_CLASSES,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(bundle.labels, bundle.predictions)),
        "macro_f1": float(
            f1_score(
                bundle.labels,
                bundle.predictions,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "per_class_f1": {name: float(report[name]["f1-score"]) for name in CANONICAL_CLASSES},
    }


def binary_metrics(bundle: PredictionBundle) -> dict[str, float]:
    return {
        "balanced_accuracy": float(balanced_accuracy_score(bundle.labels, bundle.predictions)),
        "f1_attack": float(
            f1_score(bundle.labels, bundle.predictions, pos_label=1, zero_division=0)
        ),
        "auprc_attack": float(average_precision_score(bundle.labels, bundle.probabilities[:, 1])),
    }


def predictions_frame(
    bundle: PredictionBundle,
    class_names: Sequence[str],
) -> pd.DataFrame:
    rows: list[dict] = []
    for index, (label, prediction, path) in enumerate(
        zip(bundle.labels, bundle.predictions, bundle.paths)
    ):
        row = {
            "path": path,
            "true_label_idx": int(label),
            "true_label_name": class_names[int(label)],
            "pred_label_idx": int(prediction),
            "pred_label_name": class_names[int(prediction)],
            "fake_subtype": fake_subtype_from_path(path),
        }
        for class_index, class_name in enumerate(class_names):
            row[f"prob_{class_name}"] = float(bundle.probabilities[index, class_index])
        rows.append(row)
    return pd.DataFrame(rows)


def fake_subtype_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    fake = predictions[
        (predictions["true_label_name"] == "FAKE")
        & predictions["fake_subtype"].isin(FAKE_SUBTYPES)
    ].copy()
    rows: list[dict] = []
    for subtype, group in fake.groupby("fake_subtype", sort=True):
        n = len(group)
        rows.append(
            {
                "fake_subtype": subtype,
                "n_samples": n,
                "recall_to_FAKE": float((group.pred_label_name == "FAKE").mean()),
                "rate_predicted_as_REAL": float((group.pred_label_name == "REAL").mean()),
                "rate_predicted_as_PIXEL": float((group.pred_label_name == "PIXEL").mean()),
                "rate_predicted_as_PATCH": float((group.pred_label_name == "PATCH").mean()),
                "mean_prob_FAKE": float(group["prob_FAKE"].mean()),
            }
        )
    return pd.DataFrame(rows)


def fake_subtype_bootstrap_recall(
    predictions: pd.DataFrame,
    *,
    replicates: int = 10_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Paper Table 5 FAKE recall with ordinary within-subtype bootstrap CIs."""

    fake = predictions[
        (predictions["true_label_name"] == "FAKE")
        & predictions["fake_subtype"].isin(FAKE_SUBTYPES)
    ].copy()
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for subtype, group in fake.groupby("fake_subtype", sort=True):
        successes = (group.pred_label_name == "FAKE").to_numpy(dtype=float)
        boot = np.empty(replicates, dtype=float)
        for replicate in range(replicates):
            boot[replicate] = np.mean(
                rng.choice(successes, size=len(successes), replace=True)
            )
        low, high = percentile_interval(boot)
        rows.append(
            {
                "fake_subtype": subtype,
                "n_samples": len(successes),
                "recall_to_FAKE": float(np.mean(successes)),
                "ci_low": low,
                "ci_high": high,
                "bootstrap_replicates": replicates,
                "bootstrap_seed": seed,
                "bootstrap_sampling": "ordinary_image_level_within_subtype",
            }
        )
    return pd.DataFrame(rows)


def save_evaluation(
    bundle: PredictionBundle,
    output_dir: str | Path,
    *,
    class_names: Sequence[str] = CANONICAL_CLASSES,
    title: str = "Confusion Matrix",
    bootstrap_replicates: int = 10_000,
    seed: int = 42,
) -> dict[str, float | dict]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = predictions_frame(bundle, class_names)
    frame.to_csv(output_dir / "predictions.csv", index=False)
    metrics = binary_metrics(bundle) if len(class_names) == 2 else four_class_metrics(bundle)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    labels = list(range(len(class_names)))
    cm = confusion_matrix(bundle.labels, bundle.predictions, labels=labels)
    display = ConfusionMatrixDisplay(cm, display_labels=class_names)
    figure, axis = plt.subplots(figsize=(6, 6))
    display.plot(ax=axis, cmap="Blues", values_format="d", colorbar=False)
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close(figure)

    if len(class_names) == 4 and "FAKE" in class_names:
        subtype = fake_subtype_metrics(frame)
        if not subtype.empty:
            subtype.to_csv(output_dir / "fake_subtype_metrics.csv", index=False)
            subtype_ci = fake_subtype_bootstrap_recall(
                frame,
                replicates=bootstrap_replicates,
                seed=seed,
            )
            subtype_ci.to_csv(output_dir / "fake_subtype_bootstrap_ci.csv", index=False)
    return metrics
