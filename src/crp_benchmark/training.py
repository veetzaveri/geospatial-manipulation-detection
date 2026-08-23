from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

from .reproducibility import runtime_metadata
from .stats import select_attack_f1_threshold


@dataclass(frozen=True)
class TrainingProtocol:
    name: str
    max_epochs: int
    checkpoint_metric: Literal["macro_f1", "selection_metric"]
    early_stopping_patience: int | None
    min_delta: float
    notes: str


MAIN_PROTOCOL = TrainingProtocol(
    name="main_benchmark_table_2",
    max_epochs=30,
    checkpoint_metric="macro_f1",
    early_stopping_patience=5,
    min_delta=1e-4,
    notes="Published Table 2 main-benchmark protocol.",
)

FSC_PROTOCOL = TrainingProtocol(
    name="fsc_adaptation_table_2",
    max_epochs=20,
    checkpoint_metric="selection_metric",
    early_stopping_patience=3,
    min_delta=1e-4,
    notes=(
        "Published Table 2 FSC protocol: validation macro-F1 for four-way models "
        "and validation attack-class F1 at its selected threshold for binary models."
    ),
)

@dataclass
class FitResult:
    best_epoch: int
    best_metric: float
    best_state_dict: dict[str, torch.Tensor]
    history: list[dict[str, float | int]]
    selection_metric: str


def _forward_loss(
    model: nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    criterion: nn.Module,
    binary: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    logits = model(inputs)
    if binary:
        logits = logits.reshape(-1)
        loss = criterion(logits, labels.float())
        attack_probabilities = torch.sigmoid(logits)
        predictions = (attack_probabilities >= 0.5).long()
    else:
        loss = criterion(logits, labels)
        predictions = torch.argmax(logits, dim=1)
        attack_probabilities = None
    return loss, predictions, attack_probabilities


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    optimizer: AdamW | None,
    binary: bool,
) -> dict[str, float]:
    is_training = optimizer is not None
    model.train(is_training)
    losses = 0.0
    labels_all: list[int] = []
    predictions_all: list[int] = []
    attack_probabilities_all: list[float] = []

    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        for inputs, labels, _paths in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            loss, predictions, attack_probabilities = _forward_loss(
                model, inputs, labels, criterion, binary
            )
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            losses += float(loss.item()) * inputs.size(0)
            labels_all.extend(labels.detach().cpu().numpy().astype(int).tolist())
            predictions_all.extend(predictions.detach().cpu().numpy().astype(int).tolist())
            if attack_probabilities is not None:
                attack_probabilities_all.extend(
                    attack_probabilities.detach().cpu().numpy().astype(float).tolist()
                )

    metrics = {
        "loss": losses / len(loader.dataset),
        "accuracy": float(accuracy_score(labels_all, predictions_all)),
        "macro_f1": float(f1_score(labels_all, predictions_all, average="macro", zero_division=0)),
    }
    if binary:
        labels_array = np.asarray(labels_all, dtype=np.int64)
        scores_array = np.asarray(attack_probabilities_all, dtype=np.float64)
        threshold = select_attack_f1_threshold(labels_array, scores_array)
        selected_predictions = (scores_array >= threshold).astype(np.int64)
        metrics["attack_f1"] = float(
            f1_score(labels_array, selected_predictions, pos_label=1, zero_division=0)
        )
        metrics["attack_f1_threshold"] = float(threshold)
    return metrics


def fit_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    *,
    protocol: TrainingProtocol,
    binary: bool = False,
    learning_rate: float = 3e-5,
    weight_decay: float = 1e-5,
) -> FitResult:
    criterion: nn.Module = nn.BCEWithLogitsLoss() if binary else nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)

    best_metric = -np.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []
    selection_metric = (
        "attack_f1"
        if protocol.checkpoint_metric == "selection_metric" and binary
        else "macro_f1"
    )

    model.to(device)
    for epoch in range(1, protocol.max_epochs + 1):
        train_metrics = _run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            binary=binary,
        )
        val_metrics = _run_epoch(
            model,
            val_loader,
            criterion,
            device,
            optimizer=None,
            binary=binary,
        )
        scheduler.step()
        current_lr = float(optimizer.param_groups[0]["lr"])
        row: dict[str, float | int] = {"epoch": epoch, "lr": current_lr}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"val_{key}": value for key, value in val_metrics.items()})
        history.append(row)

        candidate = val_metrics[selection_metric]
        improved = candidate >= best_metric + protocol.min_delta
        if improved:
            best_metric = candidate
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        message = (
            f"epoch={epoch:02d} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
        )
        if binary:
            message += (
                f" val_attack_f1={val_metrics['attack_f1']:.4f}"
                f" val_attack_threshold={val_metrics['attack_f1_threshold']:.8f}"
            )
        print(message)

        patience = protocol.early_stopping_patience
        if patience is not None and epochs_without_improvement >= patience:
            print(f"early stopping after epoch {epoch}")
            break

    if best_state is None:
        raise RuntimeError("Training produced no checkpoint")
    model.load_state_dict(best_state)
    return FitResult(best_epoch, float(best_metric), best_state, history, selection_metric)


def save_fit_result(
    result: FitResult,
    output_dir: str | Path,
    *,
    model_key: str,
    class_to_idx: dict[str, int],
    protocol: TrainingProtocol,
    extra_metadata: dict | None = None,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "training_history.csv"
    pd.DataFrame(result.history).to_csv(history_path, index=False)

    checkpoint_path = output_dir / "best_checkpoint.pth"
    metadata = {
        "model_key": model_key,
        "class_to_idx": class_to_idx,
        "best_epoch": result.best_epoch,
        "best_metric": result.best_metric,
        "selection_metric": result.selection_metric,
        "protocol": asdict(protocol),
        "runtime": runtime_metadata(),
        **(extra_metadata or {}),
    }
    torch.save({**metadata, "model_state_dict": result.best_state_dict}, checkpoint_path)
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return checkpoint_path
