from __future__ import annotations

import numpy as np

from .constants import CLASS_TO_INDEX
from .stats import percentile_interval


def target_class_diagnostics(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    target_name: str,
    *,
    replicates: int = 10_000,
    seed: int = 42,
) -> dict:
    target_name = target_name.upper()
    target = CLASS_TO_INDEX[target_name]
    real = CLASS_TO_INDEX["REAL"]
    mask = np.asarray(labels) == target
    predictions = np.asarray(predictions)[mask]
    probabilities = np.asarray(probabilities)[mask, target]
    if not len(predictions):
        raise ValueError(f"No true {target_name} examples")
    values = {
        "target_recall": (predictions == target).astype(float),
        "missed_as_REAL": (predictions == real).astype(float),
        "missed_as_other": ((predictions != target) & (predictions != real)).astype(float),
        f"mean_prob_{target_name}": probabilities.astype(float),
    }
    rng = np.random.default_rng(seed)
    output = {}
    for name, vector in values.items():
        boot = np.empty(replicates)
        for index in range(replicates):
            boot[index] = np.mean(rng.choice(vector, size=len(vector), replace=True))
        output[name] = {
            "estimate": float(np.mean(vector)),
            "ci": percentile_interval(boot),
        }
    output["n"] = len(predictions)
    output["bootstrap_replicates"] = replicates
    output["bootstrap_seed"] = seed
    return output
