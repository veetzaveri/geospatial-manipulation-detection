from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
)

Array = np.ndarray


def pairwise_decision_metrics(labels: Array, predictions: Array) -> dict[str, float]:
    """Balanced accuracy and attack F1 for a matched REAL/attack subset.

    Truth uses 0=REAL and 1=the evaluated attack. Predictions may additionally
    use -1 for a different attack family under Multi-E. Such a prediction is
    wrong for either true class; it must never be relabeled as REAL.

    Attack F1 is the one-vs-rest F1 of class 1. Other-family predictions on
    attack examples are false negatives, while only predictions of class 1
    on REAL examples are false positives for this specific attack class.
    """
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    if labels.ndim != 1 or predictions.ndim != 1 or labels.shape != predictions.shape:
        raise ValueError("labels and predictions must be equal-length one-dimensional arrays")
    if not len(labels) or not np.isin(labels, [0, 1]).all():
        raise ValueError("Truth must contain REAL=0 and/or attack=1")
    if not np.isin(predictions, [-1, 0, 1]).all():
        raise ValueError("Predictions must be REAL=0, attack=1, or other-family=-1")
    recalls = [
        float(np.mean(predictions[labels == label] == label))
        for label in (0, 1)
        if np.any(labels == label)
    ]
    true_positive = int(np.sum((labels == 1) & (predictions == 1)))
    false_positive = int(np.sum((labels == 0) & (predictions == 1)))
    false_negative = int(np.sum((labels == 1) & (predictions != 1)))
    denominator = 2 * true_positive + false_positive + false_negative
    return {
        "balanced_accuracy": float(np.mean(recalls)),
        "f1_attack": float(2 * true_positive / denominator) if denominator else 0.0,
    }


def select_attack_f1_threshold(labels: Array, scores: Array) -> float:
    """Select the validation threshold maximizing attack-class F1.

    Ties are resolved by choosing the threshold closest to 0.5.
    """

    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or scores.ndim != 1 or len(labels) != len(scores):
        raise ValueError("labels and scores must be one-dimensional arrays of equal length")
    if len(labels) == 0:
        raise ValueError("threshold selection requires at least one example")

    candidates = np.unique(np.concatenate(([0.0], scores, [1.0])))
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    positive_prefix = np.concatenate(([0], np.cumsum(sorted_labels == 1)))
    first_included = np.searchsorted(sorted_scores, candidates, side="left")
    predicted_positive = len(labels) - first_included
    total_positive = int(positive_prefix[-1])
    true_positive = total_positive - positive_prefix[first_included]
    false_positive = predicted_positive - true_positive
    false_negative = total_positive - true_positive
    denominator = 2 * true_positive + false_positive + false_negative
    f1_values = np.divide(
        2 * true_positive,
        denominator,
        out=np.zeros_like(denominator, dtype=float),
        where=denominator != 0,
    )
    best_f1 = np.max(f1_values)
    tied = np.flatnonzero(np.isclose(f1_values, best_f1, rtol=0.0, atol=1e-15))
    distances = np.abs(candidates[tied] - 0.5)
    # candidates are ascending, so argmin also preserves the documented lower
    # threshold tie-break when distances are identical.
    return float(candidates[tied[int(np.argmin(distances))]])


def stratified_resample_indices(
    labels: Array,
    rng: np.random.Generator,
) -> Array:
    labels = np.asarray(labels)
    pieces: list[Array] = []
    for label in np.unique(labels):
        members = np.flatnonzero(labels == label)
        pieces.append(rng.choice(members, size=len(members), replace=True))
    indices = np.concatenate(pieces)
    rng.shuffle(indices)
    return indices


def resample_indices(
    labels: Array,
    rng: np.random.Generator,
    *,
    stratified: bool = False,
) -> Array:
    """Draw an image-level bootstrap sample, optionally preserving class counts."""

    labels = np.asarray(labels)
    if stratified:
        return stratified_resample_indices(labels, rng)
    return rng.choice(np.arange(len(labels)), size=len(labels), replace=True)


def percentile_interval(values: Array, confidence: float = 0.95) -> tuple[float, float]:
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(np.asarray(values, dtype=float), [alpha, 1.0 - alpha])
    return float(low), float(high)


def bootstrap_interval(
    labels: Array,
    metric: Callable[[Array], float],
    *,
    replicates: int,
    seed: int = 42,
    confidence: float = 0.95,
    stratified: bool = False,
) -> tuple[float, float, float]:
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    point = float(metric(np.arange(len(labels))))
    values = np.empty(replicates, dtype=float)
    for replicate in range(replicates):
        indices = resample_indices(labels, rng, stratified=stratified)
        values[replicate] = metric(indices)
    low, high = percentile_interval(values, confidence)
    return point, low, high


def four_class_bootstrap(
    labels: Array,
    predictions: Array,
    *,
    replicates: int = 10_000,
    seed: int = 42,
    num_classes: int = 4,
) -> dict:
    labels = np.asarray(labels, dtype=int)
    predictions = np.asarray(predictions, dtype=int)

    metrics: dict[str, Callable[[Array], float]] = {
        "accuracy": lambda idx: accuracy_score(labels[idx], predictions[idx]),
        "macro_f1": lambda idx: f1_score(
            labels[idx],
            predictions[idx],
            labels=list(range(num_classes)),
            average="macro",
            zero_division=0,
        ),
    }
    for class_index in range(num_classes):
        metrics[f"f1_class_{class_index}"] = lambda idx, c=class_index: f1_score(
            labels[idx],
            predictions[idx],
            labels=[c],
            average="macro",
            zero_division=0,
        )

    return {
        name: dict(
            zip(
                ("estimate", "ci_low", "ci_high"),
                bootstrap_interval(
                    labels,
                    function,
                    replicates=replicates,
                    seed=seed,
                ),
            )
        )
        for name, function in metrics.items()
    }


def paired_binary_bootstrap(
    labels: Array,
    predictions_a: Array,
    scores_a: Array,
    predictions_b: Array,
    scores_b: Array,
    *,
    replicates_thresholded: int = 10_000,
    replicates_auprc: int = 1_000,
    seed: int = 42,
    stratified: bool = False,
) -> dict:
    """Paired comparison of hard decisions and scores on identical examples."""

    labels = np.asarray(labels, dtype=int)
    pred_a = np.asarray(predictions_a, dtype=int)
    scores_a = np.asarray(scores_a, dtype=float)
    pred_b = np.asarray(predictions_b, dtype=int)
    scores_b = np.asarray(scores_b, dtype=float)
    lengths = {len(labels), len(pred_a), len(scores_a), len(pred_b), len(scores_b)}
    if len(lengths) != 1:
        raise ValueError("Labels, predictions, and scores must have identical lengths")

    def ba(predictions: Array, idx: Array) -> float:
        return pairwise_decision_metrics(labels[idx], predictions[idx])["balanced_accuracy"]

    def attack_f1(predictions: Array, idx: Array) -> float:
        return pairwise_decision_metrics(labels[idx], predictions[idx])["f1_attack"]

    rng = np.random.default_rng(seed)
    ba_delta = np.empty(replicates_thresholded)
    f1_delta = np.empty(replicates_thresholded)
    for replicate in range(replicates_thresholded):
        idx = resample_indices(labels, rng, stratified=stratified)
        ba_delta[replicate] = ba(pred_b, idx) - ba(pred_a, idx)
        f1_delta[replicate] = attack_f1(pred_b, idx) - attack_f1(pred_a, idx)

    rng = np.random.default_rng(seed)
    auprc_delta = np.empty(replicates_auprc)
    for replicate in range(replicates_auprc):
        idx = resample_indices(labels, rng, stratified=stratified)
        auprc_delta[replicate] = average_precision_score(
            labels[idx], scores_b[idx]
        ) - average_precision_score(labels[idx], scores_a[idx])

    return {
        "a": {
            "balanced_accuracy": ba(pred_a, np.arange(len(labels))),
            "f1_attack": attack_f1(pred_a, np.arange(len(labels))),
            "auprc": float(average_precision_score(labels, scores_a)),
        },
        "b": {
            "balanced_accuracy": ba(pred_b, np.arange(len(labels))),
            "f1_attack": attack_f1(pred_b, np.arange(len(labels))),
            "auprc": float(average_precision_score(labels, scores_b)),
        },
        "delta_b_minus_a": {
            "balanced_accuracy": {
                "estimate": ba(pred_b, np.arange(len(labels))) - ba(pred_a, np.arange(len(labels))),
                "ci": percentile_interval(ba_delta),
            },
            "f1_attack": {
                "estimate": attack_f1(pred_b, np.arange(len(labels)))
                - attack_f1(pred_a, np.arange(len(labels))),
                "ci": percentile_interval(f1_delta),
            },
            "auprc": {
                "estimate": float(average_precision_score(labels, scores_b))
                - float(average_precision_score(labels, scores_a)),
                "ci": percentile_interval(auprc_delta),
            },
        },
        "mcnemar": mcnemar_paired_correctness(labels, pred_a, pred_b),
        "bootstrap_sampling": "class_stratified" if stratified else "ordinary_image_level",
    }


def mcnemar_paired_correctness(
    labels: Array,
    predictions_a: Array,
    predictions_b: Array,
) -> dict[str, float | int | str]:
    """Exact two-sided McNemar test on paired hard-decision correctness.

    Uses the exact binomial form and records the method in its outputs.
    """

    correct_a = np.asarray(predictions_a) == np.asarray(labels)
    correct_b = np.asarray(predictions_b) == np.asarray(labels)
    a_only = int(np.sum(correct_a & ~correct_b))
    b_only = int(np.sum(~correct_a & correct_b))
    discordant = a_only + b_only
    p_value = (
        1.0 if discordant == 0 else float(binomtest(min(a_only, b_only), discordant, 0.5).pvalue)
    )
    return {
        "a_correct_b_wrong": a_only,
        "a_wrong_b_correct": b_only,
        "discordant": discordant,
        "p_value": p_value,
        "method": "exact_two_sided_binomial",
    }


def operating_point_at_recall(
    labels: Array,
    scores: Array,
    target_recall: float = 0.90,
) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    positives = scores[labels == 1]
    if len(positives) == 0:
        raise ValueError("At least one manipulated example is required")

    required_true_positives = max(1, int(np.ceil(target_recall * len(positives))))
    ranked_positive_scores = np.sort(positives)[::-1]
    threshold = float(ranked_positive_scores[required_true_positives - 1])
    predictions = scores >= threshold
    tpr = float(np.mean(predictions[labels == 1]))
    fpr = float(np.mean(predictions[labels == 0]))
    return {"threshold": threshold, "tpr": tpr, "fpr": fpr}


def prevalence_adjusted_precision(tpr: float, fpr: float, prevalence: float) -> float:
    numerator = prevalence * tpr
    denominator = numerator + (1.0 - prevalence) * fpr
    return float(numerator / denominator) if denominator > 0 else 1.0


def prevalence_adjusted_auprc(
    labels: Array,
    scores: Array,
    prevalence: float,
) -> float:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores, kind="mergesort")
    ordered_labels = labels[order]
    ordered_scores = scores[order]
    tp_all = np.cumsum(ordered_labels == 1)
    fp_all = np.cumsum(ordered_labels == 0)
    # Evaluate once at the end of each tied-score group so AP is invariant to
    # the within-tie ordering of samples.
    group_ends = np.flatnonzero(np.r_[ordered_scores[1:] != ordered_scores[:-1], True])
    tp = tp_all[group_ends]
    fp = fp_all[group_ends]
    positives = max(1, int(np.sum(labels == 1)))
    negatives = max(1, int(np.sum(labels == 0)))
    recall = tp / positives
    tpr = recall
    fpr = fp / negatives
    adjusted_precision = np.array(
        [prevalence_adjusted_precision(a, b, prevalence) for a, b in zip(tpr, fpr)]
    )
    recall_previous = np.concatenate(([0.0], recall[:-1]))
    return float(np.sum((recall - recall_previous) * adjusted_precision))


def triage_metrics(
    labels: Array,
    scores: Array,
    *,
    prevalence: float = 0.01,
    target_recall: float = 0.90,
) -> dict[str, float]:
    point = operating_point_at_recall(labels, scores, target_recall)
    return {
        "prevalence": float(prevalence),
        "target_recall": float(target_recall),
        "threshold": point["threshold"],
        "observed_recall": point["tpr"],
        "observed_fpr": point["fpr"],
        "precision": prevalence_adjusted_precision(point["tpr"], point["fpr"], prevalence),
        "auprc": prevalence_adjusted_auprc(labels, scores, prevalence),
        "false_alarms_per_10000": float((1.0 - prevalence) * point["fpr"] * 10_000.0),
    }


def bootstrap_triage(
    labels: Array,
    scores: Array,
    *,
    prevalence: float = 0.01,
    target_recall: float = 0.90,
    replicates: int = 2_000,
    seed: int = 42,
) -> dict:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    point = triage_metrics(labels, scores, prevalence=prevalence, target_recall=target_recall)
    keys = ("precision", "auprc", "false_alarms_per_10000")
    values = {key: np.empty(replicates, dtype=float) for key in keys}
    rng = np.random.default_rng(seed)
    for replicate in range(replicates):
        idx = resample_indices(labels, rng, stratified=False)
        metrics = triage_metrics(
            labels[idx],
            scores[idx],
            prevalence=prevalence,
            target_recall=target_recall,
        )
        for key in keys:
            values[key][replicate] = metrics[key]
    return {
        **point,
        "confidence_intervals": {key: percentile_interval(values[key]) for key in keys},
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
    }


def zero_event_upper_false_alarms(
    negatives: int,
    *,
    prevalence: float = 0.01,
    confidence: float = 0.95,
) -> float:
    """Exact one-sided upper false-alarm bound after zero observed false positives."""

    if negatives <= 0:
        raise ValueError("negatives must be positive")
    alpha = 1.0 - confidence
    upper_fpr = 1.0 - alpha ** (1.0 / negatives)
    return float((1.0 - prevalence) * upper_fpr * 10_000.0)
