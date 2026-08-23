import numpy as np

from crp_benchmark.stats import (
    mcnemar_paired_correctness,
    paired_binary_bootstrap,
    prevalence_adjusted_precision,
    select_attack_f1_threshold,
    triage_metrics,
    zero_event_upper_false_alarms,
)


def test_prevalence_adjusted_precision_formula():
    value = prevalence_adjusted_precision(tpr=0.9, fpr=0.001, prevalence=0.01)
    expected = 0.009 / (0.009 + 0.00099)
    assert np.isclose(value, expected)


def test_zero_event_bound_matches_paper_note():
    value = zero_event_upper_false_alarms(542, prevalence=0.01)
    assert round(value, 1) == 54.6


def test_triage_has_expected_keys():
    labels = np.array([0] * 10 + [1] * 10)
    scores = np.array([0.01] * 10 + list(np.linspace(0.1, 1.0, 10)))
    result = triage_metrics(labels, scores, prevalence=0.01, target_recall=0.9)
    assert result["observed_recall"] >= 0.9
    assert result["observed_fpr"] == 0.0
    assert 0.0 <= result["auprc"] <= 1.0


def test_exact_mcnemar_counts_discordance():
    labels = np.array([0, 0, 1, 1])
    a = np.array([0, 1, 1, 0])
    b = np.array([0, 0, 0, 0])
    result = mcnemar_paired_correctness(labels, a, b)
    assert result["a_correct_b_wrong"] == 1
    assert result["a_wrong_b_correct"] == 1
    assert result["p_value"] == 1.0


def test_attack_f1_threshold_is_exact_and_deterministic():
    labels = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.4, 0.7, 0.9])
    assert select_attack_f1_threshold(labels, scores) == 0.4


def test_fsc_paired_bootstrap_records_stratified_sampling():
    labels = np.array([0, 0, 0, 1, 1, 1])
    predictions_a = np.array([0, 0, 1, 1, 1, 1])
    predictions_b = np.array([0, 0, 0, 1, 1, 1])
    scores_a = np.array([0.1, 0.2, 0.6, 0.5, 0.7, 0.9])
    scores_b = np.array([0.1, 0.2, 0.3, 0.6, 0.8, 0.9])
    result = paired_binary_bootstrap(
        labels,
        predictions_a,
        scores_a,
        predictions_b,
        scores_b,
        replicates_thresholded=20,
        replicates_auprc=10,
        stratified=True,
    )
    assert result["bootstrap_sampling"] == "class_stratified"
