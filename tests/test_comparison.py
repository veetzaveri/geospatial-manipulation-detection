from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


def comparison_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "compare_binary_multi.py"
    spec = importlib.util.spec_from_file_location("compare_binary_multi", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exact_family_mapping_preserves_other_family_errors():
    module = comparison_module()
    result = module.exact_family_predictions(["REAL", "FAKE", "PATCH", "PIXEL"], "FAKE")
    np.testing.assert_array_equal(result, [0, 1, -1, -1])


def test_alignment_does_not_relabel_wrong_family_as_real(tmp_path):
    module = comparison_module()
    binary = pd.DataFrame({
        "path": ["real.png", "fake.png"],
        "true_label_name": ["REAL", "FAKE"],
        "pred_label_name": ["REAL", "FAKE"],
        "prob_FAKE": [0.1, 0.9],
    })
    multi = pd.DataFrame({
        "path": ["real.png", "fake.png"],
        "true_label_name": ["REAL", "FAKE"],
        "pred_label_name": ["PATCH", "FAKE"],
        "prob_REAL": [0.2, 0.1],
        "prob_FAKE": [0.1, 0.8],
    })
    binary_path = tmp_path / "binary.csv"
    multi_path = tmp_path / "multi.csv"
    binary.to_csv(binary_path, index=False)
    multi.to_csv(multi_path, index=False)
    aligned = module.aligned(str(binary_path), str(multi_path), "FAKE", "path")
    np.testing.assert_array_equal(aligned.pred_multi_e, [-1, 1])
    np.testing.assert_array_equal(aligned.pred_multi_c, [1, 1])
    result = module.metrics(
        aligned.label.to_numpy(),
        aligned.pred_multi_e.to_numpy(),
        aligned.score_multi_e.to_numpy(),
        "exact-family",
    )
    assert result["balanced_accuracy"] == 0.5
