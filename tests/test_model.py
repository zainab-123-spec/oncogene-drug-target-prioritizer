"""Split integrity and the shape of recorded metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from odtp.model import compute_leakage, evaluate_matrix, shared_scaffold_count, split_indices


def _synthetic() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(0)
    scaffolds = np.repeat([f"s{index}" for index in range(20)], 10)
    matrix = rng.normal(size=(200, 12))
    signal = np.linspace(4.2, 8.4, 200)
    y = signal + matrix[:, 0] * 0.15
    active = (y >= 6.0).astype(np.int64)
    meta = pd.DataFrame(
        {
            "gene": ["EGFR"] * 200,
            "therapeutic_area": ["Oncology"] * 200,
            "molecule_chembl_id": [f"CHEMBL{index}" for index in range(200)],
            "smiles": ["C"] * 200,
            "scaffold": scaffolds,
        }
    )
    return matrix, y, active, scaffolds, meta


def test_scaffold_split_has_no_shared_scaffolds() -> None:
    scaffolds = np.repeat([f"s{index}" for index in range(20)], 10)
    splits = split_indices(scaffolds, seed=42)
    train_index, test_index = splits["scaffold"]
    assert shared_scaffold_count(scaffolds, train_index, test_index) == 0


def test_random_split_shares_scaffolds_when_analogs_are_repeated() -> None:
    scaffolds = np.repeat([f"s{index}" for index in range(20)], 10)
    splits = split_indices(scaffolds, seed=42)
    train_index, test_index = splits["random"]
    assert shared_scaffold_count(scaffolds, train_index, test_index) > 0


def test_evaluate_returns_required_metrics() -> None:
    matrix, y, active, scaffolds, meta = _synthetic()
    result = evaluate_matrix(matrix, y, active, scaffolds, meta, fast=True)
    assert result["splits"]["scaffold"]["shared_scaffolds"] == 0
    assert result["splits"]["random"]["shared_scaffolds"] > 0
    regressions = [
        row
        for row in result["models"]
        if row["task"] == "regression" and row["name"] == "HistGradientBoosting"
    ]
    assert len(regressions) == 2
    for row in regressions:
        assert row["rmse"] is not None and row["rmse"] > 0
        assert row["pearson_r"] is not None
        assert row["spearman_rho"] is not None
        assert row["roc_auc"] is not None
        assert row["pr_auc"] is not None


def test_classifier_metrics_do_not_invent_rmse() -> None:
    matrix, y, active, scaffolds, meta = _synthetic()
    result = evaluate_matrix(matrix, y, active, scaffolds, meta, fast=True)
    classifiers = [
        row
        for row in result["models"]
        if row["task"] == "classification" and row["name"] == "RandomForest"
    ]
    assert classifiers
    for row in classifiers:
        assert row["rmse"] is None
        assert row["pearson_r"] is None
        assert row["spearman_rho"] is None
        assert row["roc_auc"] is not None
        assert row["pr_auc"] is not None


def test_leakage_gap_positive_when_random_split_is_optimistic() -> None:
    models = [
        {
            "name": "Ridge",
            "task": "regression",
            "split": "scaffold",
            "rmse": 1.2,
            "pearson_r": 0.4,
            "spearman_rho": 0.4,
            "roc_auc": 0.70,
            "pr_auc": 0.60,
        },
        {
            "name": "Ridge",
            "task": "regression",
            "split": "random",
            "rmse": 0.9,
            "pearson_r": 0.6,
            "spearman_rho": 0.55,
            "roc_auc": 0.85,
            "pr_auc": 0.75,
        },
    ]
    gaps = compute_leakage(models)
    regression = gaps["Ridge"]["regression"]
    assert regression["rmse_gap"] == pytest.approx(0.3)
    assert regression["roc_auc_gap"] == pytest.approx(0.15)
    assert regression["pr_auc_gap"] == pytest.approx(0.15)
