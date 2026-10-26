"""Scaffold-split and random-split evaluation for potency models.

Regressors predict pIC50. Classifiers predict active (pIC50 >= 6) versus
inactive. For regressors, ROC-AUC and PR-AUC use the predicted pIC50 as a
ranking score against the active label, so every requested metric is computed
from that model's real outputs. Classifier rows leave RMSE and correlations
empty because those metrics are not defined for a class label.

A positive RMSE gap (scaffold minus random) and a positive AUC gap (random
minus scaffold) mean the random split was optimistic: structurally similar
analogs leaked across the split.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.metrics import (
    average_precision_score,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit
from sklearn.preprocessing import StandardScaler

from odtp.config import RANDOM_STATE, TEST_SIZE

try:
    from scipy.stats import pearsonr, spearmanr
except ImportError:  # pragma: no cover - scipy is a direct dependency
    pearsonr = None
    spearmanr = None


def split_indices(
    scaffolds: np.ndarray,
    *,
    seed: int = RANDOM_STATE,
    test_size: float = TEST_SIZE,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return train and test indices for a scaffold split and a random split."""

    groups = pd.factorize(pd.Series(scaffolds))[0]
    scaffold = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    random_split = ShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    placeholder = np.zeros(len(scaffolds))
    scaffold_train, scaffold_test = next(scaffold.split(placeholder, groups=groups))
    random_train, random_test = next(random_split.split(placeholder))
    return {
        "scaffold": (scaffold_train, scaffold_test),
        "random": (random_train, random_test),
    }


def shared_scaffold_count(
    scaffolds: np.ndarray,
    train_index: np.ndarray,
    test_index: np.ndarray,
) -> int:
    """How many Bemis-Murcko scaffolds appear on both sides of a split."""

    train = set(np.asarray(scaffolds)[train_index])
    test = set(np.asarray(scaffolds)[test_index])
    return len(train & test)


def _correlation(function: Any, left: np.ndarray, right: np.ndarray) -> float | None:
    if function is None or len(left) < 3:
        return None
    if np.unique(left).size < 2 or np.unique(right).size < 2:
        return None
    result = function(left, right)
    value = getattr(result, "statistic", result[0])
    if value is None or not np.isfinite(value):
        return None
    return float(value)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | None]:
    """RMSE, Pearson r, and Spearman rho for one prediction vector."""

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "rmse": rmse,
        "pearson_r": _correlation(pearsonr, y_true, y_pred),
        "spearman_rho": _correlation(spearmanr, y_true, y_pred),
    }


def ranking_metrics(active: np.ndarray, scores: np.ndarray) -> dict[str, float | None]:
    """ROC-AUC and PR-AUC. Undefined if the test fold has a single class."""

    if np.unique(active).size < 2 or np.unique(scores).size < 2:
        return {"roc_auc": None, "pr_auc": None}
    try:
        roc = float(roc_auc_score(active, scores))
    except ValueError:
        roc = None
    try:
        pr = float(average_precision_score(active, scores))
    except ValueError:
        pr = None
    if roc is not None and not np.isfinite(roc):
        roc = None
    if pr is not None and not np.isfinite(pr):
        pr = None
    return {"roc_auc": roc, "pr_auc": pr}


def _impute(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        medians = np.nanmedian(train, axis=0)
    medians = np.where(np.isnan(medians), 0.0, medians)

    def apply(matrix: np.ndarray) -> np.ndarray:
        filled = matrix.copy()
        missing = np.isnan(filled)
        filled[missing] = np.take(medians, np.where(missing)[1])
        return filled

    return apply(train), apply(test), medians


def _model_zoo(fast: bool) -> list[tuple[str, str, Any, bool]]:
    trees = 30 if fast else 200
    depth = 12 if fast else 24
    iterations = 30 if fast else 200
    return [
        ("Ridge", "regression", Ridge(alpha=1.0), True),
        (
            "RandomForest",
            "regression",
            RandomForestRegressor(
                n_estimators=trees,
                max_depth=depth,
                min_samples_leaf=2,
                max_features="sqrt",
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
            False,
        ),
        (
            "HistGradientBoosting",
            "regression",
            HistGradientBoostingRegressor(
                max_iter=iterations,
                learning_rate=0.08,
                max_depth=8,
                min_samples_leaf=20,
                l2_regularization=0.1,
                random_state=RANDOM_STATE,
                early_stopping=False,
            ),
            False,
        ),
        ("RidgeClassifier", "classification", RidgeClassifier(alpha=1.0), True),
        (
            "RandomForest",
            "classification",
            RandomForestClassifier(
                n_estimators=trees,
                max_depth=depth,
                min_samples_leaf=2,
                max_features="sqrt",
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
            False,
        ),
        (
            "HistGradientBoosting",
            "classification",
            HistGradientBoostingClassifier(
                max_iter=iterations,
                learning_rate=0.08,
                max_depth=8,
                min_samples_leaf=20,
                l2_regularization=0.1,
                random_state=RANDOM_STATE,
                early_stopping=False,
            ),
            False,
        ),
    ]


def _scores(model: Any, matrix: np.ndarray, task: str) -> np.ndarray:
    if task == "regression":
        return np.asarray(model.predict(matrix), dtype=np.float64)
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(matrix)[:, 1], dtype=np.float64)
    return np.asarray(model.decision_function(matrix), dtype=np.float64)


def compute_leakage(models: list[dict[str, Any]]) -> dict[str, Any]:
    """Gaps between random and scaffold metrics.

    RMSE gap = scaffold RMSE - random RMSE.
    Ranking gaps = random metric - scaffold metric.
    Positive values mean the random split looked better than the scaffold split.
    """

    index: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in models:
        index[(row["name"], row["task"], row["split"])] = row
    names = sorted({row["name"] for row in models})
    tasks = sorted({row["task"] for row in models})
    report: dict[str, Any] = {}
    for name in names:
        report[name] = {}
        for task in tasks:
            scaffold = index.get((name, task, "scaffold"))
            random_row = index.get((name, task, "random"))
            if scaffold is None or random_row is None:
                continue
            gaps: dict[str, float | None] = {}
            for metric in ("rmse", "pearson_r", "spearman_rho", "roc_auc", "pr_auc"):
                left = scaffold.get(metric)
                right = random_row.get(metric)
                if left is None or right is None:
                    gaps[f"{metric}_gap"] = None
                    continue
                if metric == "rmse":
                    gaps[f"{metric}_gap"] = float(left) - float(right)
                else:
                    gaps[f"{metric}_gap"] = float(right) - float(left)
            report[name][task] = gaps
    return report


def evaluate_matrix(
    matrix: np.ndarray,
    y: np.ndarray,
    active: np.ndarray,
    scaffolds: np.ndarray,
    metadata: pd.DataFrame,
    *,
    fast: bool = False,
    seed: int = RANDOM_STATE,
) -> dict[str, Any]:
    """Train every model on both splits and return metrics plus test predictions."""

    splits = split_indices(scaffolds, seed=seed)
    model_rows: list[dict[str, Any]] = []
    split_info: dict[str, Any] = {}
    prediction_frame: pd.DataFrame | None = None
    importances: list[float] | None = None

    for split_name, (train_index, test_index) in splits.items():
        y_train = y[train_index]
        y_test = y[test_index]
        active_train = active[train_index]
        active_test = active[test_index]
        shared = shared_scaffold_count(scaffolds, train_index, test_index)
        split_info[split_name] = {
            "n_train": int(len(train_index)),
            "n_test": int(len(test_index)),
            "shared_scaffolds": int(shared),
            "test_positive_rate": float(np.mean(active_test)),
            "train_positive_rate": float(np.mean(active_train)),
        }
        x_train, x_test, _medians = _impute(matrix[train_index], matrix[test_index])
        fold_predictions: dict[str, np.ndarray] = {}

        baseline_pred = np.full(y_test.shape, float(np.mean(y_train)))
        baseline = regression_metrics(y_test, baseline_pred)
        baseline.update(ranking_metrics(active_test, baseline_pred))
        baseline.update(
            {
                "name": "MeanBaseline",
                "task": "regression",
                "split": split_name,
            }
        )
        model_rows.append(baseline)
        model_rows.append(
            {
                "name": "PrevalenceBaseline",
                "task": "classification",
                "split": split_name,
                "rmse": None,
                "pearson_r": None,
                "spearman_rho": None,
                "roc_auc": 0.5,
                "pr_auc": float(np.mean(active_test)),
            }
        )

        for name, task, estimator, scale in _model_zoo(fast):
            model = estimator
            train_matrix = x_train
            test_matrix = x_test
            if scale:
                scaler = StandardScaler()
                train_matrix = scaler.fit_transform(x_train)
                test_matrix = scaler.transform(x_test)
            labels = y_train if task == "regression" else active_train
            model.fit(train_matrix, labels)
            scores = _scores(model, test_matrix, task)
            if task == "regression":
                metrics = regression_metrics(y_test, scores)
                metrics.update(ranking_metrics(active_test, scores))
                if split_name == "scaffold":
                    fold_predictions[name] = scores
                    if name == "RandomForest" and hasattr(model, "feature_importances_"):
                        importances = model.feature_importances_.tolist()
            else:
                metrics = {
                    "rmse": None,
                    "pearson_r": None,
                    "spearman_rho": None,
                }
                metrics.update(ranking_metrics(active_test, scores))
            metrics.update({"name": name, "task": task, "split": split_name})
            model_rows.append(metrics)

        if split_name == "scaffold":
            prediction_frame = metadata.iloc[test_index].reset_index(drop=True).copy()
            prediction_frame["y_true"] = y_test
            prediction_frame["active_true"] = active_test
            for name, values in fold_predictions.items():
                prediction_frame[f"pred_{name}"] = values

    return {
        "splits": split_info,
        "models": model_rows,
        "leakage": compute_leakage(model_rows),
        "predictions": prediction_frame,
        "random_forest_importances": importances,
    }


def aggregate_importances(
    feature_names: list[str], importances: list[float]
) -> list[dict[str, float | str]]:
    """Collapse fingerprint bits so the plot shows chemistry and target biology."""

    paired = list(zip(feature_names, importances, strict=True))
    rows: list[dict[str, float | str]] = []
    fingerprint_total = 0.0
    for name, value in paired:
        if name.startswith("fp_"):
            fingerprint_total += float(value)
        else:
            rows.append({"feature": name, "importance": float(value)})
    rows.append({"feature": "Morgan fingerprint (sum of bits)", "importance": fingerprint_total})
    rows.sort(key=lambda item: float(item["importance"]), reverse=True)
    return rows


def fit_demo_models(
    matrix: np.ndarray,
    y: np.ndarray,
    active: np.ndarray,
    *,
    fast: bool = False,
) -> dict[str, Any]:
    """Refit HGB models on every row for the interactive demo.

    Benchmark numbers come from ``evaluate_matrix`` held-out splits. This demo
    model is fit after that evaluation and is not the source of the tables.
    """

    filled, _, medians = _impute(matrix, matrix[:0])
    iterations = 30 if fast else 200
    regressor = HistGradientBoostingRegressor(
        max_iter=iterations,
        learning_rate=0.08,
        max_depth=8,
        min_samples_leaf=20,
        l2_regularization=0.1,
        random_state=RANDOM_STATE,
        early_stopping=False,
    )
    classifier = HistGradientBoostingClassifier(
        max_iter=iterations,
        learning_rate=0.08,
        max_depth=8,
        min_samples_leaf=20,
        l2_regularization=0.1,
        random_state=RANDOM_STATE,
        early_stopping=False,
    )
    regressor.fit(filled, y)
    classifier.fit(filled, active)
    return {"regressor": regressor, "classifier": classifier, "medians": medians}


def predict_smiles(
    bundle: dict[str, Any],
    smiles: str,
    target_row: dict[str, Any] | pd.Series,
) -> dict[str, Any]:
    """Featurize one SMILES against a saved target row and score the demo models."""

    from odtp.features import (
        MOLECULE_FEATURES,
        TARGET_FEATURES,
        compute_molecule_features,
        morgan_bits,
    )

    chemistry = compute_molecule_features(smiles)
    if chemistry is None:
        raise ValueError("RDKit could not parse that SMILES.")
    bits = morgan_bits(smiles, fp_size=fingerprint_size(bundle))
    if bits is None:
        raise ValueError("RDKit could not fingerprint that SMILES.")
    values = [float(chemistry[name]) for name in MOLECULE_FEATURES]
    for name in TARGET_FEATURES:
        raw = target_row[name]
        values.append(float(raw) if raw is not None and raw == raw else float("nan"))
    vector = np.concatenate([np.asarray(values, dtype=np.float64), bits])
    predicted = predict_vector(bundle, vector)
    predicted.update(chemistry)
    return predicted


def fingerprint_size(bundle: dict[str, Any]) -> int:
    """Number of Morgan bits the saved demo model expects."""

    if bundle.get("fp_size"):
        return int(bundle["fp_size"])
    from odtp.config import MOLECULE_FEATURES, TARGET_FEATURES

    width = int(np.asarray(bundle["medians"]).shape[0])
    return width - len(MOLECULE_FEATURES) - len(TARGET_FEATURES)


def predict_vector(
    bundle: dict[str, Any],
    vector: np.ndarray,
) -> dict[str, float]:
    """Predict pIC50 and active probability for one imputed feature row."""

    medians = np.asarray(bundle["medians"], dtype=np.float64)
    row = vector.astype(np.float64).copy()
    missing = np.isnan(row)
    row[missing] = medians[missing]
    pic50 = float(bundle["regressor"].predict(row.reshape(1, -1))[0])
    probability = float(bundle["classifier"].predict_proba(row.reshape(1, -1))[0, 1])
    return {"pIC50": pic50, "active_probability": probability}
