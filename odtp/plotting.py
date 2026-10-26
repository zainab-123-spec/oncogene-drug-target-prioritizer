"""Publication figures for the multi-disease benchmark.

Matplotlib is forced onto the Agg backend so the CLI can write PNGs on a
machine without a display.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from odtp.config import AREA_COLORS, AREA_ORDER

plt.rcParams.update(
    {
        "figure.dpi": 140,
        "savefig.dpi": 170,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "semibold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
    }
)


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _area_color(area: str) -> str:
    return AREA_COLORS.get(area, "#4B5563")


def plot_dataset_composition(model_frame: pd.DataFrame, path: Path) -> None:
    counts = (
        model_frame.groupby(["gene", "therapeutic_area"], as_index=False)
        .size()
        .rename(columns={"size": "compounds"})
    )
    counts["area_rank"] = counts["therapeutic_area"].map(
        {area: index for index, area in enumerate(AREA_ORDER)}
    )
    counts = counts.sort_values(["area_rank", "gene"])
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = [_area_color(area) for area in counts["therapeutic_area"]]
    ax.bar(counts["gene"], counts["compounds"], color=colors)
    ax.set_ylabel("Small molecules in the training table")
    ax.set_xlabel("")
    ax.set_title("Sampled ChEMBL compounds after QC, per target")
    ax.tick_params(axis="x", rotation=75)
    _legend(ax)
    _save(fig, path)


def plot_pic50(model_frame: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5))
    data = []
    labels = []
    for area in AREA_ORDER:
        values = model_frame.loc[model_frame["therapeutic_area"] == area, "pIC50"]
        if values.empty:
            continue
        data.append(values.to_numpy())
        labels.append(area)
    box = ax.boxplot(data, patch_artist=True, showfliers=False)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    for patch, area in zip(box["boxes"], labels, strict=True):
        patch.set_facecolor(_area_color(area))
        patch.set_alpha(0.75)
    ax.axhline(6.0, color="#111827", linewidth=1, linestyle=":", label="Active threshold (pIC50 6)")
    ax.set_ylabel("Measured pIC50")
    ax.set_title("Biochemical potency by therapeutic area")
    ax.legend(frameon=False)
    _save(fig, path)


def plot_heatmap(scores: pd.DataFrame, path: Path) -> None:
    table = scores.copy()
    if "therapeutic_area" not in table.columns:
        table["therapeutic_area"] = ""
    table["area_rank"] = table["therapeutic_area"].map(
        {area: index for index, area in enumerate(AREA_ORDER)}
    )
    genes = list(dict.fromkeys(table.sort_values(["area_rank", "gene"])["gene"]))
    disease_order = []
    for category in AREA_ORDER:
        names = table.loc[table["category"] == category, "disease_name"].drop_duplicates()
        disease_order.extend(list(names))
    pivot = table.pivot_table(
        index="gene",
        columns="disease_name",
        values="association_score",
        aggfunc="max",
    )
    pivot = pivot.reindex(index=genes, columns=disease_order)
    fig, ax = plt.subplots(figsize=(14, 10))
    image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=55, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_title("Open Targets association score for the curated gene panel")
    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02, label="Association score")
    _save(fig, path)


def plot_split_comparison(metrics: dict[str, Any], path: Path) -> None:
    rows = [
        row
        for row in metrics["models"]
        if row["task"] == "regression" and row["name"] != "MeanBaseline"
    ]
    names = list(dict.fromkeys(row["name"] for row in rows))
    x = np.arange(len(names))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for axis, metric, ylabel in (
        (axes[0], "rmse", "RMSE (pIC50 units)"),
        (axes[1], "roc_auc", "ROC-AUC of predicted pIC50"),
    ):
        scaffold = [_lookup(rows, name, "scaffold", metric) for name in names]
        random_split = [_lookup(rows, name, "random", metric) for name in names]
        axis.bar(x - width / 2, scaffold, width, label="Scaffold split", color="#0F6E6E")
        axis.bar(x + width / 2, random_split, width, label="Random split", color="#C2410C")
        axis.set_xticks(x)
        axis.set_xticklabels(names, rotation=15)
        axis.set_ylabel(ylabel)
        axis.legend(frameon=False, fontsize=9)
    fig.suptitle("Held-out potency models: scaffold split versus random split")
    _save(fig, path)


def plot_pred_vs_actual(predictions: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 6))
    column = "pred_HistGradientBoosting"
    for area in AREA_ORDER:
        subset = predictions.loc[predictions["therapeutic_area"] == area]
        if subset.empty or column not in subset.columns:
            continue
        ax.scatter(
            subset["y_true"],
            subset[column],
            s=16,
            alpha=0.45,
            c=_area_color(area),
            label=area,
            linewidths=0,
        )
    limits = _limits(predictions["y_true"], predictions[column])
    ax.plot(limits, limits, color="#111827", linewidth=1)
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xlabel("Measured pIC50")
    ax.set_ylabel("Predicted pIC50")
    ax.set_title("HistGradientBoosting on the scaffold-held-out test set")
    ax.legend(frameon=False, fontsize=8)
    _save(fig, path)


def plot_roc(predictions: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    y_true = predictions["active_true"].to_numpy()
    colors = {
        "pred_Ridge": "#0369A1",
        "pred_RandomForest": "#6D28D9",
        "pred_HistGradientBoosting": "#0F6E6E",
    }
    for column, color in colors.items():
        if column not in predictions.columns:
            continue
        fpr, tpr, _ = roc_curve(y_true, predictions[column].to_numpy())
        ax.plot(fpr, tpr, color=color, linewidth=2, label=column.replace("pred_", ""))
    ax.plot([0, 1], [0, 1], color="#9CA3AF", linestyle="--", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Scaffold-split ROC using predicted pIC50 as the score")
    ax.legend(frameon=False)
    _save(fig, path)


def plot_importance(rows: list[dict[str, Any]], path: Path) -> None:
    chosen = [row for row in rows if not str(row["feature"]).startswith("Morgan")][:12]
    fingerprint = next(
        (row for row in rows if str(row["feature"]).startswith("Morgan")),
        None,
    )
    chosen = list(reversed(chosen))
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(
        [row["feature"] for row in chosen],
        [row["importance"] for row in chosen],
        color="#0F6E6E",
    )
    title = "Random forest impurity importance (scaffold-split training fold)"
    if fingerprint is not None:
        title += f"\nSummed Morgan bits: {float(fingerprint['importance']):.3f}"
    ax.set_title(title)
    ax.set_xlabel("Importance")
    _save(fig, path)


def plot_lipinski(frame: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    rates = []
    labels = []
    for area in AREA_ORDER:
        subset = frame.loc[frame["therapeutic_area"] == area, "lipinski_violations"]
        if subset.empty:
            continue
        rates.append(float((subset == 0).mean()))
        labels.append(area)
    ax.bar(labels, rates, color=[_area_color(area) for area in labels])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction with zero Lipinski violations")
    ax.set_title("Rule-of-five compliance among deduplicated measured compounds")
    ax.tick_params(axis="x", rotation=20)
    _save(fig, path)


def plot_target_associations(scores: pd.DataFrame, gene: str, path: Path) -> None:
    subset = scores.loc[scores["gene"] == gene].copy()
    area_rank = {area: index for index, area in enumerate(AREA_ORDER)}
    subset["area_rank"] = subset["category"].map(area_rank)
    subset = subset.sort_values(["area_rank", "association_score"], ascending=[True, False])
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [_area_color(area) for area in subset["category"]]
    values = pd.to_numeric(subset["association_score"], errors="coerce").fillna(0)
    ax.barh(subset["disease_name"], values, color=colors)
    ax.set_xlabel("Open Targets association score")
    ax.set_title(f"{gene} across the disease panel")
    ax.invert_yaxis()
    _save(fig, path)


def plot_target_potency(compounds: pd.DataFrame, gene: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    values = compounds["pIC50"].to_numpy()
    ax.hist(values, bins=20, color="#0F6E6E", edgecolor="white")
    ax.axvline(6.0, color="#111827", linestyle=":", label="pIC50 6")
    ax.set_xlabel("pIC50")
    ax.set_ylabel("Compounds")
    ax.set_title(f"{gene} measured potency")
    ax.legend(frameon=False)
    _save(fig, path)


def write_all_figures(
    *,
    model_frame: pd.DataFrame,
    dedup_frame: pd.DataFrame,
    scores: pd.DataFrame,
    metrics: dict[str, Any],
    predictions: pd.DataFrame,
    importances: list[dict[str, Any]],
    figure_dir: Path,
) -> list[Path]:
    """Write the benchmark figure set and return the paths."""

    figure_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "dataset_composition.png": lambda path: plot_dataset_composition(model_frame, path),
        "pic50_by_area.png": lambda path: plot_pic50(model_frame, path),
        "association_heatmap.png": lambda path: plot_heatmap(scores, path),
        "lipinski_by_area.png": lambda path: plot_lipinski(dedup_frame, path),
        "split_comparison.png": lambda path: plot_split_comparison(metrics, path),
        "pred_vs_actual.png": lambda path: plot_pred_vs_actual(predictions, path),
        "roc_scaffold.png": lambda path: plot_roc(predictions, path),
        "feature_importance.png": lambda path: plot_importance(importances, path),
    }
    written: list[Path] = []
    for name, drawer in outputs.items():
        path = figure_dir / name
        drawer(path)
        written.append(path)
    return written


def _lookup(rows: list[dict[str, Any]], name: str, split: str, metric: str) -> float:
    for row in rows:
        if row["name"] == name and row["split"] == split:
            value = row.get(metric)
            return float(value) if value is not None else float("nan")
    return float("nan")


def _limits(*series: pd.Series) -> tuple[float, float]:
    values = np.concatenate([item.to_numpy(dtype=float) for item in series])
    values = values[np.isfinite(values)]
    low = float(np.min(values)) - 0.3
    high = float(np.max(values)) + 0.3
    return low, high


def _legend(axis: plt.Axes) -> None:
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor=color,
            markersize=10,
            label=area,
        )
        for area, color in AREA_COLORS.items()
    ]
    axis.legend(handles=handles, frameon=False, fontsize=8, loc="upper right")
