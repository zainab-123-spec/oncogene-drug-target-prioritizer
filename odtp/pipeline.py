"""Build the curated tables, train models, and write figures and reports."""

from __future__ import annotations

import json
import logging
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from odtp.clients.chembl import ChemblClient
from odtp.clients.opentargets import OpenTargetsClient
from odtp.clients.structures import StructureClient
from odtp.config import (
    ACTIVE_PIC50,
    CHEMBL_PAGE_SIZE,
    CHEMBL_WINDOWS,
    DISEASE_TOP_N,
    DISEASES,
    FP_SIZE,
    PER_TARGET_CAP,
    RANDOM_STATE,
    TARGETS,
    Disease,
    Target,
    feature_names,
)
from odtp.features import (
    add_molecular_features,
    assemble_matrix,
    build_target_feature_table,
    filter_small_molecules,
)
from odtp.http import HttpClient
from odtp.model import (
    aggregate_importances,
    evaluate_matrix,
    fit_demo_models,
)
from odtp.plotting import plot_target_associations, plot_target_potency, write_all_figures
from odtp.qc import apply_qc, cap_per_gene, deduplicate_pairs
from odtp.reports import render_benchmark_html, render_benchmark_markdown, render_target_report

logger = logging.getLogger(__name__)


def _versions() -> dict[str, str]:
    import matplotlib
    import numpy
    import pandas
    import rdkit
    import sklearn

    return {
        "python": platform.python_version(),
        "pandas": pandas.__version__,
        "numpy": numpy.__version__,
        "scikit-learn": sklearn.__version__,
        "rdkit": rdkit.__version__,
        "matplotlib": matplotlib.__version__,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default))


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)!r}")


def build_dataset(
    processed_dir: Path,
    *,
    http: HttpClient | None = None,
    opentargets: Any | None = None,
    chembl: Any | None = None,
    structures: Any | None = None,
    diseases: list[Disease] | None = None,
    targets: list[Target] | None = None,
    page_size: int = CHEMBL_PAGE_SIZE,
    windows: int = CHEMBL_WINDOWS,
    per_target_cap: int = PER_TARGET_CAP,
    seed: int = RANDOM_STATE,
    top_n: int = DISEASE_TOP_N,
) -> dict[str, Any]:
    """Download, QC, and save the multi-disease tables."""

    diseases = list(DISEASES if diseases is None else diseases)
    targets = list(TARGETS if targets is None else targets)
    processed_dir.mkdir(parents=True, exist_ok=True)
    if opentargets is None or chembl is None or structures is None:
        if http is None:
            raise ValueError("build_dataset needs an HttpClient or injected API clients")
    opentargets = opentargets or OpenTargetsClient(http)
    chembl = chembl or ChemblClient(http)
    structures = structures or StructureClient(http)

    rankings = opentargets.disease_rankings(diseases, top_n)
    associations = opentargets.curated_associations(diseases, targets)
    details = opentargets.target_details(targets)
    structural = structures.annotate(targets)
    for row in details:
        row.update(structural.get(row["gene"], {}))
    annotations = pd.DataFrame(details)
    association_frame = pd.DataFrame(associations)
    area_map = {target.symbol: target.category for target in targets}
    if not association_frame.empty:
        association_frame["therapeutic_area"] = association_frame["gene"].map(area_map)
    ranking_frame = pd.DataFrame(rankings)

    measurements: list[dict[str, Any]] = []
    fetch_stats: list[dict[str, Any]] = []
    for target in targets:
        rows, stats = chembl.fetch_ic50(target, page_size=page_size, windows=windows)
        measurements.extend(rows)
        fetch_stats.append(stats)

    qc_frame = apply_qc(measurements)
    dedup = deduplicate_pairs(qc_frame)
    featurized = add_molecular_features(dedup)
    if not featurized.empty:
        featurized["therapeutic_area"] = featurized["gene"].map(area_map)
    small = filter_small_molecules(featurized)
    capped = cap_per_gene(small, per_target_cap, seed)
    target_features = build_target_feature_table(association_frame, annotations)
    if capped.empty:
        model_frame = capped
    else:
        model_frame = capped.merge(target_features, on="gene", how="left")

    ranking_frame.to_csv(processed_dir / "disease_rankings.csv", index=False)
    association_frame.to_csv(processed_dir / "gene_disease_scores.csv", index=False)
    annotations.to_csv(processed_dir / "target_annotations.csv", index=False)
    featurized.to_csv(processed_dir / "activities_dedup.csv", index=False)
    model_frame.to_csv(processed_dir / "activities_model.csv", index=False)
    target_features.to_csv(processed_dir / "target_features.csv", index=False)

    funnel = _funnel(fetch_stats, qc_frame, dedup, featurized, small, model_frame, targets)
    funnel.to_csv(processed_dir / "qc_funnel.csv", index=False)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "versions": _versions(),
        "n_diseases": len(diseases),
        "n_targets": len(targets),
        "active_threshold_pic50": ACTIVE_PIC50,
        "page_size": page_size,
        "windows": windows,
        "per_target_cap": per_target_cap,
        "seed": seed,
        "label": "pIC50 = 9 - log10(IC50_nM)",
        "assay_type": "B (binding IC50 only; Ki and functional assays excluded)",
        "known_drug_note": (
            "known_drug_score uses Open Targets datatype id 'known_drug' when "
            "present, otherwise the current 'clinical' datatype."
        ),
        "funnel_totals": {
            "n_fetched": int(funnel["n_fetched"].sum()),
            "n_qc": int(funnel["n_qc"].sum()),
            "n_dedup": int(funnel["n_dedup"].sum()),
            "n_valid_smiles": int(funnel["n_valid_smiles"].sum()),
            "n_small_molecule": int(funnel["n_small_molecule"].sum()),
            "n_model": int(funnel["n_model"].sum()),
            "n_reported_binding_ic50": int(funnel["total_reported"].sum()),
        },
        "http": _http_stats(http),
    }
    _write_json(processed_dir / "summary.json", summary)
    logger.info("Dataset written to %s", processed_dir)
    logger.info("Funnel totals: %s", summary["funnel_totals"])
    return summary


def _http_stats(http: HttpClient | None) -> dict[str, int] | None:
    if http is None:
        return None
    return {"network_calls": http.network_calls, "cache_hits": http.cache_hits}


def _count(frame: pd.DataFrame, gene: str) -> int:
    if frame.empty or "gene" not in frame.columns:
        return 0
    return int((frame["gene"] == gene).sum())


def _funnel(
    fetch_stats: list[dict[str, Any]],
    qc_frame: pd.DataFrame,
    dedup: pd.DataFrame,
    featurized: pd.DataFrame,
    small: pd.DataFrame,
    model_frame: pd.DataFrame,
    targets: list[Target],
) -> pd.DataFrame:
    stats = {row["gene"]: row for row in fetch_stats}
    rows = []
    for target in targets:
        info = stats.get(target.symbol, {})
        rows.append(
            {
                "gene": target.symbol,
                "therapeutic_area": target.category,
                "chembl_id": info.get("chembl_id"),
                "total_reported": int(info.get("total_reported") or 0),
                "n_fetched": int(info.get("n_fetched") or 0),
                "n_qc": _count(qc_frame, target.symbol),
                "n_dedup": _count(dedup, target.symbol),
                "n_valid_smiles": _count(featurized, target.symbol),
                "n_small_molecule": _count(small, target.symbol),
                "n_model": _count(model_frame, target.symbol),
            }
        )
    return pd.DataFrame(rows)


def train_models(
    processed_dir: Path,
    figure_dir: Path,
    report_dir: Path,
    *,
    fast: bool = False,
    write_figures: bool = True,
) -> dict[str, Any]:
    """Evaluate scaffold and random splits and save metrics, figures, and a demo model."""

    model_frame = pd.read_csv(processed_dir / "activities_model.csv")
    dedup_frame = pd.read_csv(processed_dir / "activities_dedup.csv")
    scores = pd.read_csv(processed_dir / "gene_disease_scores.csv")
    matrix, y, active, meta = assemble_matrix(model_frame)
    if len(y) < 40:
        raise RuntimeError(f"Need at least 40 labeled compounds to train, found {len(y)}")
    evaluated = evaluate_matrix(
        matrix,
        y,
        active,
        meta["scaffold"].to_numpy(),
        meta,
        fast=fast,
    )
    names = feature_names(FP_SIZE)
    importances = []
    if evaluated["random_forest_importances"] is not None:
        importances = aggregate_importances(names, evaluated["random_forest_importances"])
    predictions = evaluated["predictions"]
    prediction_columns = [
        "gene",
        "therapeutic_area",
        "molecule_chembl_id",
        "smiles",
        "scaffold",
        "y_true",
        "active_true",
        "pred_Ridge",
        "pred_RandomForest",
        "pred_HistGradientBoosting",
    ]
    keep = [column for column in prediction_columns if column in predictions.columns]
    predictions[keep].to_csv(processed_dir / "predictions_scaffold.csv", index=False)

    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "versions": _versions(),
        "n_compounds": int(len(y)),
        "n_features": int(matrix.shape[1]),
        "active_threshold_pic50": ACTIVE_PIC50,
        "fp_size": FP_SIZE,
        "fast": fast,
        "splits": evaluated["splits"],
        "models": evaluated["models"],
        "leakage": evaluated["leakage"],
        "feature_importance": importances,
        "demo_model_note": (
            "metrics.json is computed on held-out splits. "
            "data/models/demo_hgb.joblib is refit on all modeled rows afterwards "
            "for the Streamlit demo and is not the source of these metrics."
        ),
    }
    _write_json(processed_dir / "metrics.json", metrics)
    summary_path = processed_dir / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    report_dir.mkdir(parents=True, exist_ok=True)
    markdown = render_benchmark_markdown(metrics)
    (report_dir / "benchmarks.md").write_text(markdown)
    (report_dir / "benchmark.html").write_text(render_benchmark_html(metrics, summary))
    if write_figures and importances:
        write_all_figures(
            model_frame=model_frame,
            dedup_frame=dedup_frame,
            scores=scores,
            metrics=metrics,
            predictions=predictions,
            importances=importances,
            figure_dir=figure_dir,
        )
    bundle = fit_demo_models(matrix, y, active, fast=fast)
    bundle["fp_size"] = FP_SIZE
    bundle["feature_names"] = names
    bundle["active_threshold"] = ACTIVE_PIC50
    bundle["note"] = metrics["demo_model_note"]
    model_dir = processed_dir.parent / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_dir / "demo_hgb.joblib")
    logger.info("Metrics written to %s", processed_dir / "metrics.json")
    return metrics


def analyze_target(
    gene: str,
    processed_dir: Path,
    figure_dir: Path,
    report_dir: Path,
) -> Path:
    """Write an HTML report and figures for one gene symbol."""

    symbol = gene.upper()
    dedup = pd.read_csv(processed_dir / "activities_dedup.csv")
    scores = pd.read_csv(processed_dir / "gene_disease_scores.csv")
    annotations = pd.read_csv(processed_dir / "target_annotations.csv")
    if symbol not in set(dedup["gene"].astype(str)):
        available = ", ".join(sorted(dedup["gene"].astype(str).unique()))
        raise SystemExit(f"No measured compounds for {symbol}. Available genes: {available}")
    compounds = dedup.loc[dedup["gene"] == symbol].sort_values("pIC50", ascending=False)
    gene_scores = scores.loc[scores["gene"] == symbol]
    annotation_row = annotations.loc[annotations["gene"] == symbol]
    if annotation_row.empty:
        raise SystemExit(f"No annotation row for {symbol}")
    annotation = annotation_row.iloc[0].to_dict()
    area = str(annotation.get("therapeutic_area") or compounds["therapeutic_area"].iloc[0])
    figure_dir.mkdir(parents=True, exist_ok=True)
    association_name = f"analyze_{symbol}_associations.png"
    potency_name = f"analyze_{symbol}_potency.png"
    plot_target_associations(gene_scores, symbol, figure_dir / association_name)
    plot_target_potency(compounds, symbol, figure_dir / potency_name)
    report_dir.mkdir(parents=True, exist_ok=True)
    html_path = report_dir / f"{symbol}.html"
    html_path.write_text(
        render_target_report(
            gene=symbol,
            area=area,
            annotation=annotation,
            scores=gene_scores.to_dict(orient="records"),
            compounds=compounds.to_dict(orient="records"),
            figure_names=[association_name, potency_name],
        )
    )
    logger.info("Wrote %s", html_path)
    return html_path
