"""Markdown and HTML reports. Numbers are formatted from measured metric dicts."""

from __future__ import annotations

import html
from typing import Any

REGRESSION_MODELS = (
    "MeanBaseline",
    "Ridge",
    "RandomForest",
    "HistGradientBoosting",
)
CLASSIFICATION_MODELS = (
    "PrevalenceBaseline",
    "RidgeClassifier",
    "RandomForest",
    "HistGradientBoosting",
)


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if number != number:
        return "—"
    return f"{number:.{digits}f}"


def _rows(metrics: dict[str, Any], task: str, names: tuple[str, ...]) -> list[dict[str, Any]]:
    wanted = []
    for name in names:
        for split in ("scaffold", "random"):
            for row in metrics["models"]:
                if row["name"] == name and row["task"] == task and row["split"] == split:
                    wanted.append(row)
    return wanted


def render_benchmark_markdown(metrics: dict[str, Any]) -> str:
    """Benchmark table sourced only from a metrics dictionary produced by a run."""

    lines = [
        "Numbers below are copied from the training run's `data/processed/metrics.json`.",
        "",
        "### Regression (predict pIC50)",
        "",
        "| Model | Split | RMSE | Pearson r | Spearman rho | ROC-AUC | PR-AUC |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in _rows(metrics, "regression", REGRESSION_MODELS):
        lines.append(
            "| {name} | {split} | {rmse} | {pearson} | {spearman} | {roc} | {pr} |".format(
                name=row["name"],
                split=row["split"],
                rmse=_fmt(row.get("rmse")),
                pearson=_fmt(row.get("pearson_r")),
                spearman=_fmt(row.get("spearman_rho")),
                roc=_fmt(row.get("roc_auc")),
                pr=_fmt(row.get("pr_auc")),
            )
        )
    lines.extend(
        [
            "",
            "### Classification (active if pIC50 ≥ 6)",
            "",
            "RMSE and correlations are left blank for classifiers. ROC-AUC and PR-AUC use the model's ranking score (predicted probability, or the Ridge decision function).",
            "",
            "| Model | Split | ROC-AUC | PR-AUC |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for row in _rows(metrics, "classification", CLASSIFICATION_MODELS):
        lines.append(
            "| {name} | {split} | {roc} | {pr} |".format(
                name=row["name"],
                split=row["split"],
                roc=_fmt(row.get("roc_auc")),
                pr=_fmt(row.get("pr_auc")),
            )
        )
    leakage = metrics.get("leakage") or {}
    lines.extend(["", "### Scaffold leakage gaps", ""])
    lines.append(
        "RMSE gap = scaffold RMSE − random RMSE. "
        "AUC gap = random ROC-AUC − scaffold ROC-AUC. "
        "A positive gap means the random split was optimistic."
    )
    lines.append("")
    lines.append("| Model | Task | RMSE gap | ROC-AUC gap | PR-AUC gap |")
    lines.append("| --- | --- | ---: | ---: | ---: |")
    for name, tasks in leakage.items():
        for task, gaps in tasks.items():
            lines.append(
                f"| {name} | {task} | {_fmt(gaps.get('rmse_gap'))} | "
                f"{_fmt(gaps.get('roc_auc_gap'))} | {_fmt(gaps.get('pr_auc_gap'))} |"
            )
    splits = metrics.get("splits") or {}
    lines.extend(["", "### Split sizes", ""])
    for name, info in splits.items():
        lines.append(
            f"- **{name}**: n_train={info['n_train']}, n_test={info['n_test']}, "
            f"shared scaffolds={info['shared_scaffolds']}, "
            f"test active rate={_fmt(info['test_positive_rate'])}"
        )
    lines.append("")
    return "\n".join(lines)


def render_benchmark_html(metrics: dict[str, Any], summary: dict[str, Any] | None = None) -> str:
    """A single self-contained HTML page for the benchmark."""

    table = render_benchmark_markdown(metrics)
    pre = html.escape(table)
    dataset = ""
    if summary:
        dataset = f"<pre>{html.escape(str(summary.get('funnel_totals', summary)))}</pre>"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Multi-disease bioactivity benchmark</title>
  <style>
    body {{ font-family: Georgia, serif; margin: 2.5rem auto; max-width: 920px; color: #1f2933; line-height: 1.45; }}
    pre {{ white-space: pre-wrap; font-family: ui-monospace, monospace; font-size: 0.92rem; background: #f4f7f7; padding: 1rem; }}
  </style>
</head>
<body>
  <h1>Multi-Disease Drug Target &amp; Bioactivity Prioritizer</h1>
  <p>Held-out metrics from the local training run. This page does not invent numbers; it renders <code>metrics.json</code>.</p>
  {dataset}
  <pre>{pre}</pre>
</body>
</html>
"""


def render_target_report(
    *,
    gene: str,
    area: str,
    annotation: dict[str, Any],
    scores: list[dict[str, Any]],
    compounds: list[dict[str, Any]],
    figure_names: list[str],
) -> str:
    """HTML report for one gene."""

    def cell(value: Any) -> str:
        if value is None or value == "":
            return "—"
        return html.escape(str(value))

    score_rows = []
    for row in scores:
        score_rows.append(
            "<tr>"
            f"<td>{cell(row.get('disease_name'))}</td>"
            f"<td>{cell(row.get('category'))}</td>"
            f"<td>{cell(_fmt(row.get('association_score')))}</td>"
            f"<td>{cell(_fmt(row.get('genetic_association')))}</td>"
            f"<td>{cell(_fmt(row.get('somatic_mutation')))}</td>"
            f"<td>{cell(_fmt(row.get('known_drug_score')))}</td>"
            f"<td>{cell(row.get('known_drug_source'))}</td>"
            "</tr>"
        )
    compound_rows = []
    for row in compounds[:15]:
        compound_rows.append(
            "<tr>"
            f"<td>{cell(row.get('molecule_chembl_id'))}</td>"
            f"<td>{cell(_fmt(row.get('pIC50')))}</td>"
            f"<td>{cell(_fmt(row.get('ic50_nM'), 1))}</td>"
            f"<td>{cell(_fmt(row.get('mw'), 1))}</td>"
            f"<td>{cell(_fmt(row.get('logp')))}</td>"
            f"<td>{cell(row.get('lipinski_violations'))}</td>"
            f"<td><code>{cell(row.get('canonical_smiles') or row.get('smiles'))}</code></td>"
            "</tr>"
        )
    figures = "\n".join(
        f'<figure><img src="../figures/{html.escape(name)}" alt="{html.escape(name)}" /></figure>'
        for name in figure_names
    )
    fields = [
        ("UniProt", annotation.get("uniprot")),
        ("Protein", annotation.get("protein_name")),
        ("Sequence length", annotation.get("sequence_length")),
        ("Domains", annotation.get("domain_count")),
        ("Safety liabilities", annotation.get("safety_liability_count")),
        ("Clinical candidates (Open Targets count)", annotation.get("drug_candidate_count")),
        ("Approved small-molecule drug", annotation.get("approved_drug_sm")),
        ("High-quality pocket", annotation.get("high_quality_pocket_sm")),
        ("Crystal structures", annotation.get("crystal_structure_count")),
        ("Best crystal resolution (Å)", annotation.get("best_crystal_resolution")),
        ("Best crystal PDB id", annotation.get("best_crystal_pdb_id")),
        ("Ligand-bound crystals inspected", annotation.get("ligand_bound_crystal_count")),
    ]
    fact_rows = "".join(
        f"<tr><th>{html.escape(label)}</th><td>{cell(value)}</td></tr>" for label, value in fields
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{html.escape(gene)} target report</title>
  <style>
    body {{ font-family: Georgia, serif; margin: 2rem auto; max-width: 980px; color: #1f2933; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 0.92rem; }}
    th, td {{ border-bottom: 1px solid #d9e2e2; text-align: left; padding: 0.35rem 0.45rem; vertical-align: top; }}
    img {{ max-width: 100%; height: auto; }}
    code {{ font-size: 0.78rem; }}
    .tag {{ color: #0f6e6e; letter-spacing: 0.04em; text-transform: uppercase; font-size: 0.78rem; }}
  </style>
</head>
<body>
  <p class="tag">{html.escape(area)}</p>
  <h1>{html.escape(gene)}</h1>
  <p>Association evidence is the Open Targets overall score and datatype scores for this gene.
  Potency rows are deduplicated ChEMBL binding IC50 measurements converted to pIC50.
  This is a research summary, not a treatment recommendation.</p>
  <h2>Target card</h2>
  <table>{fact_rows}</table>
  <h2>Disease panel</h2>
  <table>
    <tr><th>Disease</th><th>Area</th><th>Association</th><th>Genetic</th><th>Somatic</th><th>Known drug / clinical</th><th>Source id</th></tr>
    {"".join(score_rows)}
  </table>
  <h2>Highest measured pIC50 values</h2>
  <table>
    <tr><th>ChEMBL id</th><th>pIC50</th><th>IC50 (nM)</th><th>MW</th><th>LogP</th><th>Ro5 violations</th><th>SMILES</th></tr>
    {"".join(compound_rows)}
  </table>
  <h2>Figures</h2>
  {figures}
</body>
</html>
"""
