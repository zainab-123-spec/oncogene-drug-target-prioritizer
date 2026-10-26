"""Report text is rendered from supplied numbers, not hardcoded benchmarks."""

from odtp.model import aggregate_importances
from odtp.reports import render_benchmark_markdown, render_target_report


def test_benchmark_markdown_prints_supplied_metrics() -> None:
    metrics = {
        "models": [
            {
                "name": "Ridge",
                "task": "regression",
                "split": "scaffold",
                "rmse": 1.25,
                "pearson_r": 0.5,
                "spearman_rho": 0.45,
                "roc_auc": 0.72,
                "pr_auc": 0.81,
            },
            {
                "name": "Ridge",
                "task": "regression",
                "split": "random",
                "rmse": 1.0,
                "pearson_r": 0.6,
                "spearman_rho": 0.55,
                "roc_auc": 0.8,
                "pr_auc": 0.84,
            },
            {
                "name": "RandomForest",
                "task": "classification",
                "split": "scaffold",
                "rmse": None,
                "pearson_r": None,
                "spearman_rho": None,
                "roc_auc": 0.66,
                "pr_auc": 0.7,
            },
            {
                "name": "RandomForest",
                "task": "classification",
                "split": "random",
                "rmse": None,
                "pearson_r": None,
                "spearman_rho": None,
                "roc_auc": 0.7,
                "pr_auc": 0.74,
            },
        ],
        "leakage": {
            "Ridge": {"regression": {"rmse_gap": 0.25, "roc_auc_gap": 0.08, "pr_auc_gap": 0.03}}
        },
        "splits": {
            "scaffold": {
                "n_train": 80,
                "n_test": 20,
                "shared_scaffolds": 0,
                "test_positive_rate": 0.4,
            }
        },
    }
    text = render_benchmark_markdown(metrics)
    assert "1.250" in text
    assert "0.720" in text
    assert "shared scaffolds=0" in text
    assert "RMSE gap" in text


def test_target_report_contains_gene_and_pic50() -> None:
    html = render_target_report(
        gene="EGFR",
        area="Oncology",
        annotation={
            "uniprot": "P00533",
            "protein_name": "Epidermal growth factor receptor",
            "sequence_length": 1210,
            "domain_count": 2,
            "safety_liability_count": 3,
            "drug_candidate_count": 10,
            "approved_drug_sm": 1,
            "high_quality_pocket_sm": 1,
            "crystal_structure_count": 5,
            "best_crystal_resolution": 1.8,
            "best_crystal_pdb_id": "1M17",
            "ligand_bound_crystal_count": 2,
        },
        scores=[
            {
                "disease_name": "non-small cell lung carcinoma",
                "category": "Oncology",
                "association_score": 0.88,
                "genetic_association": 0.5,
                "somatic_mutation": 0.7,
                "known_drug_score": 0.9,
                "known_drug_source": "clinical",
            }
        ],
        compounds=[
            {
                "molecule_chembl_id": "CHEMBL68920",
                "pIC50": 8.2,
                "ic50_nM": 6.3,
                "mw": 430.0,
                "logp": 3.2,
                "lipinski_violations": 0,
                "smiles": "c1ccccc1",
            }
        ],
        figure_names=["analyze_EGFR_potency.png"],
    )
    assert "EGFR" in html
    assert "pIC50" in html
    assert "CHEMBL68920" in html
    assert "clinical" in html
    assert "1M17" in html


def test_importance_aggregation_sums_fingerprint_bits() -> None:
    rows = aggregate_importances(["mw", "fp_000", "fp_001"], [0.2, 0.3, 0.5])
    morgan = next(row for row in rows if str(row["feature"]).startswith("Morgan"))
    assert morgan["importance"] == 0.8
    assert rows[0]["feature"] == morgan["feature"]
