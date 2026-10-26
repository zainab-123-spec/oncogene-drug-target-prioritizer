"""Dataset assembly with fake API clients, plus a target report."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from odtp.config import Disease, Target
from odtp.pipeline import analyze_target, build_dataset

GEFITINIB = "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1"
IBUPROFEN = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"
CAFFEINE = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"


def _measurement(**overrides: object) -> dict:
    row = {
        "gene": "EGFR",
        "molecule_chembl_id": "CHEMBL1",
        "parent_molecule_chembl_id": "CHEMBL1",
        "target_chembl_id": "CHEMBL203",
        "smiles": GEFITINIB,
        "standard_relation": "=",
        "standard_value": 10.0,
        "standard_units": "nM",
        "standard_type": "IC50",
        "pchembl_value": None,
        "data_validity_comment": "",
        "potential_duplicate": 0,
        "activity_id": 1,
        "target_organism": "Homo sapiens",
    }
    row.update(overrides)
    return row


class FakeOpenTargets:
    def disease_rankings(self, diseases, top_n):
        disease = diseases[0]
        return [
            {
                "disease_id": disease.id,
                "disease_name": disease.name,
                "category": disease.category,
                "rank": 1,
                "gene": "EGFR",
                "ensembl_id": "ENSG00000146648",
                "target_name": "EGFR",
                "association_score": 0.88,
                "genetic_association": 0.5,
                "somatic_mutation": 0.7,
                "literature": 0.9,
                "affected_pathway": 0.2,
                "animal_model": None,
                "rna_expression": None,
                "genetic_literature": 0.4,
                "clinical": 0.95,
                "known_drug": None,
                "known_drug_score": 0.95,
                "known_drug_source": "clinical",
            }
        ]

    def curated_associations(self, diseases, targets):
        rows = []
        for disease in diseases:
            for target in targets:
                rows.append(
                    {
                        "disease_id": disease.id,
                        "disease_name": disease.name,
                        "category": disease.category,
                        "gene": target.symbol,
                        "ensembl_id": target.ensembl,
                        "association_score": 0.88,
                        "genetic_association": 0.4,
                        "somatic_mutation": 0.5,
                        "literature": 0.7,
                        "affected_pathway": 0.1,
                        "animal_model": None,
                        "rna_expression": None,
                        "genetic_literature": None,
                        "clinical": 0.9,
                        "known_drug": None,
                        "known_drug_score": 0.9,
                        "known_drug_source": "clinical",
                    }
                )
        return rows

    def target_details(self, targets):
        rows = []
        for target in targets:
            rows.append(
                {
                    "gene": target.symbol,
                    "ensembl_id": target.ensembl,
                    "approved_symbol": target.symbol,
                    "approved_name": target.symbol,
                    "therapeutic_area": target.category,
                    "uniprot": target.uniprot,
                    "symbol_matches": True,
                    "approved_drug_sm": 1,
                    "advanced_clinical_sm": 0,
                    "phase1_clinical_sm": 0,
                    "structure_with_ligand_sm": 1,
                    "high_quality_ligand_sm": 1,
                    "high_quality_pocket_sm": 1,
                    "med_quality_pocket_sm": 0,
                    "druggable_family_sm": 1,
                    "safety_liability_count": 2,
                    "safety_events": "rash",
                    "drug_candidate_count": 4,
                }
            )
        return rows


class FakeChembl:
    def fetch_ic50(self, target, page_size, windows):
        rows = [
            _measurement(activity_id=1, standard_value=10, molecule_chembl_id="CHEMBL10"),
            _measurement(activity_id=2, standard_value=100, molecule_chembl_id="CHEMBL11"),
            _measurement(
                activity_id=3,
                smiles=IBUPROFEN,
                parent_molecule_chembl_id="CHEMBL20",
                molecule_chembl_id="CHEMBL20",
                standard_value=50,
            ),
            _measurement(
                activity_id=4,
                smiles=CAFFEINE,
                parent_molecule_chembl_id="CHEMBL30",
                molecule_chembl_id="CHEMBL30",
                standard_value=1,
                standard_units="uM",
            ),
            _measurement(
                activity_id=5,
                smiles="c1ccccc1",
                parent_molecule_chembl_id="CHEMBL40",
                molecule_chembl_id="CHEMBL40",
                standard_value=20,
            ),
            _measurement(activity_id=6, standard_relation=">", smiles=GEFITINIB),
            _measurement(
                activity_id=7,
                smiles="not-a-smiles",
                parent_molecule_chembl_id="CHEMBL50",
                molecule_chembl_id="CHEMBL50",
            ),
        ]
        stats = {
            "gene": target.symbol,
            "chembl_id": "CHEMBL203",
            "total_reported": 1200,
            "n_fetched": len(rows),
            "offsets": [0, 400, 800],
        }
        return rows, stats


class FakeStructures:
    def annotate(self, targets):
        return {
            target.symbol: {
                "protein_name": "Epidermal growth factor receptor",
                "sequence_length": 1210,
                "domain_count": 2,
                "domain_names": "Protein kinase",
                "uniprot_pdb_xref_count": 3,
                "best_crystal_resolution": 1.8,
                "best_crystal_pdb_id": "1M17",
                "ligand_bound_crystal_count": 2,
                "ligand_instance_count": 3,
                "crystals_inspected": 2,
                "crystal_structure_count": 9,
            }
            for target in targets
        }


def _build(tmp_path: Path) -> Path:
    processed = tmp_path / "processed"
    build_dataset(
        processed,
        opentargets=FakeOpenTargets(),
        chembl=FakeChembl(),
        structures=FakeStructures(),
        diseases=[Disease("MONDO_0005233", "non-small cell lung carcinoma", "Oncology")],
        targets=[Target("EGFR", "P00533", "ENSG00000146648", "Oncology")],
        per_target_cap=10,
    )
    return processed


def test_build_dataset_applies_qc_and_writes_tables(tmp_path: Path) -> None:
    processed = _build(tmp_path)
    model = pd.read_csv(processed / "activities_model.csv")
    dedup = pd.read_csv(processed / "activities_dedup.csv")
    funnel = pd.read_csv(processed / "qc_funnel.csv")
    scores = pd.read_csv(processed / "gene_disease_scores.csv")
    assert set(model["gene"]) == {"EGFR"}
    assert len(model) == 3
    gefitinib = model.loc[model["parent_molecule_chembl_id"] == "CHEMBL1"]
    assert len(gefitinib) == 1
    assert gefitinib["pIC50"].iloc[0] == 7.5
    assert "not-a-smiles" not in set(dedup["smiles"])
    assert funnel.loc[0, "n_model"] == 3
    assert funnel.loc[0, "n_small_molecule"] == 3
    assert funnel.loc[0, "n_valid_smiles"] == 4
    assert funnel.loc[0, "total_reported"] == 1200
    assert scores.loc[0, "known_drug_source"] == "clinical"
    assert "max_known_drug_score" in model.columns
    assert (processed / "summary.json").exists()


def test_analyze_writes_html_report(tmp_path: Path) -> None:
    processed = _build(tmp_path)
    figures = tmp_path / "figures"
    reports = tmp_path / "reports"
    path = analyze_target("EGFR", processed, figures, reports)
    text = path.read_text()
    assert "EGFR" in text
    assert "pIC50" in text
    assert "1M17" in text
    assert (figures / "analyze_EGFR_potency.png").exists()
    assert (figures / "analyze_EGFR_associations.png").exists()
