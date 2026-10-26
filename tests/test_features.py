"""Descriptors, scaffolds, and target-feature aggregation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from odtp.features import (
    add_molecular_features,
    build_target_feature_table,
    compute_molecule_features,
    filter_small_molecules,
    lipinski_violations,
    morgan_bits,
)


def test_lipinski_clean_molecule_has_zero_violations() -> None:
    assert lipinski_violations(400, 2, 1, 4) == 0


def test_lipinski_counts_each_violation() -> None:
    assert lipinski_violations(600, 6, 6, 11) == 4


def test_morgan_fingerprint_is_deterministic() -> None:
    first = morgan_bits("c1ccccc1O")
    second = morgan_bits("c1ccccc1O")
    assert first is not None and second is not None
    assert first.shape == (512,)
    assert np.array_equal(first, second)
    assert set(np.unique(first)).issubset({0.0, 1.0})


def test_analogs_share_murcko_scaffold() -> None:
    benzene = compute_molecule_features("c1ccccc1")
    toluene = compute_molecule_features("Cc1ccccc1")
    assert benzene is not None and toluene is not None
    assert benzene["scaffold"] == toluene["scaffold"]
    assert benzene["aromatic_rings"] == 1


def test_acyclic_molecules_do_not_share_one_empty_scaffold() -> None:
    ethanol = compute_molecule_features("CCO")
    ethylamine = compute_molecule_features("CCN")
    assert ethanol is not None and ethylamine is not None
    assert ethanol["scaffold"].startswith("ACYCLIC::")
    assert ethanol["scaffold"] != ethylamine["scaffold"]


def test_invalid_smiles_dropped() -> None:
    frame = pd.DataFrame(
        [
            {
                "gene": "EGFR",
                "smiles": "not-a-molecule",
                "pIC50": 6.0,
                "molecule_chembl_id": "CHEMBL1",
                "parent_molecule_chembl_id": "CHEMBL1",
                "target_chembl_id": "CHEMBL203",
            }
        ]
    )
    assert add_molecular_features(frame).empty


def test_small_molecule_filter_drops_low_mass_fragments() -> None:
    gefitinib = "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1"
    frame = add_molecular_features(
        pd.DataFrame(
            [
                {"gene": "EGFR", "smiles": "c1ccccc1", "pIC50": 5},
                {"gene": "EGFR", "smiles": gefitinib, "pIC50": 7},
            ]
        )
    )
    kept = filter_small_molecules(frame)
    assert len(frame) == 2
    assert kept["smiles"].str.contains("ncnc").any()
    assert not kept["smiles"].eq("c1ccccc1").any()
    assert kept["mw"].min() >= 150


def test_target_feature_uses_max_scores_and_area_flags() -> None:
    associations = pd.DataFrame(
        [
            {
                "gene": "EGFR",
                "association_score": 0.2,
                "genetic_association": 0.1,
                "somatic_mutation": 0.4,
                "known_drug_score": 0.3,
                "literature": 0.2,
            },
            {
                "gene": "EGFR",
                "association_score": 0.9,
                "genetic_association": 0.8,
                "somatic_mutation": 0.1,
                "known_drug_score": 0.95,
                "literature": 0.7,
            },
        ]
    )
    annotations = pd.DataFrame(
        [
            {
                "gene": "EGFR",
                "therapeutic_area": "Oncology",
                "safety_liability_count": 4,
                "approved_drug_sm": 1,
                "high_quality_pocket_sm": 1,
                "structure_with_ligand_sm": 1,
                "drug_candidate_count": 10,
                "sequence_length": 1210,
                "domain_count": 2,
                "best_crystal_resolution": 1.5,
                "crystal_structure_count": 20,
                "ligand_bound_crystal_count": 3,
            }
        ]
    )
    table = build_target_feature_table(associations, annotations)
    assert table.loc[0, "max_association_score"] == pytest.approx(0.9)
    assert table.loc[0, "max_known_drug_score"] == pytest.approx(0.95)
    assert table.loc[0, "n_panel_diseases"] == 2
    assert table.loc[0, "area_oncology"] == 1
    assert table.loc[0, "area_metabolic"] == 0
