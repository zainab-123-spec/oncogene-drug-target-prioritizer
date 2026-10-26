"""Parsers for Open Targets, ChEMBL, UniProt, and the PDB."""

from __future__ import annotations

from odtp.clients.chembl import ChemblClient, parse_activity_page, parse_target_lookup
from odtp.clients.opentargets import (
    build_curated_query,
    known_drug_from_datatypes,
    parse_curated_block,
    parse_ranking,
    parse_target_details,
    select_exact_symbol,
)
from odtp.clients.structures import parse_crystal_entries, parse_search, parse_uniprot
from odtp.config import Disease, Target
from odtp.http import HttpClient
from tests.conftest import json_response


def test_parse_disease_ranking() -> None:
    disease = Disease("MONDO_0004975", "Alzheimer disease", "Neurodegenerative")
    payload = {
        "data": {
            "disease": {
                "id": disease.id,
                "name": disease.name,
                "associatedTargets": {
                    "rows": [
                        {
                            "score": 0.81,
                            "target": {
                                "id": "ENSG00000186868",
                                "approvedSymbol": "MAPT",
                                "approvedName": "microtubule associated protein tau",
                            },
                            "datatypeScores": [
                                {"id": "genetic_association", "score": 0.7},
                                {"id": "clinical", "score": 0.4},
                            ],
                        }
                    ]
                },
            }
        }
    }
    rows = parse_ranking(payload, disease)
    assert rows[0]["gene"] == "MAPT"
    assert rows[0]["rank"] == 1
    assert rows[0]["known_drug_score"] == 0.4
    assert rows[0]["known_drug_source"] == "clinical"
    assert rows[0]["category"] == "Neurodegenerative"


def test_known_drug_legacy_id_wins() -> None:
    score, source = known_drug_from_datatypes({"known_drug": 0.7, "clinical": 0.1})
    assert score == 0.7
    assert source == "known_drug"


def test_known_drug_falls_back_to_clinical() -> None:
    score, source = known_drug_from_datatypes({"clinical": 0.55})
    assert score == 0.55
    assert source == "clinical"


def test_short_symbol_filter_keeps_exact_gene() -> None:
    disease = Disease("MONDO_0005233", "non-small cell lung carcinoma", "Oncology")
    target = Target("MET", "P08581", "ENSG00000105976", "Oncology")
    block = {
        "associatedTargets": {
            "rows": [
                {
                    "score": 0.99,
                    "target": {"approvedSymbol": "METRN", "id": "ENSG00000106809"},
                    "datatypeScores": [],
                },
                {
                    "score": 0.42,
                    "target": {"approvedSymbol": "MET", "id": "ENSG00000105976"},
                    "datatypeScores": [{"id": "somatic_mutation", "score": 0.8}],
                },
            ]
        }
    }
    row = parse_curated_block(block, disease, target)
    assert row["gene"] == "MET"
    assert row["association_score"] == 0.42
    assert row["somatic_mutation"] == 0.8
    assert select_exact_symbol(block["associatedTargets"]["rows"], "MET")["score"] == 0.42


def test_curated_query_contains_gene_and_disease() -> None:
    disease = Disease("MONDO_0005439", "familial hypercholesterolemia", "Metabolic")
    target = Target("PCSK9", "Q8NBP7", "ENSG00000169174", "Metabolic")
    query = build_curated_query([(disease, target)])
    assert "MONDO_0005439" in query
    assert "PCSK9" in query
    assert "BFilter" in query
    assert "a0:" in query


def test_tractability_and_safety_counts() -> None:
    target = Target("EGFR", "P00533", "ENSG00000146648", "Oncology")
    payload = {
        "data": {
            "target": {
                "id": target.ensembl,
                "approvedSymbol": "EGFR",
                "approvedName": "epidermal growth factor receptor",
                "tractability": [
                    {"label": "Approved Drug", "modality": "SM", "value": True},
                    {"label": "High-Quality Pocket", "modality": "SM", "value": True},
                    {"label": "Approved Drug", "modality": "AB", "value": False},
                ],
                "safetyLiabilities": [{"event": "ALT increase"}, {"event": "rash"}],
                "drugAndClinicalCandidates": {"count": 82},
            }
        }
    }
    details = parse_target_details(payload, target)
    assert details["symbol_matches"] is True
    assert details["approved_drug_sm"] == 1
    assert details["high_quality_pocket_sm"] == 1
    assert details["safety_liability_count"] == 2
    assert details["drug_candidate_count"] == 82


def test_parse_chembl_target_and_activity_page() -> None:
    chembl_id = parse_target_lookup(
        {
            "targets": [
                {
                    "target_chembl_id": "CHEMBL3608",
                    "organism": "Mus musculus",
                    "target_type": "SINGLE PROTEIN",
                    "target_components": [{"accession": "P00533"}],
                },
                {
                    "target_chembl_id": "CHEMBL203",
                    "organism": "Homo sapiens",
                    "target_type": "SINGLE PROTEIN",
                    "target_components": [{"accession": "P00533"}],
                },
            ]
        },
        "P00533",
    )
    assert chembl_id == "CHEMBL203"
    rows, total = parse_activity_page(
        {
            "page_meta": {"total_count": 24925},
            "activities": [
                {
                    "activity_id": 5,
                    "molecule_chembl_id": "CHEMBL68920",
                    "parent_molecule_chembl_id": "CHEMBL68920",
                    "target_chembl_id": "CHEMBL203",
                    "canonical_smiles": "c1ccccc1",
                    "standard_relation": "=",
                    "standard_value": "41.0",
                    "standard_units": "nM",
                    "standard_type": "IC50",
                    "pchembl_value": "7.39",
                    "potential_duplicate": 0,
                    "target_organism": "Homo sapiens",
                }
            ],
        },
        "EGFR",
    )
    assert total == 24925
    assert rows[0]["gene"] == "EGFR"
    assert rows[0]["standard_value"] == "41.0"


def test_chembl_client_spaces_large_result(tmp_path) -> None:
    calls: list[int] = []

    class Session:
        def request(self, method, url, params=None, data=None, headers=None, timeout=None):
            offset = int(params["offset"])
            calls.append(offset)
            return json_response(
                {
                    "page_meta": {"total_count": 1000},
                    "activities": [
                        {
                            "activity_id": offset + 1,
                            "molecule_chembl_id": "CHEMBL1",
                            "parent_molecule_chembl_id": "CHEMBL1",
                            "canonical_smiles": "CCO",
                            "standard_relation": "=",
                            "standard_value": "10",
                            "standard_units": "nM",
                            "standard_type": "IC50",
                            "potential_duplicate": 0,
                            "target_organism": "Homo sapiens",
                            "target_chembl_id": "CHEMBL203",
                        }
                    ],
                }
            )

    client = ChemblClient(
        HttpClient(
            tmp_path,
            session=Session(),
            sleep=lambda _: None,
            time_fn=lambda: 0.0,
            min_interval=0,
        )
    )
    target = Target("EGFR", "P00533", "ENSG00000146648", "Oncology")
    rows, stats = client.fetch_ic50(target, page_size=200, windows=3, chembl_id="CHEMBL203")
    assert calls == [0, 400, 800]
    assert stats["total_reported"] == 1000
    assert len(rows) == 3


def test_parse_uniprot_domains_and_length() -> None:
    parsed = parse_uniprot(
        {
            "primaryAccession": "P00533",
            "proteinDescription": {
                "recommendedName": {"fullName": {"value": "Epidermal growth factor receptor"}}
            },
            "sequence": {"length": 1210},
            "features": [
                {"type": "Domain", "description": "Protein kinase"},
                {"type": "Domain", "description": "EGF-like"},
                {"type": "Chain", "description": "EGFR"},
            ],
            "uniProtKBCrossReferences": [
                {"database": "PDB", "id": "1M17"},
                {"database": "ChEMBL", "id": "CHEMBL203"},
            ],
        }
    )
    assert parsed["sequence_length"] == 1210
    assert parsed["domain_count"] == 2
    assert parsed["uniprot_pdb_xref_count"] == 1
    assert "Epidermal" in parsed["protein_name"]


def test_parse_rcsb_resolution_and_ligands() -> None:
    total, identifiers = parse_search(
        {"total_count": 12, "result_set": [{"identifier": "1M17"}, {"identifier": "5UG9"}]}
    )
    assert total == 12
    assert identifiers == ["1M17", "5UG9"]
    summary = parse_crystal_entries(
        {
            "data": {
                "entries": [
                    {
                        "rcsb_id": "1M17",
                        "exptl": [{"method": "X-RAY DIFFRACTION"}],
                        "rcsb_entry_info": {
                            "resolution_combined": [2.6],
                            "nonpolymer_entity_count": 1,
                        },
                    },
                    {
                        "rcsb_id": "5UG9",
                        "exptl": [{"method": "X-RAY DIFFRACTION"}],
                        "rcsb_entry_info": {
                            "resolution_combined": [1.33],
                            "nonpolymer_entity_count": 0,
                        },
                    },
                    {
                        "rcsb_id": "EM01",
                        "exptl": [{"method": "ELECTRON MICROSCOPY"}],
                        "rcsb_entry_info": {
                            "resolution_combined": [3.1],
                            "nonpolymer_entity_count": 2,
                        },
                    },
                ]
            }
        }
    )
    assert summary["best_crystal_resolution"] == 1.33
    assert summary["best_crystal_pdb_id"] == "5UG9"
    assert summary["ligand_bound_crystal_count"] == 1
    assert summary["crystals_inspected"] == 2
