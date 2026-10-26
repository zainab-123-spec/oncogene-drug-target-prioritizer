"""The public disease and target panel."""

from odtp.config import AREA_ORDER, DISEASES, TARGETS


def test_disease_panel_covers_five_therapeutic_areas() -> None:
    assert {disease.category for disease in DISEASES} == set(AREA_ORDER)
    names = {disease.name for disease in DISEASES}
    assert "Alzheimer disease" in names
    assert "Parkinson disease" in names
    assert "type 2 diabetes mellitus" in names
    assert "familial hypercholesterolemia" in names
    assert "rheumatoid arthritis" in names
    assert "inflammatory bowel disease" in names
    assert "HIV infectious disease" in names
    assert "breast carcinoma" in names
    assert "non-small cell lung carcinoma" in names


def test_targets_include_requested_genes() -> None:
    symbols = {target.symbol for target in TARGETS}
    expected = {
        "LRRK2",
        "BACE1",
        "MAPT",
        "ACHE",
        "GLP1R",
        "DPP4",
        "PCSK9",
        "HMGCR",
        "JAK1",
        "JAK2",
        "TNF",
        "TYK2",
        "CCR5",
        "CXCR4",
        "DHODH",
        "EGFR",
        "BRAF",
        "ERBB2",
        "ALK",
        "KRAS",
        "CDK4",
        "PARP1",
        "ABL1",
    }
    assert expected <= symbols
    areas = {target.category for target in TARGETS}
    assert areas == set(AREA_ORDER)
