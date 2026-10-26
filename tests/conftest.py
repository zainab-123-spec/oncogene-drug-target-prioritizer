"""Shared builders for API-free tests."""

from __future__ import annotations

import json
from typing import Any

import requests

from odtp.config import Disease, Target


def json_response(
    payload: dict[str, Any],
    status: int = 200,
    url: str = "https://example.test",
) -> requests.Response:
    response = requests.Response()
    response.status_code = status
    response._content = json.dumps(payload).encode()
    response.url = url
    response.encoding = "utf-8"
    response.headers["Content-Type"] = "application/json"
    return response


def measurement(**overrides: Any) -> dict[str, Any]:
    row = {
        "gene": "EGFR",
        "molecule_chembl_id": "CHEMBL1",
        "parent_molecule_chembl_id": "CHEMBL1",
        "target_chembl_id": "CHEMBL203",
        "smiles": "CCO",
        "standard_relation": "=",
        "standard_value": 10.0,
        "standard_units": "nM",
        "standard_type": "IC50",
        "pchembl_value": 8.0,
        "data_validity_comment": "",
        "potential_duplicate": 0,
        "activity_id": 1,
        "target_organism": "Homo sapiens",
    }
    row.update(overrides)
    return row


def disease() -> Disease:
    return Disease("MONDO_0005233", "non-small cell lung carcinoma", "Oncology")


def target() -> Target:
    return Target("EGFR", "P00533", "ENSG00000146648", "Oncology")
