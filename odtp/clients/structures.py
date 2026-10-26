"""UniProt sequence/domain metadata and RCSB crystal-structure summaries.

Ligand counts are taken from PDB non-polymer entities in the highest-resolution
crystal structures returned by the search (up to a small sample). This is a
count of ligand-bound crystal structures, not a geometric pocket prediction.
Open Targets supplies the separate high-quality pocket tractability flag.
"""

from __future__ import annotations

import logging
from typing import Any

from odtp.config import RCSB_CRYSTAL_ROWS, RCSB_GRAPHQL_URL, RCSB_SEARCH_URL, UNIPROT_BASE, Target
from odtp.http import HttpClient

logger = logging.getLogger(__name__)

_ENTRY_QUERY = """
query($ids: [String!]!) {
  entries(entry_ids: $ids) {
    rcsb_id
    exptl { method }
    rcsb_entry_info { resolution_combined nonpolymer_entity_count }
  }
}
"""


def parse_uniprot(payload: dict[str, Any]) -> dict[str, Any]:
    """Sequence length, recommended name, and annotated domain count."""

    description = payload.get("proteinDescription") or {}
    recommended = description.get("recommendedName") or {}
    full_name = ((recommended.get("fullName") or {}).get("value")) or ""
    if not full_name:
        submissions = description.get("submissionNames") or []
        if submissions:
            full_name = ((submissions[0].get("fullName") or {}).get("value")) or ""
    features = payload.get("features") or []
    domains = [
        feature.get("description") or "" for feature in features if feature.get("type") == "Domain"
    ]
    pdb_ids = [
        xref.get("id")
        for xref in payload.get("uniProtKBCrossReferences") or []
        if xref.get("database") == "PDB" and xref.get("id")
    ]
    sequence = payload.get("sequence") or {}
    return {
        "uniprot": payload.get("primaryAccession"),
        "protein_name": full_name,
        "sequence_length": sequence.get("length"),
        "domain_count": len(domains),
        "domain_names": "; ".join(name for name in domains if name)[:500],
        "uniprot_pdb_xref_count": len(pdb_ids),
    }


def crystal_search_body(accession: str, rows: int) -> dict[str, Any]:
    """RCSB search for X-ray entries of a UniProt accession, best resolution first."""

    return {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": (
                            "rcsb_polymer_entity_container_identifiers."
                            "reference_sequence_identifiers.database_accession"
                        ),
                        "operator": "exact_match",
                        "value": accession,
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "exptl.method",
                        "operator": "exact_match",
                        "value": "X-RAY DIFFRACTION",
                    },
                },
            ],
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": 0, "rows": rows},
            "sort": [{"sort_by": "rcsb_entry_info.resolution_combined", "direction": "asc"}],
        },
    }


def parse_search(payload: dict[str, Any]) -> tuple[int, list[str]]:
    total = int(payload.get("total_count") or 0)
    identifiers = [row.get("identifier") for row in payload.get("result_set") or []]
    return total, [str(item) for item in identifiers if item]


def parse_crystal_entries(payload: dict[str, Any]) -> dict[str, Any]:
    """Best X-ray resolution and how many inspected entries contain a ligand."""

    entries = ((payload.get("data") or {}).get("entries")) or []
    best_resolution: float | None = None
    best_id: str | None = None
    ligand_bound = 0
    ligand_instances = 0
    inspected = 0
    for entry in entries:
        if not entry:
            continue
        methods = [item.get("method") or "" for item in (entry.get("exptl") or [])]
        if methods and not any("X-RAY" in method for method in methods):
            continue
        inspected += 1
        info = entry.get("rcsb_entry_info") or {}
        nonpolymer = int(info.get("nonpolymer_entity_count") or 0)
        ligand_instances += nonpolymer
        if nonpolymer > 0:
            ligand_bound += 1
        resolutions = [
            float(value) for value in (info.get("resolution_combined") or []) if value is not None
        ]
        if not resolutions:
            continue
        resolution = min(resolutions)
        if best_resolution is None or resolution < best_resolution:
            best_resolution = resolution
            best_id = entry.get("rcsb_id")
    return {
        "best_crystal_resolution": best_resolution,
        "best_crystal_pdb_id": best_id,
        "ligand_bound_crystal_count": ligand_bound,
        "ligand_instance_count": ligand_instances,
        "crystals_inspected": inspected,
    }


class StructureClient:
    """Attach UniProt and PDB metadata to each configured target."""

    def __init__(self, http: HttpClient, crystal_rows: int = RCSB_CRYSTAL_ROWS) -> None:
        self.http = http
        self.crystal_rows = crystal_rows

    def annotate(self, targets: list[Target]) -> dict[str, dict[str, Any]]:
        annotated: dict[str, dict[str, Any]] = {}
        for target in targets:
            logger.info("Structures %s (%s)", target.symbol, target.uniprot)
            uniprot = parse_uniprot(self.http.get_json(f"{UNIPROT_BASE}/{target.uniprot}.json"))
            search = self.http.post_json(
                RCSB_SEARCH_URL,
                crystal_search_body(target.uniprot, self.crystal_rows),
            )
            crystal_count, pdb_ids = parse_search(search)
            if pdb_ids:
                details = self.http.post_json(
                    RCSB_GRAPHQL_URL,
                    {"query": _ENTRY_QUERY, "variables": {"ids": pdb_ids}},
                )
                crystals = parse_crystal_entries(details)
            else:
                crystals = parse_crystal_entries({"data": {"entries": []}})
            crystals["crystal_structure_count"] = crystal_count
            annotated[target.symbol] = {**uniprot, **crystals}
        return annotated
