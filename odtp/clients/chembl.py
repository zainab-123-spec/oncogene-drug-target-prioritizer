"""ChEMBL REST client for human IC50 bioactivities and target identifiers."""

from __future__ import annotations

import logging
from typing import Any

from odtp.config import CHEMBL_BASE, Target
from odtp.http import HttpClient
from odtp.qc import measurement_from_chembl, sample_offsets

logger = logging.getLogger(__name__)


def parse_target_lookup(payload: dict[str, Any], accession: str) -> str | None:
    """Choose the Homo sapiens SINGLE PROTEIN target for a UniProt accession."""

    matches: list[dict[str, Any]] = []
    for target in payload.get("targets") or []:
        if target.get("organism") != "Homo sapiens":
            continue
        if target.get("target_type") != "SINGLE PROTEIN":
            continue
        components = target.get("target_components") or []
        accessions = {component.get("accession") for component in components}
        if components and accession not in accessions:
            continue
        matches.append(target)
    if not matches:
        return None
    return str(matches[0]["target_chembl_id"])


def parse_activity_page(payload: dict[str, Any], gene: str) -> tuple[list[dict[str, Any]], int]:
    """Return measurement dicts and the reported total count."""

    activities = payload.get("activities") or []
    total = int((payload.get("page_meta") or {}).get("total_count") or len(activities))
    return [measurement_from_chembl(item, gene) for item in activities], total


class ChemblClient:
    """Resolve ChEMBL targets and download sampled IC50 pages."""

    def __init__(self, http: HttpClient) -> None:
        self.http = http

    def find_target_id(self, target: Target) -> str:
        payload = self.http.get_json(
            f"{CHEMBL_BASE}/target.json",
            params={"target_components__accession": target.uniprot, "limit": 20},
        )
        chembl_id = parse_target_lookup(payload, target.uniprot)
        if chembl_id is None:
            raise RuntimeError(f"No human single-protein ChEMBL target for {target.symbol}")
        logger.info("ChEMBL %s -> %s", target.symbol, chembl_id)
        return chembl_id

    def fetch_ic50(
        self,
        target: Target,
        *,
        page_size: int,
        windows: int,
        chembl_id: str | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Sample binding IC50 pages for one target.

        ``assay_type=B`` keeps biochemical binding assays. Functional cell
        assays (type F) are left out so the label is closer to a biochemical
        IC50. Ki values are not mixed in, because the training label is pIC50.
        """

        chembl_id = chembl_id or self.find_target_id(target)
        first = self.http.get_json(
            f"{CHEMBL_BASE}/activity.json",
            params={
                "target_chembl_id": chembl_id,
                "standard_type": "IC50",
                "assay_type": "B",
                "limit": page_size,
                "offset": 0,
            },
        )
        rows, total = parse_activity_page(first, target.symbol)
        offsets = sample_offsets(total, page_size, windows)
        for offset in offsets:
            if offset == 0:
                continue
            logger.info("ChEMBL %s IC50 offset %s / %s", target.symbol, offset, total)
            page = self.http.get_json(
                f"{CHEMBL_BASE}/activity.json",
                params={
                    "target_chembl_id": chembl_id,
                    "standard_type": "IC50",
                    "assay_type": "B",
                    "limit": page_size,
                    "offset": offset,
                },
            )
            extra, _ = parse_activity_page(page, target.symbol)
            rows.extend(extra)
        stats = {
            "gene": target.symbol,
            "chembl_id": chembl_id,
            "total_reported": total,
            "n_fetched": len(rows),
            "offsets": offsets,
        }
        logger.info(
            "ChEMBL %s fetched %s of %s reported binding IC50 rows",
            target.symbol,
            len(rows),
            total,
        )
        return rows, stats
