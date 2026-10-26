"""Open Targets Platform GraphQL client.

The live Platform reports clinical / known-drug evidence under the datatype id
``clinical``. Older releases used ``known_drug``. This client keeps both: if
``known_drug`` is present it wins, otherwise ``clinical`` is stored as
``known_drug_score`` and the source column says which id was used.
"""

from __future__ import annotations

import logging
from typing import Any

from odtp.config import (
    DATATYPE_FIELDS,
    OPENTARGETS_URL,
    Disease,
    Target,
)
from odtp.http import HttpClient, RequestError

logger = logging.getLogger(__name__)

_RANKING_QUERY = """
query($efo: String!, $size: Int!) {
  disease(efoId: $efo) {
    id
    name
    associatedTargets(page: {index: 0, size: $size}) {
      count
      rows {
        score
        target { id approvedSymbol approvedName }
        datatypeScores { id score }
      }
    }
  }
}
"""

_TARGET_QUERY = """
query($id: String!) {
  target(ensemblId: $id) {
    id
    approvedSymbol
    approvedName
    tractability { label modality value }
    safetyLiabilities { event }
    drugAndClinicalCandidates { count }
  }
}
"""

_SEARCH_QUERY = """
query($q: String!) {
  search(queryString: $q, entityNames: ["target"], page: {index: 0, size: 8}) {
    hits { id name }
  }
}
"""


def datatype_map(items: list[dict[str, Any]] | None) -> dict[str, float]:
    """Map Open Targets datatype scores to floats."""

    scores: dict[str, float] = {}
    for item in items or []:
        identifier = item.get("id")
        if not identifier or item.get("score") is None:
            continue
        scores[str(identifier)] = float(item["score"])
    return scores


def known_drug_from_datatypes(scores: dict[str, float]) -> tuple[float | None, str | None]:
    """Prefer the legacy ``known_drug`` id, else the current ``clinical`` id."""

    if "known_drug" in scores:
        return scores["known_drug"], "known_drug"
    if "clinical" in scores:
        return scores["clinical"], "clinical"
    return None, None


def empty_score_row(disease: Disease, target: Target, ensembl_id: str) -> dict[str, Any]:
    """A gene-disease row with missing evidence, so heatmaps stay rectangular."""

    row: dict[str, Any] = {
        "disease_id": disease.id,
        "disease_name": disease.name,
        "category": disease.category,
        "gene": target.symbol,
        "ensembl_id": ensembl_id,
        "association_score": None,
        "known_drug_score": None,
        "known_drug_source": None,
    }
    for field in DATATYPE_FIELDS:
        row[field] = None
    return row


def score_fields(scores: dict[str, float]) -> dict[str, Any]:
    known, source = known_drug_from_datatypes(scores)
    fields: dict[str, Any] = {
        "known_drug_score": known,
        "known_drug_source": source,
    }
    for field in DATATYPE_FIELDS:
        fields[field] = scores.get(field)
    return fields


def select_exact_symbol(rows: list[dict[str, Any]], symbol: str) -> dict[str, Any] | None:
    """Pick the association row whose approved symbol matches exactly.

    ``BFilter`` is a search filter. Short symbols such as MET or RET can
    otherwise return a different gene in the first position.
    """

    for row in rows:
        target = row.get("target") or {}
        if target.get("approvedSymbol") == symbol:
            return row
    return None


def build_curated_query(pairs: list[tuple[Disease, Target]]) -> str:
    """Alias-batched gene-disease association query."""

    blocks: list[str] = []
    for index, (disease, target) in enumerate(pairs):
        blocks.append(
            f"""
            a{index}: disease(efoId: "{disease.id}") {{
              id
              name
              associatedTargets(BFilter: "{target.symbol}", page: {{index: 0, size: 5}}) {{
                count
                rows {{
                  score
                  target {{ id approvedSymbol }}
                  datatypeScores {{ id score }}
                }}
              }}
            }}
            """
        )
    return "query {\n" + "\n".join(blocks) + "\n}"


def parse_ranking(payload: dict[str, Any], disease: Disease) -> list[dict[str, Any]]:
    """Flatten a disease ranking response."""

    block = (payload.get("data") or {}).get("disease") or {}
    associations = (block.get("associatedTargets") or {}).get("rows") or []
    rows: list[dict[str, Any]] = []
    for rank, item in enumerate(associations, start=1):
        target = item.get("target") or {}
        scores = datatype_map(item.get("datatypeScores"))
        row = {
            "disease_id": disease.id,
            "disease_name": block.get("name") or disease.name,
            "category": disease.category,
            "rank": rank,
            "gene": target.get("approvedSymbol"),
            "ensembl_id": target.get("id"),
            "target_name": target.get("approvedName"),
            "association_score": item.get("score"),
        }
        row.update(score_fields(scores))
        rows.append(row)
    return rows


def parse_curated_block(
    block: dict[str, Any] | None,
    disease: Disease,
    target: Target,
) -> dict[str, Any]:
    """One curated gene-disease score row."""

    if not block:
        return empty_score_row(disease, target, target.ensembl)
    associations = (block.get("associatedTargets") or {}).get("rows") or []
    chosen = select_exact_symbol(associations, target.symbol)
    if chosen is None:
        return empty_score_row(disease, target, target.ensembl)
    scores = datatype_map(chosen.get("datatypeScores"))
    hit = chosen.get("target") or {}
    row = empty_score_row(disease, target, hit.get("id") or target.ensembl)
    row["association_score"] = chosen.get("score")
    row.update(score_fields(scores))
    return row


def parse_target_details(payload: dict[str, Any], target: Target) -> dict[str, Any]:
    """Tractability flags, safety-event count, and clinical candidate count."""

    block = (payload.get("data") or {}).get("target") or {}
    symbol = block.get("approvedSymbol") or ""
    tractability = block.get("tractability") or []
    flags = {
        item.get("label"): bool(item.get("value"))
        for item in tractability
        if item.get("modality") == "SM"
    }
    events = [
        item.get("event") for item in (block.get("safetyLiabilities") or []) if item.get("event")
    ]
    drugs = block.get("drugAndClinicalCandidates") or {}
    return {
        "gene": target.symbol,
        "ensembl_id": block.get("id") or target.ensembl,
        "approved_symbol": symbol,
        "approved_name": block.get("approvedName") or "",
        "therapeutic_area": target.category,
        "uniprot": target.uniprot,
        "symbol_matches": symbol == target.symbol,
        "approved_drug_sm": int(flags.get("Approved Drug", False)),
        "advanced_clinical_sm": int(flags.get("Advanced Clinical", False)),
        "phase1_clinical_sm": int(flags.get("Phase 1 Clinical", False)),
        "structure_with_ligand_sm": int(flags.get("Structure with Ligand", False)),
        "high_quality_ligand_sm": int(flags.get("High-Quality Ligand", False)),
        "high_quality_pocket_sm": int(flags.get("High-Quality Pocket", False)),
        "med_quality_pocket_sm": int(flags.get("Med-Quality Pocket", False)),
        "druggable_family_sm": int(flags.get("Druggable Family", False)),
        "safety_liability_count": len(events),
        "safety_events": "; ".join(events[:12]),
        "drug_candidate_count": int(drugs.get("count") or 0),
    }


class OpenTargetsClient:
    """Fetch disease rankings, curated scores, and target tractability."""

    def __init__(self, http: HttpClient, batch_size: int = 6) -> None:
        self.http = http
        self.batch_size = batch_size

    def _post(self, query: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"query": query}
        if variables:
            payload["variables"] = variables
        result = self.http.post_json(OPENTARGETS_URL, payload)
        if not isinstance(result, dict):
            raise RuntimeError("Open Targets returned a non-object response")
        return result

    def resolve_ensembl(self, target: Target) -> str:
        """Confirm the configured Ensembl id, or search by exact symbol."""

        payload = self._post(_TARGET_QUERY, {"id": target.ensembl})
        block = (payload.get("data") or {}).get("target") or {}
        if block.get("approvedSymbol") == target.symbol:
            return str(block.get("id") or target.ensembl)
        logger.warning(
            "Ensembl %s did not resolve to %s (got %s); searching",
            target.ensembl,
            target.symbol,
            block.get("approvedSymbol"),
        )
        found = self._post(_SEARCH_QUERY, {"q": target.symbol})
        hits = ((found.get("data") or {}).get("search") or {}).get("hits") or []
        for hit in hits:
            if hit.get("name") == target.symbol and str(hit.get("id", "")).startswith("ENSG"):
                logger.info("Resolved %s to %s", target.symbol, hit["id"])
                return str(hit["id"])
        raise RuntimeError(f"Could not resolve an Ensembl id for {target.symbol}")

    def disease_rankings(self, diseases: list[Disease], top_n: int) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for disease in diseases:
            logger.info("Open Targets ranking: %s", disease.name)
            payload = self._post(_RANKING_QUERY, {"efo": disease.id, "size": top_n})
            rows.extend(parse_ranking(payload, disease))
        return rows

    def curated_associations(
        self,
        diseases: list[Disease],
        targets: list[Target],
    ) -> list[dict[str, Any]]:
        pairs = [(disease, target) for disease in diseases for target in targets]
        rows: list[dict[str, Any]] = []
        for start in range(0, len(pairs), self.batch_size):
            chunk = pairs[start : start + self.batch_size]
            logger.info(
                "Open Targets curated scores %s-%s of %s",
                start + 1,
                start + len(chunk),
                len(pairs),
            )
            rows.extend(self._curated_chunk(chunk))
        return rows

    def _curated_chunk(self, chunk: list[tuple[Disease, Target]]) -> list[dict[str, Any]]:
        try:
            payload = self._post(build_curated_query(chunk))
        except RequestError:
            if len(chunk) == 1:
                raise
            logger.warning("Open Targets batch failed; requesting one gene at a time")
            rows: list[dict[str, Any]] = []
            for pair in chunk:
                rows.extend(self._curated_chunk([pair]))
            return rows
        data = payload.get("data") or {}
        return [
            parse_curated_block(data.get(f"a{index}"), disease, target)
            for index, (disease, target) in enumerate(chunk)
        ]

    def target_details(self, targets: list[Target]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for target in targets:
            logger.info("Open Targets target card: %s", target.symbol)
            ensembl = target.ensembl
            payload = self._post(_TARGET_QUERY, {"id": ensembl})
            block = (payload.get("data") or {}).get("target") or {}
            if block.get("approvedSymbol") != target.symbol:
                ensembl = self.resolve_ensembl(target)
                payload = self._post(_TARGET_QUERY, {"id": ensembl})
            resolved = Target(target.symbol, target.uniprot, ensembl, target.category)
            details = parse_target_details(payload, resolved)
            if not details["symbol_matches"]:
                raise RuntimeError(
                    f"Open Targets symbol for {ensembl} was {details['approved_symbol']}, "
                    f"expected {target.symbol}"
                )
            rows.append(details)
        return rows
