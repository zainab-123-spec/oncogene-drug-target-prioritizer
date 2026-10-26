"""Quality control for ChEMBL IC50 rows.

Censored relations (``>``, ``<``, ``>=``, ``<=``) are removed rather than
treated as exact numbers. Units are converted to nanomolar, then

    pIC50 = 9 - log10(IC50_nM)

which is the same quantity as ``-log10(IC50 in molar)``. Duplicate
parent-compound / target pairs collapse to the median pIC50.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from odtp.config import ACTIVE_PIC50

_UNIT_TO_NM = {
    "nM": 1.0,
    "nmol/L": 1.0,
    "nmol.L-1": 1.0,
    "uM": 1.0e3,
    "umol/L": 1.0e3,
    "umol.L-1": 1.0e3,
    "mM": 1.0e6,
    "mmol/L": 1.0e6,
    "mmol.L-1": 1.0e6,
    "M": 1.0e9,
    "mol/L": 1.0e9,
    "mol.L-1": 1.0e9,
}


def normalize_unit(unit: str | None) -> str:
    """Map a ChEMBL unit string onto a canonical key, or return an empty string."""

    if unit is None:
        return ""
    text = str(unit).strip().replace("μ", "u").replace("µ", "u").replace(" ", "")
    aliases = {
        "nM": "nM",
        "nm": "nM",
        "NM": "nM",
        "nmol/L": "nmol/L",
        "nmol.L-1": "nmol.L-1",
        "uM": "uM",
        "um": "uM",
        "UM": "uM",
        "umol/L": "umol/L",
        "umol.L-1": "umol.L-1",
        "mM": "mM",
        "mm": "mM",
        "MM": "mM",
        "mmol/L": "mmol/L",
        "mmol.L-1": "mmol.L-1",
        "M": "M",
        "mol/L": "mol/L",
        "mol.L-1": "mol.L-1",
    }
    return aliases.get(text, "")


def to_nanomolar(value: float, unit: str | None) -> float | None:
    """Convert a concentration to nM. Unknown units return None."""

    key = normalize_unit(unit)
    factor = _UNIT_TO_NM.get(key)
    if factor is None:
        return None
    return float(value) * factor


def pic50_from_nm(ic50_nm: float) -> float:
    """Convert an exact IC50 in nanomolar to pIC50."""

    if ic50_nm <= 0 or not math.isfinite(ic50_nm):
        raise ValueError(f"IC50 must be a positive finite number, got {ic50_nm}")
    return 9.0 - math.log10(ic50_nm)


def sample_offsets(total: int, page_size: int, windows: int) -> list[int]:
    """Choose ChEMBL page offsets.

    Small result sets are read in full. Large result sets contribute one page
    from each of ``windows`` positions spread across the archive (beginning,
    middle, end) so the sample is not only the oldest assays.
    """

    if total <= 0 or page_size <= 0 or windows <= 0:
        return []
    if total <= page_size * windows:
        return list(range(0, total, page_size))
    offsets: list[int] = []
    last = max(0, total - page_size)
    for index in range(windows):
        if windows == 1:
            start = 0
        else:
            start = int(round(index * last / (windows - 1)))
        start = max(0, min(start, last))
        if start not in offsets:
            offsets.append(start)
    return offsets


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def measurement_from_chembl(raw: dict[str, Any], gene: str) -> dict[str, Any]:
    """Flatten one ChEMBL activity JSON object into the columns QC expects."""

    parent = raw.get("parent_molecule_chembl_id") or raw.get("molecule_chembl_id") or ""
    return {
        "gene": gene,
        "molecule_chembl_id": raw.get("molecule_chembl_id") or "",
        "parent_molecule_chembl_id": parent,
        "target_chembl_id": raw.get("target_chembl_id") or "",
        "smiles": raw.get("canonical_smiles") or "",
        "standard_relation": raw.get("standard_relation") or "",
        "standard_value": raw.get("standard_value"),
        "standard_units": raw.get("standard_units") or "",
        "standard_type": raw.get("standard_type") or "",
        "pchembl_value": _as_float(raw.get("pchembl_value")),
        "data_validity_comment": raw.get("data_validity_comment") or "",
        "potential_duplicate": _as_int(raw.get("potential_duplicate"), 0),
        "activity_id": _as_int(raw.get("activity_id"), 0),
        "target_organism": raw.get("target_organism") or "",
    }


def qc_reason(row: dict[str, Any]) -> str | None:
    """Return a short rejection reason, or None if the row is usable."""

    if row.get("standard_type") not in {"", "IC50"}:
        return "not_ic50"
    if row.get("standard_relation") != "=":
        return "censored_or_inexact"
    if row.get("potential_duplicate"):
        return "potential_duplicate"
    if row.get("data_validity_comment"):
        return "validity_comment"
    organism = row.get("target_organism") or ""
    if organism and organism != "Homo sapiens":
        return "non_human"
    if not row.get("smiles") or not row.get("parent_molecule_chembl_id"):
        return "missing_structure_or_id"
    value = _as_float(row.get("standard_value"))
    if value is None:
        return "missing_value"
    ic50_nm = to_nanomolar(value, row.get("standard_units"))
    if ic50_nm is None:
        return "unknown_unit"
    if ic50_nm <= 0:
        return "nonpositive_ic50"
    return None


def apply_qc(measurements: list[dict[str, Any]]) -> pd.DataFrame:
    """Keep exact, molar, human IC50 rows and attach pIC50."""

    kept: list[dict[str, Any]] = []
    for row in measurements:
        if qc_reason(row) is not None:
            continue
        value = float(row["standard_value"])
        ic50_nm = to_nanomolar(value, row["standard_units"])
        if ic50_nm is None:
            continue
        pic50 = pic50_from_nm(ic50_nm)
        kept.append(
            {
                "gene": row["gene"],
                "molecule_chembl_id": row["molecule_chembl_id"],
                "parent_molecule_chembl_id": row["parent_molecule_chembl_id"],
                "target_chembl_id": row["target_chembl_id"],
                "smiles": row["smiles"],
                "ic50_nM": ic50_nm,
                "pIC50": pic50,
                "pchembl_value": row.get("pchembl_value"),
                "activity_id": row.get("activity_id") or 0,
                "active": int(pic50 >= ACTIVE_PIC50),
            }
        )
    frame = pd.DataFrame(kept)
    if frame.empty:
        return frame
    identified = frame.loc[frame["activity_id"] != 0].drop_duplicates(subset=["activity_id"])
    unidentified = frame.loc[frame["activity_id"] == 0]
    return pd.concat([identified, unidentified], ignore_index=True)


def deduplicate_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    """Median-collapse rows that share a parent compound and a target."""

    if frame.empty:
        return frame
    ordered = frame.sort_values(["activity_id", "molecule_chembl_id"])
    rows: list[dict[str, Any]] = []
    grouped = ordered.groupby(["parent_molecule_chembl_id", "target_chembl_id"], sort=False)
    for _, group in grouped:
        median = float(group["pIC50"].median())
        nearest = (group["pIC50"] - median).abs().idxmin()
        chosen = group.loc[nearest].to_dict()
        chosen["pIC50"] = median
        chosen["ic50_nM"] = float(10 ** (9.0 - median))
        chosen["n_measurements"] = int(len(group))
        chosen["active"] = int(median >= ACTIVE_PIC50)
        rows.append(chosen)
    return pd.DataFrame(rows).reset_index(drop=True)


def cap_per_gene(frame: pd.DataFrame, cap: int, seed: int) -> pd.DataFrame:
    """Downsample each gene to ``cap`` rows so one kinase cannot dominate training."""

    if frame.empty or cap <= 0:
        return frame
    import numpy as np

    rng = np.random.default_rng(seed)
    parts: list[pd.DataFrame] = []
    for _, group in frame.groupby("gene", sort=False):
        if len(group) > cap:
            chosen = rng.choice(group.index.to_numpy(), size=cap, replace=False)
            group = group.loc[sorted(chosen)]
        parts.append(group)
    if not parts:
        return frame.iloc[0:0].copy()
    return pd.concat(parts, ignore_index=True)
