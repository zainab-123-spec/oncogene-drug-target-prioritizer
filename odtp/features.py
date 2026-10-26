"""Physicochemical descriptors, Murcko scaffolds, and the model matrix.

Fingerprints use the RDKit Morgan generator (radius 2, 512 bits). Lipinski
violations count how many of MW > 500, LogP > 5, hydrogen-bond donors > 5,
and hydrogen-bond acceptors > 10 are true. An empty Bemis-Murcko scaffold
(a molecule with no rings) is labeled with its own canonical SMILES so
unrelated chains are not glued into one giant split group.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Lipinski, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

from odtp.config import (
    ACTIVE_PIC50,
    AREA_ORDER,
    FP_RADIUS,
    FP_SIZE,
    MOLECULE_FEATURES,
    SMALL_MOL_MW_MAX,
    SMALL_MOL_MW_MIN,
    TARGET_FEATURES,
    feature_names,
)

_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_SIZE)
_CARBON = re.compile(r"[Cc]")

_AREA_COLUMNS = {
    "Neurodegenerative": "area_neurodegenerative",
    "Metabolic": "area_metabolic",
    "Autoimmune": "area_autoimmune",
    "Infectious": "area_infectious",
    "Oncology": "area_oncology",
}


def lipinski_violations(mw: float, logp: float, hbd: float, hba: float) -> int:
    """Count Lipinski rule-of-five failures."""

    return int(mw > 500) + int(logp > 5) + int(hbd > 5) + int(hba > 10)


def _scaffold_id(mol: Chem.Mol) -> str:
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    if scaffold:
        return scaffold
    return "ACYCLIC::" + Chem.MolToSmiles(mol, canonical=True)


def compute_molecule_features(smiles: str) -> dict[str, Any] | None:
    """Descriptors and scaffold for one SMILES, or None if RDKit cannot parse it."""

    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mw = float(Descriptors.MolWt(mol))
    logp = float(Descriptors.MolLogP(mol))
    hbd = float(Lipinski.NumHDonors(mol))
    hba = float(Lipinski.NumHAcceptors(mol))
    return {
        "mw": mw,
        "logp": logp,
        "hbd": hbd,
        "hba": hba,
        "rotatable_bonds": float(Lipinski.NumRotatableBonds(mol)),
        "tpsa": float(Descriptors.TPSA(mol)),
        "aromatic_rings": float(rdMolDescriptors.CalcNumAromaticRings(mol)),
        "lipinski_violations": lipinski_violations(mw, logp, hbd, hba),
        "scaffold": _scaffold_id(mol),
        "canonical_smiles": Chem.MolToSmiles(mol, canonical=True),
    }


def morgan_bits(smiles: str, fp_size: int = FP_SIZE) -> np.ndarray | None:
    """Morgan bit vector, or None if the SMILES is invalid."""

    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return None
    if fp_size == FP_SIZE:
        fingerprint = _MORGAN.GetFingerprint(mol)
        array = np.zeros((FP_SIZE,), dtype=np.float64)
    else:
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=fp_size)
        fingerprint = generator.GetFingerprint(mol)
        array = np.zeros((fp_size,), dtype=np.float64)
    DataStructs.ConvertToNumpyArray(fingerprint, array)
    return array


def add_molecular_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Append descriptors. Rows with unparseable SMILES are dropped."""

    if frame.empty:
        return frame
    records: list[dict[str, Any]] = []
    for record in frame.to_dict(orient="records"):
        features = compute_molecule_features(str(record.get("smiles", "")))
        if features is None:
            continue
        merged = dict(record)
        merged.update(features)
        records.append(merged)
    if not records:
        return frame.iloc[0:0].copy()
    return pd.DataFrame(records)


def filter_small_molecules(
    frame: pd.DataFrame,
    mw_min: float = SMALL_MOL_MW_MIN,
    mw_max: float = SMALL_MOL_MW_MAX,
) -> pd.DataFrame:
    """Keep organic small molecules inside the molecular-weight window."""

    if frame.empty:
        return frame
    smiles = frame["smiles"].astype(str)
    mask = (
        frame["mw"].between(mw_min, mw_max)
        & smiles.str.contains(_CARBON)
        & frame["scaffold"].astype(str).ne("")
    )
    return frame.loc[mask].reset_index(drop=True)


def _max_field(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return float("nan")
    return _column_max(frame[column])


def _column_max(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().any():
        return float(values.max())
    return float("nan")


def build_target_feature_table(
    associations: pd.DataFrame,
    annotations: pd.DataFrame,
) -> pd.DataFrame:
    """One row per gene: the strongest panel evidence plus structure and safety."""

    rows: list[dict[str, Any]] = []
    genes = list(annotations["gene"]) if not annotations.empty else []
    has_associations = not associations.empty and "gene" in associations.columns
    for gene in genes:
        ann = annotations.loc[annotations["gene"] == gene].iloc[0]
        if has_associations:
            subset = associations.loc[associations["gene"] == gene]
        else:
            subset = associations.iloc[0:0]
        area = str(ann.get("therapeutic_area", ""))
        if len(subset):
            score = pd.to_numeric(subset.get("association_score"), errors="coerce")
            n_panel = int((score >= 0.2).sum())
        else:
            n_panel = 0
        row: dict[str, Any] = {
            "gene": gene,
            "max_association_score": _max_field(subset, "association_score"),
            "max_genetic_association": _max_field(subset, "genetic_association"),
            "max_somatic_mutation": _max_field(subset, "somatic_mutation"),
            "max_known_drug_score": _max_field(subset, "known_drug_score"),
            "max_literature": _max_field(subset, "literature"),
            "n_panel_diseases": n_panel,
            "safety_liability_count": float(ann.get("safety_liability_count") or 0),
            "approved_drug_sm": float(ann.get("approved_drug_sm") or 0),
            "high_quality_pocket_sm": float(ann.get("high_quality_pocket_sm") or 0),
            "structure_with_ligand_sm": float(ann.get("structure_with_ligand_sm") or 0),
            "drug_candidate_count": float(ann.get("drug_candidate_count") or 0),
            "sequence_length": _optional_float(ann.get("sequence_length")),
            "domain_count": float(ann.get("domain_count") or 0),
            "best_crystal_resolution": _optional_float(ann.get("best_crystal_resolution")),
            "crystal_structure_count": float(ann.get("crystal_structure_count") or 0),
            "ligand_bound_crystal_count": float(ann.get("ligand_bound_crystal_count") or 0),
        }
        for category in AREA_ORDER:
            row[_AREA_COLUMNS[category]] = 1.0 if area == category else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def _optional_float(value: Any) -> float:
    if value is None or value == "" or (isinstance(value, float) and np.isnan(value)):
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def assemble_matrix(
    frame: pd.DataFrame,
    fp_size: int = FP_SIZE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Build X, y (pIC50), active labels, and the aligned metadata frame.

    The frame must already contain molecule descriptors, target features, SMILES,
    and pIC50. Fingerprints are computed here and are not stored in the CSV.
    """

    names = feature_names(fp_size)
    kept_rows: list[dict[str, Any]] = []
    blocks: list[np.ndarray] = []
    for record in frame.to_dict(orient="records"):
        bits = morgan_bits(str(record["smiles"]), fp_size=fp_size)
        if bits is None:
            continue
        values: list[float] = []
        ok = True
        for name in (*MOLECULE_FEATURES, *TARGET_FEATURES):
            try:
                number = float(record[name])
            except (KeyError, TypeError, ValueError):
                ok = False
                break
            values.append(number)
        if not ok:
            continue
        vector = np.concatenate([np.asarray(values, dtype=np.float64), bits])
        if vector.shape[0] != len(names):
            continue
        blocks.append(vector)
        kept_rows.append(record)
    if not blocks:
        empty = np.zeros((0, len(names)))
        return empty, empty[:, 0], empty[:, 0], frame.iloc[0:0].copy()
    matrix = np.vstack(blocks)
    meta = pd.DataFrame(kept_rows).reset_index(drop=True)
    y = meta["pIC50"].to_numpy(dtype=np.float64)
    active = (y >= ACTIVE_PIC50).astype(np.int64)
    meta["active"] = active
    return matrix, y, active, meta
