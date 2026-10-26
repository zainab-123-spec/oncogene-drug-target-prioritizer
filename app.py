"""Interactive explorer for the multi-disease target and bioactivity tables.

Run from the repository root:

    streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from odtp.config import ACTIVE_PIC50, AREA_ORDER
from odtp.model import predict_smiles

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data" / "processed"
MODEL_PATH = ROOT / "data" / "models" / "demo_hgb.joblib"

EXAMPLES = {
    "Gefitinib (oncology, EGFR)": (
        "EGFR",
        "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",
    ),
    "Donepezil (Alzheimer disease, ACHE)": (
        "ACHE",
        "COc1cc2c(cc1OC)C(=O)C(CC1CCN(Cc3ccccc3)CC1)C2",
    ),
    "Sitagliptin (type 2 diabetes, DPP4)": (
        "DPP4",
        "N[C@@H](CC(=O)N1Cc2nn(C)c(C(F)(F)F)c2C[C@H]1F)Cc1cc(F)c(F)cc1F",
    ),
    "Maraviroc (HIV, CCR5)": (
        "CCR5",
        "Cc1nnc(C(C)C)n1[C@@H]1C[C@H]2CC[C@@H](C1)N2CC[C@H](NC(=O)C1CCC(F)(F)CC1)c1ccccc1",
    ),
    "Tofacitinib (autoimmune, JAK1)": (
        "JAK1",
        "C[C@@H]1CCN(C(=O)CC#N)C[C@@H]1N(C)c1ncnc2[nH]ccc12",
    ),
}


@st.cache_data
def load_table(name: str) -> pd.DataFrame:
    path = PROCESSED / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_resource
def load_model() -> dict | None:
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def main() -> None:
    st.set_page_config(
        page_title="Multi-Disease Drug Target & Bioactivity Prioritizer",
        layout="wide",
    )
    st.title("Multi-Disease Drug Target & Bioactivity Prioritizer")
    st.caption(
        "Zainab Javed · BS Bioinformatics, Quaid-i-Azam University, Islamabad · "
        "Open Targets Platform, ChEMBL, UniProt, and the RCSB PDB"
    )
    st.write(
        "This explorer ranks human targets across neurodegenerative, metabolic, "
        "autoimmune, infectious, and oncology indications, then shows measured "
        "small-molecule potency. It is a research tool. It is not a medical device "
        "and it does not recommend treatment."
    )

    rankings = load_table("disease_rankings.csv")
    scores = load_table("gene_disease_scores.csv")
    annotations = load_table("target_annotations.csv")
    dedup = load_table("activities_dedup.csv")
    model_table = load_table("activities_model.csv")
    metrics_path = PROCESSED / "metrics.json"
    if rankings.empty or annotations.empty:
        st.warning(
            "Processed tables were not found. From the repository root, run `python -m odtp all`."
        )
        return

    overview, ranking_tab, target_tab, chemistry_tab, predict_tab, metrics_tab = st.tabs(
        [
            "Overview",
            "Disease rankings",
            "Target evidence",
            "Compounds and Lipinski",
            "Try a molecule",
            "Benchmark",
        ]
    )

    with overview:
        counts = (
            annotations.groupby("therapeutic_area")
            .size()
            .reindex(list(AREA_ORDER))
            .fillna(0)
            .astype(int)
        )
        st.subheader("Curated targets by therapeutic area")
        st.bar_chart(counts, horizontal=True)
        if not model_table.empty:
            st.subheader("Modeled small molecules")
            by_gene = (
                model_table.groupby(["therapeutic_area", "gene"])
                .size()
                .rename("compounds")
                .reset_index()
            )
            st.dataframe(by_gene, hide_index=True)
        st.info(
            "Association scores come from Open Targets. pIC50 values come from "
            "ChEMBL binding IC50 assays after censoring filters and parent-compound "
            "deduplication. pIC50 = 9 − log10(IC50 in nM)."
        )

    with ranking_tab:
        diseases = rankings["disease_name"].drop_duplicates().tolist()
        disease = st.selectbox(
            "Disease",
            diseases,
            index=_default_index(diseases, "Alzheimer disease"),
        )
        subset = rankings.loc[rankings["disease_name"] == disease].sort_values("rank")
        category = subset["category"].iloc[0] if not subset.empty else ""
        st.caption(f"Therapeutic area in this panel: {category}")
        st.dataframe(
            subset[
                [
                    "rank",
                    "gene",
                    "target_name",
                    "association_score",
                    "genetic_association",
                    "somatic_mutation",
                    "known_drug_score",
                    "known_drug_source",
                    "literature",
                ]
            ],
            hide_index=True,
        )

    with target_tab:
        gene = st.selectbox("Gene", annotations["gene"].tolist())
        card = annotations.loc[annotations["gene"] == gene].iloc[0]
        left, right, third = st.columns(3)
        left.metric("Primary area", str(card["therapeutic_area"]))
        right.metric("Safety events", int(card["safety_liability_count"]))
        third.metric("Clinical candidates", int(card["drug_candidate_count"]))
        st.write(
            f"**{card.get('approved_name', gene)}** · UniProt `{card.get('uniprot', '')}` · "
            f"best crystal {card.get('best_crystal_pdb_id') or 'n/a'} at "
            f"{card.get('best_crystal_resolution') or 'n/a'} Å"
        )
        gene_scores = scores.loc[scores["gene"] == gene].copy()
        if not gene_scores.empty:
            st.bar_chart(
                gene_scores.set_index("disease_name")["association_score"].fillna(0),
                horizontal=True,
            )
            st.dataframe(gene_scores, hide_index=True)

    with chemistry_tab:
        choices = sorted(dedup["gene"].unique()) if not dedup.empty else []
        chem_gene = st.selectbox("Target", choices)
        compounds = dedup.loc[dedup["gene"] == chem_gene].sort_values("pIC50", ascending=False)
        if compounds.empty:
            st.write("No deduplicated IC50 rows for this gene.")
        else:
            active = int((compounds["pIC50"] >= ACTIVE_PIC50).sum())
            st.metric("Compounds with pIC50 ≥ 6", f"{active} / {len(compounds)}")
            st.dataframe(
                compounds[
                    [
                        "molecule_chembl_id",
                        "pIC50",
                        "ic50_nM",
                        "mw",
                        "logp",
                        "hbd",
                        "hba",
                        "tpsa",
                        "aromatic_rings",
                        "rotatable_bonds",
                        "lipinski_violations",
                        "smiles",
                    ]
                ].head(100),
                hide_index=True,
            )

    with predict_tab:
        bundle = load_model()
        st.write(
            "The demo model is a HistGradientBoosting regressor and classifier refit on "
            "the full modeled table after the held-out benchmark was recorded. "
            "Use the Benchmark tab for the scaffold-split numbers."
        )
        if bundle is None:
            st.warning("No demo model yet. Run `python -m odtp train`.")
        else:
            example = st.selectbox("Example", ["Custom"] + list(EXAMPLES))
            default_gene = annotations["gene"].tolist()[0]
            default_smiles = ""
            if example != "Custom":
                default_gene, default_smiles = EXAMPLES[example]
            chosen = st.selectbox(
                "Predict against",
                annotations["gene"].tolist(),
                index=_default_index(annotations["gene"].tolist(), default_gene),
            )
            smiles = st.text_input("SMILES", value=default_smiles)
            if st.button("Predict", type="primary") and smiles.strip():
                features = pd.read_csv(PROCESSED / "target_features.csv")
                row = features.loc[features["gene"] == chosen]
                if row.empty:
                    st.error("That gene has no feature row.")
                else:
                    try:
                        result = predict_smiles(bundle, smiles.strip(), row.iloc[0])
                    except ValueError as exc:
                        st.error(str(exc))
                    else:
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Predicted pIC50", f"{result['pIC50']:.2f}")
                        c2.metric("P(active)", f"{result['active_probability']:.2f}")
                        c3.metric("Lipinski violations", int(result["lipinski_violations"]))
                        st.caption(
                            f"MW {result['mw']:.1f} · LogP {result['logp']:.2f} · "
                            f"HBD {result['hbd']:.0f} · HBA {result['hba']:.0f} · "
                            f"TPSA {result['tpsa']:.1f}"
                        )

    with metrics_tab:
        if not metrics_path.exists():
            st.write("Train the models to fill this tab.")
        else:
            import json

            metrics = json.loads(metrics_path.read_text())
            frame = pd.DataFrame(metrics["models"])
            st.dataframe(frame, hide_index=True)
            st.json(metrics.get("leakage", {}))


def _default_index(options: list[str], value: str) -> int:
    try:
        return options.index(value)
    except ValueError:
        return 0


main()
