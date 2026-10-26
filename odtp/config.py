"""Disease and target panel, plus the sampling choices used by the pipeline.

Identifiers were checked against the live Open Targets Platform (disease search
and ``disease(efoId:)``) and ChEMBL/UniProt accessions. Ensembl ids are
re-checked at runtime; the client falls back to a symbol search if a configured
id does not resolve to the expected gene.
"""

from __future__ import annotations

from dataclasses import dataclass

OPENTARGETS_URL = "https://api.platform.opentargets.org/api/v4/graphql"
CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"
UNIPROT_BASE = "https://rest.uniprot.org/uniprotkb"
RCSB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_GRAPHQL_URL = "https://data.rcsb.org/graphql"

USER_AGENT = "odtp/1.0 (academic; multi-disease drug-target prioritizer)"

# pIC50 = 9 - log10(IC50_nM). A value of 6.0 is an IC50 of 1 micromolar.
ACTIVE_PIC50 = 6.0
RANDOM_STATE = 42
TEST_SIZE = 0.2
FP_RADIUS = 2
FP_SIZE = 512

CHEMBL_PAGE_SIZE = 200
CHEMBL_WINDOWS = 3
PER_TARGET_CAP = 250
SMALL_MOL_MW_MIN = 150.0
SMALL_MOL_MW_MAX = 700.0
DISEASE_TOP_N = 25
RCSB_CRYSTAL_ROWS = 6

AREA_ORDER = (
    "Neurodegenerative",
    "Metabolic",
    "Autoimmune",
    "Infectious",
    "Oncology",
)

AREA_COLORS = {
    "Neurodegenerative": "#0F6E6E",
    "Metabolic": "#C2410C",
    "Autoimmune": "#6D28D9",
    "Infectious": "#0369A1",
    "Oncology": "#BE123C",
}


@dataclass(frozen=True)
class Disease:
    """One Open Targets disease used for association evidence."""

    id: str
    name: str
    category: str


@dataclass(frozen=True)
class Target:
    """A human protein with a UniProt accession and a primary therapeutic area.

    The primary area is only a display grouping. Cross-disease association
    scores are fetched for every disease in the panel, so a gene such as JAK2
    can still show oncology evidence while sitting in the autoimmune group.
    """

    symbol: str
    uniprot: str
    ensembl: str
    category: str


# Familial hypercholesterolemia (MONDO_0005439) is the disease entity used for
# the hypercholesterolemia indication. It is the inherited high-cholesterol
# disease most directly tied to PCSK9 and HMGCR. The broader phenotype term
# HP_0003124 also exists in Open Targets; this panel uses the disease record.
DISEASES: tuple[Disease, ...] = (
    Disease("MONDO_0004975", "Alzheimer disease", "Neurodegenerative"),
    Disease("MONDO_0005180", "Parkinson disease", "Neurodegenerative"),
    Disease("MONDO_0005027", "epilepsy", "Neurodegenerative"),
    Disease("MONDO_0005148", "type 2 diabetes mellitus", "Metabolic"),
    Disease("MONDO_0005439", "familial hypercholesterolemia", "Metabolic"),
    Disease("MONDO_0008383", "rheumatoid arthritis", "Autoimmune"),
    Disease("MONDO_0005265", "inflammatory bowel disease", "Autoimmune"),
    Disease("MONDO_0005109", "HIV infectious disease", "Infectious"),
    Disease("MONDO_0005136", "malaria", "Infectious"),
    Disease("MONDO_0018076", "tuberculosis", "Infectious"),
    Disease("MONDO_0100096", "COVID-19", "Infectious"),
    Disease("MONDO_0005233", "non-small cell lung carcinoma", "Oncology"),
    Disease("MONDO_0004989", "breast carcinoma", "Oncology"),
    Disease("MONDO_0005575", "colorectal cancer", "Oncology"),
    Disease("MONDO_0018177", "glioblastoma", "Oncology"),
    Disease("MONDO_0005105", "melanoma", "Oncology"),
)

TARGETS: tuple[Target, ...] = (
    Target("LRRK2", "Q5S007", "ENSG00000188906", "Neurodegenerative"),
    Target("BACE1", "P56817", "ENSG00000186331", "Neurodegenerative"),
    Target("MAPT", "P10636", "ENSG00000186868", "Neurodegenerative"),
    Target("ACHE", "P22303", "ENSG00000087085", "Neurodegenerative"),
    Target("GLP1R", "P43220", "ENSG00000112164", "Metabolic"),
    Target("DPP4", "P27487", "ENSG00000197635", "Metabolic"),
    Target("PCSK9", "Q8NBP7", "ENSG00000169174", "Metabolic"),
    Target("HMGCR", "P04035", "ENSG00000113161", "Metabolic"),
    Target("JAK1", "P23458", "ENSG00000162434", "Autoimmune"),
    Target("JAK2", "O60674", "ENSG00000096968", "Autoimmune"),
    Target("TNF", "P01375", "ENSG00000232810", "Autoimmune"),
    Target("TYK2", "P29597", "ENSG00000105397", "Autoimmune"),
    Target("CCR5", "P51681", "ENSG00000160791", "Infectious"),
    Target("CXCR4", "P61073", "ENSG00000121966", "Infectious"),
    Target("DHODH", "Q02127", "ENSG00000102967", "Infectious"),
    Target("EGFR", "P00533", "ENSG00000146648", "Oncology"),
    Target("BRAF", "P15056", "ENSG00000157764", "Oncology"),
    Target("ERBB2", "P04626", "ENSG00000141736", "Oncology"),
    Target("ALK", "Q9UM73", "ENSG00000171094", "Oncology"),
    Target("MET", "P08581", "ENSG00000105976", "Oncology"),
    Target("PIK3CA", "P42336", "ENSG00000121879", "Oncology"),
    Target("KRAS", "P01116", "ENSG00000133703", "Oncology"),
    Target("CDK4", "P11802", "ENSG00000135446", "Oncology"),
    Target("PARP1", "P09874", "ENSG00000143799", "Oncology"),
    Target("ABL1", "P00519", "ENSG00000097007", "Oncology"),
    Target("KIT", "P10721", "ENSG00000157404", "Oncology"),
    Target("RET", "P07949", "ENSG00000165731", "Oncology"),
    Target("FGFR1", "P11362", "ENSG00000077782", "Oncology"),
    Target("BTK", "Q06187", "ENSG00000010671", "Oncology"),
)

DATATYPE_FIELDS = (
    "genetic_association",
    "somatic_mutation",
    "literature",
    "affected_pathway",
    "animal_model",
    "rna_expression",
    "genetic_literature",
    "clinical",
    "known_drug",
)

MOLECULE_FEATURES = (
    "mw",
    "logp",
    "hbd",
    "hba",
    "rotatable_bonds",
    "tpsa",
    "aromatic_rings",
    "lipinski_violations",
)

TARGET_FEATURES = (
    "max_association_score",
    "max_genetic_association",
    "max_somatic_mutation",
    "max_known_drug_score",
    "max_literature",
    "n_panel_diseases",
    "safety_liability_count",
    "approved_drug_sm",
    "high_quality_pocket_sm",
    "structure_with_ligand_sm",
    "drug_candidate_count",
    "sequence_length",
    "domain_count",
    "best_crystal_resolution",
    "crystal_structure_count",
    "ligand_bound_crystal_count",
    "area_neurodegenerative",
    "area_metabolic",
    "area_autoimmune",
    "area_infectious",
    "area_oncology",
)


def feature_names(fp_size: int = FP_SIZE) -> list[str]:
    """Column order of the model matrix: descriptors, target biology, fingerprint."""

    return [
        *MOLECULE_FEATURES,
        *TARGET_FEATURES,
        *[f"fp_{index:03d}" for index in range(fp_size)],
    ]


def target_by_symbol(symbol: str) -> Target:
    """Return the configured target for a gene symbol."""

    key = symbol.upper()
    for target in TARGETS:
        if target.symbol.upper() == key:
            return target
    known = ", ".join(target.symbol for target in TARGETS)
    raise KeyError(f"Unknown gene {symbol!r}. Configured genes: {known}")
