# Learning guide

This guide explains the Multi-Disease Drug Target & Bioactivity Prioritizer in plain English. It is written for someone who can read Python and has met biology, but does not live in a computational chemistry group. The repository is [oncogene-drug-target-prioritizer](https://github.com/zainab-123-spec/oncogene-drug-target-prioritizer). The code package is still called `odtp` so the commands stay short.

Read the README for how to run the project. Read this file to understand why the pieces exist.

## 1. The question

A disease is not a single broken part. It is a process in a tissue, and many genes touch that process. Drug discovery still has to pick a **protein** to bind, then a **molecule** that binds it strongly enough, safely enough, and with the right exposure.

This project separates those two jobs, because they are different measurements:

- **Target prioritization** asks, "Does public evidence connect this human gene to this disease?" Open Targets compresses genetics, somatic mutations, literature, pathways, animal models, and clinical drug evidence into association scores between 0 and 1.
- **Bioactivity** asks, "When a chemist already has a compound and a protein, how much compound does it take to inhibit that protein by half?" ChEMBL stores that experiment as an IC50.

A high association score does not mean a compound exists. A potent IC50 does not mean the protein matters in patients. The app shows both, side by side, so you do not confuse them.

## 2. The disease panel

The panel has five areas so the project is not a cancer-only demo.

**Neurodegenerative.** Alzheimer disease, Parkinson disease, and epilepsy. LRRK2 is a kinase with genetic links to Parkinson disease. BACE1 cuts the amyloid precursor protein; it was a famous Alzheimer target and a famous clinical disappointment. MAPT is the tau protein. ACHE (acetylcholinesterase) is the target of donepezil, an older symptomatic Alzheimer drug. Symptomatic is not the same as disease-modifying. The model does not know that distinction. You should.

**Metabolic.** Type 2 diabetes and familial hypercholesterolemia. DPP4 and GLP1R are diabetes targets (gliptins and GLP-1 medicines). HMGCR is the statin target. PCSK9 is the genetics success story: people with low PCSK9 activity have lower LDL cholesterol, and medicines against PCSK9 followed. Many GLP1R and PCSK9 ligands are peptides or antibodies, so the small-molecule filter may leave few training rows. That is a result, not a bug. `qc_funnel.csv` is where you check it.

**Autoimmune.** Rheumatoid arthritis and inflammatory bowel disease. JAK1, JAK2, and TYK2 are kinases in cytokine signaling. TNF is tumor necrosis factor, a classic antibody target. JAK2 is grouped here for display and is also an oncology gene. The disease heatmap is allowed to show both.

**Infectious disease, host side.** HIV, malaria, tuberculosis, and COVID-19. CCR5 and CXCR4 are human co-receptors HIV uses to enter cells. Maraviroc blocks CCR5. DHODH is the human enzyme that helps make pyrimidines. It is a host protein. This project does not model pathogen genomes and it does not design organisms.

**Oncology.** Non-small cell lung carcinoma, breast carcinoma, colorectal cancer, glioblastoma, and melanoma, with kinases and other enzymes that already have medicines or serious chemical matter: EGFR, BRAF, ERBB2, ALK, MET, PIK3CA, KRAS, CDK4, PARP1, ABL1, KIT, RET, FGFR1, BTK.

Open Targets ids in `odtp/config.py` were checked against the live Platform API. The hypercholesterolemia indication uses familial hypercholesterolemia (`MONDO_0005439`), which is a disease record, rather than the broader phenotype term alone.

## 3. What each API is for

**Open Targets GraphQL** (`api.platform.opentargets.org`). One query returns the top associated genes for a disease. Another asks, for a specific gene such as PCSK9, "what is your score for familial hypercholesterolemia?" The second call uses `BFilter`. That filter is a search box, not an exact key. For a short symbol like MET it can return a different gene first, so the parser keeps the row whose approved symbol matches exactly.

Datatype scores travel with the overall score. The ones this project cares about most are genetic association, somatic mutation, literature, and known-drug evidence. On the current API, known-drug evidence arrives as the id `clinical`. The code prefers an id literally called `known_drug` if a future or past response includes it, and it records which id it used.

A separate target query returns tractability labels for the small-molecule modality (approved drug, high-quality pocket, structure with ligand, and others), a list of safety events, and a count of drugs and clinical candidates.

**ChEMBL REST.** For each UniProt accession we take the Homo sapiens single-protein target, then download binding assays whose standard type is IC50. Each row can include a relation (`=`, `>`, `<`), a value, a unit, a SMILES string, a parent molecule id, and flags for duplicates or suspicious values.

**UniProt.** Sequence length and the number of annotated domains. Domains are the folded modules of the protein (a kinase domain, an EGF-like domain). They are context, not a 3D dock.

**RCSB PDB.** A search for X-ray entries of that UniProt accession, best resolution first, then a small GraphQL batch for those entries. We record how many crystal structures exist, the best resolution in Ångströms (lower is sharper), and how many of the inspected crystals contain a non-polymer ligand. That ligand count is not a computed pocket. Open Targets already provides a high-quality pocket flag, and we keep that flag as its own feature.

## 4. From an IC50 to a pIC50

IC50 is the concentration that gives 50% inhibition in that assay. It might be 10 nM or 10,000 nM. Those two numbers are not a friendly regression target: the errors are skewed, and a model can look good by predicting "roughly thousands" for everything.

Chemists use the negative log:

```text
pIC50 = -log10(IC50 in molar)
       = 9 - log10(IC50 in nanomolar)
```

Check the arithmetic on two round values:

- 10 nM is `1e-8` molar. `-log10(1e-8) = 8`. Also `9 - log10(10) = 8`.
- 1 µM is 1000 nM. `9 - log10(1000) = 6`.

So pIC50 8 is tenfold more potent than pIC50 7, and pIC50 6 is the 1 µM line we call "active" for the classifier. The tests lock this formula in `tests/test_qc.py`. If someone "simplifies" the 9, the tests fail.

ChEMBL also publishes `pchembl_value`, which is their own log potency. We keep it as a column for comparison. The training label is the value we compute, so the formula in the README is the formula in the file.

## 5. Why censored rows are thrown away

A lab report sometimes says IC50 > 10,000 nM. That means "we did not see 50% inhibition at the highest dose we tried." It does not mean the IC50 is 10,000. If you train on it as 10,000, you teach the model a fake precise number.

This project keeps only `standard_relation == "="`. A survival model or a censored-regression model could use the inequalities. That is a real next step. It is not what this label claims to be. The same filter drops validity comments (ChEMBL's way of saying the number looks wrong) and rows flagged as potential duplicates.

Units are another quiet bug. `1 µM` and `1000 nM` are the same concentration. `1` with no unit is not usable. Unknown units, including mass-per-volume units that need a molecular weight to convert, are dropped.

## 6. Deduplication

One compound is often tested more than once. Salts of the same parent get different ChEMBL ids and the same parent id. If you keep every row, a popular scaffold is copied many times and a random split will put copies on both sides.

We group by parent compound and target, take the median pIC50, and remember how many measurements went into it. The median resists one wild assay better than the mean. The closest original row donates the SMILES and ids, then the pIC50 field is overwritten with the median so the label matches the rule.

## 7. Descriptors, Lipinski, fingerprints, scaffolds

**Descriptors** are numbers you can say out loud. Molecular weight. LogP, a rough love of fat versus water. Hydrogen-bond donors and acceptors. Rotatable bonds, a rough flexibility count. TPSA, polar surface area. Aromatic ring count.

**Lipinski's rule of five** is a 1997 observation about oral drugs, not a law. A violation is counted for each of: weight over 500, LogP over 5, more than 5 donors, more than 10 acceptors. Zero violations is common for classic oral drugs and a bad filter for natural products, some antibiotics, and protein-protein interaction inhibitors. TNF ligands often look nothing like a Lipinski molecule. The figure of "fraction with zero violations" is there so you can see that, not so you can worship the rule.

**Morgan fingerprints** are a different language. Walk out 2 bonds from each atom and hash what you see into 512 bits. Two molecules with similar neighborhoods light up similar bits. The bits are not individually meaningful, which is why the importance plot sums them instead of naming `fp_173`.

**Bemis-Murcko scaffold.** Strip the side chains and keep the ring systems and the linkers between them. Benzene and toluene share a scaffold. Ethanol has no rings, so the scaffold string would be empty. An empty string would glue every chain into one group and then the scaffold split would move all of them together. We label ringless molecules as `ACYCLIC::` plus their canonical SMILES, so each chain is its own group. That is conservative: it refuses to pretend that all flexible molecules are analogs.

## 8. Why a random split flatters you

Suppose the training set contains 20 analogs of gefitinib and the test set contains 2 more. A random split will do that constantly, because it cuts rows, not chemotypes. The model can memorize "this shape of quinazoline is potent on EGFR" and look accurate on the test set without ever having seen a new shape.

A scaffold split puts every analog of a scaffold entirely in train or entirely in test. The test scaffolds are new skeletons. The RMSE usually goes up. That increase is the leakage gap:

```text
RMSE gap = scaffold RMSE - random RMSE
AUC gap  = random ROC-AUC - scaffold ROC-AUC
```

Positive means the random split was the optimistic one. The code also counts shared scaffolds. For the scaffold split that count is required to be zero. For the random split it is usually not. Both facts are in `metrics.json` under `splits` and `leakage`.

This does not catch every leak. A scaffold split still lets similar scaffolds through (a pyridine versus a phenyl). It does not group by protein pocket. It is the standard first honesty check in ligand-based modeling, not the last one.

Time-split (train on old papers, test on new papers) and analog-exclusion by a similarity threshold are the usual next checks. They are named here so you can talk about them. They are not silently claimed as results.

## 9. How to read the metrics

**RMSE** is the typical size of the pIC50 error, in the same units as the label. An RMSE of 1.0 means you are often off by about tenfold in concentration, because one pIC50 unit is a factor of ten. Compare it with the mean baseline, which predicts the training average for every row. If your model does not beat that baseline on the scaffold split, it is not predicting potency.

**Pearson r** is linear correlation. **Spearman rho** is rank correlation. Spearman stays meaningful if the relationship is monotonic but not a straight line. Both need variation in the predictions. A constant prediction has no correlation, and the code records null instead of a made-up value.

**ROC-AUC** asks whether actives outrank inactives. 0.5 is a coin flip. It can look strong when almost every compound is active, which is common in published IC50 sets.

**PR-AUC** (average precision) is more sensitive to that imbalance. The prevalence baseline's PR-AUC is simply the fraction of actives in the test fold. A model that does not beat that number is not finding actives better than "guess the base rate."

Regressors get all five numbers, using predicted pIC50 as the score for the two ranking metrics. Classifiers get ROC-AUC and PR-AUC only. Their RMSE cells are empty on purpose.

## 10. The model, without mythology

Ridge is a linear model with a penalty so 512 correlated fingerprint bits do not explode. It is the interpretable baseline that can still use the fingerprint. Random forest and histogram gradient boosting are nonlinear. Trees do not need standardized inputs. Ridge does. The scaler is fit on the training fold only. Medians for missing crystal resolutions are also fit on the training fold only. Fitting them on the full matrix would leak test information into the features.

The random forest's impurity importances are summed across fingerprint bits and plotted for the descriptor and target features. Impurity importance favors features with many possible cut points. Treat the plot as a hint, not as a causal ranking. It comes from the scaffold-split training fold of the real fitted forest.

The Streamlit demo model is a second fit, on every modeled row, saved after the metrics file is written. If you quote a number in an interview, quote `metrics.json`, not a live click on gefitinib.

## 11. Module by module

**`odtp/config.py`.** The disease ids, gene symbols, UniProt accessions, Ensembl ids, feature-column order, and the sampling constants (page size, windows, cap, weight window, active threshold, fingerprint size). If you change the cap, the benchmark changes. Say so.

**`odtp/http.py`.** Every API call goes through one client. The cache key is a hash of method, URL, query, and body. A hit does not sleep and does not count as a network call. Failures (429, 500, 502, 503, 504, connection errors) retry with backoff. A GraphQL body that is only an `errors` list is not cached. A 404 is not retried and not cached.

**`odtp/clients/opentargets.py`.** Query text, parsers, and the exact-symbol guard. Parsers are ordinary functions, so tests feed them dictionaries and never the network.

**`odtp/clients/chembl.py`.** Target lookup by accession, then IC50 pages. Large targets are not downloaded in full. `sample_offsets` picks pages spread across the archive.

**`odtp/clients/structures.py`.** UniProt JSON and the RCSB search-plus-GraphQL pair.

**`odtp/qc.py`.** The scientific filters. If you want to know whether a row died, `qc_reason` returns a short string.

**`odtp/features.py`.** RDKit. Invalid SMILES become dropped rows, not zeros. Target features are the max evidence across the panel plus structure and safety, and a one-hot of the five areas.

**`odtp/model.py`.** Splits, metrics, leakage arithmetic, demo fit, and `predict_smiles` for the app.

**`odtp/plotting.py`.** Matplotlib on the Agg backend so a server without a screen can still write PNGs.

**`odtp/reports.py`.** Turns a metrics dictionary into markdown and HTML. It does not contain a hidden table of dream scores. The test `test_benchmark_markdown_prints_supplied_metrics` passes a fake 1.250 and checks that 1.250 is what comes out. That test is about the formatter. It is not a claim about ChEMBL.

**`odtp/pipeline.py`.** The order of operations: rankings, curated scores, tractability, structures, ChEMBL, QC, descriptors, cap, save, then later train and analyze.

**`odtp/cli.py` and `app.py`.** `python -m odtp build`, `train`, `analyze EGFR`, and `all`. The app reads the CSVs. It does not call the APIs on each click.

## 12. How the tests stay honest

CI has no ChEMBL credentials and should not need the network. Tests construct the JSON those APIs return and check the parsers. A fake session scripts status codes so retry and cache behavior are deterministic. A fake Open Targets / ChEMBL / structure trio runs `build_dataset` on a handful of molecules, including a censored row, an invalid SMILES, a micromolar value, a duplicate parent, and benzene (too small to stay in the model). The funnel counts are asserted.

The scaffold tests build 20 scaffolds with 10 copies each. They assert the scaffold split shares zero scaffolds and the random split shares some. A fast evaluation (`fast=True`, fewer trees) checks that RMSE and correlations exist for regressors and that classifier RMSE is null. `fast=True` is for that test. The reported benchmark is the full command, which does not pass `--fast`.

## 13. Limits you should say out loud

- Multi-lab IC50s are noisy. A median helps. It does not create a single protocol.
- ChEMBL is enriched for compounds someone thought were worth publishing. The active rate is not the active rate of chemical space.
- We sampled pages. We did not train on every EGFR assay. The funnel's `total_reported` column is the full count.
- Weight 150–700 is a small-molecule choice. It hides some real drugs that are heavier, and it hides peptides on purpose.
- Scaffold split is necessary and not sufficient.
- Predicted pIC50 is not a dose, not a selectivity claim, and not a clinical claim.
- BACE1 is in the panel specifically so that "potent and genetically interesting" is not treated as "therefore a drug."

## 14. Fifteen interview questions

These are written the way I would answer them in a Google Student Researcher or Software Engineering Intern interview. The numbers, when I quote a model, come from the latest `data/processed/metrics.json` in the repo, not from memory of a hoped-for score.

### 1. Walk me through the project in about a minute.

I built a pipeline that joins three public resources. Open Targets tells me which human genes are associated with diseases across neurology, metabolism, autoimmunity, infection, and cancer. ChEMBL tells me measured binding IC50s for small molecules against a curated set of those proteins. UniProt and the PDB add sequence length, domains, crystal resolution, and whether those crystals contain a ligand. I convert exact IC50s to pIC50, drop censored values, deduplicate parent compounds, and train linear and tree models to predict potency. The evaluation I trust is a Bemis-Murcko scaffold split, reported next to a random split so I can see how much near-analog leakage inflated the easy number. A Streamlit app lets someone browse the evidence and try a SMILES string. The demo model is fit after the benchmark, and the README says that.

### 2. Why is this not an oncology project with extra diseases pasted on?

Because the failure modes differ by area. EGFR has a huge, drug-like IC50 set and approved small molecules. PCSK9 and TNF are genetically or clinically important and often undruggable with classic oral molecules, so the small-molecule table gets thin. BACE1 has the opposite problem: lots of potent compounds and a painful clinical history. If I only showed lung cancer, a recruiter would learn that I can download EGFR. The heatmap and the funnel are there to show I looked at the cases where the data refuses to be neat.

### 3. What is an Open Targets association score, in one breath?

It is a 0-to-1 summary of how strongly that gene is connected to that disease in their evidence model. It is not a p-value, and it is not a probability that a drug will work. I also keep the pieces: genetic association, somatic mutation, literature, and the clinical / known-drug score. A gene can have a high literature score and a weak genetic score. I would not treat those as the same kind of support.

### 4. Why pIC50 instead of IC50 in nanomolar?

IC50 is a concentration. The useful range covers several orders of magnitude, so raw nanomolar error is dominated by the weak compounds. pIC50 is the negative log, so an error of 1 is a tenfold error everywhere on the scale. It matches how medicinal chemists compare potency. The conversion in this repo is `9 - log10(IC50_nM)`, and a unit test checks that 10 nM becomes 8 and 1 µM becomes 6.

### 5. Why drop `>` and `<` values instead of a fancier censored model?

Because I would rather have a smaller clean label than a larger label full of fake point values. "> 10 µM" means "not observed by the top dose." Putting 10 µM in the training column invents precision. A tobit-style or survival model is a legitimate extension. I did not implement it, so I do not claim it.

### 6. Explain a Bemis-Murcko scaffold to a biologist.

Think of the scaffold as the molecular chassis: the rings and the linkers between them, with the decorating side chains removed. Methyl, fluoro, and methoxy groups come off. What remains is the core a chemist would say "this series shares." Compounds that share a core are not independent experiments for a machine-learning test. They are cousins.

### 7. Why is a random split a bad default for molecules?

Because the cousins get separated at random. The model sees the family in training and is graded on another member of the family. Accuracy then measures "can you recognize this series?" more than "can you predict a new series?" That is a real skill, and it is not the skill I want the headline number to claim. I still report the random split, so the size of the flattery is visible.

### 8. How did you measure scaffold leakage?

I count scaffolds that appear in both train and test. The scaffold split must show zero. The random split usually shows many. Then I subtract metrics. RMSE gap is scaffold minus random. AUC gap is random minus scaffold. Positive gaps mean the random split looked better. Both the counts and the gaps are fields in `metrics.json`, computed in `odtp/model.py`, not typed into the README by hand.

### 9. Which metric would you defend, and which one would you distrust?

I defend scaffold-split RMSE against the mean baseline, plus Spearman, because potency is a number and the rank order matters to a chemist. I distrust a lone ROC-AUC on this data. Published IC50 sets are full of actives, and ROC can stay high while the model is useless at the decision you care about. I pair it with PR-AUC and with the prevalence baseline so the base rate is on the page.

### 10. ROC-AUC versus PR-AUC when most compounds are active?

ROC-AUC treats false positives and false negatives through the ranking of scores. If 80% of rows are active, a model can look excellent on ROC and still bury you in false alarms, or it can be no better than guessing "active." PR-AUC looks at precision as you walk down the ranked list. The no-skill line is the positive rate, not 0.5. That is why the prevalence baseline's PR-AUC is the test-set active fraction.

### 11. How does the HTTP client behave when an API is rude?

It pauses between network calls so we are not bursting four scientific APIs. Cache hits skip the pause. Status 429 and the usual 5xx codes, and connection errors, retry with backoff, and they honor a numeric `Retry-After` when one is present. Successful JSON is written under `data/cache/` with a hash key. Errors are not cached, including a GraphQL response that contains `errors` and no `data`. A hard 404 raises immediately. Tests script a session object, so this is checked without the network.

### 12. How would you test this in CI without live APIs?

The same way the suite is written now. Parsers take dictionaries shaped like the real responses I inspected. The dataset builder accepts client objects, so a test passes fakes that return a few IC50 rows, including rows that must be deleted. The model test builds synthetic matrices with repeated scaffolds and checks split invariants and the presence of metrics. CI installs the package and runs Ruff and pytest on Python 3.10 and 3.12. I do not want a green badge that depends on ChEMBL being awake.

### 13. Where do Lipinski's rules mislead you?

They describe a slice of oral drugs from the 1990s. They are a poor veto for many antibiotics, for some CNS drugs where the cutoffs shift, and for anything that works by covering a protein-protein surface. In this project TNF and some incretin ligands are the teaching examples: important targets, unhappy rule-of-five profiles, and sometimes no small-molecule training rows after the weight filter. I report the violation counts. I do not drop a disease because its ligands are large.

### 14. A teammate wants to ship the random-split model because the AUC is higher. What do you say?

I show them the shared-scaffold count and the gap. The higher AUC is what you get when the test set is full of analogs of the training set. Shipping that number as "expected performance on a new series" would be reporting the leak. I would ship the scaffold-split number, keep the random-split number as a diagnostic, and if we need a still harder test I would add a time split or a similarity threshold. I would not delete the random-split result. Hiding the comparison is how the leak survives review.

### 15. What are the limits, and what would you do next?

The label is a multi-lab binding IC50, sampled, capped, and restricted to drug-like weights. The model has no cell permeability, no selectivity panel, no metabolism, and no clinical outcome. Association is not causality. I would next add a time split, report per-target RMSE so EGFR cannot hide a weak PCSK9 model, and try a censored-data model so the `>` rows are used as bounds instead of discarded. I would not add a heavier network until the scaffold-split error clearly beats a fingerprint baseline by enough to matter to a chemist, which is on the order of a few tenths of a pIC50 unit, not a prettier ROC curve.

## 15. A last note on tone

If an interviewer asks for the AUC and I do not remember it, the correct move is to open `metrics.json` or the README table and read it. Inventing a metric in the room is worse than saying "the scaffold-split RMSE is the number I stand behind, and I will read it from the artifact." The code was written so that artifact is produced by the training command and rendered by `odtp/reports.py`.
