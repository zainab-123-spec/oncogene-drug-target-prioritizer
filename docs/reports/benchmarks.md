Numbers below are copied from the training run's `data/processed/metrics.json`.

### Regression (predict pIC50)

| Model | Split | RMSE | Pearson r | Spearman rho | ROC-AUC | PR-AUC |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| MeanBaseline | scaffold | 1.242 | — | — | — | — |
| MeanBaseline | random | 1.237 | — | — | — | — |
| Ridge | scaffold | 1.079 | 0.536 | 0.541 | 0.780 | 0.896 |
| Ridge | random | 0.972 | 0.628 | 0.634 | 0.827 | 0.922 |
| RandomForest | scaffold | 0.900 | 0.705 | 0.708 | 0.859 | 0.933 |
| RandomForest | random | 0.764 | 0.799 | 0.804 | 0.905 | 0.963 |
| HistGradientBoosting | scaffold | 0.900 | 0.690 | 0.686 | 0.846 | 0.929 |
| HistGradientBoosting | random | 0.764 | 0.786 | 0.788 | 0.892 | 0.957 |

### Classification (active if pIC50 ≥ 6)

RMSE and correlations are left blank for classifiers. ROC-AUC and PR-AUC use the model's ranking score (predicted probability, or the Ridge decision function).

| Model | Split | ROC-AUC | PR-AUC |
| --- | --- | ---: | ---: |
| PrevalenceBaseline | scaffold | 0.500 | 0.739 |
| PrevalenceBaseline | random | 0.500 | 0.740 |
| RidgeClassifier | scaffold | 0.794 | 0.902 |
| RidgeClassifier | random | 0.825 | 0.919 |
| RandomForest | scaffold | 0.872 | 0.948 |
| RandomForest | random | 0.902 | 0.957 |
| HistGradientBoosting | scaffold | 0.864 | 0.942 |
| HistGradientBoosting | random | 0.900 | 0.957 |

### Scaffold leakage gaps

RMSE gap = scaffold RMSE − random RMSE. AUC gap = random ROC-AUC − scaffold ROC-AUC. A positive gap means the random split was optimistic.

| Model | Task | RMSE gap | ROC-AUC gap | PR-AUC gap |
| --- | --- | ---: | ---: | ---: |
| HistGradientBoosting | classification | — | 0.036 | 0.014 |
| HistGradientBoosting | regression | 0.136 | 0.046 | 0.028 |
| MeanBaseline | regression | 0.005 | — | — |
| PrevalenceBaseline | classification | — | 0.000 | 0.001 |
| RandomForest | classification | — | 0.030 | 0.010 |
| RandomForest | regression | 0.135 | 0.046 | 0.030 |
| Ridge | regression | 0.107 | 0.047 | 0.026 |
| RidgeClassifier | classification | — | 0.031 | 0.017 |

### Split sizes

- **scaffold**: n_train=5360, n_test=1328, shared scaffolds=0, test active rate=0.739
- **random**: n_train=5350, n_test=1338, shared scaffolds=525, test active rate=0.740
