# Enhancing Limited Dataset Analysis via Synthetic Data Generation Using the Synthetic Data Vault

Research paper and supporting notebooks evaluating whether synthetic data generated using the [Synthetic Data Vault (SDV)](https://docs.sdv.dev/sdv) can improve machine learning model performance on limited datasets.

## Dataset

[DS2OS Traffic Traces](https://www.kaggle.com/datasets/francoisxa/ds2ostraffictraces) — IoT cybersecurity dataset with 357,952 rows and 13 columns. Severely imbalanced: ~97% normal traffic, ~3% anomalous (7 attack types).

## Approaches

| Approach | Dataset Size | Strategy | Result |
|---|---|---|---|
| 1. Naive Concatenation | 50,000 rows | Concatenate real + synthetic data directly | Degrades performance |
| 2. Smart Balancing | 50,000 rows | Per-class synthesis, equalize all classes to 5,000 | Baseline already near-perfect; no practical gain |
| 3. True Limited Data | 1,000 rows | Per-class synthesis, equalize all classes to 100 | Scan recall 0% → 100%, macro recall +24.5% |

## Key Finding

Synthetic data augmentation is most valuable when the dataset is genuinely limited. GaussianCopulaSynthesizer performs best in data-scarce scenarios (macro recall 0.8000 → 0.9959).

## Structure

- `Limited DataSet Analysis using SDV.docx` — Research paper
- `notebook/Notebook-1.ipynb` — Main notebook (Approaches 1 & 2)
- `notebook/Approach3-LimitedDataSimulation.ipynb` — Approach 3 notebook
- `notebook/diagrams/` — Generated charts
- `notebook/cleaned_data.csv` — Preprocessed 50k-row subset
- `changes.md` — Detailed changelog of all fixes and additions
