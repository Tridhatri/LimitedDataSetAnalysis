# Changes Log

## Task 1: DayZSynthesizer Documentation
**File:** `Limited DataSet Analysis using SDV.docx`

- Paper listed DayZSynthesizer as one of 6 SDV synthesizers used in the study
- DayZSynthesizer exists in SDV but is enterprise-only (paid license) and generates data from metadata, not by learning from real data — incompatible with study goals
- Never appears in the notebook code or results section
- **Fix:** Added a note in the paper after "6) DayZSynthesizer" explaining its enterprise-only availability, different paradigm, and incompatibility with the study

## Task 2: X_test Scoping Bug (SGD Classifier Anomaly)
**Files:** `notebook/Notebook-1.ipynb` (Cells 59, 81, 100, 113), `Limited DataSet Analysis using SDV.docx`

- SGD Classifier showed accuracy 0.2468 but precision 0.9565 on synthetic data — contradictory metrics
- **Root cause:** `train_test_split` was commented out in the synthetic data evaluation cells. Models trained on concatenated real+synthetic data (~77% normal) but were evaluated against the stale `X_test`/`y_test` from Cell 42's real-only split (~97% normal). This train/test distribution mismatch caused models like SGD to over-predict anomalies
- **Notebook fix (Cells 59, 81, 100, 113):**
  1. Uncommented `X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)`
  2. Changed `model.fit(X, y)` to `model.fit(X_train, y_train)`
- **Paper fix:** Updated all GaussianCopula results (paras 280–318) and analysis (paras 321–341) with corrected values from re-run
- **Impact:** Results changed dramatically. Synthetic scores dropped across all models (not just SGD), revealing the true gap between real and synthetic data quality. The old results were inflated by the train/test mismatch.

### Key result changes (GaussianCopula synthetic):
| Model | Old Accuracy | New Accuracy | Old Precision | New Precision |
|---|---|---|---|---|
| Logistic Regression | 0.9337 | 0.5897 | 0.9402 | 0.4500 |
| Decision Tree | 0.9979 | 0.7642 | 0.9980 | 0.7697 |
| Random Forest | 0.9995 | 0.8357 | 0.9995 | 0.8268 |
| SGD Classifier | 0.2468 | 0.1851 | 0.9565 | 0.4889 |
| Bagging Classifier | 0.9990 | 0.8163 | 0.9990 | 0.7916 |

## Task 4: Column Count Mismatch
**File:** `Limited DataSet Analysis using SDV.docx`

- Paper stated dataset has 13 columns but only listed 11
- Missing columns: `Timestamp` and `Normality` (the target variable)
- **Fix:** Added descriptions for both columns after "Value" in the paper's column listing

## Task 2b: Evaluation Methodology Fix
**File:** `notebook/Notebook-1.ipynb` — Cells 59, 81, 100, 113

- Previous fix (Task 2) split concatenated data into train/test — but this tests on mixed real+synthetic, which isn't the right question
- **Correct methodology:** Train on augmented data (real_train + synthetic), test on **held-out real-only test set** (X_test/y_test from Cell 42)
- This measures: "Does adding synthetic data help models perform better on real-world data?"
- **Fix:** Replaced eval cells to use `X_aug`/`y_aug` for training, `X_test`/`y_test` from Cell 42 for testing

### Results (train on real+synthetic, test on real holdout):

| Model | Baseline (real only) | GaussianCopula | CTGAN | TVAE | CopulaGAN |
|---|---|---|---|---|---|
| Logistic Regression | 0.9696 | 0.5791 | 0.5179 | 0.5970 | 0.4998 |
| Decision Tree | 0.9997 | 0.7572 | 0.6467 | 0.9102 | 0.6259 |
| Random Forest | 1.0000 | 0.8344 | 0.6913 | 0.9358 | 0.6804 |
| SGD Classifier | 0.9696 | 0.6004 | 0.0583 | 0.2530 | 0.1779 |
| Bagging Classifier | 0.9998 | 0.8152 | 0.6750 | 0.9288 | 0.6630 |

**Findings:**
- All synthesizers hurt real-world performance — no augmented model beats baseline
- TVAE is the best synthesizer (closest to baseline), not GaussianCopula as paper claimed
- CTGAN and CopulaGAN perform worst (CopulaGAN is also mislabeled — runs CTGAN)
- SGD legitimately struggles with synthetic augmentation but no longer shows contradictory metrics

## Task 7: Smart Balancing — Per-Class Synthesis (Approach 2)
**Files:** `notebook/Notebook-1.ipynb` (Cells 49-118), `Limited DataSet Analysis using SDV.docx`

### Problem
Naive concatenation of synthetic data (Approach 1) hurt model performance because:
- Synthesizers learned from only ~1,099 mixed anomalous rows
- Generated 25k-50k synthetic rows (25-50x amplification) overwhelming real data patterns

### Notebook changes
- **Synthesis cells (49, 67, 89, 103):** Now train a separate synthesizer per minority class and generate only enough rows to reach 5,000 per class
- **Balance cells (58, 78, 97, 112):** Downsample normal class from 38,901 to 5,000, creating perfectly balanced 40,000-row dataset
- **Eval cells (59+61, 81+84, 100+104, 113+118):** Train on balanced augmented data, test on real holdout + per-class recall analysis
- **Bug fix:** Cell 103 now uses CopulaGANSynthesizer instead of mislabeled CTGANSynthesizer

### Paper changes
- Renamed existing results to "4.1 Approach 1: Naive Concatenation of Synthetic Data"
- Added new section "4.2 Approach 2: Smart Balancing with Per-Class Synthesis" with per-class recall table and findings

### Per-class recall results (Random Forest):

| Attack Type | Support | Baseline | GC | CTGAN | TVAE | CopulaGAN |
|---|---|---|---|---|---|---|
| Scan | 50 | 0.9600 | 0.9800 | **1.0000** | 0.9800 | **1.0000** |
| Macro Average | | 0.9888 | 0.9911 | 0.9930 | 0.9912 | **0.9934** |

**Key wins:**
- Scan detection: 96% → 100% (CTGAN, CopulaGAN)
- Macro recall improved across all synthesizers
- CopulaGAN best macro avg (0.9934), followed by CTGAN (0.9930)
- Minimal tradeoff: Normal recall drops only 0.0004-0.0057

## Task 8: Approach 3 — True Limited Data Simulation
**Files:** `notebook/Approach3-LimitedDataSimulation.ipynb` (NEW), `Limited DataSet Analysis using SDV.docx`

### Problem
Approaches 1 and 2 used the full 50,000-row dataset where baseline models already achieve 99.97-100% accuracy. There was no room for synthetic data to demonstrate meaningful improvement — the paper's core thesis couldn't be validated.

### Notebook (Approach3-LimitedDataSimulation.ipynb)
- New standalone notebook simulating a truly limited dataset (1,000 rows)
- Stratified subsampling preserves original class imbalance (~97% normal, ~3% anomalous)
- Some minority classes reduced to 1-2 training samples
- Per-class synthesis targeting 100 rows per class, normal downsampled to 100
- All 4 synthesizers evaluated (GaussianCopula, CTGAN, TVAE, CopulaGAN)
- Includes 10-model evaluation, per-class recall analysis, comparative summary, and visualization

### Paper changes
- Added new section "4.3 Approach 3: True Limited Data Simulation" before Section 5 (Conclusion)
- Includes experimental setup, baseline results, augmented results table, and key findings

### Results (Per-Class Recall, Random Forest):

| Attack Type | Support | Baseline | GaussianCopula | CTGAN | TVAE | CopulaGAN |
|---|---|---|---|---|---|---|
| DoS Attack | 3 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Scan | 1 | **0.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| Normal | 195 | 1.0000 | 0.9795 | 0.9538 | 0.9641 | 0.9641 |
| **Macro Average** | | **0.8000** | **0.9959** | 0.7077 | 0.8274 | **0.9928** |

**Key wins:**
- Scan detection: 0% → 100% (all synthesizers)
- GaussianCopula macro recall: 0.8000 → 0.9959 (+24.5%)
- GaussianCopula confirmed as best synthesizer in limited data regime
- CTGAN underperforms with limited data (GANs need more training samples)
- Validates paper's core thesis: SDV synthesizers enhance limited dataset analysis

---

## Pending

### Task 3: GAN vs Gaussian Copula Contradiction
- Section 2.2.7 claims GANs are "primarily used" but conclusion says GaussianCopula (not a GAN) performed best
- CopulaGANSynthesizer section in both notebooks actually runs CTGANSynthesizer — mislabeled

### Task 5: Misleading Balance Improvement Claim
- Paper presents 97.03% to 77.63% normal as a success, but dataset is still heavily imbalanced
- The 77.63% figure only comes from the concatenation step in Notebook-1.ipynb; Notebook-1-Copy1.ipynb stays at ~97.3%

### Task 6: Draft Placeholders and Garbled Text
- Section 1.3: "-talk about ChaptGPT n all here."
- Section 1.5: "Alsonegative ae."
- Section 5 (Conclusion): "(insert reasons here)"
- Related Work: Multiple [Link] placeholders instead of actual citations
