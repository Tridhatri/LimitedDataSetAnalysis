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
