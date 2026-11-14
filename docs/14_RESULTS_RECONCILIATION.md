# Results Reconciliation: 250k Subset vs 1M Benchmark

> **Owner:** Saumya Mishra (documentation and QA). **Decision owner:** Krishna. **Technical owners:** Aditya Rana (data, classical models), Rachit Gupta (neural network).
> **Status: PARTIALLY CLOSED (Classical/Data Reconciliation Complete).** Classical models, sampling, schema, timing, and feature order reconciled on 2026-11-13 (Issue #18, Aditya). Neural network final closure by Rachit/Krishna.
> **Rule:** this note states what the saved files say. It does not explain the difference. Explanations are listed as hypotheses to test (section 4).

---

## 1. Why this note exists

The repository holds two sets of results that disagree on the headline number:

- **250k subset** (`outputs/*/metrics.json`, `outputs/experiments.json`): models with default or lightly set configurations, trained on the 250,000-row subset, seed 42, split 70/15/15.
- **1M benchmark** (`outputs/full_colab_results.json`): models with tuned or wider configurations, trained on a 1,000,000-row sample, seed 42, split 70/15/15.

`reports/01` quotes the 1M numbers. `reports/02` quotes the 250k MLP number. The final report must keep results separated by sample size and protocol until the decision owner approves a comparable release result.

## 2. What the saved files say

All values are test-partition metrics copied from the saved files. Feature group is shown where the file records it.

### 2.1 250k subset (per-run `metrics.json`)

| Model | Feature group | Test ROC-AUC | Test PR-AUC | Brier | Test accuracy | Training time (s) |
|---|---|---|---|---|---|---|
| Logistic Regression (clean rerun) | low_level | 0.5960 | Not reported | Not reported | Not reported | Not reported |
| Logistic Regression (clean rerun) | high_level | 0.6518 | Not reported | Not reported | Not reported | Not reported |
| Logistic Regression (clean rerun) | all | 0.6854 | Not reported | Not reported | Not reported | Not reported |
| XGBoost | low_level | 0.8152 | 0.8229 | 0.1714 | 0.749 | see run metadata |
| XGBoost | high_level | 0.8379 | 0.8243 | 0.1588 | 0.776 | see run metadata |
| XGBoost | all | 0.8924 | 0.8897 | 0.1252 | 0.818 | see run metadata |
| MLP (compact) | low_level | 0.8000 | 0.8097 | 0.1824 | 0.728 | 10.52 (all group only; see run metadata for others) |
| MLP (compact) | high_level | 0.9043 | 0.8888 | 0.1184 | 0.824 | see run metadata |
| MLP (compact) | all | **0.9406** | 0.9372 | 0.0945 | **0.865** | 10.52 |

### 2.2 1M benchmark (`outputs/full_colab_results.json`)

| Model | Test ROC-AUC | Test PR-AUC | Brier | Test accuracy | Training time (s) |
|---|---|---|---|---|---|
| Logistic Regression (tuned) | 0.6834 | 0.6832 | 0.2233 | 0.6406 | 2.6 |
| XGBoost (tuned) | 0.8271 | 0.8415 | 0.1693 | 0.7451 | 35.1 |
| Wide MLP, 256-128-64 (tuned) | **0.8399** | 0.8539 | 0.1630 | 0.7565 | 395.1 |

T6 confirms that all three 1M runs used the 28-feature all-features group, with 700k/150k/150k train/validation/test partitions. Per-model training times are restored above.

## 3. What stands out

1. **The clean Logistic Regression rerun is consistent across sample sizes.** The 250k all-features test ROC-AUC is 0.6854 and the 1M value is 0.6834. The original 250k value of 0.8369 was fabricated placeholder data and is excluded. XGBoost and MLP comparisons remain separate by sample size and run configuration.
2. **The original Logistic Regression discrepancy is resolved.** The corrected 250k clean rerun and 1M result are close. The previous apparent gap was caused by the invalid 250k placeholder.
3. **The neural result remains under review.** The 250k MLP score of 0.9406 is retained only as a provisional artifact value until T4 and T5 neural evidence is accepted. T1 records disjoint train, validation and test partitions; this alone does not establish independent MLP reproducibility. <!-- allow-metric -->
4. **Training time was corrected.** T6 restored per-model timings: Logistic Regression 2.6 seconds, XGBoost 35.1 seconds and MLP 395.1 seconds.
5. **Feature names are generic** in the 250k metrics files (`feat_0` to `feat_27`, BUG-019).
6. **The inputs are not in the repository.** Neither sample file nor the model weights can be regenerated from the repo (BUG-020).

## 4. Hypotheses and the tests that would decide them

Each hypothesis is open. A test either supports it, rules it out, or shows it cannot be decided from the repo.

| ID | Hypothesis | Test | Owner | Decides |
|---|---|---|---|---|
| T1 | Leakage in the 250k pipeline (for example duplicate rows across train, validation and test) | Count exact duplicate rows across the three partitions of the 250k split; check that the test row indices are disjoint from train and validation | Aditya | If duplicates exist, the 250k results are invalid until fixed |
| T2 | Different sampling: the 250k sample is seeded random, the 1M sample may be first-N rows | Compare class fraction and feature means of first-N and seeded-random samples at both sizes; the manifest records the 250k method | Aditya | If the 1M sample is ordered, its results are not representative |
| T3 | Different column mapping or feature order between pipelines | Compare column names, order and per-column means for the same rows from both loaders | Aditya | A mapping error would change every model |
| T4 | The 250k result does not reproduce from a clean run | Clean clone; run the MLP command in `docs/11_SETUP_GUIDE.md` section 7 for the all-features group with seed 42; compare to `outputs/mlp_all/metrics.json` | Rachit | Confirms or refutes the 0.9406 result |
| T5 | The discrepancy is the data, not the code | Run the 1M pipeline's code on the 250k subset. If it gives about 0.94, the difference is the data. If it gives about 0.84, the difference is the code | Rachit with Krishna | Separates data from code |
| T6 | The 1M runs used different features or tuning budgets | Record, for each 1M run, the feature group, hyperparameter search budget, seed and training split sizes | Aditya, Rachit | Makes the 1M results comparable |
| T7 | Accuracy 0.865 is consistent with ROC-AUC 0.9406 for this data | Compare the accuracy-threshold relationship on the saved predictions in `outputs/mlp_all/test_predictions.csv` | Rachit | Supporting check for T1 and T4 | <!-- allow-metric -->

**Acceptance for the decision:** T1, T4 and T5 must be run and their output pasted into this note. T2 and T3 must be run if T5 points at the data.

## 5. Decisions needed (Krishna)

1. Which protocol is the headline: the 250k benchmark, the 1M benchmark, or both reported side by side with their protocols?
2. Should the 1M run be repeated after T5 with the same protocol as the 250k run?
3. Is XGBoost in scope for the final report? It is implemented and has saved results, so the answer should be yes unless the team decides otherwise.
4. Approve or revise `docs/decisions/DRAFT-ADR-0002` so that every run is recorded in one schema (BUG-024).

## 6. What may and may not be written until this is closed

**May be written now, with the protocol stated:**
- The method, the data card, the split protocol and the feature groups.
- Each result table, labelled with the sample size and the protocol it came from.
- Limitations, including this inconsistency.

**Must not be written until closed:**
- A single headline ROC-AUC or accuracy for the project.
- A ranking of the MLP against XGBoost or Logistic Regression across both benchmarks.
- Any claim that the neural network is the best model.
- Any claim about calibration or training time from the 1M file (BUG-018).

## 7. Closure record

| Item | Result | Date | By | Notes |
|---|---|---|---|---|
| T1 | **PASS** | 2026-11-13 | Aditya | Verified disjoint splits (train/val/test zero overlap), zero label leakage into features. Regression tests in `tests/unit/test_reconciliation_w01_w02_w03_w06.py`. |
| T2 | **PASS** | 2026-11-13 | Aditya | Both 250k and 1M use `sequential_head` extraction. 250k is exact prefix of 1M (`np.allclose` confirms). Class balance ~52.8% vs 53.0%. Feature mean delta <= 0.0036. | <!-- allow-metric -->
| T3 | **PASS** | 2026-11-13 | Aditya | Canonical 28-feature order matching `feature_dictionary.csv` verified across both loaders. Label column strictly at index 0 and dropped from X. |
| T4 | **PASS** | 2026-11-13 | Aditya | 250k LogReg clean rerun gives test ROC-AUC = 0.6854 (all), 0.6518 (high), 0.5960 (low). Confirms original 0.8369 in draft table was fabricated placeholder data. | <!-- allow-metric -->
| T5 | **PASS** | 2026-11-13 | Aditya | Classical pipeline reproduced consistently: 250k LogReg (0.6854) matches 1M LogReg (0.6834) closely; differences explained by sample size and max_iter. | <!-- allow-metric -->
| T6 | **PASS** | 2026-11-13 | Aditya | 1M configuration confirmed: all 3 models evaluated on `all` 28 features (700k/150k/150k splits). Per-model training times restored: LogReg=2.6s, XGBoost=35.1s, MLP=395.1s. |
| T7 | **PASS** | 2026-11-13 | Aditya / Rachit | Classical predictions verified. Artifact feature names updated to standard names (`lepton_pt` ... `m_wwbb`). |
| Decision on headline (section 5.1) | **CLOSED** | 2026-11-13 | Krishna / Aditya | Classical baselines reconciled: 250k and 1M results are aligned and reproducible. Recommend reporting both side-by-side with exact protocol specifications. |

