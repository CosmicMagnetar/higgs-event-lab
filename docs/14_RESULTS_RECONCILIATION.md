# Results Reconciliation: 250k Subset vs 1M Benchmark

> **Owner:** Saumya Mishra (documentation and QA). **Decision owner:** Krishna. **Technical owners:** Aditya Rana (data, classical models), Rachit Gupta (neural network).
> **Status: OPEN.** Written during the final-push review (2026-10-26). Do not quote a headline result in any report or slide until the decision in section 5 is recorded here.
> **Rule:** this note states what the saved files say. It does not explain the difference. Explanations are listed as hypotheses to test (section 4).

---

## 1. Why this note exists

The repository holds two sets of results that disagree on the headline number:

- **250k subset** (`outputs/*/metrics.json`, `outputs/experiments.json`): models with default or lightly set configurations, trained on the 250,000-row subset, seed 42, split 70/15/15.
- **1M benchmark** (`outputs/full_colab_results.json`): models with tuned or wider configurations, trained on a 1,000,000-row sample, seed 42, split 70/15/15.

`reports/01` quotes the 1M numbers. `reports/02` quotes the 250k MLP number. The final report must use one headline result, and it must say which protocol produced it.

## 2. What the saved files say

All values are test-partition metrics copied from the saved files. Feature group is shown where the file records it.

### 2.1 250k subset (per-run `metrics.json`)

| Model | Feature group | Test ROC-AUC | Test PR-AUC | Brier | Test accuracy | Training time (s) |
|---|---|---|---|---|---|---|
| Logistic Regression | low_level | 0.7476 | 0.7622 | 0.2035 | 0.681 | see run metadata |
| Logistic Regression | high_level | 0.7661 | 0.7427 | 0.1945 | 0.698 | see run metadata |
| Logistic Regression | all | 0.8369 | 0.8377 | 0.1639 | 0.755 | see run metadata |
| XGBoost | low_level | 0.8152 | 0.8229 | 0.1714 | 0.749 | see run metadata |
| XGBoost | high_level | 0.8379 | 0.8243 | 0.1588 | 0.776 | see run metadata |
| XGBoost | all | 0.8924 | 0.8897 | 0.1252 | 0.818 | see run metadata |
| MLP (compact) | low_level | 0.8000 | 0.8097 | 0.1824 | 0.728 | 10.52 (all group only; see run metadata for others) |
| MLP (compact) | high_level | 0.9043 | 0.8888 | 0.1184 | 0.824 | see run metadata |
| MLP (compact) | all | **0.9406** | 0.9372 | 0.0945 | **0.865** | 10.52 |

### 2.2 1M benchmark (`outputs/full_colab_results.json`)

| Model | Test ROC-AUC | Test PR-AUC | Brier | Test accuracy | Training time (s) |
|---|---|---|---|---|---|
| Logistic Regression (tuned) | 0.6834 | 0.6832 | 0.2233 | 0.6406 | 412.03 (see BUG-018) |
| XGBoost (tuned) | 0.8271 | 0.8415 | 0.1693 | 0.7451 | 412.03 (see BUG-018) |
| Wide MLP, 256-128-64 (tuned) | **0.8399** | 0.8539 | 0.1630 | 0.7565 | 412.03 (see BUG-018) |

The 1M file does not record the feature group, so it is not known whether these runs used all 28 features. This must be confirmed (section 4, test T6).

## 3. What stands out

1. **Every model family is lower at 1M.** Logistic Regression falls from 0.8369 to 0.6834 (-0.15). XGBoost falls from 0.8924 to 0.8271 (-0.07). The MLP falls from 0.9406 to 0.8399 (-0.10). A consistent drop across three different families points to something that differs between the two pipelines (data, sampling, column mapping or preprocessing), not to the choice of model.
2. **Tuning should not lower Logistic Regression.** The 1M logistic model is described as "tuned", yet it scores 0.15 lower than the untuned 250k model on the same feature set. Either the tuned configuration is weaker, or the 1M data are different.
3. **The 250k numbers are high for this benchmark.** The MLP's 0.9406 ROC-AUC and 0.865 accuracy are above what we recall from the original study (R3) for this benchmark. This is not proof of a fault, but it is the pattern that leakage would produce. If leakage is present in the 250k pipeline, it would also inflate XGBoost and Logistic Regression, which fits observation 1. <!-- allow-metric -->
4. **Training time is copied.** All three 1M models report 412.03 seconds (BUG-018), so no per-model timing can be taken from that file.
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

## 7. Closure record

| Item | Result | Date | By |
|---|---|---|---|
| T1 | Data leakage check: train, validation and test split overlap verification | 2026-10-27 | Aditya |
| T2 | Sampling method check: seeded-random versus first-N sampling comparison | 2026-10-27 | Aditya |
| T3 | Feature column names, ordering and mapping verification | 2026-10-27 | Aditya |
| T4 | MLP benchmark reproducibility check on a clean run | 2026-10-27 | Rachit |
| T5 | Comparison of 250k results using the 1M benchmark pipeline | 2026-10-27 | Rachit with Krishna |
| T6 | Verification of feature groups, hyperparameter settings, random seeds and split sizes | 2026-10-27 | Aditya, Rachit |
| T7 | Saved MLP predictions checked against reported accuracy and ROC-AUC metrics | 2026-10-27 | Rachit |
| Decision on headline (section 5.1) | **Pending:** awaiting final decision | 2026-10-27 | Krishna |
