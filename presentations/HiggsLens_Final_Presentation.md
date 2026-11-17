% HiggsLens
% Machine-learning classification of simulated HIGGS events
% Final release-benchmark presentation

# HiggsLens: the question

- Binary classification of simulated HIGGS signal and background events.
- 21 low-level kinematic features; 7 high-level invariant-mass features; 28 combined.
- Compare Logistic Regression, XGBoost, compact MLP, deep MLP and residual MLP.
- Simulated Monte Carlo benchmark only; no particle-discovery or real-detector claim.

# Dataset and experimental design

- UCI HIGGS dataset: approximately 11 million simulated events.
- Exact duplicate events removed before sampling: **278,698**.
- Nested sample budgets: 250k, 1M and 2.5M; seed 42.
- Each budget: 70% train / 15% validation / 15% test.
- Same release script, model/feature matrix and split protocol at each budget.
- StandardScaler is fit on training rows only; validation data is used for model selection and early stopping.

# End-to-end pipeline

1. Download and validate the raw UCI data.
2. Remove exact duplicate events before sample extraction.
3. Build nested seeded samples and disjoint train/validation/test partitions.
4. Fit preprocessing on training data only.
5. Train five model families on three feature groups.
6. Save metrics and predictions; the dashboard reads artifacts and does not train at startup.

# Models compared

**Classical baselines**
- Logistic Regression: L-BFGS with training-only standardisation.
- XGBoost: depth 8, up to 3,000 trees, early stopping on validation loss.

**Neural networks**
- Compact MLP: 64 → 32, a shallow mid-semester control.
- Deep MLP: five hidden layers × 300 units.
- Residual MLP: four residual blocks of width 256.

# All-feature benchmark: test ROC-AUC

| Model | 250k | 1M | 2.5M |
|---|---:|---:|---:|
| Logistic Regression | 0.6864 | 0.6838 | 0.6842 |
| XGBoost | 0.8188 | 0.8302 | 0.8382 |
| Compact MLP (64–32) | 0.8165 | 0.8230 | 0.8255 |
| Deep MLP (5×300) | 0.8271 | 0.8474 | 0.8591 |
| Residual MLP | 0.8339 | 0.8506 | 0.8613 |

- Both deep networks outperform XGBoost at every budget.
- Best deep-network advantage over XGBoost: +0.015 at 250k; +0.020 at 1M; +0.023 at 2.5M.
- At 2.5M, Residual MLP: **0.8613**; XGBoost: **0.8382**.

# Low-level-only benchmark: test ROC-AUC

| Model | 250k | 1M | 2.5M |
|---|---:|---:|---:|
| Logistic Regression | 0.5962 | 0.5939 | 0.5938 |
| XGBoost | 0.7223 | 0.7478 | 0.7669 |
| Compact MLP (64–32) | 0.7250 | 0.7382 | 0.7448 |
| Deep MLP (5×300) | 0.7561 | 0.8141 | 0.8396 |
| Residual MLP | 0.7716 | 0.8203 | 0.8532 |

- At 2.5M, Residual MLP on 21 low-level features scores **0.8532**.
- This exceeds XGBoost on all 28 features (**0.8382**) in this benchmark.
- This is evidence of learned feature interactions in this simulated dataset, not a physics-discovery claim.

# Feature groups at 2.5M

| Model | Low-level (21) | High-level (7) | All (28) |
|---|---:|---:|---:|
| Logistic Regression | 0.5938 | 0.6471 | 0.6842 |
| XGBoost | 0.7669 | 0.7921 | 0.8382 |
| Compact MLP (64–32) | 0.7448 | 0.7882 | 0.8255 |
| Deep MLP (5×300) | 0.8396 | 0.7987 | 0.8591 |
| Residual MLP | 0.8532 | 0.7989 | 0.8613 |

- Hand-engineered invariant masses help classical models and the compact MLP.
- Deep models recover more discriminative structure from raw kinematics as the sample grows.
- One seed per budget: do not over-interpret small differences.

# Calibration and compute trade-off

| Model | Brier | ECE | Training time (s) |
|---|---:|---:|---:|
| Logistic Regression | 0.2222 | 0.0231 | 5.5 |
| XGBoost | 0.1632 | 0.0033 | 28.6 |
| Compact MLP | 0.1705 | 0.0286 | 124.2 |
| Deep MLP (5×300) | 0.1524 | 0.0156 | 425.5 |
| Residual MLP | 0.1513 | 0.0184 | 493.7 |

- XGBoost is much faster than the deep networks in this run.
- Lower Brier/ECE indicates better calibration; this is distinct from discrimination.
- Training times are hardware- and run-specific (Kaggle Tesla T4).

# Reproducibility and release checks

- Artifact contract: `higgs-experiment-v1`; split protocol: `higgs-split-v1`.

| Budget | Train | Validation | Test | Indices disjoint | Shared train/test events |
|---|---:|---:|---:|---|---:|
| 250k | 175,000 | 37,500 | 37,500 | true | 0 |
| 1M | 700,000 | 150,000 | 150,000 | true | 0 |
| 2.5M | 1,750,000 | 375,000 | 375,000 | true | 0 |

- Two CPU compact-MLP trainings with seed 42 produced identical test probabilities in the release determinism check.
- Raw data and model checkpoints are not committed; regenerate with `scripts/release_benchmark.py`.

# Limitations and conclusion

- Simulated Monte Carlo data; class balance is not representative of a rare-event search.
- One seed per budget: cross-seed variance and statistical uncertainty are not measured.
- Some deep runs reached the 60-epoch cap; further training may change the ranking.
- Release metrics do not serialize complete operating-point thresholds; none are quoted here.
- Deeper networks improve discrimination on this benchmark as sample size grows; XGBoost remains much cheaper and better calibrated in these runs.
