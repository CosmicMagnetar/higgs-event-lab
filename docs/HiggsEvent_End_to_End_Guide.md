# HiggsEvent: end-to-end explanation and viva guide

Reviewed against the workspace on 27 October 2026. Numeric results below come from `outputs/*/metrics.json` and were recomputed from the corresponding prediction CSVs. The full-data completion statement is supplied by the project team; it is separate from the locally imported metrics.

## 1. What are we actually building?

HiggsEvent is a **binary classification research workbench**. Given 28 numerical measurements describing one simulated collision event, it produces a score indicating how signal-like that event is. It compares different model families, checks their performance on held-out events, and displays the recorded evidence in Streamlit.

An event is one row, not one particle. The label is `1` for the benchmark signal and `0` for background. The UCI benchmark describes a hypothetical exotic Higgs-like signal versus a competing background process. We are not rediscovering the 125 GeV Higgs boson or analysing live LHC detector data.

The engineering outcome is a reproducible pipeline and dashboard. The scientific question is: **which models and representations separate these simulated classes well, and at what computational cost?**

## 2. Data sizes: why several numbers appear

| Experiment or resource | Events | Meaning |
|---|---:|---|
| Complete UCI source | 11,000,000 = 11 million = 1.1 crore | Complete benchmark dataset |
| Full-data experiment | 11,000,000 | Completion reported by the team; full-run result files still need importing locally |
| Current nine feature-group experiments | 250,000 = 2.5 lakh | Three models × three feature groups; these are the current dashboard comparisons |
| Saved EDA | 250,000 | Smaller dataset used to make distributions, summaries and correlations |
| Historical summary | 1,000,000 = 10 lakh | Older `outputs/full_colab_results.json`; its referenced prediction files now belong to a smaller run |

Using a smaller EDA sample does **not** mean the project never used the complete dataset. Conversely, downloading all rows does not by itself prove that a training job completed. Each experiment needs its own dataset path, count, split and result files.

“An experiment uses all 11 million events” means those events are partitioned. With a 70/15/15 protocol, the expected counts are **7,700,000 training, 1,650,000 validation and 1,650,000 test**. Only the training partition fits model weights. These are protocol-derived counts, not a recovered execution log.

## 3. What do the features mean?

- **21 low-level inputs:** lepton and jet transverse momenta, pseudorapidities, azimuthal angles, four b-tag values, and missing-energy magnitude/direction.
- **7 high-level inputs:** `m_jj`, `m_jjj`, `m_lv`, `m_jlv`, `m_bb`, `m_wbb`, `m_wwbb`. These are supplied reconstructed invariant-mass features, not new features invented by our project.
- **All 28 inputs:** both groups together.

Transverse momentum describes motion perpendicular to the beam. Pseudorapidity describes direction relative to the beam. Azimuth is the angle around it. Missing transverse energy indicates momentum imbalance. B-tag values encode information about jets associated with bottom quarks. Invariant-mass features combine measurements into quantities useful for recognising a decay pattern.

The public inputs are normalised benchmark quantities. A displayed mass value near 1 must not be interpreted directly as 1 GeV. Our additional `StandardScaler` fits the training partition of the selected experiment.

Feature-group experiments answer a useful question: can a model learn from basic kinematics, or does giving it the supplied physics-derived quantities help? In the current nine runs, **all > high-level > low-level for ROC-AUC**, for each family. That ordering is not true of every metric; Logistic Regression's high-level average precision is slightly below its low-level value.

## 4. End-to-end pipeline

1. **Obtain the data.** Notebooks download directly from UCI: `https://archive.ics.uci.edu/static/public/280/higgs.zip`. GitHub supplies project code, not the dataset.
2. **Check the input.** The data module checks 29 columns, numerical and finite values, and binary labels. Full-file row counting and checksums are distinct from validating every row's feature values.
3. **Choose the experiment input.** Current baseline notebooks read up to the configured row limit from the raw file; the full notebook passes the complete raw file to the tuned trainer. The older manifest says `seeded_random`, so do not assume it describes the current first-N-row notebook path.
4. **Separate features and labels.** `class_label` is the target, never an input feature.
5. **Split reproducibly.** `protocol.py` creates disjoint, stratified train/validation/test indices with seed 42. Stratification approximately preserves the class fraction in each partition. Equal seeds only give equal splits when row ordering and labels also match.
6. **Fit preprocessing on training data.** Standardisation is `(x - training_mean) / training_std`. Validation and test use these same training statistics. Fitting the scaler on all data would leak information.
7. **Train models.** Logistic Regression fits coefficients; XGBoost builds trees; the MLP learns layered nonlinear transformations.
8. **Use validation for decisions.** MLP early stopping watches validation ROC-AUC and restores the best weights. Classical operating thresholds target 50% validation signal efficiency. The compact MLP export currently does not save this operating-point block.
9. **Evaluate held-out events.** Compute ranking, probability-prediction and threshold metrics without fitting on test labels. The dashboard threshold slider is exploratory, not a new unbiased threshold-selection procedure.
10. **Export and inspect.** Save run metadata and test predictions. Streamlit loads the files and computes diagnostic plots; it does not train models at startup. Model Simulation loads available weights for inference on new inputs.

Chunked reading limits parsing memory, but `load_higgs_sample` concatenates selected chunks. The full trainer uses `pd.read_csv` for the complete selected file and materialises arrays. It is not an out-of-core or streaming training implementation.

## 5. Why Logistic Regression first?

Despite its name, Logistic Regression is a classifier. It computes a weighted sum `z = w·x + b`, then a sigmoid `p = 1 / (1 + exp(-z))`.

It is a sensible baseline because it is fast, regularised, relatively interpretable, and exposes how much information is available to a linear decision boundary. It also provides a low-cost check of labels, scaling, feature ordering and metrics.

Its limitation is that raw inputs enter additively: without explicit transformations it cannot freely express interactions such as a particular mass range combined with particular jet kinematics. More examples reduce estimation error but do not automatically remove this modelling limitation.

**We did not replace or discard Logistic Regression. We retained it and added nonlinear competitors.** This makes the comparison informative instead of assuming a more complicated model must be better.

## 6. Why XGBoost?

XGBoost is a gradient-boosted tree ensemble. Each new tree improves the current model by fitting information from the loss gradients, with regularisation and learning-rate shrinkage. A path through a tree can express interacting conditions: for example, a range of `m_bb` followed by a condition on missing energy.

It is particularly useful for structured tabular data. It captures nonlinear splits and interactions without manually writing all cross-features. Tree splits normally do not require standardisation, although the tuned full trainer supplies scaled arrays consistently.

The baseline settings are 150 trees, depth 6, learning rate 0.1, row/column subsampling 0.8, and histogram-based splitting. Depth limits complexity; subsampling helps regularisation; the learning rate controls each tree's contribution. The current full trainer has a separate, larger 600-tree configuration. Its `eval_set` monitors validation loss; there is no configured XGBoost early-stopping callback in that script.

## 7. Why a neural network too?

An MLP learns combinations of numerical features through several layers. It gives a different nonlinear approach from tree partitions, and lets us investigate learned representations of kinematic quantities.

The compact network is `D → 64 → 32 → 1`, where D is 21, 7 or 28. Each hidden block uses:

- **Linear layer:** learned weighted combinations.
- **BatchNorm:** normalises hidden activations during training; uses running statistics at inference.
- **ReLU:** introduces nonlinearity. Without nonlinear activations, stacked linear layers remain a linear mapping.
- **Dropout (0.2):** randomly removes hidden activations during training to discourage dependence on a few units; disabled in evaluation mode.

The output is a **logit**, not a probability until sigmoid is applied. `BCEWithLogitsLoss` combines sigmoid and binary cross-entropy numerically stably. AdamW updates weights and applies decoupled weight decay. Mini-batches of 1,024 events trade memory against computation. An epoch is one pass through training events.

The compact configuration allows 30 epochs, uses learning rate 0.001 and weight decay 0.0001, and stops after five epochs without sufficient validation improvement. All three saved compact runs contain 30 epochs; reaching the epoch limit is not proof of convergence or of failure to converge.

The wide model has hidden widths 256, 128, 64 and dropout 0.3. The current full script sets learning rate 0.0008, weight decay 0.00002 and at most 60 epochs. These are **current code settings**, not automatically the settings of an older saved run.

## 8. What do the metrics mean?

| Metric | Plain-language meaning | Important qualification |
|---|---|---|
| ROC-AUC | Probability that a random signal event outranks a random background event, with half-credit for ties | Ranking quality, not accuracy; 0.5 is chance ranking |
| Average precision (`pr_auc`) | Precision averaged over increases in recall | The code uses sklearn average precision, not trapezoidal PR integration; depends on prevalence |
| Accuracy | Fraction of correct labels at a particular threshold | Current saved accuracies use threshold 0.5 |
| Precision | Of events selected as signal, how many are signal? | Sensitive to signal prevalence |
| Recall / signal efficiency | Of actual signal events, how many are retained? | Raising the threshold generally reduces it |
| Background rejection | Fraction of background events removed: TN/(TN+FP) | Equal to 1−FPR; specify the threshold and achieved signal efficiency |
| Brier score | Mean of `(predicted_probability - label)^2` | Lower is better; reflects calibration and discrimination, not calibration alone |
| Calibration curve | Observed signal fraction versus mean score in bins | A model can rank well while its probabilities are poorly calibrated |

A threshold creates a decision from a continuous score. Lower thresholds normally retain more signal **and** more background. Higher thresholds remove more background **and** lose more signal. There is no universal best threshold independent of the use case.

For saved `xgboost_all`, the validation-selected threshold is approximately 0.6906. On test data it keeps **50.30% of signal** and rejects **89.17% of background**. Say “at a validation-targeted 50% signal efficiency,” not “at exactly 50% test efficiency.”

## 9. What do our current results actually show?

Source: the three all-feature `metrics.json` files and their 37,500-row test prediction CSVs; recomputed on 27 October 2026.

| Model | ROC-AUC | Average precision | Brier | Accuracy at 0.5 | Recorded training seconds |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.685399 | 0.681704 | 0.223111 | 0.643893 | 0.68 |
| XGBoost | 0.812343 | 0.826569 | 0.176530 | 0.733413 | 4.72 |
| Compact MLP | 0.812104 | 0.826959 | 0.176701 | 0.732187 | 50.22 |

The nonlinear models substantially outperform this linear baseline on this split. XGBoost exceeds Logistic Regression by **0.126943 AUC units**. Its lead over the compact MLP is only **0.000238**; that is a numerical lead, not established statistical superiority. The MLP has a slightly higher average precision. No repeated-seed or uncertainty analysis has established a winner between those two.

Runtime is a recorded measurement for one environment, not a universal speed guarantee. The compact MLP records CUDA; compare hardware and timing boundaries before making efficiency claims.

The three tuned quick-run prediction files reproduce the report's smaller-run scores. They do **not** reproduce the older 10-lakh summary. That summary reports a wide-MLP ROC-AUC of 0.839870 but now points to a 37,500-prediction file whose score is 0.796587. Retain it as a historical summary, not a verified current prediction-backed result. <!-- allow-metric -->

The higher values in the old report's discrepancy table match constants in the legacy synthetic fixture generator. That generator also manufactures predictions and curve coordinates. Those fixtures are not training evidence; they must not appear in the headline results.

## 10. Where is each part implemented?

| Concern | Main location |
|---|---|
| Feature names and order | `src/higgs_event_lab/features.py` |
| Data validation and loading | `src/higgs_event_lab/data.py` |
| Stratified split and artifact envelope | `src/higgs_event_lab/protocol.py` |
| Logistic Regression and XGBoost | `src/higgs_event_lab/classical.py` |
| Compact MLP and training loop | `src/higgs_event_lab/neural.py` |
| Tuned/full-data training | `scripts/train_full_colab.py` and `notebooks/03_full/00_full_higgs_colab.ipynb` |
| EDA statistics and figures | `src/higgs_event_lab/eda.py` and `outputs/eda/` |
| Run metadata / predictions | `outputs/<run_id>/metrics.json`, `test_predictions.csv` |
| Dashboard routing / file loading | `app/main.py`, `app/loader.py` |
| Charts and interactive metrics | `app/visualizations.py`, `app/metrics.py`, `app/views/` |
| Cloud entry point | `streamlit_app.py` |
| Tests | `tests/unit/`, `tests/integration/` |

The full trainer writes a summary plus flat prediction files rather than the same per-run directory layout. Sharing the schema version string does not make these layouts interchangeable. Full-run results need an explicit import/export conversion for the main registry.

## 11. How to explain and demonstrate Streamlit

1. **Overview:** introduce the dataset, the full-data experiment and the row counts of the currently loaded comparisons.
2. **Physics Explorer:** explain signal/background and the simulated benchmark boundary.
3. **Dataset Observatory:** show class balance, feature dictionary, mass separation and correlations. The EDA figures describe their recorded sample.
4. **Experiment Comparison:** show model and feature-group filters. Explain why Logistic Regression remains the baseline and why the nonlinear models help.
5. **Validation:** show ROC, average-precision context, calibration, real MLP histories and threshold trade-offs.
6. **Methods and Research:** explain reproducibility, train-only scaling, model configuration and references.
7. **Model Simulation:** score user-provided features with available saved tuned weights. This is inference, not training or discovery. The tuned weights are separate from the nine compact/baseline result folders.

The local XGBoost binary previously caused a native crash during deserialisation in the dashboard import context. Default interactive loading remains disabled. A version mismatch is a possibility, **not a proved root cause**: the file loaded in an isolated earlier process. Logistic Regression and wide MLP remain available, and saved XGBoost comparisons still work. Retraining alone does not flip the loading switch.

## 12. Short viva answers

**Why not just use accuracy?** It hides the cost of losing signal versus retaining background and depends on a chosen threshold. We examine ranking, probability quality and operating points too.

**Why use all features if high-level features already help?** Their combination gives the best current ROC-AUC for each family. Engineered masses make useful structure accessible; low-level measurements can add complementary discriminative information for these finite-capacity models.

**Why did we change models?** We extended the comparison from a linear baseline to tree-based and neural nonlinear models. We retained all three under the same split to measure the benefit.

**Did we use the entire dataset?** The team reports completing a full 11-million-event experiment. The smaller EDA and feature-group comparisons are separate experiments. Their scores must not be relabelled as full-data results.

**Does more data guarantee better results?** No. Model capacity, hyperparameters, optimisation and data quality also matter. The smaller and historical larger runs changed multiple settings, so they are not a controlled learning-curve experiment.

**Is a score of 0.9 a 90% chance of discovering a Higgs boson?** No. At most it is a model's signal-class probability estimate under the simulated benchmark distribution, and even that needs calibration checks.

**What remains to strengthen the work?** Import full-run metadata and predictions, preserve run-specific outputs, repeat seeds or use paired uncertainty estimates, document duplicate checks, and separate synthetic demo fixtures from measured results.
