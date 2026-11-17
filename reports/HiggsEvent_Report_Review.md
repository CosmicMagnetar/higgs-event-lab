# HiggsEvent: review of the attached corrected project report

**Review date:** 27 October 2026  
**Scope:** the attached corrected Word report, current source code, nine per-run artifact folders, three tuned prediction CSVs, EDA, notebooks and historical full-run summary. The original attachment is preserved.

## Executive assessment

The report's main physics framing and the current 2.5-lakh comparison tables are useful. It needs corrections before submission, particularly around **project name, experiment sizes, synthetic historical results, sampling, model histories and what was actually verified**.

The project name is **HiggsEvent**. The team reports completing an experiment using the complete **11-million-event (1.1-crore) dataset**, in addition to the smaller runs. The current full notebook is configured for that full-data workflow. The local evidence does not yet include its result bundle: the nine registry runs record 250,000 rows; the historical summary records 1,000,000. This review does not infer that the full run never happened, and does not assign smaller-run scores to it.

### Highest-priority corrections

1. Replace the old project name throughout the abstract, captions, diagrams, dashboard description, conclusion and running headers.
2. Describe **three experiment scales**: the 2.5-lakh EDA/feature-group experiments; the historical 10-lakh summary; and the team-reported completed 1.1-crore experiment. Add actual full-run results when its export is supplied.
3. Replace Section 20.3's “unexplained earlier trained results” story. Its scores match hard-coded constants in `scripts/generate_sample_artifacts.py`, which generates synthetic predictions and artificial curve coordinates. Those values are demonstration fixtures, not evidence of model training.
4. Correct the all-feature XGBoost–Logistic Regression ROC-AUC difference to **0.126943**, approximately **0.127**, rather than 0.19. <!-- allow-metric -->
5. Include compact MLP training times and genuine 30-epoch histories; the report incorrectly says they were not saved.
6. Do not call the 10-lakh summary prediction-verified: its referenced prediction CSVs now belong to a 2.5-lakh run.

## 1. What the current files establish

I recomputed ROC-AUC, sklearn average precision, Brier score and accuracy at threshold 0.5 from all twelve local prediction CSVs. The nine per-run results agree with their saved metrics to numerical precision. Each has 37,500 held-out predictions and metadata recording 175,000/37,500/37,500 train/validation/test events.

| Current run | ROC-AUC | Average precision | Brier | Accuracy | Recorded seconds |
|---|---:|---:|---:|---:|---:|
| Logistic Regression / low-level | 0.595978 | 0.622164 | 0.241766 | 0.567973 | 0.386 |
| Logistic Regression / high-level | 0.651750 | 0.621677 | 0.231491 | 0.629493 | 0.320 |
| Logistic Regression / all | 0.685399 | 0.681704 | 0.223111 | 0.643893 | 0.679 |
| XGBoost / low-level | 0.709250 | 0.723306 | 0.216494 | 0.651200 | 4.174 |
| XGBoost / high-level | 0.784830 | 0.802596 | 0.188246 | 0.704560 | 2.444 |
| XGBoost / all | 0.812343 | 0.826569 | 0.176530 | 0.733413 | 4.718 |
| Compact MLP / low-level | 0.725741 | 0.736573 | 0.211324 | 0.664960 | 47.894 |
| Compact MLP / high-level | 0.783159 | 0.802023 | 0.189245 | 0.704400 | 47.002 |
| Compact MLP / all | 0.812104 | 0.826959 | 0.176701 | 0.732187 | 50.225 |

Sources: `outputs/logreg_*/`, `outputs/xgboost_*/`, `outputs/mlp_*/`. Training times are read from metadata, not remeasured by this review. These are current **250,000-event experiment** results.

### Tuned quick-run predictions

| Prediction file prefix | Test events | ROC-AUC | Average precision | Brier | Accuracy |
|---|---:|---:|---:|---:|---:|
| `logistic_regression_tuned` | 37,500 | 0.685399 | 0.681704 | 0.223111 | 0.643893 |
| `xgboost_tuned` | 37,500 | 0.801537 | 0.817406 | 0.182032 | 0.723973 |
| `wide_mlp_tuned` | 37,500 | 0.796587 | 0.811994 | 0.183883 | 0.720187 |

These support the metric columns of Table 8. They do not establish Table 8's timings, ten-epoch history or precise hyperparameters. The currently saved notebooks have no execution outputs, and the current full script specifies different maximum training settings.

### Historical 10-lakh summary

`outputs/full_colab_results.json` records 1,000,000 events and higher tuned XGBoost/MLP scores. However, it references the same three filenames listed above, which now contain 37,500 predictions with different scores. For a 15% test split of 1,000,000 events, 150,000 predictions would be expected.

All three summary timings are 412.03 seconds. The report is right not to use them as trustworthy per-model timings. The summary should be marked **historical, matching predictions unavailable**, not silently reconciled by replacing its row count or scores.

## 2. Section-by-section changes

| Report section | Finding | Recommended correction |
|---|---|---|
| Cover, headers, figures, abstract, conclusion | Mixed project identity | Use HiggsEvent consistently, including text embedded in images. The corrected Word file's cover already uses it. |
| 1 Abstract | Describes only sample experiments; broadly says high-level beats low-level in every reproduced run | Add the team-reported complete-data experiment; specify that the feature-group ordering is for ROC-AUC, not every metric. |
| 2 Introduction | Appropriate benchmark motivation | Keep; avoid implying the classifier recreates the 2012 discovery. |
| 3 Physics | Distinguishes hypothetical benchmark signal from observed Higgs | Keep this distinction. “Higgs field gives particles mass” should not imply it explains all composite-hadron mass. |
| 4 Problem statement | Makes a subset mandatory | Say the pipeline supports full-data training and separately documented smaller comparison/EDA runs. |
| 5 Motivation | Treats missing transverse energy as simply the neutrino | Explain momentum imbalance can reflect invisible particles; avoid implying a direct neutrino measurement. |
| 6 Objectives | Says full schema validation is established and compact timing is missing | Cite a saved validation log for the former; include compact MLP histories/times from metadata. |
| 7 Scope | Restricts project to subsets; excludes live serving without distinguishing inference | Include full-data experiments. Distinguish interactive saved-model inference from production inference infrastructure. |
| 8 Data card | Uses a seeded-random manifest as if it describes the current notebook inputs | Current baseline notebooks read the first N rows of the raw file; label the older manifest as historical until the actual sample is reconciled. Add separate per-experiment provenance. |
| 8 Data card | Claims a validated 11-million-row file from Colab output | Preserve the team's account but attach the actual count/schema log and file hash. A configured expected count is not an execution record. |
| 9 EDA | Counts, class balance, missing/nonfinite counts agree with saved EDA | Keep these as 250,000-row EDA statistics, not full-data statistics. |
| 9.3 Separation | “Most informative single quantity” exceeds what mean separation establishes | Say `m_bb` has the largest measured standardised class-mean gap. It is not a model importance or single-feature AUC test. |
| 9.4 Correlations | Suggests trees are broadly unaffected | Correlation may affect feature attribution and redundancy; avoid a universal immunity claim. |
| 9.5 B-tag | Treats discrete values as necessarily nominal categories | Describe them as discrete encoded tag values. One-hot/ordinal treatment is an empirical modelling choice. |
| 10 Methodology | Claims every model exports validation-selected operating points | Classical baseline artifacts do; compact MLP artifacts currently do not. Separate implemented protocol from missing exports. |
| 10.2 Leakage | Says test was used exactly once as an established historical fact | The code intends held-out evaluation; audit logs are needed to prove all historical model-selection decisions. |
| 11 Architecture | Says every run has the same artifact layout | The tuned trainer writes a summary and flat prediction CSVs; the registry expects per-run directories. Document both contracts and the import gap. |
| 12.1 Loading | “Whole file is never held in memory” is false for full training | Chunked loading exists, but selected chunks are concatenated; the full trainer uses one `pd.read_csv` and in-memory arrays. |
| 12.3 Scaling | Mostly correct for classical baseline | Clarify that full tuned XGBoost receives scaled arrays too, although trees do not require scaling. |
| 13 Classical models | The stated 0.19 AUC gap is arithmetically wrong | Replace with approximately 0.127; retain Logistic Regression as the baseline rather than saying it was abandoned. |
| 14 MLP settings | Uses vague “recorded in script” entries | Give current full-script settings explicitly, labelled as current code; do not assign them retroactively to old quick/full runs. |
| 14.2 Compact timing | Says no time recorded | Low/high/all times are 47.89/47.00/50.22 seconds in saved metadata. |
| 14.3 Learning curve | Says compact runs have no per-epoch values | All three contain 30 real epoch records. Use these for a sourced figure; preserve a wide-model curve only with its original log. |
| 14.3, 21.2 Convergence | Epoch limit or increasing validation metric treated as proof of non-convergence | Say training reached its configured budget and additional training might help; do not claim a proved optimum or non-convergence. |
| 14.4, 20.2 | Calls 1 million “full scale” | Call it the historical 1-million-event benchmark; reserve “complete dataset” for 11 million. |
| 15 Metrics | Equates average precision with generic area under PR curve | State sklearn `average_precision_score`, a recall-increment-weighted precision sum; distinguish it from trapezoidal PR area. |
| 15 Metrics | Brier described as only calibration | Describe overall probability-prediction error, affected by both calibration and discrimination. |
| 16 Dashboard | Says all three models always score uploads and missing uploads trigger demo | Live scoring depends on weights; XGBoost deserialisation is currently disabled by default. Single-event scoring works without an upload; saved distributions have their own tab. |
| 17 Contributions | Roles are stated but not independently verified | Retain the team's agreed roles; attach sign-off or deliverable links if required by the university. |
| 18 Reproducibility | Treats notebook configuration as execution evidence | Preserve executed logs, package versions, dataset hashes and run-specific output directories. Current notebooks have cleared outputs. |
| 18.4 Artifact fields | Says every artifact has `created_at` | Current neural writer uses `created_at_utc`; describe actual producer fields and required loader schema. |
| 19 Testing | “105 passed,” “106 functions” and “not re-run” conflict | Replace with the dated command and actual latest result: 107 tests passed in this revision. Test functions and collected test cases are not always equal. |
| 20.1 Results | Current metrics are supported | Keep; call XGBoost and MLP numerically close, without claiming statistical equivalence. |
| 20.2 Interpretation | Attributes better results solely to larger data | Configurations and budgets also changed. No controlled data-size ablation establishes causation. |
| 20.3 Old scores | Calls synthetic fixture values unexplained training results | Replace with the concrete fixture-generator finding; remove obsolete plots from the main results and keep an audit note if useful. |
| 20.4 Interpretation | Generalises feature-group ordering to the 1-million run | Historical summary contains only all-feature tuned runs. The three-group ordering is established only for the nine smaller runs. |
| 21 Open issues | Says weights are absent, conflating git with the workspace | Local weights exist under `outputs/model_weights/`; they are excluded from source control. Describe availability and compatibility separately. |
| 22 Ethics | Claims every number has logged run provenance | Say which metrics were prediction-recomputed, which are metadata-only, and which are team-reported. |
| 23 Future work | Says per-epoch histories and calibration diagrams are unimplemented | They exist for current compact runs / loaded predictions. Propose extending consistent exports, recalibration and repeated seeds instead. |
| 24 Conclusion | Repeats subset-only scope and unexplained high-score story | Summarise actual comparison evidence, team-reported full-data work and outstanding full-run import. |
| 25 References | Several listed software/method references are not explicitly cited in the corresponding text | Cite them where used and check bibliographic metadata against the publisher; update numbering if edits change first-use order. |
| 26 Appendices | Repeats incorrect history/timing/issue assertions | Regenerate from corrected tables; update figure captions, cross-references and table of contents after Word repagination. |

## 3. Replacement text you can use

### Project scope / abstract paragraph

> HiggsEvent compares Logistic Regression, XGBoost and multilayer perceptrons for classifying simulated signal and background events in the UCI HIGGS benchmark. The project includes smaller feature-group and exploratory experiments and a team-reported completed experiment using the complete 11-million-event dataset. These experiments are reported separately: the currently imported nine comparison runs use 250,000 events with a stratified 70/15/15 split and seed 42. The complete-data experiment uses the full source as the input to that partitioning process, not as a single training-only partition. Its result bundle must be attached before this report assigns numerical performance to it.

### Why multiple models?

> Logistic Regression is retained as a fast regularised linear baseline. XGBoost adds nonlinear feature interactions through boosted decision trees, while the MLP learns nonlinear combinations through hidden layers. Comparing all three on the same partition measures the value of additional model complexity rather than assuming that a neural network must perform best.

### Current results paragraph

> On the saved 250,000-event comparison, all-feature XGBoost and compact MLP obtain test ROC-AUC values of 0.812343 and 0.812104, respectively, versus 0.685399 for Logistic Regression. XGBoost's numerical advantage over the MLP is 0.000238 and has not been established as statistically significant. The validation-selected XGBoost threshold retains 50.30% of signal and rejects 89.17% of background on the held-out test partition. <!-- allow-metric -->

### Replace Section 20.3

> The earlier high-score table matches constants in the repository's legacy synthetic artifact generator. That script manufactures example prediction files and curve coordinates instead of training models on HIGGS events. Those outputs are excluded from the project's measured results. The current nine experiment folders contain replacement prediction-backed results whose ROC-AUC, average precision, Brier score and accuracy were independently recomputed during this review. Automatic generation of synthetic results at dashboard startup has been removed.

This identifies a concrete code path capable of producing the earlier tables; it does not assert that every historical file's creation history has been independently recovered.

### Testing paragraph

> On 27 October 2026, `PYTHONPATH=src:. .venv/bin/python -m pytest tests/ -q` completed with 107 passed. Tests cover software behaviour and controlled fixtures; they are not evidence that the complete 11-million-event training run was executed in this review. Separate recomputation checked the metrics in the saved prediction CSVs.

## 4. What to attach for the complete-data experiment

Provide the result bundle from the runtime where the full job completed:

- Summary recording the actual input path, **11,000,000 rows**, run identifier and timestamp.
- Train/validation/test counts and seed; under the stated protocol expect **7,700,000 / 1,650,000 / 1,650,000**.
- Per-model settings, final metrics and separate measured runtimes.
- Test predictions with original row indices; normally **1,650,000 predictions per model** for this protocol, unless a different test scheme was intentionally used.
- Neural epoch history, best epoch and device/runtime information.
- Input checksum and saved row/schema-validation output.

This is an evidence import task, not a request to rerun eleven million rows unnecessarily. Never edit a 250,000-row result's metadata to make it appear full-data.

## 5. Formatting and presentation review

The review inspected the Word document's full paragraph/table text and project evidence. It does not claim a page-by-page visual layout validation of the attachment.

- Remove the extracted leading “0 0” if those characters are visible on the cover or in text boxes; verify them in Word first.
- Use one project title and consistent heading styles. Update the automatic contents and page numbers after changes.
- Keep “Higgs boson signal detection” as domain motivation, but use “simulated HIGGS event classification” for the actual technical task.
- Label each figure with experiment size, run ID and source artifact. A schematic architecture diagram is not execution evidence.
- Replace the unsourced wide-MLP quick-run curve with a current compact-MLP curve unless its original log is available.
- Put historical/demo provenance in an audit appendix, not alongside measured-model results in the headline comparison chart.
- Use consistent units: `250,000 (2.5 lakh)`, `1,000,000 (10 lakh)`, `11,000,000 (1.1 crore)` at first use; one style thereafter.
- Keep metric precision consistent (four decimals in the report; full precision in machine-readable results). Tiny differences should not be exaggerated visually.
- Cite the XGBoost, PyTorch, AdamW, batch-normalisation, dropout and metric references in the sections that use them.

## 6. Changes made alongside this review

- Standardised application and generated-deliverable branding to HiggsEvent.
- Reduced oversized UI spacing, shortened tab labels, made metric labels/values wrap and used thousand/lakh/crore count formatting with exact counts retained in tooltips/tables.
- Made experiment sizes explicit on Overview; EDA and comparison rows no longer imply the project's total data usage.
- Added a complete explanation under Methods and Research and in `docs/HiggsEvent_End_to_End_Guide.md`.
- Removed the automatic synthetic-result fallback and isolated the legacy fixture generator from the research output directory.
- Routed the legacy `app/streamlit_app.py` entry point to the same current dashboard, so launch commands no longer select different UI/evidence implementations.
- Corrected the simulation notice: an XGBoost deserialisation crash was observed, but a version mismatch was not proved to be its cause.

The original Word attachment was not rewritten. This review provides the corrections while preserving the team's 26-section report structure and authorship details.
