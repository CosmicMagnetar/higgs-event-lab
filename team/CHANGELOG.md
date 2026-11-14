# CHANGELOG

**Change Log**

*Saumya Mishra maintains this file for user-visible changes, documentation updates, known issues, and release notes. Module owners provide concise change details in their PRs.*

> **Note:** This file tracks project history and is not a place to claim model results before a run exists.

---

### Changed

* **2026-11-14:** PR #2 follow-up QA — reject hard-label outputs in interactive simulation, add regression coverage, clarify that baseline QA evidence predates this branch, neutralize viva prompts, correct stale neural benchmark reconciliation rows, and add a Docker image build to CI. Current-branch CI test suite and Docker image build passed on commit `50800faa19e1ae4fbd217aa7e9e5ddf120429b54` (runs #43573525224 and #43573521445). *(Saumya)*
* **2026-11-14:** `reports/01_report_draft_methodology_limitations.md`, `reports/03_classical_baseline_benchmarks.md` and `reports/README.md` — integrated methodology, corrected the invalid 250k Logistic Regression placeholder, separated 250k and 1M results, and marked neural results provisional pending release evidence. *(Saumya)*
* **2026-11-14:** `docs/11_SETUP_GUIDE.md` and `CITATION.cff` — aligned repository instructions and version metadata with the current repository and documented QA status. Current-branch test and Docker CI runs passed on `50800faa19e1ae4fbd217aa7e9e5ddf120429b54` (runs #43573525224 and #43573521445). *(Saumya)*
* **2026-11-14:** `team/TASK_BACKLOG.csv` — added explicit final-report integration and release-QA work packages. *(Saumya)*



### Conventions & Guidelines

* **Format:** Follows [Keep a Changelog](https://keepachangelog.com/) conventions (`Added` / `Changed` / `Fixed` / `Removed` / `Known issues`).
* **Dates:** Standard `YYYY-MM-DD` format.
* **Versioning:** Follows Semantic Versioning once the first release is tagged.
* **Adding Entries:** Place entries under `[Unreleased]`, one line per change, including the PR or issue number and the author.
* **Tagging Releases:** When Krishna tags a release, rename `[Unreleased]` to the version and date, then start a new empty `[Unreleased]` section.

---

## [Unreleased]

### Changed

* **2026-11-14:** Dashboard ranking labels now identify highest observed held-out test ROC-AUC as descriptive only; threshold workbench warns that operating-point selection must remain validation-only; simulation outputs are labelled class-1 classifier scores rather than physical probabilities. *(Saumya)*
* **2026-11-14:** QA plan and setup guide record the macOS Python 3.15 clean-clone result (143 passed), the earlier Python 3.11 hosted CI baseline, and successful current-branch Python 3.11 test and Docker CI runs; BUG-022 verified fixed. *(Saumya)*
* **2026-11-14:** Corrected stale classical baseline comparisons, softened provisional MLP interpretations, removed unsupported prevalence figures, and fixed duplicate report section numbering. *(Saumya)*

### Fixed

* **2026-11-13:** Issue #21 / BUG-021 and BUG-022: unified dashboard launch and artifact validation, removed synthetic startup fallbacks, exposed run metadata and errors, and declared dashboard/test extras with CI and regression coverage. Local verification complete; peer QA and integration pending. *(Ronit Singh)*

### Added

* **2026-10-26:** `docs/14_RESULTS_RECONCILIATION.md`: side-by-side record of the 250k and 1M results, the inconsistency found in the final review, the tests that would resolve it, and the decisions needed. *(Saumya Mishra)*
* **2026-10-26:** `team/BUG_TRACKER.md`: BUG-015 to BUG-026 added from the final-push review. *(Saumya Mishra)*
* **2026-10-26:** `docs/12_QA_TEST_PLAN.md`: final-push test log (98 non-PyTorch tests passed, 1 environment failure, PyTorch and real-file checks listed as NOT RUN). *(Saumya Mishra)*

* **2026-10-26:** `streamlit_app.py`, `.streamlit/config.toml`, `app/main.py`, `app/views/overview.py`, and `deployment/DEPLOYMENT_RUNBOOK.md` — Streamlit Community Cloud hosting integration with auto-bootstrap fallback for clean cloud deployments, custom scientific dark theme, and complete mid-semester evaluation readiness. *(Ronit & Krishna)*

* **2026-10-26:** `docs/`, `reports/01_report_draft_methodology_limitations.md`, `team/BUG_TRACKER.md`, and `team/TASK_BACKLOG.csv` — Completed mid-semester methodology and limitations synthesis with genuine empirical benchmark figures, added comprehensive Mermaid architecture, decay topology, feature hierarchy, and testing pyramid diagrams across documentation, closed all 14 bug tracker items, and marked `MID-07` as Done. *(Saumya)*

* **2026-10-25:** `outputs/mlp_<group>/`, `outputs/logreg_<group>/`, and `reports/02_neural_network_benchmarks.md` — Executed and validated multi-group MLP benchmark runs across `low_level`, `high_level`, and `all` features under `higgs-split-v1`, exported genuine `metrics.json` and `test_predictions.csv` artifacts, and completed empirical neural benchmark report (`MID-04`). *(Rachit)*

* **2026-10-25:** `app/main.py`, `app/loader.py`, `app/metrics.py`, `app/styles.py`, `app/visualizations.py`, `app/views/`, `tests/unit/test_app_loader.py`, `tests/unit/test_app_metrics.py`, and `app/README.md` — HiggsLens interactive scientific Streamlit dashboard with 6 focused sections, robust `higgs-experiment-v1` artifact loader, multi-run comparison, calibration diagrams, held-out ROC/PR curves, and interactive decision threshold operating-point workbench (`MID-06`, `END-05`). *(Ronit)*

* **2026-10-25:** `src/higgs_event_lab/classical.py`, `tests/unit/test_classical.py`, and `reports/03_classical_baseline_benchmarks.md` — Scikit-learn Logistic Regression baseline with leakage-free scaling via Pipeline, validation-only operating threshold selection, cross-feature group benchmarking (`low_level`, `high_level`, `all`), and `higgs-experiment-v1` artifact export (`MID-03`). *(Aditya)*
* **2026-10-25:** `notebooks/01_midsem/01_eda_and_feature_groups.ipynb` — Comprehensive exploratory data analysis notebook analyzing schema semantics, data hygiene, empirical class balance, discrete b-tag distribution, kinematic features, and cross-feature correlations (`MID-02`). *(Aditya)*
* **2026-10-25:** `src/higgs_event_lab/protocol.py`, `tests/unit/test_protocol.py`, and `docs/13_EXPERIMENT_PROTOCOL_AND_ARTIFACTS.md` — Shared deterministic train/validation/test split protocol (`higgs-split-v1`) and artifact contract (`higgs-experiment-v1`) for fair benchmark comparison without dataset persistence (`MID-05`, PR #4). *(Krishna)*

* **2026-10-25:** `src/higgs_event_lab/neural.py` and `tests/unit/test_neural.py` — Compact PyTorch MLP binary classifier with BatchNorm1d, Dropout, early stopping on validation ROC-AUC, best-weights restoration, and `higgs-experiment-v1` artifact export (`MID-04`, PR #5). *(Rachit)*
* **2026-10-25:** `notebooks/01_midsem/02_neural_network_baseline.ipynb` and `src/higgs_event_lab/neural.py` — Multi-group benchmark runner (`run_feature_group_benchmarks`) and interactive mid-semester notebook demonstrating MLP regularization, 21+7 cross-feature training, and artifact export (`MID-04`, PR #7). *(Rachit)*
* **2026-10-25:** `src/higgs_event_lab/evaluation.py`, `splits.py`, `artifacts.py`, `deployment/Dockerfile`, `deployment/DEPLOYMENT_RUNBOOK.md`, and `reports/02_neural_network_benchmarks.md` — Shared evaluation & threshold selection module, split/artifact wrappers, production deployment container and runbook, and neural report section (`MID-05`, `END-06`). *(Krishna & Rachit)*

* **2026-10-22:** `src/higgs_event_lab/features.py` — Schema definitions and deterministic feature group accessors (`low_level`, `high_level`, `all`) matching UCI schema and `docs/feature_dictionary.csv` (`MID-01`). *(Aditya)*
* **2026-10-22:** `src/higgs_event_lab/data.py` — Chunked streaming loader (`iter_higgs_chunks`), schema/type/value validators (`validate_raw_chunk`), and reproducible sampling workflow with later-chunk comparison (`extract_sample`) (`MID-01`). *(Aditya)*
* **2026-10-22:** `tests/unit/test_features.py`, `tests/unit/test_data_loader.py`, and `tests/integration/test_data_pipeline.py` — Unit and integration tests covering tiny uncompressed/compressed CSVs, feature group separation, error reporting, and reproducible extraction (`MID-01`). *(Aditya)*
* **2026-10-21:** `docs/08_PHYSICS_BACKGROUND.md` — Higgs field, Higgs mechanism, Higgs boson, link to the HIGGS task, glossary (`FND-05`). *(Saumya)*
* **2026-10-21:** `docs/09_DATASET_GUIDE.md` and `docs/feature_dictionary.csv` — Data card and the $21 + 7$ feature dictionary, verified against the UCI page. *(Saumya)*
* **2026-10-21:** `docs/10_OBJECTIVES_AND_METHODOLOGY.md` — Objectives, experiment matrix, pipeline, metrics, and leakage controls. Defaults marked `[PROPOSED]` need Krishna's approval. *(Saumya)*
* **2026-10-21:** `docs/11_SETUP_GUIDE.md` and `requirements.txt` (proposed dependency list). *(Saumya)*
* **2026-10-21:** `docs/12_QA_TEST_PLAN.md` and `team/BUG_TRACKER.md` — QA process, per-owner checklists, and initial bug register. *(Saumya)*
* **2026-10-21:** `tests/unit/test_repo_hygiene.py`, `test_notebook_style.py`, `test_feature_dictionary.py`, and `tests/integration/smoke_check_raw_file.py` — Automated checks for repository hygiene, notebook readability, feature schema, and the raw data file. *(Saumya)*
* **2026-10-21:** `notebooks/00_physics/00_higgs_physics_and_dataset_intro.ipynb` — Beginner notebook (Markdown-led, no model results). *(Saumya)*
* **2026-10-21:** `reports/01_report_draft_methodology_limitations.md` — Draft report sections for methodology and limitations. No numeric results. *(Saumya)*
* **2026-10-21:** `.gitignore`, `.editorconfig`, `.github/PULL_REQUEST_TEMPLATE.md`, `.github/ISSUE_TEMPLATE/*` — Files promised by repository tree but missing in scaffold (`BUG-001`, `BUG-002`). *(Saumya)*

### Changed

* **2026-10-26:** `docs/07_REFERENCES.md`: rewritten with stable numbering R1 to R19; restores the physics and dataset entries and adds XGBoost, scikit-learn, PyTorch, Streamlit and evaluation-method references. Entries not yet checked against publisher pages are marked †. *(Saumya Mishra)*
* **2026-10-26:** `docs/11_SETUP_GUIDE.md`: Streamlit prerequisite note; `app/main.py` vs `streamlit_app.py` explained; placeholder repository owner flagged (BUG-025). *(Saumya Mishra)*

* **2026-10-21:** `CITATION.cff` — Real author names, release date, dataset, and paper references (`BUG-004`). Krishna's surname remains a placeholder. *(Saumya)*
* **2026-10-21:** `docs/07_REFERENCES.md` — Added Higgs-mechanism, discovery, and Nobel references; entries numbered R1 to R8. *(Saumya)*
* **2026-10-21:** `README.md`, `docs/02_REPOSITORY_STRUCTURE.md`, `REPOSITORY_FILE_LIST.txt` — Reading order and file tree updated for new documents. *(Saumya)*
* **2026-10-21:** `team/TASK_BACKLOG.csv` — `FND-05` and `MID-07` moved to *In Review / In Progress*. *(Saumya)*
* **2026-10-21:** Converted `CHANGELOG.md` to Keep a Changelog layout. *(Saumya)*

### Removed

* **2026-10-21:** Stray `.DS_Store` file (macOS metadata) removed and git-ignored (`BUG-003`). *(Saumya)*

### Known Issues

* **2026-10-26:** Headline result not yet settled: the 250k and 1M benchmarks disagree (BUG-015). Do not quote a headline ROC-AUC in the report until the decision is recorded in `docs/14`.

* See `team/BUG_TRACKER.md` for the live list.
* As of this date, no model, loader, dashboard, or deployment code exists; no behavior has been tested beyond documentation and repository hygiene.

---

## [0.1.0-dev] - 2026-10-20

### Added

* Repository foundation (scaffold).
* Project scope, scientific limitations, and non-goals.
* Folder contracts, team ownership, GitHub workflow, and initial task backlog.
* Data-safety rules.
* *Note: No model benchmark or deployed app is claimed at this stage.*

* **2026-11-14:** Extended repository QA corrected superseded MLP values in README and reports, aligned the final report/presentation sources and generator with current artifacts and CI, corrected the HTML preview's stale QA statement, and documented that PDF/PPTX exports still require regeneration. *(Saumya)*
* **2026-11-14:** Removed a trailing empty code cell from `notebooks/01_midsem/03_classical_baseline.ipynb`; all five notebooks now pass JSON/static cell-structure inspection with no empty code cells. Full execution remains unverified. *(Saumya)*
* **2026-11-14:** Final consolidated QA commit `00de426` passed current-branch CI: editable install, **144 pytest tests passed in 14.59s** on Python 3.11, and Docker image build succeeded (run #43574590439). *(Saumya)*
