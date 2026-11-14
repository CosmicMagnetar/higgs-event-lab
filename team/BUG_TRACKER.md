# Bug and Issue Tracker

> **Owner:** Saumya Mishra (intake, reproduction, triage, verification). **Fixers:** the module owner (Aditya: data/classical; Rachit: MLP; Ronit: app; Krishna: integration/deployment; Saumya: docs, tests/checklists, packaging, low-risk cross-cutting).
> **Source of truth:** GitHub Issues with label `type:bug`. This file is the readable register that mirrors them and records the first QA sweep. When GitHub Issues exist, put the issue number in the "GH #" column.
> Process rules are in `docs/04_GITHUB_WORKFLOW.md` ("Bug process") and `docs/12_QA_TEST_PLAN.md`.

---

## 1. Severity scale

| Level | Meaning | Examples | Release effect |
|---|---|---|---|
| **P0** | Wrong or misleading results, data leakage, secrets or raw data committed, claim of discovery | Test set used for tuning; label in features; fabricated metric | **Blocks release** (Krishna decides) |
| **P1** | Main workflow broken or cannot be reproduced; schema mismatch; undocumented source | Loader crashes; dashboard cannot read artifacts; missing dependency list | Blocks release unless a workaround is documented |
| **P2** | Defect with a workaround; unclear or misleading wording | Confusing doc, minor plot error | Fix before final submission |
| **P3** | Cosmetic or consistency | Typo, formatting, line endings | Fix when convenient |

## 2. Status values

`New` > `Triaged` (reproduced, severity and owner set) > `In Progress` > `In Review` (fix PR open) > `Verified` (Saumya re-ran the reproduction on a clean clone) > `Closed`.  Also `Won't fix` and `Cannot reproduce` (with a note).

## 3. Triage procedure (Saumya)

1. Acknowledge within the team's agreed response time **[Krishna to set]**.
2. Reproduce from the report. If steps are missing, ask for them using the bug template and mark `Needs info`.
3. Reduce to a minimal failing example (a few lines or a tiny synthetic input).
4. Set severity and assign the module owner. Do not guess at internals of another person's module.
5. Link the fix PR. The fixer adds a regression test or a checklist item.
6. Verify on a clean clone, update `team/CHANGELOG.md`, then close.

---

## 4. Register: first QA sweep, 2026-10-21

Sweep scope: every file in the uploaded scaffold (60 entries), read in full and cross-checked against the other documents and against the official UCI page. No code, notebooks or data existed, so the sweep covers documentation, repository structure and process.

| ID | Sev | Area | Finding | Evidence | Owner | Status |
|---|---|---|---|---|---|---|
| BUG-001 | **P0** | Repo / data safety | `.gitignore` and `.editorconfig` were promised by `docs/02` and `REPOSITORY_FILE_LIST.txt` but absent. Without `.gitignore`, `git add .` (the command in `docs/00_START_HERE.md`) would stage raw data, model files and `.env` | Absent from the zip listing | Saumya | **Fixed** 2026-10-21: both files added; `test_gitignore_protects_data_secrets_and_artifacts` guards it |
| BUG-002 | P1 | Process | `.github/PULL_REQUEST_TEMPLATE.md` and `ISSUE_TEMPLATE/*` promised but absent, so the PR and bug process described in `docs/04` had no templates | Absent from the zip listing | Saumya (+ Krishna to review) | **Fixed**: templates added |
| BUG-003 | P3 | Repo hygiene | A macOS `.DS_Store` file was shipped in the archive | `.DS_Store`, 18,436 bytes | Saumya | **Fixed**: removed and git-ignored |
| BUG-004 | P2 | Citation | `CITATION.cff` listed "Project team / Higgs Event Lab" as the author, the version was undocumented elsewhere, and no dataset/paper references were included | `CITATION.cff` (original) | Saumya, Krishna | **Fixed** except Krishna's surname (`TODO-surname`). **Open item for Krishna.** |
| BUG-005 | **P1** | Setup / packaging | No dependency file and no installable package (`src/higgs_event_lab/` has only a README), so the setup guide cannot be fully verified and tests cannot import project code | No `requirements.txt` / `pyproject.toml` | Krishna (package layout), Saumya (docs) | **Fixed** 2026-10-25: `pyproject.toml` and verified `requirements.txt` active; package imports cleanly and 103 tests pass. |
| BUG-006 | **P1** | Data / sampling | The plan says "begin with 250,000 rows" but does not say how rows are chosen. Taking the first N lines is only valid if the file is not ordered; the exact class fraction is unmeasured | `docs/03`, `docs/05` | Aditya | **Fixed** 2026-10-22 (MID-01): `extract_sample` implements explicit row limits, seed recording, and automated comparison against later chunk signal fraction. |
| BUG-007 | P2 | Scientific wording | `docs/01` described the signal as "Higgs-associated". Per Baldi et al. (2014) the HIGGS signal is a *hypothetical exotic-Higgs* process against a top-quark background, which readers could mistake for the 2012 discovery | `docs/01_PROJECT_CHARTER.md`, original text | Saumya, Krishna review | **Fixed**: wording corrected and cross-referenced to `docs/08` |
| BUG-008 | P1 | Requirements traceability | The course brief PDF is the stated source of requirements but is not in the repository, so no one can verify the stated scope, split or deadlines | `README.md`, `docs/07` | Krishna | **Fixed** 2026-10-25: Rubric, scope, and deliverables incorporated into `docs/05_MILESTONES_AND_ACCEPTANCE.md` and `docs/10_OBJECTIVES_AND_METHODOLOGY.md`. |
| BUG-009 | P3 | Docs consistency | Label list in `docs/00_START_HERE.md` lacked `area:evaluation` and `stage:foundation`, which `docs/04` lists | Diff of the two docs | Saumya | **Fixed** |
| BUG-010 | P3 | Repo hygiene | `team/TASK_BACKLOG.csv` used Windows (CRLF) line endings while the rest of the repo is LF | `file` output on the CSV | Saumya | **Fixed**: converted to LF (also matches `.editorconfig`) |
| BUG-011 | **P1** | Protocol gap | No document fixes split proportions, the random seed, the number of repeated seeds or the target signal efficiency for the operating-point analysis | Searched all docs | Krishna | **Fixed** 2026-10-25: Confirmed in `docs/13_EXPERIMENT_PROTOCOL_AND_ARTIFACTS.md` and `src/higgs_event_lab/protocol.py` (70/15/15, seed 42, target efficiency 0.50). |
| BUG-012 | P3 | Versioning | `CITATION.cff` said version 0.1.0 while `CHANGELOG.md` said "Unreleased" only | Original files | Saumya | **Fixed**: both now say `0.1.0-dev` |
| BUG-013 | P3 | Docs consistency | `docs/02` and `REPOSITORY_FILE_LIST.txt` did not list the new documents, tests and notebook | n/a | Saumya | **Fixed** (kept in sync by `test_every_file_in_repository_file_list_exists`) |
| BUG-014 | P2 | Data assumptions | Several preprocessing assumptions (b-tag takes few discrete values, heavy-tailed pT/mass features, no missing values, class balance) are expected from physics but unmeasured | `docs/09` section 4 | Aditya | **Fixed** (MID-02): verified in EDA notebook; b-tag has discrete modes, zero missing/non-finite entries, and 52.8% signal fraction |
| BUG-015 | **P0** | Results / headline | Superseded draft MLP metrics (250k: 0.9406 / 0.9043 / 0.8000) were inconsistent with the current per-run artifacts; the separate 1M wide MLP records 0.8399. Current 250k compact-MLP artifacts are 0.8121 (all), 0.7832 (high-level), and 0.7257 (low-level). | `outputs/mlp_*/metrics.json`; `outputs/full_colab_results.json`; `README.md`; `reports/01`, `reports/02`; `docs/14` | Krishna, Rachit | **Open: neural evidence only.** Stale draft values have been replaced in the reviewed Markdown sources. Current 250k and 1M runs remain separate; T4/T5 reproduction and release decision are still pending. No cross-run ranking is approved. | <!-- allow-metric -->
| BUG-016 | P1 | Results / classical | Logistic Regression all-features test ROC-AUC is 0.8369 at 250k (C = 1.0, max_iter 1000) and 0.6834 at 1M (C = 1.0, max_iter 2000, labelled "tuned"). The regularisation is identical, so tuning does not explain the drop. The data or the code path must. | `outputs/logreg_all/metrics.json`; `outputs/full_colab_results.json`; `scripts/train_full_colab.py` | Aditya | **Fixed** 2026-11-13 (fix/dl-higgs-data-reconciliation): Investigated sampling — both 250k and 1M use `sequential_head` from HIGGS.csv.gz. 250k = rows 0–249,999; 1M = rows 0–999,999. Same method confirmed. The 0.8369 value in the original `docs/14` table was **fabricated placeholder data** — the colab-run metric was 0.6834 all along. The LogReg AUC is consistent: local clean re-run on 250k gives 0.6854 (test). <!-- allow-metric -->
| BUG-017 | P1 | Results / leakage review | The old draft accuracy 0.865 / ROC-AUC 0.9406 pair is superseded. Current `mlp_all` artifact records accuracy 0.7322 at score cutoff 0.5 and test ROC-AUC 0.8121. Independent reproduction is still missing. | `outputs/mlp_all/metrics.json`; `outputs/mlp_all/test_predictions.csv`; `docs/14` | Rachit, Krishna | **Open: neural evidence only.** T1 split disjointness passed; T4 clean reproduction and T7 accuracy consistency remain pending. Do not infer that split disjointness alone proves no leakage. | <!-- allow-metric -->
| BUG-018 | P2 | Results / reporting | `outputs/full_colab_results.json` gives the same `training_seconds` (412.03) for all three models. This is a copy error in the reporting code, so the training times are not per model. | `outputs/full_colab_results.json`; `scripts/train_full_colab.py` | Aditya, Rachit | **Fixed** 2026-11-13: `full_colab_results.json` updated with per-model times from Kaggle notebook bc03581 stdout: LogReg=2.6s, XGBoost=35.1s, MLP=395.1s. `train_full_colab.py` already computes per-model timings correctly since commit 9c58a51. |
| BUG-019 | P3 | Artifacts | The 250k `metrics.json` files list feature names as `feat_0` to `feat_27` rather than the real column names. Dashboard feature tables will show generic names. | `outputs/*/metrics.json` (`features.names`) | Rachit, Aditya | **Fixed** 2026-10-27 (commit 57cb133): all 250k metrics.json files now use real HIGGS column names (lepton_pt … m_wwbb). Verified 2026-11-13: all 9 artifact files show `has_generic: False`. |
| BUG-020 | P1 | Reproducibility | The processed samples (`data/processed/higgs_sample_250k.csv.gz`, `higgs_sample_1m.csv.gz`) and `outputs/model_weights/` are git-ignored and absent from the repository. No one can regenerate the results from the repo alone. | `.gitignore`; `outputs/README.md` | Krishna / Aditya | **Fixed** 2026-11-13: Manifests committed (`data/processed/*.manifest.json` now unignored). 250k manifest at `data/processed/higgs_sample_250k.csv.gz.manifest.json`, 1M manifest at `data/processed/higgs_sample_1m.csv.gz.manifest.json`. Regeneration commands in `data/processed/README.md`. SHA-256 prefix recorded. |
| BUG-021 | P2 | Dashboard / entry points | `app/main.py` and `streamlit_app.py` both exist, and `app/loader.py` and `app/utils/loader.py` overlap. | `app/` | Ronit | **Open; module-owner review required.** Dashboard artifact fixes merged in PR #26; verify whether entry-point and loader consolidation remains necessary before closure. |
| BUG-022 | P2 | Tests / environment | `tests/unit/test_app_views.py` requires Streamlit. It fails in an environment where only the core test tools are installed. | `tests/unit/test_app_views.py`; `docs/11` section 8 | Ronit, Saumya | **Fixed / verified 2026-11-14:** canonical `.[dev]` install and full suite passed in the current-branch Python 3.11 CI run; the user-supplied fresh macOS clone also passed 143 tests. The documented test install includes dashboard dependencies. |
| BUG-023 | P2 | Documentation / reports | The Saumya report draft (`HiggsLens_Project_Report`) states that XGBoost is not implemented and omits the 1M benchmark. Both are now in the repo. | `docs/` and report draft | Saumya | **In progress.** Saumya's report draft and classical report have been corrected against the reconciliation record. Final acceptance still depends on the neural release decision and module-owner review. |
| BUG-024 | P2 | Governance | Legacy flat artifact schema conflicts with the implemented nested `higgs-experiment-v1` contract. | `docs/decisions/DRAFT-ADR-0002-experiment-artifact-schema.md`; `app/loader.py`; `app/utils/schema.py` | Krishna / Ronit | **Decision accepted 2026-11-12:** `higgs-experiment-v1` is canonical. Legacy loader consolidation and regression tests remain open under issue #21. |
| BUG-025 | P2 | Deployment | The setup guide had a generic `<OWNER>` clone placeholder and did not identify the canonical Streamlit Cloud repository. | `docs/11_SETUP_GUIDE.md` | Krishna | **Fixed 2026-11-12 in merged PR #23.** Canonical repository is `CosmicMagnetar/higgs-event-lab`; clone URL and Streamlit Cloud repository are explicit. |
| BUG-026 | P2 | Data privacy | `data/processed/higgs_sample_250k.csv.gz.manifest.json` records the local machine path (a personal user folder). The path is not needed for reproduction. | manifest file | Krishna / Aditya | **Fixed** 2026-11-12 (PR #24) & 2026-11-13 (Issue #18): Stripped local user paths (`/Users/...`, `/content/...`) from `data/processed/higgs_sample_250k.csv.gz.manifest.json`, `data/interim/cleaning_manifest.json`, and all `outputs/*/metrics.json` artifacts. Added `to_relative_manifest_path()` in `src/higgs_event_lab/data.py` to ensure all future manifests use repository-relative paths. |

### Summary counts (2026-11-14, after clean-clone QA)

| Severity | Total | Fixed | Open |
|---|---|---|---|
| P0 | 2 | 1 | 1 (BUG-015) |
| P1 | 8 | 7 | 1 (BUG-017) |
| P2 | 10 | 8 | 2 (BUG-021, 023) |
| P3 | 6 | 6 | 0 |
| **All** | **26** | **22** | **4** |

BUG-022 is verified fixed: the canonical development install includes Streamlit and the complete test suite passes on a fresh clone and in hosted CI.

## 5. Watch list: risks to test as soon as code exists

These are not bugs yet. Each becomes a bug the moment the code fails the check. They are the highest-value regression tests for this project.

| ID | Risk | Test that must exist | Owner of the test | Status |
|---|---|---|---|---|
| W-01 | Label column leaks into the feature matrix | Assert the feature matrix has 21, 7 or 28 columns and no column equal to the label | Aditya | **Implemented & passing** (`test_reconciliation_w01_w02_w03_w06.py`) |
| W-02 | Off-by-one in column indices (UCI says "columns 2 to 22", Python starts at 0) | Compare loader column names against `docs/feature_dictionary.csv` | Aditya | **Implemented & passing** (`test_reconciliation_w01_w02_w03_w06.py`) |
| W-03 | Scaler or imputer fitted before the split | Fit on a toy set where train and test have different means; assert test mean is not used | Aditya, Rachit | **Implemented & passing** (`test_reconciliation_w01_w02_w03_w06.py`) |
| W-04 | Different models evaluated on different test rows | Assert identical test indices (or hash) across all nine experiments | Krishna | Open |
| W-05 | Threshold or hyperparameters chosen with test data | Assert the operating threshold is computed from validation arrays only | Krishna | Open |
| W-06 | Non-reproducible runs | Same seed gives identical split and near-identical metrics | Rachit, Aditya | **Implemented & passing** (`test_reconciliation_w01_w02_w03_w06.py`) |
| W-07 | MLP in train mode at inference (dropout and batch-norm active) | Assert `model.eval()` is used and predictions are deterministic | Rachit | Open |
| W-08 | Metric edge cases (one class only, ties) | Unit tests with tiny arrays | Krishna | Open |
| W-09 | Dashboard shows numbers without artifacts | Start the app with an empty `outputs/` and assert an empty-state message | Ronit | Open |
| W-10 | Dashboard accepts malformed or incompatible artifacts | Feed a bad schema and assert a clear error | Ronit | Open |
| W-11 | Accuracy reported alone, or mock data shown unlabelled | Review of the report and dashboard | Saumya | In progress; reviewed dashboard views now label held-out test leaderboard values descriptively, and the validation slider is explicitly exploratory-only. No hard-coded mock metrics were found in the reviewed views. AppTest covered all seven dashboard sections in the Issue #21 verification; final browser screenshot review and regenerated binary exports remain outstanding. |
| W-12 | Text claims discovery or real-world probability | Grep wording in the report and app (see QA plan section 6) | Saumya | In progress; reviewed the README, key project/setup/reconciliation documents, reports, presentation Markdown, five notebooks, dashboard views and core model/evaluation code. Superseded neural figures and unsupported claims were corrected in the reviewed sources. PDF/PPTX exports are stale binary copies, and module-owner review remains pending. |

---

## 6. New-bug template (copy into a GitHub issue)

```
Title: [area] short description
Severity: P0 / P1 / P2 / P3
Module / owner:
Steps to reproduce:
Expected:
Actual (paste real output):
Environment (OS, Python, commit):
Minimal failing example:
Regression test or checklist item added:  yes / no
```
