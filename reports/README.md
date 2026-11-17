# Reports and Figures

The final report is maintained collaboratively: module owners provide and verify their methods/results; **Saumya Mishra** integrates documentation, references, terminology, evidence traceability and QA; **Krishna** retains final scientific/release review responsibility.

## Folder Purpose

Store report source files and selected small figures in this folder. Place large run outputs under the ignored `outputs/` folder and copy only chosen, appropriately sized figures into `reports/` when needed.

## Rule: No Premature Numbers

> Never write numeric results until they exist in saved experiment output under `outputs/`. All reported metrics must be traceable to serialized `metrics.json` / `test_predictions.csv` artifacts.

---

## Report Sections

| Section | Description |
|---|---|
| Abstract | High-level summary of task, methods, and key findings. |
| Physics Motivation | Why the HIGGS benchmark matters; signal vs. background definition. |
| Task | Binary classification of simulated collision events. |
| Data / Provenance | UCI dataset source, schema, sampling, split protocol. |
| Methods | Preprocessing, feature groups, model architectures, training setup. |
| Experiment Protocol | `higgs-split-v1`: stratified train/val/test, seed 42, leakage prevention. |
| Results | ROC-AUC, PR-AUC, Brier score, accuracy per model and feature group. |
| Error / Threshold Analysis | Operating-point selection, calibration, confusion analysis. |
| Compute Trade-off | Training time, inference latency, model size. |
| Limitations | Simulated data only, no physical discovery claims, sample-size constraints. |
| Conclusion | Summary of findings and future work. |
| References | Academic citations and data sources. |

---

## Final Deliverables

The primary, submission-ready project deliverables are:

| File | Type | Description |
|---|---|---|
| `HiggsLens_Final_Report.md` | Markdown | Comprehensive reproducible written report with all benchmark tables and physics analysis |
| `Higgs_Boson_Deep_Learning_Report_Overleaf__2_.pdf` | PDF Document | Official 43-page compiled submission report |
| `overleaf/` | LaTeX Source | Full Overleaf LaTeX project (`main.tex`, `references.bib`, `results.tex`, `HiggsLens_Overleaf.zip`) |
| `figures/` | Visual Assets | Publication-grade plots (`feature_groups.png`, `scaling.png`) |
| `accuracy_at_05.csv` | Audit Data | Threshold $\tau=0.5$ classification accuracies across all models |
| `saved_prediction_audit.json` | Audit Data | Cryptographic and metric traceability audit for all test predictions |

---

## Release Status — 2026-11-17

All intermediate drafts and duplicate conversion files have been cleaned up and consolidated. The project report and evidence pipeline reflect the complete 45-run benchmark across 5 model families, 3 feature groups, and 3 scaling budgets (250k, 1M, 2.5M).

