# Reports and Figures

The final report is synthesized by **Krishna** from actual results. Each module owner writes the methods/results for their experiments, and **Saumya Mishra** reviews readability, references, terminology, and consistency.

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

## Current Drafts

| File | Author | Status | Contents |
|---|---|---|---|
| `01_report_draft_methodology_limitations.md` | Saumya Mishra | Revised draft; neural gate pending | Integrated scientific methodology and separately labelled 250k/1M evidence; no project-wide headline result. |
| `02_neural_network_benchmarks.md` | Rachit Gupta | Draft | Compact MLP architecture, regularization, cross-feature benchmark protocol, artifact export. |
| `03_classical_baseline_benchmarks.md` | Aditya Rana | Reconciliation correction added; module-owner review pending | Corrected 250k Logistic Regression ROC-AUC values and removed the invalid placeholder score. |
| `04_midsem_presentation_and_demo.md` | Ronit Singh | Draft | 12-slide presentation structure and live demo script. |
| `05_viva_defense_guide.md` | Krishna | Draft | Viva examination Q&A preparation pack (20 questions). |
| `06_technical_methodology_and_reproducibility.md` | Krishna | Integrated into report 01; source retained | Technical rationale for physics scope, feature groups, model choices, leakage-safe protocol, metrics, reproducibility, and scientific limitations. |

## Final generated deliverables

These files are generated from the saved artifacts under `outputs/`:

| File | Purpose |
|---|---|
| `HiggsLens_Final_Report.pdf` | Formatted artifact-backed final report; regenerate after source reconciliation before release |
| `HiggsLens_Final_Report.html` | Browser preview; QA wording refreshed, full Pandoc regeneration still recommended |
| `HiggsLens_Final_Report.md` | Reproducible report source |
| `../presentations/HiggsLens_Final_Presentation.pptx` | Presentation deck; regenerate from the corrected Markdown source before release |
| `../scripts/build_final_deliverables.py` | Rebuilds the Markdown report and presentation sources from current outputs; it does not regenerate HTML, PDF or PPTX |

Placeholders marked `[RUN: ...]` indicate where real metric values should be inserted once artifacts exist.

---


## Release export status

The Markdown report and presentation sources have been reconciled to the current saved artifacts. The committed PDF and PPTX are binary exports from an earlier source state and have not been regenerated in this QA pass. Do not treat those two binary exports as final-release deliverables until rebuilt from the corrected Markdown sources. The HTML preview has had its stale QA statement corrected but should also be regenerated with Pandoc for exact source parity.
