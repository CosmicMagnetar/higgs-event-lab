"""Build the final report source and presentation from saved experiment artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
REPORTS = ROOT / "reports"
PRESENTATIONS = ROOT / "presentations"


def load_runs() -> list[dict[str, object]]:
    runs: list[dict[str, object]] = []
    for metrics_path in sorted(OUTPUTS.glob("*/metrics.json")):
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        predictions = pd.read_csv(metrics_path.parent / "test_predictions.csv")
        test = data["evaluation"]["test"]
        validation = data["evaluation"].get("validation", {})
        training = data.get("training", {})
        operating = data["evaluation"].get("operating_point", {})
        test_accuracy = float(np.mean((predictions["score"] >= 0.5) == predictions["class_label"]))
        runs.append(
            {
                "run_id": data["run_id"],
                "model": data["model"]["name"],
                "group": data["features"]["group"],
                "features": data["features"]["count"],
                "rows": data["data"]["row_count"],
                "train_rows": data["split"].get("train_rows"),
                "validation_rows": data["split"].get("validation_rows"),
                "test_rows": data["split"].get("test_rows"),
                "val_roc": float(validation.get("roc_auc", 0.0)),
                "roc": float(test["roc_auc"]),
                "pr": float(test["pr_auc"]),
                "brier": float(test["brier_score"]),
                "accuracy": test_accuracy,
                "seconds": float(training.get("training_seconds", 0.0)),
                "device": training.get("device", "not recorded"),
                "history": training.get("history", []),
                "threshold": operating.get("validation_threshold"),
                "signal_efficiency": operating.get("test_actual_signal_efficiency"),
                "background_rejection": operating.get("test_background_rejection"),
            }
        )
    return runs


def model_label(model: str) -> str:
    return {
        "logistic_regression": "Logistic Regression",
        "xgboost": "XGBoost",
        "compact_mlp": "Compact MLP",
    }.get(model, model)


def table(runs: list[dict[str, object]]) -> str:
    ordered = sorted(runs, key=lambda item: float(item["roc"]), reverse=True)
    lines = [
        "| Model | Feature group | Features | ROC-AUC | PR-AUC | Brier | Accuracy | Time (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in ordered:
        lines.append(
            f"| {model_label(str(run['model']))} | `{run['group']}` | {run['features']} "
            f"| {float(run['roc']):.4f} | {float(run['pr']):.4f} | {float(run['brier']):.4f} "
            f"| {float(run['accuracy']):.4f} | {float(run['seconds']):.2f} |"
        )
    return "\n".join(lines)


def write_report(runs: list[dict[str, object]]) -> Path:
    REPORTS.mkdir(exist_ok=True)
    all_features = [run for run in runs if run["group"] == "all"]
    best = max(runs, key=lambda item: float(item["roc"]))
    best_all = max(all_features, key=lambda item: float(item["roc"]))
    eda = json.loads((OUTPUTS / "eda" / "summary.json").read_text(encoding="utf-8"))
    full_reference = json.loads((OUTPUTS / "full_colab_results.json").read_text(encoding="utf-8"))
    history_run = next((run for run in runs if run["run_id"] == "mlp_all"), None)
    history_epochs = len(history_run["history"]) if history_run else 0
    total_rows = int(eda["rows"])
    signal_fraction = float(eda["signal_fraction"])
    report = f"""# HiggsLens: Machine-Learning Classification of Simulated HIGGS Events

**Final artifact-backed report**  
**Project team:** Saumya Mishra · Aditya Rana · Rachit Gupta · Ronit Singh · Krishna  
**Dataset:** UCI HIGGS, dataset 280  
**Source:** <https://archive.ics.uci.edu/dataset/280/higgs>  
**Generated from:** saved artifacts under `outputs/`  

> **Scientific boundary.** HIGGS is a simulated Monte Carlo benchmark. This project evaluates classification software and makes no claim of particle discovery, real-detector performance, or physical probabilities.

## 1. Executive summary

HiggsLens is a reproducible machine-learning workbench for separating simulated HIGGS signal events from simulated background events. The implementation compares Logistic Regression, XGBoost and a regularised PyTorch multilayer perceptron across 21 low-level kinematic features, 7 high-level invariant-mass features and all 28 features.

The current validated artifact registry contains **{len(runs)} runs** over **{total_rows:,} recorded benchmark rows**, using the `higgs-split-v1` protocol with seed 42. The strongest currently loaded artifact is **{model_label(str(best['model']))} on `{best['group']}` features**, with test ROC-AUC **{float(best['roc']):.4f}**, PR-AUC **{float(best['pr']):.4f}**, Brier score **{float(best['brier']):.4f}**, and accuracy **{float(best['accuracy']):.4f}**. <!-- allow-metric -->

Among all-feature runs, **{model_label(str(best_all['model']))}** is currently highest at ROC-AUC **{float(best_all['roc']):.4f}**. <!-- allow-metric --> This identifies the highest current saved artifact in the recorded {total_rows:,}-row registry; it is not a cross-benchmark model-selection decision. The compact-MLP rows remain provisional pending independent reproduction and the release decision. The separate `outputs/full_colab_results.json` 1M reference uses a different pipeline/configuration and is not mixed into this registry.

## 2. Research question and scope

The central question is whether models can learn useful separation directly from low-level event kinematics, whether high-level invariant masses provide a measurable advantage, and how model family changes discrimination, calibration and compute cost.

In scope:

- Binary classification of simulated signal/background events.
- Logistic Regression, XGBoost and compact MLP models.
- Low-level, high-level and combined feature groups.
- Stratified train/validation/test evaluation with saved prediction artifacts.
- Streamlit inspection of metrics, curves, thresholds and EDA evidence.

Outside scope:

- Reproducing the 2012 Higgs discovery.
- Real detector data, significance estimates or systematic-uncertainty modelling.
- Treating classifier scores as physical probabilities.

## 3. Dataset and data card

The source is the UCI Machine Learning Repository HIGGS dataset (id 280), released under CC BY 4.0. The official archive contains approximately 11 million simulated events, one binary label and 28 inputs. The notebooks download the archive directly from the UCI URL; the repository contains code and artifacts, not the raw dataset.

| Item | Verified value |
|---|---|
| Dataset | UCI HIGGS, dataset 280 |
| Raw source URL | <https://archive.ics.uci.edu/static/public/280/higgs.zip> |
| Schema | 1 label + 28 features |
| Feature groups | 21 low-level + 7 high-level + 28 combined |
| Current saved EDA rows | {total_rows:,} |
| Signal events in saved EDA | {eda['label_counts']['1']:,} ({signal_fraction:.2%}) |
| Background events in saved EDA | {eda['label_counts']['0']:,} ({1 - signal_fraction:.2%}) |
| Missing values | {eda['missing_values']} |
| Non-finite values | {eda['nonfinite_values']} |
| Split | 70% train / 15% validation / 15% test, seed 42 |

The exact raw-file row count and checksum should be recorded from the runtime that downloads the final archive. The saved EDA artifact is the authoritative source for the values above.

## 4. Exploratory data analysis

The EDA artifacts are generated under `outputs/eda/` and are displayed by the Dataset Observatory in Streamlit. The saved report contains feature summaries, class balance, feature distributions, group separation and correlation outputs.

![Class balance](../outputs/eda/class_balance.png)

![Feature distributions](../outputs/eda/feature_distributions.png)

![Feature-group separation](../outputs/eda/feature_group_separation.png)

The strongest saved feature correlations include `m_wbb`–`m_wwbb` (**{eda['top_correlations'][0]['correlation']:.3f}**) and `m_jj`–`m_jjj` (**{eda['top_correlations'][1]['correlation']:.3f}**). The four b-tag features have three unique values each in the saved summary and are strongly zero-inflated, which supports treating them as discrete indicators rather than smooth measurements.

## 5. Experimental protocol

The loader validates column count, numeric values, binary labels and finite feature values. A fixed seed of 42 creates a stratified `higgs-split-v1` partition. StandardScaler is fitted on training rows only for Logistic Regression and the MLP. XGBoost uses the validated feature matrix without requiring standardisation. The operating threshold is selected on validation data for a 50% signal-efficiency target, then frozen before test evaluation.

Every run exports a `metrics.json` envelope and a `test_predictions.csv` file. The dashboard reads those files and never trains a model during startup.

## 6. Models

### Logistic Regression

The linear baseline uses an L-BFGS solver with `C=1.0` and a training-only StandardScaler. It provides a fast, interpretable reference for how much separation can be obtained without nonlinear interactions.

### XGBoost

The tree ensemble uses the project’s documented XGBoost configuration: 150 estimators, maximum depth 6, learning rate 0.1, subsample 0.8, column subsample 0.8 and histogram tree construction.

### Compact MLP

The neural baseline uses hidden widths 64 and 32, BatchNorm1d, ReLU, dropout 0.20, AdamW, learning rate 1e-3, weight decay 1e-4, batch size 1024 and validation-ROC-AUC checkpointing. The saved `mlp_all` run contains genuine per-epoch history with **{history_epochs} epochs**.

## 7. Current saved results

All values in the table below are computed from the current `outputs/*/metrics.json` and prediction files. Accuracy is recomputed at score threshold 0.5 from the saved test predictions.

{table(runs)}

![Classical ROC comparison](../outputs/roc_classical_comparison.png)

![MLP learning curves](../outputs/mlp_learning_curves.png)

The all-feature models are the strongest configuration for each family in the current registry. The all-feature XGBoost and MLP results are close, while the linear model remains lower. High-level features alone outperform low-level features for each family in the current saved run set, but the combined representation performs best.

## 8. Operating-point analysis

The threshold is selected on validation data rather than test data. For the current all-feature runs, the saved operating-point records show:

| Model | Validation threshold | Test signal efficiency | Test background rejection |
|---|---:|---:|---:|
"""
    for run in sorted(all_features, key=lambda item: float(item["roc"]), reverse=True):
        threshold = f"{float(run['threshold']):.4f}" if run["threshold"] is not None else "not recorded"
        signal_efficiency = f"{float(run['signal_efficiency']):.2%}" if run["signal_efficiency"] is not None else "not recorded"
        background_rejection = f"{float(run['background_rejection']):.2%}" if run["background_rejection"] is not None else "not recorded"
        report += f"| {model_label(str(run['model']))} | {threshold} | {signal_efficiency} | {background_rejection} |\n"
    report += f"""

Background rejection is a benchmark operating metric, not a discovery significance. The Streamlit Validation view exposes the same saved test predictions through an interactive threshold workbench.

## 9. Dashboard integration

Launch the application with:

```bash
streamlit run app/main.py
```

The dashboard now uses:

- `outputs/*/metrics.json` and `test_predictions.csv` for experiment comparison and validation plots.
- `outputs/eda/summary.json` and `feature_summary.csv` for Dataset Observatory metrics.
- Saved EDA images for class balance, distributions and correlation views.
- Genuine MLP training history when it exists in the artifact envelope.

The app reports **{len(runs)} valid runs** and skips non-run folders such as `outputs/eda/` and `outputs/model_weights/` during artifact discovery.

![Streamlit dashboard overview](dashboard_overview.png)

## 10. Testing and reproducibility

QA branch commit `50800faa19e1ae4fbd217aa7e9e5ddf120429b54` passed the GitHub Actions workflow: editable installation and the full `python -m pytest -q` suite succeeded on Python 3.11; the Docker image build also passed. See `docs/12_QA_TEST_PLAN.md` for the exact commit and run links. The hosted log does not publish a test count. Full manual notebook execution has not been completed. The raw dataset is downloaded at runtime from UCI, and generated data/model binaries remain outside version control.

## 11. Limitations and validity threats

1. The benchmark is simulated and does not represent real detector data.
2. The benchmark class balance is approximately 53/47, unlike a real rare-event search.
3. The current results use one seed and one recorded benchmark sample.
4. The historical 1M summary was produced by a different pipeline/configuration and has a copied runtime field; it is retained for traceability but not used as the current headline.
5. The current report does not claim that the full 11-million-row training run has completed. The notebooks are configured to download the full source, while the saved artifacts in this workspace record the {total_rows:,}-row benchmark.

## 12. Conclusion

HiggsLens provides a working, artifact-backed comparison of three model families over the HIGGS feature groups. The current validated registry shows that combining the low-level and high-level inputs gives the strongest result for each family, with XGBoost and the compact MLP closely matched on all features and Logistic Regression providing a fast linear baseline. The Streamlit application now presents the same saved metrics and EDA outputs used in this report, preserving traceability from experiment files to the final presentation.

## References

1. P. Baldi, P. Sadowski and D. Whiteson, “Searching for exotic particles in high-energy physics with deep learning,” *Nature Communications*, 5, 4308, 2014. <https://www.nature.com/articles/ncomms5308>
2. D. Whiteson, “HIGGS,” UCI Machine Learning Repository, dataset 280. <https://archive.ics.uci.edu/dataset/280/higgs>
3. T. Chen and C. Guestrin, “XGBoost: A scalable tree boosting system,” KDD, 2016.
4. A. Paszke et al., “PyTorch: An imperative style, high-performance deep learning library,” NeurIPS, 2019.

## Appendix: artifact paths

- `outputs/eda/summary.json`
- `outputs/eda/feature_summary.csv`
- `outputs/eda/feature_distributions.png`
- `outputs/eda/correlation_heatmap.png`
- `outputs/experiments.json`
- `outputs/mlp_*/metrics.json`
- `outputs/logreg_*/metrics.json`
- `outputs/xgboost_*/metrics.json`
"""
    output = REPORTS / "HiggsLens_Final_Report.md"
    output.write_text(report, encoding="utf-8")
    return output


def write_presentation(runs: list[dict[str, object]]) -> Path:
    PRESENTATIONS.mkdir(exist_ok=True)
    best = max(runs, key=lambda item: float(item["roc"]))
    all_features = [run for run in runs if run["group"] == "all"]
    best_all = max(all_features, key=lambda item: float(item["roc"]))
    rows = "\n".join(
        f"| {model_label(str(run['model']))} | `{run['group']}` | {float(run['roc']):.4f} | {float(run['pr']):.4f} | {float(run['brier']):.4f} |"
        for run in sorted(runs, key=lambda item: float(item["roc"]), reverse=True)
    )
    slides = f"""% HiggsLens
% Machine-learning classification of simulated HIGGS events
% Final artifact-backed presentation

# Project in one sentence

- Reproducible benchmark for classifying simulated HIGGS signal versus background events.
- Compares Logistic Regression, XGBoost and a compact PyTorch MLP.
- Dashboard consumes saved artifacts; it never trains at startup.

> **Boundary:** simulated benchmark only — no particle-discovery claim.

# Dataset and scientific question

- UCI HIGGS dataset, approximately 11 million source events.
- 28 inputs: 21 low-level kinematic features + 7 high-level invariant masses.
- Current saved benchmark: {max(run['rows'] for run in runs):,} rows, seed 42, 70/15/15 split.
- Question: do engineered invariant masses add measurable value, and can nonlinear models learn useful structure from raw kinematics?

![Saved class balance](../outputs/eda/class_balance.png)

# End-to-end pipeline

1. Download complete archive directly from UCI.
2. Validate schema, labels and finite values in chunks.
3. Create the fixed stratified split.
4. Fit preprocessing on training rows only.
5. Train models and select threshold on validation data.
6. Export metrics and predictions.
7. Explore the results in Streamlit.

# Exploratory evidence

![Feature distributions](../outputs/eda/feature_distributions.png)

- Saved EDA: {max(run['rows'] for run in runs):,} rows.
- Missing values: 0.
- Non-finite values: 0.
- Strongest saved correlation: `m_wbb` with `m_wwbb`.

# Model design

**Logistic Regression**

- L-BFGS, `C=1.0`, training-only StandardScaler.

**XGBoost**

- 150 trees, depth 6, learning rate 0.1, histogram method.

**Compact MLP**

- Hidden widths 64 → 32.
- BatchNorm + ReLU + dropout 0.20.
- AdamW, learning rate 1e-3, early stopping on validation ROC-AUC.

# Current leaderboard

| Model | Feature group | ROC-AUC | PR-AUC | Brier |
|---|---|---:|---:|---:|
{rows}

**Highest current saved artifact:** {model_label(str(best['model']))} on `{best['group']}` features, ROC-AUC **{float(best['roc']):.4f}**. This is not an approved cross-benchmark model-selection claim; neural evidence remains provisional pending independent reproduction and release decision.

# Feature-group finding

![Classical feature-group comparison](../outputs/roc_classical_comparison.png)

- Low-level-only models are weakest in the current registry.
- High-level invariant masses provide a stronger compact representation.
- Combining all 28 features gives the strongest result for each model family.

# Neural training evidence

![MLP learning curves](../outputs/mlp_learning_curves.png)

- The learning curve is loaded from genuine per-epoch history in `outputs/mlp_all/metrics.json`.
- No synthetic training history is used.

# Operating-point view

- Threshold is selected on validation data for a 50% signal-efficiency target.
- It is frozen before test evaluation.
- Background rejection is reported as a benchmark metric, not a significance estimate.
- The dashboard exposes an interactive threshold workbench over saved test predictions.

# HiggsLens Streamlit dashboard

```bash
streamlit run app/main.py
```

- Overview: artifact registry and current best run.
- Dataset Observatory: saved EDA tables and figures.
- Experiment Comparison: leaderboard, feature groups and runtime.
- Validation: ROC, PR, calibration, learning history and thresholds.
- Methods and Research: protocol, references and limitations.

![Dashboard overview](../reports/dashboard_overview.png)

# Reproducibility and checks

- Artifact contract: `higgs-experiment-v1`.
- Split protocol: `higgs-split-v1`, seed 42.
- Latest QA-branch GitHub Actions: editable install and full pytest workflow passed on Python 3.11; Docker image build passed. See `docs/12_QA_TEST_PLAN.md` for commit and run links. The hosted log does not publish a test count.
- Raw data is downloaded from UCI at runtime and is not committed.

# Limitations and conclusion

- HIGGS events are simulated Monte Carlo events.
- The class balance is not representative of real rare-event searches.
- Current saved metrics cover the {max(run['rows'] for run in runs):,}-row benchmark; the historical 1M summary is kept separate because it used a different pipeline and copied timing values.
- HiggsLens is an engineering and benchmark study, not a physics-discovery analysis.

**Takeaway:** the artifact-backed system works end to end, and the dashboard, report and slides now use the same saved evidence.
"""
    output = PRESENTATIONS / "HiggsLens_Final_Presentation.md"
    output.write_text(slides, encoding="utf-8")
    return output


def main() -> None:
    runs = load_runs()
    report = write_report(runs)
    presentation = write_presentation(runs)
    print("Report source:", report)
    print("Presentation source:", presentation)


if __name__ == "__main__":
    main()
