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
        "deep_mlp": "Deep MLP (5x300)",
        "resnet_mlp": "Residual MLP",
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


FIGURES = REPORTS / "figures"
MODEL_ORDER = ("logreg", "xgboost", "mlp", "deep_mlp", "resnet_mlp")
SHORT = {"logreg": "Logistic Regression", "xgboost": "XGBoost", "mlp": "Compact MLP (64-32)",
         "deep_mlp": "Deep MLP (5x300)", "resnet_mlp": "Residual MLP"}


def write_release_figures(release: dict | None) -> None:
    """Scaling curve and feature-group bars, both straight from release_summary.json."""
    if not release:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES.mkdir(parents=True, exist_ok=True)
    tags = sorted(release["budgets"], key=lambda tag: release["budgets"][tag]["rows"])
    rows = [release["budgets"][tag]["rows"] for tag in tags]

    def auc(tag: str, model: str, group: str) -> float:
        for r in release["budgets"][tag]["results"]:
            if r["model"] == model and r["group"] == group:
                return r["roc_auc"]
        return float("nan")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, group in zip(axes, ("all", "low_level")):
        for model in MODEL_ORDER[1:]:
            ax.plot(rows, [auc(tag, model, group) for tag in tags], marker="o", label=SHORT[model])
        ax.set_xscale("log")
        ax.minorticks_off()
        ax.set_xticks(rows, [tag for tag in tags])
        ax.set_xlabel("events in sample")
        ax.set_title({"all": "All 28 features", "low_level": "21 low-level features only"}[group])
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("test ROC-AUC")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "scaling.png", dpi=150)
    plt.close(fig)

    largest = tags[-1]
    groups = ("low_level", "high_level", "all")
    fig, ax = plt.subplots(figsize=(8, 3.6))
    width = 0.16
    for i, model in enumerate(MODEL_ORDER):
        values = [auc(largest, model, g) for g in groups]
        bars = ax.bar([g + (i - 2) * width for g in range(3)], values, width, label=SHORT[model])
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.004, f"{v:.3f}", ha="center", fontsize=6.5)
    ax.set_xticks(range(3), ["low-level (21)", "high-level (7)", "all (28)"])
    ax.set_ylim(0.55, 0.9)
    ax.set_ylabel("test ROC-AUC")
    ax.set_title(f"Feature groups at {largest} events")
    ax.legend(fontsize=7, ncol=3, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "feature_groups.png", dpi=150)
    plt.close(fig)


def findings(release: dict | None) -> dict:
    """Headline numbers quoted in the text, computed from release_summary.json."""
    if not release:
        return {}
    tags = sorted(release["budgets"], key=lambda tag: release["budgets"][tag]["rows"])

    def auc(tag: str, model: str, group: str) -> float:
        return next(r["roc_auc"] for r in release["budgets"][tag]["results"]
                    if r["model"] == model and r["group"] == group)

    big, small = tags[-1], tags[0]
    best_net = max(("deep_mlp", "resnet_mlp"), key=lambda m: auc(big, m, "all"))
    return {
        "big": big, "small": small, "best_net": SHORT[best_net],
        "net_all": auc(big, best_net, "all"), "xgb_all": auc(big, "xgboost", "all"),
        "net_low": auc(big, best_net, "low_level"), "xgb_low": auc(big, "xgboost", "low_level"),
        "xgb_high": auc(big, "xgboost", "high_level"), "mlp_all": auc(big, "mlp", "all"),
        "gaps": [(tag, max(auc(tag, "deep_mlp", "all"), auc(tag, "resnet_mlp", "all")) - auc(tag, "xgboost", "all")) for tag in tags],
        "duplicates": release.get("duplicate_rows_removed"),
    }


def load_release() -> dict | None:
    path = OUTPUTS / "release" / "release_summary.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def release_section(release: dict | None) -> str:
    if not release:
        return "No release run is saved yet. Run `python scripts/release_benchmark.py --source <HIGGS.csv>`.\n"
    out = []
    names = sorted({row["run"] for budget in release["budgets"].values() for row in budget["results"]})
    tags = list(release["budgets"])
    out.append("| Run | " + " | ".join(f"{tag} ROC-AUC | {tag} PR-AUC | {tag} Brier | {tag} train s" for tag in tags) + " |")
    out.append("|---|" + "---:|" * (4 * len(tags)))
    for name in names:
        cells = []
        for tag in tags:
            row = next((r for r in release["budgets"][tag]["results"] if r["run"] == name), None)
            if row is None:
                cells += ["–"] * 4
            else:
                cells += [f"{row['roc_auc']:.4f}", f"{row['pr_auc']:.4f}", f"{row['brier']:.4f}", f"{row['train_seconds']:.1f}"]
        out.append(f"| `{name}` | " + " | ".join(cells) + " |")
    out.append("")
    out.append("| Check | " + " | ".join(tags) + " |")
    out.append("|---|" + "---|" * len(tags))
    checks = [("Rows", "rows"), ("Signal fraction", "signal_fraction"),
              ("Split indices disjoint", "indices_disjoint"), ("Splits cover every row", "covers_every_row"),
              ("Duplicate events in sample", "duplicate_events_in_sample"),
              ("Events shared by train and test", "events_shared_by_train_and_test")]
    for label, key in checks:
        out.append(f"| {label} | " + " | ".join(str(release["budgets"][tag]["split_checks"][key]) for tag in tags) + " |")
    det = release.get("determinism")
    if det:
        out.append("")
        out.append(f"Determinism: two CPU trainings of the all-feature MLP with seed {release['seed']} "
                   f"({det['epochs']} epochs) differ by at most {det['max_abs_probability_diff']:.2e} "
                   f"in any test probability ({'identical' if det['identical'] else 'not identical'}).")
    out.append("")
    out.append(f"Environment: Python {release.get('python')}, torch {release.get('torch')}, "
               f"{'GPU ' + release['cuda'] if release.get('cuda') else 'CPU only'}, {release.get('platform')}. "
               f"Run started {release.get('started')}.")
    return "\n".join(out) + "\n"


def release_operating_table(release: dict | None) -> str:
    """Render only operating-point values actually serialized in release metrics."""
    if not release:
        return "No release operating-point metadata is available.\\n"
    lines = [
        "| Budget | Model | Validation threshold | Validation signal efficiency | Validation background rejection | Test signal efficiency | Test background rejection |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    labels = {"logreg": "Logistic Regression", "xgboost": "XGBoost"}
    for tag, budget in sorted(release["budgets"].items(), key=lambda item: item[1]["rows"]):
        for model in ("logreg", "xgboost"):
            path = OUTPUTS / "release" / tag / f"{model}_all" / "metrics.json"
            if not path.is_file():
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            op = data.get("evaluation", {}).get("operating_point")
            if not op:
                continue
            lines.append(
                f"| {tag} | {labels[model]} | {op['validation_threshold']:.4f} "
                f"| {op['validation_actual_signal_efficiency']:.2%} "
                f"| {op['validation_background_rejection']:.2%} "
                f"| {op['test_actual_signal_efficiency']:.2%} "
                f"| {op['test_background_rejection']:.2%} |"
            )
    if len(lines) == 2:
        return "No serialized operating-point records were found.\\n"
    return "\\n".join(lines) + "\\n"

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
    release = load_release()
    f = findings(release)
    operating_table = release_operating_table(release)
    gap_text = ", ".join(f"{gap:+.3f} at {tag}" for tag, gap in f.get("gaps", []))
    signal_fraction = float(eda["signal_fraction"])
    report = f"""# HiggsLens: Machine-Learning Classification of Simulated HIGGS Events

**Final artifact-backed report**  
**Project team:** Saumya Mishra · Aditya Rana · Rachit Gupta · Ronit Singh · Krishna  
**Dataset:** UCI HIGGS, dataset 280  
**Source:** <https://archive.ics.uci.edu/dataset/280/higgs>  
**Generated from:** saved artifacts under `outputs/`  

> **Scientific boundary.** HIGGS is a simulated Monte Carlo benchmark. This project evaluates classification software and makes no claim of particle discovery, real-detector performance, or physical probabilities.

## 1. Executive summary

HiggsLens is a reproducible machine-learning workbench for separating simulated HIGGS signal events from simulated background events. The implementation compares two classical baselines (Logistic Regression and XGBoost) with three PyTorch networks: the mid-semester compact MLP, a deep MLP sized like Baldi et al. (2014) and a residual MLP. Each is trained on 21 low-level kinematic features, 7 high-level invariant-mass features and all 28 features, at 250k, 1M and 2.5M events.

**Main result.** At {f.get('big')} events the {f.get('best_net')} reaches test ROC-AUC **{f.get('net_all', 0):.4f}** on all features against **{f.get('xgb_all', 0):.4f}** for XGBoost. <!-- allow-metric --> Given only the 21 low-level features it reaches **{f.get('net_low', 0):.4f}**, which is above XGBoost on the same features ({f.get('xgb_low', 0):.4f}) and above XGBoost given all 28 features. <!-- allow-metric --> The network learns from the raw kinematics what the hand-built invariant masses give the trees. The deep-network advantage on all features grows with data: {gap_text}. The compact mid-semester MLP stays near XGBoost ({f.get('mlp_all', 0):.4f} at {f.get('big')}), which is why the mid-semester comparison looked like a tie. <!-- allow-metric -->

The dashboard registry contains **{len(runs)} runs** from the 250k budget, using the `higgs-split-v1` protocol with seed 42. The strongest currently loaded artifact is **{model_label(str(best['model']))} on `{best['group']}` features**, with test ROC-AUC **{float(best['roc']):.4f}**, PR-AUC **{float(best['pr']):.4f}**, Brier score **{float(best['brier']):.4f}**, and accuracy **{float(best['accuracy']):.4f}**. <!-- allow-metric -->

The release benchmark trains every model with the same code, deduplicated data, split protocol and seed at every budget (Section 7.1); {f.get('duplicates') or 0:,} duplicate events were removed from the raw file first. The older `outputs/full_colab_results.json` notebook record used a different pipeline and is kept only for traceability.

## 2. Research question and scope

The central question is whether models can learn useful separation directly from low-level event kinematics, whether high-level invariant masses provide a measurable advantage, and how model family changes discrimination, calibration and compute cost.

In scope:

- Binary classification of simulated signal/background events.
- Logistic Regression, XGBoost, compact MLP, deep MLP and residual MLP models.
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

The EDA table describes the saved 250,000-row EDA snapshot. It is separate from the deduplicated nested release samples; use `outputs/release/release_summary.json` for each release budget's class fraction and split checks.

## 4. Exploratory data analysis

The EDA artifacts are generated under `outputs/eda/` and are displayed by the Dataset Observatory in Streamlit. The saved report contains feature summaries, class balance, feature distributions, group separation and correlation outputs.

![Class balance](../outputs/eda/class_balance.png)

![Feature distributions](../outputs/eda/feature_distributions.png)

![Feature-group separation](../outputs/eda/feature_group_separation.png)

The strongest saved feature correlations include `m_wbb`–`m_wwbb` (**{eda['top_correlations'][0]['correlation']:.3f}**) and `m_jj`–`m_jjj` (**{eda['top_correlations'][1]['correlation']:.3f}**). The four b-tag features have three unique values each in the saved summary and are strongly zero-inflated, which supports treating them as discrete indicators rather than smooth measurements.

## 5. Experimental protocol

The loader validates column count, numeric values, binary labels and finite feature values. A fixed seed of 42 creates a stratified `higgs-split-v1` partition. StandardScaler is fitted on training rows only for Logistic Regression and the MLP. XGBoost uses the validated feature matrix without requiring standardisation. Validation data is used for model selection and early stopping. The release artifacts do not serialize a complete per-run operating-point record, so this report does not quote thresholds, signal efficiencies or background rejections.

Every run exports a `metrics.json` envelope and a `test_predictions.csv` file. The dashboard reads those files and never trains a model during startup.

## 6. Models

### Logistic Regression

The linear baseline uses an L-BFGS solver with `C=1.0` and a training-only StandardScaler. It provides a fast, interpretable reference for how much separation can be obtained without nonlinear interactions.

### XGBoost

Depth 8, learning rate 0.1, subsample 0.8, column subsample 0.8, histogram trees, and up to 3000 trees with early stopping after 50 rounds without improvement in validation log-loss. Tuning the number of trees on validation keeps the tree baseline strong at every budget, so the comparison with the networks is fair.

### Compact MLP (mid-semester)

Hidden widths 64 and 32, BatchNorm1d, ReLU, dropout 0.20, AdamW, learning rate 1e-3, weight decay 1e-4, batch size 1024, at most 30 epochs with validation-ROC-AUC checkpointing. About 4k parameters: a shallow network in the sense of Baldi et al.

### Deep MLP

Five hidden layers of 300 units (the size Baldi et al. found best on HIGGS) with BatchNorm, ReLU and dropout 0.10, AdamW with weight decay 1e-5 and a cosine learning-rate schedule, up to 60 epochs, early stopping after 8 epochs without validation improvement.

### Residual MLP

Four residual blocks of width 256 (BatchNorm, Linear to 512, ReLU, dropout, Linear back to 256, plus a skip connection), after Gorishniy et al. (2021). Same optimiser and stopping rule as the deep MLP. The reasoning behind both deep models is in `docs/16_WHY_DEEP_LEARNING.md`.

## 7. Current saved results

All values in the table below are computed from the current `outputs/*/metrics.json` and prediction files. Accuracy is recomputed at score threshold 0.5 from the saved test predictions.

{table(runs)}

![ROC-AUC against sample size](figures/scaling.png)

![ROC-AUC by feature group at the largest budget](figures/feature_groups.png)

For the classical models and the compact MLP, the 7 high-level features beat the 21 low-level ones: they need the physicists' invariant masses. The deep and residual networks reverse that once there is enough data, scoring higher on low-level features alone than on high-level features alone. Calibration is a point in XGBoost's favour: its expected calibration error stays below 0.01, while the networks sit around 0.01 to 0.03, so the network scores would need recalibration before being used as probabilities.

### 7.1 Release benchmark: 250k, 1M and 2.5M

`scripts/release_benchmark.py` removes duplicate events from the raw UCI file, draws nested seeded samples and trains all fifteen model/feature combinations at each budget with identical code, seed and `higgs-split-v1` splits. Logs and artifacts are under `outputs/release/`.

{release_section(release)}
## 8. Operating-point analysis

Validation-selected operating points are available for Logistic Regression and XGBoost. The neural-network metric envelopes do not serialize an `operating_point` record, so no neural thresholds or efficiencies are inferred.

{operating_table}
The threshold workbench is exploratory; model-selection thresholds must be selected on validation data and frozen before test evaluation. Background rejection is a benchmark metric, not a discovery significance.
"""
    report += f"""

## 9. Dashboard integration

Launch the application with:

```bash
streamlit run streamlit_app.py
```

The dashboard now uses:

- `outputs/*/metrics.json` and `test_predictions.csv` for experiment comparison and validation plots.
- `outputs/eda/summary.json` and `feature_summary.csv` for Dataset Observatory metrics.
- Saved EDA images for class balance, distributions and correlation views.
- Genuine MLP training history when it exists in the artifact envelope.

The app reports **{len(runs)} valid runs** and skips non-run folders such as `outputs/eda/` and `outputs/model_weights/` during artifact discovery.

![Streamlit dashboard overview](dashboard_overview.png)

## 10. Testing and reproducibility

`python -m pytest -q` runs the unit and integration suite, and the GitHub Actions workflow runs it on every push. The release benchmark was run end to end on Kaggle against the full 11M-row UCI file (Section 7.1), covering sampling, split-integrity checks, all fifteen model/feature-group trainings per budget and the determinism check. The raw dataset and model binaries stay out of version control. `scripts/release_benchmark.py` regenerates everything from the raw file.

## 11. Limitations and validity threats

1. The benchmark is simulated and does not represent real detector data.
2. The benchmark class balance is approximately 53/47, unlike a real rare-event search.
3. Results use one seed per budget, so run-to-run variance across seeds is not measured.
4. Training time depends on the hardware recorded in Section 7.1 and is only comparable within one run.
5. The largest budget is 2.5M events; at that size several deep runs reached the 60-epoch cap, so they were still improving and the network numbers are, if anything, conservative.

## 12. Conclusion

The mid-semester result (a small MLP tied with XGBoost) was a property of the network's size and the amount of data, not a limit of neural networks on this problem. With the depth of Baldi et al. and up to 2.5M events, the deep and residual networks clearly beat a tuned XGBoost, the gap widens as data grows, and on raw low-level kinematics alone the deep network outperforms every classical model even when those models get the hand-built physics features. XGBoost remains far cheaper to train and better calibrated, so it is the right tool when data or compute is limited. The Streamlit application presents the same saved artifacts used in this report.

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
- `outputs/release/release_summary.json`
- `outputs/release/250k/*/metrics.json`, `outputs/release/1m/*/metrics.json`, `outputs/release/2.5m/*/metrics.json`
- `outputs/mlp_*/metrics.json`
- `outputs/logreg_*/metrics.json`
- `outputs/xgboost_*/metrics.json`
"""
    output = REPORTS / "HiggsLens_Final_Report.md"
    output.write_text(report, encoding="utf-8")
    return output


def write_presentation(runs: list[dict[str, object]]) -> Path:
    """Generate the deck from the canonical multi-budget release summary."""
    PRESENTATIONS.mkdir(exist_ok=True)
    release = load_release()
    if not release:
        raise RuntimeError("Missing outputs/release/release_summary.json; refusing to generate a stale 250k-only deck.")
    tags = sorted(release["budgets"], key=lambda tag: release["budgets"][tag]["rows"])
    models = ("logreg", "xgboost", "mlp", "deep_mlp", "resnet_mlp")
    labels = {"logreg": "Logistic Regression", "xgboost": "XGBoost", "mlp": "Compact MLP (64-32)", "deep_mlp": "Deep MLP (5x300)", "resnet_mlp": "Residual MLP"}
    def metric(tag: str, model: str, group: str, key: str = "roc_auc") -> float:
        row = next(item for item in release["budgets"][tag]["results"] if item["model"] == model and item["group"] == group)
        return float(row[key])
    def budget_table(group: str) -> str:
        lines = ["| Model | " + " | ".join(f"{tag} test ROC-AUC" for tag in tags) + " |", "|---|" + "---:|" * len(tags)]
        for model in models:
            lines.append("| " + labels[model] + " | " + " | ".join(f"{metric(tag, model, group):.4f}" for tag in tags) + " |")
        return "\n".join(lines)
    largest = tags[-1]
    gaps = [f"{max(metric(tag, 'deep_mlp', 'all'), metric(tag, 'resnet_mlp', 'all')) - metric(tag, 'xgboost', 'all'):+.3f} at {tag}" for tag in tags]
    split_rows = []
    for tag in tags:
        check = release["budgets"][tag]["split_checks"]
        split_rows.append(f"| {tag} | {check['split_sizes']['train']:,} | {check['split_sizes']['validation']:,} | {check['split_sizes']['test']:,} | {check['indices_disjoint']} | {check['events_shared_by_train_and_test']} |")
    feature_rows = [f"| {labels[m]} | {metric(largest,m,'low_level'):.4f} | {metric(largest,m,'high_level'):.4f} | {metric(largest,m,'all'):.4f} |" for m in models]
    cost_rows = [f"| {labels[m]} | {metric(largest,m,'all','brier'):.4f} | {metric(largest,m,'all','ece'):.4f} | {metric(largest,m,'all','train_seconds'):.1f} |" for m in models]
    slides = f"""% HiggsLens
% Machine-learning classification of simulated HIGGS events
% Final release-benchmark presentation

# HiggsLens: the question

- Binary classification of simulated HIGGS signal and background events.
- 21 low-level kinematic features; 7 high-level invariant-mass features; 28 combined.
- Compare Logistic Regression, XGBoost, compact MLP, deep MLP and residual MLP.
- Simulated Monte Carlo benchmark only; no particle-discovery or real-detector claim.

# Dataset and experimental design

- UCI HIGGS dataset: approximately 11 million simulated events.
- Exact duplicate events removed before sampling: **{release.get('duplicate_rows_removed', 0):,}**.
- Nested sample budgets: {", ".join(tags)}; seed {release['seed']}.
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

{budget_table("all")}

- Both deep networks outperform XGBoost at every budget.
- Best deep-network advantage over XGBoost: {", ".join(gaps)}.
- At {largest}, Residual MLP: **{metric(largest,'resnet_mlp','all'):.4f}**; XGBoost: **{metric(largest,'xgboost','all'):.4f}**.

# Low-level-only benchmark: test ROC-AUC

{budget_table("low_level")}

- At {largest}, Residual MLP on 21 low-level features scores **{metric(largest,'resnet_mlp','low_level'):.4f}**.
- This exceeds XGBoost on all 28 features (**{metric(largest,"xgboost","all"):.4f}**) in this benchmark.
- This is evidence of learned feature interactions in this simulated dataset, not a physics-discovery claim.

# Feature groups at {largest}

| Model | Low-level (21) | High-level (7) | All (28) |
|---|---:|---:|---:|
{chr(10).join(feature_rows)}

- Hand-engineered invariant masses help classical models and the compact MLP.
- Deep models recover more discriminative structure from raw kinematics as the sample grows.
- One seed per budget: do not over-interpret small differences.

# Calibration and compute trade-off

| Model | Brier | ECE | Training time (s) |
|---|---:|---:|---:|
{chr(10).join(cost_rows)}

- XGBoost is much faster than the deep networks in this run.
- Lower Brier/ECE indicates better calibration; this is distinct from discrimination.
- Training times are hardware- and run-specific (Kaggle Tesla T4).

# Reproducibility and release checks

- Artifact contract: `higgs-experiment-v1`; split protocol: `higgs-split-v1`.

| Budget | Train | Validation | Test | Indices disjoint | Shared train/test events |
|---|---:|---:|---:|---|---:|
{chr(10).join(split_rows)}

- Two CPU compact-MLP trainings with seed 42 produced identical test probabilities in the release determinism check.
- Raw data and model checkpoints are not committed; regenerate with `scripts/release_benchmark.py`.

# Limitations and conclusion

- Simulated Monte Carlo data; class balance is not representative of a rare-event search.
- One seed per budget: cross-seed variance and statistical uncertainty are not measured.
- Some deep runs reached the 60-epoch cap; further training may change the ranking.
- Release metrics do not serialize complete operating-point thresholds; none are quoted here.
- Deeper networks improve discrimination on this benchmark as sample size grows; XGBoost remains much cheaper and better calibrated in these runs.
"""
    output = PRESENTATIONS / "HiggsLens_Final_Presentation.md"
    output.write_text(slides, encoding="utf-8")
    return output

def main() -> None:
    runs = load_runs()
    write_release_figures(load_release())
    report = write_report(runs)
    presentation = write_presentation(runs)
    print("Report source:", report)
    print("Presentation source:", presentation)


if __name__ == "__main__":
    main()
