# HiggsEvent — Interactive Scientific Dashboard

**Owner:** Ronit Singh (Product UI / Streamlit Dashboard).

HiggsEvent provides an interactive scientific workbench for inspecting machine-learning experiments on the simulated UCI HIGGS dataset (Baldi et al., 2014). It visualizes held-out evaluation curves, compares Logistic Regression, XGBoost and compact MLP runs across feature groups, and includes a **Model Simulation** view for scoring new events with available tuned model weights.

---

## 1. Quickstart & Launch Instructions

### Prerequisites

Ensure the virtual environment is activated and project dependencies are installed:

```bash
# Install core dependencies
pip install -r requirements.txt

# Install repository in editable mode (so app/ imports work)
pip install -e .
```

### Launch the Dashboard

Run from the repository root:

```bash
python -m pip install -e ".[dev]"
streamlit run streamlit_app.py
```

This is the canonical command for local use, Docker and Streamlit Cloud.
`app/main.py` implements the application and `streamlit_app.py` at the repository
root is the only launcher. Startup and rescans never generate or overwrite experiments.
Missing results produce an empty state; invalid runs produce diagnostics.
Do not use `scripts/generate_sample_artifacts.py` as benchmark evidence: it
creates synthetic predictions and hard-coded metrics.

---

## 2. Navigation & Application Architecture

HiggsEvent is organized into seven focused scientific sections:

| Section | What it shows |
|---|---|
| **1. Overview** | Project objectives, dataset parameters, artifact registry discovery status, verified reproduction commands. |
| **2. Physics Explorer** | Standard Model electroweak symmetry breaking, exotic-Higgs benchmark decay kinematics, explicit project boundaries. |
| **3. Dataset Observatory** | Searchable 28-feature dictionary, signal vs background distribution explorer, cross-feature Pearson correlation matrix, empirical class balance observatory. |
| **4. Experiment Comparison** | Model comparison leaderboard, multi-group representation learning analysis (`low_level` vs `high_level` vs `all`), runtime vs performance trade-off. |
| **5. Validation** | ROC curves, Precision-Recall curves, calibration reliability diagrams, genuine MLP training history, interactive threshold operating-point workbench. |
| **6. Methods and Research** | `higgs-split-v1` leakage-free protocol, neural and linear model configurations, academic citations, research limitations. |
| **7. Model Simulation** | Upload a CSV of events and score them with trained logistic regression, XGBoost, and Wide MLP models; single-event prediction with sliders; score distribution histograms. |

---

## 3. Artifact Discovery & Schema Contract

HiggsEvent dynamically discovers run folders beneath `outputs/` (or a custom path entered in the sidebar). A run folder is classified as usable only when it contains both required files complying with `higgs-experiment-v1`:

### Required Files

- **`metrics.json`** — Metadata dictionary including schema version, run ID, model architecture, feature group, split seed, training configuration, runtime, and validation/test metrics.
- **`test_predictions.csv`** — Held-out test predictions containing `row_index`, `class_label` ($y \in \{0, 1\}$), and predicted class-1 probability `score` ($\in [0.0, 1.0]$).

### Validation Rules

The dashboard validates every discovered artifact before displaying it:

- `schema_version` must equal `higgs-experiment-v1`.
- `run_id` must be a non-empty string.
- `model` must be a dictionary containing `name`.
- `features.group` must be `low_level`, `high_level`, or `all`; `features.count` must be a positive integer; `features.names` must be a list.
- `split.seed` must be an integer.
- `evaluation.test` must contain `roc_auc`, `pr_auc`, and `brier_score` as finite numeric values.
- `test_predictions.csv` must contain the required columns, binary labels, and probability scores bounded in $[0, 1]`.

### Honest Empty State Policy

If `outputs/` is missing or contains no valid runs, HiggsEvent displays an empty state with commands to reproduce runs. Cloud startup never manufactures results. The legacy synthetic fixture generator is isolated under `artifacts/synthetic_demo/` and is not called by the dashboard.

---

## 4. Reproducing Experiment Runs

To populate the dashboard with genuine benchmark runs, execute the verified CLI scripts:

```bash
# 1. Extract a reproducible subset (e.g. 250,000 events)
python -m higgs_event_lab.data \
    --source data/raw/HIGGS.csv.gz \
    --output data/processed/higgs_sample_250k.csv.gz \
    --rows 250000

# 2. Train Logistic Regression baselines across all 3 feature groups
python -m higgs_event_lab.classical \
    --data data/processed/higgs_sample_250k.csv.gz \
    --all-groups \
    --output-dir outputs

# 3. Train Compact MLP neural networks across all 3 feature groups
python -m higgs_event_lab.neural \
    --data data/processed/higgs_sample_250k.csv.gz \
    --all-groups \
    --output-dir outputs
```

Or run the full Colab/Kaggle benchmark notebook (`notebooks/03_full/00_full_higgs_colab.ipynb`) to train on 1M events and export tuned models to `outputs/model_weights/`.

---

## 5. Model Simulation

The **Model Simulation** section lets you use the trained models on new data:

- **Batch CSV Upload:** Upload a CSV containing the 28 HIGGS feature columns. Each row is scored by every loaded model (logistic regression, XGBoost, Wide MLP). Download the predictions as a CSV.
- **Single-Event Prediction:** Adjust feature values with sliders and compare every model's signal probability side by side.
- **Score Distributions:** View histogram distributions of predicted probabilities across held-out test runs.

Models are loaded from `outputs/model_weights/` (git-ignored). If weights are missing, the view falls back to a demo mode showing held-out test score distributions from existing artifacts.

---

## 6. Automated Tests

Targeted unit tests for the dashboard loader, schema validation, and threshold calculations are located in `tests/unit/`:

```bash
# Run dashboard unit tests
pytest tests/unit/test_app_loader.py tests/unit/test_app_metrics.py -v

# Run entire repository test suite
pytest
```

---

## Loader

`app/loader.py` is the only loader. It owns experiment discovery, nested
`higgs-experiment-v1` validation and the saved-EDA reader. The old flat-schema
loader under `app/utils/` and the unused `app/components/` were removed in issue #21.
Flat summary JSON/CSV records are unsupported as dashboard experiment evidence.
Curves are computed from saved held-out predictions; optional metrics remain unavailable
when absent. Run IDs must be unique. Compare sources, sample sizes and splits before
interpreting differences between runs.
