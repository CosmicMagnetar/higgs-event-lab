# HiggsEvent: Particle Event Classification Benchmark

**Classify simulated particle collision events as Higgs signal or background using Logistic Regression, XGBoost, and compact neural networks — with a live interactive Streamlit dashboard.**

> **Scientific Boundary:** All collision events analysed here are Monte Carlo simulations from the UCI HIGGS dataset (Baldi et al., 2014). Model outputs are classification scores on synthetic benchmark events. This project does **not** claim physical particle detection, quantum field simulation, or any real detector analysis.

[![Tests](https://img.shields.io/badge/tests-105%20passing-brightgreen)](tests/)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](pyproject.toml)
[![Streamlit](https://img.shields.io/badge/dashboard-HiggsEvent-ff4b4b)](streamlit_app.py)

---

## Team & Ownership

| Name | Role | GitHub | Primary Tasks |
|---|---|---|---|
| **Krishna** | Team Lead | [`CosmicMagnetar`](https://github.com/CosmicMagnetar) | Split protocol, integration, cloud deployment, presentation |
| **Saumya** | Docs & QA Lead | [`Saumya-Mishra-536`](https://github.com/Saumya-Mishra-536) | Physics background, setup guide, test plan, report review |
| **Aditya Rana** | Data & Classical ML | [`A1B2C3D4E5F6G7H8I9J0164-hack`](https://github.com/A1B2C3D4E5F6G7H8I9J0164-hack) | Data loader, EDA, Logistic Regression, XGBoost baselines |
| **Rachit** | Deep Learning | [`Rachitt19`](https://github.com/Rachitt19) | Compact MLP (PyTorch), BatchNorm/Dropout, neural benchmarks |
| **Ronit** | Dashboard & UI | [`ronithi27`](https://github.com/ronithi27) | Streamlit app, artifact loader, visualisations, cloud entrypoint |

---

## System Architecture

```mermaid
flowchart TD
    subgraph Input ["1 — Data Ingestion"]
        A[("UCI HIGGS Archive\n11 M rows, 2.6 GB")] -->|auto-download in notebooks| B["data/raw/HIGGS.csv.gz"]
        B -->|chunked reader 50k rows| C["Schema Validator\nvalidate_raw_chunk()"]
        C -->|clean subset| D["data/processed/\nhiggs_sample_250k.csv.gz\nhiggs_sample_1m.csv.gz"]
    end

    subgraph Split ["2 — higgs-split-v1 Protocol"]
        D -->|seed=42, stratified| E["Train 70%\n(175k / 700k)"]
        D --> F["Validation 15%\n(37.5k / 150k)"]
        D --> G["Held-out Test 15%\n(37.5k / 150k)"]
    end

    subgraph Models ["3 — Benchmark Models"]
        E --> H["Logistic Regression\nL-BFGS · C=1.0\nStandardScaler inside Pipeline"]
        E --> I["XGBoost\n600 trees · depth 7\nlr=0.06 · subsample=0.9"]
        E --> J["Wide MLP\n256-128-64 · BatchNorm1d\nDropout 0.30 · AdamW"]
        F -->|early stopping\noperating threshold| K["Validation Tuning"]
        K --> H & I & J
    end

    subgraph Eval ["4 — Single Frozen Test Evaluation"]
        G --> L["Held-out Scoring"]
        H & I & J --> L
        L --> M["ROC-AUC · PR-AUC\nBrier Score · Accuracy\nOperating Point @50% Sig. Eff."]
        M --> N[["higgs-experiment-v1\nmetrics.json + test_predictions.csv"]]
    end

    subgraph App ["5 — Interactive Dashboard"]
        N -->|auto-discovered| O["Streamlit App\nstreamlit_app.py"]
        O --> P["Leaderboard · ROC/PR Curves\nReliability Diagrams · Feature Groups"]
    end
```

---

## Feature Groups

The HIGGS dataset has **28 features** split into two groups:

| Group | Count | Description |
|---|---|---|
| `low_level` | 21 | Raw kinematic detector measurements (momenta, pseudorapidity, azimuthal angles) |
| `high_level` | 7 | Physicist-derived composite invariant masses (m_jj, m_jjj, m_lv, m_jlv, m_bb, m_wbb, m_wwbb) |
| `all` | 28 | Combined — both groups concatenated |

---

## Benchmark Results

### 1M reference benchmark — separate pipeline

The saved 1M benchmark uses a 1,000,000-row sequential-head sample and a stratified 70/15/15 split (700,000 train, 150,000 validation, 150,000 test). These results are retained separately from the current 250k registry because the model configurations and pipeline differ. The wide-MLP result remains provisional pending independent neural reproduction and the release decision. Source: `outputs/full_colab_results.json` and `docs/14_RESULTS_RECONCILIATION.md`.

| Model | Feature Group | Test ROC-AUC | Test PR-AUC | Brier Score | Accuracy | Training time (s) |
|---|---|---:|---:|---:|---:|---:|
| Wide MLP (256-128-64) | all (28) | 0.8399 | 0.8539 | 0.1630 | 0.7565 | 395.1 |
| XGBoost | all (28) | 0.8271 | 0.8415 | 0.1693 | 0.7451 | 35.1 |
| Logistic Regression | all (28) | 0.6834 | 0.6832 | 0.2233 | 0.6406 | 2.6 |

### Current 250k saved-artifact registry

The current 250,000-row registry contains nine runs with seed 42 and the `higgs-split-v1` 70/15/15 split. Metrics below are copied from the per-run `metrics.json` artifacts. The compact-MLP rows describe the saved artifacts but remain provisional until neural reproduction and the release decision are complete.

| Model | Feature Group | Test ROC-AUC | Test PR-AUC | Brier Score | Accuracy |
|---|---|---:|---:|---:|---:|
| XGBoost | all (28) | 0.8123 | 0.8266 | 0.1765 | 0.7334 |
| Compact MLP | all (28) | 0.8121 | 0.8270 | 0.1767 | 0.7322 |
| XGBoost | high_level (7) | 0.7848 | 0.8026 | 0.1882 | 0.7046 |
| Compact MLP | high_level (7) | 0.7832 | 0.8020 | 0.1892 | 0.7044 |
| Compact MLP | low_level (21) | 0.7257 | 0.7366 | 0.2113 | 0.6650 |
| XGBoost | low_level (21) | 0.7092 | 0.7233 | 0.2165 | 0.6512 |
| Logistic Regression | all (28) | 0.6854 | 0.6817 | 0.2231 | 0.6439 |
| Logistic Regression | high_level (7) | 0.6518 | 0.6217 | 0.2315 | 0.6295 |
| Logistic Regression | low_level (21) | 0.5960 | 0.6222 | 0.2418 | 0.5680 |

> **Interpretation:** In the current 250k registry, all-feature XGBoost and compact MLP artifacts are close in test ROC-AUC. The MLP evidence is provisional; do not treat this table as an approved cross-model release ranking. The historical 1M benchmark is a different configuration and is not ranked directly against these 250k runs.

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/CosmicMagnetar/higgs-event-lab.git
cd higgs-event-lab
python -m venv .venv && source .venv/bin/activate
python -m pip install -e ".[dev]"
```

### 2. Download the Dataset (auto-handled by notebooks)

The notebooks download the **complete UCI HIGGS dataset** on first run. The archive contains approximately **11 million events**, not a pre-trimmed sample. The 250k and 1M files are generated later from this complete raw file for memory-safe benchmark runs.

Official links:

- Dataset page: <https://archive.ics.uci.edu/dataset/280/higgs>
- Complete archive: <https://archive.ics.uci.edu/static/public/280/higgs.zip>

To download the complete dataset manually:

```bash
# Option A — Official UCI zip (recommended)
curl -L https://archive.ics.uci.edu/static/public/280/higgs.zip -o /tmp/higgs.zip
unzip /tmp/higgs.zip "*.csv.gz" -d data/raw/

# Option B — Using the project download script
python scripts/download_data.py --output data/raw/HIGGS.csv.gz
```

> **Expected complete file:** `data/raw/HIGGS.csv.gz` — ~2.6 GB compressed, headerless, 29 columns, about 11,000,000 rows. This is the whole dataset used by the project.

### 3. Build the Processed Sample

```bash
# 250k mid-semester sample (fast, ~10s)
PYTHONPATH=src python -m higgs_event_lab.data \
    --source data/raw/HIGGS.csv.gz \
    --output data/processed/higgs_sample_250k.csv.gz \
    --rows 250000

# 1M full sample (slower, ~45s)
PYTHONPATH=src python -m higgs_event_lab.data \
    --source data/raw/HIGGS.csv.gz \
    --output data/processed/higgs_sample_1m.csv.gz \
    --rows 1000000
```

### 4. Train Models

```bash
# Train Logistic Regression + XGBoost across all feature groups
PYTHONPATH=src python -m higgs_event_lab.classical \
    --data data/processed/higgs_sample_250k.csv.gz \
    --model-type both \
    --all-groups \
    --output-dir outputs/

# Train compact MLP across all feature groups
PYTHONPATH=src python -m higgs_event_lab.neural \
    --data data/processed/higgs_sample_250k.csv.gz \
    --all-groups \
    --output-dir outputs/
```

### 5. Launch the Dashboard

```bash
streamlit run streamlit_app.py
```

---

## Notebooks — What to Run & In What Order

```
notebooks/
├── 00_physics/
│   └── 00_higgs_physics_and_dataset_intro.ipynb   ← Read-only, no data needed
├── 01_midsem/
│   ├── 01_eda_and_feature_groups.ipynb             ← Run FIRST (downloads data + builds 250k sample)
│   ├── 02_neural_network_baseline.ipynb            ← Run SECOND (MLP training, artifact export)
│   └── 03_classical_baseline.ipynb                 ← Run THIRD  (LogReg + XGBoost, artifact export)
└── 03_full/
    └── 00_full_higgs_colab.ipynb                   ← Run on Colab/Kaggle (full 1M benchmark)
```

| # | Notebook | Data Required | Where to Run | Time |
|---|---|---|---|---|
| 0 | `00_physics/00_higgs_physics_and_dataset_intro.ipynb` | None | Local | 1 min |
| 1 | `01_midsem/01_eda_and_feature_groups.ipynb` | Auto-downloaded (~2.6 GB) | Local | 15–30 min (first run) |
| 2 | `01_midsem/02_neural_network_baseline.ipynb` | `higgs_sample_250k.csv.gz` | Local | ~5 min |
| 3 | `01_midsem/03_classical_baseline.ipynb` | `higgs_sample_250k.csv.gz` | Local | ~3 min |
| 4 | `03_full/00_full_higgs_colab.ipynb` | Auto-downloaded | **Colab/Kaggle (GPU)** | 20–40 min |

> **Each notebook auto-downloads and auto-builds its data requirements** — just run from top to bottom and they work on a fresh machine.

---

## Project Structure

```
higgs-event-lab/
├── app/                        # Streamlit dashboard
│   ├── main.py                 # App entry point (sidebar navigation)
│   ├── loader.py               # higgs-experiment-v1 artifact loader
│   ├── visualizations.py       # ROC, PR, calibration, runtime plots
│   ├── components/             # UI components (overview, results, catalog, empty state)
│   ├── views/                  # Page-level view renderers
│   └── utils/                  # Schema validation, experiment loader
├── data/
│   ├── raw/                    # HIGGS.csv.gz (git-ignored, auto-downloaded by notebooks)
│   ├── processed/              # Sampled subsets (git-ignored, generated by pipeline)
│   ├── interim/                # Temporary processing artefacts
│   └── external/               # External reference files
├── docs/                       # Project documentation (15 numbered docs)
│   ├── 00_START_HERE.md
│   ├── 08_PHYSICS_BACKGROUND.md
│   ├── 09_DATASET_GUIDE.md
│   ├── 10_OBJECTIVES_AND_METHODOLOGY.md
│   ├── 11_SETUP_GUIDE.md
│   └── ...
├── notebooks/
│   ├── 00_physics/             # Physics intro (no execution required)
│   ├── 01_midsem/              # Mid-semester MVP (EDA + models)
│   └── 03_full/                # Full-dataset Colab/Kaggle benchmark
├── outputs/                    # Saved experiment artifacts
│   ├── logreg_*/               # LogReg runs per feature group
│   ├── xgboost_*/              # XGBoost runs per feature group
│   ├── mlp_*/                  # MLP runs per feature group
│   ├── experiments.json        # Aggregated experiment manifest
│   └── full_colab_results.json # Full 1M benchmark summary
├── reports/                    # Written reports and viva preparation
├── scripts/
│   ├── download_data.py        # CLI data downloader
│   ├── generate_sample_artifacts.py  # Synthetic fixtures; never benchmark evidence
│   └── train_and_export.py     # Full training pipeline script
├── src/higgs_event_lab/        # Core library (importable as a package)
│   ├── data.py                 # Chunked loader, sample extractor, schema validator
│   ├── features.py             # Feature group definitions (low_level, high_level, all)
│   ├── classical.py            # Logistic Regression + XGBoost training & artifact export
│   ├── neural.py               # Compact PyTorch MLP training & artifact export
│   ├── evaluation.py           # ROC-AUC, PR-AUC, Brier score, operating threshold
│   └── protocol.py             # higgs-experiment-v1 artifact schema
├── streamlit_app.py            # Canonical dashboard launcher (reads saved artifacts)
├── requirements.txt
└── tests/                      # 105 automated tests (unit + integration)
```

---

## Running Tests

```bash
# Full test suite
.venv/bin/python -m pytest tests/ -q

# Fast unit tests only
.venv/bin/python -m pytest tests/unit/ -q

# With verbose output
.venv/bin/python -m pytest tests/ -v
```

---

## Hosting on Streamlit Community Cloud

1. Fork or push this repo to your GitHub account
2. Go to [share.streamlit.io](https://share.streamlit.io) → New App
3. Set **Repository:** `CosmicMagnetar/higgs-event-lab`
4. Set **Main file path:** `streamlit_app.py`
5. Deploy with genuine saved experiment packages under `outputs/`.

The canonical launcher (`streamlit_app.py`) reads validated saved run packages. Missing outputs show an empty state; malformed runs show diagnostics. Startup and rescans never generate results. See `app/README.md` for the accepted artifact format.

---

## Citation

> Baldi, P., Sadowski, P., & Whiteson, D. (2014). Searching for exotic particles in high-energy physics with deep learning. *Nature Communications*, 5, 4308. [DOI: 10.1038/ncomms5308](https://doi.org/10.1038/ncomms5308)

```bibtex
@article{baldi2014higgs,
  title   = {Searching for exotic particles in high-energy physics with deep learning},
  author  = {Baldi, Pierre and Sadowski, Peter and Whiteson, Daniel},
  journal = {Nature Communications},
  volume  = {5},
  pages   = {4308},
  year    = {2014},
  doi     = {10.1038/ncomms5308}
}
```

---

## License

[MIT](LICENSE) — open for academic and educational use.
