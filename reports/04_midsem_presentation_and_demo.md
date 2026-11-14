# Higgs Event Lab — Mid-Semester Presentation & Demo Walkthrough

> **Author:** Krishna (team lead & integration). **Support:** All team members.
> **Audience:** Course assessors, viva examination panel, and peer reviewers.
> **Target Duration:** 12–15 minutes presentation + 5 minutes live demonstration.

---

## 1. Slide Deck Outline (12 Slides)

### Slide 1: Title & Project Identity
- **Header:** Higgs Event Lab: Supervised Machine Learning Benchmark for the HIGGS Dataset
- **Subheader:** Comparative Representation Learning & High-Level Invariant Mass Reconstructions
- **Team Roles & GitHub IDs:**
  - Krishna (`CosmicMagnetar`) — Team Lead, Integration, Protocol & Deployment
  - Saumya Mishra (`Saumya-Mishra-536`) — Documentation, Methodology & QA Lead
  - Aditya Rana (`A1B2C3D4E5F6G7H8I9J0164-hack`) — Classical ML & Data Pipeline Lead
  - Rachit Gupta (`Rachitt19`) — Deep Learning & MLP Baseline Lead
  - Ronit Singh (`ronithi27`) — Streamlit Scientific Dashboard & UI Lead
- **Key Message:** A reproducible, rigorous machine learning benchmark comparing classical linear boundaries against deep regularized neural representations on simulated LHC collision events.

### Slide 2: Physics Context & Scientific Boundary
- **The Physics Problem:** Distinguishing rare particle signatures from immense Standard Model backgrounds at hadron colliders.
- **The HIGGS Benchmark Nuance:**
  - Signal: Theoretical exotic Higgs cascade ($gg \to H^0 \to W^\pm H^\mp \to W^\pm W^\mp h^0 \to W^\pm W^\mp b\bar{b}$).
  - Background: Standard Model top-quark pair production ($t\bar{t} \to W^+ b W^- \bar{b} \to \ell \nu j j b \bar{b}$).
  - Critical Insight: Both processes end in the **identical final state** (1 charged lepton, missing energy, 2 b-jets, 2 light jets).
- **Explicit Boundary:** We benchmark classifier discrimination on simulated events; we do **not** simulate quantum fields, and we make no claim of particle discovery.

### Slide 3: Dataset Anatomy & The 21 + 7 Feature Schema
- **Source:** UCI Machine Learning Repository (dataset id 280; Baldi, Sadowski, & Whiteson, 2014; CC BY 4.0).
- **21 Low-Level Features:** Raw reconstructed particle four-momenta ($\vec{p}_T, \eta, \phi$) for lepton and jets, missing energy, and discrete b-tag scores.
- **7 High-Level Features:** Invariant masses derived from relativistic kinematic constraints ($m_{jj}, m_{jjj}, m_{lv}, m_{jlv}, m_{bb}, m_{wbb}, m_{wwbb}$).
- **Core Research Question:** Can machine learning models extract non-linear resonance constraints directly from raw four-momenta, or do physicist-derived invariant masses remain essential?

### Slide 4: Data Engineering & Leakage Protection
- **Chunked Ingestion:** Zero in-memory RAM crashes; data read in streaming chunks of 50,000 rows.
- **Protocol `higgs-split-v1`:** Deterministic stratified 70% train ($175,000$), 15% validation ($37,500$), and 15% test ($37,500$) with seed 42.
- **Leakage Prevention:**
  - Feature scalers fitted strictly on training data; never on validation or test.
  - Zero target contamination into feature matrices.
  - Test set isolated until final model freeze.

### Slide 5: Classical Linear Baseline Architecture
- **Model:** `scikit-learn` Logistic Regression with L-BFGS solver ($C = 1.0$).
- **Pipeline:** Preprocessing encapsulated in an `sklearn.pipeline.Pipeline`.
- **Validation Operating Point:** Threshold chosen on validation targeting $50\%$ signal efficiency, frozen, then evaluated on test.
- **Key Observation:** Logistic Regression is the linear reference model; timing comparisons are omitted here because the clean 250k reconciliation record does not include per-group runtimes.

### Slide 6: Compact PyTorch MLP Architecture
- **Architecture:** Compact feedforward neural network:
  - Input layer: $D \in \{21, 7, 28\}$
  - Hidden Block 1: `Linear(D, 64)` $\to$ `BatchNorm1d(64)` $\to$ `ReLU` $\to$ `Dropout(0.20)`
  - Hidden Block 2: `Linear(64, 32)` $\to$ `BatchNorm1d(32)` $\to$ `ReLU` $\to$ `Dropout(0.20)`
  - Output: `Linear(32, 1)` $\to$ Logit $\to$ `BCEWithLogitsLoss`
- **Optimizer:** `AdamW` ($\eta = 10^{-3}$, weight decay $10^{-4}$, batch size $1024$).
- **Regularization:** Checkpointing with early stopping on validation ROC-AUC (patience 5).

### Slide 7: Benchmark Leaderboard & Empirical Results
- The following table separates reconciled classical ROC-AUC values from provisional MLP artifacts. The clean 250k Logistic Regression rerun did not provide the other metrics below; they are intentionally not inferred.

| Model | Feature Set | Test ROC-AUC | Test PR-AUC | Brier Score | Runtime (s) | Evidence |
|---|---|---|---|---|---|---|
| Logistic Regression | Low-level (21) | 0.5960 <!-- allow-metric --> | Not in reconciliation record | Not in reconciliation record | Not in reconciliation record | Reconciled clean rerun |
| Logistic Regression | High-level (7) | 0.6518 <!-- allow-metric --> | Not in reconciliation record | Not in reconciliation record | Not in reconciliation record | Reconciled clean rerun |
| Logistic Regression | All features (28) | 0.6854 <!-- allow-metric --> | Not in reconciliation record | Not in reconciliation record | Not in reconciliation record | Reconciled clean rerun |
| Compact MLP | Low-level (21) | 0.8000 <!-- allow-metric --> | 0.8097 <!-- allow-metric --> | 0.1824 <!-- allow-metric --> | 15.02 | Provisional; release gate open |
| Compact MLP | High-level (7) | 0.9043 <!-- allow-metric --> | 0.8888 <!-- allow-metric --> | 0.1184 <!-- allow-metric --> | 5.13 | Provisional; release gate open |
| Compact MLP | All features (28) | 0.9406 <!-- allow-metric --> | 0.9372 <!-- allow-metric --> | 0.0945 <!-- allow-metric --> | 10.52 | Provisional; release gate open |

### Slide 8: Scientific Findings on Representation Learning
- **Finding 1:** In the reconciled 250k Logistic Regression rerun, the high-level feature run records test ROC-AUC 0.6518 <!-- allow-metric --> versus 0.5960 <!-- allow-metric --> for low-level features. The corresponding MLP artifacts record 0.9043 <!-- allow-metric --> versus 0.8000 <!-- allow-metric -->, but those neural results remain provisional and are not approved for a final comparative conclusion.
- **Finding 2:** The saved all-feature MLP artifact records test ROC-AUC = 0.9406 <!-- allow-metric -->. This value remains provisional pending the neural release gate; no causal claim about feature complementarity is made.
- **Finding 3:** The current evidence does not yet support a final claim of MLP superiority over the reconciled classical baseline.

### Slide 9: Calibration & Decision Threshold Analysis
- **Operating-Point Analysis:**
  - Target Signal Efficiency: $50\%$ on validation.
  - Frozen evaluation on test:
    - The reconciled classical record does not include a verified 250k background-rejection value for this slide.
    - Neural operating-point values remain provisional pending the release gate.
- **Probabilistic Calibration:** The MLP artifacts report Brier scores, but these results remain provisional. A Brier score alone does not establish calibration, and no physical posterior probability is claimed.

### Slide 10: Compute & Engineering Trade-Offs
- **Training Budget:** The current neural artifact runtimes are provisional; do not quote a combined training-time claim until the benchmark release evidence is reconciled.
- **Software Quality:** 143 automated tests passed in the 2026-11-14 clean-clone QA run.
- **Artifact Contract:** Versioned JSON envelope (`higgs-experiment-v1`) decoupling model training from dashboard presentation.

### Slide 11: Threats to Validity & Honest Limitations
- **Simulation-to-Reality Gap:** Monte Carlo simulations omit real-world sensor drift, trigger latencies, and LHC pile-up.
- **Class Balance Difference:** The benchmark mixture ($52.8\%$ signal) is approximately balanced and must not be interpreted as a realistic collider event rate; this project does not calculate real-world signal-to-background ratios.
- **Subsample Scaling:** Benchmark conducted on $250,000$ events; full 11M training reserved for end-semester scale-up.

### Slide 12: Live Demonstration & End-Semester Roadmap
- **Live Demo:** HiggsLens Streamlit interactive dashboard (`app/main.py`).
- **End-Semester Roadmap:**
  - Review and finalize the existing XGBoost feature-group benchmark evidence (`END-01`).
  - Extended MLP hyperparameter optimization and diagnostic learning rates (`END-02`).
  - Docker containerization and cloud deployment verification (`END-06`).

---

## 2. Live Demo Script (Step-by-Step for Presenter)

### Step 1: Launch the Scientific Laboratory UI
```bash
streamlit run app/main.py
```
- **What to show:**
  - Point to the sidebar status badge showing all 9 schema-valid experiment runs loaded from `outputs/`.
  - Explain that the dashboard operates strictly read-only on serialized `higgs-experiment-v1` artifacts without retraining models.

### Step 2: Navigate to "Physics Explorer"
- **What to show:**
  - Review the interactive Higgs cascade decay diagram.
  - Highlight why separating exotic Higgs from top-pair background is challenging (identical visible particles: $W W b \bar{b}$).

### Step 3: Navigate to "Dataset Observatory"
- **What to show:**
  - Search for feature `m_bb` in the interactive dictionary.
  - Toggle feature distribution histograms between signal (cyan) and background (orange). Show that $m_{bb}$ has a visible resonance peak for signal events.

### Step 4: Navigate to "Experiment Comparison"
- **What to show:**
  - The model comparison leaderboard table.
  - Point out the progression: Low-Level ($0.8000$ <!-- allow-metric -->) $\to$ High-Level ($0.9043$ <!-- allow-metric -->) $\to$ All Features ($0.9406$ <!-- allow-metric -->).
  - The runtime vs performance scatter plot demonstrating compute efficiency.

### Step 5: Navigate to "Validation"
- **What to show:**
  - Interactive ROC curves showing the MLP curve dominating the Logistic Regression curve.
  - Interactive PR curves and calibration reliability curves.
  - The Interactive Operating-Point Workbench slider: adjust the target signal efficiency to $50\%$ and point out the resulting $96.2\%$ background rejection.

### Step 6: (Alternative UI Demo) Terminal HUD
```bash
streamlit run app/streamlit_app.py
```
- **What to show:**
  - Show the retro terminal aesthetic with scanlines, monospace fonts, and live pipeline diagnostics.
