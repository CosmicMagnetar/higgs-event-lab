# Higgs Event Lab — Mid-Semester Presentation & Demo Walkthrough

> **Historical draft notice (2026-11-14):** This walkthrough contains superseded mid-semester benchmark figures and UI instructions. Do not present its old metric values or threshold claims as current results. Use `reports/HiggsLens_Final_Report.md` and `docs/14_RESULTS_RECONCILIATION.md` for the current evidence record; neural results remain subject to the open release decision.

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
- **Key Observation:** Logistic Regression is the linear reference model; use the current run metadata for measured runtime rather than carrying forward the original mid-semester timing.

### Slide 6: Compact PyTorch MLP Architecture
- **Architecture:** Compact feedforward neural network:
  - Input layer: $D \in \{21, 7, 28\}$
  - Hidden Block 1: `Linear(D, 64)` $\to$ `BatchNorm1d(64)` $\to$ `ReLU` $\to$ `Dropout(0.20)`
  - Hidden Block 2: `Linear(64, 32)` $\to$ `BatchNorm1d(32)` $\to$ `ReLU` $\to$ `Dropout(0.20)`
  - Output: `Linear(32, 1)` $\to$ Logit $\to$ `BCEWithLogitsLoss`
- **Optimizer:** `AdamW` ($\eta = 10^{-3}$, weight decay $10^{-4}$, batch size $1024$).
- **Regularization:** Checkpointing with early stopping on validation ROC-AUC (patience 5).

### Slide 7: Benchmark Leaderboard & Empirical Results

The original mid-semester table has been superseded. The 250k Logistic Regression values were reconciled in `docs/14_RESULTS_RECONCILIATION.md`; the earlier 0.8369 all-feature value was identified as a fabricated placeholder. The earlier MLP table is provisional and is not approved for a final comparison. Use the current artifact-backed tables in `reports/HiggsLens_Final_Report.md` and retain the 250k and 1M protocols as separate experiments.

### Slide 8: Scientific Findings on Representation Learning
- The reconciled 250k Logistic Regression results are 0.5960 (low-level), 0.6518 (high-level), and 0.6854 (all features). <!-- allow-metric -->
- The earlier MLP values in this mid-semester draft remain provisional under BUG-015 and BUG-017; do not use them to assert a final neural advantage or a causal explanation for feature-group performance.
- Interpret feature-group differences as observations on this benchmark sample, not as evidence of a universal physical rule.

### Slide 9: Calibration & Decision Threshold Analysis
- Select the operating threshold on validation data, freeze it, and evaluate the test partition once.
- Do not reuse the old 86.5% and 96.2% background-rejection figures in this draft; verify the run-specific operating-point artifacts before presenting any values.
- Brier score is a probability-error metric on this benchmark. It does not, by itself, establish calibration under real-world event prevalence.

### Slide 10: Compute & Engineering Trade-Offs
- **Training Budget:** The current 250k artifact registry records nine runs with a combined training time of approximately 156.8 seconds. The compact-MLP run artifacts record CUDA as the execution device; the exact GPU model is not recorded. Do not reuse the earlier under-32-second CPU estimate.
- **Software Quality:** The user-supplied fresh macOS clone baseline reported 143 passed. The current QA branch passed editable installation and the full pytest workflow (**144 passed in 14.59s**) on Python 3.11, and the Docker image build passed in [run #43574590439](https://github.com/Saumya-Mishra-536/Higgs_Boson_Capstone/actions/runs/43574590439).
- **Artifact Contract:** Versioned JSON envelope (`higgs-experiment-v1`) decoupling model training from dashboard presentation.

### Slide 11: Threats to Validity & Honest Limitations
- **Simulation-to-Reality Gap:** Monte Carlo simulations omit real-world sensor drift, trigger latencies, and LHC pile-up.
- **Class Balance Difference:** The benchmark mixture (about $52.8\%$ signal) is not a model of the prevalence in a real collider analysis. This project does not estimate real-world event rates.
- **Subsample Scaling:** Benchmark conducted on $250,000$ events; full 11M training reserved for end-semester scale-up.

### Slide 12: Live Demonstration & Remaining Release Work
- **Live Demo:** HiggsLens Streamlit interactive dashboard (`app/main.py`).
- **Remaining gates:** independent MLP reproduction and release decision (Rachit/Krishna), full manual notebook execution, module-owner integration review, and regeneration/inspection of the PDF and PPTX exports from the corrected Markdown sources. A successful Docker image build does not establish a live cloud deployment.

---

## 2. Live Demo Script (Step-by-Step for Presenter)

### Step 1: Launch the Scientific Laboratory UI
```bash
streamlit run app/main.py
```
- **What to show:**
  - Point to the sidebar status badge showing all 6 valid experiment runs loaded from `outputs/`.
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
  - Explain how to read the run metadata and compare models only within a compatible sample, split and protocol; neural results remain provisional.
  - The runtime vs performance scatter plot demonstrating compute efficiency.

### Step 5: Navigate to "Validation"
- **What to show:**
  - Compare the saved ROC curves and explain that neural release status is still provisional.
  - Interactive PR curves and calibration reliability curves.
  - Show the validation-selected frozen threshold and its test evaluation. Do not tune the threshold with the test-set slider.

### Step 6: (Alternative UI Demo) Terminal HUD
```bash
streamlit run app/streamlit_app.py
```
- **What to show:**
  - Show the retro terminal aesthetic with scanlines, monospace fonts, and live pipeline diagnostics.
