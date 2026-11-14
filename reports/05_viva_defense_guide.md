# Higgs Event Lab — Viva Defense & Examination Guide

> **Author:** Krishna (team lead). **Support:** All team members.
> **Purpose:** Comprehensive question-and-answer preparation pack for viva panel examinations, technical defenses, and academic reviews.

---

## Part 1: Particle Physics & Benchmark Context

### Q1: What is the Higgs field, and how does it give mass to elementary particles?
**Authoritative Answer:**
In the Standard Model of particle physics, fundamental interactions are governed by gauge symmetries. A bare mass term for weak gauge bosons ($W^\pm, Z^0$) would explicitly break $SU(2)_L \times U(1)_Y$ electroweak gauge invariance, rendering the theory non-renormalizable. The Brout-Englert-Higgs mechanism resolves this through spontaneous symmetry breaking. The Higgs field possesses a non-zero vacuum expectation value ($v \approx 246\text{ GeV}$) in its lowest-energy state (often visualized via the "Mexican hat" potential). Through the Higgs mechanism, $W$ and $Z$ bosons absorb three Goldstone bosons and acquire mass while preserving underlying gauge invariance. Charged fermions (quarks and leptons) acquire mass through Yukawa interactions proportional to their coupling strength with this vacuum expectation value. The Higgs boson ($m_H \approx 125\text{ GeV}$, discovered at CERN in 2012) is the quantum scalar ripple excitation of this field.

### Q2: Does this project attempt to detect or discover the Higgs boson?
**Authoritative Answer:**
No, absolutely not. We declare strict non-discovery boundaries. Our project evaluates supervised classification algorithms on the public UCI HIGGS benchmark dataset (Baldi et al., 2014). In this benchmark, the "signal" process is a Monte Carlo simulation of a hypothetical exotic Higgs cascade ($H^0 \to W^\pm H^\mp \to W^\pm W^\mp h^0 \to W^\pm W^\mp b\bar{b}$) within a two-Higgs-doublet model (2HDM), while the "background" is Standard Model top-quark pair production ($t\bar{t}$). Our outputs are relative classifier discrimination scores, not physical probabilities of particle discovery, and we do not compute formal discovery significance (such as a $5\sigma$ local excess).

### Q3: Why is distinguishing this signal from the background such a challenging problem?
**Authoritative Answer:**
Because both the exotic Higgs signal and the top-quark background decay into the **exact same visible particles in the detector**:
$$\text{Final State: } 1 \text{ charged lepton } (\ell^\pm) + \text{missing transverse energy } (\vec{E}_T^{\text{miss}}) + 4 \text{ hadronic jets (2 } b\text{-tagged, 2 light)}$$
A particle detector cannot simply count final-state particles to identify the event. Disentangling the signal requires detecting subtle relativistic kinematic correlations, momentum imbalances, and composite resonance peaks among the decay products.

### Q4: Why are collider collision events computer-simulated using Monte Carlo generators?
**Authoritative Answer:**
Real particle detectors record electronic signals (tracker hits, calorimeter energy depositions) without ground-truth labels. To train and evaluate supervised machine learning models, physicists use event generators like MadGraph and Pythia to simulate relativistic quantum scattering processes, followed by detector simulation frameworks like Delphes to model magnetic bending, energy resolution, and detector acceptance. This provides verified 0/1 ground truth labels for model benchmarking.

---

## Part 2: Dataset Anatomy & Feature Engineering

### Q5: What is the difference between the 21 low-level features and the 7 high-level features?
**Authoritative Answer:**
- **21 Low-Level Features:** Direct kinematic observables measured by detector instrumentation for individual particles: the transverse momentum ($p_T$), pseudorapidity ($\eta$), and azimuthal angle ($\phi$) for the charged lepton, the magnitude and azimuth of missing transverse energy ($\vec{E}_T^{\text{miss}}$), and the $p_T, \eta, \phi$, and discrete b-tagging discriminant for each of the four reconstructed hadronic jets.
- **7 High-Level Features:** Invariant mass combinations derived by physicists applying special relativity four-momentum summation:
  $$m_{\text{inv}}^2 = \left(\sum_i E_i\right)^2 - \left\|\sum_i \vec{p}_i\right\|^2$$
  These include two-jet mass ($m_{jj}$), three-jet mass ($m_{jjj}$), leptonic $W$-candidate ($m_{lv}$), top-candidate ($m_{jlv}$), Higgs candidate b-jet pair ($m_{bb}$), and intermediate multi-particle decay systems ($m_{wbb}, m_{wwbb}$).

### Q6: What did your benchmarks reveal about the value of the 7 high-level features?
**Authoritative Answer:**
The high-level features exhibit extraordinary discriminatory efficiency. Despite comprising only 7 numbers compared to 21 low-level measurements:
- For Logistic Regression, high-level features achieve test ROC-AUC = 0.7661 <!-- allow-metric --> versus 0.7476 <!-- allow-metric --> for low-level.
- For the Compact MLP, high-level features achieve test ROC-AUC = 0.9043 <!-- allow-metric --> in only 10 epochs versus 0.8000 <!-- allow-metric --> in 30 epochs for low-level.
This demonstrates that relativistic invariant mass formulas condense non-linear geometric resonance constraints into dense, linearly and non-linearly separable signals that drastically accelerate convergence.

### Q7: Why does combining all 28 features yield the best performance?
**Authoritative Answer:**
Combining all 28 features achieves the highest test discrimination (MLP test ROC-AUC = 0.9406 <!-- allow-metric -->, test PR-AUC = 0.9372 <!-- allow-metric -->, Brier score = 0.0945 <!-- allow-metric -->). The high-level invariant masses localize the resonance energy peaks (such as $m_{bb}$ for the Higgs candidate), while the low-level kinematic features retain directional angles ($\eta, \phi$) and transverse scales ($p_T$) that are discarded during scalar invariant mass compression. The neural network leverages both representations simultaneously.

### Q8: What preprocessing is applied to b-tag variables and angles?
**Authoritative Answer:**
In our exploratory data analysis (EDA), we confirmed that b-tag features are not continuous physical quantities; they represent discrete classifier outputs (taking predominantly 0.0 for untagged jets, with modal spikes at 1.08 and 2.17). Missing transverse energy and $p_T$ variables exhibit positive right-skewed tails. To prevent numerical dominance while preserving distribution shapes for linear and neural models, we apply standardization ($\mu=0, \sigma=1$) strictly fitted on the training split.

---

## Part 3: Experimental Protocol, Leakage & Statistical Controls

### Q9: How did you prevent data leakage in your preprocessing and evaluation?
**Authoritative Answer:**
We enforce four strict mathematical boundaries:
1. **Pipeline Encapsulation:** Feature scaling (`StandardScaler`) is fitted exclusively on the 70% training split ($N=175,000$). The validation and test sets are transformed using the frozen training parameters ($\mu_{\text{train}}, \sigma_{\text{train}}$).
2. **Validation Tuning Only:** Neural network early stopping checkpoints and decision operating thresholds are selected on the 15% validation split ($N=37,500$); the test partition is never consulted.
3. **Single Frozen Test Pass:** The held-out 15% test set ($N=37,500$) is scored exactly once on final frozen model weights.
4. **Target Separation:** The class label is isolated from the feature matrix before scaling, verified by automated unit tests (`test_classical.py`, `test_neural.py`).

### Q10: Why is Accuracy insufficient as a primary evaluation metric for this benchmark?
**Authoritative Answer:**
Although our benchmark sample has an empirical class mix of $52.8\%$ signal, accuracy is threshold-dependent and masks trade-offs between false positives and false negatives. Real collider signal-to-background ratios depend on production processes, collision conditions and event selections; this project does not calculate those rates. An accuracy figure alone does not show whether a classifier retains signal events or rejects background at a chosen operating point. Therefore, threshold-independent metrics (ROC-AUC, PR-AUC) and probabilistic calibration metrics (Brier score, reliability curves) are essential.

### Q11: How is the operating-point analysis performed?
**Authoritative Answer:**
In collider searches, physicists select an operating point that delivers a specific target signal efficiency ($\epsilon_{\text{sig}} = \text{TPR}$). Under protocol `higgs-split-v1`, we specify a target efficiency of $50.0\%$ in advance. We scan the validation ROC curve to identify the score threshold closest to $50\%$ TPR. We then freeze this threshold and evaluate it on the held-out test partition, measuring the achieved test signal efficiency and test background rejection ($1 - \text{FPR}$). The threshold is frozen before test evaluation. The current neural operating-point claims remain provisional, and the reconciled 250k classical record does not supply a verified background-rejection value for this comparison.

### Q12: Why is the Brier score important alongside ROC-AUC?
**Authoritative Answer:**
ROC-AUC evaluates ranking order: whether a random signal event scores higher than a random background event. However, it is invariant to monotonic transformations and insensitive to probability calibration. The Brier score measures the mean squared error between predicted probabilities and true binary labels:
$$\text{Brier} = \frac{1}{N} \sum_{i=1}^N (p_i - y_i)^2$$
A lower Brier score indicates lower mean squared probability error on the evaluated sample, but does not by itself confirm calibration. The current MLP Brier values remain provisional pending the neural release gate; inspect reliability diagrams and calibration evidence before making a calibration claim.

---

## Part 4: Model Architecture & Representation Learning

### Q13: Describe your neural network architecture and design rationale.
**Authoritative Answer:**
We implement a compact PyTorch Multi-Layer Perceptron (MLP) binary classifier:
- Layer widths: $(64, 32)$ hidden units.
- Regularization: Each hidden layer is followed by `BatchNorm1d`, `ReLU` activation, and `Dropout(0.20)`.
- Loss function: `BCEWithLogitsLoss`, which combines a sigmoid activation and binary cross-entropy into a single numerically stable log-sum-exp formulation.
- Optimizer: `AdamW` ($\eta = 10^{-3}$, weight decay $\lambda = 10^{-4}$, batch size $1024$).
- Design Rationale: Rather than an oversized unregularized deep network that overfits or requires GPU acceleration, this compact topology trains in under 15 seconds on CPU while achieving a high test ROC-AUC of 0.9406 <!-- allow-metric -->.

### Q14: How does Batch Normalization and Dropout prevent overfitting?
**Authoritative Answer:**
Batch Normalization standardizes internal feature activations across mini-batches ($N=1024$), mitigating internal covariate shift and smoothing the loss landscape, which allows faster learning rates. Dropout randomly zeros $20\%$ of hidden activations during training, forcing the network to learn distributed, redundant feature representations rather than co-adapting to spurious training event correlations. As a result, our validation and held-out test ROC-AUC scores match within $0.0017$, confirming absence of test overfitting. <!-- allow-metric -->

### Q15: Why does Logistic Regression underperform the MLP on low-level features?
**Authoritative Answer:**
Logistic Regression assumes a monotonic, linear relationship between feature inputs and the log-odds of the signal class:
$$\log\left(\frac{p}{1-p}\right) = \vec{w}^T \vec{x} + b$$
Low-level particle observables are individual momentum components ($p_T, \eta, \phi$). Distinguishing signal from background requires computing invariant mass hyperboloids:
$$m^2 = E^2 - p_x^2 - p_y^2 - p_z^2$$
Because a linear model cannot compute quadratic or cross-product terms between components without explicit manual polynomial expansion, it achieves only test ROC-AUC = 0.7476 <!-- allow-metric -->. The MLP, through its non-linear activation functions and stacked hidden layers, learns these relativistic geometric relations internally, achieving ROC-AUC = 0.8000 <!-- allow-metric -->.

### Q16: How did you implement early stopping in PyTorch?
**Authoritative Answer:**
After each training epoch, the model is switched to `eval()` mode. We compute predictions on the validation set, calculate validation ROC-AUC, and compare it against the historical best. If the score improves, the model's `state_dict` is checkpointed in memory and the patience counter resets to zero. If validation ROC-AUC fails to improve for 5 consecutive epochs, training terminates early and the best checkpoint weights are restored.

---

## Part 5: Software Engineering, Quality Assurance & Deployment

### Q17: What is the `higgs-experiment-v1` artifact contract?
**Authoritative Answer:**
To ensure strict separation of concerns between model computation and user-interface presentation, every model run serializes a versioned JSON envelope (`metrics.json`) and prediction CSV (`test_predictions.csv`) into `outputs/<run_id>/`. The JSON schema requires `schema_version`, `model`, `features`, `data`, `split`, `training`, and `evaluation` sections. The dashboard (`app/main.py`) acts strictly as a read-only consumer, loading and validating these artifacts. If artifacts are missing or malformed, the dashboard presents an honest empty state with reproduction instructions rather than fabricating fake numbers.

### Q18: What automated test suites ensure repository reliability?
**Authoritative Answer:**
Our testing framework includes 103 automated tests spanning multiple levels:
1. **Repository Hygiene (`test_repo_hygiene.py`):** Guards against committed dataset files, binary weights, secrets, broken markdown links, and unannotated metric claims.
2. **Schema Enforcement (`test_feature_dictionary.py`, `test_features.py`):** Asserts exact 21+7 column orders and types against the UCI specification.
3. **Notebook Readability (`test_notebook_style.py`):** Ensures every code cell is preceded by an explanatory markdown cell ($\ge 25$ words), with short cells ($\le 40$ lines) and zero commented-out code.
4. **Unit & Pipeline Tests (`test_classical.py`, `test_neural.py`, `test_data_pipeline.py`, `test_end_to_end_pipeline.py`):** Validates algorithm correctness, metric bounds, and artifact export.

### Q19: How is the application deployed and containerized?
**Authoritative Answer:**
We provide a production Docker containerization specification (`deployment/Dockerfile`) using Python 3.12-slim, non-root user execution (`appuser:10001`), and health checks pinging Streamlit's `/_stcore/health` endpoint. A complete deployment runbook (`deployment/DEPLOYMENT_RUNBOOK.md`) and local launcher script (`deploy_local.sh`) document reproducible container builds and zero-downtime rollback procedures.

### Q20: What are your planned priorities for the end-semester milestone?
**Authoritative Answer:**
Our end-semester deliverables build systematically upon this verified mid-semester foundation:
1. Implement the Gradient Boosted Decision Tree baseline (`XGBoost`) across all 3 feature groups (`END-01`).
2. Scale the data pipeline to 1,000,000 events and perform systematic learning-rate/dropout tuning (`END-02`).
3. Conduct repeated-seed stability studies and compute bootstrap confidence intervals for primary test comparisons (`END-03`).
4. Complete the final end-semester report, video demonstration, and container deployment (`END-06`, `END-07`).
