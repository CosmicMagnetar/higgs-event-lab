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
The reconciled 250k classical benchmark shows higher test ROC-AUC for the high-level feature group than for the low-level group. This is an observation on the recorded sample, not a universal property:
- The reconciled 250k Logistic Regression results are 0.5960 for low-level and 0.6518 for high-level features. <!-- allow-metric -->
- The earlier MLP comparisons in this guide are provisional and remain under the neural release gate. Do not use them to make a final cross-family claim.
- The benchmark results can motivate further study of invariant-mass features, but they do not prove a general physical explanation or a universal convergence advantage.

### Q7: What does the current evidence say about combining all 28 features?
**Authoritative Answer:**
The current saved 250k registry reports its strongest all-feature run for XGBoost and a closely performing all-feature compact MLP. These values are scoped to that registry and sample. The separate earlier MLP results remain provisional under BUG-015 and BUG-017; do not use them to claim a final neural best-model result until the release decision is recorded.

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
Our benchmark sample has an empirical class mix of about $52.8\%$ signal. Accuracy is threshold-dependent and can hide the trade-off between false positives and false negatives. The project does not model real-collider prevalence or background composition, so no numeric real-world signal-to-background ratio is asserted here. ROC-AUC, PR-AUC, confusion-matrix rates and Brier score provide complementary views of benchmark performance; none alone establishes real-world calibration.

### Q11: How is the operating-point analysis performed?
**Authoritative Answer:**
In this project, the target signal efficiency is specified in the protocol and the threshold is selected using validation predictions. The threshold is then frozen and evaluated on held-out test predictions. Report only the operating-point values stored in the relevant run artifact. The historical 96.2% and 86.5% figures in earlier drafts are superseded and should not be repeated.

### Q12: Why is the Brier score important alongside ROC-AUC?
**Authoritative Answer:**
ROC-AUC evaluates ranking order: whether a random signal event scores higher than a random background event. However, it is invariant to monotonic transformations and insensitive to probability calibration. The Brier score measures the mean squared error between predicted probabilities and true binary labels:
$$\text{Brier} = \frac{1}{N} \sum_{i=1}^N (p_i - y_i)^2$$
A lower Brier score indicates lower mean squared probability error on the evaluated benchmark split. It does not, by itself, confirm calibration. Review the reliability diagram and class prevalence, and do not treat the provisional MLP Brier values as final results.

---

## Part 4: Model Architecture & Representation Learning

### Q13: Describe your neural network architecture and design rationale.
**Authoritative Answer:**
We implement a compact PyTorch Multi-Layer Perceptron (MLP) binary classifier:
- Layer widths: $(64, 32)$ hidden units.
- Regularization: Each hidden layer is followed by `BatchNorm1d`, `ReLU` activation, and `Dropout(0.20)`.
- Loss function: `BCEWithLogitsLoss`, which combines a sigmoid activation and binary cross-entropy into a single numerically stable log-sum-exp formulation.
- Optimizer: `AdamW` ($\eta = 10^{-3}$, weight decay $\lambda = 10^{-4}$, batch size $1024$).
- Design Rationale: This compact topology provides a resource-conscious neural baseline. Its observed metrics remain provisional, and this report does not claim that its architecture prevents overfitting.

### Q14: What roles do Batch Normalization and Dropout play in the MLP architecture, and what does the current evidence establish about their effects?
**Authoritative Answer:**
Batch Normalization normalizes hidden activations using mini-batch statistics during training, while Dropout randomly sets a fraction of hidden activations to zero during training. In this architecture the configured dropout rate is 0.20. These are regularization design choices intended to influence optimization and reduce co-adaptation; the saved benchmark does not isolate their causal effect. The current evidence therefore does not establish that they prevented overfitting. Similar validation and test metrics cannot establish absence of leakage or test overfitting; the neural reproducibility review remains open.

### Q15: How do the current Logistic Regression and MLP results compare on low-level features, and what limits a definitive conclusion?
**Authoritative Answer:**
Logistic Regression uses a linear decision function in the input features, while the MLP can represent nonlinear interactions through its hidden layers. The low-level feature group contains event-level kinematic descriptors; the high-level group contains engineered invariant-mass quantities derived from those inputs. The current saved 250k artifacts report low-level test ROC-AUC of 0.5960 for Logistic Regression and 0.7257 for the compact MLP. <!-- allow-metric --> The MLP result remains provisional pending independent reproduction and the release decision, so it should not be presented as an approved final comparison. The feature definitions and their physical interpretation are described in `docs/06` and `reports/06`.

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
The user-supplied fresh-clone baseline on 2026-11-14 (Python 3.15 on macOS Apple Silicon) reported 143 passing tests. Separately, the current QA branch passed the editable install and full `python -m pytest -q` workflow (**144 passed in 14.59s**) on Python 3.11, and the Docker image build passed in run [#43574590439](https://github.com/Saumya-Mishra-536/Higgs_Boson_Capstone/actions/runs/43574590439).
1. **Repository Hygiene (`test_repo_hygiene.py`):** Guards against committed dataset files, binary weights, secrets, broken markdown links, and unannotated metric claims.
2. **Schema Enforcement (`test_feature_dictionary.py`, `test_features.py`):** Asserts exact 21+7 column orders and types against the UCI specification.
3. **Notebook Readability (`test_notebook_style.py`):** Ensures every code cell is preceded by an explanatory markdown cell ($\ge 25$ words), with short cells ($\le 40$ lines) and zero commented-out code.
4. **Unit & Pipeline Tests (`test_classical.py`, `test_neural.py`, `test_data_pipeline.py`, `test_end_to_end_pipeline.py`):** Validates algorithm correctness, metric bounds, and artifact export.

### Q19: How is the application deployed and containerized?
**Authoritative Answer:**
The repository provides a Docker image definition (`deployment/Dockerfile`) using Python 3.12-slim, non-root user execution (`appuser:10001`), and a Streamlit `/_stcore/health` check. The image build passed in current-branch CI. This verifies image construction, not a live cloud deployment, health-check operation in production, or rollback. The runbook documents the deployment procedure; live deployment verification remains separate.

### Q20: What are your planned priorities for the end-semester milestone?
**Authoritative Answer:**
Our end-semester deliverables build systematically upon this verified mid-semester foundation:
1. Implement the Gradient Boosted Decision Tree baseline (`XGBoost`) across all 3 feature groups (`END-01`).
2. Scale the data pipeline to 1,000,000 events and perform systematic learning-rate/dropout tuning (`END-02`).
3. Conduct repeated-seed stability studies and compute bootstrap confidence intervals for primary test comparisons (`END-03`).
4. Complete the final end-semester report, video demonstration, and container deployment (`END-06`, `END-07`).
