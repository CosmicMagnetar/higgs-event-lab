# Why deep learning for HIGGS, and what we test

The course brief for Project 10 asks for three things: an AML baseline (Logistic Regression and XGBoost), a statement of the gap, and a neural network that is compared fairly on one held-out test set with ROC-AUC, PR-AUC, calibration and training time at a fixed data budget. This note covers the research behind the gap and the extra deep-learning experiments that answer it.

## 1. The research paper

Baldi, Sadowski and Whiteson, *Searching for exotic particles in high-energy physics with deep learning*, Nature Communications 5, 4308 (2014), arXiv:1402.4735. It is the paper the UCI HIGGS dataset comes from.

**The physics problem.** The signal is a hypothetical heavy neutral Higgs boson (H⁰) that decays through a charged Higgs (H±) and W bosons into a light Higgs (h⁰ → bb̄). The background is top-quark pair production (tt̄), which leaves almost the same set of particles in the detector: one lepton, missing energy and four jets, two of them b-tagged. The two processes differ mainly in how those particles' momenta are correlated. Section 4 of `docs/08_PHYSICS_BACKGROUND.md` has the decay chain.

**Two kinds of features.**

- 21 *low-level* features: what the detector measures directly (transverse momentum, η and φ for each lepton and jet, b-tags, missing energy).
- 7 *high-level* features: invariant masses (m_jj, m_jjj, m_lv, m_jlv, m_bb, m_wbb, m_wwbb). Physicists compute these from the low-level ones by hand, because they peak at known particle masses for signal events.

**What the paper found** (Table 1, AUC on 500k held-out simulated events, with the networks trained on the rest of the simulated sample):

| Model | Low-level only | High-level only | All 28 |
|---|---:|---:|---:|
| Boosted decision trees | 0.73 | 0.78 | 0.81 |
| Shallow NN (one hidden layer) | 0.733 | 0.777 | 0.816 |
| Deep NN (5 × 300, tanh) | **0.880** | 0.800 | **0.885** |

Three things stand out:

1. Trees and shallow networks need the hand-made invariant masses. Without them they drop to about 0.73.
2. The deep network gets 0.880 **from the low-level features alone**, which is better than any model given the physicists' features. It learns the useful non-linear combinations itself.
3. Adding the high-level features to the deep network barely helps (0.880 to 0.885), because it had already discovered what they encode.

Separately from AUC, the paper estimates discovery significance under its specified physics assumptions: the deep network on all features reaches 5.0σ, against 3.7σ for the shallow network, on the same simulated sample. In high-energy physics, 5σ is the threshold for claiming a discovery.

**How they trained it.** Five hidden layers of 300 tanh units, learning rate 0.05 decaying every batch, momentum rising from 0.9 to 0.99 over 200 epochs, weight decay 1e-5, mini-batches of 100, and early stopping on a 500k validation set. Each network trained for 200 to 1000 epochs on GPUs. Unsupervised pre-training and extra layers added training time without helping.

## 2. Why our mid-semester MLP tied XGBoost

The mid-semester network had two hidden layers (64 and 32 units, about 4k parameters), saw 175k training events (70% of 250k) and stopped within 30 epochs. In the paper's terms that is a *shallow* network on a *small* sample, which is exactly the case where the paper also found neural nets and trees roughly equal. So the tie was expected, and it isn't evidence against deep learning.

The deep-learning advantage in the paper depends on three things we did not have: depth and width, enough data, and long enough training.

## 3. A second line of evidence: tabular deep learning

Gorishniy, Rubachev, Khrulkov and Babenko, *Revisiting Deep Learning Models for Tabular Data*, NeurIPS 2021 (arXiv:2106.11959), benchmark deep models against gradient-boosted trees on eleven tabular datasets, including a 98k-event HIGGS subset. Their conclusions are relevant here:

- A plain residual MLP (ResNet-style blocks: BatchNorm, Linear, ReLU, Dropout, Linear, plus a skip connection) is a strong baseline that many papers leave out. On HIGGS it beats NODE, a much more complex differentiable-tree model.
- No model wins on every dataset. Gradient boosting stays very strong on small tables, so any deep-learning claim has to be backed by a fair comparison rather than assumed.

That is why our benchmark includes both a Baldi-style deep MLP and a residual MLP, and keeps an XGBoost configuration with validation early stopping as the opponent.

## 4. What we added

| Change | Where | Why |
|---|---|---|
| `deep` preset: 5 × 300, BatchNorm, ReLU, dropout 0.1, AdamW, cosine learning-rate schedule, up to 60 epochs, patience 8 | `src/higgs_event_lab/neural.py` | Baldi-style depth and width, with modern optimisation so it trains in minutes instead of hundreds of epochs |
| `resnet` preset: 4 residual blocks of width 256 | same | Gorishniy et al. architecture; skip connections keep gradients healthy in deep stacks |
| Whole-tensor mini-batching instead of a per-row `DataLoader` | same | The old loop collated one row at a time and was the main reason training was slow |
| XGBoost: up to 3000 trees, depth 8, early stopping on validation log-loss | `src/higgs_event_lab/classical.py`, release runner | So the AML baseline is tuned too and the comparison is fair, not a weak tree against a big network |
| Duplicate events removed before sampling | `scripts/release_benchmark.py` | The raw file repeats some events verbatim. The first release run found 44 (250k) and 454 (1M) identical events in both train and test |
| Nested budgets 250k ⊂ 1M ⊂ 2.5M, one load of the raw file | same | A learning curve with the same code and seed, which is the "fixed budget before scaling" the brief asks for |
| Expected calibration error next to Brier score | same | The brief asks for calibration |

## 5. Questions the benchmark answers

1. **Fixed budget (the brief's fair comparison).** At each budget, every model sees identical train, validation and test indices. Report ROC-AUC, PR-AUC, Brier, ECE and training time.
2. **Does the gap grow with data?** Compare the XGBoost and deep-network curves across 250k, 1M and 2.5M. The paper predicts the network keeps improving after trees flatten out.
3. **Does the network learn the physics features?** Compare each model on low-level features only against high-level features only. If a network on raw kinematics beats XGBoost on the physicists' invariant masses, it has learned something like those features itself.
4. **Shallow versus deep.** The compact 64-32 MLP stays in the benchmark as the "shallow" control.

Results are in Section 6 below, `outputs/release/release_summary.json` and Section 7.1 of `reports/HiggsLens_Final_Report.md`.

## 6. Results

Kaggle run on a Tesla T4, 2026-11-14 17:22:17 UTC to 2026-11-14 18:44:53 UTC. 278,698 duplicate events were removed before sampling, and every budget's split check reports zero events shared between train and test.

**All 28 features (test ROC-AUC)**

| Model | 250k | 1M | 2.5M |
|---|---:|---:|---:|
| Logistic Regression | 0.6864 | 0.6838 | 0.6842 |
| XGBoost | 0.8188 | 0.8302 | 0.8382 |
| Compact MLP (64-32) | 0.8165 | 0.8230 | 0.8255 |
| Deep MLP (5x300) | 0.8271 | 0.8474 | 0.8591 |
| Residual MLP | 0.8339 | 0.8506 | 0.8613 |

**21 low-level features only (test ROC-AUC)**

| Model | 250k | 1M | 2.5M |
|---|---:|---:|---:|
| Logistic Regression | 0.5962 | 0.5939 | 0.5938 |
| XGBoost | 0.7223 | 0.7478 | 0.7669 |
| Compact MLP (64-32) | 0.7250 | 0.7382 | 0.7448 |
| Deep MLP (5x300) | 0.7561 | 0.8141 | 0.8396 |
| Residual MLP | 0.7716 | 0.8203 | 0.8532 |

Answers to the four questions in Section 5:

1. **Fixed budget.** At every budget both deep networks beat XGBoost on all features. The residual MLP is best: 0.8613 against 0.8382 at 2.5M.
2. **Scaling.** On all features, the best deep network leads XGBoost by +0.015 at 250k, +0.020 at 1M and +0.023 at 2.5M. On low-level features the lead grows from +0.049 to +0.086.
3. **Learned features.** At 2.5M the residual MLP on low-level features alone (0.8532) beats XGBoost on the 7 high-level features (0.7921) and even XGBoost on all 28 (0.8382). This is qualitatively consistent with the paper’s representation-learning motivation, under a different architecture, sample and split protocol; it is not an exact reproduction.
4. **Shallow versus deep.** The compact 64-32 MLP gains only +0.009 from 250k to 2.5M on all features, against +0.027 for the residual MLP. A small network can't use the extra data, which is why the mid-semester comparison looked like a tie.

**Costs.** At 2.5M XGBoost trains in about 29 s, against about 494 s for the residual MLP on a T4 GPU. XGBoost is also better calibrated (ECE 0.0033 against 0.0184). Several deep runs at 2.5M stopped at the 60-epoch cap while still improving, so the network numbers are conservative.

## 7. Limits

- We train on at most 2.5M events, a quarter of the paper's training set, with one seed per budget. Absolute AUCs will be lower than the paper's 0.885, and uncertainty has not been measured; no fixed AUC-difference cutoff establishes a reliable ranking.
- HIGGS is simulated. A better AUC here means a better classifier on this benchmark, not a physics discovery.

## References

1. P. Baldi, P. Sadowski and D. Whiteson. Searching for exotic particles in high-energy physics with deep learning. *Nature Communications* 5, 4308, 2014. arXiv:1402.4735.
2. Y. Gorishniy, I. Rubachev, V. Khrulkov and A. Babenko. Revisiting deep learning models for tabular data. *NeurIPS*, 2021. arXiv:2106.11959.
3. K. He, X. Zhang, S. Ren and J. Sun. Deep residual learning for image recognition. *CVPR*, 2016.
4. I. Loshchilov and F. Hutter. SGDR: stochastic gradient descent with warm restarts. *ICLR*, 2017 (cosine schedule).

## Review decision — 17 November 2026

Keep Logistic Regression, XGBoost and compact MLP as the core course comparison. Deep and residual MLPs are extensions, not required replacements. At 250k with all features, saved-prediction accuracy at threshold 0.5 is 73.75% for XGBoost, 74.70% for Deep MLP and 75.26% for Residual MLP. The gain must be weighed against training cost and single-seed uncertainty. Larger-budget AUC gains do not establish larger-budget accuracy gains. See the Overleaf report for the paper-to-design mapping, protocol gaps and proposed controlled experiments.
