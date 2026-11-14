"""Methods and Research section for HiggsLens Streamlit application.

Documents the strict experiment protocol (higgs-split-v1), model architectures,
academic citations, and explicit physical limitations.
"""

from __future__ import annotations

import streamlit as st


def render_methods() -> None:
    """Render Methods and Research view."""
    st.markdown(
        '<div class="section-header">'
        '<span class="material-icons icon-md" style="color:#127c80;">article</span>'
        '<h2 style="margin:0;">Methods, Experiment Protocol &amp; Research Citations</h2>'
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "Scientific integrity requires that every benchmark comparison follows a deterministic, "
        "leakage-free experimental protocol, transparently states architectural configurations, "
        "and properly credits original foundational sources."
    )

    t1, t2, t3 = st.tabs(["Experiment Protocol", "Citations &amp; References", "Research Limitations"])

    with t1:
        st.markdown("### The `higgs-split-v1` Stratified Holdout Protocol")
        st.markdown(
            "To prevent data leakage and guarantee that classical baselines and neural networks "
            "are evaluated under identical conditions, all models adhere to the protocol defined in "
            "`src/higgs_event_lab/protocol.py`:"
        )

        st.markdown(
            """
            <div class="scientific-panel">
                <ol style="font-size:0.92rem; color:#475b70; margin-bottom:0; padding-left:1.2rem;">
                    <li><strong>Stratified Partition:</strong> The dataset is partitioned into <strong>70% training,
                    15% validation, and 15% held-out test</strong> partitions using a deterministic stratified split with seed <code>42</code>.</li>
                    <li><strong>Leakage-Free Preprocessing:</strong> All feature standardisation (<code>StandardScaler</code>)
                    is fitted strictly on the 70% training partition. Means and standard deviations from validation or test rows
                    never inform model training.</li>
                    <li><strong>Validation-Only Optimization:</strong> Hyperparameters, early-stopping patience, and
                    operating decision thresholds are selected exclusively on the 15% validation partition.</li>
                    <li><strong>One-Shot Held-Out Testing:</strong> The 15% test set is evaluated exactly once after all
                    model weights and decision thresholds are frozen.</li>
                    <li><strong>Shared Row Indices:</strong> Every model evaluates the exact same row indices to enable
                    statistically valid paired comparisons.</li>
                </ol>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### Model Family Architectures")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                r"""
                <div class="scientific-panel">
                    <h4 style="color:#127c80;">Compact MLP Architecture</h4>
                    <ul style="font-size:0.88rem; color:#5f7082;">
                        <li><strong>Input Dimension:</strong> $D \in \{21, 7, 28\}$</li>
                        <li><strong>Hidden Layers:</strong> Linear(64) &rarr; BatchNorm1d &rarr; ReLU &rarr; Dropout(0.20) &rarr; Linear(32) &rarr; BatchNorm1d &rarr; ReLU &rarr; Dropout(0.20)</li>
                        <li><strong>Output:</strong> Linear(1) unconstrained logit</li>
                        <li><strong>Loss Function:</strong> BCEWithLogitsLoss</li>
                        <li><strong>Optimizer:</strong> AdamW ($lr=10^{-3}$, weight decay $= 10^{-4}$)</li>
                        <li><strong>Early Stopping:</strong> Patience of 5 epochs monitoring validation ROC-AUC with best-weights restoration.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with c2:
            st.markdown(
                r"""
                <div class="scientific-panel">
                    <h4 style="color:#4c79a4;">Logistic Regression Baseline</h4>
                    <ul style="font-size:0.88rem; color:#5f7082;">
                        <li><strong>Input Dimension:</strong> $D \in \{21, 7, 28\}$</li>
                        <li><strong>Pipeline:</strong> <code>sklearn.pipeline.Pipeline</code> embedding <code>StandardScaler</code> + <code>LogisticRegression</code></li>
                        <li><strong>Solver:</strong> L-BFGS with max iterations $= 1000$</li>
                        <li><strong>Regularization:</strong> Inverse strength $C = 1.0$, tolerance $= 10^{-4}$</li>
                        <li><strong>Thresholding:</strong> Validation threshold chosen for 50% target signal efficiency, frozen and evaluated on test.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with t2:
        st.markdown("### Academic Citations &amp; Data Sources")
        st.markdown(
            "Any academic report, publication, or presentation derived from this lab must cite "
            "the original dataset and foundational physics literature:"
        )

        st.markdown(
            """
            - **[Dataset]** Whiteson, D. (2014). *HIGGS* [Dataset]. UCI Machine Learning Repository.
              DOI: [10.24432/C5V312](https://doi.org/10.24432/C5V312).
              URL: [https://archive.ics.uci.edu/dataset/280/higgs](https://archive.ics.uci.edu/dataset/280/higgs)
            - **[Foundational Paper]** Baldi, P., Sadowski, P., &amp; Whiteson, D. (2014). Searching for exotic particles in high-energy physics with deep learning.
              *Nature Communications*, 5, 4308. DOI: [10.1038/ncomms5308](https://doi.org/10.1038/ncomms5308).
            - **[Observation of the Higgs Boson]** ATLAS Collaboration (2012). Observation of a new particle in the search for the Standard Model Higgs boson with the ATLAS detector at the LHC.
              *Physics Letters B*, 716(1), 1–29. DOI: [10.1016/j.physletb.2012.08.020](https://doi.org/10.1016/j.physletb.2012.08.020).
            - **[Observation of the Higgs Boson]** CMS Collaboration (2012). Observation of a new boson at a mass of 125 GeV with the CMS experiment at the LHC.
              *Physics Letters B*, 716(1), 30–61. DOI: [10.1016/j.physletb.2012.08.021](https://doi.org/10.1016/j.physletb.2012.08.021).
            - **[Higgs Mechanism Theory]** Englert, F., &amp; Brout, R. (1964). Broken symmetry and the mass of gauge vector mesons.
              *Physical Review Letters*, 13(9), 321–323. DOI: [10.1103/PhysRevLett.13.321](https://doi.org/10.1103/PhysRevLett.13.321).
            - **[Higgs Mechanism Theory]** Higgs, P. W. (1964). Broken symmetries and the masses of gauge bosons.
              *Physical Review Letters*, 13(16), 508–509. DOI: [10.1103/PhysRevLett.13.508](https://doi.org/10.1103/PhysRevLett.13.508).
            """
        )

    with t3:
        st.markdown("### Project Limitations &amp; Non-Goals")
        st.markdown(
            "Every scientific endeavor must clearly delineate its experimental limitations. "
            "The following constraints govern this workbench:"
        )

        st.markdown(
            r"""
            1. **Simulated Monte Carlo Data:** All collision events were generated via computational physics event generators
               (MadGraph, Pythia, Delphes fast detector simulation). While realistic, fast simulations omit complex detector noise,
               dead sensor channels, and pile-up effects present in real LHC collision data.
            2. **Benchmark Scope:** The learning problem is a binary classification task on a pre-filtered sample. It does not
               implement online real-time trigger pipelines operating at 40 MHz collision frequencies.
            3. **Decision Thresholds vs Discovery Significance:** The threshold-based operating point analysis evaluates
               trade-offs between signal efficiency and background rejection. It must **never** be equated with physical
               discovery significance (such as 5-standard-deviation Gaussian significance $\sigma = 5$).
            4. **Probabilistic Real-World Meaning:** Classifier scores output probabilities relative to the pre-balanced
               training sample. They do not represent physical posterior probabilities of a collision producing a Higgs boson
               in an actual collider run. A real-world interpretation would require externally justified event prevalence, cross-sections, selection efficiencies, detector effects and calibration studies that are outside this project.
            """
        )
