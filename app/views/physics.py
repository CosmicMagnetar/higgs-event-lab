"""Physics Explorer section for HiggsLens Streamlit application.

Presents the Standard Model foundation, electroweak symmetry breaking,
the exotic-Higgs benchmark classification task, decay kinematics, and
explicit scientific boundary controls.
"""

from __future__ import annotations

import streamlit as st


def render_physics() -> None:
    """Render Physics Explorer view."""
    st.markdown(
        '<div class="section-header">'
        '<span class="material-icons icon-md" style="color:#127c80;">science</span>'
        '<h2 style="margin:0;">Electroweak Symmetry &amp; Collision Kinematics</h2>'
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "To understand the machine-learning classification task, we must examine the physical "
        "processes producing the collision signatures in particle detectors."
    )

    t1, t2, t3 = st.tabs(["The Higgs Mechanism", "Signal vs Background Kinematics", "Scientific Boundaries"])

    with t1:
        st.markdown("### Electroweak Symmetry Breaking &amp; The Higgs Field")
        st.markdown(
            "The Standard Model of particle physics unifies the electromagnetic and weak forces into a "
            "single electroweak framework. Under gauge symmetry, gauge bosons ($W^\\pm, Z^0$) and fermions "
            "must formally remain massless. Yet experimental measurements demonstrate that $W$ and $Z$ bosons "
            "possess substantial masses ($80.4\\text{ GeV}$ and $91.2\\text{ GeV}$)."
        )

        col_left, col_right = st.columns([1.2, 1.0])
        with col_left:
            st.markdown(
                """
                - **The Non-Zero Vacuum Expectation Value:** Unlike electromagnetic or gravitational fields,
                  the scalar Higgs field has a non-zero potential minimum in the vacuum state:
                  $$v = \\frac{1}{\\sqrt{\\sqrt{2} G_F}} \\approx 246\\text{ GeV}$$
                - **Spontaneous Symmetry Breaking:** The Lagrangian remains gauge-symmetric, but the ground state
                  spontaneously selects a specific vacuum expectation value in the potential valley ("Mexican hat").
                - **Mass Generation:**
                  - $W^\\pm$ and $Z^0$ acquire mass by absorbing scalar Goldstone bosons from the Higgs doublet.
                  - Charged leptons and quarks obtain mass through Yukawa coupling to the Higgs field.
                  - Photons and gluons do not couple to the vacuum expectation value and remain strictly massless.
                """
            )

        with col_right:
            st.markdown(
                """
                <div class="scientific-panel-cyan">
                    <h4 style="color:#127c80; margin-top:0;">The Higgs Boson Ripple</h4>
                    <p style="font-size: 0.9rem; color:#475b70;">
                        The <strong>Higgs boson ($h^0$)</strong> is the scalar quantum excitation of this pervasive field.
                        Observed jointly by ATLAS and CMS at CERN on <strong>July 4, 2012</strong>, it has:
                    </p>
                    <ul style="font-size:0.85rem; color:#5f7082; margin-bottom:0;">
                        <li>Rest Mass: $m_H \\approx 125.1\\text{ GeV}$</li>
                        <li>Spin / Parity: $J^P = 0^+$ (scalar)</li>
                        <li>Electric Charge: $Q = 0$</li>
                        <li>Mean Lifetime: $\\tau \\approx 1.6 \\times 10^{-22}\\text{ s}$</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with t2:
        st.markdown("### The Benchmark Task: Exotic Higgs Signal vs Top-Pair Background")
        st.markdown(
            "> **Crucial Scientific Distinction:** The UCI HIGGS benchmark dataset (Baldi, Sadowski, &amp; Whiteson, 2014) "
            "is **not** a re-creation of the 2012 SM Higgs discovery. It simulates a hypothetical search for "
            "heavy exotic Higgs bosons ($H^0, H^\\pm$) decaying through the SM-like Higgs ($h^0$) against a dominant "
            "Standard Model top-quark background."
        )

        st.markdown("#### Identical Visible Final-State Decays")
        st.markdown(
            r"Both processes decay into the exact same visible detector particles: "
            r"**one charged lepton ($\ell$), four hadronic jets ($j$), two of which are tagged as $b$-quark jets ($b$), "
            r"and missing transverse energy ($E_T^{miss}$) from an escaping neutrino ($\nu$)**."
        )

        k1, k2 = st.columns(2)
        with k1:
            st.markdown(
                """
                <div class="scientific-panel">
                    <h4 style="color:#127c80;">Signal Process (Exotic Decay Chain)</h4>
                    <p style="font-family:monospace; font-size:0.95rem; color:#4c79a4;">
                        g g &rarr; H<sup>0</sup> &rarr; W<sup>&plusmn;</sup> H<sup>&mp;</sup> &rarr; W<sup>&plusmn;</sup> W<sup>&mp;</sup> h<sup>0</sup> &rarr; &ell; &nu; j j b b&#772;
                    </p>
                    <ul style="font-size:0.88rem; color:#5f7082;">
                        <li>Contains intermediate resonances at exotic Higgs masses.</li>
                        <li>The $b\bar{b}$ jet pair emerges directly from the SM Higgs $h^0$, concentrating around $m_{bb} \approx 125\text{ GeV}$.</li>
                        <li>Lepton and neutrino reconstruct the $W$ boson mass ($m_{lv} \approx 80.4\text{ GeV}$).</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with k2:
            st.markdown(
                """
                <div class="scientific-panel">
                    <h4 style="color:#4c79a4;">Background Process (Top-Pair Production)</h4>
                    <p style="font-family:monospace; font-size:0.95rem; color:#5f7082;">
                        g g / q q&#772; &rarr; t t&#772; &rarr; W<sup>+</sup> b W<sup>-</sup> b&#772; &rarr; &ell; &nu; j j b b&#772;
                    </p>
                    <ul style="font-size:0.88rem; color:#5f7082;">
                        <li>Dominant Standard Model strong-interaction production.</li>
                        <li>The two $b$-quarks come from separate top decay branches and do not resonate together, creating a broad combinatorial $m_{bb}$ spectrum.</li>
                        <li>Kinematic momenta ($p_T$) and invariant masses follow top decay kinematics ($m_t \approx 173\text{ GeV}$).</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("#### Why Physicists Derive High-Level Invariant Masses")
        st.markdown(
            "In four-momentum space, the invariant mass $M$ of a system of particles with energies $E_i$ and "
            "three-momenta $\\vec{p}_i$ is defined by: "
            "$$M^2 = \\left( \\sum_i E_i \\right)^2 - \\left| \\sum_i \\vec{p}_i \\right|^2$$"
            "Because invariant masses concentrate into sharp resonant peaks for true parent decays, "
            "they provide immense discriminatory power over continuous background distributions. "
            "The central machine-learning research question is whether deep neural networks can learn "
            "these nonlinear relativistic invariants directly from raw detector measurements."
        )

    with t3:
        st.markdown("### Scientific Boundaries &amp; Verification Standards")
        st.markdown(
            "In adherence to our team charter (`team/TEAM_CHARTER.md`) and project scope "
            "(`docs/03_SCIENTIFIC_SCOPE_AND_DATA.md`), all dashboard analyses observe the following boundaries:"
        )

        b1, b2 = st.columns(2)
        with b1:
            st.markdown(
                """
                <div class="scientific-panel">
                    <h4 style="color:#19745c;">What We DO</h4>
                    <ul style="font-size:0.9rem; color:#475b70;">
                        <li>Classify simulated collision events as signal or background.</li>
                        <li>Compare regularized Logistic Regression and compact MLPs.</li>
                        <li>Compare low-level (21), high-level (7), and all (28) feature groups.</li>
                        <li>Report ROC-AUC, PR-AUC, calibration curves, and compute time.</li>
                        <li>Report background rejection at validation-frozen operating thresholds.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with b2:
            st.markdown(
                """
                <div class="scientific-panel">
                    <h4 style="color:#b33b43;">What We Do NOT Do</h4>
                    <ul style="font-size:0.9rem; color:#475b70;">
                        <li>Simulate quantum fields or the Higgs mechanism.</li>
                        <li>Claim physical discovery of any elementary particle.</li>
                        <li>Produce detector-grade experimental measurements.</li>
                        <li>Convert classifier scores into real-world probabilities of being a Higgs.</li>
                        <li>Calculate experimental statistical significance (e.g. 5-sigma discovery).</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
