"""Evidence-first overview for the HiggsLens Streamlit dashboard."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from app.loader import RunDiscoveryResult


def _model_name(name: str) -> str:
    return {
        "compact_mlp": "Compact MLP",
        "logistic_regression": "Logistic Regression",
        "xgboost": "XGBoost",
    }.get(name, name.replace("_", " ").title())


def _eda_summary(base_dir: str) -> dict[str, object] | None:
    from app.utils.loader import load_saved_eda
    try:
        loaded = load_saved_eda(base_dir)
        return loaded[0] if loaded else None
    except ValueError as exc:
        st.warning(str(exc))
        return None


def _hero(best_run, valid_count: int, rows: int) -> None:
    if best_run is None:
        headline = "Awaiting validated experiment artifacts"
        detail = "Run the notebook workflow to populate the evidence registry."
    else:
        headline = f"{_model_name(best_run.model_name)} · {best_run.test_roc_auc:.4f} ROC-AUC"
        detail = f"Highest observed test ROC-AUC run (descriptive only): {best_run.run_id} · {best_run.feature_group} features · {rows:,} recorded rows"
    st.markdown(
        f"""
        <div class="hero-shell">
            <div class="hero-kicker">HIGGSLENS / EVIDENCE WORKBENCH</div>
            <div class="hero-title">Experiment overview</div>
            <div class="hero-copy">A transparent view of simulated HIGGS classification experiments, from saved EDA evidence to held-out model predictions.</div>
            <div class="proof-strip">
                <span class="proof-pill"><strong>LIVE REGISTRY</strong> {valid_count} valid runs</span>
                <span class="proof-pill"><strong>HEADLINE</strong> {headline}</span>
                <span class="proof-pill"><strong>PROTOCOL</strong> {best_run.split["protocol_version"] if best_run else "N/A"} · seed {best_run.split["seed"] if best_run else "N/A"}</span>
            </div>
            <div style="margin-top:0.8rem;color:#5f7082;font-size:0.82rem;">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_overview(discovery: RunDiscoveryResult) -> None:
    """Render the dashboard landing page from validated saved artifacts."""
    valid_runs = discovery.valid_runs
    best_run = max(valid_runs, key=lambda run: run.test_roc_auc or float("-inf"), default=None)
    all_runs = [run for run in valid_runs if run.feature_group == "all"]
    best_all = max(all_runs, key=lambda run: run.test_roc_auc or float("-inf"), default=None)
    rows = best_run.row_count if best_run else 0
    eda = _eda_summary(discovery.base_dir)

    _hero(best_run, len(valid_runs), rows)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Sample size", f"{rows:,}" if rows else "N/A")
    c2.metric("Valid runs", str(len(valid_runs)))
    c3.metric("Highest test ROC-AUC", f"{best_run.test_roc_auc:.4f}" if best_run else "N/A")
    c4.metric("Input features", "28")
    c5.metric("Missing values", str(eda.get("missing_values", "N/A")) if eda else "N/A")

    st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
    st.markdown("### Saved experiment evidence")
    proof_cols = st.columns(3)
    proof_data = [
        ("DATA PROOF", f"{eda['signal_fraction']:.2%} signal" if eda and isinstance(eda.get("signal_fraction"), (int, float)) else "EDA not loaded", "Saved EDA summary; inspect recorded quality checks."),
        ("TOP OBSERVED RUN", f"{_model_name(best_all.model_name)}" if best_all else "No run available", f"Highest observed all-feature test ROC-AUC: {best_all.test_roc_auc:.4f}. Descriptive only; not a model-selection rule." if best_all else "Run models to populate this card."),
        ("APP PROOF", "Artifact-backed", "Metrics, predictions, curves and EDA are read from outputs/; no training at startup."),
    ]
    for col, (eyebrow, value, detail) in zip(proof_cols, proof_data):
        with col:
            st.markdown(
                f'<div class="evidence-card"><div class="eyebrow">{eyebrow}</div><div class="value">{value}</div><div class="detail">{detail}</div></div>',
                unsafe_allow_html=True,
            )

    if best_all:
        st.markdown("### All-feature results")
        st.caption("Rows are ordered by held-out test ROC-AUC for descriptive comparison only. Do not use this test-set ranking to select models or tune hyperparameters.")
        leader_rows = []
        for run in sorted(all_runs, key=lambda item: item.test_roc_auc or 0, reverse=True):
            leader_rows.append({
                "Model": _model_name(run.model_name),
                "Source": run.data_source,
                "Sample size": run.row_count,
                "Split": run.split,
                "Test ROC-AUC": run.test_roc_auc,
                "PR-AUC": run.test_pr_auc,
                "Brier": run.test_brier_score,
                "Training (s)": run.training_seconds,
                "Artifact": run.run_id,
            })
        st.dataframe(pd.DataFrame(leader_rows), width="stretch", hide_index=True)

    left, right = st.columns([1.15, 1])
    with left:
        st.markdown("### Saved evidence preview")
        image = Path(discovery.base_dir) / "eda" / "feature_group_separation.png"
        if image.is_file():
            st.image(str(image), caption="Feature-group separation from saved EDA")
        else:
            st.info("Run EDA to display the saved feature-group evidence.")
    with right:
        st.markdown("### Registry health")
        health = pd.DataFrame([
            {"Check": "Schema-valid runs", "Status": "PASS" if valid_runs else "WAITING", "Value": str(len(valid_runs))},
            {"Check": "Incomplete run folders", "Status": "PASS" if not discovery.incomplete_runs else "CHECK", "Value": str(len(discovery.incomplete_runs))},
            {"Check": "Malformed artifacts", "Status": "PASS" if not discovery.malformed_runs else "CHECK", "Value": str(len(discovery.malformed_runs))},
            {"Check": "Saved EDA", "Status": "PASS" if eda else "WAITING", "Value": "outputs/eda" if eda else "missing"},
        ])
        st.dataframe(health, width="stretch", hide_index=True)
        st.caption("PASS means the dashboard found and validated the corresponding saved evidence.")

    st.markdown("### Scientific boundary")
    st.info("HIGGS is simulated Monte Carlo data. These scores compare classifiers on a benchmark; they are not particle-discovery evidence or real-detector probabilities.")

    if not valid_runs:
        st.markdown("### Reproduce the registry")
        st.code("streamlit run streamlit_app.py\n\n# Then run notebooks/03_full/00_full_higgs_colab.ipynb", language="bash")
