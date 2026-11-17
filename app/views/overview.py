"""Readable overview of saved, validated experiments."""
from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

from app.loader import RunDiscoveryResult, load_saved_eda


def _model_name(name: str) -> str:
    return {"compact_mlp": "Compact MLP", "deep_mlp": "Deep MLP",
            "resnet_mlp": "Residual MLP", "logistic_regression": "Logistic Regression",
            "xgboost": "XGBoost"}.get(name, name.replace("_", " ").title())


def render_overview(discovery: RunDiscoveryResult, outputs_path: Path | None = None) -> None:
    st.title("Experiment overview")
    st.markdown('<p class="page-intro">Simulated events. Measured comparisons. Transparent limitations.</p>', unsafe_allow_html=True)
    runs = discovery.valid_runs
    try:
        eda = load_saved_eda(discovery.base_dir)
    except ValueError as exc:
        st.warning(str(exc))
        eda = None
    if not runs:
        st.markdown("### Awaiting validated experiment artifacts")
        st.info("Import saved metrics.json and test_predictions.csv files through Artifact settings. Startup never trains or generates results.")
        return
    sizes = sorted({r.row_count for r in runs})
    left, middle, right = st.columns(3)
    left.metric("Events per loaded run" if len(sizes) == 1 else "Largest loaded sample", f"{max(sizes):,}")
    middle.metric("Saved runs", len(runs))
    right.metric("Features in the full representation", 28)
    st.divider()
    st.subheader("All-feature benchmark")
    all_runs = [r for r in runs if r.feature_group == "all"]
    order = {name: i for i, name in enumerate(["logistic_regression", "xgboost", "compact_mlp", "deep_mlp", "resnet_mlp"])}
    records = []
    for run in sorted(all_runs, key=lambda r: (r.row_count, order.get(r.model_name, 99))):
        records.append({"Model": _model_name(run.model_name), "ROC-AUC": run.test_roc_auc,
                        "PR-AUC": run.test_pr_auc, "Accuracy (0.5)": float(((run.test_scores >= 0.5) == run.test_labels).mean()), "Training seconds": run.training_seconds})
    if records:
        frame = pd.DataFrame(records)
        st.dataframe(
            frame,
            use_container_width=True,
            hide_index=True,
            column_config={
                "ROC-AUC":          st.column_config.NumberColumn(format="%.4f"),
                "PR-AUC":           st.column_config.NumberColumn(format="%.4f"),
                "Accuracy (0.5)":   st.column_config.NumberColumn(format="%.2f%%"),
                "Training seconds": st.column_config.NumberColumn(format="%.1f"),
            },
        )
    else:
        st.info("No all-feature runs loaded. Explore the available groups in Experiment Comparison.")
    if len({(r.row_count, r.data_source, str(r.split)) for r in all_runs}) > 1:
        st.warning("These runs have different data or split metadata. This table is descriptive, not a matched benchmark.")
    st.divider()
    a, b = st.columns(2, gap="large")
    with a:
        st.subheader("Why deep learning?")
        st.markdown("- Logistic Regression establishes the linear reference; XGBoost tests nonlinear feature interactions.\n"
                    "- The compact MLP satisfies the project brief. Deep and residual MLPs extend the capacity comparison.\n"
                    "- Low-level versus expert-feature experiments test whether learned interactions reduce reliance on engineered masses.")
    with b:
        st.subheader("What the evidence does not prove")
        st.markdown("- A better benchmark score is not a particle discovery or a real detector measurement.\n"
                    "- One seed does not establish reliable rankings between close models.\n"
                    "- Predictive gains do not prove a network reconstructed a specific invariant mass.\n"
                    "- XGBoost remains a useful alternative when compute cost or calibration matters.")
    st.divider()
    st.caption("Simulated UCI HIGGS benchmark. Not a particle-discovery analysis.")
    with st.expander("Inspect provenance and registry health"):
        st.dataframe(pd.DataFrame([{"Run": r.run_id, "Model": _model_name(r.model_name), "Source": r.data_source,
                                   "Rows": r.row_count, "Split": r.split} for r in runs]), hide_index=True, width="stretch")
        st.caption(f"{len(runs)} valid runs · {len(discovery.incomplete_runs)} incomplete · {len(discovery.malformed_runs)} malformed · EDA {'available' if eda else 'not loaded'}. Schema checks do not independently establish training provenance.")
        st.caption("The source dataset has 11 million events; a source-file size is not a training sample size.")
