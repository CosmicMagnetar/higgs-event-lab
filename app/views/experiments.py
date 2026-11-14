"""Experiment Comparison section for HiggsLens Streamlit application.

Displays comprehensive model leaderboards, multi-group benchmark comparisons,
and compute efficiency trade-offs across low-level, high-level, and all features.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.loader import RunDiscoveryResult
from app.visualizations import (
    plot_feature_group_comparison,
    plot_feature_group_heatmap,
    plot_model_performance_radar,
    plot_runtime_vs_auc,
)


def _model_label(name: str) -> str:
    return {
        "compact_mlp": "Compact MLP",
        "logistic_regression": "Logistic Regression",
        "xgboost": "XGBoost",
    }.get(name, name.replace("_", " ").title())


def render_experiments(discovery: RunDiscoveryResult) -> None:
    """Render Experiment Comparison view."""
    st.markdown(
        '<div class="section-header">'
        '<span class="material-icons icon-md" style="color:#127c80;">bar_chart</span>'
        '<h2 style="margin:0;">Experiment Comparison</h2>'
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "Evaluate comparative performance across Logistic Regression, XGBoost and compact MLP "
        "and feature configurations (`low_level`, `high_level`, `all`). All evaluations derive "
        "from saved run metrics and held-out predictions; partition details are shown per run."
    )

    if not discovery.has_valid_runs:
        st.warning(
            "No valid experiment artifacts are currently loaded from `outputs/`. "
            "Please run the model training scripts shown in the Overview section."
        )
        return

    all_runs = discovery.valid_runs
    chosen_ids = st.multiselect("Runs", [run.run_id for run in all_runs], default=[run.run_id for run in all_runs])
    all_runs = [run for run in all_runs if run.run_id in chosen_ids]
    model_options = sorted({_model_label(run.model_name) for run in all_runs})
    selected_models = st.multiselect("Models", model_options, default=model_options)
    selected_groups = st.multiselect(
        "Feature groups",
        ["low_level", "high_level", "all"],
        default=["low_level", "high_level", "all"],
    )
    runs = [
        run for run in all_runs
        if _model_label(run.model_name) in selected_models and run.feature_group in selected_groups
    ]
    if not runs:
        st.info("Select at least one model and feature group to compare.")
        return
    winner = max(runs, key=lambda run: run.test_roc_auc or float("-inf"))
    metric_cols = st.columns(4)
    metric_cols[0].metric("Selected runs", len(runs))
    metric_cols[1].metric("Highest test ROC-AUC run", _model_label(winner.model_name))
    metric_cols[2].metric("Highest test ROC-AUC", f"{winner.test_roc_auc:.4f}")
    metric_cols[3].metric("Group of top observed run", winner.feature_group)
    st.warning("The summary ranking uses held-out test ROC-AUC and is descriptive only. Do not use this ranking to choose models, tune settings, or make final model-selection decisions; those decisions must use training/validation evidence.")

    # Tabbed views for Leaderboard and Visual Comparative Analytics
    tab_table, tab_groups, tab_compute, tab_radar = st.tabs([
        "Comparison Leaderboard",
        "Feature-Group Comparison",
        "Compute Efficiency Trade-Off",
        "Model Performance Radar",
    ])

    with tab_table:
        st.markdown("### Held-Out Evaluation Summary")

        records = []
        for r in runs:
            records.append({
                "Run ID": r.run_id,
                    "Model": _model_label(r.model_name),
                "Family": r.model_family,
                "Group": r.feature_group,
                "Features": r.feature_count,
                "Source": r.data_source,
                "Sample size": r.row_count,
                "Split": r.split["protocol_version"],
                "Train / validation / test": f"{r.split['train_rows']} / {r.split['validation_rows']} / {r.split['test_rows']}",
                "Split Seed": r.split["seed"],
                "Runtime (s)": round(r.training_seconds, 2) if r.training_seconds is not None else None,
                "Val ROC-AUC": round(r.validation_metrics.get("roc_auc", 0.0), 4) if "roc_auc" in r.validation_metrics else None,
                "Test ROC-AUC": round(r.test_roc_auc, 4) if r.test_roc_auc is not None else None,
                    "Test PR-AUC": round(r.test_pr_auc, 4) if r.test_pr_auc is not None else None,
                    "Brier Score": round(r.test_brier_score, 4) if r.test_brier_score is not None else None,
                    "Accuracy": r.test_metrics.get("accuracy"),
                })

        df_leaderboard = pd.DataFrame(records)

        # Sort options
        sort_col = st.selectbox(
            "Sort Leaderboard By:",
            options=["Test ROC-AUC", "Test PR-AUC", "Brier Score", "Runtime (s)", "Group"],
            index=0,
        )
        ascending = sort_col in ("Brier Score", "Runtime (s)")
        df_sorted = df_leaderboard.sort_values(by=sort_col, ascending=ascending, na_position="last")

        st.dataframe(df_sorted, width="stretch", hide_index=True)

        st.markdown(
            "> **Note on Brier Score:** Lower values indicate superior probabilistic calibration, "
            "where 0.0 is perfect certainty and 0.25 represents uninformative 50/50 guessing."
        )

    with tab_groups:
        st.markdown("### Representation Learning: Low-Level (21) vs High-Level (7) vs All (28)")
        st.markdown(
            "Does the neural network learn equivalent or superior representations from raw kinematic "
            "variables alone, or do human-engineered invariant masses remain necessary?"
        )

        if len({(r.model_name, r.feature_group) for r in runs}) != len(runs):
            st.info("Select one run per model and feature group to display grouped charts. All selected runs remain in the leaderboard and validation comparison.")
        else:
            fig_groups = plot_feature_group_comparison(runs)
            st.pyplot(fig_groups)

        st.caption("Compare runs using their recorded source, sample size and split; different experiments may not be directly comparable.")

    with tab_compute:
        st.markdown("### Compute Efficiency: Training Wall-Clock Time vs Performance")
        st.markdown(
            "Compare the trade-off between computational budget (GPU/CPU wall-clock seconds) "
            "and discrimination power (held-out test ROC-AUC)."
        )

        fig_runtime = plot_runtime_vs_auc(runs)
        st.pyplot(fig_runtime)

    with tab_radar:
        st.markdown("### Model Performance Radar")
        st.markdown(
            "Multi-metric comparison across ROC-AUC, PR-AUC, calibration quality "
            "(1 - Brier), and background rejection. Each axis is normalized to [0, 1]."
        )

        fig_radar = plot_model_performance_radar(runs)
        st.pyplot(fig_radar)

        st.markdown(
            """
            <div class="scientific-panel">
                <h4 style="color:#127c80; margin-top:0;">How to Read the Radar</h4>
                <ul style="font-size:0.9rem; color:#475b70; margin-bottom:0;">
                    <li>Each vertex represents one evaluation metric; larger area indicates stronger overall performance.</li>
                    <li><strong>ROC-AUC</strong> measures ranking discrimination across all thresholds.</li>
                    <li><strong>PR-AUC</strong> measures precision-recall trade-off, sensitive to class imbalance.</li>
                    <li><strong>1 - Brier</strong> measures probabilistic calibration (higher = better calibrated).</li>
                    <li><strong>Background Rejection</strong> measures background events rejected at the validation-frozen threshold.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
