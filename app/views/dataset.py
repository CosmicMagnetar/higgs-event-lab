"""Dataset Observatory section for HiggsLens Streamlit application.

Displays verified feature dictionary, group architectures, interactive feature
distribution explorer, cross-feature correlation matrix, and empirical class balance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import json
import numpy as np
import pandas as pd
import streamlit as st

from app.styles import (
    ACCENT_AMBER,
    ACCENT_BLUE,
    ACCENT_CYAN,
    ACCENT_TEAL,
    SURFACE_BASE,
    SURFACE_BORDER,
    SURFACE_CARD,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)
from app.visualizations import (
    plot_class_balance_donut,
    plot_feature_group_heatmap,
)
from higgs_event_lab.features import (
    ALL_COLUMNS,
    ALL_FEATURES,
    FEATURE_GROUPS,
    HIGH_LEVEL_FEATURES,
    LOW_LEVEL_FEATURES,
    TARGET_COLUMN,
    get_feature_columns,
)


@st.cache_data
def load_feature_dictionary() -> pd.DataFrame:
    """Load machine-readable feature dictionary CSV from docs/."""
    dict_path = Path("docs/feature_dictionary.csv")
    if dict_path.is_file():
        return pd.read_csv(dict_path)
    # Fallback to programmatic list
    records = []
    for idx, f in enumerate(LOW_LEVEL_FEATURES):
        records.append({
            "file_column_index": idx + 1,
            "model_input_index": idx,
            "name": f,
            "group": "low_level",
            "kind": "kinematic",
            "description": f"Low-level kinematic parameter: {f}",
        })
    for idx, f in enumerate(HIGH_LEVEL_FEATURES):
        records.append({
            "file_column_index": idx + 22,
            "model_input_index": idx + 21,
            "name": f,
            "group": "high_level",
            "kind": "derived",
            "description": f"Physicist-derived invariant mass: {f}",
        })
    return pd.DataFrame(records)


def load_saved_eda(base_dir: str = "outputs"):
    """Display an actionable notice for invalid saved EDA."""
    from app.utils.loader import load_saved_eda as read_eda
    try:
        return read_eda(base_dir)
    except ValueError as exc:
        st.error(str(exc))
        return None


def render_dataset(base_dir: str = "outputs") -> None:
    """Render Dataset Observatory view."""
    st.markdown(
        '<div class="section-header">'
        '<span class="material-icons icon-md" style="color:#127c80;">dashboard</span>'
        '<h2 style="margin:0;">Dataset Observatory</h2>'
        "</div>",
        unsafe_allow_html=True,
    )
    saved_eda = load_saved_eda(base_dir)
    if saved_eda is None:
        st.warning("No saved EDA artifacts found under outputs/eda. Run the EDA notebook before using these tabs.")
    st.markdown(
        "Explore the underlying structure of the 28 input collision features, "
        "their statistical separation between exotic Higgs signal and top-quark background, "
        "and empirical class balance."
    )

    t1, t2, t3, t4 = st.tabs([
        "Feature Dictionary",
        "Distribution Explorer",
        "Cross-Feature Correlations",
        "Class Balance Observatory",
    ])

    with t1:
        st.markdown("### The 21 Low-Level and 7 High-Level Feature Architecture")
        st.markdown(
            "The HIGGS dataset defines 28 total inputs plus the target label `class_label` in column 0. "
            "Columns 1–21 represent detector kinematics, and columns 22–28 represent invariant masses."
        )

        df_dict = load_feature_dictionary()

        filter_group = st.selectbox(
            "Filter by Feature Group:",
            options=["All Groups", "low_level (21 kinematics)", "high_level (7 invariant masses)"],
            index=0,
        )

        if "low_level" in filter_group:
            df_filtered = df_dict[df_dict["group"] == "low_level"]
        elif "high_level" in filter_group:
            df_filtered = df_dict[df_dict["group"] == "high_level"]
        else:
            df_filtered = df_dict

        st.dataframe(df_filtered, width="stretch", hide_index=True)

    with t2:
        st.markdown("### Interactive Feature Distribution Explorer")
        st.markdown(
            "Compare the probability distributions of signal ($y=1$) versus background ($y=0$) "
            "across any selected kinematic or invariant mass feature."
        )

        selected_feat = st.selectbox(
            "Select Feature to Inspect:",
            options=list(ALL_FEATURES),
            index=list(ALL_FEATURES).index("m_bb") if "m_bb" in ALL_FEATURES else 0,
        )

        if saved_eda is not None:
            summary, feature_summary, _ = saved_eda
            feature_row = feature_summary[feature_summary["feature"] == selected_feat]
            figure_path = Path(base_dir) / "eda" / "feature_distributions.png"
            if figure_path.is_file():
                st.image(str(figure_path), caption="Saved full EDA feature-distribution figure")
            if not feature_row.empty:
                row = feature_row.iloc[0]
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Signal Mean", f"{row['signal_mean']:.3f}")
                c2.metric("Background Mean", f"{row['background_mean']:.3f}")
                c3.metric("Mean Difference", f"{abs(row['signal_mean'] - row['background_mean']):.3f}")
                c4.metric("Unique Values", f"{int(row['unique_count']):,}")
                st.caption(f"Measured on {int(summary['rows']):,} saved EDA rows from the recorded run.")

    with t3:
        st.markdown("### Cross-Feature Correlation Matrix")
        st.markdown(
            "Linear models rely heavily on the orthogonality of features. "
            "High-level invariant masses concentrate correlations from multiple decay products."
        )

        if saved_eda is not None:
            _, _, correlations = saved_eda
            heatmap_path = Path(base_dir) / "eda" / "correlation_heatmap.png"
            if heatmap_path.is_file():
                st.image(str(heatmap_path), caption="Saved Pearson correlation heatmap")
            if not correlations.empty:
                st.dataframe(correlations.head(10), width="stretch", hide_index=True)

    st.markdown("---")

    # Feature group performance heatmap
    st.markdown("### Model Performance Heatmap by Feature Group")
    st.markdown(
        "ROC-AUC and PR-AUC heatmap across all loaded models and feature configurations."
    )
    # We need access to runs here; this will be empty if no runs loaded.
    # Placeholder: the heatmap is rendered in the Experiments view instead.
    st.info(
        "The feature-group performance heatmap is displayed in the Experiment Comparison view "
        "when valid artifact runs are loaded."
    )

    with t4:
        st.markdown("### Class Balance Observatory")
        st.markdown(
            "In many collider analyses, signal processes are buried under overwhelming background rates. "
            "However, the UCI HIGGS benchmark was pre-filtered to establish an approximately balanced learning problem."
        )

        if saved_eda is None:
            return
        summary, _, _ = saved_eda
        counts = summary["label_counts"]
        signal_count = int(counts.get("1", 0))
        background_count = int(counts.get("0", 0))
        total_count = signal_count + background_count
        signal_fraction = signal_count / total_count if total_count else 0.0
        b1, b2, b3 = st.columns(3)
        with b1:
            st.metric("Signal Fraction (y=1)", f"{signal_fraction:.1%}", help="Verified in saved EDA outputs")
        with b2:
            st.metric("Background Fraction (y=0)", f"{1 - signal_fraction:.1%}", help="Standard top-pair collision background")
        with b3:
            st.metric("Imbalance Ratio (Sig / Bkg)", f"{signal_count / background_count:.2f} : 1", help="Nearly balanced benchmark")

        y_eda = np.concatenate([np.ones(signal_count, dtype=int), np.zeros(background_count, dtype=int)])
        class _DonutRun:
            test_labels = y_eda
        fig_donut = plot_class_balance_donut(_DonutRun())
        st.pyplot(fig_donut)

        st.markdown(
            """
            <div class="scientific-panel-cyan">
                <h4 style="color:#127c80; margin-top:0;">Why ROC-AUC &amp; PR-AUC Over Simple Accuracy?</h4>
                <p style="font-size:0.9rem; color:#475b70; margin-bottom:0;">
                    Although class balance is roughly 53/47, in physical collider operations the true background rate
                    is millions of times higher before pre-selection cuts. A classifier with 80% accuracy that achieves it by
                    allowing 20% false positives would completely flood the experiment with millions of fake events.
                    Therefore, <strong>background rejection at high signal efficiency</strong> and <strong>ROC-AUC</strong>
                    provide the true measure of scientific ranking discrimination.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
