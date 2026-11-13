"""HiggsLens — Streamlit Scientific Dashboard for Higgs Event Lab.

Provides a polished, scientific, interactive interface for exploring:
1. Overview: project objective, dataset information, available experiment results and status.
2. Physics Explorer: Higgs boson background and signal-versus-background classification.
3. Dataset Observatory: feature groups, distributions, correlations and class balance.
4. Experiment Comparison: model and feature-group metrics.
5. Validation: ROC, precision-recall, calibration and threshold analysis.
6. Methods and Research: experiment protocol, citations and limitations.
7. Model Simulation: run trained models on new data.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ and repo root are in python path
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st

from app.loader import discover_and_load_runs
from app.styles import apply_theme
from app.views.dataset import render_dataset
from app.views.experiments import render_experiments
from app.views.methods import render_methods
from app.views.overview import render_overview
from app.views.physics import render_physics
from app.views.simulation import render_simulation
from app.views.validation import render_validation

# Page Configuration
st.set_page_config(
    page_title="HiggsLens | Evidence Workbench",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    """Main dashboard entry point."""
    apply_theme()

    # Sidebar Header & Branding
    st.sidebar.markdown(
        """
        <div class="sidebar-brand">
            <h2>HiggsLens</h2>
            <p>Experiment dashboard · v0.2</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    status_slot = st.sidebar.container()
    navigation_slot = st.sidebar.container()

    # Keep setup controls available without placing them before navigation.
    with st.sidebar.expander("Artifact settings", icon=":material/folder_open:"):
        outputs_path = st.text_input(
            "Artifacts directory",
            value="outputs",
            help="Folder containing saved metrics.json and test_predictions.csv run packages.",
        )
        if st.button("Rescan artifacts", icon=":material/refresh:", width="stretch"):
            st.cache_data.clear()

    # Discover and validate experiment runs
    discovery = discover_and_load_runs(outputs_path)

    # Sidebar Run Status Badge
    valid_count = len(discovery.valid_runs)
    if valid_count > 0:
        badge_html = f"""
        <div class="scientific-badge scientific-badge-green" style="width: 100%; text-align: center; margin-bottom: 1rem;">
            <span class="material-icons icon-sm" style="vertical-align: middle; margin-right: 0.3rem;">check_circle</span>
            {valid_count} saved runs loaded
        </div>
        """
    elif discovery.dir_exists and discovery.total_scanned > 0:
        badge_html = f"""
        <div class="scientific-badge scientific-badge-amber" style="width: 100%; text-align: center; margin-bottom: 1rem;">
            <span class="material-icons icon-sm" style="vertical-align: middle; margin-right: 0.3rem;">error</span>
            0 Valid ({len(discovery.incomplete_runs)} incomplete, {len(discovery.malformed_runs)} invalid)
        </div>
        """
    else:
        badge_html = """
        <div class="scientific-badge scientific-badge-cyan" style="width: 100%; text-align: center; margin-bottom: 1rem;">
            <span class="material-icons icon-sm" style="vertical-align: middle; margin-right: 0.3rem;">schedule</span>
            Awaiting Experiment Runs
        </div>
        """
    status_slot.markdown(badge_html, unsafe_allow_html=True)

    if discovery.incomplete_runs or discovery.malformed_runs:
        with st.expander("Artifact loading diagnostics", expanded=not discovery.has_valid_runs):
            for run in discovery.incomplete_runs:
                st.warning(f"{run.run_id}: {run.reason}")
            for run in discovery.malformed_runs:
                st.error(f"{run.run_id}: {run.reason}")

    # Primary Navigation
    NAV_SECTIONS = [
        ("Overview", "home"),
        ("Physics Explorer", "science"),
        ("Dataset Observatory", "database"),
        ("Experiment Comparison", "bar_chart"),
        ("Validation", "insights"),
        ("Methods and Research", "article"),
        ("Model Simulation", "psychology"),
    ]

    navigation_slot.markdown('<div class="sidebar-nav-label">WORKSPACE</div>', unsafe_allow_html=True)
    selected_index = navigation_slot.radio(
        "Navigation",
        options=list(range(len(NAV_SECTIONS))),
        format_func=lambda i: NAV_SECTIONS[i][0],
        index=0,
        label_visibility="collapsed",
        key="dashboard_navigation",
    )

    selected_section = NAV_SECTIONS[selected_index][0]

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        <div class="sidebar-details">
            <span class="material-icons icon-sm" style="vertical-align: middle; margin-right: 0.3rem;">person</span>
            <strong>Lead UI / Product:</strong> Ronit Singh<br/>
            <span class="material-icons icon-sm" style="vertical-align: middle; margin-right: 0.3rem;">database</span>
            <strong>Dataset Benchmark:</strong> UCI HIGGS #280<br/>
            <span class="material-icons icon-sm" style="vertical-align: middle; margin-right: 0.3rem;">tune</span>
            <strong>Split:</strong> See saved metadata for each run
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Section Routing
    if selected_section == "Overview":
        render_overview(discovery)
    elif selected_section == "Physics Explorer":
        render_physics()
    elif selected_section == "Dataset Observatory":
        render_dataset(discovery.base_dir)
    elif selected_section == "Experiment Comparison":
        render_experiments(discovery)
    elif selected_section == "Validation":
        render_validation(discovery)
    elif selected_section == "Methods and Research":
        render_methods()
    elif selected_section == "Model Simulation":
        render_simulation(discovery)


if __name__ == "__main__":
    main()
