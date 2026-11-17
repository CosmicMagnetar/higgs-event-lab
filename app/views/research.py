"""Research rationale and explicitly scoped release-summary comparisons."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from app.views.overview import _model_name

REFERENCES = [
    ("Baldi et al. (2014): the HIGGS representation-learning motivation", "https://doi.org/10.1038/ncomms5308"),
    ("Glorot & Bengio (2010): activation and initialization affect deep-network optimization", "https://proceedings.mlr.press/v9/glorot10a.html"),
    ("Chen & Guestrin (2016): scalable boosted-tree baseline", "https://arxiv.org/abs/1603.02754"),
    ("Gorishniy et al. (2021): residual MLPs and FT-Transformer for tabular data", "https://arxiv.org/abs/2106.11959"),
    ("Grinsztajn et al. (2022): why trees remain strong on typical tabular tasks", "https://arxiv.org/abs/2207.08815"),
    ("UCI HIGGS: simulated events, feature definitions and original test convention", "https://archive.ics.uci.edu/dataset/280/higgs"),
]


def render_research(base_dir: str = "outputs") -> None:
    st.title("Research Evidence & Model Rationale")
    st.markdown("A hypothesis to test, a comparison to measure, and a trade-off to explain.")
    rationale, results, next_steps, sources = st.tabs(["Model choices", "Scaling evidence", "Next experiments", "Papers & report"])
    with rationale:
        st.info("Core project: Logistic Regression, XGBoost and compact MLP. Deep and residual networks are exploratory extensions. At 250k, their all-feature accuracy gains over XGBoost are modest; higher compute is not automatically justified.")
        st.subheader("Required baseline → representation gap → deeper models")
        st.markdown("The brief requires Logistic Regression, XGBoost and a compact MLP at a fixed budget. "
                    "Depth is an extension: test whether additional capacity improves classification of smooth, correlated kinematics. "
                    "Keep XGBoost in the comparison; replacing it would remove the evidence for choosing a neural model.")
        selected = st.selectbox("Inspect a neural architecture", ["Compact MLP", "Deep MLP", "Residual MLP"])
        details = {
            "Compact MLP": ("D → 64 → 32 → 1 logit", "Linear → BatchNorm → ReLU → Dropout(0.20) in each hidden layer. AdamW, lr 0.001, weight decay 0.0001; 30 epochs maximum, validation-AUC patience 5.", "Required by the brief. Small capacity and a useful control; a weak result does not rule out larger neural models."),
            "Deep MLP": ("D → 300 → 300 → 300 → 300 → 300 → 1 logit", "Linear → BatchNorm → ReLU → Dropout(0.10). AdamW, lr 0.001, weight decay 0.00001, cosine schedule; 60 epochs maximum, validation-AUC patience 8.", "Tests a larger network inspired by Baldi's depth and width. It is not an exact reproduction: activation, optimizer, regularization, data and training budget differ."),
            "Residual MLP": ("D → 256 → [residual block] × 4 → BatchNorm → ReLU → 1 logit", "Block: x + Linear(512→256)(Dropout(ReLU(Linear(256→512)(BatchNorm(x))))), followed by branch dropout. AdamW and stopping settings match the deep preset.", "Tests residual optimization inspired by tabular ResNet baselines. It also has a different parameter count, so this is not a depth-only ablation."),
        }
        architecture, training, reason = details[selected]
        st.code(architecture, language="text")
        st.write(training)
        st.info(reason)
        st.caption("D is 21, 7 or 28. All networks output logits, train with BCEWithLogitsLoss and restore the best validation-AUC weights. Sigmoid converts logits to scores at evaluation.")
        st.latex(r"h_{l+1}=h_l+F(h_l),\qquad p(y=1\mid x)=\sigma(z(x))")
        st.markdown("**Why not CNN or LSTM?** These 28 columns have no natural image grid or time sequence. "
                    "**Why not a GNN?** A graph comparison needs an explicitly justified particle/edge representation. "
                    "**Why consider FT-Transformer?** Feature-token attention is designed for tabular interactions, but needs its own fair tuning and compute budget.")
    with results:
        path = Path(base_dir) / "release" / "release_summary.json"
        if not path.is_file():
            st.info("No release summary in the selected artifact directory. Import outputs/release/release_summary.json to compare budgets.")
        else:
            try:
                summary = json.loads(path.read_text())
                budgets = summary["budgets"]
                budget = st.selectbox("Sample budget", list(budgets), format_func=lambda k: f"{budgets[k]['rows']:,} events")
                group = st.selectbox("Feature representation", ["all", "low_level", "high_level"])
                rows = [dict(r) for r in budgets[budget]["results"] if r["group"] == group]
                frame = pd.DataFrame(rows)
                frame["Model"] = frame["model"].map(lambda x: _model_name({"logreg": "logistic_regression", "mlp": "compact_mlp"}.get(x, x)))
                st.dataframe(frame[["Model", "roc_auc", "pr_auc", "brier", "ece", "train_seconds"]], hide_index=True, use_container_width=True)
                chart = pd.DataFrame([{"Events": b["rows"], "Model": r["model"], "ROC-AUC": r["roc_auc"]}
                                      for b in budgets.values() for r in b["results"] if r["group"] == group])
                st.line_chart(chart, x="Events", y="ROC-AUC", color="Model")
                st.caption("Source: the selected directory's release/release_summary.json. Seed 42 in the published release. Training time is not inference latency; ECE uses 15 equal-width bins.")
                st.warning("Saved summaries, not fresh training. Larger-budget prediction files are not included in this checkout. Models share test rows within each budget, but the test set changes across budgets; this is not a fixed-test learning curve.")
                st.download_button("Download selected comparison", frame.to_csv(index=False), f"higgs-{budget}-{group}.csv", "text/csv")
            except (OSError, ValueError, KeyError, TypeError) as exc:
                st.error(f"Cannot read release summary: {exc}")
    with next_steps:
        st.markdown("1. **Repeat seeds 42, 43 and 44**, with shared splits for every model within a seed. Separate run-to-run variability from test-sample uncertainty.\n"
                    "2. **Freeze one test set before sampling training budgets.** Select stratified, nested training subsets and record all indices.\n"
                    "3. **Match tuning budgets.** Log validation-only trials for XGBoost and neural models; early stopping alone is not a hyperparameter search.\n"
                    "4. **Test calibration on a reserved calibration partition.** Compare temperature scaling before evaluating the untouched test set.\n"
                    "5. **Ablate architecture.** Compare parameter-matched shallow/deep networks and residual on/off, keeping optimizer and epochs fixed.\n"
                    "6. **Optional FT-Transformer.** Start at 250k on all and low-level features; expand only after the baseline protocol passes.")
        st.info("The existing three neural architectures are sufficient for the project brief. These experiments strengthen the claim more than an arbitrary list of additional models.")
    with sources:
        for title, url in REFERENCES:
            st.markdown(f"- [{title}]({url})")
        st.caption("Papers motivate experiments; they do not certify the project's measured results or imply universal DL superiority.")
        report = Path(__file__).resolve().parents[2] / "reports" / "overleaf"
        for filename, label, mime in [("main.pdf", "Download research report", "application/pdf"),
                                      ("HiggsLens_Overleaf.zip", "Download Overleaf source", "application/zip")]:
            target = report / filename
            if target.is_file():
                st.download_button(label, target.read_bytes(), filename, mime)
