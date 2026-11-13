"""High-fidelity scientific visualizations for the HiggsLens dashboard.

Uses Matplotlib with high DPI, dark surfaces, and cyan/blue accents to render:
- Held-out ROC curves
- Precision-Recall curves with no-skill reference
- Reliability calibration diagrams
- Low-level vs high-level vs all feature comparisons
- MLP genuine training history (loss and validation ROC-AUC)
- Confusion matrix heatmaps for threshold operating points
- Distribution and correlation visualizers
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from app.loader import ExperimentArtifact
from app.metrics import (
    compute_calibration_points,
    compute_pr_points,
    compute_roc_points,
)
from app.styles import (
    ACCENT_AMBER,
    ACCENT_BLUE,
    ACCENT_CYAN,
    ACCENT_RED,
    ACCENT_TEAL,
    MPL_COLORS,
    SURFACE_BASE,
    SURFACE_CARD,
    SURFACE_BORDER,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)


def _configure_dark_ax(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    """Apply unified dark scientific aesthetic to a matplotlib Axes."""
    ax.set_facecolor(SURFACE_CARD)
    ax.set_title(title, color=TEXT_PRIMARY, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, color=TEXT_SECONDARY, fontsize=11, labelpad=8)
    ax.set_ylabel(ylabel, color=TEXT_SECONDARY, fontsize=11, labelpad=8)
    ax.tick_params(colors=TEXT_MUTED, labelsize=9.5)
    for spine in ax.spines.values():
        spine.set_color(SURFACE_BORDER)
        spine.set_linewidth(1.0)
    ax.grid(True, linestyle="--", alpha=0.25, color=SURFACE_BORDER)


def plot_roc_curves(runs: list[ExperimentArtifact]) -> Figure:
    """Render ROC curves for selected runs derived from held-out predictions."""
    fig, ax = plt.subplots(figsize=(7.5, 5.2), dpi=150, facecolor=SURFACE_BASE)
    _configure_dark_ax(
        ax,
        title="Receiver Operating Characteristic (ROC)",
        xlabel="False Positive Rate (1 - Background Rejection)",
        ylabel="True Positive Rate (Signal Efficiency)",
    )

    # Reference diagonal (random guessing)
    ax.plot([0, 1], [0, 1], linestyle=":", color=TEXT_MUTED, linewidth=1.2, label="Random Guess (AUC = 0.500)")

    for idx, run in enumerate(runs):
        color = MPL_COLORS[idx % len(MPL_COLORS)]
        try:
            fpr, tpr, auc_score = compute_roc_points(run.test_labels, run.test_scores)
            label = f"{run.model_name} [{run.feature_group}] (AUC = {auc_score:.4f})"
            ax.plot(fpr, tpr, color=color, linewidth=2.0, label=label)
        except Exception as exc:
            ax.text(
                0.05,
                0.95 - (idx * 0.05),
                f"{run.run_id}: evaluation error ({exc})",
                color=ACCENT_RED,
                fontsize=9,
            )

    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.02)
    legend = ax.legend(loc="lower right", facecolor=SURFACE_CARD, edgecolor=SURFACE_BORDER, fontsize=9)
    for text in legend.get_texts():
        text.set_color(TEXT_SECONDARY)

    fig.tight_layout()
    return fig


def plot_pr_curves(runs: list[ExperimentArtifact]) -> Figure:
    """Render Precision-Recall curves derived from held-out predictions."""
    fig, ax = plt.subplots(figsize=(7.5, 5.2), dpi=150, facecolor=SURFACE_BASE)
    _configure_dark_ax(
        ax,
        title="Precision-Recall Curves (PR)",
        xlabel="Recall (Signal Efficiency)",
        ylabel="Precision (Positive Predictive Value)",
    )

    # Calculate baseline signal fraction from first run if available
    if runs:
        first_labels = runs[0].test_labels
        pos_frac = float(np.mean(first_labels == 1))
        ax.axhline(
            y=pos_frac,
            linestyle=":",
            color=TEXT_MUTED,
            linewidth=1.2,
            label=f"Class Balance Baseline ({pos_frac:.1%})",
        )

    for idx, run in enumerate(runs):
        color = MPL_COLORS[idx % len(MPL_COLORS)]
        try:
            prec, rec, ap = compute_pr_points(run.test_labels, run.test_scores)
            label = f"{run.model_name} [{run.feature_group}] (PR-AUC = {ap:.4f})"
            ax.plot(rec, prec, color=color, linewidth=2.0, label=label)
        except Exception as exc:
            ax.text(
                0.05,
                0.95 - (idx * 0.05),
                f"{run.run_id}: evaluation error ({exc})",
                color=ACCENT_RED,
                fontsize=9,
            )

    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(0.0, 1.02)
    legend = ax.legend(loc="lower left", facecolor=SURFACE_CARD, edgecolor=SURFACE_BORDER, fontsize=9)
    for text in legend.get_texts():
        text.set_color(TEXT_SECONDARY)

    fig.tight_layout()
    return fig


def plot_calibration_curves(runs: list[ExperimentArtifact], n_bins: int = 10) -> Figure:
    """Render reliability diagrams (calibration curve) for selected runs."""
    fig, ax = plt.subplots(figsize=(7.5, 5.2), dpi=150, facecolor=SURFACE_BASE)
    _configure_dark_ax(
        ax,
        title="Calibration Reliability Diagram",
        xlabel="Mean Predicted Probability",
        ylabel="Observed Signal Frequency",
    )

    # Perfect calibration 45-degree line
    ax.plot([0, 1], [0, 1], linestyle="--", color=TEXT_MUTED, linewidth=1.2, label="Perfect Calibration")

    for idx, run in enumerate(runs):
        color = MPL_COLORS[idx % len(MPL_COLORS)]
        try:
            frac_pos, mean_pred, brier = compute_calibration_points(
                run.test_labels, run.test_scores, n_bins=n_bins
            )
            label = f"{run.model_name} [{run.feature_group}] (Brier = {brier:.4f})"
            ax.plot(mean_pred, frac_pos, marker="s", markersize=5, color=color, linewidth=1.8, label=label)
        except Exception as exc:
            ax.text(
                0.05,
                0.95 - (idx * 0.05),
                f"{run.run_id}: calibration error ({exc})",
                color=ACCENT_RED,
                fontsize=9,
            )

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    legend = ax.legend(loc="upper left", facecolor=SURFACE_CARD, edgecolor=SURFACE_BORDER, fontsize=9)
    for text in legend.get_texts():
        text.set_color(TEXT_SECONDARY)

    fig.tight_layout()
    return fig


def plot_feature_group_comparison(runs: list[ExperimentArtifact]) -> Figure:
    """Compare low_level, high_level, and all feature configurations across models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8), dpi=150, facecolor=SURFACE_BASE)

    _configure_dark_ax(ax1, "Test ROC-AUC by Feature Group", "Feature Group", "ROC-AUC Score")
    _configure_dark_ax(ax2, "Test PR-AUC by Feature Group", "Feature Group", "PR-AUC Score")

    group_order = ["low_level", "high_level", "all"]
    models = sorted({r.model_name for r in runs})
    model_colors = {
        "logistic_regression": ACCENT_AMBER,
        "xgboost": ACCENT_BLUE,
        "compact_mlp": ACCENT_CYAN,
    }

    width = 0.35
    x = np.arange(len(group_order))

    for m_idx, model in enumerate(models):
        model_runs = [r for r in runs if r.model_name == model]
        m_runs = {}
        for run in model_runs:
            if run.feature_group in m_runs:
                raise ValueError("Select one run per model and feature group for grouped charts")
            m_runs[run.feature_group] = run
        roc_vals = [m_runs[g].test_roc_auc if g in m_runs and m_runs[g].test_roc_auc is not None else np.nan for g in group_order]
        pr_vals = [m_runs[g].test_pr_auc if g in m_runs and m_runs[g].test_pr_auc is not None else np.nan for g in group_order]

        color = model_colors.get(model, MPL_COLORS[m_idx % len(MPL_COLORS)])
        offset = (m_idx - (len(models) - 1) / 2) * width

        bars1 = ax1.bar(x + offset, roc_vals, width, label=model, color=color, alpha=0.85, edgecolor=SURFACE_BORDER)
        bars2 = ax2.bar(x + offset, pr_vals, width, label=model, color=color, alpha=0.85, edgecolor=SURFACE_BORDER)

        # Value annotations
        for bar, val in zip(bars1, roc_vals):
            if val > 0:
                ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{val:.3f}", ha="center", va="bottom", color=TEXT_PRIMARY, fontsize=8.5)

        for bar, val in zip(bars2, pr_vals):
            if val > 0:
                ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{val:.3f}", ha="center", va="bottom", color=TEXT_PRIMARY, fontsize=8.5)

    ax1.set_xticks(x)
    ax1.set_xticklabels(["Low-level (21)", "High-level (7)", "All (28)"])
    maximum_roc = max((r.test_roc_auc or 0.0 for r in runs), default=0.9)
    maximum_pr = max((r.test_pr_auc or 0.0 for r in runs), default=0.9)
    ax1.set_ylim(max(0.4, min(0.75, maximum_roc - 0.35)), min(1.0, maximum_roc + 0.08))
    leg1 = ax1.legend(facecolor=SURFACE_CARD, edgecolor=SURFACE_BORDER, fontsize=9)
    for text in leg1.get_texts():
        text.set_color(TEXT_SECONDARY)

    ax2.set_xticks(x)
    ax2.set_xticklabels(["Low-level (21)", "High-level (7)", "All (28)"])
    ax2.set_ylim(max(0.4, min(0.75, maximum_pr - 0.35)), min(1.0, maximum_pr + 0.08))
    leg2 = ax2.legend(facecolor=SURFACE_CARD, edgecolor=SURFACE_BORDER, fontsize=9)
    for text in leg2.get_texts():
        text.set_color(TEXT_SECONDARY)

    fig.tight_layout()
    return fig


def plot_mlp_training_history(run: ExperimentArtifact) -> Figure | None:
    """Plot genuine MLP loss and validation ROC-AUC learning curves.

    Returns None if no genuine training history is present in the artifact.
    """
    if not run.has_training_history:
        return None

    history = run.training_history
    epochs = [h.get("epoch", idx + 1) for idx, h in enumerate(history)]
    train_loss = [h["train_loss"] for h in history]
    val_auc = [h["validation_roc_auc"] for h in history]

    fig, ax1 = plt.subplots(figsize=(7.5, 4.6), dpi=150, facecolor=SURFACE_BASE)
    _configure_dark_ax(
        ax1,
        title=f"MLP Training History: {run.run_id}",
        xlabel="Epoch",
        ylabel="Training BCE Loss",
    )

    line1 = ax1.plot(epochs, train_loss, color=ACCENT_CYAN, linewidth=2.0, marker="o", markersize=4, label="Train Loss")
    ax1.set_ylabel("Training Loss", color=ACCENT_CYAN, fontsize=10.5)
    ax1.tick_params(axis="y", labelcolor=ACCENT_CYAN)

    ax2 = ax1.twinx()
    ax2.set_facecolor("none")
    line2 = ax2.plot(epochs, val_auc, color=ACCENT_TEAL, linewidth=2.0, marker="s", markersize=4, linestyle="--", label="Val ROC-AUC")
    ax2.set_ylabel("Validation ROC-AUC", color=ACCENT_TEAL, fontsize=10.5)
    ax2.tick_params(axis="y", labelcolor=ACCENT_TEAL)
    for spine in ax2.spines.values():
        spine.set_color(SURFACE_BORDER)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    leg = ax1.legend(lines, labels, loc="center right", facecolor=SURFACE_CARD, edgecolor=SURFACE_BORDER, fontsize=9)
    for text in leg.get_texts():
        text.set_color(TEXT_SECONDARY)

    fig.tight_layout()
    return fig


def plot_confusion_matrix_heatmap(cm_dict: dict[str, int]) -> Figure:
    """Plot confusion matrix counts in a crisp 2x2 scientific heatmap."""
    fig, ax = plt.subplots(figsize=(4.8, 3.8), dpi=150, facecolor=SURFACE_BASE)
    _configure_dark_ax(
        ax,
        title=f"Confusion Matrix (threshold = {cm_dict['threshold']:.2f})",
        xlabel="Predicted Class",
        ylabel="True Label",
    )

    tp = cm_dict["true_positives"]
    fp = cm_dict["false_positives"]
    tn = cm_dict["true_negatives"]
    fn = cm_dict["false_negatives"]

    matrix = np.array([[tn, fp], [fn, tp]])
    im = ax.imshow(matrix, cmap="Blues", alpha=0.85)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["0 (Background)", "1 (Signal)"])
    ax.set_yticklabels(["0 (Background)", "1 (Signal)"])

    labels = [
        [f"TN\n{tn:,}", f"FP\n{fp:,}"],
        [f"FN\n{fn:,}", f"TP\n{tp:,}"],
    ]

    for i in range(2):
        for j in range(2):
            val = matrix[i, j]
            txt_color = "white" if val > matrix.max() * 0.55 else TEXT_PRIMARY
            ax.text(j, i, labels[i][j], ha="center", va="center", color=txt_color, fontsize=11, fontweight="bold")

    fig.tight_layout()
    return fig


def plot_runtime_vs_auc(runs: list[ExperimentArtifact]) -> Figure:
    """Scatter plot comparing training runtime in seconds versus held-out test ROC-AUC."""
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150, facecolor=SURFACE_BASE)
    _configure_dark_ax(
        ax,
        title="Compute Efficiency: Runtime vs Test Performance",
        xlabel="Training Time (seconds)",
        ylabel="Held-out Test ROC-AUC",
    )

    for idx, run in enumerate(runs):
        runtime = run.training_seconds
        auc = run.test_roc_auc
        if runtime is not None and auc is not None:
            color = MPL_COLORS[idx % len(MPL_COLORS)]
            marker = "o" if run.is_mlp else "^"
            ax.scatter(runtime, auc, color=color, s=120, edgecolors=TEXT_PRIMARY, linewidth=1.5, marker=marker, label=f"{run.model_name} ({run.feature_group})")
            ax.annotate(
                f"{run.run_id}",
                (runtime, auc),
                textcoords="offset points",
                xytext=(8, 4),
                ha="left",
                color=TEXT_SECONDARY,
                fontsize=8.5,
            )

    # Only show legend if there are labeled artists to avoid empty-legend warnings
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        legend = ax.legend(loc="lower right", facecolor=SURFACE_CARD, edgecolor=SURFACE_BORDER, fontsize=8.5)
        for text in legend.get_texts():
            text.set_color(TEXT_SECONDARY)

    fig.tight_layout()
    return fig


def plot_score_distributions(runs: list[ExperimentArtifact]) -> Figure:
    """Overlay predicted score distributions for selected runs, split by class."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), dpi=150, facecolor=SURFACE_BASE)

    titles = ["Signal Class (y=1) Score Distribution", "Background Class (y=0) Score Distribution"]
    for ax_idx, (ax, title) in enumerate(zip(axes, titles)):
        _configure_dark_ax(ax, title, "Predicted Signal Probability", "Event Density")
        ax.set_xlim(0.0, 1.0)

    for idx, run in enumerate(runs):
        color = MPL_COLORS[idx % len(MPL_COLORS)]
        scores = run.test_scores
        labels = run.test_labels

        # Signal class distribution
        sig_scores = scores[labels == 1]
        if len(sig_scores) > 0:
            axes[0].hist(sig_scores, bins=30, density=True, alpha=0.5, color=color, label=run.run_id, edgecolor="none")

        # Background class distribution
        bkg_scores = scores[labels == 0]
        if len(bkg_scores) > 0:
            axes[1].hist(bkg_scores, bins=30, density=True, alpha=0.5, color=color, label=run.run_id, edgecolor="none")

    for ax in axes:
        handles, labels_ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(facecolor=SURFACE_CARD, edgecolor=SURFACE_BORDER, fontsize=8)

    fig.tight_layout()
    return fig


def plot_feature_group_heatmap(runs: list[ExperimentArtifact]) -> Figure:
    """Heatmap of ROC-AUC and PR-AUC across models and feature groups."""
    group_order = ["low_level", "high_level", "all"]
    models = sorted({r.model_name for r in runs})

    roc_matrix = np.full((len(models), len(group_order)), np.nan)
    pr_matrix = np.full((len(models), len(group_order)), np.nan)

    for m_idx, model in enumerate(models):
        model_runs = [r for r in runs if r.model_name == model]
        m_runs = {}
        for run in model_runs:
            if run.feature_group in m_runs:
                raise ValueError("Select one run per model and feature group for grouped charts")
            m_runs[run.feature_group] = run
        for g_idx, group in enumerate(group_order):
            if group in m_runs and m_runs[group].test_roc_auc is not None:
                roc_matrix[m_idx, g_idx] = m_runs[group].test_roc_auc
                pr_matrix[m_idx, g_idx] = m_runs[group].test_pr_auc

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=150, facecolor=SURFACE_BASE)

    for ax_idx, (matrix, title, cmap) in enumerate([
        (roc_matrix, "Test ROC-AUC Heatmap", "Blues"),
        (pr_matrix, "Test PR-AUC Heatmap", "Greens"),
    ]):
        ax = axes[ax_idx]
        _configure_dark_ax(ax, title, "Feature Group", "Model")

        im = ax.imshow(matrix, cmap=cmap, vmin=0.4, vmax=1.0, aspect="auto")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set_xticks(range(len(group_order)))
        ax.set_xticklabels(["Low-level\n(21)", "High-level\n(7)", "All\n(28)"])
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models)

        # Annotate values
        for i in range(len(models)):
            for j in range(len(group_order)):
                val = matrix[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=TEXT_PRIMARY, fontsize=9, fontweight="bold")

    fig.tight_layout()
    return fig


def plot_model_performance_radar(runs: list[ExperimentArtifact]) -> Figure:
    """Radar/spider chart comparing models across ROC-AUC, PR-AUC, and calibration."""
    import math

    metrics = ["ROC-AUC", "PR-AUC", "1 - Brier", "Background Rejection"]
    n_metrics = len(metrics)

    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=150, facecolor=SURFACE_BASE, subplot_kw=dict(projection="polar"))

    angles = [n / float(n_metrics) * 2 * math.pi for n in range(n_metrics)]
    angles += angles[:1]

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, color=TEXT_SECONDARY, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], color=TEXT_MUTED, fontsize=8)
    ax.grid(color=SURFACE_BORDER, alpha=0.3)

    for idx, run in enumerate(runs[:5]):
        color = MPL_COLORS[idx % len(MPL_COLORS)]
        roc = run.test_roc_auc if run.test_roc_auc is not None else np.nan
        pr = run.test_pr_auc if run.test_pr_auc is not None else np.nan
        brier = run.test_brier_score if run.test_brier_score is not None else np.nan
        op = run.operating_point
        bkg_rej = op.get("test_background_rejection", np.nan) if op else np.nan

        values = [roc, pr, 1.0 - brier, bkg_rej]
        values += values[:1]

        ax.plot(angles, values, color=color, linewidth=2, label=f"{run.model_name} [{run.feature_group}]")
        ax.fill(angles, values, color=color, alpha=0.15)

    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), facecolor=SURFACE_CARD, edgecolor=SURFACE_BORDER, fontsize=8)
    ax.set_title("Model Performance Radar", color=TEXT_PRIMARY, fontsize=12, fontweight="bold", pad=20)

    fig.tight_layout()
    return fig


def plot_class_balance_donut(run: ExperimentArtifact) -> Figure:
    """Donut chart of signal vs background class balance in the held-out test set."""
    labels = run.test_labels
    n_signal = int(np.sum(labels == 1))
    n_background = int(np.sum(labels == 0))

    fig, ax = plt.subplots(figsize=(4.6, 4.2), dpi=150, facecolor=SURFACE_BASE)

    sizes = [n_signal, n_background]
    colors = [ACCENT_CYAN, ACCENT_BLUE]
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=["Signal (y=1)", "Background (y=0)"],
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        counterclock=False,
        wedgeprops=dict(width=0.45, edgecolor=SURFACE_BASE),
        textprops=dict(color=TEXT_SECONDARY, fontsize=9),
    )

    for autotext in autotexts:
        autotext.set_color(TEXT_PRIMARY)
        autotext.set_fontweight("bold")

    ax.set_title("Held-Out Test Class Balance", color=TEXT_PRIMARY, fontsize=11, fontweight="bold", pad=12)

    # Center text
    ax.text(0, 0, f"{len(labels):,}\nevents", ha="center", va="center", color=TEXT_PRIMARY, fontsize=10, fontweight="bold")

    fig.tight_layout()
    return fig


def plot_precision_recall_vs_threshold(run: ExperimentArtifact) -> Figure:
    """Precision and Recall as functions of decision threshold."""
    from app.metrics import compute_pr_points

    precision, recall, _ = compute_pr_points(run.test_labels, run.test_scores)
    # precision_recall_curve returns precision and recall arrays aligned with thresholds
    from sklearn.metrics import precision_recall_curve
    prec, rec, thresholds = precision_recall_curve(run.test_labels, run.test_scores)

    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=150, facecolor=SURFACE_BASE)
    _configure_dark_ax(ax, "Precision &amp; Recall vs Decision Threshold", "Threshold", "Score")

    # thresholds has one fewer element than precision/recall
    ax.plot(thresholds, prec[:-1], color=ACCENT_CYAN, linewidth=2, label="Precision", marker="o", markersize=3)
    ax.plot(thresholds, rec[:-1], color=ACCENT_TEAL, linewidth=2, label="Recall (Signal Efficiency)", marker="s", markersize=3)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)

    legend = ax.legend(facecolor=SURFACE_CARD, edgecolor=SURFACE_BORDER, fontsize=9)
    for text in legend.get_texts():
        text.set_color(TEXT_SECONDARY)

    fig.tight_layout()
    return fig
