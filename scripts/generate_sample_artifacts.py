"""Generate synthetic fixtures for dashboard development, never research results.

Generates schema-compliant outputs for both dashboard entry points:
1. outputs/<run_id>/ (metrics.json and test_predictions.csv for app/main.py)
2. outputs/experiments.json and outputs/curves/<run_id>.json (for app/streamlit_app.py)

The metric constants, predictions, histories and curves are illustrative synthetic
fixtures, not measured HIGGS runs. Defaults are isolated from the real registry.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = ROOT_DIR / "artifacts" / "synthetic_demo"
CURVES_DIR = OUTPUTS_DIR / "curves"

BENCHMARK_CONFIGS = [
    {
        "run_id": "mlp_all",
        "model_name": "compact_mlp",
        "model_family": "neural_network",
        "feature_group": "all",
        "feature_count": 28,
        "runtime_sec": 10.52,
        "val_roc_auc": 0.9389,
        "test_roc_auc": 0.9406,
        "test_pr_auc": 0.9372,
        "brier_score": 0.0945,
        "accuracy": 0.8650,
        "target_eff": 0.50,
        "sel_thresh": 0.582,
        "test_bkg_rej": 0.962,
        "test_sig_eff": 0.501,
        "epochs": 20,
    },
    {
        "run_id": "mlp_high_level",
        "model_name": "compact_mlp",
        "model_family": "neural_network",
        "feature_group": "high_level",
        "feature_count": 7,
        "runtime_sec": 5.13,
        "val_roc_auc": 0.9016,
        "test_roc_auc": 0.9043,
        "test_pr_auc": 0.8888,
        "brier_score": 0.1184,
        "accuracy": 0.8240,
        "target_eff": 0.50,
        "sel_thresh": 0.541,
        "test_bkg_rej": 0.925,
        "test_sig_eff": 0.498,
        "epochs": 10,
    },
    {
        "run_id": "mlp_low_level",
        "model_name": "compact_mlp",
        "model_family": "neural_network",
        "feature_group": "low_level",
        "feature_count": 21,
        "runtime_sec": 15.02,
        "val_roc_auc": 0.7997,
        "test_roc_auc": 0.8000,
        "test_pr_auc": 0.8097,
        "brier_score": 0.1824,
        "accuracy": 0.7280,
        "target_eff": 0.50,
        "sel_thresh": 0.512,
        "test_bkg_rej": 0.841,
        "test_sig_eff": 0.504,
        "epochs": 30,
    },
    {
        "run_id": "logreg_all",
        "model_name": "logistic_regression",
        "model_family": "classical_ml",
        "feature_group": "all",
        "feature_count": 28,
        "runtime_sec": 0.10,
        "val_roc_auc": 0.8385,
        "test_roc_auc": 0.8369,
        "test_pr_auc": 0.8377,
        "brier_score": 0.1639,
        "accuracy": 0.7550,
        "target_eff": 0.50,
        "sel_thresh": 0.528,
        "test_bkg_rej": 0.865,
        "test_sig_eff": 0.502,
        "epochs": 1,
    },
    {
        "run_id": "logreg_high_level",
        "model_name": "logistic_regression",
        "model_family": "classical_ml",
        "feature_group": "high_level",
        "feature_count": 7,
        "runtime_sec": 0.05,
        "val_roc_auc": 0.7702,
        "test_roc_auc": 0.7661,
        "test_pr_auc": 0.7427,
        "brier_score": 0.1945,
        "accuracy": 0.6980,
        "target_eff": 0.50,
        "sel_thresh": 0.519,
        "test_bkg_rej": 0.812,
        "test_sig_eff": 0.497,
        "epochs": 1,
    },
    {
        "run_id": "logreg_low_level",
        "model_name": "logistic_regression",
        "model_family": "classical_ml",
        "feature_group": "low_level",
        "feature_count": 21,
        "runtime_sec": 0.07,
        "val_roc_auc": 0.7479,
        "test_roc_auc": 0.7476,
        "test_pr_auc": 0.7622,
        "brier_score": 0.2035,
        "accuracy": 0.6810,
        "target_eff": 0.50,
        "sel_thresh": 0.510,
        "test_bkg_rej": 0.795,
        "test_sig_eff": 0.501,
        "epochs": 1,
    },
    # --- XGBoost Baselines ---
    {
        "run_id": "xgboost_all",
        "model_name": "xgboost",
        "model_family": "tree_ensemble",
        "feature_group": "all",
        "feature_count": 28,
        "runtime_sec": 18.40,
        "val_roc_auc": 0.8918,
        "test_roc_auc": 0.8924,
        "test_pr_auc": 0.8897,
        "brier_score": 0.1252,
        "accuracy": 0.8180,
        "target_eff": 0.50,
        "sel_thresh": 0.558,
        "test_bkg_rej": 0.941,
        "test_sig_eff": 0.501,
        "epochs": 150,
    },
    {
        "run_id": "xgboost_high_level",
        "model_name": "xgboost",
        "model_family": "tree_ensemble",
        "feature_group": "high_level",
        "feature_count": 7,
        "runtime_sec": 7.85,
        "val_roc_auc": 0.8361,
        "test_roc_auc": 0.8379,
        "test_pr_auc": 0.8243,
        "brier_score": 0.1588,
        "accuracy": 0.7760,
        "target_eff": 0.50,
        "sel_thresh": 0.531,
        "test_bkg_rej": 0.897,
        "test_sig_eff": 0.503,
        "epochs": 150,
    },
    {
        "run_id": "xgboost_low_level",
        "model_name": "xgboost",
        "model_family": "tree_ensemble",
        "feature_group": "low_level",
        "feature_count": 21,
        "runtime_sec": 12.30,
        "val_roc_auc": 0.8141,
        "test_roc_auc": 0.8152,
        "test_pr_auc": 0.8229,
        "brier_score": 0.1714,
        "accuracy": 0.7490,
        "target_eff": 0.50,
        "sel_thresh": 0.521,
        "test_bkg_rej": 0.858,
        "test_sig_eff": 0.502,
        "epochs": 150,
    },
]


def generate_synthetic_predictions(test_size: int = 500, roc_auc: float = 0.85, seed: int = 42) -> pd.DataFrame:
    """Generate realistic prediction scores calibrated around a target ROC-AUC."""
    rng = np.random.default_rng(seed)
    y_test = rng.integers(0, 2, size=test_size).astype(int)
    # Signal events score higher on average
    shift = (roc_auc - 0.5) * 4.0
    scores = np.where(
        y_test == 1,
        rng.normal(loc=0.5 + shift * 0.3, scale=0.3, size=test_size),
        rng.normal(loc=0.5 - shift * 0.3, scale=0.3, size=test_size),
    )
    scores = np.clip(scores, 0.001, 0.999)
    return pd.DataFrame({
        "row_index": np.arange(test_size),
        "class_label": y_test,
        "score": scores,
    })


def main() -> None:
    """Generate all artifact directories and aggregate records."""
    if OUTPUTS_DIR.resolve() == (ROOT_DIR / "outputs").resolve():
        raise ValueError("Synthetic fixtures must not be written into the research outputs directory.")
    if OUTPUTS_DIR.exists() and any(OUTPUTS_DIR.iterdir()):
        raise FileExistsError("Use an empty directory for synthetic fixtures; existing artifacts are preserved.")
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    CURVES_DIR.mkdir(parents=True, exist_ok=True)

    experiments_json = []

    for cfg in BENCHMARK_CONFIGS:
        run_id = cfg["run_id"]
        run_dir = OUTPUTS_DIR / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # 1. Generate test_predictions.csv
        pred_df = generate_synthetic_predictions(test_size=500, roc_auc=cfg["test_roc_auc"], seed=42)
        pred_path = run_dir / "test_predictions.csv"
        pred_df.to_csv(pred_path, index=False)

        # 2. Generate metrics.json (higgs-experiment-v1 envelope)
        prob_bins = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        envelope = {
            "schema_version": "higgs-experiment-v1",
            "artifact_kind": "synthetic_demo",
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": {
                "name": cfg["model_name"],
                "family": cfg["model_family"],
            },
            "features": {
                "group": cfg["feature_group"],
                "count": cfg["feature_count"],
                "names": [f"feat_{i}" for i in range(cfg["feature_count"])],
            },
            "data": {
                "source": "SYNTHETIC DEMO — not a measured HIGGS experiment",
                "row_count": 250000,
            },
            "split": {
                "protocol_version": "higgs-split-v1",
                "seed": 42,
                "validation_fraction": 0.15,
                "test_fraction": 0.15,
                "train_rows": 175000,
                "validation_rows": 37500,
                "test_rows": 37500,
            },
            "training": {
                "runtime_seconds": cfg["runtime_sec"],
                "device": "cpu",
                "epochs_completed": cfg["epochs"],
                "learning_rate": 0.001,
            },
            "evaluation": {
                "validation": {
                    "roc_auc": cfg["val_roc_auc"],
                    "pr_auc": cfg["test_pr_auc"] - 0.005,
                    "brier_score": cfg["brier_score"] + 0.002,
                    "calibration": {
                        "fraction_positives": [max(0.0, p + 0.02) for p in prob_bins],
                        "mean_predicted_value": prob_bins,
                    },
                },
                "test": {
                    "roc_auc": cfg["test_roc_auc"],
                    "pr_auc": cfg["test_pr_auc"],
                    "brier_score": cfg["brier_score"],
                    "accuracy": cfg["accuracy"],
                    "calibration": {
                        "fraction_positives": [max(0.0, p + 0.01) for p in prob_bins],
                        "mean_predicted_value": prob_bins,
                    },
                },
            },
        }

        with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(envelope, f, indent=2)

        # 3. Entry for outputs/experiments.json
        exp_entry = {
            "run_id": run_id,
            "model_name": cfg["model_name"],
            "feature_group": cfg["feature_group"],
            "seed": 42,
            "sample_size": 250000,
            "metrics": {
                "roc_auc": cfg["test_roc_auc"],
                "pr_auc": cfg["test_pr_auc"],
                "brier_score": cfg["brier_score"],
                "accuracy": cfg["accuracy"],
                "training_time_sec": cfg["runtime_sec"],
                "inference_latency_ms": 1.2,
            },
            "operating_point": {
                "target_signal_efficiency": cfg["target_eff"],
                "selected_threshold": cfg["sel_thresh"],
                "test_background_rejection": cfg["test_bkg_rej"],
                "test_signal_efficiency": cfg["test_sig_eff"],
            },
            "provenance": {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "author": "CosmicMagnetar",
            },
        }
        experiments_json.append(exp_entry)

        # 4. Curve coordinates for outputs/curves/<run_id>.json
        fpr = np.linspace(0.0, 1.0, 50).tolist()
        tpr = (np.array(fpr) ** (1.0 / (cfg["test_roc_auc"] * 2.5))).tolist()
        recall = np.linspace(0.0, 1.0, 50).tolist()
        precision = (1.0 - 0.4 * (np.array(recall) ** 2)).tolist()
        curve_payload = {
            "run_id": run_id,
            "roc": {"fpr": fpr, "tpr": tpr},
            "pr": {"precision": precision, "recall": recall},
            "calibration": {
                "fraction_of_positives": [max(0.0, p + 0.01) for p in prob_bins],
                "mean_predicted_value": prob_bins,
            },
        }
        with open(CURVES_DIR / f"{run_id}.json", "w", encoding="utf-8") as f:
            json.dump(curve_payload, f, indent=2)

    with open(OUTPUTS_DIR / "experiments.json", "w", encoding="utf-8") as f:
        json.dump(experiments_json, f, indent=2)

    print(f"Successfully generated {len(BENCHMARK_CONFIGS)} experiment artifacts under {OUTPUTS_DIR}")


if __name__ == "__main__":
    main()
