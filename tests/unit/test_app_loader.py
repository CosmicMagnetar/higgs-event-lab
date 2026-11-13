"""Unit tests for the HiggsLens dashboard artifact loader and schema validator.

Tests:
- Directory discovery when outputs/ is missing, empty, or populated
- Schema validation for metrics.json adhering to higgs-experiment-v1
- Detection and isolation of incomplete runs (missing metrics or predictions)
- Detection and error reporting for malformed JSON or CSV files
- Successful parsing of both MLP and Logistic Regression artifacts
"""

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd
import pytest

from app.loader import (
    ExperimentArtifact,
    RunDiscoveryResult,
    RunStatus,
    discover_and_load_runs,
    load_single_run,
    validate_metrics_json,
    validate_test_predictions_df,
)
from higgs_event_lab.protocol import ARTIFACT_SCHEMA_VERSION


@pytest.fixture
def valid_run_envelope() -> dict:
    """Fixture returning a schema-compliant higgs-experiment-v1 envelope."""
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "run_id": "test_mlp_all",
        "model": {"name": "compact_mlp", "family": "neural_network"},
        "features": {
            "group": "all",
            "count": 28,
            "names": [f"feat_{i}" for i in range(28)],
        },
        "data": {"source": "data/processed/higgs_sample_250k.csv.gz", "row_count": 1000},
        "split": {
            "protocol_version": "higgs-split-v1",
            "seed": 42,
            "validation_fraction": 0.15,
            "test_fraction": 0.15,
            "train_rows": 700,
            "validation_rows": 150,
            "test_rows": 150,
        },
        "training": {
            "training_seconds": 12.45,
            "epochs_completed": 10,
            "history": [
                {"epoch": 1, "train_loss": 0.65, "validation_roc_auc": 0.68},
                {"epoch": 2, "train_loss": 0.58, "validation_roc_auc": 0.73},
            ],
        },
        "evaluation": {
            "validation": {"roc_auc": 0.73, "pr_auc": 0.71, "brier_score": 0.20},
            "test": {"roc_auc": 0.725, "pr_auc": 0.705, "brier_score": 0.205},
        },
        "created_at_utc": "2026-10-25T00:00:00+00:00",
    }


@pytest.fixture
def valid_predictions_df() -> pd.DataFrame:
    """Fixture returning a schema-compliant predictions DataFrame."""
    rng = np.random.default_rng(42)
    n = 150
    return pd.DataFrame({
        "row_index": np.arange(n),
        "class_label": rng.integers(0, 2, size=n),
        "score": rng.uniform(0.1, 0.9, size=n),
    })


def test_discover_runs_non_existent_directory(tmp_path: Path):
    """Ensure loader handles a non-existent outputs directory without crashing."""
    missing_dir = tmp_path / "does_not_exist"
    result = discover_and_load_runs(missing_dir)

    assert isinstance(result, RunDiscoveryResult)
    assert not result.dir_exists
    assert len(result.valid_runs) == 0
    assert len(result.incomplete_runs) == 0
    assert len(result.malformed_runs) == 0
    assert not result.has_valid_runs


def test_discover_runs_empty_directory(tmp_path: Path):
    """Ensure loader handles an empty outputs directory cleanly."""
    empty_dir = tmp_path / "empty_outputs"
    empty_dir.mkdir()

    result = discover_and_load_runs(empty_dir)
    assert result.dir_exists
    assert len(result.valid_runs) == 0
    assert result.total_scanned == 0


def test_load_incomplete_run_missing_predictions(tmp_path: Path, valid_run_envelope: dict):
    """Detect run missing test_predictions.csv as incomplete."""
    run_dir = tmp_path / "incomplete_run"
    run_dir.mkdir()
    (run_dir / "metrics.json").write_text(json.dumps(valid_run_envelope), encoding="utf-8")

    artifact, status, reason, missing = load_single_run(run_dir)
    assert artifact is None
    assert status == RunStatus.INCOMPLETE
    assert "test_predictions.csv" in missing

    # Test via discover_and_load_runs
    discovery = discover_and_load_runs(tmp_path)
    assert len(discovery.valid_runs) == 0
    assert len(discovery.incomplete_runs) == 1
    assert discovery.incomplete_runs[0].run_id == "incomplete_run"


def test_load_incomplete_run_missing_metrics(tmp_path: Path, valid_predictions_df: pd.DataFrame):
    """Detect run missing metrics.json as incomplete."""
    run_dir = tmp_path / "missing_metrics_run"
    run_dir.mkdir()
    valid_predictions_df.to_csv(run_dir / "test_predictions.csv", index=False)

    artifact, status, reason, missing = load_single_run(run_dir)
    assert artifact is None
    assert status == RunStatus.INCOMPLETE
    assert "metrics.json" in missing


def test_validate_metrics_json_schema_version_mismatch(valid_run_envelope: dict):
    """Reject unknown schema versions."""
    bad = dict(valid_run_envelope)
    bad["schema_version"] = "unknown-v9"
    err = validate_metrics_json(bad)
    assert err is not None
    assert "Unsupported schema_version" in err


def test_validate_metrics_json_missing_keys(valid_run_envelope: dict):
    """Reject metrics missing required top-level keys."""
    for key in ("run_id", "model", "features", "data", "split", "training", "evaluation"):
        bad = dict(valid_run_envelope)
        del bad[key]
        err = validate_metrics_json(bad)
        assert err is not None
        assert f"Missing required top-level key: '{key}'" in err


def test_validate_metrics_json_invalid_group(valid_run_envelope: dict):
    """Reject invalid feature group names."""
    bad = dict(valid_run_envelope)
    bad["features"] = dict(bad["features"])
    bad["features"]["group"] = "invalid_group_xyz"
    err = validate_metrics_json(bad)
    assert err is not None
    assert "features.group" in err


def test_validate_predictions_df_errors(valid_predictions_df: pd.DataFrame):
    """Verify validation checks on predictions DataFrame."""
    # 1. Missing columns
    bad1 = valid_predictions_df.drop(columns=["score"])
    assert "missing required columns" in validate_test_predictions_df(bad1)

    # 2. Non-binary labels
    bad2 = valid_predictions_df.copy()
    bad2.loc[0, "class_label"] = 5
    assert "must only contain binary values {0, 1}" in validate_test_predictions_df(bad2)

    # 3. Non-finite scores
    bad3 = valid_predictions_df.copy()
    bad3.loc[0, "score"] = np.nan
    assert "non-finite" in validate_test_predictions_df(bad3)

    # 4. Out of bounds scores
    bad4 = valid_predictions_df.copy()
    bad4.loc[0, "score"] = 1.5
    assert "bounded probabilities" in validate_test_predictions_df(bad4)


def test_successful_load_mlp_and_logreg_artifacts(
    tmp_path: Path, valid_run_envelope: dict, valid_predictions_df: pd.DataFrame
):
    """Verify full end-to-end loading and property access for valid runs."""
    base_dir = tmp_path / "outputs"
    mlp_dir = base_dir / "mlp_all"
    mlp_dir.mkdir(parents=True)

    (mlp_dir / "metrics.json").write_text(json.dumps(valid_run_envelope), encoding="utf-8")
    valid_predictions_df.to_csv(mlp_dir / "test_predictions.csv", index=False)

    # Also create a Logistic Regression run
    logreg_dir = base_dir / "logreg_high_level"
    logreg_dir.mkdir(parents=True)
    logreg_envelope = dict(valid_run_envelope)
    logreg_envelope["run_id"] = "logreg_high_level"
    logreg_envelope["model"] = {"name": "logistic_regression", "family": "classical_linear"}
    logreg_envelope["features"] = {"group": "high_level", "count": 7, "names": ["m_jj", "m_jjj", "m_lv", "m_jlv", "m_bb", "m_wbb", "m_wwbb"]}
    logreg_envelope["training"] = {"training_seconds": 0.42, "n_iter": 45, "intercept": -0.15}
    logreg_envelope["evaluation"]["operating_point"] = {
        "target_signal_efficiency": 0.50,
        "validation_threshold": 0.5234,
        "test_actual_signal_efficiency": 0.51,
        "test_background_rejection": 0.74,
    }

    (logreg_dir / "metrics.json").write_text(json.dumps(logreg_envelope), encoding="utf-8")
    valid_predictions_df.to_csv(logreg_dir / "test_predictions.csv", index=False)

    discovery = discover_and_load_runs(base_dir)

    assert discovery.has_valid_runs
    assert len(discovery.valid_runs) == 2
    assert len(discovery.incomplete_runs) == 0
    assert len(discovery.malformed_runs) == 0

    runs_by_id = {r.run_id: r for r in discovery.valid_runs}

    # Verify MLP run
    mlp_run = runs_by_id["test_mlp_all"]
    assert mlp_run.is_mlp
    assert not mlp_run.is_logistic_regression
    assert mlp_run.has_training_history
    assert len(mlp_run.training_history) == 2
    assert mlp_run.test_roc_auc == 0.725
    assert mlp_run.test_pr_auc == 0.705
    assert mlp_run.test_brier_score == 0.205
    assert mlp_run.training_seconds == 12.45
    assert len(mlp_run.test_labels) == 150
    assert len(mlp_run.test_scores) == 150

    # Verify LogReg run
    lr_run = runs_by_id["logreg_high_level"]
    assert lr_run.is_logistic_regression
    assert not lr_run.is_mlp
    assert not lr_run.has_training_history
    assert lr_run.training_seconds == 0.42
    assert lr_run.feature_group == "high_level"
    assert lr_run.feature_count == 7
    assert lr_run.operating_point is not None
    assert lr_run.operating_point["validation_threshold"] == 0.5234


@pytest.mark.parametrize("section,value", [("data", None), ("training", []), ("evaluation", 42), ("features", {"group": []})])
def test_invalid_nested_sections_do_not_crash(tmp_path, valid_run_envelope, valid_predictions_df, section, value):
    valid_run_envelope[section] = value
    folder = tmp_path / "bad"
    folder.mkdir()
    (folder / "metrics.json").write_text(json.dumps(valid_run_envelope))
    valid_predictions_df.to_csv(folder / "test_predictions.csv", index=False)
    result = discover_and_load_runs(tmp_path)
    assert not result.valid_runs
    assert len(result.malformed_runs) == 1


@pytest.mark.parametrize("column,value", [("class_label", np.nan), ("row_index", -1), ("row_index", 0.5)])
def test_invalid_prediction_values(valid_predictions_df, column, value):
    valid_predictions_df[column] = valid_predictions_df[column].astype(float)
    valid_predictions_df.loc[0, column] = value
    assert validate_test_predictions_df(valid_predictions_df)


def test_missing_required_metric_rejected(valid_run_envelope):
    del valid_run_envelope["evaluation"]["test"]["roc_auc"]
    assert "roc_auc" in validate_metrics_json(valid_run_envelope)


def test_mixed_runs_and_prediction_count(tmp_path, valid_run_envelope, valid_predictions_df):
    for name in ("good", "bad"):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "metrics.json").write_text(json.dumps(valid_run_envelope))
        frame = valid_predictions_df if name == "good" else valid_predictions_df.iloc[:3]
        frame.to_csv(folder / "test_predictions.csv", index=False)
    result = discover_and_load_runs(tmp_path)
    assert len(result.valid_runs) == len(result.malformed_runs) == 1
    assert "row count" in result.malformed_runs[0].reason


def test_duplicate_run_ids_are_diagnosed(tmp_path, valid_run_envelope, valid_predictions_df):
    for name in ("first", "second"):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "metrics.json").write_text(json.dumps(valid_run_envelope))
        valid_predictions_df.to_csv(folder / "test_predictions.csv", index=False)
    result = discover_and_load_runs(tmp_path)
    assert len(result.valid_runs) == len(result.malformed_runs) == 1
    assert result.malformed_runs[0].reason == "Duplicate run_id"


@pytest.mark.parametrize("value", [True, float("nan"), -0.1, 1.1])
def test_invalid_metric_values_rejected(valid_run_envelope, value):
    valid_run_envelope["evaluation"]["test"]["roc_auc"] = value
    assert validate_metrics_json(valid_run_envelope)
