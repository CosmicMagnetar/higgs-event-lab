"""Unit tests for the dashboard schema validation, artifact loader, and pipeline checks."""

import tempfile
import json
from pathlib import Path
import pytest

from app.utils.loader import (
    get_project_root,
    inspect_pipeline_status,
    load_experiments,
    load_feature_dictionary,
)
from app.utils.schema import (
    ValidationError,
    validate_curve_payload,
    validate_experiment_payload,
)


def test_load_feature_dictionary_loads_verified_schema():
    """Verify that the feature dictionary loads all 28 features plus the target label."""
    df = load_feature_dictionary()
    assert len(df) == 29
    assert set(df.columns) >= {"file_column_index", "name", "group", "kind", "description"}
    assert (df["group"] == "low_level").sum() == 21
    assert (df["group"] == "high_level").sum() == 7
    assert (df["group"] == "target").sum() == 1
    assert df.loc[df["file_column_index"] == 0, "name"].iloc[0] == "class_label"


def test_inspect_pipeline_status_smoke():
    """Check that pipeline inspection runs without errors and reports existing directories."""
    status = inspect_pipeline_status()
    assert hasattr(status, "has_raw_data")
    assert hasattr(status, "has_processed_data")
    assert hasattr(status, "has_outputs")
    assert hasattr(status, "has_artifacts")


def test_load_experiments_empty_state_does_not_crash(tmp_path):
    """When outputs/ has no run files, load_experiments must return an empty container safely."""
    loaded = load_experiments(root=tmp_path)
    assert loaded.is_empty
    assert len(loaded.experiments) == 0
    assert len(loaded.curves) == 0


def test_legacy_flat_records_are_rejected():
    with pytest.raises(ValidationError, match="schema_version"):
        validate_experiment_payload({"run_id": "legacy", "metrics": {"roc_auc": 0.9}})


def test_validate_curve_payload():
    """Valid curve coordinates parse and validate correctly."""
    valid_curve = {
        "run_id": "test_run",
        "roc": {
            "fpr": [0.0, 0.5, 1.0],
            "tpr": [0.0, 0.7, 1.0],
        },
        "pr": {
            "precision": [0.6, 0.8, 1.0],
            "recall": [1.0, 0.5, 0.0],
        },
    }
    curve = validate_curve_payload(valid_curve)
    assert curve.run_id == "test_run"
    assert len(curve.fpr) == 3
    assert len(curve.precision) == 3


def test_validate_curve_payload_mismatched_lengths():
    """Mismatched coordinate array lengths must raise ValidationError."""
    bad_curve = {
        "run_id": "bad_curve",
        "roc": {
            "fpr": [0.0, 1.0],
            "tpr": [0.0, 0.5, 1.0],  # Mismatched length
        },
        "pr": {
            "precision": [1.0, 0.5],
            "recall": [0.5, 1.0],
        },
    }
    with pytest.raises(ValidationError, match="equal-length lists"):
        validate_curve_payload(bad_curve)


def test_validate_curve_payload_rejects_out_of_bounds_coordinates():
    """Coordinate values out of [0.0, 1.0] must raise ValidationError."""
    bad_coords = {
        "run_id": "bad_coords",
        "roc": {
            "fpr": [0.0, 1.2],  # Out of bounds
            "tpr": [0.0, 1.0],
        },
        "pr": {
            "precision": [1.0, 0.5],
            "recall": [0.0, 1.0],
        },
    }
    with pytest.raises(ValidationError, match="must be in \\[0.0, 1.0\\]"):
        validate_curve_payload(bad_coords)


def test_flat_csv_is_not_experiment_evidence(tmp_path):
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    (outputs / "experiments_summary.csv").write_text(
        "run_id,model_name,feature_group,seed,sample_size,roc_auc,pr_auc\n"
        "legacy,LR,all,42,1000,0.8,0.7\n"
    )
    assert load_experiments(tmp_path).is_empty
