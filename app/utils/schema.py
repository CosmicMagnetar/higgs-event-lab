"""Schema definitions and validation for experiment output artifacts.

Implements the contract for MID-05 / MID-06:
- Validates model metrics (ROC-AUC, PR-AUC, Brier score, accuracy, runtime).
- Validates curve coordinates (ROC, PR, calibration).
- Provides validation error structures with actionable messages without crashing the app.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


VALID_FEATURE_GROUPS = ("low_level", "high_level", "all")
VALID_SPLIT_KEYS = ("train", "val", "test")


@dataclass(frozen=True)
class OperatingPoint:
    """Operating point metrics selected on validation and evaluated on test."""

    target_signal_efficiency: float
    selected_threshold: float
    test_background_rejection: float
    test_signal_efficiency: float


@dataclass(frozen=True)
class ExperimentMetrics:
    """Core evaluation metrics for a model run."""

    roc_auc: float
    pr_auc: float
    brier_score: float | None = None
    accuracy: float | None = None
    training_time_sec: float | None = None
    inference_latency_ms: float | None = None


@dataclass(frozen=True)
class ExperimentRecord:
    """Validated representation of an individual experiment run."""

    run_id: str
    model_name: str
    feature_group: str
    seed: int
    sample_size: int
    metrics: ExperimentMetrics
    operating_point: OperatingPoint | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    raw_payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CurveData:
    """Validated coordinate arrays for ROC, PR, and Calibration curves."""

    run_id: str
    fpr: list[float]
    tpr: list[float]
    precision: list[float]
    recall: list[float]
    prob_pred: list[float] = field(default_factory=list)
    prob_true: list[float] = field(default_factory=list)


class ValidationError(ValueError):
    """Raised when an artifact payload does not conform to the expected schema."""


def validate_experiment_payload(data: dict[str, Any], source_file: str = "") -> ExperimentRecord:
    """Validate a raw experiment dictionary against the schema.

    Parameters
    ----------
    data : dict[str, Any]
        Raw dictionary loaded from JSON.
    source_file : str, optional
        File path for error diagnostics.

    Returns
    -------
    ExperimentRecord
        Validated immutable experiment record.

    Raises
    ------
    ValidationError
        If required keys are missing or values are out of bounds.
    """
    from app.loader import validate_metrics_json

    error = validate_metrics_json(data)
    if error:
        raise ValidationError(f"[{source_file}] {error}")
    test = data["evaluation"]["test"]
    op = data["evaluation"].get("operating_point")
    operating_point = None
    if op and "test_actual_signal_efficiency" in op:
        operating_point = OperatingPoint(
            op["target_signal_efficiency"], op["validation_threshold"],
            op["test_background_rejection"], op["test_actual_signal_efficiency"],
        )
    return ExperimentRecord(
        run_id=data["run_id"], model_name=data["model"]["name"],
        feature_group=data["features"]["group"], seed=data["split"]["seed"],
        sample_size=data["data"]["row_count"],
        metrics=ExperimentMetrics(
            roc_auc=test["roc_auc"], pr_auc=test["pr_auc"], brier_score=test["brier_score"],
            accuracy=test.get("accuracy"), training_time_sec=data["training"].get("training_seconds"),
        ),
        operating_point=operating_point,
        provenance={
            "source_file": data["data"]["source"],
            "split": data["split"],
            "timestamp_utc": data.get("created_at_utc"),
        },
        raw_payload=data,
    )


def validate_curve_payload(data: dict[str, Any], source_file: str = "") -> CurveData:
    """Validate curve coordinates for plotting ROC, PR, and Calibration.

    Parameters
    ----------
    data : dict[str, Any]
        Raw curve dictionary loaded from JSON.
    source_file : str, optional
        File path for error diagnostics.

    Returns
    -------
    CurveData
        Validated curve coordinate object.

    Raises
    ------
    ValidationError
        If array shapes or value ranges are invalid.
    """
    prefix = f"[{source_file}] " if source_file else ""

    if "run_id" not in data or not str(data["run_id"]).strip():
        raise ValidationError(f"{prefix}Curve payload missing 'run_id'")
    run_id = str(data["run_id"]).strip()

    # ROC validation
    roc = data.get("roc")
    if not isinstance(roc, dict) or "fpr" not in roc or "tpr" not in roc:
        raise ValidationError(f"{prefix}Curve payload must contain 'roc' dict with 'fpr' and 'tpr'")
    fpr = roc["fpr"]
    tpr = roc["tpr"]
    if not isinstance(fpr, list) or not isinstance(tpr, list) or len(fpr) != len(tpr):
        raise ValidationError(f"{prefix}'fpr' and 'tpr' must be equal-length lists")
    if len(fpr) < 2:
        raise ValidationError(f"{prefix}ROC curve requires at least 2 points")

    # PR validation
    pr = data.get("pr")
    if not isinstance(pr, dict) or "precision" not in pr or "recall" not in pr:
        raise ValidationError(f"{prefix}Curve payload must contain 'pr' dict with 'precision' and 'recall'")
    precision = pr["precision"]
    recall = pr["recall"]
    if not isinstance(precision, list) or not isinstance(recall, list) or len(precision) != len(recall):
        raise ValidationError(f"{prefix}'precision' and 'recall' must be equal-length lists")
    if len(precision) < 2:
        raise ValidationError(f"{prefix}PR curve requires at least 2 points")

    # Optional calibration validation
    calib = data.get("calibration", {})
    prob_pred: list[float] = []
    prob_true: list[float] = []
    if isinstance(calib, dict) and "prob_pred" in calib and "prob_true" in calib:
        prob_pred = calib["prob_pred"]
        prob_true = calib["prob_true"]
        if not isinstance(prob_pred, list) or not isinstance(prob_true, list) or len(prob_pred) != len(prob_true):
            raise ValidationError(f"{prefix}'calibration.prob_pred' and 'prob_true' must be equal-length lists")

    # Value range checks
    for name, arr in [("fpr", fpr), ("tpr", tpr), ("precision", precision), ("recall", recall)]:
        for val in arr:
            if not isinstance(val, (int, float)) or not (0.0 <= val <= 1.0):
                raise ValidationError(f"{prefix}Curve '{name}' values must be in [0.0, 1.0], got: {val}")

    for name, arr in [("prob_pred", prob_pred), ("prob_true", prob_true)]:
        for val in arr:
            if not isinstance(val, (int, float)) or not (0.0 <= val <= 1.0):
                raise ValidationError(f"{prefix}Calibration '{name}' values must be in [0.0, 1.0], got: {val}")

    return CurveData(
        run_id=run_id,
        fpr=[float(x) for x in fpr],
        tpr=[float(x) for x in tpr],
        precision=[float(x) for x in precision],
        recall=[float(x) for x in recall],
        prob_pred=[float(x) for x in prob_pred],
        prob_true=[float(x) for x in prob_true],
    )
