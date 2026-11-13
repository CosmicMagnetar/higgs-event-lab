"""Artifact discovery and validation loader for the HiggsLens dashboard.

Loads and validates experiment artifacts complying with the higgs-experiment-v1
schema under outputs/. Handles missing directories, malformed JSON, missing
prediction files, and invalid data types gracefully without crashing or faking data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from higgs_event_lab.protocol import ARTIFACT_SCHEMA_VERSION


class RunStatus(str, Enum):
    """Status classification for discovered experiment folders."""

    VALID = "valid"
    INCOMPLETE = "incomplete"
    MALFORMED = "malformed"


@dataclass(frozen=True)
class IncompleteRunInfo:
    """Diagnostic details for an incomplete experiment run."""

    run_dir: str
    run_id: str
    reason: str
    missing_files: list[str]


@dataclass(frozen=True)
class MalformedRunInfo:
    """Diagnostic details for a malformed or schema-violating experiment run."""

    run_dir: str
    run_id: str
    reason: str
    error_details: str


@dataclass
class ExperimentArtifact:
    """Structured, validated experiment artifact for dashboard visualization."""

    run_id: str
    run_dir: Path
    model_name: str
    model_family: str
    feature_group: str
    feature_count: int
    feature_names: list[str]
    data_source: str
    row_count: int
    split: dict[str, Any]
    training: dict[str, Any]
    evaluation: dict[str, Any]
    test_predictions: pd.DataFrame
    created_at_utc: str | None = None
    raw_metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def is_mlp(self) -> bool:
        """Return True if this run corresponds to a neural network MLP."""
        return "mlp" in self.model_name.lower() or "neural" in self.model_family.lower()

    @property
    def is_logistic_regression(self) -> bool:
        """Return True if this run corresponds to Logistic Regression."""
        return "logreg" in self.model_name.lower() or "logistic" in self.model_name.lower()

    @property
    def has_training_history(self) -> bool:
        """Return True if genuine training epoch history is exported."""
        history = self.training.get("history")
        return bool(isinstance(history, list) and len(history) > 0)

    @property
    def training_history(self) -> list[dict[str, Any]]:
        """Return genuine training history list, or empty list if unavailable."""
        history = self.training.get("history")
        if isinstance(history, list):
            return history
        return []

    @property
    def training_seconds(self) -> float | None:
        """Return training runtime in seconds if recorded."""
        val = self.training.get("training_seconds")
        if isinstance(val, (int, float)):
            return float(val)
        return None

    @property
    def test_metrics(self) -> dict[str, Any]:
        """Return test evaluation metrics dictionary."""
        return self.evaluation.get("test", {})

    @property
    def validation_metrics(self) -> dict[str, Any]:
        """Return validation evaluation metrics dictionary."""
        return self.evaluation.get("validation", {})

    @property
    def test_roc_auc(self) -> float | None:
        """Return held-out test ROC-AUC score."""
        val = self.test_metrics.get("roc_auc")
        return float(val) if isinstance(val, (int, float)) else None

    @property
    def test_pr_auc(self) -> float | None:
        """Return held-out test PR-AUC (average precision) score."""
        val = self.test_metrics.get("pr_auc")
        return float(val) if isinstance(val, (int, float)) else None

    @property
    def test_brier_score(self) -> float | None:
        """Return held-out test Brier score loss."""
        val = self.test_metrics.get("brier_score")
        return float(val) if isinstance(val, (int, float)) else None

    @property
    def test_labels(self) -> np.ndarray:
        """Return held-out true class labels as integer array."""
        return self.test_predictions["class_label"].to_numpy(dtype=np.int64)

    @property
    def test_scores(self) -> np.ndarray:
        """Return held-out predicted classifier probability scores."""
        return self.test_predictions["score"].to_numpy(dtype=np.float64)

    @property
    def operating_point(self) -> dict[str, Any] | None:
        """Return validation-selected operating point metadata if exported."""
        op = self.evaluation.get("operating_point")
        return op if isinstance(op, dict) else None


@dataclass
class RunDiscoveryResult:
    """Summary of discovered, valid, incomplete, and malformed runs."""

    valid_runs: list[ExperimentArtifact] = field(default_factory=list)
    incomplete_runs: list[IncompleteRunInfo] = field(default_factory=list)
    malformed_runs: list[MalformedRunInfo] = field(default_factory=list)
    base_dir: str = "outputs"
    dir_exists: bool = True

    @property
    def total_scanned(self) -> int:
        """Return total number of folders scanned."""
        return len(self.valid_runs) + len(self.incomplete_runs) + len(self.malformed_runs)

    @property
    def has_valid_runs(self) -> bool:
        """Return True if at least one valid run is available."""
        return len(self.valid_runs) > 0


REQUIRED_TOP_LEVEL_KEYS: tuple[str, ...] = (
    "schema_version",
    "run_id",
    "model",
    "features",
    "data",
    "split",
    "training",
    "evaluation",
)

REQUIRED_PREDICTION_COLUMNS: tuple[str, ...] = (
    "row_index",
    "class_label",
    "score",
)


def validate_metrics_json(data: dict[str, Any]) -> str | None:
    """Validate metrics dictionary against higgs-experiment-v1 contract.

    Returns None if valid, or an error description string if invalid.
    """
    if not isinstance(data, dict):
        return "metrics.json must contain a JSON object"
    for key in REQUIRED_TOP_LEVEL_KEYS:
        if key not in data:
            return f"Missing required top-level key: '{key}'"
    if data["schema_version"] != ARTIFACT_SCHEMA_VERSION:
        return f"Unsupported schema_version '{data['schema_version']}'. Expected '{ARTIFACT_SCHEMA_VERSION}'"
    if not isinstance(data["run_id"], str) or not data["run_id"].strip():
        return "run_id must be a non-empty string"
    for key in ("model", "features", "data", "split", "training", "evaluation"):
        if not isinstance(data[key], dict):
            return f"{key} section must be a dictionary"
    model, features, source, split = (data[k] for k in ("model", "features", "data", "split"))
    if not isinstance(model.get("name"), str) or not model["name"].strip():
        return "model.name must be a non-empty string"
    if features.get("group") not in ("low_level", "high_level", "all"):
        return "features.group must be 'low_level', 'high_level', or 'all'"
    count, names = features.get("count"), features.get("names")
    if type(count) is not int or count <= 0:
        return "features.count must be a positive integer"
    if not isinstance(names, list) or len(names) != count or any(not isinstance(n, str) or not n.strip() for n in names):
        return "features.names must contain one non-empty name per feature"
    if len(set(names)) != count:
        return "features.names must be unique"
    if not isinstance(source.get("source"), str) or not source["source"].strip():
        return "data.source must be a non-empty string"
    if type(source.get("row_count")) is not int or source["row_count"] <= 0:
        return "data.row_count must be a positive integer"
    if type(split.get("seed")) is not int:
        return "split.seed must be an integer"
    if split.get("protocol_version") != "higgs-split-v1":
        return "split.protocol_version must be higgs-split-v1"
    counts = [split.get(k) for k in ("train_rows", "validation_rows", "test_rows")]
    if any(type(n) is not int or n <= 0 for n in counts) or sum(counts) != source["row_count"]:
        return "split row counts must be positive integers summing to data.row_count"
    fractions = [split.get(k) for k in ("validation_fraction", "test_fraction")]
    if any(type(n) not in (int, float) or not np.isfinite(n) or not 0 < n < 1 for n in fractions) or sum(fractions) >= 1:
        return "split fractions must be positive and sum to less than 1"
    for partition in ("validation", "test"):
        metrics = data["evaluation"].get(partition)
        if not isinstance(metrics, dict):
            return f"evaluation.{partition} must be a metrics dictionary"
        for name in ("roc_auc", "pr_auc", "brier_score"):
            value = metrics.get(name)
            if type(value) not in (int, float) or not np.isfinite(value) or not 0 <= value <= 1:
                return f"evaluation.{partition}.{name} must be a finite numeric value within [0, 1]"
        if "accuracy" in metrics:
            value = metrics["accuracy"]
            if type(value) not in (int, float) or not np.isfinite(value) or not 0 <= value <= 1:
                return f"evaluation.{partition}.accuracy must be within [0, 1]"
    training = data["training"]
    if "training_seconds" in training:
        value = training["training_seconds"]
        if type(value) not in (int, float) or not np.isfinite(value) or value < 0:
            return "training.training_seconds must be finite and non-negative"
    if "history" in training:
        history = training["history"]
        if not isinstance(history, list):
            return "training.history must be a list"
        for item in history:
            if not isinstance(item, dict):
                return "training.history entries must be dictionaries"
            for key in ("epoch", "train_loss", "validation_roc_auc"):
                value = item.get(key)
                if type(value) not in (int, float) or not np.isfinite(value) or value < 0:
                    return f"training.history.{key} must be a finite non-negative number"
            if item["validation_roc_auc"] > 1:
                return "training.history.validation_roc_auc must be within [0, 1]"
    op = data["evaluation"].get("operating_point")
    if op is not None:
        if not isinstance(op, dict):
            return "evaluation.operating_point must be a dictionary"
        for key in ("target_signal_efficiency", "validation_threshold", "test_background_rejection"):
            value = op.get(key)
            if type(value) not in (int, float) or not np.isfinite(value) or not 0 <= value <= 1:
                return f"evaluation.operating_point.{key} must be within [0, 1]"
    return None


def validate_test_predictions_df(df: pd.DataFrame) -> str | None:
    """Validate predictions DataFrame against schema contract.

    Returns None if valid, or an error description string if invalid.
    """
    if len(df) == 0:
        return "test_predictions.csv contains 0 rows (empty dataset)"

    missing_cols = [c for c in REQUIRED_PREDICTION_COLUMNS if c not in df.columns]
    if missing_cols:
        return f"test_predictions.csv missing required columns: {missing_cols}"

    # Verify label values
    labels = df["class_label"]
    if not pd.api.types.is_numeric_dtype(labels):
        return "class_label column must be numeric"

    if labels.isna().any():
        return "class_label contains missing values"
    unique_labels = set(labels.unique())
    if not unique_labels.issubset({0, 1, 0.0, 1.0}):
        return f"class_label must only contain binary values {{0, 1}}, found {unique_labels}"

    if unique_labels != {0, 1}:
        return "class_label must contain both binary classes for evaluation"
    indices = df["row_index"]
    if not pd.api.types.is_numeric_dtype(indices) or indices.isna().any():
        return "row_index must contain finite non-negative integers"
    values = indices.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any() or (values != np.floor(values)).any() or indices.duplicated().any():
        return "row_index must contain unique finite non-negative integers"

    # Verify score values
    scores = df["score"]
    if not pd.api.types.is_numeric_dtype(scores):
        return "score column must be numeric"

    score_arr = scores.to_numpy(dtype=np.float64)
    if not np.isfinite(score_arr).all():
        return "score column contains non-finite (NaN or Inf) values"

    if (score_arr < 0.0).any() or (score_arr > 1.0).any():
        return "score values must be bounded probabilities within [0.0, 1.0]"

    return None


def load_single_run(run_dir: Path) -> tuple[ExperimentArtifact | None, RunStatus, str | None, list[str]]:
    """Attempt to load and validate a single run directory.

    Returns: (artifact, status, reason, missing_files)
    """
    metrics_path = run_dir / "metrics.json"
    preds_path = run_dir / "test_predictions.csv"

    missing = []
    if not metrics_path.is_file():
        missing.append("metrics.json")
    if not preds_path.is_file():
        missing.append("test_predictions.csv")

    if missing:
        return None, RunStatus.INCOMPLETE, f"Missing required file(s): {', '.join(missing)}", missing

    # Parse JSON
    try:
        metrics_text = metrics_path.read_text(encoding="utf-8")
        raw_metrics = json.loads(metrics_text)
    except Exception as exc:
        return None, RunStatus.MALFORMED, f"Unparseable metrics.json: {exc}", []

    if not isinstance(raw_metrics, dict):
        return None, RunStatus.MALFORMED, "metrics.json must contain a top-level JSON object", []

    schema_err = validate_metrics_json(raw_metrics)
    if schema_err:
        return None, RunStatus.MALFORMED, f"Schema validation error: {schema_err}", []

    # Parse CSV
    try:
        df_preds = pd.read_csv(preds_path)
    except Exception as exc:
        return None, RunStatus.MALFORMED, f"Unparseable test_predictions.csv: {exc}", []

    csv_err = validate_test_predictions_df(df_preds)
    if csv_err:
        return None, RunStatus.MALFORMED, f"Prediction file validation error: {csv_err}", []

    if len(df_preds) != raw_metrics["split"]["test_rows"]:
        return None, RunStatus.MALFORMED, "Prediction row count differs from split.test_rows", []
    if (df_preds["row_index"] >= raw_metrics["data"]["row_count"]).any():
        return None, RunStatus.MALFORMED, "Prediction row_index exceeds source sample bounds", []

    artifact = ExperimentArtifact(
        run_id=str(raw_metrics["run_id"]),
        run_dir=run_dir,
        model_name=str(raw_metrics["model"]["name"]),
        model_family=str(raw_metrics["model"].get("family", "unknown")),
        feature_group=str(raw_metrics["features"]["group"]),
        feature_count=int(raw_metrics["features"]["count"]),
        feature_names=list(raw_metrics["features"]["names"]),
        data_source=str(raw_metrics["data"].get("source", "unknown")),
        row_count=int(raw_metrics["data"].get("row_count", 0)),
        split=dict(raw_metrics["split"]),
        training=dict(raw_metrics.get("training", {})),
        evaluation=dict(raw_metrics.get("evaluation", {})),
        test_predictions=df_preds,
        created_at_utc=raw_metrics.get("created_at_utc"),
        raw_metrics=raw_metrics,
    )

    return artifact, RunStatus.VALID, None, []


def discover_and_load_runs(base_dir: str | Path = "outputs") -> RunDiscoveryResult:
    """Discover and parse experiment runs beneath base_dir.

    Never crashes if base_dir does not exist or contains invalid files.
    """
    path = Path(base_dir)
    result = RunDiscoveryResult(base_dir=str(path))

    if not path.exists():
        result.dir_exists = False
        return result

    if not path.is_dir():
        result.dir_exists = False
        result.malformed_runs.append(
            MalformedRunInfo(
                run_dir=str(path),
                run_id="root",
                reason=f"Path '{path}' is not a directory",
                error_details="Path is a regular file or link",
            )
        )
        return result

    result.dir_exists = True

    # Scan for candidate experiment directories (immediate subdirectories)
    # Also check if base_dir itself contains metrics.json and test_predictions.csv
    candidate_dirs: list[Path] = []

    if (path / "metrics.json").is_file() or (path / "test_predictions.csv").is_file():
        candidate_dirs.append(path)

    try:
        entries = sorted(path.iterdir())
    except OSError as exc:
        result.malformed_runs.append(MalformedRunInfo(str(path), "root", f"Cannot read artifact directory: {exc}", str(exc)))
        return result
    for item in entries:
        if item.is_dir() and not item.name.startswith((".", "__")):
            if item.name in {"curves", "runs", "model_weights", "eda"} and not (item / "metrics.json").is_file():
                continue
            candidate_dirs.append(item)

    seen_ids: set[str] = set()
    for run_dir in candidate_dirs:
        artifact, status, reason, missing = load_single_run(run_dir)
        if artifact is not None and artifact.run_id in seen_ids:
            artifact, status, reason = None, RunStatus.MALFORMED, "Duplicate run_id"
        if artifact is not None:
            seen_ids.add(artifact.run_id)
        folder_name = run_dir.name

        if status == RunStatus.VALID and artifact is not None:
            result.valid_runs.append(artifact)
        elif status == RunStatus.INCOMPLETE:
            result.incomplete_runs.append(
                IncompleteRunInfo(
                    run_dir=str(run_dir),
                    run_id=folder_name,
                    reason=reason or "Incomplete run",
                    missing_files=missing,
                )
            )
        elif status == RunStatus.MALFORMED:
            result.malformed_runs.append(
                MalformedRunInfo(
                    run_dir=str(run_dir),
                    run_id=folder_name,
                    reason=reason or "Malformed run",
                    error_details=reason or "Schema or data constraint violated",
                )
            )

    return result
