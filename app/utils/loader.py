"""Safe artifact loader and environment diagnostics for Higgs Event Lab.

Reads saved output files, feature definitions, and manifest files without
triggering training or falling back to hardcoded scores.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from app.utils.schema import (
    CurveData,
    ExperimentRecord,
    validate_experiment_payload,
)


def get_project_root() -> Path:
    """Return the absolute path to the repository root directory."""
    # This file lives in app/utils/loader.py -> parents[2] is the repo root
    return Path(__file__).resolve().parents[2]


@dataclass
class PipelineStatus:
    """System and folder readiness status for display on the dashboard."""

    has_raw_data: bool = False
    raw_files: list[str] = field(default_factory=list)
    has_processed_data: bool = False
    processed_files: list[str] = field(default_factory=list)
    has_manifests: bool = False
    manifest_paths: list[str] = field(default_factory=list)
    has_artifacts: bool = False
    artifact_files: list[str] = field(default_factory=list)
    has_outputs: bool = False
    output_files: list[str] = field(default_factory=list)


@dataclass
class LoadedExperiments:
    """Collection of validated experiments and curves, with any schema warnings."""

    experiments: list[ExperimentRecord] = field(default_factory=list)
    curves: dict[str, CurveData] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return len(self.experiments) == 0


def load_feature_dictionary(root: Path | None = None) -> pd.DataFrame:
    """Load the verified feature dictionary from docs/feature_dictionary.csv.

    Parameters
    ----------
    root : Path, optional
        Project root directory. If None, detected automatically.

    Returns
    -------
    pd.DataFrame
        Table of features with column indices, groups, kinds, and physical descriptions.

    Raises
    ------
    FileNotFoundError
        If docs/feature_dictionary.csv does not exist.
    """
    root = root or get_project_root()
    path = root / "docs" / "feature_dictionary.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Feature dictionary not found at {path}")

    return pd.read_csv(path)


def inspect_pipeline_status(root: Path | None = None) -> PipelineStatus:
    """Check existence of dataset files, model artifacts, and output metrics."""
    root = root or get_project_root()
    status = PipelineStatus()

    # Raw data check
    raw_dir = root / "data" / "raw"
    if raw_dir.is_dir():
        raw_items = [p.name for p in raw_dir.iterdir() if p.name != "README.md" and not p.name.startswith(".")]
        status.has_raw_data = len(raw_items) > 0
        status.raw_files = raw_items

    # Processed data check
    proc_dir = root / "data" / "processed"
    if proc_dir.is_dir():
        proc_items = [p.name for p in proc_dir.iterdir() if p.name != "README.md" and not p.name.startswith(".")]
        status.has_processed_data = len(proc_items) > 0
        status.processed_files = proc_items

        manifests = [str(p.relative_to(root)) for p in proc_dir.glob("*.manifest.json")]
        status.has_manifests = len(manifests) > 0
        status.manifest_paths = manifests

    # Artifacts check
    art_dir = root / "artifacts"
    if art_dir.is_dir():
        art_items = [p.name for p in art_dir.iterdir() if p.name != "README.md" and not p.name.startswith(".")]
        status.has_artifacts = len(art_items) > 0
        status.artifact_files = art_items

    # Outputs check
    out_dir = root / "outputs"
    if out_dir.is_dir():
        out_items = [p.name for p in out_dir.iterdir() if p.name != "README.md" and not p.name.startswith(".")]
        status.has_outputs = len(out_items) > 0
        status.output_files = out_items

    return status


def load_experiments(root: Path | None = None) -> LoadedExperiments:
    """Adapt canonical run packages for legacy components; never read flat records."""
    from app.loader import discover_and_load_runs
    from app.metrics import compute_calibration_points, compute_pr_points, compute_roc_points

    root = root or get_project_root()
    discovery = discover_and_load_runs(root / "outputs")
    result = LoadedExperiments()
    if not discovery.dir_exists:
        result.warnings.append(f"Outputs directory unavailable: {discovery.base_dir}")
    result.warnings.extend(f"{run.run_id}: {run.reason}" for run in discovery.incomplete_runs)
    result.errors.extend(f"{run.run_id}: {run.reason}" for run in discovery.malformed_runs)
    for run in discovery.valid_runs:
        result.experiments.append(validate_experiment_payload(run.raw_metrics, str(run.run_dir)))
        fpr, tpr, _ = compute_roc_points(run.test_labels, run.test_scores)
        precision, recall, _ = compute_pr_points(run.test_labels, run.test_scores)
        prob_true, prob_pred, _ = compute_calibration_points(run.test_labels, run.test_scores)
        result.curves[run.run_id] = CurveData(
            run.run_id, fpr.tolist(), tpr.tolist(), precision.tolist(), recall.tolist(),
            prob_pred.tolist(), prob_true.tolist(),
        )
    return result


def load_saved_eda(base_dir: str | Path = "outputs") -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame] | None:
    """Read saved EDA with the fields required by the dashboard validated."""
    import math

    folder = Path(base_dir) / "eda"
    summary_path, feature_path = folder / "summary.json", folder / "feature_summary.csv"
    if not summary_path.is_file() and not feature_path.is_file():
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        features = pd.read_csv(feature_path)
        correlations = pd.read_csv(folder / "top_correlations.csv") if (folder / "top_correlations.csv").is_file() else pd.DataFrame()
        if not isinstance(summary, dict):
            raise ValueError("summary must be a JSON object")
        rows = summary.get("rows")
        counts = summary.get("label_counts")
        if type(rows) is not int or rows <= 0 or not isinstance(counts, dict):
            raise ValueError("rows and label_counts are required")
        if any(type(counts.get(k)) is not int or counts[k] <= 0 for k in ("0", "1")) or counts["0"] + counts["1"] != rows:
            raise ValueError("label_counts must contain positive class counts summing to rows")
        fraction = summary.get("signal_fraction")
        if type(fraction) not in (int, float) or not math.isfinite(fraction) or not 0 <= fraction <= 1:
            raise ValueError("signal_fraction must be a finite fraction")
        columns = {"feature", "signal_mean", "background_mean", "unique_count"}
        if not columns.issubset(features.columns):
            raise ValueError("feature summary is missing required columns")
        for name in columns - {"feature"}:
            values = pd.to_numeric(features[name], errors="raise")
            if not values.map(math.isfinite).all():
                raise ValueError(f"{name} contains non-finite values")
        return summary, features, correlations
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(f"Invalid saved EDA under {folder}: {exc}") from exc
