"""Logistic Regression baselines for the HIGGS benchmark across feature groups.

This module implements Task A (MID-03):
- Trains regularized Logistic Regression models using scikit-learn.
- Enforces leakage-free scaling by embedding StandardScaler inside an sklearn Pipeline.
- Supports the low_level (21), high_level (7), and all (28) feature groups.
- Selects operating thresholds on validation data only, preserving test holdout integrity.
- Exports schema-compliant experiment artifacts (metrics.json and test_predictions.csv)
  matching the higgs-experiment-v1 protocol for downstream dashboard consumption.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from higgs_event_lab.data import load_higgs_sample, to_relative_manifest_path
from higgs_event_lab.evaluation import (
    compute_classification_metrics,
    select_operating_threshold,
)
from higgs_event_lab.features import (
    FEATURE_GROUPS,
    FeatureGroup,
    get_feature_columns,
)
from higgs_event_lab.protocol import (
    SplitConfig,
    SplitIndices,
    create_stratified_splits,
    new_experiment_artifact,
)


@dataclass(frozen=True)
class LogisticRegressionConfig:
    """Hyperparameters and execution configuration for Logistic Regression."""

    C: float = 1.0
    max_iter: int = 1000
    solver: str = "lbfgs"
    seed: int = 42
    tol: float = 1e-4
    target_signal_efficiency: float = 0.50

    def __post_init__(self) -> None:
        if self.C <= 0:
            raise ValueError("C must be strictly positive")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be strictly positive")
        if not 0 < self.target_signal_efficiency < 1:
            raise ValueError("target_signal_efficiency must be between 0 and 1")
        if self.tol <= 0:
            raise ValueError("tol must be strictly positive")


def _xgboost() -> Any:
    """Lazy import of xgboost so environments without xgboost do not fail on import."""
    try:
        import xgboost
    except ModuleNotFoundError as exc:
        raise RuntimeError("xgboost is required for XGBoost baseline. Install xgboost first.") from exc
    return xgboost


@dataclass(frozen=True)
class XGBoostConfig:
    """Hyperparameters and execution configuration for XGBoost baseline."""

    n_estimators: int = 150
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    tree_method: str = "hist"
    seed: int = 42
    target_signal_efficiency: float = 0.50
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be strictly positive")
        if self.max_depth <= 0:
            raise ValueError("max_depth must be strictly positive")
        if not 0 < self.learning_rate <= 1.0:
            raise ValueError("learning_rate must be in (0, 1]")
        if not 0 < self.subsample <= 1.0:
            raise ValueError("subsample must be in (0, 1]")
        if not 0 < self.colsample_bytree <= 1.0:
            raise ValueError("colsample_bytree must be in (0, 1]")
        if not 0 < self.target_signal_efficiency < 1.0:
            raise ValueError("target_signal_efficiency must be between 0 and 1")


def _as_feature_matrix(X: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Validate and convert input feature matrix to finite float array."""
    array = (
        X.to_numpy(dtype=np.float64, copy=True)
        if isinstance(X, pd.DataFrame)
        else np.asarray(X, dtype=np.float64)
    )
    if array.ndim != 2 or array.shape[0] < 10 or array.shape[1] == 0:
        raise ValueError(
            "X must be a two-dimensional matrix with at least 10 rows and one feature"
        )
    if not np.isfinite(array).all():
        raise ValueError("X must contain only finite, non-null values")
    return array


def build_pipeline(config: LogisticRegressionConfig) -> Pipeline:
    """Construct an sklearn Pipeline guaranteeing leakage-free scaling and training."""
    scaler = StandardScaler()
    classifier = LogisticRegression(
        C=config.C,
        max_iter=config.max_iter,
        solver=config.solver,
        random_state=config.seed,
        tol=config.tol,
    )
    return Pipeline([("scaler", scaler), ("classifier", classifier)])


def train_logistic_regression(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray | list[int],
    *,
    split: SplitIndices | None = None,
    config: LogisticRegressionConfig = LogisticRegressionConfig(),
) -> dict[str, Any]:
    """Train Logistic Regression on the training split, select threshold on validation, and evaluate on test.

    The StandardScaler and LogisticRegression are bundled into a Pipeline so the scaler is
    fitted strictly on the training partition only. Operating threshold selection occurs
    exclusively on validation predictions at the specified target signal efficiency.

    Parameters
    ----------
    X : 2D feature matrix (DataFrame or ndarray)
    y : 1D target label vector (Series, ndarray, or list of {0, 1})
    split : Optional SplitIndices; if None, created from y using config.seed
    config : Configuration dataclass

    Returns
    -------
    dict[str, Any]
        Dictionary containing pipeline, split, timings, operating threshold, and separate
        validation and test results.
    """
    x_array = _as_feature_matrix(X)
    y_array = np.asarray(y, dtype=np.int64)

    if y_array.ndim != 1:
        raise ValueError("y must be a one-dimensional array")
    if y_array.shape[0] != x_array.shape[0]:
        raise ValueError("y and X must have matching row counts")
    if set(np.unique(y_array)) != {0, 1}:
        raise ValueError("y must contain both binary classes {0, 1}")

    split = split or create_stratified_splits(y_array, SplitConfig(seed=config.seed))

    x_train = x_array[split.train]
    y_train = y_array[split.train]
    x_val = x_array[split.validation]
    y_val = y_array[split.validation]
    x_test = x_array[split.test]
    y_test = y_array[split.test]

    pipeline = build_pipeline(config)

    start_time = time.perf_counter()
    pipeline.fit(x_train, y_train)
    training_seconds = time.perf_counter() - start_time

    val_probs = pipeline.predict_proba(x_val)[:, 1]
    test_probs = pipeline.predict_proba(x_test)[:, 1]

    val_metrics = compute_classification_metrics(y_val, val_probs)
    test_metrics = compute_classification_metrics(y_test, test_probs)

    # Threshold chosen on validation partition only
    threshold = select_operating_threshold(
        y_val, val_probs, target_signal_efficiency=config.target_signal_efficiency
    )

    val_signal_mask = y_val == 1
    val_bkg_mask = y_val == 0
    test_signal_mask = y_test == 1
    test_bkg_mask = y_test == 0

    operating_point = {
        "target_signal_efficiency": config.target_signal_efficiency,
        "validation_threshold": threshold,
        "validation_actual_signal_efficiency": float(
            np.mean(val_probs[val_signal_mask] >= threshold)
        ),
        "validation_background_rejection": float(
            np.mean(val_probs[val_bkg_mask] < threshold)
        ),
        "test_actual_signal_efficiency": float(
            np.mean(test_probs[test_signal_mask] >= threshold)
        ),
        "test_background_rejection": float(
            np.mean(test_probs[test_bkg_mask] < threshold)
        ),
    }

    classifier = pipeline.named_steps["classifier"]
    coefficients = {
        "intercept": float(classifier.intercept_[0]),
        "coef": [float(c) for c in classifier.coef_[0]],
        "n_iter": int(classifier.n_iter_[0]),
    }

    return {
        "pipeline": pipeline,
        "model": classifier,
        "scaler": pipeline.named_steps["scaler"],
        "split": split,
        "training_seconds": training_seconds,
        "operating_threshold": threshold,
        "operating_point": operating_point,
        "coefficients": coefficients,
        "validation": {
            "row_indices": split.validation,
            "labels": y_val,
            "probabilities": val_probs,
            "metrics": val_metrics,
        },
        "test": {
            "row_indices": split.test,
            "labels": y_test,
            "probabilities": test_probs,
            "metrics": test_metrics,
        },
    }


def train_xgboost(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray | list[int],
    *,
    split: SplitIndices | None = None,
    config: XGBoostConfig = XGBoostConfig(),
) -> dict[str, Any]:
    """Train XGBoost gradient boosting classifier on training split and evaluate."""
    xgb = _xgboost()
    x_array = _as_feature_matrix(X)
    y_array = np.asarray(y, dtype=np.int64)

    if y_array.ndim != 1:
        raise ValueError("y must be a one-dimensional array")
    if y_array.shape[0] != x_array.shape[0]:
        raise ValueError("y and X must have matching row counts")
    if set(np.unique(y_array)) != {0, 1}:
        raise ValueError("y must contain both binary classes {0, 1}")

    split = split or create_stratified_splits(y_array, SplitConfig(seed=config.seed))

    x_train, y_train = x_array[split.train], y_array[split.train]
    x_val, y_val = x_array[split.validation], y_array[split.validation]
    x_test, y_test = x_array[split.test], y_array[split.test]

    clf_kwargs: dict[str, Any] = {
        "n_estimators": config.n_estimators,
        "max_depth": config.max_depth,
        "learning_rate": config.learning_rate,
        "subsample": config.subsample,
        "colsample_bytree": config.colsample_bytree,
        "tree_method": config.tree_method,
        "random_state": config.seed,
        "eval_metric": "logloss",
    }
    if config.device == "cuda":
        clf_kwargs["device"] = "cuda"

    model = xgb.XGBClassifier(**clf_kwargs)

    start_time = time.perf_counter()
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        verbose=False,
    )
    training_seconds = time.perf_counter() - start_time

    val_probs = model.predict_proba(x_val)[:, 1]
    test_probs = model.predict_proba(x_test)[:, 1]

    val_metrics = compute_classification_metrics(y_val, val_probs)
    test_metrics = compute_classification_metrics(y_test, test_probs)

    threshold = select_operating_threshold(
        y_val, val_probs, target_signal_efficiency=config.target_signal_efficiency
    )

    val_signal_mask = y_val == 1
    val_bkg_mask = y_val == 0
    test_signal_mask = y_test == 1
    test_bkg_mask = y_test == 0

    operating_point = {
        "target_signal_efficiency": config.target_signal_efficiency,
        "validation_threshold": threshold,
        "validation_actual_signal_efficiency": float(
            np.mean(val_probs[val_signal_mask] >= threshold)
        ),
        "validation_background_rejection": float(
            np.mean(val_probs[val_bkg_mask] < threshold)
        ),
        "test_actual_signal_efficiency": float(
            np.mean(test_probs[test_signal_mask] >= threshold)
        ),
        "test_background_rejection": float(
            np.mean(test_probs[test_bkg_mask] < threshold)
        ),
    }

    feature_importances = model.feature_importances_.tolist()

    return {
        "model": model,
        "split": split,
        "training_seconds": training_seconds,
        "operating_threshold": threshold,
        "operating_point": operating_point,
        "feature_importances": feature_importances,
        "validation": {
            "row_indices": split.validation,
            "labels": y_val,
            "probabilities": val_probs,
            "metrics": val_metrics,
        },
        "test": {
            "row_indices": split.test,
            "labels": y_test,
            "probabilities": test_probs,
            "metrics": test_metrics,
        },
    }


def write_run_artifacts(
    result: dict[str, Any],
    *,
    run_dir: str | Path,
    run_id: str,
    feature_group: str,
    feature_names: list[str],
    data_source: str,
    row_count: int,
    config: LogisticRegressionConfig | XGBoostConfig,
    model_name: str | None = None,
) -> tuple[Path, Path]:
    """Write schema-compliant metrics.json and test_predictions.csv to run_dir."""
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)

    is_xgb = isinstance(config, XGBoostConfig) or model_name == "xgboost"
    actual_model_name = "xgboost" if is_xgb else "logistic_regression"
    family = "tree_ensemble" if is_xgb else "classical_linear"

    clean_data_source = to_relative_manifest_path(data_source)
    artifact = new_experiment_artifact(
        run_id=run_id,
        model_name=actual_model_name,
        feature_group=feature_group,
        feature_names=feature_names,
        split=result["split"],
        data_source=clean_data_source,
        row_count=row_count,
    )
    artifact["model"]["family"] = family
    training_data: dict[str, Any] = {
        "config": asdict(config),
        "training_seconds": result["training_seconds"],
    }
    if not is_xgb and "coefficients" in result:
        training_data["n_iter"] = result["coefficients"]["n_iter"]
        training_data["intercept"] = result["coefficients"]["intercept"]
    elif is_xgb and "feature_importances" in result:
        training_data["feature_importances"] = result["feature_importances"]

    artifact["training"] = training_data
    artifact["evaluation"] = {
        "validation": result["validation"]["metrics"],
        "test": result["test"]["metrics"],
        "operating_point": result["operating_point"],
    }
    artifact["created_at_utc"] = datetime.now(timezone.utc).isoformat()

    metadata_path = run_path / "metrics.json"
    metadata_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    prediction_path = run_path / "test_predictions.csv"
    pd.DataFrame(
        {
            "row_index": result["test"]["row_indices"],
            "class_label": result["test"]["labels"],
            "score": result["test"]["probabilities"],
        }
    ).to_csv(prediction_path, index=False)

    return metadata_path, prediction_path


def run_feature_group_benchmarks(
    *,
    data_path: str | Path,
    row_count: int,
    output_base_dir: str | Path = "outputs",
    config: LogisticRegressionConfig = LogisticRegressionConfig(),
    xgb_config: XGBoostConfig = XGBoostConfig(),
    model_type: str = "logreg",
    groups: tuple[str, ...] = ("low_level", "high_level", "all"),
) -> dict[str, dict[str, Any]]:
    """Train and export baselines across requested feature groups."""
    base = Path(output_base_dir)
    results: dict[str, dict[str, Any]] = {}

    for group in groups:
        batch = load_higgs_sample(data_path, n_rows=row_count, feature_group=group)
        split = create_stratified_splits(batch.y.to_numpy(), SplitConfig(seed=config.seed))

        if model_type in ("logreg", "both"):
            result_lr = train_logistic_regression(batch.X, batch.y, split=split, config=config)
            run_dir_lr = base / f"logreg_{group}"
            write_run_artifacts(
                result_lr,
                run_dir=run_dir_lr,
                run_id=f"logreg_{group}",
                feature_group=group,
                feature_names=list(batch.X.columns),
                data_source=str(data_path),
                row_count=row_count,
                config=config,
                model_name="logistic_regression",
            )
            results[f"logreg_{group}"] = result_lr

        if model_type in ("xgboost", "both"):
            result_xgb = train_xgboost(batch.X, batch.y, split=split, config=xgb_config)
            run_dir_xgb = base / f"xgboost_{group}"
            write_run_artifacts(
                result_xgb,
                run_dir=run_dir_xgb,
                run_id=f"xgboost_{group}",
                feature_group=group,
                feature_names=list(batch.X.columns),
                data_source=str(data_path),
                row_count=row_count,
                config=xgb_config,
                model_name="xgboost",
            )
            results[f"xgboost_{group}"] = result_xgb

    return results


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for training classical and tree baselines."""
    parser = argparse.ArgumentParser(
        description="Train classical and tree baselines on a documented HIGGS subset."
    )
    parser.add_argument(
        "--data",
        default="data/processed/higgs_sample_250k.csv.gz",
        help="Path to data file (csv or csv.gz)",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=250_000,
        help="Number of rows to load",
    )
    parser.add_argument(
        "--model-type",
        choices=("logreg", "xgboost", "both"),
        default="logreg",
        help="Model family to train (logreg, xgboost, or both)",
    )
    parser.add_argument(
        "--feature-group",
        choices=("low_level", "high_level", "all"),
        default="all",
        help="Feature group to train on when --all-groups is not set",
    )
    parser.add_argument(
        "--all-groups",
        action="store_true",
        help="Train and export benchmarks for all 3 feature groups (low_level, high_level, all)",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Output base directory or specific run directory",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random split seed")
    parser.add_argument("--C", type=float, default=1.0, help="Inverse regularization strength for LogReg")
    parser.add_argument("--max-iter", type=int, default=1000, help="Maximum iterations for LogReg")
    parser.add_argument("--n-estimators", type=int, default=150, help="Number of trees for XGBoost")
    parser.add_argument("--max-depth", type=int, default=6, help="Maximum tree depth for XGBoost")
    parser.add_argument("--learning-rate", type=float, default=0.1, help="Learning rate for XGBoost")
    parser.add_argument(
        "--target-efficiency",
        type=float,
        default=0.50,
        help="Target signal efficiency for validation threshold selection",
    )

    args = parser.parse_args(argv)

    lr_config = LogisticRegressionConfig(
        C=args.C,
        max_iter=args.max_iter,
        seed=args.seed,
        target_signal_efficiency=args.target_efficiency,
    )
    xgb_config = XGBoostConfig(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        seed=args.seed,
        target_signal_efficiency=args.target_efficiency,
    )

    if args.all_groups:
        print(f"Running multi-group benchmark ({args.model_type}) across groups on {args.data}...")
        results = run_feature_group_benchmarks(
            data_path=args.data,
            row_count=args.rows,
            output_base_dir=args.output_dir,
            config=lr_config,
            xgb_config=xgb_config,
            model_type=args.model_type,
        )
        for key, res in results.items():
            val_auc = res["validation"]["metrics"]["roc_auc"]
            test_auc = res["test"]["metrics"]["roc_auc"]
            print(f"Run [{key}]: Val ROC-AUC = {val_auc:.4f}, Test ROC-AUC = {test_auc:.4f}")
        return 0

    batch = load_higgs_sample(args.data, n_rows=args.rows, feature_group=args.feature_group)
    split = create_stratified_splits(batch.y.to_numpy(), SplitConfig(seed=args.seed))

    if args.model_type in ("logreg", "both"):
        result = train_logistic_regression(batch.X, batch.y, split=split, config=lr_config)
        target_dir = Path(args.output_dir)
        if target_dir.name == "outputs":
            target_dir = target_dir / f"logreg_{args.feature_group}"
        meta, pred = write_run_artifacts(
            result,
            run_dir=target_dir,
            run_id=target_dir.name,
            feature_group=args.feature_group,
            feature_names=list(batch.X.columns),
            data_source=args.data,
            row_count=len(batch.X),
            config=lr_config,
            model_name="logistic_regression",
        )
        print(f"Exported Logistic Regression: {meta}")

    if args.model_type in ("xgboost", "both"):
        result = train_xgboost(batch.X, batch.y, split=split, config=xgb_config)
        target_dir = Path(args.output_dir)
        if target_dir.name == "outputs":
            target_dir = target_dir / f"xgboost_{args.feature_group}"
        meta, pred = write_run_artifacts(
            result,
            run_dir=target_dir,
            run_id=target_dir.name,
            feature_group=args.feature_group,
            feature_names=list(batch.X.columns),
            data_source=args.data,
            row_count=len(batch.X),
            config=xgb_config,
            model_name="xgboost",
        )
        print(f"Exported XGBoost: {meta}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
