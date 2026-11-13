"""Chunked data loading, schema validation, and sampling for HIGGS dataset.

Implements MID-01:
- Configurable chunk size and path (.csv and .csv.gz supported).
- Explicit 29-column schema validation (no header expected).
- Strict separation: target label (`class_label`) excluded from feature matrix.
- Deterministic feature group selection (low_level, high_level, all).
- Streaming chunk iteration avoiding full-memory loads.
- Strict validation: fails on non-numeric, NaN, infinite, or invalid label values.
- Reproducible sampling workflow with later-chunk comparison to guard against sampling bias.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, NamedTuple

import numpy as np
import pandas as pd

from higgs_event_lab.features import (
    ALL_COLUMNS,
    ALL_FEATURES,
    TARGET_COLUMN,
    get_feature_columns,
)

EXPECTED_NUM_COLUMNS: int = 29
VALID_LABELS: frozenset[float] = frozenset({0.0, 1.0})


class HiggsDataError(Exception):
    """Base exception for all HIGGS data loading and validation errors."""


class SchemaValidationError(HiggsDataError, ValueError):
    """Raised when the data does not match the expected HIGGS 29-column schema."""


class InvalidLabelError(SchemaValidationError):
    """Raised when class labels are outside {0.0, 1.0}."""


class NonNumericDataError(SchemaValidationError):
    """Raised when data cannot be converted to numeric floats (e.g. headers or text)."""


class MissingOrNonFiniteValueError(SchemaValidationError):
    """Raised when NaN, null, or infinite values are encountered."""


class HiggsBatch(NamedTuple):
    """A batch or sample of HIGGS events with separated features and target.

    Supports both attribute access (batch.X, batch.y) and tuple unpacking (X, y = batch).
    """

    X: pd.DataFrame
    y: pd.Series


def validate_raw_chunk(df: pd.DataFrame, expected_num_cols: int = EXPECTED_NUM_COLUMNS) -> None:
    """Validate a raw DataFrame chunk against HIGGS dataset constraints.

    Checks:
    1. Column count equals expected_num_cols (29).
    2. Chunk is non-empty.
    3. All values are numeric and finite (no NaN, no Inf, no empty cells).
    4. First column (class_label) contains only 0.0 or 1.0.

    Does NOT silently discard invalid rows; raises an exception immediately.

    Parameters
    ----------
    df : pd.DataFrame
        The chunk read from the file.
    expected_num_cols : int, default=29
        Expected column count.

    Raises
    ------
    SchemaValidationError
        If column count is wrong or chunk is empty.
    NonNumericDataError
        If any cell cannot be converted to float.
    MissingOrNonFiniteValueError
        If any cell contains NaN, null, or infinity.
    InvalidLabelError
        If class labels are not in {0.0, 1.0}.
    """
    if df.shape[1] != expected_num_cols:
        raise SchemaValidationError(
            f"Schema mismatch: expected {expected_num_cols} columns, but found {df.shape[1]} columns."
        )

    if len(df) == 0:
        raise SchemaValidationError("Encountered empty data chunk with 0 rows.")

    # Check numeric types
    for col_idx in range(df.shape[1]):
        col = df.iloc[:, col_idx]
        if not pd.api.types.is_numeric_dtype(col):
            try:
                converted = pd.to_numeric(col, errors="raise")
                df.iloc[:, col_idx] = converted
            except (ValueError, TypeError) as exc:
                raise NonNumericDataError(
                    f"Non-numeric value in column index {col_idx}: {exc}"
                ) from exc

    # Check finite (no NaN, no Inf)
    arr = df.to_numpy(dtype=np.float64, copy=False)
    if np.isnan(arr).any():
        nan_count = int(np.isnan(arr).sum())
        raise MissingOrNonFiniteValueError(
            f"Found {nan_count} NaN/missing value(s) in chunk. HIGGS dataset has no missing values."
        )

    if np.isinf(arr).any():
        inf_count = int(np.isinf(arr).sum())
        raise MissingOrNonFiniteValueError(
            f"Found {inf_count} infinite value(s) in chunk. All values must be finite."
        )

    # Validate target labels (column 0)
    label_values = set(df.iloc[:, 0].unique())
    unexpected_labels = label_values - VALID_LABELS
    if unexpected_labels:
        raise InvalidLabelError(
            f"Invalid class_label value(s): {sorted(unexpected_labels)}. Expected only 0.0 (background) or 1.0 (signal)."
        )


def iter_higgs_chunks(
    path: str | Path,
    chunk_size: int = 50_000,
    feature_group: str = "all",
    max_rows: int | None = None,
    dtype: type = np.float32,
    target_dtype: type = np.int64,
) -> Iterator[HiggsBatch]:
    """Iterate over HIGGS dataset in memory-safe chunks.

    Reads from .csv or .csv.gz files without loading the entire dataset into memory.
    Strictly excludes `class_label` from the feature matrix `X`.

    Parameters
    ----------
    path : str or Path
        Path to HIGGS.csv or HIGGS.csv.gz.
    chunk_size : int, default=50_000
        Number of rows per chunk. Must be positive.
    feature_group : {"low_level", "high_level", "all"}, default="all"
        Which feature group to extract.
    max_rows : int, optional
        Maximum total rows to read across all chunks.
    dtype : type, default=np.float32
        Data type for feature matrix X.
    target_dtype : type, default=np.int64
        Data type for target series y (0 or 1).

    Yields
    ------
    HiggsBatch
        Named tuple (X, y) where X is a DataFrame containing only the selected
        feature columns in deterministic order, and y is the target Series.

    Raises
    ------
    FileNotFoundError
        If path does not exist.
    ValueError
        If chunk_size <= 0 or max_rows <= 0.
    SchemaValidationError, NonNumericDataError, MissingOrNonFiniteValueError, InvalidLabelError
        If the file violates schema, data types, or label constraints.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"HIGGS dataset file not found: {path}")

    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be a positive integer, got {chunk_size}")

    if max_rows is not None and max_rows <= 0:
        raise ValueError(f"max_rows must be a positive integer if specified, got {max_rows}")

    selected_features = get_feature_columns(feature_group)
    rows_yielded = 0

    # Read without header, no column names given up front so column count is strictly verified.
    reader = pd.read_csv(
        path,
        header=None,
        chunksize=chunk_size,
        dtype=None,  # let pandas parse initial types, then validate strictly
    )

    for chunk in reader:
        validate_raw_chunk(chunk, expected_num_cols=EXPECTED_NUM_COLUMNS)
        chunk.columns = list(ALL_COLUMNS)

        if max_rows is not None:
            remaining = max_rows - rows_yielded
            if remaining <= 0:
                break
            if len(chunk) > remaining:
                chunk = chunk.iloc[:remaining]

        y = chunk[TARGET_COLUMN].astype(target_dtype)
        y.name = TARGET_COLUMN

        X = chunk[selected_features].astype(dtype)

        # Invariant checks
        assert TARGET_COLUMN not in X.columns, "Critical invariant violated: target found in features"
        assert list(X.columns) == selected_features, "Feature order mismatch"

        rows_yielded += len(chunk)
        yield HiggsBatch(X=X, y=y)

        if max_rows is not None and rows_yielded >= max_rows:
            break


def load_higgs_sample(
    path: str | Path,
    n_rows: int,
    feature_group: str = "all",
    chunk_size: int = 50_000,
    dtype: type = np.float32,
    target_dtype: type = np.int64,
) -> HiggsBatch:
    """Load a specific number of rows into memory using chunked reading.

    Parameters
    ----------
    path : str or Path
        Path to HIGGS.csv or HIGGS.csv.gz.
    n_rows : int
        Exact number of rows to load. Must be positive.
    feature_group : {"low_level", "high_level", "all"}, default="all"
        Feature group to retrieve.
    chunk_size : int, default=50_000
        Chunk size for reading.
    dtype : type, default=np.float32
        Data type for features.
    target_dtype : type, default=np.int64
        Data type for target.

    Returns
    -------
    HiggsBatch
        Combined (X, y) for the requested rows.
    """
    if n_rows <= 0:
        raise ValueError(f"n_rows must be positive, got {n_rows}")

    x_chunks = []
    y_chunks = []
    total_loaded = 0

    for batch in iter_higgs_chunks(
        path=path,
        chunk_size=chunk_size,
        feature_group=feature_group,
        max_rows=n_rows,
        dtype=dtype,
        target_dtype=target_dtype,
    ):
        x_chunks.append(batch.X)
        y_chunks.append(batch.y)
        total_loaded += len(batch.X)

    if total_loaded == 0:
        raise SchemaValidationError(f"No rows were read from {path}")

    X_all = pd.concat(x_chunks, axis=0, ignore_index=True)
    y_all = pd.concat(y_chunks, axis=0, ignore_index=True)

    return HiggsBatch(X=X_all, y=y_all)


def compare_chunks_class_fraction(
    path: str | Path,
    sample_signal_fraction: float,
    chunk_size: int = 50_000,
    skip_rows: int = 500_000,
) -> dict[str, Any]:
    """Compare a sample's signal fraction against a later chunk in the file.

    Guards against assuming the first N rows are representative without testing
    for ordering or batch effects.

    Parameters
    ----------
    path : str or Path
        Path to raw HIGGS dataset.
    sample_signal_fraction : float
        The measured signal fraction from the initial sample.
    chunk_size : int, default=50_000
        Number of rows to inspect in the later chunk.
    skip_rows : int, default=500_000
        Row offset where the later chunk is sampled.

    Returns
    -------
    dict[str, Any]
        Comparison report containing later chunk row count, label distribution,
        later signal fraction, and absolute delta.
    """
    path = Path(path)
    # Stream past skip_rows and inspect the next chunk
    reader = pd.read_csv(
        path,
        header=None,
        skiprows=skip_rows,
        nrows=chunk_size,
        dtype=None,
    )
    validate_raw_chunk(reader, expected_num_cols=EXPECTED_NUM_COLUMNS)
    labels = reader.iloc[:, 0].astype(int)

    counts = labels.value_counts().to_dict()
    total = len(labels)
    later_signal = counts.get(1, 0)
    later_fraction = later_signal / total if total > 0 else 0.0

    delta = abs(sample_signal_fraction - later_fraction)

    return {
        "later_chunk_offset_rows": skip_rows,
        "later_chunk_rows_inspected": total,
        "later_chunk_label_counts": counts,
        "later_chunk_signal_fraction": round(later_fraction, 4),
        "sample_signal_fraction": round(sample_signal_fraction, 4),
        "fraction_difference": round(delta, 4),
        "is_consistent_within_2pct": delta < 0.02,
    }


def compute_file_sha256(path: Path, max_bytes: int = 100 * 1024 * 1024) -> str:
    """Compute SHA-256 for a file (or the first max_bytes for very large files)."""
    hasher = hashlib.sha256()
    bytes_read = 0
    with open(path, "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            hasher.update(chunk)
            bytes_read += len(chunk)
            if bytes_read >= max_bytes:
                break
    return hasher.hexdigest()


def to_relative_manifest_path(path: Path | str) -> str:
    """Normalize a path to be repository-relative for privacy and portability in manifests."""
    p = Path(path)
    try:
        resolved = p.resolve()
        for parent in [resolved] + list(resolved.parents):
            if (parent / "pyproject.toml").is_file() or (parent / ".git").exists():
                return resolved.relative_to(parent).as_posix()
        if not p.is_absolute():
            return p.as_posix()
        return p.name
    except Exception:
        return str(p)


def extract_sample(
    source_path: str | Path,
    output_path: str | Path,
    n_rows: int,
    chunk_size: int = 50_000,
    seed: int | None = None,
    compare_later_chunk: bool = True,
    later_chunk_offset_rows: int = 500_000,
) -> dict[str, Any]:
    """Extract a reproducible sample of HIGGS events and save to an ignored file.

    Requirements enforced:
    - Explicit row limit (n_rows > 0).
    - Random seed recorded if randomness is used.
    - Never overwrites the source dataset.
    - Compares class fraction against a later chunk if requested.
    - Never commits generated data (caller must write to ignored path).

    Parameters
    ----------
    source_path : str or Path
        Source HIGGS dataset file.
    output_path : str or Path
        Output destination (e.g. data/processed/higgs_sample_250k.csv.gz or .parquet).
    n_rows : int
        Number of rows to extract.
    chunk_size : int, default=50_000
        Chunk size for reading.
    seed : int, optional
        Random seed for uniform sampling without replacement across the entire source
        file. If None, extracts the first n_rows sequentially.
    compare_later_chunk : bool, default=True
        Whether to inspect a later chunk to evaluate sampling consistency.
    later_chunk_offset_rows : int, default=500_000
        Row offset for the later comparison chunk.

    Returns
    -------
    dict[str, Any]
        Data manifest dictionary recording provenance, counts, fraction, and comparison.
    """
    source_path = Path(source_path).resolve()
    output_path = Path(output_path).resolve()

    if not source_path.is_file():
        raise FileNotFoundError(f"Source file not found: {source_path}")

    if source_path == output_path:
        raise ValueError(
            f"Output path cannot overwrite source dataset: {source_path}"
        )

    if n_rows <= 0:
        raise ValueError(f"n_rows must be a positive integer, got {n_rows}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows_collected = []
    total_written = 0

    # A seeded sample must be uniform over the whole file, not independently
    # sampled from each chunk. Random priorities let us retain the globally
    # smallest n_rows keys using O(n_rows + chunk_size) memory.
    rng = np.random.default_rng(seed) if seed is not None else None
    sampled_rows: pd.DataFrame | None = None
    sampled_priorities = np.empty(0, dtype=np.float64)
    sampled_source_indices = np.empty(0, dtype=np.int64)
    source_rows_seen = 0

    reader = pd.read_csv(
        source_path,
        header=None,
        chunksize=chunk_size,
        dtype=None,
    )

    for chunk in reader:
        validate_raw_chunk(chunk, expected_num_cols=EXPECTED_NUM_COLUMNS)

        if rng is None:
            remaining = n_rows - total_written
            if remaining <= 0:
                break
            sub = chunk.iloc[:remaining]
            rows_collected.append(sub)
            total_written += len(sub)
        else:
            chunk_length = len(chunk)
            candidate_rows = (
                chunk.reset_index(drop=True)
                if sampled_rows is None
                else pd.concat([sampled_rows, chunk], ignore_index=True)
            )
            candidate_priorities = np.concatenate(
                [sampled_priorities, rng.random(chunk_length)]
            )
            candidate_source_indices = np.concatenate(
                [
                    sampled_source_indices,
                    np.arange(
                        source_rows_seen,
                        source_rows_seen + chunk_length,
                        dtype=np.int64,
                    ),
                ]
            )

            if len(candidate_rows) > n_rows:
                selected = np.argpartition(candidate_priorities, n_rows - 1)[:n_rows]
                sampled_rows = candidate_rows.iloc[selected].reset_index(drop=True)
                sampled_priorities = candidate_priorities[selected]
                sampled_source_indices = candidate_source_indices[selected]
            else:
                sampled_rows = candidate_rows
                sampled_priorities = candidate_priorities
                sampled_source_indices = candidate_source_indices

            source_rows_seen += chunk_length

    if rng is not None and sampled_rows is not None:
        # Preserve original file order in the exported sample for easier auditing.
        order = np.argsort(sampled_source_indices)
        combined_df = sampled_rows.iloc[order].reset_index(drop=True)
        total_written = len(combined_df)
    else:
        if not rows_collected:
            raise SchemaValidationError("No data was extracted from source file.")
        combined_df = pd.concat(rows_collected, axis=0, ignore_index=True)
        total_written = len(combined_df)

    if total_written == 0:
        raise SchemaValidationError("No data was extracted from source file.")

    # Save format
    if output_path.suffix == ".parquet":
        combined_df.columns = list(ALL_COLUMNS)
        combined_df.to_parquet(output_path, index=False)
    elif str(output_path).endswith(".csv.gz"):
        # Headerless to preserve HIGGS format compatibility
        combined_df.to_csv(output_path, header=False, index=False, compression="gzip")
    elif output_path.suffix == ".csv":
        combined_df.to_csv(output_path, header=False, index=False)
    else:
        # Default to compressed CSV if extension unknown
        combined_df.to_csv(output_path, header=False, index=False, compression="gzip")

    label_counts = {
        0.0: int((combined_df.iloc[:, 0] == 0.0).sum()),
        1.0: int((combined_df.iloc[:, 0] == 1.0).sum()),
    }
    signal_fraction = label_counts[1.0] / total_written if total_written > 0 else 0.0

    comparison = None
    if compare_later_chunk:
        try:
            comparison = compare_chunks_class_fraction(
                path=source_path,
                sample_signal_fraction=signal_fraction,
                chunk_size=min(50_000, n_rows),
                skip_rows=later_chunk_offset_rows,
            )
        except Exception as exc:
            comparison = {
                "error": f"Could not perform later chunk comparison: {exc}"
            }

    manifest = {
        "source_path": to_relative_manifest_path(source_path),
        "source_size_bytes": source_path.stat().st_size,
        "source_sha256_prefix": compute_file_sha256(source_path),
        "output_path": to_relative_manifest_path(output_path),
        "rows_extracted": total_written,
        "sampling_method": "seeded_random" if seed is not None else "sequential_head",
        "random_seed": seed,
        "label_counts": label_counts,
        "signal_fraction": round(signal_fraction, 4),
        "later_chunk_comparison": comparison,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "columns_count": combined_df.shape[1],
    }

    manifest_path = output_path.with_name(f"{output_path.name}.manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return manifest


def stage_interim_clean_data(
    source_path: str | Path,
    output_dir: str | Path = "data/interim",
    max_rows: int = 100_000,
    chunk_size: int = 50_000,
) -> dict[str, Any]:
    """Clean, validate, and stage intermediate dataset chunks into data/interim/.

    Inspects raw data chunks, accounts for dropped or invalid rows with explicit reasons,
    writes cleaned interim chunks, and produces a cleaning manifest.
    """
    interim_path = Path(output_dir)
    interim_path.mkdir(parents=True, exist_ok=True)

    total_rows_read = 0
    total_rows_clean = 0
    dropped_breakdown: dict[str, int] = {
        "non_numeric": 0,
        "missing_or_nan": 0,
        "infinite_values": 0,
        "invalid_labels": 0,
    }

    chunks_written: list[str] = []
    chunk_index = 0

    reader = pd.read_csv(
        source_path,
        header=None,
        chunksize=chunk_size,
    )

    for chunk in reader:
        chunk_read = len(chunk)
        total_rows_read += chunk_read

        if chunk.shape[1] != EXPECTED_NUM_COLUMNS:
            raise SchemaValidationError(f"Expected {EXPECTED_NUM_COLUMNS} columns, got {chunk.shape[1]}")

        numeric_chunk = chunk.apply(pd.to_numeric, errors="coerce")
        bad_rows = numeric_chunk.isna().any(axis=1)
        if bad_rows.any():
            dropped_breakdown["non_numeric"] += int(bad_rows.sum())
            numeric_chunk = numeric_chunk[~bad_rows]

        labels = numeric_chunk.iloc[:, 0]
        invalid_label_mask = ~labels.isin(VALID_LABELS)
        if invalid_label_mask.any():
            dropped_breakdown["invalid_labels"] += int(invalid_label_mask.sum())
            numeric_chunk = numeric_chunk[~invalid_label_mask]

        inf_mask = np.isinf(numeric_chunk.to_numpy()).any(axis=1)
        if inf_mask.any():
            dropped_breakdown["infinite_values"] += int(inf_mask.sum())
            numeric_chunk = numeric_chunk[~inf_mask]

        clean_count = len(numeric_chunk)
        total_rows_clean += clean_count

        chunk_file = interim_path / f"interim_chunk_{chunk_index:04d}.csv.gz"
        numeric_chunk.columns = list(ALL_COLUMNS)
        numeric_chunk.to_csv(chunk_file, index=False, compression="gzip")
        chunks_written.append(str(chunk_file))
        chunk_index += 1

        if total_rows_clean >= max_rows:
            break

    total_dropped = sum(dropped_breakdown.values())
    manifest = {
        "source_path": to_relative_manifest_path(source_path),
        "total_rows_read": total_rows_read,
        "total_rows_clean": total_rows_clean,
        "total_dropped_rows": total_dropped,
        "dropped_reasons": dropped_breakdown,
        "chunks_written": [to_relative_manifest_path(c) for c in chunks_written],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS: All records validated successfully" if total_dropped == 0 else f"NOTE: {total_dropped} rows dropped",
    }

    manifest_path = interim_path / "cleaning_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for extracting HIGGS data subsets (python -m higgs_event_lab.data)."""
    parser = argparse.ArgumentParser(
        description="Chunked data extraction and schema validation for HIGGS dataset."
    )
    parser.add_argument(
        "--interim-clean",
        action="store_true",
        help="Stage and clean intermediate chunks into data/interim/ with cleaning manifest",
    )
    parser.add_argument(
        "--interim-dir",
        type=str,
        default="data/interim",
        help="Output directory for interim cleaning (default: data/interim)",
    )

    parser.add_argument(
        "--source",
        type=str,
        default="data/raw/HIGGS.csv.gz",
        help="Path to source HIGGS CSV/GZ file (default: data/raw/HIGGS.csv.gz)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/higgs_sample_250k.csv.gz",
        help="Path to output file (default: data/processed/higgs_sample_250k.csv.gz)",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=250_000,
        help="Number of rows to extract (default: 250000)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50_000,
        help="Chunk size for reading (default: 50000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible sampling",
    )
    parser.add_argument(
        "--skip-later-check",
        action="store_true",
        help="Skip comparison with later chunk (default: False)",
    )
    parser.add_argument(
        "--later-offset",
        type=int,
        default=500_000,
        help="Row offset for later comparison chunk (default: 500000)",
    )

    args = parser.parse_args(argv)

    if args.interim_clean:
        print(f"HIGGS Interim Data Cleaning & Staging")
        print(f"Source file        : {args.source}")
        print(f"Output directory   : {args.interim_dir}")
        print(f"Requested rows     : {args.rows}")
        print(f"Chunk size         : {args.chunk_size}")
        try:
            manifest = stage_interim_clean_data(
                source_path=args.source,
                output_dir=args.interim_dir,
                max_rows=args.rows,
                chunk_size=args.chunk_size,
            )
            print(f"\nSUCCESS: Cleaned {manifest['total_rows_clean']} rows.")
            print(f"Total dropped rows : {manifest['total_dropped_rows']}")
            print(f"Status             : {manifest['status']}")
            print(f"Cleaning manifest  : {args.interim_dir}/cleaning_manifest.json")
            return 0
        except Exception as exc:
            print(f"\nERROR: Interim cleaning failed: {exc}", file=sys.stderr)
            return 1

    print(f"HIGGS Data Access & Extraction (MID-01)")
    print(f"Source file        : {args.source}")
    print(f"Output file        : {args.output}")
    print(f"Requested rows     : {args.rows}")
    print(f"Chunk size         : {args.chunk_size}")
    print(f"Random seed        : {args.seed}")


    try:
        manifest = extract_sample(
            source_path=args.source,
            output_path=args.output,
            n_rows=args.rows,
            chunk_size=args.chunk_size,
            seed=args.seed,
            compare_later_chunk=not args.skip_later_check,
            later_chunk_offset_rows=args.later_offset,
        )
    except Exception as exc:
        print(f"\nERROR: Extraction failed: {exc}", file=sys.stderr)
        return 1

    print(f"\nSUCCESS: Extracted {manifest['rows_extracted']} rows.")
    print(f"Label counts       : {manifest['label_counts']}")
    print(f"Signal fraction    : {manifest['signal_fraction']}")

    comp = manifest.get("later_chunk_comparison")
    if comp and "later_chunk_signal_fraction" in comp:
        print(
            f"Later chunk ({comp['later_chunk_offset_rows']} rows offset) signal fraction: "
            f"{comp['later_chunk_signal_fraction']} (difference: {comp['fraction_difference']})"
        )
        if comp.get("is_consistent_within_2pct"):
            print("Sampling check     : PASS (consistent within 2%)")
        else:
            print("Sampling check     : NOTE (fraction differs by >2%; evaluate potential sampling bias)")

    print(f"Manifest saved to  : {args.output}.manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
