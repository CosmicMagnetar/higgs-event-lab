"""Unit tests for chunked HIGGS data loading and schema validation (MID-01).

Uses temporary fixtures; does not require the full HIGGS dataset.
"""

import gzip
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure src is on sys.path
SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from higgs_event_lab.data import (
    HiggsBatch,
    InvalidLabelError,
    MissingOrNonFiniteValueError,
    NonNumericDataError,
    SchemaValidationError,
    extract_sample,
    iter_higgs_chunks,
    load_higgs_sample,
    validate_raw_chunk,
)
from higgs_event_lab.features import (
    ALL_COLUMNS,
    HIGH_LEVEL_FEATURES,
    LOW_LEVEL_FEATURES,
    TARGET_COLUMN,
)


def create_synthetic_data(n_rows: int = 10, seed: int = 42) -> np.ndarray:
    """Generate deterministic synthetic data matching the 29-column HIGGS format."""
    rng = np.random.default_rng(seed)
    # Column 0: binary label in {0.0, 1.0}
    labels = rng.integers(0, 2, size=(n_rows, 1)).astype(np.float64)
    # Columns 1-28: float values
    features = rng.normal(loc=0.0, scale=1.0, size=(n_rows, 28)).astype(np.float64)
    return np.hstack([labels, features])


@pytest.fixture
def tiny_csv(tmp_path: Path) -> Path:
    """Fixture providing a temporary uncompressed CSV with 15 valid rows and no header."""
    csv_path = tmp_path / "tiny_higgs.csv"
    data = create_synthetic_data(n_rows=15, seed=123)
    np.savetxt(csv_path, data, delimiter=",", fmt="%.6f")
    return csv_path


@pytest.fixture
def tiny_csv_gz(tmp_path: Path) -> Path:
    """Fixture providing a temporary gzip-compressed CSV with 25 valid rows and no header."""
    gz_path = tmp_path / "tiny_higgs.csv.gz"
    data = create_synthetic_data(n_rows=25, seed=456)
    with gzip.open(gz_path, "wt", encoding="utf-8") as handle:
        for row in data:
            handle.write(",".join(f"{v:.6f}" for v in row) + "\n")
    return gz_path


# --- 1. Basic reading of CSV and compressed CSV without headers ---


def test_read_tiny_uncompressed_csv_no_header(tiny_csv: Path):
    chunks = list(iter_higgs_chunks(tiny_csv, chunk_size=10, feature_group="all"))
    assert len(chunks) == 2  # 15 rows / chunk_size 10 = 2 chunks (10 and 5)

    batch1 = chunks[0]
    assert isinstance(batch1, HiggsBatch)
    assert batch1.X.shape == (10, 28)
    assert batch1.y.shape == (10,)
    assert TARGET_COLUMN not in batch1.X.columns

    batch2 = chunks[1]
    assert batch2.X.shape == (5, 28)
    assert batch2.y.shape == (5,)


def test_read_tiny_compressed_csv_gz(tiny_csv_gz: Path):
    chunks = list(iter_higgs_chunks(tiny_csv_gz, chunk_size=10, feature_group="all"))
    assert len(chunks) == 3  # 25 rows = 10 + 10 + 5

    total_rows = sum(len(c.X) for c in chunks)
    assert total_rows == 25


# --- 2. Target extraction and feature group selection ---


def test_loader_never_places_class_label_in_x(tiny_csv: Path):
    for group in ("low_level", "high_level", "all"):
        for batch in iter_higgs_chunks(tiny_csv, chunk_size=5, feature_group=group):
            assert TARGET_COLUMN not in batch.X.columns
            assert TARGET_COLUMN == batch.y.name


def test_feature_group_selection_low_level(tiny_csv: Path):
    batches = list(iter_higgs_chunks(tiny_csv, chunk_size=15, feature_group="low_level"))
    assert len(batches) == 1
    X, y = batches[0]
    assert X.shape == (15, 21)
    assert list(X.columns) == list(LOW_LEVEL_FEATURES)
    assert TARGET_COLUMN not in X.columns


def test_feature_group_selection_high_level(tiny_csv: Path):
    batches = list(iter_higgs_chunks(tiny_csv, chunk_size=15, feature_group="high_level"))
    assert len(batches) == 1
    X, y = batches[0]
    assert X.shape == (15, 7)
    assert list(X.columns) == list(HIGH_LEVEL_FEATURES)
    assert TARGET_COLUMN not in X.columns


def test_feature_group_selection_all(tiny_csv: Path):
    batches = list(iter_higgs_chunks(tiny_csv, chunk_size=15, feature_group="all"))
    assert len(batches) == 1
    X, y = batches[0]
    assert X.shape == (15, 28)
    assert list(X.columns) == list(LOW_LEVEL_FEATURES + HIGH_LEVEL_FEATURES)


# --- 3. Order preservation and deterministic column order ---


def test_row_and_feature_order_is_deterministic(tiny_csv: Path):
    raw_array = np.loadtxt(tiny_csv, delimiter=",")
    expected_labels = raw_array[:, 0].astype(int)
    expected_features = raw_array[:, 1:]

    sample = load_higgs_sample(tiny_csv, n_rows=15, feature_group="all")
    assert np.array_equal(sample.y.to_numpy(), expected_labels)
    assert np.allclose(sample.X.to_numpy(), expected_features, rtol=1e-5)


# --- 4. Multiple chunks from a file larger than one chunk ---


def test_multiple_chunks_larger_than_chunk_size(tiny_csv_gz: Path):
    chunk_size = 7
    batches = list(iter_higgs_chunks(tiny_csv_gz, chunk_size=chunk_size))
    # 25 rows with chunk_size 7 -> chunks of 7, 7, 7, 4
    sizes = [len(b.X) for b in batches]
    assert sizes == [7, 7, 7, 4]
    assert sum(sizes) == 25


# --- 5. Schema validation and rejection of malformed files ---


def test_rejection_of_too_few_columns(tmp_path: Path):
    bad_csv = tmp_path / "few_cols.csv"
    # 28 columns instead of 29
    data = create_synthetic_data(n_rows=5)[:, :28]
    np.savetxt(bad_csv, data, delimiter=",", fmt="%.6f")

    with pytest.raises(SchemaValidationError, match="expected 29 columns, but found 28"):
        list(iter_higgs_chunks(bad_csv, chunk_size=5))


def test_rejection_of_too_many_columns(tmp_path: Path):
    bad_csv = tmp_path / "extra_cols.csv"
    # 30 columns instead of 29
    raw = create_synthetic_data(n_rows=5)
    extra = np.zeros((5, 1))
    data = np.hstack([raw, extra])
    np.savetxt(bad_csv, data, delimiter=",", fmt="%.6f")

    with pytest.raises(SchemaValidationError, match="expected 29 columns, but found 30"):
        list(iter_higgs_chunks(bad_csv, chunk_size=5))


def test_rejection_of_invalid_labels(tmp_path: Path):
    bad_csv = tmp_path / "bad_labels.csv"
    data = create_synthetic_data(n_rows=5)
    data[2, 0] = 2.0  # Invalid label (must be 0.0 or 1.0)
    np.savetxt(bad_csv, data, delimiter=",", fmt="%.6f")

    with pytest.raises(InvalidLabelError, match="Invalid class_label"):
        list(iter_higgs_chunks(bad_csv, chunk_size=5))


def test_rejection_of_nan_values(tmp_path: Path):
    bad_csv = tmp_path / "with_nan.csv"
    data = create_synthetic_data(n_rows=5)
    data[3, 5] = np.nan
    np.savetxt(bad_csv, data, delimiter=",", fmt="%.6f")

    with pytest.raises(MissingOrNonFiniteValueError, match="NaN/missing"):
        list(iter_higgs_chunks(bad_csv, chunk_size=5))


def test_rejection_of_infinite_values(tmp_path: Path):
    bad_csv = tmp_path / "with_inf.csv"
    data = create_synthetic_data(n_rows=5)
    data[1, 10] = np.inf
    np.savetxt(bad_csv, data, delimiter=",", fmt="%.6f")

    with pytest.raises(MissingOrNonFiniteValueError, match="infinite value"):
        list(iter_higgs_chunks(bad_csv, chunk_size=5))


def test_rejection_of_non_numeric_and_header_row(tmp_path: Path):
    bad_csv = tmp_path / "with_header.csv"
    with open(bad_csv, "w", encoding="utf-8") as f:
        # Write 29 string column names (a header row)
        f.write(",".join(ALL_COLUMNS) + "\n")
        f.write(",".join(["0.0"] + ["1.23"] * 28) + "\n")

    with pytest.raises(NonNumericDataError, match="Non-numeric value"):
        list(iter_higgs_chunks(bad_csv, chunk_size=5))


def test_rejection_of_missing_file():
    with pytest.raises(FileNotFoundError, match="not found"):
        list(iter_higgs_chunks("non_existent_file.csv"))


def test_max_rows_parameter(tiny_csv: Path):
    # File has 15 rows, ask for only 7
    batches = list(iter_higgs_chunks(tiny_csv, chunk_size=5, max_rows=7))
    total_yielded = sum(len(b.X) for b in batches)
    assert total_yielded == 7
    assert len(batches) == 2  # chunk of 5, then chunk of 2


def test_load_higgs_sample_concatenation(tiny_csv: Path):
    sample = load_higgs_sample(tiny_csv, n_rows=12, feature_group="low_level", chunk_size=5)
    assert sample.X.shape == (12, 21)
    assert sample.y.shape == (12,)
    assert TARGET_COLUMN not in sample.X.columns


def test_stage_interim_clean_data(tiny_csv: Path, tmp_path: Path):
    from higgs_event_lab.data import stage_interim_clean_data
    out_dir = tmp_path / "interim"
    manifest = stage_interim_clean_data(tiny_csv, output_dir=out_dir, max_rows=10, chunk_size=5)
    assert manifest["total_rows_clean"] == 10
    assert manifest["total_dropped_rows"] == 0
    assert (out_dir / "cleaning_manifest.json").exists()
    assert (out_dir / "interim_chunk_0000.csv.gz").exists()



def test_seeded_extract_sample_is_global_reproducible_and_without_replacement(tmp_path: Path):
    """A seeded sample must draw from the whole file, not take full early chunks."""
    raw = create_synthetic_data(n_rows=60, seed=987)
    raw[:, 1] = np.arange(len(raw), dtype=np.float64)
    source = tmp_path / "source.csv"
    np.savetxt(source, raw, delimiter=",", fmt="%.8f")

    first_output = tmp_path / "sample_a.csv.gz"
    second_output = tmp_path / "sample_b.csv.gz"
    first_manifest = extract_sample(
        source, first_output, n_rows=20, chunk_size=10, seed=17,
        compare_later_chunk=False,
    )
    extract_sample(
        source, second_output, n_rows=20, chunk_size=10, seed=17,
        compare_later_chunk=False,
    )

    first = pd.read_csv(first_output, header=None).to_numpy()
    second = pd.read_csv(second_output, header=None).to_numpy()
    selected_ids = first[:, 1].astype(int)

    assert first_manifest["sampling_method"] == "seeded_random"
    assert first_manifest["rows_extracted"] == 20
    assert np.array_equal(first, second)
    assert len(np.unique(selected_ids)) == 20
    # The old chunk-level implementation always included the first 20 rows
    # in full when n_rows and chunk_size were multiples of chunk_size.
    assert not np.array_equal(np.sort(selected_ids), np.arange(20))
    assert np.any(selected_ids >= 20)


def test_seeded_extract_sample_handles_sample_larger_than_first_chunk(tmp_path: Path):
    raw = create_synthetic_data(n_rows=35, seed=654)
    raw[:, 1] = np.arange(len(raw), dtype=np.float64)
    source = tmp_path / "source.csv"
    output = tmp_path / "sample.csv.gz"
    np.savetxt(source, raw, delimiter=",", fmt="%.8f")

    manifest = extract_sample(
        source, output, n_rows=30, chunk_size=7, seed=5,
        compare_later_chunk=False,
    )
    sample = pd.read_csv(output, header=None).to_numpy()

    assert manifest["rows_extracted"] == 30
    assert len(sample) == 30
    assert len(np.unique(sample[:, 1])) == 30
    assert np.all(np.diff(sample[:, 1]) > 0)
