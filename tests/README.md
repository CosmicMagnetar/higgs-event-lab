# Tests

Tests are added alongside implementation, not postponed until the deadline. The suite enforces repository hygiene, leak prevention, schema integrity, and artifact validity.

## Test Structure

| Folder | Purpose |
|---|---|
| `unit/` | Small, fast tests for one function or invariant at a time: data schema checks, label/feature separation, feature-group shape, metric calculations, threshold choice, artifact validation, notebook style, download logic. |
| `integration/` | Deterministic synthetic-data flows through loading, preparation, model prediction, metrics, output writing, and dashboard loading. |

## Synthetic Data Policy

Synthetic data are for testing code paths only. They must **not** be described as HIGGS benchmark results or presented as experiment evidence.

## Running Tests

```bash
# Install all test dependencies, including Streamlit and training libraries
python -m pip install -e ".[dev]"

# Run the full suite
python -m pytest -q

# Run only unit tests
pytest tests/unit/

# Run only integration tests
pytest tests/integration/

# Run with verbose output
pytest -v
```

## Coordination

Saumya Mishra coordinates the QA checklist. Module owners write and repair tests for the code they own:
- Aditya Rana: data, features, classical tests.
- Rachit Gupta: neural tests.
- Krishna: protocol, splits, evaluation, artifacts, dashboard, repo hygiene tests.

---

Dashboard regression checks:

```bash
python -m pytest tests/unit/test_app_loader.py tests/unit/test_dashboard_loader.py tests/unit/test_app_views.py tests/unit/test_app_metrics.py -q
```

CI uses the same `.[dev]` installation and full-suite command. Synthetic test
fixtures stay in temporary directories and are never demo evidence.
