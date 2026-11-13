"""Dashboard rendering uses temporary fixtures, never benchmark generation."""
import copy
import json
import runpy
from pathlib import Path

from streamlit.testing.v1 import AppTest

from app.loader import discover_and_load_runs
from app.utils.loader import load_experiments
from tests.unit.test_app_loader import valid_run_envelope, valid_predictions_df

ROOT = Path(__file__).resolve().parents[2]


def save_run(root, envelope, predictions):
    folder = root / "outputs" / envelope["run_id"]
    folder.mkdir(parents=True)
    (folder / "metrics.json").write_text(json.dumps(envelope))
    predictions.to_csv(folder / "test_predictions.csv", index=False)
    return folder


def test_adapter_uses_canonical_runs(tmp_path, valid_run_envelope, valid_predictions_df):
    save_run(tmp_path, valid_run_envelope, valid_predictions_df)
    nested = discover_and_load_runs(tmp_path / "outputs")
    adapted = load_experiments(tmp_path)
    assert len(adapted.experiments) == len(nested.valid_runs) == 1
    assert adapted.experiments[0].metrics.roc_auc == nested.valid_runs[0].test_roc_auc
    assert adapted.experiments[0].provenance["source_file"] == nested.valid_runs[0].data_source
    assert len(adapted.curves) == 1


def test_import_launcher_never_generates_artifacts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runpy.run_path(str(ROOT / "streamlit_app.py"), run_name="launcher_import_test")
    assert list(tmp_path.iterdir()) == []


def test_empty_dashboard_does_not_write_outputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    app = AppTest.from_file(str(ROOT / "streamlit_app.py")).run(timeout=30)
    assert not app.exception
    assert any("Awaiting validated experiment artifacts" in item.value for item in app.markdown)
    assert not (tmp_path / "outputs").exists()
    assert not any("Bootstrap" in button.label for button in app.button)


def test_malformed_dashboard_shows_diagnostic(tmp_path, monkeypatch):
    folder = tmp_path / "outputs" / "bad"
    folder.mkdir(parents=True)
    (folder / "metrics.json").write_text("{bad")
    (folder / "test_predictions.csv").write_text("row_index,class_label,score\n0,0,0.1\n")
    monkeypatch.chdir(tmp_path)
    app = AppTest.from_file(str(ROOT / "streamlit_app.py")).run(timeout=30)
    assert not app.exception
    assert any("bad: Unparseable metrics.json" in item.value for item in app.error)


def test_valid_multiple_run_comparison(tmp_path, monkeypatch, valid_run_envelope, valid_predictions_df):
    save_run(tmp_path, valid_run_envelope, valid_predictions_df)
    second = copy.deepcopy(valid_run_envelope)
    second["run_id"] = "second_run"
    save_run(tmp_path, second, valid_predictions_df)
    monkeypatch.chdir(tmp_path)
    app = AppTest.from_file(str(ROOT / "streamlit_app.py")).run(timeout=30)
    app.sidebar.radio[0].set_value(3).run(timeout=30)
    assert not app.exception
    table = app.dataframe[0].value
    assert set(table["Run ID"]) == {valid_run_envelope["run_id"], "second_run"}
    assert set(table["Sample size"]) == {1000}
    assert set(table["Source"]) == {valid_run_envelope["data"]["source"]}
    assert any("one run per model" in item.value for item in app.info)


def test_malformed_eda_does_not_crash_dashboard(tmp_path, monkeypatch):
    folder = tmp_path / "outputs" / "eda"
    folder.mkdir(parents=True)
    (folder / "summary.json").write_text("{broken")
    monkeypatch.chdir(tmp_path)
    app = AppTest.from_file(str(ROOT / "streamlit_app.py")).run(timeout=30)
    assert not app.exception
    assert any("Invalid saved EDA" in item.value for item in app.warning)
    app.sidebar.radio[0].set_value(2).run(timeout=30)
    assert not app.exception
    assert any("Invalid saved EDA" in item.value for item in app.error)
