"""Research views must distinguish summary evidence from predictions."""
import json
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from scripts.audit_saved_predictions import audit

ROOT = Path(__file__).resolve().parents[2]


def test_prediction_audit_rejects_mismatched_rows(tmp_path):
    for name, indices in [("a", [0, 1]), ("b", [1, 2])]:
        folder = tmp_path / name
        folder.mkdir()
        pd.DataFrame(dict(row_index=indices, class_label=[0, 1], score=[0.1, 0.9])).to_csv(folder / "test_predictions.csv", index=False)
        (folder / "metrics.json").write_text(json.dumps({"evaluation": {"test": {"roc_auc": 1, "pr_auc": 1, "brier_score": .01}}}))
    with pytest.raises(ValueError, match="Different test rows"):
        audit(tmp_path)


def test_research_architecture_and_budget_controls():
    app = AppTest.from_file(str(ROOT / "streamlit_app.py")).run(timeout=30)
    app.sidebar.radio[0].set_value(7).run(timeout=30)
    assert not app.exception
    app.selectbox[0].set_value("Residual MLP").run(timeout=30)
    assert "residual block" in app.code[0].value
    app.selectbox[1].set_value("2.5m").run(timeout=30)
    app.selectbox[2].set_value("low_level").run(timeout=30)
    assert not app.exception
    table = app.dataframe[0].value
    assert len(table) == 5
    assert table.loc[table.Model == "Residual MLP", "roc_auc"].iloc[0] == pytest.approx(.8532, abs=.0001)
    assert any("not a fixed-test" in w.value for w in app.warning)


def test_research_missing_summary_has_actionable_notice(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    app = AppTest.from_file(str(ROOT / "streamlit_app.py")).run(timeout=30)
    app.sidebar.radio[0].set_value(7).run(timeout=30)
    assert not app.exception
    assert any("No release summary" in info.value for info in app.info)
