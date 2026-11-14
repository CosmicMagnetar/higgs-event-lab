"""Regression tests for safe dashboard model loading."""

from pathlib import Path

from app.views import simulation


def test_xgboost_weight_is_disabled_by_default(tmp_path: Path, monkeypatch):
    """A stale serialized XGBoost binary must not be deserialized during page load."""
    monkeypatch.setattr(simulation, "WEIGHTS_DIR", tmp_path)
    (tmp_path / "xgboost_tuned.joblib").write_bytes(b"incompatible-binary")
    monkeypatch.delenv("HIGGSLENS_ENABLE_XGB_WEIGHTS", raising=False)

    models, scaler = simulation.load_models()

    assert models == {}
    assert scaler is None
    assert any("XGBoost interactive weights are disabled" in note for note in simulation.MODEL_LOAD_NOTICES)



def test_predict_scores_rejects_models_without_score_api(monkeypatch):
    """Hard class labels must not be presented as continuous classifier scores."""
    import numpy as np
    import pandas as pd
    import pytest
    from higgs_event_lab.features import ALL_FEATURES

    class LabelOnlyModel:
        def predict(self, values):
            return np.zeros(len(values), dtype=int)

    frame = pd.DataFrame([[0.0] * len(ALL_FEATURES)], columns=ALL_FEATURES)

    with pytest.raises(TypeError, match="hard class predictions are not valid classifier scores"):
        simulation.predict_scores({"label_only": LabelOnlyModel()}, None, frame)
