"""Tests for the serving layer: predictor wiring and health assessment."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.inference.health_score import (
    RiskLevel,
    assess_health,
    health_score,
    risk_level,
)
from src.inference.predictor import ModelRegistry, RULPredictor
from tests.conftest import WINDOW


# --- health scoring -------------------------------------------------------


def test_health_score_is_linear_and_saturates(config):
    assert health_score(0, config=config) == 0.0
    assert health_score(125, config=config) == 100.0
    assert health_score(400, config=config) == 100.0, "must saturate, not exceed 100"
    assert health_score(62.5, config=config) == pytest.approx(50.0)


def test_health_score_floors_negative_predictions(config):
    assert health_score(-10, config=config) == 0.0


def test_risk_bands_follow_the_configured_thresholds(config):
    assert risk_level(5, config) is RiskLevel.CRITICAL
    assert risk_level(20, config) is RiskLevel.CRITICAL     # boundary is inclusive
    assert risk_level(35, config) is RiskLevel.WARNING
    assert risk_level(70, config) is RiskLevel.WATCH
    assert risk_level(200, config) is RiskLevel.HEALTHY


def test_risk_levels_are_ordered_by_severity():
    severities = [level.severity for level in (
        RiskLevel.HEALTHY, RiskLevel.WATCH, RiskLevel.WARNING, RiskLevel.CRITICAL
    )]
    assert severities == sorted(severities) == [0, 1, 2, 3]


def test_assessment_carries_an_action_and_serialises(config):
    assessment = assess_health(12, config=config, engine_id=9, model="gru")

    assert assessment.risk_level is RiskLevel.CRITICAL
    assert assessment.requires_action
    assert "immediately" in assessment.recommended_action.lower()

    payload = assessment.to_dict()
    assert payload["engine_id"] == 9
    assert payload["model"] == "gru"
    assert payload["risk_level"] == "critical"
    assert payload["requires_action"] is True


def test_healthy_machines_require_no_action(config):
    assessment = assess_health(300, config=config)
    assert not assessment.requires_action
    assert assessment.health_score == 100.0


def test_inconsistent_thresholds_are_rejected(base_config):
    from src.config import Config

    payload = base_config.to_dict()
    payload["inference"]["health"]["warning_rul"] = 5  # below critical_rul
    with pytest.raises(ValueError, match="critical_rul <= warning_rul"):
        risk_level(50, Config(payload))


# --- predictor ------------------------------------------------------------


def test_predictor_reads_its_contract_from_the_checkpoint(predictor):
    assert predictor.model_name == "gru"
    assert predictor.is_loaded
    assert predictor.is_torch
    assert predictor.window_size == WINDOW
    assert predictor.n_features == 15
    assert len(predictor.feature_columns) == 15


def test_predictor_info_is_serving_metadata(predictor):
    info = predictor.info()
    assert info["name"] == "gru"
    assert info["loaded"] is True
    assert info["validation_metrics"]["RMSE"] == 28.0
    assert info["window_size"] == WINDOW


def test_predictor_scales_raw_input_before_predicting(predictor, scaler_bundle):
    """A raw window and its pre-scaled equivalent must give the same answer."""
    raw = np.full((WINDOW, 15), 505.0)
    scaled = scaler_bundle.transform(raw)

    assert predictor.predict_one(raw) == pytest.approx(
        predictor.predict_one(scaled, scaled=True), rel=1e-5
    )


def test_predictor_never_returns_negative_rul(predictor):
    predictions = predictor.predict(np.random.default_rng(0).normal(size=(6, WINDOW, 15)) * 50, scaled=True)
    assert (predictions >= 0).all()


def test_predictor_accepts_single_and_batched_windows(predictor):
    single = predictor.predict(np.zeros((WINDOW, 15)), scaled=True)
    batch = predictor.predict(np.zeros((4, WINDOW, 15)), scaled=True)

    assert single.shape == (1,)
    assert batch.shape == (4,)


def test_predictor_validates_window_shape(predictor):
    with pytest.raises(ValueError, match="cycles"):
        predictor.predict(np.zeros((3, 15)), scaled=True)
    with pytest.raises(ValueError, match="features"):
        predictor.predict(np.zeros((WINDOW, 3)), scaled=True)


def test_predictor_builds_windows_from_named_readings(predictor):
    readings = [{name: 500.0 for name in predictor.feature_columns} for _ in range(WINDOW)]
    window = predictor.window_from_readings(readings)

    assert window.shape == (WINDOW, 15)
    assert np.isfinite(predictor.predict_from_readings(readings))


def test_predictor_reorders_readings_to_the_training_order(predictor):
    """Sensor order in the payload must not change the result."""
    columns = predictor.feature_columns
    values = {name: 500.0 + index for index, name in enumerate(columns)}

    forward = [dict(values) for _ in range(WINDOW)]
    reversed_keys = [{k: values[k] for k in reversed(columns)} for _ in range(WINDOW)]

    assert predictor.predict_from_readings(forward) == pytest.approx(
        predictor.predict_from_readings(reversed_keys)
    )


def test_predictor_rejects_incomplete_readings(predictor):
    with pytest.raises(ValueError, match="missing required sensors"):
        predictor.predict_from_readings([{"sensor_2": 1.0} for _ in range(WINDOW)])


def test_predictor_assess_returns_a_decision(predictor):
    assessment = predictor.assess(np.zeros((WINDOW, 15)), scaled=True, engine_id=4)
    assert assessment.engine_id == 4
    assert assessment.model == "gru"
    assert assessment.risk_level in set(RiskLevel)


def test_predictor_assess_batch_checks_id_alignment(predictor):
    with pytest.raises(ValueError, match="engine_ids length"):
        predictor.assess_batch(np.zeros((3, WINDOW, 15)), scaled=True, engine_ids=[1])


def test_predictor_scores_a_frame_per_engine(predictor, labelled_frame):
    result = predictor.predict_frame(labelled_frame)

    assert len(result) == labelled_frame["engine_id"].nunique()
    assert set(result.columns) >= {"engine_id", "rul", "health_score", "risk_level"}
    assert (result["rul"] >= 0).all()


def test_predictor_frame_scoring_is_empty_without_engines(predictor):
    empty = pd.DataFrame(columns=["engine_id", "cycle", *predictor.feature_columns])
    assert predictor.predict_frame(empty).empty


def test_predictor_explain_is_none_for_gru(predictor):
    assert predictor.explain(np.zeros((WINDOW, 15)), scaled=True) is None


def test_attention_predictor_explains_its_prediction(config, prepared_dataset):
    """Attention pooling weights are the model's own account of what mattered."""
    from src.models import build_model
    from src.models.common import save_checkpoint
    from src.preprocessing.scaling import load_scaler

    X, _ = prepared_dataset["train"]
    model = build_model(
        "attention", input_size=15, hidden_size=32, num_layers=1, dropout=0.0, num_heads=4
    )
    save_checkpoint(
        model,
        config.path("models") / "attention.pt",
        model_type="attention",
        input_size=15,
        model_kwargs={"hidden_size": 32, "num_layers": 1, "dropout": 0.0, "num_heads": 4},
        feature_columns=load_scaler(config=config).feature_columns,
        window_size=WINDOW,
    )

    predictor = RULPredictor(model_name="attention", config=config)
    explanation = predictor.explain(np.zeros((WINDOW, 15)), scaled=True)

    assert explanation is not None
    weights = explanation["attention_weights"]
    assert len(weights) == WINDOW
    assert sum(weights) == pytest.approx(1.0, abs=1e-4)
    assert -WINDOW <= explanation["most_influential_cycle"] < 0


def test_predictor_reports_a_missing_checkpoint_actionably(config):
    with pytest.raises(FileNotFoundError, match="scripts/train.py"):
        RULPredictor(model_name="lstm", config=config)


def test_predictor_rejects_an_unknown_model_name(config):
    with pytest.raises(ValueError, match="Unknown model"):
        RULPredictor(model_name="transformer", config=config)


def test_predictor_finds_notebook_era_filenames(config, prepared_dataset):
    """``gru_baseline.pt`` is what the original notebook wrote; it must still load."""
    from src.models import build_model
    from src.models.common import save_checkpoint

    model = build_model("gru", input_size=15, hidden_size=16, num_layers=2, dropout=0.2)
    path = config.path("models") / "gru_baseline.pt"
    save_checkpoint(
        model,
        path,
        model_type="gru",
        input_size=15,
        model_kwargs={"hidden_size": 16, "num_layers": 2, "dropout": 0.2},
    )

    predictor = RULPredictor(model_name="gru", config=config)
    assert predictor.resolve_path().name in {"gru.pt", "gru_baseline.pt"}


def test_lazy_predictor_defers_loading(config, torch_checkpoint):
    predictor = RULPredictor(model_name="gru", config=config, lazy=True)
    assert not predictor.is_loaded

    predictor.predict(np.zeros((WINDOW, 15)), scaled=True)
    assert predictor.is_loaded


# --- registry -------------------------------------------------------------


def test_registry_lists_only_models_present_on_disk(config, torch_checkpoint):
    registry = ModelRegistry(config)
    assert registry.available() == ["gru"]
    assert registry.default_model == "gru"


def test_registry_caches_predictors(config, torch_checkpoint):
    registry = ModelRegistry(config)
    assert registry.get("gru") is registry.get("gru")


def test_registry_preload_survives_missing_checkpoints(config, torch_checkpoint):
    registry = ModelRegistry(config)
    assert registry.preload(["gru", "lstm"]) == ["gru"]


def test_registry_info_does_not_force_a_load(config, torch_checkpoint):
    registry = ModelRegistry(config)
    entries = registry.info()

    assert len(entries) == 1
    assert entries[0]["loaded"] is False
    assert entries[0]["size_mb"] > 0


def test_registry_unload_frees_the_predictor(config, torch_checkpoint):
    registry = ModelRegistry(config)
    registry.get("gru")

    assert registry.unload("gru") is True
    assert registry.unload("gru") is False


def test_registry_rejects_unknown_names(config):
    with pytest.raises(ValueError, match="Unknown model"):
        ModelRegistry(config).get("transformer")
