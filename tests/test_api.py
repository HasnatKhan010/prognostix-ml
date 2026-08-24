"""Tests for the HTTP API.

The client fixture injects a predictor backed by a temp checkpoint, so these
tests exercise the real request → validation → scaling → model → response path
without depending on any committed artifact.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.config import get_config
from tests.conftest import WINDOW

FEATURES = 15
PREFIX = "/api/v1"


def _window(cycles: int = WINDOW, features: int = FEATURES, seed: int = 0) -> list[list[float]]:
    rng = np.random.default_rng(seed)
    return (500.0 + rng.normal(0, 1.0, size=(cycles, features))).round(4).tolist()


# --- service endpoints ----------------------------------------------------


def test_service_info_lists_the_endpoints(api_client):
    response = api_client.get("/api/info")
    assert response.status_code == 200

    payload = response.json()
    assert payload["docs"] == "/docs"
    assert payload["dashboard"] == "/"
    assert payload["endpoints"]["predict"].endswith("/predict")


def test_root_serves_the_dashboard(api_client):
    """The API serves the static dashboard, so one container answers both."""
    response = api_client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Prognostix" in response.text


def test_dashboard_assets_are_served(api_client):
    for path, content_type in (("/app.js", "javascript"), ("/styles.css", "css")):
        response = api_client.get(path)
        assert response.status_code == 200, path
        assert content_type in response.headers["content-type"]


def test_health_reports_available_models(api_client):
    response = api_client.get(f"{PREFIX}/health")
    assert response.status_code == 200

    payload = response.json()
    assert payload["status"] == "ok"
    assert "gru" in payload["models_available"]
    assert "gru" in payload["models_loaded"]
    assert payload["default_model"] == "gru"
    assert payload["uptime_seconds"] >= 0


def test_health_is_reachable_without_the_prefix(api_client):
    """Container health checks should not need to know the API prefix."""
    assert api_client.get("/health").status_code == 200


def test_models_endpoint_describes_the_contract(api_client):
    response = api_client.get(f"{PREFIX}/models")
    assert response.status_code == 200

    payload = response.json()
    assert payload["count"] == 1
    model = payload["models"][0]
    assert model["name"] == "gru"
    assert model["window_size"] == WINDOW
    assert model["n_features"] == FEATURES
    assert len(model["feature_columns"]) == FEATURES


def test_openapi_schema_is_served(api_client):
    response = api_client.get("/openapi.json")
    assert response.status_code == 200
    assert f"{PREFIX}/predict" in response.json()["paths"]


def test_metrics_endpoint_exposes_prometheus_text(api_client):
    response = api_client.get(f"{PREFIX}/metrics")
    assert response.status_code in (200, 501)
    if response.status_code == 200:
        assert "prognostix_predictions_total" in response.text


# --- predict --------------------------------------------------------------


def test_predict_from_a_numeric_window(api_client):
    response = api_client.post(
        f"{PREFIX}/predict", json={"engine_id": 7, "window": _window()}
    )
    assert response.status_code == 200, response.text

    payload = response.json()
    assert payload["engine_id"] == 7
    assert payload["model"] == "gru"
    assert payload["unit"] == "cycles"
    assert payload["rul_cycles"] >= 0
    assert 0 <= payload["health_score"] <= 100
    assert payload["risk_level"] in {"healthy", "watch", "warning", "critical"}
    assert payload["recommended_action"]
    assert payload["window_size"] == WINDOW


def test_predict_from_named_readings(api_client, predictor):
    """Callers may send full sensor dictionaries; the model picks its columns."""
    columns = predictor.feature_columns
    readings = [
        {name: 500.0 + index * 0.1 for name in columns} for index in range(WINDOW)
    ]
    response = api_client.post(
        f"{PREFIX}/predict", json={"engine_id": "pump-3", "readings": readings}
    )

    assert response.status_code == 200, response.text
    assert response.json()["engine_id"] == "pump-3"


def test_predict_readings_may_carry_extra_sensors(api_client, predictor):
    readings = [
        {**{name: 500.0 for name in predictor.feature_columns}, "sensor_99": 1.0}
        for _ in range(WINDOW)
    ]
    assert api_client.post(f"{PREFIX}/predict", json={"readings": readings}).status_code == 200


def test_predict_rejects_missing_sensors(api_client, predictor):
    incomplete = [{predictor.feature_columns[0]: 500.0} for _ in range(WINDOW)]
    response = api_client.post(f"{PREFIX}/predict", json={"readings": incomplete})

    assert response.status_code == 422
    assert "missing required sensors" in response.json()["detail"].lower()


def test_predict_rejects_a_short_window(api_client):
    response = api_client.post(f"{PREFIX}/predict", json={"window": _window(cycles=3)})
    assert response.status_code == 422
    assert "cycles" in response.json()["detail"].lower()


def test_predict_rejects_the_wrong_feature_count(api_client):
    response = api_client.post(f"{PREFIX}/predict", json={"window": _window(features=4)})
    assert response.status_code == 422


def test_predict_rejects_a_ragged_window(api_client):
    window = _window()
    window[2] = window[2][:-1]
    response = api_client.post(f"{PREFIX}/predict", json={"window": window})

    assert response.status_code == 422
    assert "ragged" in response.json()["detail"].lower()


def test_predict_rejects_non_finite_values(api_client):
    response = api_client.post(
        f"{PREFIX}/predict", json={"window": [["nan"] * FEATURES] * WINDOW}
    )
    assert response.status_code == 422


def test_predict_requires_exactly_one_input_form(api_client):
    both = api_client.post(
        f"{PREFIX}/predict", json={"window": _window(), "readings": [{"sensor_2": 1.0}]}
    )
    neither = api_client.post(f"{PREFIX}/predict", json={"engine_id": 1})

    assert both.status_code == 422
    assert neither.status_code == 422
    assert "exactly one" in both.json()["detail"].lower()


def test_predict_rejects_an_empty_window(api_client):
    assert api_client.post(f"{PREFIX}/predict", json={"window": []}).status_code == 422


def test_predict_with_an_untrained_model_is_unavailable(api_client):
    response = api_client.post(
        f"{PREFIX}/predict", json={"model": "lstm", "window": _window()}
    )
    assert response.status_code == 503
    assert "checkpoint" in response.json()["detail"].lower()


def test_predict_with_an_unknown_model_name_is_rejected(api_client):
    response = api_client.post(
        f"{PREFIX}/predict", json={"model": "transformer", "window": _window()}
    )
    assert response.status_code == 422


def test_predict_explain_is_absent_for_non_attention_models(api_client):
    response = api_client.post(
        f"{PREFIX}/predict?explain=true", json={"window": _window()}
    )
    assert response.status_code == 200
    assert response.json()["attention"] is None


def test_predict_accepts_prescaled_windows(api_client):
    response = api_client.post(
        f"{PREFIX}/predict",
        json={"window": np.zeros((WINDOW, FEATURES)).tolist(), "scaled": True},
    )
    assert response.status_code == 200


# --- batch ----------------------------------------------------------------


def test_batch_predict_returns_a_row_per_item(api_client):
    response = api_client.post(
        f"{PREFIX}/predict/batch",
        json={
            "items": [
                {"engine_id": 1, "window": _window(seed=1)},
                {"engine_id": 2, "window": _window(seed=2)},
                {"engine_id": 3, "window": _window(seed=3)},
            ]
        },
    )
    assert response.status_code == 200, response.text

    payload = response.json()
    assert payload["count"] == 3
    assert [item["engine_id"] for item in payload["predictions"]] == [1, 2, 3]
    assert sum(payload["risk_summary"].values()) == 3
    assert payload["action_required"] <= 3


def test_batch_predict_mixes_input_forms(api_client, predictor):
    readings = [{name: 500.0 for name in predictor.feature_columns} for _ in range(WINDOW)]
    response = api_client.post(
        f"{PREFIX}/predict/batch",
        json={
            "items": [
                {"engine_id": 1, "window": _window()},
                {"engine_id": 2, "readings": readings},
            ]
        },
    )
    assert response.status_code == 200
    assert response.json()["count"] == 2


def test_batch_predict_applies_the_top_level_model(api_client):
    response = api_client.post(
        f"{PREFIX}/predict/batch",
        json={"model": "gru", "items": [{"window": _window()}]},
    )
    assert response.status_code == 200
    assert response.json()["predictions"][0]["model"] == "gru"


def test_batch_predict_fails_the_whole_request_on_a_bad_item(api_client):
    """Partial fleet results would be more dangerous than an explicit error."""
    response = api_client.post(
        f"{PREFIX}/predict/batch",
        json={"items": [{"window": _window()}, {"window": _window(cycles=2)}]},
    )
    assert response.status_code == 422


def test_batch_predict_requires_at_least_one_item(api_client):
    assert api_client.post(f"{PREFIX}/predict/batch", json={"items": []}).status_code == 422


def test_batch_predict_enforces_a_size_limit(api_client):
    response = api_client.post(
        f"{PREFIX}/predict/batch",
        json={"items": [{"window": _window()} for _ in range(300)]},
    )
    assert response.status_code == 422


# --- leaderboard and monitoring -------------------------------------------


def test_leaderboard_endpoint_returns_rows_or_empties_cleanly(api_client):
    response = api_client.get(f"{PREFIX}/leaderboard")
    assert response.status_code == 200

    payload = response.json()
    assert payload["metric"] == "RMSE"
    assert payload["count"] == len(payload["rows"])


def test_leaderboard_rejects_an_unknown_metric(api_client):
    from src.evaluation.compare import load_leaderboard

    if load_leaderboard(config=get_config()).empty:
        pytest.skip("no leaderboard on disk to validate the metric against")
    assert api_client.get(f"{PREFIX}/leaderboard?metric=nope").status_code == 400


@pytest.mark.skipif(
    not (get_config().path("data_processed") / "train_sequences.npz").exists(),
    reason="reference distribution requires prepared sequences",
)
def test_drift_endpoint_scores_live_windows(api_client):
    response = api_client.post(
        f"{PREFIX}/monitoring/drift",
        json={"windows": [_window(seed=index) for index in range(4)]},
    )
    assert response.status_code in (200, 503), response.text
    if response.status_code == 200:
        payload = response.json()
        assert payload["status"] in {"stable", "warning", "critical"}
        assert payload["n_features"] == FEATURES
        assert len(payload["details"]) == FEATURES


def test_drift_endpoint_rejects_inconsistent_windows(api_client):
    response = api_client.post(
        f"{PREFIX}/monitoring/drift",
        json={"windows": [[[1.0, 2.0], [1.0]]]},
    )
    assert response.status_code == 422


# --- error handling -------------------------------------------------------


def test_unknown_route_is_a_404(api_client):
    assert api_client.get(f"{PREFIX}/does-not-exist").status_code == 404


def test_validation_errors_use_the_shared_error_body(api_client):
    response = api_client.post(f"{PREFIX}/predict", json={"window": "not-a-matrix"})
    assert response.status_code == 422

    payload = response.json()
    assert "detail" in payload
    assert payload["error_type"] == "validation_error"
    assert "timestamp" in payload
