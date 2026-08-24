"""Tests for drift detection, performance tracking and alerting."""

from __future__ import annotations

import json

import numpy as np
import pytest

from monitoring.alerts import Alert, AlertManager, Severity, alert_from_assessment
from monitoring.drift import (
    detect_drift,
    flatten_sequences,
    population_stability_index,
)
from monitoring.performance import (
    append_history,
    baseline_metrics,
    load_history,
    track_performance,
)

RNG = np.random.default_rng(11)


def _matrix(loc: float = 0.0, scale: float = 1.0, rows: int = 4000, columns: int = 6):
    return RNG.normal(loc, scale, size=(rows, columns))


# --- PSI ------------------------------------------------------------------


def test_psi_is_near_zero_for_the_same_distribution():
    reference = _matrix()[:, 0]
    current = RNG.normal(0, 1, size=4000)
    assert population_stability_index(reference, current) < 0.1


def test_psi_grows_with_the_size_of_the_shift():
    reference = RNG.normal(0, 1, size=4000)
    small = population_stability_index(reference, RNG.normal(0.3, 1, size=4000))
    large = population_stability_index(reference, RNG.normal(3.0, 1, size=4000))

    assert large > small > 0


def test_psi_handles_degenerate_input():
    assert population_stability_index(np.array([]), np.array([1.0, 2.0])) == 0.0
    assert population_stability_index(np.ones(100), np.ones(100)) == 0.0


def test_psi_is_finite_when_a_bin_empties():
    """Disjoint supports would divide by zero without the epsilon floor."""
    value = population_stability_index(RNG.normal(0, 1, 1000), RNG.normal(50, 1, 1000))
    assert np.isfinite(value) and value > 0


def test_flatten_sequences_collapses_the_time_axis():
    assert flatten_sequences(np.zeros((5, 10, 3))).shape == (50, 3)
    assert flatten_sequences(np.zeros((5, 3))).shape == (5, 3)
    with pytest.raises(ValueError, match="2-D or 3-D"):
        flatten_sequences(np.zeros(4))


# --- drift report ---------------------------------------------------------


def test_stable_data_is_reported_as_stable(config):
    report = detect_drift(_matrix(), _matrix(), config=config)

    assert report.status == "stable"
    assert report.drifted_features == []
    assert report.n_features == 6
    assert report.feature_share == 0.0


def test_a_wholesale_shift_is_reported_as_critical(config):
    report = detect_drift(_matrix(), _matrix(loc=4.0), config=config)

    assert report.status == "critical"
    assert len(report.drifted_features) == 6
    assert report.max_psi > 0.25


def test_a_single_drifting_sensor_is_isolated(config):
    reference = _matrix()
    current = _matrix()
    current[:, 2] += 5.0  # one recalibrated probe

    report = detect_drift(reference, current, [f"sensor_{i}" for i in range(6)], config=config)
    assert "sensor_2" in report.drifted_features
    assert len(report.drifted_features) == 1


def test_drift_accepts_sequence_shaped_input(config):
    report = detect_drift(
        RNG.normal(size=(200, 10, 4)), RNG.normal(size=(50, 10, 4)), config=config
    )
    assert report.n_features == 4
    assert report.n_samples == 500  # 50 windows x 10 cycles


def test_drift_report_serialises_and_saves(tmp_path, config):
    report = detect_drift(_matrix(), _matrix(loc=2.0), config=config)
    path = report.save(tmp_path / "drift.json")

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == report.status
    assert len(payload["details"]) == 6
    assert "psi_warning" in payload["thresholds"]
    assert "PSI" in report.summary() or "psi" in report.summary().lower()


def test_drift_rejects_mismatched_feature_counts(config):
    with pytest.raises(ValueError, match="Feature count mismatch"):
        detect_drift(_matrix(columns=6), _matrix(columns=4), config=config)


def test_drift_rejects_wrong_length_name_lists(config):
    with pytest.raises(ValueError, match="feature name"):
        detect_drift(_matrix(), _matrix(), ["only_one"], config=config)


# --- performance tracking -------------------------------------------------


def test_performance_is_ok_when_error_matches_the_baseline(config):
    y_true = np.arange(100, dtype=float)
    report = track_performance(
        y_true, y_true + 5.0, model="gru", config=config,
        baseline={"RMSE": 5.0, "MAE": 5.0}, raise_alerts=False,
    )

    assert report.status == "ok"
    assert not report.degraded
    assert report.metrics["RMSE"] == pytest.approx(5.0)


def test_performance_flags_critical_degradation(config):
    y_true = np.arange(100, dtype=float)
    report = track_performance(
        y_true, y_true + 20.0, model="gru", config=config,
        baseline={"RMSE": 5.0}, raise_alerts=False,
    )

    assert report.status == "critical"
    assert report.degradation_pct["RMSE"] > 25.0


def test_performance_flags_warning_degradation(config):
    y_true = np.arange(100, dtype=float)
    report = track_performance(
        y_true, y_true + 5.75, model="gru", config=config,
        baseline={"RMSE": 5.0}, raise_alerts=False,
    )
    assert report.status == "warning"


def test_small_samples_do_not_trigger_a_verdict(config):
    """Ten points cannot distinguish a bad model from a bad afternoon."""
    y_true = np.arange(10, dtype=float)
    report = track_performance(
        y_true, y_true + 50.0, model="gru", config=config,
        baseline={"RMSE": 5.0}, raise_alerts=False,
    )

    assert report.status == "ok"
    assert "required" in (report.note or "")


def test_missing_baseline_is_reported_not_guessed(config):
    y_true = np.arange(100, dtype=float)
    report = track_performance(
        y_true, y_true + 1.0, model="gru", config=config, baseline={}, raise_alerts=False
    )

    assert report.status == "ok"
    assert "No baseline" in (report.note or "")


def test_performance_history_appends_and_reloads(config):
    y_true = np.arange(100, dtype=float)
    for offset in (5.0, 6.0):
        track_performance(
            y_true, y_true + offset, model="gru", config=config,
            baseline={"RMSE": 5.0}, raise_alerts=False,
        )

    history = load_history(config=config)
    assert len(history) == 2
    assert {"timestamp", "model", "status", "RMSE"} <= set(history.columns)


def test_append_history_writes_the_header_once(config):
    from monitoring.performance import PerformanceReport

    for _ in range(3):
        append_history(
            PerformanceReport(status="ok", model="gru", metrics={"RMSE": 1.0}), config=config
        )
    assert len(load_history(config=config)) == 3


def test_baseline_metrics_prefer_the_configured_values(base_config, config):
    from src.config import Config

    payload = config.to_dict()
    payload["monitoring"]["performance"]["baseline_rmse"] = 17.5
    assert baseline_metrics("gru", Config(payload))["RMSE"] == 17.5


def test_baseline_metrics_fall_back_to_the_leaderboard(config):
    from src.evaluation.compare import update_leaderboard

    update_leaderboard({"Model": "gru", "MAE": 12.0, "RMSE": 19.0}, config=config)
    assert baseline_metrics("gru", config)["RMSE"] == 19.0


def test_baseline_metrics_are_empty_without_any_reference(config):
    assert baseline_metrics("gru", config) == {}


# --- alerts ---------------------------------------------------------------


def test_alerts_are_written_to_the_jsonl_log(config):
    manager = AlertManager(config)
    manager.emit(
        Alert(
            title="Critical RUL",
            message="Inspect immediately",
            severity=Severity.CRITICAL,
            metric="rul_cycles",
            value=8.0,
            threshold=20.0,
            entity=17,
        )
    )

    records = manager.history()
    assert len(records) == 1
    assert records[0]["severity"] == "critical"
    assert records[0]["entity"] == 17
    assert manager.log_path.exists()


def test_alert_history_accumulates_across_managers(config):
    AlertManager(config).emit(Alert(title="one", message="first"))
    AlertManager(config).emit(Alert(title="two", message="second"))

    assert len(AlertManager(config).history()) == 2


def test_alert_severity_filter_drops_quiet_alerts(config):
    manager = AlertManager(config, min_severity=Severity.WARNING)

    assert manager.emit(Alert(title="fyi", message="noise", severity=Severity.INFO)) is False
    assert manager.emit(Alert(title="act", message="real", severity=Severity.WARNING)) is True
    assert manager.summary() == {"info": 0, "warning": 1, "critical": 0}


def test_alert_formats_a_readable_line():
    line = Alert(
        title="Drift", message="4 features moved", severity=Severity.WARNING,
        metric="max_psi", value=0.31, threshold=0.1, entity="fleet",
    ).format()

    assert "[WARNING]" in line
    assert "max_psi=0.31" in line
    assert "fleet" in line


def test_alert_from_assessment_only_fires_when_action_is_needed(config):
    from src.inference.health_score import assess_health

    assert alert_from_assessment(assess_health(300, config=config, engine_id=1)) is None

    alert = alert_from_assessment(assess_health(9, config=config, engine_id=2))
    assert alert is not None
    assert alert.severity is Severity.CRITICAL
    assert alert.entity == 2


def test_alert_from_assessment_maps_warning_band(config):
    from src.inference.health_score import assess_health

    alert = alert_from_assessment(assess_health(40, config=config, engine_id=3))
    assert alert is not None and alert.severity is Severity.WARNING


def test_emit_many_counts_delivered_alerts(config):
    manager = AlertManager(config)
    sent = manager.emit_many(
        [Alert(title=f"a{index}", message="m") for index in range(4)]
    )
    assert sent == 4


def test_webhook_failure_does_not_propagate(config):
    """A broken sink must not take down the job that raised the alert."""
    manager = AlertManager(config, webhook_url="http://127.0.0.1:1/nope")
    assert manager.emit(Alert(title="still logged", message="webhook will fail")) is True
    assert len(manager.history()) == 1
