"""Tests for validation, cleaning, scaling and sequence construction.

The invariants here are the ones that quietly ruin a RUL project when broken:
engine leakage between splits, windows that straddle two machines, and a scaler
fitted on data the model is later evaluated on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ingestion.loader import add_rul
from src.ingestion.validator import (
    ValidationError,
    validate_raw_frame,
    validate_sequences,
    validate_window,
)
from src.preprocessing.cleaning import (
    clip_rul,
    find_constant_sensors,
    prepare_frame,
    remove_duplicate_cycles,
    select_feature_columns,
)
from src.preprocessing.scaling import apply_scaler, fit_scaler, load_scaler, save_scaler
from src.preprocessing.sequences import (
    create_sequences,
    last_window_per_engine,
    split_by_engine,
    split_engines,
)
from tests.conftest import CONSTANT_SENSORS, WINDOW


# --- RUL target -----------------------------------------------------------


def test_add_rul_counts_down_to_zero(raw_frame):
    labelled = add_rul(raw_frame)
    for engine_id, group in labelled.groupby("engine_id"):
        group = group.sort_values("cycle")
        assert group["RUL"].iloc[-1] == 0, f"engine {engine_id} should end at RUL 0"
        assert group["RUL"].iloc[0] == len(group) - 1
        # RUL must fall by exactly one cycle per step.
        assert (group["RUL"].diff().dropna() == -1).all()


def test_add_rul_offsets_by_ground_truth(raw_frame):
    """A test engine stops before failure, so its final RUL is the truth value."""
    truth = pd.Series({engine: 10.0 for engine in raw_frame["engine_id"].unique()})
    labelled = add_rul(raw_frame, final_rul=truth)

    last_values = labelled.groupby("engine_id")["RUL"].min()
    assert (last_values == 10.0).all()


def test_add_rul_rejects_unknown_engine(raw_frame):
    with pytest.raises(ValueError, match="No ground-truth RUL"):
        add_rul(raw_frame, final_rul=pd.Series({999: 5.0}))


def test_clip_rul_caps_the_target(labelled_frame):
    capped = clip_rul(labelled_frame, cap=15.0)
    assert capped["RUL"].max() == 15.0
    assert labelled_frame["RUL"].max() > 15.0, "original frame must not be mutated"


def test_clip_rul_without_cap_is_a_noop(labelled_frame):
    assert clip_rul(labelled_frame, cap=None) is labelled_frame


# --- cleaning -------------------------------------------------------------


def test_find_constant_sensors(labelled_frame, config):
    found = find_constant_sensors(labelled_frame, config=config)
    assert set(found) == set(CONSTANT_SENSORS)


def test_select_feature_columns_drops_constants_and_keeps_order(labelled_frame, config):
    columns = select_feature_columns(labelled_frame, config)

    assert len(columns) == 21 - len(CONSTANT_SENSORS)
    assert not set(columns) & set(CONSTANT_SENSORS)
    # Order must follow the sensor index, since it defines the feature axis.
    assert columns == sorted(columns, key=lambda name: int(name.split("_")[1]))


def test_select_feature_columns_raises_when_nothing_survives(config):
    flat = pd.DataFrame({f"sensor_{i}": [1.0, 1.0] for i in range(1, 22)})
    with pytest.raises(ValueError, match="No usable sensor columns"):
        select_feature_columns(flat, config)


def test_remove_duplicate_cycles(labelled_frame, config):
    doubled = pd.concat([labelled_frame, labelled_frame.head(3)], ignore_index=True)
    cleaned = remove_duplicate_cycles(doubled, config)
    assert len(cleaned) == len(labelled_frame)


def test_prepare_frame_sorts_and_selects(labelled_frame, config):
    shuffled = labelled_frame.sample(frac=1.0, random_state=0)
    frame, columns = prepare_frame(shuffled, config, cap=20.0)

    assert frame["RUL"].max() == 20.0
    assert len(columns) == 15
    assert frame.groupby("engine_id")["cycle"].apply(
        lambda series: series.is_monotonic_increasing
    ).all()


# --- splitting ------------------------------------------------------------


def test_split_engines_is_disjoint_and_covers_everything():
    engines = np.arange(1, 101)
    train, val, test = split_engines(engines, test_size=0.30, val_ratio=0.50, random_state=42)

    assert len(train) == 70 and len(val) == 15 and len(test) == 15
    assert not set(train) & set(val)
    assert not set(train) & set(test)
    assert not set(val) & set(test)
    assert set(train) | set(val) | set(test) == set(engines)


def test_split_engines_is_deterministic():
    engines = np.arange(1, 51)
    first = split_engines(engines, random_state=42)
    second = split_engines(engines, random_state=42)
    assert all(np.array_equal(a, b) for a, b in zip(first, second))


def test_split_engines_needs_three_engines():
    with pytest.raises(ValueError, match="at least 3 engines"):
        split_engines([1, 2])


def test_split_by_engine_selects_only_requested_ids(labelled_frame):
    part = split_by_engine(labelled_frame, [1, 2], "engine_id")
    assert set(part["engine_id"]) == {1, 2}


# --- scaling --------------------------------------------------------------


def test_scaler_standardises_the_training_split(labelled_frame, feature_columns):
    bundle = fit_scaler(labelled_frame, feature_columns)
    scaled = apply_scaler(labelled_frame, bundle)

    assert np.allclose(scaled[feature_columns].mean(), 0.0, atol=1e-6)
    assert np.allclose(scaled[feature_columns].std(ddof=0), 1.0, atol=1e-6)


def test_scaler_uses_training_statistics_on_other_splits(labelled_frame, feature_columns):
    """Validation data must be transformed, never re-fitted - that would leak."""
    train = labelled_frame[labelled_frame["engine_id"] <= 8]
    val = labelled_frame[labelled_frame["engine_id"] > 8]

    bundle = fit_scaler(train, feature_columns)
    scaled_val = apply_scaler(val, bundle)

    manual = (val[feature_columns] - train[feature_columns].mean()) / train[
        feature_columns
    ].std(ddof=0)
    assert np.allclose(scaled_val[feature_columns].to_numpy(), manual.to_numpy(), atol=1e-6)


def test_scaler_roundtrip_preserves_the_contract(config, labelled_frame, feature_columns):
    bundle = fit_scaler(labelled_frame, feature_columns, window_size=WINDOW, rul_cap=125)
    save_scaler(bundle, config=config)
    restored = load_scaler(config=config)

    assert restored.feature_columns == feature_columns
    assert restored.window_size == WINDOW
    assert restored.rul_cap == 125
    assert np.allclose(
        restored.transform(labelled_frame[feature_columns]),
        bundle.transform(labelled_frame[feature_columns]),
    )


def test_scaler_transform_sequences_matches_flat_transform(sequences, scaler_bundle):
    X, _ = sequences
    scaled = scaler_bundle.transform_sequences(X)

    assert scaled.shape == X.shape
    assert np.allclose(scaled[0], scaler_bundle.transform(X[0]))


def test_scaler_rejects_wrong_feature_count(scaler_bundle):
    with pytest.raises(ValueError, match="Expected 15 features"):
        scaler_bundle.transform(np.zeros((4, 3)))


def test_load_scaler_missing_file_is_actionable(config):
    with pytest.raises(FileNotFoundError, match="prepare_data.py"):
        load_scaler(config=config)


# --- sequences ------------------------------------------------------------


def test_create_sequences_shapes(sequences, feature_columns):
    X, y = sequences
    assert X.ndim == 3
    assert X.shape[1] == WINDOW
    assert X.shape[2] == len(feature_columns)
    assert len(X) == len(y)


def test_create_sequences_window_content_is_the_preceding_cycles(
    labelled_frame, feature_columns
):
    """Window i must hold cycles [i-window, i) and be labelled with RUL at i."""
    single = labelled_frame[labelled_frame["engine_id"] == 1].sort_values("cycle")
    X, y = create_sequences(single, feature_columns, "RUL", window_size=WINDOW)

    expected = single[feature_columns].to_numpy()[0:WINDOW]
    assert np.allclose(X[0], expected)
    assert y[0] == single["RUL"].to_numpy()[WINDOW]
    assert len(X) == len(single) - WINDOW


def test_create_sequences_never_crosses_engine_boundaries(labelled_frame, feature_columns):
    per_engine = {
        engine: len(group) - WINDOW
        for engine, group in labelled_frame.groupby("engine_id")
        if len(group) > WINDOW
    }
    X, _ = create_sequences(labelled_frame, feature_columns, "RUL", window_size=WINDOW)
    assert len(X) == sum(per_engine.values())


def test_create_sequences_returns_engine_ids(labelled_frame, feature_columns):
    X, y, ids = create_sequences(
        labelled_frame, feature_columns, "RUL", WINDOW, return_ids=True
    )
    assert len(ids) == len(X)
    # The engine too short to window must be absent.
    assert 12 not in set(ids)


def test_create_sequences_empty_when_window_exceeds_history(labelled_frame, feature_columns):
    X, y = create_sequences(labelled_frame, feature_columns, "RUL", window_size=500)
    assert X.shape == (0, 500, len(feature_columns))
    assert len(y) == 0


def test_create_sequences_validates_columns(labelled_frame):
    with pytest.raises(ValueError, match="Missing feature columns"):
        create_sequences(labelled_frame, ["nope"], "RUL", WINDOW)
    with pytest.raises(ValueError, match="Missing target column"):
        create_sequences(labelled_frame, ["sensor_2"], "missing", WINDOW)


def test_last_window_per_engine_takes_the_most_recent_cycles(
    labelled_frame, feature_columns
):
    X, engine_ids = last_window_per_engine(labelled_frame, feature_columns, WINDOW)

    assert X.shape == (labelled_frame["engine_id"].nunique(), WINDOW, len(feature_columns))
    single = labelled_frame[labelled_frame["engine_id"] == 1].sort_values("cycle")
    position = list(engine_ids).index(1)
    assert np.allclose(X[position], single[feature_columns].to_numpy()[-WINDOW:])


def test_last_window_per_engine_pads_short_histories(labelled_frame, feature_columns):
    X, engine_ids = last_window_per_engine(labelled_frame, feature_columns, WINDOW, pad=True)
    assert 12 in set(engine_ids)  # the 8-cycle engine was padded, not dropped

    skipped, ids = last_window_per_engine(
        labelled_frame, feature_columns, WINDOW, pad=False
    )
    assert 12 not in set(ids)
    assert len(skipped) == len(X) - 1


# --- validation -----------------------------------------------------------


def test_validate_raw_frame_accepts_clean_data(raw_frame, config):
    report = validate_raw_frame(raw_frame, config)
    assert report.ok, report.errors
    assert report.stats["n_engines"] == 12
    assert set(report.stats["constant_sensors"]) == set(CONSTANT_SENSORS)


def test_validate_raw_frame_flags_missing_columns(raw_frame, config):
    report = validate_raw_frame(raw_frame.drop(columns=["sensor_3"]), config)
    assert not report.ok
    assert "Missing columns" in report.errors[0]


def test_validate_raw_frame_flags_nulls_and_duplicates(raw_frame, config):
    broken = raw_frame.copy()
    broken.loc[0, "sensor_2"] = np.nan
    broken = pd.concat([broken, broken.iloc[[5]]], ignore_index=True)

    report = validate_raw_frame(broken, config)
    assert not report.ok
    assert any("Null values" in error for error in report.errors)
    assert any("duplicate" in error for error in report.errors)


def test_validate_raw_frame_flags_non_monotonic_cycles(raw_frame, config):
    broken = raw_frame.copy()
    broken.loc[broken.index[:2], "cycle"] = [5, 1]

    report = validate_raw_frame(broken, config)
    assert not report.ok
    assert any("not increasing" in error for error in report.errors)


def test_validation_report_raises_with_every_error(raw_frame, config):
    report = validate_raw_frame(raw_frame.drop(columns=["cycle"]), config)
    with pytest.raises(ValidationError, match="Data validation failed"):
        report.raise_for_errors()


def test_validate_sequences_catches_bad_shapes(sequences, config):
    X, y = sequences
    assert validate_sequences(X, y, window_size=WINDOW, config=config).ok

    assert not validate_sequences(X[:, :3, :], y, window_size=WINDOW, config=config).ok
    assert not validate_sequences(X, y[:-1], window_size=WINDOW, config=config).ok
    assert not validate_sequences(X, -y - 1, window_size=WINDOW, config=config).ok


def test_validate_window_checks_length_and_finiteness():
    good = np.zeros((WINDOW, 15))
    assert validate_window(good, WINDOW, 15).ok

    assert not validate_window(np.zeros((3, 15)), WINDOW, 15).ok
    assert not validate_window(np.zeros((WINDOW, 4)), WINDOW, 15).ok

    with_nan = good.copy()
    with_nan[0, 0] = np.inf
    assert not validate_window(with_nan, WINDOW, 15).ok
