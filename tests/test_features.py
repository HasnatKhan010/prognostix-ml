"""Tests for feature engineering.

Two things are easy to get wrong and expensive to notice later: the block order of
flattened window statistics (which must match the feature names, or a saved model
reads its inputs in the wrong order), and leakage of one engine's history into
another through a rolling or lag window.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.features.engineering import (
    DEFAULT_STATS,
    STAT_FUNCTIONS,
    build_tabular_frame,
    create_statistical_features,
    statistical_feature_names,
)
from src.features.lag_features import add_diff_features, add_lag_features
from src.features.rolling_features import add_expanding_features, add_rolling_features

# --- window statistics ----------------------------------------------------


def test_statistical_features_shape(sequences):
    X, _ = sequences
    features = create_statistical_features(X)
    assert features.shape == (len(X), len(DEFAULT_STATS) * X.shape[2])


def test_statistical_features_values_are_correct():
    """Each block must hold the aggregation it claims, in declared order."""
    X = np.array(
        [
            [[1.0, 10.0], [2.0, 20.0], [6.0, 30.0]],
        ]
    )
    features = create_statistical_features(X, stats=["mean", "std", "min", "max", "last", "trend"])
    n = 2

    assert np.allclose(features[0, 0:n], [3.0, 20.0])                    # mean
    assert np.allclose(features[0, n : 2 * n], X[0].std(axis=0))          # std
    assert np.allclose(features[0, 2 * n : 3 * n], [1.0, 10.0])           # min
    assert np.allclose(features[0, 3 * n : 4 * n], [6.0, 30.0])           # max
    assert np.allclose(features[0, 4 * n : 5 * n], [6.0, 30.0])           # last
    assert np.allclose(features[0, 5 * n : 6 * n], [5.0, 20.0])           # trend


def test_statistical_features_accept_custom_stats(sequences):
    X, _ = sequences
    features = create_statistical_features(X, stats=["mean", "range"])
    assert features.shape == (len(X), 2 * X.shape[2])


def test_statistical_feature_names_align_with_columns(sequences, feature_columns):
    X, _ = sequences
    names = statistical_feature_names(feature_columns)
    assert len(names) == create_statistical_features(X).shape[1]
    assert names[0] == f"{feature_columns[0]}_mean"
    assert names[-1] == f"{feature_columns[-1]}_trend"


def test_statistical_features_reject_unknown_stat(sequences):
    X, _ = sequences
    with pytest.raises(ValueError, match="Unknown aggregation"):
        create_statistical_features(X, stats=["mean", "kurtosis"])


def test_statistical_features_reject_wrong_dimensionality():
    with pytest.raises(ValueError, match="Expected a 3-D array"):
        create_statistical_features(np.zeros((4, 5)))


def test_statistical_features_handle_empty_input():
    features = create_statistical_features(np.empty((0, 10, 15)))
    assert features.shape == (0, len(DEFAULT_STATS) * 15)


def test_every_registered_stat_reduces_the_time_axis(sequences):
    X, _ = sequences
    for name, function in STAT_FUNCTIONS.items():
        assert function(X).shape == (len(X), X.shape[2]), name


# --- rolling --------------------------------------------------------------


def test_rolling_features_are_added_per_window_and_stat(labelled_frame):
    columns = ["sensor_2", "sensor_3"]
    result = add_rolling_features(labelled_frame, columns, windows=[3, 5], stats=["mean", "std"])

    for window in (3, 5):
        for stat in ("mean", "std"):
            for column in columns:
                assert f"{column}_roll{window}_{stat}" in result.columns


def test_rolling_mean_never_mixes_two_engines(labelled_frame):
    """The first cycle of an engine must equal itself, not the previous engine."""
    result = add_rolling_features(labelled_frame, ["sensor_2"], windows=[5], stats=["mean"])

    for engine_id, group in result.groupby("engine_id"):
        group = group.sort_values("cycle")
        assert np.isclose(
            group["sensor_2_roll5_mean"].iloc[0], group["sensor_2"].iloc[0]
        ), f"engine {engine_id} leaked history from another engine"


def test_rolling_mean_matches_manual_computation(labelled_frame):
    result = add_rolling_features(labelled_frame, ["sensor_2"], windows=[4], stats=["mean"])
    single = result[result["engine_id"] == 1].sort_values("cycle")

    manual = single["sensor_2"].rolling(4, min_periods=1).mean()
    assert np.allclose(single["sensor_2_roll4_mean"], manual)


def test_expanding_features_grow_with_history(labelled_frame):
    result = add_expanding_features(labelled_frame, ["sensor_2"], stats=["mean"])
    single = result[result["engine_id"] == 1].sort_values("cycle")

    assert np.allclose(
        single["sensor_2_expanding_mean"], single["sensor_2"].expanding().mean()
    )


def test_rolling_features_validate_inputs(labelled_frame):
    with pytest.raises(ValueError, match="Missing columns"):
        add_rolling_features(labelled_frame, ["nope"], windows=[3])
    with pytest.raises(ValueError, match="Unsupported statistic"):
        add_rolling_features(labelled_frame, ["sensor_2"], windows=[3], stats=["skew"])
    with pytest.raises(ValueError, match="Window sizes must be"):
        add_rolling_features(labelled_frame, ["sensor_2"], windows=[0])


# --- lags and differences -------------------------------------------------


def test_lag_features_shift_within_each_engine(labelled_frame):
    result = add_lag_features(labelled_frame, ["sensor_2"], lags=[1, 2])
    single = result[result["engine_id"] == 1].sort_values("cycle")

    assert np.isnan(single["sensor_2_lag1"].iloc[0])
    assert np.isnan(single["sensor_2_lag2"].iloc[1])
    assert np.isclose(single["sensor_2_lag1"].iloc[1], single["sensor_2"].iloc[0])
    # One NaN per lag per engine.
    assert result["sensor_2_lag1"].isna().sum() == labelled_frame["engine_id"].nunique()


def test_diff_features_measure_change(labelled_frame):
    result = add_diff_features(labelled_frame, ["sensor_2"], periods=[1])
    single = result[result["engine_id"] == 1].sort_values("cycle")

    expected = single["sensor_2"].iloc[1] - single["sensor_2"].iloc[0]
    assert np.isclose(single["sensor_2_diff1"].iloc[1], expected)


def test_lag_features_validate_offsets(labelled_frame):
    with pytest.raises(ValueError, match="Offsets must be"):
        add_lag_features(labelled_frame, ["sensor_2"], lags=[0])


# --- combined frame -------------------------------------------------------


def test_build_tabular_frame_adds_and_reports_columns(labelled_frame, config, feature_columns):
    result, engineered = build_tabular_frame(
        labelled_frame,
        feature_columns[:3],
        config=config,
        windows=[3],
        lags=[1],
        include_diff=True,
    )

    assert engineered, "no features were engineered"
    assert all(name in result.columns for name in engineered)
    # Warm-up rows with incomplete history are dropped.
    assert not result[engineered].isna().any().any()
    assert len(result) < len(labelled_frame)


def test_build_tabular_frame_can_keep_warmup_rows(labelled_frame, config, feature_columns):
    result, engineered = build_tabular_frame(
        labelled_frame,
        feature_columns[:2],
        config=config,
        windows=[],
        lags=[2],
        include_diff=False,
        dropna=False,
    )
    assert len(result) == len(labelled_frame)
    assert result[engineered].isna().any().any()
