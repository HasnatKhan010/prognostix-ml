"""Tests for the model zoo, the shared trainer, metrics and the leaderboard."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.evaluation.compare import build_leaderboard, load_leaderboard, update_leaderboard
from src.evaluation.metrics import (
    evaluate_model,
    mae,
    mape,
    nasa_score,
    regression_metrics,
    rmse,
    within_tolerance,
)
from src.models import MODEL_NAMES, TORCH_MODELS, build_model, is_torch_model
from src.models.baseline.naive import MeanBaseline, MedianBaseline, build_naive
from src.models.baseline.random_forest import (
    TabularRULModel,
    build_linear_regression,
    build_random_forest,
)
from src.models.common import (
    EarlyStopping,
    count_parameters,
    fit,
    load_checkpoint,
    make_loaders,
    predict,
    save_checkpoint,
)
from tests.conftest import WINDOW

BATCH, FEATURES = 8, 15


def _batch(window: int = WINDOW, features: int = FEATURES) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(BATCH, window, features)


# --- architectures --------------------------------------------------------


@pytest.mark.parametrize("name", sorted(TORCH_MODELS))
def test_sequence_models_return_one_value_per_window(name):
    model = build_model(name, input_size=FEATURES, hidden_size=32, num_layers=1, dropout=0.0)
    output = model(_batch())

    assert output.shape == (BATCH,)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("name", sorted(TORCH_MODELS))
def test_sequence_models_are_window_length_agnostic(name):
    """A trained model must tolerate a longer window at inference time."""
    model = build_model(name, input_size=FEATURES, hidden_size=32, num_layers=1, dropout=0.0)
    assert model(_batch(window=WINDOW * 2)).shape == (BATCH,)


@pytest.mark.parametrize("name", sorted(TORCH_MODELS))
def test_sequence_models_are_trainable(name):
    model = build_model(name, input_size=FEATURES, hidden_size=16, num_layers=1, dropout=0.0)
    loss = model(_batch()).sum()
    loss.backward()

    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_single_layer_model_does_not_apply_inter_layer_dropout():
    """PyTorch warns when dropout is set with one layer; it must be zeroed."""
    model = build_model("gru", input_size=FEATURES, num_layers=1, dropout=0.3)
    assert model.gru.dropout == 0.0


def test_bidirectional_widens_the_head():
    model = build_model("lstm", input_size=FEATURES, hidden_size=32, bidirectional=True)
    assert model.fc.in_features == 64
    assert model(_batch()).shape == (BATCH,)


def test_attention_weights_form_a_distribution_over_the_window():
    model = build_model("attention", input_size=FEATURES, hidden_size=32, num_layers=1, num_heads=4)
    prediction, weights = model(_batch(), return_attention=True)

    assert prediction.shape == (BATCH,)
    assert weights.shape == (BATCH, WINDOW)
    assert torch.allclose(weights.sum(dim=1), torch.ones(BATCH), atol=1e-5)
    assert (weights >= 0).all()


def test_attention_rejects_indivisible_head_count():
    with pytest.raises(ValueError, match="divisible"):
        build_model("attention", input_size=FEATURES, hidden_size=30, num_heads=4)


def test_build_model_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unknown sequence model"):
        build_model("transformer", input_size=FEATURES)


def test_model_registry_contents():
    assert set(TORCH_MODELS) == {"lstm", "gru", "attention"}
    assert is_torch_model("gru") and not is_torch_model("random_forest")
    assert "random_forest" in MODEL_NAMES


def test_invalid_input_size_is_rejected():
    with pytest.raises(ValueError, match="input_size"):
        build_model("gru", input_size=0)


# --- trainer --------------------------------------------------------------


def test_fit_reduces_training_loss_on_a_learnable_signal():
    """The target is the last cycle's first feature, so loss must drop."""
    torch.manual_seed(0)
    X = np.random.default_rng(0).normal(size=(160, WINDOW, 4)).astype("float32")
    y = X[:, -1, 0] * 3.0

    model = build_model("gru", input_size=4, hidden_size=16, num_layers=1, dropout=0.0)
    loaders = make_loaders({"train": (X[:120], y[:120]), "val": (X[120:], y[120:])}, batch_size=16)

    result = fit(
        model,
        loaders["train"],
        loaders["val"],
        epochs=6,
        learning_rate=0.02,
        early_stopping_patience=None,
        verbose=False,
    )

    assert result.history["train_loss"][-1] < result.history["train_loss"][0]
    assert result.epochs_run == 6
    assert result.n_parameters == count_parameters(result.model)


def test_fit_stops_early_and_restores_the_best_weights():
    """Validation targets contradict training, so val loss can only worsen."""
    rng = np.random.default_rng(1)
    X_train = rng.normal(size=(64, WINDOW, 3)).astype("float32")
    y_train = (X_train[:, -1, 0] * 3.0).astype("float32")

    X_val = rng.normal(size=(32, WINDOW, 3)).astype("float32")
    y_val = (-X_val[:, -1, 0] * 3.0).astype("float32")  # inverted relationship

    model = build_model("lstm", input_size=3, hidden_size=16, num_layers=1, dropout=0.0)
    loaders = make_loaders({"train": (X_train, y_train), "val": (X_val, y_val)}, batch_size=16)

    result = fit(
        model,
        loaders["train"],
        loaders["val"],
        epochs=40,
        learning_rate=0.02,
        early_stopping_patience=2,
        verbose=False,
    )

    assert result.epochs_run < 40
    assert result.stopped_early
    assert result.best_epoch <= result.epochs_run


def test_early_stopping_tracks_and_restores_the_best_epoch():
    model = build_model("gru", input_size=2, hidden_size=4, num_layers=1, dropout=0.0)
    stopper = EarlyStopping(patience=2, min_delta=0.0)

    assert stopper.step(1.0, 1, model) is True
    assert stopper.step(0.5, 2, model) is True
    assert stopper.step(0.6, 3, model) is False
    assert not stopper.should_stop
    assert stopper.step(0.7, 4, model) is False
    assert stopper.should_stop
    assert stopper.best_epoch == 2 and stopper.best_loss == 0.5

    stopper.restore(model)  # must not raise


def test_early_stopping_rejects_zero_patience():
    with pytest.raises(ValueError, match="patience"):
        EarlyStopping(patience=0)


def test_make_loaders_shuffles_only_training_data():
    X = np.zeros((10, WINDOW, 2), dtype="float32")
    y = np.arange(10, dtype="float32")
    loaders = make_loaders({"train": (X, y), "val": (X, y)}, batch_size=4)

    assert loaders["train"].batch_size == 4
    assert loaders["val"].sampler.__class__.__name__ == "SequentialSampler"


def test_predict_accepts_a_single_window():
    model = build_model("gru", input_size=FEATURES, hidden_size=8, num_layers=1, dropout=0.0)
    single = predict(model, np.zeros((WINDOW, FEATURES)))
    batched = predict(model, np.zeros((3, WINDOW, FEATURES)))

    assert single.shape == (1,)
    assert batched.shape == (3,)


# --- checkpoints ----------------------------------------------------------


def test_checkpoint_roundtrip_reproduces_predictions(tmp_path):
    model = build_model("gru", input_size=FEATURES, hidden_size=16, num_layers=2, dropout=0.1)
    batch = _batch()
    model.eval()
    with torch.no_grad():
        before = model(batch).numpy()

    path = save_checkpoint(
        model,
        tmp_path / "gru.pt",
        model_type="gru",
        input_size=FEATURES,
        model_kwargs={"hidden_size": 16, "num_layers": 2, "dropout": 0.1},
        feature_columns=[f"sensor_{i}" for i in range(FEATURES)],
        window_size=WINDOW,
        metrics={"val": {"RMSE": 25.0}},
    )

    payload = load_checkpoint(path)
    assert payload["model_type"] == "gru"
    assert payload["window_size"] == WINDOW
    assert payload["hidden_size"] == 16  # flat copy for notebook compatibility
    assert payload["model_kwargs"]["dropout"] == 0.1

    restored = build_model("gru", input_size=payload["input_size"], **payload["model_kwargs"])
    restored.load_state_dict(payload["model_state_dict"])
    restored.eval()
    with torch.no_grad():
        after = restored(batch).numpy()

    assert np.allclose(before, after)


def test_load_checkpoint_reports_a_missing_file_actionably(tmp_path):
    with pytest.raises(FileNotFoundError, match="Train the model first"):
        load_checkpoint(tmp_path / "absent.pt")


def test_load_checkpoint_rejects_a_foreign_file(tmp_path):
    path = tmp_path / "junk.pt"
    torch.save({"weights": 1}, path)
    with pytest.raises(ValueError, match="not a Prognostix checkpoint"):
        load_checkpoint(path)


# --- baselines ------------------------------------------------------------


def test_mean_baseline_predicts_the_training_mean(sequences):
    X, y = sequences
    model = MeanBaseline().fit(X, y)

    predictions = model.predict(X)
    assert predictions.shape == (len(X),)
    assert np.allclose(predictions, y.mean())


def test_median_baseline_is_robust_to_the_rul_tail():
    y = np.array([1.0, 2.0, 3.0, 400.0])
    assert MedianBaseline().fit(None, y).predict(np.zeros((4, 1))) [0] == 2.5


def test_quantile_baseline_is_conservative(sequences):
    X, y = sequences
    conservative = build_naive("quantile", quantile=0.1).fit(X, y)
    assert conservative.predict(X)[0] < y.mean()


def test_naive_baseline_rejects_bad_configuration():
    with pytest.raises(ValueError, match="Unknown strategy"):
        build_naive("magic")
    with pytest.raises(ValueError, match="quantile must be"):
        build_naive("quantile", quantile=1.5)
    with pytest.raises(RuntimeError, match="not fitted"):
        MeanBaseline().predict(np.zeros((2, 1)))


def test_tabular_model_flattens_then_fits(sequences, feature_columns):
    X, y = sequences
    model = TabularRULModel(
        estimator=build_linear_regression(), feature_columns=feature_columns
    ).fit(X, y)

    assert model.window_size == WINDOW
    predictions = model.predict(X)
    assert predictions.shape == (len(X),)
    assert np.isfinite(predictions).all()
    # A single window is accepted too.
    assert model.predict(X[0]).shape == (1,)


def test_tabular_model_exposes_named_importances(sequences, feature_columns, config):
    X, y = sequences
    model = TabularRULModel(
        estimator=build_random_forest(config, n_estimators=5),
        feature_columns=feature_columns,
    ).fit(X, y)

    importances = model.feature_importances()
    assert importances is not None
    assert len(importances) == len(feature_columns) * 6
    assert all(name.rsplit("_", 1)[0] in feature_columns for name in importances)


def test_tabular_model_roundtrip(tmp_path, sequences, feature_columns):
    X, y = sequences
    model = TabularRULModel(
        estimator=build_linear_regression(), feature_columns=feature_columns
    ).fit(X, y)

    path = model.save(tmp_path / "linear.joblib")
    restored = TabularRULModel.load(path)

    assert restored.feature_columns == feature_columns
    assert np.allclose(restored.predict(X), model.predict(X))


def test_tabular_model_adopts_a_bare_estimator(tmp_path, sequences, feature_columns):
    """The committed *_baseline.joblib files hold raw sklearn estimators."""
    import joblib

    from src.features.engineering import create_statistical_features

    X, y = sequences
    estimator = build_linear_regression().fit(create_statistical_features(X), y)
    path = tmp_path / "bare.joblib"
    joblib.dump(estimator, path)

    wrapped = TabularRULModel.load(path, feature_columns=feature_columns)
    assert wrapped.metadata["source"] == "bare_estimator"
    assert np.allclose(wrapped.predict(X), estimator.predict(create_statistical_features(X)))


def test_tabular_model_rejects_a_non_estimator(tmp_path):
    import joblib

    path = tmp_path / "not_a_model.joblib"
    joblib.dump({"hello": "world"}, path)
    with pytest.raises(TypeError, match="no predict"):
        TabularRULModel.load(path)


# --- metrics --------------------------------------------------------------


def test_metrics_match_hand_computed_values():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 33.0])

    assert mae(y_true, y_pred) == pytest.approx((2 + 2 + 3) / 3)
    assert rmse(y_true, y_pred) == pytest.approx(np.sqrt((4 + 4 + 9) / 3))
    assert within_tolerance(y_true, y_pred, tolerance=2.0) == pytest.approx(2 / 3)


def test_mape_survives_zero_rul():
    """RUL legitimately reaches zero, so the denominator must be floored."""
    value = mape(np.array([0.0, 10.0]), np.array([1.0, 11.0]))
    assert np.isfinite(value)


def test_nasa_score_penalises_late_predictions_harder():
    """Over-estimating remaining life means the machine fails in service."""
    y_true = np.array([50.0])
    late = nasa_score(y_true, np.array([70.0]))     # predicted more life than real
    early = nasa_score(y_true, np.array([30.0]))    # conservative

    assert late > early > 0
    assert nasa_score(y_true, y_true) == pytest.approx(0.0)


def test_nasa_score_does_not_overflow_on_extreme_errors():
    assert np.isfinite(nasa_score(np.array([0.0]), np.array([1e6])))


def test_regression_metrics_reports_the_full_set():
    y_true = np.arange(1, 21, dtype=float)
    metrics = regression_metrics(y_true, y_true + 1.0)

    for key in ("MAE", "RMSE", "R2", "MAPE", "Within10", "Bias", "MaxError", "N", "NASAScore"):
        assert key in metrics
    assert metrics["N"] == 20
    assert metrics["Bias"] == pytest.approx(1.0)


def test_metrics_reject_mismatched_or_empty_inputs():
    with pytest.raises(ValueError, match="Shape mismatch"):
        mae([1.0, 2.0], [1.0])
    with pytest.raises(ValueError, match="empty"):
        rmse([], [])


def test_evaluate_model_produces_a_leaderboard_row():
    row = evaluate_model("GRU", [10.0, 20.0], [11.0, 19.0])
    assert row["Model"] == "GRU" and "RMSE" in row


# --- leaderboard ----------------------------------------------------------


def test_build_leaderboard_sorts_by_rmse():
    frame = build_leaderboard(
        [
            {"Model": "worse", "MAE": 30.0, "RMSE": 40.0},
            {"Model": "better", "MAE": 10.0, "RMSE": 15.0},
        ]
    )
    assert list(frame["Model"]) == ["better", "worse"]


def test_update_leaderboard_replaces_rather_than_duplicates(config):
    update_leaderboard({"Model": "GRU", "MAE": 20.0, "RMSE": 30.0}, config=config)
    merged = update_leaderboard({"Model": "GRU", "MAE": 15.0, "RMSE": 22.0}, config=config)

    assert len(merged[merged["Model"] == "GRU"]) == 1
    assert merged.loc[merged["Model"] == "GRU", "RMSE"].iloc[0] == 22.0

    reloaded = load_leaderboard(config=config)
    assert reloaded.loc[reloaded["Model"] == "GRU", "MAE"].iloc[0] == 15.0


def test_update_leaderboard_keeps_other_models(config):
    update_leaderboard({"Model": "LSTM", "MAE": 20.0, "RMSE": 30.0}, config=config)
    merged = update_leaderboard({"Model": "GRU", "MAE": 15.0, "RMSE": 22.0}, config=config)
    assert set(merged["Model"]) == {"LSTM", "GRU"}


def test_update_leaderboard_requires_a_model_column(config):
    with pytest.raises(ValueError, match="Model"):
        update_leaderboard({"MAE": 1.0}, config=config)


def test_load_leaderboard_is_empty_when_absent(config):
    assert load_leaderboard(config=config).empty
