"""End-to-end training run for the sequence models.

The three architectures share one runner so a comparison between them differs
only by architecture, never by training protocol: same splits, same seed, same
optimiser settings, same metrics, same artifacts.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from src.config import Config, get_config, get_device, set_seed
from src.evaluation.compare import update_leaderboard
from src.evaluation.metrics import regression_metrics
from src.evaluation.plots import (
    plot_actual_vs_predicted,
    plot_residuals,
    plot_training_history,
)
from src.ingestion.loader import load_sequences
from src.ingestion.validator import validate_sequences
from src.models import build_model
from src.models.common import fit, make_loaders, save_checkpoint, validate

logger = logging.getLogger(__name__)

__all__ = ["checkpoint_path_for", "run_sequence_training"]


def checkpoint_path_for(model_name: str, config: Config | None = None) -> Path:
    """Canonical checkpoint location for a model name."""
    config = config or get_config()
    return config.path("models") / f"{model_name}.pt"


def run_sequence_training(
    model_name: str,
    config: Config | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    device: str | None = None,
    evaluate_test: bool = False,
    save: bool = True,
    make_plots: bool = True,
    model_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train one sequence model and persist its checkpoint, metrics and figures.

    Parameters
    ----------
    evaluate_test:
        Also score the held-out test split. Left off by default so the test set
        stays untouched during architecture selection.

    Returns
    -------
    A summary dict with validation metrics, the training history and the
    checkpoint path.
    """
    config = config or get_config()
    training = config.training
    set_seed(int(config.project.seed))

    resolved_device = get_device(device or training.get("device", "auto"))
    epochs = int(epochs or training.epochs)
    batch_size = int(batch_size or training.batch_size)
    learning_rate = float(learning_rate or training.learning_rate)

    # --- data ---------------------------------------------------------
    X_train, y_train = load_sequences("train", config)
    X_val, y_val = load_sequences("val", config)
    validate_sequences(X_train, y_train, config=config).raise_for_errors()
    validate_sequences(X_val, y_val, config=config).raise_for_errors()

    splits = {"train": (X_train, y_train), "val": (X_val, y_val)}
    if evaluate_test:
        splits["test"] = load_sequences("test", config)
    loaders = make_loaders(
        splits,
        batch_size=batch_size,
        num_workers=int(training.get("num_workers", 0)),
    )

    # --- model --------------------------------------------------------
    model_kwargs = dict(config.models.get(model_name, {}) or {})
    model_kwargs.update(model_overrides or {})
    input_size = int(X_train.shape[2])
    model = build_model(model_name, input_size=input_size, **model_kwargs)
    logger.info(
        "Training %s on %s | %d train / %d val windows of shape (%d, %d)",
        model_name,
        resolved_device,
        len(X_train),
        len(X_val),
        X_train.shape[1],
        input_size,
    )

    # --- fit ----------------------------------------------------------
    result = fit(
        model,
        loaders["train"],
        loaders["val"],
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=float(training.get("weight_decay", 0.0)),
        device=resolved_device,
        grad_clip=training.get("grad_clip"),
        early_stopping_patience=training.get("early_stopping_patience"),
        min_delta=float(training.get("min_delta", 0.0)),
        lr_scheduler=str(training.get("lr_scheduler", "plateau")),
        lr_patience=int(training.get("lr_patience", 3)),
        lr_factor=float(training.get("lr_factor", 0.5)),
    )

    # --- evaluate -----------------------------------------------------
    import torch.nn as nn

    criterion = nn.MSELoss()
    _, val_predictions, val_targets = validate(
        result.model, loaders["val"], criterion, resolved_device
    )
    val_metrics = regression_metrics(val_targets, val_predictions)
    logger.info(
        "%s validation: MAE %.3f | RMSE %.3f | R2 %.3f",
        model_name,
        val_metrics["MAE"],
        val_metrics["RMSE"],
        val_metrics["R2"],
    )

    test_metrics: dict[str, float] | None = None
    if evaluate_test:
        _, test_predictions, test_targets = validate(
            result.model, loaders["test"], criterion, resolved_device
        )
        test_metrics = regression_metrics(test_targets, test_predictions)
        logger.info(
            "%s test: MAE %.3f | RMSE %.3f",
            model_name,
            test_metrics["MAE"],
            test_metrics["RMSE"],
        )

    # --- persist ------------------------------------------------------
    checkpoint = None
    if save:
        checkpoint = save_checkpoint(
            result.model,
            checkpoint_path_for(model_name, config),
            model_type=model_name,
            input_size=input_size,
            model_kwargs=model_kwargs,
            feature_columns=_feature_columns(config),
            window_size=int(X_train.shape[1]),
            metrics={"val": val_metrics, **({"test": test_metrics} if test_metrics else {})},
            history=result.history,
            extra={
                "epochs_run": result.epochs_run,
                "best_epoch": result.best_epoch,
                "n_parameters": result.n_parameters,
                "rul_cap": config.data.get("rul_cap"),
            },
        )
        result.checkpoint_path = checkpoint

        update_leaderboard(
            {
                "Model": model_name,
                **val_metrics,
                "Params": result.n_parameters,
                "Epochs": result.epochs_run,
                "TrainSeconds": round(result.duration_seconds, 1),
            },
            config=config,
        )

    if make_plots:
        _write_figures(model_name, result.history, val_targets, val_predictions, config)

    return {
        "model": model_name,
        "input_size": input_size,
        "window_size": int(X_train.shape[1]),
        "device": str(resolved_device),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "history": result.history,
        "best_epoch": result.best_epoch,
        "best_val_loss": result.best_val_loss,
        "epochs_run": result.epochs_run,
        "stopped_early": result.stopped_early,
        "n_parameters": result.n_parameters,
        "duration_seconds": result.duration_seconds,
        "checkpoint": str(checkpoint) if checkpoint else None,
    }


def _feature_columns(config: Config) -> list[str] | None:
    """Feature order recorded by the preprocessing run, when available."""
    try:
        from src.preprocessing.scaling import load_scaler

        return load_scaler(config=config).feature_columns
    except Exception:  # a checkpoint is still usable without this metadata
        logger.debug("No scaler bundle found; checkpoint will omit feature names")
        return None


def _write_figures(
    model_name: str,
    history: dict[str, list[float]],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    config: Config,
) -> None:
    """Save the standard diagnostic figures for a run."""
    import matplotlib.pyplot as plt

    figures = config.path("figures")
    label = model_name.upper()
    try:
        plot_training_history(
            {k: v for k, v in history.items() if k != "lr"},
            title=f"{label} training history",
            save_path=figures / f"{model_name}_training_history.png",
        )
        plot_actual_vs_predicted(
            y_true,
            y_pred,
            title=f"{label} - actual vs predicted RUL",
            save_path=figures / f"{model_name}_actual_vs_predicted.png",
        )
        plot_residuals(
            y_true,
            y_pred,
            title=f"{label} residuals",
            save_path=figures / f"{model_name}_residuals.png",
        )
    finally:
        plt.close("all")
