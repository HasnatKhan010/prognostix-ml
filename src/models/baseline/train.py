"""Train the baseline models and refresh ``baseline_results.csv``.

Reproduces the baseline notebook: naive mean, Linear Regression and Random
Forest, all evaluated on the same windows the sequence models use, so the
leaderboard compares like with like.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.config import Config, get_config, set_seed, setup_logging
from src.evaluation.compare import save_leaderboard, update_leaderboard
from src.evaluation.metrics import regression_metrics
from src.ingestion.loader import load_sequences
from src.models.baseline.naive import build_naive
from src.models.baseline.random_forest import TabularRULModel, build_tabular_estimator

logger = logging.getLogger(__name__)

__all__ = ["train_baseline", "train_baselines"]

#: Filenames match the artifacts committed by the original notebook, so a
#: retrain overwrites them instead of leaving two versions side by side.
ARTIFACT_NAMES = {
    "linear": "linear_regression_baseline.joblib",
    "random_forest": "random_forest_baseline.joblib",
}

DISPLAY_NAMES = {
    "mean": "Naive Mean",
    "linear": "Linear Regression",
    "random_forest": "Random Forest",
}


def train_baseline(
    model_name: str,
    config: Config | None = None,
    evaluate_test: bool = False,
    save: bool = True,
) -> dict[str, Any]:
    """Train one baseline and return its metrics.

    ``model_name`` is ``mean``, ``linear`` or ``random_forest``.
    """
    config = config or get_config()
    set_seed(int(config.project.seed))

    X_train, y_train = load_sequences("train", config)
    X_val, y_val = load_sequences("val", config)

    feature_columns = _feature_columns(config)
    if model_name == "mean":
        model: Any = build_naive("mean").fit(X_train, y_train)
    else:
        model = TabularRULModel(
            estimator=build_tabular_estimator(model_name, config=config),
            feature_columns=feature_columns,
            window_size=int(X_train.shape[1]),
        ).fit(X_train, y_train)

    metrics = regression_metrics(y_val, model.predict(X_val))
    logger.info(
        "%s validation: MAE %.3f | RMSE %.3f | R2 %.3f",
        DISPLAY_NAMES.get(model_name, model_name),
        metrics["MAE"],
        metrics["RMSE"],
        metrics["R2"],
    )

    test_metrics = None
    if evaluate_test:
        X_test, y_test = load_sequences("test", config)
        test_metrics = regression_metrics(y_test, model.predict(X_test))
        logger.info(
            "%s test: MAE %.3f | RMSE %.3f",
            DISPLAY_NAMES.get(model_name, model_name),
            test_metrics["MAE"],
            test_metrics["RMSE"],
        )

    path: Path | None = None
    if save and model_name in ARTIFACT_NAMES:
        path = config.path("models") / ARTIFACT_NAMES[model_name]
        model.save(path)

    if save:
        # The leaderboard is keyed by the model's config name so monitoring can
        # look a model up by the same identifier the CLI uses; the friendly name
        # is kept for baseline_results.csv.
        update_leaderboard({"Model": model_name, **metrics}, config=config)

    return {
        "model": model_name,
        "display_name": DISPLAY_NAMES.get(model_name, model_name),
        "val_metrics": metrics,
        "test_metrics": test_metrics,
        "artifact": str(path) if path else None,
        "estimator": model,
    }


def train_baselines(
    config: Config | None = None,
    models: list[str] | None = None,
    evaluate_test: bool = False,
    save: bool = True,
) -> dict[str, Any]:
    """Train every baseline and write ``baseline_results.csv``.

    Keeps the two-column ``Model,MAE,RMSE`` shape of the original file while the
    combined leaderboard carries the full metric set.
    """
    config = config or get_config()
    models = models or ["mean", "linear", "random_forest"]

    results = [
        train_baseline(name, config=config, evaluate_test=evaluate_test, save=save)
        for name in models
    ]

    if save:
        import pandas as pd

        summary = pd.DataFrame(
            [
                {
                    "Model": result["display_name"],
                    "MAE": round(result["val_metrics"]["MAE"], 4),
                    "RMSE": round(result["val_metrics"]["RMSE"], 4),
                }
                for result in results
            ]
        ).sort_values("RMSE")
        save_leaderboard(summary, config=config, filename="baseline_results.csv")

    return {result["model"]: result for result in results}


def _feature_columns(config: Config) -> list[str] | None:
    """Sensor names from the scaler bundle, when preprocessing has been run."""
    try:
        from src.preprocessing.scaling import load_scaler

        return load_scaler(config=config).feature_columns
    except Exception:
        return None


if __name__ == "__main__":  # pragma: no cover - manual entry point
    setup_logging()
    train_baselines()
