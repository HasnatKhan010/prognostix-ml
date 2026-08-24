"""Model zoo: naive and tabular baselines plus three sequence architectures.

``build_model`` is the single entry point used by the training scripts and the
inference layer, so a model name in a config file or CLI flag resolves the same
way everywhere.
"""

from __future__ import annotations

from typing import Any

from src.models.attention.model import AttentionRegressor
from src.models.gru.model import GRURegressor
from src.models.lstm.model import LSTMRegressor

#: Sequence models, keyed by the name used in configs, CLI flags and checkpoints.
TORCH_MODELS: dict[str, type] = {
    "lstm": LSTMRegressor,
    "gru": GRURegressor,
    "attention": AttentionRegressor,
}

#: Tabular / naive models trained through scikit-learn.
SKLEARN_MODELS: tuple[str, ...] = ("linear", "random_forest", "mean")

#: Every model name the training script accepts.
MODEL_NAMES: tuple[str, ...] = (*SKLEARN_MODELS, *TORCH_MODELS)

__all__ = [
    "AttentionRegressor",
    "GRURegressor",
    "LSTMRegressor",
    "MODEL_NAMES",
    "SKLEARN_MODELS",
    "TORCH_MODELS",
    "build_model",
    "is_torch_model",
]


def is_torch_model(name: str) -> bool:
    """True when ``name`` refers to a PyTorch sequence model."""
    return name in TORCH_MODELS


def build_model(name: str, input_size: int, **kwargs: Any):
    """Instantiate a sequence model by name.

    Unknown keyword arguments are rejected by the model constructors, which
    keeps a stale checkpoint from silently training the wrong architecture.
    """
    try:
        model_class = TORCH_MODELS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown sequence model {name!r}; choose from {sorted(TORCH_MODELS)}"
        ) from exc
    return model_class(input_size=input_size, **kwargs)
