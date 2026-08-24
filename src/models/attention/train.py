"""Train the attention regressor.

Run directly with ``python -m src.models.attention.train`` or through
``python scripts/train.py --model attention``.
"""

from __future__ import annotations

from typing import Any

from src.config import Config, setup_logging
from src.models.runner import run_sequence_training

__all__ = ["train_attention"]


def train_attention(config: Config | None = None, **kwargs: Any) -> dict[str, Any]:
    """Train, evaluate and checkpoint the attention model.

    Accepts the same overrides as
    :func:`src.models.runner.run_sequence_training` (``epochs``,
    ``batch_size``, ``learning_rate``, ``device``, ``evaluate_test``, ...).
    """
    return run_sequence_training("attention", config=config, **kwargs)


if __name__ == "__main__":  # pragma: no cover - manual entry point
    setup_logging()
    train_attention()
