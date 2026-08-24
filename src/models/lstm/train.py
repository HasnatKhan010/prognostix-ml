"""Train the LSTM regressor.

Run directly with ``python -m src.models.lstm.train`` or through
``python scripts/train.py --model lstm``.
"""

from __future__ import annotations

from typing import Any

from src.config import Config, setup_logging
from src.models.runner import run_sequence_training

__all__ = ["train_lstm"]


def train_lstm(config: Config | None = None, **kwargs: Any) -> dict[str, Any]:
    """Train, evaluate and checkpoint the LSTM model.

    Accepts the same overrides as
    :func:`src.models.runner.run_sequence_training` (``epochs``,
    ``batch_size``, ``learning_rate``, ``device``, ``evaluate_test``, ...).
    """
    return run_sequence_training("lstm", config=config, **kwargs)


if __name__ == "__main__":  # pragma: no cover - manual entry point
    setup_logging()
    train_lstm()
