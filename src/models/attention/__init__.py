"""Attention-based sequence model.

The trainer lives in :mod:`src.models.attention.train` and is imported on demand -
importing it here would create a cycle, since the trainer depends on the model
registry in :mod:`src.models`.
"""

from src.models.attention.model import AttentionRegressor

__all__ = ["AttentionRegressor"]
