"""GRU sequence model.

The trainer lives in :mod:`src.models.gru.train` and is imported on demand -
importing it here would create a cycle, since the trainer depends on the model
registry in :mod:`src.models`.
"""

from src.models.gru.model import GRURegressor

__all__ = ["GRURegressor"]
