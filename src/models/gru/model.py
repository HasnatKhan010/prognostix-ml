"""GRU regressor for Remaining Useful Life."""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["GRURegressor"]


class GRURegressor(nn.Module):
    """Stacked GRU that maps a sensor window to a single RUL value.

    A GRU has two gates instead of the LSTM's three, so it trains faster with
    roughly 25% fewer parameters - which on CMAPSS costs little accuracy and is
    why this is the default served model. Layer names (``gru``, ``fc``) match the
    committed ``gru_baseline.pt`` checkpoint.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ):
        super().__init__()
        if input_size < 1:
            raise ValueError(f"input_size must be >= 1, got {input_size}")
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.fc = nn.Linear(hidden_size * (2 if bidirectional else 1), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(batch, window, input_size)`` to ``(batch,)`` RUL estimates."""
        output, _hidden = self.gru(x)

        last_output = output[:, -1, :]

        prediction = self.fc(last_output)
        return prediction.squeeze(1)
