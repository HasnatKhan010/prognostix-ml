"""LSTM regressor for Remaining Useful Life."""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["LSTMRegressor"]


class LSTMRegressor(nn.Module):
    """Stacked LSTM that maps a sensor window to a single RUL value.

    The prediction is read from the final timestep's hidden state, which by
    construction has seen the whole window. Layer names (``lstm``, ``fc``) match
    the checkpoints produced by the original notebooks so older weights load
    without translation.

    Parameters
    ----------
    input_size:
        Number of sensor channels per cycle.
    hidden_size:
        Units per LSTM layer.
    num_layers:
        Stacked LSTM layers. ``dropout`` is ignored when this is 1, since PyTorch
        applies dropout only *between* layers.
    bidirectional:
        Reading the window backwards as well doubles the representation width.
        Legitimate here because a window is a completed history, not a live
        stream.
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

        self.lstm = nn.LSTM(
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
        output, (_hidden, _cell) = self.lstm(x)

        # Representation from the final timestep.
        last_output = output[:, -1, :]

        prediction = self.fc(last_output)
        return prediction.squeeze(1)
