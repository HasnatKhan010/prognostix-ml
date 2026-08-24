"""LSTM encoder with attention pooling over the sensor window.

Reading only the final timestep, as the plain LSTM/GRU regressors do, forces the
recurrent state to carry every earlier cycle through a single vector. Attention
removes that bottleneck twice over:

1. **Self-attention** lets each cycle look directly at every other cycle.
2. **Additive attention pooling** replaces "take the last hidden state" with a
   learned weighted average, and those weights are readable - they show which
   cycles in the window drove a prediction, which is the difference between a
   number a maintenance planner can act on and one they have to trust blindly.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["AttentionRegressor"]


class AttentionRegressor(nn.Module):
    """LSTM encoder + multi-head self-attention + attention-pooled regression head.

    Parameters
    ----------
    input_size:
        Number of sensor channels per cycle.
    hidden_size:
        Encoder width. Must be divisible by ``num_heads``.
    num_layers:
        Stacked LSTM layers in the encoder.
    dropout:
        Applied between encoder layers, inside attention and in the head.
    num_heads:
        Self-attention heads; each can specialise on a different degradation
        pattern.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        num_heads: int = 4,
    ):
        super().__init__()
        if input_size < 1:
            raise ValueError(f"input_size must be >= 1, got {input_size}")
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        if hidden_size % num_heads:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by "
                f"num_heads ({num_heads})"
            )

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_heads = num_heads

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_size)

        # Additive (Bahdanau-style) scorer producing one weight per timestep.
        self.pool_score = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Map ``(batch, window, input_size)`` to ``(batch,)`` RUL estimates.

        Parameters
        ----------
        return_attention:
            Also return the pooling weights, shape ``(batch, window)``, which sum
            to 1 across the window.
        """
        encoded, _ = self.lstm(x)

        attended, _ = self.self_attention(encoded, encoded, encoded)
        # Residual connection keeps the raw recurrent signal available.
        encoded = self.norm(encoded + attended)

        weights = torch.softmax(self.pool_score(encoded).squeeze(-1), dim=1)
        pooled = torch.bmm(weights.unsqueeze(1), encoded).squeeze(1)

        prediction = self.head(pooled).squeeze(1)
        return (prediction, weights) if return_attention else prediction

    @torch.no_grad()
    def attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Pooling weights for a batch, for explaining a prediction."""
        self.eval()
        _, weights = self.forward(x, return_attention=True)
        return weights
