"""Deep learning architectures for cryptocurrency volatility forecasting in PyTorch.

Implements:
- CryptoLSTM: Multi-layer LSTM with dropout and positive volatility projection head.
- CryptoGRU: Gated Recurrent Unit with residual projection head.
- CryptoMultimodalAttention: Temporal attention mechanism over multimodal representations.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CryptoLSTM(nn.Module):
    """Multi-layer LSTM forecaster with Softplus projection head for non-negative volatility."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        fc_in_dim = hidden_size * self.num_directions
        self.head = nn.Sequential(
            nn.Linear(fc_in_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: Tensor of shape (batch_size, seq_len, input_size).

        Returns:
            Tensor of shape (batch_size, 1) representing forecast volatility.
        """
        # lstm_out: (batch_size, seq_len, hidden_size * num_directions)
        lstm_out, _ = self.lstm(x)

        # Take representation from the final time step
        last_step = lstm_out[:, -1, :]
        raw_output = self.head(last_step)

        # Softplus ensures predicted volatility is strictly non-negative
        return F.softplus(raw_output) + 1e-6


class CryptoGRU(nn.Module):
    """Multi-layer GRU forecaster with dropout and positive projection head."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: Tensor of shape (batch_size, seq_len, input_size).

        Returns:
            Tensor of shape (batch_size, 1).
        """
        gru_out, _ = self.gru(x)
        last_step = gru_out[:, -1, :]
        raw_output = self.head(last_step)
        return F.softplus(raw_output) + 1e-6


class CryptoMultimodalAttention(nn.Module):
    """Temporal attention mechanism over sequential multimodal market and NLP features."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_heads: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.input_proj = nn.Linear(input_size, hidden_size)
        self.positional_encoding = nn.Parameter(torch.randn(1, 100, hidden_size) * 0.02)

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_size)

        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: Tensor of shape (batch_size, seq_len, input_size).

        Returns:
            Tensor of shape (batch_size, 1).
        """
        batch_size, seq_len, _ = x.shape
        proj = self.input_proj(x)
        # Add positional encoding up to seq_len
        proj = proj + self.positional_encoding[:, :seq_len, :]

        attn_out, _ = self.attention(proj, proj, proj)
        pooled = self.norm(attn_out + proj).mean(dim=1)

        raw_output = self.head(pooled)
        return F.softplus(raw_output) + 1e-6
