"""Bidirectional LSTM for ECG arrhythmia classification."""

import torch
import torch.nn as nn


class BiLSTMModel(nn.Module):
    """
    Two-layer Bidirectional LSTM.

    Input:  (batch, n_leads, window) → transposed to (batch, window, n_leads)
    Output: (batch, n_classes) logits

    The last time-step hidden states from both directions are concatenated
    and passed through a classification head.
    """

    def __init__(self, n_leads: int = 2, hidden_size: int = 128,
                 num_layers: int = 2, n_classes: int = 5,
                 dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_leads,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_size * 2)
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_leads, window) → (batch, window, n_leads)
        x = x.permute(0, 2, 1)
        out, (h_n, _) = self.lstm(x)

        # concatenate last forward and backward hidden states
        h_fwd = h_n[-2]   # (batch, hidden_size)
        h_bwd = h_n[-1]   # (batch, hidden_size)
        h = torch.cat([h_fwd, h_bwd], dim=-1)  # (batch, hidden_size*2)
        h = self.norm(h)
        return self.head(h)
