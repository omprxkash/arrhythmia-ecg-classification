"""CNN-Transformer hybrid model for ECG arrhythmia classification."""

import math
import torch
import torch.nn as nn


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        return x + self.pe(positions)


class CNNTransformer(nn.Module):
    """
    CNN feature extractor followed by a Transformer encoder.

    Pipeline:
      CNN stem → sequence of feature vectors → positional encoding
      → Transformer encoder → mean pooling → classification head

    Input:  (batch, 2, 360)
    Output: (batch, 5) logits
    """

    def __init__(self, n_leads: int = 2, n_classes: int = 5,
                 d_model: int = 128, nhead: int = 4,
                 num_layers: int = 3, dim_feedforward: int = 256,
                 dropout: float = 0.1, max_seq_len: int = 128):
        super().__init__()

        # CNN stem: extracts local temporal features
        self.cnn_stem = nn.Sequential(
            nn.Conv1d(n_leads, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(64, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True),
        )

        self.pos_enc = LearnedPositionalEncoding(d_model, max_len=max_seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_leads, 360)
        feat = self.cnn_stem(x)              # (batch, d_model, T)
        feat = feat.permute(0, 2, 1)         # (batch, T, d_model)
        feat = self.pos_enc(feat)
        feat = self.transformer(feat)
        feat = feat.mean(dim=1)              # global average over time
        return self.head(feat)
