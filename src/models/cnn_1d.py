"""1D CNN baseline for ECG arrhythmia classification."""

import torch
import torch.nn as nn


class CNN1D(nn.Module):
    """
    3-block 1D convolutional network.
    Input:  (batch, 2, 360)
    Output: (batch, 5) logits
    """

    def __init__(self, n_leads: int = 2, n_classes: int = 5,
                 dropout: float = 0.5):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(n_leads, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            # Block 2
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            # Block 3
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            # Block 4
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)
