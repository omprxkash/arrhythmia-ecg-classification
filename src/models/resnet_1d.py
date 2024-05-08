"""ResNet-style 1D convolutional network for ECG arrhythmia classification."""

import torch
import torch.nn as nn


class ResidualBlock1D(nn.Module):
    """
    Pre-activation residual block for 1D signals.
    Supports optional channel expansion and temporal downsampling via stride.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 stride: int = 1, dropout: float = 0.1):
        super().__init__()
        pad = kernel_size // 2

        self.block = nn.Sequential(
            nn.BatchNorm1d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=pad, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel_size, stride=1, padding=pad, bias=False),
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Conv1d(in_ch, out_ch, kernel_size=1,
                                      stride=stride, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + self.shortcut(x)


class ResNet1D(nn.Module):
    """
    ResNet-1D for ECG beat classification.

    Architecture:
      Stem:   Conv(2→64, k=7, stride=2) → BN → ReLU → MaxPool
      Layer1: 2× ResBlock(64→64)
      Layer2: 2× ResBlock(64→128, stride=2)
      Layer3: 2× ResBlock(128→256, stride=2)
      Layer4: 2× ResBlock(256→512, stride=2)
      Head:   AdaptiveAvgPool → Linear(512→n_classes)

    Input:  (batch, 2, 360)
    Output: (batch, 5) logits
    """

    def __init__(self, n_leads: int = 2, n_classes: int = 5,
                 dropout: float = 0.2):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(n_leads, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = self._make_layer(64,  64,  blocks=2, stride=1, dropout=dropout)
        self.layer2 = self._make_layer(64,  128, blocks=2, stride=2, dropout=dropout)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2, dropout=dropout)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2, dropout=dropout)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(512, n_classes),
        )

    def _make_layer(self, in_ch, out_ch, blocks, stride, dropout):
        layers = [ResidualBlock1D(in_ch, out_ch, stride=stride, dropout=dropout)]
        for _ in range(1, blocks):
            layers.append(ResidualBlock1D(out_ch, out_ch, dropout=dropout))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.head(x)

    def get_last_conv_layer(self):
        """Return the last conv layer for Grad-CAM."""
        return self.layer4[-1].block[-1]
