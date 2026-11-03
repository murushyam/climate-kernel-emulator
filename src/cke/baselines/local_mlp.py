"""Per-gridcell MLP: shared weights applied independently at each cell (1x1 convs) -> no spatial context."""
import torch
from torch import nn


class LocalMLP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden=(32, 32), dropout: float = 0.0):
        super().__init__()
        layers, c = [], in_channels
        for h in hidden:
            layers += [nn.Conv2d(c, h, 1), nn.ReLU(), nn.Dropout(dropout)]
            c = h
        layers.append(nn.Conv2d(c, out_channels, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):  # (B, C, H, W) -> (B, V, H, W)
        return self.net(x)
