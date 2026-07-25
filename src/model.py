"""The neural network: a small residual CNN policy that predicts the next move.

Policy-only (no value head) on purpose — predicting moves is all the demo needs,
and it keeps training and inference simple and fast on CPU.

Input : (batch, 17, 8, 8)
Output: (batch, 4096) move logits
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoding import ACTION_SIZE, NUM_PLANES


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class ChessPolicyNet(nn.Module):
    """ResNet policy. Defaults (128 channels, 10 blocks, ~4-5M params) are the
    sweet spot: strong enough when trained on lots of GM data, yet one forward
    pass takes ~0.1-0.3s on CPU, so inference stays snappy."""

    def __init__(self, channels: int = 128, num_blocks: int = 10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(NUM_PLANES, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_blocks)]
        )
        # Policy head: reduce to 2 planes, then a linear layer to 4096 logits.
        self.policy_conv = nn.Conv2d(channels, 2, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 8 * 8, ACTION_SIZE)

        # Remember the hyper-parameters so checkpoints are self-describing.
        self.config = {"channels": channels, "num_blocks": num_blocks}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.flatten(start_dim=1)
        return self.policy_fc(p)
