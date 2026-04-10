"""
Value-only ANIL model for MAML on abstract strategy games.

Architecture:
  Trunk (frozen in inner loop):
    4x Conv3x3+ReLU, 64 filters → bottleneck Linear(5184, 64)

  Value head (adapted in inner loop):
    Linear(64, 64) + ReLU → Linear(64, 1)
    Total adapted params: 64*64 + 64 + 64 + 1 = 4,225

The bottleneck compresses the trunk's spatial features into a compact
game-relevant representation. The value head learns to read it for the
current task. With ANIL, only the value head is updated in the inner loop.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """3x3 Conv + ReLU, no batch norm (MAML-friendly)."""

    def __init__(self, in_ch: int, out_ch: int = 64):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        return F.relu(self.conv(x), inplace=True)


class ValueNet(nn.Module):
    """
    Value-only network for ANIL-style MAML.

    Parameters
    ----------
    in_channels : int
        Number of input feature planes (e.g. 47 for the unified chess/shogi spec).
    trunk_hidden : int
        Number of filters per conv block.
    n_conv_blocks : int
        Number of stacked conv blocks in the trunk.
    bottleneck_dim : int
        Dimension of the bottleneck between trunk and value head.
        Controls the information bandwidth available to the inner loop.
    value_hidden : int
        Hidden dimension in the value head (the adapted part).
    """

    # Names of parameter groups for ANIL inner-loop filtering
    TRUNK_PREFIX = "trunk."
    BOTTLENECK_PREFIX = "bottleneck."
    VALUE_PREFIX = "value_head."

    def __init__(
        self,
        in_channels: int,
        trunk_hidden: int = 64,
        n_conv_blocks: int = 4,
        bottleneck_dim: int = 64,
        value_hidden: int = 64,
    ):
        super().__init__()

        # --- Trunk (frozen in inner loop) ---
        blocks = []
        ch = in_channels
        for _ in range(n_conv_blocks):
            blocks.append(ConvBlock(ch, trunk_hidden))
            ch = trunk_hidden
        self.trunk = nn.Sequential(*blocks)

        feat_dim = trunk_hidden * 9 * 9  # 5184 for 9x9 board

        # Bottleneck: compress spatial features (frozen in inner loop)
        self.bottleneck = nn.Sequential(
            nn.Linear(feat_dim, bottleneck_dim),
            nn.ReLU(inplace=True),
        )

        # --- Value head (adapted in inner loop) ---
        self.value_head = nn.Sequential(
            nn.Linear(bottleneck_dim, value_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(value_hidden, 1),
        )

    def forward(self, x):
        h = self.trunk(x)
        h = h.flatten(1)
        h = self.bottleneck(h)
        v = self.value_head(h).squeeze(-1)
        return v

    def trunk_params(self):
        """Parameters frozen in the inner loop."""
        for name, p in self.named_parameters():
            if name.startswith(self.TRUNK_PREFIX) or name.startswith(self.BOTTLENECK_PREFIX):
                yield name, p

    def head_params(self):
        """Parameters adapted in the inner loop."""
        for name, p in self.named_parameters():
            if name.startswith(self.VALUE_PREFIX):
                yield name, p

    def head_param_count(self):
        return sum(p.numel() for _, p in self.head_params())

    def total_param_count(self):
        return sum(p.numel() for p in self.parameters())
