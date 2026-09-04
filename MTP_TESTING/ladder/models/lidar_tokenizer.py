"""
BEV LiDAR tokeniser -- trained from scratch.

There is no useful pretrained model for 2-bin height-histogram BEV rasters at
this scale, so unlike the RGB branch this one trains. TransFuser does the same:
modality-specific encoders, not a shared tokeniser across perspective RGB and
top-down BEV. The two frames have incompatible statistics and sharing weights
between them is not what TransFuser does, despite what the legacy docstrings in
this repository claimed.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import config as C


class BEVTokenizer(nn.Module):
    """(B, 2, 256, 256) -> (B, 196, embed_dim)."""

    def __init__(self, in_channels: int = C.BEV_BINS, embed_dim: int = C.D_MODEL,
                 width: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        chans = (width, width * 2, width * 2, embed_dim)
        layers, cin = [], in_channels
        for cout in chans:
            layers += [
                nn.Conv2d(cin, cout, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.GELU(),
            ]
            cin = cout
        self.stem = nn.Sequential(*layers)      # 256 -> 16
        self.norm = nn.LayerNorm(embed_dim)
        self.grid = int(round(C.TOKENS_LIDAR ** 0.5))   # 14

    def forward(self, bev: torch.Tensor) -> torch.Tensor:
        if bev.ndim != 4 or bev.shape[1] != C.BEV_BINS:
            raise ValueError(
                f"BEV input is {tuple(bev.shape)}, expected (B, {C.BEV_BINS}, H, W)")
        f = self.stem(bev)
        f = F.adaptive_avg_pool2d(f, (self.grid, self.grid))
        t = f.flatten(2).transpose(1, 2)
        assert t.shape[1] == C.TOKENS_LIDAR, f"got {t.shape[1]} lidar tokens"
        return self.norm(t)
