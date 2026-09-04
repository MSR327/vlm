"""
Stage-1 supervision heads.

The auxiliary heads are the cheapest large win available: LMDrive's ablation is
DS 36.2 -> 16.9 when visual pre-training is removed, and that pre-training is
exactly this -- detection, traffic-light state and waypoints, NOT the
steer-and-speed regression the legacy documents attributed to it.

Every head is supervised by privileged state and reads only the latent. At
evaluation time the prediction comes from the sensors; ground truth appears in
a loss term during Stage 1 and nowhere else.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .. import config as C


def _mlp(din: int, dhidden: int, dout: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(din, dhidden), nn.GELU(), nn.Linear(dhidden, dout))


class WaypointHead(nn.Module):
    """Latent -> K future ego-frame waypoints. The primary Stage-1 objective."""

    def __init__(self, latent_dim: int = C.D_LATENT, k: int = C.N_WAYPOINTS, hidden: int = 128):
        super().__init__()
        self.k = k
        self.net = _mlp(latent_dim, hidden, k * 2)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).view(-1, self.k, 2)


class TrafficLightHead(nn.Module):
    """Latent -> {none, red, yellow, green}. The thing the legacy env cheated on."""

    def __init__(self, latent_dim: int = C.D_LATENT, hidden: int = 64, n_classes: int = 4):
        super().__init__()
        self.net = _mlp(latent_dim, hidden, n_classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class DrivableBEVHead(nn.Module):
    """Latent -> coarse drivable-area occupancy logits. Teaches geometry."""

    def __init__(self, latent_dim: int = C.D_LATENT, grid: int = 32, hidden: int = 512):
        super().__init__()
        self.grid = grid
        self.net = _mlp(latent_dim, hidden, grid * grid)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).view(-1, self.grid, self.grid)


class InstructionCompleteHead(nn.Module):
    """Latent -> has the current instruction been carried out? (LMDrive)."""

    def __init__(self, latent_dim: int = C.D_LATENT, hidden: int = 64):
        super().__init__()
        self.net = _mlp(latent_dim, hidden, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)
