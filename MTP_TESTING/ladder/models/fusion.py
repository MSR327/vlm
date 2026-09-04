"""
Fusion operators.

All three produce exactly D_LATENT dimensions regardless of how many modalities
feed them. That is the property the ladder needs: adding LiDAR adds tokens, not
output width, so "richer sensors" can never confound with "wider policy input".

CrossAttentionFusion is the default. Concat and FiLM exist as the ablation arms
for the Phase 5 offline comparison -- they are not dead code and not fallbacks.

A correction carried forward from the audit: mean-pooling is NOT the disaster an
earlier draft claimed. TransFuser average-pools to a 64-d vector and was SOTA.
The case for learned queries is referring expressions and spatial grounding --
"the van on your right" -- not that pooling is broken.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from .. import config as C


class _Block(nn.Module):
    """Pre-norm cross-attention or self-attention block."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float = 2.0):
        super().__init__()
        self.nq = nn.LayerNorm(dim)
        self.nk = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.nm = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, q: torch.Tensor, kv: torch.Tensor | None = None) -> torch.Tensor:
        k = self.nq(q) if kv is None else self.nk(kv)
        a, _ = self.attn(self.nq(q), k, k, need_weights=False)
        q = q + a
        return q + self.mlp(self.nm(q))


class CrossAttentionFusion(nn.Module):
    """
    Perceiver / Q-Former style. N learned queries cross-attend over the union of
    all modality tokens, then self-attend among themselves.

    Precedent: InterFuser's transformer decoder queries over fused multi-view +
    LiDAR tokens; LMDrive's Q-Former with learnable queries producing BEV,
    waypoint and traffic-light tokens.

    Cost is O(n_queries x n_tokens), linear in token count -- which is what keeps
    the eventual edge budget reachable. Joint self-attention over all tokens is
    quadratic and was rejected for exactly that reason.
    """

    name = "cross_attention"

    def __init__(self, dim: int = C.D_MODEL, latent_dim: int = C.D_LATENT,
                 n_queries: int = C.N_QUERIES, heads: int = 8,
                 n_cross: int = 2, n_self: int = 2, mlp_ratio: float = 2.0):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, n_queries, dim) * 0.02)
        self.cross = nn.ModuleList(_Block(dim, heads, mlp_ratio) for _ in range(n_cross))
        self.selfattn = nn.ModuleList(_Block(dim, heads, mlp_ratio) for _ in range(n_self))
        self.out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, latent_dim),
                                 nn.LayerNorm(latent_dim))
        self.latent_dim = latent_dim

    def forward(self, token_sets: Sequence[torch.Tensor]) -> torch.Tensor:
        if not token_sets:
            raise ValueError("fusion received no token sets")
        kv = torch.cat(list(token_sets), dim=1)
        q = self.queries.expand(kv.shape[0], -1, -1)
        for blk in self.cross:
            q = blk(q, kv)
        for blk in self.selfattn:
            q = blk(q)
        return self.out(q.mean(dim=1))


class ConcatFusion(nn.Module):
    """Ablation arm: mean-pool each modality, concatenate, MLP. No interaction."""

    name = "concat"

    def __init__(self, dim: int = C.D_MODEL, latent_dim: int = C.D_LATENT,
                 max_sets: int = 3, hidden: int = 512, mlp_ratio: float = 2.0):
        hidden = int(hidden * mlp_ratio / 2.0)
        super().__init__()
        self.max_sets = max_sets
        self.proj = nn.Sequential(
            nn.LayerNorm(dim * max_sets), nn.Linear(dim * max_sets, hidden), nn.GELU(),
            nn.Linear(hidden, latent_dim), nn.LayerNorm(latent_dim))
        self.latent_dim = latent_dim

    def forward(self, token_sets: Sequence[torch.Tensor]) -> torch.Tensor:
        pooled = [t.mean(dim=1) for t in token_sets]
        if len(pooled) > self.max_sets:
            raise ValueError(f"ConcatFusion built for {self.max_sets} sets, got {len(pooled)}")
        while len(pooled) < self.max_sets:      # keep width fixed across configs
            pooled.append(torch.zeros_like(pooled[0]))
        return self.proj(torch.cat(pooled, dim=-1))


class FiLMFusion(nn.Module):
    """
    Ablation arm: one global gamma/beta per frame, applied identically to every
    visual token. This is what the legacy encoder did.

    It can express "drive cautiously". It structurally cannot express "the van
    on your right", because a single global affine cannot select a location.
    That limitation is the point of including it.
    """

    name = "film"

    def __init__(self, dim: int = C.D_MODEL, latent_dim: int = C.D_LATENT,
                 hidden: int = 256, mlp_ratio: float = 2.0):
        super().__init__()
        hidden = int(hidden * mlp_ratio / 2.0)
        self.gen = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.LayerNorm(hidden),
                                 nn.Linear(hidden, 2 * dim))
        self.out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, latent_dim),
                                 nn.LayerNorm(latent_dim))
        self.latent_dim = latent_dim

    def forward(self, token_sets: Sequence[torch.Tensor]) -> torch.Tensor:
        visual = torch.cat(list(token_sets[:-1]) or [token_sets[0]], dim=1)
        cond = token_sets[-1].mean(dim=1)
        gamma, beta = self.gen(cond).chunk(2, dim=-1)
        modulated = (1.0 + gamma.unsqueeze(1)) * visual + beta.unsqueeze(1)
        return self.out(modulated.mean(dim=1))


FUSIONS = {f.name: f for f in (CrossAttentionFusion, ConcatFusion, FiLMFusion)}


def build_fusion(name: str = "cross_attention", **kwargs) -> nn.Module:
    if name not in FUSIONS:
        raise ValueError(f"unknown fusion {name!r}; available: {tuple(FUSIONS)}")
    return FUSIONS[name](**kwargs)
