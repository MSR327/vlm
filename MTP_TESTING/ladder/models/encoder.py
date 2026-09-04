"""
The fast pathway: 4 RGB views + LiDAR BEV (+ optional instruction) -> z in R^128.

Every configuration on the ladder emits exactly D_LATENT dimensions. That is the
control that makes the ablation readable, and it is also what lets the eventual
edge model be produced by distilling the encoder alone while the PPO policy head
stays bit-identical.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import config as C
from .backbones import Backbone, build_backbone
from .fusion import build_fusion
from .lidar_tokenizer import BEVTokenizer


class BlindEncoder(nn.Module):
    """
    The R0 control: sees only [ego, nav], still emits 128 dimensions.

    Exists so the PPO input is literally identical in shape and normalisation to
    every other configuration, making R0 a true control rather than a
    differently-shaped model. Its job is to fail. If it scores competitively the
    observation contract is still leaking and the target point must be pushed
    further out.
    """

    def __init__(self, latent_dim: int = C.D_LATENT, hidden: int = 256):
        super().__init__()
        din = C.DIM_EGO + C.DIM_NAV
        self.net = nn.Sequential(
            nn.Linear(din, hidden), nn.GELU(),
            nn.Linear(hidden, latent_dim), nn.LayerNorm(latent_dim))
        self.latent_dim = latent_dim

    def forward(self, ego: torch.Tensor, nav: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([ego, nav], dim=-1))


class FastPathwayEncoder(nn.Module):
    """
    Args:
        backbone: name from ladder.models.backbones.AVAILABLE. Frozen.
        n_views: 1 (front only) or 4 (surround). The multi-view treatment.
        use_lidar: the LiDAR treatment.
        use_language: the language treatment. Instruction embeddings are looked
            up from a precomputed table, never encoded in the loop.
        fusion: "cross_attention" (default) | "concat" | "film".
    """

    def __init__(
        self,
        backbone: str = "resnet34",
        n_views: int = C.N_VIEWS,
        use_lidar: bool = True,
        use_language: bool = False,
        fusion: str = "cross_attention",
        d_model: int = C.D_MODEL,
        latent_dim: int = C.D_LATENT,
        fusion_mlp_ratio: float = 2.0,
        backbone_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        if n_views not in (1, C.N_VIEWS):
            raise ValueError(f"n_views must be 1 or {C.N_VIEWS}, got {n_views}")
        self.n_views = n_views
        self.use_lidar = use_lidar
        self.use_language = use_language
        self.latent_dim = latent_dim
        self.view_names = C.VIEW_NAMES[:1] if n_views == 1 else C.VIEW_NAMES

        self.backbone: Backbone = build_backbone(backbone, freeze=True, **(backbone_kwargs or {}))
        self.rgb_proj = nn.Linear(self.backbone.embed_dim, d_model)

        # View and time identity. Shared backbone weights across views, so the
        # tokens are otherwise indistinguishable.
        self.view_embed = nn.Parameter(torch.randn(1, C.N_VIEWS, 1, d_model) * 0.02)
        self.time_embed = nn.Parameter(torch.randn(1, C.N_FRAMES, 1, d_model) * 0.02)

        if use_lidar:
            self.bev_tokenizer = BEVTokenizer(embed_dim=d_model)
        if use_language:
            self.text_proj = nn.Linear(C.DIM_TEXT, C.N_TEXT_TOKENS * d_model)

        self.fusion = build_fusion(fusion, dim=d_model, latent_dim=latent_dim,
                                   mlp_ratio=fusion_mlp_ratio)
        self.d_model = d_model
        self.grid = int(round(C.TOKENS_PER_VIEW ** 0.5))    # 8

    # -- token construction ------------------------------------------------
    def _rgb_tokens(self, rgb: dict) -> torch.Tensor:
        missing = [v for v in self.view_names if v not in rgb]
        if missing:
            raise ValueError(f"encoder configured for {self.view_names}, missing {missing}")

        per_view = []
        for vi, name in enumerate(self.view_names):
            x = rgb[name]
            if x.ndim != 5 or x.shape[1] != C.N_FRAMES or x.shape[2] != 3:
                raise ValueError(
                    f"rgb[{name!r}] is {tuple(x.shape)}, expected "
                    f"(B, {C.N_FRAMES}, 3, {C.IMG_SIZE}, {C.IMG_SIZE})")
            b, f = x.shape[:2]
            flat = x.reshape(b * f, *x.shape[2:])
            with torch.no_grad():                       # frozen: no activations retained
                tok = self.backbone(flat)               # (B*F, N, D_bb)
            tok = self.rgb_proj(tok)                    # (B*F, N, d_model)

            # pool the backbone's native token grid to a fixed 8x8 so token count
            # is not a hidden variable in the backbone comparison
            n = tok.shape[1]
            side = int(round(n ** 0.5))
            if side * side != n:
                raise RuntimeError(f"backbone returned {n} tokens, not a square grid")
            tok = tok.transpose(1, 2).reshape(b * f, self.d_model, side, side)
            tok = F.adaptive_avg_pool2d(tok, (self.grid, self.grid))
            tok = tok.flatten(2).transpose(1, 2).reshape(b, f, C.TOKENS_PER_VIEW, self.d_model)

            tok = tok + self.view_embed[:, vi:vi + 1] + self.time_embed[:, :, :, :]
            per_view.append(tok.reshape(b, f * C.TOKENS_PER_VIEW, self.d_model))

        return torch.cat(per_view, dim=1)

    def _text_tokens(self, emb: torch.Tensor) -> torch.Tensor:
        if emb.shape[-1] != C.DIM_TEXT:
            raise ValueError(f"instruction embedding is {tuple(emb.shape)}, "
                             f"expected (B, {C.DIM_TEXT})")
        return self.text_proj(emb).view(-1, C.N_TEXT_TOKENS, self.d_model)

    # -- forward -----------------------------------------------------------
    def forward(
        self,
        rgb: dict,
        bev: Optional[torch.Tensor] = None,
        instruction_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        sets: list[torch.Tensor] = [self._rgb_tokens(rgb)]

        if self.use_lidar:
            if bev is None:
                raise ValueError("encoder configured with use_lidar=True but bev is None. "
                                 "Refusing to substitute zeros -- a silent stand-in would "
                                 "make the LiDAR ablation meaningless.")
            sets.append(self.bev_tokenizer(bev))
        elif bev is not None:
            raise ValueError("bev supplied to an encoder built with use_lidar=False")

        if self.use_language:
            if instruction_embedding is None:
                raise ValueError("encoder configured with use_language=True but no "
                                 "instruction embedding was supplied")
            sets.append(self._text_tokens(instruction_embedding))
        elif instruction_embedding is not None:
            raise ValueError("instruction embedding supplied to a non-language encoder")

        z = self.fusion(sets)
        assert z.shape[-1] == self.latent_dim, \
            f"encoder emitted {z.shape[-1]} dims, invariant requires {self.latent_dim}"
        return z

    # -- accounting --------------------------------------------------------
    def parameter_report(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}


def solve_capacity_match(
    configs: dict,
    target: Optional[int] = None,
    tol: float = C.PARAM_MATCH_TOLERANCE,
    lo: float = 0.25,
    hi: float = 64.0,
) -> dict:
    """
    Choose a per-configuration fusion `mlp_ratio` so that every configuration on
    the ladder has the same TRAINABLE parameter count to within `tol`.

    Without this, R3 carries the BEV tokeniser and R4 carries the text
    projection, so the richer configurations are simply bigger models and
    "more sensors helped" cannot be separated from "more parameters helped".
    The extra capacity is added to the fusion stage -- the part that actually
    learns -- rather than padded in somewhere inert.

    Args:
        configs: name -> kwargs for FastPathwayEncoder (without fusion_mlp_ratio).
        target: trainable parameter budget. Defaults to the largest configuration
            at ratio 2.0, so thin configurations are widened and no configuration
            is ever shrunk below its natural size.

    Returns:
        name -> {"mlp_ratio", "trainable"}.
    """
    def trainable(kw, ratio):
        enc = FastPathwayEncoder(fusion_mlp_ratio=ratio, **kw)
        return enc.parameter_report()["trainable"]

    if target is None:
        target = max(trainable(kw, 2.0) for kw in configs.values())

    solved = {}
    for name, kw in configs.items():
        a, b = lo, hi
        best = None
        for _ in range(40):
            mid = (a + b) / 2.0
            n = trainable(kw, mid)
            if best is None or abs(n - target) < abs(best[1] - target):
                best = (mid, n)
            if n < target:
                a = mid
            else:
                b = mid
        ratio, count = best
        solved[name] = {"mlp_ratio": round(ratio, 4), "trainable": count,
                        "within_tolerance": abs(count - target) / target <= tol}
    solved["_target"] = target
    return solved


def check_parameter_match(reports: Sequence[dict], tol: float = C.PARAM_MATCH_TOLERANCE) -> dict:
    """
    Verify the parameter-matching invariant across ladder configurations.

    Compares TRAINABLE parameters: the frozen backbone is identical across
    configs by construction, so including it would mask real differences in the
    parts that actually learn.
    """
    counts = [r["trainable"] for r in reports]
    lo, hi = min(counts), max(counts)
    spread = (hi - lo) / hi if hi else 0.0
    return {"min": lo, "max": hi, "spread": spread, "within_tolerance": spread <= tol}
