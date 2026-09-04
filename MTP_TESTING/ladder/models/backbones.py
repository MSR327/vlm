"""
Frozen pretrained RGB backbones, behind one interface.

Selection between them is a Phase 5 offline experiment ranked on waypoint ADE,
not a decision to make by argument. The interface exists so that swapping the
backbone changes one string.

HARD RULE: no silent fallbacks. A missing dependency raises with the install
command. The previous codebase silently substituted a random 4-layer CNN for
MobileNetV3 when torchvision was absent -- same class name, same output shape,
entirely different semantics, one printed warning. That bug is the reason this
module is written the way it is.

`scratch_tiny` is NOT a fallback. It is the from-scratch arm of the backbone
comparison and must be requested by name.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .. import config as C


class Backbone(nn.Module):
    """(B, 3, H, W) -> (B, N, embed_dim). Frozen unless explicitly unfrozen."""

    embed_dim: int

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - interface
        raise NotImplementedError

    def freeze(self) -> "Backbone":
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()
        return self


def _require(module_name: str, backbone: str, install: str):
    try:
        return __import__(module_name)
    except ImportError as exc:
        raise ImportError(
            f"Backbone {backbone!r} requires {module_name!r}, which is not installed.\n"
            f"    pip install {install}\n"
            f"Refusing to substitute a different backbone: a silent swap would make "
            f"every result in the backbone comparison meaningless."
        ) from exc


class ScratchTiny(Backbone):
    """
    From-scratch convolutional arm of the backbone comparison.

    Deliberately small and deliberately untrained-at-init. Its role is to answer
    "is pretraining worth it on our data" with a number rather than a citation.
    """

    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.embed_dim = embed_dim
        ch = (32, 64, 128, embed_dim)
        layers, cin = [], 3
        for cout in ch:
            layers += [
                nn.Conv2d(cin, cout, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.GELU(),
            ]
            cin = cout
        self.stem = nn.Sequential(*layers)   # 224 -> 14
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.stem(x)                                  # (B, D, 14, 14)
        t = f.flatten(2).transpose(1, 2)                  # (B, 196, D)
        return self.norm(t)


class TorchvisionResNet(Backbone):
    """
    ImageNet ResNet, the TransFuser / InterFuser choice.

    Directly precedented and trivially exportable, which makes it the sensible
    default for the first pass even though its classification features are
    weaker for dense spatial tasks than modern self-supervised ones.
    """

    def __init__(self, arch: str = "resnet34", weights: str = "DEFAULT"):
        super().__init__()
        tv = _require("torchvision", arch, "torchvision")
        models = tv.models
        if not hasattr(models, arch):
            raise ValueError(f"torchvision has no model {arch!r}")
        net = getattr(models, arch)(weights=weights)
        self.stem = nn.Sequential(
            net.conv1, net.bn1, net.relu, net.maxpool,
            net.layer1, net.layer2, net.layer3, net.layer4,
        )                                                  # 224 -> 7
        self.embed_dim = 512 if arch in ("resnet18", "resnet34") else 2048
        self.norm = nn.LayerNorm(self.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.stem(x)
        t = f.flatten(2).transpose(1, 2)
        return self.norm(t)


class DINOv2Small(Backbone):
    """Self-supervised ViT-S/14. Stronger dense features; no aligned text tower."""

    def __init__(self):
        super().__init__()
        _require("timm", "dinov2_s", "timm")
        import timm
        self.net = timm.create_model(
            "vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0)
        self.embed_dim = 384
        self.norm = nn.LayerNorm(self.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = self.net.forward_features(x)
        if t.ndim != 3:
            raise RuntimeError(f"expected token output (B, N, D), got {tuple(t.shape)}")
        return self.norm(t[:, 1:])          # drop CLS


class MobileCLIPS0(Backbone):
    """
    MobileCLIP-S0 image tower.

    Preferred once available: it is the cheapest of the three, and its text
    tower shares an embedding space with the image tower, so cross-attention
    between instruction tokens and visual tokens is far better conditioned than
    with an unaligned sentence encoder.
    """

    def __init__(self, checkpoint: str | None = None):
        super().__init__()
        _require("open_clip", "mobileclip_s0", "open_clip_torch")
        import open_clip
        if checkpoint is None:
            raise ValueError(
                "mobileclip_s0 needs an explicit `checkpoint` path. Refusing to guess: "
                "a wrong checkpoint would silently change what every cached feature means.")
        model, _, _ = open_clip.create_model_and_transforms(
            "MobileCLIP-S0", pretrained=checkpoint)
        self.visual = model.visual
        self.embed_dim = 512
        self.norm = nn.LayerNorm(self.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = self.visual.forward_features(x) if hasattr(self.visual, "forward_features") \
            else self.visual(x)
        if t.ndim == 4:
            t = t.flatten(2).transpose(1, 2)
        return self.norm(t)


_REGISTRY = {
    "scratch_tiny": ScratchTiny,
    "resnet18": lambda: TorchvisionResNet("resnet18"),
    "resnet34": lambda: TorchvisionResNet("resnet34"),
    "dinov2_s": DINOv2Small,
    "mobileclip_s0": MobileCLIPS0,
}

AVAILABLE = tuple(_REGISTRY)


def build_backbone(name: str, freeze: bool = True, **kwargs) -> Backbone:
    if name not in _REGISTRY:
        raise ValueError(f"unknown backbone {name!r}; available: {AVAILABLE}")
    bb = _REGISTRY[name](**kwargs) if kwargs else _REGISTRY[name]()
    return bb.freeze() if freeze else bb
