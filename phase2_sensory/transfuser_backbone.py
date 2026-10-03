"""
TransFuser Multi-Modal Cross-Attention Backbone for Phase 2 Autonomous Driving.

Fuses Front RGB Camera (256x256x3) with 2-bin BEV LiDAR (256x256x2) via
multi-scale Transformer cross-attention, emitting a fixed-width 128-d latent
representation vector for PPO reinforcement learning.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import phase2_sensory.config as C


# --- Standard Residual Blocks (Native PyTorch, Self-Contained) ----------------
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_c)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c)
            )

    def forward(self, x):
        return F.relu(self.conv(x) + self.shortcut(x), inplace=True)


class ResNetEncoder(nn.Module):
    """
    Modular ResNet encoder supporting arbitrary input channels (RGB=3, BEV=2).
    Emits feature maps at scales:
        Stage 1: 64x64,  64-d
        Stage 2: 32x32, 128-d
        Stage 3: 16x16, 256-d  <-- Cross-Attention Fusion Scale 1
        Stage 4:  8x8,  512-d  <-- Cross-Attention Fusion Scale 2
    """
    def __init__(self, in_channels=3, depth='resnet34'):
        super().__init__()
        num_blocks = [3, 4, 6, 3] if depth == 'resnet34' else [2, 2, 2, 2]

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.stage1 = self._make_stage(64, 64, num_blocks[0], stride=1)
        self.stage2 = self._make_stage(64, 128, num_blocks[1], stride=2)
        self.stage3 = self._make_stage(128, 256, num_blocks[2], stride=2)
        self.stage4 = self._make_stage(256, 512, num_blocks[3], stride=2)

    def _make_stage(self, in_c, out_c, blocks, stride):
        layers = [ConvBlock(in_c, out_c, stride=stride)]
        for _ in range(1, blocks):
            layers.append(ConvBlock(out_c, out_c, stride=1))
        return nn.Sequential(*layers)


# --- Multi-Scale Cross-Attention Transformer Fusion ---------------------------
class CrossAttentionFusionBlock(nn.Module):
    """
    Bidirectional Cross-Attention Transformer between Vision and LiDAR tokens.
    Vision tokens query 3D metric distances from LiDAR.
    LiDAR tokens query semantic lane/traffic context from Vision.
    """
    def __init__(self, dim, num_heads=4, spatial_res=16):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_tokens = spatial_res * spatial_res

        # 2D Positional Embeddings
        self.pos_img = nn.Parameter(torch.randn(1, self.num_tokens, dim) * 0.02)
        self.pos_lidar = nn.Parameter(torch.randn(1, self.num_tokens, dim) * 0.02)

        # Multi-Head Cross-Attention: Image queries LiDAR
        self.cross_attn_img = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_img1 = nn.LayerNorm(dim)
        self.norm_img2 = nn.LayerNorm(dim)
        self.mlp_img = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )

        # Multi-Head Cross-Attention: LiDAR queries Image
        self.cross_attn_lidar = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_lidar1 = nn.LayerNorm(dim)
        self.norm_lidar2 = nn.LayerNorm(dim)
        self.mlp_lidar = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, img_feat, lidar_feat):
        """
        Args:
            img_feat:   (B, C, H, W)
            lidar_feat: (B, C, H, W)
        Returns:
            fused_img, fused_lidar: (B, C, H, W)
        """
        B, C, H, W = img_feat.shape
        # Flatten spatial dims to tokens: (B, H*W, C)
        t_img = img_feat.flatten(2).transpose(1, 2)
        t_lidar = lidar_feat.flatten(2).transpose(1, 2)

        # Add positional encodings
        q_img = t_img + self.pos_img[:, :t_img.shape[1], :]
        q_lidar = t_lidar + self.pos_lidar[:, :t_lidar.shape[1], :]

        # Cross-Attention: Vision queries LiDAR depth
        attn_img_out, _ = self.cross_attn_img(query=q_img, key=q_lidar, value=t_lidar)
        t_img = self.norm_img1(t_img + attn_img_out)
        t_img = self.norm_img2(t_img + self.mlp_img(t_img))

        # Cross-Attention: LiDAR queries Vision semantics
        attn_lidar_out, _ = self.cross_attn_lidar(query=q_lidar, key=q_img, value=t_img)
        t_lidar = self.norm_lidar1(t_lidar + attn_lidar_out)
        t_lidar = self.norm_lidar2(t_lidar + self.mlp_lidar(t_lidar))

        # Unflatten back to spatial feature maps
        img_out = t_img.transpose(1, 2).reshape(B, C, H, W)
        lidar_out = t_lidar.transpose(1, 2).reshape(B, C, H, W)
        return img_out, lidar_out


# --- Full TransFuser Backbone -------------------------------------------------
class TransFuserBackbone(nn.Module):
    """
    Complete Multi-Modal TransFuser Backbone.
    Input:
        rgb:   (B, 3, 256, 256)
        lidar: (B, 2, 256, 256)
    Output:
        latent: (B, 128) - Fixed latent vector z invariant
    """
    def __init__(self, latent_dim=C.LATENT_DIM):
        super().__init__()
        self.latent_dim = latent_dim

        # Dual encoders
        self.img_encoder = ResNetEncoder(in_channels=3, depth=C.IMAGE_BACKBONE)
        self.lidar_encoder = ResNetEncoder(in_channels=C.BEV_CHANNELS, depth=C.LIDAR_BACKBONE)

        # Multi-scale cross-attention fusion blocks
        self.fusion_stage3 = CrossAttentionFusionBlock(dim=256, num_heads=C.ATTN_HEADS, spatial_res=16)
        self.fusion_stage4 = CrossAttentionFusionBlock(dim=512, num_heads=C.ATTN_HEADS, spatial_res=8)

        # Global average pooling
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fixed Latent Invariant Projection: [512 (img) + 512 (lidar)] -> 128-d
        self.proj_head = nn.Sequential(
            nn.Linear(1024, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, latent_dim),
            nn.LayerNorm(latent_dim)
        )

        # Initialize weights with Kaiming Normal
        self._init_weights()

        # Attempt to load torchvision ImageNet pretrained weights for RGB ResNet-34
        self.load_pretrained_image_weights()

    def _init_weights(self):
        """Kaiming Normal initialization for conv and linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def load_pretrained_image_weights(self):
        """Transfers torchvision ImageNet pretrained weights into self.img_encoder if available."""
        try:
            import torchvision.models as tv_models
            weights = tv_models.ResNet34_Weights.DEFAULT
            tv_model = tv_models.resnet34(weights=weights)

            # Copy stem
            self.img_encoder.stem[0].weight.data.copy_(tv_model.conv1.weight.data)
            self.img_encoder.stem[1].weight.data.copy_(tv_model.bn1.weight.data)
            self.img_encoder.stem[1].bias.data.copy_(tv_model.bn1.bias.data)
            self.img_encoder.stem[1].running_mean.data.copy_(tv_model.bn1.running_mean.data)
            self.img_encoder.stem[1].running_var.data.copy_(tv_model.bn1.running_var.data)

            # Stages 1 to 4
            stages = [
                (self.img_encoder.stage1, tv_model.layer1),
                (self.img_encoder.stage2, tv_model.layer2),
                (self.img_encoder.stage3, tv_model.layer3),
                (self.img_encoder.stage4, tv_model.layer4),
            ]
            for my_stage, tv_layer in stages:
                for my_blk, tv_blk in zip(my_stage, tv_layer):
                    my_blk.conv[0].weight.data.copy_(tv_blk.conv1.weight.data)
                    my_blk.conv[1].weight.data.copy_(tv_blk.bn1.weight.data)
                    my_blk.conv[1].bias.data.copy_(tv_blk.bn1.bias.data)
                    my_blk.conv[1].running_mean.data.copy_(tv_blk.bn1.running_mean.data)
                    my_blk.conv[1].running_var.data.copy_(tv_blk.bn1.running_var.data)

                    my_blk.conv[3].weight.data.copy_(tv_blk.conv2.weight.data)
                    my_blk.conv[4].weight.data.copy_(tv_blk.bn2.weight.data)
                    my_blk.conv[4].bias.data.copy_(tv_blk.bn2.bias.data)
                    my_blk.conv[4].running_mean.data.copy_(tv_blk.bn2.running_mean.data)
                    my_blk.conv[4].running_var.data.copy_(tv_blk.bn2.running_var.data)

                    if len(my_blk.shortcut) > 0 and tv_blk.downsample is not None:
                        my_blk.shortcut[0].weight.data.copy_(tv_blk.downsample[0].weight.data)
                        my_blk.shortcut[1].weight.data.copy_(tv_blk.downsample[1].weight.data)
                        my_blk.shortcut[1].bias.data.copy_(tv_blk.downsample[1].bias.data)
                        my_blk.shortcut[1].running_mean.data.copy_(tv_blk.downsample[1].running_mean.data)
                        my_blk.shortcut[1].running_var.data.copy_(tv_blk.downsample[1].running_var.data)
            print("[TransFuserBackbone] ImageNet pretrained weights successfully transferred to RGB ResNet-34!")
            return True
        except Exception as e:
            print(f"[TransFuserBackbone] ImageNet pretraining transfer bypassed ({e}); running with Kaiming Normal weights.")
            return False

    def forward(self, rgb, lidar):
        """
        Forward pass processing camera and BEV LiDAR streams through multi-scale attention.
        """
        # Stem & Stages 1-2
        img_f = self.img_encoder.stage1(self.img_encoder.stem(rgb))
        img_f = self.img_encoder.stage2(img_f)

        lidar_f = self.lidar_encoder.stage1(self.lidar_encoder.stem(lidar))
        lidar_f = self.lidar_encoder.stage2(lidar_f)

        # Stage 3 (16x16, 256-d) + Cross-Attention Fusion
        img_f = self.img_encoder.stage3(img_f)
        lidar_f = self.lidar_encoder.stage3(lidar_f)
        img_f, lidar_f = self.fusion_stage3(img_f, lidar_f)

        # Stage 4 (8x8, 512-d) + Cross-Attention Fusion
        img_f = self.img_encoder.stage4(img_f)
        lidar_f = self.lidar_encoder.stage4(lidar_f)
        img_f, lidar_f = self.fusion_stage4(img_f, lidar_f)

        # Spatial pooling: (B, 512, 1, 1) -> (B, 512)
        v_img = self.pool(img_f).flatten(1)
        v_lidar = self.pool(lidar_f).flatten(1)

        # Concatenate & Project to fixed 128-d latent
        v_cat = torch.cat([v_img, v_lidar], dim=-1)
        z = self.proj_head(v_cat)
        return z

    def freeze_backbone(self):
        """Freezes all backbone weights for representation preservation during RL."""
        for param in self.parameters():
            param.requires_grad = False
        print("[TransFuserBackbone] Frozen all visual-LiDAR backbone weights.")
