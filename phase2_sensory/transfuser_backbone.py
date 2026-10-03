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


# --- 2D Sinusoidal Positional Embeddings & TransFuser Fusion Block ------------
def build_2d_sincos_position_embedding(h, w, embed_dim, temperature=10000.0):
    """Generates fixed 2D sin-cos positional encodings (DETR / TransFuser style)."""
    grid_w = torch.arange(w, dtype=torch.float32)
    grid_h = torch.arange(h, dtype=torch.float32)
    grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='xy')
    assert embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos pos embedding'
    pos_dim = embed_dim // 4
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
    omega = 1.0 / (temperature ** omega)
    out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
    out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
    pos = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)
    return pos.unsqueeze(0)  # (1, H*W, embed_dim)


class TransFuserFusionBlock(nn.Module):
    """
    TransFuser Multi-Modal Transformer Fusion Block (CVPR 2021 / TPAMI 2022).

    1. Reduces channel dimension from in_channels -> embed_dim with 1x1 conv.
    2. Spatially downsamples high-resolution feature maps to (8, 8) with adaptive avg pool.
    3. Adds fixed 2D sinusoidal positional encodings.
    4. Concatenates Image and LiDAR tokens along sequence length:
       T = [T_img, T_lidar] in R^(B, 2*8*8, embed_dim) = R^(B, 128, embed_dim).
    5. Applies TransformerEncoderLayer (Multi-Head Self-Attention + MLP + LayerNorm).
    6. Splits back into Image and LiDAR tokens, upsamples to input resolution (H, W).
    7. Projects back to in_channels with 1x1 conv and adds residual connection.
    """
    def __init__(self, in_channels, embed_dim=C.TRANSFUSER_EMBED_DIM, num_heads=C.ATTN_HEADS, spatial_pool=8):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.spatial_pool = spatial_pool
        self.num_tokens = spatial_pool * spatial_pool  # 64 tokens per modality

        # 1x1 conv channel reductions
        self.reduce_img = nn.Conv2d(in_channels, embed_dim, kernel_size=1, bias=False)
        self.reduce_lidar = nn.Conv2d(in_channels, embed_dim, kernel_size=1, bias=False)

        # 2D Sin-Cos Positional Encodings (fixed, non-trainable buffer)
        pos = build_2d_sincos_position_embedding(spatial_pool, spatial_pool, embed_dim)
        self.register_buffer('pos_embedding', pos)

        # Standard Transformer Encoder Layer with Self-Attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # 1x1 conv projections back to input channel dimension
        self.expand_img = nn.Sequential(
            nn.Conv2d(embed_dim, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels)
        )
        self.expand_lidar = nn.Sequential(
            nn.Conv2d(embed_dim, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, img_feat, lidar_feat):
        """
        Args:
            img_feat:   (B, C, H, W)
            lidar_feat: (B, C, H, W)
        Returns:
            fused_img, fused_lidar: (B, C, H, W) with residual connection
        """
        B, C, H, W = img_feat.shape

        # 1. Channel reduction: (B, C, H, W) -> (B, embed_dim, H, W)
        r_img = self.reduce_img(img_feat)
        r_lidar = self.reduce_lidar(lidar_feat)

        # 2. Downsample to (8, 8) if H, W > spatial_pool
        if H != self.spatial_pool or W != self.spatial_pool:
            p_img = F.adaptive_avg_pool2d(r_img, (self.spatial_pool, self.spatial_pool))
            p_lidar = F.adaptive_avg_pool2d(r_lidar, (self.spatial_pool, self.spatial_pool))
        else:
            p_img = r_img
            p_lidar = r_lidar

        # 3. Flatten spatial dimensions into tokens: (B, 64, embed_dim)
        t_img = p_img.flatten(2).transpose(1, 2)
        t_lidar = p_lidar.flatten(2).transpose(1, 2)

        # 4. Add 2D sinusoidal positional encodings
        t_img = t_img + self.pos_embedding
        t_lidar = t_lidar + self.pos_embedding

        # 5. Concatenate tokens along sequence dimension: (B, 128, embed_dim)
        tokens = torch.cat([t_img, t_lidar], dim=1)

        # 6. Multi-Head Self-Attention across both intra- and cross-modal tokens
        tokens_fused = self.transformer(tokens)

        # 7. Split back into Image and LiDAR tokens: each (B, 64, embed_dim)
        f_img = tokens_fused[:, :self.num_tokens, :]
        f_lidar = tokens_fused[:, self.num_tokens:, :]

        # 8. Reshape back to (B, embed_dim, 8, 8)
        f_img = f_img.transpose(1, 2).reshape(B, self.embed_dim, self.spatial_pool, self.spatial_pool)
        f_lidar = f_lidar.transpose(1, 2).reshape(B, self.embed_dim, self.spatial_pool, self.spatial_pool)

        # 9. Upsample back to (H, W) if downsampled
        if H != self.spatial_pool or W != self.spatial_pool:
            f_img = F.interpolate(f_img, size=(H, W), mode='bilinear', align_corners=False)
            f_lidar = F.interpolate(f_lidar, size=(H, W), mode='bilinear', align_corners=False)

        # 10. Expand channels & add residual skip connection
        out_img = F.relu(img_feat + self.expand_img(f_img), inplace=True)
        out_lidar = F.relu(lidar_feat + self.expand_lidar(f_lidar), inplace=True)

        return out_img, out_lidar


# --- Full TransFuser Backbone -------------------------------------------------
class TransFuserBackbone(nn.Module):
    """
    Complete Multi-Modal TransFuser Backbone (CVPR 2021 / TPAMI 2022).
    Input:
        rgb:   (B, 3, 256, 256)
        lidar: (B, 2, 256, 256)
    Output:
        latent: (B, 128) - Fixed latent vector z invariant
    """
    def __init__(self, latent_dim=C.LATENT_DIM):
        super().__init__()
        self.latent_dim = latent_dim

        # Dual ResNet-34 encoders
        self.img_encoder = ResNetEncoder(in_channels=3, depth=C.IMAGE_BACKBONE)
        self.lidar_encoder = ResNetEncoder(in_channels=C.BEV_CHANNELS, depth=C.LIDAR_BACKBONE)

        # Multi-scale Transformer fusion blocks across all 4 stages
        self.fusion_stage1 = TransFuserFusionBlock(in_channels=64, embed_dim=C.TRANSFUSER_EMBED_DIM, num_heads=C.ATTN_HEADS)
        self.fusion_stage2 = TransFuserFusionBlock(in_channels=128, embed_dim=C.TRANSFUSER_EMBED_DIM, num_heads=C.ATTN_HEADS)
        self.fusion_stage3 = TransFuserFusionBlock(in_channels=256, embed_dim=C.TRANSFUSER_EMBED_DIM, num_heads=C.ATTN_HEADS)
        self.fusion_stage4 = TransFuserFusionBlock(in_channels=512, embed_dim=C.TRANSFUSER_EMBED_DIM, num_heads=C.ATTN_HEADS)

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
        Forward pass processing camera and BEV LiDAR streams through 4-stage multi-scale Transformer fusion.
        """
        # Stem
        img_f = self.img_encoder.stem(rgb)
        lidar_f = self.lidar_encoder.stem(lidar)

        # Stage 1 (64x64, 64-d) + Transformer Fusion
        img_f = self.img_encoder.stage1(img_f)
        lidar_f = self.lidar_encoder.stage1(lidar_f)
        img_f, lidar_f = self.fusion_stage1(img_f, lidar_f)

        # Stage 2 (32x32, 128-d) + Transformer Fusion
        img_f = self.img_encoder.stage2(img_f)
        lidar_f = self.lidar_encoder.stage2(lidar_f)
        img_f, lidar_f = self.fusion_stage2(img_f, lidar_f)

        # Stage 3 (16x16, 256-d) + Transformer Fusion
        img_f = self.img_encoder.stage3(img_f)
        lidar_f = self.lidar_encoder.stage3(lidar_f)
        img_f, lidar_f = self.fusion_stage3(img_f, lidar_f)

        # Stage 4 (8x8, 512-d) + Transformer Fusion
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
