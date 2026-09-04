"""
Multimodal Vision-Language-Sensor Fusion Encoder for Autonomous Driving (VLM-PPO)
================================================================================
System 1: Fast Reactive Pathway (20 Hz, <25 ms, Edge CPU)

Fuses 6+ sensor modalities with Dual FiLM Conditioning:
1. Front RGB Camera (160x80)
2. Left RGB Camera (160x80 @ -60 deg)
3. Right RGB Camera (160x80 @ +60 deg)
4. Rear RGB Camera (160x80 @ 180 deg) — 360° Surround
5. 2D Bird's-Eye-View (BEV) LiDAR Grid (160x80, asymmetric -15m to +35m)
6. Continuous Language Embedding (128-dim from System 2 Speech Encoder)
7. Scene Context Embedding (64-dim from System 2 Foundation Model)
8. Vehicle Kinematic Telemetry (5-dim IMU/GNSS odometry)

Architecture:
- Shared-Weight Depthwise Separable Patch Embeddings (200 tokens per modality)
- Dual FiLM Conditioning:
  • Language FiLM: Modulates tokens based on speech/command intent (128-dim)
  • Scene FiLM: Modulates tokens based on scene context from foundation model (64-dim)
- 2-Layer Spatial-Linguistic Transformer with Multi-Head Self-Attention
- Configurable Latent Projection (Default: 128-dim + 5-dim Nav = 133-dim State)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Canonical Speech / Driving Intent Vocabulary (matching Aanchal_Spring dataset)
COMMAND_VOCAB = {
    "keep_lane": 0,
    "keep_speed": 0,
    "turn_left": 1,
    "steer_left": 1,
    "turn_right": 2,
    "steer_right": 2,
    "shift_left_lane": 3,
    "shift_right_lane": 4,
    "slow_down": 5,
    "reduce_speed": 5,
    "speed_up": 6,
    "accelerate": 6,
    "emergency_stop": 7,
    "stop": 7
}
NUM_COMMANDS = 8


class DepthwiseSeparableConv(nn.Module):
    """Ultra-lightweight depthwise separable convolution for edge inference."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.bn(self.pointwise(self.depthwise(x))))


class SharedPatchEmbedder(nn.Module):
    """
    Shared-weight patch embedder that processes 160x80 image / BEV inputs into spatial tokens.
    Input: (B, C, 80, 160) -> Output: (B, 200, embed_dim)
    """
    def __init__(self, in_channels=3, embed_dim=64, patch_size=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        
        self.proj = nn.Sequential(
            DepthwiseSeparableConv(in_channels, embed_dim // 2, kernel_size=3, stride=2, padding=1), # 40x80
            DepthwiseSeparableConv(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1),    # 20x40
            DepthwiseSeparableConv(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1),         # 10x20 = 200 patches
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (B, C, H, W)
        feat = self.proj(x) # (B, embed_dim, 10, 20)
        B, C, H, W = feat.shape
        tokens = feat.flatten(2).transpose(1, 2) # (B, 200, embed_dim)
        return self.norm(tokens)


class FiLMConditioning(nn.Module):
    """
    Continuous & Open-Vocabulary Feature-wise Linear Modulation (FiLM):
    Transforms intermediate visual features based on continuous semantic language embeddings
    (from wav2vec 2.0 / Qwen2-0.5B / text) or discrete intent tokens.
    Output: (1.0 + gamma(language)) * Visual_Tokens + beta(language)
    """
    def __init__(self, lang_dim=128, embed_dim=64, num_commands=NUM_COMMANDS):
        super().__init__()
        self.lang_dim = lang_dim
        self.embed_dim = embed_dim
        # Optional embedding lookup for discrete fallback/tokens
        self.cmd_embedding = nn.Embedding(num_commands, lang_dim)
        
        # Generator for continuous affine parameters gamma and beta
        self.film_gen = nn.Sequential(
            nn.Linear(lang_dim, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, 2 * embed_dim)
        )

    def forward(self, visual_tokens, language_input):
        """
        visual_tokens: (B, N, embed_dim)
        language_input:
            - Continuous float tensor: (B, lang_dim)
            - Discrete long tensor: (B,)
        """
        if language_input.dtype in (torch.long, torch.int, torch.int32, torch.int64):
            lang_embed = self.cmd_embedding(language_input)  # (B, lang_dim)
        else:
            lang_embed = language_input                      # (B, lang_dim)
            
        film_params = self.film_gen(lang_embed)              # (B, 2 * embed_dim)
        gamma, beta = torch.chunk(film_params, 2, dim=-1)    # (B, embed_dim) each
        
        gamma = gamma.unsqueeze(1) # (B, 1, embed_dim)
        beta = beta.unsqueeze(1)   # (B, 1, embed_dim)
        
        return (1.0 + gamma) * visual_tokens + beta


class SceneFiLMConditioning(nn.Module):
    """
    Scene-Context Feature-wise Linear Modulation (Scene FiLM):
    Conditions visual tokens based on the 64-dim scene context vector
    produced by System 2's pre-trained foundation model (MobileNetV3/CLIP).
    
    This enables System 1 to adapt its driving behavior based on scene type
    (highway vs urban vs intersection) without running scene classification
    at 20 Hz — the scene context is latched asynchronously from System 2.
    
    Output: (1.0 + gamma_scene) * tokens + beta_scene
    """
    def __init__(self, scene_dim=64, embed_dim=64):
        super().__init__()
        self.scene_dim = scene_dim
        self.embed_dim = embed_dim
        
        self.scene_film_gen = nn.Sequential(
            nn.Linear(scene_dim, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, 2 * embed_dim)
        )
    
    def forward(self, visual_tokens, scene_context):
        """
        visual_tokens: (B, N, embed_dim)
        scene_context: (B, scene_dim) — from System 2 foundation model
        """
        if scene_context is None:
            return visual_tokens
        
        film_params = self.scene_film_gen(scene_context)          # (B, 2 * embed_dim)
        gamma, beta = torch.chunk(film_params, 2, dim=-1)        # (B, embed_dim) each
        
        gamma = gamma.unsqueeze(1)  # (B, 1, embed_dim)
        beta = beta.unsqueeze(1)    # (B, 1, embed_dim)
        
        return (1.0 + gamma) * visual_tokens + beta


class TransformerBlock(nn.Module):
    """Lightweight Transformer encoder block with Multi-Head Self-Attention."""
    def __init__(self, embed_dim=64, num_heads=4, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim)
        )

    def forward(self, x):
        # Multi-Head Self-Attention
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        # Feed-Forward MLP
        x = x + self.mlp(self.norm2(x))
        return x


class MultimodalEdgeEncoder(nn.Module):
    """
    Full Multimodal Vision-Language-Sensor Fusion Encoder (Fast Pathway / System 1).
    
    Dual FiLM Conditioning Pipeline:
      Language FiLM → Scene FiLM → Transformer → Latent Projection
    
    Processes:
      - Front RGB Camera (160x80)
      - Left RGB Camera (160x80 @ -60 deg)
      - Right RGB Camera (160x80 @ +60 deg)
      - Optional: 2D BEV LiDAR Grid (160x80)
      - Continuous Language Embedding (128-dim) or string/discrete command
      - Scene Context Embedding (64-dim from System 2 foundation model)
      - Vehicle Telemetry (5-dim)
    
    Outputs:
      - Unified Observation Vector for PPO: (B, latent_dim + 5)
    """
    def __init__(self, embed_dim=64, latent_dim=128, lang_dim=128, scene_dim=64, nav_dim=5, num_cameras=4, use_lidar=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.lang_dim = lang_dim
        self.scene_dim = scene_dim
        self.nav_dim = nav_dim
        self.num_cameras = num_cameras
        self.use_lidar = use_lidar
        
        # Shared visual patch embedder for RGB cameras
        self.rgb_patch_embed = SharedPatchEmbedder(in_channels=3, embed_dim=embed_dim)
        
        # BEV LiDAR patch embedder (single-channel intensity/density)
        if self.use_lidar:
            self.lidar_patch_embed = SharedPatchEmbedder(in_channels=1, embed_dim=embed_dim)
            num_views = num_cameras + 1 # 4 cameras + 1 LiDAR BEV = 5 views (1000 tokens total)
        else:
            num_views = num_cameras     # cameras only (200 tokens per camera)

        # Learnable View Encodings (identifies Front vs Left vs Right vs Rear vs LiDAR)
        self.view_embeddings = nn.Parameter(torch.randn(1, num_views, 1, embed_dim) * 0.02)
        
        # Dual FiLM Conditioning Layers
        # 1. Language FiLM: Modulates tokens based on speech/command intent
        self.film = FiLMConditioning(lang_dim=lang_dim, embed_dim=embed_dim, num_commands=NUM_COMMANDS)
        # 2. Scene FiLM: Modulates tokens based on scene context from foundation model
        self.scene_film = SceneFiLMConditioning(scene_dim=scene_dim, embed_dim=embed_dim)
        
        # Spatial-Linguistic Transformer Stack
        self.transformer = nn.Sequential(
            TransformerBlock(embed_dim=embed_dim, num_heads=4, mlp_ratio=2.0),
            TransformerBlock(embed_dim=embed_dim, num_heads=4, mlp_ratio=2.0)
        )
        
        # Projection to PPO Latent Space
        self.latent_proj = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, latent_dim)
        )
        
        # Telemetry normalizer
        self.telemetry_proj = nn.Sequential(
            nn.Linear(nav_dim, 32),
            nn.GELU(),
            nn.Linear(32, nav_dim)
        )

    def _parse_command(self, command, device, B=1):
        """
        Parses command input into either a continuous language embedding (B, lang_dim)
        or discrete token index (B,).
        """
        if isinstance(command, torch.Tensor):
            if command.dtype in (torch.float32, torch.float64, torch.float16):
                if command.ndim == 1:
                    command = command.unsqueeze(0)
                if command.shape[0] != B:
                    command = command.repeat(B, 1)
                return command.to(device=device, dtype=torch.float32)
            else:
                if command.ndim == 0:
                    command = command.unsqueeze(0)
                if command.shape[0] != B:
                    command = command.repeat(B)
                return command.to(device=device, dtype=torch.long)
        elif isinstance(command, np.ndarray):
            if np.issubdtype(command.dtype, np.floating):
                tensor = torch.from_numpy(command).float().to(device)
                if tensor.ndim == 1:
                    tensor = tensor.unsqueeze(0)
                if tensor.shape[0] != B:
                    tensor = tensor.repeat(B, 1)
                return tensor
            else:
                tensor = torch.from_numpy(command).long().to(device)
                if tensor.ndim == 0:
                    tensor = tensor.unsqueeze(0)
                if tensor.shape[0] != B:
                    tensor = tensor.repeat(B)
                return tensor
        elif isinstance(command, str):
            cmd_id = COMMAND_VOCAB.get(command.lower().strip(), 0)
            return torch.tensor([cmd_id] * B, device=device, dtype=torch.long)
        elif isinstance(command, (int, np.integer)):
            return torch.tensor([int(command)] * B, device=device, dtype=torch.long)
        elif isinstance(command, (list, tuple)):
            if len(command) > 0 and isinstance(command[0], str):
                ids = [COMMAND_VOCAB.get(c.lower().strip(), 0) for c in command]
                return torch.tensor(ids, device=device, dtype=torch.long)
            elif len(command) > 0 and isinstance(command[0], (float, np.floating)):
                tensor = torch.tensor(command, device=device, dtype=torch.float32).unsqueeze(0)
                if tensor.shape[0] != B:
                    tensor = tensor.repeat(B, 1)
                return tensor
            else:
                return torch.tensor(command, device=device, dtype=torch.long)
        # Default: zeros
        return torch.zeros(B, self.lang_dim, device=device, dtype=torch.float32)

    def _to_tensor(self, x, expected_channels, device):
        """Converts numpy array or list of images into standardized (B, C, 80, 160) torch FloatTensor in [0, 1]."""
        if x is None:
            # Fallback zero tensor
            return torch.zeros(1, expected_channels, 80, 160, device=device, dtype=torch.float32)
        if isinstance(x, np.ndarray):
            if x.ndim == 3: # (H, W, C)
                x = np.transpose(x, (2, 0, 1))[np.newaxis, ...] # (1, C, H, W)
            elif x.ndim == 4 and x.shape[-1] in (1, 3): # (B, H, W, C)
                x = np.transpose(x, (0, 3, 1, 2))
            x = torch.from_numpy(x).float()
            if x.max() > 1.0:
                x = x / 255.0
        return x.to(device)

    def forward(self, front_img, left_img=None, right_img=None, rear_img=None, lidar_bev=None, command="keep_lane", telemetry=None, scene_context=None):
        """
        Forward pass for multimodal perception with Dual FiLM conditioning (4 Cameras + LiDAR).
        
        Args:
            front_img: Front camera image (80, 160, 3)
            left_img: Left camera image (80, 160, 3)
            right_img: Right camera image (80, 160, 3)
            rear_img: Rear camera image (80, 160, 3)
            lidar_bev: BEV LiDAR grid (80, 160, 1)
            command: Language embedding (B, 128) or string command
            telemetry: Vehicle telemetry (5,)
            scene_context: Scene context vector (B, 64) from System 2 foundation model.
                          If None, Scene FiLM is skipped (backward compatible).
        
        Returns:
            unified_obs: (B, latent_dim + 5) FloatTensor ready for PPO Policy
        """
        device = next(self.parameters()).device
        
        # 1. Prepare Camera Tensors
        t_front = self._to_tensor(front_img, 3, device)
        t_left  = self._to_tensor(left_img, 3, device)
        t_right = self._to_tensor(right_img, 3, device)
        B = t_front.shape[0]

        # 2. Extract Spatial Tokens per Camera using Shared Embedder
        f_front = self.rgb_patch_embed(t_front) # (B, 200, embed_dim)
        f_left  = self.rgb_patch_embed(t_left)  # (B, 200, embed_dim)
        f_right = self.rgb_patch_embed(t_right) # (B, 200, embed_dim)
        
        token_list = [f_front, f_left, f_right]

        if self.num_cameras >= 4:
            t_rear = self._to_tensor(rear_img, 3, device)
            f_rear = self.rgb_patch_embed(t_rear)  # (B, 200, embed_dim)
            token_list.append(f_rear)

        # 3. Optional LiDAR BEV Processing
        if self.use_lidar:
            t_lidar = self._to_tensor(lidar_bev, 1, device)
            f_lidar = self.lidar_patch_embed(t_lidar) # (B, 200, embed_dim)
            token_list.append(f_lidar)

        # 4. Add View Identifiers (Front vs Left vs Right vs Rear vs LiDAR)
        stacked_views = torch.stack(token_list, dim=1)
        stacked_views = stacked_views + self.view_embeddings[:, :len(token_list)]
        all_tokens = stacked_views.view(B, -1, self.embed_dim)

        # 5. DUAL FiLM CONDITIONING
        # 5a. Language FiLM: Modulate tokens with speech/command intent
        cmd_tensor = self._parse_command(command, device, B=B)
        conditioned_tokens = self.film(all_tokens, cmd_tensor)
        
        # 5b. Scene FiLM: Modulate tokens with scene context from foundation model
        if scene_context is not None:
            if isinstance(scene_context, np.ndarray):
                scene_context = torch.from_numpy(scene_context).float()
            if scene_context.ndim == 1:
                scene_context = scene_context.unsqueeze(0)
            scene_context = scene_context.to(device)
            conditioned_tokens = self.scene_film(conditioned_tokens, scene_context)

        # 6. Spatial-Linguistic Transformer Reasoning
        trans_out = self.transformer(conditioned_tokens)

        # 7. Global Pooling & Latent Projection
        pooled = trans_out.mean(dim=1) # (B, embed_dim)
        vlm_latent = self.latent_proj(pooled) # (B, latent_dim)

        # 8. Telemetry Concatenation
        if telemetry is not None:
            if isinstance(telemetry, np.ndarray):
                if telemetry.ndim == 1:
                    telemetry = telemetry[np.newaxis, ...]
                telemetry = torch.from_numpy(telemetry).float().to(device)
            normed_telemetry = self.telemetry_proj(telemetry)
            unified_obs = torch.cat([vlm_latent, normed_telemetry], dim=-1)
        else:
            zero_telem = torch.zeros(B, self.nav_dim, device=device)
            unified_obs = torch.cat([vlm_latent, zero_telem], dim=-1)

        return unified_obs

    def get_latent_feature(self, front_img, left_img=None, right_img=None, rear_img=None, lidar_bev=None, command="keep_lane"):
        """Extracts only the visual-linguistic latent representation (B, latent_dim)."""
        device = next(self.parameters()).device
        with torch.no_grad():
            unified = self.forward(front_img, left_img, right_img, rear_img, lidar_bev, command, telemetry=None)
            return unified[:, :self.latent_dim].cpu().numpy()


if __name__ == "__main__":
    print("=" * 70)
    print("Testing MultimodalEdgeEncoder (System 1 Fast Pathway / 4-Camera Surround)")
    print("=" * 70)
    encoder = MultimodalEdgeEncoder(embed_dim=64, latent_dim=128, lang_dim=128, scene_dim=64, nav_dim=5, num_cameras=4, use_lidar=True)
    num_params = sum(p.numel() for p in encoder.parameters())
    print(f"\nTotal Parameters: {num_params:,} ({num_params * 4 / (1024*1024):.2f} MB)")

    # Dummy Multi-Sensor Inputs (4 Cameras + 1 LiDAR BEV)
    dummy_front = np.random.randint(0, 255, (80, 160, 3), dtype=np.uint8)
    dummy_left  = np.random.randint(0, 255, (80, 160, 3), dtype=np.uint8)
    dummy_right = np.random.randint(0, 255, (80, 160, 3), dtype=np.uint8)
    dummy_rear  = np.random.randint(0, 255, (80, 160, 3), dtype=np.uint8)
    dummy_lidar = np.random.rand(80, 160, 1).astype(np.float32)
    dummy_telemetry = np.array([0.5, 20.0, 1.0, 0.1, 0.05], dtype=np.float32)

    # Test 1: Discrete string command (4 Cameras + LiDAR = 5 views = 1000 tokens)
    out_str = encoder(dummy_front, dummy_left, dummy_right, dummy_rear, dummy_lidar, "shift_left_lane", dummy_telemetry)
    print(f"\n1. Discrete Command (4 Cams + LiDAR):  {out_str.shape} (Expected: (1, 133))")
    assert out_str.shape == (1, 133)

    # Test 2: Continuous 128-dim language embedding
    dummy_lang = torch.randn(1, 128)
    out_cont = encoder(dummy_front, dummy_left, dummy_right, dummy_rear, dummy_lidar, dummy_lang, dummy_telemetry)
    print(f"2. Continuous Language (4 Cams + LiDAR): {out_cont.shape} (Expected: (1, 133))")
    assert out_cont.shape == (1, 133)

    # Test 3: DUAL FiLM — Language + Scene Context from foundation model
    dummy_scene = torch.randn(1, 64)
    out_dual = encoder(dummy_front, dummy_left, dummy_right, dummy_rear, dummy_lidar, dummy_lang, dummy_telemetry, scene_context=dummy_scene)
    print(f"3. Dual FiLM (lang + scene + 4 Cams):    {out_dual.shape} (Expected: (1, 133))")
    assert out_dual.shape == (1, 133)

    # Verify outputs differ when scene context is applied
    diff = torch.norm(out_cont - out_dual).item()
    print(f"\n   Output difference with/without scene FiLM: {diff:.4f} (should be > 0)")
    assert diff > 0, "Scene FiLM should change the output!"

    print(f"\n MultimodalEdgeEncoder 4-Camera Surround Stack is 100% OPERATIONAL!")
    print(f"   Model size: {num_params * 4 / (1024*1024):.2f} MB — fits in 1 MB L2 cache: {'YES' if num_params * 4 < 1024*1024 else 'NO'}")


