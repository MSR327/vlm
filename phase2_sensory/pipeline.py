"""
End-to-End Multimodal Inference & Perception Pipeline for Phase 2 TransFuser-PPO.

Stitches raw sensor inputs, BEV LiDAR rasterization, cross-attention fusion,
and PPO actor-critic decision making into a clean, unified interface.
"""
import time
import numpy as np
import torch

import phase2_sensory.config as C
from phase2_sensory.bev_lidar import BEVLidarProjector
from phase2_sensory.transfuser_backbone import TransFuserBackbone
from phase2_sensory.ppo_agent import PPOAgent


class TransFuserPPOPipeline:
    """
    Unified multimodal driver pipeline:
        Sensors (RGB, LiDAR) -> TransFuser Cross-Attention -> z (128-d) -> PPO (144-d) -> Actions
    """
    def __init__(self, device=None):
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Pipeline] Initializing TransFuser-PPO Pipeline on device: {self.device}")

        self.projector = BEVLidarProjector()
        self.backbone = TransFuserBackbone(latent_dim=C.LATENT_DIM).to(self.device)
        self.agent = PPOAgent(obs_dim=C.OBS_DIM, action_dim=C.ACTION_DIM, device=self.device)

        # ImageNet normalization tensors
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std  = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    def process_sensors(self, sensor_obs):
        """
        Converts raw sensor observation dictionary into the 144-d PPO state vector.
        Args:
            sensor_obs: dict with keys ['rgb', 'lidar', 'ego', 'nav']
        Returns:
            state_np: np.ndarray of shape (144,) float32
            latency_ms: float (encoder forward time in milliseconds)
        """
        t0 = time.perf_counter()

        # 1. Prepare RGB Tensor: (H, W, 3) -> (1, 3, H, W)
        rgb_np = sensor_obs['rgb']
        rgb_t = torch.from_numpy(rgb_np).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        rgb_t = (rgb_t - self.mean) / self.std

        # 2. Rasterize BEV LiDAR Grid: (N, 4) -> (1, 2, H, W)
        lidar_pts = sensor_obs['lidar']
        bev_t = self.projector.project(lidar_pts).unsqueeze(0).to(self.device)

        # 3. TransFuser Multi-Scale Cross-Attention Forward Pass
        with torch.no_grad():
            z_t = self.backbone(rgb_t, bev_t)  # (1, 128)
            z_np = z_t.squeeze(0).cpu().numpy()

        # 4. Assemble State Vector: [z(128) || ego(8) || nav(8)] = 144-d
        ego_np = np.asarray(sensor_obs['ego'], dtype=np.float32)
        nav_np = np.asarray(sensor_obs['nav'], dtype=np.float32)
        state_np = np.concatenate([z_np, ego_np, nav_np], axis=-1)

        dt_ms = (time.perf_counter() - t0) * 1000.0
        return state_np, dt_ms

    def act(self, sensor_obs, deterministic=False):
        """
        Takes raw sensor observation, returns action_clamped, action_raw, log_prob, value, state_np, latency_ms.
        """
        state_np, latency_ms = self.process_sensors(sensor_obs)
        action_clamped, action_raw, log_prob, value = self.agent.select_action(state_np, deterministic=deterministic)
        return action_clamped, action_raw, log_prob, value, state_np, latency_ms
