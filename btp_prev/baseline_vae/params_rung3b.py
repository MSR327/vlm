"""
RUNG 3b -- VAE + 3 cameras (Front + Left + Right surround) + 2D BEV LiDAR with Panoramic Max-Pooling.

    OBSERVATION_DIM = 95 (pooled vision) + 95 (BEV LiDAR) + 5 (nav) = 195

Matches Rung 3's compact 195-dimensional state representation, eliminating parameter
bloat and sample inefficiency in PPO. The 3 surround camera latents (285 dims) are
channel-wise max-pooled into a single 95-dimensional panoramic visual token, preserving
the peak obstacle and lane edge activations across the full 210-degree forward arc
without increasing policy complexity.
"""
from params_vae_base import *
from params_vae_base import make_paths

RUNG            = '3b'
RUNG_NAME       = '3cam_lidar'
NUM_CAMERAS_VAE = 3
USE_LIDAR_VAE   = True
CAMERA_YAWS     = CAMERA_YAWS_3
POOL_CAMERAS    = True

OBSERVATION_DIM = 195   # 95 pooled vision + 95 lidar + 5 nav
globals().update(make_paths(RUNG_NAME))
