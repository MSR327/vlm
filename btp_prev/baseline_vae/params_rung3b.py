"""
RUNG 3b -- VAE + 3 cameras (Front + Left + Right surround, 210 deg FOV) + 2D BEV LiDAR.

    OBSERVATION_DIM = 95 * 3 (cams) + 95 * 1 (lidar) + 5 (nav) = 385

This provides an exact dimension-matched control against Rung 2a (4-cam visual baseline, 385-dim).
At the identical observation width of 385 dimensions, it evaluates replacing
redundant/conflicting rear vision with orthogonal 360-degree metric LiDAR geometry.

The 3 cameras cover 210 degrees of lateral semantic context (front, left, right),
while the BEV LiDAR covers 360 degrees of metric obstacle/free-space geometry.
"""
from params_vae_base import *
from params_vae_base import derive_observation_dim, make_paths

RUNG            = '3b'
RUNG_NAME       = '3cam_lidar'
NUM_CAMERAS_VAE = 3
USE_LIDAR_VAE   = True
CAMERA_YAWS     = CAMERA_YAWS_3

OBSERVATION_DIM = derive_observation_dim(NUM_CAMERAS_VAE, USE_LIDAR_VAE)   # 385
globals().update(make_paths(RUNG_NAME))
