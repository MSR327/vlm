"""
RUNG 2 -- VAE + 3 cameras (Front + Left + Right surround, 210 deg panoramic FOV).

    OBSERVATION_DIM = 95 * 3 + 5 = 290

The frozen VAE is shared across all three views: each image is encoded by the
same weights and the three 95-dim latents are concatenated. No retraining.

This is the rung that answers the obvious reviewer objection -- "you improved
because you added sensors, not because of your architecture."

Usage:  in main.py / test.py replace
            from parameters import *
        with
            from params_rung2 import *
"""
from params_vae_base import *
from params_vae_base import derive_observation_dim, make_paths

RUNG            = 2
RUNG_NAME       = '3cam'
NUM_CAMERAS_VAE = 3
USE_LIDAR_VAE   = False
CAMERA_YAWS     = CAMERA_YAWS_3

OBSERVATION_DIM = derive_observation_dim(NUM_CAMERAS_VAE, USE_LIDAR_VAE)   # 290
globals().update(make_paths(RUNG_NAME))
