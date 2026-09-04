"""
RUNG 2 -- VAE + 4 cameras (360 deg surround).

    OBSERVATION_DIM = 95 * 4 + 5 = 385

The frozen VAE is shared across all four views: each image is encoded by the
same weights and the four 95-dim latents are concatenated. No retraining.

This is the rung that answers the obvious reviewer objection -- "you improved
because you added sensors, not because of your architecture."

Caveat to state in the write-up: the VAE was trained on front-camera frames,
so applying it to the left/right/rear views is a mild distribution shift.
Acceptable because all views use the same semantic-segmentation palette.

Usage:  in main.py / test.py replace
            from parameters import *
        with
            from params_rung2 import *
"""
from params_vae_base import *
from params_vae_base import derive_observation_dim, make_paths

RUNG            = 2
RUNG_NAME       = '4cam'
NUM_CAMERAS_VAE = 4
USE_LIDAR_VAE   = False
CAMERA_YAWS     = CAMERA_YAWS_4

OBSERVATION_DIM = derive_observation_dim(NUM_CAMERAS_VAE, USE_LIDAR_VAE)   # 385
globals().update(make_paths(RUNG_NAME))
