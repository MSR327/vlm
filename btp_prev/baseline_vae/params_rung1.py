"""
RUNG 1 -- VAE + 1 camera.  The baseline as originally handed over.

    OBSERVATION_DIM = 95 * 1 + 5 = 100

Reference numbers already exist in Results_05/test_results_gpu.csv.
Re-run only if you need them regenerated under the current protocol.

Usage:  in main.py / test.py replace
            from parameters import *
        with
            from params_rung1 import *
"""
from params_vae_base import *
from params_vae_base import derive_observation_dim, make_paths

RUNG            = 1
RUNG_NAME       = '1cam'
NUM_CAMERAS_VAE = 1
USE_LIDAR_VAE   = False
CAMERA_YAWS     = CAMERA_YAWS_1

OBSERVATION_DIM = derive_observation_dim(NUM_CAMERAS_VAE, USE_LIDAR_VAE)   # 100
globals().update(make_paths(RUNG_NAME))
