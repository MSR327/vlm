"""
RUNG 3 -- VAE + 1 front camera + 2D BEV LiDAR.

    OBSERVATION_DIM = 95 * 2 + 5 = 195

Adds a complementary geometric sensor (LiDAR BEV) to the single front camera
instead of adding redundant visual sensors (more cameras). This tests whether
orthogonal spatial information (3D obstacle geometry) helps driving more than
redundant visual information (side/rear camera views).

The BEV occupancy grid is encoded by a SECOND VAE, not the camera one.
The pretrained camera VAE was trained on semantic-segmentation RGB; a BEV
occupancy grid is badly out of distribution for it, so reusing it here would
produce meaningless latents and understate the baseline.

That second encoder does not exist yet -- it needs a short training run on
BEV frames (see collect_data.py:45 project_lidar_to_bev for the grid, and
VAE/VAE_Trainer_dont_touch.py for the training recipe). Until BEV_VAE_PATH
exists, check_rung3.py will report it as missing rather than fail obscurely.

Usage:  in main.py / test.py replace
            from parameters import *
        with
            from params_rung3 import *
"""
from params_vae_base import *
from params_vae_base import derive_observation_dim, make_paths

RUNG            = 3
RUNG_NAME       = '1cam_lidar'
NUM_CAMERAS_VAE = 1
USE_LIDAR_VAE   = True
CAMERA_YAWS     = CAMERA_YAWS_1

OBSERVATION_DIM = derive_observation_dim(NUM_CAMERAS_VAE, USE_LIDAR_VAE)   # 195
globals().update(make_paths(RUNG_NAME))
