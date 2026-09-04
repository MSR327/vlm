"""
Shared constants for the legacy VAE baseline ladder.

Do NOT import this directly as a run config -- import one of the
per-rung modules (params_rung1 / params_rung2 / params_rung3), which
set NUM_CAMERAS_VAE / USE_LIDAR_VAE and derive OBSERVATION_DIM.

Kept separate from the VLM's parameters.py on purpose: the two systems have
different observation widths and therefore different PPO networks, so they
can never share a run or a checkpoint.

CONTROLLED-COMPARISON CONTRACT
------------------------------
Everything in this file is held FIXED across rungs 1/2/3. The only things a
rung module is allowed to change are:

    NUM_CAMERAS_VAE, USE_LIDAR_VAE, CAMERA_YAWS
    -> and therefore OBSERVATION_DIM, and the output directory.

If you need to change a PPO hyperparameter, change it HERE so all three rungs
move together. Changing it in one rung module invalidates the ablation.
"""
import os

# --- VAE geometry -----------------------------------------------------------
# Confirmed against the saved model signature (verified on TF 2.20):
#   input_1 : (None, 160, 80, 3)
#   output_1: (None, 95)
LATENT_DIM = 95                 # VAE_Trainer_dont_touch.py:11
NAV_DIM    = 5                  # [throttle, velocity, norm_velocity,
                                #  dist_from_center, angle]  (main.py:358)
IM_WIDTH   = 160
IM_HEIGHT  = 80

# --- normalisation arm ------------------------------------------------------
# The VAE was TRAINED on [0,1] inputs (VAE_Trainer_dont_touch.py:126,
# `rescale=1.0/255`) but the deployed EncodeState feeds raw 0-255 floats
# (main.py:638). Measured on the real weights over 8 held-out frames:
#
#     input     latent std   abs max   mean pairwise cos
#     [0,1]           8.73      27.0             0.978
#     [0,255]      2863.41    7835.4             0.992
#
# The actor only clips at +/-1e8 and has no input normalisation, so the
# as-deployed values enter `tanh` fully saturated, and the latents are LESS
# frame-discriminative than in the corrected arm.
#
# Both arms are run so the ablation is defensible AND still comparable to the
# existing Results_05 rung-1 reference numbers.
#
#   BTP_NORM=raw    -> 1.0     as-deployed, bug-faithful, matches Results_05
#   BTP_NORM=scaled -> 1/255   as the VAE was trained
NORM_ARM = os.environ.get('BTP_NORM', 'raw').strip().lower()
if NORM_ARM not in ('raw', 'scaled'):
    raise ValueError(f"BTP_NORM must be 'raw' or 'scaled', got {NORM_ARM!r}")
VAE_INPUT_SCALE = 1.0 if NORM_ARM == 'raw' else (1.0 / 255.0)

# --- pretrained encoder locations -------------------------------------------
# EncodeState does os.path.join(VAR_AUTO_MODEL_PATH, 'var_auto_encoder_model')
VAR_AUTO_MODEL_PATH = 'VAE'
BEV_VAE_PATH        = 'VAE/bev_encoder_model'   # rung 3 only; trained separately

# --- camera rig ---------------------------------------------------------------
# Yaws match the 360 deg surround layout used by the new multimodal system,
# so the baseline and the proposed method see the same viewpoints.
CAMERA_YAWS_4 = [0.0, -60.0, 60.0, 180.0]       # front, left, right, rear
CAMERA_YAWS_1 = [0.0]                           # original single front camera

# Full rig placement. The front entry is byte-for-byte the original single
# camera (main.py:535-539), so rung 1 is an exact reproduction of the baseline.
CAMERA_RIG = {
     0.0: dict(x= 2.4, y= 0.0, z=1.5, pitch=-10.0, fov=125),   # front
   -60.0: dict(x= 1.0, y=-0.4, z=1.5, pitch= -5.0, fov=90),    # left
    60.0: dict(x= 1.0, y= 0.4, z=1.5, pitch= -5.0, fov=90),    # right
   180.0: dict(x=-1.5, y= 0.0, z=1.5, pitch=-10.0, fov=125),   # rear
}

# The legacy baseline uses CARLA's semantic_segmentation camera (main.py:523),
# i.e. ground-truth labels from the simulator. The proposed multimodal method
# uses true RGB. Recorded here so the write-up can state it explicitly.
CAMERA_SENSOR_NAME = 'sensor.camera.semantic_segmentation'
CAMERA_FOV         = 125

# --- BEV LiDAR (rung 3) -------------------------------------------------------
# Reuses project_lidar_to_bev() from collect_data.py:45 -- same grid geometry.
BEV_GRID_W, BEV_GRID_H = 160, 80
BEV_MIN_X, BEV_MAX_X   = -15.0, 35.0
BEV_MAX_Y              = 25.0

LIDAR_SPECS = {
    'channels': 64,
    'points_per_second': 100000,
    'range': 50.0,
    'upper_fov': 10.0,
    'lower_fov': -30.0,
    'rotation_frequency': 20.0,
    'x': 0.0, 'y': 0.0, 'z': 2.4,
}

# --- PPO hyperparameters ------------------------------------------------------
# Held IDENTICAL across all three rungs. Values match vlm/parameters.py, which
# is the config the Results_05 run was produced under.
ACTION_DIM        = 2
ACTION_STD_INIT   = 0.2
LEARNING_RATE     = 1e-4
BATCH_SIZE        = 1
POLICY_CLIP       = 0.2
GAMMA             = 0.99
LAMBDA            = 0.95
NO_OF_ITERATIONS  = 15
SEED              = 42

# --- run protocol -------------------------------------------------------------
TRAIN_TIMESTEPS     = 1e6
EPISODE_LENGTH      = 10000
TEST_EPISODES       = 50
NO_OF_TEST_EPISODES = 10
CHECKPOINT_LOAD     = False

TRAIN_TOWN = 'Town01'
TEST_TOWN  = 'Town02'
TOWN       = TRAIN_TOWN

CAR_NAME             = 'model3'
NUMBER_OF_VEHICLES   = 30
NUMBER_OF_PEDESTRIAN = 10
CONTINUOUS_ACTION    = True
VISUAL_DISPLAY       = False        # off for headless ladder runs

EXCEL_LOG_PATH = 'training_log.xlsx'

# --- CARLA server / determinism ------------------------------------------------
CARLA_HOST = os.environ.get('CARLA_HOST', 'localhost')
CARLA_PORT = int(os.environ.get('CARLA_PORT', '2000'))
CARLA_TM_PORT = int(os.environ.get('CARLA_TM_PORT', '8000'))
CARLA_TIMEOUT = 30.0

# SYNCHRONOUS MODE -- required for a fair ladder, and NOT what main.py does.
#
# main.py never touches world settings, so the simulator runs ASYNCHRONOUSLY:
# the server advances in real time while Python encodes the observation. The
# three rungs have measured encoder latencies of roughly 4 / 6 / 14 ms, so in
# async mode a slower rung issues control commands after MORE simulated time
# has elapsed. The comparison would then be partly "which encoder is faster",
# not "which sensor set is more informative" -- the exact confound this study
# exists to avoid.
#
# In synchronous mode the server advances exactly FIXED_DELTA_SECONDS per
# action for every rung, so wall-clock cost is reported separately (it is a
# real deployment constraint) without contaminating the driving metrics.
#
# Set BTP_SYNC=0 to reproduce the original asynchronous behaviour, e.g. to
# regenerate numbers comparable to Results_05.
SYNCHRONOUS_MODE = os.environ.get('BTP_SYNC', '1') != '0'
FIXED_DELTA_SECONDS = 0.05          # 20 Hz, matching vlm/collect_data.py

# CARLA's own RNG, separate from numpy/random/tf. Without this, pedestrian
# placement (create_pedestrians uses world.get_random_location_from_navigation,
# a SERVER-side draw) differs between rungs and the routes are not identical.
CARLA_SEED = 42
TRAFFIC_MANAGER_SEED = 42

# main.py defines set_other_vehicles() but NEVER CALLS IT, so NUMBER_OF_VEHICLES
# has had no effect on any run to date -- including the Results_05 reference.
# Left off by default so the ladder matches that history. Turn on with
# BTP_NPC=1 for a harder, collision-rich protocol; if you do, turn it on for
# ALL THREE rungs or the comparison is void.
ENABLE_NPC_VEHICLES = os.environ.get('BTP_NPC', '0') == '1'

WEATHER_PRESET = 'CloudyNoon'       # fixed across rungs; set in ClientConnection

# --- evaluation routes ---------------------------------------------------------
# The canonical route is the one main.py hardcodes: spawn point 30 in Town02,
# 500 waypoints at 1 m spacing. Every rung uses it, so the comparison is on
# identical geometry.
#
# With one route, N test episodes are N stochastic rollouts of the SAME road
# (the policy samples its action at test time), which measures consistency but
# not generalisation across road types. EVAL_SPAWN_POINTS lets you evaluate on
# several fixed routes instead; the list is shared by all rungs, and any index
# whose route fails to build is skipped identically for every rung.
#
# Leave as [None] for the canonical single-route protocol.
EVAL_SPAWN_POINTS = [None]
EVAL_SPAWN_POINTS_TOWN02_MULTI = [30, 10, 50, 70, 90]

# --- episode termination / success criteria -----------------------------------
# main.py hard-codes these inside reset() (lines 161-165). Lifted here so the
# three rungs provably share them and so test-time success can be defined
# against the same thresholds the reward uses.
TARGET_SPEED              = 22.0    # km/h
MAX_SPEED                 = 35.0
MIN_SPEED                 = 15.0
MAX_DISTANCE_FROM_CENTER  = 3.0     # m
ROUTE_LENGTH_TOWN01       = 500     # waypoints, 1 m apart
ROUTE_LENGTH_TOWN02       = 500

# An episode counts as a SUCCESS when the agent reaches the end of the route
# without collision / lane departure / stall / overspeed.
SUCCESS_COMPLETION_THRESHOLD = 0.95     # fraction of route waypoints


def derive_observation_dim(num_cameras, use_lidar):
    """OBSERVATION_DIM is always derived, never hand-typed.

    The 133-vs-100 mismatch previously in btp_prev is exactly the failure this
    prevents: it does not crash, it silently builds the PPO network at the
    wrong input width.
    """
    streams = num_cameras + (1 if use_lidar else 0)
    return LATENT_DIM * streams + NAV_DIM


def make_paths(tag):
    """Per-rung, per-arm output tree.

    The normalisation arm is part of the path, so the two arms of the same rung
    cannot overwrite each other's numbers either.
    """
    root = f'Results_VAE_{tag}_{NORM_ARM}'
    return dict(
        RESULTS_PATH    = root,
        PPO_MODEL_PATH  = f'{root}/ppo_model',
        CHECKPOINT_PATH = f'{root}/checkpoints',
        TEST_IMAGES     = f'{root}/test_images',
        LOG_PATH_TRAIN  = f'{root}/runs/train',
        LOG_PATH_TEST   = f'{root}/runs/test',
    )
