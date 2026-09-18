"""
Dynamic Configuration for VAE Capacity & Latent Scaling Experiments.

Selects architecture, latent dimension, and sets observation dimensions:
    VAE_ARCH:    'baseline', 'wide', 'deep'
    LATENT_DIM:  16, 32, 64, 95, 128, 190, 256
"""
import os

# --- Architecture & Bottleneck Selection --------------------------------------
VAE_ARCH    = os.environ.get('VAE_ARCH', 'baseline').strip().lower()
LATENT_DIM  = int(os.environ.get('VAE_LATENT', '95'))
NAV_DIM     = 5

# Derived Observation Dimension
OBSERVATION_DIM = LATENT_DIM + NAV_DIM

IM_WIDTH  = 160
IM_HEIGHT = 80
VAE_INPUT_SCALE = 1.0 / 255.0

# --- Model Path Resolution ----------------------------------------------------
custom_model_dir = os.path.join(
    os.path.dirname(__file__),
    'models',
    f'vae_{VAE_ARCH}_{LATENT_DIM}',
    'var_auto_encoder_model'
)

if os.path.isdir(custom_model_dir):
    VAR_AUTO_MODEL_PATH = custom_model_dir
elif VAE_ARCH == 'baseline' and LATENT_DIM == 95 and os.path.isdir('../btp_prev/VAE/var_auto_encoder_model'):
    VAR_AUTO_MODEL_PATH = '../btp_prev/VAE/var_auto_encoder_model'
else:
    VAR_AUTO_MODEL_PATH = custom_model_dir

# --- Output Directories -------------------------------------------------------
RESULTS_TAG     = f"{VAE_ARCH}_d{LATENT_DIM}"
RESULTS_PATH    = f"Results_VAE_{RESULTS_TAG}"
PPO_MODEL_PATH  = f"{RESULTS_PATH}/ppo_model"
CHECKPOINT_PATH = f"{RESULTS_PATH}/checkpoints"
TEST_IMAGES     = f"{RESULTS_PATH}/test_images"
LOG_PATH_TRAIN  = f"{RESULTS_PATH}/runs/train"
LOG_PATH_TEST   = f"{RESULTS_PATH}/runs/test"

# --- Camera Sensor Setup ------------------------------------------------------
CAMERA_SENSOR_NAME = 'sensor.camera.semantic_segmentation'
CAMERA_FOV         = 125
CAMERA_RIG = {
    0.0: dict(x=2.4, y=0.0, z=1.5, pitch=-10.0, fov=125)
}
CAMERA_YAWS = [0.0]
NUM_CAMERAS = 1

# --- PPO Hyperparameters ------------------------------------------------------
ACTION_DIM        = 2
ACTION_STD_INIT   = 0.2
LEARNING_RATE     = 1e-4
BATCH_SIZE        = 1
POLICY_CLIP       = 0.2
GAMMA             = 0.99
LAMBDA            = 0.95
NO_OF_ITERATIONS  = 15
SEED              = 42

# --- Evaluation Protocol ------------------------------------------------------
TRAIN_TIMESTEPS     = 1e6
EPISODE_LENGTH      = 10000
TEST_EPISODES       = 20
NO_OF_TEST_EPISODES = 10
CHECKPOINT_LOAD     = False

TRAIN_TOWN = 'Town01'
TEST_TOWN  = 'Town02'
TOWN       = TRAIN_TOWN

CAR_NAME             = 'model3'
NUMBER_OF_VEHICLES   = 30
NUMBER_OF_PEDESTRIAN = 10
CONTINUOUS_ACTION    = True
VISUAL_DISPLAY       = False

# --- CARLA Server & Sync Settings ---------------------------------------------
CARLA_HOST = os.environ.get('CARLA_HOST', 'localhost')
CARLA_PORT = int(os.environ.get('CARLA_PORT', '2000'))
CARLA_TM_PORT = int(os.environ.get('CARLA_TM_PORT', '8000'))
CARLA_TIMEOUT = 30.0

SYNCHRONOUS_MODE = True
FIXED_DELTA_SECONDS = 0.05          # 20 Hz
CARLA_SEED = 42
TRAFFIC_MANAGER_SEED = 42
ENABLE_NPC_VEHICLES = False
WEATHER_PRESET = 'CloudyNoon'

TARGET_SPEED              = 22.0    # km/h
MAX_SPEED                 = 35.0
MIN_SPEED                 = 15.0
MAX_DISTANCE_FROM_CENTER  = 3.0     # m
ROUTE_LENGTH_TOWN01       = 500
ROUTE_LENGTH_TOWN02       = 500
SUCCESS_COMPLETION_THRESHOLD = 0.95
EVAL_SPAWN_POINTS = [None]
