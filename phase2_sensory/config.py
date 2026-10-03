"""
Configuration parameters for Phase 2: Sensory-Only TransFuser + PPO Architecture.
"""
import os

# --- Sensor Configurations ----------------------------------------------------
IM_WIDTH  = 256
IM_HEIGHT = 256
CAMERA_FOV = 90.0
CAMERA_POS = dict(x=1.5, y=0.0, z=2.4, pitch=-10.0, yaw=0.0, roll=0.0)

# LiDAR Bird's-Eye-View (BEV) Projection Parameters (Vehicle Ground Frame)
LIDAR_POS = dict(x=1.3, y=0.0, z=2.5)
LIDAR_RANGE_X = (-4.0, 28.0)     # meters forward/backward in ego vehicle frame (32m span)
LIDAR_RANGE_Y = (-16.0, 16.0)   # meters lateral in ego vehicle frame (32m span)
LIDAR_GROUND_HEIGHT = 0.3       # meters above road surface; partition threshold above/below ground
LIDAR_MIN_HEIGHT = -0.5         # meters relative to road surface; cutoff below road
LIDAR_MAX_HEIGHT = 3.5          # meters relative to road surface; cutoff above road
BEV_GRID_SIZE = 256             # pixels (256x256 grid -> 0.125m resolution per pixel isotropic)
BEV_CHANNELS = 2                # Channel 0: obstacles (> 0.3m), Channel 1: ground plane (<= 0.3m)

# --- Feature & Latent Dimensions ----------------------------------------------
LATENT_DIM = 128                # D_z: Fixed latent output invariant from cross-attention
EGO_DIM    = 8                  # Ego telemetry [v_x, v_y, yaw_rate, a_x, a_y, prev_steer, prev_throttle, prev_brake]
NAV_DIM    = 8                  # GPS Target [dx, dy, one_hot_cmd(6)]
OBS_DIM    = LATENT_DIM + EGO_DIM + NAV_DIM  # 144-d total PPO observation
ACTION_DIM = 2                  # Continuous control: [steer in [-1, 1], accel in [-1, 1]]

# --- TransFuser Cross-Attention Architecture ----------------------------------
IMAGE_BACKBONE = 'resnet34'
LIDAR_BACKBONE = 'resnet18'
FUSION_SCALES  = [16, 8]        # Feature resolutions where cross-attention exchanges tokens
ATTN_HEADS     = 4
TRANSFUSER_EMBED_DIM = 64

# --- PPO Hyperparameters ------------------------------------------------------
LEARNING_RATE      = 3e-4
GAMMA              = 0.99
LAMBDA             = 0.95
POLICY_CLIP        = 0.2
VALUE_COEF         = 0.5
ENTROPY_COEF       = 0.01
ACTION_STD_INIT    = 0.2
N_EPOCHS           = 10
BATCH_SIZE         = 64
EPISODES_PER_BATCH = 5
MAX_STEPS_PER_EP   = 2000

# --- CARLA Environment Settings -----------------------------------------------
CARLA_HOST         = os.environ.get('CARLA_HOST', 'localhost')
CARLA_PORT         = int(os.environ.get('CARLA_PORT', '2000'))
CARLA_TM_PORT      = int(os.environ.get('CARLA_TM_PORT', '8000'))
CARLA_TIMEOUT      = 30.0
SYNCHRONOUS_MODE   = True
FIXED_DELTA_SECONDS = 0.05      # 20 Hz simulation tick
DEFAULT_TOWN       = 'Town01'
WEATHER_PRESET     = 'ClearNoon'
TARGET_SPEED       = 20.0       # km/h
MAX_SPEED          = 35.0       # km/h
MIN_SPEED          = 5.0        # km/h
MAX_CENTER_DEV     = 3.0        # meters before lane departure termination
DEFAULT_SPAWN_IDX  = 12

# --- Anti-Stall & Termination Settings ----------------------------------------
STALL_SPEED_THRESH = 1.0        # km/h; speed below which vehicle is considered stalled
MAX_STALL_STEPS    = 100        # steps (5.0s at 20 Hz) before episode terminates with stall penalty
STALL_PENALTY      = -10.0      # reward penalty for standing still / deadlocking

# --- Paths & Logging ----------------------------------------------------------
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR  = os.path.join(BASE_DIR, 'checkpoints')
RESULTS_DIR     = os.path.join(BASE_DIR, 'results')
LOG_DIR         = os.path.join(BASE_DIR, 'logs')
