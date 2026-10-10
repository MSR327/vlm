# parameters.py
"""
Configuration parameters for Multimodal VLM-PPO Autonomous Driving System.
Supports 6 sensor modalities: 4x RGB Cameras (360° Surround), 2D BEV LiDAR, IMU/GNSS, Speech, Collision.
Training on Town01, Testing/Generalization on Town02.
"""

# ==============================================================================
# SENSOR & ENCODER DIMENSIONS
# ==============================================================================
IM_WIDTH = 160
IM_HEIGHT = 80
NUM_CAMERAS = 4          # Front (0 deg), Left (-60 deg), Right (+60 deg), Rear (180 deg)
ENABLE_LIDAR = True      # 2D BEV Projected LiDAR grid (160x80)

# Visual-Linguistic Latent Space (Foundation Model output)
VLM_LATENT_DIM = 128     # Rich multimodal latent representation (sweet spot: 128)
LANG_EMBED_DIM = 128     # Continuous semantic speech intent embedding dimension
SCENE_EMBED_DIM = 64     # Scene context embedding dimension (System 2 visual perception)
NAV_DIM = 5              # Telemetry: [throttle, velocity, norm_velocity, dist_center, angle]

# Dual-System Cognitive Execution Rates
SLOW_PATHWAY_RATE_HZ = 2.0   # System 2: Language/Reasoning + Scene Perception (asynchronous)
FAST_PATHWAY_RATE_HZ = 20.0  # System 1: Real-time Edge Transformer (synchronous)
SCENE_PATHWAY_RATE_HZ = 2.0  # Scene context update rate (same as slow pathway)
SCENE_UPDATE_INTERVAL = 10   # Update scene context every N environment steps

# RL Observation Space
OBSERVATION_DIM = VLM_LATENT_DIM + NAV_DIM  # 128 + 5 = 133 dimensions
ACTION_DIM = 2           # Continuous: [Steering (-1 to 1), Longitudinal (-1=full brake, +1=full throttle)]

# ==============================================================================
# PPO REINFORCEMENT LEARNING HYPERPARAMETERS
# ==============================================================================
ACTION_STD_INIT = 0.2
LEARNING_RATE = 1e-4
BATCH_SIZE = 1
POLICY_CLIP = 0.2
GAMMA = 0.99
LAMBDA = 0.95
NO_OF_ITERATIONS = 15
SEED = 42

# Training limits
TRAIN_TIMESTEPS = 1e6
EPISODE_LENGTH = 10000
TEST_EPISODES = 50
NO_OF_TEST_EPISODES = 10

# Action std decay schedule
ACTION_STD_DECAY_RATE = 0.05
ACTION_STD_DECAY_FREQ = 50000    # Decay every 50K timesteps
ACTION_STD_MIN = 0.05            # Minimum exploration noise

# ==============================================================================
# CARLA SIMULATION PARAMETERS
# ==============================================================================
TRAIN_TOWN = 'Town01'    # Map used for training and domain fine-tuning
TEST_TOWN = 'Town02'     # Map used for zero-shot generalization testing
TOWN = TRAIN_TOWN

CAR_NAME = 'model3'
NUMBER_OF_VEHICLES = 30
NUMBER_OF_PEDESTRIAN = 10
CONTINUOUS_ACTION = True
VISUAL_DISPLAY = True

# Multi-Camera Specifications (4-Camera 360 Surround Suite)
CAMERA_SPECS = {
    'front': {'x': 2.4,  'y': 0.0,  'z': 1.5, 'pitch': -10.0, 'yaw': 0.0,    'fov': 125},
    'left':  {'x': 1.0,  'y': -0.4, 'z': 1.5, 'pitch': -5.0,  'yaw': -60.0,  'fov': 90},
    'right': {'x': 1.0,  'y': 0.4,  'z': 1.5, 'pitch': -5.0,  'yaw': 60.0,   'fov': 90},
    'rear':  {'x': -1.5, 'y': 0.0,  'z': 1.5, 'pitch': -10.0, 'yaw': 180.0,  'fov': 125}
}

# LiDAR Specifications (Asymmetric 360 Surround BEV: -15m behind to +35m forward)
LIDAR_SPECS = {
    'channels': 64,
    'points_per_second': 100000,
    'range': 50.0,
    'min_x': -15.0,
    'max_x': 35.0,
    'max_y': 25.0,
    'upper_fov': 10.0,
    'lower_fov': -30.0,
    'x': 0.0, 'y': 0.0, 'z': 2.4
}

# ==============================================================================
# PROCESSOR-IN-THE-LOOP (PIL) NETWORKING
# ==============================================================================
SIMULATION_IP = '127.0.0.1'
EDGE_IP = '0.0.0.0'
PORT = 5000

# ==============================================================================
# PATHS & DIRECTORIES
# ==============================================================================
RESULTS_PATH = 'Results_VLM'
DATASET_PATH = 'data_collected_town01'
PRETRAINED_MODEL_PATH = 'models/pretrained_vlm_encoder.pth'

PPO_MODEL_PATH = f'{RESULTS_PATH}/ppo_model'
CHECKPOINT_PATH = f'{RESULTS_PATH}/checkpoints'
LOG_PATH_TRAIN = f'{RESULTS_PATH}/runs/train'
LOG_PATH_TEST = f'{RESULTS_PATH}/runs/test'
LOG_PATH_PI = f'{RESULTS_PATH}/runs/pi_test'
TF_LITE_PATH = f'{RESULTS_PATH}/tf_lite_models'
