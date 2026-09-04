"""
Ladder invariants.

Every value here is fixed across all configurations on the ladder. A run that
uses a different value for any of them is not comparable to the others and does
not count. See PHASE2_BRIEF.md section 1.

Nothing in this module may be overridden per-experiment. Values that ARE meant
to vary between experiments live in the experiment config, not here.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Fixed representation widths
# ---------------------------------------------------------------------------
D_LATENT = 128          # encoder output width -- identical for every config
D_MODEL = 256           # internal fusion width
N_QUERIES = 32          # learned latent queries
N_WAYPOINTS = 4         # K, following TransFuser (T = 4)

DIM_EGO = 8             # speed, yaw_rate, accel_long, accel_lat, prev_action(2), speed_hist(2)
DIM_NAV = 8             # target_point(2), command one-hot(6)
DIM_TAU = N_WAYPOINTS * 2   # nominal trajectory, once the waypoint head lands

# PPO observation width. Resize PPO's input layer ONCE to each of these, in
# order, then never again.
DIM_POLICY_OBS_STAGE1 = D_LATENT + DIM_EGO + DIM_NAV            # 144
DIM_POLICY_OBS_STAGE2 = D_LATENT + DIM_EGO + DIM_NAV + DIM_TAU  # 152

# ---------------------------------------------------------------------------
# Sensors
# ---------------------------------------------------------------------------
IMG_SIZE = 224          # not 160x80 -- no pretrained backbone recovers unsampled detail
N_VIEWS = 4             # front, left, right, rear
N_FRAMES = 2            # t and t-0.2s, uniform across every modality
VIEW_NAMES = ("front", "left", "right", "rear")

# Front FOV is pre-registered narrow so that the multi-view comparison has
# headroom. TransFuser's ablation finds camera FOV is the single largest factor
# in driving score; a 125-degree front camera already covers most of what the
# side cameras would add, and the multi-view result would come back null for
# the wrong reason.
CAMERA_FOV = {"front": 65.0, "left": 90.0, "right": 90.0, "rear": 90.0}

# ---------------------------------------------------------------------------
# LiDAR BEV -- TransFuser's 2-bin height histogram.
# Deliberately NO goal channel: TransFuser rasterises the goal into the BEV,
# which means their "LiDAR helps" result partly measures a better goal
# encoding. Keeping it out is required for the ablation to be clean.
# ---------------------------------------------------------------------------
BEV_RANGE_M = 32.0      # 32m x 32m, ego-centred laterally, forward-biased
BEV_RES_M = 0.125       # metres per pixel
BEV_PIXELS = int(BEV_RANGE_M / BEV_RES_M)   # 256
BEV_BINS = 2            # points below / above the ground plane
BEV_GROUND_Z = -1.8     # ground plane in LiDAR frame, metres
BEV_FORWARD_M = 24.0    # forward extent; remainder is behind the ego

# ---------------------------------------------------------------------------
# Language -- finite instruction set, embeddings precomputed offline.
# ---------------------------------------------------------------------------
DIM_TEXT = 128
N_TEXT_TOKENS = 8

# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------
NAV_COMMANDS = ("follow", "left", "right", "straight", "lane_left", "lane_right")
NAV_TARGET_MIN_M = 25.0     # closer than 20m and the target point becomes the answer
NAV_TARGET_MAX_M = 40.0
NAV_TARGET_NOISE_M = 1.0

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEEDS = (0, 1, 2)

# ---------------------------------------------------------------------------
# Parameter matching. Encoder parameter counts must agree to within this
# fraction across configs, or "more sensors helped" is indistinguishable from
# "more parameters helped".
# ---------------------------------------------------------------------------
PARAM_MATCH_TOLERANCE = 0.10

# Tokens kept per view-frame after pooling the backbone's token grid. Fixed
# across backbones so that token count is not a hidden variable in the
# backbone comparison.
TOKENS_PER_VIEW = 64        # 8 x 8
TOKENS_LIDAR = 196          # 14 x 14
