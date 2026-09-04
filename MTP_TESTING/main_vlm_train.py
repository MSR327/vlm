"""
main_vlm_train.py
=============================================================================
Multimodal VLM-PPO Reinforcement Learning Training & Testing Pipeline.
- Multi-Sensor Environment: 4x RGB Cameras (360° Surround) + 2D BEV LiDAR + Telemetry
- Training Mode: Trains PPO on CARLA Town01
- Testing Mode: Evaluates Zero-Shot Generalization on CARLA Town02

Usage:
    Train: python main_vlm_train.py --mode train --town Town01 --timesteps 1000000
    Test:  python main_vlm_train.py --mode test  --town Town02 --episodes 50
=============================================================================
"""

import os
import sys
import glob
import time
import queue
import math
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter

# CARLA Python API path setup
try:
    sys.path.append(glob.glob('carla/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

try:
    import carla
    CARLA_AVAILABLE = True
except ImportError:
    carla = None
    CARLA_AVAILABLE = False

from multimodal_encoder import MultimodalEdgeEncoder, COMMAND_VOCAB
from continuous_speech_encoder import SlowCognitivePathway
from parameters import (
    IM_WIDTH, IM_HEIGHT, NUM_CAMERAS, VLM_LATENT_DIM, LANG_EMBED_DIM, NAV_DIM, OBSERVATION_DIM, ACTION_DIM,
    ACTION_STD_INIT, ACTION_STD_DECAY_FREQ, ACTION_STD_DECAY_RATE, ACTION_STD_MIN,
    LEARNING_RATE, POLICY_CLIP, GAMMA, LAMBDA, NO_OF_ITERATIONS,
    TRAIN_TOWN, TEST_TOWN, CAR_NAME, NUMBER_OF_VEHICLES, NUMBER_OF_PEDESTRIAN,
    CAMERA_SPECS, LIDAR_SPECS, RESULTS_PATH, PRETRAINED_MODEL_PATH,
    PPO_MODEL_PATH, CHECKPOINT_PATH, LOG_PATH_TRAIN, LOG_PATH_TEST
)


# ==============================================================================
# 1. PPO ACTOR-CRITIC NETWORKS (PyTorch)
# ==============================================================================
class ActorCritic(nn.Module):
    def __init__(self, state_dim=OBSERVATION_DIM, action_dim=ACTION_DIM, action_std_init=ACTION_STD_INIT):
        super().__init__()
        self.action_dim = action_dim
        self.register_buffer('action_var', torch.full((action_dim,), action_std_init * action_std_init))

        # Actor MLP (500 -> 300 -> 100 -> action_dim)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 500),
            nn.Tanh(),
            nn.Linear(500, 300),
            nn.Tanh(),
            nn.Linear(300, 100),
            nn.Tanh(),
            nn.Linear(100, action_dim),
            nn.Tanh() # Continuous control in [-1, 1]
        )

        # Critic MLP (500 -> 300 -> 100 -> 1)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 500),
            nn.Tanh(),
            nn.Linear(500, 300),
            nn.Tanh(),
            nn.Linear(300, 100),
            nn.Tanh(),
            nn.Linear(100, 1)
        )

    def set_action_std(self, new_action_std):
        self.action_var.fill_(new_action_std * new_action_std)

    def forward(self):
        raise NotImplementedError

    def act(self, state, device, deterministic=False):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32, device=device)
        if state.ndim == 1:
            state = state.unsqueeze(0)

        action_mean = self.actor(state)
        if deterministic:
            return action_mean.detach().cpu().numpy()[0], None, None

        cov_mat = torch.diag(self.action_var).to(device)
        dist = Normal(action_mean, torch.sqrt(self.action_var.to(device)))
        action = dist.sample()
        action_logprob = dist.log_prob(action).sum(dim=-1)
        state_val = self.critic(state)

        return action.detach().cpu().numpy()[0], action_logprob.detach(), state_val.detach()

    def evaluate(self, state, action, device):
        action_mean = self.actor(state)
        action_var = self.action_var.expand_as(action_mean).to(device)
        dist = Normal(action_mean, torch.sqrt(action_var))
        
        action_logprobs = dist.log_prob(action).sum(dim=-1)
        dist_entropy = dist.entropy().sum(dim=-1)
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy


class PPOMemory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


class PPOAgent:
    def __init__(self, state_dim=OBSERVATION_DIM, action_dim=ACTION_DIM, lr=LEARNING_RATE, gamma=GAMMA, K_epochs=NO_OF_ITERATIONS, eps_clip=POLICY_CLIP, device="cpu"):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.device = device

        self.buffer = PPOMemory()
        self.policy = ActorCritic(state_dim, action_dim, ACTION_STD_INIT).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCritic(state_dim, action_dim, ACTION_STD_INIT).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def select_action(self, state, deterministic=False):
        with torch.no_grad():
            action, log_prob, state_val = self.policy_old.act(state, self.device, deterministic=deterministic)
        if not deterministic:
            self.buffer.states.append(torch.tensor(state, dtype=torch.float32))
            self.buffer.actions.append(torch.tensor(action, dtype=torch.float32))
            self.buffer.logprobs.append(log_prob)
            self.buffer.state_values.append(state_val)
        return action

    def update(self):
        # Convert lists to tensors
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(self.device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(self.device)

        rewards = torch.tensor(self.buffer.rewards, dtype=torch.float32).to(self.device)
        is_terminals = torch.tensor(self.buffer.is_terminals, dtype=torch.float32).to(self.device)

        # GAE-Lambda advantage estimation
        advantages = torch.zeros_like(rewards).to(self.device)
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0.0
            else:
                next_value = old_state_values[t + 1]
            next_non_terminal = 1.0 - is_terminals[t]
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - old_state_values[t]
            gae = delta + self.gamma * LAMBDA * next_non_terminal * gae
            advantages[t] = gae

        # Compute returns from advantages
        returns = advantages + old_state_values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions, self.device)
            state_values = torch.squeeze(state_values)

            # Ratios for PPO surrogate
            ratios = torch.exp(logprobs - old_logprobs.detach())

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # Clipped surrogate loss + value loss - entropy bonus
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, returns) - 0.01 * dist_entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()

        # Copy new weights to old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save({
            'policy_state_dict': self.policy_old.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, checkpoint_path)

    def load(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'policy_state_dict' in checkpoint:
            self.policy_old.load_state_dict(checkpoint['policy_state_dict'])
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            # Backward compatibility with old checkpoints
            self.policy_old.load_state_dict(checkpoint)
            self.policy.load_state_dict(checkpoint)


# ==============================================================================
# 2. CARLA MULTI-SENSOR ENVIRONMENT (Town01 / Town02)
# ==============================================================================
def project_lidar_to_bev(point_cloud, grid_width=160, grid_height=80, min_x=-15.0, max_x=35.0, max_y=25.0):
    """
    Projects a 3D LiDAR point cloud into a 2D Bird's-Eye-View (BEV) asymmetric 360 surround height grid.
    Range: x in [-15m, +35m] (rear to front), y in [-25m, +25m] (left to right).
    Returns: (80, 160, 1) float32 array normalized to [0, 1].
    """
    raw_data = np.frombuffer(point_cloud.raw_data, dtype=np.dtype('f4'))
    points = np.reshape(raw_data, (int(raw_data.shape[0] / 4), 4))
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    mask = (x >= min_x) & (x <= max_x) & (np.abs(y) <= max_y)
    x_filt, y_filt, z_filt = x[mask], y[mask], z[mask]
    bev_grid = np.zeros((grid_height, grid_width), dtype=np.float32)
    if len(x_filt) == 0:
        return bev_grid[:, :, np.newaxis]
    grid_x = np.clip(((max_x - x_filt) / (max_x - min_x) * (grid_height - 1)).astype(np.int32), 0, grid_height - 1)
    grid_y = np.clip(((y_filt + max_y) / (2 * max_y) * (grid_width - 1)).astype(np.int32), 0, grid_width - 1)
    norm_z = np.clip((z_filt + 2.0) / 4.0, 0.0, 1.0)
    bev_grid[grid_x, grid_y] = np.maximum(bev_grid[grid_x, grid_y], norm_z)
    return bev_grid[:, :, np.newaxis]


class CarlaEnvVLM:
    def __init__(self, town=TRAIN_TOWN, host="127.0.0.1", port=2000, encoder=None, device="cpu"):
        if not CARLA_AVAILABLE:
            raise ImportError("CARLA Python API is not installed. Please run this on a host with CARLA simulator.")
        self.town = town
        self.host = host
        self.port = port
        self.device = device
        self.encoder = encoder

        self.client = carla.Client(host, port)
        self.client.set_timeout(20.0)
        self.world = self.client.load_world(town)
        self.map = self.world.get_map()
        self.bp_lib = self.world.get_blueprint_library()

        # Synchronous settings (20 FPS)
        self.settings = self.world.get_settings()
        self.settings.synchronous_mode = True
        self.settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(self.settings)

        self.actor_list = []
        self.vehicle = None
        self.collision_sensor = None
        self.collision_history = []

        self.sensor_queues = {
            'front': queue.Queue(),
            'left': queue.Queue(),
            'right': queue.Queue(),
            'rear': queue.Queue(),
            'lidar': queue.Queue()
        }

        self.current_command = "keep_lane"
        self.system2_slow = SlowCognitivePathway(device=self.device)
        self.latched_speech_embedding = self.system2_slow.encode_text("keep_lane")
        self.latched_scene_embedding = torch.zeros(1, 64, device=self.device)  # Scene context from foundation model
        self.step_count = 0
        self.prev_steer = 0.0
        # Store latest camera frames for System 2 scene perception (4 cameras)
        self.latest_front_img = None
        self.latest_left_img = None
        self.latest_right_img = None
        self.latest_rear_img = None

    def set_speech_instruction(self, instruction):
        """Asynchronously updates the latched continuous speech embedding (System 2 -> System 1)."""
        if isinstance(instruction, torch.Tensor):
            self.latched_speech_embedding = instruction.to(self.device)
            self.current_command = "continuous_embedding"
        elif isinstance(instruction, str):
            self.current_command = instruction
            self.latched_speech_embedding = self.system2_slow.encode_text(instruction)
        else:
            self.current_command = "keep_lane"
            self.latched_speech_embedding = self.system2_slow.encode_text("keep_lane")

    def _update_scene_context(self, front_img, left_img, right_img, rear_img=None):
        """
        Asynchronously updates the latched scene context embedding using
        System 2's pre-trained foundation model (MobileNetV3-Small / ImageNet).
        Called every SCENE_UPDATE_INTERVAL steps (not every frame).
        """
        self.latched_scene_embedding = self.system2_slow.encode_scene(front_img, left_img, right_img, rear_img)

    def reset(self, command=None):
        self._cleanup()
        self.collision_history = []
        self.step_count = 0
        self.prev_steer = 0.0
        
        # Latch open-vocabulary speech instruction into System 2
        cmd = command if command else random.choice(list(COMMAND_VOCAB.keys()))
        self.set_speech_instruction(cmd)

        # Spawn Vehicle
        vehicle_bp = self.bp_lib.filter('vehicle.tesla.model3')[0]
        spawn_points = self.map.get_spawn_points()
        spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        self.actor_list.append(self.vehicle)

        # Attach 4 RGB Cameras + 1 LiDAR + Collision Sensor
        self._setup_sensors()
        self._spawn_npcs()

        # Let vehicle drop to ground
        for _ in range(10):
            self.world.tick()

        # Drain stale sensor data from warmup ticks
        for key in self.sensor_queues:
            while not self.sensor_queues[key].empty():
                try:
                    self.sensor_queues[key].get_nowait()
                except queue.Empty:
                    break

        # Tick once more to get fresh data
        self.world.tick()

        return self._get_obs()

    def _spawn_npcs(self, num_vehicles=NUMBER_OF_VEHICLES, num_pedestrians=NUMBER_OF_PEDESTRIAN):
        """Spawn NPC traffic for realistic training conditions."""
        spawn_points = self.map.get_spawn_points()
        random.shuffle(spawn_points)

        # Spawn NPC vehicles
        vehicle_bps = self.bp_lib.filter('vehicle.*')
        for i, sp in enumerate(spawn_points[:min(num_vehicles, len(spawn_points) - 1)]):
            bp = random.choice(vehicle_bps)
            npc = self.world.try_spawn_actor(bp, sp)
            if npc:
                npc.set_autopilot(True)
                self.actor_list.append(npc)

    def _setup_sensors(self):
        # 1. Front Camera
        cam_bp = self.bp_lib.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', str(IM_WIDTH))
        cam_bp.set_attribute('image_size_y', str(IM_HEIGHT))
        cam_bp.set_attribute('fov', str(CAMERA_SPECS['front']['fov']))
        front_tf = carla.Transform(
            carla.Location(x=CAMERA_SPECS['front']['x'], z=CAMERA_SPECS['front']['z']),
            carla.Rotation(pitch=CAMERA_SPECS['front']['pitch'])
        )
        self.front_cam = self.world.spawn_actor(cam_bp, front_tf, attach_to=self.vehicle)
        self.front_cam.listen(self.sensor_queues['front'].put)
        self.actor_list.append(self.front_cam)

        # 2. Left Camera
        left_tf = carla.Transform(
            carla.Location(x=CAMERA_SPECS['left']['x'], y=CAMERA_SPECS['left']['y'], z=CAMERA_SPECS['left']['z']),
            carla.Rotation(pitch=CAMERA_SPECS['left']['pitch'], yaw=CAMERA_SPECS['left']['yaw'])
        )
        self.left_cam = self.world.spawn_actor(cam_bp, left_tf, attach_to=self.vehicle)
        self.left_cam.listen(self.sensor_queues['left'].put)
        self.actor_list.append(self.left_cam)

        # 3. Right Camera
        right_tf = carla.Transform(
            carla.Location(x=CAMERA_SPECS['right']['x'], y=CAMERA_SPECS['right']['y'], z=CAMERA_SPECS['right']['z']),
            carla.Rotation(pitch=CAMERA_SPECS['right']['pitch'], yaw=CAMERA_SPECS['right']['yaw'])
        )
        self.right_cam = self.world.spawn_actor(cam_bp, right_tf, attach_to=self.vehicle)
        self.right_cam.listen(self.sensor_queues['right'].put)
        self.actor_list.append(self.right_cam)

        # 4. Rear Camera (360 Surround View)
        rear_tf = carla.Transform(
            carla.Location(x=CAMERA_SPECS['rear']['x'], y=CAMERA_SPECS['rear']['y'], z=CAMERA_SPECS['rear']['z']),
            carla.Rotation(pitch=CAMERA_SPECS['rear']['pitch'], yaw=CAMERA_SPECS['rear']['yaw'])
        )
        self.rear_cam = self.world.spawn_actor(cam_bp, rear_tf, attach_to=self.vehicle)
        self.rear_cam.listen(self.sensor_queues['rear'].put)
        self.actor_list.append(self.rear_cam)

        # 5. LiDAR
        lidar_bp = self.bp_lib.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('channels', str(LIDAR_SPECS['channels']))
        lidar_bp.set_attribute('points_per_second', str(LIDAR_SPECS['points_per_second']))
        lidar_bp.set_attribute('range', str(LIDAR_SPECS['range']))
        lidar_tf = carla.Transform(carla.Location(x=LIDAR_SPECS['x'], z=LIDAR_SPECS['z']))
        self.lidar = self.world.spawn_actor(lidar_bp, lidar_tf, attach_to=self.vehicle)
        self.lidar.listen(self.sensor_queues['lidar'].put)
        self.actor_list.append(self.lidar)

        # 6. Collision Sensor
        col_bp = self.bp_lib.find('sensor.other.collision')
        col_tf = carla.Transform(carla.Location(x=1.3, z=0.5))
        self.collision_sensor = self.world.spawn_actor(col_bp, col_tf, attach_to=self.vehicle)
        self.collision_sensor.listen(lambda event: self.collision_history.append(event))
        self.actor_list.append(self.collision_sensor)

    def _get_obs(self):
        # Fetch synchronized queues (4 Cameras + LiDAR)
        front_raw = self.sensor_queues['front'].get(timeout=2.0)
        left_raw  = self.sensor_queues['left'].get(timeout=2.0)
        right_raw = self.sensor_queues['right'].get(timeout=2.0)
        rear_raw  = self.sensor_queues['rear'].get(timeout=2.0)
        lidar_raw = self.sensor_queues['lidar'].get(timeout=2.0)

        front_img = np.frombuffer(front_raw.raw_data, dtype=np.uint8).reshape((IM_HEIGHT, IM_WIDTH, 4))[:, :, :3]
        left_img  = np.frombuffer(left_raw.raw_data, dtype=np.uint8).reshape((IM_HEIGHT, IM_WIDTH, 4))[:, :, :3]
        right_img = np.frombuffer(right_raw.raw_data, dtype=np.uint8).reshape((IM_HEIGHT, IM_WIDTH, 4))[:, :, :3]
        rear_img  = np.frombuffer(rear_raw.raw_data, dtype=np.uint8).reshape((IM_HEIGHT, IM_WIDTH, 4))[:, :, :3]
        lidar_bev = project_lidar_to_bev(lidar_raw, IM_WIDTH, IM_HEIGHT)

        # Cache latest camera frames for System 2 scene perception
        self.latest_front_img = front_img
        self.latest_left_img  = left_img
        self.latest_right_img = right_img
        self.latest_rear_img  = rear_img

        # System 2 Scene Context Update (asynchronous, every N steps)
        # Uses pre-trained MobileNetV3-Small foundation model backbone
        if self.step_count % 10 == 0:  # ~2 Hz at 20 FPS
            self._update_scene_context(front_img, left_img, right_img, rear_img)

        # Telemetry
        vel = self.vehicle.get_velocity()
        speed_kmh = 3.6 * np.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
        norm_speed = min(speed_kmh / 25.0, 1.0)
        
        veh_loc = self.vehicle.get_location()
        current_wp = self.map.get_waypoint(veh_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        dist_from_center = veh_loc.distance(current_wp.transform.location) if current_wp else 0.0
        norm_dist = min(dist_from_center / 3.0, 1.0)

        # Angle deviation
        veh_forward = self.vehicle.get_transform().get_forward_vector()
        wp_forward  = current_wp.transform.get_forward_vector() if current_wp else veh_forward
        dot = veh_forward.x * wp_forward.x + veh_forward.y * wp_forward.y
        norm_angle = math.acos(np.clip(dot, -1.0, 1.0)) / math.pi

        telemetry = np.array([self.prev_steer, speed_kmh, norm_speed, norm_dist, norm_angle], dtype=np.float32)

        # Pass through Multimodal Encoder (Fast Pathway / System 1) with DUAL FiLM (4 Cameras + LiDAR)
        with torch.no_grad():
            unified_obs = self.encoder(
                front_img, left_img, right_img, rear_img, lidar_bev,
                self.latched_speech_embedding,   # Language FiLM (from System 2 speech)
                telemetry,
                scene_context=self.latched_scene_embedding  # Scene FiLM (from System 2 foundation model)
            )
            return unified_obs.cpu().numpy()[0], dist_from_center, speed_kmh, norm_angle

    def step(self, action):
        self.step_count += 1
        raw_steer, raw_longitudinal = float(action[0]), float(action[1])

        # Smooth steering with EMA (CAPS-inspired action smoothing)
        steer = float(np.clip(self.prev_steer * 0.8 + raw_steer * 0.2, -1.0, 1.0))
        self.prev_steer = steer

        # Signed longitudinal control: positive = throttle, negative = brake
        if raw_longitudinal >= 0:
            throttle = float(np.clip(raw_longitudinal, 0.0, 1.0))
            brake = 0.0
        else:
            throttle = 0.0
            brake = float(np.clip(-raw_longitudinal, 0.0, 1.0))

        # Apply CARLA Vehicle Control
        self.vehicle.apply_control(carla.VehicleControl(steer=steer, throttle=throttle, brake=brake))
        self.world.tick()

        obs, dist_from_center, speed_kmh, heading_angle = self._get_obs()

        # === Reward Calculation ===
        # R_center: Lane centering factor
        centering_factor = max(1.0 - dist_from_center / 3.0, 0.0)
        # R_speed: Speed factor (encourage 15-22 km/h)
        if speed_kmh <= 20.0:
            speed_factor = speed_kmh / 20.0
        elif speed_kmh <= 30.0:
            speed_factor = 1.0
        else:
            speed_factor = max(1.0 - (speed_kmh - 30.0) / 10.0, 0.0)
        # R_angle: Heading alignment factor (from baseline, was missing)
        angle_factor = max(1.0 - abs(heading_angle) / (math.pi / 9), 0.0)  # 20 degrees
        # R_speech: Speech adherence reward
        speech_reward = self._compute_speech_reward(speed_kmh, steer, heading_angle)

        reward = speed_factor * centering_factor * angle_factor + 0.3 * speech_reward

        done = False
        # Terminal conditions
        if len(self.collision_history) > 0:
            reward = -10.0
            done = True
        elif dist_from_center > 3.5:
            reward = -5.0
            done = True
        elif speed_kmh < 1.0 and self.step_count > 200:
            reward = -3.0
            done = True
        elif self.step_count >= 1000:
            done = True

        info = {"speed_kmh": speed_kmh, "dist_from_center": dist_from_center, 
                "collision": len(self.collision_history) > 0, "heading_angle": heading_angle}
        return obs, reward, done, info

    def _compute_speech_reward(self, speed_kmh, steer, heading_angle):
        """Compute reward for following the speech command intent.
        Returns a value in [-1.0, 1.0] indicating how well behavior matches command."""
        cmd = self.current_command
        reward = 0.0

        if cmd == "keep_lane":
            # Reward small steering, penalize large steering
            reward = 1.0 - min(abs(steer) / 0.3, 1.0)
        elif cmd == "turn_left":
            # Reward negative steering (left turn)
            reward = max(-steer, 0.0)  # steer < 0 means left
        elif cmd == "turn_right":
            # Reward positive steering (right turn)
            reward = max(steer, 0.0)
        elif cmd == "shift_left_lane":
            reward = max(-steer * 0.5, 0.0)
        elif cmd == "shift_right_lane":
            reward = max(steer * 0.5, 0.0)
        elif cmd == "speed_up":
            reward = min(speed_kmh / 25.0, 1.0)
        elif cmd == "slow_down":
            reward = max(1.0 - speed_kmh / 15.0, 0.0)
        elif cmd == "emergency_stop":
            reward = max(1.0 - speed_kmh / 5.0, 0.0)  # High reward for near-zero speed

        return float(np.clip(reward, -1.0, 1.0))

    def _cleanup(self):
        for actor in self.actor_list:
            if actor is not None and actor.is_alive:
                actor.destroy()
        self.actor_list.clear()

    def close(self):
        self._cleanup()
        self.settings.synchronous_mode = False
        self.world.apply_settings(self.settings)


# ==============================================================================
# 3. TRAINING AND TESTING LOOPS
# ==============================================================================
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Training VLM-PPO on {args.town} | Device: {device}")

    os.makedirs(PPO_MODEL_PATH, exist_ok=True)
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)
    writer = SummaryWriter(LOG_PATH_TRAIN)

    # 1. Initialize Multimodal Encoder
    encoder = MultimodalEdgeEncoder(latent_dim=VLM_LATENT_DIM, num_cameras=NUM_CAMERAS, use_lidar=True).to(device)
    if os.path.exists(PRETRAINED_MODEL_PATH):
        print(f" Loading Pre-trained Encoder from: {PRETRAINED_MODEL_PATH}")
        encoder.load_state_dict(torch.load(PRETRAINED_MODEL_PATH, map_location=device))
    encoder.eval() # Freeze encoder during initial PPO training for stability

    # 2. Initialize PPO Agent & CARLA Environment
    agent = PPOAgent(state_dim=OBSERVATION_DIM, action_dim=ACTION_DIM, device=device)
    env = CarlaEnvVLM(town=args.town, host=args.host, port=args.port, encoder=encoder, device=device)

    timestep = 0
    episode = 0

    print(f" Starting Training Loop for {args.timesteps:,} timesteps...")

    while timestep < args.timesteps:
        state, _, _, _ = env.reset()
        ep_reward = 0
        ep_steps = 0

        for _ in range(1000):
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            agent.buffer.rewards.append(reward)
            agent.buffer.is_terminals.append(done)

            state = next_state
            ep_reward += reward
            ep_steps += 1
            timestep += 1

            # Decay action std
            if timestep % ACTION_STD_DECAY_FREQ == 0:
                new_std = max(ACTION_STD_INIT - (timestep // ACTION_STD_DECAY_FREQ) * ACTION_STD_DECAY_RATE, ACTION_STD_MIN)
                agent.policy.set_action_std(new_std)
                agent.policy_old.set_action_std(new_std)
                print(f"  Action Std decayed to: {new_std:.4f}")

            # Update PPO policy every 2000 steps
            if timestep % 2000 == 0:
                agent.update()

            if done:
                break

        episode += 1
        writer.add_scalar("Reward/Episode", ep_reward, episode)
        writer.add_scalar("Steps/Episode", ep_steps, episode)

        print(f" Episode {episode:4d} | Timesteps: {timestep:7d}/{args.timesteps} | Reward: {ep_reward:7.2f} | Steps: {ep_steps:4d}")

        if episode % 50 == 0:
            ckpt_path = os.path.join(CHECKPOINT_PATH, f"ppo_ckpt_{episode}.pth")
            agent.save(ckpt_path)
            agent.save(os.path.join(PPO_MODEL_PATH, "actor_latest.pth"))
            print(f"  --> Checkpoint saved: {ckpt_path}")

    env.close()
    writer.close()
    print(" Training Complete!")


def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f" ZERO-SHOT GENERALIZATION TEST ON {args.town}")
    print(f"=======================================================")

    encoder = MultimodalEdgeEncoder(latent_dim=VLM_LATENT_DIM, num_cameras=NUM_CAMERAS, use_lidar=True).to(device)
    if os.path.exists(PRETRAINED_MODEL_PATH):
        encoder.load_state_dict(torch.load(PRETRAINED_MODEL_PATH, map_location=device))
    encoder.eval()

    agent = PPOAgent(state_dim=OBSERVATION_DIM, action_dim=ACTION_DIM, device=device)
    model_file = os.path.join(PPO_MODEL_PATH, "actor_latest.pth")
    if os.path.exists(model_file):
        agent.load(model_file)
        print(f" Loaded Trained Policy: {model_file}")

    env = CarlaEnvVLM(town=args.town, host=args.host, port=args.port, encoder=encoder, device=device)
    results = []

    for ep in range(1, args.episodes + 1):
        cmd = random.choice(list(COMMAND_VOCAB.keys()))
        state, _, _, _ = env.reset(command=cmd)
        ep_reward = 0
        ep_steps = 0
        speeds = []
        deviations = []
        start_time = time.time()

        for step_i in range(500):
            t0 = time.time()
            action = agent.select_action(state, deterministic=True)
            latency_ms = (time.time() - t0) * 1000.0

            next_state, reward, done, info = env.step(action)
            state = next_state

            ep_reward += reward
            ep_steps += 1
            speeds.append(info["speed_kmh"])
            deviations.append(info["dist_from_center"])

            if done:
                break

        total_time = time.time() - start_time
        avg_speed = np.mean(speeds) if speeds else 0.0
        avg_dev = np.mean(deviations) if deviations else 0.0
        collision = info["collision"]

        results.append({
            "episode": ep,
            "command": cmd,
            "reward": ep_reward,
            "steps": ep_steps,
            "time_sec": total_time,
            "avg_speed_kmh": avg_speed,
            "avg_dist_center_m": avg_dev,
            "collision": collision,
            "success": not collision and ep_steps >= 200
        })

        print(f" Test Ep {ep:2d}/{args.episodes} | Cmd: {cmd:15s} | Reward: {ep_reward:6.2f} | Speed: {avg_speed:4.1f} km/h | Collided: {collision}")

    env.close()

    df = pd.DataFrame(results)
    out_csv = os.path.join(RESULTS_PATH, f"test_results_{args.town.lower()}.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n Benchmark Complete! Saved to: {out_csv}")
    print(f" Success Rate: {df['success'].mean()*100:.1f}% | Collision Rate: {df['collision'].mean()*100:.1f}% | Avg Speed: {df['avg_speed_kmh'].mean():.2f} km/h")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multimodal VLM-PPO Driver")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"], help="Mode: train or test")
    parser.add_argument("--town", type=str, default=TRAIN_TOWN, help="CARLA Town (default: Town01 for train, Town02 for test)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="CARLA host IP")
    parser.add_argument("--port", type=int, default=2000, help="CARLA port")
    parser.add_argument("--timesteps", type=int, default=1000000, help="Total training timesteps")
    parser.add_argument("--episodes", type=int, default=50, help="Total testing episodes")
    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    else:
        test(args)
