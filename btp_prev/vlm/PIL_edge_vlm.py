"""
PIL_edge_vlm.py
Processor-In-The-Loop (PIL) Edge Node with Multimodal Vision-Language Encoder
=============================================================================
This edge node connects to the CARLA simulation server over TCP/IP sockets,
receives camera frames and navigation telemetry, passes them through the
Multimodal Edge Encoder (4-Camera 360° Surround + LiDAR BEV), and streams
control actions (steer, throttle) back in real-time.
"""

import os
import sys
import time
import socket
import struct
import numpy as np
import torch

from parameters import *
from multimodal_encoder import MultimodalEdgeEncoder
from continuous_speech_encoder import SlowCognitivePathway
from main_vlm_train import ActorCritic


def run_edge_client(command_intent="keep_lane"):
    print(f"Connecting to CARLA Simulation Node at {SIMULATION_IP}:{PORT}...")
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        client_socket.connect((SIMULATION_IP, PORT))
        print(" Connected to CARLA Simulation Server!")
    except Exception as e:
        print(f"❌ Failed to connect to {SIMULATION_IP}:{PORT} -> {e}")
        print("Note: Ensure main.py or PIL_simulation.py is running on the CARLA host.")
        return

    # Initialize lightweight VLM edge models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    slow_system = SlowCognitivePathway(device=device)
    latched_lang_emb = slow_system.encode_text(command_intent)

    encoder = MultimodalEdgeEncoder(
        latent_dim=VLM_LATENT_DIM,
        lang_dim=LANG_EMBED_DIM,
        num_cameras=NUM_CAMERAS,
        use_lidar=True
    ).to(device)
    # Load pre-trained encoder weights if available
    if os.path.exists(PRETRAINED_MODEL_PATH):
        encoder.load_state_dict(torch.load(PRETRAINED_MODEL_PATH, map_location=device))
    encoder.eval()

    actor_critic = ActorCritic(state_dim=OBSERVATION_DIM, action_dim=ACTION_DIM).to(device)
    # Load trained policy weights if available
    model_path = os.path.join(PPO_MODEL_PATH, 'actor_latest.pth')
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'policy_state_dict' in checkpoint:
            actor_critic.load_state_dict(checkpoint['policy_state_dict'])
        else:
            actor_critic.load_state_dict(checkpoint)
    actor_critic.eval()
    print(f" Loaded VLM Edge Models on {device} (133-dim state, {NUM_CAMERAS} cameras)")

    step_count = 0
    total_time = 0.0

    try:
        while True:
            # 1. Receive Header: [H, W, 4 cameras * 3 ch + 1 BEV ch = 13 ch]
            header = client_socket.recv(12)
            if not header or len(header) < 12:
                print("Simulation finished or connection closed.")
                break

            h, w, c = struct.unpack("3I", header)
            image_size = h * w * c
            info_size = 5  # telemetry scalar count

            # 2. Receive Multi-Sensor Stream
            sensor_bytes = b""
            while len(sensor_bytes) < image_size:
                chunk = client_socket.recv(min(8192, image_size - len(sensor_bytes)))
                if not chunk:
                    break
                sensor_bytes += chunk

            # 3. Receive Navigation Telemetry
            info_bytes = client_socket.recv(info_size * 4)
            if not info_bytes or len(info_bytes) < info_size * 4:
                break

            t0 = time.perf_counter()

            sensor_array = np.frombuffer(sensor_bytes, dtype=np.uint8).reshape((h, w, c))
            nav_array = np.frombuffer(info_bytes, dtype=np.float32)

            # Separate multi-view feeds (4 Cameras + 1 LiDAR BEV)
            cam_front = sensor_array[:, :, 0:3]
            cam_left  = sensor_array[:, :, 3:6]
            cam_right = sensor_array[:, :, 6:9]
            cam_rear  = sensor_array[:, :, 9:12]
            lidar_bev = sensor_array[:, :, 12:13]

            # 4. Multimodal Vision-Language Feature Extraction (4 Cameras + LiDAR)
            with torch.no_grad():
                z_vlm = encoder(cam_front, cam_left, cam_right, cam_rear, lidar_bev, latched_lang_emb, nav_array)
                action_mean, _, _ = actor_critic.act(z_vlm, device, deterministic=True)

            t_infer = (time.perf_counter() - t0) * 1000.0  # in ms
            step_count += 1
            total_time += t_infer

            # 5. Transmit Action (steer, longitudinal) back to CARLA
            action_data = struct.pack("2f", float(action_mean[0]), float(action_mean[1]))
            client_socket.sendall(action_data)

            if step_count % 50 == 0:
                print(f"[Step {step_count:04d}] Steer: {action_mean[0]:+.3f}, Longitudinal: {action_mean[1]:.3f} | Latency: {t_infer:.2f} ms (Avg: {total_time/step_count:.2f} ms)")

    except KeyboardInterrupt:
        print("\nStopping edge client gracefully...")
    finally:
        client_socket.close()
        print("Socket disconnected.")


if __name__ == "__main__":
    intent = sys.argv[1] if len(sys.argv) > 1 else "keep_lane"
    run_edge_client(command_intent=intent)
