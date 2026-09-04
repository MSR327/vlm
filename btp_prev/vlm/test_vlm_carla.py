"""
test_vlm_carla.py
=================
Live CARLA Simulator Verification Script for Multimodal Vision-Language Agent.
1. Connects to CARLA on 127.0.0.1:2000.
2. Uses REAL RGB Camera (sensor.camera.rgb) — NO semantic segmentation.
3. Streams live frames into MultimodalEdgeEncoder.
4. Generates real actions and computes empirical per-frame latency in CARLA.
"""

import sys
import time
import math
import weakref
import numpy as np
import pygame
import torch

try:
    import carla
except ImportError:
    print("Warning: carla Python egg not in local python path. Run this on the machine with CARLA installed.")

from parameters import *
from multimodal_encoder import MultimodalEdgeEncoder
from main_vlm import PyTorchActorCritic


class RGBCameraSensor:
    def __init__(self, vehicle, img_h=80, img_w=160):
        self.parent = vehicle
        self.img_h = img_h
        self.img_w = img_w
        self.latest_frame = None
        world = self.parent.get_world()
        
        bp = world.get_blueprint_library().find('sensor.camera.rgb')
        bp.set_attribute('image_size_x', str(img_w))
        bp.set_attribute('image_size_y', str(img_h))
        bp.set_attribute('fov', '125')
        
        self.sensor = world.spawn_actor(
            bp,
            carla.Transform(carla.Location(x=2.4, z=1.5), carla.Rotation(pitch=-10)),
            attach_to=self.parent
        )
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda image: RGBCameraSensor._callback(weak_self, image))

    @staticmethod
    def _callback(weak_self, image):
        self = weak_self()
        if not self:
            return
        # Extract raw RGB array (H, W, 3)
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        self.latest_frame = array[:, :, :3]  # drop alpha channel


def run_live_test(num_steps=300, voice_command="go_straight"):
    print("=" * 65)
    print("  🚗 LIVE CARLA MULTIMODAL VLM VERIFICATION")
    print("=" * 65)

    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    print("✅ Connected to CARLA Simulator!")

    # Set weather and map
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter("model3")[0]
    
    spawn_points = world.get_map().get_spawn_points()
    spawn_point = spawn_points[0] if spawn_points else carla.Transform()
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    print(f"✅ Vehicle spawned at: ({spawn_point.location.x:.1f}, {spawn_point.location.y:.1f})")

    # Spawn real RGB camera
    rgb_cam = RGBCameraSensor(vehicle, img_h=IM_HEIGHT, img_w=IM_WIDTH)
    time.sleep(1.0)  # Wait for sensor pipeline to stream

    # Initialize PyTorch Models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = MultimodalEdgeEncoder(
        img_h=IM_HEIGHT,
        img_w=IM_WIDTH,
        in_channels=3,
        embed_dim=64,
        latent_dim=LATENT_DIM,
        nav_dim=5
    ).to(device)
    encoder.eval()

    actor_critic = PyTorchActorCritic(obs_dim=100, action_dim=2).to(device)
    actor_critic.eval()
    print(f"✅ VLM Models loaded on {device}")

    print(f"\n🚀 Starting live drive test with voice command: '{voice_command}'...\n")
    latencies = []

    try:
        for step in range(1, num_steps + 1):
            while rgb_cam.latest_frame is None:
                time.sleep(0.001)

            frame = rgb_cam.latest_frame.copy()
            
            # Read vehicle telemetry
            vel = vehicle.get_velocity()
            speed_kmh = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2) * 3.6
            nav_data = np.array([0.5, speed_kmh / 3.6, speed_kmh / 30.0, 0.05, 0.02], dtype=np.float32)

            t0 = time.perf_counter()

            # End-to-end inference
            with torch.no_grad():
                img_tensor = torch.from_numpy(frame).float().unsqueeze(0).to(device)
                nav_tensor = torch.from_numpy(nav_data).float().unsqueeze(0).to(device)
                
                obs_100d = encoder(img_tensor, nav_tensor, command_id=voice_command)
                action, _ = actor_critic.get_action(obs_100d, deterministic=True)

            t_infer = (time.perf_counter() - t0) * 1000.0  # ms
            latencies.append(t_infer)

            # Apply vehicle control
            steer = float(np.clip(action[0], -1.0, 1.0))
            throttle = float(np.clip((action[1] + 1.0) / 2.0, 0.0, 1.0))
            vehicle.apply_control(carla.VehicleControl(steer=steer, throttle=throttle))

            if step % 20 == 0:
                print(f"[Frame {step:03d}/{num_steps}] Speed: {speed_kmh:4.1f} km/h | Steer: {steer:+5.2f} | Throttle: {throttle:4.2f} | VLM Latency: {t_infer:5.2f} ms")

            time.sleep(0.05)  # ~20 FPS step rate

        avg_lat = np.mean(latencies)
        print("\n" + "=" * 65)
        print(f"🎉 VERIFICATION COMPLETE:")
        print(f"  • Total Frames Processed:  {num_steps}")
        print(f"  • Average VLM Latency:     {avg_lat:.2f} ms")
        print(f"  • Real-time Frame Rate:    {1000.0/avg_lat:.1f} FPS (Target: > 20 FPS)")
        print("=" * 65)

    finally:
        print("\nCleaning up CARLA actors...")
        rgb_cam.sensor.destroy()
        vehicle.destroy()
        print("Cleanup complete.")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "go_straight"
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    run_live_test(num_steps=steps, voice_command=cmd)
