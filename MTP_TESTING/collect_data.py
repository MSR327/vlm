"""
collect_data.py
=============================================================================
Automated Multi-Sensor Data Collection in CARLA (Town01).
Spawns ego-vehicle with Autopilot / Traffic Manager and records synchronized:
1. Front RGB Camera (160x80)
2. Left RGB Camera (160x80 @ -60 deg)
3. Right RGB Camera (160x80 @ +60 deg)
4. Rear RGB Camera (160x80 @ 180 deg) — 360° Surround
5. 2D Bird's-Eye-View (BEV) LiDAR Grid (160x80, asymmetric -15m to +35m)
6. Driving Telemetry & Labels (Steering, Throttle, Brake, Speed, Deviation)

Usage:
    python collect_data.py --town Town01 --frames 20000 --out data_collected_town01
=============================================================================
"""

import os
import sys
import glob
import time
import queue
import argparse
import random
import math
import numpy as np
import pandas as pd

# CARLA Python API path setup
try:
    sys.path.append(glob.glob('carla/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
from parameters import (
    IM_WIDTH, IM_HEIGHT, CAMERA_SPECS, LIDAR_SPECS,
    NUMBER_OF_VEHICLES, NUMBER_OF_PEDESTRIAN
)


def project_lidar_to_bev(point_cloud, grid_width=160, grid_height=80, min_x=-15.0, max_x=35.0, max_y=25.0):
    """
    Projects a 3D LiDAR point cloud into a 2D Bird's-Eye-View (BEV) asymmetric 360 surround height grid.
    Range: x in [-15m, +35m] (rear to front), y in [-25m, +25m] (left to right).
    Returns: (80, 160, 1) float32 array normalized to [0, 1].
    """
    raw_data = np.frombuffer(point_cloud.raw_data, dtype=np.dtype('f4'))
    points = np.reshape(raw_data, (int(raw_data.shape[0] / 4), 4))
    
    x = points[:, 0] # Forward (+x) / Rear (-x)
    y = points[:, 1] # Right (+y) / Left (-y)
    z = points[:, 2] # Height (+z)

    # 360 Asymmetric Envelope: -15m behind to +35m ahead, ±25m lateral
    mask = (x >= min_x) & (x <= max_x) & (np.abs(y) <= max_y)
    x_filt, y_filt, z_filt = x[mask], y[mask], z[mask]

    bev_grid = np.zeros((grid_height, grid_width), dtype=np.float32)
    if len(x_filt) == 0:
        return bev_grid[:, :, np.newaxis]

    # Map x -> grid_height (0 to 80, 0=front +35m, 79=rear -15m), y -> grid_width (0 to 160)
    grid_x = np.clip(((max_x - x_filt) / (max_x - min_x) * (grid_height - 1)).astype(np.int32), 0, grid_height - 1)
    grid_y = np.clip(((y_filt + max_y) / (2 * max_y) * (grid_width - 1)).astype(np.int32), 0, grid_width - 1)

    # Normalize height z into [0, 1]
    norm_z = np.clip((z_filt + 2.0) / 4.0, 0.0, 1.0)
    bev_grid[grid_x, grid_y] = np.maximum(bev_grid[grid_x, grid_y], norm_z)

    return bev_grid[:, :, np.newaxis]


def main():
    parser = argparse.ArgumentParser(description="CARLA Multi-Sensor Data Collector")
    parser.add_argument("--town", type=str, default="Town01", help="CARLA Town (default: Town01)")
    parser.add_argument("--frames", type=int, default=20000, help="Total frames to collect")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="CARLA Host IP")
    parser.add_argument("--port", type=int, default=2000, help="CARLA Port")
    parser.add_argument("--out", type=str, default="data_collected_town01", help="Output directory")
    args = parser.parse_args()

    # Create output directories
    os.makedirs(os.path.join(args.out, "frames"), exist_ok=True)
    labels_csv_path = os.path.join(args.out, "labels.csv")

    actor_list = []
    sensor_queues = {
        'front': queue.Queue(),
        'left': queue.Queue(),
        'right': queue.Queue(),
        'rear': queue.Queue(),
        'lidar': queue.Queue()
    }

    try:
        print(f" Connecting to CARLA on {args.host}:{args.port}...")
        client = carla.Client(args.host, args.port)
        client.set_timeout(20.0)

        world = client.load_world(args.town)
        world_map = world.get_map()
        bp_lib = world.get_blueprint_library()

        # Set Synchronous Mode (Fixed 20 FPS)
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)

        # Weather presets to cycle through
        weather_presets = [
            ("ClearNoon", carla.WeatherParameters.ClearNoon),
            ("WetSunset", carla.WeatherParameters.WetSunset),
            ("HardRainNoon", carla.WeatherParameters.HardRainNoon),
            ("CloudyNight", carla.WeatherParameters.CloudyNight)
        ]
        world.set_weather(weather_presets[0][1])
        current_weather_name = weather_presets[0][0]

        # Spawn Ego Vehicle
        vehicle_bp = bp_lib.filter('vehicle.tesla.model3')[0]
        spawn_points = world_map.get_spawn_points()
        spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        actor_list.append(vehicle)
        print(f" Ego Vehicle spawned at {spawn_point.location}")

        # Enable Autopilot via Traffic Manager
        tm = client.get_trafficmanager(8000)
        tm.set_synchronous_mode(True)
        vehicle.set_autopilot(True, tm.get_port())
        tm.ignore_lights_percentage(vehicle, 0.0)
        tm.auto_lane_change(vehicle, True)

        # 1. Front Camera
        cam_bp = bp_lib.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', str(IM_WIDTH))
        cam_bp.set_attribute('image_size_y', str(IM_HEIGHT))
        cam_bp.set_attribute('fov', str(CAMERA_SPECS['front']['fov']))
        front_tf = carla.Transform(
            carla.Location(x=CAMERA_SPECS['front']['x'], z=CAMERA_SPECS['front']['z']),
            carla.Rotation(pitch=CAMERA_SPECS['front']['pitch'])
        )
        front_cam = world.spawn_actor(cam_bp, front_tf, attach_to=vehicle)
        front_cam.listen(sensor_queues['front'].put)
        actor_list.append(front_cam)

        # 2. Left Camera
        cam_bp_left = bp_lib.find('sensor.camera.rgb')
        cam_bp_left.set_attribute('image_size_x', str(IM_WIDTH))
        cam_bp_left.set_attribute('image_size_y', str(IM_HEIGHT))
        cam_bp_left.set_attribute('fov', str(CAMERA_SPECS['left']['fov']))
        left_tf = carla.Transform(
            carla.Location(x=CAMERA_SPECS['left']['x'], y=CAMERA_SPECS['left']['y'], z=CAMERA_SPECS['left']['z']),
            carla.Rotation(pitch=CAMERA_SPECS['left']['pitch'], yaw=CAMERA_SPECS['left']['yaw'])
        )
        left_cam = world.spawn_actor(cam_bp_left, left_tf, attach_to=vehicle)
        left_cam.listen(sensor_queues['left'].put)
        actor_list.append(left_cam)

        # 3. Right Camera
        cam_bp_right = bp_lib.find('sensor.camera.rgb')
        cam_bp_right.set_attribute('image_size_x', str(IM_WIDTH))
        cam_bp_right.set_attribute('image_size_y', str(IM_HEIGHT))
        cam_bp_right.set_attribute('fov', str(CAMERA_SPECS['right']['fov']))
        right_tf = carla.Transform(
            carla.Location(x=CAMERA_SPECS['right']['x'], y=CAMERA_SPECS['right']['y'], z=CAMERA_SPECS['right']['z']),
            carla.Rotation(pitch=CAMERA_SPECS['right']['pitch'], yaw=CAMERA_SPECS['right']['yaw'])
        )
        right_cam = world.spawn_actor(cam_bp_right, right_tf, attach_to=vehicle)
        right_cam.listen(sensor_queues['right'].put)
        actor_list.append(right_cam)

        # 4. Rear Camera (360 Surround View)
        cam_bp_rear = bp_lib.find('sensor.camera.rgb')
        cam_bp_rear.set_attribute('image_size_x', str(IM_WIDTH))
        cam_bp_rear.set_attribute('image_size_y', str(IM_HEIGHT))
        cam_bp_rear.set_attribute('fov', str(CAMERA_SPECS['rear']['fov']))
        rear_tf = carla.Transform(
            carla.Location(x=CAMERA_SPECS['rear']['x'], y=CAMERA_SPECS['rear']['y'], z=CAMERA_SPECS['rear']['z']),
            carla.Rotation(pitch=CAMERA_SPECS['rear']['pitch'], yaw=CAMERA_SPECS['rear']['yaw'])
        )
        rear_cam = world.spawn_actor(cam_bp_rear, rear_tf, attach_to=vehicle)
        rear_cam.listen(sensor_queues['rear'].put)
        actor_list.append(rear_cam)

        # 5. LiDAR Sensor
        lidar_bp = bp_lib.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('channels', str(LIDAR_SPECS['channels']))
        lidar_bp.set_attribute('points_per_second', str(LIDAR_SPECS['points_per_second']))
        lidar_bp.set_attribute('range', str(LIDAR_SPECS['range']))
        lidar_tf = carla.Transform(carla.Location(x=LIDAR_SPECS['x'], z=LIDAR_SPECS['z']))
        lidar = world.spawn_actor(lidar_bp, lidar_tf, attach_to=vehicle)
        lidar.listen(sensor_queues['lidar'].put)
        actor_list.append(lidar)

        print(f" All 5 multi-sensor actors attached (4 Cameras + 1 LiDAR).")
        print(f" Starting data collection in {args.town} for {args.frames} frames...")

        collected_records = []
        frame_idx = 0

        # Let vehicle accelerate for 30 ticks
        for _ in range(30):
            world.tick()

        # FIX-08: Drain stale sensor data accumulated during warmup
        # Without this, the first ~30 frames have a 1.5-second lag between
        # sensor images and telemetry labels.
        for q_name in sensor_queues:
            while not sensor_queues[q_name].empty():
                try:
                    sensor_queues[q_name].get_nowait()
                except queue.Empty:
                    break
        print(f" Drained sensor queues after warmup (removed 30 stale frames)")

        # FIX-09: Spawn NPC traffic for realistic data collection
        npc_list = []
        vehicle_bps = bp_lib.filter('vehicle.*')
        npc_spawn_points = world_map.get_spawn_points()
        random.shuffle(npc_spawn_points)
        for sp in npc_spawn_points[:NUMBER_OF_VEHICLES]:
            bp = random.choice(vehicle_bps)
            npc = world.try_spawn_actor(bp, sp)
            if npc:
                npc.set_autopilot(True, tm.get_port())
                npc_list.append(npc)
                actor_list.append(npc)
        print(f" Spawned {len(npc_list)} NPC vehicles for realistic data")

        # Tick once more to get fresh synchronized data
        world.tick()

        while frame_idx < args.frames:
            world.tick()

            # Retrieve synchronized sensor data (4 Cameras + LiDAR)
            front_raw = sensor_queues['front'].get(timeout=2.0)
            left_raw  = sensor_queues['left'].get(timeout=2.0)
            right_raw = sensor_queues['right'].get(timeout=2.0)
            rear_raw  = sensor_queues['rear'].get(timeout=2.0)
            lidar_raw = sensor_queues['lidar'].get(timeout=2.0)

            # Process Images to (80, 160, 3) RGB uint8
            front_img = np.frombuffer(front_raw.raw_data, dtype=np.uint8).reshape((IM_HEIGHT, IM_WIDTH, 4))[:, :, :3]
            left_img  = np.frombuffer(left_raw.raw_data, dtype=np.uint8).reshape((IM_HEIGHT, IM_WIDTH, 4))[:, :, :3]
            right_img = np.frombuffer(right_raw.raw_data, dtype=np.uint8).reshape((IM_HEIGHT, IM_WIDTH, 4))[:, :, :3]
            rear_img  = np.frombuffer(rear_raw.raw_data, dtype=np.uint8).reshape((IM_HEIGHT, IM_WIDTH, 4))[:, :, :3]

            # Process LiDAR to (80, 160, 1) BEV grid
            lidar_bev = project_lidar_to_bev(lidar_raw, IM_WIDTH, IM_HEIGHT)

            # Retrieve Vehicle Telemetry & Control Actions
            control = vehicle.get_control()
            vel = vehicle.get_velocity()
            speed_kmh = 3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
            
            # Waypoint Alignment
            veh_loc = vehicle.get_location()
            current_wp = world_map.get_waypoint(veh_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
            dist_from_center = veh_loc.distance(current_wp.transform.location) if current_wp else 0.0

            # Save arrays (4 Cameras + LiDAR)
            frame_prefix = f"{frame_idx:06d}"
            np.savez_compressed(
                os.path.join(args.out, "frames", f"{frame_prefix}.npz"),
                front=front_img,
                left=left_img,
                right=right_img,
                rear=rear_img,
                lidar=lidar_bev
            )

            # Record Telemetry Label
            collected_records.append({
                "frame_id": frame_prefix,
                "steer": control.steer,
                "throttle": control.throttle,
                "brake": control.brake,
                "speed_kmh": speed_kmh,
                "dist_from_center": dist_from_center,
                "weather": current_weather_name
            })

            frame_idx += 1

            # Weather randomization every 2,500 frames
            if frame_idx % 2500 == 0:
                w_name, w_preset = random.choice(weather_presets)
                world.set_weather(w_preset)
                current_weather_name = w_name
                print(f" [Frame {frame_idx}/{args.frames}] Switched weather to {current_weather_name}")

            if frame_idx % 500 == 0:
                print(f" Collected {frame_idx}/{args.frames} frames ({frame_idx/args.frames*100:.1f}%) | Speed: {speed_kmh:.1f} km/h | Steer: {control.steer:.2f}")

        # Save metadata CSV
        df = pd.DataFrame(collected_records)
        df.to_csv(labels_csv_path, index=False)
        print(f"\n Data Collection Complete! {len(df)} frames saved to {args.out}")

    except KeyboardInterrupt:
        print("\n Data collection interrupted by user.")
    except Exception as e:
        print(f"\n Error during collection: {e}")
    finally:
        print(" Cleaning up CARLA actors...")
        try:
            settings = world.get_settings()
            settings.synchronous_mode = False
            world.apply_settings(settings)
        except Exception:
            pass
        for actor in actor_list:
            if actor is not None and actor.is_alive:
                actor.destroy()
        print(" All actors cleaned up successfully.")


if __name__ == "__main__":
    main()
