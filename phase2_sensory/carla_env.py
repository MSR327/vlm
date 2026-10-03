"""
CARLA Closed-Loop Sensory Environment with Strict Two-Record Observation Architecture.

Eliminates ground-truth map telemetry leaks:
    step(action) -> (sensor_obs, privileged_state, reward, done, info)
Policy sees ONLY legitimate onboard sensors (RGB, LiDAR, IMU/wheel-speed, noisy GPS target).
Reward reads strictly from privileged ground truth.
"""
import math
import queue
import time
import weakref
import numpy as np

import phase2_sensory.config as C

try:
    import carla
except ImportError:
    carla = None


class CarlaSensoryEnvironment:
    """
    Closed-loop CARLA environment providing synchronized Front RGB, 3D LiDAR,
    and vehicle proprioception.
    """
    def __init__(self, town=C.DEFAULT_TOWN, host=C.CARLA_HOST, port=C.CARLA_PORT, timeout=C.CARLA_TIMEOUT):
        if carla is None:
            raise ImportError("carla python egg/package is not installed.")

        self.town = town
        self.host = host
        self.port = port
        self.timeout = timeout

        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(self.timeout)
        self.world = self.client.load_world(self.town)
        self.map = self.world.get_map()

        self._apply_determinism()

        self.vehicle = None
        self.camera_sensor = None
        self.lidar_sensor = None
        self.collision_sensor = None

        self.actor_list = []
        self.sensor_list = []

        # Synchronous sensor queues & buffers
        self.camera_queue = queue.Queue()
        self.lidar_queue = queue.Queue()
        self.latest_rgb = None
        self.latest_lidar_points = None
        self.collision_events = []

        # Vehicle internal state
        self.velocity = 0.0
        self.prev_steer = 0.0
        self.prev_throttle = 0.0
        self.prev_brake = 0.0
        self.timesteps = 0
        self.stall_steps = 0
        self.distance_covered = 0.0
        self.route_waypoints = []
        self.current_wp_idx = 0
        self.total_route_length = 500

    def _apply_determinism(self):
        settings = self.world.get_settings()
        settings.synchronous_mode = C.SYNCHRONOUS_MODE
        settings.fixed_delta_seconds = C.FIXED_DELTA_SECONDS
        self.world.apply_settings(settings)

        tm = self.client.get_trafficmanager(C.CARLA_TM_PORT)
        tm.set_synchronous_mode(C.SYNCHRONOUS_MODE)
        print(f"[CarlaEnv] Synchronous mode enabled (dt={C.FIXED_DELTA_SECONDS}s, 20 Hz).")

    def reset(self, spawn_idx=C.DEFAULT_SPAWN_IDX):
        self._teardown()

        bp_lib = self.world.get_blueprint_library()
        vehicle_bp = bp_lib.filter('vehicle.tesla.model3')[0]

        spawn_points = self.map.get_spawn_points()
        idx = spawn_idx % len(spawn_points)
        transform = spawn_points[idx]

        self.vehicle = self.world.try_spawn_actor(vehicle_bp, transform)
        if self.vehicle is None:
            # Fallback to random unoccupied spawn point
            for sp in spawn_points:
                self.vehicle = self.world.try_spawn_actor(vehicle_bp, sp)
                if self.vehicle is not None:
                    break
        self.actor_list.append(self.vehicle)

        # Build ground-truth route waypoints
        self._build_route(self.vehicle.get_transform().location)

        # Spawn sensors
        self._spawn_sensors(bp_lib)

        # Reset states
        self.velocity = 0.0
        self.prev_steer = 0.0
        self.prev_throttle = 0.0
        self.prev_brake = 0.0
        self.timesteps = 0
        self.stall_steps = 0
        self.distance_covered = 0.0
        self.current_wp_idx = 0
        self.collision_events.clear()

        # Clear any stale queue items from prior episodes
        while not self.camera_queue.empty():
            try:
                self.camera_queue.get_nowait()
            except queue.Empty:
                break
        while not self.lidar_queue.empty():
            try:
                self.lidar_queue.get_nowait()
            except queue.Empty:
                break

        # Prime sensors with initial tick
        frame_id = self.world.tick()
        self._retrieve_sensor_data(frame_id, timeout=3.0)
        while self.latest_rgb is None or self.latest_lidar_points is None:
            frame_id = self.world.tick()
            self._retrieve_sensor_data(frame_id, timeout=3.0)

        sensor_obs = self._get_sensor_obs()
        return sensor_obs

    def _spawn_sensors(self, bp_lib):
        # 1. Front RGB Camera
        cam_bp = bp_lib.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', str(C.IM_WIDTH))
        cam_bp.set_attribute('image_size_y', str(C.IM_HEIGHT))
        cam_bp.set_attribute('fov', str(C.CAMERA_FOV))
        cam_tf = carla.Transform(
            carla.Location(x=C.CAMERA_POS['x'], y=C.CAMERA_POS['y'], z=C.CAMERA_POS['z']),
            carla.Rotation(pitch=C.CAMERA_POS['pitch'], yaw=C.CAMERA_POS['yaw'], roll=C.CAMERA_POS['roll'])
        )
        self.camera_sensor = self.world.spawn_actor(cam_bp, cam_tf, attach_to=self.vehicle)
        weak_self = weakref.ref(self)
        self.camera_sensor.listen(lambda img: CarlaSensoryEnvironment._on_camera(weak_self, img))
        self.sensor_list.append(self.camera_sensor)

        # 2. 3D LiDAR (Raycast)
        lidar_bp = bp_lib.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('range', '50.0')
        lidar_bp.set_attribute('channels', '32')
        lidar_bp.set_attribute('points_per_second', '100000')
        lidar_bp.set_attribute('rotation_frequency', '20')  # Exactly matches 20 Hz tick
        lidar_tf = carla.Transform(carla.Location(x=C.LIDAR_POS['x'], y=C.LIDAR_POS['y'], z=C.LIDAR_POS['z']))
        self.lidar_sensor = self.world.spawn_actor(lidar_bp, lidar_tf, attach_to=self.vehicle)
        self.lidar_sensor.listen(lambda pts: CarlaSensoryEnvironment._on_lidar(weak_self, pts))
        self.sensor_list.append(self.lidar_sensor)

        # 3. Collision Sensor
        col_bp = bp_lib.find('sensor.other.collision')
        col_tf = carla.Transform(carla.Location(x=1.3, z=0.5))
        self.collision_sensor = self.world.spawn_actor(col_bp, col_tf, attach_to=self.vehicle)
        self.collision_sensor.listen(lambda event: CarlaSensoryEnvironment._on_collision(weak_self, event))
        self.sensor_list.append(self.collision_sensor)

    @staticmethod
    def _on_camera(weak_self, img):
        me = weak_self()
        if me is not None:
            me.camera_queue.put(img)

    @staticmethod
    def _on_lidar(weak_self, pts):
        me = weak_self()
        if me is not None:
            me.lidar_queue.put(pts)

    def _retrieve_sensor_data(self, frame_id, timeout=2.0):
        """Thread-safe lockstep barrier: blocks until camera and LiDAR arrive for frame_id."""
        t0 = time.time()
        img_data = None
        lidar_data = None

        while img_data is None or lidar_data is None:
            if time.time() - t0 > timeout:
                break
            if img_data is None:
                try:
                    c = self.camera_queue.get(timeout=0.05)
                    if c.frame >= frame_id:
                        img_data = c
                except queue.Empty:
                    pass
            if lidar_data is None:
                try:
                    l = self.lidar_queue.get(timeout=0.05)
                    if l.frame >= frame_id:
                        lidar_data = l
                except queue.Empty:
                    pass

        if img_data is not None:
            arr = np.frombuffer(img_data.raw_data, dtype=np.uint8).reshape((C.IM_HEIGHT, C.IM_WIDTH, 4))
            self.latest_rgb = arr[:, :, :3][:, :, ::-1].astype(np.float32) / 255.0

        if lidar_data is not None:
            raw = np.frombuffer(lidar_data.raw_data, dtype=np.float32).reshape((-1, 4)).copy()
            # Shift raw points from LiDAR sensor frame into Ego Vehicle Ground frame
            raw[:, 0] += C.LIDAR_POS['x']
            raw[:, 1] += C.LIDAR_POS['y']
            raw[:, 2] += C.LIDAR_POS['z']
            self.latest_lidar_points = raw

    @staticmethod
    def _on_collision(weak_self, event):
        me = weak_self()
        if me is None:
            return
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        me.collision_events.append(intensity)

    def _build_route(self, ego_loc):
        self.route_waypoints = []
        cur_wp = self.map.get_waypoint(ego_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        self.route_waypoints.append(cur_wp)

        for x in range(self.total_route_length):
            nxts = cur_wp.next(1.0)
            if not nxts:
                break
            nxt = nxts[-1] if x < 300 else nxts[0]
            self.route_waypoints.append(nxt)
            cur_wp = nxt

    def step(self, action):
        """
        Executes continuous action [steer in [-1, 1], accel in [-1, 1]].
        Returns (sensor_obs, privileged_state, reward, done, info)
        """
        self.timesteps += 1

        steer_cmd = float(action[0])
        accel_cmd = float(action[1])

        # Smooth action filter
        applied_steer = self.prev_steer * 0.8 + steer_cmd * 0.2

        # Smooth Throttle vs. Brake Separation (smoothly decaying opposite pedal)
        if accel_cmd >= 0.0:
            applied_throttle = self.prev_throttle * 0.8 + accel_cmd * 0.2
            applied_brake = self.prev_brake * 0.8
        else:
            applied_throttle = self.prev_throttle * 0.8
            applied_brake = self.prev_brake * 0.8 + (-accel_cmd) * 0.2

        self.vehicle.apply_control(carla.VehicleControl(
            steer=float(np.clip(applied_steer, -1.0, 1.0)),
            throttle=float(np.clip(applied_throttle, 0.0, 1.0)),
            brake=float(np.clip(applied_brake, 0.0, 1.0))
        ))
        self.prev_steer = applied_steer
        self.prev_throttle = applied_throttle
        self.prev_brake = applied_brake

        # Advance simulator by exactly one tick and retrieve synchronized sensor frames
        frame_id = self.world.tick()
        self._retrieve_sensor_data(frame_id, timeout=2.0)

        # Update telemetry
        vel = self.vehicle.get_velocity()
        self.velocity = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2) * 3.6  # km/h
        ego_tf = self.vehicle.get_transform()
        ego_loc = ego_tf.location

        # Update closest waypoint along route sequentially (prevent skipping curves)
        max_idx = len(self.route_waypoints) - 1
        for i in range(self.current_wp_idx, min(self.current_wp_idx + 4, max_idx)):
            wp_loc = self.route_waypoints[i].transform.location
            dist_to_wp = math.sqrt((ego_loc.x - wp_loc.x)**2 + (ego_loc.y - wp_loc.y)**2)
            if dist_to_wp < 6.0:
                fwd = self.route_waypoints[i].transform.get_forward_vector()
                dot = (fwd.x * (ego_loc.x - wp_loc.x) + fwd.y * (ego_loc.y - wp_loc.y))
                if dot > 0.0:
                    self.current_wp_idx = i

        cur_wp = self.route_waypoints[self.current_wp_idx]
        nxt_wp = self.route_waypoints[min(self.current_wp_idx + 1, max_idx)]

        # Privileged ground-truth metrics (FOR REWARD AND BENCHMARK ONLY)
        lane_dev = self._distance_to_line_2d(cur_wp.transform.location, nxt_wp.transform.location, ego_loc)
        wp_fwd = cur_wp.transform.get_forward_vector()
        veh_fwd = ego_tf.get_forward_vector()
        heading_err = math.degrees(math.acos(np.clip(wp_fwd.x * veh_fwd.x + wp_fwd.y * veh_fwd.y, -1.0, 1.0)))

        route_comp = float(self.current_wp_idx) / float(max(1, len(self.route_waypoints)))
        self.distance_covered = float(self.current_wp_idx)

        # Standstill / Stall detection
        if self.velocity < C.STALL_SPEED_THRESH:
            self.stall_steps += 1
        else:
            self.stall_steps = 0

        # Check termination criteria
        done = False
        term_reason = 'running'
        reward = 0.0

        if len(self.collision_events) > 0:
            done = True
            term_reason = 'collision'
            reward = -10.0
        elif lane_dev > C.MAX_CENTER_DEV:
            done = True
            term_reason = 'lane_departure'
            reward = -10.0
        elif self.stall_steps >= C.MAX_STALL_STEPS:
            done = True
            term_reason = 'stall'
            reward = C.STALL_PENALTY
        elif route_comp >= 0.95:
            done = True
            term_reason = 'route_complete'
            reward = 100.0
        elif self.timesteps >= C.MAX_STEPS_PER_EP:
            done = True
            term_reason = 'max_steps'

        # Continuous driving reward
        if not done:
            centering_factor = max(0.0, 1.0 - (lane_dev / C.MAX_CENTER_DEV))
            angle_factor = max(0.0, 1.0 - (abs(heading_err) / 20.0))
            speed_factor = min(self.velocity / C.TARGET_SPEED, 1.0)
            reward = speed_factor * centering_factor * angle_factor
            # Idle penalty to discourage standstill exploitation
            if self.velocity < C.STALL_SPEED_THRESH:
                reward -= 0.05

        privileged_state = dict(
            lane_dev=lane_dev,
            heading_err=heading_err,
            route_comp=route_comp,
            distance_covered=self.distance_covered,
            term_reason=term_reason,
            collided=len(self.collision_events) > 0
        )

        sensor_obs = self._get_sensor_obs()

        info = {
            'route_completion': route_comp,
            'distance_covered': self.distance_covered,
            'center_lane_deviation': lane_dev,
            'termination_reason': term_reason,
            'collided': len(self.collision_events) > 0,
            'mean_speed_kmh': self.velocity
        }

        return sensor_obs, privileged_state, reward, done, info

    def _get_sensor_obs(self):
        """Constructs policy-visible observation record (Tier S) with NO telemetry leakage."""
        ego_tf = self.vehicle.get_transform()
        ego_loc = ego_tf.location
        fwd_vec = ego_tf.get_forward_vector()
        right_vec = ego_tf.get_right_vector()

        # Telemetry: onboard IMU and wheel speed only (NO absolute compass yaw)
        ang_vel = self.vehicle.get_angular_velocity()
        yaw_rate_norm = float(np.clip(ang_vel.z / 50.0, -1.0, 1.0))

        # Vehicle accelerations projected into ego body frame (Forward +X, Right +Y)
        accel = self.vehicle.get_acceleration()
        accel_fwd = (accel.x * fwd_vec.x + accel.y * fwd_vec.y + accel.z * fwd_vec.z)
        accel_right = (accel.x * right_vec.x + accel.y * right_vec.y + accel.z * right_vec.z)

        # 8-dim Ego Telemetry
        ego_vec = np.array([
            self.velocity / 30.0,
            float(np.clip(accel_fwd / 10.0, -1.0, 1.0)),
            float(np.clip(accel_right / 10.0, -1.0, 1.0)),
            yaw_rate_norm,
            self.prev_steer,
            self.prev_throttle,
            self.prev_brake,
            1.0  # Forward Gear flag
        ], dtype=np.float32)

        # 8-dim GPS Navigation Target (sampled ~30m ahead with 1.0m sensor noise)
        target_idx = min(self.current_wp_idx + 30, len(self.route_waypoints) - 1)
        tgt_loc = self.route_waypoints[target_idx].transform.location

        # World displacement vector
        dx_w = (tgt_loc.x - ego_loc.x) + np.random.normal(0.0, 1.0)
        dy_w = (tgt_loc.y - ego_loc.y) + np.random.normal(0.0, 1.0)
        dz_w = (tgt_loc.z - ego_loc.z)

        # Project into Ego Vehicle Frame (Forward +X, Right +Y)
        dx_ego = dx_w * fwd_vec.x + dy_w * fwd_vec.y + dz_w * fwd_vec.z
        dy_ego = dx_w * right_vec.x + dy_w * right_vec.y + dz_w * right_vec.z

        nav_vec = np.zeros(8, dtype=np.float32)
        nav_vec[0] = float(np.clip(dx_ego / 50.0, -1.0, 1.0))
        nav_vec[1] = float(np.clip(dy_ego / 50.0, -1.0, 1.0))
        nav_vec[2] = 1.0  # One-hot: Follow Lane command

        return {
            'rgb': self.latest_rgb,
            'lidar': self.latest_lidar_points,
            'ego': ego_vec,
            'nav': nav_vec
        }

    @staticmethod
    def _distance_to_line_2d(A, B, p):
        num = abs((B.y - A.y) * p.x - (B.x - A.x) * p.y + B.x * A.y - B.y * A.x)
        den = math.sqrt((B.y - A.y)**2 + (B.x - A.x)**2)
        return float(num / (den + 1e-8))

    def _teardown(self):
        for s in self.sensor_list:
            if s is not None and s.is_alive:
                s.destroy()
        self.sensor_list.clear()

        for a in self.actor_list:
            if a is not None and a.is_alive:
                a.destroy()
        self.actor_list.clear()

    def close(self):
        self._teardown()
        try:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)
        except Exception:
            pass
