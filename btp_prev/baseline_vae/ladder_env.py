"""
CARLA environment for the VAE baseline ladder (rungs 1-3).

Subclasses main.py's CarlaEnvironment so the *dynamics* -- route generation,
waypoint tracking, reward shaping, termination thresholds, pedestrian/NPC
spawning -- are literally the same code across all three rungs. Only two
things are overridden:

  reset() / step()   to drive an N-camera rig (+ optional LiDAR) instead of
                     the single hard-coded front camera, and to return the
                     observation in the layout EncodeStateVAE expects.

  info               to carry the metrics the ablation reports. main.py
                     returned only [distance_covered, center_lane_deviation],
                     which is why the existing Results_05 CSV has no collision
                     rate, no success rate and no route completion. Those are
                     now first-class.

The reward function is NOT touched. Copying it rather than importing it would
have been the easy way to accidentally desync the rungs; overriding step()
keeps one definition of reward in main.py:318-332 and reuses it verbatim.
"""
import time

import numpy as np

import carla

from main import CarlaEnvironment, CollisionSensor, CameraSensorEnv
from multi_sensor import MultiCameraSensor, LidarBEVSensor


# Termination reasons, in the order main.py tests them.
COLLISION      = 'collision'
LANE_DEPARTURE = 'lane_departure'
STALL          = 'stall'
OVERSPEED      = 'overspeed'
ROUTE_COMPLETE = 'route_complete'
STEP_LIMIT     = 'step_limit'
RUNNING        = 'running'


class LadderEnvironment(CarlaEnvironment):

    def __init__(self, client, world, town, params,
                 checkpoint_frequency=100, continuous_action=True):
        self.p = params
        self._original_settings = world.get_settings()
        self._apply_determinism(client, world, params)
        super().__init__(client, world, town, checkpoint_frequency, continuous_action)
        self.lidar_obj = None
        self.termination_reason = RUNNING
        self.collision_count = 0
        self.spawn_point_override = None      # set by run_ladder for multi-route eval
        if params.ENABLE_NPC_VEHICLES:
            # main.py defines set_other_vehicles() but never calls it. Opt-in.
            self.set_other_vehicles()

    # ------------------------------------------------------------------
    @staticmethod
    def _apply_determinism(client, world, params):
        """Synchronous mode + seeded RNGs, so all rungs see the same world.

        Without this the server free-runs while Python encodes, and the rung
        with the slower encoder gets more simulated time per action -- the
        driving comparison would then be partly a latency comparison.
        """
        settings = world.get_settings()
        if params.SYNCHRONOUS_MODE:
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = params.FIXED_DELTA_SECONDS
        else:
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
        world.apply_settings(settings)

        try:
            world.set_pedestrians_seed(params.CARLA_SEED)
        except AttributeError:
            pass                        # older CARLA builds

        tm = client.get_trafficmanager(params.CARLA_TM_PORT)
        tm.set_random_device_seed(params.TRAFFIC_MANAGER_SEED)
        if params.SYNCHRONOUS_MODE:
            tm.set_synchronous_mode(True)

        print(f"[ladder_env] synchronous={params.SYNCHRONOUS_MODE} "
              f"dt={params.FIXED_DELTA_SECONDS} "
              f"carla_seed={params.CARLA_SEED} "
              f"tm_seed={params.TRAFFIC_MANAGER_SEED} "
              f"npc_vehicles={params.ENABLE_NPC_VEHICLES}")

    def _tick(self):
        if self.p.SYNCHRONOUS_MODE:
            self.world.tick()

    def restore_settings(self):
        """Put the server back to async so the next process is not stuck."""
        try:
            self.world.apply_settings(self._original_settings)
            self.client.get_trafficmanager(self.p.CARLA_TM_PORT).set_synchronous_mode(False)
        except Exception:
            pass

    # ------------------------------------------------------------------
    def _spawn_sensors(self):
        self.camera_obj = MultiCameraSensor(self.vehicle, self.p)
        self._tick()
        while not self.camera_obj.ready():
            self._tick() if self.p.SYNCHRONOUS_MODE else time.sleep(0.0001)
        self.sensor_list.extend(self.camera_obj.sensors)

        if self.p.USE_LIDAR_VAE:
            self.lidar_obj = LidarBEVSensor(self.vehicle, self.p)
            self._tick()
            while not self.lidar_obj.ready():
                self._tick() if self.p.SYNCHRONOUS_MODE else time.sleep(0.0001)
            self.sensor_list.append(self.lidar_obj.sensor)

    def _observation(self):
        """Pack sensors into the layout EncodeStateVAE.process expects."""
        frames = self.camera_obj.get_frames()
        # Rung 1 keeps the original single-array form, so the encoder path is
        # byte-identical to the baseline rather than a length-1 list.
        images = frames[0] if self.p.NUM_CAMERAS_VAE == 1 else frames
        if self.p.USE_LIDAR_VAE:
            return [images, self.lidar_obj.get_frame(), self.navigation_obs]
        return [images, self.navigation_obs]

    # ------------------------------------------------------------------
    def reset(self):
        try:
            if len(self.actor_list) != 0 or len(self.sensor_list) != 0:
                self.client.apply_batch([carla.command.DestroyActor(x) for x in self.sensor_list])
                self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
                self.sensor_list.clear()
                self.actor_list.clear()

            self.remove_sensors()

            vehicle_bp = self.get_vehicle(self.p.CAR_NAME)

            spawn_points = self.map.get_spawn_points()
            if self.town == "Town07":
                default_idx, self.total_distance = 20, 750
            elif self.town == "Town02":
                default_idx, self.total_distance = 30, self.p.ROUTE_LENGTH_TOWN02
            else:
                default_idx, self.total_distance = 12, self.p.ROUTE_LENGTH_TOWN01

            # Same index for every rung; only run_ladder's --eval-routes changes it,
            # and it walks the same list in the same order for all three rungs.
            idx = default_idx if self.spawn_point_override is None \
                else self.spawn_point_override % len(spawn_points)
            self.current_spawn_index = idx
            transform = spawn_points[idx]

            self.vehicle = self.world.try_spawn_actor(vehicle_bp, transform)
            self.actor_list.append(self.vehicle)

            self._spawn_sensors()

            if self.display_on:
                self.env_camera_obj = CameraSensorEnv(self.vehicle)
                self.sensor_list.append(self.env_camera_obj.sensor)

            self.collision_obj = CollisionSensor(self.vehicle)
            self.collision_history = self.collision_obj.collision_data
            self.sensor_list.append(self.collision_obj.sensor)

            self.timesteps = 0
            self.rotation = self.vehicle.get_transform().rotation.yaw
            self.previous_location = self.vehicle.get_location()
            self.distance_traveled = 0.0
            self.center_lane_deviation = 0.0
            self.target_speed = self.p.TARGET_SPEED
            self.max_speed = self.p.MAX_SPEED
            self.min_speed = self.p.MIN_SPEED
            self.max_distance_from_center = self.p.MAX_DISTANCE_FROM_CENTER
            self.throttle = float(0.0)
            self.brake = float(0.0)
            self.previous_steer = float(0.0)
            self.velocity = float(0.0)
            self.distance_from_center = float(0.0)
            self.angle = float(0.0)
            self.distance_covered = 0.0

            self.termination_reason = RUNNING
            self.collision_count = 0
            self.speed_history = []

            if self.fresh_start:
                self.current_waypoint_index = 0
                self.route_waypoints = list()
                self.waypoint = self.map.get_waypoint(
                    self.vehicle.get_location(), project_to_road=True,
                    lane_type=carla.LaneType.Driving)
                current_waypoint = self.waypoint
                self.route_waypoints.append(current_waypoint)

                for x in range(self.total_distance):
                    nxts = current_waypoint.next(1.0)
                    if not nxts:
                        # Dead end. Only reachable with a non-default spawn point;
                        # the route is simply shorter, and it is the SAME shorter
                        # route for every rung because the index list is shared.
                        print(f"  [route] dead end after {x} m from spawn "
                              f"{self.current_spawn_index}; route truncated")
                        break
                    if self.town == "Town07":
                        nxt = nxts[0] if x < 650 else nxts[-1]
                    elif self.town == "Town02":
                        nxt = nxts[-1] if x > 100 else nxts[0]
                    else:
                        nxt = nxts[-1] if x < 300 else nxts[0]
                    self.route_waypoints.append(nxt)
                    current_waypoint = nxt
            else:
                waypoint = self.route_waypoints[self.checkpoint_waypoint_index % len(self.route_waypoints)]
                self.vehicle.set_transform(waypoint.transform)
                self.current_waypoint_index = self.checkpoint_waypoint_index

            self.route_start_index = self.current_waypoint_index
            normalized_velocity = float(np.clip(self.velocity / self.target_speed, 0.0, 2.0))
            normalized_distance_from_center = float(np.clip(self.distance_from_center / self.max_distance_from_center, -1.0, 1.0))
            normalized_angle = float(np.clip(self.angle / np.deg2rad(20), -1.0, 1.0))
            self.navigation_obs = np.array(
                [self.throttle, self.velocity, normalized_velocity,
                 normalized_distance_from_center, normalized_angle], dtype=np.float32)
            # Settle the physics. main.py used time.sleep(0.5), which advances a
            # wall-clock-dependent number of frames; a fixed tick count is
            # identical for every rung.
            if self.p.SYNCHRONOUS_MODE:
                for _ in range(int(0.5 / self.p.FIXED_DELTA_SECONDS)):
                    self.world.tick()
            else:
                time.sleep(0.5)
            self.collision_history.clear()

            self.episode_start_time = time.time()
            return self._observation()

        except Exception as e:
            print(f"Error while Resetting : {e}")
            self._emergency_cleanup()
            raise

    # ------------------------------------------------------------------
    def step(self, action_idx):
        try:
            self.timesteps += 1
            self.fresh_start = False

            velocity = self.vehicle.get_velocity()
            self.velocity = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) * 3.6
            self.speed_history.append(self.velocity)

            if self.continous_action_space:
                steer = max(min(float(action_idx[0]), 1.0), -1.0)
                # Map [-1,1] → [0,1] exactly like main.py (Rung 1, 39% completion).
                # Random init (tanh output ~0, noise std=0.2) → throttle ~0.5 → car drives.
                # No brake channel: deceleration comes from drag + low throttle.
                throttle = float((action_idx[1] + 1.0) / 2.0)
                throttle = max(min(throttle, 1.0), 0.0)

                applied_steer = self.previous_steer * 0.6 + steer * 0.4
                applied_throttle = self.throttle * 0.7 + throttle * 0.3

                self.vehicle.apply_control(carla.VehicleControl(
                    steer=applied_steer,
                    throttle=applied_throttle,
                    brake=0.0))
                self.previous_steer = applied_steer
                self.throttle = applied_throttle

            if self.vehicle.is_at_traffic_light():
                tl = self.vehicle.get_traffic_light()
                if tl.get_state() == carla.TrafficLightState.Red:
                    tl.set_state(carla.TrafficLightState.Green)

            # Advance the simulation by exactly one fixed step. Every rung gets
            # the same simulated dt per action regardless of encoder cost.
            self._tick()

            self.collision_history = self.collision_obj.collision_data
            self.rotation = self.vehicle.get_transform().rotation.yaw
            self.location = self.vehicle.get_location()

            waypoint_index = self.current_waypoint_index
            for _ in range(len(self.route_waypoints)):
                next_waypoint_index = waypoint_index + 1
                wp = self.route_waypoints[next_waypoint_index % len(self.route_waypoints)]
                dot = np.dot(self.vector(wp.transform.get_forward_vector())[:2],
                             self.vector(self.location - wp.transform.location)[:2])
                if dot > 0.0:
                    waypoint_index += 1
                else:
                    break

            self.current_waypoint_index = waypoint_index
            self.current_waypoint = self.route_waypoints[self.current_waypoint_index % len(self.route_waypoints)]
            self.next_waypoint = self.route_waypoints[(self.current_waypoint_index + 1) % len(self.route_waypoints)]

            self.distance_from_center = self.distance_to_line(
                self.vector(self.current_waypoint.transform.location),
                self.vector(self.next_waypoint.transform.location),
                self.vector(self.location))
            self.center_lane_deviation += self.distance_from_center

            fwd = self.vector(self.vehicle.get_velocity())
            wp_fwd = self.vector(self.current_waypoint.transform.rotation.get_forward_vector())
            self.angle = self.angle_diff(fwd, wp_fwd)

            done = False
            reward = 0

            # --- termination criteria --------------------------------------
            if len(self.collision_history) != 0:
                done, reward = True, -10
                self.termination_reason = COLLISION
                self.collision_count = len(self.collision_history)
            elif self.distance_from_center > self.max_distance_from_center:
                done, reward = True, -10
                self.termination_reason = LANE_DEPARTURE
            elif self._elapsed_sim_s() > 10.0 and self.velocity < 1.0:
                done, reward = True, -10
                self.termination_reason = STALL
            elif self.velocity > 60.0:
                # Extreme runaway overspeed safeguard (not normal highway cruising)
                done, reward = True, -10
                self.termination_reason = OVERSPEED

            centering_factor = max(1.0 - self.distance_from_center / self.max_distance_from_center, 0.0)
            angle_factor = max(1.0 - abs(self.angle / np.deg2rad(20)), 0.0)

            if not done:
                if self.velocity < self.min_speed:
                    reward = (self.velocity / self.min_speed) * centering_factor * angle_factor
                elif self.velocity <= self.target_speed:
                    reward = 1.0 * centering_factor * angle_factor
                elif self.velocity <= self.max_speed:
                    reward = (1.0 - (self.velocity - self.target_speed) /
                              (self.max_speed - self.target_speed)) * centering_factor * angle_factor
                else:
                    # velocity > max_speed (35 km/h): negative speeding penalty, removes reward hack
                    reward = -1.0 * min((self.velocity - self.max_speed) / 10.0, 2.0)

            if self.current_waypoint_index >= len(self.route_waypoints) - 2:
                done = True
                self.termination_reason = ROUTE_COMPLETE
                self.fresh_start = True
                if self.checkpoint_frequency is not None:
                    if self.checkpoint_frequency < self.total_distance // 2:
                        self.checkpoint_frequency += 2
                    else:
                        self.checkpoint_frequency = None
                        self.checkpoint_waypoint_index = 0

            while not self.camera_obj.ready():
                self._tick() if self.p.SYNCHRONOUS_MODE else time.sleep(0.0001)
            if self.p.USE_LIDAR_VAE:
                while not self.lidar_obj.ready():
                    self._tick() if self.p.SYNCHRONOUS_MODE else time.sleep(0.0001)

            wp_vec = self.vector(self.next_waypoint.transform.location)[:2] - self.vector(self.current_waypoint.transform.location)[:2]
            veh_vec = self.vector(self.location)[:2] - self.vector(self.current_waypoint.transform.location)[:2]
            cross_2d = float(wp_vec[0] * veh_vec[1] - wp_vec[1] * veh_vec[0])
            lat_sign = 1.0 if cross_2d >= 0.0 else -1.0
            signed_distance = lat_sign * self.distance_from_center

            normalized_velocity = float(np.clip(self.velocity / self.target_speed, 0.0, 2.0))
            normalized_distance_from_center = float(np.clip(signed_distance / self.max_distance_from_center, -1.0, 1.0))
            normalized_angle = float(np.clip(self.angle / np.deg2rad(20), -1.0, 1.0))
            self.navigation_obs = np.array(
                [self.throttle, self.velocity, normalized_velocity,
                 normalized_distance_from_center, normalized_angle], dtype=np.float32)

            obs = self._observation()
            info = self._info(done)

            if done:
                self.center_lane_deviation = self.center_lane_deviation / max(self.timesteps, 1)
                info = self._info(done)
                self._teardown()

            return obs, reward, done, info

        except Exception as e:
            print(f"Error while step : {e}")
            self._emergency_cleanup()
            raise

    # ------------------------------------------------------------------
    def _elapsed_sim_s(self):
        """Seconds of SIMULATED time since episode start.

        The stall check in main.py used wall-clock (`episode_start_time + 10 <
        time.time()`). Under a slower encoder that fires after fewer simulated
        seconds, so rung 3 would be judged stalled earlier than rung 1 for the
        same driving. In synchronous mode simulated time is exact.
        """
        if self.p.SYNCHRONOUS_MODE:
            return self.timesteps * self.p.FIXED_DELTA_SECONDS
        return time.time() - self.episode_start_time

    def _info(self, done):
        route_len = max(len(self.route_waypoints) - 2, 1)
        advanced = self.current_waypoint_index - self.route_start_index
        completion = float(np.clip(advanced / max(route_len - self.route_start_index, 1), 0.0, 1.0))
        self.distance_covered = abs(self.current_waypoint_index - self.checkpoint_waypoint_index)

        collided = self.termination_reason == COLLISION
        success = (self.termination_reason == ROUTE_COMPLETE) and not collided

        return dict(
            distance_covered=self.distance_covered,          # metres (1 m per waypoint)
            center_lane_deviation=self.center_lane_deviation,
            route_completion=completion,
            termination_reason=self.termination_reason,
            collided=bool(collided),
            collision_count=int(self.collision_count),
            success=bool(success),
            timesteps=int(self.timesteps),
            spawn_index=int(getattr(self, 'current_spawn_index', -1)),
            route_length=int(len(self.route_waypoints)),
            mean_speed_kmh=float(np.mean(self.speed_history)) if self.speed_history else 0.0,
        )

    def _teardown(self):
        for sensor in self.sensor_list:
            if sensor is not None and sensor.is_alive:
                sensor.destroy()
        self.sensor_list.clear()
        self.remove_sensors()
        for actor in self.actor_list:
            if actor is not None and actor.is_alive:
                actor.destroy()
        self.actor_list.clear()

    def _emergency_cleanup(self):
        try:
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.sensor_list])
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.walker_list])
        except Exception:
            pass
        self.sensor_list.clear()
        self.actor_list.clear()
        self.remove_sensors()

    def remove_sensors(self):
        super().remove_sensors()
        self.lidar_obj = None
