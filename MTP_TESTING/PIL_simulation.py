import os
import sys
import glob
import math
import weakref
import pygame
import time
import random
import struct
import csv
import cv2
import pickle
import socket
import time
import math
import numpy as np
import pandas as pd
import logging
from datetime import datetime
import tensorflow as tf

from parameters import*
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((EDGE_IP, PORT))
server_socket.listen(1)
print()
print(f"waiting for connection on {EDGE_IP}:{PORT}.....")

client_socket, client_address = server_socket.accept()
print(f"Connection established with {client_address}:{client_socket}")



try:
    sys.path.append(glob.glob('./carla/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    print('Couldn\'t import Carla egg properly')


import carla

class ClientConnection:

    def __init__(self):
        self.host ="localhost"
        self.town = TOWN
        self.client = None
        self.port = 2000
        self.timeout = 40.0

    def setup(self):
        try:
            self.client = carla.Client(self.host, self.port)
            self.client.set_timeout(self.timeout)
            self.world = self.client.load_world(self.town)
            self.world.set_weather(carla.WeatherParameters.CloudyNoon)

            return self.client, self.world

        except Exception as e:
            print('Failed to make a connection with the server: {}'.format(e))

            if self.client.get_client_version != self.client.get_server_version:
                print("There is a Client and Server version mismatch! Please install or download the right versions.")



class CarlaEnvironment():

    def __init__(self, client, world, town, checkpoint_frequency=100, continuous_action=True) -> None:


        self.client = client
        self.world = world
        self.blueprint_library = self.world.get_blueprint_library()
        self.map = self.world.get_map()
        self.action_space = self.get_discrete_action_space()
        self.continous_action_space = True
        self.display_on = VISUAL_DISPLAY
        self.vehicle = None
        self.settings = None
        self.current_waypoint_index = 0
        self.checkpoint_waypoint_index = 0
        self.fresh_start=True
        self.checkpoint_frequency = checkpoint_frequency
        self.route_waypoints = None
        self.town = town
        
        # Objects to be kept alive
        self.camera_obj = None
        self.env_camera_obj = None
        self.collision_obj = None
        self.lane_invasion_obj = None

        # Two very important lists for keeping track of our actors and their observations.
        self.sensor_list = list()
        self.actor_list = list()
        self.walker_list = list()
        self.create_pedestrians()

    def reset(self):

        try:
            
            if len(self.actor_list) != 0 or len(self.sensor_list) != 0:
                self.client.apply_batch([carla.command.DestroyActor(x) for x in self.sensor_list])
                self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
                self.sensor_list.clear()
                self.actor_list.clear()
            
            self.remove_sensors()


            # Blueprint of our main vehicle
            vehicle_bp = self.get_vehicle(CAR_NAME)

            if self.town == "Town07":
                transform = self.map.get_spawn_points()[20] #Town7  is 38 
                self.total_distance = 750
            elif self.town == "Town02":
                transform = self.map.get_spawn_points()[30] #Town2 is 30
                self.total_distance = 500
                #self.total_distance = 200

            else:
                transform = self.map.get_spawn_points()[12] #40 nocd 
                self.total_distance = 500

            self.vehicle = self.world.try_spawn_actor(vehicle_bp, transform)
            if self.vehicle is None:
                for sp in self.map.get_spawn_points():
                    self.vehicle = self.world.try_spawn_actor(vehicle_bp, sp)
                    if self.vehicle is not None:
                        break
            if self.vehicle is None:
                raise RuntimeError("Failed to spawn ego vehicle at any spawn point!")
            self.actor_list.append(self.vehicle)

            # Camera Sensor
            self.camera_obj = CameraSensor(self.vehicle)
            t_wait = time.time()
            while(len(self.camera_obj.front_camera) == 0):
                time.sleep(0.002)
                if time.time() - t_wait > 5.0:
                    print("[WARNING] Front camera wait timed out during reset!")
                    break
            if len(self.camera_obj.front_camera) > 0:
                self.image_obs = self.camera_obj.front_camera.pop(-1)
            else:
                self.image_obs = np.zeros((80, 160, 3), dtype=np.uint8)
            self.sensor_list.append(self.camera_obj.sensor)

            # Third person view of our vehicle in the Simulated env
            if self.display_on:
                self.env_camera_obj = CameraSensorEnv(self.vehicle)
                self.sensor_list.append(self.env_camera_obj.sensor)

            # Collision sensor
            self.collision_obj = CollisionSensor(self.vehicle)
            self.collision_history = self.collision_obj.collision_data
            self.sensor_list.append(self.collision_obj.sensor)

            
            self.timesteps = 0
            self.rotation = self.vehicle.get_transform().rotation.yaw
            self.previous_location = self.vehicle.get_location()
            self.distance_traveled = 0.0
            self.center_lane_deviation = 0.0
            self.target_speed = 22 #km/h
            self.max_speed = 40.0
            self.min_speed = 15.0
            self.max_distance_from_center = 3
            self.throttle = float(0.0)
            self.previous_steer = float(0.0)
            self.velocity = float(0.0)
            self.distance_from_center = float(0.0)
            self.angle = float(0.0)
            self.center_lane_deviation = 0.0
            self.distance_covered = 0.0


            if self.fresh_start:
                #print("fresh start")

                self.current_waypoint_index = 0
                self.route_waypoints = list()
                self.waypoint = self.map.get_waypoint(self.vehicle.get_location(), project_to_road=True, lane_type=(carla.LaneType.Driving))
                current_waypoint = self.waypoint
                self.route_waypoints.append(current_waypoint)

                for x in range(self.total_distance):
                    if self.town == "Town07":
                        if x < 650:
                            next_waypoint = current_waypoint.next(1.0)[0]
                        else:
                            next_waypoint = current_waypoint.next(1.0)[-1]
                    elif self.town == "Town02": #200
                        # if x < 200:
                        #     next_waypoint = current_waypoint.next(1.0)[-1]
                        # else:
                        #     next_waypoint = current_waypoint.next(1.0)[0]
                        if x > 100:
                            next_waypoint = current_waypoint.next(1.0)[-1]
                        else:
                            next_waypoint = current_waypoint.next(1.0)[0]
                        

                    else:
                        if x < 300:
                            next_waypoint = current_waypoint.next(1.0)[-1]
                        else:
                            next_waypoint = current_waypoint.next(1.0)[0]

                    self.route_waypoints.append(next_waypoint)
                    current_waypoint = next_waypoint

 
            
            else:
                #print("waypoint update")
                # Teleport vehicle to last checkpoint
                waypoint = self.route_waypoints[self.checkpoint_waypoint_index % len(self.route_waypoints)]
                transform = waypoint.transform
                self.vehicle.set_transform(transform)
                self.current_waypoint_index = self.checkpoint_waypoint_index


            if self.vehicle is not None:
                try:
                    veh_tf = self.vehicle.get_transform()
                    spec_tf = carla.Transform(
                        veh_tf.location + veh_tf.get_forward_vector() * (-6.0) + carla.Location(z=2.8),
                        carla.Rotation(pitch=-12, yaw=veh_tf.rotation.yaw)
                    )
                    self.world.get_spectator().set_transform(spec_tf)
                except Exception:
                    pass

            self.navigation_obs = np.array([self.throttle, self.velocity, self.previous_steer, self.distance_from_center, self.angle])                        
            time.sleep(0.5)
            self.collision_history.clear()

            self.episode_start_time = time.time()
            return [self.image_obs, self.navigation_obs]

        except Exception as e:
            print(f"Error while Resetting : {e}")
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.sensor_list])
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.walker_list])
            self.sensor_list.clear()
            self.actor_list.clear()
            self.remove_sensors()
            if self.display_on:
                pygame.quit()

    def step(self, action_idx):
        try:

            self.timesteps+=1
            self.fresh_start = False

 
            velocity = self.vehicle.get_velocity()
            self.velocity = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) * 3.6
            
     
            if self.continous_action_space:
                steer = float(action_idx[0])
                steer = max(min(steer, 1.0), -1.0)
                throttle = float((action_idx[1] + 1.0)/2)
                throttle = max(min(throttle, 1.0), 0.0) 
                #throttle = max(min(throttle, 1.0), 0.0)
                self.vehicle.apply_control(carla.VehicleControl(steer=self.previous_steer*0.9 + steer*0.1, throttle=self.throttle*0.9 + throttle*0.1))
                self.previous_steer = steer
                self.throttle = throttle

            

            if self.vehicle.is_at_traffic_light():
                traffic_light = self.vehicle.get_traffic_light()
                if traffic_light.get_state() == carla.TrafficLightState.Red:
                    traffic_light.set_state(carla.TrafficLightState.Green)


            self.collision_history = self.collision_obj.collision_data            
            self.rotation = self.vehicle.get_transform().rotation.yaw
            self.location = self.vehicle.get_location()

            if self.vehicle is not None:
                try:
                    veh_tf = self.vehicle.get_transform()
                    spec_tf = carla.Transform(
                        veh_tf.location + veh_tf.get_forward_vector() * (-6.0) + carla.Location(z=2.8),
                        carla.Rotation(pitch=-12, yaw=veh_tf.rotation.yaw)
                    )
                    self.world.get_spectator().set_transform(spec_tf)
                except Exception:
                    pass

            waypoint_index = self.current_waypoint_index
            for _ in range(len(self.route_waypoints)):
                next_waypoint_index = waypoint_index + 1
                wp = self.route_waypoints[next_waypoint_index % len(self.route_waypoints)]
                dot = np.dot(self.vector(wp.transform.get_forward_vector())[:2],self.vector(self.location - wp.transform.location)[:2])
                if dot > 0.0:
                    waypoint_index += 1
                else:
                    break

            self.current_waypoint_index = waypoint_index
            self.current_waypoint = self.route_waypoints[ self.current_waypoint_index    % len(self.route_waypoints)]
            self.next_waypoint = self.route_waypoints[(self.current_waypoint_index+1) % len(self.route_waypoints)]

            self.distance_from_center = self.distance_to_line(self.vector(self.current_waypoint.transform.location),self.vector(self.next_waypoint.transform.location),self.vector(self.location))
            self.center_lane_deviation += self.distance_from_center

           
            fwd    = self.vector(self.vehicle.get_velocity())
            wp_fwd = self.vector(self.current_waypoint.transform.rotation.get_forward_vector())
            self.angle  = self.angle_diff(fwd, wp_fwd)

            #Update checkpoint for training
            # if not self.fresh_start:
            #     if self.checkpoint_frequency is not None:
            #         self.checkpoint_waypoint_index = (self.current_waypoint_index // self.checkpoint_frequency) * self.checkpoint_frequency

            
            done = False
            reward = 0
            self.termination_reason = "RUNNING"

            if len(self.collision_history) != 0:
                self.termination_reason = "COLLISION"
                done = True
                reward = -10
            elif self.distance_from_center > self.max_distance_from_center:
                self.termination_reason = "OFF_LANE"
                done = True
                reward = -10
            elif self.episode_start_time + 10 < time.time() and self.velocity < 1.0:
                self.termination_reason = "STALLED"
                reward = -10
                done = True
            elif self.velocity > self.max_speed:
                self.termination_reason = "OVERSPEED"
                reward = -10
                done = True

            centering_factor = max(1.0 - self.distance_from_center / self.max_distance_from_center, 0.0)
            angle_factor = max(1.0 - abs(self.angle / np.deg2rad(20)), 0.0)

            if not done:
                if self.continous_action_space:
                    if self.velocity < self.min_speed:
                        reward = (self.velocity / self.min_speed) * centering_factor * angle_factor    
                    elif self.velocity > self.target_speed:           
                        reward = (1.0 - (self.velocity-self.target_speed) / (self.max_speed-self.target_speed)) * centering_factor * angle_factor  
                    else:                                       
                        reward = 1.0 * centering_factor * angle_factor 
                else:
                    reward = 1.0 * centering_factor * angle_factor

            if self.timesteps >= 2e6:
                self.termination_reason = "MAX_TIMESTEPS"
                done = True
            elif self.current_waypoint_index >= len(self.route_waypoints) - 2:
                print("Reached destination -- Repeat")
                self.termination_reason = "REACHED_DESTINATION"
                done = True
                self.fresh_start = True
                if self.checkpoint_frequency is not None:
                    if self.checkpoint_frequency < self.total_distance//2:
                        self.checkpoint_frequency += 2
                    else:
                        self.checkpoint_frequency = None
                        self.checkpoint_waypoint_index = 0

            t_wait = time.time()
            while(len(self.camera_obj.front_camera) == 0):
                time.sleep(0.001)
                if time.time() - t_wait > 3.0:
                    print("[WARNING] Front camera wait timed out during step!")
                    break

            if len(self.camera_obj.front_camera) > 0:
                self.image_obs = self.camera_obj.front_camera.pop(-1)
            else:
                self.image_obs = np.zeros((80, 160, 3), dtype=np.uint8)

            normalized_velocity = self.velocity/self.target_speed
            normalized_distance_from_center = self.distance_from_center / self.max_distance_from_center
            normalized_angle = abs(self.angle / np.deg2rad(20))
            self.navigation_obs = np.array([self.throttle, self.velocity, normalized_velocity, normalized_distance_from_center, normalized_angle])

            if done:
                self.center_lane_deviation = self.center_lane_deviation / max(self.timesteps, 1)
                self.distance_covered = abs(self.current_waypoint_index - self.checkpoint_waypoint_index)
                
                for sensor in self.sensor_list:
                    try:
                        sensor.destroy()
                    except:
                        pass
                
                self.remove_sensors()
                
                for actor in self.actor_list:
                    try:
                        actor.destroy()
                    except:
                        pass
            
            return [self.image_obs, self.navigation_obs], reward, done, [self.distance_covered, self.center_lane_deviation, self.termination_reason]

        except Exception as e:
            print(f"Error while step  : {e}")
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.sensor_list])
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.walker_list])
            self.sensor_list.clear()
            self.actor_list.clear()
            self.remove_sensors()
            if self.display_on:
                pygame.quit()


    def create_pedestrians(self):
        try:

            # Our code for this method has been broken into 3 sections.

            # 1. Getting the available spawn points in  our world.
            # Random Spawn locations for the walker
            walker_spawn_points = []
            for i in range(NUMBER_OF_PEDESTRIAN):
                spawn_point_ = carla.Transform()
                loc = self.world.get_random_location_from_navigation()
                if (loc != None):
                    spawn_point_.location = loc
                    walker_spawn_points.append(spawn_point_)

            # 2. We spawn the walker actor and ai controller
            # Also set their respective attributes
            for spawn_point_ in walker_spawn_points:
                walker_bp = random.choice(
                    self.blueprint_library.filter('walker.pedestrian.*'))
                walker_controller_bp = self.blueprint_library.find(
                    'controller.ai.walker')
                # Walkers are made visible in the simulation
                if walker_bp.has_attribute('is_invincible'):
                    walker_bp.set_attribute('is_invincible', 'false')
                # They're all walking not running on their recommended speed
                if walker_bp.has_attribute('speed'):
                    walker_bp.set_attribute(
                        'speed', (walker_bp.get_attribute('speed').recommended_values[1]))
                else:
                    walker_bp.set_attribute('speed', 0.0)
                walker = self.world.try_spawn_actor(walker_bp, spawn_point_)
                if walker is not None:
                    walker_controller = self.world.spawn_actor(
                        walker_controller_bp, carla.Transform(), walker)
                    self.walker_list.append(walker_controller.id)
                    self.walker_list.append(walker.id)
            all_actors = self.world.get_actors(self.walker_list)

            # set how many pedestrians can cross the road
            #self.world.set_pedestrians_cross_factor(0.0)
            # 3. Starting the motion of our pedestrians
            for i in range(0, len(self.walker_list), 2):
                # start walker
                all_actors[i].start()
            # set walk to random point
                all_actors[i].go_to_location(
                    self.world.get_random_location_from_navigation())

        except:
            self.client.apply_batch(
                [carla.command.DestroyActor(x) for x in self.walker_list])

    def set_other_vehicles(self):
        try:
            # NPC vehicles generated and set to autopilot
            # One simple for loop for creating x number of vehicles and spawing them into the world
            for _ in range(0, NUMBER_OF_VEHICLES):
                spawn_point = random.choice(self.map.get_spawn_points())
                bp_vehicle = random.choice(self.blueprint_library.filter('vehicle'))
                other_vehicle = self.world.try_spawn_actor(
                    bp_vehicle, spawn_point)
                if other_vehicle is not None:
                    other_vehicle.set_autopilot(True)
                    self.actor_list.append(other_vehicle)
            print("NPC vehicles have been generated in autopilot mode.")
        except:
            self.client.apply_batch(
                [carla.command.DestroyActor(x) for x in self.actor_list])

    def change_town(self, new_town):
        self.world = self.client.load_world(new_town)

    def get_world(self) -> object:
        return self.world

    def get_blueprint_library(self) -> object:
        return self.world.get_blueprint_library()

    def angle_diff(self, v0, v1):
        angle = np.arctan2(v1[1], v1[0]) - np.arctan2(v0[1], v0[0])
        if angle > np.pi: angle -= 2 * np.pi
        elif angle <= -np.pi: angle += 2 * np.pi
        return angle

    def distance_to_line(self, A, B, p):
        num   = np.linalg.norm(np.cross(B - A, A - p))
        denom = np.linalg.norm(B - A)
        if np.isclose(denom, 0):
            return np.linalg.norm(p - A)
        return num / denom

    def vector(self, v):
        if isinstance(v, carla.Location) or isinstance(v, carla.Vector3D):
            return np.array([v.x, v.y, v.z])
        elif isinstance(v, carla.Rotation):
            return np.array([v.pitch, v.yaw, v.roll])

    def get_discrete_action_space(self):
        action_space = \
            np.array([
            -0.50,
            -0.30,
            -0.10,
            0.0,
            0.10,
            0.30,
            0.50
            ])
        return action_space

    def get_vehicle(self, vehicle_name):
        blueprint = self.blueprint_library.filter(vehicle_name)[0]
        if blueprint.has_attribute('color'):
            color = random.choice(
                blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        return blueprint

    def set_vehicle(self, vehicle_bp, spawn_points):
        # Main vehicle spawned into the env
        spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
        self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)

    def remove_sensors(self):
        self.camera_obj = None
        self.collision_obj = None
        self.lane_invasion_obj = None
        self.env_camera_obj = None
        self.front_camera = None
        self.collision_history = None
        self.wrong_maneuver = None



class CameraSensor():

    def __init__(self, vehicle):
        self.sensor_name = 'sensor.camera.semantic_segmentation'
        self.parent = vehicle
        self.front_camera = list()
        world = self.parent.get_world()
        self.sensor = self._set_camera_sensor(world)
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda image: CameraSensor._get_front_camera_data(weak_self, image))

    # Main front camera is setup and provide the visual observations for our network.
    def _set_camera_sensor(self, world):
        front_camera_bp = world.get_blueprint_library().find(self.sensor_name)
        front_camera_bp.set_attribute('image_size_x', f'160')
        front_camera_bp.set_attribute('image_size_y', f'80')
        front_camera_bp.set_attribute('fov', f'125')
        front_camera = world.spawn_actor(front_camera_bp, carla.Transform(
            carla.Location(x=2.4, z=1.5), carla.Rotation(pitch= -10)), attach_to=self.parent)
        return front_camera

    @staticmethod
    def _get_front_camera_data(weak_self, image):
        self = weak_self()
        if not self:
            return
        image.convert(carla.ColorConverter.CityScapesPalette)
        placeholder = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        placeholder1 = placeholder.reshape((image.width, image.height, 4))
        target = placeholder1[:, :, :3]
        self.front_camera.append(target)#/255.0)

class CameraSensorEnv:
    _display = None
    _pygame_initialized = False

    def __init__(self, vehicle):
        if not CameraSensorEnv._pygame_initialized:
            pygame.init()
            CameraSensorEnv._display = pygame.display.set_mode((720, 720), pygame.HWSURFACE | pygame.DOUBLEBUF)
            pygame.display.set_caption("CARLA PIL 3rd Person View")
            CameraSensorEnv._pygame_initialized = True

        self.display = CameraSensorEnv._display
        self.sensor_name = 'sensor.camera.rgb'
        self.parent = vehicle
        self.surface = None
        world = self.parent.get_world()
        self.sensor = self._set_camera_sensor(world)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda image: CameraSensorEnv._get_third_person_camera(weak_self, image))

    # Third camera is setup and provide the visual observations for our environment.

    def _set_camera_sensor(self, world):

        thrid_person_camera_bp = world.get_blueprint_library().find(self.sensor_name)
        thrid_person_camera_bp.set_attribute('image_size_x', f'720')
        thrid_person_camera_bp.set_attribute('image_size_y', f'720')
        third_camera = world.spawn_actor(thrid_person_camera_bp, carla.Transform(
            carla.Location(x=-4.0, z=2.0), carla.Rotation(pitch=-12.0)), attach_to=self.parent)
        return third_camera

    @staticmethod
    def _get_third_person_camera(weak_self, image):
        self = weak_self()
        if not self:
            return
        pygame.event.pump()
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        placeholder1 = array.reshape((image.width, image.height, 4))
        placeholder2 = placeholder1[:, :, :3]
        placeholder2 = placeholder2[:, :, ::-1]
        self.surface = pygame.surfarray.make_surface(placeholder2.swapaxes(0, 1))
        self.display.blit(self.surface, (0, 0))
        pygame.display.flip()

class CollisionSensor:

    def __init__(self, vehicle) -> None:
        self.sensor_name = 'sensor.other.collision'
        self.parent = vehicle
        self.collision_data = list()
        world = self.parent.get_world()
        self.sensor = self._set_collision_sensor(world)
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: CollisionSensor._on_collision(weak_self, event))

    # Collision sensor to detect collisions occured in the driving process.
    def _set_collision_sensor(self, world) -> object:
        collision_sensor_bp = world.get_blueprint_library().find(self.sensor_name)
        sensor_relative_transform = carla.Transform(
            carla.Location(x=1.3, z=0.5))
        collision_sensor = world.spawn_actor(
            collision_sensor_bp, sensor_relative_transform, attach_to=self.parent)
        return collision_sensor

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        self.collision_data.append(intensity)


def data_processing(observation):

    image_array = observation[0].astype(np.uint8)
    info_array = np.array(observation[1],dtype = np.float32)

    print(f'Image data:{image_array[0]}')
    print(f'info array:{info_array}')

    image_shape = image_array.shape
    image_bytes = image_array.tobytes()
    info_bytes = info_array.tobytes()

    data = struct.pack("3I", *image_shape) + image_bytes + info_bytes 

    #print(f'transmission_data:{data}')

    return data


def run():
    
    np.random.seed(SEED)
    random.seed(SEED)
    tf.random.set_seed(SEED)


    try:
        client, world = ClientConnection().setup()
        print("CONNECTION HAS BEEN STEUP SUCCESSFULLY.")
        print()
    except:
        ConnectionRefusedError
        print("CONNECTION HAS BEEN REFUSED BY THE SERVER.")
        print()
   

    if not os.path.exists(LOG_PATH_PI):
        os.makedirs(LOG_PATH_PI)
    
    summary_writer = tf.summary.create_file_writer(LOG_PATH_PI)

    env = CarlaEnvironment(client, world,TOWN)


    episode = 0
    print()
    print('TESTING.....')
    print()
    step_telemetry_file = f'{RESULTS_PATH}/PIL_step_telemetry.csv'
    detailed_episodes_file = f'{RESULTS_PATH}/PIL_detailed_analysis.csv'
    legacy_results_file = f'{RESULTS_PATH}/PIL_test_results_16bit.csv'

    model_name = sys.argv[2] if len(sys.argv) > 2 else "SAPPO"

    # Initialize CSV headers if files do not exist
    if not os.path.exists(step_telemetry_file) or os.path.getsize(step_telemetry_file) == 0:
        with open(step_telemetry_file, mode="w", newline="") as f:
            csv.writer(f).writerow(["Model", "Episode", "Step", "Timestamp", "Steer", "Throttle", "Velocity_kmh", "DistFromCenter_m", "Angle_deg", "Reward", "Latency_ms", "Steer_Delta"])

    if not os.path.exists(detailed_episodes_file) or os.path.getsize(detailed_episodes_file) == 0:
        with open(detailed_episodes_file, mode="w", newline="") as f:
            csv.writer(f).writerow(["Model", "Episode", "Duration_sec", "Reward", "Distance_m", "Termination_Reason", "Mean_Latency_ms", "P50_Latency_ms", "P95_Latency_ms", "P99_Latency_ms", "Latency_Std_ms", "Effective_Hz", "Avg_Speed_mps", "Mean_Lane_Deviation_m", "Mean_Steer_Jitter", "Steps_Count"])

    episode = 0
    print()
    print('======================================================================')
    print(f' PROCESSOR-IN-THE-LOOP (PIL) EVALUATION: {model_name}')
    print('======================================================================')
    print()

    client_socket.settimeout(20.0)

    target_episodes = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else NO_OF_TEST_EPISODES
    while episode < target_episodes:

        total_time = 0
        current_ep_reward = 0
        deviation_from_center = 0
        distance_covered = 0
        t1 = datetime.now()
        avg_latency = []
        episode_steer_deltas = []
        prev_steer = 0.0

        observation = env.reset()
        t3 = datetime.now()

        data = data_processing(observation)
        client_socket.sendall(data)

        step_count = 0
        for i in range(EPISODE_LENGTH):
            step_count += 1
            try:
                d = client_socket.recv(8)
                if not d or len(d) < 8:
                    print("[WARNING] Socket closed by client.")
                    break
                action = struct.unpack('2f', d)
            except socket.timeout:
                print(f"[ERROR] Socket timed out waiting for action at step {step_count}!")
                break

            t4 = datetime.now()
            step_lat_sec = abs((t4 - t3).total_seconds())
            avg_latency.append(step_lat_sec)

            steer_delta = abs(action[0] - prev_steer)
            episode_steer_deltas.append(steer_delta)
            prev_steer = action[0]

            observation, reward, done, info = env.step(action)
            t3 = datetime.now()

            # Per-step telemetry logging
            with open(step_telemetry_file, mode="a", newline="") as f_step:
                csv.writer(f_step).writerow([
                    episode, step_count, datetime.now().isoformat(),
                    round(action[0], 4), round(action[1], 4),
                    round(env.velocity, 2), round(env.distance_from_center, 4),
                    round(env.angle, 4), round(reward, 4),
                    round(step_lat_sec * 1000.0, 2), round(steer_delta, 4)
                ])

            data = data_processing(observation)
            client_socket.sendall(data)
            current_ep_reward += reward

            if done:
                episode += 1
                break

        deviation_from_center += info[1]
        distance_covered += info[0]
        term_reason = info[2] if len(info) > 2 else "DONE"

        t2 = datetime.now()
        total_time = abs((t2 - t1).total_seconds())

        lat_arr = np.array(avg_latency) * 1000.0 if len(avg_latency) > 0 else np.array([0.0])
        mean_lat = float(np.mean(lat_arr))
        p50_lat = float(np.percentile(lat_arr, 50))
        p95_lat = float(np.percentile(lat_arr, 95))
        p99_lat = float(np.percentile(lat_arr, 99))
        std_lat = float(np.std(lat_arr))
        effective_hz = round(1000.0 / mean_lat, 1) if mean_lat > 0 else 0.0
        avg_speed = round(info[0] / max(total_time, 0.001), 2)
        mean_steer_jitter = float(np.mean(episode_steer_deltas)) if len(episode_steer_deltas) > 0 else 0.0

        print(f"\n[EPISODE {episode:02d}/{TEST_EPISODES}] Status: {term_reason} | Time: {total_time:.2f}s | Reward: {current_ep_reward:+.2f} | Dist: {info[0]}m | Latency: {mean_lat:.1f}ms (P95: {p95_lat:.1f}ms, {effective_hz} Hz) | Steer Jitter (ASR): {mean_steer_jitter:.3f}")

        with summary_writer.as_default():
            tf.summary.scalar('Metrics/Time Taken', total_time, step=episode)
            tf.summary.scalar('Metrics/Reward', current_ep_reward, step=episode)
            tf.summary.scalar('Metrics/Distance Covered', info[0], step=episode)
            tf.summary.scalar('Metrics/Mean Latency ms', mean_lat, step=episode)
            tf.summary.scalar('Metrics/P95 Latency ms', p95_lat, step=episode)
            tf.summary.scalar('Metrics/Effective Hz', effective_hz, step=episode)
            tf.summary.scalar('Metrics/Steer Jitter', mean_steer_jitter, step=episode)
            summary_writer.flush()

        # Detailed analysis CSV
        with open(detailed_episodes_file, mode="a", newline="") as f_det:
            csv.writer(f_det).writerow([
                model_name, episode, round(total_time, 2), round(current_ep_reward, 2), info[0],
                term_reason, round(mean_lat, 2), round(p50_lat, 2), round(p95_lat, 2),
                round(p99_lat, 2), round(std_lat, 2), effective_hz, avg_speed,
                round(info[1], 4), round(mean_steer_jitter, 4), step_count
            ])

        # Legacy backward-compatible CSV
        with open(legacy_results_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            if file.tell() == 0:
                writer.writerow(["Episode", "TimeTaken (sec)", "Reward", "Distance Covered (m)", "Avg Latency (msec)", "Avg speed (m/sec)"])
            writer.writerow([episode, total_time, current_ep_reward, info[0], mean_lat, avg_speed])

    sys.exit()



if __name__ == "__main__":

    try:
        run()

    except KeyboardInterrupt:
        sys.exit()
        
    finally:
        try:
            client_socket.close()
        except Exception:
            pass
        try:
            server_socket.close()
        except Exception:
            pass
        print("\nTerminating...")