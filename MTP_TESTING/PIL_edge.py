import os
import sys
import random
import socket
import struct
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import time
layers = tf.keras.layers
from parameters import*
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.settimeout(5.0)

connected = False
for attempt in range(1, 11):
    try:
        print(f"Connecting to Simulation Node at {SIMULATION_IP}:{PORT} (Attempt {attempt}/10)...")
        client_socket.connect((SIMULATION_IP, PORT))
        client_socket.settimeout(None)
        connected = True
        print("[SUCCESS] Connection Established with Simulation Node!")
        break
    except socket.error as e:
        print(f"  Attempt {attempt} failed ({e}). Retrying in 2 seconds...")
        time.sleep(2)

if not connected:
    print(f"[ERROR] Could not connect to {SIMULATION_IP}:{PORT}.")
    print("Troubleshooting checklist:")
    print("  1. Is CARLA and PIL_simulation.py running on the PC?")
    print("  2. Is SIMULATION_IP set to the PC's actual LAN IP (not 127.0.0.1)?")
    print("  3. Is Windows Defender Firewall allowing port 5000?")
    sys.exit(1)


@tf.keras.utils.register_keras_serializable()
class Encoder(tf.keras.Model):

    def __init__(self,name = 'ENCODER',**kwargs):
        super().__init__(name = name ,**kwargs)

        self.latent_dim = LATENT_DIM

        self.conv1 = layers.Conv2D(32, (4, 4), activation='relu', strides=2, padding='same')
        self.conv2 = layers.Conv2D(64, (3, 3), activation='relu', strides=2, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.conv3 = layers.Conv2D(128, (4, 4), activation='relu', strides=2, padding='same')
        self.conv4 = layers.Conv2D(256, (3, 3), activation='relu', strides=2, padding='same')
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(1024, activation='relu')
        self.mu = layers.Dense(self.latent_dim)
        self.sigma = layers.Dense(self.latent_dim)

        self.N = tfp.distributions.Normal(loc=0.0, scale=1.0)
        self.kl = 0  


    def call(self, inputs):

        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.bn1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        x = self.dense1(x)

        mu = self.mu(x)

        sigma = tf.exp(self.sigma(x))  
        z = mu + sigma * tfd.Normal(0.0, 1.0).sample(tf.shape(mu))  
        self.kl = tf.reduce_sum(sigma**2 + mu**2 - tf.math.log(sigma) - 0.5) 

        return z


    def process(self,observation):

        image_obs = tf.convert_to_tensor(observation[0], dtype=tf.float32)
        image_obs = tf.expand_dims(image_obs, axis=0)  
        image_obs = self(image_obs, training=False)  
        navigation_obs = tf.convert_to_tensor(observation[1], dtype=tf.float32)
        observation = tf.concat([tf.reshape(tf.cast(image_obs, tf.float32), [-1]),tf.cast(navigation_obs, tf.float32)], axis=-1)
       
        return observation


@tf.keras.utils.register_keras_serializable()
class Actor(tf.keras.Model):

    def __init__(self,name = 'ACTOR',**kwargs):
        super().__init__(name = name ,**kwargs)

        self.obs_dim = OBSERVATION_DIM
        self.action_dim = ACTION_DIM
        self.action_std_init = ACTION_STD_INIT
       
        self.dense1 = layers.Dense(500, activation='tanh')
        self.dense2 = layers.Dense(300, activation='tanh')
        self.dense3 = layers.Dense(100, activation='tanh')
        self.output_layer = layers.Dense(self.action_dim, activation='tanh')


    def call(self, obs):

        if isinstance(obs, np.ndarray):
            obs = tf.convert_to_tensor(obs, dtype=tf.float32)

        if len(obs.shape) == 1: 
            obs = tf.expand_dims(obs, axis=0)
        
        obs = self.normalize(obs)

        #print(obs)

        x = self.dense1(obs)
        x = self.dense2(x)
        x = self.dense3(x)
        mean = self.output_layer(x)

        log_std = tf.Variable(tf.fill((self.action_dim,), self.action_std_init), trainable=False, dtype=tf.float32)
        dist = tfd.MultivariateNormalDiag(loc=mean, scale_diag=log_std)
        action = dist.sample()

        return action

    def normalize(self, obs):

        obs = tf.clip_by_value(obs, clip_value_min=-1e8, clip_value_max=1e8)
        return obs





def data_processing():

    header = client_socket.recv(12)
    h,w,c = struct.unpack("3I",header)
    info_size = 5
    image_size = h*w*c

    image_bytes = b""

    while len(image_bytes)<image_size:
        image_bytes += client_socket.recv(image_size - len(image_bytes))

    info_bytes = client_socket.recv(info_size*4)

    image_array = np.frombuffer(image_bytes,dtype = np.uint8).reshape((h,w,c))
    info_array = np.frombuffer(info_bytes,dtype=np.float32)


    image_obs = tf.convert_to_tensor(image_array, dtype=tf.float32)
    image_obs = tf.expand_dims(image_obs, axis=0)  

    info_obs = tf.convert_to_tensor(info_array,dtype = tf.float32)

    return image_obs,info_obs



def run():

    np.random.seed(SEED)
    random.seed(SEED)
    tf.random.set_seed(SEED)

    agent = Actor()
    encoder = Encoder()

    encoder = tf.keras.models.load_model(VAE_MODEL_PATH+'/var_auto_encoder_model')

    print(f'Encoder Model is Loaded from {VAE_MODEL_PATH}')

    agent = Actor()
    agent = tf.keras.models.load_model(PPO_MODEL_PATH + '/actor')
    print(f"actor Model is  loaded from {PPO_MODEL_PATH}")
    print()


    while True:

        image_obs ,info_obs = data_processing()

        obs_1 = encoder(image_obs)

        observation = tf.concat([tf.reshape(tf.cast(obs_1, tf.float32), [-1]),tf.cast(info_obs, tf.float32)], axis=-1)

        if observation is None:
            break

        action = agent(observation).numpy().flatten()

        print(action)

        data = struct.pack('2f',*action)

        client_socket.sendall(data)
        print("action sent")

    
    client_socket.close()

    sys.exit()


if __name__ == "__main__":
    run()

