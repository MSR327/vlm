import os
import sys
import random
import socket
import struct
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
layers = tf.keras.layers
from parameters import*
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

client_socket = None

def connect_to_simulation():
    global client_socket
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.settimeout(5.0)
    print(f"\n[PIL Edge] Connecting to CARLA Simulation Node at {SIMULATION_IP}:{PORT}...")
    for attempt in range(1, 16):
        try:
            client_socket.connect((SIMULATION_IP, PORT))
            client_socket.settimeout(None)
            print(f"[PIL Edge] ✅ Connection Established Successfully on attempt {attempt}!")
            return client_socket
        except socket.error as e:
            print(f"[PIL Edge] Attempt {attempt}/15 failed: {e}. Retrying in 2 seconds...")
            import time as _t; _t.sleep(2)
    print(f"\n[PIL Edge] ❌ Could not connect to {SIMULATION_IP}:{PORT}.")
    print("Troubleshooting Guide:")
    print(" 1. Ensure PIL_simulation.py is running on your host PC first.")
    print(" 2. Verify SIMULATION_IP in parameters.py is set to your PC's IP (not 127.0.0.1).")
    print(" 3. Ensure Port 5000 is open in Windows Firewall.")
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
    import time
    np.random.seed(SEED)
    random.seed(SEED)
    tf.random.set_seed(SEED)

    connect_to_simulation()

    print(f'[PIL Edge] Loading VAE Encoder from {VAE_MODEL_PATH}...')
    encoder = tf.keras.models.load_model(VAE_MODEL_PATH + '/var_auto_encoder_model')
    print(f'[PIL Edge] ✅ Encoder Model Loaded Successfully!')

    print(f'[PIL Edge] Loading PPO Actor from {PPO_MODEL_PATH}...')
    agent = tf.keras.models.load_model(PPO_MODEL_PATH + '/actor')
    print(f"[PIL Edge] ✅ Actor Model Loaded Successfully!\n")

    step_count = 0
    total_vae_ms = 0.0
    total_ppo_ms = 0.0

    print("=" * 72)
    print("   PROCESSOR-IN-THE-LOOP (PIL) EDGE INFERENCE RUNNING ON RASPBERRY PI")
    print("=" * 72)

    while True:
        try:
            image_obs, info_obs = data_processing()
        except Exception as e:
            print(f"[PIL Edge] Socket disconnected by host: {e}")
            break

        step_count += 1

        # 1. Profile VAE Encoder Latency (Compressing to 95 Latents)
        t_vae_0 = time.time()
        obs_1 = encoder(image_obs)
        t_vae_ms = (time.time() - t_vae_0) * 1000.0
        total_vae_ms += t_vae_ms

        # 2. Perception Explainability: Analyze the 95 Latent Values
        latent_np = obs_1.numpy().flatten()
        latent_norm = float(np.linalg.norm(latent_np))
        latent_mean = float(np.mean(latent_np))
        latent_var  = float(np.var(latent_np))

        # 3. Form 100-d Observation: 95 Latents + 5 Nav Features
        observation = tf.concat([tf.reshape(tf.cast(obs_1, tf.float32), [-1]), tf.cast(info_obs, tf.float32)], axis=-1)
        if observation is None:
            break

        # 4. Profile PPO Actor Policy Latency
        t_ppo_0 = time.time()
        action = agent(observation).numpy().flatten()
        t_ppo_ms = (time.time() - t_ppo_0) * 1000.0
        total_ppo_ms += t_ppo_ms

        edge_inference_ms = t_vae_ms + t_ppo_ms
        edge_freq_hz = 1000.0 / edge_inference_ms if edge_inference_ms > 0 else 0.0

        # 5. Measure Action Divergence (ADIV) from ICML 2024 (Sun et al.)
        # Evaluates policy sensitivity: ||pi(s + delta) - pi(s)||_2 / (2 * epsilon)
        eps_adiv = 0.05
        noise_delta = tf.random.normal(tf.shape(observation), mean=0.0, stddev=0.05, dtype=tf.float32)
        perturbed_obs = observation + noise_delta
        action_perturbed = agent(perturbed_obs).numpy().flatten()
        adiv_val = float(np.linalg.norm(action_perturbed - action) / (2.0 * eps_adiv))

        # Print real-time edge telemetry every 10 steps
        if step_count % 10 == 0:
            print(f"Step {step_count:4d} | VAE: {t_vae_ms:4.1f}ms | PPO: {t_ppo_ms:4.1f}ms | "
                  f"Edge: {edge_freq_hz:4.1f}Hz | LatentNorm: {latent_norm:5.2f} | "
                  f"Steer: {action[0]:+5.2f} | Thr: {action[1]:+5.2f} | ADIV: {adiv_val:.3f}")

        data = struct.pack('2f', *action)
        try:
            client_socket.sendall(data)
        except Exception as e:
            print(f"[PIL Edge] Connection lost on send: {e}")
            break

    if client_socket:
        client_socket.close()

    if step_count > 0:
        print("\n" + "=" * 60)
        print("           RASPBERRY PI EDGE INFERENCE SUMMARY")
        print("=" * 60)
        print(f" Total Steps Processed:       {step_count}")
        print(f" Mean VAE Inference:          {total_vae_ms / step_count:.2f} ms")
        print(f" Mean PPO Actor Inference:    {total_ppo_ms / step_count:.2f} ms")
        print(f" Mean Total Edge Inference:   {(total_vae_ms + total_ppo_ms) / step_count:.2f} ms")
        mean_edge_freq = 1000.0 / ((total_vae_ms + total_ppo_ms) / step_count)
        print(f" Mean Edge Frequency:         {mean_edge_freq:.2f} Hz  (Target Planning ~14 Hz)")
        print("=" * 60 + "\n")

    sys.exit()


if __name__ == "__main__":
    run()

