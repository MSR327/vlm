"""
Measured encoder + policy latency for each ladder configuration.

Reports p50/p95, not just the mean: the ladder's control loop runs at 20 Hz
(50 ms budget) and the tail is what breaks it.

Rung 3's BEV encoder does not exist yet, so its LiDAR stream is timed against
a freshly-initialised encoder of the IDENTICAL architecture. Latency and cost
depend only on layer geometry and input shape, not on weight values, so this
number is exact for rung 3 even before the BEV VAE is trained. Only rung 3's
*information* metrics have to wait for real training.
"""
import json
import os
import sys
import time

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

import numpy as np
import tensorflow as tf

sys.path.insert(0, 'study')
from cost_model import RUNGS, config_cost, LATENT_DIM, NAV_DIM

N_WARMUP, N_ITERS = 20, 100


def load_camera_encoder():
    sm = tf.saved_model.load('VAE/var_auto_encoder_model')
    fn = sm.signatures['serving_default']
    return lambda x: fn(input_1=tf.convert_to_tensor(x, tf.float32))['output_1']


def build_untrained_encoder():
    """Stand-in for the not-yet-trained BEV encoder.

    Layer-for-layer identical to VAE_Trainer_dont_touch.Encoder (verified: the
    analytic param count of this geometry, 13,749,662, matches the saved camera
    encoder's variable count exactly). Rebuilt here rather than imported
    because that module pulls in tensorflow_probability, which is only needed
    for the training-time KL sampling -- irrelevant to inference cost.
    """
    layers = tf.keras.layers
    enc = tf.keras.Sequential([
        layers.Input(shape=(160, 80, 3)),
        layers.Conv2D(32, (4, 4), activation='relu', strides=2, padding='same'),
        layers.Conv2D(64, (3, 3), activation='relu', strides=2, padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (4, 4), activation='relu', strides=2, padding='same'),
        layers.Conv2D(256, (3, 3), activation='relu', strides=2, padding='same'),
        layers.Flatten(),
        layers.Dense(1024, activation='relu'),
        layers.Dense(LATENT_DIM),          # mu head; sigma head timed below
    ])
    sigma = layers.Dense(LATENT_DIM)
    enc(tf.zeros((1, 160, 80, 3), tf.float32))
    sigma(tf.zeros((1, 1024), tf.float32))

    def _call(x):
        return enc(tf.convert_to_tensor(x, tf.float32), training=False)
    return _call


def build_actor(obs_dim):
    layers = tf.keras.layers
    m = tf.keras.Sequential([
        layers.Input(shape=(obs_dim,)),
        layers.Dense(500, activation='tanh'),
        layers.Dense(300, activation='tanh'),
        layers.Dense(100, activation='tanh'),
        layers.Dense(2, activation='tanh'),
    ])
    m(tf.zeros((1, obs_dim)))
    return m


def timeit(fn, n_warmup=N_WARMUP, n_iters=N_ITERS):
    for _ in range(n_warmup):
        fn()
    ts = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000.0)
    a = np.asarray(ts)
    return dict(mean_ms=float(a.mean()), p50_ms=float(np.percentile(a, 50)),
                p95_ms=float(np.percentile(a, 95)), std_ms=float(a.std()))


def main():
    cam = load_camera_encoder()
    bev = build_untrained_encoder()
    rng = np.random.default_rng(0)

    rows = []
    for rung, cfg in RUNGS.items():
        n_cam = cfg['n_cam']
        cam_batch = rng.integers(0, 256, (n_cam, 160, 80, 3)).astype(np.float32)
        bev_batch = rng.random((1, 160, 80, 3)).astype(np.float32) * 255.0

        cam_t = timeit(lambda: np.asarray(cam(cam_batch)))
        if cfg['lidar']:
            bev_t = timeit(lambda: np.asarray(bev(bev_batch)))
            enc_mean = cam_t['mean_ms'] + bev_t['mean_ms']
            enc_p95 = cam_t['p95_ms'] + bev_t['p95_ms']
        else:
            bev_t = None
            enc_mean, enc_p95 = cam_t['mean_ms'], cam_t['p95_ms']

        c = config_cost(rung)
        actor = build_actor(c['obs_dim'])
        obs = tf.constant(rng.standard_normal((1, c['obs_dim'])), tf.float32)
        pol_t = timeit(lambda: actor(obs, training=False).numpy())

        rows.append(dict(
            rung=rung, name=cfg['name'], obs_dim=c['obs_dim'],
            n_streams=c['n_streams'],
            cam_encoder_mean_ms=cam_t['mean_ms'], cam_encoder_p95_ms=cam_t['p95_ms'],
            bev_encoder_mean_ms=(bev_t['mean_ms'] if bev_t else 0.0),
            encoder_mean_ms=enc_mean, encoder_p95_ms=enc_p95,
            policy_mean_ms=pol_t['mean_ms'], policy_p95_ms=pol_t['p95_ms'],
            pipeline_mean_ms=enc_mean + pol_t['mean_ms'],
            pipeline_p95_ms=enc_p95 + pol_t['p95_ms'],
            max_rate_hz=1000.0 / (enc_mean + pol_t['mean_ms']),
            encoder_params=c['encoder_params'], actor_params=c['actor_params'],
            critic_params=c['critic_params'], total_params=c['total_params'],
            inference_macs=c['inference_macs'],
            inference_fp32_mb=c['inference_fp32_mb'], fp32_mb=c['fp32_mb'],
        ))

    os.makedirs('results/study', exist_ok=True)
    with open('results/study/cost_latency.json', 'w') as f:
        json.dump(dict(device='CPU (arm64 macOS)', tf=tf.__version__,
                       n_iters=N_ITERS, rows=rows), f, indent=2)

    hdr = (f"{'rung':<12}{'obs':>5}{'enc ms':>9}{'p95':>8}{'pol ms':>8}"
           f"{'total ms':>10}{'Hz':>7}{'MACs(M)':>10}{'infer MB':>10}")
    print(hdr)
    print('-' * len(hdr))
    for r in rows:
        print(f"{r['name']:<12}{r['obs_dim']:>5}{r['encoder_mean_ms']:>9.2f}"
              f"{r['encoder_p95_ms']:>8.2f}{r['policy_mean_ms']:>8.2f}"
              f"{r['pipeline_mean_ms']:>10.2f}{r['max_rate_hz']:>7.1f}"
              f"{r['inference_macs']/1e6:>10.1f}{r['inference_fp32_mb']:>10.2f}")
    print("\nwrote results/study/cost_latency.json")


if __name__ == '__main__':
    main()
