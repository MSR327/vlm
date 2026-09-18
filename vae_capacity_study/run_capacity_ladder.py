"""
Closed-Loop CARLA Benchmark Runner for VAE Capacity & Latent Scaling.

Usage:
    python run_capacity_ladder.py --arch baseline --latent 95 --mode test
    python run_capacity_ladder.py --arch wide     --latent 95 --mode test
    python run_capacity_ladder.py --arch deep     --latent 95 --mode test
    python run_capacity_ladder.py --arch deep     --latent 190 --mode test
"""
import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime

import numpy as np

# Parse CLI arguments first to configure environment variables before imports
parser = argparse.ArgumentParser(description="Run VAE Capacity Scaling in CARLA")
parser.add_argument('--arch', type=str, default='baseline', choices=['baseline', 'wide', 'deep'],
                    help="VAE Architecture: baseline, wide, deep")
parser.add_argument('--latent', type=int, default=95,
                    help="Latent space dimension dz (e.g. 16, 32, 64, 95, 128, 190, 256)")
parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'],
                    help="Execution mode: train or test")
parser.add_argument('--town', type=str, default='Town01',
                    help="CARLA benchmark town (Town01 or Town02)")
parser.add_argument('--episodes', type=int, default=20,
                    help="Number of evaluation episodes")
args = parser.parse_args()

os.environ['VAE_ARCH'] = args.arch
os.environ['VAE_LATENT'] = str(args.latent)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import params_capacity as P
import tensorflow as tf

from main import ClientConnection, PPOAgent
from ladder_env import LadderEnvironment
from encode_state_capacity import EncodeStateCapacity


CSV_FIELDS = [
    'arch', 'latent_dim', 'obs_dim', 'town', 'episode',
    'reward', 'success', 'collided', 'collision_count', 'route_completion',
    'distance_covered_m', 'center_lane_deviation_m', 'termination_reason',
    'timesteps', 'mean_speed_kmh', 'episode_wall_time_s',
    'encoder_latency_mean_ms', 'encoder_latency_p95_ms',
    'step_latency_mean_ms'
]


def _connect(town):
    conn = ClientConnection()
    conn.host = getattr(P, 'CARLA_HOST', 'localhost')
    conn.port = getattr(P, 'CARLA_PORT', 2000)
    conn.timeout = getattr(P, 'CARLA_TIMEOUT', 60.0)
    conn.town = town
    client, world = conn.setup()
    if client is None or world is None:
        raise ConnectionError(
            f"Could not reach CARLA server at {conn.host}:{conn.port}. Is CarlaUE4.exe running?"
        )
    print(f"CARLA connection established ({town}) at {conn.host}:{conn.port}.")
    return client, world


def run_evaluation(env, agent, encoder, n_episodes, out_csv):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    f_csv = open(out_csv, 'w', newline='')
    writer = csv.DictWriter(f_csv, fieldnames=CSV_FIELDS)
    writer.writeheader()

    print(f"\n=======================================================")
    print(f" CARLA CLOSED-LOOP BENCHMARK: {P.VAE_ARCH.upper()} (dz={P.LATENT_DIM})")
    print(f" Town: {P.TOWN} | Episodes: {n_episodes} | Obs Dim: {P.OBSERVATION_DIM}")
    print(f" Results CSV: {out_csv}")
    print(f"=======================================================\n")

    summary_rewards = []
    summary_completions = []
    summary_deviations = []

    for ep in range(1, n_episodes + 1):
        t0_ep = time.time()
        obs_raw = env.reset()
        state = encoder.process(obs_raw)

        total_reward = 0.0
        step_count = 0
        step_latencies = []
        encoder.timings.clear()

        done = False
        info = {}

        while not done and step_count < P.EPISODE_LENGTH:
            t0_step = time.perf_counter()
            action, _ = agent(state, False)

            next_obs_raw, reward, done, info = env.step(action)
            next_state = encoder.process(next_obs_raw)

            total_reward += reward
            step_count += 1
            state = next_state

            dt_step = (time.perf_counter() - t0_step) * 1000.0
            step_latencies.append(dt_step)

        wall_time = time.time() - t0_ep
        success = int(info.get('success', False))
        collided = int(info.get('collided', False))
        collision_count = info.get('collision_count', 0)
        completion = info.get('route_completion', 0.0)
        distance = info.get('distance_covered', 0.0)
        lane_dev = info.get('center_lane_deviation', 0.0)
        reason = info.get('termination_reason', 'max_steps')

        enc_mean = float(np.mean(encoder.timings)) if encoder.timings else 0.0
        enc_p95  = float(np.percentile(encoder.timings, 95)) if encoder.timings else 0.0
        step_mean = float(np.mean(step_latencies)) if step_latencies else 0.0

        row = {
            'arch': P.VAE_ARCH,
            'latent_dim': P.LATENT_DIM,
            'obs_dim': P.OBSERVATION_DIM,
            'town': P.TOWN,
            'episode': ep,
            'reward': round(total_reward, 2),
            'success': success,
            'collided': collided,
            'collision_count': collision_count,
            'route_completion': round(completion * 100.0, 1),
            'distance_covered_m': round(distance, 1),
            'center_lane_deviation_m': round(lane_dev, 3),
            'termination_reason': reason,
            'timesteps': step_count,
            'mean_speed_kmh': round(info.get('mean_speed_kmh', 0.0), 1),
            'episode_wall_time_s': round(wall_time, 1),
            'encoder_latency_mean_ms': round(enc_mean, 2),
            'encoder_latency_p95_ms': round(enc_p95, 2),
            'step_latency_mean_ms': round(step_mean, 2)
        }
        writer.writerow(row)
        f_csv.flush()

        summary_rewards.append(total_reward)
        summary_completions.append(completion * 100.0)
        summary_deviations.append(lane_dev)

        print(f"Ep {ep:2d}/{n_episodes:2d} | "
              f"Reward: {total_reward:7.1f} | "
              f"Comp: {completion*100.0:4.1f}% ({distance:5.1f}m) | "
              f"Dev: {lane_dev:.2f}m | "
              f"Term: {reason:<15} | "
              f"Enc: {enc_mean:.1f}ms")

    f_csv.close()

    print("\n" + "=" * 60)
    print(f" EVALUATION SUMMARY ({P.VAE_ARCH.upper()}, dz={P.LATENT_DIM})")
    print(f" Mean Cumulative Reward:    {np.mean(summary_rewards):.2f} +/- {np.std(summary_rewards):.2f}")
    print(f" Mean Route Completion:    {np.mean(summary_completions):.1f}% +/- {np.std(summary_completions):.1f}%")
    print(f" Mean Lane Center Dev:     {np.mean(summary_deviations):.2f} m")
    print("=" * 60 + "\n")


def main():
    P.TOWN = args.town
    for d in [P.RESULTS_PATH, P.CHECKPOINT_PATH, P.LOG_PATH_TRAIN, P.LOG_PATH_TEST]:
        os.makedirs(d, exist_ok=True)

    print(f"[init] Connecting to CARLA ({getattr(P, 'CARLA_HOST', 'localhost')}:{getattr(P, 'CARLA_PORT', 2000)}) ...")
    client, world = _connect(P.TOWN)

    print(f"[init] Initializing LadderEnvironment in {P.TOWN} ...")
    env = LadderEnvironment(client, world, P.TOWN, P)

    print(f"[init] Initializing PPOAgent with observation_dim={P.OBSERVATION_DIM} ...")
    agent = PPOAgent()

    if args.mode == 'test':
        if os.path.isdir(os.path.join(P.PPO_MODEL_PATH, 'actor')):
            agent.load()
            print(f"[PPOAgent] Loaded trained model from {P.PPO_MODEL_PATH}")
        elif P.OBSERVATION_DIM == 100:
            candidates = [
                '../btp_prev/results/Results_05/ppo_model',
                '../MTP_TESTING/Results_05/ppo_model',
                '../btp_prev/Results_05/ppo_model',
                'Results_05/ppo_model'
            ]
            loaded = False
            for c in candidates:
                if os.path.isdir(os.path.join(c, 'actor')):
                    agent.models_dir = c
                    agent.load()
                    print(f"[PPOAgent] Loaded pre-trained baseline policy for evaluation from: {c}")
                    loaded = True
                    break
            if not loaded:
                print("[PPOAgent] Warning: No baseline actor found. Driving with initialized policy.")

    encoder = EncodeStateCapacity(P)

    out_csv = os.path.join(P.RESULTS_PATH, f"eval_{P.TOWN}_{args.arch}_d{args.latent}.csv")
    run_evaluation(env, agent, encoder, args.episodes, out_csv)


if __name__ == '__main__':
    main()
