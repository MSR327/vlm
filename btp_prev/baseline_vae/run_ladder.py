"""
Single entry point for the VAE baseline ladder -- rungs 1, 2 and 3.

    cd btp_prev                       # ALWAYS: every path literal is CWD-relative

    BTP_RUNG=1 python baseline_vae/run_ladder.py --mode train
    BTP_RUNG=2 python baseline_vae/run_ladder.py --mode train
    BTP_RUNG=3 python baseline_vae/run_ladder.py --mode train

    BTP_RUNG=1 python baseline_vae/run_ladder.py --mode test --town Town02
    ...

Why one script and not three: the ONLY thing that differs between rungs is the
sensor set, which BTP_RUNG selects. PPO hyperparameters, reward, route, towns,
seeds, episode budget and termination thresholds all come from
params_vae_base.py and are therefore provably identical. Three scripts would
have drifted.

Set BTP_NORM=raw (default, as-deployed, comparable to Results_05) or
BTP_NORM=scaled (as the VAE was trained). Results land in
Results_VAE_<rung>_<arm>/ so no run can clobber another.

Writes per-episode CSVs with the metrics the ablation needs:
reward, success, collision, route completion, latency (encoder vs step),
lane deviation, termination reason.
"""
import argparse
import csv
import json
import os
import random
import sys
import time
from datetime import datetime

import numpy as np
import tensorflow as tf

from parameters import *          # noqa: F401,F403  -- rung-selected config
import parameters as P

from main import ClientConnection, PPOAgent
from ladder_env import LadderEnvironment
from encode_state_vae import EncodeStateVAE


TEST_FIELDS = [
    'rung', 'rung_name', 'norm_arm', 'obs_dim', 'town', 'episode',
    'reward', 'success', 'collided', 'collision_count', 'route_completion',
    'distance_covered_m', 'center_lane_deviation_m', 'termination_reason',
    'timesteps', 'mean_speed_kmh', 'episode_wall_time_s',
    'spawn_index', 'route_length',
    'encoder_latency_mean_ms', 'encoder_latency_p95_ms',
    'step_latency_mean_ms', 'policy_latency_mean_ms',
]


def _write_manifest(args):
    """Record the exact protocol next to the numbers.

    Without this, a CSV six weeks from now cannot be checked for whether it
    was produced under the same seed, sync mode and traffic settings as the
    rung it is being compared against.
    """
    import platform
    man = dict(
        rung=RUNG, rung_name=RUNG_NAME, norm_arm=NORM_ARM,
        vae_input_scale=VAE_INPUT_SCALE, observation_dim=OBSERVATION_DIM,
        num_cameras=NUM_CAMERAS_VAE, use_lidar=USE_LIDAR_VAE,
        camera_yaws=CAMERA_YAWS, camera_sensor=CAMERA_SENSOR_NAME,
        town=args.town, mode=args.mode,
        eval_routes=args.eval_routes or EVAL_SPAWN_POINTS,
        episodes=args.episodes or NO_OF_TEST_EPISODES,
        seed=SEED, carla_seed=CARLA_SEED, tm_seed=TRAFFIC_MANAGER_SEED,
        synchronous=SYNCHRONOUS_MODE, fixed_delta_seconds=FIXED_DELTA_SECONDS,
        npc_vehicles=ENABLE_NPC_VEHICLES,
        number_of_pedestrians=NUMBER_OF_PEDESTRIAN,
        weather=WEATHER_PRESET,
        ppo=dict(action_std_init=ACTION_STD_INIT, lr=LEARNING_RATE,
                 gamma=GAMMA, lam=LAMBDA, clip=POLICY_CLIP,
                 iterations=NO_OF_ITERATIONS, batch_size=BATCH_SIZE),
        target_speed=TARGET_SPEED, max_speed=MAX_SPEED, min_speed=MIN_SPEED,
        max_distance_from_center=MAX_DISTANCE_FROM_CENTER,
        frame_layout=__import__('multi_sensor').FRAME_LAYOUT,
        tf_version=tf.__version__, python=platform.python_version(),
        platform=platform.platform(),
        timestamp=datetime.now().isoformat(timespec='seconds'),
    )
    os.makedirs(RESULTS_PATH, exist_ok=True)
    path = f'{RESULTS_PATH}/run_manifest_{args.mode}_{args.town.lower()}.json'
    with open(path, 'w') as f:
        json.dump(man, f, indent=2)
    print(f"[manifest] {path}")


def _seed_everything():
    np.random.seed(SEED)
    random.seed(SEED)
    tf.random.set_seed(SEED)


def _connect(town):
    """Connect using the params host/port rather than main.py's hardcoded ones.

    ClientConnection hardcodes localhost:2000 and a 20 s timeout. Loading a
    town on a cold Windows server routinely takes longer than that, so the
    timeout is raised. Nothing else about the connection changes.
    """
    conn = ClientConnection()
    conn.host = CARLA_HOST
    conn.port = CARLA_PORT
    conn.timeout = CARLA_TIMEOUT
    conn.town = town
    client, world = conn.setup()
    if client is None or world is None:
        raise ConnectionError(
            f"Could not reach the CARLA server at {CARLA_HOST}:{CARLA_PORT}. "
            f"Is CarlaUE4.exe running?")
    print(f"CARLA connection established ({town}) at {CARLA_HOST}:{CARLA_PORT}.")
    return client, world


def _append_csv(path, fields, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow(row)


# ======================================================================
def train(args):
    _seed_everything()
    client, world = _connect(args.town)

    os.makedirs(LOG_PATH_TRAIN, exist_ok=True)
    summary_writer = tf.summary.create_file_writer(LOG_PATH_TRAIN)

    env = LadderEnvironment(client, world, args.town, P)
    encoder = EncodeStateVAE(P)
    agent = PPOAgent()

    if CHECKPOINT_LOAD:
        episode, timestep, cumulative_score = agent.chkpt_load()
        agent.load()
    else:
        episode = timestep = 0
        cumulative_score = 0

    scores = []
    train_csv = f'{RESULTS_PATH}/train_episodes.csv'
    train_fields = ['rung', 'norm_arm', 'episode', 'timestep', 'reward',
                    'cumulative_reward', 'success', 'collided',
                    'route_completion', 'distance_covered_m',
                    'center_lane_deviation_m', 'termination_reason']

    budget = args.timesteps or TRAIN_TIMESTEPS
    print(f"TRAINING rung {RUNG} ({RUNG_NAME}) arm={NORM_ARM} "
          f"obs_dim={OBSERVATION_DIM} for {budget:.0f} timesteps")

    while timestep < budget:
        observation = encoder.process(env.reset())
        current_ep_reward = 0
        t1 = datetime.now()

        for _ in range(EPISODE_LENGTH):
            action, _ = agent(observation, True)
            observation, reward, done, info = env.step(action)
            observation = encoder.process(observation)

            agent.memory.rewards.append(reward)
            agent.memory.dones.append(done)
            timestep += 1
            current_ep_reward += reward
            if done:
                episode += 1
                break

        scores.append(current_ep_reward)
        cumulative_score = float(np.mean(scores))

        print(f"Ep {episode} | t {timestep} | R {current_ep_reward:.2f} "
              f"| avgR {cumulative_score:.2f} | comp {info['route_completion']:.2f} "
              f"| {info['termination_reason']}")

        _append_csv(train_csv, train_fields, dict(
            rung=RUNG, norm_arm=NORM_ARM, episode=episode, timestep=timestep,
            reward=current_ep_reward, cumulative_reward=cumulative_score,
            success=info['success'], collided=info['collided'],
            route_completion=info['route_completion'],
            distance_covered_m=info['distance_covered'],
            center_lane_deviation_m=info['center_lane_deviation'],
            termination_reason=info['termination_reason']))

        with summary_writer.as_default():
            tf.summary.scalar("Episodic Reward/episode", current_ep_reward, step=episode)
            tf.summary.scalar("Cumulative Reward/episode", cumulative_score, step=episode)
            tf.summary.scalar("Route Completion/episode", info['route_completion'], step=episode)
            tf.summary.scalar("Success/episode", float(info['success']), step=episode)
            tf.summary.scalar("Collision/episode", float(info['collided']), step=episode)
            summary_writer.flush()

        if episode % 5 == 0:
            agent.learn()
        if episode % 50 == 0:
            agent.save()
            agent.chkpt_save(episode, timestep, cumulative_score)

    agent.save()
    agent.chkpt_save(episode, timestep, cumulative_score)
    env.restore_settings()
    print("TRAINING COMPLETE")


# ======================================================================
def test(args):
    _seed_everything()
    client, world = _connect(args.town)

    os.makedirs(LOG_PATH_TEST, exist_ok=True)
    env = LadderEnvironment(client, world, args.town, P)
    encoder = EncodeStateVAE(P)
    agent = PPOAgent()
    agent.load()

    _write_manifest(args)
    n_episodes = args.episodes or NO_OF_TEST_EPISODES
    out_csv = f'{RESULTS_PATH}/test_results_{args.town.lower()}.csv'
    print(f"TESTING rung {RUNG} ({RUNG_NAME}) arm={NORM_ARM} "
          f"obs_dim={OBSERVATION_DIM} on {args.town}, {n_episodes} episodes")

    routes = args.eval_routes or EVAL_SPAWN_POINTS

    for episode in range(1, n_episodes + 1):
        # Cycle the shared route list. Rung r episode k always gets the same
        # spawn point as rung r' episode k.
        env.spawn_point_override = routes[(episode - 1) % len(routes)]
        env.fresh_start = True
        encoder.timings.clear()
        observation = encoder.process(env.reset())

        current_ep_reward = 0
        step_lat, policy_lat = [], []
        t1 = time.perf_counter()

        for _ in range(EPISODE_LENGTH):
            ts = time.perf_counter()
            tp = time.perf_counter()
            action, _ = agent(observation, False)
            policy_lat.append((time.perf_counter() - tp) * 1000.0)

            observation, reward, done, info = env.step(action)
            observation = encoder.process(observation)
            step_lat.append((time.perf_counter() - ts) * 1000.0)

            current_ep_reward += reward
            if done:
                break

        wall = time.perf_counter() - t1
        lat = encoder.latency_summary()

        row = dict(
            rung=RUNG, rung_name=RUNG_NAME, norm_arm=NORM_ARM,
            obs_dim=OBSERVATION_DIM, town=args.town, episode=episode,
            reward=current_ep_reward,
            success=int(info['success']), collided=int(info['collided']),
            collision_count=info['collision_count'],
            route_completion=info['route_completion'],
            distance_covered_m=info['distance_covered'],
            center_lane_deviation_m=info['center_lane_deviation'],
            termination_reason=info['termination_reason'],
            timesteps=info['timesteps'], mean_speed_kmh=info['mean_speed_kmh'],
            spawn_index=info['spawn_index'], route_length=info['route_length'],
            episode_wall_time_s=wall,
            encoder_latency_mean_ms=lat['mean_ms'],
            encoder_latency_p95_ms=lat['p95_ms'],
            step_latency_mean_ms=float(np.mean(step_lat)) if step_lat else 0.0,
            policy_latency_mean_ms=float(np.mean(policy_lat)) if policy_lat else 0.0,
        )
        _append_csv(out_csv, TEST_FIELDS, row)

        print(f"Ep {episode:3d} | R {current_ep_reward:8.2f} | succ {info['success']:d} "
              f"| coll {info['collided']:d} | comp {info['route_completion']:.3f} "
              f"| enc {lat['mean_ms']:.2f} ms | {info['termination_reason']}")

    env.restore_settings()

    df_cols = ['reward', 'success', 'collided', 'route_completion']
    import csv as _csv
    with open(out_csv) as f:
        rows = list(_csv.DictReader(f))
    n = len(rows)
    summary = {c: sum(float(r[c]) for r in rows) / n for c in df_cols}
    print(f"\n--- rung {RUNG} ({RUNG_NAME}) arm={NORM_ARM} on {args.town}, "
          f"{n} episodes ---")
    print(f"  mean reward      {summary['reward']:.2f}")
    print(f"  success rate     {summary['success']*100:.1f}%")
    print(f"  collision rate   {summary['collided']*100:.1f}%")
    print(f"  route completion {summary['route_completion']*100:.1f}%")
    print(f"\nWrote {out_csv}")


# ======================================================================
def main():
    ap = argparse.ArgumentParser(description="VAE baseline ladder runner")
    ap.add_argument('--mode', choices=['train', 'test'], required=True)
    ap.add_argument('--town', default=None,
                    help="default: Town01 for train, Town02 for test")
    ap.add_argument('--episodes', type=int, default=None)
    ap.add_argument('--timesteps', type=float, default=None)
    ap.add_argument('--eval-routes', type=int, nargs='+', default=None,
                    metavar='IDX',
                    help="spawn-point indices to evaluate on, cycled across "
                         "episodes. Omit for the canonical single route. Use "
                         "the SAME list for all three rungs.")
    ap.add_argument('--mask-rear', action='store_true', default=False,
                    help="mask rear camera with zeros (Town02 simulator fault-tolerant workaround)")
    args = ap.parse_args()

    if args.mask_rear:
        os.environ['BTP_MASK_REAR'] = '1'
        P.MASK_REAR_CAMERA = True

    if args.town is None:
        args.town = TRAIN_TOWN if args.mode == 'train' else TEST_TOWN

    if args.mode == 'train':
        train(args)
    else:
        test(args)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit()
    finally:
        print("\nTerminating...")
