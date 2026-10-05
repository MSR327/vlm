"""
CARLA Closed-Loop Benchmark Runner for Phase 2 TransFuser-PPO.

Usage:
    python -m phase2_sensory.run_benchmark --mode test --town Town01 --episodes 20
    python -m phase2_sensory.run_benchmark --mode train --town Town01 --episodes 20
"""
import argparse
import csv
import os
import time
import numpy as np

import phase2_sensory.config as C
from phase2_sensory.pipeline import TransFuserPPOPipeline
from phase2_sensory.visualizer import SensoryHUDVisualizer

CSV_FIELDS = [
    'episode', 'reward', 'route_completion_pct', 'distance_covered_m',
    'lane_deviation_m', 'termination_reason', 'collided',
    'timesteps', 'mean_speed_kmh', 'latency_mean_ms', 'latency_p95_ms'
]


def run_training(env, pipeline, n_episodes, save_path, visualizer=None):
    print(f"\n=======================================================")
    print(f" PHASE 2 TRANSFUSER-PPO TRAINING RUN")
    print(f" Town: {env.town} | Episodes: {n_episodes} | Obs Dim: {C.OBS_DIM}")
    print(f" Checkpoint Out: {save_path}")
    print(f" Visual HUD Render: {'ENABLED' if visualizer else 'HEADLESS'}")
    print(f"=======================================================\n")

    agent = pipeline.agent
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for ep in range(1, n_episodes + 1):
        sensor_obs = env.reset()
        ep_reward = 0.0
        done = False
        step = 0
        info = {}

        while not done and step < C.MAX_STEPS_PER_EP:
            step += 1
            action, u, log_prob, value, state_np, lat_ms = pipeline.act(sensor_obs, deterministic=False)
            next_sensor_obs, priv_state, reward, done, info = env.step(action)

            truncated = (priv_state.get('term_reason') == 'max_steps')
            next_val = 0.0
            if truncated:
                next_state_np, _ = pipeline.process_sensors(next_sensor_obs)
                next_val = agent.get_value(next_state_np)

            agent.remember(state_np, action, u, log_prob, reward, done, value, truncated=truncated, next_value=next_val)
            ep_reward += reward

            # Real-time HUD Visualization
            if visualizer is not None:
                chase_img = env.get_chase_image()
                bev_tensor = pipeline.projector.project(sensor_obs['lidar'])
                telemetry = {
                    'episode': ep,
                    'step': step,
                    'reward': ep_reward,
                    'speed': priv_state.get('velocity_kmh', env.velocity),
                    'target_speed': C.TARGET_SPEED,
                    'steer': float(action[0]),
                    'throttle': float(env.prev_throttle),
                    'brake': float(env.prev_brake),
                    'lane_deviation': priv_state.get('distance_to_center', 0.0),
                    'heading_error': priv_state.get('heading_error_deg', 0.0),
                    'route_completion': priv_state.get('route_completion', 0.0) * 100.0,
                    'distance_covered': priv_state.get('distance_covered', 0.0),
                    'stall_steps': env.stall_steps,
                    'max_stall_steps': C.MAX_STALL_STEPS,
                    'value': value,
                    'latency_ms': lat_ms
                }
                action_hud = visualizer.render(chase_img, sensor_obs['rgb'], bev_tensor, telemetry)
                if action_hud == 'quit':
                    print("[HUD] User requested termination via 'Q'.")
                    done = True

            sensor_obs = next_sensor_obs

        # Update policy every EPISODES_PER_BATCH episodes
        if ep % C.EPISODES_PER_BATCH == 0 and len(agent.buffer) > 0:
            agent.learn()

        comp = info.get('route_completion', 0.0) * 100.0
        dist = info.get('distance_covered', 0.0)
        dev = info.get('center_lane_deviation', 0.0)
        term = info.get('termination_reason', 'done')
        print(f"Train Ep {ep:3d}/{n_episodes:3d} | Reward: {ep_reward:7.1f} | Comp: {comp:4.1f}% ({dist:4.1f}m) | Dev: {dev:.2f}m | Term: {term:<12} | Steps: {step}")

        if ep % 10 == 0:
            agent.save(save_path)

    if len(agent.buffer) > 0:
        agent.learn()
    agent.save(save_path)
    print(f"[PPOAgent] Training complete. Checkpoint saved to: {save_path}\n")


def run_evaluation(env, pipeline, n_episodes, out_csv, visualizer=None):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    f_csv = open(out_csv, 'w', newline='')
    writer = csv.DictWriter(f_csv, fieldnames=CSV_FIELDS)
    writer.writeheader()

    print(f"\n=======================================================")
    print(f" PHASE 2 TRANSFUSER-PPO BENCHMARK EVALUATION")
    print(f" Town: {env.town} | Episodes: {n_episodes} | Obs Dim: {C.OBS_DIM}")
    print(f" Results CSV: {out_csv}")
    print(f" Visual HUD Render: {'ENABLED' if visualizer else 'HEADLESS'}")
    print(f"=======================================================\n")

    summary_rewards = []
    summary_completions = []
    summary_deviations = []
    summary_collisions = []
    latencies = []

    for ep in range(1, n_episodes + 1):
        sensor_obs = env.reset()
        ep_reward = 0.0
        done = False
        step = 0
        info = {}
        ep_latencies = []

        while not done and step < C.MAX_STEPS_PER_EP:
            step += 1
            action, _, _, value, _, lat_ms = pipeline.act(sensor_obs, deterministic=True)
            next_sensor_obs, priv_state, reward, done, info = env.step(action)

            ep_reward += reward
            ep_latencies.append(lat_ms)

            # Real-time HUD Visualization
            if visualizer is not None:
                chase_img = env.get_chase_image()
                bev_tensor = pipeline.projector.project(sensor_obs['lidar'])
                telemetry = {
                    'episode': ep,
                    'step': step,
                    'reward': ep_reward,
                    'speed': priv_state.get('velocity_kmh', env.velocity),
                    'target_speed': C.TARGET_SPEED,
                    'steer': float(action[0]),
                    'throttle': float(env.prev_throttle),
                    'brake': float(env.prev_brake),
                    'lane_deviation': priv_state.get('distance_to_center', 0.0),
                    'heading_error': priv_state.get('heading_error_deg', 0.0),
                    'route_completion': priv_state.get('route_completion', 0.0) * 100.0,
                    'distance_covered': priv_state.get('distance_covered', 0.0),
                    'stall_steps': env.stall_steps,
                    'max_stall_steps': C.MAX_STALL_STEPS,
                    'value': value,
                    'latency_ms': lat_ms
                }
                action_hud = visualizer.render(chase_img, sensor_obs['rgb'], bev_tensor, telemetry)
                if action_hud == 'quit':
                    print("[HUD] User requested termination via 'Q'.")
                    done = True

            sensor_obs = next_sensor_obs

        comp = info.get('route_completion', 0.0) * 100.0
        dist = info.get('distance_covered', 0.0)
        dev = info.get('center_lane_deviation', 0.0)
        term = info.get('termination_reason', 'done')
        collided = int(info.get('collided', False))

        # Automatically save termination snapshot if visualizer is active
        if visualizer is not None:
            chase_img = env.get_chase_image()
            bev_tensor = pipeline.projector.project(sensor_obs['lidar'])
            telemetry = {
                'episode': ep,
                'step': step,
                'reward': ep_reward,
                'speed': env.velocity,
                'target_speed': C.TARGET_SPEED,
                'steer': float(env.prev_steer),
                'throttle': float(env.prev_throttle),
                'brake': float(env.prev_brake),
                'lane_deviation': dev,
                'heading_error': info.get('heading_error_deg', 0.0),
                'route_completion': comp,
                'distance_covered': dist,
                'stall_steps': env.stall_steps,
                'max_stall_steps': C.MAX_STALL_STEPS,
                'value': 0.0,
                'latency_ms': 0.0
            }
            canvas_snap = visualizer.build_canvas(chase_img, sensor_obs['rgb'], bev_tensor, telemetry)
            visualizer.save_screenshot(canvas_snap, telemetry, custom_tag=f"{term}")

        lat_mean = float(np.mean(ep_latencies)) if ep_latencies else 0.0
        lat_p95  = float(np.percentile(ep_latencies, 95)) if ep_latencies else 0.0
        latencies.extend(ep_latencies)

        row = {
            'episode': ep,
            'reward': round(ep_reward, 2),
            'route_completion_pct': round(comp, 1),
            'distance_covered_m': round(dist, 1),
            'lane_deviation_m': round(dev, 3),
            'termination_reason': term,
            'collided': collided,
            'timesteps': step,
            'mean_speed_kmh': round(info.get('mean_speed_kmh', 0.0), 1),
            'latency_mean_ms': round(lat_mean, 2),
            'latency_p95_ms': round(lat_p95, 2)
        }
        writer.writerow(row)
        f_csv.flush()

        summary_rewards.append(ep_reward)
        summary_completions.append(comp)
        summary_deviations.append(dev)
        summary_collisions.append(collided)

        print(f"Ep {ep:2d}/{n_episodes:2d} | "
              f"Reward: {ep_reward:7.1f} | "
              f"Comp: {comp:4.1f}% ({dist:5.1f}m) | "
              f"Dev: {dev:.2f}m | "
              f"Term: {term:<15} | "
              f"Enc: {lat_mean:.1f}ms")

    f_csv.close()

    mean_r = float(np.mean(summary_rewards))
    std_r  = float(np.std(summary_rewards))
    mean_c = float(np.mean(summary_completions))
    std_c  = float(np.std(summary_completions))
    mean_d = float(np.mean(summary_deviations))
    mean_coll = float(np.mean(summary_collisions)) * 100.0 if summary_collisions else 0.0
    mean_lat = float(np.mean(latencies)) if latencies else 0.0
    fps = 1000.0 / mean_lat if mean_lat > 0 else 0.0

    print("\n" + "=" * 60)
    print(f" EVALUATION SUMMARY (PHASE 2 TRANSFUSER-PPO)")
    print(f" Mean Cumulative Reward:    {mean_r:.2f} +/- {std_r:.2f}")
    print(f" Mean Route Completion:    {mean_c:.1f}% +/- {std_c:.1f}%")
    print(f" Mean Collision Rate:      {mean_coll:.1f}%")
    print(f" Mean Lane Center Dev:     {mean_d:.2f} m")
    print(f" Mean Encoder Latency:     {mean_lat:.2f} ms ({fps:.0f} FPS)")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run Phase 2 TransFuser-PPO in CARLA")
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'])
    parser.add_argument('--town', type=str, default=C.DEFAULT_TOWN)
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--checkpoint', type=str, default=os.path.join(C.CHECKPOINT_DIR, 'transfuser_ppo.pth'))
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--render', action='store_true', help='Enable real-time OpenCV HUD visualizer')
    args = parser.parse_args()

    # Deferred import to ensure CARLA availability
    from phase2_sensory.carla_env import CarlaSensoryEnvironment

    print(f"[init] Initializing CARLA environment ({args.town}) ...")
    env = CarlaSensoryEnvironment(town=args.town, enable_render=args.render)
    visualizer = SensoryHUDVisualizer() if args.render else None

    pipeline = TransFuserPPOPipeline()

    if os.path.isfile(args.checkpoint):
        pipeline.agent.load(args.checkpoint)

    try:
        if args.mode == 'train':
            run_training(env, pipeline, args.episodes, args.checkpoint, visualizer=visualizer)
        else:
            tag_str = f"_{args.tag}" if args.tag else ""
            out_csv = os.path.join(C.RESULTS_DIR, f"eval_{args.town}_transfuser_ppo{tag_str}.csv")
            run_evaluation(env, pipeline, args.episodes, out_csv, visualizer=visualizer)
    finally:
        if visualizer is not None:
            visualizer.close()
        env.close()


if __name__ == '__main__':
    main()
