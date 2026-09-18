"""
Aggregator and Table Generator for VAE Capacity & Latent Scaling Study.

Scans evaluation CSVs and latent probe JSONs, summarizing:
  - Model Parameters & Layers
  - Reconstruction MSE & KL Divergence
  - Effective Rank & Information Decodability (R^2)
  - Closed-Loop CARLA Performance (Completion %, Reward, Latency)
Outputs comparison tables in Markdown and LaTeX.
"""
import glob
import json
import os
import pandas as pd
import numpy as np


def collect_probe_results(models_dir):
    probe_files = glob.glob(os.path.join(models_dir, '*', 'latent_probe_results.json'))
    data = []
    for pf in probe_files:
        try:
            with open(pf) as f:
                d = json.load(f)
            folder = os.path.basename(os.path.dirname(pf))
            parts = folder.split('_')
            arch = parts[1] if len(parts) > 1 else 'unknown'
            dz = int(parts[2]) if len(parts) > 2 else d.get('latent_dim', 95)
            data.append({
                'arch': arch,
                'latent_dim': dz,
                'effective_rank': d.get('effective_rank', np.nan),
                'r2_mean': d.get('r2_mean', np.nan),
                'r2_lane_offset': d.get('r2_scores', {}).get('lane_offset', np.nan),
                'r2_heading': d.get('r2_scores', {}).get('lane_heading', np.nan),
                'latency_mean_ms': d.get('latency', {}).get('mean_ms', np.nan),
                'fps': d.get('latency', {}).get('fps', np.nan),
            })
        except Exception as e:
            print(f"Warning: could not parse {pf}: {e}")
    return pd.DataFrame(data)


def collect_carla_results(results_dir):
    csv_files = glob.glob(os.path.join(results_dir, 'Results_VAE_*', '*.csv'))
    rows = []
    for cf in csv_files:
        try:
            df = pd.read_csv(cf)
            if df.empty:
                continue
            arch = df['arch'].iloc[0] if 'arch' in df else 'unknown'
            dz = df['latent_dim'].iloc[0] if 'latent_dim' in df else 95
            rows.append({
                'arch': arch,
                'latent_dim': dz,
                'route_completion_mean': df['route_completion'].mean(),
                'route_completion_std': df['route_completion'].std(),
                'reward_mean': df['reward'].mean(),
                'reward_std': df['reward'].std(),
                'lane_dev_mean': df['center_lane_deviation_m'].mean(),
                'step_latency_ms': df['step_latency_mean_ms'].mean() if 'step_latency_mean_ms' in df else np.nan,
                'n_episodes': len(df)
            })
        except Exception as e:
            print(f"Warning: could not parse {cf}: {e}")
    return pd.DataFrame(rows)


def main():
    root = os.path.dirname(__file__)
    models_dir = os.path.join(root, 'models')
    df_probe = collect_probe_results(models_dir)
    df_carla = collect_carla_results(root)

    print("\n" + "=" * 80)
    print(" VAE CAPACITY & LATENT SCALING COMPARATIVE SUMMARY")
    print("=" * 80)

    if not df_probe.empty:
        print("\n--- Latent Space Quality & Probing Results ---")
        print(df_probe.to_markdown(index=False))

    if not df_carla.empty:
        print("\n--- CARLA Closed-Loop Driving Benchmark Results ---")
        print(df_carla.to_markdown(index=False))

    if df_probe.empty and df_carla.empty:
        print("No evaluation results found yet. Run training and evaluation first:")
        print("  1. python train_vae_capacity.py --arch wide --latent 95")
        print("  2. python eval_latent_probe.py --model models/vae_wide_95/var_auto_encoder_model")
        print("  3. python run_capacity_ladder.py --arch wide --latent 95 --mode test")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
