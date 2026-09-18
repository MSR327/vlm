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
    
    # Also include baseline probe results if present
    baseline_candidates = [
        os.path.join(models_dir, '..', '..', 'btp_prev', 'VAE', 'latent_probe_results.json'),
        os.path.join(models_dir, '..', '..', 'VAE', 'latent_probe_results.json'),
        '../btp_prev/VAE/latent_probe_results.json',
        '../../btp_prev/VAE/latent_probe_results.json'
    ]
    for bc in baseline_candidates:
        if os.path.isfile(bc) and bc not in probe_files:
            probe_files.append(bc)
            break

    data = []
    for pf in probe_files:
        try:
            with open(pf) as f:
                d = json.load(f)
            folder = os.path.basename(os.path.dirname(pf))
            if 'baseline' in pf.lower() or folder == 'VAE':
                arch = 'baseline'
                dz = 95
            else:
                parts = folder.split('_')
                arch = parts[1] if len(parts) > 1 else 'unknown'
                dz = int(parts[2]) if len(parts) > 2 else d.get('latent_dim', 95)
            
            data.append({
                'arch': arch,
                'latent_dim': dz,
                'effective_rank': round(d.get('effective_rank', np.nan), 2),
                'r2_mean': round(d.get('r2_mean', np.nan), 4),
                'r2_lane_offset': round(d.get('r2_scores', {}).get('lane_offset', np.nan), 4),
                'r2_heading': round(d.get('r2_scores', {}).get('lane_heading', np.nan), 4),
                'r2_road_frac': round(d.get('r2_scores', {}).get('road_frac', np.nan), 4),
                'latency_mean_ms': round(d.get('latency', {}).get('mean_ms', np.nan), 2),
                'fps': round(d.get('latency', {}).get('fps', np.nan), 1),
            })
        except Exception as e:
            print(f"Warning: could not parse {pf}: {e}")
    df = pd.DataFrame(data)
    if not df.empty:
        df = df.sort_values(by=['arch', 'latent_dim']).reset_index(drop=True)
    return df


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
                'episodes': len(df),
                'route_completion_mean': round(df['route_completion'].mean(), 1),
                'reward_mean': round(df['reward'].mean(), 2),
                'reward_std': round(df['reward'].std(), 2),
                'lane_dev_mean': round(df['center_lane_deviation_m'].mean(), 3),
                'collision_rate': round(df['collided'].mean() * 100.0, 1),
                'enc_latency_p95': round(df['encoder_latency_p95_ms'].mean(), 2) if 'encoder_latency_p95_ms' in df else np.nan
            })
        except Exception as e:
            print(f"Warning: could not parse {cf}: {e}")
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(by=['arch', 'latent_dim']).reset_index(drop=True)
    return df


def _print_df(df):
    try:
        print(df.to_markdown(index=False))
    except Exception:
        print(df.to_string(index=False))


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
        _print_df(df_probe)

    if not df_carla.empty:
        print("\n--- CARLA Closed-Loop Driving Benchmark Results ---")
        _print_df(df_carla)

    if df_probe.empty and df_carla.empty:
        print("No evaluation results found yet.")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
