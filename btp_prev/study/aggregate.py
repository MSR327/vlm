"""
Aggregate the ladder into the comparison table.

Reads whatever exists and says plainly what is missing, rather than silently
producing a half-empty table:

  results/study/cost_latency.json          <- study/bench_latency.py   (offline)
  results/study/latent_probe.json          <- study/latent_probe.py    (offline)
  Results_VAE_<rung>_<arm>/test_results_*.csv  <- run_ladder.py --mode test (CARLA)

Usage:  python study/aggregate.py [--out results/study/ladder_summary.md]
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

RUNGS = [
    (1, '1cam',       'VAE + 1 RGB (100-dim)'),
    (2, '4cam',       'VAE + 4 RGB 360° (385-dim)'),
    (2, '3cam',       'VAE + 3 RGB Surround (290-dim)'),
    (3, '1cam_lidar', 'VAE + 1 RGB + LiDAR (195-dim)'),
]
ARMS = ['raw', 'scaled']


def load_closed_loop():
    rows = []
    for rung, tag, label in RUNGS:
        for arm in ARMS:
            for path in glob.glob(f'Results_VAE_{tag}_{arm}/test_results_*.csv'):
                df = pd.read_csv(path)
                if df.empty:
                    continue
                rows.append(dict(
                    rung=rung, config=label, arm=arm,
                    town=df['town'].iloc[0] if 'town' in df else '?',
                    episodes=len(df),
                    obs_dim=int(df['obs_dim'].iloc[0]),
                    reward_mean=df['reward'].mean(), reward_std=df['reward'].std(),
                    success_rate=df['success'].mean(),
                    collision_rate=df['collided'].mean(),
                    route_completion_mean=df['route_completion'].mean(),
                    route_completion_std=df['route_completion'].std(),
                    distance_m_mean=df['distance_covered_m'].mean(),
                    lane_dev_mean=df['center_lane_deviation_m'].mean(),
                    enc_latency_mean_ms=df['encoder_latency_mean_ms'].mean(),
                    enc_latency_p95_ms=df['encoder_latency_p95_ms'].mean(),
                    mean_speed_kmh=df['mean_speed_kmh'].mean(),
                    source=path))
    return pd.DataFrame(rows)


def fmt_table(df, cols, headers=None):
    headers = headers or cols
    out = ['| ' + ' | '.join(headers) + ' |',
           '|' + '|'.join(['---'] * len(cols)) + '|']
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(f'{v:.3f}' if isinstance(v, (float, np.floating)) else str(v))
        out.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results/study/ladder_summary.md')
    args = ap.parse_args()

    md = ['# VAE baseline ladder — results summary', '']

    # ---- cost / latency -------------------------------------------------
    cl_path = 'results/study/cost_latency.json'
    if os.path.exists(cl_path):
        cl = json.load(open(cl_path))
        df = pd.DataFrame(cl['rows'])
        df['MACs_M'] = df['inference_macs'] / 1e6
        md += ['## Computational cost (measured on %s, TF %s)' % (cl['device'], cl['tf']), '',
               fmt_table(df,
                         ['name', 'obs_dim', 'encoder_params', 'actor_params',
                          'total_params', 'MACs_M', 'inference_fp32_mb',
                          'encoder_mean_ms', 'encoder_p95_ms', 'policy_mean_ms',
                          'pipeline_mean_ms', 'max_rate_hz'],
                         ['config', 'obs dim', 'encoder params', 'actor params',
                          'total params', 'MACs (M)', 'infer size (MB)',
                          'encoder ms', 'encoder p95', 'policy ms',
                          'pipeline ms', 'max Hz']), '']
    else:
        md += ['## Computational cost', '', '_missing — run `python study/bench_latency.py`_', '']

    # ---- latent probe ---------------------------------------------------
    lp_path = 'results/study/latent_probe.json'
    if os.path.exists(lp_path):
        lp = json.load(open(lp_path))
        rows = []
        for c in lp['conditions']:
            row = dict(layout=c['layout'], arm=c['arm'],
                       latent_std=c['latent_std'],
                       eff_rank=c['effective_rank'],
                       nonfinite_pct=100 * c.get('nonfinite_frac_train', 0.0),
                       mean_R2=c['r2_mean'])
            row.update({f'R2_{k}': v for k, v in c['r2'].items()})
            rows.append(row)
        df = pd.DataFrame(rows)
        md += [f"## Latent information (n_train={lp['n_train']}, n_test={lp['n_test']})", '',
               fmt_table(df, list(df.columns)), '',
               f"Pixel baseline (16x8 grey, 128 dims): mean R² = "
               f"{lp['pixel_baseline_mean']:.4f}", '',
               '### OOD response — BEV grid through the camera VAE', '',
               fmt_table(pd.DataFrame([lp['ood']]), list(lp['ood'].keys())), '']
    else:
        md += ['## Latent information', '', '_missing — run `python study/latent_probe.py`_', '']

    # ---- multi-view fusion proxy ----------------------------------------
    mv_path = 'results/study/multiview_proxy.json'
    if os.path.exists(mv_path):
        mv = json.load(open(mv_path))
        rows = [dict(variant=k, dims=v['dims'],
                     eff_rank=v['effective_rank'], mean_R2=v['r2_mean'],
                     **{f'R2_{t}': r for t, r in v['r2'].items()})
                for k, v in mv['variants'].items()]
        md += ['## Multi-view fusion mechanism (proxy: left/right half-frames)', '',
               fmt_table(pd.DataFrame(rows), list(rows[0].keys())), '',
               f"Gain from concatenating a second view: "
               f"**+{mv['fusion_gain']:.4f}** mean R² over the better single view "
               f"({mv['best_single_r2']:.4f} → {mv['concat_r2']:.4f}).", '',
               f"Rank additivity: {mv['rank_if_independent']:.2f} expected if "
               f"independent, {mv['variants']['concat [A, B]']['effective_rank']:.2f} "
               f"actual = **{mv['rank_additivity']*100:.0f}% additive**. "
               f"The 190 concatenated dims carry only "
               f"{mv['concat_dim_utilisation']*100:.1f}% as many effective "
               f"dimensions.", '']
    else:
        md += ['## Multi-view fusion mechanism', '',
               '_missing — run `python study/multiview_proxy.py`_', '']

    # ---- closed loop ----------------------------------------------------
    cl_df = load_closed_loop()
    if len(cl_df):
        md += ['## Closed-loop driving (CARLA)', '',
               fmt_table(cl_df.sort_values(['arm', 'rung']),
                         ['config', 'arm', 'town', 'episodes', 'obs_dim',
                          'reward_mean', 'reward_std', 'success_rate',
                          'collision_rate', 'route_completion_mean',
                          'distance_m_mean', 'lane_dev_mean',
                          'enc_latency_mean_ms', 'mean_speed_kmh'],
                         ['config', 'arm', 'town', 'eps', 'obs',
                          'reward', '±', 'success', 'collision',
                          'route compl.', 'distance m', 'lane dev m',
                          'enc ms', 'speed km/h']), '']
    else:
        md += ['## Closed-loop driving (CARLA)', '',
               '_no results yet. On the CARLA machine:_', '', '```bash',
               'for ARM in raw scaled; do for R in 1 2 3; do',
               '  BTP_RUNG=$R BTP_NORM=$ARM python baseline_vae/run_ladder.py --mode train',
               '  BTP_RUNG=$R BTP_NORM=$ARM python baseline_vae/run_ladder.py --mode test --town Town02',
               'done; done', '```', '']

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out, 'w').write('\n'.join(md))
    print('\n'.join(md))
    print(f"\nwrote {args.out}")


if __name__ == '__main__':
    main()
