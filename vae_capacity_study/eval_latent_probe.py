"""
Offline Latent Probe & Representation Quality Analyzer.

Evaluates how well a trained VAE encoder preserves driving-critical affordances:
  - Linear probe (ridge regression) R^2 for lane offset, heading, road fraction, etc.
  - Effective dimensionality / participation ratio: (sum lambda)^2 / sum (lambda^2)
  - Latent numerical stability (std, absmax, NaN/inf rates)
  - Forward-pass inference latency (mean ms, p95 ms, FPS)
"""
import argparse
import glob
import json
import os
import sys
import time

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import numpy as np
from PIL import Image
import tensorflow as tf

from semantic_targets import extract_targets, TARGET_NAMES
from vae_loader import load_vae_encoder


def load_dataset_frames(data_dir, split='test', limit=2000):
    candidates = [
        os.path.join(data_dir, split, 'class1', '*.png'),
        os.path.join(data_dir, split, '*.png'),
        os.path.join('../btp_prev/VAE/dataset', split, 'class1', '*.png')
    ]
    paths = []
    for c in candidates:
        found = sorted(glob.glob(c))
        if found:
            paths = found
            break

    if not paths:
        raise FileNotFoundError(f"Could not find {split} images in {data_dir}")

    if limit:
        paths = paths[:limit]

    n = len(paths)
    x_frames = np.empty((n, 160, 80, 3), dtype=np.float32)
    y_targets = np.empty((n, len(TARGET_NAMES)), dtype=np.float32)

    for i, p in enumerate(paths):
        im = Image.open(p).convert('RGB')
        arr = np.asarray(im, dtype=np.uint8)
        # Resize to (80, 160) then transpose to (160, 80, 3) in [0, 1]
        im_resized = im.resize((80, 160), Image.BILINEAR)
        x_frames[i] = np.asarray(im_resized, dtype=np.float32) / 255.0
        y_targets[i] = extract_targets(arr)

    return x_frames, y_targets


def effective_rank(z):
    """
    Participation ratio of the covariance spectrum: (sum lambda)^2 / sum (lambda^2).
    Measures the number of active, non-redundant orthogonal dimensions in latent space.
    """
    zc = z - z.mean(axis=0)
    cov = np.cov(zc, rowvar=False)
    if cov.ndim == 0:
        return 1.0
    ev = np.linalg.eigvalsh(cov)
    ev = np.clip(ev, 0.0, None)
    total_ev = ev.sum()
    if total_ev <= 1e-8:
        return 0.0
    return float((total_ev ** 2) / np.sum(ev ** 2))


def linear_probe_r2(z_train, y_train, z_test, y_test, alpha=10.0):
    """Fits linear ridge regression from latent z to driving affordance targets y."""
    n_features = z_train.shape[1]
    # Center variables
    ymu = y_train.mean(axis=0)
    ysd = y_train.std(axis=0) + 1e-8
    y_tr_norm = (y_train - ymu) / ysd
    y_te_norm = (y_test - ymu) / ysd

    # Ridge solution: W = (Z^T Z + alpha * I)^-1 Z^T Y
    I = np.eye(n_features)
    reg = alpha * I
    try:
        W = np.linalg.solve(z_train.T @ z_train + reg, z_train.T @ y_tr_norm)
        pred_norm = z_test @ W
        pred_y = pred_norm * ysd + ymu

        ss_res = np.sum((pred_y - y_test) ** 2, axis=0)
        ss_tot = np.sum((y_test - ymu) ** 2, axis=0) + 1e-8
        r2 = 1.0 - (ss_res / ss_tot)
        return r2
    except np.linalg.LinAlgError:
        return np.zeros(y_train.shape[1])


def benchmark_latency(encoder_fn, dummy_input, n_warmup=20, n_runs=100):
    for _ in range(n_warmup):
        _ = encoder_fn(dummy_input)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = encoder_fn(dummy_input)
        times.append((time.perf_counter() - t0) * 1000.0)

    times = np.array(times)
    return {
        'mean_ms': float(np.mean(times)),
        'p50_ms':  float(np.median(times)),
        'p95_ms':  float(np.percentile(times, 95)),
        'p99_ms':  float(np.percentile(times, 99)),
        'fps':     float(1000.0 / np.mean(times))
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help="Path to SavedModel encoder directory")
    parser.add_argument('--data', default='../btp_prev/VAE/dataset', help="Path to dataset directory")
    parser.add_argument('--train_limit', type=int, default=4000, help="Number of train frames to probe")
    parser.add_argument('--test_limit',  type=int, default=1000, help="Number of test frames to probe")
    parser.add_argument('--out', default=None, help="Output JSON path")
    args = parser.parse_args()

    print(f"\nEvaluating Latent Quality for: {args.model}")
    encoder_fn = load_vae_encoder(args.model)

    print("Loading test and train frames ...")
    x_tr, y_tr = load_dataset_frames(args.data, 'train', limit=args.train_limit)
    x_te, y_te = load_dataset_frames(args.data, 'test',  limit=args.test_limit)

    print("Encoding frames ...")
    z_tr = np.asarray(encoder_fn(x_tr))
    z_te = np.asarray(encoder_fn(x_te))

    latent_dim = z_tr.shape[1]
    er = effective_rank(z_tr)
    r2_scores = linear_probe_r2(z_tr, y_tr, z_te, y_te)

    # Benchmark latency with single frame (batch=1)
    dummy_frame = x_te[:1]
    lat_stats = benchmark_latency(encoder_fn, dummy_frame)

    results = {
        'model_path': args.model,
        'latent_dim': latent_dim,
        'effective_rank': er,
        'effective_rank_ratio': float(er / latent_dim),
        'latent_std': float(np.std(z_tr)),
        'latent_absmax': float(np.max(np.abs(z_tr))),
        'has_nan': bool(np.isnan(z_tr).any()),
        'has_inf': bool(np.isinf(z_tr).any()),
        'r2_scores': {name: float(score) for name, score in zip(TARGET_NAMES, r2_scores)},
        'r2_mean': float(np.mean(r2_scores)),
        'latency': lat_stats
    }

    print("\n" + "=" * 60)
    print(f" LATENT QUALITY & EFFICIENCY REPORT")
    print(f" Latent Dimension:        {latent_dim}")
    print(f" Effective Rank:          {er:.2f} / {latent_dim} ({results['effective_rank_ratio']*100:.1f}%)")
    print(f" Latent Std Dev:          {results['latent_std']:.3f} (Max: {results['latent_absmax']:.2f})")
    print(f" Mean Affordance R^2:     {results['r2_mean']:.4f}")
    for name, s in results['r2_scores'].items():
        print(f"   - {name:<16}: R^2 = {s:.4f}")
    print(f" Inference Latency:       {lat_stats['mean_ms']:.2f} ms (p95: {lat_stats['p95_ms']:.2f} ms, {lat_stats['fps']:.1f} FPS)")
    print("=" * 60 + "\n")

    out_json = args.out or os.path.join(os.path.dirname(args.model), 'latent_probe_results.json')
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {out_json}")


if __name__ == '__main__':
    main()
