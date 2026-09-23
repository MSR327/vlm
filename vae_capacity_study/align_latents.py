"""
Latent Representation Alignment Adapter for Zero-Shot Policy Transfer.

Solves the coordinate rotation and scale mismatch between newly trained
capacity VAEs (Wide, Deep) and the pre-trained baseline PPO policy.

Fits a closed-form Ridge Regression mapping:
    Z_baseline ≈ Z_target @ W_align + b_align

Saves alignment.npz to the model directory, enabling EncodeStateCapacity
to automatically translate coordinates for zero-shot closed-loop driving.
"""
import argparse
import glob
import os
import sys
import time

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import numpy as np
import tensorflow as tf

from vae_loader import load_vae_encoder


def find_dataset(data_arg):
    candidates = [
        data_arg,
        os.path.join(os.path.dirname(__file__), data_arg),
        os.path.join(os.path.dirname(__file__), '../btp_prev/VAE/dataset'),
        'btp_prev/VAE/dataset',
        'VAE/dataset',
    ]
    for c in candidates:
        if os.path.isdir(os.path.join(c, 'train')):
            return os.path.abspath(c)
    raise FileNotFoundError(f"Could not locate dataset in any candidate path: {candidates}")


def load_sample_images(data_dir, num_samples=2000, target_size=(160, 80)):
    """Loads a random sample of num_samples images as float32 in [0, 1]."""
    img_paths = glob.glob(os.path.join(data_dir, 'train', '**', '*.png'), recursive=True)
    if not img_paths:
        img_paths = glob.glob(os.path.join(data_dir, '**', '*.png'), recursive=True)
    if not img_paths:
        raise FileNotFoundError(f"No PNG images found in {data_dir}")

    rng = np.random.default_rng(42)
    selected = rng.choice(img_paths, size=min(num_samples, len(img_paths)), replace=False)
    print(f"[align] Loading {len(selected)} image frames from {data_dir} ...")

    images = []
    for p in selected:
        img_raw = tf.io.read_file(p)
        img = tf.image.decode_png(img_raw, channels=3)
        img = tf.image.resize(img, target_size)
        img = tf.cast(img, tf.float32) / 255.0
        images.append(img.numpy())

    return np.stack(images, axis=0)  # (N, 160, 80, 3)


def fit_ridge_alignment(z_target, z_baseline, alpha=1.0):
    """
    Fits Ridge Regression: Z_target @ W + b ≈ Z_baseline
    Returns W, b, and mean R^2 score.
    """
    N, d_in = z_target.shape
    _, d_out = z_baseline.shape

    # Center target and baseline
    mean_target = np.mean(z_target, axis=0)
    mean_base = np.mean(z_baseline, axis=0)

    Z_c = z_target - mean_target
    Y_c = z_baseline - mean_base

    # Closed-form Ridge: W = (Z^T Z + alpha * I)^{-1} Z^T Y
    reg = alpha * np.eye(d_in, dtype=np.float32)
    W = np.linalg.solve(Z_c.T @ Z_c + reg, Z_c.T @ Y_c)
    b = mean_base - mean_target @ W

    # Evaluate fit
    z_pred = z_target @ W + b
    ss_res = np.sum((z_baseline - z_pred) ** 2, axis=0)
    ss_tot = np.sum((z_baseline - mean_base) ** 2, axis=0) + 1e-8
    r2_per_dim = 1.0 - (ss_res / ss_tot)
    mean_r2 = float(np.mean(r2_per_dim))

    return W.astype(np.float32), b.astype(np.float32), mean_r2, r2_per_dim


def main():
    parser = argparse.ArgumentParser(description="Fit Latent Alignment Adapter for VAE Capacity Transfer")
    parser.add_argument('--arch', type=str, required=True, choices=['baseline', 'wide', 'deep'],
                        help="Target VAE architecture to align")
    parser.add_argument('--latent', type=int, default=95,
                        help="Target VAE latent dimension dz")
    parser.add_argument('--samples', type=int, default=2000,
                        help="Number of image samples for alignment fitting")
    parser.add_argument('--alpha', type=float, default=1.0,
                        help="Ridge regularization strength")
    parser.add_argument('--data', type=str, default='../btp_prev/VAE/dataset',
                        help="Path to dataset directory")
    parser.add_argument('--raw-base', dest='raw_base', action='store_true', default=True,
                        help="Feed unscaled [0, 255] frames to baseline encoder matching Results_05 deployed policy (default: True)")
    parser.add_argument('--scaled-base', dest='raw_base', action='store_false',
                        help="Feed scaled [0, 1] frames to baseline encoder")
    args = parser.parse_args()

    data_dir = find_dataset(args.data)
    images = load_sample_images(data_dir, num_samples=args.samples)

    # 1. Load Baseline Encoder
    base_candidates = [
        '../btp_prev/VAE/var_auto_encoder_model',
        'btp_prev/VAE/var_auto_encoder_model',
        'VAE/var_auto_encoder_model'
    ]
    base_model_path = None
    for bc in base_candidates:
        if os.path.isdir(bc):
            base_model_path = bc
            break
    if not base_model_path:
        raise FileNotFoundError(f"Could not locate baseline VAE model in {base_candidates}")

    print(f"[align] Loading baseline VAE from: {base_model_path}")
    base_encoder = load_vae_encoder(base_model_path)

    # 2. Load Target Capacity Encoder
    model_dir = os.path.join(
        os.path.dirname(__file__),
        'models',
        f'vae_{args.arch}_{args.latent}'
    )
    target_candidates = [
        os.path.join(model_dir, 'var_auto_encoder_model'),
        os.path.join(model_dir, 'encoder'),
        model_dir
    ]
    target_model_path = None
    for tc in target_candidates:
        if os.path.isdir(tc):
            target_model_path = tc
            break
    if not target_model_path:
        raise FileNotFoundError(f"Could not locate target VAE model in {target_candidates}")

    print(f"[align] Loading target {args.arch.upper()}-{args.latent} VAE from: {target_model_path}")
    target_encoder = load_vae_encoder(target_model_path)

    # 3. Extract Representations
    print(f"[align] Computing latent representations (raw_base={args.raw_base}) ...")
    batch_size = 64
    z_base_list = []
    z_target_list = []

    for i in range(0, len(images), batch_size):
        b = images[i:i + batch_size]
        b_base = b * 255.0 if args.raw_base else b
        zb = np.asarray(base_encoder(b_base))
        zb = np.clip(np.nan_to_num(zb, nan=0.0, posinf=1e8, neginf=-1e8), -1e8, 1e8)
        zt = np.asarray(target_encoder(b))
        zt = np.clip(np.nan_to_num(zt, nan=0.0, posinf=1e8, neginf=-1e8), -1e8, 1e8)
        z_base_list.append(zb)
        z_target_list.append(zt)

    z_base = np.concatenate(z_base_list, axis=0)
    z_target = np.concatenate(z_target_list, axis=0)

    print(f"[align] Baseline Z shape: {z_base.shape} | mean: {z_base.mean():.3f}, std: {z_base.std():.3f}")
    print(f"[align] Target   Z shape: {z_target.shape} | mean: {z_target.mean():.3f}, std: {z_target.std():.3f}")

    # 4. Fit Ridge Alignment
    print(f"[align] Fitting Ridge regression adapter (alpha={args.alpha}) ...")
    W, b, mean_r2, r2_per_dim = fit_ridge_alignment(z_target, z_base, alpha=args.alpha)

    print(f"\n=======================================================")
    print(f" ALIGNMENT ADAPTER FIT: {args.arch.upper()} (dz={args.latent}) -> BASELINE (dz=95)")
    print(f" Mean Cross-Latent R^2: {mean_r2:.4f}")
    print(f" Top 5 Dim R^2: {np.sort(r2_per_dim)[-5:]}")
    print(f" Transformation Shape: W {W.shape}, b {b.shape}")
    print(f"=======================================================\n")

    # 5. Save alignment artifact
    out_file = os.path.join(model_dir, 'alignment.npz')
    np.savez_compressed(
        out_file,
        W=W,
        b=b,
        mean_r2=mean_r2,
        r2_per_dim=r2_per_dim,
        source_arch=args.arch,
        source_latent=args.latent
    )
    print(f"[align] Successfully saved alignment adapter to: {out_file}")


if __name__ == '__main__':
    main()
