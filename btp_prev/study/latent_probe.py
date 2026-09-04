"""
How much driving-relevant information survives the frozen 95-d VAE latent?

This is the offline half of the ladder study. It cannot measure what four
cameras add (there is no multi-view data without CARLA), but it measures the
ENCODER that all three rungs share -- so every finding here propagates to
every rung, and it settles two things the closed-loop runs cannot separate:

  1. the normalisation arm            (raw 0-255 vs scaled 0-1)
  2. the frame-layout defect          (main.py's reshape((W,H,4)) byte
                                       reinterpretation vs a real image)

METHOD
  * 12,000 train / 2,000 test CityScapesPalette frames (VAE/dataset).
  * Targets are read off the segmentation labels themselves
    (study/semantic_targets.py) -- ground truth, not estimates.
  * Probe = ridge regression, latent -> target, fit on train, R^2 on test.
    Linear on purpose: it asks "is the information linearly available to the
    PPO MLP", which is the question that matters for the downstream policy.
  * Pixel baseline = ridge on a 16x8 grey downsample (128 dims) of the true
    frame. Not an upper bound on what a perfect encoder could do, but a
    reference point: a 95-d learned latent that loses to 128 raw grey pixels
    is not earning its 13.7M parameters.
  * Effective rank of the latent (participation ratio of the PCA spectrum)
    bounds what concatenating N views can add: if one view only occupies k
    effective dimensions, N views give at most N*k, not 95*N.
"""
import glob
import json
import os
import sys

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

import numpy as np
from PIL import Image

sys.path.insert(0, 'study')
from semantic_targets import extract_targets, TARGET_NAMES

BATCH = 256


# ----------------------------------------------------------------------
def carla_legacy_layout(img_hwc):
    """Reproduce main.py:549 exactly.

    CARLA hands over a row-major (H, W, 4) BGRA buffer and main.py does
    reshape((image.width, image.height, 4)). Since H*W*4 == W*H*4 that does not
    raise -- it reinterprets the bytes. Each output row of 80 pixels straddles
    half of a true 160-pixel row, so the encoder never sees a picture.
    """
    h, w = img_hwc.shape[:2]
    bgra = np.dstack([img_hwc[:, :, ::-1], np.full((h, w, 1), 255, np.uint8)])
    return np.frombuffer(bgra.tobytes(), np.uint8).reshape((w, h, 4))[:, :, :3]


def train_style_layout(pil_img):
    """Reproduce VAE_Trainer_dont_touch.py:126-132.

    flow_from_directory(target_size=(160, 80)) means (height=160, width=80),
    so the 160x80 (WxH) PNG is resized to 80 wide by 160 tall -- a real image,
    just stretched. This is what the VAE's weights were actually fitted to.
    """
    return np.asarray(pil_img.resize((80, 160), Image.BILINEAR), dtype=np.uint8)


# ----------------------------------------------------------------------
def load_split(split, limit=None):
    paths = sorted(glob.glob(f'VAE/dataset/{split}/class1/*.png'))
    if limit:
        paths = paths[:limit]
    n = len(paths)
    true_hwc = np.empty((n, 80, 160, 3), np.uint8)
    x_train_style = np.empty((n, 160, 80, 3), np.uint8)
    x_carla = np.empty((n, 160, 80, 3), np.uint8)
    targets = np.empty((n, len(TARGET_NAMES)), np.float32)

    for i, p in enumerate(paths):
        im = Image.open(p).convert('RGB')
        arr = np.asarray(im, dtype=np.uint8)
        true_hwc[i] = arr
        x_train_style[i] = train_style_layout(im)
        x_carla[i] = carla_legacy_layout(arr)
        targets[i] = extract_targets(arr)
        if (i + 1) % 4000 == 0:
            print(f"    {split}: {i+1}/{n}")
    return dict(true=true_hwc, train_style=x_train_style, carla=x_carla,
                y=targets, n=n)


def encode_all(encoder, x_uint8, scale):
    """Encode, then apply the SAME sanitisation the deployed actor applies.

    In the raw arm the encoder's `sigma = tf.exp(dense(x))` head overflows on
    some frames and emits inf/NaN latents. Actor.normalize (main.py:701) is
    `tf.clip_by_value(obs, -1e8, 1e8)`, which clamps +/-inf to +/-1e8 but
    passes NaN straight through into the policy. We reproduce that clamp and
    additionally zero the NaNs so the probe is computable, and report the rate
    -- a NaN reaching the PPO update would poison every gradient in the batch.
    """
    out = np.empty((len(x_uint8), 95), np.float64)
    n_nan = n_inf = 0
    for i in range(0, len(x_uint8), BATCH):
        chunk = x_uint8[i:i + BATCH].astype(np.float32) * scale
        z = np.asarray(encoder(chunk), dtype=np.float64)
        n_nan += int(np.isnan(z).sum())
        n_inf += int(np.isinf(z).sum())
        z = np.clip(np.nan_to_num(z, nan=0.0, posinf=1e8, neginf=-1e8), -1e8, 1e8)
        out[i:i + BATCH] = z
    stats = dict(nan_elements=n_nan, inf_elements=n_inf,
                 nonfinite_frac=float((n_nan + n_inf) / out.size))
    return out, stats


# ----------------------------------------------------------------------
def ridge_probe(z_tr, y_tr, z_te, y_te, alphas=(1e-3, 1e-1, 1e1, 1e3, 1e5)):
    """Standardised ridge with alpha picked on a held-out slice of train.

    Returns per-target R^2 on test. Standardisation matters here: the raw arm's
    latents have std ~2900, and an unscaled ridge penalty would be meaningless
    across arms.
    """
    z_tr = np.asarray(z_tr, np.float64)
    z_te = np.asarray(z_te, np.float64)
    y_tr = np.asarray(y_tr, np.float64)
    y_te = np.asarray(y_te, np.float64)

    mu, sd = z_tr.mean(0), z_tr.std(0) + 1e-8
    Ztr, Zte = (z_tr - mu) / sd, (z_te - mu) / sd
    ymu, ysd = y_tr.mean(0), y_tr.std(0) + 1e-8

    cut = int(len(Ztr) * 0.8)
    A, B = Ztr[:cut], Ztr[cut:]
    ya, yb = y_tr[:cut], y_tr[cut:]

    def fit(X, Y, a):
        Xb = np.hstack([X, np.ones((len(X), 1), np.float32)])
        G = Xb.T @ Xb
        reg = a * np.eye(G.shape[0]); reg[-1, -1] = 0.0
        return np.linalg.solve(G + reg, Xb.T @ Y)

    def pred(W, X):
        return np.hstack([X, np.ones((len(X), 1), np.float32)]) @ W

    best_a, best = alphas[-1], -np.inf
    for a in alphas:
        try:
            W = fit(A, (ya - ymu) / ysd, a)
        except np.linalg.LinAlgError:
            continue
        r = 1 - ((pred(W, B) - (yb - ymu) / ysd) ** 2).sum() / \
            (((yb - ymu) / ysd - ((ya - ymu) / ysd).mean(0)) ** 2).sum()
        if np.isfinite(r) and r > best:
            best, best_a = r, a

    W = fit(Ztr, (y_tr - ymu) / ysd, best_a)
    P = pred(W, Zte) * ysd + ymu
    ss_res = ((P - y_te) ** 2).sum(0)
    ss_tot = ((y_te - y_tr.mean(0)) ** 2).sum(0)
    return (1 - ss_res / ss_tot), best_a


def effective_rank(z):
    """Participation ratio of the covariance spectrum: (sum l)^2 / sum l^2.

    A latent using all 95 dimensions equally scores 95. One dominated by a
    single direction scores ~1. This bounds what concatenating views can add.
    """
    zc = z - z.mean(0)
    ev = np.linalg.eigvalsh(np.cov(zc, rowvar=False))
    ev = np.clip(ev, 0, None)
    if ev.sum() <= 0:
        return 0.0
    return float(ev.sum() ** 2 / (ev ** 2).sum())


# ----------------------------------------------------------------------
def main():
    import tensorflow as tf
    sm = tf.saved_model.load('VAE/var_auto_encoder_model')
    fn = sm.signatures['serving_default']
    def encoder(x):
        return fn(input_1=tf.convert_to_tensor(x, tf.float32))['output_1']

    print("Loading dataset ...")
    tr = load_split('train')
    te = load_split('test')
    print(f"  train {tr['n']}  test {te['n']}")

    conditions = [
        ('train_style', 'scaled', 'train_style', 1 / 255.0),
        ('train_style', 'raw',    'train_style', 1.0),
        ('carla_legacy', 'scaled', 'carla',      1 / 255.0),
        ('carla_legacy', 'raw',    'carla',      1.0),     # <- as deployed
    ]

    results = []
    for layout, arm, key, scale in conditions:
        print(f"\n--- layout={layout} arm={arm} ---")
        ztr, s_tr = encode_all(encoder, tr[key], scale)
        zte, s_te = encode_all(encoder, te[key], scale)
        r2, alpha = ridge_probe(ztr, tr['y'], zte, te['y'])
        er = effective_rank(ztr)
        results.append(dict(layout=layout, arm=arm, alpha=float(alpha),
                            nonfinite_frac_train=s_tr['nonfinite_frac'],
                            nonfinite_frac_test=s_te['nonfinite_frac'],
                            nan_elements_train=s_tr['nan_elements'],
                            inf_elements_train=s_tr['inf_elements'],
                            latent_std=float(ztr.std()),
                            latent_absmax=float(np.abs(ztr).max()),
                            effective_rank=er,
                            r2={n: float(v) for n, v in zip(TARGET_NAMES, r2)},
                            r2_mean=float(np.mean(r2))))
        print(f"  latent std {ztr.std():.3f}  eff.rank {er:.2f}/95  "
              f"mean R2 {np.mean(r2):.4f}  "
              f"nonfinite {s_tr['nonfinite_frac']*100:.4f}% "
              f"(nan={s_tr['nan_elements']}, inf={s_tr['inf_elements']})")
        for n, v in zip(TARGET_NAMES, r2):
            print(f"    {n:<15} R2 {v:>8.4f}")

    # --- pixel reference -------------------------------------------------
    print("\n--- pixel baseline (16x8 grey = 128 dims) ---")
    def pix(a):
        g = a.mean(axis=3)                       # (N, 80, 160)
        g = g.reshape(len(g), 8, 10, 16, 10).mean(axis=(2, 4))
        return g.reshape(len(g), -1).astype(np.float32)
    r2p, _ = ridge_probe(pix(tr['true']), tr['y'], pix(te['true']), te['y'])
    print(f"  mean R2 {np.mean(r2p):.4f}")
    for n, v in zip(TARGET_NAMES, r2p):
        print(f"    {n:<15} R2 {v:>8.4f}")

    # --- OOD: BEV-like input through the CAMERA encoder ------------------
    # Justifies rung 3 needing its own encoder rather than reusing this one.
    print("\n--- OOD probe: BEV-like occupancy through the camera VAE ---")
    rng = np.random.default_rng(0)
    bev_like = np.zeros((512, 160, 80, 3), np.float32)
    for i in range(512):
        g = np.zeros((160, 80), np.float32)
        for _ in range(rng.integers(3, 12)):
            cx, cy = rng.integers(0, 160), rng.integers(0, 80)
            g[max(0, cx-4):cx+4, max(0, cy-3):cy+3] = rng.random()
        bev_like[i] = np.repeat(g[:, :, None], 3, axis=2) * 255.0
    z_bev, _ = encode_all(encoder, bev_like.astype(np.uint8), 1.0)
    z_cam, _ = encode_all(encoder, te['carla'][:512], 1.0)
    ood = dict(
        bev_effective_rank=effective_rank(z_bev),
        cam_effective_rank=effective_rank(z_cam),
        bev_latent_std=float(z_bev.std()),
        cam_latent_std=float(z_cam.std()),
        bev_mean_pairwise_cos=float(_mean_cos(z_bev[:128])),
        cam_mean_pairwise_cos=float(_mean_cos(z_cam[:128])),
    )
    for k, v in ood.items():
        print(f"  {k:<26} {v:.4f}")

    os.makedirs('results/study', exist_ok=True)
    with open('results/study/latent_probe.json', 'w') as f:
        json.dump(dict(n_train=tr['n'], n_test=te['n'], targets=TARGET_NAMES,
                       conditions=results,
                       pixel_baseline={n: float(v) for n, v in zip(TARGET_NAMES, r2p)},
                       pixel_baseline_mean=float(np.mean(r2p)),
                       ood=ood), f, indent=2)
    print("\nwrote results/study/latent_probe.json")


def _mean_cos(z):
    zn = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)
    C = zn @ zn.T
    iu = np.triu_indices(len(z), 1)
    return C[iu].mean()


if __name__ == '__main__':
    main()
