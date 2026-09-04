"""
Does concatenating latents from a SHARED frozen encoder actually fuse
information across views?

That is exactly the mechanism rung 2 relies on: four views, one frozen VAE,
`np.concatenate` the four 95-d latents. There is no multi-view data without
CARLA, so this tests the mechanism on real data with a controlled proxy:

    view A = left  half of the frame, resized back to the encoder's input
    view B = right half of the frame, resized back to the encoder's input

Each half is a genuine, partial, non-overlapping view of the same scene, run
through the same frozen encoder that rung 2 uses on all four cameras. Targets
are computed on the FULL frame, so neither half alone can be sufficient.

Then compare linear-probe R^2 from
    z(A)            one view                       95 dims,  1 encoder pass
    z(B)            the other view                 95 dims,  1 encoder pass
    [z(A), z(B)]    concatenated, as EncodeStateVAE does
                                                  190 dims,  2 encoder passes
    z(full)         whole frame, one pass          95 dims,  1 encoder pass

NOTE on the last row: it is an equal-LATENT-BUDGET reference, not an upper
bound. Two halves each get the encoder's full 160x80 input grid, so the concat
variant spends twice the encoder compute and sees twice the pixels. Concat
scoring above it is expected and is not evidence of anything surprising.

The result to read is the RANK SUB-ADDITIVITY: if two views were independent,
the concatenated effective rank would be rank(A) + rank(B). How far short it
falls is how redundant the shared frozen encoder's codes are -- and that is
precisely what determines whether rung 2's extra 285 dimensions buy
information or just cost the policy capacity.

This is a proxy for viewpoint diversity, not a substitute for the real
4-camera run. It bounds the mechanism, not the rung.
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
from latent_probe import ridge_probe, effective_rank, encode_all, carla_legacy_layout

N_TRAIN, N_TEST = 12000, 2000
SCALE = 1 / 255.0          # 'scaled' arm: the encoder's best case, so a weak
                           # result here cannot be blamed on the norm defect.


def halves(pil_img):
    """(left, right) halves, each resized to the encoder's (160, 80) input."""
    w, h = pil_img.size                       # 160 x 80
    left = pil_img.crop((0, 0, w // 2, h)).resize((80, 160), Image.BILINEAR)
    right = pil_img.crop((w // 2, 0, w, h)).resize((80, 160), Image.BILINEAR)
    return np.asarray(left, np.uint8), np.asarray(right, np.uint8)


def load(split, limit):
    paths = sorted(glob.glob(f'VAE/dataset/{split}/class1/*.png'))[:limit]
    n = len(paths)
    A = np.empty((n, 160, 80, 3), np.uint8)
    B = np.empty((n, 160, 80, 3), np.uint8)
    F = np.empty((n, 160, 80, 3), np.uint8)
    Y = np.empty((n, len(TARGET_NAMES)), np.float32)
    for i, p in enumerate(paths):
        im = Image.open(p).convert('RGB')
        A[i], B[i] = halves(im)
        F[i] = np.asarray(im.resize((80, 160), Image.BILINEAR), np.uint8)
        Y[i] = extract_targets(np.asarray(im))
    return A, B, F, Y


def main():
    import tensorflow as tf
    sm = tf.saved_model.load('VAE/var_auto_encoder_model')
    fn = sm.signatures['serving_default']
    enc = lambda x: fn(input_1=tf.convert_to_tensor(x, tf.float32))['output_1']

    print(f"Loading {N_TRAIN} train / {N_TEST} test ...")
    Atr, Btr, Ftr, Ytr = load('train', N_TRAIN)
    Ate, Bte, Fte, Yte = load('test', N_TEST)

    za_tr, _ = encode_all(enc, Atr, SCALE); za_te, _ = encode_all(enc, Ate, SCALE)
    zb_tr, _ = encode_all(enc, Btr, SCALE); zb_te, _ = encode_all(enc, Bte, SCALE)
    zf_tr, _ = encode_all(enc, Ftr, SCALE); zf_te, _ = encode_all(enc, Fte, SCALE)

    cat_tr = np.hstack([za_tr, zb_tr])
    cat_te = np.hstack([za_te, zb_te])

    variants = [
        ('view A (left half)',   za_tr, za_te),
        ('view B (right half)',  zb_tr, zb_te),
        ('concat [A, B]',        cat_tr, cat_te),
        ('full frame (1 pass)',  zf_tr, zf_te),
    ]

    results = {}
    print(f"\n{'variant':<24}{'dims':>6}{'eff.rank':>10}{'mean R2':>10}   per-target")
    print('-' * 100)
    for name, ztr, zte in variants:
        r2, _ = ridge_probe(ztr, Ytr, zte, Yte)
        er = effective_rank(ztr)
        results[name] = dict(dims=int(ztr.shape[1]), effective_rank=er,
                             r2_mean=float(np.mean(r2)),
                             r2={n: float(v) for n, v in zip(TARGET_NAMES, r2)})
        per = '  '.join(f"{n[:9]}={v:.3f}" for n, v in zip(TARGET_NAMES, r2))
        print(f"{name:<24}{ztr.shape[1]:>6}{er:>10.2f}{np.mean(r2):>10.4f}   {per}")

    ra = results['view A (left half)']
    rb = results['view B (right half)']
    rc = results['concat [A, B]']
    rf = results['full frame (1 pass)']

    best_single = max(ra['r2_mean'], rb['r2_mean'])
    fusion_gain = rc['r2_mean'] - best_single

    rank_if_independent = ra['effective_rank'] + rb['effective_rank']
    rank_additivity = rc['effective_rank'] / rank_if_independent

    print(f"\nbest single view       {best_single:.4f}")
    print(f"concat [A, B]          {rc['r2_mean']:.4f}   "
          f"(+{fusion_gain:.4f} over best single view)")
    print(f"full frame, 1 pass     {rf['r2_mean']:.4f}   "
          f"(equal latent budget, half the encoder compute)")
    print(f"\nRANK SUB-ADDITIVITY")
    print(f"  rank(A) {ra['effective_rank']:.2f} + rank(B) {rb['effective_rank']:.2f}"
          f" = {rank_if_independent:.2f} if independent")
    print(f"  rank([A,B]) actually  {rc['effective_rank']:.2f}"
          f"   -> {rank_additivity*100:.0f}% additive")
    print(f"  {rc['dims']} concatenated dims carry"
          f" {rc['effective_rank']:.2f} effective dimensions"
          f" ({rc['effective_rank']/rc['dims']*100:.1f}% utilisation)")

    os.makedirs('results/study', exist_ok=True)
    json.dump(dict(n_train=N_TRAIN, n_test=N_TEST, scale=SCALE,
                   variants=results,
                   best_single_r2=best_single, concat_r2=rc['r2_mean'],
                   full_frame_r2=rf['r2_mean'], fusion_gain=fusion_gain,
                   rank_if_independent=rank_if_independent,
                   rank_additivity=rank_additivity,
                   concat_dim_utilisation=rc['effective_rank'] / rc['dims']),
              open('results/study/multiview_proxy.json', 'w'), indent=2)
    print("\nwrote results/study/multiview_proxy.json")


if __name__ == '__main__':
    main()
