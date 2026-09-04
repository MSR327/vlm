"""
Train the BEV-LiDAR VAE encoder required by rung 3.

    cd btp_prev
    python baseline_vae/train_bev_vae.py --data data_collected_town01 \
                                         --out  VAE/bev_encoder_model

WHY A SECOND ENCODER
--------------------
The pretrained camera VAE was trained on CityScapesPalette semantic RGB. A BEV
occupancy/height grid is badly out of distribution for it; reusing it would
produce meaningless latents and understate rung 3, making the LiDAR look
useless for the wrong reason. Rung 3 therefore gets its own encoder.

WHY IT MUST BE THE SAME ARCHITECTURE
------------------------------------
This trainer imports Encoder/Decoder/VariationalAutoencoder straight from
VAE/VAE_Trainer_dont_touch.py and uses the same LATENT_DIM (95), the same
optimiser, the same learning rate, the same loss and the same epoch count.

That is the whole point. If the BEV encoder were deeper or wider, rung 3's
gain over rung 2 would be confounded with extra encoder capacity, and the
ablation would no longer isolate the modality. Any change here must be
mirrored in the camera VAE or the ladder is invalid.

INPUT PIPELINE
--------------
collect_data.py stores lidar as (80, 160, 1) float32 in [0, 1]. The frozen VAE
architecture takes (160, 80, 3), so each grid is transposed to (160, 80) and
tiled to 3 channels -- exactly what multi_sensor.project_lidar_to_bev emits at
run time, minus the *255 that the 'raw' arm applies.

Training therefore sees [0, 1] inputs, matching how the camera VAE was trained
(VAE_Trainer_dont_touch.py:126, rescale=1/255). Both modalities then sit in the
same relation to their encoder in both norm arms.
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, 'VAE')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import tensorflow as tf

# Reuse the EXACT baseline architecture and recipe.
from VAE_Trainer_dont_touch import (
    VariationalAutoencoder, LATENT_DIM, BATCH_SIZE, LEARNING_RATE, EPOCHS, LOSS,
)


def load_bev_frames(data_dir, limit=None):
    """data_collected_town01/frames/*.npz -> (N, 160, 80, 3) float32 in [0,1]."""
    paths = sorted(glob.glob(os.path.join(data_dir, 'frames', '*.npz')))
    if not paths:
        raise FileNotFoundError(
            f"No frames in '{data_dir}/frames'. Run the collector first:\n"
            f"    python vlm/collect_data.py --town Town01 --frames 20000 "
            f"--out {data_dir}")
    if limit:
        paths = paths[:limit]

    out = np.empty((len(paths), 160, 80, 3), dtype=np.float32)
    for i, p in enumerate(paths):
        with np.load(p) as z:
            bev = z['lidar']                      # (80, 160, 1) in [0,1]
        grid = np.squeeze(bev, axis=-1)           # (80, 160)
        grid = np.transpose(grid, (1, 0))         # (160, 80)
        out[i] = np.repeat(grid[:, :, None], 3, axis=2)
        if (i + 1) % 2000 == 0:
            print(f"  loaded {i+1}/{len(paths)}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data_collected_town01')
    ap.add_argument('--out', default='VAE/bev_encoder_model')
    ap.add_argument('--epochs', type=int, default=EPOCHS)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--val-split', type=float, default=0.2)
    args = ap.parse_args()

    print(f"Loading BEV frames from {args.data} ...")
    x = load_bev_frames(args.data, args.limit)
    print(f"  {x.shape[0]} frames, shape {x.shape[1:]}, "
          f"occupancy mean {x.mean():.4f}")

    rng = np.random.default_rng(42)
    idx = rng.permutation(len(x))
    n_val = int(len(x) * args.val_split)
    val, train = x[idx[:n_val]], x[idx[n_val:]]
    print(f"  train {len(train)} / val {len(val)}")

    model = VariationalAutoencoder()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss=LOSS)

    model.fit(train, train,
              batch_size=BATCH_SIZE,
              epochs=args.epochs,
              validation_data=(val, val),
              verbose=1)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    model.encoder.save_model(args.out)

    enc = tf.keras.models.load_model(args.out)
    z = enc(val[:4].astype(np.float32))
    print(f"\nSaved BEV encoder to {args.out}")
    print(f"  latent shape {z.shape}  (expected (4, {LATENT_DIM}))")
    print(f"  latent std   {np.std(np.asarray(z)):.4f}")
    print("\nRung 3 is now unblocked:")
    print("    BTP_RUNG=3 python baseline_vae/check_rung3.py")


if __name__ == '__main__':
    main()
