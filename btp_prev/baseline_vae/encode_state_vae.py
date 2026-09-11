"""
Multi-camera / LiDAR drop-in replacement for the legacy EncodeState
(main.py:622).

The original encodes ONE semantic-segmentation frame:

    image_obs -> VAE -> (95,)  ;  concat nav (5,)  ->  (100,)

This version encodes N camera frames with the SAME frozen VAE and
concatenates the latents, optionally appending a BEV-LiDAR latent from a
second encoder:

    rung 1 : 95*1 + 5 = 100
    rung 2 : 95*4 + 5 = 385
    rung 3 : 95*5 + 5 = 480

FIDELITY NOTES -- deliberately matching the original, do not "improve":
  * Normalisation is an explicit experiment ARM, not a fix. params.VAE_INPUT_SCALE
    is 1.0 in the 'raw' arm (what main.py:638 actually does) and 1/255 in the
    'scaled' arm (what VAE_Trainer_dont_touch.py:126 trained on). Both are run.
  * Camera frames arrive as (W, H, C) = (160, 80, 3), which is what the
    saved signature expects -- see main.py:549 reshape((width, height, 4)).

MEASURED CORRECTION to an earlier comment here: the saved encoder is
effectively DETERMINISTIC at inference. Over repeated calls on identical
input, mean |z1 - z2| = 5.7e-5 against a latent std of 8.7 -- the sampling
noise is ~1e-5 relative. Treat it as deterministic; no seed control needed
for the latents, and repeated-call variance is not a confound in the ladder.
"""
import time
import numpy as np


class EncodeStateVAE:
    """Frozen-VAE observation encoder for rungs 1-3.

    Exposes `.last_encode_ms` and `.timings` so encoder latency can be reported
    separately from whole-step latency -- the ladder's cost axis is about the
    ENCODER, and whole-step latency is dominated by the CARLA server tick.
    """

    def __init__(self, params):
        """params: one of the params_rung{1,2,3} modules."""
        import os
        from vae_loader import load_vae_encoder

        self.p = params
        self.num_cameras = params.NUM_CAMERAS_VAE
        self.use_lidar   = params.USE_LIDAR_VAE
        self.latent_dim  = params.LATENT_DIM
        self.scale       = params.VAE_INPUT_SCALE

        cam_path = os.path.join(params.VAR_AUTO_MODEL_PATH, 'var_auto_encoder_model')
        self.cam_encoder = load_vae_encoder(cam_path)

        self.bev_encoder = None
        if self.use_lidar:
            self.bev_encoder = load_vae_encoder(params.BEV_VAE_PATH)

        self.expected_dim = params.OBSERVATION_DIM
        self.timings = []
        self.last_encode_ms = 0.0

        print(f"[EncodeStateVAE] rung {params.RUNG} "
              f"({self.num_cameras} cam, lidar={self.use_lidar}, "
              f"arm={params.NORM_ARM}, scale={self.scale:g}) "
              f"-> observation dim {self.expected_dim}")

    # ------------------------------------------------------------------
    def process(self, observation):
        """
        observation:
            [images, nav]            where images is either a single
                                     (W,H,3) array (rung 1) or a list/array
                                     of N such arrays,
            [images, bev, nav]       when use_lidar is True.

        returns: 1-D float32 numpy array of length OBSERVATION_DIM
        """
        t0 = time.perf_counter()

        if self.use_lidar:
            images, bev, nav = observation[0], observation[1], observation[2]
        else:
            images, nav = observation[0], observation[1]
            bev = None

        batch = self._as_batch(images)
        if batch.shape[0] != self.num_cameras:
            raise ValueError(
                f"expected {self.num_cameras} camera frame(s), got {batch.shape[0]}"
            )

        # One batched forward pass rather than N calls: cheaper, and latency
        # is one of the reported metrics. Safe because BatchNorm runs with
        # frozen moving statistics in inference mode.
        cam_latents = np.asarray(self.cam_encoder(batch))
        if getattr(self.p, 'POOL_CAMERAS', False) and cam_latents.ndim == 2 and cam_latents.shape[0] > 1:
            # Channel-wise max-pooling across the surround views -> (95,)
            latents = np.max(cam_latents, axis=0)
        else:
            latents = cam_latents.reshape(-1)

        # Protect against unscaled latent explosion in raw arm (std ~2984, max ~14000)
        # to prevent complete tanh saturation (499/500 dead neurons) and telemetry drowning
        lat_std = float(np.std(latents))
        if lat_std > 10.0:
            latents = (latents - np.mean(latents)) / (lat_std + 1e-7)

        parts = [latents]

        if self.use_lidar:
            bev_batch = self._as_batch(bev)
            bev_lat = np.asarray(self.bev_encoder(bev_batch)).reshape(-1)
            bev_std = float(np.std(bev_lat))
            if bev_std > 10.0:
                bev_lat = (bev_lat - np.mean(bev_lat)) / (bev_std + 1e-7)
            parts.append(bev_lat)

        parts.append(np.asarray(nav, dtype=np.float32).reshape(-1))
        out = np.concatenate(parts).astype(np.float32)

        if out.shape[0] != self.expected_dim:
            raise ValueError(
                f"observation width {out.shape[0]} != OBSERVATION_DIM "
                f"{self.expected_dim}. The PPO network would be built at the "
                f"wrong input size."
            )

        self.last_encode_ms = (time.perf_counter() - t0) * 1000.0
        self.timings.append(self.last_encode_ms)
        return out

    # ------------------------------------------------------------------
    def _as_batch(self, imgs):
        """Accept (W,H,3) or [(W,H,3), ...] -> (N, W, H, 3) float32, scaled."""
        arr = np.asarray(imgs, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[None, ...]
        if self.scale != 1.0:
            arr = arr * self.scale
        return arr

    # ------------------------------------------------------------------
    def latency_summary(self):
        """mean / p50 / p95 encoder latency in ms over the run so far."""
        if not self.timings:
            return dict(n=0, mean_ms=0.0, p50_ms=0.0, p95_ms=0.0)
        t = np.asarray(self.timings)
        return dict(n=int(t.size), mean_ms=float(t.mean()),
                    p50_ms=float(np.percentile(t, 50)),
                    p95_ms=float(np.percentile(t, 95)))
