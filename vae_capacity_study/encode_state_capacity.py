"""
Observation state encoder for VAE Capacity Scaling Study in CARLA.

Encodes camera frames through the specified capacity VAE encoder,
extracting deterministic latent vector mu, and concatenating vehicle telemetry:
    s_t = [ z_cam in R^{dz} || x_nav in R^5 ] in R^{dz + 5}
"""
import os
import time
import numpy as np

from vae_loader import load_vae_encoder


class EncodeStateCapacity:
    def __init__(self, params):
        self.p = params
        self.latent_dim = params.LATENT_DIM
        self.scale = getattr(params, 'VAE_INPUT_SCALE', 1.0 / 255.0)
        self.model_path = params.VAR_AUTO_MODEL_PATH

        print(f"[EncodeStateCapacity] Loading VAE encoder from: {self.model_path}")
        self.encoder = load_vae_encoder(self.model_path)
        self.expected_dim = params.OBSERVATION_DIM
        self.timings = []
        self.last_encode_ms = 0.0

        print(f"[EncodeStateCapacity] Arch: {params.VAE_ARCH}, Latent Dim: {self.latent_dim}, "
              f"Obs Dim: {self.expected_dim}")

    def _prepare_image(self, img):
        if not isinstance(img, np.ndarray):
            img = np.asarray(img)
        # Ensure float32 in [0, 1]
        if img.dtype == np.uint8:
            arr = img.astype(np.float32) * self.scale
        else:
            arr = img.astype(np.float32)
            if self.scale != 1.0 and arr.max() > 1.5:
                arr = arr * self.scale

        # Ensure shape (1, 160, 80, 3)
        if arr.ndim == 3:
            arr = np.expand_dims(arr, axis=0)
        return arr

    def process(self, observation):
        """
        observation: [image_obs, nav_telemetry]
        returns: 1D float32 array of shape (OBSERVATION_DIM,)
        """
        t0 = time.perf_counter()
        image_obs, nav = observation[0], observation[1]

        batch = self._prepare_image(image_obs)
        latent = np.asarray(self.encoder(batch))

        # Flatten if needed
        if latent.ndim > 1:
            latent = latent.reshape(-1)

        # Telemetry: [speed, dist_center, heading, steer_prev, throttle_prev]
        nav_vec = np.asarray(nav, dtype=np.float32).reshape(-1)
        if nav_vec.shape[0] != self.p.NAV_DIM:
            # Pad or slice to NAV_DIM
            if nav_vec.shape[0] < self.p.NAV_DIM:
                nav_vec = np.pad(nav_vec, (0, self.p.NAV_DIM - nav_vec.shape[0]))
            else:
                nav_vec = nav_vec[:self.p.NAV_DIM]

        obs = np.concatenate([latent, nav_vec], axis=0).astype(np.float32)

        # Sanitize against any stray non-finite values
        if not np.all(np.isfinite(obs)):
            obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)

        dt_ms = (time.perf_counter() - t0) * 1000.0
        self.last_encode_ms = dt_ms
        self.timings.append(dt_ms)

        return obs
