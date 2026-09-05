"""
Shared checker for the VAE baseline rungs. Runs WITHOUT CARLA.

Feeds synthetic frames of the right shape through the real pretrained VAE
and asserts the observation vector comes out at exactly OBSERVATION_DIM.
The point is to catch width/shape/loader errors on a dev machine rather
than after copying everything to the CARLA box.

Run via check_rung1.py / check_rung2.py / check_rung3.py.
"""
import os
import sys
import time
import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


def run_check(params):
    name = f"RUNG {params.RUNG} ({params.RUNG_NAME})"
    print("=" * 62)
    print(f"  {name}")
    print("=" * 62)
    print(f"  cameras        : {params.NUM_CAMERAS_VAE}  yaws={params.CAMERA_YAWS}")
    print(f"  lidar          : {params.USE_LIDAR_VAE}")
    print(f"  latent dim     : {params.LATENT_DIM}")
    print(f"  nav dim        : {params.NAV_DIM}")
    print(f"  OBSERVATION_DIM: {params.OBSERVATION_DIM}")
    print(f"  results dir    : {params.RESULTS_PATH}")
    print("-" * 62)

    if params.USE_LIDAR_VAE and not os.path.isdir(params.BEV_VAE_PATH):
        print(f"  SKIPPED: BEV encoder not trained yet.")
        print(f"           expected at '{params.BEV_VAE_PATH}'")
        print(f"           train it before running rung 3 (see params_rung3 docstring)")
        return None

    from encode_state_vae import EncodeStateVAE

    try:
        enc = EncodeStateVAE(params)
    except Exception as e:
        print(f"  FAIL: could not build encoder\n        {type(e).__name__}: {e}")
        return False

    # Synthetic frames in the same layout CARLA delivers:
    # (W, H, 3) uint8-valued floats, 0-255, NOT normalised.
    rng = np.random.default_rng(0)
    imgs = [rng.integers(0, 256, (params.IM_WIDTH, params.IM_HEIGHT, 3)).astype(np.float32)
            for _ in range(params.NUM_CAMERAS_VAE)]
    images = imgs[0] if params.NUM_CAMERAS_VAE == 1 else imgs
    nav = np.array([0.5, 8.3, 0.42, 0.11, 0.03], dtype=np.float32)

    if params.USE_LIDAR_VAE:
        bev = rng.integers(0, 256, (params.BEV_GRID_W, params.BEV_GRID_H, 3)).astype(np.float32)
        obs_in = [images, bev, nav]
    else:
        obs_in = [images, nav]

    try:
        t0 = time.perf_counter()
        out = enc.process(obs_in)
        dt = (time.perf_counter() - t0) * 1000.0
    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {e}")
        return False

    ok = (out.shape == (params.OBSERVATION_DIM,))
    print(f"  output shape   : {out.shape}   expected ({params.OBSERVATION_DIM},)")
    print(f"  dtype          : {out.dtype}")
    print(f"  finite         : {bool(np.isfinite(out).all())}")
    print(f"  nav tail intact: {np.allclose(out[-params.NAV_DIM:], nav)}")
    print(f"  first pass     : {dt:.1f} ms (includes graph warm-up)")

    # Steady-state timing, warm.
    t0 = time.perf_counter()
    for _ in range(10):
        enc.process(obs_in)
    print(f"  warm latency   : {(time.perf_counter()-t0)*100:.1f} ms/call")

    print(f"  RESULT         : {'PASS' if ok else 'FAIL'}")
    return ok
