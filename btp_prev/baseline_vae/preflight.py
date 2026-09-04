"""
Run this FIRST on the CARLA machine, before any training.

Every check below has cost someone hours at some point. It takes ~30 seconds
and tells you exactly which of them would have bitten.

    cd btp_prev
    python baseline_vae/preflight.py                 # rung 1, checks everything
    BTP_RUNG=3 python baseline_vae/preflight.py      # also checks the BEV encoder
"""
import glob
import os
import sys
import traceback

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

PASS, FAIL, WARN = 'PASS', 'FAIL', 'WARN'
results = []


def check(name, fn, fatal=True):
    try:
        status, detail = fn()
    except Exception as e:
        status, detail = FAIL, f"{type(e).__name__}: {e}"
        if os.environ.get('BTP_VERBOSE'):
            traceback.print_exc()
    results.append((name, status, detail, fatal))
    print(f"  [{status}] {name}: {detail}")
    return status


# ---------------------------------------------------------------- cwd
def c_cwd():
    if not os.path.isdir('VAE/var_auto_encoder_model'):
        return FAIL, ("VAE/ not found. You must run from the btp_prev root: "
                      "`cd btp_prev` then `python baseline_vae/preflight.py`")
    return PASS, f"running from {os.getcwd()}"


# ---------------------------------------------------------------- python deps
def c_tf():
    import tensorflow as tf
    major, minor = (int(x) for x in tf.__version__.split('.')[:2])
    gpu = tf.config.list_physical_devices('GPU')
    dev = f"GPU x{len(gpu)}" if gpu else "CPU only"
    if (major, minor) >= (2, 16):
        return WARN, (f"TF {tf.__version__} ({dev}) is Keras 3. "
                      f"PPOAgent.call is patched for it, but agent.save()/load() "
                      f"in main.py use the legacy SavedModel format that Keras 3 "
                      f"rejects. Use TF 2.15 or lower. See RUNBOOK step 4.")
    return PASS, f"TF {tf.__version__} ({dev}) -- Keras 2, fully supported"


def c_tfp():
    import tensorflow_probability as tfp
    _ = tfp.distributions.MultivariateNormalDiag
    return PASS, f"tensorflow_probability {tfp.__version__}"


def c_deps():
    missing = []
    for m in ('numpy', 'pandas', 'cv2', 'pygame', 'scipy'):
        try:
            __import__(m)
        except ImportError:
            missing.append(m)
    if missing:
        return FAIL, f"missing: {', '.join(missing)}  (pip install -r requirements.txt)"
    return PASS, "numpy, pandas, cv2, pygame, scipy present"


# ---------------------------------------------------------------- carla
def c_carla_egg():
    try:
        import carla
    except ImportError:
        eggs = glob.glob('carla/carla-*.egg')
        return FAIL, (f"`import carla` failed. Eggs found in ./carla: {eggs or 'none'}. "
                      f"Either `pip install carla==<server version>` or drop the "
                      f"matching .egg into ./carla/.")
    return PASS, f"carla module at {carla.__file__}"


def c_carla_server():
    import carla
    from parameters import CARLA_HOST, CARLA_PORT, CARLA_TIMEOUT
    client = carla.Client(CARLA_HOST, CARLA_PORT)
    client.set_timeout(CARLA_TIMEOUT)
    cv, sv = client.get_client_version(), client.get_server_version()
    if cv != sv:
        return FAIL, (f"VERSION MISMATCH -- client {cv}, server {sv}. "
                      f"The Python API must match CarlaUE4.exe exactly.")
    return PASS, f"client {cv} == server {sv} at {CARLA_HOST}:{CARLA_PORT}"


def c_towns():
    import carla
    from parameters import CARLA_HOST, CARLA_PORT, TRAIN_TOWN, TEST_TOWN, CARLA_TIMEOUT
    client = carla.Client(CARLA_HOST, CARLA_PORT)
    client.set_timeout(CARLA_TIMEOUT)
    avail = [t.split('/')[-1] for t in client.get_available_maps()]
    missing = [t for t in (TRAIN_TOWN, TEST_TOWN) if t not in avail]
    if missing:
        return FAIL, f"missing maps {missing}; server has {sorted(set(avail))}"
    return PASS, f"{TRAIN_TOWN} and {TEST_TOWN} available"


def c_sensors():
    import carla
    from parameters import CARLA_HOST, CARLA_PORT, CAMERA_SENSOR_NAME, CARLA_TIMEOUT
    client = carla.Client(CARLA_HOST, CARLA_PORT)
    client.set_timeout(CARLA_TIMEOUT)
    bl = client.get_world().get_blueprint_library()
    need = [CAMERA_SENSOR_NAME, 'sensor.lidar.ray_cast', 'sensor.other.collision']
    missing = [n for n in need if not len(bl.filter(n))]
    if missing:
        return FAIL, f"blueprints missing: {missing}"
    return PASS, f"all sensor blueprints present ({len(need)})"


# ---------------------------------------------------------------- our config
def c_config():
    import parameters as P
    return PASS, (f"rung {P.RUNG} ({P.RUNG_NAME}) arm={P.NORM_ARM} "
                  f"obs_dim={P.OBSERVATION_DIM} cams={P.NUM_CAMERAS_VAE} "
                  f"lidar={P.USE_LIDAR_VAE} sync={P.SYNCHRONOUS_MODE} "
                  f"-> {P.RESULTS_PATH}")


def c_encoder():
    import numpy as np
    import parameters as P
    from encode_state_vae import EncodeStateVAE
    enc = EncodeStateVAE(P)
    rng = np.random.default_rng(0)
    imgs = [rng.integers(0, 256, (P.IM_WIDTH, P.IM_HEIGHT, 3)).astype(np.float32)
            for _ in range(P.NUM_CAMERAS_VAE)]
    images = imgs[0] if P.NUM_CAMERAS_VAE == 1 else imgs
    nav = np.zeros(P.NAV_DIM, np.float32)
    obs_in = [images, nav]
    if P.USE_LIDAR_VAE:
        bev = rng.integers(0, 256, (P.BEV_GRID_W, P.BEV_GRID_H, 3)).astype(np.float32)
        obs_in = [images, bev, nav]
    out = enc.process(obs_in)
    if out.shape != (P.OBSERVATION_DIM,):
        return FAIL, f"got {out.shape}, expected ({P.OBSERVATION_DIM},)"
    return PASS, f"observation {out.shape}, {enc.last_encode_ms:.1f} ms"


def c_bev_encoder():
    import parameters as P
    if not P.USE_LIDAR_VAE:
        return PASS, "not needed for this rung"
    if not os.path.isdir(P.BEV_VAE_PATH):
        return FAIL, (f"rung 3 needs a BEV encoder at '{P.BEV_VAE_PATH}'. "
                      f"Collect data then run baseline_vae/train_bev_vae.py "
                      f"(RUNBOOK step 7).")
    return PASS, f"BEV encoder present at {P.BEV_VAE_PATH}"


def c_agent():
    import numpy as np
    import parameters as P
    from main import PPOAgent
    agent = PPOAgent()
    obs = np.zeros(P.OBSERVATION_DIM, np.float32)
    a, m = agent(obs, False)                 # legacy positional call
    if a.shape != (P.ACTION_DIM,):
        return FAIL, f"action shape {a.shape}"
    if agent.obs_dim != P.OBSERVATION_DIM:
        return FAIL, f"actor built at {agent.obs_dim}, config says {P.OBSERVATION_DIM}"
    return PASS, f"PPOAgent obs_dim={agent.obs_dim}, action {a.shape}"


def c_disk():
    import shutil
    free_gb = shutil.disk_usage('.').free / 1024 ** 3
    if free_gb < 5:
        return FAIL, f"only {free_gb:.1f} GB free"
    return PASS, f"{free_gb:.1f} GB free"


# ----------------------------------------------------------------
def main():
    print("=" * 72)
    print("  PREFLIGHT -- VAE baseline ladder")
    print("=" * 72)

    print("\n[environment]")
    if check("working directory", c_cwd) == FAIL:
        sys.exit(1)
    check("tensorflow", c_tf, fatal=False)
    check("tensorflow_probability", c_tfp)
    check("python deps", c_deps)
    check("disk space", c_disk, fatal=False)

    print("\n[carla]")
    egg_ok = check("carla python api", c_carla_egg)
    if egg_ok == PASS:
        srv = check("carla server + version match", c_carla_server)
        if srv == PASS:
            check("required towns", c_towns)
            check("sensor blueprints", c_sensors)

    print("\n[ladder]")
    check("config", c_config)
    check("bev encoder", c_bev_encoder)
    check("vae encoder forward pass", c_encoder)
    check("ppo agent", c_agent)

    print("\n" + "=" * 72)
    fails = [r for r in results if r[1] == FAIL and r[3]]
    warns = [r for r in results if r[1] == WARN]
    for n, _, d, _ in warns:
        print(f"  WARN  {n}: {d}")
    if fails:
        print(f"  {len(fails)} BLOCKING FAILURE(S) -- do not start the run:")
        for n, _, d, _ in fails:
            print(f"    - {n}: {d}")
        sys.exit(1)
    print("  ALL CHECKS PASSED -- safe to start the run.")
    print("=" * 72)


if __name__ == '__main__':
    main()
