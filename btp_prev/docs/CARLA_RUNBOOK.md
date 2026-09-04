# CARLA closed-loop runbook — Windows machine

Step-by-step for running the three-rung ablation. Every command here
corresponds to code that exists in this repo; nothing is hypothetical.

Throughout: **`<REPO>` is the folder containing `baseline_vae/`, `VAE/`,
`vlm/`, `study/`.** Every command runs from `<REPO>`, never from inside a
subfolder — all path literals in the code are relative to it.

---

## 0. TL;DR — what you will actually type

```bat
cd C:\btp_prev
conda activate carla-ladder

REM one-time, per rung:
set BTP_NORM=raw

set BTP_RUNG=1
python baseline_vae/preflight.py
python baseline_vae/run_ladder.py --mode train --timesteps 300000
python baseline_vae/run_ladder.py --mode test  --town Town02 --episodes 20

set BTP_RUNG=2
python baseline_vae/preflight.py
python baseline_vae/run_ladder.py --mode train --timesteps 300000
python baseline_vae/run_ladder.py --mode test  --town Town02 --episodes 20

REM rung 3 needs the BEV encoder first — see step 7
set BTP_RUNG=3
python baseline_vae/preflight.py
python baseline_vae/run_ladder.py --mode train --timesteps 300000
python baseline_vae/run_ladder.py --mode test  --town Town02 --episodes 20
```

The rest of this document explains each of those and what to do when one fails.

---

## 1. Start CARLA

Open a terminal in your CARLA install folder and run the server **windowed and
at low quality** — you do not need pretty pixels, you need throughput, and the
observation cameras are 160×80 regardless.

```bat
cd C:\CARLA_0.9.15\WindowsNoEditor
CarlaUE4.exe -quality-level=Low -windowed -ResX=800 -ResY=600 -carla-rpc-port=2000
```

For maximum speed, run headless (no render window at all):

```bat
CarlaUE4.exe -RenderOffScreen -quality-level=Low -carla-rpc-port=2000
```

Leave this running in its own terminal for the whole session. Every command in
the rest of this runbook goes in a **second** terminal.

Notes:
- The port `2000` must match `CARLA_PORT` (default 2000). Override with
  `set CARLA_PORT=2000`.
- The scripts put the server into **synchronous mode** and set
  `fixed_delta_seconds = 0.05`. If a run crashes hard, the server can be left
  in synchronous mode and will appear frozen. Restart `CarlaUE4.exe` — or run
  `python -c "import carla;w=carla.Client('localhost',2000).get_world();s=w.get_settings();s.synchronous_mode=False;s.fixed_delta_seconds=None;w.apply_settings(s)"`.
  `run_ladder.py` calls `env.restore_settings()` on a clean exit.

---

## 2. Verify the Python API matches the server

This is the single most common failure. The `carla` Python package version must
equal the `CarlaUE4.exe` version **exactly**.

Find the server version:

```bat
python -c "import carla; c=carla.Client('localhost',2000); c.set_timeout(20.0); print('client', c.get_client_version()); print('server', c.get_server_version())"
```

Both lines must print the same string, e.g. `0.9.15`. If `import carla` fails
or the versions differ, install the matching API:

```bat
REM Option A - pip (works for the versions that publish wheels)
pip install carla==0.9.15

REM Option B - the .egg shipped with your server build
REM   C:\CARLA_0.9.15\WindowsNoEditor\PythonAPI\carla\dist\carla-0.9.15-py3.8-win-amd64.egg
REM Copy it into <REPO>\carla\  -- main.py globs ./carla/carla-*.egg on startup.
mkdir C:\btp_prev\carla
copy C:\CARLA_0.9.15\WindowsNoEditor\PythonAPI\carla\dist\carla-*-win-amd64.egg C:\btp_prev\carla\
```

**The egg's Python version pins your interpreter.** An egg named `py3.8` only
works on Python 3.8. Pick your conda Python to match the egg you have — see
step 4.

`python baseline_vae/preflight.py` checks all of this for you and refuses to
start if the versions disagree.

---

## 3. Files to copy to the Windows machine

Copy the whole repo **except** the two big items you do not need for closed-loop
runs:

| Path | Copy? | Why |
|---|---|---|
| `baseline_vae/` | **yes** | the ladder — all runner code |
| `VAE/var_auto_encoder_model/` | **yes** | the frozen encoder. Without it nothing runs |
| `VAE/VAE_Trainer_dont_touch.py` | **yes** | `train_bev_vae.py` imports the architecture from it |
| `vlm/collect_data.py`, `vlm/parameters.py` | **yes** | needed only for rung 3's BEV data |
| `study/` | yes (small) | lets you run `aggregate.py` on the box |
| `requirements.txt`, `README_LAYOUT.md`, `docs/` | yes | reference |
| `VAE/dataset/`, `VAE/dataset.zip` | **no** | 14k PNGs, ~42 MB, only for retraining the camera VAE or the offline probes |
| `results/Results_04`, `results/Results_05` | **no** | prior outputs, not inputs |
| `VAE/var_encoder_model.pth` | **no** | 40 MB PyTorch file, unused by the ladder |

Minimum viable set is roughly 60 MB. Verify after copying:

```bat
cd C:\btp_prev
dir VAE\var_auto_encoder_model\saved_model.pb
```

If that file is missing, everything else is pointless.

---

## 4. Python environment

**Critical constraint: use TensorFlow ≤ 2.15.**

`main.py`'s `agent.save()` / `agent.load()` use the legacy SavedModel format
(`model.save('<dir>')` + `tf.keras.models.load_model('<dir>')`). Keras 3
(TF ≥ 2.16) removed that format, and its single shared `Adam` optimizer in
`learn()` also fails under Keras 3. I patched the one issue you asked about
(the positional `train` argument), but the save/load and optimizer issues are
in the untouched training logic. **TF 2.15 or lower avoids all three.**

On native Windows there is a second constraint: **TF > 2.10 has no GPU support
on native Windows** (it requires WSL2). If you want GPU, use TF 2.10.

Recommended, GPU on native Windows:

```bat
conda create -n carla-ladder python=3.8 -y
conda activate carla-ladder

conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0 -y
pip install "tensorflow==2.10.1" "tensorflow-probability==0.18.0"
pip install "protobuf<3.21" numpy pandas opencv-python pygame scipy tqdm tensorboard
pip install carla==0.9.15
```

If you do not need GPU, or your egg pins a different Python:

```bat
conda create -n carla-ladder python=3.8 -y
conda activate carla-ladder
pip install "tensorflow==2.15.1" "tensorflow-probability==0.23.0"
pip install numpy pandas opencv-python pygame scipy tqdm tensorboard
```

Verify:

```bat
python -c "import tensorflow as tf; print(tf.__version__, tf.config.list_physical_devices('GPU'))"
python -c "import tensorflow_probability as tfp; print(tfp.__version__)"
```

`tensorflow-probability` must import cleanly — `main.py` uses
`tfd.MultivariateNormalDiag` for the policy distribution. The
version pairing matters: tfp 0.18 ↔ TF 2.10, tfp 0.23 ↔ TF 2.15.

### Preflight

Before every rung:

```bat
python baseline_vae/preflight.py
```

It checks: working directory, TF version and Keras generation, tfp, deps, disk,
the `carla` import, **client/server version match**, Town01+Town02 present,
sensor blueprints present, the rung config, the BEV encoder (rung 3), a real VAE
forward pass, and a real `PPOAgent` forward pass at the right width. It exits
non-zero on anything blocking. Run it — it takes 30 seconds and saves hours.

---

## 5–7. The three runs

All three use **one script**, `baseline_vae/run_ladder.py`. The rung is selected
by the `BTP_RUNG` environment variable, and nothing else differs. That is
deliberate: PPO hyperparameters, reward, route, towns, seeds, weather, traffic,
termination thresholds and episode budget all come from
`baseline_vae/params_vae_base.py`, which all three rungs import.

Set the normalisation arm once for the whole session:

```bat
set BTP_NORM=raw
```

(`raw` = as-deployed, comparable to `Results_05`. Run the whole ladder in one
arm before starting the other.)

### 5. Rung 1 — VAE + 1 RGB camera (obs 100)

```bat
set BTP_RUNG=1
python baseline_vae/preflight.py
python baseline_vae/run_ladder.py --mode train --timesteps 300000
python baseline_vae/run_ladder.py --mode test --town Town02 --episodes 20
```

### 6. Rung 2 — VAE + 4 RGB cameras (obs 385)

```bat
set BTP_RUNG=2
python baseline_vae/preflight.py
python baseline_vae/run_ladder.py --mode train --timesteps 300000
python baseline_vae/run_ladder.py --mode test --town Town02 --episodes 20
```

Cameras are at yaw 0 / −60 / +60 / 180, placements in
`params_vae_base.CAMERA_RIG`. The yaw-0 camera is byte-identical to rung 1's.

### 7. Rung 3 — VAE + 4 RGB + BEV LiDAR (obs 480)

**Rung 3 has a prerequisite.** The BEV grid is far out of distribution for the
camera VAE, so rung 3 uses a second encoder of *identical architecture* trained
on BEV frames. It does not exist yet. Two extra steps, once:

```bat
REM 7a. collect synchronised 4-camera + LiDAR frames (~30-60 min for 20k)
python vlm/collect_data.py --town Town01 --frames 20000 --out data_collected_town01

REM 7b. train the BEV encoder (same recipe as the camera VAE: 95-d, 10 epochs)
python baseline_vae/train_bev_vae.py --data data_collected_town01 --out VAE/bev_encoder_model
```

Then:

```bat
set BTP_RUNG=3
python baseline_vae/preflight.py
python baseline_vae/run_ladder.py --mode train --timesteps 300000
python baseline_vae/run_ladder.py --mode test --town Town02 --episodes 20
```

`preflight.py` fails with a clear message if `VAE/bev_encoder_model` is absent,
rather than crashing mid-run.

---

## 8. How much training

**`--timesteps 300000` per rung.** The default in `params_vae_base.py` is
`TRAIN_TIMESTEPS = 1e6`; at roughly 20–50 env steps/second that is 6–14 hours
per rung, so 3 rungs would be a two-day run. 300k lands around 2–4 hours each,
about 8–12 hours for the ladder — one overnight run.

**The number matters far less than using the same number for all three.** The
comparison is only valid if rungs 1, 2 and 3 get an identical budget. Do not
give rung 3 more because it "seems to need it".

PPO updates every 5 episodes and checkpoints every 50 (`run_ladder.train`).
Watch the learning curve:

```bat
tensorboard --logdir Results_VAE_1cam_raw/runs/train
```

If a rung's reward is still climbing steeply at 300k, say so when you send the
results — I will read the per-episode CSV and tell you whether the ranking is
safe to trust or whether the budget needs raising for all three.

---

## 9. How much evaluation

**`--episodes 20` on Town02.**

Town02 is the zero-shot generalisation map — no rung ever trains on it.
`NO_OF_TEST_EPISODES` defaults to 10; 20 gives a usable standard error on
success and collision rate, which are the two noisiest metrics.

By default every episode drives the **same canonical route** — Town02 spawn
point 30, 500 waypoints at 1 m spacing, the route `main.py` has always used.
The 20 episodes differ only in policy sampling and pedestrian motion. That is a
consistency measurement, not a generalisation-across-roads measurement.

To evaluate on several fixed routes instead, pass the same list to all three
rungs:

```bat
python baseline_vae/run_ladder.py --mode test --town Town02 --episodes 20 --eval-routes 30 10 50 70 90
```

Do the single-route protocol first — it is the one that reproduces prior work.
Treat multi-route as a second pass.

**Fixed and identical across all three rungs**, so you do not have to manage any
of it: weather `CloudyNoon`; synchronous mode at `fixed_delta_seconds = 0.05`;
`SEED = 42` for numpy/random/tf; `CARLA_SEED = 42`; `TRAFFIC_MANAGER_SEED = 42`;
10 pedestrians; NPC vehicles **off** (`main.py` defines `set_other_vehicles()`
but never calls it, so no prior run had traffic either — turn it on for all
three with `set BTP_NPC=1` if you want a collision-rich protocol);
PPO 500-300-100 tanh, `ACTION_STD_INIT=0.2`, `lr=1e-4`, `γ=0.99`, `λ=0.95`,
`clip=0.2`, 15 iterations.

Each run writes a `run_manifest_*.json` recording every one of those values, so
we can prove after the fact that the three runs were comparable.

---

## 10. Where results are saved

One directory per (rung, arm), created under `<REPO>`:

```
Results_VAE_1cam_raw/
Results_VAE_4cam_raw/
Results_VAE_4cam_lidar_raw/
```

(and `_scaled` variants if you run the second arm). Each contains:

```
train_episodes.csv                  per-episode training log
test_results_town02.csv             per-episode evaluation results  <- the key file
run_manifest_train_town01.json      exact protocol used
run_manifest_test_town02.json       exact protocol used
ppo_model/actor/                    trained policy (SavedModel dir)
ppo_model/critic/
ppo_model/log_std.npy
checkpoints/checkpoint.pickle       episode/timestep/score
runs/train/                         TensorBoard scalars
runs/test/
```

`test_results_town02.csv` columns: `rung, rung_name, norm_arm, obs_dim, town,
episode, reward, success, collided, collision_count, route_completion,
distance_covered_m, center_lane_deviation_m, termination_reason, timesteps,
mean_speed_kmh, episode_wall_time_s, spawn_index, route_length,
encoder_latency_mean_ms, encoder_latency_p95_ms, step_latency_mean_ms,
policy_latency_mean_ms`.

Nothing overwrites anything: the rung tag and the arm are both in the directory
name, and the CSVs append with a header written once.

---

## 11. What to send back

Zip these and send the archive:

```bat
powershell Compress-Archive -Path Results_VAE_*_raw -DestinationPath ladder_results_raw.zip
```

Specifically I need, for each of the three rungs:

| File | Why |
|---|---|
| `Results_VAE_*/test_results_town02.csv` | **essential** — every headline metric |
| `Results_VAE_*/run_manifest_test_town02.json` | proves the three runs were comparable |
| `Results_VAE_*/run_manifest_train_town01.json` | same, for training |
| `Results_VAE_*/train_episodes.csv` | learning curves — tells us if a rung was under-trained |
| `Results_VAE_*/checkpoints/checkpoint.pickle` | final episode/timestep/score |
| console output of `preflight.py` for each rung | records TF/CARLA versions actually used |

Not needed: `ppo_model/` (large), `runs/` (TensorBoard binaries) — unless
something looks wrong and we need to dig.

If you also run the offline study on that machine:

```bat
python study/bench_latency.py
python study/aggregate.py
```

then send `results/study/*.json` and `results/study/ladder_summary.md` too —
GPU latency numbers from the real deployment box are more useful than the CPU
numbers I measured here.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Couldn't import Carla egg properly` then `ModuleNotFoundError: carla` | no API installed | step 2 |
| Client/server version mismatch | wrong `carla` package | step 2 |
| `Only input tensors may be passed as positional arguments` | TF ≥ 2.16 | step 4 — use TF ≤ 2.15 |
| `Invalid filepath extension for saving` | TF ≥ 2.16 | step 4 |
| `Unknown variable: <Variable path=CRITIC/...>` in `learn()` | TF ≥ 2.16 | step 4 |
| Server appears frozen after a crash | left in synchronous mode | restart `CarlaUE4.exe`, or the one-liner in step 1 |
| `RuntimeError: time-out of 30000ms` on `load_world` | cold map load | rerun; raise `CARLA_TIMEOUT` in `params_vae_base.py` |
| `observation width N != OBSERVATION_DIM M` | `BTP_RUNG` changed between train and test | retrain, or set `BTP_RUNG` back — widths are not interchangeable |
| Rung 3 preflight fails on BEV encoder | not trained | step 7a/7b |
| Very low FPS | rendering at high quality | `-quality-level=Low -RenderOffScreen` |
