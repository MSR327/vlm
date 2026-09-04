# VAE baseline ladder — controlled comparison protocol

Baseline ablation for the multimodal VLM project. Question: **how much
driving performance do we buy by adding sensory information**, along the
ladder 1 RGB → 4 RGB → 4 RGB + LiDAR, with the VAE/PPO pipeline held fixed?

## 1. Configurations

| Rung | Sensors | Streams | OBS_DIM | Config |
|------|---------|---------|---------|--------|
| 1 | 1 semantic camera (front, yaw 0) | 1 | 95·1 + 5 = **100** | `params_rung1` |
| 2 | 4 semantic cameras (0/−60/+60/180) | 4 | 95·4 + 5 = **385** | `params_rung2` |
| 3 | 4 cameras + 2D BEV LiDAR | 5 | 95·5 + 5 = **480** | `params_rung3` |

Rung 1 reproduces the handed-over baseline exactly: same sensor type, same
resolution, same placement, same frame-decode path.

## 2. What is held fixed (the control)

Everything below lives in `baseline_vae/params_vae_base.py` and is imported by
all three rungs, so it *cannot* drift between them:

- **Encoder**: the same frozen 95-d VAE (`VAE/var_auto_encoder_model`,
  13,749,662 params) encodes every camera view. No retraining, no per-view
  weights.
- **PPO**: `ACTION_STD_INIT=0.2`, `LEARNING_RATE=1e-4`, `GAMMA=0.99`,
  `LAMBDA=0.95`, `POLICY_CLIP=0.2`, `NO_OF_ITERATIONS=15`, `BATCH_SIZE=1`,
  actor/critic = 500-300-100 tanh MLP.
- **Reward, termination, route**: one definition, in `main.py`, reused by
  `ladder_env.py` via subclassing rather than copying.
- **Protocol**: `SEED=42`, train Town01, test Town02, `TRAIN_TIMESTEPS=1e6`,
  `EPISODE_LENGTH=10000`, 10 test episodes.

The **only** free variables are `NUM_CAMERAS_VAE`, `USE_LIDAR_VAE`,
`CAMERA_YAWS`, and therefore `OBSERVATION_DIM`.

### Confounds that remain, and how they are handled

| Confound | Handling |
|---|---|
| More sensors ⇒ wider observation ⇒ bigger first PPO layer | Unavoidable and *intrinsic* to the comparison. Reported explicitly: the actor grows 50k → 193k → 240k params. Any gain must be read as "modality + the capacity it forces", and the cost table makes the price visible. |
| VAE trained on **front** views, applied to left/right/rear | Stated as a limitation of rung 2. All views use the same semantic palette, so it is a mild shift, but rung 2 is a *lower bound* on what 4 views could give with per-view encoders. |
| BEV grid is out of distribution for the camera VAE | Rung 3 gets a **separate** BEV encoder of identical architecture (`train_bev_vae.py`). Same capacity ⇒ the +95 dims are attributable to the modality, not to a bigger encoder. Measured OOD evidence for this decision is in §5. |
| Input normalisation defect | Run as an explicit **arm**, see §3. |
| Frame-layout defect | Reproduced in all rungs (`FRAME_LAYOUT='legacy'`), quantified offline, see §5. |

## 3. The normalisation arms

The VAE was trained on `[0,1]` inputs (`VAE_Trainer_dont_touch.py:126`,
`rescale=1/255`) but the deployed encoder feeds raw `0–255`
(`main.py:638`). Every rung is therefore run twice:

- `BTP_NORM=raw` — as deployed. Comparable to `Results_05/test_results_gpu.csv`.
- `BTP_NORM=scaled` — as the VAE was trained.

Results go to `Results_VAE_<rung>_<arm>/`, so no run overwrites another.

## 4. Run order

```bash
cd btp_prev            # ALWAYS — every path literal is CWD-relative

# 0. shape/loader check on any machine, no CARLA needed
for R in 1 2 3; do BTP_RUNG=$R python baseline_vae/check_rung$R.py; done

# 1. rung 3 only: collect BEV data, then train the BEV encoder
python vlm/collect_data.py --town Town01 --frames 20000 --out data_collected_town01
python baseline_vae/train_bev_vae.py --data data_collected_town01 --out VAE/bev_encoder_model

# 2. train — 6 runs (3 rungs x 2 arms)
for ARM in raw scaled; do for R in 1 2 3; do
  BTP_RUNG=$R BTP_NORM=$ARM python baseline_vae/run_ladder.py --mode train
done; done

# 3. test — zero-shot generalisation to Town02
for ARM in raw scaled; do for R in 1 2 3; do
  BTP_RUNG=$R BTP_NORM=$ARM python baseline_vae/run_ladder.py --mode test --town Town02
done; done

# 4. aggregate
python study/aggregate.py
```

Each rung must be trained from scratch: the observation widths differ, so a
checkpoint can never transfer between rungs. That is enforced by construction —
separate `PPO_MODEL_PATH` per rung and arm.

## 5. Metrics

**Closed-loop (needs CARLA)** — `run_ladder.py` writes one row per episode to
`Results_VAE_<rung>_<arm>/test_results_town02.csv`:

| Metric | Definition |
|---|---|
| Reward | Sum of per-step reward (`main.py:318-332`) |
| Success rate | Fraction of episodes terminating in `route_complete` with no collision |
| Collision rate | Fraction of episodes terminating in `collision` |
| Route completion | Waypoints advanced ÷ route length, per episode |
| Distance covered | Metres (waypoints are 1 m apart) |
| Centre-lane deviation | Mean lateral error over the episode |
| Termination reason | `collision` / `lane_departure` / `stall` / `overspeed` / `route_complete` / `step_limit` |
| Encoder latency | mean and **p95** ms per `process()` call |
| Step / policy latency | mean ms |
| Mean speed | km/h |

The prior `Results_05` CSV had no success, collision or route-completion
column — those three are new, and are why `ladder_env.py` exists.

**Offline (no CARLA)** — `study/`:

| Script | Produces |
|---|---|
| `study/cost_model.py` | Analytic params / MACs / model size per rung |
| `study/bench_latency.py` | Measured encoder + policy latency, p50/p95 |
| `study/latent_probe.py` | Linear-probe R² of driving-relevant targets from the 95-d latent, under both norm arms and both frame layouts; latent effective rank; OOD response to BEV input |

## 6. Reporting

Report mean ± std over the 10 test episodes, and state the seed. With 10
episodes and a single seed, treat differences smaller than roughly one
standard deviation as unresolved rather than as a result. If a rung's ordering
matters to the VLM argument, re-run it at ≥3 seeds before claiming it.
