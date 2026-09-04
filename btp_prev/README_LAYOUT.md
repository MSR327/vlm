# btp_prev — layout

Baseline-ladder workspace. The goal is to show that the proposed multimodal
encoder beats progressively stronger VAE baselines, rather than just beating
the original single-camera one.

The controlled protocol for the ladder is **`docs/EXPERIMENT_PROTOCOL.md`**.
Read that before running anything.

## IMPORTANT: always run from THIS directory

Every path literal in the code (`'VAE/var_auto_encoder_model'`,
`'data_collected_town01'`, `'models/...'`) resolves against the current
working directory, and `VAE/` lives at the root. So:

    cd btp_prev
    python baseline_vae/check_rung2.py     # correct
    cd baseline_vae && python check_rung2.py   # WRONG - won't find VAE/

Running `python <folder>/<script>.py` puts that folder on `sys.path`, which
is what lets each cluster import its own modules with no code changes.

## Folders

    baseline_vae/   legacy TF VAE pipeline + the rung ladder (rungs 1-3)
    vlm/            new PyTorch multimodal stack (rung 4, the proposed method)
    VAE/            pretrained encoder weights + training set  [must stay at root]
    study/          offline analysis that needs no CARLA (cost, latency, latents)
    results/        Results_04, Results_05 - prior run outputs
                    results/study/ - offline analysis outputs
    docs/           architecture diagrams, training log, EXPERIMENT_PROTOCOL.md

`baseline_vae/` and `vlm/` each contain their OWN `parameters.py`. This is
deliberate. The two systems have different observation widths:

    baseline_vae -> OBSERVATION_DIM = 95*N + 5   (100 / 385 / 480)
    vlm          -> OBSERVATION_DIM = 128 + 5    (133)

They therefore need different PPO networks and can never share a run or a
checkpoint.

## The rung ladder

| Rung | Config           | Sensors                  | OBS_DIM | Results dir                       |
|------|------------------|--------------------------|---------|-----------------------------------|
| 1    | `params_rung1`   | VAE + 1 camera           | 100     | `Results_VAE_1cam_<arm>`          |
| 2    | `params_rung2`   | VAE + 4 cameras          | 385     | `Results_VAE_4cam_<arm>`          |
| 3    | `params_rung3`   | VAE + 4 cameras + LiDAR  | 480     | `Results_VAE_4cam_lidar_<arm>`    |
| 4    | `vlm/parameters` | multimodal encoder       | 133     | `Results_VLM`                     |

Rung 1 reference numbers already exist in `results/Results_05/test_results_gpu.csv`.

### Selecting a rung

No file edits. Both selectors are environment variables:

    BTP_RUNG=1|2|3      which rung
    BTP_NORM=raw|scaled input normalisation arm (see below)

    BTP_RUNG=2 BTP_NORM=raw python baseline_vae/run_ladder.py --mode train
    BTP_RUNG=2 BTP_NORM=raw python baseline_vae/run_ladder.py --mode test --town Town02

Each (rung, arm) writes to its own directory, so a run cannot overwrite
another's numbers.

### The two normalisation arms

The VAE was trained on `[0,1]` inputs (`VAE_Trainer_dont_touch.py:126`) but
the deployed encoder feeds raw `0-255` (`main.py:638`). `BTP_NORM` selects
which is used. `raw` is the default because it reproduces the existing
`Results_05` numbers; `scaled` is what the weights were actually fitted to.
Both are run. See `docs/EXPERIMENT_PROTOCOL.md` §3.

## Ladder files (new)

    baseline_vae/params_vae_base.py   ALL shared constants -- PPO, reward
                                      thresholds, towns, seed. Held fixed
                                      across rungs; this is the control.
    baseline_vae/parameters.py        BTP_RUNG selector
    baseline_vae/multi_sensor.py      N-camera rig + LiDAR->BEV projection
    baseline_vae/ladder_env.py        CarlaEnvironment subclass; adds collision /
                                      success / route-completion metrics
    baseline_vae/run_ladder.py        one entry point for train+test, all rungs
    baseline_vae/encode_state_vae.py  N-stream VAE encoder, norm arm, latency
    baseline_vae/train_bev_vae.py     trains rung 3's BEV encoder

## Checks that run WITHOUT CARLA

    BTP_RUNG=1 python baseline_vae/check_rung1.py    # -> (100,)  PASS
    BTP_RUNG=2 python baseline_vae/check_rung2.py    # -> (385,)  PASS
    BTP_RUNG=3 python baseline_vae/check_rung3.py    # -> SKIPPED until BEV VAE exists

    python study/cost_model.py       # analytic params / MACs / size per rung
    python study/bench_latency.py    # measured encoder + policy latency
    python study/latent_probe.py     # what the 95-d latent actually retains
    python study/aggregate.py        # build the comparison table

## Known defects, deliberately preserved

Both are reproduced in all three rungs so the ladder stays internally
consistent, and both are quantified offline in `study/latent_probe.py`:

1. **Normalisation** - see above. Run as an explicit arm.
2. **Frame layout** - `main.py:549` does `reshape((image.width, image.height, 4))`
   on a row-major `(H, W, 4)` CARLA buffer. That does not transpose the image,
   it reinterprets the bytes, so the encoder never sees a picture.
   `multi_sensor.FRAME_LAYOUT` keeps `'legacy'`; `'correct'` exists for
   measurement only. Do not flip the default mid-study.

## Still to do

- Rung 3 needs BEV data + a trained BEV encoder:
      python vlm/collect_data.py --town Town01 --frames 20000 --out data_collected_town01
      python baseline_vae/train_bev_vae.py --data data_collected_town01
- `vlm/main_vlm.py` is dead code: calls `MultimodalEdgeEncoder` with
  `img_h`/`img_w`/`in_channels`, which the constructor does not accept, and
  references an undefined `LATENT_DIM`. `main_vlm_train.py` calls it correctly.
