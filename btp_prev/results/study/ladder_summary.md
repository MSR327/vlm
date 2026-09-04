# VAE baseline ladder — results summary

## Computational cost (measured on CPU (arm64 macOS), TF 2.20.0)

| config | obs dim | encoder params | actor params | total params | MACs (M) | infer size (MB) | encoder ms | encoder p95 | policy ms | pipeline ms | max Hz |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1cam | 100 | 13749662 | 231102 | 14211765 | 74.204 | 53.332 | 4.036 | 4.370 | 1.982 | 6.019 | 166.153 |
| 4cam | 385 | 13749662 | 373602 | 14496765 | 296.268 | 53.876 | 6.431 | 7.007 | 1.982 | 8.412 | 118.872 |
| 4cam_lidar | 480 | 27499324 | 421102 | 28341427 | 370.289 | 106.508 | 14.225 | 15.582 | 1.994 | 16.218 | 61.658 |

## Latent information (n_train=12000, n_test=2000)

| layout | arm | latent_std | eff_rank | nonfinite_pct | mean_R2 | R2_road_frac | R2_lane_offset | R2_lane_heading | R2_roadline_frac | R2_obstacle_frac | R2_free_ahead |
|---|---|---|---|---|---|---|---|---|---|---|---|
| train_style | scaled | 9.928 | 7.741 | 0.000 | 0.836 | 0.973 | 0.976 | 0.817 | 0.954 | 0.452 | 0.843 |
| train_style | raw | 362753.012 | 1.000 | 0.001 | 0.808 | 0.947 | 0.949 | 0.782 | 0.955 | 0.362 | 0.853 |
| carla_legacy | scaled | 7.052 | 7.128 | 0.000 | 0.769 | 0.973 | 0.975 | 0.729 | 0.878 | 0.379 | 0.680 |
| carla_legacy | raw | 2984.424 | 5.837 | 0.000 | 0.758 | 0.956 | 0.971 | 0.706 | 0.865 | 0.326 | 0.725 |

Pixel baseline (16x8 grey, 128 dims): mean R² = 0.6648

### OOD response — BEV grid through the camera VAE

| bev_effective_rank | cam_effective_rank | bev_latent_std | cam_latent_std | bev_mean_pairwise_cos | cam_mean_pairwise_cos |
|---|---|---|---|---|---|
| 18.937 | 8.909 | 6129285.770 | 2931.085 | 0.138 | 0.905 |

## Multi-view fusion mechanism (proxy: left/right half-frames)

| variant | dims | eff_rank | mean_R2 | R2_road_frac | R2_lane_offset | R2_lane_heading | R2_roadline_frac | R2_obstacle_frac | R2_free_ahead |
|---|---|---|---|---|---|---|---|---|---|
| view A (left half) | 95 | 8.114 | 0.622 | 0.633 | 0.542 | 0.593 | 0.863 | 0.273 | 0.831 |
| view B (right half) | 95 | 5.908 | 0.688 | 0.923 | 0.961 | 0.579 | 0.643 | 0.334 | 0.687 |
| concat [A, B] | 190 | 8.843 | 0.875 | 0.982 | 0.982 | 0.895 | 0.966 | 0.555 | 0.869 |
| full frame (1 pass) | 95 | 7.741 | 0.836 | 0.973 | 0.976 | 0.817 | 0.954 | 0.452 | 0.843 |

Gain from concatenating a second view: **+0.1870** mean R² over the better single view (0.6878 → 0.8748).

Rank additivity: 14.02 expected if independent, 8.84 actual = **63% additive**. The 190 concatenated dims carry only 4.7% as many effective dimensions.

## Closed-loop driving (CARLA)

_no results yet. On the CARLA machine:_

```bash
for ARM in raw scaled; do for R in 1 2 3; do
  BTP_RUNG=$R BTP_NORM=$ARM python baseline_vae/run_ladder.py --mode train
  BTP_RUNG=$R BTP_NORM=$ARM python baseline_vae/run_ladder.py --mode test --town Town02
done; done
```
