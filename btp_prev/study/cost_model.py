"""
Analytic + measured computational cost of each ladder configuration.

Two things are reported and they answer different questions:

  ANALYTIC (params, MACs, bytes) -- hardware-independent, reproducible, and
  the right number for a paper table. Derived from the layer geometry of the
  frozen VAE encoder (VAE/VAE_Trainer_dont_touch.py) and the PPO actor/critic
  (baseline_vae/main.py:666-740).

  MEASURED (latency) -- wall-clock on THIS machine. Reported with p50/p95 over
  many warm calls, because a mean alone hides the tail that matters for a
  20 Hz control loop.

The camera VAE is shared across all views, so rung 2 is one batched forward
pass of batch 4, not four passes. Rung 3 adds a second, architecturally
identical encoder for the BEV grid.
"""
import numpy as np

LATENT_DIM = 95
NAV_DIM = 5
IN_H, IN_W, IN_C = 160, 80, 3


def _conv(h, w, cin, cout, k, stride):
    """-> (out_h, out_w, params, macs) for 'same' padding."""
    oh, ow = int(np.ceil(h / stride)), int(np.ceil(w / stride))
    params = k * k * cin * cout + cout
    macs = oh * ow * k * k * cin * cout
    return oh, ow, params, macs


def vae_encoder_cost():
    """Encoder: conv(32,4,s2) conv(64,3,s2) BN conv(128,4,s2) conv(256,3,s2)
    flatten dense(1024) -> mu(95) + sigma(95)."""
    h, w, c = IN_H, IN_W, IN_C
    params = macs = 0
    for cout, k, s in [(32, 4, 2), (64, 3, 2), (128, 4, 2), (256, 3, 2)]:
        h, w, p, m = _conv(h, w, c, cout, k, s)
        params += p
        macs += m
        c = cout
        if cout == 64:                      # BatchNormalization after conv2
            params += 4 * cout              # gamma, beta, moving_mean, moving_var
            macs += h * w * cout
    flat = h * w * c
    params += flat * 1024 + 1024            # dense1
    macs += flat * 1024
    for _ in range(2):                      # mu and sigma heads
        params += 1024 * LATENT_DIM + LATENT_DIM
        macs += 1024 * LATENT_DIM
    return dict(params=params, macs=macs, flat=flat, final_hw=(h, w, c))


def mlp_cost(in_dim, hidden=(500, 300, 100), out_dim=2):
    params = macs = 0
    d = in_dim
    for hdim in hidden:
        params += d * hdim + hdim
        macs += d * hdim
        d = hdim
    params += d * out_dim + out_dim
    macs += d * out_dim
    return dict(params=params, macs=macs)


RUNGS = {
    1: dict(name='1cam',       n_cam=1, lidar=False),
    2: dict(name='4cam',       n_cam=4, lidar=False),
    3: dict(name='4cam_lidar', n_cam=4, lidar=True),
}


def config_cost(rung):
    cfg = RUNGS[rung]
    enc = vae_encoder_cost()
    n_streams = cfg['n_cam'] + (1 if cfg['lidar'] else 0)
    obs_dim = LATENT_DIM * n_streams + NAV_DIM

    # Camera VAE weights are SHARED across views -> counted once.
    # A LiDAR BEV encoder is a SEPARATE set of weights -> counted again.
    encoder_param_sets = 1 + (1 if cfg['lidar'] else 0)
    enc_params = enc['params'] * encoder_param_sets
    enc_macs = enc['macs'] * n_streams          # one forward pass per stream

    actor = mlp_cost(obs_dim, out_dim=2)
    critic = mlp_cost(obs_dim, out_dim=1)

    total_params = enc_params + actor['params'] + critic['params']
    return dict(
        rung=rung, name=cfg['name'], n_cameras=cfg['n_cam'],
        lidar=cfg['lidar'], n_streams=n_streams, obs_dim=obs_dim,
        encoder_params=enc_params, encoder_macs=enc_macs,
        actor_params=actor['params'], critic_params=critic['params'],
        policy_macs=actor['macs'],                 # inference uses the actor only
        total_params=total_params,
        inference_params=enc_params + actor['params'],
        inference_macs=enc_macs + actor['macs'],
        fp32_mb=total_params * 4 / 1024 ** 2,
        inference_fp32_mb=(enc_params + actor['params']) * 4 / 1024 ** 2,
    )


if __name__ == '__main__':
    for r in (1, 2, 3):
        c = config_cost(r)
        print(f"rung {r} {c['name']:>11s}  obs {c['obs_dim']:4d}  "
              f"enc_params {c['encoder_params']:>10,}  "
              f"MACs {c['inference_macs']/1e6:8.2f}M  "
              f"size {c['inference_fp32_mb']:6.2f} MB")
