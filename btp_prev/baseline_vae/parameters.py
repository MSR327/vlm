"""
Rung selector for the legacy VAE baseline scripts.

The legacy scripts (main.py, test.py, mian_test.py, accuracy_check.py, ...)
all do `from parameters import *`. Before the reorg that picked up the NEW
multimodal config at the repo root, which defines OBSERVATION_DIM = 133
(128 + 5) -- wrong for the VAE, which emits 95 + 5 = 100. That mismatch did
not crash; it silently built the PPO network at the wrong input width.

This module redirects that import to the correct per-rung config, so the
legacy scripts need no edits.

SELECTING A RUNG
----------------
Preferred (no file edits, so one machine can run all three back to back):

    BTP_RUNG=1 python baseline_vae/run_ladder.py --mode test
    BTP_RUNG=2 python baseline_vae/run_ladder.py --mode test
    BTP_RUNG=3 python baseline_vae/run_ladder.py --mode test

      BTP_RUNG=1 -> VAE + 1 camera            OBSERVATION_DIM = 100
      BTP_RUNG=2 -> VAE + 4 cameras           OBSERVATION_DIM = 385
      BTP_RUNG=3 -> VAE + 4 cameras + LiDAR   OBSERVATION_DIM = 480

    BTP_NORM=raw    -> feed 0-255, as deployed (default; matches Results_05)
    BTP_NORM=scaled -> feed 0-1,   as the VAE was trained

Each (rung, arm) pair writes to its own Results_VAE_<tag>_<arm> directory, so
no run can overwrite another's numbers.
"""
import os

_RUNG = os.environ.get('BTP_RUNG', '1').strip()
if _RUNG not in ('1', '2', '3', '3b', '3B'):
    raise ValueError(f"BTP_RUNG must be 1, 2, 3, or 3b -- got {_RUNG!r}")

if _RUNG == '1':
    from params_rung1 import *
    import params_rung1 as _cfg
elif _RUNG == '2':
    from params_rung2 import *
    import params_rung2 as _cfg
elif _RUNG in ('3b', '3B'):
    from params_rung3b import *
    import params_rung3b as _cfg
else:
    from params_rung3 import *
    import params_rung3 as _cfg

print(f"[parameters] legacy VAE baseline -- rung {_cfg.RUNG} ({_cfg.RUNG_NAME}), "
      f"norm arm '{_cfg.NORM_ARM}' (scale={_cfg.VAE_INPUT_SCALE:g}), "
      f"OBSERVATION_DIM={_cfg.OBSERVATION_DIM}, out={_cfg.RESULTS_PATH}")
