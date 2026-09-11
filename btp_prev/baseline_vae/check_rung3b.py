"""Local shape check for rung 3b (3 cams + LiDAR). No CARLA required."""
import params_rung3b as P
from vae_selfcheck import run_check
import sys

result = run_check(P)
sys.exit(0 if result in (True, None) else 1)
