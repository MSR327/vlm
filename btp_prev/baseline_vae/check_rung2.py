"""Local shape check for rung 2. No CARLA required."""
import params_rung2 as P
from vae_selfcheck import run_check
import sys

result = run_check(P)
sys.exit(0 if result in (True, None) else 1)
