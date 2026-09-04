# `ladder` — Phase 2 perception stack

Implements the target architecture in `PHASE2_BRIEF.md` §3 and the slow→fast
seam in §7. Nothing here imports the legacy `MTP_TESTING` modules; the legacy
tree is untouched.

## Status

| Built | Module | Brief |
|---|---|---|
| ✅ | `config.py` — every ladder invariant in one place | §1 |
| ✅ | `contracts.py` — Tier S / Tier P separation, enforced by types | §2 |
| ✅ | `sensors/bev.py` — 2-bin height histogram, no goal channel | §3 |
| ✅ | `models/backbones.py` — frozen registry, hard-fail on missing deps | §3 |
| ✅ | `models/lidar_tokenizer.py` — trained-from-scratch BEV branch | §3 |
| ✅ | `models/fusion.py` — cross-attention + concat + FiLM arms | §3, §8 Phase 5 |
| ✅ | `models/heads.py` — waypoint, traffic light, drivable BEV, instruction-done | §8 Phase 4 |
| ✅ | `models/encoder.py` — assembled fast pathway + capacity solver | §1, §3 |
| ✅ | `intent/oracle.py` — scripted oracle + constant ablation arm | §7 |
| ✅ | `tests/test_ladder.py` — 30 invariant tests, no CARLA or data needed | §9 r4 |

**Blocked, not built** — every one needs a decision or an asset that does not exist yet:

| Not built | Blocked on |
|---|---|
| CARLA env, route set, reward, metrics (§8 Phase 1) | CARLA version — §12 open item |
| Stage 0 collection (§8 Phase 2) | CARLA version; expert choice (coach vs autopilot) |
| Feature cache (§8 Phase 3) | dataset; ~60 GB disk confirmation |
| Stage 1 training + Stage 2 selection (§8 Phases 4–5) | dataset |
| PPO integration (§8 Phase 6) | all of the above |

## Invariants, and how they are enforced

- **`D_LATENT = 128` for every configuration.** Asserted at the end of
  `FastPathwayEncoder.forward` and covered by
  `test_every_config_emits_the_invariant_width`.
- **Tier P cannot reach the policy.** `PrivilegedState` has no
  `to_policy_vector`, and `assert_no_privileged_leak()` fails if one is added or
  if a field name migrates between tiers.
- **Parameter matching to ±10%.** `solve_capacity_match()` widens the fusion MLP
  of thinner configurations to a common trainable budget. Without it the spread
  across R1–R4 is 26% and "more sensors helped" is inseparable from "more
  parameters helped". `test_unmatched_configs_fail_the_invariant` guards the
  guard.
- **No silent fallbacks.** A missing backbone dependency, a missing BEV on a
  LiDAR config, a missing instruction on a language config, or reading the
  intent seam before it is latched all raise.

## Configurations

```python
from ladder.models.encoder import FastPathwayEncoder, BlindEncoder, solve_capacity_match

CFGS = {
    "R1": dict(backbone="resnet34", n_views=1, use_lidar=False, use_language=False),
    "R2": dict(backbone="resnet34", n_views=4, use_lidar=False, use_language=False),
    "R3": dict(backbone="resnet34", n_views=4, use_lidar=True,  use_language=False),
    "R4": dict(backbone="resnet34", n_views=4, use_lidar=True,  use_language=True),
}
ratios = solve_capacity_match(CFGS)          # run once, record the ratios
enc = FastPathwayEncoder(fusion_mlp_ratio=ratios["R3"]["mlp_ratio"], **CFGS["R3"])
```

R0 is `BlindEncoder` — `[ego, nav]` through an MLP to the same 128 dims, so the
PPO input is literally identical in shape and normalisation.

## Running the tests

```
cd MTP_TESTING && python3 -m unittest discover -s ladder/tests -t .
```

No CARLA, no dataset, no GPU. On this machine `torchvision` is absent, so use
`backbone="scratch_tiny"` locally; the Windows box needs
`pip install torchvision` before `resnet34` will build — it will raise rather
than substitute.
