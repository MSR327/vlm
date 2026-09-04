# Phase 2 Implementation Brief — Perception Stack Rebuild

**For:** the implementing agent working in `MTP_TESTING/`
**Date:** 2026-08-31
**Prerequisite reading:** `findings/` — the VLM-PPO Stack Audit, the Perception Ladder Spec, and the Ladder Precedent Audit. Read all three before writing code. This brief assumes their findings and does not repeat the evidence.

---

## 0 · Goal, in one paragraph

Replace the from-scratch 167K-parameter `MultimodalEdgeEncoder` with a **pretrained** multimodal perception stack that consumes 4 RGB views + LiDAR, fuses them with learned-query cross-attention, and emits a fixed-width 128-d latent to PPO. PPO's algorithm is **not** being changed. The language pathway is a frozen text encoder whose embeddings are precomputed offline into a lookup table. A typed slow→fast interface is cut now, backed by a scripted oracle, so a real VLM can be swapped in later without a rewrite.

**The single most important fact in this brief:** the current environment leaks the answer into the observation. Until that is fixed, a better perception stack cannot show up in any measurement. Fix it first. Everything else is wasted work until it is done.

---

## 1 · Non-negotiables

These hold for every experiment. A run that violates one is not comparable and does not count.

| Invariant | Value | Why |
|---|---|---|
| Latent width | `D_z = 128` | Every encoder config projects to the same width, so "richer sensors" never confounds with "wider policy input". Also what makes edge distillation work later. |
| PPO observation | `z(128) + ego(8) + nav(8)` = 144-d, growing to 152-d once the waypoint head exists (`+ tau_nom(8)`) | Shape and semantics fixed. Resize PPO's input layer **once**, then never again. |
| PPO algorithm | unchanged | Not in scope. See §4 for exactly what does and does not change. |
| Encoder during PPO | **frozen** | Perception is the independent variable; PPO is the measuring instrument. If the encoder adapts during RL, "richer perception helped" becomes inseparable from "richer perception adapted more easily under RL". |
| Encoder params | matched to ±10% across configs | Otherwise "more sensors helped" is indistinguishable from "more parameters helped". |
| Reward | identical function, identical weights, all configs | Reads from privileged state, never from the observation. |
| Seeds | 3 seeds: 0, 1, 2 — the same three everywhere | Paired comparison per seed. Currently **nothing** is seeded despite `SEED = 42` existing in `parameters.py`. |
| Visual history | 2 frames: `t` and `t−0.2s`, every modality | Uniform, so no config has a temporal advantage. |

---

## 2 · Fix the leak — do this before anything else

### The defect

`main_vlm_train.py:_get_obs()` computes two of the five telemetry dimensions from `self.map.get_waypoint()` — CARLA's ground-truth road graph:

```python
dist_from_center = veh_loc.distance(current_wp.transform.location)
norm_angle       = acos(dot(veh_fwd, wp_fwd)) / pi
```

These are concatenated into the observation **after** the transformer, so they reach the policy untouched by perception. And `main_vlm_train.py:step()` computes:

```python
reward = speed_factor * centering_factor * angle_factor + 0.3 * speech_reward
```

All three factors are arguments the policy already receives for free. A five-input MLP maximises this reward with every camera unplugged, and gradient descent will find that solution because it is by far the easiest one.

### The fix — a two-record environment

`step()` returns two disjoint records. The policy's signature accepts only the first. This is enforced by **types**, not by discipline:

```python
step() -> (SensorObs, PrivilegedState, reward, done, info)

policy:     SensorObs -> Action                          # cannot see PrivilegedState
reward_fn:  (SensorObs, PrivilegedState, Action) -> float
metrics_fn: PrivilegedState -> dict
```

**Tier S — policy-visible.** Membership test: *could a production vehicle obtain this from its own hardware, with no map oracle?*

| Field | Shape | Notes |
|---|---|---|
| `rgb_front / left / right / rear` | `2 × 3 × 224 × 224` | True RGB. See §5 for the BGRA bug. Per-camera FOV from a **per-camera blueprint**. |
| `lidar_sweep` | raw points `N × 4` | Stored raw. BEV rasterisation is a model-side choice so it can change without re-collecting. |
| `ego.speed` | 1 | Wheel-speed / IMU. Legitimate. |
| `ego.yaw_rate, accel_long, accel_lat` | 3 | IMU. Legitimate. |
| `ego.prev_action` | 2 | Proprioception. |
| `ego.speed_hist` | 2 | Speed at `t−0.2s`, `t−0.4s`. |
| `nav.target_point` | 2 | Next route node in ego frame, sampled **25–40 m ahead**, with σ = 1.0 m Gaussian noise. |
| `nav.command` | 6 (one-hot) | follow / left / right / straight / lane-left / lane-right. |
| `instruction_embedding` | 128 | Language config only. Cached lookup — see §6. |

**Tier P — privileged, reward and metrics only.** Lane-centre distance, heading error, lane id/type, junction geometry, collision and lane-invasion and red-light events, route completion, distance-along-route, blocked timer, ground-truth actor list, expert future trajectory, semantic segmentation and depth.

> **The rule that makes Tier P usable:** privileged information may be a **training label**; it may never be a **deployed input**. An auxiliary head predicting traffic-light state from pixels is supervised by ground truth, but at inference the prediction comes from the camera. The test is mechanical: trace whether the quantity appears in the forward pass at evaluation time.

**Tier F — forbidden anywhere, including as a label.** Any Tier-P quantity in the policy's evaluation-time forward pass; the full future route beyond the single sampled target point; other agents' future trajectories; episode seed / route id / town id / weather id as model inputs; evaluation-set statistics in normalisation constants; any reward-derived signal fed back as an observation.

### Why the nav target point is allowed but lane geometry is not

The line is not "was the map consulted". It is **"is this quantity the answer the perception is supposed to compute"**. Every CARLA Leaderboard agent receives a sparse GPS route plus a high-level command; a real car has a nav system. The 25–40 m sampling distance and the 1 m noise are what keep it a hint rather than a target to servo onto. Dense lane-centre distance is the answer. A target point closer than 20 m becomes the answer.

---

## 3 · Target architecture

```
4× RGB @224², 2 frames ──▶ FROZEN pretrained backbone (shared across views)
                           + view embedding + time embedding
                           → ~64 pooled tokens per view-frame

LiDAR raw points       ──▶ 2-bin height histogram BEV (above/below ground)
                           NO goal channel  ──▶ small TRAINED tokeniser → ~196 tokens

instruction (optional) ──▶ frozen text encoder, run OFFLINE
                           → cached lookup table → 8 tokens
                                        │
        ┌───────────────────────────────┘
        ▼
   ┌──────────────────────────────────────┐
   │  32 learned latent queries           │   ← the fusion stage
   │  2× cross-attn  (Q ← all tokens)     │
   │  2× self-attn   (among Q)            │
   └──────────────────────────────────────┘
        │
   pool → Linear → LayerNorm → z (128-d)   ◀── FROZEN after Stage 1
        │
   ┌────┴────────┬──────────────────┬─────────────────────────┐
   ▼             ▼                  ▼                         ▼
waypoint      aux heads          PPO actor-critic       (later) DrivingIntent
head          TL state           [z, ego, nav, τ]        consumer
4×(x,y)       BEV occupancy      → action
FROZEN        depth / seg        TRAINED (unchanged algo)
after S1      Stage-1 only
```

### Why learned-query cross-attention

It is the only fusion operator whose **output width is independent of how many modalities feed it**. Adding LiDAR adds tokens, not output dimensions — which is exactly what the fixed-latent invariant requires. Cost is linear in token count, so the edge budget survives. Attention maps are inspectable evidence rather than only a number. And it subsumes concatenation and FiLM, which remain available as ablations.

**Rejected:** FiLM alone — one global γ/β applied identically to every token can express "drive cautiously" but structurally cannot express "the van on your right". Joint self-attention over all tokens — quadratic at ~2000 tokens, and the single line item that would end the edge story.

**Correction to carry forward:** the earlier audit called mean-pooling "the biggest architectural flaw". That was too strong — TransFuser average-pools to a 64-d vector and was SOTA. Keep the queries, but justify them by **referring expressions and spatial grounding**, not by pooling being broken.

### Backbone selection

At 8–12 GB with CARLA co-resident:

| Candidate | Params | Aligned text tower? | Note |
|---|---|---|---|
| **MobileCLIP-S0** | ~11M | **yes** | Cheapest. The aligned text tower means instruction embeddings and visual tokens share a space, so cross-attention between them is far better conditioned. |
| **DINOv2-S/14** | ~21M | no | Stronger dense spatial features. Pair with a separate sentence encoder. |
| SigLIP-Base/16 | ~86M | yes | ~17 GFLOPs per view-frame × 8 per control step. **Will not hold 20 Hz** next to CARLA on this hardware. Do not use. |

Default to **MobileCLIP-S0** unless an offline comparison on waypoint ADE says otherwise. Run that comparison in Stage 2 — it costs no CARLA time.

### LiDAR representation

Use TransFuser's **2-bin height histogram** (points above / below ground plane), *not* the current single max-height channel. **Do not rasterise the goal location into the BEV** — TransFuser does, and it means their "LiDAR helps" result partly measures a better goal encoding. Keeping the goal out is required for the ablation to be clean; say so explicitly in any write-up.

---

## 4 · The PPO boundary — exactly what changes

**Does not change:** the PPO algorithm, the clip/GAE/epoch structure, the actor-critic architecture, the optimiser, the update rule. Do not touch `ActorCritic` beyond the input dimension.

**Does change, and must:**

1. **The observation vector's composition.** Two leaked dims come out; nav fields go in. PPO's input layer is resized **once**, to 144-d (and to 152-d when the waypoint head lands). Then frozen in shape forever.
2. **Where the reward reads from.** It now takes `PrivilegedState`, not the observation. Same arithmetic for now — the reward function is an invariant and must not be tuned mid-ladder.
3. **The checkpoint load must become mandatory.** `train()` currently loads `models/pretrained_vlm_encoder.pth` only `if os.path.exists(...)`, and that file does not exist — so PPO has been optimising a 495K-parameter MLP on top of a **fixed random projection of the images**, printing no warning. Same pattern in `test()` and `PIL_edge_vlm.py`. Make it raise.
4. **Seeding.** `random`, `numpy` and `torch` seeded from `SEED`, and the seed recorded in every results file.
5. **The success metric.** `success = not collision and ep_steps >= 200` currently scores a **parked car at 100%**, because the stall terminal fires at step 201. Replace with route completion, infraction score, driving score = RC × IS, and a success criterion a stationary vehicle fails.

**Known, deferred, do not solve now:** the PPO sample budget. The Precedent Audit found 1.5M steps is two orders of magnitude below CaRL's 300M and 7× below Roach's 10M. Run a **pilot** — one config at several budgets, plot the learning curve — before fixing any number. Do not silently adopt 1.5M.

---

## 5 · Sensor and preprocessing defects to fix during collection

All of these are in `main_vlm_train.py` and must be fixed before Stage 0, because they contaminate the dataset:

- **`_setup_sensors` reuses one blueprint** for all four cameras, so left/right silently inherit the front camera's 125° FOV instead of 90°. `collect_data.py` creates four separate blueprints correctly — so the pretraining dataset and the RL environment currently have **different camera intrinsics**. Fix the env to match the collector.
- **Pre-register the front camera FOV at ~60–70°.** This decides whether the multi-view comparison can measure anything at all. TransFuser's ablation finds camera FOV has the largest single impact on driving score; a 125° front camera already covers most of what side cameras would add, and the multi-view result would come back null for the wrong reason.
- **BGRA → RGB.** CARLA delivers BGRA; `[:, :, :3]` yields **BGR**, labelled RGB throughout. Harmless for a from-scratch CNN, **wrong for any pretrained backbone** — which is now the whole plan. Fix it.
- **LiDAR `rotation_frequency` is never set.** CARLA's default is 10 Hz against a 20 FPS sim tick, so each frame captures roughly half a revolution and the "360° surround BEV" is an alternating half-sweep. Set it explicitly and confirm against the CARLA version in use.
- **`_spawn_npcs` calls `set_autopilot(True)` without binding the Traffic Manager to synchronous mode.** `collect_data.py` does this correctly. In sync mode this is a known source of stalls.
- **Resolution 224², not 160×80.** No pretrained backbone can recover detail that was never sampled. This is the decision that makes the plan viable or not.

---

## 6 · The language pathway

**The existing `encode_text` is discarded entirely.** It is a substring match against nine hard-coded anchors, falling back to `torch.randn` seeded by the sum of the string's character codes. Measured in the audit: `cos("steer_left", "turn_left") = 0.085` for two strings that map to the *same* command id. It is a hash, not an encoder.

**Replacement:** a frozen sentence encoder — MobileCLIP-S0's text tower (preferred, aligned with the vision tower) or MiniLM-L6 (22M). Trainable 128-d projection on top; the encoder itself never trains.

**The instruction set is finite and precomputed.** ~40 templates × ~18 LLM paraphrases ≈ 700 strings. Encode all of them **once**, offline, store a lookup table. The text encoder therefore never runs in the control loop — on the training machine or on the edge device. The language pathway costs a 128-float table lookup at inference. This is an honest architecture, not a shortcut, and it is what makes language affordable at 8–12 GB.

**Split on two axes, and report them separately:**
- Held-out **paraphrases** of seen templates → surface-form generalisation. The easy test.
- Held-out **templates** — new structures, new referent types → compositional generalisation. The real test. A system can pass the first and fail this completely.

**Do not add a compliance reward term.** The reward stays byte-identical across configs so that any difference is a pure input difference. Instead make compliance *instrumentally necessary*: construct instruction-critical routes where ignoring the instruction reduces route progress, and suppress `nav.command` at those junctions so only the sentence disambiguates the branch. The legacy `_compute_speech_reward` is directly hackable — it rewards steering sign, so a policy can "comply" by swerving.

**Centre the language claim on notice instructions, not junctions.** A hazard notice delivered before the hazard is sensor-visible ("careful, a cyclist is coming from your right" behind an occlusion) supplies information no amount of perception richness could supply. No verified paper shows language adding *navigational* value on top of an intact nav command — LMDrive removes the command entirely. The notice case is where the effect is real. Note this is **not novel**: it is LMDrive's LangAuto-Notice track. Position it as a controlled replication at ~1/300th the model scale, which is itself the interesting question — does notice-following require an LLM at all?

**Also add a misleading-instruction condition.** Cheap, and an agent that follows an unsafe instruction has learned the wrong thing.

---

## 7 · The slow→fast seam (build now, fill later)

Cut this interface **now**, even though nothing sophisticated sits behind it yet.

```python
@dataclass(frozen=True)
class DrivingIntent:
    target_waypoints: np.ndarray   # (K, 2) ego-frame, metres
    lane_action:      LaneAction   # follow | left | right | stop
    speed_cap:        float        # m/s
    hazard:           bool
    instruction_done: bool
    valid_until:      float        # timestamp; latched between updates
```

**First and only adapter for now: a scripted oracle** driven by the route planner and `PrivilegedState`. Perfect by construction.

This is not busywork. It buys four concrete things:

1. The entire fast pathway gets built, trained and validated against a known-good intent stream — so when something scores badly you know it is not the upstream module.
2. The oracle's score is the **ceiling** any future VLM must approach. That number tells you whether a VLM is worth adding at all, before you build one.
3. The whole slow pathway is ablatable by swapping in a constant.
4. A real VLM later is a swap behind an existing, already-validated interface rather than a rewrite.

Note: `DrivingIntent` is *not* on the policy's input path during the current phase. It is infrastructure. Do not let it become a route for privileged state to reach the policy — the oracle reads Tier P, so anything derived from it that reaches the policy at evaluation time is Tier F. When a model replaces the oracle, that model sees only Tier S.

---

## 8 · Build order, with gates

### Phase 1 — Harness. No learning at all.
Two-record environment with the enforced signature. Route set (40 train / 30 eval). Weather schedule. Reward reading `PrivilegedState`. Full metric pipeline. Seeding. Parallel-env wrapper (**2–4 envs**, not 8 — CARLA and the model share one GPU).

**Validation:** a scripted pure-pursuit agent driven by the privileged route scores near-perfectly; the same controller with the route withheld fails. That pair of runs proves the harness measures driving. Nothing else does.

### Phase 2 — Stage 0 collection. The long pole.
~150K frames on Town01 (+ Town03/04 if time allows), 6 weather presets. **Review the storage schema before starting the run** — you will change the BEV representation and the target-point distance at least once, and re-collecting is the expensive mistake.

Store **raw**: raw LiDAR points, the full route, all Tier-P state, instructions per frame.

**Include ~20% perturbation-recovery frames.** Inject a bounded lateral or heading perturbation into the ego, then record the expert's recovery trajectory as the label. Without this, behaviour cloning gives a head that is excellent on-distribution and useless once the car drifts. ChauffeurNet flags sensor-space perturbation as the hard case — budget for it.

**Expert choice:** the CARLA autopilot is a weak expert. Roach's central result is that an RL-trained privileged coach generates better IL data and its students do better. A privileged coach is the right answer but is real work — if it is out of scope for now, use the autopilot and **write down that this caps the ceiling**.

### Phase 3 — Backbone feature cache.
Run the frozen backbone once over the dataset, pool tokens 256→64, store fp16. At 150K frames × 4 views × 2 frames × 64 tokens × 384 dims this is roughly **60 GB**. Confirm you have the disk before starting.

This is the step that makes 8–12 GB viable: no backbone forward and no backbone activations during Stage 1 training.

### Phase 4 — Stage 1 supervised pretraining. No CARLA.
Train the BEV tokeniser, latent queries, cross-attention, projection, waypoint head and auxiliary heads on cached features.

Losses: waypoint L1 (primary), traffic-light-state cross-entropy, BEV drivable-area/occupancy BCE, instruction-completion BCE. **The auxiliary heads are the cheapest large win available** — LMDrive's 16.9 → 36.2 comes from exactly this.

**Split by route and weather, never by shuffled row index.** Driving frames are temporally correlated; a random split leaks neighbouring frames into validation.

### Phase 5 — Offline architecture selection. No CARLA.
Backbone choice, query count, resolution, and the fusion-operator comparison (concat / FiLM / cross-attention / multi-resolution) all decided here on waypoint ADE/FDE and auxiliary-head accuracy. Cheap, fast, parallelisable, and it prunes the expensive closed-loop matrix.

**Include a multi-resolution fusion variant.** TransFuser fuses RGB and LiDAR at several feature-map scales, and that is the paper with the cleanest RGB+LiDAR ablation. Without this variant, a null "LiDAR didn't help" result is ambiguous between *the sensor doesn't help* and *we fused too late*.

### Phase 6 — PPO. Frozen encoder, unchanged algorithm.
Freeze encoder, waypoint head, auxiliary heads, text encoder and normalisation statistics. Train the actor-critic only.

**⛔ GATE — do not build anything richer until all four pass:**

- **`DS(blind) < 0.6 × DS(front-RGB)`.** The blind config sees only `[ego, nav]` through a 2-layer MLP that outputs 128-d, so the PPO input is literally identical in shape and normalisation. If it scores competitively, the observation contract is **still leaking** and the correct response is to push the target point further out — not to conclude the environment is broken. **Decide the threshold before seeing the number.**
- **`DS(front-RGB)` meaningfully > `DS(blind)`**, with the gap larger than the seed spread. Otherwise the environment does not reward perception and no amount of additional sensors can help.
- **`ΔDS = DS(PPO) − DS(IL-only)` > 0.** Evaluate every config twice — once with the policy, once with the residual forced to zero. If PPO adds nothing anywhere, that is a finding about the interface, not about RL.
- **Seed spread small enough** that a plausible config-to-config effect would be visible above it. If not, more seeds or more evaluation routes — decided now, not after five configs.

Every one of these is cheap to fix here and enormously expensive to discover later.

---

## 9 · Hard engineering rules

The previous codebase failed in specific, repeatable ways. These are not style preferences.

1. **No silent fallbacks.** Missing checkpoint, missing `torchvision`, missing dataset → **raise**. The current code has `if os.path.exists(checkpoint)` around a required file and trains on random weights without a warning. `SceneContextEncoder` silently substitutes a random 4-layer CNN with a different feature dimension when `torchvision` is absent — same class name, same output shape, entirely different semantics, one printed warning.
2. **No stubs shaped like components.** Three modules in the current tree have plausible names, plausible signatures, plausible output shapes, and are functionally random number generators (`encode_text`, the `SceneContextEncoder` projection head, `telemetry_proj`). If it is not implemented, `raise NotImplementedError`. Never return a plausibly-shaped tensor.
3. **Assert shapes and dtypes at every module boundary.** Three separate train/inference mismatches got through because nothing checked — FiLM input type flipping between a string list and a continuous vector, Scene FiLM applied at RL time but never trained, `telemetry_proj` never trained at all.
4. **Every module that is constructed must be reachable from a test.** If nothing calls it, delete it.
5. **Measure throughput on day one.** Four views × two frames is ~8 backbone forwards per control step. Establish achieved env-steps/second with CARLA running before building on top of the number. If it is half of what was assumed, cut the step budget uniformly — never cut configurations.
6. **No claim without an artifact on disk.** The previous design documents contained latency tables, driving scores and route-completion targets for a system that had never been run end to end. Every number in a results table must be traceable to a file.
7. **Delete the dead files** rather than repairing them: `main_vlm.py`, `test_vlm_carla.py`, `PIL_edge_vlm.py`, `convert.py`, `tf_multimodal_encoder.py`. They target an architecture that no longer exists.
8. **Repair or re-clone the git object store.** `MTP_TESTING/.git` is corrupt — the packfile does not match the index — so no history is readable and diff-based review is impossible. Do this before more work lands.

---

## 10 · Explicitly do NOT do

- Do not build the language config before the gate in Phase 6 passes.
- Do not tune the reward. It is an invariant; tuning it mid-sequence invalidates every completed run.
- Do not fine-tune the backbone. Not the vision tower, not the text encoder. Frozen means frozen.
- Do not let PPO gradients reach any pretrained parameter. High-variance policy gradients over a few thousand correlated samples will destroy features that took orders of magnitude more data to learn.
- Do not port anything to ONNX / TF-Lite / TensorRT yet.
- Do not add pedestrians, buy hardware, or optimise latency.
- Do not use the word **VLM** for what this builds. It is a **language-conditioned multimodal driving encoder**. The term becomes earned only when visual tokens are projected into a language model's embedding space and that language model does the reasoning. Fix the naming now; it costs nothing today and a great deal at a viva.

---

## 11 · Framing, for the write-up

Two things the implementing agent should know because they affect what gets logged and measured:

**The architecture is not the novelty.** Multi-view + LiDAR + query-based fusion + waypoint output + auxiliary pretraining + frozen encoder is almost exactly the InterFuser / LMDrive / TransFuser consensus. Building the well-validated stack at smaller scale and ablating it more carefully than the original papers did is a legitimate contribution — their perception ablations are thin and none controls for parameter count. But it is a methods contribution, not an architecture one.

**The exposure is where PPO sits.** Every strong CARLA RL result puts RL *upstream* on privileged state and imitation *downstream* on sensors — Roach and CaRL both. Nobody runs PPO downstream of a frozen learned sensor encoder as the primary driving policy. That paradigm is validated in manipulation and navigation, not driving. This is not a reason to abandon it; it is the reason to name it as the research question: *does the frozen-pretrained-representation paradigm transfer to closed-loop driving?* Framed that way it is interesting and this design is the right instrument. Framed as an obvious engineering choice, the first reviewer who knows Roach asks why you did not do it the other way.

**What is genuinely ours:** fixed latent width plus matched parameters as an explicit control for a perception-richness ablation; the blind config as a numeric leakage gate; the counterfactual instruction triad; and encoder-only distillation with a bit-identical frozen policy head. Log whatever is needed to support those four claims.

---

## 12 · Open items — flag, do not guess

- **CARLA version.** 0.9.15 recommended. This determines LiDAR semantics, route API and traffic-manager behaviour, and changing it later invalidates collected data. Confirm before Phase 2.
- **Privileged coach or autopilot** as the Stage 0 expert. Coach is better and is real work. Decide explicitly and record the choice.
- **Disk available** for the ~60 GB feature cache. Confirm before Phase 3.
- **Edge target device.** Jetson-class or Raspberry Pi. Constrains the research model's size now, not later. Note for the record: four cameras plus LiDAR plus language at 20 Hz on a Pi CPU is **not achievable**, and no amount of quantisation changes that.
- **Eight literature claims marked NEEDS CHECK** in the Ladder Precedent Audit are unverified — notably residual RL (Silver 2018 / Johannink 2019), which is the load-bearing citation for the residual action design, and the counter-evidence to Parisi on frozen representations. Do not cite any of them until opened.
