# Independent Review: Multimodal VLM-PPO Autonomous Driving Plan

## Bottom line
The project direction is plausible, but the current plan is not yet executable or defensible. The literature trail contains wrong citations, the new code has multiple runtime-breaking API mismatches, the edge-deployment path still targets the old TensorFlow baseline, and the experimental design does not yet isolate whether language conditioning or multimodal fusion actually helps.

## Phase 1 — What already exists

### Main research / execution documents
- `context/project_papers_and_execution_roadmap.md`
- `context/implementation_plan.md`
- `context/task_level_design_doc.md`
- `context/related_work_analysis.md`
- `context/codebase_analysis.md`
- `context/operational_guide.md`

### Existing implementation artifacts
- Baseline CARLA/PPO/VAE pipeline: `MTP_TESTING/main.py`
- New PyTorch multimodal encoder: `MTP_TESTING/multimodal_encoder.py`
- New VLM wrapper: `MTP_TESTING/main_vlm.py`
- New CARLA env / PPO training: `MTP_TESTING/main_vlm_train.py`
- New data collection: `MTP_TESTING/collect_data.py`
- New pretraining script: `MTP_TESTING/pretrain_encoder.py`
- New latency benchmark: `MTP_TESTING/benchmark_latency.py`
- New PIL edge client: `MTP_TESTING/PIL_edge_vlm.py`
- Old PIL baseline: `MTP_TESTING/PIL_simulation.py`
- Speech sidecar: `Aanchal_Spring/inference.py`, `Aanchal_Spring/speech_to_text.py`
- Baseline artifacts: `MTP_TESTING/Results_04`, `MTP_TESTING/Results_05`, `Aanchal_Spring/test_results/*`

## Phase 2 — Fresh literature verification

| Claim | Status | Evidence / correction |
|---|---|---|
| “LMDrive” is the cited 5-camera + LiDAR + Q-Former + LLaMA-7B system | **Unverified / likely wrong citation** | The cited arXiv ID `2406.10165` is **CarLLaVA: Vision language models for camera-only closed-loop driving**, not LMDrive. It is camera-only and uses LLaVA/LLaMA backbone. |
| “DriveLM” is a real language-driving paper | **Verified, but mischaracterized** | `2312.14150` is **DriveLM: Driving with Graph Visual Question Answering**. It is a graph-VQA / reasoning benchmark, not a direct lightweight edge control model. |
| “DriveMLM” plugs MLLMs into behavior planning and Apollo | **Verified** | The repo/abstract describe decision-state standardization, MLLM behavior planning, and Apollo plug-in use. |
| “InternVL2” is the cited 2024 backbone paper | **Incorrect citation** | `2404.16821` is **InternVL 1.5**, not InternVL2. The repo also shows larger models and newer lightweight variants such as `InternViT-300M`. |
| “CaRL” / PPO scaling claim from `2306.09345` | **Wrong citation** | `2306.09345` is **Evaluating Data Attribution for Text-to-Image Models**, unrelated to driving. |
| “Phi-3.5-vision” is an efficient multimodal model for constrained environments | **Verified** | Microsoft’s model card explicitly lists memory/compute constrained and latency-bound use cases. |
| “MobileCLIP” is smaller/faster than CLIP and aimed at mobile use | **Verified** | Apple’s repo states MobileCLIP S0/S2 variants are smaller/faster, with mobile/iOS inference support. |

### Literature gap that should change the plan
The plan currently frames the work as if the main choice is “full VLM vs tiny custom FiLM encoder.” Recent open alternatives already give you better off-the-shelf lightweight backbones:
- MobileCLIP (`apple/ml-mobileclip`)
- Phi-3.5-vision
- InternViT-300M / Mini-InternVL variants

That means the current “from scratch tiny transformer” is not obviously the best first implementation choice.

## Phase 3 — Architecture red-team

### What the code actually does
- `multimodal_encoder.py:148-284` implements a real lightweight PyTorch encoder with:
  - 3 RGB views
  - optional 2D BEV LiDAR
  - discrete command ID FiLM conditioning
  - 2 transformer blocks
  - 128-d latent + 5 telemetry = 133-d state
- Parameter count is **133,065** (~0.508 MB), which is numerically plausible.

### Where the architecture sounds plausible but breaks

1. **`main_vlm.py` will not run as written.**
   - It calls `MultimodalEdgeEncoder(img_h=..., img_w=..., in_channels=..., latent_dim=LATENT_DIM, ...)` at `main_vlm.py:39-46`, but the class only accepts `embed_dim, latent_dim, nav_dim, num_cameras, use_lidar` (`multimodal_encoder.py:148-154`).
   - It also calls `self.model(..., command_id=command)` at `main_vlm.py:61-65`, but the encoder expects `command=...`, not `command_id`.
   - `LATENT_DIM` does not exist in `parameters.py`; only `VLM_LATENT_DIM` does (`parameters.py:17-22`).

2. **`PIL_edge_vlm.py` will not run as written.**
   - It imports `PyTorchActorCritic` from `main_vlm`, but instantiates `PPOActorCritic` (`PIL_edge_vlm.py:20,46`).
   - It also passes `command_id=command_intent` into the encoder (`PIL_edge_vlm.py:97`), which the encoder does not accept.

3. **The sim/edge socket protocol still mismatches the new edge client.**
   - Old PIL sim sends `struct.pack("3I", *image_shape) + image_bytes + info_bytes` (`PIL_simulation.py:637,687`).
   - New edge client expects a merged multi-sensor tensor with 10 channels and a 12-byte header (`PIL_edge_vlm.py:55-98`).
   - This is a protocol mismatch, not a cosmetic difference.

4. **Brake is still not implemented.**
   - The new stack keeps `ACTION_DIM = 2` (`parameters.py:20-22`).
   - The policy applies `brake=0.0` always (`main_vlm_train.py:384-385`).
   - Yet the design docs ask for steer + throttle + brake. That is inconsistent.

5. **The “speech” path is not actually integrated into driving.**
   - `Aanchal_Spring` is a standalone Wav2Vec2 + Qwen2 audio-to-text/intent pipeline (`Aanchal_Spring/speech_to_text.py:23-30,128-238`; `Aanchal_Spring/inference.py:22-30,87-130`).
   - Driving code consumes a command token/string directly; there is no async speech bus, no timestamp alignment, and no end-to-end audio-to-control path.

## Phase 4 — Implementation-plan audit

### Main blockers

| Blocker | Why it matters | Evidence | Fix |
|---|---|---|---|
| Broken constructor/API calls | Current “new” scripts fail before inference | `main_vlm.py:39-65`, `PIL_edge_vlm.py:36-47,97` | Make one canonical encoder API and update all wrappers |
| Old TensorFlow conversion path | No edge export for the new PyTorch model | `convert.py:25-67` still converts baseline TF actor/VAE only | Replace with ONNX/TorchScript/quantization for the new model |
| No brake action | Safety and spec mismatch | `parameters.py:21-22`, `main_vlm_train.py:384-385` | Either add brake now or explicitly remove brake from the spec |
| Speech not wired in | Central thesis claim is unproven | `Aanchal_Spring/*` is standalone | Add a command bus, or stage speech as a later milestone |
| No realistic latency proof | Benchmark is synthetic only | `benchmark_latency.py:45-114` uses random tensors, not CARLA/Pi/quantized deployment | Measure end-to-end on the target device with real preprocessing |
| Weak eval signal | Can’t prove language conditioning | `main_vlm_train.py:509-559` uses random command per episode and success = collision-free + 200 steps | Add command-following, route completion, and ablations |

### Data pipeline problems
- `collect_data.py` records RGB + LiDAR + telemetry, but no real speech labels or IMU/GNSS sensor stream (`collect_data.py:76-240`).
- `pretrain_encoder.py` invents command labels from steer/speed heuristics (`pretrain_encoder.py:75-85`), so “language conditioning” is not supervised by actual language.
- The train/val split is a sequential slice (`pretrain_encoder.py:45-51`), which is weak for temporally correlated driving data.

## Phase 5 — Things the previous agent missed

### 🔴 Critical
1. **Wrong literature citations**  
   **Why it matters:** If the base references are wrong, the whole novelty argument is shaky.  
   **Evidence:** `2406.10165` = CarLLaVA; `2312.14150` = DriveLM; `2404.16821` = InternVL 1.5; `2306.09345` = text-to-image attribution.  
   **Fix:** Rebuild the literature section from verified titles/IDs.

2. **Broken runtime API surface**  
   **Why it matters:** The code path fails before a single CARLA step.  
   **Evidence:** `main_vlm.py:39-65`, `PIL_edge_vlm.py:36-47,97`.  
   **Fix:** Standardize constructor names and call signatures.

3. **Edge-deployment story is incomplete**  
   **Why it matters:** The new model is PyTorch, but the export path is still TF baseline-only.  
   **Evidence:** `convert.py:25-67`.  
   **Fix:** Add TorchScript/ONNX/quantization for the new encoder/policy.

4. **Brake is missing**  
   **Why it matters:** The thesis spec says steer/throttle/brake; the code cannot do that.  
   **Evidence:** `parameters.py:21-22`, `main_vlm_train.py:384-385`.  
   **Fix:** Either add brake or narrow the claim.

### 🟠 Important
5. **Speech is not actually part of the driving loop**  
   **Why it matters:** This is one of the core thesis claims.  
   **Evidence:** `Aanchal_Spring/*` is separate; driving code uses command IDs.  
   **Fix:** Add an async speech-intent service or downgrade speech to an auxiliary demo.

6. **Metrics are too weak**  
   **Why it matters:** A model can look good without proving the intended contribution.  
   **Evidence:** `main_vlm_train.py:539-559`.  
   **Fix:** Use route completion, infractions, command adherence, and variance across seeds.

7. **Pretraining labels are heuristic**  
   **Why it matters:** Steering/speed are not language supervision.  
   **Evidence:** `pretrain_encoder.py:79-85`.  
   **Fix:** Treat this as auxiliary representation learning, not command grounding.

8. **Sequential train/val split is weak**  
   **Why it matters:** Temporal correlation makes the validation set non-independent.  
   **Evidence:** `pretrain_encoder.py:45-51`.  
   **Fix:** Split by route/episode/weather, not raw row order.

### 🟡 Minor
9. **“Zero extra parameters” and “full 3D metric awareness” are overclaims**  
   **Why it matters:** It reads like a theorem but is only an intuition.  
   **Evidence:** `multimodal_encoder.py:156-191` only does 2D BEV projection plus shared tokenization.  
   **Fix:** Rephrase as an efficiency tradeoff, not a guarantee.

10. **`main_vlm_train.py` says GAE but implements Monte Carlo returns**  
    **Why it matters:** The algorithm description is inaccurate.  
    **Evidence:** `main_vlm_train.py:161-173`.  
    **Fix:** Either implement true GAE or rename the comment/docs.

## Phase 6 — Scope creep audit

| Component | Verdict | Reason |
|---|---|---|
| 3 RGB cameras | **MUST HAVE** | Core perception input |
| 2D BEV LiDAR | **MUST HAVE** | Central sensor fusion claim |
| Telemetry (5-dim) | **MUST HAVE** | Needed for control state |
| PPO actor-critic | **MUST HAVE** | Core learning method |
| Town01 → Town02 split | **MUST HAVE** | Core generalization claim |
| Speech command conditioning | **SHOULD HAVE** | Important, but currently not wired end-to-end |
| Brake action | **MUST HAVE if claimed** | Safety/spec consistency |
| PIL edge deployment | **SHOULD HAVE** | Important, but only after the core policy works |
| Quantization | **SHOULD HAVE** | Needed for edge, but after runtime correctness |
| Full multimodal LLM backbone | **REMOVE** | Too heavy and not aligned with current codebase |
| “Statistically prove SOTA” language | **REMOVE** | Too strong without proper baselines and runs |

### Minimum viable research system
1. RGB + LiDAR BEV + telemetry encoder
2. PPO policy with consistent action space
3. Town01 training, Town02 evaluation
4. A single verified command-conditioning path
5. A real edge export path

## Phase 7 — Corrected roadmap

| Existing plan | Problem | Recommended change | Reason |
|---|---|---|---|
| “Use the cited 6 core papers” | Several citations are wrong | Rebuild the literature map from verified paper IDs/titles | Prevents a false novelty story |
| “Implement `main_vlm.py` / `PIL_edge_vlm.py` as drop-in” | API mismatches and undefined symbols | Unify all call signatures around `MultimodalEdgeEncoder(front,left,right,lidar,command,telemetry)` | Makes the code runnable |
| “Keep TensorFlow export pipeline” | New model is PyTorch | Replace with TorchScript/ONNX + quantization | Makes edge deployment real |
| “Train with random command sampling” | Doesn’t prove language grounding | Add real command source or command-ablation experiment | Isolates contribution |
| “Evaluate on success rate only” | Weak proof | Use route completion, infractions, collisions, command adherence, and seeds | Actually validates the claim |

### Corrected execution order
1. **Lock the claim set**  
   Decide whether the project really claims brake support and real speech conditioning.

2. **Repair the code surface**  
   Fix constructor names, undefined symbols, and sim/edge packet protocol.

3. **Choose one deployment stack**  
   Prefer PyTorch end-to-end; drop the old TF conversion path unless you rewrite it.

4. **Build the real data pipeline**  
   Collect multi-view RGB + LiDAR + telemetry with proper splits and metadata.

5. **Pretrain the encoder**  
   Keep it auxiliary-only unless you can prove joint finetuning is stable.

6. **Train PPO**  
   Use a consistent action space, meaningful reward, and command ablations.

7. **Evaluate properly**  
   Town02, multiple seeds, route completion, command adherence, failure analysis.

8. **Only then add edge packaging**  
   Quantize/export the final model and validate on target hardware.

## Phase 8 — Experimental design audit

### Can the current experiments prove the thesis?
Not yet.

### Why not
- The model can ignore the command input and still do okay on lane keeping.
- The current metrics do not separate “drives well” from “understands language.”
- The train/val split is weak for temporally correlated driving data.
- There are no ablations for:
  - no command vs correct command
  - RGB-only vs RGB+LiDAR
  - with vs without auxiliary pretraining
  - with vs without brake
  - quantized vs non-quantized edge inference

### Single most important experiment
**Command-conditioning ablation on the same Town02 routes**

Run the same trained policy under:
1. correct command
2. shuffled/random command
3. no command

Measure:
- route completion
- collisions/infractions
- command adherence
- variance across seeds

If this does not move the metrics, the language-conditioning claim is not supported.

### Could the project get a better result without validating the claimed contribution?
Yes. A policy can improve just by:
- better RGB/LiDAR fusion
- stronger pretraining
- reward tuning
- route memorization

That is why the ablations above are mandatory.

## Phase 9 — Final assessment

### Scores
- Research quality: **4/10**
- Technical correctness: **3/10**
- Architecture: **4/10**
- Implementation feasibility: **3/10**
- Experimental rigor: **2/10**
- Novelty: **6/10**
- Reproducibility: **3/10**
- Overall readiness: **3/10**

### Direct answers
1. **What did the previous agent do well?**  
   It identified a coherent target direction: lightweight multimodal driving with Town01→Town02 generalization and edge deployment.

2. **Biggest technical flaw?**  
   The new code does not run as written because of API/name mismatches and mismatched deployment tooling.

3. **Biggest research gap?**  
   The literature section uses wrong citations and overstates what the cited papers actually show.

4. **Biggest implementation risk?**  
   Edge deployment for the new PyTorch model is not wired up; the current conversion path is still for the old TensorFlow baseline.

5. **Most important thing it missed?**  
   A proper experiment that proves language conditioning matters.

6. **What should be changed before implementation begins?**  
   Fix the claims, references, runtime APIs, action space, and evaluation protocol.

7. **What should I implement first?**  
   A runnable PyTorch end-to-end baseline with one canonical encoder API and a matching sim/edge protocol.

8. **What could invalidate the entire approach?**  
   If command-conditioning ablations show no measurable effect, the language/speech contribution collapses.

9. **If I owned the project, what would I do differently?**  
   I would first prove a strong RGB+LiDAR+telemetry PPO baseline, then add speech as an isolated conditioning channel, then only after that claim edge-ready multimodal autonomy.
