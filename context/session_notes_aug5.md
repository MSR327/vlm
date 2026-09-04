# 📝 BTP Session Notes — Aug 5, 2026

## Session Summary

Two major things happened today:
1. **Professor meeting debrief** — clarified the exact project scope
2. **Deep codebase analysis** — tore apart the existing MTP_TESTING code and identified 10 critical flaws

---

## §1. What the Professor Actually Wants (Meeting Aug 5)

### The Existing System (already built by previous student)
```
1 Semantic Segmentation Camera → VAE Encoder → PPO → [steer, throttle]
```

### Your Job (modify it to)
```
Multiple Sensors (cameras, LiDAR, GPS, IMU)  ─┐
                                               ├→ Multimodal LLM → PPO → [steer, throttle, brake]
User Voice (audio, parallel track)            ─┘
```

### Key Requirements from Professor
| Requirement | Detail |
|---|---|
| **Multi-sensor input** | Not just 1 camera — multiple sensors of different data types |
| **Multimodal LLM** | Replace VAE with an LLM that can process images + text + sensor data |
| **Voice commands** | User's verbal input processed on a parallel async track |
| **Edge deployment** | Must be lightweight enough to run on edge hardware (not a cloud GPU) |
| **Read papers first** | Come up with your own critical inputs before coding |

### Professor Also Provided
- The existing codebase: `/Users/msr/claude_code/BTP_Project/MTP_TESTING/`
- 5 research papers (AM-PPO, HEPPO, Thesis VAE+PPO, Learning to Drive in a Day, Parallel DRL Survey)

---

## §2. Codebase Analysis — What Exists

> [!NOTE]
> Full detailed analysis in [codebase_analysis.md](file:///Users/msr/.gemini/antigravity/brain/3c50afdf-4044-46b1-871f-8bbdba2217a7/codebase_analysis.md)

### Architecture Map
```
CARLA Simulator (Town02)
  └─ Semantic Segmentation Camera (160×80, FOV 125°)
       └─ VAE Encoder (Conv2D layers → z ∈ ℝ^95)
            └─ concat with nav_obs [throttle, velocity, norm_vel, norm_dist, norm_angle]
                 └─ state ∈ ℝ^100
                      └─ PPO Actor (MLP: 500→300→100→2, tanh) → [steer, throttle]
                      └─ PPO Critic (MLP: 500→300→100→1) → V(s)
```

### Key Files
| File | Role |
|---|---|
| [main.py](file:///Users/msr/claude_code/BTP_Project/MTP_TESTING/main.py) | Everything — CARLA env, VAE, PPO, train/test loops (1427 lines) |
| [parameters.py](file:///Users/msr/claude_code/BTP_Project/MTP_TESTING/parameters.py) | All hyperparameters and config |
| [PIL_edge.py](file:///Users/msr/claude_code/BTP_Project/MTP_TESTING/PIL_edge.py) | Edge device inference (VAE + Actor over TCP socket, CPU-only) |
| [PIL_simulation.py](file:///Users/msr/claude_code/BTP_Project/MTP_TESTING/PIL_simulation.py) | CARLA sim side — sends frames, receives actions via socket |
| [convert.py](file:///Users/msr/claude_code/BTP_Project/MTP_TESTING/convert.py) | TFLite conversion (FP16 + INT8) for edge deployment |
| [main_cpu_dont_touch.py](file:///Users/msr/claude_code/BTP_Project/MTP_TESTING/main_cpu_dont_touch.py) | Working backup — DO NOT MODIFY |

### Edge Deployment (PIL = Processor-In-the-Loop)
```
CARLA Server ──[TCP socket: image + nav_obs]──→ Edge Device (CPU)
                                                  ├─ VAE Encoder inference
                                                  ├─ Actor inference
CARLA Server ←──[TCP socket: steer, throttle]──── └─ Sends action back
```

---

## §3. 10 Critical Flaws Found in Existing Work

### 🔴 Severe Flaws

| # | Flaw | Why It Matters |
|---|---|---|
| 1 | **Semantic segmentation camera = cheating** | Ground-truth pixel labels don't exist in real cars. The VAE just compresses already-solved images. Zero transfer to real world. |
| 2 | **Only 1 sensor = massive blind spots** | No rear view, no side view, no depth. This IS the uMDP partial observability problem. |
| 3 | **No braking action** (`ACTION_DIM=2`) | Car can only steer + accelerate. Code cheats by [turning red lights green](file:///Users/msr/claude_code/BTP_Project/MTP_TESTING/main.py#L262-L265) to avoid needing to stop. |
| 4 | **Reward function issues** | Collision penalty (-10) is too small vs cumulative driving reward (~100+). No smoothness penalty for jerky steering. Hardcoded speed target (22 km/h). |
| 5 | **Fake "normalization"** | [L701-704](file:///Users/msr/claude_code/BTP_Project/MTP_TESTING/main.py#L701-L704): clips at ±100 million instead of actual zero-mean unit-variance normalization. Different feature scales slow learning. |

### 🟡 Design Concerns

| # | Flaw | Why It Matters |
|---|---|---|
| 6 | **VAE latent dim = 95 — arbitrary?** | 404× compression ratio. No ablation showing 95 is optimal. Could be losing critical scene info. |
| 7 | **PPO updates every 5 episodes** | Non-standard. Buffer grows unbounded. Stale early-episode data contaminates updates. |
| 8 | **TensorFlow in 2025/26** | Research ecosystem moved to PyTorch. All new VLMs (InternVL, SigLIP, CLIP) are PyTorch-native. |
| 9 | **PIL is synchronous blocking** | CARLA freezes while waiting for edge response. Doesn't test real-world timing where the car keeps moving during inference. |
| 10 | **No generalization testing** | Only Town02, only CloudyNoon weather. Likely overfits to one town layout. |

---

## §4. Key Architectural Insight — Don't Use Full LLM

> [!IMPORTANT]
> A full Multimodal LLM (e.g., LLaVA-7B) takes ~800ms per inference. At 60 km/h, the car moves 13 meters blind. **Not viable for edge deployment.**

### The Solution: Use Only the Vision Encoder

```
A Multimodal LLM has 2 parts:

┌──────────────┐    ┌───────────────────┐
│ Vision       │    │ Language Model    │
│ Encoder      │    │ (the heavy part)  │
│ ~300M params │    │ ~1-7B params 🐘   │
│ ~10ms ⚡      │    │ ~500ms+ 🐢        │
└──────────────┘    └───────────────────┘

Use THIS only ──────────── Skip THIS
```

### Lightweight Edge-Compatible Models

| Model | Size | Speed on T4 | Edge-deployable? |
|---|---|---|---|
| SigLIP-B | ~400M | ~8ms | ✅ Yes |
| CLIP ViT-B/32 | ~150M | ~5ms | ✅ Yes |
| MobileCLIP | ~60M | ~3ms | ✅ Yes (designed for mobile) |
| LLaVA-7B | 7B | ~800ms | ❌ No |

---

## §5. What to Present to Professor

### Your Critical Analysis
1. Semantic segmentation camera is unrealistic — switching to RGB with a learned vision encoder is necessary for real-world transfer
2. Single camera creates the exact partial observability (uMDP) problem — more sensors directly address this
3. No braking action is a fundamental limitation — need 3D action space [steer, throttle, brake]
4. Full LLM is too heavy for edge — propose using only the vision encoder part
5. Voice commands on parallel async track so driving never pauses

### Your Proposed Architecture
```
RGB Cameras ──→ Lightweight Vision Encoder (SigLIP/MobileCLIP) ──→ z_visual
LiDAR/GPS/IMU ──→ Normalize + project ──→ z_sensor                  │
User Voice ──→ Whisper STT → Text Encoder ──→ z_text (async)        │
                                                                     ▼
                                                              Gated Fusion
                                                                     │
                                                                     ▼
                                                              PPO (AM-PPO)
                                                                     │
                                                                     ▼
                                                        [steer, throttle, brake]
```

---

## §6. Infrastructure Status

| Platform | Status | Use For |
|---|---|---|
| **Google Colab** | ✅ Ready (free T4 GPU) | Running CARLA + training |
| **Your Mac** | ✅ Ready | Code editing, text encoder dev, paper reading |
| **Azure for Students** | ⏳ Verification failed, Q&A post submitted | GPU VM (backup option) |
| **GCP** | ⏳ $300 free trial pending small verification payment | GPU VM (backup option) |
| **Lab server** | ❓ Need to ask professor | Best option if available |

---

## §7. Open Items for Next Session

- [ ] Present critical analysis + proposal to professor (Aug 6)
- [ ] Get professor's feedback on: vision encoder choice, braking action, TF vs PyTorch
- [ ] Ask for chatbot dialogue dataset
- [ ] Ask about lab GPU server access
- [ ] Start reading DriveMLM §3 and SimLingo §3 (the method sections)
- [ ] After professor feedback → update implementation plan and start coding
