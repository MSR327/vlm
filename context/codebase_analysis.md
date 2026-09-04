# 🔬 Codebase Analysis: MTP_TESTING

## What Has Already Been Built

The existing code is a **complete, working pipeline** for autonomous driving in CARLA using VAE + PPO, **including edge deployment**. This is more complete than I expected. Here's the full picture:

---

## The Current Pipeline (What Exists)

```
┌──────────────────────────────────────────────────────────────────────┐
│                    EXISTING PIPELINE                                 │
│                                                                      │
│  CARLA Simulator                                                     │
│  ┌─────────────────────┐                                            │
│  │ Semantic Segmentation│   (NOT RGB! They use segmentation masks)   │
│  │ Camera (160×80)      │──────────────────────┐                    │
│  │ FOV: 125°            │                      │                    │
│  └─────────────────────┘                      ▼                    │
│                                         ┌──────────┐               │
│  Navigation Obs (5 floats):             │   VAE    │               │
│  [throttle, velocity,                   │ Encoder  │               │
│   norm_velocity,                        │          │               │
│   norm_dist_center,                     │ Conv2D   │               │
│   norm_angle]                           │ layers   │               │
│       │                                 │    ↓     │               │
│       │                                 │ z ∈ ℝ^95 │               │
│       │                                 └────┬─────┘               │
│       │                                      │                     │
│       └──────────────┬───────────────────────┘                     │
│                      │ concat                                      │
│                      ▼                                             │
│               state ∈ ℝ^100                                        │
│               (95 VAE + 5 nav)                                     │
│                      │                                             │
│                      ▼                                             │
│               ┌──────────┐                                         │
│               │   PPO    │                                         │
│               │ Actor:   │ MLP: 500 → 300 → 100 → 2 (tanh)       │
│               │ Critic:  │ MLP: 500 → 300 → 100 → 1               │
│               └────┬─────┘                                         │
│                    │                                               │
│                    ▼                                               │
│            [steer, throttle]                                        │
│            (continuous, ∈ [-1, 1])                                  │
└──────────────────────────────────────────────────────────────────────┘
```

---

## File-by-File Breakdown

### Core Files

| File | Purpose | Key Details |
|---|---|---|
| [main.py](file:///Users/msr/claude_code/BTP_Project/MTP_TESTING/main.py) | **Main training + testing** | Contains EVERYTHING: CARLA env, VAE encoder, PPO agent, training loop, testing loop, data capture. 1427 lines. |
| [parameters.py](file:///Users/msr/claude_code/BTP_Project/MTP_TESTING/parameters.py) | **Hyperparameters** | All config in one place: image dims, latent dim, PPO params, CARLA settings, network IPs |

### Edge Deployment Files (PIL = "Processing In the Loop")

| File | Purpose | Key Details |
|---|---|---|
| [PIL_simulation.py](file:///Users/msr/claude_code/BTP_Project/MTP_TESTING/PIL_simulation.py) | **Simulation side** of edge deployment | Runs CARLA, captures frames, sends sensor data over TCP socket to edge device, receives actions back |
| [PIL_edge.py](file:///Users/msr/claude_code/BTP_Project/MTP_TESTING/PIL_edge.py) | **Edge device side** | Receives sensor data via socket, runs VAE+Actor inference on CPU (no GPU!), sends actions back |
| [server.py](file:///Users/msr/claude_code/BTP_Project/MTP_TESTING/server.py) | Simple TCP echo server | For testing socket connection between sim and edge |

### Utility Files

| File | Purpose | Key Details |
|---|---|---|
| [convert.py](file:///Users/msr/claude_code/BTP_Project/MTP_TESTING/convert.py) | **TFLite conversion** | Converts VAE + Actor to TFLite (FP16 and INT8) for edge deployment |
| [accuracy_check.py](file:///Users/msr/claude_code/BTP_Project/MTP_TESTING/accuracy_check.py) | TFLite accuracy validation | Compares TFLite model outputs vs full TensorFlow model outputs |
| [main_capture.py](file:///Users/msr/claude_code/BTP_Project/MTP_TESTING/main_capture.py) | Data capture variant | Runs agent and saves every frame + action as CSV for analysis |
| [mian_test.py](file:///Users/msr/claude_code/BTP_Project/MTP_TESTING/mian_test.py) | Testing variant | (typo in filename) Another test script variant |
| [test.py](file:///Users/msr/claude_code/BTP_Project/MTP_TESTING/test.py) | Full test pipeline | Complete test with TFLite inference |
| [main_cpu_dont_touch.py](file:///Users/msr/claude_code/BTP_Project/MTP_TESTING/main_cpu_dont_touch.py) | **Working backup** | CPU-only version that works — DO NOT MODIFY |
| [tf_check.py](file:///Users/msr/claude_code/BTP_Project/MTP_TESTING/tf_check.py) | TF version check | Prints TF version and GPU availability |

---

## Key Architecture Details I Discovered

### 1. Camera Uses SEMANTIC SEGMENTATION (Not RGB!)
```python
# Line 523 in main.py
self.sensor_name = 'sensor.camera.semantic_segmentation'
```
> [!IMPORTANT]
> The existing system uses **semantic segmentation** camera, NOT a raw RGB camera! CARLA's semantic segmentation provides pixel-level class labels (road, vehicle, pedestrian, etc.) rendered as colored masks. This is a **huge simplification** because the VAE doesn't need to learn object detection — it just compresses an already-labeled image.
>
> **Your multimodal LLM approach will use raw RGB** instead, which means the LLM vision encoder needs to do the understanding that segmentation was doing for free.

### 2. VAE Architecture (from PIL_edge.py — the Encoder class)
```
Input: (160, 80, 3) semantic segmentation image
  → Conv2D(32, 4×4, stride=2) → ReLU
  → Conv2D(64, 3×3, stride=2) → ReLU → BatchNorm
  → Conv2D(128, 4×4, stride=2) → ReLU
  → Conv2D(256, 3×3, stride=2) → ReLU
  → Flatten → Dense(1024) → ReLU
  → μ = Dense(95), σ = Dense(95)
  → z = μ + σ * N(0,1)    [reparameterization trick]
Output: z ∈ ℝ^95 (latent vector)
```

### 3. State Vector Construction
```python
# Line 642 in main.py (EncodeState.process)
observation = tf.concat([tf.reshape(image_obs, [-1]), navigation_obs], axis=-1)
# Result: [z_95 | nav_5] = 100-dimensional state vector
```

### 4. Edge Deployment Architecture (PIL)
```
┌─────────────────┐        TCP Socket        ┌──────────────────┐
│  CARLA Server    │ ───── image + nav ────→  │  Edge Device     │
│  (GPU machine)   │                          │  (Raspberry Pi   │
│                  │ ←──── [steer, throt] ──  │   or similar)    │
│  PIL_simulation  │                          │  PIL_edge.py     │
│  .py             │                          │  (CPU-only!)     │
└─────────────────┘                          └──────────────────┘
```
> [!NOTE]
> The PIL (Processor-In-the-Loop) architecture already does what your professor wants! The edge device runs **only** the VAE encoder + Actor inference on CPU. Your job is to replace the VAE with a lightweight vision encoder that can also run on the edge.

### 5. TFLite Conversion (Edge Optimization)
The existing `convert.py` already converts models to:
- **FP16** (float16 quantization)
- **INT8** (full integer quantization for maximum speed on edge)

This is critical for edge deployment and you'll need to do the same for your new models.

### 6. Reward Function
```python
# Negative rewards (episode ends):
collision         → -10, done=True
off-lane (>3m)    → -10, done=True
stuck (v<1 km/h)  → -10, done=True
too fast (>35km/h) → -10, done=True

# Positive reward (continuous):
reward = speed_factor × centering_factor × angle_factor
  where:
    centering = max(1.0 - dist_from_center / 3.0, 0.0)
    angle     = max(1.0 - |angle| / 20°, 0.0)
    speed     = velocity / target_speed (capped)
```

---

## What's MISSING — What YOU Need to Build

### ❌ 1. Multi-Sensor Input (Currently: only 1 segmentation camera)
**What exists**: Single front-facing semantic segmentation camera (160×80)
**What you need**: Multiple sensors — RGB cameras, LiDAR, GPS, IMU

### ❌ 2. LLM/VLM Vision Encoder (Currently: VAE)
**What exists**: Custom 4-layer Conv VAE → 95-dim latent
**What you need**: Replace with a lightweight vision encoder (MobileCLIP, SigLIP, or EfficientViT) that can:
- Process RGB images (not segmentation)
- Run fast enough for real-time edge inference
- Still fit in TFLite conversion pipeline

### ❌ 3. Audio/Voice Input Pipeline (Currently: none)
**What exists**: Nothing
**What you need**: Speech-to-Text → Text Encoder → feature vector (on parallel async track)

### ❌ 4. Multi-Modal Fusion Module (Currently: simple concat)
**What exists**: `tf.concat([z_vae, nav_obs])` — just concatenation
**What you need**: Gated fusion or cross-attention to merge visual + text + sensor features

### ❌ 5. uMDP / Belief State (Currently: standard MDP)
**What exists**: Single-frame observation, no memory
**What you need**: LSTM/GRU to process observation history for partial observability

### ✅ 6. PPO Agent — EXISTS (may need AM-PPO upgrade)
**What exists**: Full PPO with GAE, clipping, actor-critic, advantage normalization
**What you may upgrade to**: AM-PPO (advantage modulation from your professor's paper)

### ✅ 7. CARLA Environment Wrapper — EXISTS
**What exists**: Complete Gym-like wrapper with reset/step/reward
**You'll modify**: Add more sensors, change camera to RGB, update observation space

### ✅ 8. Edge Deployment (PIL) — EXISTS
**What exists**: Socket-based edge inference pipeline + TFLite conversion
**You'll modify**: Replace VAE+Actor with VisionEncoder+FusionModule+Actor

---

## Where Exactly to Make Changes (Code Locations)

| What to Change | File | Lines | What to Do |
|---|---|---|---|
| Camera type | [main.py](file:///Users/msr/claude_code/BTP_Project/MTP_TESTING/main.py#L523) | L523 | Change `semantic_segmentation` → `rgb` |
| Add more sensors | [main.py](file:///Users/msr/claude_code/BTP_Project/MTP_TESTING/main.py#L138-L153) | L138-153 | Add LiDAR, side cameras, GPS, IMU sensors |
| Replace VAE | [main.py](file:///Users/msr/claude_code/BTP_Project/MTP_TESTING/main.py#L622-L644) | L622-644 | Replace `EncodeState` class with VLM encoder |
| State vector | [main.py](file:///Users/msr/claude_code/BTP_Project/MTP_TESTING/main.py#L642) | L642 | Change concat to fusion module |
| Observation dim | [parameters.py](file:///Users/msr/claude_code/BTP_Project/MTP_TESTING/parameters.py#L9) | L9 | Update `OBSERVATION_DIM = 100` to new size |
| Latent dim | [parameters.py](file:///Users/msr/claude_code/BTP_Project/MTP_TESTING/parameters.py#L6) | L6 | Update `LATENT_DIM = 95` to VLM output dim |
| Edge encoder | [PIL_edge.py](file:///Users/msr/claude_code/BTP_Project/MTP_TESTING/PIL_edge.py#L21-L58) | L21-58 | Replace `Encoder` class with lightweight VLM |
| TFLite conversion | [convert.py](file:///Users/msr/claude_code/BTP_Project/MTP_TESTING/convert.py#L25-L67) | L25-67 | Add VLM + fusion module to conversion pipeline |
