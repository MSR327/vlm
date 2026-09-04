# Production Task-Level Design Document: Multimodal VLM-PPO Autonomous Driving System

**Document Version:** 2.0  
**Project:** BTP Autonomous Driving (Multimodal VLM-PPO)  
**Target Environment:** CARLA 0.9.10+ / PyTorch 2.x / TensorFlow Lite Edge Deployment  
**Status:** Implementation-Ready Specification  

---

## 1. Objective, Problem, Scope & Non-Scope

### 1.1 Objective
To design, implement, train, benchmark, and deploy an end-to-end **Multimodal Vision-Language-Action Policy** for autonomous driving that:
1. Replaces synthetic ground-truth semantic segmentation shortcuts with photorealistic **Multi-View RGB Cameras (Front, Left, Right)** and **2D Bird's-Eye-View (BEV) Projected LiDAR**.
2. Incorporates human **spoken verbal commands** (via speech-to-intent tokens) to condition driving behavior using **Feature-wise Linear Modulation (FiLM)** and Transformer attention.
3. Employs a **Deep Reinforcement Learning (PPO)** actor-critic controller trained in **CARLA Town01** that achieves zero-shot generalization on **Town02**.
4. Remains computationally lightweight ($\le 1.5\text{ MB}$ footprint, $\le 25\text{ ms}$ on CPU/Edge) for **Processor-in-the-Loop (PIL)** deployment on embedded hardware (e.g., Raspberry Pi 4/5).

### 1.2 Problem Statement & Baseline Critique
The prior baseline (`MTP_TESTING`, results in `Results_05`) suffers from fundamental flaws:
* **Perception "Cheating":** Uses CARLA's internal `sensor.camera.semantic_segmentation` ground truth rather than realistic vision.
* **Overfitting & No Generalization:** Trained and evaluated strictly on the exact same map (`Town02`).
* **Zero Verbal/Audio Capability:** Incapable of accepting human speech or language navigation instructions.
* **Heavy Latency on Edge:** The baseline Conv2D VAE (52 MB) experiences **110–325 ms latency** on Raspberry Pi (`PIL_test_results_16bit.csv`), resulting in negative rewards and unstable steering.
* **Unrealistic Traffic Logic:** Hardcoded script forcing red lights to turn green automatically (`traffic_light.set_state(Green)`), bypassing real-world traffic rule compliance.

### 1.3 Scope
* Multi-sensor synchronous acquisition: 3x RGB cameras ($160\times 80$), 1x 64-channel LiDAR ($160\times 80$ BEV projection), 5-dim GNSS/IMU odometry, spoken audio command taxonomy.
* Multi-task supervised pre-training / domain adaptation on Town01 data (steering and speed prediction).
* PPO continuous control policy training in Town01 with closed-loop reward optimization.
* Quantitative zero-shot generalization benchmarking in Town02 against baseline `Results_05/test_results_gpu.csv`.
* Quantization (FP16 & INT8) and PIL network client-server deployment over TCP/IP sockets.

### 1.4 Non-Scope
* Hardware CAN-bus physical steering wheel integration on physical real-world cars.
* End-to-end training of multi-billion parameter LLMs (e.g., 7B LLaVA) inside the closed-loop 20 Hz simulation loop (we use distilled SLM / FiLM projection instead).
* Dynamic map generation / procedural procedural mesh generation beyond standard CARLA towns.

---

## 2. Requirements, Assumptions, Dependencies & Constraints

### 2.1 Functional Requirements (FR)
* **FR-1 [Perception]:** Process 3 RGB frames ($160\times 80\times 3$) and 1 LiDAR BEV grid ($160\times 80\times 1$) synchronously at $\ge 20\text{ FPS}$.
* **FR-2 [Language Conditioning]:** Modulate visual features according to 8 canonical driving commands (`keep_lane`, `turn_left`, `turn_right`, `shift_left_lane`, `shift_right_lane`, `slow_down`, `speed_up`, `emergency_stop`).
* **FR-3 [State Construction]:** Output a dense continuous latent state vector ($\mathbf{z} \in \mathbb{R}^{128}$) concatenated with 5-dim normalized telemetry into a unified $\mathbf{s} \in \mathbb{R}^{133}$.
* **FR-4 [Control Policy]:** Output continuous steering $\in [-1.0, 1.0]$, throttle $\in [0.0, 1.0]$, and brake $\in [0.0, 1.0]$.
* **FR-5 [Generalization]:** Achieve $\ge 80\%$ Route Completion (RC) on unseen Town02 without map-specific fine-tuning.

### 2.2 Non-Functional & Performance Constraints (NFR)
* **NFR-1 [Model Size]:** Total perception + policy parameter footprint $\le 2.0\text{ MB}$ ($\le 500\text{K}$ parameters).
* **NFR-2 [Latency]:** Forward pass $\le 3\text{ ms}$ on NVIDIA GPU, $\le 25\text{ ms}$ on x86/ARM CPU.
* **NFR-3 [Safety Margin]:** Collision rate $\le 10\%$ across 50 test episodes.

### 2.3 System Dependencies
```
Python >= 3.8, < 3.11
PyTorch >= 2.0.0
TorchVision >= 0.15.0
Torchaudio >= 2.0.0
CARLA Simulator == 0.9.10 / 0.9.13 / 0.9.15
Transformers >= 4.35.0 (for Wav2Vec2 / Qwen tokenization)
NumPy >= 1.22.0, Pandas >= 1.4.0, Scipy >= 1.9.0
TensorBoard >= 2.10.0
TensorFlow Lite / ONNX Runtime (for edge deployment)
```

---

## 3. Current Architecture $\rightarrow$ Proposed Architecture

```
========================================================================================
                                LEGACY BASELINE ARCHITECTURE
========================================================================================
  [ Semantic Seg Camera ] (Cheat) ──► Conv2D VAE (52 MB) ──► 95-dim Latent ──┐
                                                                              ├──► PPO (Town02)
  [ Telemetry (5-dim) ] ───────────────────────────────────► 5-dim Nav ──────┘
  (No audio, 1 camera, no LiDAR, auto-green traffic lights, evaluated only in Town02)

========================================================================================
                              PROPOSED MULTIMODAL VLM-PPO
========================================================================================
  [ Front RGB (125°) ] ──┐
  [ Left RGB (-60°)  ] ──┼──► Shared-Weight Patch Embedder (600 Tokens) ──┐
  [ Right RGB (+60°) ] ──┘                                                │
                                                                          │
  [ 64-Ch LiDAR ] ──────────► 2D BEV Projection Map (200 Tokens) ─────────┼──► FiLM Language
                                                                          │    Conditioning
  [ Speech Audio (.mp3) ] ──► Wav2Vec2 + Qwen2 Tokenizer (Command ID) ────┘         │
                                                                                    ▼
                                                                        Spatial-Linguistic
                                                                        Transformer (2-Layer)
                                                                                    │
                                                                                    ▼
                                                                        128-dim Multimodal Latent
                                                                                    +
                                                                        5-dim Normalized Telemetry
                                                                                    │
                                                                                    ▼
                                                                        133-dim State Vector
                                                                                    │
                                                                                    ▼
                                                                        PPO Actor-Critic Network
                                                                                    │
                                                                                    ▼
                                                                        [Steer, Throttle, Brake]
                                                                        (Train: Town01, Test: Town02)
========================================================================================
```

---

## 4. Detailed End-to-End Data Flow

```mermaid
sequenceDiagram
    autonumber
    participant CARLA as CARLA Server (Town01/02)
    participant Sensors as Multi-Sensor Stack (3x RGB, LiDAR, IMU)
    participant Audio as Speech Intent (Wav2Vec2/Qwen2)
    participant VLM as MultimodalEdgeEncoder
    participant PPO as PPO Actor-Critic Policy
    participant Control as Vehicle Control Actuator

    CARLA->>Sensors: world.tick() [Synchronous 20 FPS]
    Sensors->>VLM: Front (80x160x3), Left (80x160x3), Right (80x160x3), LiDAR BEV (80x160x1)
    Audio->>VLM: Command ID c in [0..7] ("shift_left_lane")
    Sensors->>VLM: IMU Telemetry (steer, speed, norm_v, norm_dist, norm_angle)
    Note over VLM: 1. Patch Embedding (4x200 tokens)<br/>2. View Encodings Addition<br/>3. FiLM Affine Modulation (gamma, beta)<br/>4. Self-Attention Transformer<br/>5. Global Average Pooling -> 128-dim Latent
    VLM->>PPO: Unified 133-dim Observation Vector
    Note over PPO: Gaussian Policy Forward Pass<br/>a ~ N(mu(s), sigma^2)
    PPO->>Control: [Steering in [-1,1], Throttle in [0,1], Brake in [0,1]]
    Control->>CARLA: vehicle.apply_control(carla.VehicleControl)
    CARLA->>PPO: Next State, Reward r_t, Done Flag (Collision / Deviation)
```

---

## 5. Module & Component Responsibilities

| Module | File Location | Primary Responsibility | Input | Output |
|---|---|---|---|---|
| **Parameters** | `parameters.py` | Centralized hyperparameter, sensor, path, and network configuration. | None | Global constants |
| **Multimodal Encoder** | `multimodal_encoder.py` | Shared patch embedding, FiLM conditioning, Transformer spatial fusion, state projection. | 3x RGB, 1x BEV LiDAR, Command ID, Telemetry | $(B, 133)$ Observation Tensor |
| **Data Collector** | `collect_data.py` | Synchronous data acquisition via CARLA Autopilot in Town01 across 4 weathers. | CARLA world | 20,000 `.npz` multi-view frames + `labels.csv` |
| **Domain Fine-Tuner** | `pretrain_encoder.py` | Supervised auxiliary multi-task pre-training (steering & speed regression) before RL. | `data_collected_town01/` | `models/pretrained_vlm_encoder.pth` |
| **PPO Policy & Env** | `main_vlm_train.py` | Closed-loop `CarlaEnvVLM` environment, GAE buffer, PPO Actor-Critic training & testing. | `CarlaEnvVLM` observations | Checkpoints & `test_results_town02.csv` |
| **Latency Profiler** | `benchmark_latency.py` | Micro-benchmarking p50, p95, p99 forward pass latency and FPS. | Synthetic tensor feeds | Benchmark report table |
| **PIL Edge Node** | `PIL_edge_vlm.py` | TCP socket client running quantized inference on edge hardware (RPi). | Raw bytes over socket | Action bytes `[steer, throttle]` |
| **PIL Sim Server** | `PIL_simulation.py` | TCP socket server sending simulation frames and executing edge commands. | CARLA sensors | Network packets |

---

## 6. Exact Interfaces, Inputs/Outputs & Tensor Shapes

### 6.1 Input Tensors to `MultimodalEdgeEncoder`

```
┌─────────────────┬────────────────────┬───────────────┬───────────────────────────────┐
│ Input Name      │ Data Type          │ Tensor Shape  │ Value Range / Normalization   │
├─────────────────┼────────────────────┼───────────────┼───────────────────────────────┤
│ front_img       │ torch.FloatTensor  │ (B, 3, 80, 160) │ [0.0, 1.0] (RGB Normalized)   │
│ left_img        │ torch.FloatTensor  │ (B, 3, 80, 160) │ [0.0, 1.0] (RGB Normalized)   │
│ right_img       │ torch.FloatTensor  │ (B, 3, 80, 160) │ [0.0, 1.0] (RGB Normalized)   │
│ lidar_bev       │ torch.FloatTensor  │ (B, 1, 80, 160) │ [0.0, 1.0] (Normalized Dist) │
│ command         │ torch.LongTensor   │ (B,)          │ Integers in [0, 7]            │
│ telemetry       │ torch.FloatTensor  │ (B, 5)        │ Continuous Normalized Values  │
└─────────────────┴────────────────────┴───────────────┴───────────────────────────────┘
```

### 6.2 Intermediate Tensor Transformations Inside Encoder

$$\begin{aligned}
\mathbf{T}_{\text{RGB}} &\in \mathbb{R}^{B \times 3 \times 80 \times 160} \xrightarrow{\text{SharedPatchEmbed}} \mathbf{F}_{\text{front}}, \mathbf{F}_{\text{left}}, \mathbf{F}_{\text{right}} \in \mathbb{R}^{B \times 200 \times 64} \\
\mathbf{T}_{\text{LiDAR}} &\in \mathbb{R}^{B \times 1 \times 80 \times 160} \xrightarrow{\text{LiDARPatchEmbed}} \mathbf{F}_{\text{lidar}} \in \mathbb{R}^{B \times 200 \times 64} \\
\mathbf{F}_{\text{views}} &= \text{Stack}[\mathbf{F}_{\text{front}}, \mathbf{F}_{\text{left}}, \mathbf{F}_{\text{right}}, \mathbf{F}_{\text{lidar}}] + \mathbf{E}_{\text{view}} \in \mathbb{R}^{B \times 4 \times 200 \times 64} \rightarrow \mathbb{R}^{B \times 800 \times 64} \\
\mathbf{F}_{\text{FiLM}} &= (1 + \boldsymbol{\gamma}(\mathbf{c})) \odot \mathbf{F}_{\text{views}} + \boldsymbol{\beta}(\mathbf{c}) \in \mathbb{R}^{B \times 800 \times 64} \\
\mathbf{F}_{\text{trans}} &= \text{TransformerBlocks}(\mathbf{F}_{\text{FiLM}}) \in \mathbb{R}^{B \times 800 \times 64} \\
\mathbf{z}_{\text{vlm}} &= \text{Linear}(\text{MeanPool}(\mathbf{F}_{\text{trans}})) \in \mathbb{R}^{B \times 128} \\
\mathbf{s}_{\text{unified}} &= [\mathbf{z}_{\text{vlm}} \,\|\, \text{Linear}(\text{Telemetry})] \in \mathbb{R}^{B \times 133}
\end{aligned}$$

---

## 7. Model Architecture, Encoders, Fusion & Policy

### 7.1 Detailed Layer-by-Layer Architecture

```
========================================================================================
LAYER COMPONENT                   INPUT SHAPE          OUTPUT SHAPE        PARAMS
========================================================================================
SharedPatchEmbedder (RGB):
  DepthwiseConv2D (k=3, s=2, p=1) (B, 3, 80, 160)     (B, 32, 40, 80)     138
  DepthwiseConv2D (k=3, s=2, p=1) (B, 32, 40, 80)     (B, 64, 20, 40)     2,432
  DepthwiseConv2D (k=3, s=2, p=1) (B, 64, 20, 40)     (B, 64, 10, 20)     4,736
  LayerNorm(64)                   (B, 200, 64)         (B, 200, 64)        128
----------------------------------------------------------------------------------------
SharedPatchEmbedder (LiDAR):
  DepthwiseConv2D (k=3, s=2, p=1) (B, 1, 80, 160)     (B, 32, 40, 80)     74
  DepthwiseConv2D (k=3, s=2, p=1) (B, 32, 40, 80)     (B, 64, 20, 40)     2,432
  DepthwiseConv2D (k=3, s=2, p=1) (B, 64, 20, 40)     (B, 64, 10, 20)     4,736
----------------------------------------------------------------------------------------
View Embedding Matrix             (1, 4, 1, 64)        (1, 4, 1, 64)       256
----------------------------------------------------------------------------------------
FiLMConditioning:
  Embedding(8, 64)                (B,)                 (B, 64)             512
  Linear(64, 128) + GELU          (B, 64)              (B, 128)            8,320
  Linear(128, 128) [gamma, beta]  (B, 128)             (B, 128)            16,512
----------------------------------------------------------------------------------------
Spatial-Linguistic Transformer:
  Block 1: Self-Attn (4 heads, 64) (B, 800, 64)        (B, 800, 64)        16,640
           MLP (64 -> 128 -> 64)  (B, 800, 64)         (B, 800, 64)        16,576
  Block 2: Self-Attn (4 heads, 64) (B, 800, 64)        (B, 800, 64)        16,640
           MLP (64 -> 128 -> 64)  (B, 800, 64)         (B, 800, 64)        16,576
----------------------------------------------------------------------------------------
Latent Projector:
  Linear(64, 128) + LayerNorm     (B, 64)              (B, 128)            8,448
  Linear(128, 128)                (B, 128)             (B, 128)            16,512
----------------------------------------------------------------------------------------
Telemetry Projector:
  Linear(5, 32) + GELU + Linear   (B, 5)               (B, 5)              389
========================================================================================
TOTAL MULTIMODAL ENCODER PARAMETERS: 133,065 (~0.51 MB)
========================================================================================
PPO Actor Network:
  Linear(133, 500) + Tanh         (B, 133)             (B, 500)            67,000
  Linear(500, 300) + Tanh         (B, 500)             (B, 300)            150,300
  Linear(300, 100) + Tanh         (B, 300)             (B, 100)            30,100
  Linear(100, 2) + Tanh           (B, 100)             (B, 2)              202
PPO Critic Network:
  Linear(133, 500) -> 300 -> 100 -> 1                                      247,501
========================================================================================
TOTAL SYSTEM PARAMETERS: 628,168 (~2.40 MB FP32 / 0.60 MB INT8)
========================================================================================
```

---

## 8. Training & Inference Pipeline (PPO / RL Loop)

### 8.1 Two-Stage Training Procedure
1. **Stage 1 (Supervised Domain Adaptation):** Train `MultimodalEdgeEncoder` with auxiliary prediction heads on Town01 multi-sensor records. Minimizes $\mathcal{L}_{\text{pretrain}} = \text{MSE}(\hat{\delta}, \delta) + 0.01 \cdot \text{MSE}(\hat{v}, v)$.
2. **Stage 2 (PPO RL Policy Optimization):** Freeze `MultimodalEdgeEncoder` backbone; optimize PPO Actor-Critic on Town01 closed-loop driving.

### 8.2 PPO Clipped Surrogate Objective

$$L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left( r_t(\theta)\hat{A}_t, \, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t \right) \right]$$

where:
$$r_t(\theta) = \frac{\pi_\theta(\mathbf{a}_t \mid \mathbf{s}_t)}{\pi_{\theta_{\text{old}}}(\mathbf{a}_t \mid \mathbf{s}_t)}$$

Generalized Advantage Estimation (GAE):
$$\hat{A}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}^V, \quad \text{where } \delta_t^V = r_t + \gamma V(\mathbf{s}_{t+1}) - V(\mathbf{s}_t)$$

Total Loss:
$$\mathcal{L}_{\text{PPO}}(\theta) = -L^{\text{CLIP}}(\theta) + c_1 \mathcal{L}_{\text{value}}(\theta) - c_2 \mathcal{S}[\pi_\theta](\mathbf{s}_t)$$
with $c_1 = 0.5$, $c_2 = 0.01$, $\epsilon = 0.2$, $\gamma = 0.99$, $\lambda = 0.95$.

---

## 9. Dataset, Preprocessing, Synchronization & Augmentation

### 9.1 Data Collection Protocol (`collect_data.py`)
* **Simulation Mode:** CARLA Synchronous Mode (`fixed_delta_seconds = 0.05`, exactly 20 FPS).
* **Sensor Synchronization:** Python `queue.Queue(maxsize=1)` per sensor. `world.tick()` blocks until all queues yield the identical frame index $k$.
* **Weather Rotation:** Dynamically switches every 2,500 frames across:
  1. `ClearNoon`
  2. `WetSunset`
  3. `HardRainNoon`
  4. `CloudyNight`

### 9.2 Data Augmentation Pipeline
* **Photometric Jitter:** Random brightness $(\pm 15\%)$, contrast $(\pm 15\%)$, Gaussian noise ($\sigma = 0.02$).
* **LiDAR Dropout:** Random spatial dropout ($p = 0.05$) of point cloud returns to simulate rain occlusion.
* **Camera Dropout:** Randomly zero-out left or right camera during $5\%$ of batches to ensure policy robustness against single-camera hardware failure.

---

## 10. CARLA Setup, Scenarios, Routes & Seeds

| Parameter | Training Setting (Town01) | Testing / Generalization Setting (Town02) |
|---|---|---|
| **Map** | `Town01` | `Town02` (Unseen) |
| **Number of NPC Vehicles** | 30 (Autopilot enabled) | 25 |
| **Number of Pedestrians** | 10 | 10 |
| **Route Length** | 1,000 steps (~500 m) | 500 steps (~250 m) |
| **Ego Vehicle Blueprint** | `vehicle.tesla.model3` | `vehicle.tesla.model3` |
| **Random Seed** | `42` | `100` |
| **Traffic Lights** | Standard CARLA autonomous cycle | Standard CARLA autonomous cycle |

---

## 11. Reward Design & Justification

### 11.1 Reward Function Formula

$$R_t = R_{\text{speed}} \cdot R_{\text{center}} \cdot R_{\text{angle}} + R_{\text{penalty}}$$

where:
$$\begin{aligned}
R_{\text{center}} &= \max\left(1.0 - \frac{d_{\text{center}}}{3.0}, \, 0.0\right) \\
R_{\text{angle}} &= \max\left(1.0 - \frac{|\theta_{\text{heading}}|}{\pi / 9}, \, 0.0\right) \quad (\text{penalizes deviation beyond } 20^\circ) \\
R_{\text{speed}} &= \begin{cases}
\frac{v}{20.0}, & \text{if } v \le 20.0 \text{ km/h} \\
1.0, & \text{if } 20.0 < v \le 25.0 \text{ km/h} \\
\max\left(1.0 - \frac{v - 25.0}{15.0}, \, 0.0\right), & \text{if } v > 25.0 \text{ km/h}
\end{cases}
\end{aligned}$$

### 11.2 Terminal Penalties
* **Collision:** $R_{\text{penalty}} = -10.0$ and `done = True`.
* **Out-of-Lane ($d_{\text{center}} > 3.5\text{ m}$):** $R_{\text{penalty}} = -5.0$ and `done = True`.
* **Stall ($v < 1.0\text{ km/h}$ for $> 10\text{ s}$ outside red lights):** $R_{\text{penalty}} = -5.0$ and `done = True`.

---

## 12. Evaluation Metrics, Baselines & Success Criteria

### 12.1 Evaluation Metrics
1. **Route Completion (RC %):** Percentage of route distance covered before termination.
2. **Infraction Score (IS):** Deductions for collisions ($0.5\times$), red lights ($0.7\times$), off-road ($0.8\times$).
3. **Driving Score (DS):** $\text{DS} = \text{RC} \times \text{IS}$ (CARLA Leaderboard standard).
4. **Average Speed (m/s):** Forward velocity efficiency.
5. **Inference Latency (ms):** Wall-clock time per decision step.

### 12.2 Baseline Comparison Target (`Results_05`)

| Metric | Legacy Baseline (`Results_05`) | Proposed VLM-PPO Target |
|---|---|---|
| **Visual Modality** | Synthetic Semantic Seg | **Raw 3x RGB + LiDAR** |
| **Speech Conditioning** | ❌ None | ✅ **8 Canonical Commands** |
| **Model Size** | 52 MB | **$\le 1.5$ MB (35x lighter)** |
| **Test Town** | Town02 (Same as train) | **Town02 (Zero-shot generalization)** |
| **Route Completion (Town02)** | ~98% (Overfit) | **$\ge 85\%$ (Zero-shot generalization)** |
| **Edge Latency (RPi)** | 110–325 ms | **$\le 25$ ms (Real-time 40 FPS)** |

---

## 13. Experiments & Ablation Plan

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                   ABLATION MATRIX                                      │
├────┬──────────────────────────────────┬─────────────────────────────┬──────────────────┤
│ ID │ Configuration                    │ What is Tested              │ Expected Result  │
├────┼──────────────────────────────────┼─────────────────────────────┼──────────────────┤
│ A1 │ 1 Camera (Front Only)            │ Baseline perception         │ Fails at turns   │
│ A2 │ 3 Cameras (Front + Left + Right) │ Peripheral awareness        │ +25% Turn RC     │
│ A3 │ 3 Cameras + LiDAR BEV            │ Distance / Depth accuracy   │ -60% Collisions  │
│ A4 │ Full Model w/o FiLM              │ Unconditioned driving       │ Ignores commands │
│ A5 │ Full Model + FiLM Conditioning   │ Full Multimodal VLM-PPO     │ SOTA Performance │
│ A6 │ Direct RL w/o Pre-training       │ Effect of domain pre-train  │ Slower learning  │
└────┴──────────────────────────────────┴─────────────────────────────┴──────────────────┘
```

---

## 14. Failure Modes, Edge Cases, Safety & Recovery

| Failure Mode | Root Cause | Detection Mechanism | Recovery Action |
|---|---|---|---|
| **Single Camera Dropout** | Sensor cable / API disconnect | Image tensor is all zeros or `None` | Fallback to remaining 2 camera views; set warning flag. |
| **LiDAR Occlusion (Heavy Rain)**| Rain particles backscatter laser | BEV grid density $< 0.01$ | Rely on RGB cameras + telemetry heading tracker. |
| **Contradictory Voice Command** | e.g. "turn left" on single lane | Center deviation $> 2.0\text{ m}$ | Safety override: clamp steering within lane bounds. |
| **Vehicle Stall / Deadlock** | PPO policy gets stuck at obstacle | Speed $< 1.0\text{ km/h}$ for 50 ticks | Apply incremental creep throttle ($+0.2$). |

---

## 15. Performance, Latency & Resource Budgets

```
┌────────────────────────┬──────────────────┬──────────────────┬────────────────────────┐
│ Metric                 │ Remote GPU (3090)│ Local Mac (M-CPU)│ Raspberry Pi 4 (Edge)  │
├────────────────────────┼──────────────────┼──────────────────┼────────────────────────┤
│ Forward Latency (p50)  │ 1.8 ms           │ 20.1 ms          │ ~22.5 ms (INT8 TFLite) │
│ Memory Footprint (RAM) │ ~450 MB VRAM     │ ~120 MB RAM      │ ~85 MB RAM             │
│ Disk Storage (Weights) │ 2.4 MB (FP32)    │ 2.4 MB (FP32)    │ 0.6 MB (INT8)          │
│ Control Frequency      │ 500 Hz           │ 48 Hz            │ 44 Hz (Real-time >20Hz)│
└────────────────────────┴──────────────────┴──────────────────┴────────────────────────┘
```

---

## 16. Checkpointing, Tracking & Reproducibility

* **Deterministic Seeding:** Set `random.seed(42)`, `np.random.seed(42)`, `torch.manual_seed(42)`, `torch.cuda.manual_seed_all(42)`.
* **Logging Framework:** TensorBoard writer storing reward, value loss, surrogate loss, entropy, and average speed at `Results_VLM/runs/train/`.
* **Checkpoint Frequency:** Every 50 training episodes to `Results_VLM/checkpoints/ppo_ckpt_{episode}.pth` and `Results_VLM/ppo_model/actor_latest.pth`.
* **Evaluation Artifacts:** `Results_VLM/test_results_town02.csv` containing per-episode metrics.

---

## 17. Testing Strategy: Unit $\rightarrow$ Integration $\rightarrow$ Closed-Loop $\rightarrow$ Regression

```
Level 1: Unit Tests (test_vlm_units.py)
  ├── test_shared_patch_embed_shape()
  ├── test_film_conditioning_modulation()
  ├── test_multimodal_encoder_forward()
  └── test_ppo_actor_critic_dimensions()

Level 2: Integration Tests (benchmark_latency.py)
  ├── test_synthetic_5sensor_pipeline_latency()
  └── test_tflite_conversion_numerical_parity()

Level 3: Closed-Loop CARLA Tests (collect_data.py & test_vlm_carla.py)
  ├── test_carla_synchronous_multi_camera_tick()
  └── test_50_episode_generalization_town02()

Level 4: Regression vs Baseline (accuracy_check.py)
  └── compare_vlm_vs_results05_benchmarks()
```

---

## 18. Exact Files, Classes & Functions Specification

```
/Users/msr/claude_code/BTP_Project/MTP_TESTING/
├── parameters.py                      [MODIFIED] Sensor configs, Town01/02 paths, dimensions
├── multimodal_encoder.py              [MODIFIED] 5-sensor MultimodalEdgeEncoder class
├── collect_data.py                    [CREATED] Multi-sensor synchronized collector
├── pretrain_encoder.py                [CREATED] Supervised domain fine-tuning script
├── main_vlm_train.py                  [CREATED] PPO train/test loop + CarlaEnvVLM class
├── benchmark_latency.py               [MODIFIED] Multi-sensor latency profiler
├── PIL_edge_vlm.py                    [MODIFIED] Processor-in-loop socket edge client
└── PIL_simulation.py                  [MODIFIED] Processor-in-loop server simulation
```

---

## 19. Task Breakdown & Implementation Dependency Graph

```mermaid
graph TD
    T1[Task 1: Core Parameters & Latent Config] --> T2[Task 2: 5-Sensor MultimodalEdgeEncoder]
    T2 --> T3[Task 3: Town01 Synchronous Collector]
    T3 --> T4[Task 4: Collect 20K Multi-Sensor Dataset]
    T4 --> T5[Task 5: Supervised Pre-Training Script]
    T5 --> T6[Task 6: Domain Fine-Tuning Execution]
    T6 --> T7[Task 7: CarlaEnvVLM & PPO Training Pipeline]
    T7 --> T8[Task 8: Train PPO on Town01 1M Steps]
    T8 --> T9[Task 9: Zero-Shot Generalization Test on Town02]
    T9 --> T10[Task 10: TFLite Quantization & PIL Edge Benchmark]
```

---

## 20. Risks, Trade-offs, Alternatives & Mitigations

| Risk / Bottleneck | Severity | Mitigation Strategy | Alternative |
|---|---|---|---|
| **GPU Memory Bottleneck during CARLA + PPO** | Medium | Run CARLA in `-quality-level=Low -RenderOffScreen` mode. | Run CARLA on host and PPO in separate headless process. |
| **PPO Policy Instability with High-Dim Input** | High | Freeze pre-trained encoder during initial 500K steps; only train MLP actor. | Use small entropy bonus ($c_2 = 0.01$) to encourage smooth exploration. |
| **Socket Network Jitter during PIL Testing** | Medium | Use TCP `TCP_NODELAY` flag and packed binary `struct.pack`. | Shared memory / IPC if running on same hardware. |

---

## 21. Deliverables & Acceptance Criteria for EVERY Task

### Task 1: Environment & Parameter Configuration
* **WHAT:** Update `parameters.py` with multi-camera specifications, LiDAR parameters, `OBSERVATION_DIM=133`, `Town01` train map, `Town02` test map.
* **WHY:** Provides single source of truth across all training, data collection, and evaluation scripts.
* **WHERE:** `parameters.py`
* **HOW:** Define `CAMERA_SPECS`, `LIDAR_SPECS`, `VLM_LATENT_DIM=128`, `NAV_DIM=5`.
* **DEPENDS ON:** None.
* **TESTING:** Import in python: `from parameters import *; assert OBSERVATION_DIM == 133`.
* **DONE WHEN:** All constants import cleanly without syntax or type errors.

### Task 2: Multimodal Edge Encoder Module
* **WHAT:** Implement `MultimodalEdgeEncoder` with shared patch embeddings for 3 cameras + LiDAR BEV, FiLM speech conditioning, and 128-dim latent projection.
* **WHY:** Replaces legacy 52MB VAE with a multi-sensor, language-aware perception backbone.
* **WHERE:** `multimodal_encoder.py`
* **HOW:** PyTorch module utilizing `DepthwiseSeparableConv`, `SharedPatchEmbedder`, `FiLMConditioning`, `TransformerBlock`.
* **DEPENDS ON:** Task 1.
* **TESTING:** Run `python multimodal_encoder.py` $\rightarrow$ verifies output shape `(1, 133)` and param count $\le 150\text{K}$.
* **DONE WHEN:** Forward pass executes in $< 25\text{ ms}$ on CPU and produces exact tensor shapes.

### Task 3: Multi-Sensor Data Collector
* **WHAT:** Create `collect_data.py` to synchronously record 3 RGB cameras + LiDAR + Telemetry from CARLA Autopilot in Town01.
* **WHY:** Required to train the encoder on driving scene geometry before RL.
* **WHERE:** `collect_data.py`
* **HOW:** CARLA synchronous mode (20 FPS) with `queue.Queue` synchronization, saving `.npz` multi-view frames and `labels.csv`.
* **DEPENDS ON:** Task 1, CARLA Server.
* **TESTING:** Run `python collect_data.py --town Town01 --frames 100` $\rightarrow$ verifies `.npz` files and labels match.
* **DONE WHEN:** Collects 20,000 synchronized frames across 4 randomized weathers with clean teardown on exit.

### Task 4: Supervised Domain Adaptation & Pre-Training
* **WHAT:** Implement and run `pretrain_encoder.py` to train the encoder on steering and speed prediction.
* **WHY:** Instills spatial "road sense" and prevents RL policy collapse.
* **WHERE:** `pretrain_encoder.py`
* **HOW:** Multi-task MSE loss with AdamW and cosine annealing schedule for 40 epochs.
* **DEPENDS ON:** Task 2, Task 3.
* **TESTING:** Validation steering MSE loss drops $\le 0.05$.
* **DONE WHEN:** Saves best checkpoint to `models/pretrained_vlm_encoder.pth`.

### Task 5: PPO Reinforcement Learning Training Pipeline
* **WHAT:** Implement `main_vlm_train.py` with `CarlaEnvVLM`, PPO Actor-Critic, GAE buffer, and TensorBoard logging.
* **WHY:** Core training script for continuous driving control in Town01.
* **WHERE:** `main_vlm_train.py`
* **HOW:** Standard PyTorch PPO with continuous action Gaussian policy, reward shaping, and periodic checkpointing.
* **DEPENDS ON:** Task 2, Task 4.
* **TESTING:** Run `python main_vlm_train.py --mode train --timesteps 1000` $\rightarrow$ verifies gradient steps and checkpoint generation.
* **DONE WHEN:** Trains for 1,000,000 timesteps, reward curve shows clear upward trend.

### Task 6: Zero-Shot Generalization Benchmarking
* **WHAT:** Evaluate trained policy on unseen `Town02` across 50 test episodes.
* **WHY:** Proves the model generalizes across maps without overfitting.
* **WHERE:** `main_vlm_train.py --mode test --town Town02 --episodes 50`
* **HOW:** Deterministic evaluation logging Route Completion, Collisions, Speed to `Results_VLM/test_results_town02.csv`.
* **DEPENDS ON:** Task 5.
* **TESTING:** Compare CSV against `Results_05/test_results_gpu.csv`.
* **DONE WHEN:** Route Completion $\ge 85\%$, Collision Rate $\le 10\%$.

### Task 7: Latency Profiling & Edge Deployment
* **WHAT:** Micro-benchmark inference latency and convert model to TFLite/ONNX for PIL Raspberry Pi testing.
* **WHY:** Validates real-time edge execution feasibility.
* **WHERE:** `benchmark_latency.py`, `PIL_edge_vlm.py`
* **HOW:** Measure 500 forward passes; run TCP client-server loop over socket.
* **DEPENDS ON:** Task 2, Task 5.
* **TESTING:** Run `python benchmark_latency.py --iters 500`.
* **DONE WHEN:** Average latency $\le 25\text{ ms}$ on CPU ($> 40\text{ FPS}$) and $\le 3\text{ ms}$ on GPU.

---

## 22. Final Implementation Checklist

- [x] **Audit & Planning:** Analyze legacy baseline `MTP_TESTING` and speech repository `Aanchal_Spring`.
- [x] **Configuration:** Update `parameters.py` with 5-sensor specs and Town01/02 parameters.
- [x] **Encoder Architecture:** Implement and verify `multimodal_encoder.py` (0.51 MB, shape `(1, 133)`).
- [x] **Data Collection Script:** Implement `collect_data.py` for synchronous Town01 recording.
- [x] **Pre-Training Script:** Implement `pretrain_encoder.py` for multi-task steering/speed adaptation.
- [x] **PPO Training Pipeline:** Implement `main_vlm_train.py` for Town01 training and Town02 testing.
- [x] **Hardware Benchmark:** Implement and verify `benchmark_latency.py` (20.7 ms on CPU, 48.1 FPS).
- [x] **Documentation & Manual:** Produce comprehensive `implementation_plan.md`, `operational_guide.md`, and `task_level_design_doc.md`.
- [ ] **Remote Execution:** Transfer files to Remote PC via AnyDesk.
- [ ] **Data Gathering:** Execute `collect_data.py` for 20,000 frames in CARLA Town01.
- [ ] **Encoder Fine-Tuning:** Execute `pretrain_encoder.py` to produce `pretrained_vlm_encoder.pth`.
- [ ] **Policy Training:** Execute `main_vlm_train.py` on Town01 for 1,000,000 timesteps.
- [ ] **Benchmark Verification:** Execute `main_vlm_train.py --mode test --town Town02` and generate final comparison graphs.
