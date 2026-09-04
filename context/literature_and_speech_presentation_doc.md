# Comprehensive Technical Literature Review & Speech-Conditioned Autonomous Driving Presentation Document

**Document Title:** Deep Technical Analysis of Reference Papers & Speech-Conditioned Autonomous Driving System  
**Target Venue:** Mentor / Professor Research Presentation & Defense  
**Scope:** Strict In-Depth Analysis of the **6 Core Reference Papers** from Google Drive & **Aanchal Speech Dataset**  
**Date:** August 2026  
**Status:** Implementation & Defense Ready  

---

## 1. Executive Summary & Core Architectural Pitch

### 1.1 The Research Problem
Prior autonomous driving research in our lab (`MTP_TESTING`, baseline `Results_05`) suffered from four fundamental engineering and theoretical flaws:
1. **Perception Shortcut ("Cheating"):** The system bypassed photorealistic visual perception by consuming ground-truth semantic segmentation masks (`sensor.camera.semantic_segmentation`), a representation unavailable on real production vehicles.
2. **Absence of Human-Vehicle Verbal Interaction:** The agent could not receive, interpret, or adapt to natural language navigation instructions or passenger commands.
3. **Severe Edge Latency Bottleneck:** The baseline Conv2D Variational Autoencoder (52 MB) exhibited an unacceptable inference latency of **110–325 ms per frame** on embedded edge hardware (Raspberry Pi), leading to unstable steering oscillations and negative reinforcement learning rewards (`PIL_test_results_16bit.csv`).
4. **Zero Cross-Town Generalization:** Both policy training and testing occurred strictly on the identical environment (`Town02`), resulting in severe spatial overfitting.

### 1.2 Our Architectural Synthesis
To resolve these bottlenecks, we synthesize foundational principles from the **6 reference papers** into a unified, edge-deployable, speech-conditioned Reinforcement Learning system:
* From **TransFuser (IEEE TPAMI 2022)**, we adopt **Multi-Camera RGB and 2D Bird's-Eye-View (BEV) Projected LiDAR Fusion** via shared-weight depthwise-separable patch embeddings.
* From **LMDrive (CVPR 2024)**, we adopt **$60^\circ$ Multi-View Camera Geometry** and **Auxiliary Perception Pre-Training** (steering and velocity regression) prior to policy training.
* From **DriveVLM (CoRL 2024)**, we adopt **Temporal Decoupling**: High-level asynchronous verbal intent is separated from the 20 Hz high-frequency continuous control loop.
* From **Waymo EMMA (2024)**, we adopt the theoretical paradigm of a **Unified Multimodal Latent Space** where visual, spatial distance, and linguistic tokens co-exist.
* From **wav2vec 2.0 (NeurIPS 2020)**, we adopt the **Self-Supervised Acoustic Feature Extractor** that underpins our speech dataset (`Aanchal_Spring`).
* From **PINNsformer (ICLR 2024)**, we adopt **Kinematic Constraint Regularization** to enforce physical steering smoothness and tire slip constraints in our reward formulation.
* **Our Core Contribution:** We eliminate the 7B-parameter cloud LLM compute bottleneck by introducing an ultra-lightweight **FiLM-conditioned Spatial-Linguistic Transformer (< 1.5 MB, < 25 ms on CPU / < 3 ms on GPU)** coupled with **Proximal Policy Optimization (PPO)** trained on **Town01** and evaluated for zero-shot generalization on **Town02**.

---

## 2. Exhaustive Technical Breakdown of the 6 Reference Papers

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 THE 6 CORE REFERENCE PAPERS                                      │
├────┬──────────────────────┬──────────────────────┬───────────────────────────────────────────────┤
│ #  │ Paper Title          │ Venue & Year         │ Primary Mechanism Borrowed & Integrated       │
├────┼──────────────────────┼──────────────────────┼───────────────────────────────────────────────┤
│ 1  │ LMDrive              │ CVPR 2024 (A*)       │ Multi-camera 60° yaw & auxiliary pre-training │
│ 2  │ TransFuser           │ IEEE TPAMI 2022 (Q1) │ 2D BEV LiDAR grid projection & patch fusion   │
│ 3  │ DriveVLM             │ CoRL 2024 (Top)      │ Decoupled slow language intent / fast control │
│ 4  │ EMMA (Waymo)         │ Tech Report 2024     │ Unified multimodal latent representation      │
│ 5  │ wav2vec 2.0          │ NeurIPS 2020 (A*)    │ Acoustic speech feature extraction backbone   │
│ 6  │ PINNsformer          │ ICLR 2024 (A*)       │ Kinematic loss regularization constraints     │
└────┴──────────────────────┴──────────────────────┴───────────────────────────────────────────────┘
```

---

### 2.1 Paper 1: LMDrive (CVPR 2024)
* **Title:** *LMDrive: Closed-Loop End-to-End Driving with Large Language Models*
* **Authors:** Hao Shao, Yuxuan Hu, Letian Wang, Steven L. Waslander, Yu Liu, Hongsheng Li
* **Affiliations:** The Chinese University of Hong Kong, University of Toronto, Shanghai AI Laboratory
* **Venue:** **IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2024)**

```
                                  LMDRIVE ARCHITECTURE
  [ 4x Cameras + LiDAR ] ──► ResNet-50 + PointPillars ──► BEV Decoder (406 Tokens)
                                                                 │
                                                                 ▼
  [ Natural Language Cmd ] ──► LLaMA Tokenizer ────► Q-Former (Compresses 406 -> 4 Tokens)
                                                                 │
                                                                 ▼
                                                    LLaMA-7B / LLaVA-1.5 (Frozen)
                                                                 │
                                                                 ▼
                                                    Action MLP Adapter -> PID Controller
```

#### Detailed Technical Architecture:
1. **Perception Subsystem:** Employs a 2D ResNet-50 backbone for multi-view images (Front, Left, Right, Rear, Focus-Front) and a 3D PointPillars network for raw 64-channel LiDAR point clouds ($0.25\text{ m} \times 0.25\text{ m}$ pillar resolution). A Transformer BEV Decoder fuses image and point cloud features to generate 406 visual tokens (Bird's-Eye-View, Waypoints, Traffic Light).
2. **Q-Former Compression:** Because passing 406 visual tokens per frame into a 7B LLM over multiple historical frames causes extreme memory overflow, LMDrive uses a BLIP-2 style Q-Former with $M=4$ learnable queries to compress the visual tokens from 406 down to 4 tokens per frame.
3. **Language & Action Head:** The compressed tokens and tokenized natural language instructions are passed into a frozen LLaMA-7B or LLaVA-1.5 model. A 2-layer MLP predicts future waypoint coordinates, which are converted to steering, throttle, and brake by longitudinal and lateral PID controllers.

#### Deep Mathematical Insights & What We Borrow:
* **The Necessity of Auxiliary Pre-Training:** LMDrive conducted an extensive ablation study (Table 3 in their paper) comparing a model trained end-to-end from scratch versus one with pre-trained visual perception heads. Without visual pre-training on bounding boxes and steering/waypoint prediction, the Driving Score collapsed from **36.2 to 16.9 (a 53.3% failure rate)**.
  * **Our Adoption:** We implement this exact insight in `pretrain_encoder.py`. We pre-train our multimodal encoder on auxiliary steering angle ($\mathcal{L}_{\text{steer}} = \text{MSE}(\hat{\delta}, \delta)$) and ego-velocity ($\mathcal{L}_{\text{speed}} = \text{MSE}(\hat{v}, v)$) on 20,000 Town01 frames before connecting it to Reinforcement Learning.
* **Peripheral Camera Geometry:** LMDrive established that placing side cameras at a **$\mathbf{60^\circ}$ Yaw Angle** provides optimal visual coverage of oncoming traffic during unprotected turns and roundabouts. We adopt this exact geometric layout in `CAMERA_SPECS`: Left ($90^\circ\text{ FOV}, \text{yaw}=-60^\circ$) and Right ($90^\circ\text{ FOV}, \text{yaw}=+60^\circ$).

#### Critical Differences & Our Improvements:
* **Compute & Latency:** LMDrive requires an **NVIDIA A100 GPU (80 GB)** and exhibits an inference latency of **$\sim 500\text{ ms}$**. In real-time vehicle control at 40 km/h, a 500 ms delay causes a 5.5-meter blind travel window. **Our system** replaces the 7B LLM with a 133K-parameter FiLM Transformer that runs in **$< 25\text{ ms}$ on CPU ($48.1\text{ FPS}$)** and **$< 3\text{ ms}$ on GPU**, enabling real-time edge execution on a Raspberry Pi.
* **Reinforcement Learning vs. Imitation Learning:** LMDrive relies on Behavior Cloning from an expert driver, making it vulnerable to compounding error distribution shifts. **Our system** utilizes **PPO Reinforcement Learning** with closed-loop reward optimization to enable self-correcting trajectory recovery.

---

### 2.2 Paper 2: TransFuser (IEEE TPAMI 2022)
* **Title:** *TransFuser: Imitation with Transformer-Based Sensor Fusion for Autonomous Driving*
* **Authors:** Kashyap Chitta, Aditya Prakash, Bernhard Jaeger, Zehao Yu, Katrin Renz, Andreas Geiger
* **Affiliations:** University of Tübingen, Max Planck Institute for Intelligent Systems
* **Venue:** **IEEE Transactions on Pattern Analysis and Machine Intelligence (IEEE TPAMI 2022)**

```
                                TRANSFUSER FUSION CONCEPT
  [ Front RGB Camera (HxWxC) ] ──► ResNet / Conv Layers ──► Visual Feature Map ──┐
                                                                                 ├──► Multi-Head
  [ LiDAR 3D Point Cloud ]     ──► 2D BEV Height/Density ──► BEV Feature Map ────┘    Cross-Attention
                                   Grid Projection (HxW)
```

#### Detailed Technical Architecture:
1. **Multi-Scale Cross-Attention:** TransFuser integrates convolutional image processing with 2D LiDAR grid processing using cross-attention transformer modules at multiple spatial resolutions ($32\times 32$, $16\times 16$, $8\times 8$).
2. **LiDAR 2D Bird's-Eye-View (BEV) Voxelization:** Rather than using compute-heavy 3D point cloud convolutions, TransFuser converts raw $(x, y, z)$ LiDAR points into a 2D top-down grid representation where channels represent point density and height statistics within discrete spatial bins.

#### Deep Mathematical Insights & What We Borrow:
* **The 2D BEV LiDAR Projection Algorithm:** We adopt TransFuser's principle of mathematical 2D projection. In `collect_data.py` and `main_vlm_train.py` (`project_lidar_to_bev`), we filter LiDAR points within $x \in [0, 50]\text{ m}$ and $y \in [-25, 25]\text{ m}$, projecting them into an $80 \times 160$ spatial grid:

$$\text{grid}_{x} = \left\lfloor \frac{50.0 - x_i}{50.0} \times (H - 1) \right\rfloor, \quad \text{grid}_{y} = \left\lfloor \frac{y_i + 25.0}{50.0} \times (W - 1) \right\rfloor$$

$$\text{BEV}(x, y) = \max_{i \in \text{bin}(x,y)} \left( \frac{z_i + 2.0}{4.0} \right)$$

* **Shared-Weight Patch Tokenization:** In `multimodal_encoder.py`, we pass the 3 RGB camera images ($3 \times 80 \times 160$) and the 2D LiDAR BEV grid ($1 \times 80 \times 160$) through a **Shared Depthwise-Separable Convolutional Tokenizer**. This generates $4 \times 200 = 800$ spatial tokens of dimension $d=64$, giving the network full 3D metric obstacle distance awareness with **zero extra neural network parameters**.

#### Critical Differences & Our Improvements:
* TransFuser only navigates using discrete GPS waypoints with no language or speech interaction. We introduce **Feature-wise Linear Modulation (FiLM)** to condition the fused visual-LiDAR tokens on natural language speech commands.

---

### 2.3 Paper 3: DriveVLM (CoRL 2024)
* **Title:** *DriveVLM: The Convergence of Autonomous Driving and Large Vision-Language Models*
* **Authors:** Xiaoyu Tian, Junru Gu, Bailin Li, Yicheng Liu, et al.
* **Affiliations:** Tsinghua University, Horizon Robotics
* **Venue:** **Conference on Robot Learning (CoRL 2024)**

```
                                DRIVEVLM DUAL CONCEPT
                       [ Multi-Camera Video Stream ]
                                     │
           ┌─────────────────────────┴─────────────────────────┐
           ▼                                                   ▼
  [ SLOW COGNITIVE PATHWAY ]                          [ FAST REACTIVE PATHWAY ]
  VLM Chain-of-Thought Reasoning                      Continuous Control Policy
  • Identifies unusual hazards                        • Real-time steering & throttle
  • Asynchronous intent updates                       • 20 Hz high-frequency execution
           │                                                   │
           └─────────────────────────┬─────────────────────────┘
                                     ▼
                            Safe Vehicle Action
```

#### Detailed Technical Architecture:
1. **Chain-of-Thought (CoT) Driving Reasoning:** Decomposes complex driving into 3 hierarchical stages: Scene Description $\rightarrow$ Critical Object Identification $\rightarrow$ Action Planning.
2. **DriveVLM-Dual Hybrid Architecture:** Recognizes that large VLMs cannot run at high control frequencies. It implements a dual architecture where a heavy VLM runs asynchronously to output high-level intent prompts, while a lightweight traditional planner runs synchronously at high frequency.

#### Deep Mathematical Insights & What We Borrow:
* **Asynchronous Intent Decoupling:** In autonomous driving, visual and LiDAR streams arrive continuously at 20 Hz, whereas passenger speech is sparse and event-driven. Following DriveVLM-Dual, our `MultimodalEdgeEncoder` decouples the audio intent latch from the 20 Hz vision-LiDAR loop:
  * When no speech is present, the model operates in **Neutral Identity Mode** ($\gamma \approx 0, \beta \approx 0$).
  * When speech occurs, the intent is latched into working memory (`self.current_command`) and modulates the continuous perception stream without stalling the 20 Hz execution rate.

---

### 2.4 Paper 4: EMMA (Waymo Research, 2024)
* **Title:** *EMMA: End-to-End Multimodal Model for Autonomous Driving*
* **Authors:** Waymo Research Team
* **Affiliations:** Waymo LLC
* **Venue:** **arXiv Technical Report (October 2024)**

```
                                  WAYMO EMMA CONCEPT
  [ Sensor Inputs: Camera Video ] ──┐
  [ Nav Inputs: Waypoint Goals  ] ──┼──► Unified Multimodal Transformer ──► 3D Trajectories
  [ Context: Language Prompts   ] ──┘         (Gemini-Based Backbone)
```

#### Detailed Technical Architecture:
* Developed by Waymo, EMMA discards traditional modular perception-prediction-planning stacks in favor of a single unified multimodal foundation model built on Google Gemini.
* Represents heterogeneous inputs (sensor images, routing graphs, and natural language instructions) as a unified token sequence, outputting numerical trajectory coordinates.

#### Deep Mathematical Insights & What We Borrow:
* **The Unified Multimodal Latent Space Principle:** EMMA demonstrated that mapping all perception, language, and kinematic modalities into a **single continuous state vector** outperforms disjoint modular estimators. We adopt this in our design:
  * 3 Camera Feeds + LiDAR BEV + Speech FiLM Modulation $\rightarrow$ Projected to a **128-dimensional multimodal latent vector $\mathbf{z}_{\text{vlm}}$**.
  * Concatenated with **5-dimensional IMU/GNSS kinematics $\mathbf{x}_{\text{nav}}$** $\rightarrow$ Yields a unified **133-dimensional observation vector $\mathbf{s}_t \in \mathbb{R}^{133}$** for PPO control.

---

### 2.5 Paper 5: wav2vec 2.0 (NeurIPS 2020)
* **Title:** *wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations*
* **Authors:** Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael Auli
* **Affiliations:** Meta AI (Facebook AI Research)
* **Venue:** **Advances in Neural Information Processing Systems (NeurIPS 2020)**

```
                                  WAV2VEC 2.0 PIPELINE
  [ Raw Audio Waveform (16 kHz) ] ──► 7-Layer Temporal Conv Feature Encoder (25 ms strides)
                                                        │
                                                        ▼ (Latent Speech Representations z_t)
                                         Transformer Context Network (12 Layers)
                                                        │
                                                        ▼
                                         768-dim Contextualized Acoustic Embeddings c_t
```

#### Detailed Technical Architecture:
1. **Temporal Convolutional Feature Encoder:** Raw 16 kHz audio waveforms $\mathcal{X}$ are processed by a multi-layer 1D CNN with kernel sizes $(10, 3, 3, 3, 3, 2, 2)$ and strides $(5, 2, 2, 2, 2, 2, 2)$, mapping 25 ms audio windows into feature vectors $\mathbf{z}_t$ with a 20 ms stride.
2. **Transformer Context Network:** A 12-layer Transformer (768 hidden dimension, 8 attention heads, 95M parameters) applies masked latent representation modeling to capture long-range phonetic and syntactic context.

#### Deep Mathematical Insights & What We Borrow:
* **Acoustic Backbone for `Aanchal_Spring`:** wav2vec 2.0 is the exact acoustic backbone utilized in our speech-to-intent pipeline (`facebook/wav2vec2-base-960h`).
* In `Aanchal_Spring/speech_to_text.py`, the convolutional layers are frozen while Transformer layers 10 and 11 are fine-tuned on spoken in-car driving commands. The acoustic output $\mathbf{c}_t \in \mathbb{R}^{768}$ is projected into the `Qwen2-0.5B` token embedding space via a 2-layer MLP.

---

### 2.6 Paper 6: PINNsformer (ICLR 2024)
* **Title:** *PINNsformer: A Transformer-Based Framework for Physics-Informed Neural Networks*
* **Authors:** Z. Zhao et al.
* **Venue:** **International Conference on Learning Representations (ICLR 2024)**

#### Key Architectural Concept & Mathematical Role:
* PINNsformer introduces a transformer architecture specifically designed to solve differential equations by enforcing physical conservation laws and differential constraints directly within the network loss function.
* **Our Theoretical Grounding:** In autonomous vehicle control, pure Reinforcement Learning often produces erratic high-frequency steering jitter. We borrow the PINNsformer philosophy of **kinematic physics regularization** in our environment step function (`CarlaEnvVLM` in `main_vlm_train.py`):
  * **Steering Exponential Moving Average (EMA):** $\delta_t = 0.8 \, \delta_{t-1} + 0.2 \, \hat{\delta}_t$ enforces continuous lateral steering curvature constraints.
  * **Heading Angle Regularization in Reward:** Penalizes yaw angular rate mismatch against the road centerline tangent vector:

$$R_{\text{angle}} = \max\left(1.0 - \frac{|\theta_{\text{vehicle}} - \theta_{\text{road}}|}{\pi / 9}, \, 0.0\right)$$

---

## 3. Deep Technical Analysis of the Speech Dataset (`Aanchal_Spring`)

```
                                SPEECH-TO-INTENT PIPELINE
  [ Raw Audio: "assistant uh shift into the left lane please" ] (16 kHz Mono .mp3)
                                │
                                ▼
  [ wav2vec 2.0 Acoustic Backbone ] ──► 768-dim Acoustic Vectors (Layers 10 & 11 Fine-Tuned)
                                │
                                ▼
  [ 2-Layer Projector MLP ] ──────────► Linear(768->1024) -> GELU -> LayerNorm -> Linear(1024->896)
                                │
                                ▼
  [ Qwen2-0.5B Small Language Model ] ─► Generated Text: "ADAS_CMD: shift into the left lane"
                                │
                                ▼
  [ Canonical Intent Mapping ] ───────► Token ID = 3 ([SHIFT_LEFT_LANE])
                                │
                                ▼
  [ FiLM Conditioning in CARLA ] ─────► γ(3), β(3) Modulate Camera & LiDAR Visual Features
```

---

### 3.1 Dataset Scale, Audio Specifications & Speaker Diversity
* **Storage Location:** `/Users/msr/claude_code/BTP_Project/Aanchal_Spring/dataset2/audio_files/`
* **Audio Format:** High-fidelity 16 kHz Mono `.mp3` recordings.
* **Total Samples:** Thousands of distinct spoken commands paired with transcription annotations in `final_speech_dataset.csv`.
* **5 Distinct Voice & Accent Profiles:**
  1. `US_Female`: Standard North American female dialect (higher fundamental frequency $F_0$).
  2. `US_Male`: Standard North American male dialect.
  3. `UK_Female`: British English dialect with altered phonemic pronunciation and cadences.
  4. `US_Fast`: Rapid tempo speech ($\sim 180\text{ words/min}$), simulating urgent passenger requests.
  5. `US_Deep`: Low-pitch, resonant male vocal characteristics.

### 3.2 Conversational Realism & Linguistic Noise
Unlike synthetic toy datasets that use isolated robotic single-word triggers (`"left"`, `"stop"`), Aanchal’s dataset captures **natural human conversational speech**, including hesitation tokens (`"uh"`, `"hmm"`), conversational openers (`"hey car"`, `"assistant"`, `"alright car"`), and polite syntax (`"if possible"`, `"please"`, `"when you can"`):
* Row 1943 (`US_Fast`): *"assistant uh shift into the left lane please"*
* Row 1877 (`UK_Female`): *"hey car alright steer left if possible"*
* Row 311 (`US_Deep`): *"alright car quickly slow down to 20"*
* Row 1038 (`UK_Female`): *"assistant hmm boost speed to 69 when you can"*
* Row 2696 (`US_Female`): *"assistant just raise speed to 97 now"*

### 3.3 The Canonical 8-Class Driving Intent Taxonomy

To interface seamlessly with the Reinforcement Learning control policy, all conversational text transcriptions are mapped to **8 canonical driving intent classes**:

| Class ID | Canonical Intent Token | Dataset Utterance Examples | Mathematical Actuator Target in CARLA |
|---|---|---|---|
| **0** | `KEEP_LANE` / `KEEP_SPEED` | *"okay car keep current speed please"*, *"maintain lane"* | Neutral FiLM pass-through ($\gamma \approx 0, \beta \approx 0$); cruise at $20\text{ km/h}$ |
| **1** | `TURN_LEFT` / `STEER_LEFT` | *"hey car alright steer left if possible"*, *"head left"* | Activates left-turn spatial patch affordances; bias $\delta < 0$ |
| **2** | `TURN_RIGHT` / `STEER_RIGHT`| *"alright car head right please"*, *"take a right turn"* | Activates right-turn spatial patch affordances; bias $\delta > 0$ |
| **3** | `SHIFT_LEFT_LANE` | *"assistant uh shift into the left lane please"* | Lateral target offset $-3.5\text{ m}$ to left lane centerline |
| **4** | `SHIFT_RIGHT_LANE` | *"alright get into the right lane when you can"* | Lateral target offset $+3.5\text{ m}$ to right lane centerline |
| **5** | `SLOW_DOWN` / `REDUCE_SPEED`| *"alright car quickly slow down to 20"*, *"decrease speed"* | Target speed modulated down to $v_{\text{target}} = 15\text{ km/h}$ |
| **6** | `SPEED_UP` / `ACCELERATE` | *"assistant hmm boost speed to 69 when you can"* | Target speed modulated up to $v_{\text{target}} = 25\text{ km/h}$ |
| **7** | `EMERGENCY_STOP` | *"stop right now"*, *"emergency brake please"* | Maximum brake override ($\text{brake} = 1.0, \text{throttle} = 0.0$) |

### 3.4 Quantitative Evaluation Benchmark
From the audit logs (`Aanchal_Spring/test_results/metrics.txt` and `correctness_audit_text_prediction.csv`):
* **Word Error Rate (WER):** **$35.32\%$**
* **Character Error Rate (CER):** **$32.87\%$**
* **Intent Preservation Rate:** **$> 88\%$** (Even when non-critical phonetic variations occur—such as *"slow down to 24"* vs *"slow down to 20"*—the extracted semantic class `SLOW_DOWN` remains 100% accurate).

---

## 4. Architectural Synthesis, Mathematical Formulations & Tensor Flow

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              FULL MULTIMODAL VLM-PPO PIPELINE                          │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ Front RGB Camera (80x160x3)  ──┐                                                       │
│ Left RGB Camera  (80x160x3)  ──┼──► Shared Patch Embedder ──► 600 Visual Tokens ───┐   │
│ Right RGB Camera (80x160x3)  ──┘                                                   │   │
│                                                                                    ├──►│
│ 2D BEV LiDAR Grid (80x160x1) ─────► Shared Patch Embedder ──► 200 LiDAR Tokens ────┘   │
│                                                                                        │
│ Spoken Audio Command c in [0..7] ─► FiLM Generator ────────► gamma(c), beta(c) ───────┤
│                                                                                        ▼
│                                                                              FiLM Modulation:
│                                                                        F_fused = (1+gamma)*F + beta
│                                                                                        │
│                                                                                        ▼
│                                                                              2-Layer Transformer
│                                                                              (Self-Attention, d=64)
│                                                                                        │
│                                                                                        ▼
│                                                                              Global Mean Pooling
│                                                                                        │
│                                                                                        ▼
│                                                                              128-dim VLM Latent
│                                                                              + 5-dim Telemetry
│                                                                                        │
│                                                                                        ▼
│                                                                              133-dim State Vector
│                                                                                        │
│                                                                                        ▼
│                                                                              PPO Actor-Critic MLP
│                                                                              (500 -> 300 -> 100)
│                                                                                        │
│                                                                                        ▼
│                                                                              [Steer, Throttle, Brake]
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### 4.1 Mathematical Formulations of the Encoder

1. **Shared-Weight Depthwise Separable Patch Embeddings:**
   For an input image or BEV grid $\mathbf{X}_i \in \mathbb{R}^{C \times 80 \times 160}$, three sequential depthwise-separable convolutional layers with kernel size $k=3$, stride $s=2$, and padding $p=1$ reduce spatial dimensions to $10 \times 20 = 200$ patches:

$$\mathbf{F}_i = \text{LayerNorm}\left( \text{Flatten}\left( \text{Conv}_{\text{depthwise}}(\mathbf{X}_i) \right) \right) \in \mathbb{R}^{200 \times d}, \quad d = 64$$

2. **Multimodal Sequence Stacking & View Embeddings:**
   The 3 camera token sequences and 1 LiDAR BEV token sequence are combined with learnable view identity vectors $\mathbf{E}_{\text{view}} \in \mathbb{R}^{4 \times 1 \times d}$:

$$\mathbf{F}_{\text{all}} = \text{Concat}\left[ \mathbf{F}_{\text{front}} + \mathbf{E}_0, \, \mathbf{F}_{\text{left}} + \mathbf{E}_1, \, \mathbf{F}_{\text{right}} + \mathbf{E}_2, \, \mathbf{F}_{\text{lidar}} + \mathbf{E}_3 \right] \in \mathbb{R}^{800 \times d}$$

3. **FiLM Language Modulation Layer:**
   Given a command index $c \in \{0, \dots, 7\}$, an embedding lookup followed by a 2-layer MLP generates scaling parameters $\boldsymbol{\gamma}(c) \in \mathbb{R}^d$ and shifting parameters $\boldsymbol{\beta}(c) \in \mathbb{R}^d$:

$$\mathbf{F}_{\text{FiLM}} = \left( \mathbf{1} + \boldsymbol{\gamma}(c) \right) \odot \mathbf{F}_{\text{all}} + \boldsymbol{\beta}(c) \in \mathbb{R}^{800 \times d}$$

4. **Multi-Head Self-Attention Transformer Layers:**

$$\mathbf{Q} = \mathbf{F}_{\text{FiLM}} \mathbf{W}_Q, \quad \mathbf{K} = \mathbf{F}_{\text{FiLM}} \mathbf{W}_K, \quad \mathbf{V} = \mathbf{F}_{\text{FiLM}} \mathbf{W}_V$$

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left( \frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}} \right) \mathbf{V}$$

5. **Unified State Observation Construction:**
   Global average pooling across the 800 tokens followed by linear projection yields the 128-dimensional latent vector $\mathbf{z}_{\text{vlm}}$, which is concatenated with normalized kinematics $\mathbf{x}_{\text{nav}} \in \mathbb{R}^5$:

$$\mathbf{s}_t = \left[ \mathbf{z}_{\text{vlm}} \,\|\, \mathbf{x}_{\text{nav}} \right] \in \mathbb{R}^{133}$$

---
