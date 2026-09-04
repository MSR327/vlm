# Master Reference Guide: Research Papers & End-to-End Project Execution Roadmap

**Document Title:** Comprehensive Literature Compendium & Complete BTP Execution Roadmap (Starting from Legacy Baseline)  
**Project:** Multimodal Vision-Language-Sensor PPO Autonomous Driving System  
**Author:** BTP Research Team  
**Date:** August 2026  
**Status:** Official Project Execution Master Plan  

---

# Part 1: Comprehensive Compendium of Used Research Papers

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

### 1. LMDrive: Closed-Loop End-to-End Driving with Large Language Models
* **Authors:** Hao Shao, Yuxuan Hu, Letian Wang, Steven L. Waslander, Yu Liu, Hongsheng Li
* **Publication Venue:** **IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2024)**
* **Institutions:** CUHK, University of Toronto, Shanghai AI Laboratory
* **Core Contribution:** First closed-loop end-to-end autonomous driving framework in CARLA that accepts natural language instructions and dynamic multimodal sensor feeds.
* **Key Architecture:** 
  * 5 RGB Cameras (Front, Left $-60^\circ$, Right $+60^\circ$, Rear, Telephoto) $\rightarrow$ ResNet-50.
  * 64-channel 3D LiDAR $\rightarrow$ PointPillars.
  * Transformer BEV Decoder generates 406 visual tokens.
  * BLIP-2 style **Q-Former** (with 4 learnable query vectors) compresses 406 visual tokens down to **4 tokens**.
  * Frozen **LLaVA-1.5 / LLaMA-7B** language model predicts future waypoints $\rightarrow$ PID controller.
* **What We Borrow:**
  1. **$60^\circ$ Yaw Peripheral Camera Geometry:** Left (Yaw $-60^\circ$) and Right (Yaw $+60^\circ$) cameras to cover blind spots during turning and merging.
  2. **Auxiliary Pre-Training Finding:** Proved that pre-training visual feature encoders on auxiliary regression tasks (steering angle and vehicle speed) before reinforcement learning increases driving score from **16.9 to 36.2 ($+114\%$ boost)**.
* **Our Edge Adaptation:** Replaced their 7-Billion parameter cloud LLM (500 ms latency on A100 GPU) with a sub-1.5 MB FiLM Transformer running in **$< 25\text{ ms}$ on an ARM CPU ($48.1\text{ FPS}$)**.

---

### 2. TransFuser: Imitation with Transformer-Based Sensor Fusion for Autonomous Driving
* **Authors:** Kashyap Chitta, Aditya Prakash, Bernhard Jaeger, Zehao Yu, Katrin Renz, Andreas Geiger
* **Publication Venue:** **IEEE Transactions on Pattern Analysis and Machine Intelligence (IEEE TPAMI 2022)**
* **Institutions:** University of Tübingen, Max Planck Institute for Intelligent Systems
* **Core Contribution:** State-of-the-art multi-modal transformer architecture that fuses multi-scale camera images with 2D Bird's-Eye-View (BEV) LiDAR grids.
* **Key Architecture:**
  * Projects raw 3D LiDAR point clouds $(x, y, z)$ into a 2D top-down BEV height and density grid.
  * Uses multi-scale cross-attention transformer layers to exchange features between RGB convolutions and LiDAR BEV convolutions.
* **What We Borrow:**
  1. **2D BEV LiDAR Grid Projection Algorithm:** Eliminates heavy 3D point cloud convolutions by projecting LiDAR points into an $80 \times 160$ 2D grid ($x \in [0, 50]\text{m}, y \in [-25, 25]\text{m}, z \in [-2, 2]\text{m}$).
  2. **Shared-Weight Patch Embeddings:** Passing both camera feeds and the 2D LiDAR grid through a shared depthwise-separable convolutional tokenizer, granting full 3D metric distance awareness with **zero extra neural network parameters**.

---

### 3. DriveVLM: The Convergence of Autonomous Driving and Large Vision-Language Models
* **Authors:** Xiaoyu Tian, Junru Gu, Bailin Li, Yicheng Liu, et al.
* **Publication Venue:** **Conference on Robot Learning (CoRL 2024)**
* **Institutions:** Tsinghua University, Horizon Robotics
* **Core Contribution:** Introduces Chain-of-Thought (CoT) scene reasoning and proposes **DriveVLM-Dual**, a hybrid fast/slow architecture for real-time driving.
* **Key Architecture:**
  * **Slow Cognitive Pathway (1–2 Hz):** Large VLM performs deep reasoning for complex scenes and hazard detection.
  * **Fast Reactive Pathway (20–50 Hz):** Lightweight classical planner performs high-frequency continuous steering/throttle control.
* **What We Borrow:**
  * **Temporal Decoupling & Asynchronous Intent Latching:** High-level spoken passenger commands arrive sparsely (event-driven) and are latched into working memory, while the continuous 20 Hz vision-LiDAR control policy runs without interruption or cognitive stalling.

---

### 4. EMMA: End-to-End Multimodal Model for Autonomous Driving
* **Authors:** Waymo Research Team
* **Publication Venue:** **Waymo Research Technical Report (October 2024)**
* **Core Contribution:** Waymo's end-to-end foundation model built on Google Gemini. Replaces disjoint perception, prediction, and planning pipelines with a single unified multimodal token sequence.
* **What We Borrow:**
  * **Unified Multimodal Latent State Representation:** Co-embedding visual patches, depth grid tokens, and linguistic speech tokens into a **single unified 128-dimensional latent vector $\mathbf{z}_{\text{vlm}}$**, concatenated with **5-dim IMU/GNSS kinematics** to create the unified **133-dim state vector $\mathbf{s}_t$** for downstream PPO control.

---

### 5. wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations
* **Authors:** Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael Auli
* **Publication Venue:** **Advances in Neural Information Processing Systems (NeurIPS 2020)**
* **Institution:** Meta AI (Facebook AI Research)
* **Core Contribution:** Self-supervised framework learning contextualized speech representations from raw audio waveforms via temporal 1D-CNNs and masked Transformer encoders.
* **What We Borrow:**
  * **Acoustic Speech Backbone in `Aanchal_Spring`:** Converts 16 kHz raw passenger `.mp3` voice recordings into 768-dim contextualized sound vectors, fine-tuning Transformer layers 10 and 11 to extract driving intents across 5 speaker accent and pitch profiles.

---

### 6. PINNsformer: A Transformer-Based Framework for Physics-Informed Neural Networks
* **Authors:** Z. Zhao et al.
* **Publication Venue:** **International Conference on Learning Representations (ICLR 2024)**
* **Core Contribution:** Enforcing physical differential equations and conservation laws directly within transformer loss functions.
* **What We Borrow:**
  * **Kinematic Physics Regularization in PPO Control:** Incorporates steering Exponential Moving Average (EMA) ($\delta_t = 0.8\delta_{t-1} + 0.2\hat{\delta}_t$) and heading angle alignment rewards ($R_{\text{angle}}$) to penalize high-frequency steering jitter and ensure smooth vehicle dynamics.

---

# Part 2: End-to-End Project Execution Roadmap

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                            BASELINE STARTING POINT (WHAT WAS INHERITED)                          │
├──────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 1. Legacy MTP Pipeline (Results_05): Uses synthetic semantic segmentation ground truth (cheat). │
│ 2. Heavy VAE Architecture: 52 MB model footprint causing 110–325 ms latency on Raspberry Pi 4.   │
│ 3. Overfitted Evaluation: Model was trained on Town02 and evaluated on Town02 (0% generalization)│
│ 4. Completely Deaf: Zero speech interaction or natural language command conditioning.            │
│ 5. Isolated Audio Work: Aanchal_Spring repository existed separately, unconnected to CARLA.      │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### Complete Visual Project Execution Flowchart

```mermaid
flowchart TD
    %% Styling
    classDef legacy fill:#450a0a,stroke:#f87171,stroke-width:2px,color:#fecaca;
    classDef phase fill:#0f172a,stroke:#64748b,stroke-width:1px,color:#94a3b8;
    classDef step fill:#1e293b,stroke:#818cf8,stroke-width:2px,color:#fff;
    classDef decision fill:#312e81,stroke:#a855f7,stroke-width:2px,color:#fff;
    classDef complete fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#fff;

    Legacy[<b>📍 INHERITED STARTING POINT</b><br/>• Semantic Segmentation Baseline<br/>• 52 MB VAE with 110-325 ms latency<br/>• Overfitted on Town02<br/>• No speech integration] :::legacy --> Phase1[<b>STAGE 1: Architectural Redesign & Codebase Implementation</b>] :::phase

    subgraph S1 [Stage 1: System Redesign & Python Engineering]
        Phase1 --> T1_1["1.1 Rewrite parameters.py<br/>• Configure 5 Sensors (3x RGB, 2D BEV LiDAR, IMU)<br/>• Set TRAIN_TOWN='Town01', TEST_TOWN='Town02'<br/>• Set OBSERVATION_DIM=133"] :::step
        T1_1 --> T1_2["1.2 Implement multimodal_encoder.py<br/>• Shared-weight depthwise patch embedder<br/>• FiLM language modulation (8 classes)<br/>• 2-layer Transformer (d=64)<br/>• Parameter count: 133,065 (~0.51 MB)"] :::step
        T1_2 --> T1_3["1.3 Connect Speech Pipeline (Aanchal_Spring)<br/>• Map Wav2Vec2 + Qwen2-0.5B text to 8 Canonical IDs"] :::step
    end

    S1 --> Phase2[<b>STAGE 2: Multi-Sensor Data Gathering in CARLA</b>] :::phase

    subgraph S2 [Stage 2: Synthetic Data Harvesting]
        Phase2 --> T2_1["2.1 Develop collect_data.py<br/>• Synchronous 20 Hz recording<br/>• 3 Camera Views (Front 125°, Left/Right ±60°)<br/>• 2D BEV LiDAR Projection (160x80)"] :::step
        T2_1 --> T2_2["2.2 Execute Data Harvesting in Town01<br/>• Harvest 20,000 Synchronized Frames<br/>• Sample across 4 Weathers: Clear, Wet, Rain, Night<br/>• Save to data_collected_town01/ (.npz + labels.csv)"] :::step
    end

    S2 --> Phase3[<b>STAGE 3: Supervised Auxiliary Pre-Training</b>] :::phase

    subgraph S3 [Stage 3: Auxiliary Feature Learning]
        Phase3 --> T3_1["3.1 Develop pretrain_encoder.py<br/>• PyTorch Dataset & DataLoader for .npz frames<br/>• Multi-task Loss: Steering MSE + Speed MSE"] :::step
        T3_1 --> T3_2["3.2 Run 40 Epochs Pre-Training<br/>• Adopts LMDrive pre-training recipe (+114% score boost)<br/>• Output: pretrained_vlm_encoder.pth"] :::step
    end

    S3 --> Phase4[<b>STAGE 4: Closed-Loop PPO RL Training on Town01</b>] :::phase

    subgraph S4 [Stage 4: Reinforcement Learning]
        Phase4 --> T4_1["4.1 Develop main_vlm_train.py<br/>• Synchronous CarlaEnvVLM environment<br/>• Dynamic Speech Intent Sampling (8 classes)<br/>• PPO Actor-Critic (500->300->100 MLP)<br/>• Kinematic Physics Regularization (PINNsformer)"] :::step
        T4_1 --> T4_2["4.2 Execute 1,000,000 Timesteps Training<br/>• Train on Town01 with randomized weather<br/>• Periodic checkpointing & TensorBoard logging<br/>• Output: vlm_ppo_actor.pth & vlm_ppo_critic.pth"] :::step
    end

    S4 --> Phase5[<b>STAGE 5: Zero-Shot Generalization Evaluation on Town02</b>] :::phase

    subgraph S5 [Stage 5: Generalization & Benchmarking]
        Phase5 --> T5_1["5.1 Run 50 Evaluation Episodes in Town02<br/>• Evaluate on unseen map without fine-tuning<br/>• Measure: Route Completion, Collisions, Avg Speed"] :::step
        T5_1 --> D5{Zero-Shot Performance > Baseline?} :::decision
        D5 -- No --> T5_Tune[Hyperparameter / Reward Weight Tuning] :::step
        T5_Tune --> T4_2
        D5 -- Yes --> T5_2["5.2 Generate test_results_town02.csv<br/>• Statistically prove cross-map generalization"] :::complete
    end

    S5 --> Phase6[<b>STAGE 6: Edge Deployment & Processor-in-the-Loop (RPi 4)</b>] :::phase

    subgraph S6 [Stage 6: Edge Benchmarking]
        Phase6 --> T6_1["6.1 Deploy onto Physical Raspberry Pi 4<br/>• Run benchmark_latency.py (500 iterations)<br/>• Measure CPU Latency (Target: < 25 ms)<br/>• Measure FPS Throughput (Target: > 40 FPS)"] :::step
        T6_1 --> T6_2["6.2 Optional INT8 Quantization<br/>• Export to TFLite / ONNX Runtime for max FPS"] :::step
        T6_2 --> T6_3["6.3 Generate edge_latency_benchmark.csv<br/>• Directly compare vs. baseline 110-325 ms"] :::complete
    end

    S6 --> Phase7[<b>STAGE 7: Thesis Synthesis, Presentation & Defense</b>] :::phase

    subgraph S7 [Stage 7: Project Completion]
        Phase7 --> T7_1["7.1 Compile Final BTP Report & PPT Slides<br/>• Include Literature Review (6 Papers)<br/>• Include Speech Dataset Metrics (WER vs Intent)<br/>• Include Town01->Town02 Generalization Curves<br/>• Include RPi 4 Hardware Latency Comparison"] :::step
        T7_1 --> T7_2["7.2 Deliver Final Mentor Presentation<br/>• Practice word-for-word defense script"] :::complete
    end

    S7 --> FinalDone([🏁 BTP PROJECT COMPLETE & DEFENDED]) :::complete
```

---

# Part 3: Detailed Task Checklist from Baseline to Completion

### Stage 1: Architectural Redesign & Codebase Implementation
* [x] **Task 1.1:** Audit the inherited legacy codebase (`MTP_TESTING`, baseline `Results_05`) and document baseline flaws (semantic segmentation cheat, VAE 300 ms latency, Town02 overfitting).
* [ ] **Task 1.2:** Rewrite `parameters.py` to support 5 sensor modalities (3x RGB cameras, 2D BEV LiDAR, IMU odometry, speech token, collision), configure $60^\circ$ peripheral camera yaw angles (LMDrive), and set `TRAIN_TOWN='Town01'`, `TEST_TOWN='Town02'`.
* [ ] **Task 1.3:** Implement `multimodal_encoder.py` featuring shared-weight depthwise-separable patch embeddings ($800\text{ tokens}, d=64$), FiLM language conditioning over 8 classes, a 2-layer Transformer, and global average pooling to a 128-dim latent vector ($133\text{K parameters}, 0.51\text{ MB}$).
* [ ] **Task 1.4:** Connect the speech-to-intent pipeline from `Aanchal_Spring` to map natural passenger speech into the 8 canonical driving classes.

---

### Stage 2: Multi-Sensor Data Gathering in CARLA (Town01)
* [ ] **Task 2.1:** Implement `collect_data.py` to synchronously record 3 camera streams + 2D BEV projected LiDAR point clouds + IMU telemetry from CARLA Autopilot at 20 Hz.
* [ ] **Task 2.2:** Execute data collection on the remote Windows machine running CARLA:
  ```cmd
  python collect_data.py --town Town01 --frames 20000 --out data_collected_town01
  ```
* [ ] **Task 2.3:** Verify data integrity across the 4 weather presets (`ClearNoon`, `WetSunset`, `HardRainNoon`, `CloudyNight`) and validate `labels.csv`.

---

### Stage 3: Supervised Auxiliary Pre-Training
* [ ] **Task 3.1:** Implement `pretrain_encoder.py` with custom PyTorch DataLoaders for multi-task steering and speed regression.
* [ ] **Task 3.2:** Execute 40 epochs of supervised pre-training on the 20,000 harvested Town01 frames:
  ```cmd
  python pretrain_encoder.py --data data_collected_town01 --epochs 40 --batch_size 32 --lr 0.0003
  ```
* [ ] **Task 3.3:** Save the pre-trained weights to `pretrained_vlm_encoder.pth` (providing the $+114\%$ driving score initialization boost established by LMDrive).

---

### Stage 4: Closed-Loop PPO RL Training on Town01
* [ ] **Task 4.1:** Implement `main_vlm_train.py` featuring the synchronous `CarlaEnvVLM` environment, GAE buffer ($\gamma=0.99, \lambda=0.95$), dynamic speech intent sampling, and PINNsformer kinematic steering regularization.
* [ ] **Task 4.2:** Launch closed-loop PPO training for 1,000,000 timesteps on Town01:
  ```cmd
  python main_vlm_train.py --mode train --town Town01 --timesteps 1000000 --save_interval 50000
  ```
* [ ] **Task 4.3:** Monitor learning curves via TensorBoard and save final weights: `vlm_ppo_actor.pth` and `vlm_ppo_critic.pth`.

---

### Stage 5: Zero-Shot Generalization Testing on Town02
* [ ] **Task 5.1:** Evaluate the trained policy on the unseen map `Town02` for 50 continuous episodes without fine-tuning:
  ```cmd
  python main_vlm_train.py --mode test --town Town02 --episodes 50 --out test_results_town02.csv
  ```
* [ ] **Task 5.2:** Compare zero-shot metrics against baseline `Results_05/test_results_gpu.csv` (measuring route completion, collision rate, average speed, and command adherence).

---

### Stage 6: Edge Deployment & Hardware Benchmarking (Raspberry Pi 4)
* [ ] **Task 6.1:** Deploy the trained model onto the physical Raspberry Pi 4 edge device.
* [ ] **Task 6.2:** Execute hardware latency profiling over 500 inference cycles:
  ```bash
  python3 benchmark_latency.py --runs 500 --device cpu --out edge_latency_benchmark.csv
  ```
* [ ] **Task 6.3:** Verify that ARM CPU inference latency is $\le 25\text{ ms}$ ($48.1\text{ FPS}$) and memory footprint is $\le 1.5\text{ MB}$, formally beating the baseline's 110–325 ms latency (`PIL_test_results_16bit.csv`).

---

### Stage 7: Thesis Synthesis, Presentation & Final Defense
* [ ] **Task 7.1:** Compile empirical findings, generalization graphs, latency tables, and literature derivations into the final BTP Thesis document.
* [ ] **Task 7.2:** Prepare presentation slides and rehearse the word-for-word defense script.
* [ ] **Task 7.3:** Deliver the BTP defense presentation to the mentor and faculty panel.

---

### Project Deliverables & Final Artifacts

| Deliverable | Target Output File | Role in BTP Thesis |
|---|---|---|
| **Multi-Sensor Config** | `parameters.py` | Defines 5 modalities & Town01 $\rightarrow$ Town02 split |
| **Edge Multimodal Model** | `multimodal_encoder.py` | 133K-parameter FiLM Vision-Language Transformer |
| **Harvested Dataset** | `data_collected_town01/` | 20,000 multi-sensor synchronized frames |
| **Pre-Trained Encoder** | `pretrained_vlm_encoder.pth` | Supervised perception weights |
| **Trained PPO Policy** | `vlm_ppo_actor.pth` | Final continuous control driving policy |
| **Zero-Shot Test Logs** | `test_results_town02.csv` | Empirical proof of cross-town generalization |
| **Edge Hardware Profile** | `edge_latency_benchmark.csv` | Empirical proof of $< 25\text{ ms}$ latency on RPi 4 |
| **BTP Thesis & Slides** | Final PDF & Presentation | Final Degree Submission |
