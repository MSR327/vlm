# Operational Execution Manual: Step-by-Step Guide

This manual details the step-by-step procedure to execute the complete **Multimodal VLM-PPO Autonomous Driving** pipeline.

---

## 🗺️ Execution Overview

| Phase | Description | Machine | Estimated Time |
|---|---|---|---|
| **Phase 1** | Script Creation & Verification | Local Mac | ✅ Completed |
| **Phase 2** | Multi-Sensor Data Collection (Town01) | Remote Desktop (CARLA) | ~15–20 minutes |
| **Phase 3** | Domain Adaptation Pre-Training | Remote Desktop (GPU) | ~15–25 minutes |
| **Phase 4** | PPO Reinforcement Learning Training | Remote Desktop (GPU) | ~4–6 hours (Overnight) |
| **Phase 5** | Zero-Shot Generalization Benchmark (Town02) | Remote Desktop (GPU) | ~20–30 minutes |
| **Phase 6** | Raspberry Pi PIL Edge Evaluation | Raspberry Pi + GPU | ~1–2 hours |

---

## 🛠️ Step-by-Step Execution Instructions

### STEP 1: Transfer Updated Files to Remote Windows Desktop (via AnyDesk)
In AnyDesk File Manager:
* **Left Window (Your Mac):** `/Users/msr/claude_code/BTP_Project/MTP_TESTING/`
* **Right Window (Remote PC):** `C:\Users\User\Documents\SKY\MTP_TESTING\`
* **Transfer these 5 updated/new files:**
  1. `parameters.py`
  2. `multimodal_encoder.py`
  3. `collect_data.py`
  4. `pretrain_encoder.py`
  5. `main_vlm_train.py`
  6. `benchmark_latency.py`

---

### STEP 2: Multi-Sensor Data Collection in CARLA (Town01)
1. **Start CARLA Simulator** on the Remote Windows PC:
   ```cmd
   CarlaUE4.exe
   ```
2. **Open Command Prompt / Terminal** in the project folder and run:
   ```cmd
   python collect_data.py --town Town01 --frames 20000 --out data_collected_town01
   ```
   * *What happens:* Spawns ego-vehicle with Autopilot, connects 3 RGB cameras + LiDAR, drives across Town01 through 4 randomized weather conditions, and records 20,000 synchronized `.npz` multi-sensor frames and `labels.csv`.
   * *Duration:* ~15 minutes at 20 FPS.

---

### STEP 3: Domain Fine-Tuning / Pre-Training of Multimodal Encoder
Once data collection finishes:
```cmd
python pretrain_encoder.py --data data_collected_town01 --epochs 40 --batch_size 32
```
* *What happens:* Trains the `MultimodalEdgeEncoder` on multi-task steering angle prediction and speed prediction. Instills spatial "road sense" and obstacle distance awareness.
* *Output:* Saves fine-tuned weights to `models/pretrained_vlm_encoder.pth`.
* *Duration:* ~15–20 minutes on GPU.

---

### STEP 4: PPO Policy Training on Town01
Start continuous PPO reinforcement learning:
```cmd
python main_vlm_train.py --mode train --town Town01 --timesteps 1000000
```
* *What happens:* The car learns closed-loop lane keeping, speed regulation, and speech-command following using the pre-trained perception backbone.
* *Checkpoints:* Automatically saved to `Results_VLM/checkpoints/` every 50 episodes.
* *Logging:* Live curves logged to TensorBoard (`tensorboard --logdir Results_VLM/runs/train`).
* *Duration:* 4–6 hours (can be left running overnight).

---

### STEP 5: Zero-Shot Generalization Benchmark on Town02
Evaluate the trained policy on unseen **Town02** routes:
```cmd
python main_vlm_train.py --mode test --town Town02 --episodes 50
```
* *What happens:* Runs 50 test episodes across diverse commands (`shift_left_lane`, `slow_down`, `keep_lane`, etc.) without exploration noise.
* *Output:* Produces `Results_VLM/test_results_town02.csv` with Route Completion %, Infraction Score, Average Speed, and Latency.
* *Comparison:* Directly compare these numbers against baseline `Results_05/test_results_gpu.csv` to prove superior generalization!

---

### STEP 6: Hardware Latency Profiling
Run the latency benchmark to measure the GPU/CPU inference speed:
```cmd
python benchmark_latency.py --iters 500
```
* *Screenshot the results* for the BTP Report and presentation tables!
