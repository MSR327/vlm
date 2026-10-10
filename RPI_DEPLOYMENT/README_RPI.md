# Raspberry Pi 4 Edge Deployment Package (Processor-In-The-Loop)

This package contains standalone deployment files for the **6 Robust PPO Variants** requested by the mentor, matching the methodologies from **ATLA (ICLR 2021)** and **WocaR-RL (NeurIPS 2022)**.

---

## 1. Directory Structure

```text
RPI_DEPLOYMENT/
├── models/
│   ├── actor_fp16.tflite                # 16-bit Float Quantized Actor Model (~455 KB)
│   ├── actor_int8.tflite                # 8-bit Integer Quantized Actor Model (~232 KB)
│   ├── var_auto_encoder_model_fp16.tflite # 16-bit VAE Encoder Model (~26.2 MB)
│   └── var_auto_encoder_model_int8.tflite # 8-bit VAE Encoder Model (~13.1 MB)
├── PIL_edge_1_sappo.py                  # SA-PPO (State-Adversarial, NeurIPS 2020)
├── PIL_edge_2_radial_ppo.py             # RADIAL-PPO (Certified Adversarial Loss, NeurIPS 2021)
├── PIL_edge_3_atla_ppo.py               # ATLA-PPO (Alternating Training with Learned Adversary, ICLR 2021)
├── PIL_edge_4_pa_atla_ppo.py            # PA-ATLA-PPO (Policy-Adversary ATLA, 2021)
├── PIL_edge_5_sppo.py                   # S-PPO (Smoothed PPO, ICML 2024 / Kumar 2021)
├── PIL_edge_6_wocar_ppo.py              # WocaR-PPO (Worst-Case-Aware Robust PPO, NeurIPS 2022)
└── README_RPI.md                        # Deployment Guide
```

---

## 2. Raspberry Pi Prerequisites

On the Raspberry Pi 4 Model B (Raspberry Pi OS 64-bit / 32-bit):

```bash
# Lightweight installation (recommended, fast & low memory footprint):
pip3 install tflite-runtime numpy

# Alternatively (if full TensorFlow is already installed):
pip3 install tensorflow numpy
```

---

## 3. How to Transfer to Raspberry Pi

From your PC terminal, copy the entire deployment directory to the Raspberry Pi:

```powershell
scp -r D:\SKY-shifted_temprorary\BTP\MTP_TESTING_ORIGINAL\RPI_DEPLOYMENT pi@<RPI_IP_ADDRESS>:~/
```

---

## 4. Running Each Model on the Raspberry Pi

Make sure `PIL_simulation.py` is started on the Simulation PC first. Then run the desired variant on the Pi:

### Variant 1: SA-PPO (NeurIPS 2020)
```bash
python3 PIL_edge_1_sappo.py --sim-ip <PC_IP_ADDRESS> --port 5000 --eps 0.05
```

### Variant 2: RADIAL-PPO (NeurIPS 2021)
```bash
python3 PIL_edge_2_radial_ppo.py --sim-ip <PC_IP_ADDRESS> --port 5000 --bound 0.08
```

### Variant 3: ATLA-PPO (ICLR 2021)
```bash
python3 PIL_edge_3_atla_ppo.py --sim-ip <PC_IP_ADDRESS> --port 5000
```

### Variant 4: PA-ATLA-PPO (Sun et al. 2021)
```bash
python3 PIL_edge_4_pa_atla_ppo.py --sim-ip <PC_IP_ADDRESS> --port 5000
```

### Variant 5: S-PPO (Smoothed DRL, M=5 samples)
```bash
python3 PIL_edge_5_sppo.py --sim-ip <PC_IP_ADDRESS> --port 5000 --samples 5
```

### Variant 6: WocaR-PPO (NeurIPS 2022)
```bash
python3 PIL_edge_6_wocar_ppo.py --sim-ip <PC_IP_ADDRESS> --port 5000
```

---

## 5. Recorded Metrics

Each script logs the exact edge telemetry required by the mentor:
- **$T_{\text{inf}}$:** Hardware inference latency on the ARM Cortex-A72 CPU in milliseconds.
- **$f_{\text{ctrl}}$:** Effective control frequency ($1000 / T_{\text{loop}}$ Hz).
- **Steer & Throttle commands:** Transmitted in real-time over TCP to the CARLA simulation vehicle.
