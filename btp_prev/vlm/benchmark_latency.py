"""
benchmark_latency.py
=============================================================================
Hardware Latency & Throughput Benchmark for Multimodal VLM-PPO Perception Stack.
Measures forward pass latency for:
- 4x RGB Cameras (Front, Left, Right, Rear) — 360° Surround
- 1x 2D BEV LiDAR Grid (Asymmetric -15m to +35m)
- 1x Spoken Command FiLM Conditioning
- 1x Scene Context FiLM Conditioning
- 1x Kinematic Telemetry Projection
- 1x PPO Actor Network Forward Pass

Usage:
    python benchmark_latency.py --iters 500
=============================================================================
"""

import time
import argparse
import numpy as np
import torch
import torch.nn as nn

from multimodal_encoder import MultimodalEdgeEncoder
from parameters import VLM_LATENT_DIM, NUM_CAMERAS, OBSERVATION_DIM, ACTION_DIM


class BenchmarkActor(nn.Module):
    def __init__(self, state_dim=OBSERVATION_DIM, action_dim=ACTION_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 500),
            nn.Tanh(),
            nn.Linear(500, 300),
            nn.Tanh(),
            nn.Linear(300, 100),
            nn.Tanh(),
            nn.Linear(100, action_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)


def run_benchmark(num_iters=500):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n===============================================================")
    print(f" 🚀 MULTIMODAL VLM-PPO HARDWARE LATENCY BENCHMARK")
    print(f"===============================================================")
    print(f" Target Device: {device}")
    print(f" Sensor Suite:  {NUM_CAMERAS} Cameras + 1 LiDAR BEV = {NUM_CAMERAS + 1} views")
    if device.type == "cuda":
        print(f" GPU Device: {torch.cuda.get_device_name(0)}")

    encoder = MultimodalEdgeEncoder(latent_dim=VLM_LATENT_DIM, num_cameras=NUM_CAMERAS, use_lidar=True).to(device)
    actor = BenchmarkActor(state_dim=OBSERVATION_DIM, action_dim=ACTION_DIM).to(device)

    encoder.eval()
    actor.eval()

    total_params = sum(p.numel() for p in encoder.parameters()) + sum(p.numel() for p in actor.parameters())
    encoder_params = sum(p.numel() for p in encoder.parameters())
    print(f" Encoder Params: {encoder_params:,} ({encoder_params * 4 / (1024*1024):.2f} MB)")
    print(f" Total Params:   {total_params:,} ({total_params * 4 / (1024*1024):.2f} MB)")

    # Synthetic multi-sensor inputs (4 Cameras + 1 LiDAR BEV)
    front_t = torch.randn(1, 3, 80, 160, device=device)
    left_t  = torch.randn(1, 3, 80, 160, device=device)
    right_t = torch.randn(1, 3, 80, 160, device=device)
    rear_t  = torch.randn(1, 3, 80, 160, device=device)
    lidar_t = torch.randn(1, 1, 80, 160, device=device)
    cmd     = "shift_left_lane"
    telem_t = torch.randn(1, 5, device=device)
    scene_t = torch.randn(1, 64, device=device)  # Scene context from System 2

    # Warmup
    print(" Warming up hardware...")
    with torch.no_grad():
        for _ in range(50):
            obs = encoder(front_t, left_t, right_t, rear_t, lidar_t, cmd, telem_t, scene_context=scene_t)
            act = actor(obs)

    # Benchmarking
    print(f" Profiling {num_iters} end-to-end forward passes...")
    latencies = []

    with torch.no_grad():
        for _ in range(num_iters):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            obs = encoder(front_t, left_t, right_t, rear_t, lidar_t, cmd, telem_t, scene_context=scene_t)
            act = actor(obs)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            latencies.append((t1 - t0) * 1000.0) # ms

    latencies = np.array(latencies)
    avg_latency = np.mean(latencies)
    p50_latency = np.percentile(latencies, 50)
    p95_latency = np.percentile(latencies, 95)
    p99_latency = np.percentile(latencies, 99)
    fps = 1000.0 / avg_latency

    print(f"\n===============================================================")
    print(f" 📊 BENCHMARK RESULTS ({NUM_CAMERAS} Cameras + LiDAR, Dual FiLM)")
    print(f"===============================================================")
    print(f" Average Latency: {avg_latency:.3f} ms")
    print(f" Median (p50):    {p50_latency:.3f} ms")
    print(f" 95th Percentile: {p95_latency:.3f} ms")
    print(f" 99th Percentile: {p99_latency:.3f} ms")
    print(f" Throughput:      {fps:.1f} FPS (Target: > 20 FPS)")
    print(f" Observation Dim: {obs.shape[-1]} (Latent: {VLM_LATENT_DIM} + Nav: 5)")
    print(f" Real-Time Grade: {'✅ SOTA EDGE READY' if avg_latency < 20.0 else '⚠️ GPU ONLY'}")
    print(f"===============================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=500, help="Number of benchmark iterations")
    args = parser.parse_args()
    run_benchmark(args.iters)
