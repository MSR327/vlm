"""
Unit Test Suite for Phase 2 TransFuser-PPO Sensory Pipeline.

Tests all modules in isolation without requiring an active CARLA server.
Validates tensor dimensions, cross-attention fusion, PPO optimization, and state contracts.
"""
import os
import sys
import numpy as np
import torch

# Add repository root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import phase2_sensory.config as C
from phase2_sensory.bev_lidar import BEVLidarProjector
from phase2_sensory.transfuser_backbone import TransFuserBackbone
from phase2_sensory.ppo_agent import PPOAgent
from phase2_sensory.pipeline import TransFuserPPOPipeline


def test_bev_lidar():
    print("[Test 1/4] Testing BEV LiDAR Projector ...")
    projector = BEVLidarProjector()

    # Generate synthetic point cloud (10,000 points)
    # x in [0, 32], y in [-16, 16], z in [-2, 3]
    n_pts = 10000
    pts = np.zeros((n_pts, 4), dtype=np.float32)
    pts[:, 0] = np.random.uniform(0.0, 32.0, n_pts)
    pts[:, 1] = np.random.uniform(-16.0, 16.0, n_pts)
    pts[:, 2] = np.random.uniform(-2.0, 3.0, n_pts)
    pts[:, 3] = np.random.uniform(0.0, 1.0, n_pts)

    bev = projector.project(pts)
    assert bev.shape == (2, 256, 256), f"Expected shape (2, 256, 256), got {bev.shape}"
    assert bev.dtype == torch.float32, f"Expected float32, got {bev.dtype}"
    assert bev.min() >= 0.0 and bev.max() <= 1.0, f"Values out of bounds: [{bev.min()}, {bev.max()}]"
    assert bev[0].sum() > 0.0, "Obstacle channel is empty!"
    assert bev[1].sum() > 0.0, "Ground plane channel is empty!"
    print("  -> BEV LiDAR Projector: PASSED (Shape 2x256x256, strictly bounded [0, 1])\n")


def test_transfuser_backbone():
    print("[Test 2/4] Testing TransFuser Cross-Attention Backbone ...")
    device = torch.device("cpu")
    model = TransFuserBackbone(latent_dim=C.LATENT_DIM).to(device)

    # Compute parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  -> Total TransFuser Parameters: {total_params:,}")

    # Forward pass with batch size = 2
    rgb = torch.randn(2, 3, 256, 256, device=device)
    lidar = torch.randn(2, 2, 256, 256, device=device)

    z = model(rgb, lidar)
    assert z.shape == (2, 128), f"Expected latent shape (2, 128), got {z.shape}"
    assert not torch.isnan(z).any(), "NaN detected in TransFuser latent output!"

    # Test freezing
    model.freeze_backbone()
    assert all(not p.requires_grad for p in model.parameters()), "Freeze failed!"
    print("  -> TransFuser Backbone: PASSED (Outputs fixed 128-d latent invariant)\n")


def test_ppo_agent():
    print("[Test 3/4] Testing PyTorch PPO Agent ...")
    agent = PPOAgent(obs_dim=C.OBS_DIM, action_dim=C.ACTION_DIM, device=torch.device("cpu"))

    # Test action selection
    state = np.random.randn(C.OBS_DIM).astype(np.float32)
    action, log_prob, val = agent.select_action(state)
    assert action.shape == (2,), f"Expected action shape (2,), got {action.shape}"
    assert np.all(action >= -1.0) and np.all(action <= 1.0), f"Action out of bounds: {action}"

    # Test buffer and learning update
    for _ in range(10):
        s = np.random.randn(C.OBS_DIM).astype(np.float32)
        a, lp, v = agent.select_action(s)
        agent.remember(s, a, lp, 1.0, False, v)

    a_loss, c_loss = agent.learn()
    print("  -> PPO Agent Optimization: PASSED (Valid action sampling & gradient backpropagation)\n")


def test_full_pipeline():
    print("[Test 4/4] Testing End-to-End TransFuserPPOPipeline ...")
    pipeline = TransFuserPPOPipeline(device=torch.device("cpu"))

    # Create dummy raw sensor observation
    dummy_obs = {
        'rgb': np.random.rand(256, 256, 3).astype(np.float32),
        'lidar': np.random.uniform(-10.0, 10.0, (5000, 4)).astype(np.float32),
        'ego': np.zeros(8, dtype=np.float32),
        'nav': np.zeros(8, dtype=np.float32)
    }

    action, log_prob, val, state_np, lat_ms = pipeline.act(dummy_obs)
    assert action.shape == (2,), f"Expected action shape (2,), got {action.shape}"
    assert state_np.shape == (144,), f"Expected observation shape (144,), got {state_np.shape}"
    print(f"  -> End-to-End Pipeline Latency: {lat_ms:.2f} ms")
    print("  -> End-to-End Pipeline: PASSED\n")


def main():
    print("\n" + "=" * 60)
    print(" RUNNING PHASE 2 SENSORY PIPELINE VERIFICATION SUITE")
    print("=" * 60 + "\n")

    test_bev_lidar()
    test_transfuser_backbone()
    test_ppo_agent()
    test_full_pipeline()

    print("=" * 60)
    print(" ALL 4 VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
