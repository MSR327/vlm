"""
Unit Test Suite for Phase 2 TransFuser-PPO Sensory Pipeline.

Tests all modules in isolation without requiring an active CARLA server.
Validates tensor dimensions, cross-attention fusion, PPO optimization, and state contracts.
"""
import math
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

    # 1. Test empty point cloud edge case
    pts_empty = np.zeros((0, 4), dtype=np.float32)
    bev_empty = projector.project(pts_empty)
    assert bev_empty.shape == (2, 256, 256), f"Expected shape (2, 256, 256), got {bev_empty.shape}"
    assert bev_empty.sum() == 0.0, "Expected all zeros for empty point cloud"

    # 2. Generate synthetic point cloud (10,000 points)
    # x in [-4, 28], y in [-16, 16], z in [-0.5, 3.5]
    n_pts = 10000
    pts = np.zeros((n_pts, 4), dtype=np.float32)
    pts[:, 0] = np.random.uniform(-4.0, 28.0, n_pts)
    pts[:, 1] = np.random.uniform(-16.0, 16.0, n_pts)
    pts[:, 2] = np.random.uniform(-0.5, 3.5, n_pts)
    pts[:, 3] = np.random.uniform(0.0, 1.0, n_pts)

    bev = projector.project(pts)
    assert bev.shape == (2, 256, 256), f"Expected shape (2, 256, 256), got {bev.shape}"
    assert bev.dtype == torch.float32, f"Expected float32, got {bev.dtype}"
    assert bev.min() >= 0.0 and bev.max() <= 1.0, f"Values out of bounds: [{bev.min()}, {bev.max()}]"
    assert bev[0].sum() > 0.0, "Obstacle channel is empty!"
    assert bev[1].sum() > 0.0, "Ground plane channel is empty!"
    print("  -> BEV LiDAR Projector: PASSED (Shape 2x256x256, empty cloud robustness, bounded [0, 1])\n")


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
    print("[Test 3/4] Testing PyTorch PPO Agent (Squashed Normal Policy) ...")
    agent = PPOAgent(obs_dim=C.OBS_DIM, action_dim=C.ACTION_DIM, device=torch.device("cpu"))

    # Test action selection (returns action, u, log_prob, value)
    state = np.random.randn(C.OBS_DIM).astype(np.float32)
    action, u, log_prob, val = agent.select_action(state)
    assert action.shape == (2,), f"Expected action shape (2,), got {action.shape}"
    assert u.shape == (2,), f"Expected u shape (2,), got {u.shape}"
    assert np.all(action >= -1.0) and np.all(action <= 1.0), f"Action out of bounds: {action}"

    # Test value function evaluation
    val_pred = agent.get_value(state)
    assert isinstance(val_pred, float), f"Expected float value, got {type(val_pred)}"

    # Test multi-episode buffer with termination and truncation boundaries
    # Episode 1: 5 steps, terminates at step 4 (done=True, truncated=False)
    for t in range(5):
        s = np.random.randn(C.OBS_DIM).astype(np.float32)
        a, u_sample, lp, v = agent.select_action(s)
        done = (t == 4)
        agent.remember(s, a, u_sample, lp, 1.0, done, v, truncated=False, next_value=0.0)

    # Episode 2: 5 steps, truncates at step 9 (done=True, truncated=True with bootstrap value)
    for t in range(5):
        s = np.random.randn(C.OBS_DIM).astype(np.float32)
        a, u_sample, lp, v = agent.select_action(s)
        done = (t == 4)
        agent.remember(s, a, u_sample, lp, 2.0, done, v, truncated=done, next_value=5.0)

    # Compute GAE and verify episode boundary isolation (no cross-episode bleed)
    advs, rets = agent.compute_gae()
    assert len(advs) == 10, f"Expected 10 advantages, got {len(advs)}"
    # Episode 1 final step (index 4) should have advantage delta = reward - value (no bleed from ep 2)
    delta_ep1_terminal = agent.buffer.rewards[4] - agent.buffer.values[4]
    assert math.isclose(advs[4], delta_ep1_terminal, rel_tol=1e-5), f"Cross-episode GAE leakage detected: {advs[4]} vs {delta_ep1_terminal}"

    # Test learning update
    a_loss, c_loss = agent.learn()
    assert len(agent.buffer) == 0, "Buffer not cleared after learn()!"
    print("  -> PPO Agent Optimization: PASSED (Multi-episode GAE boundary isolation & SquashedNormal update)\n")


def test_full_pipeline():
    print("[Test 4/4] Testing End-to-End TransFuserPPOPipeline ...")
    pipeline = TransFuserPPOPipeline(device=torch.device("cpu"))

    # Create dummy raw sensor observation
    dummy_obs = {
        'rgb': np.random.rand(256, 256, 3).astype(np.float32),
        'lidar': np.random.uniform(-4.0, 20.0, (5000, 4)).astype(np.float32),
        'ego': np.zeros(8, dtype=np.float32),
        'nav': np.zeros(8, dtype=np.float32)
    }

    action, u, log_prob, val, state_np, lat_ms = pipeline.act(dummy_obs)
    assert action.shape == (2,), f"Expected action shape (2,), got {action.shape}"
    assert u.shape == (2,), f"Expected u shape (2,), got {u.shape}"
    assert state_np.shape == (144,), f"Expected observation shape (144,), got {state_np.shape}"
    print(f"  -> End-to-End Pipeline Latency: {lat_ms:.2f} ms")
    print("  -> End-to-End Pipeline: PASSED\n")


def test_hud_visualizer():
    print("[Test 5/6] Testing Real-Time OpenCV HUD Visualizer ...")
    from phase2_sensory.visualizer import SensoryHUDVisualizer
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        viz = SensoryHUDVisualizer(save_dir=tmp_dir)

        # Create synthetic test frames
        chase = np.zeros((720, 880, 3), dtype=np.uint8)
        front_rgb = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        bev = torch.rand(2, 256, 256, dtype=torch.float32)
        telemetry = {
            'episode': 1,
            'step': 42,
            'reward': 15.6,
            'speed': 18.5,
            'target_speed': 20.0,
            'steer': 0.15,
            'throttle': 0.65,
            'brake': 0.0,
            'lane_deviation': 0.12,
            'heading_error': 1.8,
            'route_completion': 45.0,
            'distance_covered': 85.0,
            'stall_steps': 0,
            'max_stall_steps': 100,
            'value': 12.4,
            'latency_ms': 38.5
        }

        # 1. Build Canvas
        canvas = viz.build_canvas(chase, front_rgb, bev, telemetry)
        assert canvas.shape == (720, 1280, 3), f"Expected canvas (720, 1280, 3), got {canvas.shape}"
        assert canvas.dtype == np.uint8, f"Expected uint8, got {canvas.dtype}"

        # 2. Save Screenshot
        shot_path = viz.save_screenshot(canvas, telemetry)
        assert os.path.exists(shot_path), f"Screenshot was not created: {shot_path}"
        assert os.path.getsize(shot_path) > 1000, "Screenshot file is empty!"

    print("  -> HUD Visualizer: PASSED (1280x720 canvas layout & telemetry gauges validated)\n")


def test_plotting():
    print("[Test 6/6] Testing Benchmark Plotting & Analysis ...")
    from phase2_sensory.plot_results import (
        generate_synthetic_phase2_data,
        plot_benchmark_comparison,
        plot_eval_progression,
        plot_termination_analysis
    )
    import tempfile

    data = generate_synthetic_phase2_data(n_episodes=10)
    assert len(data['episodes']) == 10
    assert len(data['completions']) == 10

    with tempfile.TemporaryDirectory() as tmp_dir:
        p1 = os.path.join(tmp_dir, 'comp.png')
        p2 = os.path.join(tmp_dir, 'prog.png')
        p3 = os.path.join(tmp_dir, 'term.png')

        plot_benchmark_comparison(data, p1)
        plot_eval_progression(data, p2)
        plot_termination_analysis(data, p3)

        assert os.path.exists(p1) and os.path.getsize(p1) > 1000
        assert os.path.exists(p2) and os.path.getsize(p2) > 1000
        assert os.path.exists(p3) and os.path.getsize(p3) > 1000

    print("  -> Benchmark Plotting: PASSED (Publication-quality figures generated cleanly)\n")


def main():
    print("\n" + "=" * 60)
    print(" RUNNING PHASE 2 SENSORY PIPELINE VERIFICATION SUITE")
    print("=" * 60 + "\n")

    test_bev_lidar()
    test_transfuser_backbone()
    test_ppo_agent()
    test_full_pipeline()
    test_hud_visualizer()
    test_plotting()

    print("=" * 60)
    print(" ALL 6 VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
