"""
Invariant tests. These run with no CARLA, no dataset and no GPU.

Every module in `ladder` must be reachable from here (brief section 9, rule 4).
Most of these encode a specific defect found in the audit of the previous
codebase; the docstrings say which.
"""

from __future__ import annotations

import unittest

import numpy as np
import torch

from ladder import config as C
from ladder.contracts import (
    DrivingIntent, EgoState, LaneAction, NavState, PrivilegedState, SensorObs,
    assert_no_privileged_leak,
)
from ladder.intent.oracle import ConstantIntentSource, OracleIntentSource
from ladder.models.backbones import AVAILABLE, build_backbone
from ladder.models.encoder import (
    BlindEncoder, FastPathwayEncoder, check_parameter_match, solve_capacity_match,
)
from ladder.models.fusion import FUSIONS, build_fusion
from ladder.models.heads import (
    DrivableBEVHead, InstructionCompleteHead, TrafficLightHead, WaypointHead,
)
from ladder.models.lidar_tokenizer import BEVTokenizer
from ladder.sensors.bev import rasterise_bev

B = 2


def _ego() -> EgoState:
    return EgoState(speed=5.0, yaw_rate=0.1, accel_long=0.3, accel_lat=0.0,
                    prev_action=np.zeros(2, np.float32), speed_hist=np.zeros(2, np.float32))


def _nav(dist: float = 30.0) -> NavState:
    return NavState(target_point=np.array([dist, 2.0], np.float32), command="left")


def _priv(**over) -> PrivilegedState:
    base = dict(dist_from_lane_centre=0.2, heading_error=0.05, lane_id=1, is_junction=False,
                collision=False, lane_invasion=False, red_light_violation=False,
                off_road=False, blocked_seconds=0.0, route_completion=0.5,
                distance_along_route=100.0,
                expert_future_xy=np.zeros((C.N_WAYPOINTS, 2), np.float32),
                traffic_light_state=0)
    base.update(over)
    return PrivilegedState(**base)


class TestObservationContract(unittest.TestCase):
    """The leak that made every previous measurement uninterpretable."""

    def test_structural_leak_audit(self):
        assert_no_privileged_leak()

    def test_privileged_state_cannot_become_a_policy_input(self):
        self.assertFalse(hasattr(PrivilegedState, "to_policy_vector"))
        self.assertFalse(hasattr(PrivilegedState, "to_vector"))

    def test_policy_vector_width_is_fixed(self):
        obs = SensorObs(
            rgb={v: np.zeros((C.N_FRAMES, 3, C.IMG_SIZE, C.IMG_SIZE), np.float32)
                 for v in C.VIEW_NAMES},
            lidar_points=np.zeros((10, 4), np.float32), ego=_ego(), nav=_nav())
        z = np.zeros(C.D_LATENT, np.float32)
        self.assertEqual(obs.to_policy_vector(z).shape, (C.DIM_POLICY_OBS_STAGE1,))
        tau = np.zeros(C.DIM_TAU, np.float32)
        self.assertEqual(obs.to_policy_vector(z, tau).shape, (C.DIM_POLICY_OBS_STAGE2,))

    def test_close_nav_target_is_rejected(self):
        """Closer than 20 m the target point stops being a hint and becomes the answer."""
        with self.assertRaises(ValueError):
            NavState(target_point=np.array([5.0, 0.0], np.float32), command="follow")

    def test_unknown_nav_command_is_rejected(self):
        with self.assertRaises(ValueError):
            NavState(target_point=np.array([30.0, 0.0], np.float32), command="u_turn")

    def test_wrong_image_shape_is_rejected(self):
        with self.assertRaises(ValueError):
            SensorObs(rgb={v: np.zeros((C.N_FRAMES, 3, 80, 160), np.float32)
                           for v in C.VIEW_NAMES},
                      lidar_points=np.zeros((10, 4), np.float32), ego=_ego(), nav=_nav())


class TestBEV(unittest.TestCase):
    def test_shape_and_two_bins_only(self):
        """Two channels: below and above ground. No goal channel -- see bev.py."""
        pts = np.random.default_rng(0).uniform(-20, 20, (2000, 4)).astype(np.float32)
        g = rasterise_bev(pts)
        self.assertEqual(g.shape, (C.BEV_BINS, C.BEV_PIXELS, C.BEV_PIXELS))
        self.assertEqual(g.shape[0], 2)

    def test_empty_cloud_is_not_an_error(self):
        g = rasterise_bev(np.zeros((0, 4), np.float32))
        self.assertEqual(g.shape, (C.BEV_BINS, C.BEV_PIXELS, C.BEV_PIXELS))
        self.assertEqual(float(g.max()), 0.0)

    def test_ground_split_is_respected(self):
        below = np.zeros((100, 4), np.float32); below[:, 2] = C.BEV_GROUND_Z - 1.0
        above = np.zeros((100, 4), np.float32); above[:, 2] = C.BEV_GROUND_Z + 1.0
        self.assertEqual(float(rasterise_bev(below)[1].max()), 0.0)
        self.assertEqual(float(rasterise_bev(above)[0].max()), 0.0)

    def test_malformed_input_raises(self):
        with self.assertRaises(ValueError):
            rasterise_bev(np.zeros((10, 3), np.float32))


class TestBackbones(unittest.TestCase):
    def test_registry_is_non_empty(self):
        self.assertIn("scratch_tiny", AVAILABLE)

    def test_unknown_backbone_raises(self):
        with self.assertRaises(ValueError):
            build_backbone("resnet9000")

    def test_frozen_by_default(self):
        bb = build_backbone("scratch_tiny")
        self.assertEqual(sum(p.numel() for p in bb.parameters() if p.requires_grad), 0)

    def test_missing_dependency_raises_rather_than_substituting(self):
        """The legacy SceneContextEncoder silently swapped in a random CNN instead."""
        try:
            import torchvision  # noqa: F401
            self.skipTest("torchvision present; cannot exercise the failure path here")
        except ImportError:
            with self.assertRaises(ImportError):
                build_backbone("resnet34")


class TestFusion(unittest.TestCase):
    def test_every_arm_emits_the_invariant_width(self):
        sets = [torch.randn(B, 512, C.D_MODEL), torch.randn(B, 196, C.D_MODEL),
                torch.randn(B, 8, C.D_MODEL)]
        for name in FUSIONS:
            with self.subTest(fusion=name):
                self.assertEqual(build_fusion(name)(sets).shape, (B, C.D_LATENT))

    def test_unknown_fusion_raises(self):
        with self.assertRaises(ValueError):
            build_fusion("magic")


class TestEncoder(unittest.TestCase):
    CFGS = {
        "R1": dict(backbone="scratch_tiny", n_views=1, use_lidar=False, use_language=False),
        "R2": dict(backbone="scratch_tiny", n_views=4, use_lidar=False, use_language=False),
        "R3": dict(backbone="scratch_tiny", n_views=4, use_lidar=True, use_language=False),
        "R4": dict(backbone="scratch_tiny", n_views=4, use_lidar=True, use_language=True),
    }

    @staticmethod
    def _inputs(kw):
        views = C.VIEW_NAMES[:1] if kw["n_views"] == 1 else C.VIEW_NAMES
        rgb = {v: torch.rand(B, C.N_FRAMES, 3, C.IMG_SIZE, C.IMG_SIZE) for v in views}
        bev = torch.rand(B, C.BEV_BINS, C.BEV_PIXELS, C.BEV_PIXELS) if kw["use_lidar"] else None
        ins = torch.randn(B, C.DIM_TEXT) if kw["use_language"] else None
        return rgb, bev, ins

    def test_every_config_emits_the_invariant_width(self):
        for name, kw in self.CFGS.items():
            with self.subTest(config=name):
                enc = FastPathwayEncoder(**kw)
                z = enc(*self._inputs(kw))
                self.assertEqual(z.shape, (B, C.D_LATENT))

    def test_blind_encoder_matches_the_same_width(self):
        z = BlindEncoder()(torch.randn(B, C.DIM_EGO), torch.randn(B, C.DIM_NAV))
        self.assertEqual(z.shape, (B, C.D_LATENT))

    def test_backbone_stays_frozen(self):
        enc = FastPathwayEncoder(**self.CFGS["R3"])
        self.assertEqual(
            sum(p.numel() for p in enc.backbone.parameters() if p.requires_grad), 0)

    def test_missing_lidar_raises_rather_than_zero_filling(self):
        enc = FastPathwayEncoder(**self.CFGS["R3"])
        rgb, _, _ = self._inputs(self.CFGS["R3"])
        with self.assertRaises(ValueError):
            enc(rgb, None, None)

    def test_supplying_lidar_to_a_camera_only_config_raises(self):
        enc = FastPathwayEncoder(**self.CFGS["R2"])
        rgb, _, _ = self._inputs(self.CFGS["R2"])
        bev = torch.rand(B, C.BEV_BINS, C.BEV_PIXELS, C.BEV_PIXELS)
        with self.assertRaises(ValueError):
            enc(rgb, bev, None)

    def test_capacity_solver_satisfies_the_matching_invariant(self):
        solved = solve_capacity_match(self.CFGS)
        solved.pop("_target")
        reports = [FastPathwayEncoder(fusion_mlp_ratio=solved[k]["mlp_ratio"],
                                      **kw).parameter_report()
                   for k, kw in self.CFGS.items()]
        self.assertTrue(check_parameter_match(reports)["within_tolerance"])

    def test_unmatched_configs_fail_the_invariant(self):
        """Guards the guard: the check must be able to fail."""
        reports = [FastPathwayEncoder(**kw).parameter_report() for kw in self.CFGS.values()]
        self.assertFalse(check_parameter_match(reports)["within_tolerance"])


class TestHeads(unittest.TestCase):
    def test_shapes(self):
        z = torch.randn(B, C.D_LATENT)
        self.assertEqual(WaypointHead()(z).shape, (B, C.N_WAYPOINTS, 2))
        self.assertEqual(TrafficLightHead()(z).shape, (B, 4))
        self.assertEqual(DrivableBEVHead()(z).shape, (B, 32, 32))
        self.assertEqual(InstructionCompleteHead()(z).shape, (B,))

    def test_bev_tokenizer(self):
        t = BEVTokenizer()(torch.rand(B, C.BEV_BINS, C.BEV_PIXELS, C.BEV_PIXELS))
        self.assertEqual(t.shape, (B, C.TOKENS_LIDAR, C.D_MODEL))


class TestIntentSeam(unittest.TestCase):
    def test_latch_and_staleness(self):
        src = OracleIntentSource(rate_hz=2.0)
        src.update(_priv(), "left", now=0.0)
        self.assertFalse(src.is_stale(0.1))
        self.assertTrue(src.is_stale(1.0))

    def test_get_before_update_raises(self):
        """No plausible-looking default: that is the failure mode the seam prevents."""
        with self.assertRaises(RuntimeError):
            OracleIntentSource().get(0.0)

    def test_red_light_forces_stop(self):
        i = OracleIntentSource().update(_priv(traffic_light_state=1), "left", 0.0)
        self.assertIs(i.lane_action, LaneAction.STOP)
        self.assertEqual(i.speed_cap, 0.0)

    def test_constant_arm_ignores_state(self):
        i = ConstantIntentSource().update(_priv(traffic_light_state=1), "left", 0.0)
        self.assertIs(i.lane_action, LaneAction.FOLLOW)

    def test_malformed_waypoints_rejected(self):
        with self.assertRaises(ValueError):
            DrivingIntent(target_waypoints=np.zeros((2, 2), np.float32),
                          lane_action=LaneAction.FOLLOW, speed_cap=5.0, hazard=False,
                          instruction_done=False, valid_until=1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
