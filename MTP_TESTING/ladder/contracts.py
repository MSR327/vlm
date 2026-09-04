"""
The observation contract.

The previous environment leaked the answer into the observation: lane-centre
distance and heading error were computed from CARLA's road graph, concatenated
into the policy input after the transformer, and then used to compute the
reward. A five-input MLP maximised that reward with every camera unplugged.

This module makes that class of bug a type error rather than a review comment.

    SensorObs        Tier S -- policy-visible. Could a production vehicle obtain
                     this from its own hardware, with no map oracle?
    PrivilegedState  Tier P -- reward, metrics and training labels ONLY.
    (Tier F)         Forbidden anywhere. Enforced by absence: there is no field
                     for it on either record.

Only SensorObs carries `to_policy_vector`. PrivilegedState has no method that
produces a policy input and must never acquire one. The rule that makes Tier P
usable: privileged information may be a training LABEL; it may never be a
deployed INPUT. The test is mechanical -- trace whether the quantity appears in
the forward pass at evaluation time.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Optional

import numpy as np

from . import config as C


class LaneAction(Enum):
    FOLLOW = "follow"
    LEFT = "left"
    RIGHT = "right"
    STOP = "stop"


# ---------------------------------------------------------------------------
# Tier S -- policy-visible
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EgoState:
    """Proprioception. Everything here comes from the vehicle's own sensors."""
    speed: float                    # m/s, wheel encoder / IMU
    yaw_rate: float                 # rad/s, IMU
    accel_long: float               # m/s^2, IMU
    accel_lat: float                # m/s^2, IMU
    prev_action: np.ndarray         # (2,) the command the vehicle last issued
    speed_hist: np.ndarray          # (2,) speed at t-0.2s, t-0.4s

    def to_vector(self) -> np.ndarray:
        v = np.concatenate([
            np.array([self.speed, self.yaw_rate, self.accel_long, self.accel_lat],
                     dtype=np.float32),
            np.asarray(self.prev_action, dtype=np.float32).reshape(2),
            np.asarray(self.speed_hist, dtype=np.float32).reshape(2),
        ])
        assert v.shape == (C.DIM_EGO,), f"ego vector is {v.shape}, expected ({C.DIM_EGO},)"
        return v


@dataclass(frozen=True)
class NavState:
    """
    Output of a navigation system, not of the map oracle.

    A real car has GPS and a route; every CARLA Leaderboard agent receives a
    sparse route plus a high-level command. What makes this a hint rather than
    the answer is the sampling distance (25-40 m) and the injected noise. A
    target point closer than 20 m becomes the answer and must be rejected.
    """
    target_point: np.ndarray        # (2,) ego frame, metres, noisy
    command: str                    # one of C.NAV_COMMANDS

    def __post_init__(self):
        if self.command not in C.NAV_COMMANDS:
            raise ValueError(
                f"unknown nav command {self.command!r}; expected one of {C.NAV_COMMANDS}")
        d = float(np.linalg.norm(np.asarray(self.target_point, dtype=np.float64)))
        if d < 20.0:
            raise ValueError(
                f"nav target point is {d:.1f} m away. Closer than 20 m it stops being a "
                f"navigation hint and becomes the answer the perception is supposed to "
                f"compute. Sample it in [{C.NAV_TARGET_MIN_M}, {C.NAV_TARGET_MAX_M}] m.")

    def to_vector(self) -> np.ndarray:
        onehot = np.zeros(len(C.NAV_COMMANDS), dtype=np.float32)
        onehot[C.NAV_COMMANDS.index(self.command)] = 1.0
        v = np.concatenate([np.asarray(self.target_point, dtype=np.float32).reshape(2), onehot])
        assert v.shape == (C.DIM_NAV,), f"nav vector is {v.shape}, expected ({C.DIM_NAV},)"
        return v


@dataclass(frozen=True)
class SensorObs:
    """Everything the policy is allowed to see. Nothing else reaches it."""
    rgb: dict                       # view name -> (N_FRAMES, 3, 224, 224) float32 in [0,1], RGB
    lidar_points: np.ndarray        # (N, 4) raw x,y,z,intensity -- rasterised model-side
    ego: EgoState
    nav: NavState
    instruction_embedding: Optional[np.ndarray] = None   # (128,) cached lookup, language configs

    def __post_init__(self):
        missing = set(C.VIEW_NAMES) - set(self.rgb)
        if missing:
            raise ValueError(f"SensorObs is missing camera views: {sorted(missing)}")
        for name, arr in self.rgb.items():
            expected = (C.N_FRAMES, 3, C.IMG_SIZE, C.IMG_SIZE)
            if tuple(arr.shape) != expected:
                raise ValueError(f"rgb[{name!r}] has shape {tuple(arr.shape)}, expected {expected}")
        if self.lidar_points.ndim != 2 or self.lidar_points.shape[1] != 4:
            raise ValueError(
                f"lidar_points has shape {tuple(self.lidar_points.shape)}, expected (N, 4)")

    def to_policy_vector(self, z: np.ndarray, tau_nom: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Assemble the PPO observation. This is the ONLY path from an observation
        record to a policy input, and PrivilegedState has no equivalent.
        """
        z = np.asarray(z, dtype=np.float32).reshape(-1)
        if z.shape != (C.D_LATENT,):
            raise ValueError(f"latent is {z.shape}, expected ({C.D_LATENT},)")
        parts = [z, self.ego.to_vector(), self.nav.to_vector()]
        if tau_nom is not None:
            tau = np.asarray(tau_nom, dtype=np.float32).reshape(-1)
            if tau.shape != (C.DIM_TAU,):
                raise ValueError(f"tau_nom is {tau.shape}, expected ({C.DIM_TAU},)")
            parts.append(tau)
        v = np.concatenate(parts)
        expected = C.DIM_POLICY_OBS_STAGE2 if tau_nom is not None else C.DIM_POLICY_OBS_STAGE1
        assert v.shape == (expected,), f"policy obs is {v.shape}, expected ({expected},)"
        return v


# ---------------------------------------------------------------------------
# Tier P -- reward, metrics and training labels only
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PrivilegedState:
    """
    Simulator ground truth. Reachable from the reward function, the metric
    logger and the offline label generator.

    Deliberately has NO to_policy_vector and must never gain one. If you find
    yourself wanting to add one, the quantity you want belongs in SensorObs and
    has to be justified against the Tier S membership test.
    """
    # lane geometry -- the previous leak lived here
    dist_from_lane_centre: float
    heading_error: float
    lane_id: int
    is_junction: bool

    # events
    collision: bool
    lane_invasion: bool
    red_light_violation: bool
    off_road: bool
    blocked_seconds: float

    # route
    route_completion: float         # [0, 1]
    distance_along_route: float     # metres

    # labels for Stage 1 supervision (never inputs at evaluation time)
    expert_future_xy: np.ndarray    # (K, 2) ego frame -- the waypoint label
    traffic_light_state: int        # 0 none / 1 red / 2 yellow / 3 green
    drivable_bev: Optional[np.ndarray] = None   # (H, W) occupancy label
    actors: tuple = field(default_factory=tuple)

    def reward_inputs(self) -> dict:
        return {
            "route_completion": self.route_completion,
            "collision": self.collision,
            "red_light_violation": self.red_light_violation,
            "off_road": self.off_road,
            "blocked_seconds": self.blocked_seconds,
        }


# ---------------------------------------------------------------------------
# The slow -> fast seam. Infrastructure, not a policy input in this phase.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DrivingIntent:
    """
    What the slow pathway hands the fast pathway.

    Typed rather than a latent blob so that it can be logged, unit-tested,
    produced by a scripted oracle, and ablated by swapping in a constant. The
    first and only adapter for now is the oracle in ladder.intent.oracle.

    NOTE: the oracle reads Tier P. Anything derived from it that reaches the
    policy at evaluation time is therefore Tier F. During this phase
    DrivingIntent is infrastructure only. When a model replaces the oracle,
    that model sees only Tier S.
    """
    target_waypoints: np.ndarray    # (K, 2) ego frame, metres
    lane_action: LaneAction
    speed_cap: float                # m/s
    hazard: bool
    instruction_done: bool
    valid_until: float              # timestamp; latched between slow-pathway updates

    def __post_init__(self):
        expected = (C.N_WAYPOINTS, 2)
        if tuple(self.target_waypoints.shape) != expected:
            raise ValueError(
                f"target_waypoints is {tuple(self.target_waypoints.shape)}, expected {expected}")


# ---------------------------------------------------------------------------
# Leak audit -- called by the test suite, and cheap enough to call at startup
# ---------------------------------------------------------------------------
PRIVILEGED_FIELD_NAMES = frozenset(f.name for f in fields(PrivilegedState))


def assert_no_privileged_leak() -> None:
    """
    Structural check that Tier P cannot reach the policy.

    Verifies that PrivilegedState exposes no vector-producing method and that
    no Tier S field shares a name with a Tier P field -- a rename is the most
    likely way this contract would quietly erode.
    """
    banned = {"to_policy_vector", "to_vector"}
    present = banned & set(dir(PrivilegedState))
    if present:
        raise AssertionError(
            f"PrivilegedState exposes {sorted(present)}. Tier P must have no path "
            f"to a policy input.")

    sensor_names = set()
    for rec in (SensorObs, EgoState, NavState):
        sensor_names |= {f.name for f in fields(rec)}
    overlap = sensor_names & PRIVILEGED_FIELD_NAMES
    if overlap:
        raise AssertionError(
            f"Field names {sorted(overlap)} appear in both Tier S and Tier P. "
            f"One of them is misfiled.")
