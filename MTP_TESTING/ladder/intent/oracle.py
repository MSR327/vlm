"""
The scripted oracle -- first and only adapter behind the slow -> fast seam.

It reads privileged state and the route plan, so it is perfect by construction.
That is the point: it lets the whole fast pathway be built, trained and
validated against a known-good intent stream, so when something scores badly
you know the upstream module is not the cause. Its score is also the ceiling
any future model must approach, which tells you whether a VLM is worth building
before you build one.

BOUNDARY: the oracle reads Tier P. Anything derived from it that reaches the
policy at evaluation time is Tier F. During this phase DrivingIntent is
infrastructure and is not on the policy's input path. When a model replaces the
oracle, that model sees only Tier S.
"""

from __future__ import annotations

import numpy as np

from .. import config as C
from ..contracts import DrivingIntent, LaneAction, PrivilegedState

_COMMAND_TO_ACTION = {
    "follow": LaneAction.FOLLOW,
    "straight": LaneAction.FOLLOW,
    "left": LaneAction.LEFT,
    "right": LaneAction.RIGHT,
    "lane_left": LaneAction.LEFT,
    "lane_right": LaneAction.RIGHT,
}


class OracleIntentSource:
    """
    Produces a DrivingIntent at the slow-pathway rate from privileged state.

    Args:
        rate_hz: slow-pathway update rate. The fast pathway latches the last
            value between updates and never blocks on this.
        hazard_radius_m: an actor closer than this counts as a hazard.
        default_speed_cap: m/s when nothing constrains speed.
    """

    def __init__(self, rate_hz: float = 2.0, hazard_radius_m: float = 12.0,
                 default_speed_cap: float = 8.0):
        if rate_hz <= 0:
            raise ValueError("rate_hz must be positive")
        self.period = 1.0 / rate_hz
        self.hazard_radius_m = hazard_radius_m
        self.default_speed_cap = default_speed_cap
        self._latched: DrivingIntent | None = None

    def _build(self, priv: PrivilegedState, nav_command: str, now: float) -> DrivingIntent:
        wp = np.asarray(priv.expert_future_xy, dtype=np.float32)
        if wp.shape != (C.N_WAYPOINTS, 2):
            raise ValueError(
                f"expert_future_xy is {wp.shape}, expected ({C.N_WAYPOINTS}, 2)")

        hazard = any(
            float(np.hypot(a[0], a[1])) < self.hazard_radius_m
            for a in priv.actors
        ) if priv.actors else False

        if priv.traffic_light_state == 1 or hazard:      # red light or close actor
            action = LaneAction.STOP
            speed_cap = 0.0
        else:
            action = _COMMAND_TO_ACTION.get(nav_command, LaneAction.FOLLOW)
            speed_cap = self.default_speed_cap
            if priv.is_junction:
                speed_cap = min(speed_cap, 5.0)

        return DrivingIntent(
            target_waypoints=wp,
            lane_action=action,
            speed_cap=float(speed_cap),
            hazard=bool(hazard),
            instruction_done=bool(priv.route_completion >= 1.0),
            valid_until=now + self.period,
        )

    def update(self, priv: PrivilegedState, nav_command: str, now: float) -> DrivingIntent:
        """Force a slow-pathway update. Call at `rate_hz`, not every control step."""
        self._latched = self._build(priv, nav_command, now)
        return self._latched

    def get(self, now: float) -> DrivingIntent:
        """
        Read the latched intent. Never blocks and never triggers a slow-pathway
        run -- the fast pathway must be able to call this every control step.
        """
        if self._latched is None:
            raise RuntimeError(
                "no intent has been latched yet; call update() once before the first "
                "control step. Refusing to fabricate a default -- a plausible-looking "
                "stand-in is exactly the failure mode this seam exists to prevent.")
        return self._latched

    def is_stale(self, now: float) -> bool:
        return self._latched is None or now > self._latched.valid_until


class ConstantIntentSource(OracleIntentSource):
    """
    Ablation arm: the slow pathway replaced by a constant.

    Swapping this in measures how much the slow pathway contributes at all.
    """

    def _build(self, priv: PrivilegedState, nav_command: str, now: float) -> DrivingIntent:
        straight = np.stack([np.linspace(2.0, 8.0, C.N_WAYPOINTS),
                             np.zeros(C.N_WAYPOINTS)], axis=1).astype(np.float32)
        return DrivingIntent(
            target_waypoints=straight,
            lane_action=LaneAction.FOLLOW,
            speed_cap=self.default_speed_cap,
            hazard=False,
            instruction_done=False,
            valid_until=now + self.period,
        )
