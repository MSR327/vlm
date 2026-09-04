"""
LiDAR -> BEV rasterisation.

TransFuser's 2-bin height histogram: points below and above the ground plane,
as two channels of a top-down grid. This replaces the previous single
max-height channel, which discarded point density entirely.

Deliberately NO goal channel. TransFuser rasterises the 2D goal location into
the same BEV grid as a third channel, which means their "LiDAR helps" result
partly measures a better goal encoding rather than the sensor. Keeping the goal
out is required for the LiDAR ablation to be clean, and should be stated
explicitly in any write-up.

Rasterisation happens model-side, from raw stored points, so the representation
can be changed without re-collecting the dataset.
"""

from __future__ import annotations

import numpy as np

from .. import config as C


def rasterise_bev(
    points: np.ndarray,
    pixels: int = C.BEV_PIXELS,
    range_m: float = C.BEV_RANGE_M,
    forward_m: float = C.BEV_FORWARD_M,
    ground_z: float = C.BEV_GROUND_Z,
) -> np.ndarray:
    """
    Args:
        points: (N, 4) raw LiDAR returns, CARLA sensor frame (x forward, y right, z up).
        pixels: side length of the square output grid.
        range_m: extent of the grid in metres, both axes.
        forward_m: how much of `range_m` lies ahead of the ego.
        ground_z: ground plane height in the sensor frame.

    Returns:
        (2, pixels, pixels) float32. Channel 0 counts points at or below the
        ground plane, channel 1 counts points above it. Counts are log1p-scaled
        then normalised to [0, 1] -- raw counts have a very long tail and would
        otherwise be dominated by the few cells containing a nearby wall.

        Row 0 is the furthest point ahead; column 0 is furthest to the left.
    """
    if points.ndim != 2 or points.shape[1] != 4:
        raise ValueError(f"points has shape {tuple(points.shape)}, expected (N, 4)")

    grid = np.zeros((C.BEV_BINS, pixels, pixels), dtype=np.float32)
    if points.shape[0] == 0:
        return grid

    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    back_m = range_m - forward_m
    half = range_m / 2.0
    in_range = (x >= -back_m) & (x < forward_m) & (y >= -half) & (y < half)
    if not np.any(in_range):
        return grid
    x, y, z = x[in_range], y[in_range], z[in_range]

    # x: forward_m -> row 0, -back_m -> row (pixels-1)
    row = ((forward_m - x) / range_m * pixels).astype(np.int32)
    col = ((y + half) / range_m * pixels).astype(np.int32)
    np.clip(row, 0, pixels - 1, out=row)
    np.clip(col, 0, pixels - 1, out=col)

    above = (z > ground_z).astype(np.int32)

    flat = (above * pixels + row) * pixels + col
    counts = np.bincount(flat, minlength=C.BEV_BINS * pixels * pixels)
    grid = counts.reshape(C.BEV_BINS, pixels, pixels).astype(np.float32)

    grid = np.log1p(grid)
    peak = float(grid.max())
    if peak > 0.0:
        grid /= peak
    return grid
