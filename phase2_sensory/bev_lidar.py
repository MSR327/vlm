"""
Vectorized Bird's-Eye-View (BEV) LiDAR Projection for Phase 2 TransFuser.

Converts raw CARLA 3D LiDAR point clouds (N, 4) into a calibrated 2-channel
metric spatial grid (2, 256, 256) partitioning obstacle points from ground plane points.
"""
import numpy as np
import torch
import phase2_sensory.config as C


class BEVLidarProjector:
    """
    Projects 3D point cloud into 2-channel BEV height-partitioned grid.
    Channel 0: Obstacles above ground plane (z > 0.3m) -> Normalized height
    Channel 1: Ground plane / road surface (z <= 0.3m) -> Point density
    """
    def __init__(self,
                 range_x=C.LIDAR_RANGE_X,
                 range_y=C.LIDAR_RANGE_Y,
                 ground_height=C.LIDAR_GROUND_HEIGHT,
                 min_height=C.LIDAR_MIN_HEIGHT,
                 max_height=C.LIDAR_MAX_HEIGHT,
                 grid_size=C.BEV_GRID_SIZE):
        self.x_min, self.x_max = range_x
        self.y_min, self.y_max = range_y
        self.ground_height = ground_height
        self.min_height = min_height
        self.max_height = max_height
        self.grid_size = grid_size

        self.res_x = (self.x_max - self.x_min) / float(self.grid_size)
        self.res_y = (self.y_max - self.y_min) / float(self.grid_size)

    def project(self, points):
        """
        Args:
            points: np.ndarray of shape (N, 3) or (N, 4) [x, y, z, (intensity)]
        Returns:
            torch.Tensor of shape (2, grid_size, grid_size) float32 in range [0, 1]
        """
        if points is None or len(points) == 0:
            return torch.zeros((2, self.grid_size, self.grid_size), dtype=torch.float32)

        if not isinstance(points, np.ndarray):
            points = np.asarray(points, dtype=np.float32)

        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

        # Spatial bounding box filter
        mask = (
            (x >= self.x_min) & (x < self.x_max) &
            (y >= self.y_min) & (y < self.y_max) &
            (z >= self.min_height) & (z < self.max_height)
        )
        x_filt = x[mask]
        y_filt = y[mask]
        z_filt = z[mask]

        if len(x_filt) == 0:
            return torch.zeros((2, self.grid_size, self.grid_size), dtype=torch.float32)

        # Discretize into grid pixel coordinates
        # x_px: Longitudinal distance forward (0 to 32m -> rows 0 to grid_size-1)
        # y_px: Lateral distance (-16m to +16m -> cols 0 to grid_size-1)
        x_px = np.clip(np.floor((x_filt - self.x_min) / self.res_x).astype(np.int64), 0, self.grid_size - 1)
        y_px = np.clip(np.floor((y_filt - self.y_min) / self.res_y).astype(np.int64), 0, self.grid_size - 1)

        bev = np.zeros((2, self.grid_size, self.grid_size), dtype=np.float32)

        # Split above/below ground plane
        above_mask = z_filt > self.ground_height
        below_mask = ~above_mask

        # Channel 0: Normalized obstacle height (z - ground_height) / (max_height - ground_height)
        if np.any(above_mask):
            z_norm = np.clip(
                (z_filt[above_mask] - self.ground_height) / (self.max_height - self.ground_height),
                0.0, 1.0
            )
            # Use max-height per cell to preserve obstacle silhouette
            np.maximum.at(bev[0], (x_px[above_mask], y_px[above_mask]), z_norm)

        # Channel 1: Ground plane occupancy / density (normalized count)
        if np.any(below_mask):
            np.add.at(bev[1], (x_px[below_mask], y_px[below_mask]), 1.0)
            # Compress count with log1p and normalize
            bev[1] = np.clip(np.log1p(bev[1]) / 3.0, 0.0, 1.0)

        return torch.from_numpy(bev).float()
