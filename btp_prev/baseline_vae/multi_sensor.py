"""
Multi-camera rig + BEV LiDAR sensors for the VAE baseline ladder.

Replaces main.py's single `CameraSensor` (main.py:520) with an N-camera rig
driven by params.CAMERA_YAWS, plus an optional LiDAR whose point cloud is
projected to the same 2D BEV grid the VLM stack uses
(vlm/collect_data.py:45 project_lidar_to_bev).

CONTROL NOTES
-------------
* The yaw-0 camera is byte-for-byte the original: semantic_segmentation,
  160x80, fov 125, at (x=2.4, z=1.5, pitch=-10). So rung 1 through this class
  is an exact reproduction of the handed-over baseline, and any rung-1 delta
  against Results_05 is a bug in this file, not a protocol change.
* All cameras are the SAME sensor type and resolution. Only yaw/placement/fov
  differ, per params.CAMERA_RIG.
* Frames are returned in params.CAMERA_YAWS order, so latent slice k always
  corresponds to the same physical view across episodes and rungs.

FRAME LAYOUT -- a known defect, reproduced on purpose
-----------------------------------------------------
The original does `raw_data.reshape((image.width, image.height, 4))`
(main.py:549). CARLA's buffer is row-major (H, W, 4) = (80, 160, 4), so
reshaping to (160, 80, 4) does not transpose the image, it REINTERPRETS the
bytes -- each output row of 80 pixels straddles half of a true 160-pixel row.
The result is a deterministic, invertible scramble, not a picture.

We keep it (FRAME_LAYOUT='legacy') because rung 1 must reproduce Results_05.
'correct' is provided so the cost of this defect can be measured separately
without touching the ladder. Do not change the default mid-study.
"""
import os
import weakref

import numpy as np

try:
    import carla
except ImportError:      # allows import on a dev box without the CARLA egg
    carla = None


FRAME_LAYOUT = 'legacy'          # 'legacy' (as deployed) | 'correct'


def decode_camera_frame(image, layout=None):
    """CARLA image -> (W, H, 3) float-ready uint8 array.

    layout='legacy'  : reshape((W, H, 4))  -- byte reinterpretation, as deployed
    layout='correct' : reshape((H, W, 4)) then transpose to (W, H, 3)
    """
    layout = layout or FRAME_LAYOUT
    buf = np.frombuffer(image.raw_data, dtype=np.uint8)
    if layout == 'legacy':
        return buf.reshape((image.width, image.height, 4))[:, :, :3]
    hwc = buf.reshape((image.height, image.width, 4))[:, :, :3]
    return np.transpose(hwc, (1, 0, 2))


def project_lidar_to_bev(point_cloud, grid_width=160, grid_height=80,
                         min_x=-15.0, max_x=35.0, max_y=25.0):
    """3D point cloud -> 2D BEV max-height grid, returned as (W, H, 3).

    Grid geometry is identical to vlm/collect_data.py:45 so the BEV VAE trained
    for rung 3 sees exactly the representation the VLM stack will use.

    Returned as (W, H, 3) -- tiled to 3 channels and transposed -- because the
    frozen VAE architecture takes (160, 80, 3). Rung 3 therefore spends the
    SAME encoder capacity on LiDAR as on one camera, which is what makes the
    +95 dims attributable to the modality rather than to a bigger encoder.
    """
    raw = np.frombuffer(point_cloud.raw_data, dtype=np.dtype('f4'))
    points = np.reshape(raw, (int(raw.shape[0] / 4), 4))

    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    mask = (x >= min_x) & (x <= max_x) & (np.abs(y) <= max_y)
    x_f, y_f, z_f = x[mask], y[mask], z[mask]

    grid = np.zeros((grid_height, grid_width), dtype=np.float32)
    if len(x_f):
        gx = np.clip(((max_x - x_f) / (max_x - min_x) * (grid_height - 1)).astype(np.int32),
                     0, grid_height - 1)
        gy = np.clip(((y_f + max_y) / (2 * max_y) * (grid_width - 1)).astype(np.int32),
                     0, grid_width - 1)
        norm_z = np.clip((z_f + 2.0) / 4.0, 0.0, 1.0)
        np.maximum.at(grid, (gx, gy), norm_z)

    # (H, W) -> (W, H, 3), scaled to 0-255 so the LiDAR stream enters the
    # encoder on the same numeric scale as the camera streams. Without this the
    # two modalities would sit on different scales and the 'raw' arm would be
    # comparing apples to oranges.
    grid = np.transpose(grid, (1, 0))
    return np.repeat(grid[:, :, None], 3, axis=2) * 255.0


class MultiCameraSensor:
    """N semantic-segmentation cameras at params.CAMERA_YAWS."""

    def __init__(self, vehicle, params):
        self.p = params
        self.parent = vehicle
        self.yaws = list(params.CAMERA_YAWS)
        self.buffers = {yaw: [] for yaw in self.yaws}
        self.sensors = []

        mask_rear = getattr(params, 'MASK_REAR_CAMERA', False) or (os.environ.get('BTP_MASK_REAR') == '1')
        self.masked_yaws = set([180.0]) if mask_rear else set()

        world = vehicle.get_world()
        bp_lib = world.get_blueprint_library()

        for yaw in self.yaws:
            if yaw in self.masked_yaws:
                continue
            spec = params.CAMERA_RIG[yaw]
            bp = bp_lib.find(params.CAMERA_SENSOR_NAME)
            bp.set_attribute('image_size_x', str(params.IM_WIDTH))
            bp.set_attribute('image_size_y', str(params.IM_HEIGHT))
            bp.set_attribute('fov', str(spec['fov']))
            sensor = world.spawn_actor(
                bp,
                carla.Transform(
                    carla.Location(x=spec['x'], y=spec['y'], z=spec['z']),
                    carla.Rotation(pitch=spec['pitch'], yaw=yaw),
                ),
                attach_to=vehicle,
            )
            weak_self = weakref.ref(self)
            sensor.listen(
                lambda image, y=yaw: MultiCameraSensor._on_image(weak_self, y, image))
            self.sensors.append(sensor)

    @staticmethod
    def _on_image(weak_self, yaw, image):
        self = weak_self()
        if not self:
            return
        image.convert(carla.ColorConverter.CityScapesPalette)
        self.buffers[yaw].append(decode_camera_frame(image))

    # ------------------------------------------------------------------
    def ready(self):
        return all(len(self.buffers[y]) for y in self.yaws if y not in self.masked_yaws)

    def get_frames(self):
        """Latest frame from each view, in CAMERA_YAWS order -> list of (W,H,3)."""
        frames = []
        for yaw in self.yaws:
            if yaw in self.masked_yaws:
                frames.append(np.zeros((self.p.IM_WIDTH, self.p.IM_HEIGHT, 3), dtype=np.uint8))
                continue
            buf = self.buffers[yaw]
            frames.append(buf.pop(-1))
            buf.clear()          # drop backlog; we always want the freshest frame
        return frames

    def destroy(self):
        for s in self.sensors:
            if s is not None and s.is_alive:
                s.destroy()
        self.sensors = []


class LidarBEVSensor:
    """Raycast LiDAR projected to a 2D BEV occupancy/height grid."""

    def __init__(self, vehicle, params):
        self.p = params
        self.parent = vehicle
        self.buffer = []

        world = vehicle.get_world()
        spec = params.LIDAR_SPECS
        bp = world.get_blueprint_library().find('sensor.lidar.ray_cast')
        bp.set_attribute('channels', str(spec['channels']))
        bp.set_attribute('points_per_second', str(spec['points_per_second']))
        bp.set_attribute('range', str(spec['range']))
        bp.set_attribute('upper_fov', str(spec['upper_fov']))
        bp.set_attribute('lower_fov', str(spec['lower_fov']))
        bp.set_attribute('rotation_frequency', str(spec['rotation_frequency']))

        self.sensor = world.spawn_actor(
            bp,
            carla.Transform(carla.Location(x=spec['x'], y=spec['y'], z=spec['z'])),
            attach_to=vehicle,
        )
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda pc: LidarBEVSensor._on_scan(weak_self, pc))

    @staticmethod
    def _on_scan(weak_self, point_cloud):
        self = weak_self()
        if not self:
            return
        p = self.p
        self.buffer.append(project_lidar_to_bev(
            point_cloud, p.BEV_GRID_W, p.BEV_GRID_H,
            p.BEV_MIN_X, p.BEV_MAX_X, p.BEV_MAX_Y))

    def ready(self):
        return len(self.buffer) > 0

    def get_frame(self):
        bev = self.buffer.pop(-1)
        self.buffer.clear()
        return bev

    def destroy(self):
        if self.sensor is not None and self.sensor.is_alive:
            self.sensor.destroy()
        self.sensor = None
