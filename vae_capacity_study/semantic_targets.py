"""
Driving-relevant scalar targets derived from CARLA semantic-segmentation frames.

The VAE dataset (VAE/dataset) has no action labels, but the frames ARE
ground-truth semantic labels from the simulator, so the quantities a lateral
controller actually needs can be read off the image exactly. That makes them
honest probe targets: if a 95-d latent cannot linearly predict "where is the
road centre", it cannot support lane keeping either.

Targets (all computed on the TRUE (H, W, 3) image, independent of whatever
preprocessing is later fed to the encoder):

  road_frac        fraction of road pixels                -- free space
  lane_offset      road centroid in the lower third,      -- the steering signal
                   as a signed fraction of half-width
  lane_heading     road centroid shift between the lower  -- path curvature
                   and middle bands
  roadline_frac    lane-marking pixels                    -- lane structure
  obstacle_frac    vehicle + pedestrian pixels            -- what you must not hit
  free_ahead       how far up the centre column stays     -- longitudinal
                   drivable, as a fraction of height         headroom

CARLA CityScapesPalette RGB values, CARLA 0.9.x.
"""
import numpy as np

PALETTE = {
    'unlabeled':   (0, 0, 0),
    'building':    (70, 70, 70),
    'fence':       (100, 40, 40),
    'other':       (55, 90, 80),
    'pedestrian':  (220, 20, 60),
    'pole':        (153, 153, 153),
    'road_line':   (157, 234, 50),
    'road':        (128, 64, 128),
    'sidewalk':    (244, 35, 232),
    'vegetation':  (107, 142, 35),
    'vehicle':     (0, 0, 142),
    'wall':        (102, 102, 156),
    'traffic_sign':(220, 220, 0),
}

TARGET_NAMES = ['road_frac', 'lane_offset', 'lane_heading',
                'roadline_frac', 'obstacle_frac', 'free_ahead']


def _mask(img, name, tol=12):
    """Nearest-colour mask, tolerant to the PNG round-trip."""
    ref = np.array(PALETTE[name], dtype=np.int16)
    return (np.abs(img.astype(np.int16) - ref).sum(axis=2) <= tol)


def extract_targets(img):
    """img: (H, W, 3) uint8 CityScapesPalette frame -> (6,) float32."""
    h, w = img.shape[:2]
    road = _mask(img, 'road') | _mask(img, 'road_line')
    line = _mask(img, 'road_line')
    obst = _mask(img, 'vehicle') | _mask(img, 'pedestrian')

    road_frac = road.mean()
    roadline_frac = line.mean()
    obstacle_frac = obst.mean()

    def band_centroid(y0, y1):
        band = road[y0:y1]
        cols = band.sum(axis=0)
        if cols.sum() == 0:
            return 0.0
        centre = (cols * np.arange(w)).sum() / cols.sum()
        return float((centre - (w - 1) / 2.0) / ((w - 1) / 2.0))   # [-1, 1]

    lower = band_centroid(int(h * 0.66), h)
    middle = band_centroid(int(h * 0.33), int(h * 0.66))
    lane_offset = lower
    lane_heading = middle - lower

    # Free space straight ahead: scan up the centre columns until road stops.
    cc = road[:, int(w * 0.45):int(w * 0.55)].any(axis=1)
    rows = np.flatnonzero(cc)
    free_ahead = float((h - rows.min()) / h) if rows.size else 0.0

    return np.array([road_frac, lane_offset, lane_heading,
                     roadline_frac, obstacle_frac, free_ahead], dtype=np.float32)
