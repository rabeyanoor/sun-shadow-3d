"""A tiny synthetic street: a ground plane and a few boxes, seen by a pinhole camera.

Everything is known exactly (depth, shadows, sun), so it separates two questions that the
KITTI pilot could not: what a shadow constrains *in principle*, and what our detector and
screen-space renderer lose in practice.

Camera frame as in KITTI: x right, y down, z forward. The ground is the plane y = cam_height.
"""
import numpy as np


def rays(K, h, w):
    u, v = np.meshgrid(np.arange(w) + 0.0, np.arange(h) + 0.0)
    d = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1)
    return d  # z component is 1, so the ray parameter is the depth


def _hit_box(d, lo, hi):
    """Slab test for rays from the origin with direction d (z = 1). Returns entry depth or inf."""
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = lo / d
        t2 = hi / d
    tmin = np.nanmax(np.minimum(t1, t2), axis=-1)
    tmax = np.nanmin(np.maximum(t1, t2), axis=-1)
    hit = (tmax >= tmin) & (tmax > 0)
    return np.where(hit, np.maximum(tmin, 0), np.inf)


def make_scene(K, h, w, cam_height=1.65, boxes=None):
    """Depth map (metres) of the ground plane plus boxes. Sky (no hit) is 0.

    boxes: list of (lo, hi) corners in camera coordinates; a default street is used if None.
    """
    if boxes is None:
        g = cam_height
        boxes = [
            (np.array([-6.0, g - 3.0, 12.0]), np.array([-3.5, g, 30.0])),    # wall on the left
            (np.array([2.5, g - 1.5, 10.0]), np.array([4.3, g, 14.5])),      # parked car
            (np.array([-1.0, g - 4.0, 22.0]), np.array([-0.6, g, 22.4])),    # pole
            (np.array([4.0, g - 6.0, 24.0]), np.array([9.0, g, 40.0])),      # building on the right
        ]
    d = rays(K, h, w)
    with np.errstate(divide="ignore"):
        t_ground = np.where(d[..., 1] > 1e-6, cam_height / d[..., 1], np.inf)
    depth = t_ground
    for lo, hi in boxes:
        depth = np.minimum(depth, _hit_box(d, lo, hi))
    depth = np.where(np.isfinite(depth) & (depth < 120), depth, 0.0)
    ground = np.isclose(depth, t_ground) & (depth > 0)
    return depth.astype(np.float64), ground, boxes


def exact_shadow(K, h, w, depth, boxes, sun_dir):
    """Ground-truth cast shadows by ray casting towards the sun in 3D (not screen space)."""
    d = rays(K, h, w)
    P = d * depth[..., None]
    shadow = np.zeros((h, w), bool)
    valid = depth > 0
    for lo, hi in boxes:
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (lo - P) / sun_dir
            t2 = (hi - P) / sun_dir
        tmin = np.nanmax(np.minimum(t1, t2), axis=-1)
        tmax = np.nanmin(np.maximum(t1, t2), axis=-1)
        shadow |= (tmax >= tmin) & (tmin > 1e-3)
    return shadow & valid


def sun_vector(azimuth_deg, elevation_deg):
    """Sun direction in camera coordinates for a camera looking along +z with y down.
    azimuth 0 = straight ahead, positive = to the right."""
    az, el = np.radians(azimuth_deg), np.radians(elevation_deg)
    return np.array([np.cos(el) * np.sin(az), -np.sin(el), np.cos(el) * np.cos(az)])
