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


# ---------------------------------------------------------------- random streets (benchmark)
# KITTI colour camera at the quarter resolution the study renders at
K_KITTI_Q = np.array([[180.2, 0, 152.3], [0, 180.2, 43.1], [0, 0, 1]])
HW_KITTI_Q = (94, 310)


def random_street(rng, cam_height=1.65, trees=True, behind=True):
    """A random street: building rows, trees (trunk + raised canopy), parked cars and poles.

    Many casters are out of view by design, as on KITTI: canopies and upper facades above the
    image, and everything at z < 0 behind the camera. Returns the list of boxes (lo, hi).
    """
    g = cam_height
    boxes = []
    z_min = -40.0 if behind else 2.0
    for side in (-1, 1):
        z = z_min
        while z < 80:  # building row with gaps
            length = rng.uniform(6, 20)
            if rng.random() < 0.75:
                x0 = rng.uniform(6, 10)
                x1 = x0 + rng.uniform(6, 15)
                top = rng.uniform(4, 18)
                xs = sorted((side * x0, side * x1))
                boxes.append((np.array([xs[0], g - top, z]), np.array([xs[1], g, z + length])))
            z += length + rng.uniform(0, 8)
        if trees:
            for zt in rng.uniform(z_min, 60, rng.integers(0, 6)):
                xt = side * rng.uniform(4, 6.5)
                r = rng.uniform(1.0, 2.5)
                lo_c, hi_c = rng.uniform(2.5, 4.0), rng.uniform(6, 11)
                boxes.append((np.array([xt - 0.15, g - lo_c, zt - 0.15]), np.array([xt + 0.15, g, zt + 0.15])))
                boxes.append((np.array([xt - r, g - hi_c, zt - r]), np.array([xt + r, g - lo_c, zt + r])))
        for zc in rng.uniform(4, 45, rng.integers(0, 4)):  # parked cars
            xc = side * rng.uniform(2.3, 3.5)
            boxes.append((np.array([xc - 0.9, g - 1.5, zc]), np.array([xc + 0.9, g, zc + 4.2])))
        for zp in rng.uniform(z_min, 50, rng.integers(0, 3)):  # poles
            xp = side * rng.uniform(3.8, 5.0)
            boxes.append((np.array([xp - 0.12, g - rng.uniform(4, 9), zp]), np.array([xp + 0.12, g, zp + 0.24])))
    return boxes


def visible_boxes(K, h, w, depth, boxes):
    """Boxes that own at least one pixel of the depth map."""
    d = rays(K, h, w)
    out = []
    for lo, hi in boxes:
        t = _hit_box(d, lo, hi)
        if np.any(np.isclose(t, depth) & (depth > 0)):
            out.append((lo, hi))
    return out


def random_sun(rng, el_range=(20, 60)):
    return sun_vector(rng.uniform(-180, 180), rng.uniform(*el_range))


def simulated_detector(rng, shadow, receiver, strength=0.45, albedo=0.25, noise=0.12, blobs=6):
    """A soft shadow score as a real detector sees it: shadow darkening, dark albedo patches
    (oil, patches of new asphalt, drains) and pixel noise. Returns (soft score, binary mask)."""
    from scipy import ndimage

    h, w = shadow.shape
    patch = np.zeros((h, w))
    for _ in range(rng.poisson(blobs)):
        cy, cx = rng.integers(h // 3, h), rng.integers(0, w)
        ry, rx = rng.uniform(1.5, 6), rng.uniform(4, 25)
        yy, xx = np.ogrid[:h, :w]
        patch[((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 < 1] = 1
    soft = strength * shadow + albedo * patch + noise * rng.standard_normal((h, w))
    soft = ndimage.gaussian_filter(soft, 0.7)
    binary = (soft > 0.5 * strength) & receiver
    return soft, binary
