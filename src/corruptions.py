"""The corruption protocol: one factor at a time, five values each, the truth in the middle."""
import numpy as np


def warp(d, g):
    """Shape warp d -> m (d / m)^g about the median m: g < 1 compresses depth towards m."""
    m = np.median(d[d > 0])
    o = d.copy()
    o[d > 0] = m * (d[d > 0] / m) ** g
    return o


def scale_focal(K, k):
    K = K.copy()
    K[0, 0] *= k
    K[1, 1] *= k
    return K


def rotate_about_up(v, deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]]) @ v


def raise_sun(v, deg):
    """Change the sun elevation by deg, keeping its azimuth (camera y is down)."""
    el = np.arcsin(np.clip(-v[1], -1, 1)) + np.radians(deg)
    h = np.hypot(v[0], v[2])
    return np.array([v[0] / h * np.cos(el), -np.sin(el), v[2] / h * np.cos(el)])


CORRUPTIONS = {
    "scale": ([0.5, 0.75, 1.0, 1.5, 2.0], lambda d, K, s, a: (d * a, K, s)),
    "shape": ([0.6, 0.8, 1.0, 1.25, 1.5], lambda d, K, s, g: (warp(d, g), K, s)),
    "focal": ([0.6, 0.8, 1.0, 1.25, 1.5], lambda d, K, s, k: (d, scale_focal(K, k), s)),
    "sun_azimuth": ([-60, -30, 0, 30, 60], lambda d, K, s, a: (d, K, rotate_about_up(s, a))),
}
TRUTH = {"scale": 1.0, "shape": 1.0, "focal": 1.0, "sun_azimuth": 0}
# fine focal grid for self-calibration, log-spaced 0.5 .. 2
K_GRID = np.round(np.exp(np.linspace(np.log(0.5), np.log(2.0), 13)), 4)
