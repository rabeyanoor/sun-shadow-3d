"""Render a cast-shadow mask from a depth map and a known sun direction.

Screen-space ray marching: every pixel is lifted to 3D, we walk from it towards the sun
and check whether the walk goes behind a visible surface. Cheap, runs fine on CPU at
quarter resolution.
"""
import numpy as np


def backproject(depth, K):
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    x = (u - K[0, 2]) / K[0, 0] * depth
    y = (v - K[1, 2]) / K[1, 1] * depth
    return np.stack([x, y, depth], -1)


def render_shadow(depth, K, sun_dir, max_dist=25.0, n_steps=64, eps=0.15, eps_rel=0.03, thickness=3.0, only=None):
    """Return a boolean mask: True where the pixel's 3D point cannot see the sun.

    depth    : (H, W) metric depth in metres (0 or nan = invalid)
    K        : 3x3 intrinsics matching depth's resolution
    sun_dir  : unit vector towards the sun in camera coordinates
    eps      : tolerance so a surface does not shadow itself; the total tolerance is
               eps + eps_rel * depth because depth noise grows with distance
    thickness: how deep (m) we assume an occluder is behind its visible surface
    only     : optional (H, W) mask; only these pixels are rendered (the rest is False)
    """
    h, w = depth.shape
    valid = np.isfinite(depth) & (depth > 0)
    # non-uniform steps: fine near the surface, coarse far away
    ts = max_dist * np.linspace(0.02, 1.0, n_steps) ** 2
    shadow = np.zeros(h * w, bool)
    idx = np.flatnonzero(valid & (True if only is None else only))
    P = backproject(np.where(valid, depth, 0), K).reshape(-1, 3)[idx]
    for t in ts:
        if idx.size == 0:
            break
        Q = P + t * sun_dir
        z = Q[:, 2]
        zc = np.maximum(z, 1e-3)
        u = np.round(K[0, 0] * Q[:, 0] / zc + K[0, 2]).astype(int)
        v = np.round(K[1, 1] * Q[:, 1] / zc + K[1, 2]).astype(int)
        inside = (z > 0.1) & (u >= 0) & (u < w) & (v >= 0) & (v < h)  # rays that leave the image stop
        d_hit = depth[np.clip(v, 0, h - 1), np.clip(u, 0, w - 1)]
        behind = inside & np.isfinite(d_hit) & (d_hit > 0) & (z - d_hit > eps + eps_rel * d_hit) & (z - d_hit < thickness)
        shadow[idx[behind]] = True
        keep = inside & ~behind
        idx, P = idx[keep], P[keep]
    return shadow.reshape(h, w) & valid


def resize_K(K, sx, sy):
    K = K.copy()
    K[0] *= sx
    K[1] *= sy
    return K
