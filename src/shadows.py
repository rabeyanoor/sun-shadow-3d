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


def render_shadow(depth, K, sun_dir, max_dist=25.0, n_steps=64, eps=0.15, eps_rel=0.03, thickness=3.0):
    """Return a boolean mask: True where the pixel's 3D point cannot see the sun.

    depth    : (H, W) metric depth in metres (0 or nan = invalid)
    K        : 3x3 intrinsics matching depth's resolution
    sun_dir  : unit vector towards the sun in camera coordinates
    eps      : tolerance so a surface does not shadow itself; the total tolerance is
               eps + eps_rel * depth because depth noise grows with distance
    thickness: how deep (m) we assume an occluder is behind its visible surface
    """
    h, w = depth.shape
    valid = np.isfinite(depth) & (depth > 0)
    P = backproject(np.where(valid, depth, 0), K)

    # non-uniform steps: fine near the surface, coarse far away
    ts = max_dist * np.linspace(0.02, 1.0, n_steps) ** 2
    shadow = np.zeros((h, w), bool)
    alive = valid.copy()
    for t in ts:
        Q = P + t * sun_dir
        z = Q[..., 2]
        in_front = z > 0.1
        u = np.round(K[0, 0] * Q[..., 0] / np.maximum(z, 1e-3) + K[0, 2]).astype(int)
        v = np.round(K[1, 1] * Q[..., 1] / np.maximum(z, 1e-3) + K[1, 2]).astype(int)
        inside = in_front & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        # rays that leave the image can no longer be tested
        alive &= inside
        uu, vv = np.clip(u, 0, w - 1), np.clip(v, 0, h - 1)
        d_hit = depth[vv, uu]
        tol = eps + eps_rel * d_hit
        behind = alive & np.isfinite(d_hit) & (d_hit > 0) & (z - d_hit > tol) & (z - d_hit < thickness)
        shadow |= behind
        alive &= ~behind
    return shadow & valid


def resize_K(K, sx, sy):
    K = K.copy()
    K[0] *= sx
    K[1] *= sy
    return K
