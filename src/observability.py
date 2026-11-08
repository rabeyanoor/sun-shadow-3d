"""Observability-aware shadow checking.

The plain renderer (shadows.render_shadow) answers "lit" for every pixel whose ray towards the
sun does not hit a *visible* surface, also when the ray leaves the image while still low enough
to be blocked by something out of view. On KITTI most road shadows come from such casters
(trees, facades above the image or behind the camera), so "lit" is mostly a guess.

render_shadow3 gives each pixel one of three states:
    shadow       the ray hits a visible surface
    lit          the ray climbs above h_max while still inside the view, without a hit
    undetermined the ray leaves the view (or reaches max_dist) below h_max
Only shadow and lit pixels are *determined*, and only they are scored.

Scores on the determined pixels:
    auc   area under the ROC curve of a soft observed shadow score (a detector probability,
          or plain road darkness) for rendered-shadow against rendered-lit pixels. If the
          observed score is independent of the rendered mask its expectation is 1/2 for every
          mask size, so a larger rendered shadow cannot win by coverage (prop. 3 in the paper).
    iou, kappa on the determined pixels, for comparison.
"""
import numpy as np
from scipy.stats import rankdata

from shadows import backproject

SHADOW, LIT, UNDETERMINED = 1, 0, -1


def ground_level(depth, K, receiver):
    """Height of the ground plane in camera y (y down): median y of the receiver points."""
    P = backproject(np.where(depth > 0, depth, 0), K)
    y = P[..., 1][receiver & (depth > 0)]
    return float(np.median(y)) if y.size else np.nan


def render_shadow3(depth, K, sun_dir, ground_y, h_max=10.0, n_steps=96, eps=0.15, eps_rel=0.03, thickness=3.0,
                   max_dist=None, only=None):
    """Three-state cast shadow from a depth map, without completing truncated casters.

    depth    : (H, W) metres, 0 = invalid / sky
    ground_y : camera-frame y of the ground plane (positive: below the camera)
    h_max    : height (m) above the ground plane above which nothing is assumed to cast a shadow
    only     : optional (H, W) mask; only these pixels are rendered (the others are UNDETERMINED)
    Returns an int8 map with SHADOW, LIT or UNDETERMINED (invalid pixels are UNDETERMINED).
    """
    h, w = depth.shape
    valid = np.isfinite(depth) & (depth > 0)
    rise = max(-sun_dir[1], 1e-3)  # height gained per metre along the ray
    if max_dist is None:
        max_dist = (h_max + 2.0) / rise  # long enough for a ray from the lowest receiver to reach h_max
    ts = max_dist * np.linspace(0.01, 1.0, n_steps) ** 2
    state = np.full(h * w, UNDETERMINED, np.int8)
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
        inside = (z > 0.1) & (u >= 0) & (u < w) & (v >= 0) & (v < h)  # left the view below h_max: undetermined
        d_hit = depth[np.clip(v, 0, h - 1), np.clip(u, 0, w - 1)]
        behind = inside & np.isfinite(d_hit) & (d_hit > 0) & (z - d_hit > eps + eps_rel * d_hit) & (z - d_hit < thickness)
        state[idx[behind]] = SHADOW
        high = inside & ~behind & (ground_y - Q[:, 1] >= h_max)
        state[idx[high]] = LIT
        keep = inside & ~behind & ~high
        idx, P = idx[keep], P[keep]
    return state.reshape(h, w)


def auc(score, positive, negative, min_count=20):
    """P(score at a positive pixel > score at a negative pixel), ties count 1/2 (Mann-Whitney)."""
    a, b = score[positive], score[negative]
    if a.size < min_count or b.size < min_count:
        return np.nan
    r = rankdata(np.concatenate([a, b]))
    return float((r[: a.size].sum() - a.size * (a.size + 1) / 2) / (a.size * b.size))


def darkness(img, receiver):
    """Detector-free soft shadow score: how much darker than the sunlit road a pixel is."""
    lum = img.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
    lit = np.percentile(lum[receiver], 80) if receiver.any() else 1.0
    return 1.0 - lum / max(lit, 1e-3)


def scores(state, soft, binary, region, min_count=20):
    """All scores of a three-state rendering against an observed soft score and binary mask."""
    det = region & (state != UNDETERMINED)
    sh, lit = det & (state == SHADOW), det & (state == LIT)
    out = dict(det_frac=float(det.sum() / max(region.sum(), 1)), det_shadow_frac=float(sh.sum() / max(det.sum(), 1)),
               n_det=int(det.sum()), auc=auc(soft, sh, lit, min_count),
               auc_am=auc(soft, region & (state == SHADOW), region & (state != SHADOW), min_count),
               amodal_frac=float((region & (state == SHADOW)).sum() / max(region.sum(), 1)))
    if det.sum() >= 2 * min_count:
        a, b = sh[det], binary[det]
        union = (a | b).sum()
        out["iou_det"] = float((a & b).sum() / union) if union else np.nan
        pa, pb = a.mean(), b.mean()
        chance = pa * pb + (1 - pa) * (1 - pb)
        out["kappa_det"] = float(((a == b).mean() - chance) / (1 - chance)) if chance < 1 else np.nan
    else:
        out["iou_det"] = out["kappa_det"] = np.nan
    return out


def chance_iou(r, o):
    """Expected IoU of two independent masks with coverages r and o. Increasing in r (prop. 3)."""
    return r * o / (r + o - r * o) if r + o > 0 else np.nan


def focal_sensitivity(sun_dir):
    """d(equivalent sun error)/d(log k) at k = 1, in radians: sin(2 psi) / 2 (cor. 1)."""
    psi = np.arccos(np.clip(abs(sun_dir[2]) / np.linalg.norm(sun_dir), 0, 1))
    return 0.5 * np.sin(2 * psi)


def equivalent_sun(sun_dir, k):
    """Sun that a focal factor k (or, to first order, a shape warp gamma = k) is equivalent to (prop. 2)."""
    s = np.array([k * sun_dir[0], k * sun_dir[1], sun_dir[2]])
    return s / np.linalg.norm(s)


def focal_from_sun(sun_true, sun_est):
    """Invert prop. 2: the focal factor k whose equivalent sun is closest to an estimated sun.

    s_k is proportional to (k s_x, k s_y, s_z), so k = |s_est,xy| s_z / (|s_xy| s_est,z) when both
    suns lie on the same side of the image plane. Returns nan when the sun is on the optical axis
    or in the image plane, where k is not observable.
    """
    a = np.hypot(sun_true[0], sun_true[1]) / max(abs(sun_true[2]), 1e-9)
    b = np.hypot(sun_est[0], sun_est[1]) / max(abs(sun_est[2]), 1e-9)
    if a < 1e-3 or a > 1e3 or np.sign(sun_true[2]) != np.sign(sun_est[2]):
        return np.nan
    return float(b / a)


def render_shadow_amodal(depth, K, sun_dir, ground_y, h_max=20.0, n_steps=96, eps=0.15, eps_rel=0.03,
                         thickness=3.0, max_dist=None, sky=None, return_state=False, only=None):
    """Two-state cast shadow with casters cut by the top image border completed upwards.

    An image column u is the vertical plane X = (u - c_x) Z / f_x through the camera, and it meets
    any vertical surface in a vertical line of constant depth Z. So a facade, trunk or pole that
    leaves the image at the top continues above it with the depth of the top row: the completion is
    exact for vertical surfaces (camera upright) and needs no learning. A ray that leaves through the
    top border is tested against that column depth up to the height prior h_max. Columns whose top
    pixel is sky (or invalid) have nothing to complete.
    With return_state, rays that leave through the sides or behind the camera are UNDETERMINED.
    only: optional (H, W) mask of the pixels to render (others are not shadow / UNDETERMINED).
    """
    h, w = depth.shape
    valid = np.isfinite(depth) & (depth > 0)
    top = np.where(valid[0], depth[0], 0.0)
    if sky is not None:
        top = np.where(sky[0], 0.0, top)
    rise = max(-sun_dir[1], 1e-3)
    if max_dist is None:
        max_dist = (h_max + 2.0) / rise
    ts = max_dist * np.linspace(0.01, 1.0, n_steps) ** 2
    shadow = np.zeros(h * w, bool)
    lost = np.zeros(h * w, bool)
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
        cols = (z > 0.1) & (u >= 0) & (u < w)
        low = ground_y - Q[:, 1] < h_max
        inside = cols & (v >= 0) & (v < h)
        above = cols & (v < 0) & low
        stay = inside | above
        lost[idx[~stay & low]] = True  # left through the sides or behind the camera below h_max
        uu, vv = np.clip(u, 0, w - 1), np.clip(v, 0, h - 1)
        d_hit = np.where(above, top[uu], depth[vv, uu])
        behind = stay & (d_hit > 0) & (z - d_hit > eps + eps_rel * d_hit) & (z - d_hit < thickness)
        shadow[idx[behind]] = True
        keep = stay & ~behind
        idx, P = idx[keep], P[keep]
    shadow = shadow.reshape(h, w) & valid
    if not return_state:
        return shadow
    state = np.where(shadow, SHADOW, np.where(lost.reshape(h, w), UNDETERMINED, LIT)).astype(np.int8)
    state[~valid] = UNDETERMINED
    if only is not None:
        state[~only] = UNDETERMINED
    return state
