"""SunCheck: a label-free shadow test for a depth map, in one call.

    import suncheck
    s = suncheck.sun_in_camera_frame(lat, lon, "2011-09-26 11:02:25", heading_deg=10.0)
    r = suncheck.check(depth, K, s, road, suncheck.darkness_score(image, road))
    print(r.auc, r.coverage)

What it measures. The depth map and the camera are rendered into cast shadows under the sun from GPS and
time, with three states per road pixel (shadow, lit, undetermined) and casters cut by the top image border
completed upwards (exact in depth for vertical surfaces, up to the caster height prior h_max). The score
is the AUC of an observed shadow score (road darkness, or a detector probability) between the rendered
shadow and the rendered lit pixels: 0.5 means no agreement beyond chance, whatever the size of the
rendered shadow. Always read it together with the coverage, the fraction of the road it is computed on.

What it cannot measure. Global scale is invisible to shadows; focal length and depth compression are seen
only together, as one effective focal factor, and a single frame cannot resolve a 20-40 % error of it.
The test needs direct sun, an upright camera (roll removed) and a known camera orientation.
"""
from dataclasses import dataclass, field

import numpy as np

import observability as obs
from corruptions import K_GRID, scale_focal

__all__ = ["SunCheckResult", "sun_in_camera_frame", "darkness_score", "check", "compare", "calibrate_focal"]


@dataclass
class SunCheckResult:
    auc: float  # nan when either class has fewer than n_min pixels
    coverage: float  # determined road pixels / road pixels
    n_shadow: int
    n_lit: int
    state: np.ndarray = field(repr=False)  # int8 map: 1 shadow, 0 lit, -1 undetermined (or not road)


def sun_in_camera_frame(lat, lon, when_utc, heading_deg, pitch_deg=0.0, roll_deg=0.0):
    """Unit vector towards the sun in camera coordinates (x right, y down, z forward).

    heading_deg: direction of the optical axis, clockwise from north; pitch_deg: optical axis above the
    horizon; roll_deg: rotation about the optical axis (clockwise as seen from behind the camera).
    """
    import sun  # pvlib is only needed here

    az, el = sun.sun_az_el(lat, lon, when_utc)
    s = sun.enu_from_az_el(az, el)
    h, p, r = np.radians([heading_deg, pitch_deg, roll_deg])
    fwd = np.array([np.sin(h) * np.cos(p), np.cos(h) * np.cos(p), np.sin(p)])
    right = np.array([np.cos(h), -np.sin(h), 0.0])
    down = np.cross(fwd, right)
    right, down = np.cos(r) * right + np.sin(r) * down, -np.sin(r) * right + np.cos(r) * down
    v = np.array([s @ right, s @ down, s @ fwd])
    return v / np.linalg.norm(v)


def darkness_score(image, road):
    """Detector-free shadow score: 1 - luminance / (80th percentile of road luminance)."""
    return obs.darkness(np.asarray(image), np.asarray(road, bool))


def _region(depth, road):
    return np.asarray(road, bool) & np.isfinite(depth) & (depth > 0)


def _render(depth, K, sun_dir, region, sky, h_max):
    gy = obs.ground_level(depth, K, region)
    return obs.render_shadow_amodal(depth, K, np.asarray(sun_dir, float), gy, h_max=h_max, sky=sky,
                                    return_state=True, only=region)


def _score(state, score, scored, n_min):
    sh, lit = scored & (state == obs.SHADOW), scored & (state == obs.LIT)
    return obs.auc(score, sh, lit, n_min), int(sh.sum()), int(lit.sum())


def check(depth, K, sun_dir, road, score, sky=None, h_max=20.0, n_min=20):
    """SunCheck of one depth map: AUC of `score` between rendered shadow and rendered lit road pixels.

    depth: (H, W) metres, 0 or nan where invalid; K: 3x3 intrinsics at the depth resolution;
    sun_dir: unit vector towards the sun in camera coordinates; road: (H, W) bool mask of the receiver;
    score: (H, W) observed shadow score, higher = darker (darkness_score() or a detector probability);
    sky: optional (H, W) bool sky mask (columns with sky at the top are not completed).
    """
    depth = np.asarray(depth, float)
    region = _region(depth, road)
    state = _render(depth, np.asarray(K, float), sun_dir, region, sky, h_max)
    det = region & (state != obs.UNDETERMINED)
    auc, n_sh, n_lit = _score(state, score, det, n_min)
    return SunCheckResult(auc=auc, coverage=float(det.sum() / max(region.sum(), 1)), n_shadow=n_sh, n_lit=n_lit,
                          state=state)


def compare(candidates, road, score, sky=None, h_max=20.0, n_min=20):
    """Score several candidate geometries of one image on their common support.

    candidates: {name: (depth, K, sun_dir)}. Each candidate is scored on its own determined pixels
    (auc, coverage) and on the pixels that every candidate determines (auc_common, common_coverage),
    so that no candidate can win by determining only easy pixels.
    """
    states, regions = {}, {}
    for name, (d, K, s) in candidates.items():
        d = np.asarray(d, float)
        regions[name] = _region(d, road)
        states[name] = _render(d, np.asarray(K, float), s, regions[name], sky, h_max)
    support = np.logical_and.reduce([regions[n] & (states[n] != obs.UNDETERMINED) for n in candidates])
    road_n = max(int(np.asarray(road, bool).sum()), 1)
    out = {}
    for n in candidates:
        det = regions[n] & (states[n] != obs.UNDETERMINED)
        out[n] = dict(auc=_score(states[n], score, det, n_min)[0], coverage=float(det.sum() / road_n),
                      auc_common=_score(states[n], score, support, n_min)[0],
                      common_coverage=float(support.sum() / road_n))
    return out


def refine_argmax(x, y):
    """Maximum of y over the grid x, refined by a parabola through the best point and its neighbours."""
    y = np.asarray(y, float)
    if not np.isfinite(y).any():
        return np.nan
    i = int(np.nanargmax(y))
    if 0 < i < len(y) - 1 and np.isfinite(y[i - 1]) and np.isfinite(y[i + 1]):
        den = y[i - 1] - 2 * y[i] + y[i + 1]
        d = 0.5 * (y[i - 1] - y[i + 1]) / den if den < 0 else 0.0
        return float(x[i] + np.clip(d, -0.5, 0.5) * (x[i + 1] - x[i]))
    return float(x[i])


def calibrate_focal(frames, k_grid=K_GRID, h_max=20.0, n_min=20, bias=1.0):
    """Effective focal factor of one camera from several frames, by pooling SunCheck over a focal grid.

    frames: iterable of dicts with depth, K, sun_dir, road, score and optionally sky. Each frame is weighted
    by its observability sin(2 psi) / 2 (psi: angle between sun and optical axis). Returns
    (k_hat, curve): k_hat is the refined best factor divided by `bias` (the renderer bias, measured on
    synthetic scenes: 0.79 for the KITTI camera), curve the pooled score per grid value.
    The estimate is the product of the focal error and the depth compression (they are confounded), and
    it is coarse: tens of percent per drive on KITTI.
    """
    num, den = np.zeros(len(k_grid)), np.zeros(len(k_grid))
    for f in frames:
        w = obs.focal_sensitivity(np.asarray(f["sun_dir"], float))
        for j, k in enumerate(k_grid):
            r = check(f["depth"], scale_focal(np.asarray(f["K"], float), k), f["sun_dir"], f["road"], f["score"],
                      sky=f.get("sky"), h_max=h_max, n_min=n_min)
            if np.isfinite(r.auc):
                num[j] += w * r.auc
                den[j] += w
    curve = np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)
    k_hat = np.exp(refine_argmax(np.log(np.asarray(k_grid, float)), curve))
    return k_hat / bias, curve
