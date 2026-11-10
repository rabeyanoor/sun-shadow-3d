"""End-to-end pieces of the study pipeline, on synthetic data (no KITTI download, no GPU)."""
import sys
from pathlib import Path

import numpy as np
import pytest

from corruptions import CORRUPTIONS, TRUTH, raise_sun, rotate_about_up, scale_focal, warp

import synthetic as syn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
K, (H, W) = syn.K_KITTI_Q, syn.HW_KITTI_Q


# ---------------------------------------------------------------- corruption protocol
def test_truth_is_one_of_the_protocol_values():
    for corr, (values, _) in CORRUPTIONS.items():
        assert TRUTH[corr] in values and len(values) == 5


def test_warp_keeps_the_median_and_compresses_towards_it():
    d = np.linspace(5, 50, 1000)
    w = warp(d, 0.6)
    assert np.isclose(np.median(w), np.median(d), rtol=1e-3)
    assert np.ptp(w) < np.ptp(d) and np.allclose(warp(d, 1.0), d)


def test_scale_focal_keeps_the_principal_point():
    Kk = scale_focal(K, 1.25)
    assert np.isclose(Kk[0, 0], 1.25 * K[0, 0]) and Kk[0, 2] == K[0, 2] and Kk[1, 2] == K[1, 2]


def test_sun_rotations_keep_unit_length_and_move_the_sun():
    s = syn.sun_vector(-40, 30)
    r = rotate_about_up(s, 30)
    assert np.isclose(np.linalg.norm(r), 1) and np.isclose(r[1], s[1])  # azimuth only
    up = raise_sun(s, 10)
    assert np.isclose(np.linalg.norm(up), 1) and np.isclose(np.degrees(np.arcsin(-up[1])), 40, atol=1e-6)


# ---------------------------------------------------------------- LiDAR reference
def test_fit_gamma_recovers_the_warp_exponent():
    import lidar_reference as lr

    rng = np.random.default_rng(0)
    L = rng.uniform(4, 60, 5000)
    m = np.median(L)
    for g in (0.8, 1.0, 1.2):
        d = 1.3 * m * (L / m) ** g * np.exp(rng.normal(0, 0.02, L.size))  # any scale, small noise
        assert abs(lr.fit_gamma(d, L) - g) < 0.02


def test_project_puts_a_point_on_the_optical_axis_at_the_principal_point():
    import lidar_reference as lr

    calib = {"R_rect": np.eye(4), "T_velo2cam": np.eye(4), "K2": np.diag([720.0, 720.0, 1.0]),
             "P2": np.array([[720.0, 0, 600, 0], [0, 720, 180, 0], [0, 0, 1, 0]])}
    Kq = np.array([[180.0, 0, 150], [0, 180, 45], [0, 0, 1]])
    depth = lr.project(np.array([[0.0, 0.0, 10.0, 1.0]], np.float32), calib, Kq, (94, 310))
    assert depth[45, 150] == pytest.approx(10.0) and (depth > 0).sum() == 1


# ---------------------------------------------------------------- synthetic benchmark
def test_synthetic_benchmark_scene_produces_the_focal_grid():
    import synthetic_benchmark as sb

    rows, grid = sb.run_scene((3, 20.0, True))
    assert rows == [] and grid, "grid-only run should give grid rows only"
    cols = {"k", "auc3", "auc_cs", "cs_frac", "auc_noam", "auc_shape", "equiv_iou", "iou", "focal_sens"}
    assert cols <= set(grid[0])
    truth = [g for g in grid if np.isclose(g["k"], 1.0)]
    assert all(g["equiv_iou"] == 1.0 for g in truth)  # gamma = k = 1 is the same depth map


# ---------------------------------------------------------------- KITTI evaluation on a synthetic "frame"
@pytest.fixture()
def fake_frame(tmp_path):
    depth, ground, boxes = syn.make_scene(K, H, W)
    s = syn.sun_vector(-40, 30)
    road = ground & (depth < 50)
    exact = syn.exact_shadow(K, H, W, depth, boxes, s) & road
    img = np.full((H, W, 3), 180, np.uint8)
    img[exact] = 70
    path = tmp_path / "2011_09_26_drive_0001_0000000007.npz"
    np.savez(path, receiver=road, img=img, sun=s, K=K, sky=np.zeros((H, W), bool),
             prob=exact.astype(np.float16), **{"depth_dav2-s": depth.astype(np.float16)})
    return path


def test_shadow_eval_scores_a_frame_with_every_experiment(fake_frame):
    import shadow_eval as se

    rows = se.run_frame((str(fake_frame), 20.0, {}, None))
    exps = {r["exp"] for r in rows}
    assert {"scale", "shape", "focal", "sun_azimuth", "focal_grid", "sun_az_grid", "sun_el_grid"} <= exps
    assert {r["obs"] for r in rows} == {"net", "thr"}
    truth = [r for r in rows if r["exp"] == "focal_grid" and np.isclose(r["value"], 1.0) and r["obs"] == "net"][0]
    assert truth["auc"] > 0.8 and 0 < truth["det_frac"] <= 1
    assert all("auc_cs" in r and "cs_frac" in r for r in rows)
    grid = [r for r in rows if r["exp"] == "sun_az_grid" and r["obs"] == "net"]
    assert len({r["cs_frac"] for r in grid}) == 1  # one common support per group


def test_shadow_eval_scores_detector_v2_like_v1(fake_frame, tmp_path):
    import shadow_eval as se

    z = np.load(fake_frame)
    prob2 = tmp_path / f"{fake_frame.stem}.npy"
    np.save(prob2, z["prob"])  # same probability as v1 -> identical scores
    rows = se.run_frame((str(fake_frame), 20.0, {}, str(prob2)))
    assert {r["obs"] for r in rows} == {"net", "thr", "net2", "fuse"}
    v1 = [r["auc"] for r in rows if r["obs"] == "net"]
    v2 = [r["auc"] for r in rows if r["obs"] == "net2"]
    np.testing.assert_allclose(v1, v2)
