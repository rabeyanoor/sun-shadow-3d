"""Sun direction, shadow rendering and the synthetic scene."""
from datetime import datetime

import numpy as np
import pytest

import shadows
import sun
import synthetic

K = np.array([[180.0, 0, 309.5], [0, 180.0, 93.5], [0, 0, 1]])
H, W = 188, 620


def test_noon_sun_is_south_and_high_in_karlsruhe():
    # 26 Sep 2011, solar noon in Karlsruhe is about 11:25 UTC
    az, el = sun.sun_az_el(49.015, 8.434, datetime(2011, 9, 26, 11, 25))
    assert 170 < az < 190 and 35 < el < 45


def test_rotation_is_orthonormal():
    R = sun.rot_imu_to_enu(0.02, -0.01, 1.3)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(R), 1.0)


def test_sun_vector_points_up_in_camera_frame():
    s = synthetic.sun_vector(30, 40)
    assert np.isclose(np.linalg.norm(s), 1.0) and s[1] < 0  # y is down


def test_synthetic_ground_depth_matches_plane():
    depth, ground, _ = synthetic.make_scene(K, H, W, boxes=[])
    v = np.arange(H)[:, None] + np.zeros((1, W))
    below = (v > K[1, 2] + 2) & ground
    expected = 1.65 * K[1, 1] / (v - K[1, 2])
    np.testing.assert_allclose(depth[below], expected[below], rtol=1e-9)


@pytest.mark.parametrize("a", [0.5, 2.0])
def test_render_is_scale_invariant_when_tolerances_scale(a):
    depth, ground, boxes = synthetic.make_scene(K, H, W)
    s = synthetic.sun_vector(25, 35)
    kw = dict(max_dist=40.0, n_steps=64, eps=0.15, thickness=3.0)
    ref = shadows.render_shadow(depth, K, s, **kw)
    scaled = shadows.render_shadow(depth * a, K, s, **{**kw, "max_dist": 40.0 * a, "eps": 0.15 * a, "thickness": 3.0 * a})
    assert (ref == scaled).mean() > 0.999


def test_exact_shadow_is_on_the_far_side_from_the_sun():
    depth, ground, boxes = synthetic.make_scene(K, H, W)
    sh = synthetic.exact_shadow(K, H, W, depth, boxes, synthetic.sun_vector(-40, 30)) & ground
    assert sh.any()


def test_kappa_is_zero_for_independent_masks_and_one_for_equal():
    import scene

    rng = np.random.default_rng(0)
    region = np.ones((200, 200), bool)
    a = rng.random((200, 200)) < 0.3
    b = rng.random((200, 200)) < 0.6
    assert abs(scene.kappa(a, b, region)) < 0.02
    assert np.isclose(scene.kappa(a, a, region), 1.0)
    # IoU of independent masks grows with the coverage of the prediction, kappa does not
    big = rng.random((200, 200)) < 0.9
    assert scene.iou(big, a, region) > scene.iou(b, a, region)
    assert abs(scene.kappa(big, a, region)) < 0.02
