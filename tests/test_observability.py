"""Three-state renderer, prevalence-invariant score and the focal / sun inversion."""
import numpy as np
import pytest

import observability as obs
from corruptions import raise_sun, rotate_about_up

import shadows
import synthetic as syn

K, (H, W) = syn.K_KITTI_Q, syn.HW_KITTI_Q


def test_auc_is_half_for_independent_masks_of_any_size():
    rng = np.random.default_rng(0)
    soft = rng.random((300, 300))
    for cover in (0.05, 0.3, 0.7):
        m = rng.random((300, 300)) < cover
        assert abs(obs.auc(soft, m, ~m) - 0.5) < 0.01


def test_auc_is_one_for_a_perfect_score():
    m = np.zeros((50, 50), bool)
    m[:20] = True
    assert obs.auc(m.astype(float) + 0.0, m, ~m) == 1.0


def test_chance_iou_grows_with_rendered_coverage():
    r = np.linspace(0.01, 0.9, 50)
    assert np.all(np.diff([obs.chance_iou(x, 0.3) for x in r]) > 0)


@pytest.mark.parametrize("k", [0.6, 0.8, 1.25, 1.5])
def test_focal_from_sun_inverts_the_equivalent_sun(k):
    s = syn.sun_vector(30, 40)
    assert np.isclose(obs.focal_from_sun(s, obs.equivalent_sun(s, k)), k)


def test_focal_sensitivity_vanishes_on_axis_and_in_image_plane():
    assert obs.focal_sensitivity(np.array([0, 0, 1.0])) < 1e-9
    assert obs.focal_sensitivity(np.array([0, -1.0, 0])) < 1e-9
    assert np.isclose(obs.focal_sensitivity(np.array([0, -1.0, 1.0]) / np.sqrt(2)), 0.5)


def test_sun_rotations_keep_unit_length_and_elevation():
    s = syn.sun_vector(20, 35)
    assert np.isclose(np.linalg.norm(rotate_about_up(s, 40)), 1) and np.isclose(rotate_about_up(s, 40)[1], s[1])
    up = raise_sun(s, 10)
    assert np.isclose(np.degrees(-np.arcsin(up[1])), 45)


def test_three_state_agrees_with_exact_shadow_where_determined():
    depth, ground, boxes = syn.make_scene(K, H, W)
    s = syn.sun_vector(0, 35)  # sun ahead: rays stay in view longest
    st = obs.render_shadow3(depth, K, s, ground_y=1.65, h_max=4.0)
    exact = syn.exact_shadow(K, H, W, depth, boxes, s)
    det = ground & (st != obs.UNDETERMINED)
    assert det.sum() > 100
    assert ((st == obs.SHADOW) == exact)[det].mean() > 0.9


def test_lit_needs_the_h_max_point_in_view():
    # observability horizon: a ground point is determined lit only if G + H / sin(el) * s projects into the image
    depth, ground, _ = syn.make_scene(K, H, W, boxes=[])
    s, hm = syn.sun_vector(0, 35), 4.0
    st = obs.render_shadow3(depth, K, s, ground_y=1.65, h_max=hm)
    Q = shadows.backproject(depth, K) + (hm / -s[1]) * s
    v = K[1, 1] * Q[..., 1] / Q[..., 2] + K[1, 2]
    in_view = (Q[..., 2] > 0) & (v >= 0)
    agree = ((st == obs.LIT) == in_view)[ground]
    assert agree.mean() > 0.97


def test_fuse_is_mean_rank_on_road():
    import observability as obs

    region = np.array([[True, True, True, False]])
    a = np.array([[0.1, 0.5, 0.9, 7.0]])
    b = np.array([[3.0, 2.0, 1.0, 7.0]])
    f = obs.fuse([a, b], region)
    assert f[0, 3] == 0
    np.testing.assert_allclose(f[0, :3], [(1 / 3 + 1) / 2, 2 / 3, (1 + 1 / 3) / 2], rtol=1e-6)
    same = obs.fuse([a], region)
    assert obs.auc(same, region & (a > 0.4), region & (a < 0.4), min_count=1) == obs.auc(a, region & (a > 0.4),
                                                                                         region & (a < 0.4), min_count=1)
