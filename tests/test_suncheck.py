"""Public SunCheck API: the score, common support, the sun in the camera frame and the fast renderers."""
import sys
from pathlib import Path

import numpy as np
import pytest

import observability as obs
import suncheck
from corruptions import scale_focal

import shadows
import synthetic as syn

K, (H, W) = syn.K_KITTI_Q, syn.HW_KITTI_Q


@pytest.fixture(scope="module")
def street():
    depth, ground, boxes = syn.make_scene(K, H, W)
    s = syn.sun_vector(-40, 30)
    road = ground & (depth < 50)
    exact = syn.exact_shadow(K, H, W, depth, boxes, s) & road
    return depth, road, s, exact


def test_true_geometry_scores_well_and_noise_scores_chance(street):
    depth, road, s, exact = street
    r = suncheck.check(depth, K, s, road, exact.astype(float))
    assert r.auc > 0.8 and 0 < r.coverage <= 1 and r.n_shadow >= 20 and r.n_lit >= 20
    noise = np.random.default_rng(0).random(depth.shape)
    assert abs(suncheck.check(depth, K, s, road, noise).auc - 0.5) < 0.1


def test_scale_is_invisible(street):
    depth, road, s, exact = street
    a = suncheck.check(depth, K, s, road, exact.astype(float), h_max=20.0)
    b = suncheck.check(2 * depth, K, s, road, exact.astype(float), h_max=40.0)  # metric prior scaled along
    # not exact: the self-shadow tolerance and occluder thickness stay in metres (prop. 1 remark)
    assert np.mean(a.state[road] == b.state[road]) > 0.95


def test_compare_scores_every_candidate_on_one_common_support(street):
    depth, road, s, exact = street
    out = suncheck.compare({"truth": (depth, K, s), "short focal": (depth, scale_focal(K, 0.6), s)},
                           road, exact.astype(float))
    assert out["truth"]["common_coverage"] == out["short focal"]["common_coverage"]
    assert out["truth"]["common_coverage"] <= min(out["truth"]["coverage"], out["short focal"]["coverage"])


def test_sun_in_camera_frame_for_a_camera_facing_the_sun():
    import sun

    az, el = sun.sun_az_el(49.0, 8.4, "2011-09-26 10:00:00")
    v = suncheck.sun_in_camera_frame(49.0, 8.4, "2011-09-26 10:00:00", heading_deg=az)
    e = np.radians(el)
    assert np.allclose(v, [0.0, -np.sin(e), np.cos(e)], atol=1e-6)  # straight ahead and up (y is down)
    left = suncheck.sun_in_camera_frame(49.0, 8.4, "2011-09-26 10:00:00", heading_deg=az + 90)
    assert left[0] < -0.5  # sun on the left when the camera turns right


def test_refine_argmax_finds_the_vertex_of_a_parabola():
    x = np.linspace(-1, 1, 9)
    assert np.isclose(suncheck.refine_argmax(x, -(x - 0.13) ** 2), 0.13)


@pytest.mark.parametrize("sun_az", [-40, 150])
def test_road_only_renderers_match_the_full_image(street, sun_az):
    depth, road, _, _ = street
    s = syn.sun_vector(sun_az, 35)
    gy = obs.ground_level(depth, K, road)
    full = obs.render_shadow_amodal(depth, K, s, gy, return_state=True)
    part = obs.render_shadow_amodal(depth, K, s, gy, return_state=True, only=road)
    assert (full[road] == part[road]).all()
    assert (obs.render_shadow3(depth, K, s, gy, h_max=20)[road]
            == obs.render_shadow3(depth, K, s, gy, h_max=20, only=road)[road]).all()
    assert (shadows.render_shadow(depth, K, s)[road] == shadows.render_shadow(depth, K, s, only=road)[road]).all()


def test_cli_runs_on_arrays(tmp_path, street):
    depth, road, s, exact = street
    np.save(tmp_path / "d.npy", depth)
    np.save(tmp_path / "r.npy", road)
    np.save(tmp_path / "p.npy", exact.astype(np.float32))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import suncheck_cli

    r = suncheck_cli.main(["--depth", str(tmp_path / "d.npy"), "--road", str(tmp_path / "r.npy"),
                           "--prob", str(tmp_path / "p.npy"), "--K", str(K[0, 0]), str(K[1, 1]), str(K[0, 2]),
                           str(K[1, 2]), "--sun", *map(str, s), "--png", str(tmp_path / "s.png")])
    assert r.auc > 0.8 and (tmp_path / "s.png").exists()
