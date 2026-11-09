"""PandaSet geometry used for the independent validation: poses, world-to-ENU rotation, LiDAR projection, sun."""
import numpy as np
import pytest

import pandaset as ps
import suncheck


def quat_from_R(R):
    w = np.sqrt(max(1e-12, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    return dict(w=w, x=(R[2, 1] - R[1, 2]) / (4 * w), y=(R[0, 2] - R[2, 0]) / (4 * w), z=(R[1, 0] - R[0, 1]) / (4 * w))


def camera_R(heading_deg):
    """Camera -> world (east, north, up) for a level camera looking along the heading (x right, y down, z fwd)."""
    h = np.radians(heading_deg)
    fwd, right, down = np.array([np.sin(h), np.cos(h), 0]), np.array([np.cos(h), -np.sin(h), 0]), np.array([0, 0, -1.0])
    return np.c_[right, down, fwd]


def test_quaternion_round_trip_is_a_rotation():
    R = camera_R(37.0)
    Q = ps.quat_R(quat_from_R(R))
    assert np.allclose(Q, R, atol=1e-9) and np.allclose(Q.T @ Q, np.eye(3), atol=1e-9)


def test_world_to_enu_recovers_a_yaw_offset():
    lat0, lon0 = 37.4, -122.1
    t = np.linspace(0, 80, 40)
    E, N = 3 * t, 1.5 * t + 0.02 * t ** 2  # a curved track, metres
    gps = [dict(lat=lat0 + np.degrees(n / 6378137.0), long=lon0 + np.degrees(e / (6378137.0 * np.cos(np.radians(lat0)))))
           for e, n in zip(E, N)]
    a = np.radians(5.0)  # the world frame is the ENU frame turned by -5 degrees
    W = np.c_[E, N] @ np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    poses = [dict(position=dict(x=x, y=y, z=0.0)) for x, y in W]
    R = ps.world_to_enu(poses, gps)
    enu = np.c_[E, N]
    assert np.allclose((W - W.mean(0)) @ R[:2, :2].T, enu - enu.mean(0), atol=0.5)  # metres, small-angle GPS approx
    assert np.isclose(np.degrees(np.arctan2(R[1, 0], R[0, 0])), 5.0, atol=0.2)


def test_short_tracks_keep_the_world_frame():
    gps = [dict(lat=37.0, long=-122.0)] * 5
    poses = [dict(position=dict(x=0.0, y=0.0, z=0.0))] * 5
    assert np.allclose(ps.world_to_enu(poses, gps), np.eye(3))


def test_lidar_depth_projects_a_point_ahead_to_the_principal_point():
    K = np.array([[300.0, 0, 160], [0, 300.0, 90], [0, 0, 1]])
    Rc, tc = camera_R(90.0), np.array([10.0, 5.0, 1.8])  # looking east
    pts = np.array([[30.0, 5.0, 1.8], [-30.0, 5.0, 1.8]])  # 20 m ahead, and one behind the camera
    d = ps.lidar_depth(pts, Rc, tc, K, (180, 320))
    assert d[90, 160] == pytest.approx(20.0) and (d > 0).sum() == 1


def test_sun_from_the_pose_matches_the_public_api():
    when, lat, lon, heading = "2019-05-11 16:35:46", 37.41, -122.12, 135.0
    import sun

    az, el = sun.sun_az_el(lat, lon, when)
    s_pose = camera_R(heading).T @ sun.enu_from_az_el(az, el)
    s_api = suncheck.sun_in_camera_frame(lat, lon, when, heading_deg=heading)
    assert np.allclose(s_pose, s_api, atol=1e-9)
