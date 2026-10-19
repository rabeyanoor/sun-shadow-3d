"""Sun direction from GPS + time, expressed in the KITTI camera frame."""
from datetime import timedelta

import numpy as np
import pandas as pd
import pvlib


def sun_az_el(lat, lon, when_utc):
    """Solar azimuth (deg, clockwise from north) and apparent elevation (deg)."""
    t = pd.DatetimeIndex([pd.Timestamp(when_utc).tz_localize("UTC")])
    pos = pvlib.solarposition.get_solarposition(t, lat, lon)
    return float(pos["azimuth"].iloc[0]), float(pos["apparent_elevation"].iloc[0])


def enu_from_az_el(az_deg, el_deg):
    az, el = np.radians(az_deg), np.radians(el_deg)
    return np.array([np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), np.sin(el)])


def rot_imu_to_enu(roll, pitch, yaw):
    # same convention as the KITTI devkit (convertOxtsToPose): R = Rz(yaw) Ry(pitch) Rx(roll)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def sun_in_camera(oxts, i, timestamp_local, calib, utc_offset_hours):
    """Unit vector pointing from the scene towards the sun, in rectified cam2 coordinates."""
    when_utc = timestamp_local - timedelta(hours=utc_offset_hours)
    az, el = sun_az_el(oxts["lat"][i], oxts["lon"][i], when_utc)
    s_enu = enu_from_az_el(az, el)
    R = rot_imu_to_enu(oxts["roll"][i], oxts["pitch"][i], oxts["yaw"][i])
    s_cam = calib["R_imu2cam"] @ (R.T @ s_enu)
    return s_cam / np.linalg.norm(s_cam), az, el
