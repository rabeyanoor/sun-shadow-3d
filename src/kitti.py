"""Small readers for KITTI raw drives (oxts, timestamps, calibration, images, velodyne)."""
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

OXTS_FIELDS = [
    "lat", "lon", "alt", "roll", "pitch", "yaw",
    "vn", "ve", "vf", "vl", "vu", "ax", "ay", "az", "af", "al", "au",
    "wx", "wy", "wz", "wf", "wl", "wu", "pos_accuracy", "vel_accuracy",
    "navstat", "numsats", "posmode", "velmode", "orimode",
]


def read_timestamps(path):
    # lines look like "2011-09-26 13:02:25.951199488" -> drop the extra nanosecond digits
    out = []
    for line in Path(path).read_text().split("\n"):
        line = line.strip()
        if line:
            out.append(datetime.strptime(line[:26], "%Y-%m-%d %H:%M:%S.%f"))
    return out


def read_oxts(drive_dir):
    files = sorted((Path(drive_dir) / "oxts" / "data").glob("*.txt"))
    rows = [np.array(f.read_text().split(), dtype=np.float64) for f in files]
    data = np.stack(rows)
    return {name: data[:, i] for i, name in enumerate(OXTS_FIELDS)}


def _read_calib_file(path):
    out = {}
    for line in Path(path).read_text().split("\n"):
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        try:
            out[key] = np.array(val.split(), dtype=np.float64)
        except ValueError:
            pass  # calib_time line
    return out


def read_calib(date_dir):
    date_dir = Path(date_dir)
    imu2velo = _read_calib_file(date_dir / "calib_imu_to_velo.txt")
    velo2cam = _read_calib_file(date_dir / "calib_velo_to_cam.txt")
    cam = _read_calib_file(date_dir / "calib_cam_to_cam.txt")

    def rt(d):
        T = np.eye(4)
        T[:3, :3] = d["R"].reshape(3, 3)
        T[:3, 3] = d["T"]
        return T

    R_rect = np.eye(4)
    R_rect[:3, :3] = cam["R_rect_00"].reshape(3, 3)
    P2 = cam["P_rect_02"].reshape(3, 4)

    # velodyne -> rectified cam2 pixels:  P2 @ R_rect @ T_velo2cam
    T_imu2velo = rt(imu2velo)
    T_velo2cam = rt(velo2cam)
    return {
        "T_imu2velo": T_imu2velo,
        "T_velo2cam": T_velo2cam,
        "R_rect": R_rect,
        "P2": P2,
        "K2": P2[:, :3],
        # rotation that takes a direction in the IMU frame to the rectified camera frame
        "R_imu2cam": (R_rect @ T_velo2cam @ T_imu2velo)[:3, :3],
    }


def load_image(drive_dir, idx, cam=2):
    path = Path(drive_dir) / f"image_0{cam}" / "data" / f"{idx:010d}.png"
    return np.asarray(Image.open(path).convert("RGB"))


def load_velo(drive_dir, idx):
    path = Path(drive_dir) / "velodyne_points" / "data" / f"{idx:010d}.bin"
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)


def velo_to_depth(points, calib, h, w):
    """Project lidar points into cam2 and return a sparse depth map (0 = no point)."""
    xyz1 = np.c_[points[:, :3], np.ones(len(points))]
    cam = (calib["R_rect"] @ calib["T_velo2cam"] @ xyz1.T)
    cam = cam[:, cam[2] > 0.5]
    uvw = calib["P2"] @ cam
    u = np.round(uvw[0] / uvw[2]).astype(int)
    v = np.round(uvw[1] / uvw[2]).astype(int)
    z = cam[2]
    keep = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    depth = np.zeros((h, w), np.float32)
    # write far points first so the nearest point wins when two land on one pixel
    order = np.argsort(-z[keep])
    depth[v[keep][order], u[keep][order]] = z[keep][order]
    return depth
