"""PandaSet geometry: camera poses, the world-to-ENU rotation and LiDAR projection (used by
scripts/cache_pandaset.py; kept free of torch so it can be tested)."""
import gzip
import pickle

import numpy as np


def quat_R(q):
    """Rotation matrix of a unit quaternion {w, x, y, z} (camera -> world for PandaSet poses)."""
    w, x, y, z = q["w"], q["x"], q["y"], q["z"]
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def world_to_enu(poses, gps):
    """Rotation about the vertical that takes the world x, y to east, north (Procrustes on the trajectory)."""
    lat, lon = np.array([g["lat"] for g in gps]), np.array([g["long"] for g in gps])
    E = np.radians(lon - lon[0]) * 6378137.0 * np.cos(np.radians(lat[0]))
    N = np.radians(lat - lat[0]) * 6378137.0
    P = np.array([[q["position"]["x"], q["position"]["y"]] for q in poses])
    n = min(len(P), len(E))
    A, B = P[:n] - P[:n].mean(0), np.c_[E[:n], N[:n]] - np.c_[E[:n], N[:n]].mean(0)
    if np.linalg.norm(np.diff(B, axis=0), axis=1).sum() < 20:  # too short to fit: the world frame is ENU
        return np.eye(3)
    U, _, Vt = np.linalg.svd(A.T @ B)
    R2 = (U @ Vt).T
    if np.linalg.det(R2) < 0:
        return np.eye(3)
    R = np.eye(3)
    R[:2, :2] = R2
    return R


def read_lidar(path):
    """World x, y, z of a PandaSet LiDAR sweep (.pkl or .pkl.gz pandas frame)."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        df = pickle.load(f)
    return np.asarray(df[["x", "y", "z"]].values, np.float64)


def lidar_depth(pts_world, Rc, tc, K, shape):
    """Sparse depth image of world points seen by a camera with rotation Rc (camera -> world) and centre tc."""
    cam = (pts_world - tc) @ Rc  # world -> camera: R^T (p - t)
    cam = cam[cam[:, 2] > 1.0]
    u = np.round(K[0, 0] * cam[:, 0] / cam[:, 2] + K[0, 2]).astype(int)
    v = np.round(K[1, 1] * cam[:, 1] / cam[:, 2] + K[1, 2]).astype(int)
    h, w = shape
    keep = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    depth = np.zeros(shape, np.float32)
    order = np.argsort(-cam[keep, 2])  # nearest point wins
    depth[v[keep][order], u[keep][order]] = cam[keep, 2][order]
    return depth
