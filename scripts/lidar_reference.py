"""LiDAR reference for every cached KITTI frame: what the shadow check *should* find.

    python scripts/lidar_reference.py --cache outputs/cache --out outputs/study/kitti_lidar.csv

Reads the velodyne scan of each cached frame straight out of the remote KITTI zip (HTTP range
requests, about 2 MB a frame) and projects it into the quarter-resolution camera of the cache.
Per frame and depth model it stores
    absrel        mean |d - L| / L of the metric prediction
    absrel_med    the same after median scaling (shape error only)
    gamma         slope of log d against log L: the depth-shape exponent of the warp
                  d = m (L / m)^gamma; gamma < 1 is depth compressed towards its median
    gamma_road    the same on road pixels only
A shadow sees a focal factor k and a shape exponent gamma only through their product (the
focal-compression corollary of the paper), so with the calibrated camera the shadow-optimal focal
factor should track gamma, and with the model's own camera k_own * gamma.
Finished drives are kept in --parts, so the script can be stopped and restarted.
"""
import argparse
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import kitti  # noqa: E402

S3 = "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data"
MODELS = ["dav2-s", "dav2-l", "depthpro", "moge2-l"]


def calib_for(date, cache_dir):
    from remotezip import RemoteZip

    d = Path(cache_dir) / date
    if not (d / "calib_cam_to_cam.txt").exists():
        d.mkdir(parents=True, exist_ok=True)
        with RemoteZip(f"{S3}/{date}_calib.zip") as z:
            for n in z.namelist():
                if n.endswith(".txt"):
                    (d / Path(n).name).write_bytes(z.read(n))
    return kitti.read_calib(d)


def project(points, calib, K_q, shape):
    """Sparse LiDAR depth in the quarter-resolution camera whose intrinsics are K_q."""
    sx, sy = K_q[0, 0] / calib["K2"][0, 0], K_q[1, 1] / calib["K2"][1, 1]
    P = np.diag([sx, sy, 1.0]) @ calib["P2"]
    P[0, 2] = K_q[0, 2]  # keep the principal point of the cache exactly
    P[1, 2] = K_q[1, 2]
    xyz1 = np.c_[points[:, :3], np.ones(len(points))]
    cam = calib["R_rect"] @ calib["T_velo2cam"] @ xyz1.T
    cam = cam[:, cam[2] > 1.0]
    uvw = P @ cam
    u, v, z = np.round(uvw[0] / uvw[2]).astype(int), np.round(uvw[1] / uvw[2]).astype(int), cam[2]
    h, w = shape
    keep = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    depth = np.zeros(shape, np.float32)
    order = np.argsort(-z[keep])  # nearest point wins
    depth[v[keep][order], u[keep][order]] = z[keep][order]
    return depth


def fit_gamma(d, L):
    """Slope of log d on log L, after dropping the 5% largest residuals of a first fit."""
    x, y = np.log(L), np.log(d)
    if x.size < 50 or np.ptp(x) < 0.3:
        return np.nan
    g, c = np.polyfit(x, y, 1)
    r = np.abs(y - (g * x + c))
    keep = r <= np.quantile(r, 0.95)
    return float(np.polyfit(x[keep], y[keep], 1)[0])


def frame_rows(npz_path, L):
    z = np.load(npz_path)
    receiver = z["receiver"]
    rows = []
    for m in MODELS:
        if f"depth_{m}" not in z:
            continue
        d = z[f"depth_{m}"].astype(np.float64)
        ok = (L > 1.0) & (L < 80) & (d > 0)
        if ok.sum() < 50:
            continue
        dl, ll = d[ok], L[ok]
        s = np.median(ll / dl)
        road = ok & receiver
        rows.append(dict(model=m, n_lidar=int(ok.sum()), absrel=float(np.mean(np.abs(dl - ll) / ll)),
                         absrel_med=float(np.mean(np.abs(s * dl - ll) / ll)), scale_ratio=float(s),
                         gamma=fit_gamma(dl, ll), gamma_road=fit_gamma(d[road], L[road]) if road.sum() >= 50 else np.nan))
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default=str(ROOT / "outputs" / "cache"))
    p.add_argument("--calib", default=str(ROOT / "outputs" / "kitti_calib"))
    p.add_argument("--parts", default=str(ROOT / "outputs" / "study" / "lidar_parts"))
    p.add_argument("--out", default=str(ROOT / "outputs" / "study" / "kitti_lidar.csv"))
    p.add_argument("--drives", nargs="*", default=None, help="only these drives")
    a = p.parse_args()
    from remotezip import RemoteZip

    index = pd.read_csv(Path(a.cache) / "frames.csv")
    parts = Path(a.parts)
    parts.mkdir(parents=True, exist_ok=True)
    for drive, g in index.groupby("drive"):
        if a.drives and drive not in a.drives:
            continue
        done = parts / f"{drive}.csv"
        if done.exists():
            continue
        rows = None
        for attempt in range(4):  # S3 resets connections now and then
            try:
                calib = calib_for(drive[:10], a.calib)
                rows = []
                with RemoteZip(f"{S3}/{drive}/{drive}_sync.zip") as zz:
                    velo = {int(Path(n).stem): n for n in zz.namelist()
                            if "/velodyne_points/data/" in n and n.endswith(".bin")}
                    for _, r in g.iterrows():
                        npz = Path(a.cache) / f"{r['key']}.npz"
                        if int(r["frame"]) not in velo:
                            continue
                        pts = np.frombuffer(zz.read(velo[int(r["frame"])]), np.float32).reshape(-1, 4)
                        z = np.load(npz)
                        L = project(pts, calib, z["K"], z["receiver"].shape)
                        rows += [dict(key=r["key"], drive=drive, frame=int(r["frame"]), **e) for e in frame_rows(npz, L)]
                break
            except (zipfile.BadZipFile, KeyError) as e:
                print(f"skip {drive}: {e!r}", flush=True)
                break
            except Exception as e:  # network: RemoteIOError, ConnectionError, OSError
                print(f"retry {drive} ({attempt + 1}): {e!r}"[:200], flush=True)
                rows = None
                time.sleep(10 * (attempt + 1))
        if rows is None:
            continue
        tmp = done.with_suffix(".tmp")
        pd.DataFrame(rows).to_csv(tmp, index=False)
        tmp.rename(done)
        print(drive, len(rows), "rows", flush=True)
    df = pd.concat([pd.read_csv(f) for f in sorted(parts.glob("*.csv")) if f.stat().st_size > 1], ignore_index=True)
    df.to_csv(a.out, index=False)
    print(len(df), "rows ->", a.out)
    print(df.groupby("model")[["absrel", "absrel_med", "gamma", "gamma_road"]].median().round(3).to_string())
