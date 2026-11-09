"""LiDAR reference for the PandaSet cache: the same per-frame metrics as scripts/lidar_reference.py (KITTI).

    python scripts/lidar_panda.py --cache outputs/cache_panda --out outputs/study/panda_lidar.csv

The cache already holds the LiDAR depth of each frame (both PandaSet sensors, world points projected
into the camera with its pose), so this only scores the depth models against it.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from lidar_reference import frame_rows  # noqa: E402

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default=str(ROOT / "outputs" / "cache_panda"))
    p.add_argument("--out", default=str(ROOT / "outputs" / "study" / "panda_lidar.csv"))
    a = p.parse_args()
    rows = []
    for k in pd.read_csv(Path(a.cache) / "frames.csv").key:
        z = np.load(Path(a.cache) / f"{k}.npz")
        if "lidar" not in z:
            continue
        drive, frame = k.rsplit("_", 1)
        for r in frame_rows(Path(a.cache) / f"{k}.npz", z["lidar"].astype(np.float64)):
            rows.append(dict(key=k, drive=drive, frame=int(frame), **r))
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    print(len(df), "rows,", df.key.nunique(), "frames ->", a.out)
