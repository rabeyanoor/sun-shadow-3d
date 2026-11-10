"""Pixel-level agreement at the true geometry: precision, recall, F1 (= Dice) and false-positive rate.

    python scripts/pixel_metrics.py --cache outputs/cache --out outputs/study/kitti_pixel_metrics.csv [--prob2 DIR]

For every cached frame and depth model, at the calibrated camera and the sun from GPS and time, the rendered
shadow is compared with the binary observed shadow (net: detector probability > 0.5; thr: dark-asphalt
threshold; net2: detector v2 > 0.5, with --prob2), pixel by pixel:
    two_state  plain renderer on all road pixels (lit = no visible hit)
    suncheck   three-state amodal renderer, on the determined road pixels only
Rendered shadow is the positive class: precision = TP / rendered shadow, recall = TP / observed shadow,
FPR = FP / observed lit. The counts are stored, so that metrics can be pooled over frames or drives.
"""
import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from shadow_eval import MODELS, detector_obs, threshold_mask  # noqa: E402

import observability as obs  # noqa: E402

import shadows  # noqa: E402


def counts(rendered, observed, region):
    r, o = rendered & region, observed & region
    return dict(tp=int((r & o).sum()), fp=int((r & ~o).sum()), fn=int((~r & o).sum()), tn=int((~r & ~o & region).sum()))


def run(job):
    path, prob2 = job
    z = np.load(path)
    receiver, img, s, K, sky = z["receiver"], z["img"], z["sun"], z["K"], z["sky"]
    observed = {"net": detector_obs(z["prob"], receiver)[1], "thr": threshold_mask(img, receiver)}
    if prob2:
        observed["net2"] = detector_obs(np.load(prob2), receiver)[1]
        observed["fuse"] = observed["net2"] & observed["thr"]
    key = Path(path).stem
    drive = key.rsplit("_", 1)[0]
    rows = []
    for m in MODELS:
        if f"depth_{m}" not in z:
            continue
        d = z[f"depth_{m}"].astype(np.float64)
        region = receiver & (d > 0)
        if region.sum() < 200:
            continue
        gy = obs.ground_level(d, K, region)
        st = obs.render_shadow_amodal(d, K, s, gy, h_max=20.0, sky=sky, return_state=True, only=region)
        two = shadows.render_shadow(d, K, s, only=region)
        det = region & (st != obs.UNDETERMINED)
        for o, b in observed.items():
            rows.append(dict(key=key, drive=drive, model=m, obs=o, renderer="two_state", coverage=1.0,
                             **counts(two, b, region)))
            rows.append(dict(key=key, drive=drive, model=m, obs=o, renderer="suncheck",
                             coverage=float(det.sum() / region.sum()), **counts(st == obs.SHADOW, b, det)))
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default=str(ROOT / "outputs" / "cache"))
    p.add_argument("--out", default=str(ROOT / "outputs" / "study" / "kitti_pixel_metrics.csv"))
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--prob2", default=None, help="folder of detector-v2 probabilities <key>.npy")
    a = p.parse_args()
    keys = pd.read_csv(Path(a.cache) / "frames.csv").key
    jobs = [(str(Path(a.cache) / f"{k}.npz"), str(Path(a.prob2) / f"{k}.npy") if a.prob2 else None) for k in keys]
    with Pool(a.workers, maxtasksperchild=50) as pool:
        res = pool.map(run, jobs, chunksize=4)
    df = pd.DataFrame([r for rows in res for r in rows])
    df.to_csv(a.out, index=False)
    print(len(df), "rows,", df.key.nunique(), "frames ->", a.out)
