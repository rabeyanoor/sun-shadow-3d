"""Score every cached KITTI frame with the original and the proposed shadow checks.

    python scripts/shadow_eval.py --cache outputs/cache --workers 6

Reads the per-frame .npz files written by scripts/cache_kitti.py. For each frame and depth model
it renders under
    the corruption protocol (scale, shape, focal, sun azimuth; five values each),
    a fine focal grid (13 log-spaced factors 0.5..2) for self-calibration,
    a sun grid (azimuth -60..60 and elevation -20..20 degrees) for sun localisation,
    the model's own predicted camera (Depth Pro, MoGe-2) against the calibrated one,
with the two-state and the three-state renderer, and scores both against two observations:
    net  ISTD-trained SegFormer: soft = probability, binary = probability > 0.5
    thr  no detector: soft = road darkness, binary = dark-asphalt threshold
One CSV row per (frame, model, observation, experiment, value).
"""
import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import observability as obs  # noqa: E402
from corruptions import CORRUPTIONS, K_GRID, raise_sun, rotate_about_up, scale_focal  # noqa: E402

import shadows  # noqa: E402

MODELS = ["dav2-s", "dav2-l", "depthpro", "moge2-l"]
SUN_AZ = list(range(-60, 61, 10))
SUN_EL = list(range(-20, 21, 5))


def threshold_mask(img, receiver, dark_ratio=0.55):
    # same rule as scene.observed_shadow, without importing torch in the workers
    lum = img.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
    dark = (lum < dark_ratio * np.percentile(lum[receiver], 80)) & receiver
    dark = ndimage.binary_opening(dark, iterations=1)
    return ndimage.median_filter(dark.astype(np.uint8), size=5).astype(bool) & receiver


def score_all(state3, rendered2, observations, region):
    out = {}
    for name, (soft, binary) in observations.items():
        e = obs.scores(state3, soft, binary, region)
        a, b = rendered2 & region, binary & region
        n = region.sum()
        r, o = a.sum() / n, b.sum() / n
        union = (a | b).sum()
        chance = r * o + (1 - r) * (1 - o)
        e.update(iou=float((a & b).sum() / union) if union else np.nan,
                 kappa=float(((a == b)[region].mean() - chance) / (1 - chance)) if chance < 1 else np.nan,
                 chance_iou=obs.chance_iou(r, o), rendered_frac=float(r), obs_frac=float(o),
                 auc2=obs.auc(soft, a, region & ~rendered2))
        out[name] = e
    return out


def run_frame(args):
    path, h_max, f_pred = args
    z = np.load(path)
    receiver, img, s, K, sky = z["receiver"], z["img"], z["sun"], z["K"], z["sky"]
    observations = {
        "net": (z["prob"].astype(np.float32),
                ndimage.median_filter((z["prob"] > 0.5).astype(np.uint8), size=5).astype(bool) & receiver),
        "thr": (obs.darkness(img, receiver), threshold_mask(img, receiver)),
    }
    key = Path(path).stem
    drive, frame = key.rsplit("_", 1)  # KITTI 2011_09_26_drive_0001_0000000007, PandaSet panda_080_0000000003
    meta = dict(key=key, drive=drive, frame=int(frame), focal_sens=float(obs.focal_sensitivity(s)),
                sun_x=float(s[0]), sun_y=float(s[1]), sun_z=float(s[2]))
    rows = []
    for m in MODELS:
        if f"depth_{m}" not in z:
            continue
        d = z[f"depth_{m}"].astype(np.float64)
        region = receiver & (d > 0)
        if region.sum() < 200:
            continue

        groups = {}

        def add(exp, value, dc, Kc, sc, noam=False):
            gy = obs.ground_level(dc, Kc, region)
            st = obs.render_shadow_amodal(dc, Kc, sc, gy, h_max=h_max, sky=sky, return_state=True, only=region)
            r2 = shadows.render_shadow(dc, Kc, sc, only=region)
            st_noam = obs.render_shadow3(dc, Kc, sc, gy, h_max=h_max, only=region) if noam else None
            for o, e in score_all(st, r2, observations, region).items():
                if noam:
                    e["auc_noam"] = obs.auc(observations[o][0], region & (st_noam == obs.SHADOW),
                                            region & (st_noam == obs.LIT))
                    e["det_frac_noam"] = float((region & (st_noam != obs.UNDETERMINED)).sum() / region.sum())
                rows.append(dict(**meta, model=m, obs=o, exp=exp, value=value, **e))
                groups.setdefault(exp, []).append((len(rows) - 1, st))

        for corr, (values, fn) in CORRUPTIONS.items():
            for v in values:
                add(corr, v, *fn(d.copy(), K.copy(), s.copy(), v))
        for k in K_GRID:
            add("focal_grid", float(k), d, scale_focal(K, k), s, noam=True)
        for a in SUN_AZ:
            add("sun_az_grid", a, d, K, rotate_about_up(s, a), noam=True)
        for e in SUN_EL:
            add("sun_el_grid", e, d, K, raise_sun(s, e))
        if np.isfinite(f_pred.get(m, np.nan)):
            k_own = f_pred[m]  # model focal / calibrated focal
            add("own_camera", 0, d, K, s)
            add("own_camera", 1, d, scale_focal(K, k_own), s)
            for r in rows[-4:]:
                r["k_own"] = k_own
        # common support: score every candidate of a group on the pixels that all of them determine
        for members in groups.values():
            support = region.copy()
            for _, st in members:
                support &= st != obs.UNDETERMINED
            for i, st in members:
                soft = observations[rows[i]["obs"]][0]
                rows[i]["auc_cs"] = obs.auc(soft, support & (st == obs.SHADOW), support & (st == obs.LIT))
                rows[i]["cs_frac"] = float(support.sum() / region.sum())
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default=str(ROOT / "outputs" / "cache"))
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--h-max", type=float, default=20.0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", default=str(ROOT / "outputs" / "study" / "kitti_eval.csv"))
    p.add_argument("--parts", default=None, help="directory of per-frame CSVs; finished frames are skipped")
    p.add_argument("--shard", action="append", default=None,
                   help="i/n: only frames i, i+n, i+2n, ... (repeatable), to split the work over machines")
    a = p.parse_args()
    index = pd.read_csv(Path(a.cache) / "frames.csv")
    jobs = []
    for _, r in index.iterrows():
        f_pred = {m: r[f"f_{m}"] / r["f_true"] for m in MODELS if f"f_{m}" in r and np.isfinite(r[f"f_{m}"])}
        jobs.append((str(Path(a.cache) / f"{r['key']}.npz"), a.h_max, f_pred))
    jobs = jobs[: a.limit]
    if a.shard:
        picks = [tuple(map(int, x.split("/"))) for x in a.shard]
        jobs = [j for i, j in enumerate(jobs) if any(i % n == k for k, n in picks)]
    parts = Path(a.parts) if a.parts else None
    if parts:
        parts.mkdir(parents=True, exist_ok=True)
        todo = [j for j in jobs if not (parts / f"{Path(j[0]).stem}.csv").exists()]
        print(f"{len(jobs) - len(todo)} frames done, {len(todo)} to go", flush=True)
        with Pool(a.workers, maxtasksperchild=20) as pool:
            for i, (job, rows) in enumerate(zip(todo, pool.imap(run_frame, todo, chunksize=1))):
                tmp = parts / f"{Path(job[0]).stem}.tmp"
                pd.DataFrame(rows).to_csv(tmp, index=False)
                tmp.rename(tmp.with_suffix(".csv"))  # atomic: a killed run never leaves half a frame
                print(f"{i + 1}/{len(todo)}", flush=True)
        frames = [pd.read_csv(f) for f in sorted(parts.glob("*.csv")) if f.stat().st_size > 1]
        df = pd.concat(frames, ignore_index=True)
    else:
        with Pool(a.workers) as pool:
            res = pool.map(run_frame, jobs, chunksize=1)
        df = pd.DataFrame([r for rows in res for r in rows])
    df = df.merge(index[["key", "stamp", "azimuth", "elevation"]], on="key", how="left")
    df.to_csv(a.out, index=False)
    print(len(df), "rows,", df.key.nunique(), "frames ->", a.out)
