"""Random-street benchmark: when does a shadow check score the true geometry best?

    python scripts/synthetic_benchmark.py --scenes 300 --workers 6

Every scene is a random street (buildings, trees, cars, poles) under a random sun, seen by the
KITTI camera at quarter resolution. Three observation conditions separate the failure causes:
    visible    exact shadows of the casters that are in view
    offscreen  exact shadows of all casters, also those above the image or behind the camera
    detector   offscreen + a simulated detector (darkening, dark albedo patches, noise)
Each check renders shadows from corrupted depth and scores them on the road:
    iou, kappa    two-state renderer, all road pixels (the original check)
    auc2          two-state renderer, AUC of the soft observed score (prevalence-invariant only)
    auc3          three-state renderer, AUC on the determined pixels (the proposed check)
Writes outputs/study/synthetic_benchmark.csv and synthetic_focal_grid.csv.
"""
import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import observability as obs  # noqa: E402
from corruptions import CORRUPTIONS, K_GRID, scale_focal, warp  # noqa: E402

import shadows  # noqa: E402
import synthetic as syn  # noqa: E402

K, (H, W) = syn.K_KITTI_Q, syn.HW_KITTI_Q
R2 = dict(max_dist=40.0, n_steps=96)


def evaluate(state3, rendered2, observed, region):
    soft, binary = observed
    out = obs.scores(state3, soft, binary, region)
    a, b = rendered2 & region, binary & region
    union = (a | b).sum()
    r, o = a.sum() / region.sum(), b.sum() / region.sum()
    pa, pb = r, o
    chance = pa * pb + (1 - pa) * (1 - pb)
    out.update(iou=float((a & b).sum() / union) if union else np.nan,
               kappa=float(((a == b)[region].mean() - chance) / (1 - chance)) if chance < 1 else np.nan,
               chance_iou=obs.chance_iou(r, o), rendered_frac=float(r), obs_frac=float(o),
               auc2=obs.auc(soft, a, region & ~rendered2))
    return out


def run_scene(args):
    seed, h_max, grid_only = args
    rng = np.random.default_rng(seed)
    boxes = syn.random_street(rng)
    depth, ground, _ = syn.make_scene(K, H, W, boxes=boxes)
    P = shadows.backproject(depth, K)
    region = ground & (np.abs(P[..., 0]) < 5) & (depth < 50)
    if region.sum() < 300:
        return [], []
    s = syn.random_sun(rng)
    truth_all = syn.exact_shadow(K, H, W, depth, boxes, s) & region
    truth_vis = syn.exact_shadow(K, H, W, depth, syn.visible_boxes(K, H, W, depth, boxes), s) & region
    soft_det, bin_det = syn.simulated_detector(rng, truth_all, region)
    conds = {"visible": (truth_vis.astype(float), truth_vis), "offscreen": (truth_all.astype(float), truth_all),
             "detector": (soft_det, bin_det)}
    meta = dict(scene=seed, azimuth=float(np.degrees(np.arctan2(s[0], s[2]))), elevation=float(np.degrees(-np.arcsin(s[1]))),
                offscreen_frac=float((truth_all & ~truth_vis).sum() / max(truth_all.sum(), 1)),
                focal_sens=float(obs.focal_sensitivity(s)))
    rows, grid = [], []

    def render(d, Kc, sc):
        state = obs.render_shadow_amodal(d, Kc, sc, obs.ground_level(d, Kc, region), h_max=h_max, return_state=True,
                                         only=region)
        return state, shadows.render_shadow(d, Kc, sc, only=region, **R2)

    for corr, (values, fn) in ({} if grid_only else CORRUPTIONS).items():
        for v in values:
            st, r2 = render(*fn(depth.copy(), K.copy(), s.copy(), v))
            for c, o in conds.items():
                rows.append(dict(**meta, condition=c, corruption=corr, value=v, **evaluate(st, r2, o, region)))
    states = {}
    for k in K_GRID:  # fine focal grid for self-calibration, and the same factors as a shape warp
        Kk = scale_focal(K, k)
        st, r2 = render(depth, Kk, s)
        noam = obs.render_shadow3(depth, Kk, s, obs.ground_level(depth, Kk, region), h_max=h_max, only=region)
        dg = warp(depth, k)
        st_g = obs.render_shadow_amodal(dg, K, s, obs.ground_level(dg, K, region), h_max=h_max, return_state=True,
                                        only=region)
        a, b = region & (st == obs.SHADOW), region & (st_g == obs.SHADOW)
        equiv = float((a & b).sum() / max((a | b).sum(), 1))  # k and gamma = k cast the same shadow? (cor. confound)
        states[k] = st
        for c in ("offscreen", "detector"):
            e = evaluate(st, r2, conds[c], region)
            soft = conds[c][0]
            grid.append(dict(**meta, condition=c, k=k, auc3=e["auc"], iou=e["iou"], kappa=e["kappa"], auc2=e["auc2"],
                             auc_am=e["auc_am"], det_frac=e["det_frac"],
                             auc_noam=obs.auc(soft, region & (noam == obs.SHADOW), region & (noam == obs.LIT)),
                             auc_shape=obs.auc(soft, region & (st_g == obs.SHADOW), region & (st_g == obs.LIT)),
                             equiv_iou=equiv))
    support = region.copy()
    for st in states.values():
        support &= st != obs.UNDETERMINED
    for g in grid:
        st, soft = states[g["k"]], conds[g["condition"]][0]
        g["auc_cs"] = obs.auc(soft, support & (st == obs.SHADOW), support & (st == obs.LIT))
        g["cs_frac"] = float(support.sum() / region.sum())
    return rows, grid


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", type=int, default=300)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--h-max", type=float, default=20.0)
    p.add_argument("--seed0", type=int, default=0, help="first scene seed (the h_max ablation uses other seeds)")
    p.add_argument("--tag", default="")
    p.add_argument("--grid-only", action="store_true", help="only the focal/shape grid (synthetic_focal_grid*.csv)")
    p.add_argument("--out", default=str(ROOT / "outputs" / "study"))
    a = p.parse_args()
    with Pool(a.workers) as pool:
        res = pool.map(run_scene, [(a.seed0 + i, a.h_max, a.grid_only) for i in range(a.scenes)], chunksize=2)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if not a.grid_only:
        pd.DataFrame([r for rows, _ in res for r in rows]).to_csv(out / f"synthetic_benchmark{a.tag}.csv", index=False)
    pd.DataFrame([g for _, grid in res for g in grid]).to_csv(out / f"synthetic_focal_grid{a.tag}.csv", index=False)
    print("scenes used:", sum(bool(g) for _, g in res))
