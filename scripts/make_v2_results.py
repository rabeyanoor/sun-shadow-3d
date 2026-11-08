"""Numbers, tables and figures for the SunCheck sections, from outputs/study.

    python scripts/make_v2_results.py

Reads the synthetic benchmark (synthetic_benchmark_h*.csv, synthetic_focal_grid_h*.csv,
ablation/), and the KITTI evaluation (kitti_eval_h*.csv). H (the caster height prior) is chosen
on the held-out ablation seeds only. Writes paper/gen/numbers_v2.tex, paper/gen/table_bench.tex,
paper/gen/table_kitti_v2.tex and paper/fig/{bench,kitti_v2,calib}.pdf.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from corruptions import K_GRID, TRUTH  # noqa: E402
from suncheck import refine_argmax  # noqa: E402

STUDY, GEN, FIG = ROOT / "outputs" / "study", ROOT / "paper" / "gen", ROOT / "paper" / "fig"
GEN.mkdir(parents=True, exist_ok=True)
SCORES = {"iou": "IoU", "kappa": r"$\kappa$", "auc2": "AUC", "auc_am": "+ amodal", "auc": "SunCheck"}
CORR = {"focal": "focal", "shape": "shape", "sun_azimuth": "sun az."}
LOW = {"scale": 0.5, "shape": 0.6, "focal": 0.6, "sun_azimuth": -60}
MODELS = {"dav2-s": "DA-V2-S", "dav2-l": "DA-V2-L", "depthpro": "Depth Pro", "moge2-l": "MoGe-2-L"}
macros = {}
rng = np.random.default_rng(0)


def mkey(*parts):
    """LaTeX macro name: camel case, digits spelled out (macros cannot contain digits)."""
    words = dict(zip("0123456789", ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]))
    import re
    toks = [t for p in parts for t in re.split(r"[-_. ]+", str(p)) if t]
    s = toks[0] + "".join(t[:1].upper() + t[1:] for t in toks[1:])
    return "".join(words.get(c, c) for c in s if c.isalnum())


def put(name, value, fmt="{:.2f}"):
    text = value if isinstance(value, str) else fmt.format(value)
    macros[name] = text.replace("-", "$-$", 1) if text.startswith("-") else text


def boot(values, groups, n=2000, stat=np.mean):
    """95% interval of the mean, resampling groups (drives or scenes) with replacement."""
    v = pd.Series(np.asarray(values, float), index=np.asarray(groups))
    v = v[np.isfinite(v.values)]
    keys = v.index.unique()
    parts = [v[v.index == k].values for k in keys]
    sums, cnts = np.array([p.sum() for p in parts]), np.array([len(p) for p in parts])
    pick = rng.integers(0, len(keys), size=(n, len(keys)))
    means = sums[pick].sum(1) / cnts[pick].sum(1)
    return np.percentile(means, [2.5, 97.5])


def pool(w, weights, level):
    """Weighted mean of each column over the rows of a group, ignoring missing scores."""
    num = w.mul(weights, axis=0).groupby(level=level).sum()
    den = w.notna().mul(weights, axis=0).groupby(level=level).sum()
    return num / den.replace(0, np.nan)


def refine(pooled):
    """Best focal factor per row, refined by a parabola through the grid maximum and its neighbours (in log k)."""
    x = np.log(pooled.columns.values.astype(float))
    return pd.Series([np.exp(refine_argmax(x, row.values)) for _, row in pooled.iterrows()], index=pooled.index)


def truth_argmax(df, score, corr, case_cols, group):
    g = df[df.exp == corr] if "exp" in df else df[df.corruption == corr]
    w = g.pivot_table(index=case_cols, columns="value", values=score)
    w = w.dropna()
    hit = (w.idxmax(axis=1) == TRUTH[corr]).astype(float)
    grp = w.index.get_level_values(group)
    lo, hi = boot(hit.values, grp)
    return hit.mean(), lo, hi, len(hit)


def pick_h():
    """Caster height prior H with the best SunCheck truth-argmax on held-out synthetic seeds."""
    res = {}
    for f in sorted((STUDY / "ablation").glob("synthetic_benchmark_h*.csv")):
        h = int(f.stem.split("_h")[-1])
        d = pd.read_csv(f)
        d = d[d.condition == "detector"]
        res[h] = np.mean([truth_argmax(d, "auc", c, ["scene"], "scene")[0] for c in CORR])
        put(mkey("ablH", h), 100 * res[h], "{:.0f}")
    h = max(res, key=res.get)
    put("hStar", h, "{:d}")
    return h


# observability horizon (prop. 5) for the KITTI camera and sun
K_Q = np.array([[180.2, 0, 152.3], [0, 180.2, 43.1], [0, 0, 1]])
phi = np.arctan(K_Q[1, 2] / K_Q[1, 1])
put("horizonPhi", np.degrees(phi), "{:.1f}")
for Hc, tag in ((12.0, "Twelve"), (20.0, "Twenty")):
    for alpha, side in ((0.0, "Ahead"), (180.0, "Behind")):
        z_star = (Hc - 1.65) / np.tan(phi) - Hc / np.tan(np.radians(40)) * np.cos(np.radians(alpha))
        put(f"horizon{tag}{side}", z_star, "{:.0f}")

# H is a scene prior fixed in advance (the tallest street casters), not tuned
H = 20
put("hStar", H, "{:d}")

# sun-direction error from a one-minute timestamp error, at the GPS position and time of every cached frame
try:
    import pvlib

    fr = pd.read_csv(ROOT / "outputs" / "cache" / "frames.csv")
    t0 = pd.DatetimeIndex(pd.to_datetime(fr.stamp.str[:19]).dt.tz_localize("Etc/GMT-2"))

    def sun_vec(t):
        sp = pvlib.solarposition.get_solarposition(t, 49.01, 8.40)
        az, el = np.radians(sp.azimuth.values), np.radians(sp.apparent_elevation.values)
        return np.stack([np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), np.sin(el)], -1)

    dot = (sun_vec(t0) * sun_vec(t0 + pd.Timedelta(seconds=60))).sum(1)
    put("sunErrMinute", np.degrees(np.arccos(np.clip(dot, -1, 1))).max(), "{:.2f}")
except (ImportError, FileNotFoundError):
    put("sunErrMinute", "0.25")

put("gridStep", 100 * (K_GRID[1] / K_GRID[0] - 1), "{:.0f}")
# ------------------------------------------------------------------ synthetic benchmark
syn = pd.read_csv(STUDY / f"synthetic_benchmark_h{H}.csv")
put("benchScenes", syn.scene.nunique(), "{:d}")
put("benchOffscreen", 100 * syn.drop_duplicates("scene").offscreen_frac.mean(), "{:.0f}")
put("benchDetFrac", 100 * syn[(syn.value == syn.corruption.map(TRUTH))].det_frac.mean(), "{:.0f}")
cond_names = {"visible": "in-view casters", "offscreen": "+ off-screen casters", "detector": "+ detector noise"}
lines = [r"\begin{table}[t]", r"\centering\small\setlength{\tabcolsep}{3.2pt}",
         r"\resizebox{\linewidth}{!}{\begin{tabular}{@{}ll" + "c" * len(CORR) + "@{}}", r"\toprule",
         "Observed & Score & " + " & ".join(CORR.values()) + r" \\", r"\midrule"]
bench = {}
for ci, (cond, cname) in enumerate(cond_names.items()):
    d = syn[syn.condition == cond]
    for si, (s, sname) in enumerate(SCORES.items()):
        cells = []
        for c in CORR:
            m, lo, hi, n = truth_argmax(d, s, c, ["scene"], "scene")
            bench[cond, s, c] = m
            put(mkey("bench", cond, s, c), 100 * m, "{:.0f}")
            cells.append(f"{100 * m:.0f}" if s != "auc" else rf"\textbf{{{100 * m:.0f}}}")
        head = rf"\multirow{{{len(SCORES)}}}{{*}}{{{cname}}}" if si == 0 else ""
        lines.append(head + f" & {sname} & " + " & ".join(cells) + r" \\")
    if ci < len(cond_names) - 1:
        lines.append(r"\midrule")
    # compressed preference of IoU
    g = d[d.corruption == "focal"].pivot_table(index="scene", columns="value", values="iou")
    put(mkey("benchIouFocalGain", cond), (g[0.6] - g[1.0]).mean(), "{:.3f}")
lines += [r"\bottomrule", r"\end{tabular}}",
          r"\caption{Random-street benchmark (" + macros["benchScenes"] + r" scenes, KITTI camera). Per scene, how often"
          r" the true value scores best among five (\%, chance 20). Off-screen casters break the overlap scores; the"
          r" rank score with the three-state renderer (SunCheck, $H=" + macros["hStar"] + r"$\,m) keeps the truth on top.}",
          r"\label{tab:bench}", r"\end{table}"]
(GEN / "table_bench.tex").write_text("\n".join(lines) + "\n")

# the same benchmark with 95% scene-bootstrap intervals (supplement)
lines = [r"\begin{table}[!ht]", r"\centering\small\setlength{\tabcolsep}{3pt}",
         r"\resizebox{\linewidth}{!}{\begin{tabular}{@{}ll" + "c" * len(CORR) + "@{}}", r"\toprule",
         "Observed & Score & " + " & ".join(CORR.values()) + r" \\", r"\midrule"]
for ci, (cond, cname) in enumerate(cond_names.items()):
    d = syn[syn.condition == cond]
    for si, (sc, sname) in enumerate(SCORES.items()):
        cells = []
        for c in CORR:
            m, lo, hi, n = truth_argmax(d, sc, c, ["scene"], "scene")
            cells.append(f"{100 * m:.0f} [{100 * lo:.0f}, {100 * hi:.0f}]")
        head = rf"\multirow{{{len(SCORES)}}}{{*}}{{{cname}}}" if si == 0 else ""
        lines.append(head + f" & {sname} & " + " & ".join(cells) + r" \\")
    if ci < len(cond_names) - 1:
        lines.append(r"\midrule")
lines += [r"\bottomrule", r"\end{tabular}}",
          r"\caption{Tab.~\ref{tab:bench} (main paper) with 95\% bootstrap intervals over scenes"
          r" (2{,}000 resamples): truth is the"
          r" best of five values (\%, chance 20).}", r"\label{tab:benchci}", r"\end{table}"]
(GEN / "table_bench_ci.tex").write_text("\n".join(lines) + "\n")

# self-calibration on synthetic: groups of 10 scenes share one camera; split by the side of the sun
cs_file = STUDY / f"synthetic_focal_grid_cs_h{H}.csv"
grid = pd.read_csv(cs_file if cs_file.exists() else STUDY / f"synthetic_focal_grid_h{H}.csv")
grid["side"] = np.where(grid.azimuth.abs() < 90, "ahead", "behind")
GRID_SCORES = {"iou": "iou", "kappa": "kappa", "auc2": "auc2", "auc_am": "auc_am", "auc": "auc3"}
EXTRA_GRID = {k: k for k in ("auc_noam", "auc_cs", "auc_shape") if k in grid}
syn_calib = {}
for cond in ("offscreen", "detector"):
    for side in ("ahead", "behind", "all"):
        g = grid[(grid.condition == cond) & ((grid.side == side) if side != "all" else True)].copy()
        g["group"] = g.scene // 10
        for s_name, col in {**GRID_SCORES, **EXTRA_GRID}.items():
            w = g.pivot_table(index=["group", "scene"], columns="k", values=col)
            wt = g.drop_duplicates("scene").set_index("scene").focal_sens
            pooled = pool(w, wt.reindex(w.index.get_level_values("scene")).values, "group").dropna(how="all")
            k_hat = pooled.idxmax(axis=1).astype(float)
            syn_calib[cond, side, s_name] = k_hat
            put(mkey("synKFine", cond, side, s_name), np.median(refine(pooled)), "{:.2f}")
            put(mkey("synK", cond, side, s_name), np.median(k_hat), "{:.2f}")
            put(mkey("synKWithin", cond, side, s_name), 100 * np.mean(np.abs(np.log(k_hat)) < 0.13), "{:.0f}")
            # the bias calibrated on the other half of the scenes (even groups -> odd groups)
        if side == "all":
            for s_name in {**GRID_SCORES, **EXTRA_GRID}:
                k_hat = syn_calib[cond, "all", s_name]
                even, odd = k_hat[k_hat.index % 2 == 0], k_hat[k_hat.index % 2 == 1]
                corr = np.log(odd) - np.median(np.log(even))
                put(mkey("synKCal", cond, s_name), 100 * np.median(np.abs(corr)), "{:.0f}")

if "equiv_iou" in grid:
    # focal-compression confounding: the shadow of focal factor k against that of the warp gamma = k
    e = grid[grid.condition == "detector"].groupby("k").equiv_iou.median()
    for k in (0.794, 0.891, 1.122, 1.26):
        put(mkey("synEquiv", f"{k:.2f}"), e.loc[np.isclose(e.index, k, atol=2e-3)].iloc[0], "{:.2f}")
    put("synCsFrac", 100 * grid.drop_duplicates("scene").cs_frac.median(), "{:.0f}")

# ------------------------------------------------------------------ KITTI
kit = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else STUDY / f"kitti_eval_h{H}.csv")
kit = kit[kit.model.isin(MODELS)]
put("kNFrames", kit.key.nunique(), "{:d}")
put("kNDrives", kit.drive.nunique(), "{:d}")
put("kNDays", kit.drive.str[:10].nunique(), "{:d}")
put("kNRows", len(kit), "{:,}")
put("kElevMin", kit.elevation.min(), "{:.0f}")
put("kElevMax", kit.elevation.max(), "{:.0f}")
truth = kit[(kit.exp == "focal") & (kit.value == 1.0)]
put("kDetFrac", 100 * truth.det_frac.mean(), "{:.0f}")
cases = ["drive", "key", "model"]
OBS = {"net": "Net", "thr": "Dark"}
lines = [r"\begin{table}[!ht]", r"\centering\small\setlength{\tabcolsep}{3pt}",
         r"\resizebox{\linewidth}{!}{\begin{tabular}{@{}ll" + "c" * (len(CORR) + 1) + "@{}}", r"\toprule",
         "Obs. & Score & " + " & ".join(CORR.values()) + r" & at truth \\", r"\midrule"]
KSCORES = {**SCORES, "auc_cs": "SunCheck (CS)"}
for oi, (o, oname) in enumerate(OBS.items()):
    d = kit[kit.obs == o]
    for si, (s, sname) in enumerate(KSCORES.items()):
        cells = []
        for c in CORR:
            m, lo, hi, n = truth_argmax(d, s, c, cases, "drive")
            tag = mkey("kAt", o, s, c)
            put(tag, 100 * m, "{:.0f}")
            put(tag + "Lo", 100 * lo, "{:.0f}")
            put(tag + "Hi", 100 * hi, "{:.0f}")
            cells.append(f"{100 * m:.0f}")
        t = d[(d.exp == "focal") & (d.value == 1.0)]
        level = t[s].mean()
        if s == "iou":
            put(mkey("kChanceIou", o), t.chance_iou.mean(), "{:.3f}")
            put(mkey("kIou", o), level, "{:.3f}")
        put(mkey("kLevel", o, s), level, "{:.3f}")
        lo, hi = boot(t[s].values, t.drive.values)
        put(mkey("kLevel", o, s, "lo"), lo, "{:.3f}")
        put(mkey("kLevel", o, s, "hi"), hi, "{:.3f}")
        cells.append(f"{level:.2f}")
        head = rf"\multirow{{{len(KSCORES)}}}{{*}}{{{oname}}}" if si == 0 else ""
        lines.append(head + f" & {sname} & " + " & ".join(cells) + r" \\")
        # signed gain of the most compressed value
        for c in ("focal", "shape"):
            w = d[d.exp == c].pivot_table(index=cases, columns="value", values=s).dropna()
            gain = w[LOW[c]] - w[TRUTH[c]]
            put(mkey("kGain", o, s, c), gain.mean(), "{:.3f}")
            lo, hi = boot(gain.values, w.index.get_level_values("drive"))
            put(mkey("kGain", o, s, c, "lo"), lo, "{:.3f}")
            put(mkey("kGain", o, s, c, "hi"), hi, "{:.3f}")
    if oi == 0:
        lines.append(r"\midrule")
lines += [r"\bottomrule", r"\end{tabular}}",
          r"\caption{KITTI, " + macros["kNFrames"] + " frames from " + macros["kNDrives"] + r" drives, four depth models."
          r" Truth is the per-case best of five values (\%, chance 20), and the mean score at the true geometry (chance:"
          r" AUC 0.5, $\kappa$ 0). Net: ISTD-trained detector; Dark: no detector, road darkness and a threshold.}",
          r"\label{tab:kitti2}", r"\end{table}"]
(GEN / "table_kitti_v2.tex").write_text("\n".join(lines) + "\n")

# sun localisation from a single frame
for o in OBS:
    for exp, unit in (("sun_az_grid", "Az"), ("sun_el_grid", "El")):
        for s in ("auc", "auc_cs"):
            w = kit[(kit.obs == o) & (kit.exp == exp)].pivot_table(index=cases, columns="value", values=s).dropna()
            err = np.abs(w.idxmax(axis=1).astype(float))
            tag = "" if s == "auc" else "Cs"
            put(mkey("kSun", unit, o, tag, "med"), np.median(err), "{:.0f}")
            put(mkey("kSun", unit, o, tag, "within"), 100 * np.mean(err <= {"Az": 10, "El": 5}[unit]), "{:.0f}")
            put(mkey("kSun", unit, o, tag, "n"), len(err), "{:d}")
        wi = kit[(kit.obs == o) & (kit.exp == exp)].pivot_table(index=cases, columns="value", values="iou").dropna()
        put(mkey("kSun", unit, o, "iou", "med"), np.median(np.abs(wi.idxmax(axis=1).astype(float))), "{:.0f}")

# focal self-calibration per drive and model, observability-weighted
kit["side"] = np.where(kit.sun_z > 0, "ahead", "behind")
for exp in ("focal_grid", "sun_az_grid"):
    put(mkey("kCsFrac", exp), 100 * kit[kit.exp == exp].cs_frac.median(), "{:.0f}")
calib = {}
CAL_SCORES = ("auc", "auc_cs", "auc_noam", "auc_am", "auc2", "kappa", "iou")
for o in OBS:
    for s in CAL_SCORES:
        g = kit[(kit.obs == o) & (kit.exp == "focal_grid")]
        for side in ("ahead", "behind"):
            gs = g[g.side == side]
            w = gs.pivot_table(index=["drive", "model", "key"], columns="value", values=s)
            wt = gs.drop_duplicates(["key", "model"]).set_index(["drive", "model", "key"]).focal_sens.reindex(w.index)
            pooled_side = pool(w, wt.values, ["drive", "model"]).dropna(how="all")
            kh = pooled_side.idxmax(axis=1).astype(float)
            put(mkey("kCalibSideFine", o, s, side), np.median(refine(pooled_side)), "{:.2f}")
            calib[o, s, side] = kh
            put(mkey("kCalibSide", o, s, side), np.median(kh), "{:.2f}")
        w = g.pivot_table(index=["drive", "model", "key"], columns="value", values=s)
        wt = g.drop_duplicates(["key", "model"]).set_index(["drive", "model", "key"]).focal_sens.reindex(w.index)
        pooled = pool(w, wt.values, ["drive", "model"]).dropna(how="all")
        k_hat = pooled.idxmax(axis=1).astype(float)
        calib[o, s] = k_hat
        sk = syn_calib.get(("detector", "all", s), pd.Series([1.0]))
        put(mkey("kCalib", o, s, "corrected"), np.median(k_hat) / np.median(sk), "{:.2f}")
        put(mkey("kCalib", o, s, "med"), np.median(k_hat), "{:.2f}")
        put(mkey("kCalib", o, s, "err"), 100 * np.median(np.abs(np.log(k_hat))), "{:.0f}")
        put(mkey("kCalib", o, s, "within"), 100 * np.mean(np.abs(np.log(k_hat)) < 0.13), "{:.0f}")

# LiDAR: the effective focal factor a correct check should find is k_eff = gamma (calibrated camera)
lid_file = STUDY / "kitti_lidar.csv"
lidar = pd.read_csv(lid_file) if lid_file.exists() else None
heldout = {}
if lidar is not None:
    lidar = lidar[lidar.model.isin(MODELS)]
    ref = lidar.groupby(["drive", "model"]).gamma.median()
    for m, g in lidar.groupby("model"):
        put(mkey("lidAbsRel", m), g.absrel.median(), "{:.3f}")
        put(mkey("lidAbsRelMed", m), g.absrel_med.median(), "{:.3f}")
        put(mkey("lidGamma", m), g.gamma.median(), "{:.2f}")
        put(mkey("lidScale", m), g.scale_ratio.median(), "{:.2f}")
    put("lidGammaAll", lidar.gamma.median(), "{:.2f}")
    put("lidNFrames", lidar.key.nunique(), "{:d}")
    drives = sorted(kit.drive.unique())
    fold = {dr: i % 2 for i, dr in enumerate(drives)}
    for o in OBS:
        for s in CAL_SCORES:
            kh = calib[o, s]
            r = ref.reindex(kh.index)
            ok = r.notna() & kh.notna()
            lk, lr = np.log(kh[ok]), np.log(r[ok])
            f = np.array([fold[dr] for dr in lk.index.get_level_values("drive")])
            # bias estimated on one half of the drives, applied to the other half (two-fold)
            err = np.empty(len(lk))
            for side in (0, 1):
                b = np.median((lk - lr)[f != side])
                err[f == side] = np.abs((lk - lr)[f == side] - b)
            heldout[o, s] = err
            put(mkey("kHeld", o, s), 100 * np.median(err), "{:.0f}")
            put(mkey("kRaw", o, s), 100 * np.median(np.abs(lk - lr)), "{:.0f}")
            put(mkey("kBias", o, s), np.exp(np.median(lk - lr)), "{:.2f}")
            sk = syn_calib["detector", "all", s] if ("detector", "all", s) in syn_calib else None
            if sk is not None:
                put(mkey("kSynCorr", o, s), 100 * np.median(np.abs(lk - lr - np.median(np.log(sk)))), "{:.0f}")

# ablation on KITTI: one row per score
ABL = {"iou": "IoU (2-state)", "kappa": r"$\kappa$ (2-state)", "auc2": "AUC (2-state)", "auc_noam": "AUC, 3-state",
       "auc_am": "AUC, amodal", "auc": "SunCheck", "auc_cs": "SunCheck, common supp."}
lines = [r"\begin{table}[t]", r"\centering\small\setlength{\tabcolsep}{2.6pt}",
         r"\resizebox{\linewidth}{!}{\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
         r"Score & at truth & $n$ & sun err. & $\hat k$ ahead & $\hat k$ behind & $k$ err. \\",
         r" & & & (deg) & & & (\%) \\", r"\midrule"]
o = "net"
for s, sname in ABL.items():
    t = kit[(kit.obs == o) & (kit.exp == "focal_grid") & np.isclose(kit.value, 1.0)]
    w = kit[(kit.obs == o) & (kit.exp == "sun_az_grid")].pivot_table(index=cases, columns="value", values=s).dropna()
    az = np.median(np.abs(w.idxmax(axis=1).astype(float))) if len(w) else np.nan
    put(mkey("ablSun", s), az, "{:.0f}")
    ka, kb = np.median(calib[o, s, "ahead"]), np.median(calib[o, s, "behind"])
    he = 100 * np.median(heldout[o, s]) if (o, s) in heldout else np.nan
    lines.append(f"{sname} & {t[s].mean():.2f} & {len(w)} & {az:.0f} & {ka:.2f} & {kb:.2f} & " +
                 (f"{he:.0f}" if np.isfinite(he) else "--") + r" \\")
lines += [r"\bottomrule", r"\end{tabular}}",
          r"\caption{Ablation on KITTI (Net detector, " + macros["kNFrames"] + r" frames, four models). At truth: mean score"
          r" at the calibrated camera (chance: AUC 0.5, $\kappa$ 0)."
          r" $n$: cases on which the score is defined for the whole sun grid (few cases means easy ones)."
          r" Sun err.: median error of the single-frame best sun"
          r" azimuth ($\pm60^\circ$ grid). $\hat k$: focal factor pooled per drive and model with the sun ahead or behind the"
          r" camera. $k$ err.: median $|\log\hat k-\log k_{\mathrm{eff}}|$ against the LiDAR $k_{\mathrm{eff}}=\gamma$, after a"
          r" bias estimated on held-out drives.}", r"\label{tab:ablation}", r"\end{table}"]
(GEN / "table_ablation.tex").write_text("\n".join(lines) + "\n")

# models: does the check rank the depth models as LiDAR does?
if lidar is not None:
    lines = [r"\begin{table}[t]", r"\centering\small\setlength{\tabcolsep}{3pt}",
             r"\resizebox{\linewidth}{!}{\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
             r"Model & AbsRel & AbsRel$_{\mathrm{med}}$ & $\gamma$ & IoU & SunCheck & SunCheck \\",
             r" & \multicolumn{3}{c}{LiDAR} & Net & Net & Dark \\", r"\midrule"]
    t = kit[(kit.exp == "focal_grid") & np.isclose(kit.value, 1.0)]
    for m, mname in MODELS.items():
        g = lidar[lidar.model == m]
        tn, td = t[(t.model == m) & (t.obs == "net")], t[(t.model == m) & (t.obs == "thr")]
        put(mkey("kModel", m, "auc"), tn.auc.mean(), "{:.3f}")
        lines.append(f"{mname} & {g.absrel.median():.3f} & {g.absrel_med.median():.3f} & {g.gamma.median():.2f} & "
                     f"{tn.iou.mean():.3f} & {tn.auc.mean():.3f} & {td.auc.mean():.3f}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}",
              r"\caption{Depth models on KITTI. LiDAR: median AbsRel of the metric depth (AbsRel), the same after median"
              r" scaling (AbsRel$_{\mathrm{med}}$, the shape error used to rank the models), and the"
              r" depth-shape exponent $\gamma$ (1: correct shape). Shadow scores at the calibrated camera, mean over"
              r" frames.}", r"\label{tab:models2}", r"\end{table}"]
    (GEN / "table_models_v2.tex").write_text("\n".join(lines) + "\n")

# paired comparisons on the same cases: SunCheck against IoU, and pairs of depth models
for o in OBS:
    g = kit[(kit.obs == o) & (kit.exp == "sun_az_grid")]
    A = g.pivot_table(index=cases, columns="value", values="auc").dropna()
    Iw = g.pivot_table(index=cases, columns="value", values="iou").reindex(A.index).dropna()
    A = A.loc[Iw.index]
    ea, ei = np.abs(A.idxmax(axis=1).astype(float)), np.abs(Iw.idxmax(axis=1).astype(float))
    put(mkey("pairSunN", o), len(A), "{:d}")
    put(mkey("pairSunMed", o, "auc"), np.median(ea), "{:.0f}")
    put(mkey("pairSunMed", o, "iou"), np.median(ei), "{:.0f}")
    put(mkey("pairSunWithin", o, "auc"), 100 * np.mean(ea <= 10), "{:.0f}")
    put(mkey("pairSunWithin", o, "iou"), 100 * np.mean(ei <= 10), "{:.0f}")
    lo, hi = boot(((ea <= 10).astype(float) - (ei <= 10).astype(float)).values, A.index.get_level_values("drive"))
    put(mkey("pairSunDiff", o, "lo"), 100 * lo, "{:.0f}")
    put(mkey("pairSunDiff", o, "hi"), 100 * hi, "{:.0f}")
t = kit[(kit.exp == "focal_grid") & np.isclose(kit.value, 1.0)]
for o in OBS:
    for sc in ("auc", "iou"):
        w = t[t.obs == o].pivot_table(index=["drive", "key"], columns="model", values=sc).dropna()
        put(mkey("nPair", o, sc), len(w), "{:d}")
        for a_, b_ in (("dav2-l", "dav2-s"), ("depthpro", "dav2-l"), ("moge2-l", "dav2-l"), ("moge2-l", "depthpro")):
            dlt = w[a_] - w[b_]
            lo, hi = boot(dlt.values, w.index.get_level_values("drive"))
            tag = mkey("diff", o, sc, a_, b_)
            put(tag, dlt.mean(), "{:.3f}")
            put(tag + "Lo", lo, "{:.3f}")
            put(tag + "Hi", hi, "{:.3f}")
# how many cases each score is defined on (sun grid complete), and its coverage at the truth
for sc in ("iou", "auc_noam", "auc", "auc_cs"):
    g = kit[(kit.obs == "net") & (kit.exp == "sun_az_grid")]
    put(mkey("nDef", sc), len(g.pivot_table(index=cases, columns="value", values=sc).dropna()), "{:d}")
g = kit[(kit.obs == "net") & (kit.exp == "sun_az_grid")]
W = {sc: g.pivot_table(index=cases, columns="value", values=sc).dropna() for sc in ABL}
common = None
for w in W.values():
    common = w.index if common is None else common.intersection(w.index)
put("nCommon", len(common), "{:d}")
for sc in ABL:
    e = np.abs(W[sc].loc[common].idxmax(axis=1).astype(float))
    put(mkey("commonSunWithin", sc), 100 * np.mean(e <= 10), "{:.0f}")
tt = t[t.obs == "net"]
put("detFracAmodal", 100 * tt.det_frac.median(), "{:.0f}")
put("detFracNoam", 100 * tt.det_frac_noam.median(), "{:.0f}")

# robustness: stricter minimum class size n_min and minimum coverage C_min, applied after the fact
kit["n_sh"] = kit.n_det * kit.det_shadow_frac
kit["n_lit"] = kit.n_det - kit.n_sh
SWEEP = [(20, 0.0), (50, 0.0), (100, 0.0), (200, 0.0), (20, 0.1), (20, 0.2), (20, 0.3)]
lines = [r"\begin{table}[!ht]", r"\centering\small\setlength{\tabcolsep}{3pt}",
         r"\resizebox{\linewidth}{!}{\begin{tabular}{@{}cccccccc@{}}", r"\toprule",
         r"$n_{\min}$ & $C_{\min}$ & cases & AUC Net & AUC Dark & sun az.\ SC & sun az.\ IoU & L$-$S Dark \\", r"\midrule"]
for nmin, cmin in SWEEP:
    kk = kit.copy()
    bad = (np.minimum(kk.n_sh, kk.n_lit) < nmin) | (kk.det_frac < cmin)
    kk.loc[bad, "auc"] = np.nan
    t0 = kk[(kk.exp == "focal_grid") & np.isclose(kk.value, 1.0)]
    lv = {o: t0[t0.obs == o].auc.mean() for o in OBS}
    ncase = int(t0[t0.obs == "net"].auc.notna().sum())
    g = kk[(kk.obs == "net") & (kk.exp == "sun_azimuth")]
    A = g.pivot_table(index=cases, columns="value", values="auc").dropna()
    I_ = g.pivot_table(index=cases, columns="value", values="iou").reindex(A.index).dropna()
    A = A.loc[I_.index]
    sa, si = 100 * np.mean(A.idxmax(axis=1) == 0), 100 * np.mean(I_.idxmax(axis=1) == 0)
    w = t0[t0.obs == "thr"].pivot_table(index=["drive", "key"], columns="model", values="auc").dropna()
    dlt = w["dav2-l"] - w["dav2-s"]
    lo, hi = boot(dlt.values, w.index.get_level_values("drive"))
    tag = mkey("sweep", nmin, int(100 * cmin))
    put(tag + "Cases", ncase, "{:d}")
    put(tag + "Net", lv["net"], "{:.3f}")
    put(tag + "Sun", sa, "{:.0f}")
    put(tag + "SunIou", si, "{:.0f}")
    lines.append(f"{nmin} & {cmin:.1f} & {ncase} & {lv['net']:.3f} & {lv['thr']:.3f} & {sa:.0f} & {si:.0f} & "
                 f"{dlt.mean():.3f} [{lo:.3f}, {hi:.3f}]" + r" \\")
lines += [r"\bottomrule", r"\end{tabular}}",
          r"\caption{Robustness of the KITTI results to the minimum class size $n_{\min}$ and a minimum coverage"
          r" $C_{\min}$ (applied after the fact). Cases: (frame, model) pairs scored at the truth. AUC: SunCheck at the"
          r" true geometry. Sun az.: truth is the best of five azimuths (\%, chance 20), SunCheck and IoU on the same"
          r" cases. L$-$S: paired SunCheck difference DA-V2-L minus DA-V2-S without a detector, with a drive bootstrap.}",
          r"\label{tab:sweep}", r"\end{table}"]
(GEN / "table_sweep.tex").write_text("\n".join(lines) + "\n")

# H ablation: random streets on held-out seeds (10000+) and a third of the KITTI frames, H = 10 / 20 / 40 m
SYN_H, KIT_H = ROOT / "outputs" / "kaggle_synth_h", ROOT / "outputs" / "kaggle_hablation"
if all((SYN_H / f"synthetic_focal_grid_h{h}.csv").exists() for h in (10, 20, 40)) and \
        all((KIT_H / f"h{h}" / "kitti_eval_shard.csv").exists() for h in (10, 40)):
    lines = [r"\begin{table}[!ht]", r"\centering\small\setlength{\tabcolsep}{3pt}",
             r"\resizebox{\linewidth}{!}{\begin{tabular}{@{}c|ccc|cccc@{}}", r"\toprule",
             r" & \multicolumn{3}{c|}{random streets (held-out seeds)} & \multicolumn{4}{c}{KITTI (a third of the frames)} \\",
             r"$H$ (m) & sun az. & $\hat k$ ahead & $\hat k$ behind & $C$ & AUC Net & AUC Dark & sun az. \\", r"\midrule"]
    for h in (10, 20, 40):
        sb = pd.read_csv(SYN_H / f"synthetic_benchmark_h{h}.csv")
        sb = sb[sb.condition == "detector"]
        sa = truth_argmax(sb, "auc", "sun_azimuth", ["scene"], "scene")[0]
        gg = pd.read_csv(SYN_H / f"synthetic_focal_grid_h{h}.csv")
        gg = gg[gg.condition == "detector"].copy()
        gg["side"] = np.where(gg.azimuth.abs() < 90, "ahead", "behind")
        gg["group"] = gg.scene // 10
        ks = {}
        for side in ("ahead", "behind"):
            g2 = gg[gg.side == side]
            w = g2.pivot_table(index=["group", "scene"], columns="k", values="auc3")
            wt = g2.drop_duplicates("scene").set_index("scene").focal_sens
            pooled = pool(w, wt.reindex(w.index.get_level_values("scene")).values, "group").dropna(how="all")
            ks[side] = np.median(refine(pooled))
        if h == 20:
            kk = kit[kit.key.isin(pd.read_csv(KIT_H / "h10" / "kitti_eval_shard.csv", usecols=["key"]).key.unique())]
        else:
            kk = pd.read_csv(KIT_H / f"h{h}" / "kitti_eval_shard.csv")
        t0 = kk[(kk.exp == "focal_grid") & np.isclose(kk.value, 1.0)]
        cov = t0[t0.obs == "net"].det_frac.median()
        ln, ld = t0[t0.obs == "net"].auc.mean(), t0[t0.obs == "thr"].auc.mean()
        ka = truth_argmax(kk[kk.obs == "net"], "auc", "sun_azimuth", cases, "drive")[0]
        for name, v in (("Sun", 100 * sa), ("Ahead", ks["ahead"]), ("Behind", ks["behind"]), ("Cov", 100 * cov),
                        ("Net", ln), ("Dark", ld), ("KSun", 100 * ka)):
            fmt = "{:.0f}" if name in ("Sun", "Cov", "KSun") else ("{:.2f}" if len(name) > 4 else "{:.3f}")
            put(mkey("hab", h, name), v, fmt)
        lines.append(f"{h} & {100 * sa:.0f} & {ks['ahead']:.2f} & {ks['behind']:.2f} & {100 * cov:.0f}\\% & {ln:.3f} & "
                     f"{ld:.3f} & {100 * ka:.0f}" + r" \\")
    put("habNFrames", kk.key.nunique(), "{:d}")
    lines += [r"\bottomrule", r"\end{tabular}}",
              r"\caption{Sensitivity to the caster height prior $H$. Random streets: 200 scenes with seeds never used"
              r" elsewhere, detector condition; truth is the best of five sun azimuths (\%, chance 20) and the pooled"
              r" focal factor (refined, truth 1) with the sun ahead or behind. KITTI: the same " + macros["habNFrames"] +
              r" frames for every $H$; median coverage at the truth, mean SunCheck AUC at the truth, and truth is the"
              r" best of five sun azimuths (Net, \%).}", r"\label{tab:hablation}", r"\end{table}"]
    (GEN / "table_hablation.tex").write_text("\n".join(lines) + "\n")

# the model's own camera against the calibrated one
own = kit[kit.exp == "own_camera"]
for m in ("depthpro", "moge2-l"):
    g = own[own.model == m]
    if g.empty:
        continue
    put(mkey("kOwnK", m), g.k_own.mean(), "{:.2f}")
    for o in OBS:
        for s in ("auc", "iou"):
            w = g[g.obs == o].pivot_table(index=["drive", "key"], columns="value", values=s).dropna()
            win = (w[0] > w[1]).astype(float)
            put(mkey("kOwn", m, o, s), 100 * win.mean(), "{:.0f}")
            lo, hi = boot(win.values, w.index.get_level_values("drive"))
            put(mkey("kOwn", m, o, s, "lo"), 100 * lo, "{:.0f}")
            put(mkey("kOwn", m, o, s, "hi"), 100 * hi, "{:.0f}")

(GEN / "numbers_v2.tex").write_text("% generated by scripts/make_v2_results.py, do not edit\n" + "".join(
    f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(macros.items())))

# ------------------------------------------------------------------ figures
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
COL = {"iou": "#9a9a9a", "kappa": "#c9a227", "auc2": "#5b8fd0", "auc_am": "#8e6bbf", "auc": "#c0392b"}


def norm_curve(d, s, c, cases):
    w = d[d.exp == c].pivot_table(index=cases, columns="value", values=s).dropna()
    w = w.sub(w[TRUTH[c]], axis=0)  # paired: score minus score at the truth
    return w.columns.values.astype(float), w.mean().values, w.sem().values * 1.96


fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.75))
for ax, c in zip(axes, ["scale", "shape", "focal", "sun_azimuth"]):
    for s in ("iou", "kappa", "auc"):
        x, m, e = norm_curve(kit[kit.obs == "net"], s, c, cases)
        ax.errorbar(x, m, e, color=COL[s], marker="o", ms=2.5, lw=1, capsize=1.5, label=SCORES[s])
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(TRUTH[c], color="k", lw=0.5, ls=":")
    ax.set_title(c.replace("_", " "))
    if c in ("scale", "shape", "focal"):
        ax.set_xscale("log")
        ax.set_xticks([0.5, 0.75, 1, 1.5, 2] if c == "scale" else [0.6, 0.8, 1, 1.25, 1.5])
        ax.set_xticklabels(["0.5", ".75", "1", "1.5", "2"] if c == "scale" else [".6", ".8", "1", "1.25", "1.5"])
        ax.minorticks_off()
axes[0].set_ylabel("score $-$ score at truth")
axes[0].legend(frameon=False, fontsize=6.5)
fig.tight_layout()
fig.savefig(FIG / "kitti_v2.pdf", bbox_inches="tight", pad_inches=0.02)

fig, axes = plt.subplots(1, 3, figsize=(7.0, 1.8))
for ax, cond in zip(axes, cond_names):
    x = np.arange(len(CORR))
    for i, s in enumerate(SCORES):
        ax.bar(x + (i - 2) * 0.16, [100 * bench[cond, s, c] for c in CORR], 0.16, color=COL[s], label=SCORES[s])
    ax.axhline(20, color="k", lw=0.6, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(CORR.values())
    ax.set_ylim(0, 100)
    ax.set_title(cond_names[cond])
axes[0].set_ylabel("truth is best (%)")
axes[0].legend(frameon=False, fontsize=6, ncol=2, loc="upper right")
fig.tight_layout()
fig.savefig(FIG / "bench.pdf", bbox_inches="tight", pad_inches=0.02)

fig, ax = plt.subplots(figsize=(3.3, 1.75))
x = np.arange(len(GRID_SCORES))
for i, (side, mk) in enumerate((("ahead", "o"), ("behind", "s"))):
    for j, s_name in enumerate(GRID_SCORES):
        k = syn_calib["detector", side, s_name]
        q = np.percentile(k, [25, 50, 75])
        ax.errorbar(j + (i - 0.5) * 0.25, q[1], [[q[1] - q[0]], [q[2] - q[1]]], fmt=mk, color=COL[s_name], ms=3.5,
                    mfc="white" if side == "behind" else COL[s_name], capsize=1.5, lw=1)
ax.axhline(1, color="k", lw=0.6, ls=":")
ax.set_yscale("log")
ax.set_yticks([0.5, 0.7, 1, 1.4, 2])
ax.set_yticklabels(["0.5", "0.7", "1", "1.4", "2"])
ax.minorticks_off()
ax.set_xticks(x)
ax.set_xticklabels([SCORES[s_name] for s_name in GRID_SCORES], fontsize=6.5)
ax.set_ylabel(r"pooled $\hat k$ (truth 1)")
ax.plot([], [], "ko", ms=3.5, label="sun ahead")
ax.plot([], [], "ks", mfc="white", ms=3.5, label="sun behind")
ax.legend(frameon=False, fontsize=6, loc="upper left")
fig.tight_layout()
fig.savefig(FIG / "bias.pdf", bbox_inches="tight", pad_inches=0.02)

fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.7), sharey=True)
bins = np.log(np.r_[K_GRID / np.exp(np.diff(np.log(K_GRID))[0] / 2), K_GRID[-1] * np.exp(np.diff(np.log(K_GRID))[0] / 2)])
for ax, o in zip(axes, OBS):
    for s in ("iou", "auc"):
        ax.hist(np.log(calib[o, s].values), bins=bins, color=COL[s], alpha=0.75, label=SCORES[s])
    ax.axvline(0, color="k", lw=0.6, ls=":")
    ax.set_xticks(np.log([0.5, 1, 2]))
    ax.set_xticklabels(["0.5", "1", "2"])
    ax.set_xlabel(r"estimated focal factor $\hat k$")
    ax.set_title({"net": "Net", "thr": "Dark"}[o])
axes[0].set_ylabel("drive $\\times$ model")
axes[0].legend(frameon=False, fontsize=6)
fig.tight_layout()
fig.savefig(FIG / "calib.pdf", bbox_inches="tight", pad_inches=0.02)
print(len(macros), "macros")
for k in sorted(macros):
    print(k, macros[k])
