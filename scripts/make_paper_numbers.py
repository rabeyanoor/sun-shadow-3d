"""Every number in the paper, as LaTeX macros, from the result files in outputs/study.

    python scripts/make_paper_numbers.py        # writes paper/gen/numbers.tex and paper/gen/table_*.tex

The synthetic street is cheap, so it is recomputed here rather than read from a notebook.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import scene  # noqa: E402
import shadows  # noqa: E402
import synthetic  # noqa: E402

STUDY = ROOT / "outputs" / "study"
GEN = ROOT / "paper" / "gen"
GEN.mkdir(parents=True, exist_ok=True)
TRUE = {"scale": 1.0, "shape": 1.0, "focal": 1.0, "sun_azimuth": 0.0}
LOW = {"scale": 0.5, "shape": 0.6, "focal": 0.6, "sun_azimuth": -60.0}
MODELS = {"dav2-s": "DA-V2-S", "dav2-l": "DA-V2-L", "depthpro": "Depth Pro", "moge2-l": "MoGe-2-L"}
DET = {"threshold": "Thr", "segformer-istd": "Net"}
macros = {}


def key(*parts):
    words = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five", "6": "Six", "7": "Seven",
             "8": "Eight", "9": "Nine"}
    s = "".join(str(p).replace("-", " ").replace("_", " ").title().replace(" ", "") for p in parts)
    return "".join(words.get(c, c) for c in s if c.isalnum())


def put(name, value, fmt="{:.3f}"):
    text = fmt.format(value) if not isinstance(value, str) else value
    # a typeset minus, not a hyphen
    macros[name] = text.replace("-", "$-$", 1) if text.startswith("-") else text


def num(x, fmt):
    return fmt.format(x).replace("-", "$-$")


def pct(x):
    return f"{100 * x:.0f}"


def boot_drive(v, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    per = v.groupby(level="drive").agg(["sum", "count"])
    pick = rng.integers(0, len(per), size=(n, len(per)))
    means = per["sum"].values[pick].sum(1) / per["count"].values[pick].sum(1)
    return np.percentile(means, [2.5, 97.5])


# ------------------------------------------------------------------ KITTI study
d = pd.read_csv(STUDY / "shadow_study.csv")
net = json.loads((STUDY / "shadow_net.json").read_text())
has_kappa = "kappa" in d.columns
put("nDrives", d.drive.nunique(), "{:d}")
put("nFrames", int(d.groupby("drive").frame.nunique().sum()), "{:d}")
put("nRows", len(d), "{:,}")
put("istdBer", net["istd_test_ber"], "{:.1f}")
put("elevMin", d.elevation.min(), "{:.1f}")
put("elevMax", d.elevation.max(), "{:.1f}")
for det, tag in DET.items():
    put(f"obsFrac{tag}", d[d.detector == det].obs_frac.mean() * 100, "{:.0f}")

scores = ["iou", "kappa"] if has_kappa else ["iou"]
rows = []
for (det, c, score), _ in [((det, c, s), None) for det in DET for c in TRUE for s in scores]:
    g = d[(d.detector == det) & (d.corruption == c)]
    w = g.pivot_table(index=["drive", "frame", "model"], columns="value", values=score)
    at_truth = (w.idxmax(axis=1) == TRUE[c]).astype(float)
    gain = w[LOW[c]] - w[TRUE[c]]
    lo, hi = boot_drive(at_truth)
    glo, ghi = boot_drive(gain)
    rows.append(dict(detector=det, corruption=c, score=score, at=at_truth.mean(), lo=lo, hi=hi,
                     gain=gain.mean(), glo=glo, ghi=ghi, n=len(at_truth)))
    t = key(DET[det], c, score)
    put(f"at{t}", pct(at_truth.mean()))
    put(f"at{t}Lo", pct(lo))
    put(f"at{t}Hi", pct(hi))
    put(f"gain{t}", gain.mean())
    put(f"gain{t}Lo", glo)
    put(f"gain{t}Hi", ghi)
    for v, m in g.groupby("value")[score].mean().items():
        put(f"mean{t}{key(str(v).replace('.', 'p').replace('-', 'm'))}", m)
truth = pd.DataFrame(rows)
put("nCases", int(truth.n.iloc[0]), "{:d}")

rank = d[(d.corruption == "scale") & (d.value == 1.0)].groupby(["detector", "model"])
for (det, m), g in rank:
    put(f"rank{key(DET[det], m)}Iou", g.iou.mean())
    if has_kappa:
        put(f"rank{key(DET[det], m)}Kappa", g.kappa.mean())
rank_tab = rank.iou.mean().unstack()[list(MODELS)].rename(columns=MODELS)

if has_kappa:
    # chance level of the IoU at the true depth, over the whole study
    t = d[(d.corruption == "scale") & (d.value == 1.0)].copy()
    r, o = t.rendered_frac, t.obs_frac
    t["chance"] = r * o / (r + o - r * o)
    for det, tag in DET.items():
        g = t[t.detector == det].set_index("drive")
        ex = g.iou - g.chance
        lo, hi = boot_drive(ex)
        put(f"study{tag}Iou", g.iou.mean())
        put(f"study{tag}Chance", g.chance.mean())
        put(f"study{tag}Excess", ex.mean())
        put(f"study{tag}ExcessLo", lo)
        put(f"study{tag}ExcessHi", hi)
        put(f"study{tag}Kappa", g.kappa.mean())
        put(f"study{tag}RenderFrac", g.rendered_frac.mean() * 100, "{:.1f}")
        a = d[(d.detector == det) & (d.corruption.isin(["shape", "focal"]))]
        a = a.pivot_table(index=["drive", "frame", "model", "corruption"], columns="value", values="rendered_frac")
        put(f"study{tag}AreaRatio", (a[0.6] / a[1.0].clip(lower=1e-6)).median(), "{:.2f}")
        dd = d[d.detector == det]
        within = dd.assign(**{c: dd[c] - dd.groupby(["drive", "frame", "model"])[c].transform("mean")
                              for c in ["iou", "kappa", "rendered_frac"]})
        put(f"study{tag}CorrIou", np.corrcoef(within.rendered_frac, within.iou)[0, 1], "{:.2f}")
        put(f"study{tag}CorrKappa", np.corrcoef(within.rendered_frac, within.kappa)[0, 1], "{:.2f}")

# ------------------------------------------------------------------ drive-0001 coverage study (notebook 06)
ch = pd.read_csv(STUDY / "coverage_bias_chance.csv", index_col=0)["value"]
for k_, v in ch.items():
    put(f"cov{key(k_)}", v)
cs = pd.read_csv(STUDY / "coverage_bias_summary.csv")
ck = pd.read_csv(STUDY / "coverage_bias_kitti.csv")
put("covFrames", ck.frame.nunique(), "{:d}")
cw = ck.assign(**{c: ck[c] - ck.groupby("frame")[c].transform("mean") for c in ["iou", "kappa", "rendered_frac"]})
put("covCorrIou", np.corrcoef(cw.rendered_frac, cw.iou)[0, 1], "{:.2f}")
put("covCorrKappa", np.corrcoef(cw.rendered_frac, cw.kappa)[0, 1], "{:.2f}")
put("covObsFrac", ck.obs_frac.mean() * 100, "{:.0f}")
put("covAreaRatio", cs.area_06_over_true.mean(), "{:.1f}")
put("covRenderTrue", ck[ck.value == 1.0].rendered_frac.mean() * 100, "{:.1f}")
put("covRenderLow", ck[ck.value == 0.6].rendered_frac.mean() * 100, "{:.1f}")
eq = pd.read_csv(STUDY / "equivalent_sun_error.csv", index_col=0)
for v, a in eq["focal"].items():
    put(f"eqSun{key(str(v).replace('.', 'p'))}", a, "{:.1f}")
for v in [0.6, 0.8, 1.25, 1.5]:
    put(f"eqBound{key(str(v).replace('.', 'p'))}", np.degrees(np.arctan(abs(1 - v) / (2 * np.sqrt(v)))), "{:.1f}")

# ------------------------------------------------------------------ synthetic street (recomputed)
H, W = 188, 620
K = np.array([[180.0, 0, 309.5], [0, 180.0, 93.5], [0, 0, 1]])
depth, ground, boxes = synthetic.make_scene(K, H, W)
SUNS = [(-40, 30), (-20, 45), (25, 35), (50, 25)]
RENDER = dict(max_dist=40.0, n_steps=96, eps=0.15, thickness=3.0)
truths = [synthetic.exact_shadow(K, H, W, depth, boxes, synthetic.sun_vector(a, e)) for a, e in SUNS]
rendered = [shadows.render_shadow(depth, K, synthetic.sun_vector(a, e), **RENDER) & ground for a, e in SUNS]
put("synRenderMiss", 100 * (1 - np.mean([r.sum() / (t & ground).sum() for r, t in zip(rendered, truths)])), "{:.0f}")
put("synRenderIou", np.mean([scene.iou(shadows.render_shadow(depth, K, synthetic.sun_vector(a, e), **RENDER), t, ground)
                             for (a, e), t in zip(SUNS, truths)]))


def syn_score(dc, Kc, tol=1.0, fn=scene.iou):
    kw = dict(RENDER, max_dist=40.0 * tol, eps=0.15 * tol, thickness=3.0 * tol)
    return float(np.mean([fn(shadows.render_shadow(dc, Kc, synthetic.sun_vector(a, e), **kw), t, ground)
                          for (a, e), t in zip(SUNS, truths)]))


scale_vals = [syn_score(depth * a, K, tol=a) for a in [0.5, 0.75, 1.0, 1.5, 2.0]]
put("synScaleMin", min(scale_vals))
put("synScaleMax", max(scale_vals))
put("synScaleRaw", syn_score(depth * 2.0, K))
sy = pd.read_csv(STUDY / "coverage_bias_synthetic.csv").groupby(["corruption", "value"])[["iou", "kappa"]].mean()
for (c, v), r in sy.iterrows():
    put(f"syn{key(c)}{key(str(v).replace('.', 'p'))}Iou", r.iou)
    put(f"syn{key(c)}{key(str(v).replace('.', 'p'))}Kappa", r.kappa)

# ------------------------------------------------------------------ tables
VALS = {"scale": [0.5, 0.75, 1.0, 1.5, 2.0], "shape": [0.6, 0.8, 1.0, 1.25, 1.5], "focal": [0.6, 0.8, 1.0, 1.25, 1.5],
        "sun_azimuth": [-60.0, -30.0, 0.0, 30.0, 60.0]}
NAME = {"scale": "scale $\\lambda$", "shape": "shape $\\gamma$", "focal": "focal $k$", "sun_azimuth": "sun az.\\ (deg)"}
lines = [r"\begin{table*}[t]", r"\centering\small", r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{llccccc@{\hskip 10pt}cc}", r"\toprule",
         r"& & \multicolumn{5}{c}{mean road IoU at corruption value} & truth is argmax & IoU(wrong)$-$IoU(truth) \\",
         r"\cmidrule(lr){3-7}\cmidrule(lr){8-8}\cmidrule(lr){9-9}"]
for det, label in [("threshold", "threshold"), ("segformer-istd", "SegFormer-ISTD")]:
    lines.append(rf"\multicolumn{{9}}{{l}}{{\emph{{shadow detector: {label}}}}} \\")
    for c in TRUE:
        g = d[(d.detector == det) & (d.corruption == c)].groupby("value").iou.mean()
        vals = VALS[c]
        best = g.idxmax()
        cells = []
        for v in vals:
            s = f"{g[v]:.3f}"
            if v == best:
                s = rf"\textbf{{{s}}}"
            if v == TRUE[c]:
                s = rf"\underline{{{s}}}"
            cells.append(s)
        r = truth[(truth.detector == det) & (truth.corruption == c) & (truth.score == "iou")].iloc[0]
        head = " / ".join(f"{v:g}" for v in vals)
        lines.append(rf"& {NAME[c]} \tiny{{({head})}} & " + " & ".join(cells)
                     + rf" & {100 * r['at']:.0f}\% [{100 * r.lo:.0f}, {100 * r.hi:.0f}] & "
                     + rf"{num(r.gain, '{:+.3f}')} [{num(r.glo, '{:+.3f}')}, {num(r.ghi, '{:+.3f}')}] \\")
lines += [r"\bottomrule", r"\end{tabular}",
          r"\caption{The shadow score on KITTI (\nFrames{} frames from \nDrives{} drives, four depth models). Columns are the "
          r"five corruption values in the order given in brackets; the true value is underlined, the best mean in bold. "
          r"`Truth is argmax' is the share of (frame, model) cases whose best value is the true one (chance 20\%), and the "
          r"last column the paired IoU gain of the most compressed value (scale 0.5, $\gamma$ and $k$ 0.6, azimuth "
          r"$-60^\circ$) over the truth; intervals are 95\% bootstraps over drives.}",
          r"\label{tab:kitti}", r"\end{table*}"]
(GEN / "table_kitti.tex").write_text("\n".join(lines) + "\n")

lines = [r"\begin{table}[t]", r"\centering\small", r"\begin{tabular}{lcccc}", r"\toprule",
         "detector & " + " & ".join(MODELS.values()) + r" \\", r"\midrule"]
for det, label in [("threshold", "threshold"), ("segformer-istd", "SegFormer-ISTD")]:
    row = rank_tab.loc[det]
    lines.append(label + " & " + " & ".join(f"{x:.3f}" for x in row.values) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Mean road shadow IoU per depth model at the true camera and sun. A useful check should rank "
          r"DA-V2-L above DA-V2-S; it does the opposite.}", r"\label{tab:models}", r"\end{table}"]
(GEN / "table_models.tex").write_text("\n".join(lines) + "\n")

(GEN / "numbers.tex").write_text("% generated by scripts/make_paper_numbers.py, do not edit\n"
                                 + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(macros.items())))
print(len(macros), "macros", "(with kappa)" if has_kappa else "(no kappa columns yet)")
