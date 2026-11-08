"""Paper figures as vector PDFs at column width, from outputs/study and the local KITTI drive.

    python scripts/make_paper_figures.py            # all figures into paper/fig
    python scripts/make_paper_figures.py --no-teaser   # skip the one that runs a depth model
"""
import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import scene  # noqa: E402
import shadows  # noqa: E402
import synthetic  # noqa: E402

STUDY = ROOT / "outputs" / "study"
FIG = ROOT / "paper" / "fig"
FIG.mkdir(parents=True, exist_ok=True)
COL, FULL = 3.3, 6.9
plt.rcParams.update({"font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "legend.fontsize": 6,
                     "xtick.labelsize": 6, "ytick.labelsize": 6, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42, "lines.linewidth": 1.2, "lines.markersize": 3})
MODELS = ["dav2-s", "dav2-l", "depthpro", "moge2-l"]
LABEL = {"dav2-s": "DA-V2-S", "dav2-l": "DA-V2-L", "depthpro": "Depth Pro", "moge2-l": "MoGe-2-L"}
COLOUR = {"dav2-s": "#E69F00", "dav2-l": "#D55E00", "depthpro": "#0072B2", "moge2-l": "#009E73"}
IOU_C, KAPPA_C = "#D55E00", "#0072B2"
VALUES = [0.6, 0.8, 1.0, 1.25, 1.5]


def log_axis(ax, ticks):
    ax.set_xscale("log")
    ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
    ax.minorticks_off()


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", name)


# ------------------------------------------------------------------ 1. teaser on a KITTI frame
def teaser():
    import torch
    import torch.nn.functional as F
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    import kitti
    import sun

    date_dir = ROOT / "data/raw/2011_09_26"
    drive = date_dir / "2011_09_26_drive_0001_sync"
    oxts, stamps, calib = kitti.read_oxts(drive), kitti.read_timestamps(drive / "oxts/timestamps.txt"), kitti.read_calib(date_dir)
    repo = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"
    proc, model = AutoImageProcessor.from_pretrained(repo), AutoModelForDepthEstimation.from_pretrained(repo).eval()
    i, sc = 36, 0.25
    img = kitti.load_image(drive, i)
    h, w = img.shape[:2]
    hs, ws = int(h * sc), int(w * sc)

    def down(x, mode="nearest"):
        t = torch.from_numpy(np.ascontiguousarray(x)).float()[None, None]
        return F.interpolate(t, size=(hs, ws), mode=mode)[0, 0].numpy()

    with torch.no_grad():
        pred = model(**proc(images=img, return_tensors="pt")).predicted_depth[:, None]
    pred = F.interpolate(pred, size=(h, w), mode="bilinear", align_corners=False)[0, 0].numpy()
    labels = scene.segment(img)
    pred[labels == scene.SKY] = 0
    d = down(pred)
    receiver = down(scene.receiver_mask(labels)) > 0.5
    region = receiver & (d > 0)
    obs = scene.observed_shadow(np.repeat(down(img.mean(-1), "area")[..., None], 3, -1), receiver)
    s, az, el = sun.sun_in_camera(oxts, i, stamps[i], calib, 2)
    K = shadows.resize_K(calib["K2"], ws / w, hs / h)
    Kf = K.copy()
    Kf[0, 0] *= 0.6
    Kf[1, 1] *= 0.6
    small = img[::4, ::4][:hs, :ws]
    panels = [("KITTI 0001, frame 36\n(sun from GPS + time)", None), ("observed shadow\n(threshold detector)", obs),
              ("rendered, true focal", shadows.render_shadow(d, K, s)),
              ("rendered, focal $\\times$0.6", shadows.render_shadow(d, Kf, s))]
    fig, axes = plt.subplots(2, 2, figsize=(COL, COL * hs / ws + 0.45), gridspec_kw=dict(wspace=0.03, hspace=0.45))
    for ax, (title, m) in zip(axes.ravel(), panels):
        ax.imshow(small)
        if m is not None:
            over = np.zeros((*m.shape, 4))
            over[region] = (1, 1, 1, 0.25)
            over[m & region] = (0.0, 0.25, 1.0, 0.75)
            ax.imshow(over)
            if title.startswith("rendered"):
                title += f"\nIoU {scene.iou(m, obs, region):.3f}, $\\kappa$ {scene.kappa(m, obs, region):+.3f}"
        ax.set_title(title, fontsize=6, pad=2)
        ax.axis("off")
    save(fig, "teaser")


# ------------------------------------------------------------------ 2. synthetic street
def synthetic_fig():
    H, W = 188, 620
    K = np.array([[180.0, 0, 309.5], [0, 180.0, 93.5], [0, 0, 1]])
    depth, ground, boxes = synthetic.make_scene(K, H, W)
    s = synthetic.sun_vector(-40, 30)
    truth = synthetic.exact_shadow(K, H, W, depth, boxes, s)
    syn = pd.read_csv(STUDY / "coverage_bias_synthetic.csv")
    fig = plt.figure(figsize=(COL, 2.35))
    ax0 = fig.add_axes([0.0, 0.55, 1.0, 0.45])
    ax0.imshow(np.where(depth > 0, depth, np.nan), cmap="turbo_r")
    ax0.imshow(np.ma.masked_where(~truth, truth), cmap="gray_r", alpha=0.6, vmin=0, vmax=1)
    ax0.set_title("synthetic street: depth and exact shadows (sun az $-40^\\circ$, el $30^\\circ$)")
    ax0.axis("off")
    ax1 = fig.add_axes([0.11, 0.08, 0.38, 0.36])
    ax2 = fig.add_axes([0.61, 0.08, 0.38, 0.36], sharey=ax1)
    for ax, c, xl in [(ax1, "shape", "shape warp $\\gamma$"), (ax2, "focal", "focal factor $k$")]:
        m = syn[syn.corruption == c].groupby("value")[["iou", "kappa"]].mean()
        ax.plot(m.index, m.iou, "o-", color=IOU_C, label="IoU")
        ax.plot(m.index, m.kappa, "s-", color=KAPPA_C, label="$\\kappa$")
        ax.axvline(1.0, color="k", lw=0.6, ls=":")
        log_axis(ax, VALUES)
        ax.set_xlabel(xl)
    ax1.set_ylabel("score vs exact shadow")
    ax1.legend(frameon=False)
    save(fig, "synthetic")


# ------------------------------------------------------------------ 3. the KITTI study
def kitti_fig():
    d = pd.read_csv(STUDY / "shadow_study.csv")
    TRUE = {"scale": 1.0, "shape": 1.0, "focal": 1.0, "sun_azimuth": 0.0}
    fig, axes = plt.subplots(2, 4, figsize=(FULL, 2.7), sharey="row")
    for i, (det, name) in enumerate([("threshold", "threshold"), ("segformer-istd", "SegFormer-ISTD")]):
        for j, (c, xl) in enumerate([("scale", "global scale $\\lambda$"), ("shape", "shape warp $\\gamma$"),
                                     ("focal", "focal factor $k$"), ("sun_azimuth", "sun azimuth error (deg)")]):
            ax = axes[i, j]
            for m in MODELS:
                g = d[(d.detector == det) & (d.corruption == c) & (d.model == m)].groupby("value").iou.mean()
                ax.plot(g.index, g.values, "o-", color=COLOUR[m], label=LABEL[m], lw=1.0)
            ax.axvline(TRUE[c], color="k", lw=0.6, ls=":")
            if c in ("scale", "shape", "focal"):
                log_axis(ax, sorted(d[d.corruption == c].value.unique()))
            if i == 1:
                ax.set_xlabel(xl)
        axes[i, 0].set_ylabel(f"road IoU\n({name})")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), frameon=False, ncol=4, loc="upper center",
               bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(pad=0.3)
    save(fig, "kitti_study")


# ------------------------------------------------------------------ 4. coverage bias and the sun-error bound
def coverage_fig():
    k = pd.read_csv(STUDY / "coverage_bias_kitti.csv")
    eq = pd.read_csv(STUDY / "equivalent_sun_error.csv", index_col=0)
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 1.75))
    for c, ls in [("shape", "-"), ("focal", "--")]:
        m = k[k.corruption == c].groupby("value").mean(numeric_only=True)
        axes[0].plot(m.index, 100 * m.rendered_frac, "o" + ls, color="#555555", label=f"{c}")
        axes[1].plot(m.index, m.iou, "o" + ls, color=IOU_C, label=f"IoU, {c}")
        axes[1].plot(m.index, m.kappa, "s" + ls, color=KAPPA_C, label=f"$\\kappa$, {c}")
    t = k[k.value == 1.0].drop_duplicates("frame")
    r, o = t.rendered_frac, t.obs_frac
    axes[1].axhline((r * o / (r + o - r * o)).mean(), color=IOU_C, lw=0.6, ls=":", label="chance IoU")
    axes[1].axhline(0, color=KAPPA_C, lw=0.6, ls=":")
    axes[0].set(ylabel="rendered shadow (% of road)", title="compressed depth casts more shadow")
    axes[1].set(ylabel="score", title="IoU follows area; $\\kappa\\approx0$")
    for ax in axes[:2]:
        ax.axvline(1.0, color="k", lw=0.6, ls=":")
        log_axis(ax, VALUES)
        ax.set_xlabel("$\\gamma$ or $k$")
    axes[0].legend(frameon=False, fontsize=5)
    axes[1].legend(frameon=False, fontsize=5, ncol=2, loc="center", bbox_to_anchor=(0.5, 0.52))
    ks = np.exp(np.linspace(np.log(0.5), np.log(2.0), 200))
    axes[2].plot(ks, np.degrees(np.arctan(np.abs(1 - ks) / (2 * np.sqrt(ks)))), "k-", lw=1.0, label="worst case (Cor. 2)")
    axes[2].plot(eq.index, eq.focal, "o", color="#009E73", label="KITTI 0001 sun")
    axes[2].axhline(30, color="#888888", lw=0.6, ls="--", label="sun error not seen per frame")
    log_axis(axes[2], [0.5, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0])
    axes[2].set(xlabel="focal factor $k$ (or $\\gamma$)", ylabel="equivalent sun error (deg)",
                title="a depth error is a small sun error")
    axes[2].legend(frameon=False, fontsize=5)
    fig.tight_layout(pad=0.3)
    save(fig, "coverage")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--no-teaser", action="store_true")
    a = p.parse_args()
    synthetic_fig()
    kitti_fig()
    coverage_fig()
    if not a.no_teaser:
        teaser()
