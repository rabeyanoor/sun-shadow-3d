"""Qualitative KITTI figure: frames where SunCheck agrees best and worst with the observed shadow.

    python scripts/make_qualitative.py [outputs/study/kitti_eval_h20.csv]

For Depth Pro at the calibrated camera, ranks frames by the Net SunCheck AUC and shows the two best
and the two worst: RGB, detector probability on the road, the two-state rendering, and the three
states of SunCheck (shadow, lit, undetermined). Writes paper/fig/qualitative.pdf and prints the
chosen frames, so the failure cases can be inspected and described.
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
import observability as obs  # noqa: E402

import shadows  # noqa: E402

MODEL, H = "depthpro", 20.0
csv = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "outputs" / "study" / "kitti_eval_h20.csv"
kit = pd.read_csv(csv)
t = kit[(kit.model == MODEL) & (kit.obs == "net") & (kit.exp == "focal_grid") & np.isclose(kit.value, 1.0)]
t = t.dropna(subset=["auc"])
t = t[t.det_frac >= 0.3].sort_values("auc")  # extreme AUCs on a few pixels are not informative
picks = list(t.key.iloc[-2:][::-1]) + list(t.key.iloc[:2])
labels = ["best", "2nd best", "worst", "2nd worst"]

fig, axes = plt.subplots(len(picks), 4, figsize=(7.0, 0.72 * len(picks)))
for row, (key, lab) in enumerate(zip(picks, labels)):
    z = np.load(ROOT / "outputs" / "cache" / f"{key}.npz")
    d = z[f"depth_{MODEL}"].astype(np.float64)
    region = z["receiver"] & (d > 0)
    gy = obs.ground_level(d, z["K"], region)
    st = obs.render_shadow_amodal(d, z["K"], z["sun"], gy, h_max=H, sky=z["sky"], return_state=True, only=region)
    r2 = shadows.render_shadow(d, z["K"], z["sun"], only=region)
    img = z["img"]
    prob = np.where(region, z["prob"].astype(float), np.nan)
    three = np.full(d.shape + (3,), 0.92)
    three[region & (st == obs.SHADOW)] = (0.75, 0.22, 0.17)
    three[region & (st == obs.LIT)] = (0.98, 0.85, 0.35)
    three[region & (st == obs.UNDETERMINED)] = (0.55, 0.55, 0.55)
    two = np.full(d.shape + (3,), 0.92)
    two[region & r2] = (0.75, 0.22, 0.17)
    two[region & ~r2] = (0.98, 0.85, 0.35)
    r = t[t.key == key].iloc[0]
    panels = [(img, f"{key[11:21]} #{int(key[22:])}  ({lab})"), (prob, "detector prob."),
              (two, f"2-state, IoU {r.iou:.2f}"), (three, f"SunCheck AUC {r.auc:.2f}, C {100 * r.det_frac:.0f}%")]
    for ax, (im, title) in zip(axes[row], panels):
        ax.imshow(im, cmap="magma", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(title, fontsize=6, pad=2)
        ax.axis("off")
    print(lab, key, f"auc={r.auc:.3f} iou={r.iou:.3f} det={r.det_frac:.2f} az={r.azimuth:.0f} el={r.elevation:.0f}")
fig.tight_layout(pad=0.2, h_pad=0.3, w_pad=0.2)
out = ROOT / "paper" / "fig" / "qualitative.pdf"
fig.savefig(out, bbox_inches="tight", pad_inches=0.02, dpi=300)
fig.savefig(out.with_suffix(".png"), bbox_inches="tight", pad_inches=0.02, dpi=150)
print("->", out)
