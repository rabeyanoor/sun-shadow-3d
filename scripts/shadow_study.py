"""Shadow study v2: learned shadow detector, stronger depth and segmentation models, more drives.

    python scripts/shadow_study.py --drives 2011_09_26_drive_0001 2011_09_26_drive_0009 --every 10

For every frame, every depth model and both shadow detectors (dark-asphalt threshold and the
ISTD-trained SegFormer), it renders shadows from the predicted depth under the same four
corruptions as notebook 03 and scores them on the road. One CSV row per
(frame, model, detector, corruption, value).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import depth_models  # noqa: E402
import kitti  # noqa: E402
import scene  # noqa: E402
import shadow_net  # noqa: E402
import shadows  # noqa: E402
import sun  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--raw", default=str(ROOT / "data" / "raw"))
p.add_argument("--drives", nargs="+", default=["2011_09_26_drive_0001"])
p.add_argument("--every", type=int, default=10)
p.add_argument("--max-frames", type=int, default=15)
p.add_argument("--models", nargs="+", default=list(depth_models.MODELS))
p.add_argument("--shadow-epochs", type=int, default=8)
p.add_argument("--shadow-weights", default=None, help="load a trained detector instead of training one")
p.add_argument("--istd-limit", type=int, default=None, help="smoke test: use only this many ISTD images")
p.add_argument("--out", default=str(ROOT / "outputs" / "study"))
args = p.parse_args()

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
SCALE, UTC_OFFSET = 0.25, 2

# ---------------------------------------------------------------- 1. shadow detector
t0 = time.time()
if args.shadow_weights:
    net = shadow_net.build()
    net.load_state_dict(torch.load(args.shadow_weights, map_location=shadow_net.DEVICE))
    net.eval()
else:
    net = shadow_net.train(epochs=args.shadow_epochs, log=lambda m: print(m, flush=True), limit=args.istd_limit)
istd_ber = shadow_net.evaluate(net, limit=args.istd_limit)
print(f"ISTD test BER {istd_ber:.2f}%  ({(time.time() - t0) / 60:.1f} min)", flush=True)
(out / "shadow_net.json").write_text(json.dumps({"istd_test_ber": istd_ber, "epochs": args.shadow_epochs,
                                                 "backbone": "nvidia/mit-b2", "size": shadow_net.SIZE}))
torch.save(net.state_dict(), out / "shadow_net_segformer_b2_istd.pt")


# ---------------------------------------------------------------- 2. frames
def down(x, hs, ws, mode="nearest"):
    t = torch.from_numpy(np.ascontiguousarray(x)).float()[None, None]
    return F.interpolate(t, size=(hs, ws), mode=mode)[0, 0].numpy()


frames = []
for drive in args.drives:
    date_dir = Path(args.raw) / drive[:10]
    ddir = date_dir / f"{drive}_sync"
    calib = kitti.read_calib(date_dir)
    oxts = kitti.read_oxts(ddir)
    stamps = kitti.read_timestamps(ddir / "oxts" / "timestamps.txt")
    for i in list(range(0, len(stamps), args.every))[: args.max_frames]:
        img = kitti.load_image(ddir, i)
        h, w = img.shape[:2]
        hs, ws = int(h * SCALE), int(w * SCALE)
        labels = scene.segment_m2f(img)
        receiver = down(scene.receiver_mask(labels), hs, ws) > 0.5
        if receiver.sum() < 200:
            continue
        gray = down(img.mean(-1), hs, ws, "area")
        s_cam, az, el = sun.sun_in_camera(oxts, i, stamps[i], calib, UTC_OFFSET)
        frames.append(dict(
            drive=drive, frame=i, img=img, sky=labels == scene.SKY, receiver=receiver, sun=s_cam, el=el,
            K=shadows.resize_K(calib["K2"], ws / w, hs / h), hw=(hs, ws),
            obs={"threshold": scene.observed_shadow(np.repeat(gray[..., None], 3, -1), receiver),
                 "segformer-istd": scene.observed_shadow_net(down(shadow_net.predict(net, img), hs, ws, "area"), receiver)}))
print(len(frames), "frames", flush=True)
np.savez_compressed(out / "example_frame.npz", img=frames[0]["img"], **{f"obs_{k}": v for k, v in frames[0]["obs"].items()})


# ---------------------------------------------------------------- 3. corruptions
def warp(d, g):
    m = np.median(d[d > 0])
    o = d.copy()
    o[d > 0] = m * (d[d > 0] / m) ** g
    return o


def scale_focal(K, k):
    K = K.copy()
    K[0, 0] *= k
    K[1, 1] *= k
    return K


def rotate_about_up(v, deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]]) @ v


CORRUPTIONS = {
    "scale": ([0.5, 0.75, 1.0, 1.5, 2.0], lambda d, K, s, a: (d * a, K, s)),
    "shape": ([0.6, 0.8, 1.0, 1.25, 1.5], lambda d, K, s, g: (warp(d, g), K, s)),
    "focal": ([0.6, 0.8, 1.0, 1.25, 1.5], lambda d, K, s, k: (d, scale_focal(K, k), s)),
    "sun_azimuth": ([-60, -30, 0, 30, 60], lambda d, K, s, a: (d, K, rotate_about_up(s, a))),
}

rows = []
for name in args.models:
    t1 = time.time()
    model = depth_models.MODELS[name]()
    for f in frames:
        d = model(f["img"])
        d[f["sky"]] = 0
        d = down(d, *f["hw"])
        region = f["receiver"] & (d > 0)
        for corr, (values, fn) in CORRUPTIONS.items():
            for v in values:
                dc, Kc, sc = fn(d.copy(), f["K"].copy(), f["sun"].copy(), v)
                rendered = shadows.render_shadow(dc, Kc, sc)
                for det, obs in f["obs"].items():
                    rows.append(dict(drive=f["drive"], frame=f["frame"], elevation=f["el"], model=name, detector=det,
                                     corruption=corr, value=v, iou=scene.iou(rendered, obs, region),
                                     kappa=scene.kappa(rendered, obs, region),
                                     rendered_frac=float(rendered[region].mean()) if region.any() else np.nan,
                                     obs_frac=float(obs[region].mean()) if region.any() else np.nan))
    del model
    torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(out / "shadow_study.csv", index=False)
    print(f"{name}: done in {(time.time() - t1) / 60:.1f} min", flush=True)
print("all done", flush=True)
