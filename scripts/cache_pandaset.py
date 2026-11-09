"""Cache PandaSet frames in the same format as scripts/cache_kitti.py, for an independent validation.

    python scripts/cache_pandaset.py --root /kaggle/input/pandaset --weights shadow_net_segformer_b2_istd.pt \
        --out cache_panda --per-seq 8 --min-elevation 2

PandaSet (Scale AI / Hesai, CC BY 4.0): San Francisco and the Peninsula, May 2019, 1920x1080 front camera
(f = 1970 px) with real UNIX timestamps, GPS per frame and camera poses in a world frame whose x, y, z are
east, north, up (checked against GPS: the fitted rotation is under 1 degree; we still apply the per-sequence
fit). The sun direction in the camera follows from the full camera rotation, so pitch and roll are included.
Per frame it stores, at 1/6 resolution: the image, the road and sky masks (Mask2Former), the ISTD shadow
probability, the sun in camera coordinates, the intrinsics, the depth of every model, and the LiDAR depth
(both sensors, world points projected with the camera pose) for the LiDAR reference.
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pandaset import lidar_depth, quat_R, read_lidar, world_to_enu  # noqa: E402

import depth_models  # noqa: E402
import scene  # noqa: E402
import shadow_net  # noqa: E402
import shadows  # noqa: E402
import sun  # noqa: E402

SCALE = 1 / 6

p = argparse.ArgumentParser()
p.add_argument("--root", required=True, help="folder that contains the sequence folders (001, 002, ...)")
p.add_argument("--seqs", nargs="*", help="sequence names; default: all with the sun above --min-elevation")
p.add_argument("--per-seq", type=int, default=8)
p.add_argument("--min-elevation", type=float, default=2.0)
p.add_argument("--models", nargs="+", default=list(depth_models.MODELS))
p.add_argument("--weights", required=True)
p.add_argument("--out", default=str(ROOT / "outputs" / "cache_panda"))
args = p.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)


def down(x, hs, ws, mode="nearest"):
    t = torch.from_numpy(np.ascontiguousarray(x)).float()[None, None]
    return F.interpolate(t, size=(hs, ws), mode=mode)[0, 0].numpy()


seqs = args.seqs or sorted(d.name for d in Path(args.root).iterdir() if (d / "meta" / "gps.json").exists())
net = shadow_net.build()
net.load_state_dict(torch.load(args.weights, map_location=shadow_net.DEVICE))
net.eval()

t0 = time.time()
frames = []
if seqs:
    print("lidar files:", sorted(p.name for p in (Path(args.root) / seqs[0] / "lidar").glob("*"))[:3], flush=True)
for name in seqs:
    d = Path(args.root) / name
    try:
        cam = d / "camera" / "front_camera"
        poses, stamps = json.load(open(cam / "poses.json")), json.load(open(cam / "timestamps.json"))
        gps = json.load(open(d / "meta" / "gps.json"))
        intr = json.load(open(cam / "intrinsics.json"))
        K = np.array([[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1.0]])
        R_we = world_to_enu(poses, gps)
        n = min(len(poses), len(stamps), len(gps))
        when0 = datetime.fromtimestamp(stamps[0], tz=timezone.utc).replace(tzinfo=None)
        if sun.sun_az_el(gps[0]["lat"], gps[0]["long"], when0)[1] < args.min_elevation:
            continue
        for i in np.unique(np.linspace(0, n - 1, args.per_seq).round().astype(int)):
            when = datetime.fromtimestamp(stamps[i], tz=timezone.utc).replace(tzinfo=None)
            az, el = sun.sun_az_el(gps[i]["lat"], gps[i]["long"], when)
            Rc = quat_R(poses[i]["heading"])  # camera -> world, camera axes x right, y down, z forward
            tc = np.array([poses[i]["position"][k] for k in "xyz"])
            s_cam = Rc.T @ (R_we.T @ sun.enu_from_az_el(az, el))
            img = np.asarray(Image.open(cam / f"{i:02d}.jpg").convert("RGB"))
            h, w = img.shape[:2]
            hs, ws = int(round(h * SCALE)), int(round(w * SCALE))
            Kq = shadows.resize_K(K, ws / w, hs / h)
            labels = scene.segment_m2f(img)
            lid = next((d / "lidar" / f"{i:02d}{ext}" for ext in (".pkl.gz", ".pkl")
                        if (d / "lidar" / f"{i:02d}{ext}").exists()), d / "lidar" / "missing")
            tilt = float(np.degrees(np.arccos(np.clip(Rc[:, 1] @ np.array([0, 0, -1.0]), -1, 1))))
            frames.append(dict(
                drive=f"panda_{name}", frame=int(i), stamp=str(when), az=az, el=el, sun=s_cam / np.linalg.norm(s_cam),
                img=img, sky=labels == scene.SKY, receiver=down(scene.receiver_mask(labels), hs, ws) > 0.5, K=Kq,
                f_true=float(K[0, 0]), tilt=tilt,
                img_q=np.stack([down(img[..., c], hs, ws, "area") for c in range(3)], -1).round().astype(np.uint8),
                prob=down(shadow_net.predict(net, img), hs, ws, "area").astype(np.float16),
                sky_q=down(labels == scene.SKY, hs, ws) > 0.5,
                lidar=(lidar_depth(read_lidar(lid), Rc, tc, Kq, (hs, ws)).astype(np.float16) if lid.exists() else None),
                depth={}, f_pred={}))
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as e:
        print(f"skip {name}: {e!r}", flush=True)
        continue
    print(f"{name}: {len(frames)} frames so far ({(time.time() - t0) / 60:.1f} min)", flush=True)
del net
torch.cuda.empty_cache()

for mname in args.models:
    t1 = time.time()
    model = depth_models.MODELS[mname]()
    for f in frames:
        dm = model(f["img"])
        dm[f["sky"]] = 0
        f["depth"][mname] = down(dm, *f["receiver"].shape).astype(np.float16)
        if getattr(model, "last_focal", None) is not None:
            f["f_pred"][mname] = float(model.last_focal)
    del model
    torch.cuda.empty_cache()
    print(f"{mname}: {len(frames)} frames in {(time.time() - t1) / 60:.1f} min", flush=True)

index = []
for f in frames:
    key = f"{f['drive']}_{f['frame']:010d}"
    extra = {} if f["lidar"] is None else {"lidar": f["lidar"]}
    np.savez_compressed(out / f"{key}.npz", img=f["img_q"], receiver=f["receiver"], sky=f["sky_q"], prob=f["prob"],
                        sun=f["sun"], K=f["K"], **extra, **{f"depth_{m}": v for m, v in f["depth"].items()})
    index.append(dict(key=key, drive=f["drive"], frame=f["frame"], stamp=f["stamp"], azimuth=f["az"], elevation=f["el"],
                      f_true=f["f_true"], tilt=f["tilt"], **{f"f_{m}": v for m, v in f["f_pred"].items()}))
pd.DataFrame(index).to_csv(out / "frames.csv", index=False)
print(f"wrote {len(index)} frames to {out} ({(time.time() - t0) / 60:.1f} min)", flush=True)
