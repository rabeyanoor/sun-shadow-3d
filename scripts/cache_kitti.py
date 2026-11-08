"""Cache everything the shadow analysis needs from KITTI raw, so all scoring runs on a laptop.

    python scripts/cache_kitti.py --weights shadow_net_segformer_b2_istd.pt --out cache \
        --drives 2011_09_26_drive_0001 2011_09_29_drive_0026 --per-drive 15

Frames are read straight out of the remote KITTI zips with HTTP range requests (remotezip),
so a 4 GB drive costs a few MB. Per frame it stores, at quarter resolution: the RGB image, the
road and sky masks (Mask2Former Swin-L), the ISTD-trained shadow probability, the sun direction
in the camera frame and the intrinsics; then the depth of every model (float16), and the focal
length that Depth Pro and MoGe-2 predict for themselves.
"""
import argparse
import io
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import depth_models  # noqa: E402
import kitti  # noqa: E402
import scene  # noqa: E402
import shadow_net  # noqa: E402
import shadows  # noqa: E402
import sun  # noqa: E402

S3 = "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data"
SCALE, UTC_OFFSET = 0.25, 2  # KITTI timestamps are CEST (notebook 01)

p = argparse.ArgumentParser()
p.add_argument("--drives", nargs="+", required=True)
p.add_argument("--per-drive", type=int, default=15, help="frames per drive, spread evenly over the drive")
p.add_argument("--models", nargs="+", default=list(depth_models.MODELS))
p.add_argument("--weights", required=True, help="ISTD shadow detector state dict")
p.add_argument("--out", default=str(ROOT / "outputs" / "cache"))
p.add_argument("--calib", default="/tmp/kitti_calib")
args = p.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)


def down(x, hs, ws, mode="nearest"):
    t = torch.from_numpy(np.ascontiguousarray(x)).float()[None, None]
    return F.interpolate(t, size=(hs, ws), mode=mode)[0, 0].numpy()


def calib_for(date):
    from remotezip import RemoteZip

    d = Path(args.calib) / date
    if not (d / "calib_cam_to_cam.txt").exists():
        d.mkdir(parents=True, exist_ok=True)
        with RemoteZip(f"{S3}/{date}_calib.zip") as z:
            for n in z.namelist():
                if n.endswith(".txt"):
                    (d / Path(n).name).write_bytes(z.read(n))
    return kitti.read_calib(d)


def read_drive(drive):
    """Yield (frame index, timestamp, oxts row, RGB image) for evenly spread frames of a drive."""
    from remotezip import RemoteZip

    with RemoteZip(f"{S3}/{drive}/{drive}_sync.zip") as z:
        names = z.namelist()
        ts_name = next(n for n in names if n.endswith("oxts/timestamps.txt"))
        tmp = Path("/tmp") / f"{drive}_ts.txt"
        tmp.write_bytes(z.read(ts_name))
        stamps = kitti.read_timestamps(tmp)
        imgs = {int(Path(n).stem): n for n in names if "/image_02/data/" in n and n.endswith(".png")}
        oxts = {int(Path(n).stem): n for n in names if "/oxts/data/" in n and n.endswith(".txt")}
        n = min(len(stamps), len(imgs), len(oxts))
        for i in np.unique(np.linspace(0, n - 1, args.per_drive).round().astype(int)):
            row = np.array(z.read(oxts[i]).decode().split(), dtype=np.float64)
            img = np.asarray(Image.open(io.BytesIO(z.read(imgs[i]))).convert("RGB"))
            yield int(i), stamps[i], row, img


net = shadow_net.build()
net.load_state_dict(torch.load(args.weights, map_location=shadow_net.DEVICE))
net.eval()

# ---------------------------------------------------------------- 1. frames, masks, sun, detector
t0 = time.time()
frames = []
for drive in args.drives:
    try:
        calib = calib_for(drive[:10])
        for i, stamp, row, img in read_drive(drive):
            oxts = {k: np.array([row[j]]) for j, k in enumerate(kitti.OXTS_FIELDS)}
            h, w = img.shape[:2]
            hs, ws = int(h * SCALE), int(w * SCALE)
            labels = scene.segment_m2f(img)
            receiver = down(scene.receiver_mask(labels), hs, ws) > 0.5
            s_cam, az, el = sun.sun_in_camera(oxts, 0, stamp, calib, UTC_OFFSET)
            frames.append(dict(
                drive=drive, frame=i, stamp=str(stamp), az=az, el=el, sun=s_cam, img=img, sky=labels == scene.SKY,
                receiver=receiver, K=shadows.resize_K(calib["K2"], ws / w, hs / h), f_true=float(calib["K2"][0, 0]),
                img_q=np.stack([down(img[..., c], hs, ws, "area") for c in range(3)], -1).round().astype(np.uint8),
                prob=down(shadow_net.predict(net, img), hs, ws, "area").astype(np.float16),
                sky_q=down(labels == scene.SKY, hs, ws) > 0.5, depth={}, f_pred={}))
    except (zipfile.BadZipFile, StopIteration, OSError, KeyError) as e:  # a drive that is missing or broken
        print(f"skip {drive}: {e!r}", flush=True)
        continue
    print(f"{drive}: {len(frames)} frames so far ({(time.time() - t0) / 60:.1f} min)", flush=True)
del net
torch.cuda.empty_cache()

# ---------------------------------------------------------------- 2. depth models
for name in args.models:
    t1 = time.time()
    model = depth_models.MODELS[name]()
    for f in frames:
        d = model(f["img"])
        d[f["sky"]] = 0
        f["depth"][name] = down(d, *f["receiver"].shape).astype(np.float16)
        if getattr(model, "last_focal", None) is not None:
            f["f_pred"][name] = float(model.last_focal)
    del model
    torch.cuda.empty_cache()
    print(f"{name}: {len(frames)} frames in {(time.time() - t1) / 60:.1f} min", flush=True)

# ---------------------------------------------------------------- 3. write
index = []
for f in frames:
    key = f"{f['drive']}_{f['frame']:010d}"
    np.savez_compressed(out / f"{key}.npz", img=f["img_q"], receiver=f["receiver"], sky=f["sky_q"], prob=f["prob"],
                        sun=f["sun"], K=f["K"], **{f"depth_{m}": d for m, d in f["depth"].items()})
    index.append(dict(key=key, drive=f["drive"], frame=f["frame"], stamp=f["stamp"], azimuth=f["az"], elevation=f["el"],
                      f_true=f["f_true"], **{f"f_{m}": v for m, v in f["f_pred"].items()}))
pd.DataFrame(index).to_csv(out / "frames.csv", index=False)
print(f"wrote {len(index)} frames to {out} ({(time.time() - t0) / 60:.1f} min)", flush=True)
