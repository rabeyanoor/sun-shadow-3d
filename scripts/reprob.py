"""Run a shadow detector on the full-resolution images of frames that are already cached.

    python scripts/reprob.py --data kitti --frames outputs/cache/frames.csv --weights v2.pt --backbone nvidia/mit-b5 \
        --size 512 --tta --out prob2_kitti [--check-weights shadow_net_segformer_b2_istd.pt --check-cache cache/]
    python scripts/reprob.py --data panda --root /kaggle/input/pandaset --frames outputs/cache_panda/frames.csv ...

Writes one float16 array <key>.npy per frame at the cache resolution, downsampled exactly as the cache
scripts do (area interpolation of the full-resolution probability). shadow_eval.py --prob2 reads them as the
observation "net2". With --check-weights and --check-cache, the old detector is rerun on the first frames and
compared with the cached probability, which shows that the images read here are the cached ones.
"""
import argparse
import io
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import shadow_net  # noqa: E402

S3 = "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data"
SCALE = {"kitti": 0.25, "panda": 1 / 6}


def down(x, hs, ws):
    t = torch.from_numpy(np.ascontiguousarray(x)).float()[None, None]
    return F.interpolate(t, size=(hs, ws), mode="area")[0, 0].numpy()


def kitti_images(frames):
    from remotezip import RemoteZip

    for drive, rows in frames.groupby("drive", sort=False):
        with RemoteZip(f"{S3}/{drive}/{drive}_sync.zip") as z:
            imgs = {int(Path(n).stem): n for n in z.namelist() if "/image_02/data/" in n and n.endswith(".png")}
            for _, r in rows.iterrows():
                yield r["key"], np.asarray(Image.open(io.BytesIO(z.read(imgs[int(r["frame"])]))).convert("RGB"))


def panda_images(frames, root):
    for _, r in frames.iterrows():
        seq = r["drive"].removeprefix("panda_")
        path = Path(root) / seq / "camera" / "front_camera" / f"{int(r['frame']):02d}.jpg"
        yield r["key"], np.asarray(Image.open(path).convert("RGB"))


def load(weights, backbone):
    net = shadow_net.build(backbone)
    net.load_state_dict(torch.load(weights, map_location=shadow_net.DEVICE))
    return net.eval()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", choices=["kitti", "panda"], required=True)
    p.add_argument("--frames", required=True, help="frames.csv of the cache")
    p.add_argument("--root", help="PandaSet folder with the sequence folders")
    p.add_argument("--weights", required=True)
    p.add_argument("--backbone", default="nvidia/mit-b5")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--tta", action="store_true")
    p.add_argument("--out", required=True)
    p.add_argument("--check-weights", help="old detector (mit-b2, 384 px) to compare with the cached probability")
    p.add_argument("--check-cache", help="cache folder with the <key>.npz files")
    p.add_argument("--check-n", type=int, default=3)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    frames = pd.read_csv(a.frames)
    net = load(a.weights, a.backbone)
    old = load(a.check_weights, "nvidia/mit-b2") if a.check_weights else None
    images = kitti_images(frames) if a.data == "kitti" else panda_images(frames, a.root)
    t0, done, diffs = time.time(), 0, []
    for key, img in images:
        h, w = img.shape[:2]
        hs, ws = (int(h * SCALE["kitti"]), int(w * SCALE["kitti"])) if a.data == "kitti" else \
            (int(round(h * SCALE["panda"])), int(round(w * SCALE["panda"])))
        prob = down(shadow_net.predict(net, img, a.size, a.tta), hs, ws).astype(np.float16)
        np.save(out / f"{key}.npy", prob)
        if old is not None and a.check_cache and len(diffs) < a.check_n:
            ref = np.load(Path(a.check_cache) / f"{key}.npz")["prob"].astype(np.float32)
            again = down(shadow_net.predict(old, img), hs, ws)
            diffs.append(float(np.abs(again - ref).max()))
            print(f"check {key}: max |old detector now - cached| = {diffs[-1]:.4f}", flush=True)
        done += 1
        if done % 50 == 0:
            print(f"{done}/{len(frames)} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    print(f"wrote {done} of {len(frames)} frames to {out}; check diffs {diffs}", flush=True)
