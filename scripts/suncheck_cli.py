"""Run SunCheck on your own depth map.

    python scripts/suncheck_cli.py --depth depth.npy --road road.png --image image.png \
        --K 721.5 721.5 609.6 172.9 --lat 49.011 --lon 8.416 --time "2011-09-26 09:02:25" --heading 12

    python scripts/suncheck_cli.py --kitti outputs/cache/2011_09_26_drive_0001_0000000000.npz --model depthpro

Inputs: a depth map in metres (.npy, 0 = invalid), the road (receiver) mask (.npy or image, nonzero = road),
the RGB image (for the detector-free darkness score) or a shadow probability map (--prob .npy), the
intrinsics at the depth resolution, and the sun: either --sun x y z in camera coordinates (x right, y down,
z forward) or GPS position, UTC time and camera heading (--heading, clockwise from north; --pitch, --roll).
Prints a JSON line with the AUC (0.5: no agreement beyond chance), the coverage and the class sizes; with
--png, also saves the three rendered states (red shadow, yellow lit, grey undetermined).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import suncheck  # noqa: E402


def load_mask(path):
    if path.endswith(".npy"):
        return np.load(path) > 0
    from PIL import Image

    return np.asarray(Image.open(path).convert("L")) > 0


def load_image(path):
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kitti", help="a cached KITTI frame (.npz from scripts/cache_kitti.py) instead of the inputs below")
    p.add_argument("--model", default="depthpro", help="depth model in the cached frame")
    p.add_argument("--depth")
    p.add_argument("--road")
    p.add_argument("--image")
    p.add_argument("--prob", help="shadow probability map (.npy), or 'net' with --kitti; default: image darkness")
    p.add_argument("--sky", help="sky mask; columns with sky at the top are not completed")
    p.add_argument("--K", nargs=4, type=float, metavar=("FX", "FY", "CX", "CY"))
    p.add_argument("--sun", nargs=3, type=float, metavar=("X", "Y", "Z"))
    p.add_argument("--lat", type=float)
    p.add_argument("--lon", type=float)
    p.add_argument("--time", help="UTC, e.g. '2011-09-26 09:02:25'")
    p.add_argument("--heading", type=float)
    p.add_argument("--pitch", type=float, default=0.0)
    p.add_argument("--roll", type=float, default=0.0)
    p.add_argument("--h-max", type=float, default=20.0, help="caster height prior (m)")
    p.add_argument("--n-min", type=int, default=20)
    p.add_argument("--png", help="save the rendered states")
    a = p.parse_args(argv)

    if a.kitti:
        z = np.load(a.kitti)
        depth, road, K, s, sky = z[f"depth_{a.model}"].astype(float), z["receiver"], z["K"], z["sun"], z["sky"]
        score = z["prob"].astype(np.float32) if a.prob == "net" else suncheck.darkness_score(z["img"], road)
    else:
        if not (a.depth and a.road and a.K and (a.image or a.prob)):
            p.error("--depth, --road, --K and --image or --prob are required (or use --kitti)")
        depth, road = np.load(a.depth).astype(float), load_mask(a.road)
        K = np.array([[a.K[0], 0, a.K[2]], [0, a.K[1], a.K[3]], [0, 0, 1]])
        sky = load_mask(a.sky) if a.sky else None
        score = np.load(a.prob).astype(np.float32) if a.prob else suncheck.darkness_score(load_image(a.image), road)
        if a.sun:
            s = np.asarray(a.sun, float) / np.linalg.norm(a.sun)
        elif None not in (a.lat, a.lon, a.time, a.heading):
            s = suncheck.sun_in_camera_frame(a.lat, a.lon, a.time, a.heading, a.pitch, a.roll)
        else:
            p.error("give --sun, or --lat --lon --time --heading")
        if depth.shape != road.shape or depth.shape != score.shape:
            p.error(f"shapes differ: depth {depth.shape}, road {road.shape}, score {score.shape}")

    r = suncheck.check(depth, K, s, road, score, sky=sky, h_max=a.h_max, n_min=a.n_min)
    print(json.dumps(dict(auc=None if np.isnan(r.auc) else round(r.auc, 4), coverage=round(r.coverage, 4),
                          n_shadow=r.n_shadow, n_lit=r.n_lit, sun=[round(float(v), 4) for v in s])))
    if a.png:
        from PIL import Image

        rgb = np.full(r.state.shape + (3,), 235, np.uint8)
        rgb[r.state == 1] = (191, 56, 43)
        rgb[r.state == 0] = (250, 217, 89)
        rgb[(r.state == -1) & np.asarray(road, bool)] = (140, 140, 140)
        Image.fromarray(rgb).save(a.png)
    return r


if __name__ == "__main__":
    main()
