# Sun-Shadow 3D

Can the sun check a monocular depth map without labels? GPS and a timestamp give the exact sun direction. A correct depth map should then cast the shadows the camera actually sees.

This repo is the complete pilot study. Its main finding is partly negative, and it is written up as such.

## Findings

| Notebook | Question | Result |
|---|---|---|
| 01 `kitti_sun_direction` | Is our sun vector right on KITTI? | Yes. KITTI timestamps are local time (UTC+2). Pole shadows match the UTC+2 sun (az 175°, el 40°). |
| 02 `depth_and_rendered_shadows` | Do shadows rendered from depth match the real ones? | Only weakly. Road IoU is 0.06–0.30 from predicted depth (DA-V2-S) and 0.01–0.10 from LiDAR, because the casters (trees, facades) are above the LiDAR field of view. |
| 03 `corruption_study` | On real KITTI, does the score react to depth errors? | No. IoU stays ~0.07 for global scale ×0.5–2, shape warp and wrong focal. It does react to a wrong sun azimuth (±60° → 0.03), so the sun pipeline itself is sound. |
| 04 `synthetic_identifiability` | Physics or pipeline? (exact synthetic street) | **Both.** See below. |

What the synthetic street (notebook 04) shows:

- **Global metric scale is unidentifiable from shadows.** A sun at infinity casts the same image-space shadows on a scaled scene. When the renderer's metric tolerances scale with the scene, the IoU is exactly constant (0.637 at every scale from 0.5 to 2).
- **Relative shape and focal length are identifiable.** With exact shadows, IoU falls from 0.64 to 0.41–0.45 at ±50 % warp or focal error.
- **So on KITTI the shape and focal signal was lost to the pipeline:** the dark-asphalt shadow detector and casters that are off-screen or above the LiDAR.

**Consequence:** the original claim, "a label-free check for *metric* depth", is impossible in principle. A shadow check can at most test relative shape and camera intrinsics. Even that needs a much better shadow detector and scenes with visible casters. `PLAN.md` keeps the full reasoning and the reviewer concerns.

![synthetic](outputs/figures/04_synthetic_corruptions.png)

## Layout

```
sun-shadow-3d/
├── src/
│   ├── kitti.py        KITTI raw readers: oxts, timestamps, calibration chain, velodyne -> depth
│   ├── sun.py          sun azimuth/elevation (pvlib) and its direction in the camera frame
│   ├── shadows.py      screen-space shadow rendering from a depth map
│   ├── scene.py        SegFormer road/sky masks, dark-asphalt shadow detector, IoU
│   └── synthetic.py    exact synthetic street: depth by ray casting, exact 3D shadows
├── notebooks/          01-04, jupytext .py + executed .ipynb, run one at a time
├── tests/              pytest: sun position, rotations, ground plane, scale invariance
├── scripts/download_kitti.sh
├── outputs/figures/
├── PLAN.md             plan v2, reviewer concerns, prior work, pilot findings
└── requirements.txt
```

## Run

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cpu torch
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m ipykernel install --user --name sun-shadow-3d
scripts/download_kitti.sh                     # ~0.45 GB, drive 2011_09_26_0001
.venv/bin/python -m pytest -q tests

cd notebooks
../.venv/bin/jupytext --to ipynb 04_synthetic_identifiability.py
../.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.kernel_name=sun-shadow-3d 04_synthetic_identifiability.ipynb
```

Everything runs on a CPU. Notebooks 02 and 03 take a few minutes each, because they run DA-V2-S and SegFormer-b0.

## Prior work

- Abrams, Miskell, Pless. *The Episolar Constraint*, CVPR 2013. Sparse depth from sun and shadow correspondences on webcams.
- Lalonde et al. *What do the sun and sky tell us about the camera?*, IJCV 2010.
- OutCast (2204.09341) and ShadowNeuS (2211.14086). Shadows for relighting and per-scene shape.
