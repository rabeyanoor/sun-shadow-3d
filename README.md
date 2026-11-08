# Sun-Shadow 3D

Can the sun check a monocular depth map without labels? GPS and a timestamp give the exact sun direction. A correct depth map should then cast the shadows the camera actually sees.

The repo has two parts. The **pilot study** (notebooks 01–06) shows that a naive overlap check fails on KITTI, and why. **SunCheck** (`src/observability.py`, `scripts/`) is the fix: an observability-aware, coverage-invariant shadow test, evaluated on 400 random streets and 600 KITTI frames.

## SunCheck: main results (600 KITTI frames, 40 drives, 4 depth models)

| Question | Result |
|---|---|
| Is there a signal at the true geometry? | Yes. SunCheck AUC 0.71 [0.68, 0.74] with the ISTD detector and 0.71 with no detector at all (chance 0.5). |
| Does it rank depth models like LiDAR? | Yes, by shape. Without a detector it orders MoGe-2-L > Depth Pro > DA-V2-L > DA-V2-S, the order of the LiDAR AbsRel after median scaling, and every neighbouring difference is positive (drive bootstrap). IoU ranks DA-V2-L below DA-V2-S. |
| Can it see metric scale? | No (Prop. 1). MoGe-2-L's scale is off by 1.30x, and it still scores best. |
| Can one frame see a 20–40 % focal or shape error? | No, for any score (truth is best in 5–15 % of cases, chance 20 %). This is Cor. 1. |
| Sun localisation | Truth best of 5 azimuths: SunCheck 44 %, IoU 29 %. |
| Pooled focal calibration | IoU's bias flips sign with the sun side (k̂ 0.56 ahead, 1.78 behind), as on synthetic streets. SunCheck's median 0.79 equals its synthetic bias, so the corrected value is 1.00. Held-out error against LiDAR: 40 % (31 % on common support), against 81 % for IoU. Unbiased, but coarse. |
| Ablation | The 3-state renderer without completion is defined on only 142 of 2389 cases (6 % of the road vs 18 %), so its good numbers come from easy frames. See `paper/gen/table_ablation.tex`. |

Theory checks on real depth maps (`scripts/verify_theory.py`): Prop. 1 and Prop. 2 hold pixel for pixel (IoU 1.00 over 232 cases). The focal/compression equivalence holds only locally (IoU 0.71).

## Findings

| Notebook | Question | Result |
|---|---|---|
| 01 `kitti_sun_direction` | Is our sun vector right on KITTI? | Yes. KITTI timestamps are local time (UTC+2). Pole shadows match the UTC+2 sun (az 175°, el 40°). |
| 02 `depth_and_rendered_shadows` | Do shadows rendered from depth match the real ones? | Only weakly. Road IoU is 0.06–0.30 from predicted depth (DA-V2-S) and 0.01–0.10 from LiDAR, because the casters (trees, facades) are above the LiDAR field of view. |
| 03 `corruption_study` | On real KITTI, does the score react to depth errors? | No. IoU stays ~0.07 for global scale ×0.5–2, shape warp and wrong focal. It does react to a wrong sun azimuth (±60° → 0.03), so the sun pipeline itself is sound. |
| 04 `synthetic_identifiability` | Physics or pipeline? (exact synthetic street) | **Both.** See below. |
| 05 `kitti_study` | Full study: 5 drives, 71 frames, DA-V2-S/L, Depth Pro, MoGe-2-L, threshold and a SegFormer-b2 shadow net trained on ISTD (test BER 1.9 %) | **No.** Scale is flat. Shape and focal move the score the wrong way: IoU is highest for compressed depth, and per frame the truth wins only 5-10 % of the time (chance 20 %). The score ranks DA-V2-L below DA-V2-S. See below. |
| 06 `coverage_bias` | Why does compressed depth score better? | **Coverage bias.** Compressed depth casts 1.3x more shadow and IoU follows the area. At the true depth the IoU (0.064) equals that of two independent masks of the same coverage (0.063), kappa -0.003: no agreement beyond chance. |

What the synthetic street (notebook 04) shows:

- **Global metric scale is unidentifiable from shadows.** A sun at infinity casts the same image-space shadows on a scaled scene. When the renderer's metric tolerances scale with the scene, the IoU is exactly constant (0.637 at every scale from 0.5 to 2).
- **Relative shape and focal length are identifiable.** With exact shadows, IoU falls from 0.64 to 0.41–0.45 at ±50 % warp or focal error.
- **So on KITTI the shape and focal signal was lost to the pipeline:** the dark-asphalt shadow detector and casters that are off-screen or above the LiDAR.

What the full KITTI study (notebook 05, run on Kaggle) adds:

- **A better shadow detector raises the IoU level (0.12 vs 0.08) but not its meaning.** The score still rises as depth is compressed. Per frame the true focal has the best score only 5 % of the time (95 % drive-bootstrap interval 3-8 %), far below the 20 % of chance; on average a 0.6 focal factor scores 0.042 IoU better than the true one. This fits a coverage bias: longer rendered shadows overlap more of a detector that marks about a fifth of the road.
- **One frame says little even about the sun.** The mean curve peaks at the true sun azimuth, but a single frame has its best score there only 26 % of the time.
- **It does not rank depth models.** MoGe-2-L scores highest and DA-V2-L lowest.

![kitti study](outputs/figures/05_kitti_study.png)

**Consequence:** the original claim, "a label-free check for *metric* depth", is impossible in principle. A shadow check can at most test relative shape and camera intrinsics. Even that needs a much better shadow detector and scenes with visible casters. `PLAN.md` (kept on the laptop) has the full reasoning and the reviewer concerns.

![synthetic](outputs/figures/04_synthetic_corruptions.png)

## Layout

```
sun-shadow-3d/
├── src/
│   ├── suncheck.py     public API: check, compare (common support), calibrate_focal, sun_in_camera_frame
│   ├── observability.py SunCheck internals: 3-state amodal renderer, AUC, coverage, focal/sun inversion
│   ├── corruptions.py  scale / shape / focal / sun corruption protocol and the fine focal grid
│   ├── kitti.py        KITTI raw readers: oxts, timestamps, calibration chain, velodyne -> depth
│   ├── sun.py          sun azimuth/elevation (pvlib) and its direction in the camera frame
│   ├── shadows.py      screen-space shadow rendering from a depth map
│   ├── scene.py        SegFormer / Mask2Former road masks, dark-asphalt and learned shadow detectors, IoU, kappa
│   ├── shadow_net.py   SegFormer mit-b2 shadow detector trained on ISTD (train, evaluate, BER)
│   ├── depth_models.py DA-V2-S/L, Depth Pro, MoGe-2-L behind one predict() call
│   └── synthetic.py    exact synthetic street: depth by ray casting, exact 3D shadows
├── notebooks/          01-06, jupytext .py + executed .ipynb, run one at a time
├── tests/              pytest: sun position, rotations, ground plane, scale invariance
├── scripts/            pilot: download_kitti.sh, shadow_study.py, kaggle_build.py, make_paper_numbers.py,
│                       make_paper_figures.py, kaggle_paper.py
│                       SunCheck: cache_kitti.py + kaggle_cache.py (frames, masks, depth on a Kaggle GPU),
│                       shadow_eval.py (+ kaggle_eval_shards.py), synthetic_benchmark.py, lidar_reference.py,
│                       verify_theory.py, make_v2_results.py, make_qualitative.py, suncheck_cli.py,
│                       kaggle_synthetic_h.py (H ablation)
├── kaggle/             one-file Kaggle kernel template for the study (GPU)
├── outputs/figures/
├── outputs/study/      study CSV, shadow-net metrics, argmax-at-truth and ranking tables
├── logs/kaggle/        Kaggle run logs
├── paper/              SunCheck_paper.pdf, SunCheck_supplementary.pdf; sources main.tex, supp.tex,
│                       sec/, gen/ (generated numbers and tables), fig/
├── PLAN.md             plan, reviewer concerns and findings log (laptop only, not in git)
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

## Paper

The compiled paper is `paper/SunCheck_paper.pdf` (8 pages + references) and the supplementary material
`paper/SunCheck_supplementary.pdf`. Sources: `paper/main.tex` and `paper/supp.tex`.

## Tests

```bash
.venv/bin/python -m pytest -q        # 39 tests: theory, renderers, SunCheck API and CLI, pipeline pieces
.venv/bin/ruff check src scripts tests
```

The tests need no KITTI download and no GPU: the pipeline tests run on synthetic streets and a synthetic
cached frame.

## Use SunCheck on your own data

SunCheck is a consistency test for a depth map, not a replacement for ground truth. It needs direct sun,
the road (or another flat receiver) mask, the intrinsics, and the sun in the camera frame (or GPS, UTC time
and the camera heading). Read the AUC (0.5 = no agreement beyond chance) together with the coverage.

```python
import sys; sys.path.insert(0, "src")
import suncheck

s = suncheck.sun_in_camera_frame(lat, lon, "2011-09-26 09:02:25", heading_deg=12.0)
r = suncheck.check(depth, K, s, road_mask, suncheck.darkness_score(image, road_mask))
print(r.auc, r.coverage)                       # r.state: 1 shadow, 0 lit, -1 undetermined
suncheck.compare({"model A": (dA, K, s), "model B": (dB, K, s)}, road_mask, score)   # common support
k_hat, curve = suncheck.calibrate_focal(frames, bias=0.79)   # coarse effective focal factor, many frames
```

```bash
python scripts/suncheck_cli.py --depth depth.npy --road road.png --image image.png \
    --K 721.5 721.5 609.6 172.9 --lat 49.011 --lon 8.416 --time "2011-09-26 09:02:25" --heading 12 --png states.png
python scripts/suncheck_cli.py --kitti outputs/cache/2011_09_26_drive_0056_0000000209.npz --model depthpro
```

It cannot see global scale, and one frame cannot see a 20–40 % focal or depth-compression error; these
need many frames, and even then the estimate is coarse.

## Run SunCheck (the paper's experiments)

```bash
python scripts/kaggle_cache.py <kaggle-user> && kaggle kernels push -p kaggle/build/sun-shadow-cache   # 600 frames, ~1 h GPU
kaggle kernels output <kaggle-user>/sun-shadow-cache -p outputs && tar xf outputs/cache.tar -C outputs
.venv/bin/python scripts/shadow_eval.py --cache outputs/cache --workers 4 --parts outputs/study/parts_h20 \
    --out outputs/study/kitti_eval_h20.csv          # resumable; --shard i/n splits it over machines
.venv/bin/python scripts/synthetic_benchmark.py --scenes 400 --workers 4 --tag _h20
.venv/bin/python scripts/synthetic_benchmark.py --scenes 400 --workers 4 --grid-only --tag _cs_h20
.venv/bin/python scripts/lidar_reference.py         # resumable per drive, reads velodyne scans remotely
.venv/bin/python scripts/verify_theory.py
.venv/bin/python scripts/make_v2_results.py         # numbers_v2.tex, tables, figures
.venv/bin/python scripts/make_qualitative.py
```

## Compile the paper

`paper/main.tex` ("When Can a Shadow Check Depth?") has the identifiability and observability theory, SunCheck,
the pilot and the full KITTI study. Every number in it is a macro generated from the result files
(`make_paper_numbers.py` for the pilot, `make_v2_results.py` and `verify_theory.py` for SunCheck).
The laptop has no LaTeX, so it is compiled on a Kaggle CPU kernel:

```bash
.venv/bin/python scripts/make_paper_numbers.py
.venv/bin/python scripts/make_paper_figures.py
.venv/bin/python scripts/kaggle_paper.py <kaggle-user>
kaggle datasets version -p kaggle/build/paper_data -m update --dir-mode zip   # first time: datasets create
kaggle kernels push -p kaggle/build/paper
kaggle kernels output <kaggle-user>/sun-shadow-3d-paper -p paper/build   # main.pdf + LaTeX logs
```

## Prior work

- Abrams, Miskell, Pless. *The Episolar Constraint*, CVPR 2013. Sparse depth from sun and shadow correspondences on webcams.
- Lalonde et al. *What do the sun and sky tell us about the camera?*, IJCV 2010.
- OutCast (2204.09341) and ShadowNeuS (2211.14086). Shadows for relighting and per-scene shape.
