# SunCheck: Can Sunlight Audit Monocular Depth?

Code and results for the paper **"Can Sunlight Audit Monocular Depth? Identifiability, Observability, and a
Label-Free Shadow Test for Automated-Driving Perception"** (R. Noor; under review at IEEE Transactions on
Intelligent Transportation Systems).

## What this research does

Self-driving cars and roadside cameras increasingly estimate **depth from a single image** (monocular depth).
In the field there is usually no laser scanner to tell whether that depth is right. This project asks whether
**the sun can serve as a free reference**:

1. The car's **GPS position and the time** give the exact direction of the sun.
2. If the predicted depth map is correct, rebuilding the scene in 3D and **lighting it with that sun** should
   cast shadows on the road **where the camera actually sees shadows**.
3. Comparing the rendered and observed shadows therefore checks the depth map **without any labels**.

<p align="center">
  <img src="docs/images/idea.png" width="520" alt="Sun direction from GPS and time, observed shadow, and shadows rendered from depth">
</p>

The work has two parts:

- **Theory: what shadows can and cannot see.** Shadows never reveal the overall scale of a depth map, a focal
  length error looks exactly like a sun direction error, and a driving camera's narrow view hides the objects
  (trees, buildings) that cast most road shadows. A naive overlap score is therefore misleading.
- **SunCheck: a test built on that theory.** Each road pixel is rendered as *shadow*, *lit* or *undetermined*
  (when the object casting the shadow is out of view), and agreement is scored only where the answer is known.
  On 600 KITTI and 664 PandaSet frames, SunCheck ranks four modern depth models in the same order as a laser
  scanner, using no ground truth at all.

<p align="center">
  <img src="docs/images/examples.png" width="900" alt="KITTI frames where SunCheck agrees best and worst with the observed shadow">
  <br><sub>KITTI frames where SunCheck agrees best (top) and worst (bottom) with the observed shadow.
  Red = rendered shadow, yellow = lit, grey = undetermined. Images: KITTI raw data, CC BY-NC-SA 3.0.</sub>
</p>

## Key results

| Question | Answer |
|---|---|
| Can shadows see metric scale? | **No.** Scale is invisible to a sun at infinity (Prop. 1, verified pixel for pixel on KITTI depth maps). |
| Can one frame see a 20–40 % focal or depth-compression error? | **No.** Such errors equal a sun error of a few degrees; the true value is best in about 10 % of frames (chance 20 %). |
| Does a naive overlap (IoU) check work? | **No.** IoU is won by the larger rendered shadow and never reproduces the laser ranking of depth models. |
| Is there a signal with SunCheck? | **Yes.** AUC 0.71 on 600 KITTI frames with no shadow detector (chance 0.5). |
| Does it rank depth models like a laser scanner? | **Yes, by shape.** Same order as the LiDAR shape error in 98 % of drive bootstrap resamples (KITTI) and 100 % (PandaSet, high sun). |
| Does a better shadow detector help? | It raises agreement (AUC 0.75 KITTI, 0.86 PandaSet day) but **not** the ranking; plain road darkness stays the best observation. |
| Does it work at a low sun? | Only weakly: learned detectors degrade (0.51–0.60), road darkness gives 0.67, and the test is rarely defined. |
| Can it calibrate the camera? | Coarsely: pooling 160 frames estimates the effective focal factor to about 21 %, with a bias that depends on the sun regime. |

Every number in the paper is generated from the result files in `outputs/study/`.

## Repository layout

```
sun-shadow-3d/
├── src/                     library code (importable after `sys.path.insert(0, "src")`)
│   ├── suncheck.py          public API: check, compare, calibrate_focal, sun_in_camera_frame
│   ├── observability.py     three-state amodal renderer, AUC score, coverage, observation fusion
│   ├── shadows.py           screen-space shadow rendering from a depth map
│   ├── sun.py               sun position (pvlib) and its direction in the camera frame
│   ├── corruptions.py       scale / shape / focal / sun corruptions and the focal grid
│   ├── synthetic.py         exact synthetic street scenes (ray-cast depth and shadows)
│   ├── kitti.py, pandaset.py  dataset readers (poses, calibration, LiDAR depth)
│   ├── depth_models.py      Depth Anything V2 S/L, Depth Pro, MoGe-2-L behind one predict()
│   ├── scene.py             road segmentation and shadow observations
│   └── shadow_net.py        SegFormer shadow detectors (v1: ISTD; v2: SBU+ISTD, low-sun augmentation)
├── scripts/                 experiment pipeline (see "Reproducing the paper")
├── tests/                   pytest suite (no dataset download or GPU needed)
├── kaggle/                  Kaggle kernel template used by the build scripts
├── outputs/study/           result tables (CSV) behind every number in the paper
└── docs/images/             figures used in this README
```

## Installation

Python 3.12. Everything except the depth models and detector training runs on a CPU.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cpu torch
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m pytest          # 48 tests
.venv/bin/ruff check src scripts tests
```

## Quick start

SunCheck needs a depth map, the camera intrinsics `K`, a road (receiver) mask, an observed shadow score and
the sun in the camera frame (or GPS, UTC time and the camera heading). Read the AUC (0.5 = no agreement
beyond chance) together with the coverage.

```python
import sys; sys.path.insert(0, "src")
import suncheck

s = suncheck.sun_in_camera_frame(lat, lon, "2011-09-26 09:02:25", heading_deg=12.0)
r = suncheck.check(depth, K, s, road_mask, suncheck.darkness_score(image, road_mask))
print(r.auc, r.coverage)             # r.state: 1 shadow, 0 lit, -1 undetermined

# compare depth models on the pixels that all of them determine
suncheck.compare({"model A": (depth_a, K, s), "model B": (depth_b, K, s)}, road_mask, score)

# coarse effective focal factor, pooled over many frames
k_hat, curve = suncheck.calibrate_focal(frames, bias=0.79)
```

Command line:

```bash
python scripts/suncheck_cli.py --depth depth.npy --road road.png --image image.png \
    --K 721.5 721.5 609.6 172.9 --lat 49.011 --lon 8.416 --time "2011-09-26 09:02:25" --heading 12 --png states.png
```

## Reproducing the paper

Heavy steps (depth models, detector training, the 600 + 664 frame evaluations) run on Kaggle kernels; the
`kaggle_*.py` scripts build each kernel. `<user>` is your Kaggle username.

**1. Cache frames, masks and depth** (GPU)

```bash
python scripts/kaggle_cache.py <user>        && kaggle kernels push -p kaggle/build/sun-shadow-cache        # KITTI (cache_kitti.py)
python scripts/kaggle_cache_panda.py <user>  && kaggle kernels push -p kaggle/build/sun-shadow-cache-panda  # PandaSet (cache_pandaset.py)
```

**2. Shadow detector v2** (GPU; trained on SBU + ISTD, design fixed on their test sets only)

```bash
python scripts/kaggle_detector.py <user> && kaggle kernels push -p kaggle/build/sun-shadow-detector-v2
# runs train_detector.py (weights + balanced error rates) and reprob.py (v2 probabilities for every cached frame)
```

**3. Evaluate SunCheck** (CPU, sharded over Kaggle kernels or run locally)

```bash
python scripts/kaggle_eval_shards.py <user> 5 0 1 2 3 4 --v2                 # KITTI
python scripts/kaggle_eval_shards.py <user> 4 0 1 2 3 --data=panda --v2      # PandaSet
python scripts/shadow_eval.py --cache outputs/cache --prob2 outputs/v2/prob2_kitti --out outputs/study/kitti_eval_h20.csv
python scripts/pixel_metrics.py --prob2 outputs/v2/prob2_kitti                # pixel-level agreement
python scripts/synthetic_benchmark.py --scenes 400 --tag _h20                 # random streets
python scripts/kaggle_synthetic_h.py <user>                                   # caster-height ablation
```

**4. LiDAR reference and theory checks**

```bash
python scripts/lidar_reference.py            # KITTI
python scripts/lidar_panda.py --cache outputs/cache_panda --out outputs/study/panda_lidar.csv
python scripts/verify_theory.py              # propositions on real depth maps
```

**5. Tables and figures**

```bash
python scripts/make_paper_numbers.py         # pilot numbers
python scripts/make_v2_results.py outputs/study/kitti_eval_v1only_h20.csv   # benchmark and KITTI
python scripts/make_paper_figures.py && python scripts/make_qualitative.py
```

The generators write LaTeX tables and PDF figures to `paper_tits/gen` and `paper_tits/fig`; the manuscript
itself is not part of this repository.

`make_v2_results.py` reads the KITTI evaluation with the original observations only (Net and Dark), i.e. the
output of `shadow_eval.py` run without `--prob2`, saved as `kitti_eval_v1only_h20.csv`. With the full
`kitti_eval_h20.csv` (which also holds Net-v2 and Fused) a few bootstrap bounds move in the third decimal.

The pilot study (`scripts/shadow_study.py`, `scripts/kaggle_build.py`) shows why a naive
overlap check fails on KITTI; it motivated SunCheck and supplies the pilot numbers of the paper.

## Data and licenses

| Dataset | Use | License |
|---|---|---|
| [KITTI raw](https://www.cvlibs.net/datasets/kitti/raw_data.php) | 600 frames, 40 drives | CC BY-NC-SA 3.0 |
| [PandaSet](https://pandaset.org) | 664 frames, independent validation | CC BY 4.0 |
| ISTD | shadow detector training | research use |
| SBU Shadow | shadow detector v2 training | non-commercial research use |

No dataset, cached frame or model weight is stored in this repository; the scripts download or read them.

The code is released under the [MIT License](LICENSE). The datasets keep their own licenses above
(KITTI and SBU allow non-commercial use only).

## Citation

```bibtex
@article{noor2026suncheck,
  title   = {Can Sunlight Audit Monocular Depth? Identifiability, Observability, and a Label-Free
             Shadow Test for Automated-Driving Perception},
  author  = {Noor, Rabeya},
  journal = {IEEE Transactions on Intelligent Transportation Systems},
  note    = {under review},
  year    = {2026}
}
```
