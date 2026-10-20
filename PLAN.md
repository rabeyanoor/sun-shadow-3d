# Sun-Shadow 3D: research plan (v2, 2026-10-18)

Working title: **Solar Geometry as a Label-Free Diagnostic for Metric Monocular Depth**

v1 was "test + proof + fine-tune at web scale". After asking ChatGPT and Gemini to review it
as tough CVPR/ICCV reviewers, the plan is narrowed: the main claim is the diagnostic,
and fine-tuning is the second contribution, done only once the diagnostic is shown to work.

## What the reviews agreed on

| Concern | What we do about it |
|---|---|
| A wrong shadow can come from wrong depth, a bad shadow mask, missing occluders or lighting, not only depth | Controlled corruption study: on KITTI, corrupt LiDAR depth in *scale* only and in *shape* only, and check the score reacts to each. Report when the signal is ambiguous |
| Soft shadows, sky light, clouds | Score only high-confidence hard-shadow *boundaries* on clear-sky frames. Model the sun as a 0.53 deg disc (penumbra width), not a point |
| Dark paint or asphalt vs real shadow (albedo) | Restrict to albedo-neutral receivers (road, pavement) first; ablate with and without an intrinsic / shadow-detector mask |
| Off-screen casters, 2.5D depth | Use only caster-shadow pairs where the caster is visible; report the percentage of observable pixels |
| Sun elevation conditioning (no constraint at noon, unstable at dusk) | Derive sensitivity from h = L * tan(alpha). Give each image an **observability score** and do not score images below a threshold |
| Unknown yaw on web data is circular (yaw can hide bad depth) | Phase 1 is KITTI only (yaw is known). Web data comes later, with yaw recovery tested on held-out known-yaw sequences first |
| KITTI sun vector is not automatically exact | Check the timestamp timezone, propagate GPS/IMU/calibration uncertainty into a sun-angle error bar (notebook 01) |
| VGGT is a different task from single-image depth | Separate protocols: single-image models vs multi-frame models |

## Prior work to cite (checked)

- Abrams, Miskell, Pless. *The Episolar Constraint: Monocular Shape from Shadow Correspondence*. CVPR 2013. The closest work: it uses sun and shadows on webcams to get sparse depth. We are different: we diagnose and adapt learned metric-depth models; we do not reconstruct.
- Lalonde et al. *What do the sun and sky tell us about the camera?* (IJCV 2010), plus their work on weather and time from photos.
- Satellite NeRF with a known sun direction (S-NeRF, Derksen and Izzo 2021): multi-view satellite images, not single-image depth.
- OutCast 2204.09341 and ShadowNeuS 2211.14086: shadows used for relighting and for per-scene shape.
- AdapterShadow 2311.08891 (SAM-based shadow detection): a possible tool.
- Not found on HF, so not cited until checked: a 2025 preprint "Physics-Grounded Shadow Generation from Monocular 3D Geometry Priors" that ChatGPT mentioned.

## Contributions (in priority order)

1. **Observability theory.** When can a shadow constrain metric scale, relative shape, or neither? This gives a per-image observability score.
2. **Diagnostic.** A shadow-boundary consistency score. On KITTI it predicts LiDAR metric error, beyond the model's own uncertainty and beyond ensemble disagreement (risk-coverage curves).
3. **Controlled validation.** Scale-only vs shape-only depth corruption shows what the score is sensitive to.
4. **Adaptation (second).** A confidence-weighted shadow loss with no regression on shadow-free and indoor data.
5. **Benchmark.** An observability-stratified KITTI split, plus webcams later.

## Notebook order (run one at a time)

1. `01_kitti_sun_direction`: sun vector in the camera frame, timezone check by eye.
2. `02_depth_and_rendered_shadows`: shadows rendered from LiDAR vs from predicted depth.
3. `03_corruption_study`: corrupt LiDAR depth (scale / shape), and check how the score responds.
4. `04_shadow_masks`: real shadow masks (detector trained on ISTD, or AdapterShadow) and boundary extraction.
5. `05_kitti_audit`: several models and drives, score vs LiDAR error, risk-coverage.
6. Later: observability theory, uncertainty baselines, adaptation (GPU, Kaggle).

## Compute note

The laptop is CPU only (8 cores, 14 GB RAM). Notebooks 01-05 are designed to run on CPU at quarter resolution with small models. Fine-tuning needs a GPU (Kaggle or similar).

## Pilot findings (2026-10-18, KITTI 2011_09_26_drive_0001, DA-V2-Small metric, CPU)

- Notebook 01: KITTI timestamps are **local time (UTC+2)**. Pole shadows match the UTC+2 sun (azimuth 175, elevation 40), not UTC+0.
- Notebook 02: LiDAR is a poor shadow oracle. Most road shadows come from tree canopies and house fronts above the LiDAR vertical FOV, so LiDAR geometry has no caster (IoU 0.01-0.10). The predicted depth covers the full image (IoU 0.06-0.30). The simple dark-asphalt detector misreads tram-track gravel as shadow, so receivers are now road only.
- Notebook 03 (9 frames, mean road IoU about 0.075):
  - global depth scale x0.5..x2: IoU 0.070-0.077, i.e. **flat**. This confirms the theory: shadows from a sun at infinity are invariant to global scale. **The shadow test cannot diagnose metric scale.**
  - shape warp (gamma 0.6-1.5) and wrong focal (x0.6-1.5): nearly flat (0.069-0.080). With this detector and renderer the test is barely sensitive to shape either.
  - sun azimuth offset +-60 deg: IoU drops to 0.03, so the pipeline does respond to the sun direction (sanity check passes).
- Conclusion: the main claim "label-free check for *metric* depth" is not supported. Any further version needs scenes with visible casters, a better shadow detector, and a claim about shape/camera rather than scale. Decision pending with the user.

## Synthetic identifiability (notebook 04, 2026-10-19)

- On an exact synthetic street, global scale is exactly invariant once the renderer's metric tolerances scale with the scene: IoU is 0.637 at every scale.
- Shape warp and wrong focal length do lower IoU (0.64 -> 0.41-0.45 at +-50 %).
- So metric scale is impossible in principle, while shape and focal are possible; on KITTI their signal was lost to the shadow detector and to off-screen casters.
- The project stops here as a documented pilot; the zoom-equivariance project continues the "free physical ruler" line.

## Full KITTI study (notebook 05, Kaggle, 2026-10-20)

- 5 drives, 71 frames; DA-V2-S/L, Depth Pro, MoGe-2-L; dark-asphalt threshold and a SegFormer mit-b2 shadow detector trained on ISTD (test BER 1.9 %).
- Scale is flat (as theory says). Shape and focal move the IoU monotonically towards *compressed* depth, so the score is highest at the wrong value; per frame the truth wins 5-10 % of the time (chance 20 %). (A first version of the drive bootstrap averaged a drive drawn twice into one copy; it now resamples per-case values.) Consistent with a coverage bias of IoU.
- The learned detector raises IoU from about 0.08 to 0.12 but keeps the trend. The score ranks DA-V2-L below DA-V2-S.
- Per frame even the sun azimuth is at its best only 26 % of the time; the signal exists only on average.
- Final: shadow IoU is not a usable label-free depth check on real driving data. A future version would need an area-unbiased score (shadow boundaries, caster-shadow correspondences) and scenes with visible casters.
