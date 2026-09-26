# Three-frame spine proportion comparison

User-requested experiment, September 26, 2026. Ratios are ordered
**sacrolumbar : thoracic : cervical**. Compare the existing 18:20:6.3 result
with 20:20:12, 20:20:13 and 20:20:14. These are experimental proportions,
not verified anthropometric standards or fixed percentages of body height.

The Ceres length-proportion residual is unchanged: each length prefers its
normalized share of the current three-length total, with 50 mm residual scale.
The total remains free. Compared with the baseline, both the lumbar/thoracic
balance and the cervical share change; this is not a cervical-only ablation.

All fits use 222 frames, three active frames, two fixed history frames, original
time weights, function tolerance 1e-6 and maximum 200 iterations. Keypoint targets,
initial seeds, person scale, SC offsets, rigid clavicle lengths, shoulder linkage
residuals, rest-pose residuals and temporal residuals are unchanged. There is no
global refinement. Cached baseline geometry is reused without rerunning it.

## Viewer and commands

Open <http://127.0.0.1:8773/recording_spine_ratio_comparison.html>. The compact
Spine column identifies each ratio. The selected 20:20:12 fit is initially visible. Use Solo, then Details to inspect neck-center and SC height at the
current frame. Heights are projected along the hip-to-shoulder axis, not world Z.
They compare fitted model landmarks with keypoint-derived geometry; there is
no independently measured SC position serving as anatomical ground truth.

From SkellyForge:

```powershell
uv run --no-sync poe solver-viewer-spine-ratios
uv run --no-sync poe solver-viewer-spine-ratios --publish-only
```

`--resume` reuses completed cases after an interrupted experiment; cached ratio
values must match the requested case. Source hashes, geometry, recording context,
processing settings and other model settings are checked before overlaying.
Generated JSON overwrites fixed files in `build/spine_ratio_comparison/`.
The prepared recording remains read-only. The owner selected 20:20:12 after visual review; that is now the default
ratio for recording experiments.

## Results

| Ratio | Ceres seconds | Read + fit seconds | Windows converged | Keypoint-target RMS, mm |
| --- | ---: | ---: | ---: | ---: |
| 18:20:6.3 | 83.9 | historical adapter | 220/220 | 15.951 |
| 20:20:12 | 80.2 | 86.6 | 219/220 | 15.964 |
| 20:20:13 | 79.0 | 85.7 | 220/220 | 15.964 |
| 20:20:14 | 81.2 | 91.0 | 220/220 | 15.970 |

| Ratio | Frame 188 SC height, mm | Frame 192 | Frame 203 | Frame 188 cervical length, mm |
| --- | ---: | ---: | ---: | ---: |
| 18:20:6.3 | 77.0 | -8.2 | 66.5 | 122.4 |
| 20:20:12 | 12.8 | 22.0 | 31.7 | 167.6 |
| 20:20:13 | -0.5 | -7.0 | 21.7 | 180.6 |
| 20:20:14 | -12.1 | -14.3 | 10.6 | 192.7 |

The revised proportions substantially reduce the high SC placement at frame 188.
Changes vary with pose; use the complete playback before selecting a default.
Similar target RMS does not mean the fitted landmarks occupy the same positions.
The baseline read/fit time used the old slow viewer adapter and is not comparable
to the current end-to-end timings.

Validation: 19 focused Python tests passed; the four-fit viewer passes numerical
geometry and playback checks. No native code or prepared recording was changed.

Frames 216–221 contain no mapped keypoint targets and remain explicitly marked
as unsupported by measurements. Exclude that tail when assessing agreement with
recorded motion. Convergence and keypoint RMS do not establish anatomical accuracy.
