# Soft equality of the two axial spine lengths

Experiment 17 adds a single Ceres residual per frame to the existing lower-SC,
relaxed-shoulder, free-spine fit. It compares **sacrolumbar** and **thoracic**
lengths in millimeters. Cervical geometry stays fixed.

```text
r = sqrt(time_weight) * (L_sacrolumbar - L_thoracic) / scale_mm
```

The residual references the two existing scalar length parameter blocks. No
new parameter blocks are introduced. It prefers equal absolute lengths, not
equal ratios to their reference lengths. Their sum remains free, subject to
the same keypoint, pose, linkage and chest-line residuals as the reference fit.
This is a geometric modeling preference, not an anatomical measurement.
It is not a temporal smoothness residual: both lengths can still change together
between frames. The time weight matches the existing trapezoidal sequence objective.

`kDefaultLengthEqualityScale = 50 mm` in
`cpp/include/skellyforge/chain_solver_constants.h` is exposed as
`SPINE_LENGTH_EQUALITY_SCALE_MM` in `scripts/solver_fit_settings.py`.
Smaller values give stronger equality preference. It is a residual scale, not
a maximum allowed difference. Native input `LengthEqualityPrior` explicitly
names the two axial segment indices and validates their identities and scale.

Run `uv run --no-sync poe solver-viewer-spine-equality` from SkellyForge.
It reads the saved reference fit, checks that recording, initialization, reference
geometry and other settings match, runs the new fit, then regenerates both viewers.
The prepared recording is read-only. Current comparison: frames 180–213.

| Metric | Reference | Equal-length preference |
| --- | ---: | ---: |
| Target RMS | 23.732 mm | 23.450 mm |
| Spine length difference RMS | 151.823 mm | 2.907 mm |
| Maximum length difference | 544.259 mm | 10.553 mm |
| Frames at zero length | 2 | 0 |
| Sacrolumbar range | 0–544.259 mm | 202.043–297.308 mm |
| Thoracic range | 0–451.093 mm | 199.775–297.556 mm |
| Ceres solve time | 8.98 s | 28.18 s |
| Termination | CONVERGENCE | CONVERGENCE (136 iterations) |

New problem: 2244 parameter blocks, 6273 residual blocks (+34 residuals).
Native residual-family costs reproduce final cost within 1e-12; attachment
equations agree within 5e-13 mm. The changed objective can lead to a different
local optimum; these results do not establish anatomical ground truth or
global optimality.

The comparison viewer overlays the prior fit in green and the equal-length fit
in yellow. Its table names the equality preference and residual scale. The shared
inspector reports current-frame lengths and their difference. The original solver
lab's experiment 17 includes the extra residual nodes, connections to both scalar
length blocks, native block counts, and the existing length time-series plot.

Validation: 41 relevant Python tests, including eight new native equality tests;
two CTest checks; both viewer checks. Tests cover soft preference strength,
unchanged parameter count, the additional residual count, unconstrained total
length, symmetry, invalid definitions, and reconstruction of residual cost.

The subsequent [full-recording and three-length proportion report](full_recording_fit_report.md)
contains the 222-frame timing and the cervical-flexibility comparison. The numbers
above remain the original 34-frame experiment, not a full-recording benchmark.
