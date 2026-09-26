# Full-recording baseline and three-length ratio experiment

2026-09-26. SkellyForge lab results, not a change to FreeMoCap's production pipeline.

## Open the results

- [Full recording, frames 0–221](http://127.0.0.1:8773/recording_comparison.html): the
  preceding equal-sacrolumbar/thoracic-length fit, now over all 222 frames.
- [Ratio comparison, frames 180–213](http://127.0.0.1:8773/recording_window_comparison.html):
  purple is the new three-flexible-length fit; gold is the preceding two-length
  equality fit. Solo, overlay, and Details expose both. Details shows actual and
  preferred length percentages per frame.
- [Solver lab](http://127.0.0.1:8773/solver_viewer.html): experiment 19 includes
  the new Ceres residual connections and all three length traces.

The short window was a development interval around shoulder elevation and
bending, not the complete recording. These pages intentionally do not overlay
results from different frame ranges. Refresh the browser to load new artifacts.

## Measured results

| Quantity | Previous equality, 180–213 | Full equality, 0–221 | New proportions, 180–213 |
| --- | ---: | ---: | ---: |
| Frames | 34 | 222 | 34 |
| Ceres solve seconds | 28.18 | 908.34 | 10.91 |
| Read/fit/output-adaptation seconds | not recorded | 1024.32 | 16.01 |
| Ceres iterations reported | 136 | 201 | 48 |
| Termination | CONVERGENCE | NO_CONVERGENCE | CONVERGENCE |
| Mapped-keypoint target RMS, mm | 23.45 | 16.12 | 23.63 |
| Ceres parameter blocks | 2244 | 14652 | 2278 |
| Ceres residual blocks | 6273 | 41368 | 6273 |

The full solve reached the configured 200-iteration limit (the report includes
the initial iteration). This does not mean no solution exists. It is not suitable
as the default fast iteration loop at this runtime. The compiled solver was an
optimized Release build with sparse normal Cholesky, one thread, and function,
gradient, and parameter tolerances of 1e-10. The adapter currently saves Ceres'
BriefReport; detailed derivative/factorization timings still need to be exposed
before attributing the runtime to a particular bottleneck.

Wall times exclude subsequent annotated-preview/context preparation and viewer
publication. These are single-run measurements, not controlled performance
statistics. A few lightweight checks ran during the full fit. An additional
full no-equality control was stopped after the requested ratio experiment became
the priority; no partial control result is displayed.

The lower full-recording target RMS is not proof of an improvement: it covers a
different set of frames and keypoints. All target errors are distances to mapped
keypoint measurements, not errors against anatomical ground truth.

Full equality: sacrolumbar 198.90–512.86 mm; thoracic 198.96–528.20 mm. Their
difference is 1.75 mm RMS, with a 15.34 mm maximum. Neither reaches zero length.
Shoulder displacement is 11.82 mm RMS, 64.42 mm maximum. Attachment-equation
error is below 4.6e-13 mm; relaxed shoulder displacement is explicitly included
in that equation, not mistaken for a disconnected joint.

## What is being solved

Each frame retains the complete 61-segment skeleton. Root XYZ, unit WXYZ segment
quaternions, selected axial lengths, and two parent-local shoulder XYZ
displacements are Ceres parameter blocks. Quaternions use QuaternionManifold.
Other translations follow the connected skeleton equations. Saved person scale,
keypoints, mappings, and reference geometry are not re-estimated during the fit.

Current modeling residual scales:

| Residual | Scale |
| --- | ---: |
| Mapped-keypoint position | 10 mm |
| Root acceleration | 3000 mm/s² |
| Segment quaternion acceleration | 20 rad/s² |
| Relative T-pose quaternion | 2 rad |
| Chest-center lateral/front-back distance to mapped centerline | 50 mm |
| Additional anterior-only chest-center penalty | 20 mm |
| Shoulder linkage displacement from zero | 10 mm |
| Shoulder parent-local displacement acceleration | 300 mm/s² |
| Two-length equality, previous fit | 50 mm |
| Three-length proportion, new fit | 50 mm |

These are residual scales, not hard bounds or independently established
measurement uncertainties. The two 50 mm residual scales act on different
formulas and do not imply identical effective stiffness.

SC attachments retain the earlier fixed lowering of 10% of reference thoracic
extent (26.95 mm for this model). Clavicle lengths and transverse dimensions
remain fixed. Variable axial lengths are nonnegative, with no upper bound,
reference-length penalty, or temporal length penalty. Changing thoracic axial
length also changes its attachment Z positions through the existing axial model.

## New proportionality experiment

The owner supplied cervical : thoracic : sacrolumbar = **6.3 : 20 : 18**.
We normalize by 44.3, giving 14.22% : 45.15% : 40.63% of their combined length.
These are ratios only; we do not force the total to be 44.3% of body height.
No verified Winter/de Leva source for these exact ratios has been established.
The existing de Leva YAML supplies inertia and mass properties, not these lengths.

The cervical segment now has its own scalar length parameter from neck_center
to craniocervical_junction; the skull remains rigid. One residual block per frame
replaces the old equality block. For each of the three lengths:

    residual_i = sqrt(time_weight) * (length_i - fraction_i * total_length) / 50 mm

There are three components with two independent relationships. The block adds
no parameters; enabling cervical deformation adds one parameter per frame.
Total length remains free. This is a soft preference, not exact ratio enforcement.

The fitted ranges are sacrolumbar 217.22–339.32 mm, thoracic 250.99–365.34 mm,
and cervical 27.47–188.69 mm. Average fitted shares are 38.40%, 43.29%, 18.31%
respectively, versus targets 40.63%, 45.15%, 14.22%. Share-of-total length
deviation is 31.42 mm RMS across frames and segments. Thus this initial 50 mm
scale leaves substantial freedom, especially in the cervical segment. It is
not a claim that the requested ratio was maintained exactly.

## Missing measurements and interpretation

The recording spans 0–36.83 seconds at 6 Hz. Frames 216–221 contain no saved
keypoints or root poses. Their initialization uses the nearest saved root pose;
they add no measurement residuals and are explicitly marked in the viewer.
The skeleton and all landmarks remain defined. Temporal and model residuals
support the fitted poses, but there is no temporal length residual to determine
common spine length in this tail. Do not interpret those frames as measured poses.

The earlier calibration-board portion contains occlusion. Assess it alongside
the annotated videos, rather than treating every tracked keypoint as ground truth.
Neither visual plausibility nor Ceres convergence establishes anatomical accuracy.

## Validation and next steps

Python checks cover synthetic recovery, variable total length, ratio
normalization, soft-prior strength, exact deformed attachments, invalid inputs,
residual cost accounting, and missing-keypoint initialization. Native CTest checks
the new AutoDiff Jacobian and its free-total-length nullspace, including zero
length. Viewer checks exercise saved geometry, frame coverage, overlay controls,
playback, unsupported-tail labels, and Ceres block connections.

Next: review the short-window ratio fit against the gold baseline, especially
bending and head movement. Decide the appropriate proportion strength separately
from whether to add temporal length regularization. Then benchmark overlapping
Ceres windows with the same objective and explicit temporal boundary handling.
Do not assume the current sparse full-sequence solve is inherently quadratic.
Keep person scaling outside those windows. See
[the window and anthropometry plan](spine_proportions_and_window_plan.md).

Saved numeric results are overwritten at `build/full_recording/report.json`,
`build/full_recording/equal_spine_lengths.json`, and `build/spine_proportions.json`.
The source Parquet and prepared recording outputs were not modified.
