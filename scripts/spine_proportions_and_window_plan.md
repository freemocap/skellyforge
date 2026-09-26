# Next comparisons: spine proportions and overlapping Ceres windows

Updated 2026-09-26. The full-recording equality baseline and a short-window
three-length ratio experiment are now implemented; moving windows remain a plan.
The owner supplied cervical:thoracic:sacrolumbar = 6.3:20:18 for the experiment.
See [the measured results](full_recording_fit_report.md). Production defaults
remain unchanged. Change one factor at a time in the comparison viewer.

## Spine proportions

The preceding residual prefers equal sacrolumbar and thoracic lengths. Its cervical
segment remains rigid. The new experimental replacement makes cervical length variable
and prefers person-appropriate proportions among all three axial lengths.

Use the actual authored connections:

| Segment | Proximal landmark | Distal landmark |
| --- | --- | --- |
| sacrolumbar | pelvis_origin | chest_center |
| thoracic | chest_center | neck_center |
| cervical_spine | neck_center | craniocervical_junction |

The skull remains attached at craniocervical_junction. Head-center keypoints are
not substituted for this landmark. Landmarks and segment definitions remain
complete regardless of keypoint availability.

For fixed target fractions pL, pT, pC, summing to one, and per-frame Ceres scalar
length parameter blocks LL, LT, LC, define S = LL + LT + LC. A possible residual
block is:

    sqrt(time_weight) * [(LL - pL*S)/sigmaL,
                         (LT - pT*S)/sigmaT,
                         (LC - pC*S)/sigmaC]

Only two proportional relationships are independent. This block has rank two,
adds no parameter blocks, and has zero cost at any total length with the target
proportions. It does not fix total length, smooth length over time, or constrain
width. The sigmas are named residual scales in millimeters, not hard bounds or
automatically established measurement uncertainty. Test the block's nullspace
and derivatives, including zero lengths; no division by fitted length is needed.

Use a soft preference rather than a hard fixed ratio: the straight distances
between our landmarks need not shorten proportionally during bending. Estimate
person scale separately, then hold reference proportions fixed during pose
fitting. A ratio prior alone cannot determine total length in frames without
keypoint residuals. Whether to add explicit temporal length residuals is a
separate, visible model choice.

### Evidence and unresolved mapping

[Glaudot et al. (2025)](https://link.springer.com/article/10.1007/s00276-025-03681-1)
report mean vertebral-column regional lengths of 121 mm cervical, 292 mm thoracic,
and 158 mm lumbar in 20 embalmed older adults. Measurements followed the posterior
vertebral-body surface using tape; sacral canal length was excluded. These are
neither straight endpoint distances nor a direct sacrolumbar definition. They
are useful anatomical context, not approved model ratios. Spinal-cord lengths
in the same article are a different quantity and must not be used here.

[Bruno et al. (2015)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5101035/) describe
an OpenSim thoracolumbar model based on CT geometry, published spinal curvature,
and subject scaling. Its coordinate geometry could inform endpoint-compatible
reference distances; it does not directly supply our three-segment ratio.

Before promoting the experimental fractions, trace the current authored geometry's source
and compare published endpoints with pelvis_origin, chest_center, neck_center,
and craniocervical_junction. Do not label an authored or inferred ratio as a
published anthropometric measurement. Report population and variability.

## Moving-window solve

Keep person scale and model reference geometry fixed across the recording.
Use the same per-frame root translation, unit WXYZ quaternion, axial length,
and shoulder displacement parameter blocks. Retain measurement, rest-pose,
chest-line, linkage, proportion, and three-frame temporal residual definitions.

Fit overlapping windows, initializing new frames from existing segment fits and
the overlap from the preceding connected solve. Preserve temporal residuals
across boundaries and the original full-sequence time weights. Do not independently
normalize each window or average quaternion components to hide seams.

The boundary policy must be explicit. A fixed-lag implementation can retain
finalized boundary states as constant parameter blocks and optimize the active
window, but fixing those states discards uncertainty and is not equivalent to a
full-sequence optimum. Marginalization can retain a linearized information prior
from discarded states; this is additional machinery, not automatic Ceres behavior.
[Demmel et al. (2021)](https://arxiv.org/abs/2109.02182) provide relevant prior art
for square-root marginalization, in visual odometry rather than human skeletons.

Choose window duration in seconds, then derive frame count from timestamps.
At 6 Hz, seven samples span one second; the same seven samples span only 0.1 s
at 60 Hz. Three-frame acceleration support does not mean three frames provide
enough evidence during keypoint occlusion.

With fixed window size, stride, and bounded per-window work, total work grows
approximately linearly with recording length. The existing sparse full-sequence
solve is not inherently quadratic: its time coupling is local, and ordering,
factorization fill, derivative evaluation, and iteration count all matter.
Expose Ceres FullReport timing before attributing the current runtime to a cause,
as recommended by the [Ceres performance guide](https://ceres-solver.readthedocs.io/latest/solving_faqs.html).

## Order and checks

1. Completed: publish the unchanged 222-frame equal-length baseline. The additional
   full no-equality control was stopped to prioritize the new ratio experiment.
2. Implemented for review: cervical axial freedom and the soft proportion residual
   using the owner's supplied ratios on the established short movement window.
   Verify endpoint-compatible published attribution before treating the ratios as
   an anthropometric default.
3. Expose detailed Ceres timings; implement an overlapping-window comparison
   using the same objective and explicit boundary policy.
4. Compare runtime, keypoint residuals, length ratios, attachment equations,
   boundary velocity/acceleration, missing-keypoint intervals, and forward versus
   reverse processing. Show the active window and committed frames in the viewer.

Neither a low keypoint residual nor Ceres convergence establishes anatomical
ground truth. Non-convergence means the stopping criterion was not reached; it
does not by itself establish that the posed problem has no solution.
