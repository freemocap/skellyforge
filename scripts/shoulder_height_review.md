# Sternoclavicular height review

September 26, 2026. Saved `windows_3` fit, unchanged geometry. Three active
frames and function tolerance 1e-6 are now the recording experiment defaults;
global refinement remains opt-in. The viewer initially selects only that fit.
Its dense table moves settings into Details, including per-frame neck-center
and SC-midpoint heights relative to the measured shoulder midpoint.

## Findings

The prior is wired in the intended order: sacrolumbar : thoracic : cervical =
18 : 20 : 6.3. It is a soft proportion preference, not a hard ratio or an
absolute length. No swapped segment indices were found.

Heights below are projections along the normalized hip-center to shoulder-center
vector, not world Z. Positive means toward the head. These are comparisons with
keypoint-derived geometry, not independent measurements of internal joints.

| Frame | Cervical length, mm | Neck center above shoulder midpoint, mm | SC midpoint above shoulder midpoint, mm |
| --- | ---: | ---: | ---: |
| 188 | 122.4 | 99.9 | 77.0 |
| 192 | 155.4 | 30.2 | -8.2 |
| 203 | 151.7 | 94.4 | 66.5 |

At frame 188 the SC offset has two projected contributions: the fixed local
forward offset contributes +12.6 mm and the local axial lowering contributes
-35.5 mm. The net -22.9 mm lowers SC relative to the neck center. Most of its
height above the shoulders therefore comes from the neck-center placement.

The current saved skull geometry places head_center at its origin and the two
ear landmarks symmetrically at local Z=0. The cervical endpoint connects to that
origin. At frame 188 it is 209.2 mm above the shoulder midpoint; the cervical
segment supplies only 109.3 mm of that height, leaving the neck center 99.9 mm
above the shoulders. The cervical's saved person-scaled reference length is
206.6 mm, substantially longer than its fitted 122.4 mm. That reference is an
existing scale estimate, not ground truth.

The ratio prior's preferred cervical length at that frame's current total is
approximately 114.1 mm. Thus it favors shortening the fitted cervical there,
not lengthening it. This supports investigating the ratio hypothesis, but does
not prove that it alone caused the selected pose.

There is a second coupling: rigid clavicle lengths are 184.8 and 184.9 mm. At
frame 188 their distal-minus-proximal height components are -75.4 and -88.8 mm;
their perpendicular components are 168.7 and 162.2 mm. Simply lowering SC while
holding everything else fixed would shorten those distances and violate the
rigid lengths. A new fit would have to change other parameters too.

## Next controlled comparison

Follow-up: the owner subsequently requested 20:20:12, 20:20:13 and 20:20:14
instead of the person-reference control proposed below. Those results are in
`spine_ratio_comparison.md`; the proposal below records the preceding review.

Keep the three-frame solve, offsets, clavicle lengths, and all residual strengths
fixed. Compare the current proportion prior with proportions derived from the
existing person-scaled axial reference lengths. This is a diagnostic control,
not a claim that those scale estimates are correct anthropometry. Compare neck
and SC placement, head/shoulder keypoint residuals, and clavicle slopes across
upright, bent and raised-arm frames. Then evaluate clavicle scale/attachment
geometry separately if the discrepancy persists.

No ratio, attachment geometry, segment length, or fitted pose was changed during
this review. Existing fits were republished with additional diagnostic values.
