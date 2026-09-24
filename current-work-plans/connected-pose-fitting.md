# Connected spine and clavicle fitting

`core/skeleton/pose/fit_connected_pose.py` fits multiple landmark targets in one
connected skeleton. It reuses the existing forward kinematics for every trial
pose, keeping fitted dimensions, attachment offsets and root transform fixed.

For the first torso integration, select pelvis, sacrolumbar, thoracic and both
clavicles, plus any desired descendants with available rotations. Supply mapped
left/right acromion positions as targets. Allow sacrolumbar, thoracic and the
clavicles to rotate. Other selected parent-relative rotations remain unchanged;
their world transforms follow their parents. Do not add constructed SC points
as independent observations. No SkellyTracker or FreeMoCap import is required.

## Explicit fitting choices

The caller supplies a position tolerance for each target, in the same units as
the pose, and a rotation tolerance in radians for each movable segment. These
are modeling weights, not measured confidence intervals or anatomical limits.
Smaller position tolerance favors closer target agreement; smaller rotation
tolerance favors keeping the reference local rotation. Each angular tolerance
applies equally to all three rotation-vector components for that segment.

Objective (plain text):

    sum(squared_position_error / position_tolerance_squared)
      + sum(squared_rotation_change / rotation_tolerance_squared)

Rotation changes are parent-frame rotation vectors applied to the supplied
reference local quaternions. The solve starts at those references. It jointly
adjusts the available rotations instead of saturating the spine before allowing
clavicles to move. The required positive rotation penalty limits unconstrained
twist and selects a nearby solution when shoulder positions alone are ambiguous.
This is a prior, not proof of a unique physical pose.

The optimizer uses central finite differences and damped least-squares steps
with an acceptance check that the objective decreases. Numerical damping is
separate from model weights and does not introduce temporal lag. The reference
rotation-vector neighborhood excludes magnitude pi or larger; that is a local
parameterization restriction, not an anatomical joint limit. See Samuel Buss's
[IK survey](https://math.ucsd.edu/~sbuss/ResearchWeb/ikmethods/iksurvey.pdf) for the
Jacobian/damped least-squares method underlying the numerical step.

Results include initial/final objective, per-target distances, iteration count,
and termination (`stationary`, `stalled`, or `iteration_limit`). Target tolerance
satisfaction is reported separately: a stationary regularized compromise can
still miss a target. This is a local solve, with no global-optimum guarantee.
Callers must retain/report termination and residuals rather than treating every
returned pose as a successful match. Empty/nonfinite targets are rejected;
unavailable targets may be explicitly omitted. No identity or rest poses are
silently invented.

## Validation and integration

Tests cover an unchanged exact pose, joint fitting of asymmetric shoulders,
preserved root/connections/lengths, unreachable targets, world-coordinate
invariance, omitted targets, invalid evidence and bounded jitter. Test weights
(5 position units and 0.5 radians) are explicit test conditions, not production
defaults. Test agreement is a geometric contract, not anatomical ground truth.

Develop and review this solver within SkellyForge using
`scripts/generate_real_skeleton_viewer.py --serve`. It reads the prepared Parquet
at the standard home-folder location, without importing FreeMoCap or SkellyTracker.
The synthetic viewer remains separate and shares its display helpers with the
real viewer. Model weights are CLI options; termination and residuals are shown
per frame. The real viewer reuses saved processed landmarks and refits fixed
dimensions once with current Forge, retaining the saved scale-voting selection.
It does not rerun tracker mapping. After fitting the upper body, the viewer
recomputes descendant local rotations to preserve their reconstructed world
rotations; their origins still follow the connected skeleton.
Only update FreeMoCap when ready to integrate the reviewed feature into its actual
pipeline. A FreeMoCap dependency refresh is not required to iterate on Forge.

Not yet implemented: anatomical joint limits, variable spine lengths, temporal
regularization, automatic target confidence weighting, or BVH/glTF writers.

## Prepared recording mismatch found during visual review

The current prepared test recording retains the old SC offset: the distance from
neck_center to the mean SC point is exactly 0.1 times shoulder width (median
37.6 mm in the final quarter). Current Tracker mappings use anterior coefficient
0.234362559. Forge's fixed thoracic template places SC attachments approximately
94 mm anterior at this recording's fitted thoracic scale. The shoulder solver
can therefore pull the torso backward to accommodate incompatible geometry.
The clavicle joints correctly connect at sternoclavicular, not neck_center.

Next integration step: in FreeMoCap, inspect and refresh derived landmarks from
saved keypoints with the corrected mapping, then regenerate this Forge viewer.
Reuse saved reconstruction evidence where supported; do not rerun video detection
or calibration unnecessarily. Refitting dimensions alone does not refresh mapped
landmark offsets. Do not compensate by moving the viewer or changing attachments
to neck_center. After refreshing, check the remaining difference between offsets
scaled by shoulder width and offsets scaled by thoracic length; those are distinct
dimension policies and may still need a consistent subject-fit treatment.
