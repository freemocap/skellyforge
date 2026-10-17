# Roll resolution contract

The supported resolver uses minimum-swing transport in the moving parent's
frame. It never constructs a roll reference from parent/child origin differences
or a bend-plane normal. Authored rest rotations are required explicitly.

## Calculation

For the first frame, use the resolved parent orientation composed with the
child's authored local rest orientation. For subsequent consecutive observations,
carry the child's previous orientation with the parent's change in rotation.
Apply the shortest swing from that reference's primary axis to the observed
primary direction. Rigid-fit orientations remain unchanged. Apply the existing
model-declared terminal twist backfill, then carry the final orientations.

Twist backfill projects in the parent's local frame. Its relative change is
`q_current_relative * inverse(q_rest_relative)`, not the reversed product, which
expresses the change in the child's rest frame. A nonidentity-rest test checks
this distinction, and the noisy-body checks include both clavicles.

If a parent is missing, transport in world space. On reacquisition, require two
consecutive parent observations before inheriting its rotational increment.
Missing segments are omitted and initialize from rest on reacquisition. Reset
between takes and process chronologically. An exact opposite-direction step has
no unique swing axis; use the reference's transverse axis for that half-turn.

There are no bend-angle tuning thresholds and no legacy mode. Segment-level
resolution uses the same calculation without an observed parent. Rest rotations
must cover the full skeleton; malformed input fails instead of assuming identity.

## Meaning and limits

The measured long-axis direction is preserved. Axial rotation supplied by transport
is a convention, with path dependence/geometric phase; it is not a claim of
measured anatomical twist. No smoothing or velocity estimation is performed.
Sparse sampling, an ambiguous initial orientation, occlusions, and badly estimated
rigid-body orientations remain limitations. Terminal twist attribution is an
explicit model assumption and is tested with hand pronation and wrist flexion.
This is not a complete constrained anatomical IK solver.

Blender's [Damped Track](https://docs.blender.org/manual/en/latest/animation/constraints/tracking/damped_track.html)
uses a minimal swing from the incoming owner orientation. This resolver uses
that mathematical operation, with an explicitly temporal parent-relative reference.
It does not claim equivalence to an entire Blender rig or constraint stack.
The add-on also applies joint limits and independent roll references, which must
be specified in Forge's own anatomical frames before being implemented here.
[OpenSim IK](https://opensimconfluence.atlassian.net/wiki/spaces/OpenSim/pages/53090032/Getting%20Started%20with%20Inverse%20Kinematics)
provides a reference for the later task of fitting model coordinates to weighted
observations under the chosen model.

## Validation and review

`test_roll_resolution.py` exercises static landmark jitter across the major body
segments at 1 and 3 mm, two seeds, 300 frames per case, checking step sizes,
excursions from the initial orientation, and preservation of primary directions.
It also exercises full parent rotation, straight-knee crossing, overhead direction
transport, reset, missing parents and dictionary ordering. Existing terminal-twist
checks require recovery of hand pronation without mistaking wrist flexion for roll.

The viewer uses the same rest orientations and terminal-backfill configuration as
production core. The first plot selects any major body segment's inter-frame
orientation change. Use all motion off and noise on first; then test actual motion.
Restart the Python server to load changes and verify its loaded-code hash.
Real-recording validation and full anatomical constraints remain subsequent work.

Validation on 2026-09-24: full suite 581 passed, one existing skip. The viewer's
60-frame static sequence at 1 mm noise has a maximum step of 4.56 degrees across
the 13 plotted body segments (sacrolumbar 1.12, knees 0.88/0.89). Zero-noise
static motion reports zero steps at displayed precision. These finite synthetic
checks do not establish anatomical accuracy or stability for arbitrary noise.
