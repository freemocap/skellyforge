# Testing handoff: secondary exports

2026-09-23. Owner narrowed the testing effort to support secondary exports.
The proposal is in FreeMoCap's `current-work-plans/02-pipeline/secondary-exports.md`;
its format choices remain proposals. No exporters are implemented by this chunk.

## Workspace checkpoint

- SkellyCam is accepted as sufficient for this phase. Its latest focused run had
  40 passing tests covering capture, recording lifecycle, playback, and save-first
  shutdown. The lifecycle changes were still uncommitted at the last inspection.
  Deferred: combined multiprocess camera replay, full UI/build closeout, remaining
  test-cruft audit, physical-camera endurance, audio, and timed replay.
- SkellyTracker and utilities testing work is deferred. Blender add-on remains out
  of scope. These are intentional deferrals, not claims of complete coverage.
- Core already has preparation/reuse of the 222-frame recording, fresh calibration
  and posthoc processing, four prepared-data consumer checks, and two realtime
  replay checks. These were not rerun for this handoff. Sample preparation through
  that helper, process-mode integration, numerical accuracy findings, and legacy
  fixture cleanup remain deferred.

## Small first stage in SkellyForge

`test_animation_math_contract.py` exercises existing production APIs:

- Parent-relative rotations reconstruct a branching hierarchy with asymmetric,
  noncommuting frames, out-of-order storage, opposite quaternion signs, and angles
  around 180 degrees. Missing parents do not produce fabricated local rotations.
- Coordinate conversion preserves posed attachment geometry and lengths, including
  handedness changes. This checks basis conversion, not millimeter-to-meter policy.
- Rotation resampling respects irregular timestamps and a nonzero time origin,
  takes the short path across 180 degrees, and leaves source arrays unchanged.

These tests use analytical matrix/constant-speed answers without inference or data
downloads. Existing Euler, rest-pose, coordinate-frame, scale-fit, and synthetic
hydration tests remain valuable; retain them rather than duplicating them wholesale.

Validation on 2026-09-23: all 12 new cases passed. The full installed SkellyForge
suite passed 573 tests with one existing skip: the shipped model has no adjacent
rigid-fit pairs for that chain-closure check. No production code was changed.

Run from this repository using the installed environment:

```powershell
.\.venv\Scripts\python.exe -B -m pytest skellyforge/tests/test_animation_math_contract.py
```

## Connected fitted poses (2026-09-24)

`core/skeleton/chain/synthesis.py::synthesize_fitted_pose` now computes one
connected pose from the existing skeleton definition, a `ModelScaleFit`, explicit
root position/world rotation, and parent-relative rotations keyed by non-root
segment name. It returns world orientations, origins, and landmark positions.
The existing module names/layout are retained; reorganization is deferred.

Each child attaches to its parent's scaled connection landmark. Each landmark
uses its owning segment's scale. Use the same fit and selection across frames to
keep dimensions and attachments fixed. Units remain those of the fit and root
position; the function performs no conversion or resampling. Rotations already
include the rest orientation, which must not be applied a second time.

The optional `segment_names` selection must contain the root and every ancestor
of every selected segment. This permits body-only motion without fabricating
finger rotations. All selected non-root rotations must be supplied, and the fit
must cover the full definition with lengths consistent with its geometry.

`test_fitted_synthesis.py` checks changing root motion and noncommuting rotations,
unequal scales, parent-owned attachments, unchanged template geometry, explicit
missing-motion rejection, and agreement with a uniformly scaled rest pose.
These are exact mathematical checks. Agreement with noisy measured origins is
not an exactness requirement: this calculation connects independently estimated
rotations; it does not optimize an armature against measured trajectories.

Run the focused checks from this repository:

```powershell
.\.venv\Scripts\python.exe -B -m pytest skellyforge/tests/test_fitted_synthesis.py
```

Validation: full SkellyForge suite passed 576 tests with one existing chain-closure
skip on 2026-09-24. No real recording or file-format interoperability is claimed
by this validation.

For visual review, regenerate `scripts/skeleton_viewer.html`; instructions are in
`scripts/README.md`. The updated viewer overlays connected poses on independent
segment poses and synthetic source motion, with frame scrubbing and origin
displacement plots. Generation was checked with zero noise, default noise, and
5 mm noise with unequal scales; all 60 frames in each case contained finite data.
JavaScript syntax checks passed. Interactive rendering was not verified in the
agent session because no browser connection was available.

The viewer now also supports `--serve` for independent root/shoulder/elbow/head
motion controls and optional deterministic landmark noise, starting still with
noise off. Main-body orientation arrows can be placed on independent or connected
origins. Diagnostics verified the still baseline, each motion switch, fixed root
with root motion disabled, deterministic noise, live HTTP recomputation, invalid
request rejection, and JavaScript syntax. Browser rendering remains unverified.

### Roll resolution

The resolver uses parent-relative minimum-swing transport with explicit rest
orientations and the declared terminal twist pass. There is no bend-plane anchor
or angular-threshold workaround. The viewer and core use the same rest/backfill
configuration. See `ROLL_STABILITY_DESIGN.md` for the calculation, scope and
validation. The viewer's first plot selects all major body segments; restart its
Python server and verify the loaded-code hash before review.

## Remaining integration and export stages

After the human commits and pushes SkellyForge, core can consume this API with
the prepared recording's definition, frozen fit, root motion, and local rotations.
Select an explicit usable interval and ancestor-complete set of segments; report
excluded data. Preserve measured poses separately and characterize connected-pose
differences rather than treating zero measurement residual as a pass criterion.
No dataset downloads, processing, Parquet changes, or exporters belong to this
SkellyForge chunk.

1. Extend core's existing prepared-Parquet hierarchy diagnostic to exercise the
   fitted-pose API. Measure time-varying parent-relative translations and
   fixed-offset forward-kinematics differences, including branching attachments.
   Do not assume segment lengths define every offset or that a conventional BVH
   hierarchy reproduces saved positions. Reuse processed test data; do not rerun
   tracking to test an export consumer.
2. Once motion preparation has a concrete API, test full ancestor validity, gap
   policy, translation resampling, quaternion sign continuity, and unit conversion.
   The existing resampler clamps out-of-range times and interpolates across spans;
   it is not an export gap-policy implementation. This stage does not assert raw
   quaternion component continuity, only correct represented rotations.
3. Keep Parquet selection, CSV/NPY/NPZ round trips, timestamps, names, validity,
   revisions, and artifact publication tests in core. SkellyForge receives numeric
   inputs and existing scientific types, with no core/Tracker imports.
4. When writers exist, independently validate/import output and compare world
   positions/orientations. Test rest orientation/alignment applied exactly once,
   moving roots, fixed-offset approximation errors, and missing-data behavior.

Passing these stages establishes mathematical prerequisites, not glTF/BVH
interchange correctness or real-recording playback parity. No format defaults
or new dependency is introduced.
