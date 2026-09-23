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

## Next stages, deliberately not implemented here

1. In core, read one prepared Parquet snapshot and its saved definitions/fit. Measure
   time-varying parent-relative translations and fixed-offset forward-kinematics
   error, including branching attachments. Do not assume segment lengths define
   every offset or that a conventional BVH hierarchy reproduces saved positions.
   Reuse processed test data; do not rerun tracking to test an export consumer.
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

Passing this first stage establishes mathematical prerequisites, not glTF/BVH
interchange correctness or real-recording playback parity. No format defaults,
animation contract, or new dependency is introduced.
