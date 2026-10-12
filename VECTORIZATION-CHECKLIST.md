# NumPy vectorization checklist

Checked means implemented, parity-tested and measured, or investigated with an
explicit no-change conclusion. It does not mean every loop has been removed.
This pass belongs to SkellyForge; application integration follows its independent
human commit/push boundary. Preserve scientific definitions, missing-data handling,
coordinate conventions, validation, and temporal dependencies.

- [x] Anchored joint/landmark rotations, fitted-dimension checks, scalar rotation.
- [x] Remaining landmark transforms: completion, forward synthesis, COM preparation.
- [x] Parent-relative quaternion calculations — measured, slower; candidate removed.
- [x] Joint-angle conversion grouped by rotation convention.
- [x] Segment directions, lengths, and pose recovery from keypoints (ordinary two-point solver).
- [x] Center-of-mass/inertia reductions and per-segment inertia tensors.
- [x] Scale-fitting reductions and temporal calculations reviewed.
- [x] Final regression/reference checks; ready for human integration handoff.

The first item is documented in VECTORIZATION.md. Its sample synthesis timing was
7.986 to 1.322 seconds. New items use incremental baselines: original functions
with already-improved shared helpers, so savings are not credited twice.

Remaining landmark transforms (sample): completion 1.446 → 1.232 s;
COM landmark preparation 0.444 → 0.218 s; forward synthesis 1.083 → 0.845 s.
Test recording: 0.278 → 0.226 s, 0.085 → 0.043 s, 0.213 → 0.167 s.
All output values matched exactly. Timings include object/array packing, exclude
reference loading and parity assertions. Completion uses frames with a saved root;
forward synthesis uses frames with a complete saved pose. Raw counts and timings
are in `.test-artifacts/vectorization/checklist.json`.

Parent-relative products: packing pairs for NumPy and reconstructing validated
quaternion objects was slower at per-pose batch sizes. Test 0.055 → 0.064 s;
sample 0.366 → 0.415 s. Exact numerical parity, but no speedup: production code
restored unchanged. Candidate saved locally under ignored
`.test-artifacts/vectorization/parent_relative_candidate.py`; measurements in
`relative.json` beside it. Larger cross-frame batching would require a different
caller interface and is not an obvious local loop replacement.

Joint-angle conversion: test 0.330 → 0.144 s; sample 2.175 → 0.767 s.
Reference outputs, offsets, provenance and quaternion values matched exactly.
123 focused tests passed, one existing skip, including all twelve Euler sequences
and exact/near-lock cases. `.test-artifacts/vectorization/angles.json` holds timings.

Pose recovery: test 0.977 → 0.457 s; sample 5.265 → 1.867 s. Exact reference
output parity. Batches lengths, cross/dot products and shortest-arc geometry for
ordinary two-point segments. Quaternion construction, singular cases,
observation-frame precedence, rigid-fit precedence and error paths stay intact.
General ragged SVD/observation-frame batching is not implemented by this item.
Raw results: `.test-artifacts/vectorization/hydration.json`.

COM/inertia: sample whole-body COM 27.8 → 17.8 ms; inertia reduction
185.5 → 37.5 ms. Test: 3.39 → 1.76 ms and 31.86 → 9.86 ms.
Per-segment tensors then batched too: body-inertial-properties calculation with
the preceding reduction improvements already enabled went from 1.186 → 0.549 s
(sample), 0.226 → 0.111 s (test). Exact reference parity throughout. The benchmark
uses a fixed 70 kg illustrative mass, not a measured subject mass. These APIs
are not evidence of a measured application reconstruction speedup. 17 focused
tests passed, including forward-axis fallback, empty and zero-length cases.
Raw files: `reductions.json` and `inertia.json` in the ignored vectorization folder.

Scale summaries now group equal-length sample arrays and compute medians/MADs
along an axis, without padding missing evidence. For **100 repeated fits** on
recording-derived samples: test 0.311 → 0.159 s; sample 0.631 → 0.432 s.
That is 3.11 → 1.59 ms and 6.31 → 4.32 ms per fit, not hundreds of milliseconds
saved on one recording. Exact output parity; 18 focused tests pass.
`.test-artifacts/vectorization/scale.json` contains the raw measurements.

Temporal review: velocity/acceleration differences and angular-velocity math
already operate on arrays. Roll propagation depends on prior frames/parents;
its ordering is retained. Ground-contact window selection uses irregular
timestamps and stateful episode boundaries; wholesale batching would require
a separately tested algorithm, not a direct independent-loop replacement.
Ragged rigid-fit SVD groups and observation-frame solvers are further research
candidates, not claimed as implemented. GPU batching is outside this pass.

## Final validation

- Full suite: 678 passed, 1 skipped, 1 pre-existing forearm-pronation failure.
  The failure remains visible and is documented in TESTING.md/VECTORIZATION.md.
- Reference tests: 222 frames (216 hydrated, 6 absent) and 1,108 frames
  (1,079 hydrated, 29 absent), both passed; source hashes unchanged.
- All reference before/after comparisons above had zero numeric difference.
  Synthetic tests additionally cover all Euler conventions/locks, direction
  singularities, partial poses, invalid scales and inertia fallback/empty cases.
- `git diff --check` passed. Generated reports and rejected candidate source are
  ignored under `.test-artifacts/`; no generated artifacts are tracked.
- No Git state changes, dependency refreshes or FreeMoCap source edits occurred.

Reproduce individual measurements from the repository with its existing Python:

```powershell
.venv/Scripts/python.exe -B scripts/benchmark_vectorization_checklist.py --item landmarks
.venv/Scripts/python.exe -B scripts/benchmark_vectorization_checklist.py --item angles
.venv/Scripts/python.exe -B scripts/benchmark_vectorization_checklist.py --item hydration
.venv/Scripts/python.exe -B scripts/benchmark_vectorization_checklist.py --item reductions
.venv/Scripts/python.exe -B scripts/benchmark_vectorization_checklist.py --item inertia
.venv/Scripts/python.exe -B scripts/benchmark_vectorization_checklist.py --item scale
```

`--item relative` now compares the retained original implementation to itself;
its rejected experiment is preserved only in the ignored local candidate file.
Timings are warmed medians of three alternating before/after passes. Reference
loading, benchmark setup and output comparisons are outside the measured region;
packing arrays inside the production function is included. Independent operations
must not be summed into a claimed full application speedup.

## Cross-frame follow-up — 2026-10-12

This follow-up is separate from the completed per-pose optimization pass above.
The following checks mean **SkellyForge-owned work is complete**. They do not
claim application integration or completion of FreeMoCap-owned serialization.

- [x] Skeleton reconstruction: frame-batched pose recovery, connected positions,
  joint rotations/angles, COM, residuals, and array input for recording scale fits.
- [x] Preparing results for saving: numeric batch outputs with explicit named
  axes, presence masks and solver provenance; consumer serialization remains in core.
- [x] Skeleton setup: reusable geometry/index/convention/measurement arrays,
  prepared outside the frame calculations.
- [x] Parquet writing: no production writer belongs to Forge. Vectorize its
  reference-data reader; retain application writer work for core integration.
- [x] CSV exports: ownership audit complete; no CSV exporter exists in Forge.

`core/skeleton/chain/anchored_batch.py` adds `AnchoredSkeletonGeometry` and
`ConnectedPositionBatch`. Prepare fixed geometry with `from_skeleton`, then pass
`positions` world wxyz arrays `(frames, segments, 4)`, a boolean presence mask,
and `(frames, 3)` anchor origins. Segment/landmark names are explicit ordered axes
on the prepared geometry. Outputs preserve that ordering and the input world
frame/units. Missing values remain absent; disconnected selections are rejected.
The model and fit are copied into read-only numeric geometry, not retained by reference.

The calculation batches both frames and landmarks. A loop over tree depths remains
because child positions depend on parents; each depth operates over all frames and
branches together. The caller supplies bounded chunks (128 in the benchmark).
No temporal orientation recovery, filtering, scale fitting, or provenance is changed.
Callers must retain their existing orientation provenance alongside these positions.

`scripts/benchmark_anchored_batch.py` compares the current optimized per-frame API
against this array API using the existing prepared reference recordings. Three
alternating measured passes after warmup, with no concurrent test suite:

| Recording | Motion frames | Position calculation before | After | Speedup |
| --- | ---: | ---: | ---: | ---: |
| Test | 222 | 0.2850 s | 0.0196 s | 14.55x |
| Sample | 1,108 | 1.5223 s | 0.0987 s | 15.43x |

This includes one-time geometry preparation and conversion of existing per-frame
objects into batch inputs. It excludes reference reading, parity assertions, and
rebuilding legacy output objects: the intended consumer uses the returned arrays.
This is **only connected positions**, not full reconstruction or application timing.
All 119,322 test and 593,658 sample position values matched exactly; source hashes
were unchanged. Raw repeats are in ignored `.test-artifacts/anchored-batch.json`.

Initial focused tests: 20 passed, including scalar comparisons at four anchors, varying
connected subsets, empty frames, invalid inputs, and chunk-boundary equivalence.
Full suite before the final additional invalid-input test: 684 passed, 1 skipped,
the same previously documented forearm-pronation failure. No new failures.
FreeMoCap has not been edited or refreshed and does not yet call this API.

### Remaining batch interfaces implemented in this pass

| Interface | Responsibility |
| --- | --- |
| `LandmarkPoseGeometry.from_skeleton(...).recover(positions, landmark_present)` | `(F,L,3)` landmarks to `(F,S,3)` origins, `(F,S,4)` wxyz rotations, scales, status and solver provenance. Observation frames retain precedence; rigid fits group equal visible-landmark selections and batch SVD; two-point directions batch their geometry and quaternion conversion. |
| `JointBatchGeometry.from_skeleton(...).calculate(world_orientations, segment_present, segment_solutions)` | Parent/child quaternion products and Euler conversion across frames and joints, grouped by convention. Input solver provenance survives on both sides of each joint. |
| `CenterOfMassGeometry.from_definitions(...).calculate(positions, landmark_present)` | Segment/body centers of mass, retaining visible-landmark/mass renormalization and accumulation order. |
| `ResidualBatchGeometry.from_skeleton(...).calculate(scales, segment_present)` | Measured lengths and residuals over the complete frame/segment axes. Prepared reference lengths and reference kind remain explicit. |
| `fit_model_scale_arrays(...)` | Direct array evidence for a whole-recording scale fit, preserving first-observation segment ordering and excluding missing evidence. Not a live rolling-window replacement. |

Geometry axes must be used in their declared order. Callers keep the recording,
frame-number, time, coordinate-frame and unit identities alongside the arrays.
None of these operations changes axes/units, imputes observations, or authorizes
publication of incomplete scientific results. LandmarkPoseGeometry retains model
definitions: do not mutate them during a run. Other prepared numeric geometry
uses read-only snapshots. Output arrays are owned by the caller.

Latest isolated component measurements, seconds for the complete reference take:

| Operation | Test before → after | Sample before → after |
| --- | ---: | ---: |
| Pose recovery | 0.291351 → 0.034354 | 1.686404 → 0.153892 |
| Joint rotations and angles | 0.134040 → 0.022072 | 0.709013 → 0.107600 |
| Center of mass | 0.012953 → 0.005895 | 0.073404 → 0.034774 |
| Measurement residuals | 0.024557 → 0.002557 | 0.166730 → 0.011380 |
| Scale fit, including conversion from old pose objects | 0.002073 → 0.003361 | 0.012960 → 0.015638 |
| Scale fit, arrays already available | 0.002402 → 0.001252 | 0.014966 → 0.003132 |
| Full reference-recording read | 2.474869 → 1.008863 | 12.428937 → 5.668142 |

**Do not replace the old scale fitter with an object-to-array detour:** that is
slower. Its new interface serves the arrays already produced by batched recovery.
The two scale rows have different input preparation boundaries, explicitly named
in the raw report. Standalone COM measurements use illustrative fixed masses;
tests additionally exercise unequal segment masses and varying visibility.

Other component timers include prepared geometry and input object-to-array
packing; exclude reference loading, output comparison and legacy object rebuilding.
Times are medians of three alternating measured passes after warmup. They are
not end-to-end application measurements and must not be added to claim a pipeline
speedup. Outputs from new APIs are arrays rather than legacy per-frame objects.
Maximum reference differences: recovered origins/scales exact, recovered
quaternions 1.11e-16, joint quaternions/angles 8.88e-16; COM/residuals/fits exact.
Solver provenance and presence match. Reference file hashes remain unchanged.

Raw reports in ignored `.test-artifacts/`: `hydration-batch.json`,
`joint-batch.json`, `reconstruction-reductions.json`, `recording-reader.json`.
Reproduce with the corresponding `scripts/benchmark_hydration_batch.py`,
`benchmark_joint_batch.py`, `benchmark_reconstruction_reductions.py`, and
`benchmark_recording_reader.py`. Reader baseline is Git revision
`0ede33e0a6e323dcd4f4b9de2e1b4fc9eff5e41d`; pass `--baseline` to retain it
after committing. No benchmark imports another repository or modifies input data.

### Parquet reader scope

`tools/recording_data.py` now filters requested channels with Arrow before Python
conversion, then checks duplicates, timestamps, units and missingness using
column arrays. Final per-frame dictionaries remain solely for the existing
standalone viewer API. Whole-take viewer output remains resident in memory; it
is not the bounded production pipeline transport. The producer-side batch math
uses caller-supplied chunks (128 in the comparisons). There is no Forge Parquet
writer or CSV exporter to replace; those are owned by FreeMoCap.

### Final validation and integration boundary

Final full suite: **702 passed, 1 skipped, 1 existing forearm-pronation failure**
in 40.41 seconds. Rigid-marker batch tests cover subsets, collinear observations,
direction fallback, empty frames, and no-joint skeletons. The reader additionally
covers empty/all-null input. No tests were weakened or skipped by this work.
Coverage also includes four position anchors, chunk boundaries, random partial
poses, all Euler conventions, quaternion branch cases, SVD reflection correction,
fit evidence ordering, residual availability, COM missingness, and reader ambiguity.

FreeMoCap integration remains a separate change:

1. Keep measured-only fitting inputs separate from filtered/interpolated motion.
   Recover their scales in chunks, preserve required fit identity, then fit once.
2. Recover motion poses in chunks. Carry existing roll state sequentially across
   chunk boundaries, including the existing reset behavior on empty inputs.
   Do not replace roll recovery or hide the known pronation test failure.
3. Feed resolved orientation arrays to connected-position and joint calculations.
   Compute each frame's anchor-connected selection; do not silently bridge gaps.
4. Carry numeric outputs directly into core's channel-array/Arrow serializers.
   Retain provenance, frame IDs, timestamps, units, and coordinate identity.
5. Run full application/reference stage, export, cancellation, completeness and
   performance comparisons. Core still owns all final save/export validation.

Remaining execution loops are over bounded chunks, model segments, tree depths,
Euler conventions or distinct rigid-fit observation subsets. The scientific
roll recurrence remains frame-ordered. Public legacy object APIs and the viewer
still require object adaptation; they have not been falsely advertised as removed.

The next required action is the human commit/push of **SkellyForge**, branch
`development-streaming`. Only after confirmation may FreeMoCap refresh this
dependency and run the complete app/pipeline comparisons. Those integration
measurements are not yet available.
