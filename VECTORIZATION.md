# Reconstruction math measurements — 2026-10-12

This report records the initial optimization slice. Subsequent implementations,
measurements, rejected candidates and current completion state are recorded in
[VECTORIZATION-CHECKLIST.md](VECTORIZATION-CHECKLIST.md).

Joint-attachment and landmark position calculation is about six times faster on
the existing prepared reference recordings. These are isolated anchored-synthesis
measurements, not full reconstruction or application pipeline timings.

| Recording | Motion frames | Before | After | Saved | Before ms/frame | After ms/frame |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Test | 222 | 1.571 s | 0.259 s | 1.312 s | 7.07 | 1.16 |
| Sample | 1,108 | 7.845 s | 1.306 s | 6.539 s | 7.08 | 1.18 |

Frames mean motion-capture frames, not camera images. Empty frames remain empty.
The test recording has 216 saved connected poses and the sample has 1,076.
All **1,026,188 compared numeric values were exactly equal**: orientations,
segment origins, and landmark positions. Output keys also matched. Both original
Parquet SHA256 hashes remained unchanged.

## What changed

This is a first implementation slice, not completion of every optimization in
the audit. The original headline combined the changes inside anchored synthesis;
it did not measure each change independently. Also, orientations in that test
were supplied inputs, not newly estimated rotations. The separate breakdown
below includes recomputing measured segment orientations from saved keypoints.

- Joint attachments: pack matching rotations and offsets into arrays and rotate
  all attachments together, instead of invoking quaternion rotation per joint.
- Landmark positions: rotate all independent local offsets together, then add
  segment origins as one array operation.
- Fitted geometry validation: compare all scales and lengths together with the
  original tolerances. Invalid input still raises; validation was not removed.
- Single-vector rotation elsewhere: expand the fixed three-component cross
  products directly instead of making two generic `np.cross` calls. This is a
  scalar specialization, not batching; it preserves the original arithmetic.

`rotate_vector_pairs` is O(N), returning one vector per quaternion. The existing
`rotate_vectors_batch` has different semantics: every quaternion against every
vector, returning an N-by-M array. That API is unchanged.

The graph walk still follows parent/child dependencies. Frame ordering and roll
history are unchanged. Packing named objects and constructing validated output
objects still use loops; the independent rotation math is batched. Euler angles,
direction-based hydration and scale summaries were subsequent checklist items.

## Reproduce

From the SkellyForge repository with its existing environment and prepared data:

```powershell
.venv/Scripts/python.exe -B scripts/benchmark_synthesis.py --baseline 0ede33e0a6e323dcd4f4b9de2e1b4fc9eff5e41d
.venv/Scripts/python.exe -B scripts/benchmark_math_breakdown.py
.venv/Scripts/python.exe -B -m pytest
.venv/Scripts/python.exe -B -m pytest tests/reference --dataset test_data -s
.venv/Scripts/python.exe -B -m pytest tests/reference --dataset sample_data -s
```

The benchmark uses `recording_path` and `read_recording`, the same reference-data
loader as the repository's tests. It reads baseline source with `git show` without
changing the checkout. It applies runtime type checking to baseline functions,
temporarily substitutes the old rotation method only inside the benchmark, and
restores it in `finally`. All other code and prepared inputs are shared.

Saved world rotations and fitted dimensions are inputs to this comparison;
hydration, roll resolution, fitting, I/O, and application scheduling are outside
the timed region. Preparation and output comparison are also outside that region.
One warm-up pair precedes three alternating before/after measurements. No other
tests or benchmarks were launched alongside the reported run. Medians are shown.
Sample baseline timings were 7.749, 7.845, 11.419 seconds; candidate timings were
1.302, 1.321, 1.306 seconds. The baseline outlier is retained, not discarded.
Raw local results: `.test-artifacts/vectorization/measurements.json` (ignored).

Input hashes:

- Test: `0d1930e726ce8a0503714c1afab04ee7e1efb9bfdcd96aa62f99601020fb1402`
- Sample: `0bd8a0bbf34928d19966ea77f7fc67af98cd919d1039ddc548710856bfad75e0`

## Separate measurements of each change

`scripts/benchmark_math_breakdown.py` records one warm-up and three measured
passes per case. Raw results are in ignored
`.test-artifacts/vectorization/breakdown.json`. No production code was changed
for this follow-up measurement.

For **10,000 independent vector rotations**, with the same inputs and exact
output agreement:

| Implementation | Median time |
| --- | ---: |
| Original scalar rotation loop | 320.408 ms |
| Optimized scalar rotation loop | 23.535 ms |
| Paired NumPy rotation array | 0.635 ms |

Array packing and quaternion construction are outside this microbenchmark.
The array result is not a prediction of whole-recording speedup.

To attribute changes inside position calculation, enable them successively:

| Enabled changes | Test, 222 frames | Sample, 1,108 frames |
| --- | ---: | ---: |
| Original code | 2.416 s | 7.986 s |
| Faster single-vector rotation only | 0.639 s | 2.184 s |
| Also array-based fitted-dimension validation | 0.372 s | 1.529 s |
| Also batched joint/landmark rotations | 0.305 s | 1.322 s |

These are cumulative configurations, not independent savings to add together.
Sample marginal savings are 5.802 s, 0.655 s, and 0.207 s in that order.
The test baseline varied from 1.685 to 2.836 s in this run; its absolute timing
is noisier than the sample, whose baseline ranged from 7.780 to 8.210 s.
Every configuration retained identical keys and zero numeric difference.

Additional isolated operations:

| Operation | Test before → after | Sample before → after |
| --- | ---: | ---: |
| Validate all fitted dimensions once per frame | 0.156 → 0.025 s | 0.729 → 0.115 s |
| Recompute measured segment positions/orientations/scales from mapped keypoints | 0.948 → 0.713 s | 3.291 → 3.366 s |

The latter is `hydrate_skeleton`, not full reconstruction. Its sample result
shows **no speedup** (2.3% slower in this measurement); do not claim a general
hydration improvement. Its 627,984 output numbers matched exactly, including
newly calculated orientations. Each case was warmed, then measured three times;
hydration and isolated microbenchmarks ran before then after, unlike the
alternating cumulative synthesis configurations. These observations do not
establish that a small timing difference is a reproducible regression.

At this initial measurement, Euler/joint-angle extraction and scale summaries
were still unchanged; both have since been addressed in the checklist.
Temporal roll resolution, tracking, triangulation, exports and application
scheduling remain unchanged. The shared scalar helper can affect callers, but their end-to-end
timings are not established by these measurements. No new whole-app baseline
exists until the updated dependency is integrated.

## Validation and integration boundary

Both reference hydration tests pass (222 and 1,108 frames). Randomized paired
and scalar rotation tests match the original cross-product formula exactly.
Existing synthesis tests cover reversed traversal, partial connected components,
isolated anchors, missing connections, fitted attachments, and landmark ownership.
Additional tests reject zero, negative, infinite, and NaN fitted scales.

The full suite reports **657 passed, 1 skipped, 1 failed** and retains the previously documented failure in
`test_twist_backfill::test_pure_pronation_is_recovered_during_resolution`:
0 degrees recovered where the fixture expects 60. It was reproduced with the
original scalar rotation and original anchored-synthesis/validation functions.
It was not skipped or weakened. This change does not solve that scientific issue.

The human must commit/push SkellyForge on `development-streaming` before core
refreshes its dependency. FreeMoCap's environment and lockfile have not been
changed for this work. Full application timings and output comparisons belong
to that subsequent integration step; do not interpret this component speedup as
a measured reduction in the whole application's runtime.
