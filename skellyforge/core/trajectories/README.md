# Keypoint trajectory preparation

SkellyForge owns 3D keypoint gap filling. Inputs are one tracked person's
`(frames, keypoints, 3)` array in millimeters and strictly increasing timestamps
in seconds. A missing point has three NaNs. Partial XYZ and infinities are errors.

```python
from skellyforge.core.trajectories import fill_trajectory_gaps

prepared, report = fill_trajectory_gaps(points=raw, timestamps_s=times)
measured_support = report.measured_support(prepared)
for start, stop in report.active_spans:
    # Smooth and reconstruct this interval, then fit it independently.
    interval = prepared[start:stop]
```

The implementation is extracted from FreeMoCap's version-3 gap filler. It keeps
timestamp-linear interpolation, the 100 ms trajectory-support rule and provenance.
Consecutive observations form a supported run at any frame rate; short gaps can
join support when their bracketing timestamps are at most 100 ms apart. Runs
shorter than 100 ms, including singletons, are discarded. Once supported anchors
exist, interpolation has no maximum interior gap length. It can therefore miss
real motion during a long occlusion; filling is an estimate, not a measurement.

Version 4 adds whole-person absence boundaries. Any all-keypoint-missing frame
splits the input. Support rejection may create further blank frames. Filling
never crosses these boundaries. No endpoint extrapolation is performed, and
never-observed keypoints remain missing. Thus the precise guarantee is **no
interior gaps between supported anchors in a visible interval**, not that every
anatomical keypoint is observed. Skeleton geometry is complete independently.

`active_spans` and `blank_spans` use half-open frame indices on the original grid.
They partition the recording. A single blank frame is a boundary too; the array
alone cannot distinguish detector failure from a person leaving. This API does
not infer identities, detect a person from one point, or join multiple people.
Intervals too short for a caller's sequence solver need an explicit skip or
closed-form reconstruction policy; the three-frame window solver cannot fit them.

Keep `report.measured_support(prepared)` alongside the trajectories through
smoothing/alignment. Interpolated samples may guide fitting, but must not count
as independent measured evidence for person dimensions or ground alignment.
`original_support` recovers input availability including discarded fragments.
`to_dict()` supplies a serialization-ready report without adding a Pydantic
dependency to SkellyForge.

## Inspect the behavior

From the SkellyForge repository:

```powershell
uv run poe experiment-partial-observations --serve
```

Open http://127.0.0.1:8777/. Without `--serve`, the same command only saves the
self-contained page to `.test-artifacts/viewers/partial_observations/index.html`.
It compares raw, prepared and fitted geometry for five deterministic dropout
cases using the production connected window solver. Reference truth never enters
the fit. It is a controlled chain experiment, not the accepted full-human fit.

## FreeMoCap integration handoff

After this SkellyForge change is committed and pushed on `development-streaming`:

1. Refresh FreeMoCap's pinned SkellyForge revision and installed native extension.
2. Replace the algorithm in `core/reconstruction/trajectory_gap_filling.py` with
   a thin adapter to this API. Preserve reading legacy version-2/3 saved reports;
   adapt version-4 dataclass reports into FreeMoCap's existing persisted schema.
3. Keep gap filling before smoothing and preserve measured-support masks for
   alignment and scale. Do not independently fill the same data again.
4. Split sequence fitting at `active_spans`, retain the original frame/time grid
   with blank outputs at absence, and define how intervals shorter than three
   frames are represented. Do not borrow root seeds across separate appearances.
5. Emit all modeled landmarks for present skeletons and keep observation arrays
   separate. Reprocess `test_data`, then `sample_data`, through standard commands.

FreeMoCap's calibration-specific interpolation is separate: this person-presence
contract must not be applied to calibration-board data as an incidental change.
