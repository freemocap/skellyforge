# Keypoint trajectory preparation

SkellyForge owns 3D keypoint gap filling. Inputs are one tracked person's
`(frames, keypoints, 3)` array in millimeters and strictly increasing timestamps
in seconds. A missing point has three NaNs. Partial XYZ and infinities are errors.

```python
from skellyforge.core.trajectories import fill_trajectory_gaps

prepared, report = fill_trajectory_gaps(points=raw, timestamps_s=times)
measured_support = report.measured_support(prepared)
for start, stop in report.active_spans:
    # Smooth and reconstruct this interval independently.
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

Keep `report.measured_support(prepared)` alongside the trajectories through
smoothing/alignment. Interpolated samples may guide fitting, but must not count
as independent measured evidence for person dimensions or ground alignment.
`original_support` recovers input availability including discarded fragments.
`to_dict()` supplies a serialization-ready report without adding a Pydantic
dependency to SkellyForge.

## Validation and integration

Run `uv run --no-sync poe test` for deterministic dropout and support regressions.
The connected-fit dropout viewer is preserved on `development-skelly-fit`.

FreeMoCap owns detector mapping, calibration, filtering, recording publication and
end-to-end reference-data runs. Keep gap filling before smoothing and preserve
measured-support masks through alignment and scale estimation. Do not independently
fill the same data twice. Validate both test_data and sample_data after integration.
Calibration-board interpolation is a separate contract from person presence.
