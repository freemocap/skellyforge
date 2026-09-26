# Sequential Ceres processing comparison

The model and residual strengths are unchanged: three variable spine lengths,
18:20:6.3 sacrolumbar/thoracic/cervical proportion preference, chest-center line
preference, lowered SC attachments, and relaxed shoulder linkages. These are
experimental fits of prepared real data, not production pipeline output.

## Run and inspect

From SkellyForge, after rebuilding the extension with `uv run --no-sync poe native-install`:

```powershell
uv run --no-sync poe solver-viewer-processing
```

This now defaults to the accepted three-active-frame pass, without global
refinement, at function tolerance 1e-6. For the original four-way comparison,
add `--windows 3 5 7 --refine`.
`--start 180 --end 213` selects the established short movement interval.
The low-level native function tolerance remains 1e-10; only this recording
experiment's default is 1e-6, matching the reviewed result.
Maximum iterations remain 200; gradient and parameter tolerances remain 1e-10.
Add `--publish-only` to rebuild the viewer from those saved results without
running Ceres again. Requested cached variants must exist.

Open `http://127.0.0.1:8773/recording_processing_comparison.html` when the existing
scripts viewer server is running. The page is updated after each completed fit.
The table supports overlays and isolation. Details identify the window that
finalized the selected frame, its Ceres report, total window time, and optional
global-refinement time. The timeline band shows fixed history in amber and
active frames in cyan. Playback shows final fitted poses, not intermediate
optimizer iterations.

Artifacts overwrite stable filenames under `build/processing_comparison/full/`
or `movement_window/`; they do not accumulate per-run directories. The source
Parquet and prepared recording are read-only. Annotated video uses the existing
frame-image cache. Previous viewer pages remain available separately.

## What the Ceres problems contain

Each active frame has 67 parameter blocks: one root XYZ, 61 WXYZ unit quaternion
blocks with QuaternionManifold, three axial lengths, and two shoulder XYZ
displacements. Segment widths and other attachment geometry remain fixed.

For a three-frame pass, the first solve adjusts frames 0–2 and finalizes frame 0.
The next adjusts 1–3 with frame 0 constant. Subsequent problems retain two
constant history frames: e.g. frames 0–1 constant while 2–4 remain adjustable.
`SetParameterBlockConstant` fixes every parameter type in that prefix, allowing
the existing three-frame acceleration residuals to cross the boundary. The
last window finalizes its remaining frames. Every frame is published once.

Overlapping poses retain the preceding solution. A newly introduced frame copies
the preceding fitted root, all quaternions, lengths and shoulder displacements.
The first window uses saved segment/root initialization. There is no averaging,
blending, quaternion-component interpolation, or post-fit smoothing.

Full-recording trapezoid weights are sliced into each window so the measurement
and per-frame prior weights do not change at window endpoints. Window costs
overlap and must not be summed. A final Ceres evaluation, with no optimization,
scores the assembled trajectory under the complete recording objective. Optional
refinement initializes all parameters from that trajectory, releases the fixed
prefix, and solves the whole recording once.

This is a fixed-history approximation, not marginalization. Earlier decisions
cannot be revised by later windows. Global refinement can revise them. Neither
convergence nor a low mapped-keypoint residual establishes anatomical truth.

## Measurements, September 26, 2026

Prepared recording: 222 frames, indices 0–221, 6 Hz, 36.833 seconds; 61 segments.
The final six frames have no mapped keypoint targets. Their model landmarks
remain defined, but fitted movement there is supported by temporal/model
residuals rather than measurements. No keypoints are fabricated.

Full-recording results at function tolerance 1e-6 follow. Native times measure
Ceres Solve, separately from Python adaptation and file writing.

| Processing | Window seconds | Refinement seconds | Total Ceres seconds | Target RMS, mm | Full objective | Termination |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 3 active | 83.925 | — | 83.925 | 15.951 | 3440.297 | 220/220 windows converged |
| 5 active | 167.555 | — | 167.555 | 15.955 | 3377.550 | 216/218 converged |
| 7 active | 230.622 | — | 230.622 | 15.969 | 3368.347 | 215/216 converged |
| 3 active + full refinement | 83.925 | 323.175 | 407.100 | 15.956 | 3361.360 | Refinement converged, 82 reported iterations |

Five-frame windows 189–193 and 209–213 reached the iteration limit. Results
remain usable but did not satisfy a convergence criterion. Processing wall
times excluding the viewer adapter were 85.663, 170.224, 233.475 and 409.161
seconds respectively. Three-frame throughput is approximately 2.65 frames/sec,
below this recording's 6 Hz rate: this is not real-time performance.

Global refinement reduced the total objective by 2.29%, with similar keypoint
RMS. Ceres spent 277.643 seconds in linear solving and 43.875 seconds evaluating
Jacobians/residuals. This contrasts with the derivative-dominated small-window
example below. The large sparse solve remains expensive despite the warm start.

Across frames 0–215, refinement changed fitted points by median 0.360 mm,
95th percentile 2.643 mm, maximum 55.676 mm. Including the unobserved tail, the
maximum was 259.048 mm. Summed quaternion-acceleration costs for 3/5/7/refined
were 410.628 / 382.230 / 374.388 / 364.327. The longer/global solves do improve
the combined objective; target RMS alone does not capture that improvement.
Maximum exact-attachment equation error stayed below 5.2e-13 mm in all four
fits. Relaxed shoulder separation is a distinct, intentional parameter.

Recommendation: keep three active frames as the fast comparison candidate;
retain global refinement as an explicitly optional expensive pass. Inspect the
saved motion before choosing a production default. Further speed work should
use the Ceres timing breakdown rather than assume window size is the only cost.

Three versus five active frames also illustrates why RMS alone is insufficient:
across frames 0–215, fitted-point displacement has median 0.370 mm, 95th percentile
3.613 mm, and maximum 101.028 mm. Across the unobserved tail, the maximum grows
to 1773.210 mm at the right index fingertip in frame 221. Those six frames have
no mapped keypoint targets: temporal extrapolation is not a recovered
measurement. The viewer explicitly marks that lack of measurement support.
This iteration does not introduce a new gap-filling or stopping policy.

The previous 908-second full-recording run used the older two-length equality
model and stricter tolerance. It is not an apples-to-apples speed baseline for
these runs. A cold full-recording solve of this exact model and tolerance is
still needed to isolate the benefit of windowing from tolerance changes.

### Tolerance experiment on frames 180–213

| Processing | Function tolerance | Ceres seconds | Windows not converged | Target RMS, mm |
| --- | --- | ---: | ---: | ---: |
| 3 active | 1e-10 | 36.48 | 12/32 | 23.375 |
| 5 active | 1e-10 | 74.37 | 14/30 | 23.368 |
| 7 active | 1e-10 | 87.52 | 12/28 | 23.397 |
| 3 active + global refinement | 1e-10 | 36.48 + 10.19 | final refinement converged | 23.390 |
| 3 active | 1e-6 | 13.29 | 0/32 | 23.412 |

Strict versus looser three-frame output: median fitted landmark displacement
0.096 mm, 95th percentile 5.303 mm, maximum 35.072 mm. Global objective increased
from 1128.099 to 1139.052. Similar RMS therefore does not mean identical poses.

One slow strict three-frame window used 201 reported iterations (initial state
plus the 200-iteration limit), with 1.671 seconds of residual/Jacobian evaluation
versus 0.249 seconds of linear solving. Reducing the problem size alone does not
remove derivative work or the need for a practical convergence criterion.
The current Ceres build also has `MINIGLOG_MAX_LOG_LEVEL=2`, which emits verbose
allocation messages even with solver progress logging disabled. This was left
unchanged throughout the benchmark; logging configuration is a follow-up
profiling concern, not an attributed speed improvement here.

### Viewer adaptation overhead

The full benchmark began before a separate Python adapter repair. Repeated
pybind vector-property access copied the entire sequence inside the frame/body
loop. The adapter now reads each sequence array once. A 222-frame read,
evaluation-only Ceres call and viewer adaptation took 4.70 seconds in a separate
diagnostic. This is not a fitting-time measurement or a rerun of the saved fit.
Saved benchmark wall times still include the old adapter overhead.
The redirected PowerShell benchmark returned status 1 with a `NativeCommandError`
record for Ceres stderr logging; it nevertheless saved all four completed fits,
with no Python traceback. Separate saved-result republishing returned status 0,
and the four-fit/222-frame geometry and viewer checks passed.

## Limits and next checks

This is an offline processing experiment, not an end-to-end live pipeline:
person scale, seed data and timestamp weights come from the prepared recording.
Three active frames imply at least two intervals of lookahead; five and seven
imply four and six. At 6 Hz those delays are 0.333, 0.667 and 1.0 seconds before
computation. Frame count must be reconsidered for higher-rate recordings.

Before selecting a production default, inspect bending, raised arms, occlusion
and window boundaries in the comparison. Compare temporal residual costs and
the final full-refinement displacement, not just target RMS. Reverse traversal,
a matching cold full solve, and longer/higher-rate recordings remain follow-up
experiments. No full sample-data reprocessing is needed for this iteration.

Synthetic tests cover unchanged fixed boundaries, complete parameter warm
starts, exact frame coverage, missing-keypoint windows, a single-window/full-fit
equivalence, full-objective rescoring, and refinement initialization. Viewer
checks compare displayed geometry with saved numerical output and verify each
frame's finalizing-window report.

Validation: 47 focused Python tests passed, three native CTest checks passed,
and the new four-fit viewer plus both existing comparison pages passed their
Node geometry/playback checks. A headless Edge render was also inspected.
