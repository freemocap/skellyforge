# Connected sequence fitting

This package owns connected model preparation, native argument construction,
the accepted human-fit configuration and moving-window execution. Recording
loading and viewer-specific grouping/diagnostics remain in scripts. The package
accepts prepared in-memory trajectories; it does not rerun tracking, triangulation,
gap filling, filtering, ground alignment or person scaling.

## Human fitting entry point

```python
from skellyforge.core.skeleton.fitting import fit_human

fitted = fit_human(skeleton, saved_model, segment_scales, records,
                   inspect_windows=(0,))
sequence = fitted.sequence
quaternions = sequence.quaternions
translations = sequence.translations
segment_names = fitted.model["names"]
```

Inputs use the existing recording adapter contract:

- `skeleton`: restored SkellyForge skeleton definition.
- `saved_model`: mapping definitions (`mappings`, including prefix and entries)
  and authored relative `rest_pose.orientations` in wxyz order.
- `segment_scales`: previously estimated scale keyed by segment name.
- `records`: ordered dictionaries with `number`, `time` (seconds), `keypoints`
  (source names to XYZ), `points` (mapped landmark names to XYZ), `rotations`
  (segment world wxyz quaternions), and `origins` (segment world XYZ).
  Vector entries follow the existing NumPy-array adapter contract.

`HumanFit.model` contains the ordered geometry and source mappings actually used.
`preparation` contains initialization/target bookkeeping; `sequence` is the same
`WindowSequenceFit` documented below. Model landmarks remain fully defined.
Available direct keypoint targets are counted once even when an attachment shares
that mapping. Saved mapped axial landmark positions supply explicit correlated
position preferences, not additional independent measurements. Preparation keeps
the existing initialization fallbacks; it does not manufacture measured targets.

The caller supplies gap-filled, filtered (where appropriate), aligned trajectories.
This extraction preserves existing missing-keypoint handling for compatibility;
it does not add a second gap-filling algorithm inside the solver.

### Numerical code ownership

- `body_model.py`: tree ordering, scaled local geometry, mappings and axial extents.
- `reference_geometry.py`: fixed SC reference-offset profile before optimization.
- `centerline.py`, `axis_priors.py`, `spine_priors.py`, `sc_prior.py`: existing
  geometric preparation for native residuals. Viewer drawing remains in scripts.
- `preparation.py`: native inputs, initialization, configured residual objects.
- `human.py`: accepted combination and public `fit_human` entry point.
- `settings.py`: shared numerical scales; Ceres defaults are read from `_native`.
- `window_sequence.py`: window execution and optional full-recording refinement.

`human_fit_options()` returns the accepted configuration: three flexible axial
lengths; soft 20:20:12 proportions; rest-length fraction 0.5; existing chest-line,
shoulder-axis and twist preferences; lowered SC with forward factor 0.75; relaxed
shoulder connections; axial landmark position scale 5 mm; keypoint Huber transition
30 mm; three active frames, 200 maximum iterations and function tolerance 1e-6.
The ratios are the accepted experimental values, not verified anthropometric
measurements. Residual scales are preferences, not hard anatomical limits.


## Execution and ownership

1. The caller supplies local segment geometry, linkage attachments, keypoint
   targets, timestamps, initial segment quaternions/root positions and configured
   priors through the existing native argument dictionary.
2. `fit_windows` prepares each window and calls `_native.fit_chain_sequence`.
3. `cpp/bindings/module.cpp` converts Python inputs to the C++ declarations in
   `cpp/include/skellyforge/chain_sequence.h`.
4. `cpp/src/chain_sequence.cpp` constructs parameter blocks and residual blocks,
   applies quaternion manifolds, fixed history and bounds, then calls Ceres.
5. Python retains the fitted active states and advances the window. It neither
   averages poses nor applies a second smoothing pass.
6. A final native evaluation scores the assembled recording without optimizing
   it. Optional `refine_window_result` is a separate full-recording optimization.

The native extension contains the numerical implementation, not the viewer.
This package imports NumPy and that extension; it does not import scripts,
FreeMoCap, SkellyTracker, Parquet readers or browser assets.

## API

```python
from skellyforge.core.skeleton.fitting import fit_windows

fit = fit_windows(
    arguments,              # prepared arguments for the native chain solve
    active_frames=3,
    inspect_windows=(0,),   # optional; default captures no problem graphs
)
quaternions = fit.quaternions
translations = fit.translations
window_reports = fit.processing["windows"]
```

`WindowSequenceFit.final` holds the native result. Attribute forwarding exposes
its fitted arrays and numerical diagnostics. `processing` records each window's
frame interval, termination, iterations and timing. `seconds` contains native
solve/evaluation time; `processing["wall_seconds"]` also includes controller work.
Use both when measuring performance. Ceres convergence and result usability are
separate properties; neither alone proves an anatomically good fit.

Inputs retain the existing native conventions: millimeters, right-handed +Z up,
+X forward, and wxyz quaternions. Segment quaternions in this solver are world
quaternions. Child translations are derived by connected forward kinematics;
only explicitly relaxed linkages have displacement parameters. Keypoints are
observations mapped to model landmarks; landmark definitions remain present
independently of observation availability.

### Window behavior

The default is three active frames and up to two fixed preceding frames. Frame
indices in `inspect_windows` identify the first active frame. The first window
uses the supplied initialization. Later windows retain overlapping fitted states
and initialize the new frame from its predecessor. Fixed history is checked for
exact equality after every solve. The final window retains all remaining active
frames. No marginalization is performed.

The controller uses full-recording timestamp weights and caller-prepared scale
and initialization. It is currently an offline controller, not a complete live
pipeline. Do not describe three frames as a frame-rate-independent time horizon.

### Inspection

Requested windows retain the actual native Ceres problem before and after their
solve. Parameters include frame/segment identities and quantities; residuals
include purpose, actual connected parameter IDs, weighted values and robust cost.
Frame identities inside native snapshots are window-local; the snapshot's
`frame_start` provides the recording offset. IDs are local to each problem.
Inspecting a window does not reconstruct its problem from the final trajectory.
The recording adapter supplies names from the same model used for fitting.

## Building and changing the native code

The source build uses scikit-build-core -> CMake -> a C++17 compiler, with
pybind11 exposing `_native`. CMake fetches pinned, SHA256-checked Ceres/Eigen
archives. See [cpp/README.md](../../../../cpp/README.md) for prerequisites,
bootstrap commands, platform caveats and wheel packaging status.

From the repository, with the native tooling already installed:

```powershell
uv run --no-sync poe native-check
uv run --no-sync poe native-install
uv run --no-sync poe test-native
uv run --no-sync poe test-native-python
```

`native-install` configures and incrementally builds Release, then installs the
extension into this checkout. Restart Python after changing C++ or bindings.
Python-only controller changes require no native rebuild. Do not commit build
outputs. Do not judge timings from a Debug or unoptimized native build.

A wheel includes this Python package and the compiled extension; the controller
has no dependency on the repository's `scripts` directory. Cross-platform wheel
publication remains a separate release task; a passing checkout test is not a
portable-wheel validation.

## Validation and integration status

`test-fitting` exercises window boundaries, initialization, optional refinement,
priors and native inspection. The recording-body and shoulder-offset tests also
exercise the extracted model preparation and viewer adapter.

To compare against the saved accepted test/sample fits without overwriting them:

```powershell
uv run --no-sync python -m scripts.validate_packaged_human_fit --recording test
uv run --no-sync python -m scripts.validate_packaged_human_fit --recording sample
```

These diagnostics require prepared recordings and the saved accepted candidates
under `build/spine_positions`. They check source SHA, segment order and all fitted
quaternions, translations, axial lengths and landmark positions; timing and results
are saved under `build/package_extraction`. They never reprocess the source videos
or overwrite prepared recordings. Timing is machine/load dependent.

The accepted recording viewer uses the same packaged preparation and execution.
It retains rendering and reporting code in scripts. Installed-wheel validation,
packaging the inspector frontend and FreeMoCap integration remain separate steps.
No FreeMoCap dependency update is needed to develop or validate this package here.


## Logging ownership and progress

Library imports never configure root handlers. FreeMoCap owns SkellyLogs setup
when using `fit_human`. Standalone `python -m skellyforge` and the accepted
`scripts.solver_spine_positions` entry point configure SkellyLogs with
`use_websocket=False`: console/file logs without an unconsumed IPC queue.
Run with `python -X utf8` in legacy Windows redirected consoles; the installed
SkellyLogs formatter contains Unicode characters not supported by CP1252.

Fit progress uses the terminal table only; duplicate formatted logging records
are not emitted. The progress callback still runs after every window, preserving
application progress and cancellation. Unusable results print a failure line and
raise an exception.

A separate terminal-only table writes directly to stderr, never to log handlers
or the WebSocket queue. It begins with settings and a definition for every column
and live-status field. Every completed window occupies exactly one table row, including usable
nonconverged windows. Use a wide terminal (about
180 columns) to avoid terminal soft wrapping. Interactive terminals have a single
updating bottom status line; redirected output has only permanent lines and no
ANSI escapes. Colors reuse SkellyLogs' `LOG_COLOR_CODES` palette; `NO_COLOR`
disables color. No terminal library or root-logger configuration is installed.

The live mean uses the last 20 completed-window intervals (all available if fewer).
ETA is remaining windows times that mean, excludes final evaluation, and is only
an estimate. Updates happen between native solves, not within a solve. Extra time
is measured native-call wall time minus native-reported time, not all Python
overhead. Window costs belong to different objectives and are not a sequence-wide
convergence curve. The final summary includes median/p95/max iterations and native
solve time, the slowest call, convergence counts, and evaluation/total wall time.

Ceres miniglog is compiled with `MINIGLOG_MAX_LOG_LEVEL=-1`, preserving native
warnings/errors/fatal checks while removing INFO/VLOG diagnostics. Rebuild the
native extension to apply this setting; Python logging levels cannot change it.
Nonconverged but usable results remain accepted by the existing policy, with a
WARNING summary. Unusable windows log ERROR and raise. No tolerances, residuals,
initialization or callback semantics are changed. Per-window native reports and
problem inspection remain available in the returned result; logs do not replace
the inspector or create another representation of its Ceres graph.
