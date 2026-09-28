# Connected sequence fitting

This package owns the sequential window controller previously implemented in
`scripts/solver_window_sequence.py`. The extraction preserves the numerical
implementation. Recording loading, body-model preparation and the accepted human
fit configuration still live in the recording scripts; they are the next
extraction stage. This is not yet a one-call recording-to-skeleton API.

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
uv run --no-sync poe test-fitting
```

`native-install` configures and incrementally builds Release, then installs the
extension into this checkout. Restart Python after changing C++ or bindings.
Python-only controller changes require no native rebuild. Do not commit build
outputs. Do not judge timings from a Debug or unoptimized native build.

A wheel includes this Python package and the compiled extension; the controller
has no dependency on the repository's `scripts` directory. Cross-platform wheel
publication remains a separate release task; a passing checkout test is not a
portable-wheel validation.

## Validation and remaining extraction

`test-fitting` exercises window boundaries, initialization, optional refinement,
priors and native inspection through the package imports. Changes to residual
math also require native and geometric tests. Changes to the human configuration
require test/sample recording comparisons of quaternions, fitted landmarks,
lengths, residuals and timing, plus visual review. Preserve the accepted fit.

Next: move recording-independent body-model/argument preparation and the accepted
configuration out of experiments, then have recording adapters and the inspector
call that single implementation. Do not move HTML generation or Parquet loading
into the numerical package. FreeMoCap integration is a later, separate handoff.
