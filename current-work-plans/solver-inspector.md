# Supported fitting pipeline and Solver Inspector

## Implementation checkpoint: direct problem inspection

Implemented the first native foundation in `cpp/{include/skellyforge,src}/problem_inspection.*`.
`ChainSolveOptions.inspect_problem` defaults to false. When enabled, the native
result exposes `problem_initial` and `problem_final`, captured from the same
live `ceres::Problem` immediately before solving and after solving/evaluation.
Parameter/residual enumeration, connections, ambient/tangent dimensions,
constant state, bounds, manifolds, raw weighted residuals and loss-adjusted costs
come from Ceres inspection APIs. There is no copied connection/formula registry.

`fit_windows(..., inspect_windows=[...])` retains only requested windows, with
their actual initial/final block states and frame interval. These records remain
separate from the final assembled trajectory and survive optional refinement as
historical records of the earlier window solves.

Current block IDs are local to the assembled problem, never serialized addresses.
Runtime C++ type names expose the actual cost/loss/manifold implementation, but
are compiler-specific; they are not a portable semantic naming contract. Binding
blocks to existing named skeleton entities, browser transport and renderer wiring
are still pending. Do not present the old JavaScript schematic as this new data.

20 focused native/window tests pass. Inspected and uninspected fits are identical
in tested numerical outputs. Tests check complete counts, summed costs, registered
connections, constant history, bounds, quaternion tangent dimensions, robust loss,
and automatic appearance of an added position residual without inspector changes.

The owner clarified that the inspector must not own a second model or graph schema.
All future presentation must consume these working objects and their existing
entity identities. Serialization transports facts; it must not infer the problem.

## Scope and invariant

The accepted real-data fit becomes a supported SkellyForge pipeline. The inspector
calls that pipeline and displays its results; it does not implement another fit.
Preserve the accepted objective, parameterization, initialization, window order,
weights, robust loss, and thresholds during extraction. Temporal improvements and
FreeMoCap integration are separate stages.

This plan follows inspection of the committed September 27 implementation.
`scripts/solver_spine_positions.py` selects the accepted configuration;
`solver_recording_body.py` currently combines model preparation, native argument
assembly, fitting, diagnostics and viewer serialization. `solver_window_sequence.py`
owns the three-frame controller. `cpp/src/chain_sequence.cpp` creates the actual
Ceres problem. `solver_problem.js` reconstructs a schematic from settings; it is
explicitly not native problem introspection. The standard inspector must replace
that schematic as the authority for the displayed connections.

## Proposed locations

```text
skellyforge/
  core/skeleton/fitting/
    inputs.py                 # skeleton, scale, timestamps, mapped targets, initial poses
    configuration.py          # compositions of supported residuals and parameter policies
    prepare_problem.py        # resolve existing segment/landmark/linkage references
    fit_sequence.py           # native solve entry point and result assembly
    window_sequence.py        # extracted existing moving-window controller
    result.py                 # fitted poses, axial lengths, linkage displacements, diagnostics
    problem_trace.py          # Python representation of native problem records
    profiles/
      human.py                # accepted human configuration, not a solver subclass
  tools/solver_inspector/
    __main__.py               # standalone command, load/run/serve explicit operations
    recording.py              # prepared Parquet adapter, optional PyArrow dependency
    session.py                # saved fit and trace manifest; references source recording
    server.py                 # local inspector and frame/trace data endpoints
    web/
      index.html
      layout.css
      scene.js                # reuse current comparison scene
      playback.js             # reuse current scrub/play/video synchronization
      problem_map.js          # reuse interactive graph presentation
      inspector.js            # selections, values, visibility, window controls
cpp/
  include/skellyforge/problem_trace.h
  src/problem_trace.cpp       # capture actual Ceres block graph and evaluations
  src/chain_sequence.cpp      # existing objective, tagged block registration
  bindings/module.cpp        # expose traces and existing numerical results
```

These are proposed responsibilities, not empty files to scaffold up front.
Add a file only when its implementation exists. Keep native mathematics in C++.
There must be no runtime imports from `scripts/`, Tracker, or FreeMoCap in the
installed fitting package or inspector.

Use `core/skeleton/fitting` because this operation jointly fits the complete
skeleton. It does not belong in the landmark definition or single-chain layer.
The human profile names the existing spine segments and shoulder linkages;
generic fitting operations accept resolved entities and numerical definitions.
Existing `fit_connected_pose` / `fit_connected_sequence` use an older SciPy path.
Do not silently redirect their callers or preserve two competing production
defaults. Review callers before explicit replacement/deprecation during migration.

## Data contract in project vocabulary

Inputs retain the existing SkeletonDefinition, ModelScaleFit, RestPose and mapping
definitions. Keypoint trajectories remain source data. Direct mapped landmark
targets and derived landmark position preferences retain their distinct identities
and provenance. Each target references its existing landmark and source mapping.
Reference geometry changes such as the accepted SC offsets are explicit profile
settings applied before fitting, with original and effective geometry recorded.

Outputs include fitted segment quaternions (wxyz), origins, fitted landmark
positions, axial lengths, linkage displacements, window reports, and acceptance
results. Do not squeeze flexible geometry into a rigid SkeletonPose while losing
the fitted lengths or relaxed connections. Reuse existing pose objects where their
contracts apply; carry the additional fitted quantities explicitly.

Recording access and viewer serialization sit outside the numerical API. A caller
can fit synthetic geometry or in-memory trajectories without Parquet, video,
HTTP, or human-specific defaults. A future rigid object case should compose
position targets and quaternion parameters without spine or shoulder terms.

## Composition

Compose explicit parameter policies and residual definitions around the existing
objects: root translation, segment quaternions, axial lengths, linkage displacement;
mapped landmark targets, derived landmark position preferences, length/rest/ratio
preferences, geometric line/axis preferences, relative quaternion preferences, and
temporal residuals. A profile chooses components and their settings. Components
are assembled into one Ceres problem; they are not sequential independent fits.

Exact forward-kinematics connections, fixed parameter blocks, parameter bounds,
and quaternion manifolds are separate from soft residuals. Robust loss belongs
to its residual block. Acceptance checks are separate from both. The inspector
must show these distinctions instead of labeling everything a constraint.

Start with the supported components already present. No inheritance tree of
tracker-specific solvers, plugin registry, new entity ontology, or new residual
mathematics is required for this extraction.

## Native trace is the authority

Assign stable logical IDs using frame, entity and parameter/residual family;
never serialize memory addresses. Capture connections from Ceres' registered
residual blocks and parameter blocks. Registration sites supply semantic names,
units, scales, targets and provenance; introspection supplies actual connectivity,
dimensions, fixed/variable state, bounds and manifold information.

For a selected window expose:

- root XYZ, each segment's wxyz quaternion, axial length, and linkage displacement;
- parameter ambient/tangent dimensions, initial/final values and constant status;
- every enabled residual family and its actual parameter connections;
- raw residual components, robustified cost, loss type and transition scale;
- time weights, physical units, mapped targets and fixed reference values;
- exact linkage equations and source definition identities;
- iteration summary, termination reason, timing and geometry acceptance results.

Fixed history and adjustable frames must be labeled. A finalized frame's pose is
not the full state of the window when it was optimized: retain the selected window's
actual initial/final states, rather than constructing a historical trace from the
final assembled trajectory. Separately identify full-sequence evaluate-only costs.
Do not sum overlapping window costs and call them the recording objective.

Make trace detail selectable: summary for normal fitting; full initial/final block
records for inspection. Record detailed traces in bounded, replaceable session
outputs with a manifest. Load one window's graph at a time; do not inline all
windows into a hundred-megabyte HTML file. Do not silently rerun a solve to invent
a missing historical trace. Older sessions without traces are labeled accordingly.

## Standard tool interaction

Reuse the accepted recording viewer's 3D scene, annotated video, synchronized
bottom transport and comparison controls. Reuse the solver lab graph's pan/zoom,
hover connections, pin selection, family filters and body grouping. Every panel
is resizable and collapsible; labels and axes have independent controls.

Default visible content concerns the actual selected solve:
fitted skeleton, its landmark targets/preferences, and optional annotated video.
Source keypoints, saved segment poses, axial reference geometry and relaxed
shoulder connectors are independently selectable. Derived landmarks and keypoints
remain visibly distinct. Only geometry relevant to enabled components gets a
default overlay; historical experiment options stay in the old lab.

Selecting a segment or landmark highlights its parameter blocks and residuals.
Selecting a residual highlights the target, fitted point or linkage it constrains.
Selecting a window shows its fixed history and active frames. The inspector can
compare saved sessions without changing either result. Hover must not resize the
graph or panels. Graph state survives playback and selection changes where valid.

Proposed user-facing entry point: `poe solver-inspector`, backed by
`python -m skellyforge.tools.solver_inspector`. Distinguish opening a saved session
from starting a new solve. Package frontend assets so the installed tool works
outside a repository checkout. Preserve optional recording/video dependencies;
the numerical fitting API must not require them.

## Ordered implementation and review checkpoints

1. **Native trace foundation.** Instrument existing block registration, expose a
   selected window's true graph and values. Prove parameter/residual counts,
   connectivity, bounds, constant blocks and costs match Ceres. Compare tracing
   enabled/disabled solutions; numerical output must remain unchanged. Show this
   graph with the existing interactive map before rearranging the solver.
2. **Extract the accepted numerical pipeline.** Move window execution and problem
   preparation into the package; have the current viewer call it. Preserve one
   implementation behind old script entry points while migrating callers. Compare
   full test/sample outputs to the committed accepted configuration. Do not use
   unchanged global RMS alone as the equivalence criterion.
3. **Package Solver Inspector.** Combine the recording scene and true problem map,
   with linked selection and frame/window distinction. Add saved sessions, lazy
   window loading, resizable panels, explicit solve/open actions and Poe command.
   Validate installed-package assets and operation from outside the checkout.
4. **Regression and handoff.** Run geometry/cost/trace checks plus the real recording
   and synthetic rigid/linkage cases. Verify the inspector shows actual configured
   residuals and no invented ones. Report numerical equivalence and tracing overhead.
   The human commits/pushes Forge; FreeMoCap integration is a separate stage.

Every checkpoint includes something inspectable, not just a report that tests
passed. Preserve the approved fit as the baseline throughout. Temporal tuning,
new scapular mechanics, BVH/GLTF conversion and FreeMoCap UI changes are outside
this extraction and inspector work.

## Native inspection preview checkpoint

`scripts/problem_inspector.html` now displays actual Ceres snapshots for test
recording windows 0, 193 and 219, before and after optimization. Connectivity,
parameter values, manifolds, bounds, constant flags, residual values and costs
come from the native problem. Python transports exposed read-only properties;
the viewer does not reconstruct the graph from skeleton definitions.

The preview reuses the existing interactive graph renderer and embeds the
recording viewer below it. These are distinct views: a selected window's captured
optimization state versus the finalized recording poses. Selection is not yet
linked. Blocks currently have native problem-local IDs and implementation types,
not anatomical names. Package extraction and linked entity selection remain next.

Validation: 21 focused Python tests pass; native graph checks cover three windows
in both states, every displayed connection, filtering, zoom and stable hover.
Existing solver-viewer checks pass. Capturing these windows left every test-data
frame's quaternion, fitted landmark position and axial length exactly unchanged
against the pre-capture result (222 frames). Browser pixel inspection remains
outstanding; HTTP and DOM/graph checks passed.


### Named frame layout

Native parameter registration now retains frame index, segment index and quantity
(root translation, quaternion, axial length or linkage displacement). Residual
purpose comes from the solver's existing residual-family collections. Segment
names travel from the same body model used for the fit. Ceres remains the source
of every connection and numerical value.

The inspector defaults to frame columns with fixed-history/active headings,
segment parameter groups and residual-purpose groups. Disabling grouping exposes
individual blocks. Group connections are unions of the captured native edges;
group costs are sums of their captured residual costs. They do not define a
second mathematical model. Selection still does not drive the 3D recording.


### Package extraction, first step

The moving-window implementation now lives in
`skellyforge/core/skeleton/fitting/window_sequence.py`. All script/test callers
import the package; the old script implementation was removed. Executable ASTs
were compared before/after the move and matched exactly. The 41 focused fitting
tests pass. No residual mathematics or native build configuration changed in
this extraction. The package README documents the API and runtime/build boundary;
the native README no longer incorrectly describes the extension as smoke-only.

Body-model preparation and the accepted human configuration remain in scripts.
Their extraction, installed-wheel validation and fresh test/sample comparisons
remain outstanding. Existing viewer traces remain available; this move alone
does not require regenerating their unchanged numerical results.


### Accepted human configuration and preparation extracted

`fitting.fit_human` now composes numerical body-model preparation, existing
reference geometry/priors, native argument construction and the packaged window
controller. Recording reads and viewer diagnostics stay in scripts. The accepted
viewer calls the same preparation and fit configuration. The old settings module
was removed and callers use the package constants. No C++ or objective changes.

Validation: 50 focused tests pass. Full saved comparisons for the 222-frame test
and 1108-frame sample have exactly zero differences in world quaternions,
translations, axial lengths and fitted landmarks. Timing in this run: test
100.60 seconds native / 103.81 wall; sample 59.75 native / 75.98 wall. Timing is
not a controlled benchmark. Reports: build/package_extraction/{test,sample}.json.
Installed-wheel validation and packaging the inspector are still outstanding;
FreeMoCap integration remains a separate stage.

All 15 residual-family cost comparisons also match exactly on both recordings.


### Installed-wheel checkpoint

Built and installed the Windows CPython 3.12 wheel into an isolated environment
outside the repository. Python -I loaded the package and native module only from
that environment. Synthetic window fitting and native inspection passed; the
complete 222-frame fit matched saved quaternions, translations and axial lengths
exactly. Runtime: 91.21 s native / 93.85 s wall. Build inputs and commands are in
cpp/WHEEL_VALIDATION.md. Clean installation exposed an unused, unavailable
skellylogs runtime requirement; removed it and its unused source, refreshed the
lock offline without advancing Git revisions. Other-platform and clean-machine
runtime validation remain outstanding. Inspector frontend packaging is next.


### Logging correction / supersedes wheel dependency removal

The SkellyLogs removal was rejected: shared logging is required. Restored the
committed dependency declaration, source declaration, lockfile and temporary
logging-workaround comment. The earlier wheel's successful numerical check does
not validate the accepted dependency configuration. Investigate/fix SkellyLogs
lifecycle in its repository, then integrate through the human commit/push boundary.
No solver mathematics has changed.


### Logging integration and remaining delivery stages

Removed the temporary import-time basicConfig workaround. Standalone entry
points use SkellyLogs console/file handlers; application consumers retain
ownership of handlers and IPC relay. Added preparation and bounded-frequency
window progress logs, per-window DEBUG details and convergence/timing summaries.
43 focused tests pass, including exact numerical equality with logging enabled.
Shared logging smoke test passes in UTF-8 mode. Legacy Windows CP1252 redirected
console formatting still fails on SkellyLogs Unicode decorations; this is a
shared formatter issue, not a solver failure. No native mathematics changed.

Remaining order of work:
1. Close installed-wheel validation with the required SkellyLogs dependency,
   including supported build/install instructions and dependency availability.
   The previous diagnostic wheel is not the release acceptance evidence.
2. Package the existing recording inspector as a supported Forge tool. Preserve
   native parameter/residual snapshots and existing real-data checks; do not
   add a separately maintained visualization model.
3. Confirm packaged accepted fits on both prepared recordings and document
   runtime, residuals, convergence limitations and reproduction commands.
   Human commits/pushes Forge before consumer integration.
4. In FreeMoCap, add an explicit optional post-hoc skeleton-fitting stage after
   trajectory preparation, person alignment and person-scale preparation.
   Specify saved fitted geometry and solver provenance using the current Parquet
   conventions; preserve keypoints and existing segment outputs. Add UI options,
   progress/error reporting and save/reload integration tests. This is a separate
   core stage, not a change to the live segment pipeline.
5. After pipeline and Parquet round-trip acceptance, implement exports from the
   saved result: CSV wide/tall and NumPy first, then glTF and BVH. Flexible spine
   lengths and relaxed shoulder connections require explicit representation or
   projection decisions for a fixed-offset BVH armature; do not silently discard
   those differences or claim exact equivalence to the fitted geometry.


### Supported viewer entry point (2026-09-28)

`poe viewer` / `skellyforge-viewer` now presents the synthetic hydration tool,
accepted test/sample recording fits, and native Ceres captures together on port
8774. Synthetic and comparison Python code live in tools/viewer; historical
script entry points delegate to those modules. Existing browser assets are
installed unchanged by CMake. Standalone streams and Poe Python tasks use UTF-8.

The new optional preparation command `poe viewer --prepare-synthetic-fit` runs a
60-frame all-motion, ideal full-landmark fixture through `fit_human`. Synthetic
keypoints have explicit identity mappings; this is NOT a sparse COCO simulation.
58/58 windows converged (2.556 native seconds, 3.029 sequence wall seconds in this
run). Its rendered connected bases use the returned Ceres quaternions. Synthetic
captures are invalidated by fixture/fitting/native source changes. Recording
views verify the saved source Parquet hash. All existing recording mathematics
and accepted fit artifacts are unchanged.

Validation: 55 focused Python tests; existing Node recording control/geometry
and native graph checks pass. HTTP checks loaded all three saved fit views.
Browser visual review is pending because no browser connection was available.
Clean installed-wheel validation is blocked by SkellyLogs package discovery;
see cpp/WHEEL_VALIDATION.md. Recording-fit preparation still lives in scripts
and must be packaged before complete Forge sign-off. See tools/viewer/README.md
for the supported commands and explicit remaining scope.
