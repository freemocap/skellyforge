# Skeleton viewer

From the SkellyForge repository root, using its installed environment:

```powershell
.\.venv\Scripts\python.exe -B scripts/generate_skeleton_viewer.py --serve
```

Open **http://127.0.0.1:8766/**. Keep the terminal running; Ctrl+C stops the server.
It binds only to localhost, serves the viewer and recomputes poses in Python. It
does not read recordings, download data, or write a new file for each combination.
Use `--port` to choose another port if needed.

The starting pose is still, with **all motion and noise disabled**. Enable root
translation, shoulders, elbows or head independently. Root movement is not
required. Changes retain the current frame and camera while recomputing the
observations, recovered poses, connected poses and plots. Controls disable during
computation; an error leaves the previous data visible with an explicit message.

Noise has its own switch and a standard deviation in millimetres (0â€“20 per
coordinate). It is independent Gaussian landmark noise, with a fixed seed for
repeatable comparisons. With noise off, residuals cannot be attributed to injected
noise; inspect the reconstruction and orientation conventions instead.

Body orientation arrows show +X red, +Y green and +Z blue on the torso, head,
clavicles, arms and legs. Finger axes have their own visibility switch and length
slider. Choose independent or connected
origins and adjust arrow length. Both representations use the same rotations;
connecting them changes the origins. Turn overlays off to reduce clutter.

Use **Left hand** or **Right hand** to focus the camera. Finger bones and landmark
markers are smaller so axes remain legible. **Spread fingers** starts enabled and
poses the metacarpals in a fan (thumb -35Â°, index -12Â°, middle 0Â°, ring 10Â°, pinky
22Â°, mirrored on the other side). It exercises the normal FK/observation/recovery
path; these display-pose angles are not anatomical limits or edits to the shared
rest skeleton. Disable it to inspect the original straight-finger pose.

For a standalone snapshot, omit `--serve` and open `scripts/skeleton_viewer.html`.
The generated HTML includes its data and JavaScript libraries and opens offline,
but motion/noise input controls require the local server. Playback, scrubbing,
axes and overlay controls work offline.
Each generation overwrites the same HTML file. It is currently tracked despite
its existing ignore rule, so this change includes the regenerated snapshot.
Edit the generator, then regenerate after changing the skeleton or calculation code.

The viewer compares three representations of a synthetic moving subject:

- White: the source motion used to generate landmark observations.
- Red/blue/gray: segment poses recovered independently from noisy observations.
- Cyan: `synthesize_fitted_pose` using those recovered rotations and root, with
  fixed parent attachments and known subject dimensions.

Pause and drag the frame slider, or click any chart to seek. Toggle overlays to
inspect each representation alone; hover cyan bones for their origin displacement
in millimetres. Drag the scene to orbit and scroll to zoom. Orientation axes and
center of mass belongs to the independent poses and is initially hidden.

Useful comparisons:

```powershell
# No measurement noise: check the synthetic baseline.
.\.venv\Scripts\python.exe -B scripts/generate_skeleton_viewer.py --noise-mm 0

# Amplify differences and exercise unequal segment dimensions.
.\.venv\Scripts\python.exe -B scripts/generate_skeleton_viewer.py --all-motion --noise-mm 5 --unequal-scales
```

Reload an offline snapshot after regenerating. The default noise is off;
the optional unequal scales vary by up to 8% around the 1700 mm model scale
and stay fixed throughout the animation. The random seed is fixed for comparisons.

The first plot selects a major body segment and shows its inter-frame rotation (direction and roll),
with zero shown at the first frame because there is no preceding frame. It exposes
noise-amplified spinning independently of origin residuals. The next plot measures
connected versus independent origin displacement. This
is not an assertion that either representation perfectly matches measurements.
The remaining plots compare landmark noise and recovered motion against the known
source. This demo does not read recordings, estimate subject dimensions from
observations, optimize a skeleton against trajectories, or validate an exporter.
For saved real recordings, use the companion viewer below.

The resolver starts from authored rest orientations and carries orientation with
its parent, applying only the minimum swing needed to match each new measured
direction. Model-declared terminal orientation evidence supplies twist where
available. There are no bend-plane thresholds. Transported twist remains a
convention, with geometric path dependence; this is not full anatomical IK.

Restart `--serve` after Python changes. The loaded-code panel identifies the
resolver path, source hashes and server start time. The server checks both roll
resolution and twist-backfill code, as well as the viewer script, for changes.
Refresh alone does not reload Python.
# Real recording companion viewer

From the SkellyForge repository:

```powershell
.\.venv\Scripts\python.exe -B scripts/generate_real_skeleton_viewer.py --serve
```

Open http://127.0.0.1:8771/real_skeleton_viewer.html. This overwrites
`scripts/real_skeleton_viewer.html`, leaving `scripts/skeleton_viewer.html` intact.
Both viewers share the vendored Three.js/OrbitControls assets and cylinder drawing
helpers. No FreeMoCap or SkellyTracker import is used by the real-data viewer.

For a fresh environment, the optional Parquet reader is installed with
`uv sync --extra recording-viewer`. The generator defaults to:

```text
~/freemocap_data/testing/prepared/freemocap_test_data/current/recordings/freemocap_test_data/freemocap_test_data_data.parquet
```

It falls back to `~/freemocap_data/recordings/freemocap_test_data/` with the same
Parquet filename. `--dataset sample` selects the corresponding sample-data paths;
it is never selected automatically. `--parquet PATH` overrides discovery and
`--sensor-group NAME` disambiguates recordings containing multiple model streams.
Missing processed data produces a preparation instruction; the viewer does not
download raw videos or run another repository's pipeline.

Orange bones display saved post hoc segment origins and world rotations. Their
endpoints use the skeleton geometry and fixed scale fit embedded in the same
recording. Local default definitions, hydration, roll resolution, scale fitting
and shoulder optimization are not run by this viewer. The previous gray/cyan
experimental layers have been removed because they were not production outputs.

Pink wireframe spheres are saved Forge landmarks; smaller green solid spheres are
Tracker keypoints. Both point layers start visible. Hover any point or segment for
its type and name; overlapping objects list together. Keypoints come from the exact
saved raw/filtered 3D input channel named by the reconstruction metadata. All
channels must match frame numbers, timestamps, coordinate frame and units.

Bones have a 13 mm base diameter and a 3.25 mm distal diameter. Drawing endpoints
are origin + recorded rotation applied to the saved, fixed-scale primary vector;
no endpoint is snapped to a landmark. Missing poses remain absent. Invalid
quaternions and inconsistent saved geometry/dimensions fail rather than being
repaired. Gaps between independent segments remain visible.

Use Play, frame scrubbing, last-quarter/whole-recording playback, layer toggles,
segment XYZ axes and Focus shoulders to inspect the result. The source panel
identifies the recording checksum, selected run and input channels. To change the
reconstruction, run the production post hoc reconstruction in FreeMoCap and then
regenerate this viewer. FreeMoCap's prepared-recording refresh helper can reuse
saved 3D keypoints without detecting or calibrating the videos again.

Saved data is opened read-only and checksum-checked. Only the single HTML artifact
is overwritten. Ctrl+C stops the local server. No new dependencies are required
beyond the existing optional Parquet reader.

## Connected root fitting demonstration

From the SkellyForge checkout, run `.\.venv\Scripts\python.exe -B -m scripts.generate_connected_fit_viewer`, then open `scripts/connected_fit_viewer.html` in a browser. It uses the existing offline viewer assets and geometry.

The slider selects whole-body displacement (0–2), overhead arms (3–5), and four static 1 mm noise samples (6–9). Each noiseless group shows the initial pose, fixed-pelvis fit, and free-pelvis fit. Orange is fitted geometry, blue is the known synthetic geometry, and pink spheres are the 11 target observations. Short saturated axes belong to the fit; longer pale axes belong to the reference. Hover for names and select segments to inspect rotation and origin errors.

These are independent static comparisons, not a recording or a temporal solve. Noise is independent Gaussian XYZ noise with a fixed seed. Every fit starts from the same T-pose, not the preceding result. Lengths and connections stay fixed. The reference generates the target points but is not passed as the fitter's initial pose. Explicit demonstration priors are listed in the page; they are not production defaults. Small point residuals can coexist with different joint rotations because sparse points do not determine every degree of freedom. The saved-recording viewer keeps its original defaults; optional comparison layers are used only by this synthetic page.

## Connected sequence fitting comparison

Run `.\.venv\Scripts\python.exe -B -m scripts.generate_sequence_fit_viewer` from
SkellyForge, then open `scripts/sequence_fit_viewer.html`. SciPy is a required
runtime dependency; normal `uv sync` installs it. Generation can take minutes
because it runs the actual fitting algorithms.

Choose the static-noise or moving-arms/pelvis case. Both use four frames at 30 Hz,
11 synthetic landmark observations and independent Gaussian XYZ noise with SD
1 mm. Orange fits each frame independently. Green solves the whole window jointly
with explicit pose and velocity priors. Blue is the known generating pose; pink
shows the same noisy inputs for both fitters. The known pose is not supplied as
the initial pose or prior: all fits start from the T-pose. The axes selector lets
you compare either fit with the known reference. Point errors, convergence and
the selected segment's world rotation/origin errors remain visible.

The window objective integrates point/pose residuals with trapezoidal timestamp
weights and squared velocity over actual elapsed seconds. A fully missing frame
is supported by its prior and neighbors, not treated as observed. This short
demonstration is not a full recording test, anatomical validation, causal realtime
filter, or long-window performance claim. Inspect the moving case for attenuation;
smoother output alone is not evidence of greater accuracy. Settings are disclosed
on the page and remain experimental, not production defaults.

## Real recording: experimental connected fit

Run `.\.venv\Scripts\python.exe -B -m scripts.generate_recording_fit_viewer`.
Open `scripts/recording_fit_viewer.html`. It starts at recording frame 200 and
reviews the eight-frame window 196–203 by default. Use `--first-frame`, `--count`,
`--dataset sample` or `--parquet` to select another window/source.

The reader uses the same prepared/canonical locations as the saved-recording
viewer. It restores the saved skeleton snapshot rather than current YAML,
preserves the saved scale fit, and initializes the sequence fit from saved root
poses and parent-relative rotations. Joint pose preferences use the saved model's
authored rest rotations, independently of that initialization. The weak root
preference remains relative to the saved root and is disclosed. The 11 selected targets must have unique,
direct saved mappings; saved landmark coordinates are checked against their
source keypoints. No Tracker or FreeMoCap imports, downloads, video processing,
calibration, or Parquet writes occur. Source SHA-256 is checked before/after.

Orange is unchanged saved posthoc output. Blue is the experimental connected
upper-body result. Pink and green remain saved landmarks and input keypoints.
Both may be shown at once, with hover labels and selectable axes. Neither pose
layer is ground truth. Legs/hands are saved context, not newly fitted output.

The page records source identity, input/prior policy, solver convergence, target
residuals and all weights. CLI options expose target/pose/velocity scales and the
evaluation budget. Defaults are experimental review settings, not uncertainty
estimates or production configuration. A missing initialization pose fails
explicitly rather than silently substituting a current template. Only the HTML
review artifact is overwritten. Production adoption still belongs in FreeMoCap
after the Forge implementation and real-data behavior have been reviewed.

The review also reports lumbar/thoracic axis bend and rigid-target distance
incompatibilities. For two observations attached to the same rigid segment,
half the difference between observed and fixed point separation is a lower bound
on the larger endpoint error. Joint-origin identities are respected in this check;
an articulated chain is not treated as one rigid body. These quantities diagnose
model/observation compatibility, not anatomical accuracy.
# Ceres solver development viewer

Time-series panels now use bundled Plotly: scroll/box zoom, pan, reset axes,
hover values, component checkboxes and SVG export. Drag the bottom-right corner of
each individual chart panel to change its width and height, or use Expand to
fill the window (Restore/Escape returns). Chart zoom survives playback and
fitting-method changes. Click a data point to select its frame. Sidebar sections
also have vertical resize handles; the 3D/plots and main/sidebar splitters remain.

The lab now includes stationary and moving sequences (41 frames over two seconds),
with independent fits, two velocity preferences, and an acceleration preference. Bottom
panels plot translation XYZ, quaternion WXYZ, and a selectable cube landmark's
world XYZ. Component colors are red/green/blue, with gray W; solid is fitted,
dashed is known, and landmark dots are observations. Click a plot to select its
frame. Both the sidebar divider and the horizontal plot divider are draggable.
Quaternion signs are made continuous only for plotting, without modifying saved
solver output. Component values are not Euler angles.

Sequence fits initialize from independent Ceres fits, never known poses. They
use trapezoidal time weights on scaled landmark residuals and neighboring
quaternion/translation motion residuals divided by sqrt(dt) and explicit speed
scales. The quaternion penalty is 4*sin(theta/2)^2, not squared angle. There are
no endpoint pins or robust losses. Every frame has at least five observed
non-collinear landmarks in this experiment; fully missing frames and arbitrary
per-frame landmark subsets are not yet supported by this native sequence API.
The outlier occurs at frame 20 only. The stronger setting intentionally exposes
attenuation of real motion. It is not a recommended production setting.

The generated bank has 32 static cases and 64 sequences, each with four fitting
methods. Changing controls selects saved solves; it never fits in JavaScript.
The objective chart shows the selected frame's independent solve or the entire
sequence solve, as labeled. Native tests check timing/unit consistency, cost
accounting, static noise reduction, and retention of moving trajectories.

From the Forge repository, run `uv run --no-sync poe native-install`, then
`uv run --no-sync poe solver-viewer`. Open `scripts/solver_viewer.html` directly
or run `uv run --no-sync poe solver-viewer-serve` and visit
http://127.0.0.1:8773/solver_viewer.html . All JavaScript assets are local.

The static experiment fits a 200 mm cube with Ceres AutoDiff and a wxyz
QuaternionManifold. Controls select 32 precomputed runs varying noise, missing
landmarks, an outlier and seed. The page does not solve in JavaScript; regenerate
after changing native code. Known geometry, initial pose, fitted pose, targets,
axes and residuals can be inspected separately. JSON download preserves the
selected inputs, poses, cost history and diagnostics. Costs are ordinary squared
landmark errors, not robust or temporal losses. Rotation-vector poses are not
used. This experiment does not modify the skeleton pipeline or recorded data.

The acceleration comparison uses differences between adjacent interval velocities,
weighted by the time between interval midpoints. Translation scale is 3000 mm/s?;
angular scale is 20 rad/s?. These are illustrative settings, not production defaults.
Angular velocities use the principal quaternion logarithm in a shared world basis;
only this residual calculation uses three-component angular increments. All pose
parameters and outputs remain wxyz quaternions on the Ceres quaternion manifold.
Inter-frame rotation must be below 180 degrees to identify that principal increment.
There are no acceleration residuals outside the interior interval midpoints and no
endpoint pins. Constant linear and constant world angular velocity have zero penalty,
including with unequal time intervals. Genuine acceleration can still be attenuated.

The lab now includes two rigid segments joined at a shared point. Choose experiment
04, then compare independent segments with the connected fit. The connected fit
uses two quaternion blocks and one shared world attachment point: coincidence is
exact, not a weighted penalty. This stage has no temporal prior or joint-angle limits.
The eight local landmarks on each segment provide roll information; this is not yet
a solution to the sparse landmark observability of a human skeleton.

The experiment selector only exposes applicable controls. The segment selector
controls the three shared trajectory plots and residual table. Initial poses are
hidden by default; settings, iteration histories, residual tables and pose values
are expandable. Every plot and the scene/sidebar splits remain resizable.
See [solver_lab.md](solver_lab.md) for the viewer organization and extension contract.


Experiment 05 adds a single Ceres Problem over both linked segments and all frames.
Select **One temporal Problem**, then compare **Per-frame Problems** during the amber
child-observation gap. The child has no per-frame fitted pose in that interval.
The temporal fit retains both quaternion blocks and the shared joint-position block,
connected across frames by acceleration residual blocks. See `solver_lab.md` for the
parameterization, initialization, reported residual costs and current input limits.


Experiment 06 adds the three-segment chain. Select Middle segment, inspect frame 20,
and toggle the gray initialized pose against the orange Ceres fit. Root and distal
landmarks remain observed during the middle gap. ChainLandmarkResidual blocks use
root position and all upstream world-quaternion blocks; both linkage attachments
remain exact. The Ceres inspector shows those parameter dependencies.


Experiment 07 adds a bounded displacement parameter at the second linkage while
keeping every segment rigid and observed. Compare Fixed linkage and Bounded
displacement. The map includes scalar parameter blocks, prior residuals and temporal
residuals; the displacement plot shows fitted versus known values. The gold connection
line represents modeled separation, not a broken attachment equation. Settings are
for this synthetic experiment only; see `solver_lab.md`.


Experiment 08 combines bounded displacement with a middle observation gap. Compare
Fixed linkage and Bounded displacement on identical observations; select Middle
segment and frame 20. The shaded interval has no middle landmark residual blocks.
Inspect the displacement plot, quaternion plots, residual-family costs and bound
saturation count. No production skeleton fitting behavior changes in this step.


Experiment 09 adds one parent and two fixed branches. Select Branch A and compare
0, 5 and 11 missing frames. Frame 20 is centered in the gap. Both attachment errors
should stay near numerical zero. The Ceres map shows each branch using the shared
parent quaternion/root position and its own quaternion, not its sibling's. Fit
diagnostics report landmark and quaternion angular errors separately for each body.


Experiment 10 validates a five-segment tree with dense synthetic observations.
It is an infrastructure check, not a fit using only hip and shoulder landmarks.
The map follows arbitrary ancestor paths; a branch never uses its sibling's
quaternion. The four-landmark torso solve is the next modeling stage.


Experiment 11 uses the existing rigid human torso definitions and only four observed
landmarks. Compare No rest-pose residuals / With rest-pose residuals; select shoulder
elevation, left shrug or torso twist. Inspect the per-segment angular errors as well
as observed landmark RMS: matching four points does not determine every segment
rotation. The Ceres map includes filterable RelativePoseResidual blocks. Non-converged
runs remain visible and labeled; no production fitting or FreeMoCap code changes.


Experiment 12 reviews the prepared real recording with the native rigid torso solver.
From the SkellyForge root run `uv run --no-sync poe solver-viewer-recording`, then
refresh the viewer on port 8773. An explicit path can be supplied with
`python -m scripts.solver_recording_torso --parquet <path>`.
Generate the synthetic viewer first if the HTML is missing. Regenerating the
synthetic bank replaces the HTML; run solver-viewer-recording afterward to add the
real recording again. No reconstruction or source-recording writes occur.

The current test recording review covers frames 0-215 at saved timestamps. Frames
216-221 lack targets and are reported as omitted. Interior gaps are rejected rather
than bridged or interpolated. The four direct landmark/keypoint correspondences
are verified before fitting. The model uses saved skeleton, T-pose and segment
scales. Initial pose comes from observed targets and the T-pose, never saved segment
rotations. There is no reference motion: known-pose layers/curves and angular
accuracy scores are absent. Fit errors compare against observed targets only.


To add full-body context and annotated camera previews to experiment 12, run
`uv run --no-sync poe solver-viewer-context` after generating the recording fit.
This reads saved full-body segments, landmarks and tracker keypoints as separate
context layers. They are not additional Ceres results or targets.

The camera panel decodes the recording's annotated_videos (never raw video). ffprobe
checks every video frame timestamp and count against the saved recording grid;
ffmpeg extracts sequential JPEG previews without inserting or dropping frames.
Scrubbing selects the matching frame number. A loading/error caption replaces an
outdated image; the source videos are never modified. Choose a camera and resize,
collapse or expand the panel. This is frame-sequence playback, without video audio.

Previews live in ignored scripts/.solver_media/recording_torso, with stable camera
and frame names reused between generations. Current three-camera previews occupy
about 108 MiB. Source hashes invalidate the cache; no per-run folders accumulate.
`--no-videos` adds only 3D context if ffmpeg/ffprobe are unavailable. Regenerate this
context after regenerating the recording fit. The local scripts HTTP server serves
the previews; keep the cache beside the generated HTML.


Experiment 12 now compares Rigid spine and Flexible axial spine on the same real
recording. Inspect frame 192 and the new Spine lengths plot. Dashed lines are the
saved reference lengths, not measured lengths. Length bounds, prior costs and
acceleration costs are exposed in the Ceres map/settings. The recording task now
reattaches saved full-body context and annotated previews automatically. Experiment
13 provides the corresponding known synthetic contraction check.
