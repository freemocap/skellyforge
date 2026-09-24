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

Noise has its own switch and a standard deviation in millimetres (0–20 per
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
poses the metacarpals in a fan (thumb -35°, index -12°, middle 0°, ring 10°, pinky
22°, mirrored on the other side). It exercises the normal FK/observation/recovery
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

Orange shows current Forge hydration/roll resolution from **saved processed
landmarks**. Gray shows their connected FK before fitting. Cyan shows connected
FK after joint shoulder fitting. Pink wireframe spheres are saved Forge landmarks;
smaller green solid spheres are Tracker keypoints. Both point layers start visible
and are wider than the bones. Hover any point or segment for its type and name;
overlapping objects list together. Keypoints come from the exact saved raw/filtered
3D channel named by the reconstruction metadata, with matching frames, timestamps,
coordinate frame and units. The viewer does not remap or refilter them.
Bones taper from a wide proximal end to a narrow distal tip.
This is a local Forge calculation experiment, not a replay or validation of the
production posthoc algorithm. The viewer's hydration, dimension fitting and
shoulder optimization must not be treated as production pipeline output.
Fixed dimensions are refitted once from the saved landmarks using current Forge
and the recording's saved scale-voting selection. Old and new lengths are recorded
in the source panel. Tracker mapping is not rerun: older mapped landmarks remain
the inputs. After shoulder fitting, arms and head retain their reconstructed world
rotations while their origins follow the connected skeleton.
The source panel records this provenance and hashes the local calculation files.

Use Play, the frame slider, last-quarter/whole-recording playback, layer toggles,
segment XYZ axes, and Focus shoulders to inspect the result. Missing segments
stay absent. Fitter status and shoulder-target distances appear for each frame;
these are not anatomical accuracy scores. Default modeling tolerances are 5 mm
for shoulder targets and 30 degrees for movable local rotations. Override with
`--position-tolerance-mm` and `--rotation-tolerance-deg`, then regenerate. These
are adjustable fitting preferences, not measured uncertainty or joint limits.

Saved data is opened read-only and checksum-checked. All per-frame calculations
are in memory; only the single HTML artifact is overwritten. Ctrl+C stops the
local server. Generation currently solves each frame offline and may take time.
