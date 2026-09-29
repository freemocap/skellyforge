# SkellyForge pipeline viewer

From the SkellyForge checkout:

```powershell
uv run --no-sync poe diagnostic-viewer
# First time, or after changing synthetic/fitting code:
uv run --no-sync poe diagnostic-viewer --prepare-synthetic-fit
```

Open http://127.0.0.1:8774/. Equivalent entry points are
`python -m skellyforge.tools.viewer` and the installed `skellyforge-viewer` command.
Standalone launch configures SkellyLogs and UTF-8 stdout/stderr. Poe sets
PYTHONUTF8=1 for its Python tasks. Importing the library leaves host streams and
logging handlers alone.

## Views

- **Synthetic humanoid:** shipped rest pose with independently switchable root,
  shoulder, elbow and head motion, hand spread, and optional known landmark noise.
  Uses the existing hydration and connected-reconstruction calculations. Cyan is
  the fixed-length connected reconstruction here, not Ceres.
- **Synthetic Ceres fit:** 60-frame all-motion fixture through the actual accepted
  `fit_human` function. Cyan is the Ceres output. It uses explicit identity mappings
  from synthetic keypoints to every landmark, an ideal full-landmark fixture,
  not a COCO observation model. The same generated inputs feed hydration and fit.
  Controls that would alter saved inputs are disabled. The captured Ceres window
  is available in **Ceres / Synthetic**.
- **Test / Sample recording:** accepted saved fit against source keypoints,
  mapped landmarks and saved segments, with independent visibility controls,
  hover identities, full playback and existing annotated video previews.
- **Ceres / Test:** captured native problem blocks, residual costs, manifolds,
  bounds, fixed history and actual connections. The graph and finished recording
  show different stages: a selected window's state versus finalized poses.

Synthetic connected axes use the fitted quaternions, not the hydration axes.
The synthetic diagnostic plots retain their explicit hydration labels; the
connected-origin plot is replaced with distances for the displayed Ceres fit.
Those distances are not errors against ground truth.

## Inputs and installation

The viewer reads existing accepted artifacts; it never runs calibration, tracking,
reconstruction, filtering or alignment. Missing or changed Parquet inputs disable
that recording view with an explicit error. Synthetic captures are invalidated
when the fixture, fitting Python code or native extension changes.

Defaults from the checkout are:

- `--results build/spine_positions`: sample candidate.json; test/candidate.json;
  synthetic.json and synthetic_window_0.json.
- `--captures scripts/.solver_inspections`: existing named native window captures.
- `--media scripts/.solver_media`: existing annotated JPEG previews.

These paths are explicit arguments when running an installed wheel from another
folder. Saved candidates contain the exact prepared Parquet path/hash. Default
prepared recordings remain under `~/freemocap_data/testing/prepared/...`, with
source recordings under `~/freemocap_data/recordings/...`; the viewer does not
copy or move them. Regenerating a recording fit remains a separate preparation
operation, currently `poe experiment-solver-viewer-spine-positions --recording test` or
`--recording sample`. Packaging that preparation command is still pending.

The synthetic and comparison Python implementations now live in this package;
old script entry points are compatibility wrappers. CMake installs the existing
JS/CSS/vendor sources unchanged into the wheel. There is no second solver or
separately authored Ceres graph. The HTTP server binds only to loopback and serves
only viewer assets, designated captures and designated JPEG previews.

## Remaining sign-off work

- A sparse synthetic observation fixture using the existing mapping machinery;
  the ideal full-landmark fixture does not establish sparse-tracker performance.
- Package recording-fit preparation and its Parquet reader so regeneration no
  longer needs repository scripts. Keep one numerical entry point.
- Visual review in a real browser. Automated geometry/control checks cannot
  establish layout quality; the agent's browser connection was unavailable.
- Platform wheel validation beyond this Windows development machine.

FreeMoCap stage/UI/Parquet integration and exporters remain separate repository
stages after Forge sign-off.


## Simple motion review

`uv run --no-sync poe diagnostic-motion-viewer` starts the new streamlined viewer at
http://127.0.0.1:8775/simple. The detailed viewer at port 8774 remains unchanged.
Both use the same accepted saved results; no fit is recomputed by switching
views. The new HTML/CSS/JS are under `tools/viewer/simple/`.

Choose Synthetic, Test or Sample. Fitted sticks use red for the left side, blue
for the right, and pale gray for central segments. Reference segments are a
bright amber overlay (hydrated segments for synthetic data, saved post-hoc segments
for recordings). Keypoints are solid teal dots; mapped landmarks are gold wire
spheres. Hover identifies each object. Synthetic keypoint display is disabled
because that saved visualization provides landmark inputs rather than a separate
tracker point set. Synthetic playback is the existing all-motion Ceres fixture.

Annotated video sits to the left with a draggable divider. Bottom controls offer
scrubbing, previous/next frame, play/pause, loop and playback speed. No solver
settings, tables, charts, axis annotations or residual graphs are added here;
the Detailed viewer link retains access to those diagnostics. This is a
presentation-only change, with unchanged fitted geometry and source data.

With the server running, `poe diagnostic-test-motion-viewer` checks all three datasets,
rendered stick endpoints, layer toggles, playback and stale video-image rejection.


Simple-view axes: **Skeleton axes** and **Reference axes** are independent
visibility toggles, separate from the sticks. X/Y/Z use red/green/blue. Each
helper has an origin sphere with a hover identity (solid pale fitted origins,
wire amber reference origins). Reference recording bases come from saved segment
rotations; fitted bases come from the saved Ceres wxyz quaternions. Synthetic
bases use their saved hydration/Ceres basis vectors. Roll is never reconstructed
from endpoints. Hand helpers are smaller to keep the fingers readable.
