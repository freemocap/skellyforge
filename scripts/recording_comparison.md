# Recording comparison viewer

From the SkellyForge root, run `uv run --no-sync poe recording-comparison`.
With the existing scripts server on port 8773, open
<http://127.0.0.1:8773/recording_comparison.html>.
If no server is running, serve the scripts directory with
`python -m http.server 8773 --bind 127.0.0.1 --directory scripts`.

This is a separate, simplified viewer for saved full-body recording fits.
The original `solver_viewer.html` retains synthetic experiments, time-series
plots, and the Ceres map. Neither fitting code nor source recordings are changed
by generating or using this viewer.

The generator reads the existing lab artifact, checks recording identity,
frame numbers, timestamps, skeleton identities and shared recording context,
then exports a smaller presentation dataset. It retains exact saved fitted
landmark coordinates, quaternions, translations and per-method reference geometry.
It neither reruns Ceres nor downloads or processes recording data. Regenerate
after adding new saved fits to the solver lab. The generated HTML is ignored by
Git; viewer source files live beside it.

## Controls

- Each saved fit is a row in the audit table. Columns show display controls,
  spine length policy, SC reference offsets, shoulder linkage policy, convergence
  and solve time, and target RMS. Prior scales describe residual strengths, not
  hard length limits. Check rows to overlay them; edit color/opacity or use Solo.
  Current-frame zero-spine warnings appear in the solve column.
- Details selects a shared inspector without changing visibility. It exposes
  readable settings, exact saved settings, recording path/hash, native binary
  hash, saved metrics and the Ceres report. Solve times are historical results;
  different native builds need not have comparable runtimes.
  Reset comparison restores the initial visible pair.
- When available, the starting pair is lower-SC relaxed shoulders with and
  without the equal-spine-length preference. Otherwise it is lower-SC exact
  versus relaxed shoulders.
  Recent shoulder comparisons appear first, followed by older spine cases.
- Bottom transport: play/pause, previous/next, first/last, looping, playback
  speed, timeline scrubbing and recording-frame entry. Space and arrow keys work
  when focus is outside an input/control.
- Cyan solid spheres are shared tracker keypoints. Solution-colored wire
  spheres are fitted landmarks. Bone shapes use each fit's saved geometry;
  relaxed shoulder connectors expose attachment separation. Hover labels identify
  the fit, segment, landmark or keypoint. Point coordinates refer to sphere centers.
- Geometry/reference layers and body-region filtering affect display only.
  Axes are optional, with labeled origin spheres. Hand markers use half-size radii.
- Annotated video sits to the left of the 3D view; the inventory occupies the
  right sidebar. Both vertical dividers resize their adjacent panels and accept
  left/right arrow keys. Annotated image previews use the
  existing frame-numbered cache, decode before display, and reject stale loads.
  While loading, the previous image remains and its unmatched status is explicit.

## Validation

`python -m pytest skellyforge/tests/test_recording_comparison.py -q`
checks immutable export and rejection of mismatched frame/context inputs.

`node scripts/check_recording_comparison.cjs` checks all saved fits and frames
using real Three geometry with a mocked DOM/WebGL: landmark positions, bone
endpoints, overlays, visibility, isolation, body filtering, playback, looping,
scrubbing, and out-of-order image completion. It does not assess visual quality.
An actual headless Edge screenshot was also inspected during development.
