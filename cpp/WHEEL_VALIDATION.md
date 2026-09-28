# Installed-wheel validation

A checkout passing tests is not proof that an installed wheel works. This check
builds the wheel, installs its declared runtime dependencies into a separate
virtual environment, and runs Python with `-I` from outside the repository.
No editable installs, sibling paths, or `PYTHONPATH` are involved.

## Repeat the check (Windows / Python 3.12)

Run the first commands from the SkellyForge checkout with native and development
tools already installed. The input preparation also needs the recording-viewer
extra, the prepared test recording, and the saved accepted candidate at
`build/spine_positions/test/candidate.json`.

```powershell
uv run --no-sync poe build-wheel
uv run --no-sync poe wheel-validation-inputs

$forgeRepo = (Get-Location).Path
$wheelCheck = Join-Path $env:TEMP 'skellyforge-wheel-check'
# Use a fresh directory/environment for a clean-install test.
uv venv --python 3.12 "$wheelCheck\.venv"
$wheelPython = Join-Path $wheelCheck '.venv\Scripts\python.exe'
$forgeWheel = Get-ChildItem "$forgeRepo\dist\*.whl" |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
# SkellyLogs is currently distributed from GitHub, not the package registry.
# This published revision includes the explicit package-discovery fix.
uv pip install --python $wheelPython $forgeWheel.FullName `
    'skellylogs @ https://github.com/freemocap/skellylogs/archive/85a1382cbb56f1f9f1c013ef42369eec2afb495f.zip'

Push-Location $wheelCheck
try {
    & $wheelPython -I "$forgeRepo\scripts\validate_installed_wheel.py" run `
        --inputs "$forgeRepo\build\wheel_validation\inputs.json.gz" `
        --report "$forgeRepo\build\wheel_validation\report.json"
} finally {
    Pop-Location
}
```

Use the wheel matching the test interpreter and platform; the example assumes
that the most recently built wheel is that wheel. The runner checks package and
extension import locations against the isolated environment. It imports no
repository adapters during the installed-package run. Its `prepare` command
serializes only in-memory solver inputs and baseline data; it checks the source
recording SHA and never changes prepared recordings. The compressed fixture and
reports overwrite fixed paths under ignored `build/wheel_validation`.

After validation, the temporary environment is disposable. Preserve the report
and wheel hash if retaining evidence for a release. Do not commit wheels,
virtual environments, real recording fixtures or generated logs.

## What is checked

- The wheel includes the Python fitting implementation and loadable native module.
- Normal dependency resolution works without `--no-deps` or manually supplied
  undeclared runtime packages.
- A synthetic connected window sequence matches checkout results and exposes
  actual named parameter/residual inspection, including fixed history.
- The complete 222-frame prepared test recording fits through installed
  `fit_human`, comparing all world quaternions, translations and axial lengths
  against the saved accepted result. Tolerances are 1e-8 absolute / 1e-12 relative
  to allow separate native builds; actual maximum differences are reported.
- Native and wall durations are reported. This is a correctness/packaging test,
  not a controlled performance benchmark.

## Dependency distribution

Clean registry installation failed because wheel metadata requires `skellylogs`,
which is not available from the configured package registry. A diagnostic wheel
built without that requirement passed the numerical checks below, but removing
SkellyLogs violates the project's shared logging contract. That dependency removal
has been reverted. The diagnostic wheel is not the accepted distribution.

The published SkellyLogs package-discovery fix now installs successfully from
the archive above. The accepted wheel retains its SkellyLogs requirement.
Consumers must supply that source until a registry release is available; core
already declares a Git source. No sibling checkout or dependency removal is
needed. The historical diagnostic result below is superseded by the final
validation checkpoint.

## Scope

This validates this machine's Windows x64 / CPython 3.12 build. It does not prove
that MSVC runtime DLLs are available on a clean Windows machine, validate Python
3.11, or establish Linux/macOS wheel portability. Those require separate CI and
platform testing before distribution. The inspector and simple-viewer frontends
are now packaged, including their shared JavaScript and offline vendor assets.


## Local result: 2026-09-28

Windows x64, CPython 3.12.14, Ceres 2.2.0: the diagnostic wheel with the
logging requirement removed installed successfully. This is not the approved
dependency configuration.
The synthetic sequence and named native inspection passed. All 222 recording
frames matched saved world quaternions, translations and axial lengths exactly
(maximum absolute differences all zero). Native time: 91.21 s; wall: 93.85 s.
Both Python and extension import paths were inside the isolated site-packages.
The disposable environment was removed after validation; the wheel, compressed
inputs, report and artifact SHA are retained in ignored build/dist directories.
Six focused extension/inspection tests also passed in the development environment.


## Supported-viewer wheel checkpoint (2026-09-28)

Built the Windows CPython 3.12 wheel with the required SkellyLogs dependency
retained. It contains the supported viewer Python modules, HTML, shared JS/CSS
and offline vendor assets (23 entries under tools/viewer). Packaged Python/HTML
were byte-compared with the source checkout. Source runtime checks pass.

A fresh isolated environment could not install the current published SkellyLogs
archive at commit `8a11690f11ca88e54adf8da12a88e3e7aeddb5c8`: setuptools reports
multiple top-level packages `notes` and `skellylogs`. No dependency was removed
or replaced by a sibling checkout. This requires explicit package discovery in
SkellyLogs, human commit/push, then a repeat clean install and numerical check.
Therefore installed-wheel acceptance remains OPEN. The generated wheel is not
claimed to be validated in a clean runtime environment.

## Final installed-package checkpoint (2026-09-28)

Supersedes the blockers above. Rebuilt the current viewer/fitting wheel and
installed it with the published SkellyLogs revision
`85a1382cbb56f1f9f1c013ef42369eec2afb495f`. Both validation processes exited zero.
Python and native imports came from isolated site-packages, outside the checkout.

| Recording | Frames | Native time | Fit wall time | Maximum quaternion / translation / axial-length differences |
| --- | ---: | ---: | ---: | --- |
| Test | 222 | 72.08 s | 74.23 s | 0 / 0 / 0 |
| Sample | 1108 | 45.02 s | 57.61 s | 0 / 0 / 0 |

These compare the accepted saved candidates, not anatomical ground truth. The
recordings have different timestamps and optimization trajectories; these are
individual correctness runs, not a frame-count scaling benchmark. No fit settings
or residuals changed. Fit wall time excludes input deserialization and installation.

To prepare the sample fixture, run from the checkout:

```powershell
python -m scripts.validate_installed_wheel prepare --recording sample --inputs build/wheel_validation/sample-inputs.json.gz
```

Then use the same isolated `run` command with that input and a separate report.
The sample candidate is `build/spine_positions/candidate.json`. This is the actual
1108-frame prepared sample currently on disk; no assumption of 2000 frames is made.

Installed viewer HTTP checks passed for both real-data payloads, simple HTML/JS/CSS,
shared rendering assets, native inspector assets and Three.js. The installed
synthetic generator produced a fresh 60-frame fit and native inspection; its source
hashes passed reload validation. Existing source synthetic caches correctly become
stale when the native build changes. Four supported-viewer tests also passed.
HTTP checks do not establish browser layout quality.

Wheel SHA256:
`6f1a032da31cd642a4442148129c4cb374eff524a823c9d658cd28a1e283eee2`.
Reports and logs are in the workspace's ignored `.test-artifacts/`, named
`forge-wheel-test-report.json`, `forge-wheel-sample-report.json` and corresponding
`.log` files. This closes the local Windows installed-package check. Platform CI,
supported recording-preparation packaging, and FreeMoCap integration remain
separate work.
