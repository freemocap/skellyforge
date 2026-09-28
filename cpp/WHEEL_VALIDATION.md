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
uv pip install --python $wheelPython $forgeWheel.FullName

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

## Unresolved dependency distribution

Clean registry installation failed because wheel metadata requires `skellylogs`,
which is not available from the configured package registry. A diagnostic wheel
built without that requirement passed the numerical checks below, but removing
SkellyLogs violates the project's shared logging contract. That dependency removal
has been reverted. The diagnostic wheel is not the accepted distribution.

SkellyLogs lifecycle and sandbox support are under investigation in its own
repository. A subsequent validated wheel must retain SkellyLogs and explicitly
provide its supported installation source. Numerical validation below remains
useful evidence but does not close the logging or distribution work.

## Scope

This validates this machine's Windows x64 / CPython 3.12 build. It does not prove
that MSVC runtime DLLs are available on a clean Windows machine, validate Python
3.11, or establish Linux/macOS wheel portability. Those require separate CI and
platform testing before distribution. The inspector frontend is not yet packaged.


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
