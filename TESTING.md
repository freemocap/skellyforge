# Testing the reconstruction package

`development-streaming` contains closed-form geometry, skeleton hydration, person
scale estimation, forward synthesis, biomechanics and trajectory preparation.
Connected optimization, Ceres, native builds and fitting experiments are preserved
on `development-skelly-fit`, not required by this branch.

Run from the SkellyForge checkout using its own environment:

| Check | Command |
| --- | --- |
| Core regression suite | `uv run --no-sync poe test` |
| Prepared 222-frame test recording | `uv run --no-sync poe test-test-data` |
| Prepared 1,108-frame sample recording | `uv run --no-sync poe test-sample-data` |
| Both reference recordings | `uv run --no-sync poe test-all-data` |
| Offline synthetic hydration viewer | `uv run --no-sync poe viewer` |

Reference tests require PyArrow (`recording-viewer` extra) and prepared recordings.
Missing data fails explicitly; tests never download data or import FreeMoCap.
An explicit input may be selected with `--parquet PATH --sensor-group GROUP`.
Default discovery prefers `~/freemocap_data/testing/prepared/RECORDING/current/recordings/RECORDING/`.

The reference tests read mapped observations and the saved skeleton definition,
rehydrate every frame, and validate finite poses, unit rotations, positive scales
and absence handling. They print the input hash and assert the original file is
unchanged. These checks exercise the geometry on real data; they do not certify
anatomical accuracy or replace FreeMoCap's full pipeline and export acceptance tests.
Those integration checks run in FreeMoCap after the human commit/push handoff.

Keep both synthetic regressions and real-data runs when changing reconstruction.
Place generated artifacts in ignored `.test-artifacts/` or `build/`. The root
pytest configuration creates a separate scratch directory for each invocation.
Do not overwrite prepared source recordings or regenerate tracked HTML snapshots.

The package uses Flit and must build a `py3-none-any.whl`, without CMake, a C++
compiler or a bundled native extension. Validate release contents from a clean
source copy: ignored binaries left by previous native builds are not source.

The historical tracker-mapping authoring script and its development dependency
are separate architectural debt. This extraction does not expand those imports
or move integration ownership into Forge.

## Extraction validation (2026-10-08)

Both prepared recordings passed source and isolated-wheel hydration checks:
222 frames (216 hydrated, 6 absent) and 1,108 frames (1,079 hydrated, 29 absent).
The original Parquet hashes were unchanged. The standalone CLI and synthetic
viewer also ran successfully. Clean-source and direct-checkout wheels passed
`scripts/validate_python_wheel.py`; neither contained a native extension.

Final source suite: **643 passed, 1 skipped, 1 failed**. The core suite has a pre-existing failure:
`test_twist_backfill.py::test_pure_pronation_is_recovered_during_resolution`
recovers 0 degrees where its fixture expects 60. It fails identically in a source
snapshot of preserved commit `8a28aab7f57a6d7e268e5e67baca026bccaf6bcd` and in
the extracted package. It has not been skipped, weakened or fixed by this cleanup.
The suite's existing skip concerns the lack of adjacent rigid-fit pairs.

Locally, the old ignored `_native.cp312-win_amd64.pyd` and generated dependency
licenses were moved out of the package to `.test-artifacts/extraction/old-native/`.
Keep generated binaries outside `skellyforge/` when building a pure Python wheel.
The existing `.venv` was not synchronized; wheel checks used a separate environment.

After review, commit and push this repository on `development-streaming` before
refreshing FreeMoCap's dependency. FreeMoCap still needs its own fitting-removal
changes and full recording/reprocessing/export tests; this is the Forge handoff.
