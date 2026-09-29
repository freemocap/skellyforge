# Tests, diagnostics, and experiments

Run commands from the **SkellyForge repository**, `project/repos/skellyforge`,
using its existing environment. Prefix Poe tasks with `uv run --no-sync poe`.

| Purpose | Command | Inputs and outcome |
| --- | --- | --- |
| Core regression tests | `uv run --no-sync poe test` | Synthetic fixtures; geometry and solver contracts; pass/fail |
| Standard fit on test data | `uv run --no-sync poe test-test-data` | Prepared `test_data`, 222 frames; accepted `fit_human` only |
| Standard fit on sample data | `uv run --no-sync poe test-sample-data` | Prepared `sample_data`, 1,108 frames; accepted `fit_human` only |
| Both reference recordings | `uv run --no-sync poe test-all-data` | Test data first; a failure stops sample data |
| Diagnostic tooling tests | `uv run --no-sync poe test-diagnostics` | Readers, viewers, and diagnostic calculations |
| Experimental tooling tests | `uv run --no-sync poe test-experiments` | Comparison machinery and experimental assertions; some need local recordings |
| All three regression suites | `uv run --no-sync poe test-all` | Core, diagnostic tooling, experimental tooling; no full reference fits |

`poe test` does not download recordings, open viewers, or sweep solver options.
The native extension must already be built. Reference tests additionally need
the `recording-viewer` extra (PyArrow) and prepared data from FreeMoCap. Missing
inputs fail explicitly. SkellyForge reads data; it never imports FreeMoCap or
reruns calibration, tracking, triangulation, alignment, or person scaling.

Reference tests call the public accepted fitter with its defaults, check complete
finite returned segment state and unit quaternions, and verify the source file
is unchanged. They do not use the experimental 50 mm spine-distance gate, claim
anatomical accuracy, or require arbitrary coverage percentages. Numerical contracts
and regression cases belong in tests; error distributions belong in diagnostics;
changes to solver choices belong in experiments until deliberately accepted.

Default data discovery prefers FreeMoCap's `testing/prepared/.../current` recording
and falls back to the original recording directory. The input path and SHA256 are
printed. To select a particular prepared recording or disambiguate its group:

```powershell
uv run --no-sync poe test-test-data --parquet C:\data\recording_data.parquet --sensor-group GROUP
```

No fit output is saved by these tests. They do not replace FreeMoCap's accepted
recording. Each pytest invocation gets its own temporary directory beneath
`.test-artifacts/pytest/run-*`, avoiding the shared Windows `pytest-of-USER`
directory and collisions between concurrent runs. Explicit `--basetemp` overrides
are respected. Run directories remain available for debugging; old runs are not
automatically pruned. Any persistent generated artifacts
must be ignored. Pytest's shared cache is disabled by default to avoid the same
cross-session permission problem in `.pytest_cache/`. Other generated files
must go under ignored `.test-artifacts/` or `build/`, or outside the repository.
Check ignore rules before creating outputs. Historical tracked HTML snapshots
remain for now; do not regenerate them as part of normal test execution.

## Locations and exploratory work

- `skellyforge/tests/`: core regression tests (the default pytest collection).
- `tests/reference/`: explicit full-recording tests.
- `diagnostics/tests/`: regression checks for diagnostic tooling.
- `experiments/tests/`: regression checks for exploratory tooling.
- `diagnostics/poe_tasks.toml`: `diagnostic-*` viewer/report commands.
- `experiments/poe_tasks.toml`: `experiment-*` solver comparisons.

Existing exploratory scripts and results are retained under `scripts/`. Their
Poe entry points now have explicit prefixes; for example,
`solver-viewer-spine-ratios` becomes `experiment-solver-viewer-spine-ratios`.
Python module paths remain compatible. The shared recording reader now lives in
`skellyforge/tools/recording_data.py`; `scripts.recording_data` remains an alias.
Tests of individual native primitives remain core tests even where the primitive
also supports an experiment. Moving them out merely because they have many
parameter cases would remove useful regression coverage.

## Remaining work, in order

1. Finish disentangling shared synthetic fixtures from experimental scripts and
   move generators/assets together into their owning diagnostic or experiment
   directories. Audit relative paths before moving them; keep historical results.
2. Resolve **complete skeleton output from partial observations** in SkellyForge.
   Keypoints may be absent; modeled segments must retain all their landmarks.
   Specify rest-relative completion, observed/inferred provenance, and the wholly
   unobservable case. Do not feed inferred positions back as measured evidence.
   This is a separate mathematical change, not a workaround in the dataset test.
3. Revalidate the two standard-data tests against newly processed FreeMoCap data.
   Fresh test-data currently exposes `KeyError: pelvis_origin` in position-prior
   preparation. Older prepared data may pass; that does not resolve this blocker.
4. After the human commit/push handoff, validate FreeMoCap's adapter against the
   completed-skeleton contract and finish end-to-end dataset acceptance there.

The suite reorganization does not alter fitting mathematics or decide the missing
observation policy. Experimental quality thresholds remain explicitly experimental.
