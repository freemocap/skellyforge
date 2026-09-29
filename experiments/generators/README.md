# Solver comparison generators

Synthetic and recording-based experiments live here. They use production fitting
primitives or the accepted fitter with explicitly selected experimental options.
They are not run by `poe test` or the standard-data tests.

`uv run poe experiment-partial-observations --serve` opens a controlled dropout
experiment at http://127.0.0.1:8777/. Compare raw keypoints, gap-filled inputs and
complete connected output, including exit/re-entry and never-observed segments.
See the [trajectory contract](../../skellyforge/core/trajectories/README.md).
The page is saved under `.test-artifacts/viewers/partial_observations/`.

Use the `experiment-*` Poe tasks in `experiments/poe_tasks.toml`. Comparisons that
read a preceding result under `build/` require that experiment to have been run;
missing prerequisites remain explicit errors. Do not run every variant as a test.

Shared deterministic inputs live in `test_support`. Shared web assets live in
`skellyforge/tools/viewer/web`. New HTML, media and inspector captures go to
ignored `.test-artifacts/viewers/`; numeric experiment outputs stay under ignored
`build/`. Legacy `scripts.*` entry points remain compatibility aliases.

Historical pages and reports under `scripts/` remain unchanged. Opening the
supported diagnostic viewer does not regenerate them or start experiments.

`uv run poe experiment-hand-fit --serve` compares the current synthetic fit with
the same fixed lengths and observations under greatly weakened angular smoothing.
Open http://127.0.0.1:8776/ to inspect both fits. Measurements and standalone pages
are saved in `.test-artifacts/hand-fit-audit/`. This audits full landmark coverage
and target bone lengths; it does not change production tuning or establish a
solution for partially observed recording data.

Start the synthetic lab with `uv run poe experiment-solver-viewer --serve`, then
open http://127.0.0.1:8773/. Omit `--serve` to generate only. Use
`uv run poe diagnostic-solver-viewer-serve` to reopen saved output without refitting.
