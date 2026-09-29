# Diagnostics

From the SkellyForge repository, run `uv run poe diagnostic-viewer` and open
http://127.0.0.1:8774/. Stop with Ctrl+C. This starts the supported viewer without
rerunning recording processing or automatically fitting missing saved results.

`generators/` contains saved-data views and report calculations; `checks/` contains
JavaScript UI checks; `tests/` verifies the diagnostic tools. Run the Python checks
with `uv run poe test-diagnostics`.

Source assets are shared with installed viewers in `skellyforge/tools/viewer/web`.
Generated HTML and media go under ignored `.test-artifacts/viewers/`. To serve
already-generated experimental pages, use `uv run poe diagnostic-solver-viewer-serve`
and visit http://127.0.0.1:8773/solver_viewer.html. Generate the selected experiment
first; serving a page does not run it. Legacy captures can still be selected with
the viewer's explicit `--captures` and `--media` options.

For generation and serving in one step, use `uv run poe experiment-solver-viewer --serve`.
The separate serve command reopens existing output without rerunning fits. Both
serve the same page; the root URL redirects to it. Stop with Ctrl+C.
