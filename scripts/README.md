# Local tools

- `generate_skeleton_viewer.py`: synthetic, closed-form hydration and forward
  synthesis viewer. Outputs ignored `.test-artifacts/viewers/skeleton_viewer.html`.
- `validate_python_wheel.py WHEEL`: checks the actual wheel for pure Python tags,
  scientific definitions, offline assets, and absence of native/fitting code.
- `generate_tracker_mapping_ratios.py`: historical cross-repository mapping
  authoring helper. Its relocation into FreeMoCap is a separate integration task.

Connected-fitting scripts, native build helpers, optimizer diagnostics and
experiments are preserved on `development-skelly-fit`.
