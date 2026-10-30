# Local tools

- `generate_skeleton_viewer.py`: synthetic, closed-form hydration and forward
  synthesis viewer. Outputs ignored `.test-artifacts/viewers/skeleton_viewer.html`.
- `validate_python_wheel.py WHEEL`: checks the actual wheel for pure Python tags,
  scientific definitions, offline assets, and absence of native/fitting code.

Tracker mapping authoring lives in FreeMoCap because it composes Tracker mappings
with Forge skeleton definitions. From a FreeMoCap installation, run
`python -m freemocap.tools.generate_tracker_mapping_ratios <mapping.yaml> [...]`.
The command prints proposed ratios and errors without modifying mapping files.

Connected-fitting scripts, native build helpers, optimizer diagnostics and
experiments are preserved on `development-skelly-fit`.
