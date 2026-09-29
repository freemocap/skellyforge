# Shared synthetic fixtures

These modules construct deterministic geometry, observations, and initial states
for core regression tests and exploratory generators. They do not invoke a solver,
render a viewer, choose a winning method, or write files.

- `chain.py`: three-segment chain or branch, controlled noise and missing frames.
- `tree.py`: five-segment tree with optional axial deformation.
- `torso.py`: authored torso geometry, four observations, and initialization.
- `geometry.py`: shared synthetic rotation and axial deformation helpers.

Keep expected truth separate from observed targets. Preserve seeds and numerical
values when reorganizing these fixtures. They are repository tooling, not part of
the installed SkellyForge API. Core tests enforce that neither this directory nor
the core test suite imports exploratory scripts.
