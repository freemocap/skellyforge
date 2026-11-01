# SkellyForge agent guidance

Forge owns authored skeleton geometry, reconstruction and kinematics. Skeletons
are generic: the human is one definition, not the contract. Detector keypoints
and mapping evaluation belong to Tracker; Forge must work without imports from
Tracker, Cam or Core, including tests. Core owns cross-component integration.

## Task continuity

At milestones, checkpoint the objective, decisions and rationale, exact checkout,
existing interfaces, relevant definitions/plan stage, tests actually run and next
action. Include targeted source paths and the question each answers. After
compaction or handoff, read the checkpoint, this file and only the relevant
architecture sections before editing; inspect the current implementation and diff.

- Ownership, layering and invariants: [CLAUDE.md](CLAUDE.md), especially
  Conventions and Invariants; dated status counts are not fresh evidence.
- Authored geometry: [definitions](skellyforge/definitions/) and
  [skeleton model](skellyforge/core/skeleton/). Preserve the distinction between
  reference-unit geometry and measured coordinates, with explicit scale.
- Coordinate boundaries: [coordinate definitions](skellyforge/definitions/coordinate_systems/coordinate_systems.yaml)
  and [conversion implementation](skellyforge/core/math/geometry/coordinate_systems/).
- Hydration and scale: [pose implementation](skellyforge/core/skeleton/pose/).
- Validation: [TESTING.md](TESTING.md) and current `pyproject.toml`/task bodies;
  historical branch labels and old environment instructions need verification.

Search existing math and typed model interfaces before adding another solver or
definition. Continue authorized work once the next action's meaning and owner
are clear. Ask only when targeted inspection cannot resolve consequential intent.
Follow containing workspace policy when present; preserve edits and leave Git
state to the human. Use ignored or external temporary output paths. Documentation
commands do not grant dependency-installation or environment-change permission.
