# CLAUDE.md

Guidance for Claude Code working in **skellyforge** (the standard human + kinematics, a
sub-skelley of the FreeMoCap polyrepo). Workspace orientation, repo boundaries, and the
cross-repo model live in [`../CLAUDE.md`](../CLAUDE.md) — read it first. **Never touch
git** — the user owns commits; make changes on disk and report stopping points.

## What this repo is (current architecture)

The **standard human**, built on the seven-layer ontology (keypoint → mapping → landmark →
segment → linkage → chain → skeleton), defined in **YAML** and compiled into typed objects
whose references are objects, not strings. See `freemocap/current-work-plans/ontology.md`
first. Layout:

```
skellyforge/
├── core/
│   ├── math/
│   │   ├── geometry/                     the affine algebra + frames
│   │   │   ├── spatial_vectors.py        #   Point / Displacement / UnitVector
│   │   │   ├── rotation_quaternion.py    #   RotationQuaternion (wxyz) + vectorized ops
│   │   │   ├── transform_math.py         #   Transform (rotation + translation)
│   │   │   ├── point_ring_buffer.py      #   PointRingBuffer (streaming, zero-copy window)
│   │   │   ├── numeric_tolerances.py     #   every tolerance, derived from one base
│   │   │   └── orthonormal_basis/        #   SpatialAxis, ReferenceFrameDefinition,
│   │   │                                 #   OrthonormalBasis, calculate_orthonormal_basis
│   │   └── kinematics/                   closed-form solvers on observed positions
│   │       ├── rigid_point_set.py        #   Kabsch fit + RigidPointSet
│   │       └── coordinate_frame_ops.py   #   shortest-arc rotation, default perpendicular
│   ├── skeleton/                         the typed model
│   │   ├── skeleton_definition.py        #   SkeletonDefinition.from_yaml, global checks
│   │   ├── skeleton_pose.py              #   SegmentPose / SkeletonPose / PoseSolution
 │   │   ├── components/                   #   the leaf model types + naming/resolver helpers
 │   │   │   ├── anatomical_landmark.py    #   AnatomicalLandmark
 │   │   │   ├── rigid_body_segment.py     #   RigidBodySegment + calculate_bases_for_segments
 │   │   │   ├── segment_basis_solver.py   #   per-segment basis solver
 │   │   │   ├── landmark_name_resolver.py #   alias → canonical, resolved once at load
 │   │   │   ├── face_blendshapes.py       #   FaceBlendShapes (52 ARKit, NOT a component)
 │   │   │   └── naming.py                 #   snake_case + alias rules
 │   │   ├── linkage/                      # the joint layer: JointDefinition, EulerConvention,
 │   │   │                                 #   relative_orientation, JointPose + provenance
 │   │   ├── chain/                        # the multi-segment layer: KinematicChain declarations,
 │   │   │                                 #   forward synthesis, two-bone IK + FABRIK,
 │   │   │                                 #   twist backfill from rigid terminals
 │   │   ├── pose/                         #   hydration + rest pose + roll resolution
 │   │   │                                 #   (anchored secondary axes w/ transport fallback)
│   │   ├── loading/                      #   the YAML loader pipeline
│   │   │   ├── component_building.py     #   $include → lowercase → sided → reference frames → objects
│   │   │   ├── include_resolution.py     #   $include
│   │   │   ├── name_lowercasing.py       #   lowercase
│   │   │   ├── sided_expansion.py        #   left_/right_ prefixing
│   │   │   └── reference_frame_building.py  #   reference frames
│   │   └── pose/                         #   pose + hydration
│   │       ├── rest_pose.py              #   RestPose.from_yaml + forward kinematics
│   │       ├── hydration.py              #   hydrate_segment / hydrate_skeleton
│   │       ├── roll_resolution.py        #   ContinuousRollResolver (parallel transport)
│   │       └── segment_length_estimation.py  # per-subject length calibration
│   └── biomechanics/                     the derived layer (mass, CoM, inertia)
│       ├── anthropometric_parameters.py  #   de Leva (1996) masses + radii of gyration
│       ├── center_of_mass.py             #   per-segment COM = weighted landmark sums
│       ├── segment_mapping.py            #   61 skeleton segments → 16 anatomical segments
│       ├── segment_inertia.py            #   per-segment inertia tensor
│       ├── composite_inertia.py          #   whole-body CoM + inertia (parallel axis)
│       ├── ground_reference.py           #   CoP / XCoM / CMP
│       └── derived_kinematics.py         #   CoM velocity / acceleration
├── definitions/human_skeleton/           authored YAML (the static source of truth)
│   ├── human_skeleton.yaml               #   components: pelvis, spine, skull, arm, hand, leg, foot
│   ├── rest_pose.yaml                    #   the T-pose: per-segment relative orientations only
│   ├── anthropometric_parameters.yaml    #   de Leva (1996) masses + radii of gyration
│   ├── center_of_mass.yaml               #   per-segment COM = weighted landmark sums
│   ├── face.yaml                         #   52 blendshapes (FaceBlendShapes, not the skeleton)
│   ├── default-vrm.gltf.json5            #   the VRM humanoid the rest pose was derived against
│   └── components/                       #   one .yaml per component
├── type_overloads.py                     FloatArray + the name-string vocabulary
└── tests/
```

**Status.** The whole human skeleton loads (61 segments / 124 landmarks / 52 face blendshapes /
60 joints / 5 chains) and hydrates: `RestPose`, `hydrate_skeleton`, `ContinuousRollResolver` and
`estimate_segment_lengths` all work on the shipped definitions, and the viewer exercises
the whole path end to end. Landmark coordinates are authored as **body-height proportions**
(`H = 1.0` = floor-to-skull-top), not millimetres. The **linkage layer is built**: `human_skeleton.yaml`'s
`joints:` section is the authoritative topology (bilateral joints authored once via `sided: true`),
`relative_orientation` + per-joint euler conventions decompose to named angles, and every `JointPose`
carries input provenance. The **chain layer's declarations and forward synthesis are built**:
declared chains compile with contiguity validation, and `synthesize_pose` walks joint angles ->
whole-body poses, gated by the FK-closure tests. Roll resolution anchors to parent-origin directions
at the skeleton level (deterministic per frame) and falls back to parallel transport when no anchor
exists; twist backfill fills a chain's proximal roll from its measured rigid-fit terminal.
Chain IK is built: closed-form two-bone solving and iterative FABRIK, both fail-loud on unreachable
targets and iteration exhaustion. Every segment declares its `anatomical_segment` (de Leva chunk)
in its component YAML — `segment_mapping.py` reads the declarations rather than a hardcoded dict.
The spine/thorax redesign is landed (`sacrolumbar`/`thoracic`/`cervical_spine`). Next work: the
**body-fitting step** that scales the proportional template to measured millimetres, then the
pelvis split, face component, and finger coupling ratios.

## Commands

```bash
uv sync                                                            # install (incl. dev deps)
uv run --with pytest pytest skellyforge/tests/ -q -o addopts=""     # full suite (~3 s)
uv run python -m skellyforge                                        # smoke test: load and describe
python scripts/generate_skeleton_viewer.py                          # regenerate the viewer
```

`pytest` is NOT in the default env — always `uv run --with pytest`. `ruff` is not
installed in the default env either (no lint gate here yet).

## Conventions

- Frozen dataclasses with `__post_init__` `ValueError`s for fail-loud validation (no
  Pydantic anywhere — it is not a dependency).
- **Fail loudly, always.** Nothing here repairs, skips, or swallows. If a value is missing
  or malformed, raise, and name the thing that was wrong. A silently short result or a
  bare `except` is a bug even when the shipped data never triggers it.
- Hot-path code: no per-frame allocations beyond necessary; dict-backed indices built once
  at load.
- Every numeric tolerance comes from `numeric_tolerances.py`. No bare `1e-10` in a
  comparison.
- Keyword-only arguments throughout, including module-level functions.
- The authored data carries **provenance comments** (sourced vs. estimated-with-said-so).
- Boundary rule: skellyforge **never imports** skellytracker or freemocap — it must have a
  standalone existence and expose functionality that `freemocap` consumes.
- Layering: `core/math/` (pure algebra) → `core/skeleton/` (the typed model) →
  `core/biomechanics/` (the derived layer). Each layer reads the ones below it and is
  never imported by them; the skeleton never imports biomechanics, and math never
  imports either.
- **Canonical coordinate system: Blender's** — right-handed, `+x` right, `+y` forward,
  `+z` up, ground plane at `z = 0`. Authored `local_position`s are **body-height proportions**
  (`H = 1.0` = floor-to-skull-top), so the template is body-agnostic; the body-fitting step
  scales them to measured mm. Every other convention (VRM/glTF, ROS, ISB, Unreal, Unity, and any a
  user defines) lives in `definitions/coordinate_systems/coordinate_systems.yaml` and is
  entered or left only at an I/O boundary, through `CoordinateSystemTransform`.

## Vocabulary

A **keypoint** is tracker-side — a point measured by a detector, triangulated to 3D. A
**landmark** is model-side — a named point in a segment's local frame with a static rest
definition and a per-frame world hydration. A **segment** is a VRM-1.0-aligned rigid body
(origin / orientation / length) whose landmarks are declared explicitly.

## Invariants worth knowing before you change anything

- **A segment's own origin landmark sits at `[0, 0, 0]`.** Enforced in
  `RigidBodySegment.__post_init__`. `length`, the rest pose's forward kinematics and
  hydration all read local positions as being in the segment's frame, which is only true
  if this holds.
- **A fully-specified segment solves to the identity basis from its own rest positions.**
  Enforced by `test_every_fully_specified_segment_solves_to_its_own_authoring_frame`. This
  is what keeps `reference_geometry` (Gram-Schmidt) and `local_position` (Kabsch, forward
  kinematics) from being two different answers to "which way does this segment face".
- **Left and right local frames mean the same thing.** The loader negates x-axis
  declarations on the right side, so both sides get local `+x` toward the subject's right,
  `+y` forward, `+z` up — Blender's convention, right-handed on both. A right-handed triad
  cannot mirror all three axes, so `+x` is medial on the left and lateral on the right.
- **The rest pose has exactly one root, and every segment has an entry.** A `connect_at`
  must be owned by the parent, whether it was named or defaulted.
- **The feet stand on one flat ground plane.** Enforced by
  `test_both_feet_stand_on_one_flat_ground_plane`, which ties the heel orientation, the
  heel length and the foot orientation together so none can drift alone.
- **Most of the skeleton has free roll.** 58 of 61 segments are underspecified (only
  `pelvis`, `thoracic`, `skull` name a secondary axis). `ContinuousRollResolver` supplies the
  rest by convention — anchored secondary axes with parallel-transport fallback and twist
  backfill — and `SegmentPose.solved_by` says which you are looking at.
