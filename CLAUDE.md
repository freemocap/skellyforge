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
│   └── skeleton_parts/                   the typed model
│       ├── anatomical_landmark.py        #   AnatomicalLandmark
│       ├── rigid_body_segment.py         #   RigidBodySegment + calculate_bases_for_segments
│       ├── skeleton_definition.py        #   SkeletonDefinition.from_yaml, global checks
│       ├── skeleton_yaml_loader.py       #   $include → lowercase → sided → reference frames
│       ├── rest_pose.py                  #   RestPose.from_yaml + forward kinematics
│       ├── skeleton_hydration.py         #   hydrate_segment / hydrate_skeleton
│       ├── skeleton_pose.py              #   SegmentPose / SkeletonPose / PoseSolution
│       ├── roll_resolution.py            #   ContinuousRollResolver (parallel transport)
│       ├── segment_length_estimation.py  #   per-subject length calibration
│       ├── landmark_name_resolver.py     #   alias → canonical, resolved once at load
│       ├── face_blendshapes.py           #   FaceBlendShapes (52 ARKit, NOT a component)
│       ├── segment_linkage.py            #   SegmentLinkage (placeholder, layer pending)
│       ├── kinematic_chain.py            #   KinematicChain (placeholder, layer pending)
│       └── naming.py
├── definitions/human_skeleton/           authored YAML (the static source of truth)
│   ├── human_skeleton.yaml               #   components: pelvis, spine, skull, arm, hand, leg, foot
│   ├── rest_pose.yaml                    #   the T-pose: parent tree + relative orientations
│   ├── face.yaml                         #   52 blendshapes (FaceBlendShapes, not the skeleton)
│   ├── default-vrm.gltf.json5            #   the VRM humanoid the rest pose was derived against
│   └── components/                       #   one .yaml per component
├── type_overloads.py                     FloatArray + the name-string vocabulary
└── tests/
```

**Status.** The whole human skeleton loads (61 segments / 124 landmarks / 52 face
blendshapes) and hydrates: `RestPose`, `hydrate_skeleton`, `ContinuousRollResolver` and
`estimate_segment_lengths` all work on the shipped definitions, and the viewer exercises
the whole path end to end. The **linkage and chain layers are still placeholders** — the
hierarchy currently lives in `rest_pose.yaml`'s `parent` / `connect_at` fields, and
reconciling those two is what building the linkage layer means.

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
- Layering: `core/math/` knows nothing about `core/skeleton_parts/`. Anything needing a
  skeleton or a pose belongs in `skeleton_parts`, however mathematical it is.

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
  declarations on the right side, so both sides get local `+x` toward the subject's left,
  `+y` up, `+z` forward — VRM's convention, right-handed on both. A right-handed triad
  cannot mirror all three axes, so `+x` is lateral on the left and medial on the right.
- **The rest pose has exactly one root, and every segment has an entry.** A `connect_at`
  must be owned by the parent, whether it was named or defaulted.
- **The feet stand on one flat ground plane.** Enforced by
  `test_both_feet_stand_on_one_flat_ground_plane`, which ties the heel orientation, the
  heel length and the foot orientation together so none can drift alone.
- **Most of the skeleton has free roll.** 56 of 61 segments are direction-only.
  `ContinuousRollResolver` supplies their roll by convention, not by measurement, and
  `SegmentPose.solved_by` says which you are looking at.
