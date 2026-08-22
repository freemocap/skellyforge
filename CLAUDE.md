# CLAUDE.md

Guidance for Claude Code working in **skellyforge** (the standard human + kinematics, a sub-skelley of
the FreeMoCap polyrepo). Workspace orientation, repo boundaries, and the cross-repo model live in
[`../CLAUDE.md`](../CLAUDE.md) — read it first. **Never touch git** — the user owns commits; make changes
on disk and report stopping points.

## What this repo is (current architecture, 2026-08)

The **standard human**, rebuilt onto the seven-layer ontology (keypoint → mapping → landmark → segment →
linkage → chain → skeleton), defined in **YAML** and compiled into typed objects whose references are
objects, not strings. See `freemocap/current-work-plans/ontology.md` first. Layout:

```
skellyforge/
├── core/
│   ├── math/
│   │   └── geometry/                     # Phase 1 (done): the affine algebra + frames
│   │       ├── spatial_vectors.py        #   Point / Displacement / UnitVector
│   │       ├── rotation_quaternion.py    #   RotationQuaternion (wxyz) + vectorized ops
│   │       ├── transform_math.py         #   Transform (rotation + translation)
│   │       ├── point_ring_buffer.py      #   PointRingBuffer (streaming, zero-copy window)
│   │       ├── numeric_tolerances.py     #   shared tolerances, derived from one base
│   │       └── orthonormal_basis/        #   SpatialAxis, ReferenceFrameDefinition,
│   │                                     #   OrthonormalBasis, calculate_orthonormal_basis
│   ├── skeleton_parts/                   # Phase 2 (done): the typed model
│   │   ├── anatomical_landmark.py        #   AnatomicalLandmark
│   │   ├── rigid_body_segment.py         #   RigidBodySegment + calculate_bases_for_segments
│   │   ├── segment_linkage.py            #   SegmentLinkage (placeholder, linkage layer pending)
│   │   ├── kinematic_chain.py            #   KinematicChain (placeholder, chain layer pending)
│   │   ├── skeleton_definition.py        #   SkeletonDefinition.from_yaml / from_component_yaml
│   │   ├── skeleton_yaml_loader.py       #   $include → lowercase → sided → reference frames
│   │   ├── landmark_name_resolver.py     #   alias → canonical, resolved once at load
│   │   ├── face_blendshapes.py           #   FaceBlendShapes (52 ARKit blendshapes, NOT a component)
│   │   └── naming.py
│   └── post_processing/                  # filters + interpolation (register-based)
├── definitions/human_skeleton/           # authored YAML (the static source of truth)
│   ├── human_skeleton.yaml               #   components: pelvis, spine, skull, arm, hand, leg, foot
│   ├── face.yaml                         #   52 blendshapes (loaded by FaceBlendShapes, not the skeleton)
│   └── components/                       #   one .yaml/.yml file per component
├── type_overloads.py                     # FloatArray (beartype-checkable float64 ndarray)
└── tests/
```

**Pipeline status:** Phase 1 (geometry) and Phase 2 (static definitions) are **done** — the whole
human skeleton loads (61 segments / 124 landmarks / 52 face blendshapes). Phase 3 (hydration) is **not
started**: `rotation_quaternion.py`, `transform_math.py`, `PointRingBuffer`, and
`calculate_bases_for_segments` have no callers outside their own tests yet. Do not delete them. The old
`core/math/kinematics/` module (pre-bloodbath solvers on the retired `skellymodels` types) was deleted; the
rest pose and hydration solvers are being rebuilt from scratch on the new types.

## Commands

```bash
uv sync                                                      # install (incl. dev deps)
uv run --with pytest pytest skellyforge/tests/ -q -o addopts=""   # full suite
```

`pytest` is NOT in the default env — always `uv run --with pytest`. `ruff` is not installed in the
default env either (no lint gate here yet).

## Conventions

- Frozen dataclasses with `__post_init__` `ValueError`s for fail-loud validation (no Pydantic in the
  hot path).
- Hot-path code: no per-frame allocations beyond necessary; dict-backed indices built once at load.
- The authored data carries **provenance comments** (sourced vs. estimated-with-said-so) — the honesty
  rules in `freemocap/current-work-plans/archive/phase-1-work-plans/09-segment-model.md` §7.
- Boundary rule: skellyforge **never imports** skellytracker or freemocap - it must have a standalone existence and expose functionality that `freemocap` consumes
- Vocabulary: **keypoint / landmark / segment**. A **keypoint** is tracker-side — a point measured by
  a detector, triangulated to 3D. A **landmark** is model-side — a named point in a segment's local
  frame with a static rest definition and a per-frame world hydration (the mapping hydrates its name
  from tracker keypoints). A **segment** is a VRM-1.0-aligned rigid body (origin / orientation /
  length) whose landmarks are declared explicitly.
