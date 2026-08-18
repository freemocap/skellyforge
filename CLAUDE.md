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
├── skellymodels/standard_human/
│   ├── anatomical_landmark.py   # AnatomicalLandmark (name + anatomical_definition + rest_position + reference_frame)
│   ├── rigid_body_segment.py    # RigidBodySegment + AxisDefinition (length derived from rest_position)
│   ├── joint_linkage.py         # JointLinkage (parent + child + shared landmark, derived from parent edges)
│   ├── kinematic_chain.py       # KinematicChain (start → end path; the IK/FABRIK unit)
│   ├── human_skeleton.py        # HumanSkeleton.from_yaml (parts + sidedness + Y-mirroring + $include)
│   ├── face_blendshapes.py      # FaceBlendShapes (52 ARKit blendshapes; eyes/ears/nose are skull LANDMARKS)
│   ├── config_types.py          # typed YAML config shapes (cls(**data), no string-key indexing)
│   ├── definitions/             # flat part files: standard_human, pelvis, axial, arm, hand, leg, foot, face
│   ├── segment_definition.py    # OLD — SegmentDefinition (being retired)
│   ├── reference_geometry.py    # OLD — ReferenceGeometry / SegmentReferenceGeometry (being retired)
│   ├── rest_pose.py             # OLD — RestSegment / RestLandmark (wire projection)
│   └── … (body_part.py / hand_part.py / face_part.py / standard_human_model.py — OLD, being retired)
└── kinematics/
    ├── quaternion_math.py       # RotationQuaternion (wxyz) + vectorized ops
    ├── coordinate_frame_ops.py  # basis construction, Kabsch, rotation_between_vectors
    ├── orientation_solver.py    # solve_frame_orientations (being ported to Kabsch for 3+ landmarks)
    ├── critically_damped_orientation.py  # the D3/D4 filter (per-segment state, time-constant based)
    └── …
```

**Old-architecture (being retired — do NOT build on them):** `segment_definition.py` (`SegmentDefinition`),
`reference_geometry.py` (`ReferenceGeometry`/`SegmentReferenceGeometry`), `rest_pose.py`
(`RestSegment`/`RestLandmark`), the Python-authored `body_part.py` / `hand_part.py` / `face_part.py` /
`standard_human_model.py`, plus `skellymodels/models/` + `managers/`,
`tracker_info/*.yaml`, `skellyforge/biomechanics/` (dead), `pipelines/dlc_pipeline.py` (dead).

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
- Boundary rule: skellyforge **never imports** skellytracker or freemocap — with one sanctioned
  exception. `skellymodels/standard_human/tracker_contract.py` imports skellytracker's `core.io`
  mapping machinery (mapping-path registry + `TrackerMapping` — base install only, no detector or
  onnxruntime/mediapipe extras) to validate the tracker→standard-human completeness contract at load
  time. The boundary deliberately leaks in exactly that one module; nothing else in skellyforge may
  import skellytracker, and `skellyforge/__init__.py` must NOT import `tracker_contract`.
- Vocabulary: **keypoint / landmark / segment**. A **keypoint** is tracker-side — a point measured by
  a detector, triangulated to 3D. A **landmark** is model-side — a named point in a segment's local
  frame with a static rest definition and a per-frame world hydration (the mapping hydrates its name
  from tracker keypoints). A **segment** is a VRM-1.0-aligned rigid body (origin / orientation /
  length) whose landmarks are declared explicitly.
