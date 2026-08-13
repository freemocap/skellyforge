# CLAUDE.md

Guidance for Claude Code working in **skellyforge** (the standard human + kinematics, a sub-skelley of
the FreeMoCap polyrepo). Workspace orientation, repo boundaries, and the cross-repo model live in
[`../CLAUDE.md`](../CLAUDE.md) — read it first. **Never touch git** — the user owns commits; make changes
on disk and report stopping points.

## What this repo is (current architecture, 2026-08)

The **segment model** of the canonical standard human — VRM 1.0 rigid bodies whose reference geometry is
defined **directly from tracker keypoints** (no landmark layer; see the streaming-compatibility docs in
the freemocap repo). Layout:

```
skellyforge/
├── skellymodels/standard_human/
│   ├── segment_definition.py     # SegmentDefinition / RotationLimits / ParentAttachment — the authored unit
│   ├── segment_parts.py          # SegmentPart + compose_parts (prefixing + midline name-agreement fallback)
│   ├── body_part.py              # BODY_MIDLINE_PART + BODY_LIMB_PART (20 segments)
│   ├── hand_part.py              # HAND_PART (16 segments)
│   ├── face_part.py              # FACE_PART (8 segments: 3 VRM face bones + 5 face-detail)
│   ├── standard_human_model.py   # frozen StandardHuman + compose_standard_human() — 60 segments
│   ├── reference_geometry.py     # build_reference_geometry → ReferenceGeometry (T-pose, mirroring)
│   ├── human_bone_aliases.py     # BONE_ALIASES (60: vrm + unreal targets)
│   └── human_blendshapes.py      # 52 ARKit BlendShapeChannel declarations
└── kinematics/
    ├── quaternion_math.py        # RotationQuaternion (wxyz) + vectorized ops
    ├── coordinate_frame_ops.py   # basis construction, Kabsch, rotation_between_vectors
    ├── orientation_solver.py     # solve_frame_orientations — keypoint-declared two-tier twist
    ├── critically_damped_orientation.py  # the D3/D4 filter (per-segment state, time-constant based)
    ├── online_segment_lengths.py # SegmentLengthEstimator (window_seconds=None = unbounded posthoc)
    └── … (rigid_body_kinematics, skeleton_rigidifier, segment_lengths, inertial/)
```

**Still old-architecture** (pending Phase D/E of the plan — do NOT build on them): `skellymodels/models/`
+ `managers/` (the pre-standard-human model layer), `skellymodels/tracker_info/canonical_body.yaml` +
`canonical_hand.yaml`, `skellyforge/biomechanics/` (dead duplicate), `pipelines/dlc_pipeline.py` (dead).

## Commands

```bash
uv sync                                                      # install (incl. dev deps)
uv run --with pytest pytest skellyforge/tests/ -q -o addopts=""   # full suite — 94 passing as of 2026-08-13
```

`pytest` is NOT in the default env — always `uv run --with pytest`. `ruff` is not installed in the
default env either (no lint gate here yet).

## Conventions

- Frozen dataclasses with `__post_init__` `ValueError`s for fail-loud validation (no Pydantic in the
  hot path).
- Hot-path code: no per-frame allocations beyond necessary; dict-backed indices built once at load.
- The authored data carries **provenance comments** (sourced vs. estimated-with-said-so) — the honesty
  rules in `freemocap/docs/streaming-compatibility/phase-1/09-segment-model.md` §7.
- Boundary rule: skellyforge **never imports** skellytracker or freemocap — with one sanctioned
  exception. `skellymodels/standard_human/tracker_contract.py` imports skellytracker's `core.io`
  mapping machinery (mapping-path registry + `TrackerMapping` — base install only, no detector or
  onnxruntime/mediapipe extras) to validate the tracker→standard-human completeness contract at load
  time. The boundary deliberately leaks in exactly that one module; nothing else in skellyforge may
  import skellytracker, and `skellyforge/__init__.py` must NOT import `tracker_contract`.
- Vocabulary: **keypoint / segment** only. "Landmark" and "canonical" are retired (MediaPipe's own
  `PoseLandmarker`-style product names are the only exception).
