# skellyforge

**The standard human.** A skeleton definition — landmarks, rigid-body segments, and the
reference frames they realize — authored in YAML and compiled into typed objects, plus the
closed-form math that hydrates it from observed 3D positions.

Part of the [FreeMoCap](https://freemocap.org) project. skellyforge never imports
skellytracker or freemocap: it stands alone, and freemocap consumes it.

```bash
uv sync
uv run python -m skellyforge          # load the standard human and describe it
uv run --with pytest pytest skellyforge/tests/ -q -o addopts=""
```

`python -m skellyforge` prints:

```
skeleton `human`
  segments            61
  landmarks           124
  face blendshapes    52
  fully specified     3 (roll pinned by a secondary landmark)
  underspecified      58 (roll resolved downstream)
rest pose `human`
  root segment        pelvis
  resolved landmarks  124
```

## What is in here

```
skellyforge/
├── core/
│   ├── math/
│   │   ├── geometry/            the affine algebra and reference frames
│   │   └── kinematics/          closed-form solvers: Kabsch, shortest-arc rotation
│   └── skeleton_parts/          the typed model: landmarks, segments, skeleton, pose
├── definitions/human_skeleton/  the authored YAML — the static source of truth
└── tests/
```

### Vocabulary

A **keypoint** is tracker-side: a point measured by a detector and triangulated to 3D. A
**landmark** is model-side: a named point in a segment's local frame, with a static rest
definition and a per-frame world position. A **segment** is a VRM-aligned rigid body —
origin, orientation, length — whose landmarks are declared explicitly.

### Coordinate system

VRM convention, right-handed, millimetres, 50th-percentile adult:
`+x` is the subject's **left**, `+y` is **up**, `+z` is **forward**.

Sided structures are authored once for the left side; the loader emits `left_*` and
`right_*`, mirrors x, and negates x-axis declarations on the right so both sides' local
frames mean the same thing. Sidedness is a prefix, never a `.L`/`.R` suffix.

## The three types every value is one of

`Point`, `Displacement` and `UnitVector` wrap the same `(..., 3)` float64 array and are
deliberately not interchangeable, because they obey different algebras:

```
Point        - Point        -> Displacement
Point        + Displacement -> Point
Displacement + Displacement -> Displacement
Point        + Point        -> TypeError
```

Rotating a displacement just rotates it; rotating a point requires centering it first,
and that is the bug the distinction exists to catch. Leading dimensions are free, so one
landmark is `(3,)`, a trajectory is `(num_frames, 3)`, and every operation is vectorized
over whatever is in front — batching is the normal case here, not a special one.

## Hydration

`hydrate_skeleton` recovers each segment's pose from observed landmark positions, in
closed form, one frame at a time.

- A segment whose own landmarks **span a plane** is fit as a rigid body with Kabsch over
  every one of them that is observed, which pins the full orientation. On the shipped
  human that is 5 segments of 61.
- Every other segment yields a **direction** — two landmarks fix the long axis and leave
  roll about it free. `ContinuousRollResolver` supplies that roll by parallel transport,
  carrying the previous frame's secondary axis forward so the result is continuous across
  frames and lag-free.

Every `SegmentPose` records which of the two produced it, because "measured" and
"well-behaved" are different claims about the world.

## Conventions

- Frozen dataclasses that validate in `__post_init__` and raise. **The loader repairs
  nothing** — unknown keys, malformed geometry, missing origins, half-specified axes and
  segments with no reference geometry all fail loudly, at load, naming what was wrong.
- Data entering the type system is validated; data derived within it is trusted
  (`from_array` versus `from_prevalidated_array`). That split is what keeps a streaming
  rolling window O(1) per frame instead of O(window).
- Numeric tolerances live in one file, `numeric_tolerances.py`, each derived from one base.
- The authored YAML carries **provenance**: what a number is, and whether it was sourced
  or estimated.

## Viewer

```bash
python scripts/generate_skeleton_viewer.py   # writes scripts/skeleton_viewer.html
```

Synthesizes a looped upper-limb motion, projects it to landmarks, adds noise, hydrates it
back, and plays the result next to time-series of landmark noise, bone-direction error,
joint angles and rigid-fit orientation error — so solver error can be told apart from the
noise it was given. The output is self-contained (three.js is vendored and inlined) and
gitignored; regenerate it rather than committing it.

## License

AGPL-3.0-or-later. See [LICENSE](LICENSE).
