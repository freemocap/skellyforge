# Work plan: geometry → skeleton definition → hydration

**Status as of 2026-08-21.** 222 tests passing. Phase 2 (static definitions) is complete: the whole
human skeleton loads — 59 segments, 167 landmarks, 52 face blendshapes.

This document covers the work started after the "commit b4 bloodbath" checkpoint: rebuilding the
geometry layer, then walking the human skeleton YAML component by component until the whole
skeleton loads.

---

## The three phases

| phase | what it produces | state |
|---|---|---|
| **1. Geometry** | the math the rest of it stands on | **done** |
| **2. Static definitions** | `SkeletonDefinition` loaded from YAML — every landmark and segment, with rest positions and reference frames | **done** |
| **3. Hydration** | observed data driving those definitions: per-frame poses, orientations, filtering | **not started** |

Phase 3 is why `rotation_quaternion.py` and `transform_math.py` exist. They currently have no
callers outside their own tests. That is expected — nothing hydrates yet. Do not delete them.

---

## Phase 1: Geometry (done)

`skellyforge/core/math/geometry/`

### Type ontology

```
Vector3Array          (..., 3) float64, finite. Storage + validation + dot.
├── Point             a location. Point - Point -> Displacement. Point + Point is blocked.
├── Displacement      a difference. Adds, scales, negates, normalizes.
└── UnitVector        a direction. Unit length enforced at construction.
```

These are **siblings, not substitutes**. The affine algebra is enforced at runtime by `isinstance`
guards returning `NotImplemented`, not by type hints alone — so it holds even where beartype is
inactive. `Point.__sub__` carries `@overload`s so callers never get handed a `Displacement | Point`
union to narrow by hand.

`from_prevalidated_array` bypasses `__init__` (and therefore beartype) for values *derived* from
already-valid ones. Anything holding raw or external data goes through `from_array`. That split is
the difference between 0.21 µs and 4.31 µs per construction, and it is the reason batched solves
are cheap.

### Reference frames

- **`SpatialAxis`** — one enum, six members (`X`, `NEGATIVE_Y`, …), each carrying `index` and `sign`.
  `cyclic_sign_toward` deliberately consults only the axes, never their directions; callers fold
  their own signs into the vectors. This is the subtlest thing in the package — see the
  verification note below before changing it.
- **`ReferenceFrameDefinition`** — origin point, a primary signed axis pointing *exactly* at its
  landmark, and optionally a secondary signed axis pointing *approximately* at its landmark. With
  no secondary it is **underspecified**: a direction with unresolved roll.
- **`OrthonormalBasis`** — three `UnitVector` axes plus an origin. Orthogonality and handedness are
  checked once at construction, so nothing downstream re-checks. `at_batch_index` slices a
  validated batch without re-validating, which is what makes batched solving worth doing.
- **`calculate_orthonormal_basis`** — Gram-Schmidt. Fully vectorized over leading dimensions, so a
  single frame, a rolling window, and a whole take all go through the same code path.
- **`Handedness`** — right-handed is the assumption everywhere; left-handed is buildable so geometry
  authored here can be converted to a left-handed external convention later, and every construction
  of one warns loudly.

### Streaming

`PointRingBuffer` stores `(num_points, 2 * capacity, 3)` and writes every sample twice, so the
newest `capacity` samples are always a **contiguous zero-copy slice**. O(1) append, no reallocation.

Measured: 1.36 / 1.46 / 1.84 µs per append at capacity 100 / 1000 / 10000, against 3.93 / 68 / 944 µs
for the naive concatenate.

> **Aliasing hazard.** A retained window view goes *incoherent* after an append, it does not slide.
> Buffer `[0,1,2]`, append 99 → the retained view reads `[99,1,2]` while a fresh window reads
> `[1,2,99]`. Use `window_copy` if the window must outlive the next append.

### Verification

The math was checked by execution, not inspection:

- **2400 combinations** — every ordered pair of the 6 signed axes × both handednesses × 50 random
  point sets. Primary point lands exactly on its named signed half-axis with zero off-axis
  component; secondary lands on the positive half of its named axis; `det` matches the requested
  handedness to 1e-9. Zero failures.
- Basis round trips (`world → local → world`), matrix inverses, origin → local zero.
- Quaternions, 300 random: `det(R)=1`, `R·Rᵀ=I`, matrix and axis-angle round trips,
  `rotate_vector` == matrix product, `q·q⁻¹ = I`.
- Ring buffer against a plain list across 23 appends at capacity 5, including the wrap.

> **Note on `cyclic_sign_toward`.** A check asserting it should equal the sign of the cross product
> of the two *signed* basis vectors reports six failures. That check is wrong, not the code. The
> signs cancel: `pa.sign` and `sa.sign` are baked into both the stored rows and the cross product,
> so `det = cyclic_unsigned(i,j) × handedness × cyclic_unsigned(i,j) = handedness`. The
> 2400-combination test is the authority here.

---

## Phase 2: Static definitions (done)

### The pipeline

```
component .yaml
  │
  ├─ resolve_includes      {$include: path} -> that file's contents, pasted.
  │                        Include-only mappings; paths relative to the INCLUDING file;
  │                        cycle-guarded.
  ├─ lowercase_names       one recursive walk. Every key and every string goes lowercase.
  │                        Prose under `definition` is the one exception.
  ├─ expand_sided_entries  sided: true -> left_<name> and right_<name>. Right mirrors x.
  │                        Aliases get sided too. References inside a sided entry resolve
  │                        to the same side.
  └─ build_reference_frame_definition
                           type: exact -> primary axis. type: approximate -> secondary.
                           negate: true -> the NEGATIVE_* variant. No approximate ->
                           underspecified segment.
  │
  ▼
(landmarks, segments)  ──merge──▶  SkeletonDefinition   (global cross-validation)
```

`build_component` returns two plain dicts rather than a component object. A component is a *file*,
not a thing that outlives loading, and what a file contributes is landmarks and segments.

Cross-component checks live on `SkeletonDefinition`, which is the first place that can see every
component at once:

- global landmark name/alias uniqueness (via `LandmarkNameResolver`)
- every landmark's `reference_frame` names a real segment
- every segment's reference geometry names landmarks that exist
- no name claimed by two components

A landmark in one file may legitimately name a `reference_frame` that lives in another, which is
exactly why per-component validation would be wrong.

### Conventions locked in

**Coordinate system** (VRM): `+x` = subject's LEFT, `+y` = UP, `+z` = FORWARD. Millimetres.
50th-percentile adult.

**Sidedness**: `sided: true` at the file level or per entry. Authored for the **left** side only;
the loader emits `left_*` and `right_*` and mirrors x across the sagittal plane. A sided landmark's
x may be **negative** — mirror-symmetric structures (pelvis, skull) sit on one side of the sagittal
plane, but fan-shaped ones (hand, foot) legitimately span both (the thumb is +x, the pinky −x).

**Naming**: `SCREAMING_SNAKE` in the YAML because it makes the document scannable; lowercase at
runtime. The loader lowercases, nothing enforces the screaming. Sidedness is a **prefix**
(`left_hip_socket`), never a `.L`/`.R` suffix — that convention is gone.

**Fail loudly.** The loader repairs nothing. Unknown keys, malformed geometry, missing origins,
half-specified secondary axes, and segments with no `reference_geometry` all raise.

### Decisions made while bringing up the components

- **Shared joints (option 1).** A segment's **origin** may name a landmark owned by its parent
  segment — the elbow belongs to the upper arm but is the lower arm's origin; the CMC joints belong
  to the carpals but are the metacarpals' origins. Primary and secondary must still be owned. This
  is the ontology's "linkage = shared point" model, applied within a component.
- **Self-contained components.** At the *cross-component* boundaries each component owns its own
  proximal landmark rather than referencing a neighbor's: the hand owns `carpal_origin`, the leg
  owns `hip_joint`, the foot owns `ankle_origin` — distinct from the arm's `wrist`, the pelvis's
  `hip_socket`, and the leg's `ankle`. Wiring those coincident points together is the linkage pass.
- **Skull collapsed to one segment.** Eyes/ears/nose/chin are skull *landmarks*, not segments
  (per the ontology). `EYE_BONE` and `JAW` were dropped.
- **Face is a separate thing.** The 52 ARKit blendshapes live in `face.yaml`, held by
  `FaceBlendShapes`, not a skeleton component. Driving them is future work.

### Component walk

Working down the `$include` list in `human_skeleton.yaml`, fixing each as we reach it.

| component | segments | state |
|---|---|---|
| **pelvis** | 1 | **done** — 15 landmarks (5 midline + 5 sided × 2), `PELVIS` fully specified |
| **spine** | 3 | **done** — 52 landmarks (24 vertebra base + 24 `_top` + 4 sternum), `CHEST` fully specified, lumbar/cervical underspecified |
| **skull** | 1 | **done** — 14 landmarks, fully specified (collapsed from a 3-segment scaffold) |
| **arm** | 2 | **done** — shoulder/elbow/wrist, elbow shared between upper and lower arm |
| **hand** | 20 | **done** — 33 landmarks (carpals + 5 digits), all joints shared via the carpals/metacarpals |
| **leg** | 2 | **done** — hip/knee/ankle, knee shared between upper and lower leg |
| **foot** | 3 | **done** — heel/foot/toes, simplified to a midline ball and toe tip |
| **face** | — | **done (holder)** — 52 ARKit blendshapes in `face.yaml`, loaded by `FaceBlendShapes` |

Whole skeleton: **59 segments, 167 landmarks, 52 blendshapes**. Fully specified: pelvis, chest, skull
(they carry a roll landmark); the limbs and spine are still roll-underspecified.

### Rest pose (T-pose)

The rest pose is now defined in `definitions/human_skeleton/rest_pose.yaml`, loaded by
`RestPose.from_yaml` in `core/skeleton_parts/rest_pose.py`. Each segment names its `parent`,
an optional `connect_at` (the parent landmark its origin sits on — defaulting to the segment's own
origin landmark, which is exactly right for a shared joint), and an optional `orientation` (a wxyz
quaternion, parent-relative). Walking the tree composes these into per-segment world transforms.

It matches the VRM default humanoid: the trunk runs straight up, the arms point out to ±x, the legs
point down, the feet slope forward-and-down, and the toes point forward — with only ~10 non-identity
orientations spelled out. The old `build_standard_human_tpose` (pre-bloodbath `skellymodels` code)
was deleted along with the rest of `core/math/kinematics/`.

**Still deferred:** the rest pose nails the *shape* (orientations). The *positions* it produces use our
anatomical lengths (not the VRM's), and the cross-component joints (wrist, hip, ankle) remain two
landmarks at one point — reconciling them into a single shared point is the linkage pass.

---

## Phase 3: Hydration (next)

Observed data driving the static definitions. This is where the currently-orphaned tier gets used:

- `rotation_quaternion.py` — `RotationQuaternion` (wxyz), slerp, batched ops
- `transform_math.py` — `Transform` (rotation + translation)
- `PointRingBuffer` — the streaming path
- `calculate_bases_for_segments` — the batched solve
- hydration solvers (tpose, rigidifier, orientation solver, D3/D4 filter, length estimation) — rebuilt from scratch on the new types; the pre-bloodbath `core/math/kinematics/` module was deleted

### Performance, already established

`calculate_orthonormal_basis` costs ~100 µs per **call** and ~0.23 µs per **frame** once batched.
The cost is almost entirely fixed per-call overhead, so the batch axis matters more than the batch
size.

`calculate_bases_for_segments` therefore groups segments by `(primary axis, secondary axis,
handedness)` and issues one call per convention. Twenty segments, one frame each: 2105 µs solving
one at a time versus ~300 µs grouped — **7× faster**. That batch axis is *segments*, and it composes
with time: each segment's points may be a single frame, a rolling window, or a whole take.

Three data regimes, one code path:

| regime | shape | source |
|---|---|---|
| realtime single frame | `(3,)` per point | `buffer.latest_by_name()` |
| rolling window | `(window, 3)` | `buffer.window_by_name()` |
| batch / post hoc | `(num_frames, 3)` | the whole take |

---

## Cleanup — done

- ~~`skellyforge/core/math/geometry/orthonormal_basis/_to_delete/` and
  `skellyforge/core/skeleton_parts/_to_delete/`~~ — deleted (the `_to_delete` tree is gone).
- ~~`kinematic_chain.py` imports `skellyforge.skeleton_parts.…`, missing `core.`~~ — fixed.
- ~~`segment_linkage.py` types `segments: tuple[RigidBodySegment]`~~ — now `tuple[RigidBodySegment, ...]`.
- ~~`CLAUDE.md` still describes the pre-bloodbath `skellymodels/standard_human/` layout~~ — rewritten.
- ~~Six tolerance constants across four files, chosen ad hoc~~ — centralized in
  `numeric_tolerances.py`, each derived from one base (`MINIMUM_VECTOR_NORM = 1e-9`).
- ~~`core/math/kinematics/` still writes `NDArray[np.float64]` inline (silently unchecked)~~ —
  swept to `FloatArray` across 7 files, so beartype now checks them.
