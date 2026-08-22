# Work plan: geometry → skeleton definition → hydration

**Status as of 2026-08-22.** 317 tests passing. Phases 1 and 2 are complete and Phase 3
(hydration) is working end to end: the whole human skeleton loads — 61 segments, 124
landmarks, 52 face blendshapes — resolves to a rest pose, and hydrates back from observed
positions with its free roll resolved. What is left of the plan is the **linkage and chain
layers**, which are still placeholders.

This document covers the work started after the "commit b4 bloodbath" checkpoint: rebuilding the
geometry layer, then walking the human skeleton YAML component by component until the whole
skeleton loads.

---

## The three phases

| phase | what it produces | state |
|---|---|---|
| **1. Geometry** | the math the rest of it stands on | **done** |
| **2. Static definitions** | `SkeletonDefinition` loaded from YAML — every landmark and segment, with rest positions and reference frames | **done** |
| **3. Hydration** | observed data driving those definitions: per-frame poses, orientations, roll | **done** |
| **4. Linkage + chain** | the two ontology layers above `segment`; today the hierarchy lives in `rest_pose.yaml` | **not started** |

Phase 3 is why `rotation_quaternion.py` and `transform_math.py` exist, and both are now
called: `Transform` by the Kabsch fit, the scalar quaternion throughout the rest pose and
hydration. The **vectorized** half of `rotation_quaternion.py` still has no callers — it is
for the per-frame hot loop, which does not exist until something streams — but it is now
covered by parity tests against its scalar twin (96%), so it can be relied on when that
arrives.

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
| **spine** | 4 | **done** — 9 landmarks (3 vertebral-junction boundaries + 4 sternum + 2 acromion); per-vertebra landmarks removed, lumbar/cervical are shared-origin direction segments |
| **skull** | 1 | **done** — 14 landmarks, fully specified (collapsed from a 3-segment scaffold) |
| **arm** | 2 | **done** — shoulder/elbow/wrist, elbow shared between upper and lower arm |
| **hand** | 20 | **done** — 33 landmarks (carpals + 5 digits), all joints shared via the carpals/metacarpals |
| **leg** | 2 | **done** — hip/knee/ankle, knee shared between upper and lower leg |
| **foot** | 3 | **done** — heel/foot/toes, simplified to a midline ball and toe tip |
| **face** | — | **done (holder)** — 52 ARKit blendshapes in `face.yaml`, loaded by `FaceBlendShapes` |

Whole skeleton: **61 segments, 124 landmarks, 52 blendshapes**. Fully specified: pelvis, chest, skull
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

## Phase 3: Hydration (done)

Observed landmark positions in, per-segment poses out, in closed form and one frame at a
time. `hydrate_skeleton` branches on a STATIC property of each segment rather than on what
a fit happens to throw:

- **`RigidBodySegment.supports_rigid_fit`** — whether the segment's own landmarks span a
  plane. If they do, Kabsch fits every observed one of them and pins the full orientation.
  On the shipped human that is 5 segments of 61 (pelvis, chest, skull, both carpals).
- **Otherwise** the origin-to-primary direction is recovered by shortest arc, leaving roll
  about the long axis free. **`ContinuousRollResolver`** then supplies that roll by
  parallel transport — carrying the previous frame's secondary axis forward and
  orthonormalizing it against the new direction, which is continuous through the pole where
  a per-frame shortest-arc roll flips, and adds no lag.

`SegmentPose.solved_by` records which of the three (`RIGID_FIT`, `DIRECTION`,
`TRANSPORTED_ROLL`) produced a pose, so no consumer has to re-derive the branch — and
because "measured" and "merely well-behaved" are different claims about the world.

`estimate_segment_lengths` calibrates per-subject lengths as the median observed
origin-to-primary distance, and refuses to measure a segment whose landmarks are missing
rather than returning a silently short result.

### Still to build

- **The linkage layer.** Cross-component joints are already exactly coincident in the rest
  pose (`wrist` / `carpal_origin`, `hip_socket` / `hip_joint`, `ankle` / `ankle_origin`,
  `acromion` / `shoulder` all resolve to the same world point, to 0.00 mm), so this is a
  naming and identity job rather than a geometric one — smaller than it looks.
- **The chain layer**, above linkages.
- **Filtering and the streaming path.** `PointRingBuffer` and the vectorized quaternion
  functions are the pieces waiting for it.

### Performance, already established

`calculate_orthonormal_basis` costs almost entirely fixed per-call overhead, so the batch
axis matters more than the batch size. `calculate_bases_for_segments` groups segments by
`(primary axis, secondary axis, handedness)` and issues one call per convention: measured
over twenty segments, 5948 µs solved one at a time against 752 µs batched, a factor of
about 8. Note the scope — only a fully specified segment has a basis to solve, so on the
shipped human that is three segments; it earns its keep on dense rigid bodies and on
batched time, not on the limbs.

Three data regimes, one code path:

| regime | shape | source |
|---|---|---|
| realtime single frame | `(3,)` per point | `buffer.latest_by_name()` |
| rolling window | `(window, 3)` | `buffer.window_by_name()` |
| batch / post hoc | `(num_frames, 3)` | the whole take |

## Cleanup — done

- ~~`skellyforge/core/math/geometry/orthonormal_basis/_to_delete/` and
  `skellyforge/core/skeleton_parts/_to_delete/`~~ — deleted (the `_to_delete` tree is gone).
- ~~`kinematic_chain.py` imports `skellyforge.skeleton_parts.…`, missing `core.`~~ — fixed.
- ~~`segment_linkage.py` types `segments: tuple[RigidBodySegment]`~~ — now `tuple[RigidBodySegment, ...]`.
- ~~`CLAUDE.md` still describes the pre-bloodbath `skellymodels/standard_human/` layout~~ — rewritten.
- ~~Six tolerance constants across four files, chosen ad hoc~~ — centralized in
  `numeric_tolerances.py`, each derived from one base (`MINIMUM_VECTOR_NORM = 1e-9`).
  Finished properly in the audit sweep: `rotation_quaternion.py` had quietly reintroduced
  fourteen of them, and `rigid_point_set.py` was using machine epsilon.
- ~~`core/math/kinematics/` still writes `NDArray[np.float64]` inline (silently unchecked)~~ —
  swept to `FloatArray` across 7 files, so beartype now checks them.

---

## Audit sweep (2026-08-22)

A full audit and its fixes are recorded in [`../AUDIT_REPORT.md`](../AUDIT_REPORT.md). The
things that changed behaviour, rather than only tidying:

- **The heel was upside down.** Its rest quaternion had `w` and `x` swapped, putting the
  calcaneus 19.8 mm above the ankle; the calcaneus was also 45 mm rather than the ~80 mm
  the anatomy actually is. Both feet now stand on one flat ground plane.
- **The pelvis frame was 20.22° from its own coordinates**, because its approximate y axis
  pointed at `sacrum_top`, which sits up *and* back. It points at `left_iliac_crest` now,
  and the invariant is a test over every fully-specified segment.
- **Left and right local frames now mean the same thing.** The loader negates x-axis
  declarations on the right side (VRM convention, right-handed on both sides). This
  surfaced a latent half-turn in hydration: it compared a signed world direction against
  an unsigned local one, so any `NEGATIVE_*` primary axis would have hydrated 180° out.
- **The rest pose fails loud.** Unknown or missing segment entries, more than one root, a
  self-parent, and a `connect_at` the parent does not own — including a defaulted one —
  all raise. A one-character typo used to silently move a limb to the world origin.
- **`hydrate_segment` no longer swallows `ValueError`.** The branch is a static property.
- **The roll convention moved into the library** from the viewer script.
- **`core/post_processing/` was deleted** — eight modules that had never imported.
