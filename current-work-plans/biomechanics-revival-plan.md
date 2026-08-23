# Biomechanics Revival Plan

**Goal.** Restore the Center of Mass, inertial, and ground-reference functionality that
the refactors swept away, rebuilt on the current system (typed geometry, YAML-defined
skeleton, `SkeletonPose` hydration, VRM y-up coordinates).

---

## 1. What was swept away (the audit)

Git history shows three generations of biomechanics code, all deleted:

### Gen 1 — keypoint-based (oldest)
`skellyforge/biomechanics/` and `skellyforge/skellymodels/biomechanics/` (deleted in
`c85b753` 'rm biomech/'):

- `calculations/calculate_center_of_mass.py` — per-segment CoM as
  `proximal + (distal - proximal) * com_fraction`, then a mass-fraction-weighted
  whole-body sum. Keypoint-level (no segment model).
- `calculations/enforce_rigid_bones.py` — median bone-length rigidifier: measure median
  length per joint pair, then snap distal joints to that length along a joint hierarchy
  (recursive child adjustment).
- `anatomical_calculations.py` — `CenterOfMassCalculation`, `RigidBonesEnforcement`,
  `CalculationPipeline` — all built on the retired `Aspect`/`Trajectory` framework.

### Gen 2 — segment-based (newest, the one worth reviving)
`skellyforge/core/math/kinematics/` (deleted in `e1aceb3` 'theoreticallytpose'):

- `anthropometric_parameters.py` — **de Leva (1996)** body-segment inertial parameters:
  `SegmentInertialParameters(mass_fraction, com_fraction, k_sagittal, k_transverse,
  k_longitudinal)`, with female / male / mean tables for 8 segments (head, trunk,
  upper_arm, forearm, hand, thigh, shank, foot).
- `center_of_mass.py` — de Leva 8-segment model mapped onto the standard-human segment
  layer; whole-body CoM = mass-weighted sum; **mass redistribution** up limb chains when a
  distal segment is occluded (foot→shank→thigh, hand→forearm→upper_arm, then trunk);
  `CoMConfidence` tiers (invalid/low/medium/high) and a `directly_observed_mass` fraction.
- `composite_inertia.py` — composite centroidal inertia (CCRBI) via the parallel-axis
  theorem, principal axes/moments via `eigh`, and equimomental ellipsoid semi-axes. Its
  per-segment self-inertia term (`segment_inertias`) was declared but deferred ('Phase 2').
- `ground_reference.py` — **CoP** (vertical ground projection of CoM, 'estimated'), **XCoM**
  (extrapolated CoM / Hof 2008 capture point), **CMP** (centroidal moment pivot from CoM
  acceleration). Note: this file assumed **z-up**.
- `rigid_body_kinematics.py` — linear velocity/acceleration and angular acceleration via
  finite differences (forward/central/backward), a lazy `RigidBodyKinematics` batch model
  (velocity, acceleration, angular velocity, Euler angles, resample, shift), and keypoint
  world-projection.

### Swept test
- `tests/test_segment_center_of_mass.py` (deleted in `98cb783`) — three tests: mass sums
  to 1; thigh CoM sits at the de Leva com_fraction along the segment; an occluded foot rolls
  its mass up the chain and drops `directly_observed_mass`.

---

## 2. What has ALREADY been revived (do not duplicate)

| Old | New home |
|---|---|
| `tpose.py` / `build_standard_human_tpose` | `rest_pose.py` (`RestPose` + `build_rest_pose`) |
| `skeleton_rigidifier.py` / `rigid_point_set.py` | `hydrate_segment` (Kabsch) + `rigid_point_set.py` |
| `orientation_solver.py`, `critically_damped_orientation.py` | `roll_resolution.py` (`ContinuousRollResolver`) + `rotation_quaternion` vectorized ops (now tested) |
| `segment_lengths.py`, `segment_length_estimation.py` | `segment_length_estimation.py` |
| `quaternion_math.py` | `rotation_quaternion.py` (incl. `compute_angular_velocity`) |

So the **angular-velocity half** of `rigid_body_kinematics` already exists; the
**linear velocity / acceleration / angular-acceleration** half does not.

---

## 3. What to revive (the five targets)

1. **Anthropometric inertial parameters** — de Leva BSIP table + dataclass.
2. **Center of mass** — de Leva → standard-human segment mapping, whole-body CoM, mass
   redistribution, confidence.
3. **Composite inertia** — CCRBI + principal axes + ellipsoid, **plus** the per-segment
   world-frame inertia the old code deferred ('Phase 2').
4. **Ground-reference points** — CoP, XCoM, CMP.
5. **Linear kinematics** — linear velocity / acceleration / angular acceleration, so XCoM and
   CMP have their `com_velocity` / `com_acceleration` inputs.

---

## 4. Key adaptations for the new system

1. **Axis convention: already z-up.** Since this plan was written, the whole package was
   re-authored into Blender's convention (**+x right, +y forward, +z up**, ground plane at
   `z = 0`). The old `ground_reference.py` already used 'z is vertical, ground plane
   z = 0', so its vertical axis now MATCHES the canonical frame - no z/y flip is needed.
   `test_both_feet_stand_on_one_flat_ground_plane` now asserts foot contacts share a `z`
   height. CoP/XCoM/CMP keep `z` as the vertical axis with `g = (0, 0, -g)`, and any
   residual horizontal-convention mismatch goes through `CoordinateSystemTransform`.
2. **Naming: `.L`/`.R` suffix → `left_`/`right_` prefix.** And segment renames:
   `forearm` → `lower_arm`, `thigh` → `upper_leg`, `shank` → `lower_leg`,
   `hand` → `carpals`+digits, `foot` → `foot` (origin `ankle_origin`, primary `ball`).
3. **Landmark renames for the composite spans.** Old → new:
   `neck_center` → `cervicothoracic_junction`; `hips_center` → `pelvis_origin`
   (its alias is `hips_center`); `head_vertex` → `head_vertex` (unchanged);
   `hand_middle_finger_tip` → `middle_tip`; `wrist` → `wrist` (unchanged).
4. **Types.** Raw `np.ndarray` → `Point`/`Displacement`/`FloatArray`; keyword-only
   arguments; frozen dataclasses; fail-loud (no `except: pass`, no silent skipping).
5. **Input contract.** `HumanSkeleton` + `dict[landmark, np.ndarray]` →
   `SkeletonDefinition` + a `Mapping[LandmarkNameString, Point]` of world positions (the
   same contract `hydrate_skeleton` already takes), or a `SkeletonPose`. Segment origin /
   primary names now come from `frame_definition.origin_point_name` /
   `frame_definition.primary_point_name` instead of `origin_landmark.name` /
   `primary_axis.target_landmark`.
6. **Layering.** Pure, skeleton-agnostic math (inertia summation, ground-reference points,
   finite-difference derivatives) → `core/math/kinematics/`. Anything that needs a skeleton
   or pose (the BSIP table, the de Leva→segment mapping, per-segment inertia construction) →
   `core/skeleton_parts/`. (This is the rule `roll_resolution.py` already follows.)
7. **Tolerances** come from `numeric_tolerances.py`. `GRAVITY_MM_S2 = 9810.0` is a physical
   constant, not a tolerance, but should be a named module constant.
8. **Provenance.** The de Leva table is *sourced* (not estimated) — carry the citation
   comment through, per the honesty rules.

---

## 5. Proposed module layout

```
core/math/kinematics/
  rigid_body_kinematics.py     # linear/angular velocity & acceleration (finite differences)
  composite_inertia.py         # CCRBI sum, principal axes, equimomental ellipsoid (pure math)
  ground_reference.py          # CoP / XCoM / CMP (pure math, y-up)
core/skeleton_parts/
  anthropometric_parameters.py # de Leva 1996 BSIP (frozen dataclass + tables, sourced comments)
  center_of_mass.py            # de Leva → standard-human mapping + whole-body CoM + redistribution
  segment_inertia.py           # per-segment world-frame inertia from BSIP + pose ('Phase 2')
```

`composite_inertia` and `ground_reference` take plain arrays (no skeleton dependency), so
they belong in `core/math/kinematics`; `center_of_mass`, `anthropometric_parameters`, and
`segment_inertia` are model-aware and belong in `skeleton_parts`.

---

## 6. Order of work (each step lands with its own tests)

1. **`anthropometric_parameters.py`** — pure data, no deps. Tests: the 8 mass fractions
   sum to 1 (both sexes); every `com_fraction` ∈ (0,1); every radius of gyration > 0;
   `mean_with` averages correctly.
2. **`center_of_mass.py`** — the hard part is the mapping table. Test against the rest pose
   (T-pose): whole-body CoM lands near the sagittal midline and near the pelvis height; mass
   sums to 1; the thigh CoM sits at the de Leva fraction along `hip_joint→knee`; occluding
   `left_ball` rolls the foot's mass up the leg and drops `directly_observed_mass`. Port
   and extend the three swept tests.
3. **`rigid_body_kinematics.py`** — reuse the finite-difference scheme already in
   `compute_angular_velocity`; add `compute_linear_velocity`, `compute_linear_acceleration`,
   `compute_angular_acceleration` (global + local). Tests: a constant-velocity trajectory
   gives constant velocity / zero acceleration; a constant-acceleration trajectory gives
   the right slope; strict-timestamp rejection.
4. **`segment_inertia.py`** — per-segment world-frame inertia: `J_world = R J_local Rᵀ`
   with `J_local = m · diag((k_sag·L)², (k_trans·L)², (k_long·L)²)`, using the segment's
   `length` (or `estimate_segment_lengths`) and `SegmentPose.orientation`. Tests: tensor
   symmetric & positive semi-definite; principal axes align with the segment's long axis at
   rest.
5. **`composite_inertia.py`** — sum + principal axes + ellipsoid (mostly ported, typed).
   Tests: a two-point-mass dumbbell has the analytic `m d²` about the perpendicular axis;
   the ellipsoid semi-axes invert the moments; triangle-inequality violation raises.
6. **`ground_reference.py`** — CoP/XCoM/CMP re-expressed y-up. Tests: quiet stance → CoP
   directly under CoM; a forward CoM velocity puts XCoM ahead of CoM; non-positive vertical
   reaction force raises for CMP; a non-positive CoM height raises for XCoM.

The natural end-to-end acceptance test: **rest pose → `hydrate_skeleton` →
`ContinuousRollResolver` → `calculate_center_of_mass` → `segment_inertia` →
`composite_centroidal_inertia` → ground-reference points**, asserted finite and sensible in
the T-pose (CoM on the midline, inertia symmetric, CoP under CoM).

---

## 7. Decisions to make (flagging, not guessing)

- **de Leva is an 8-segment approximation of a 61-segment skeleton.** The hand→`carpals`+
  digits and head/trunk composite spans need explicit, hand-authored landmark choices. The
  Gen-2 mapping is the starting point; the new landmark names are in §4.
- **Mass redistribution chains** must be re-keyed to the new names (`foot→lower_leg→
  upper_leg`, `hand→lower_arm→upper_arm`).
- **CoP honesty.** With no force plate this is always 'estimated' — keep that flag explicit
  in the API rather than burying it.
- **de Leva table: Python vs YAML.** Recommend Python (`frozen` dataclass + module constants)
  for the lowest-friction revival; it is a fixed published table, not authored geometry. Can
  move to YAML later if the 'static source of truth' convention demands it.

---

## 8. Explicitly out of scope (do not revive)

- **`enforce_rigid_bones.py`** — superseded by Kabsch hydration (`hydrate_segment`). The old
  forward-pass length-snapping rigidifier should stay dead.
- **`anatomical_calculations.py` / `CalculationPipeline`** — built on the retired
  `Aspect`/`Trajectory` model. The new 'pipeline' is just composed pure functions
  (`hydrate → CoM → inertia → ground reference`); no framework needed.
- **Gen-1 keypoint-based CoM** — the Gen-2 segment-based approach is the right one; the
  keypoint version is strictly worse (no segment model, no redistribution).

---

## 9. Suggested first step

Revive `anthropometric_parameters.py` + `center_of_mass.py` together (steps 1–2). They are
self-contained, directly answer 'where is the CoM', and their tests are already half-written
(the three swept tests). Then add the kinematics and ground-reference pieces in a second
batch, since XCoM/CMP need `com_velocity`/`com_acceleration` from step 3.