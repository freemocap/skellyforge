# Skelly Forge — Audit Report & Fix Log

> **Two audits are merged here.** The original three-pass report is preserved verbatim
> from section 1 onward. Section 0 is the working TODO: it is the union of that report's
> findings and a second independent three-pass audit, and it is what gets checked off as
> fixes land. Where the two audits found the same thing, both IDs are listed.

---

## 0. Fix log

Legend: `[x]` landed and verified by execution · `[ ]` outstanding.
`C/H/M/L` are the second audit's IDs; `#n` are this document's section-3 numbers.

### Chunk 1 — Critical ✅ complete (258 tests passing)

- [x] **C1 / #2 — `core/post_processing/` deleted.** Eight modules that had never
      imported: wrong package paths (`skellyforge.post_processing.*`), three phantom
      modules, and two undeclared dependencies (`pydantic`, `tqdm`).
- [x] **C2 / #1 — `__main__.py` rewritten around a real `run()`.** Loads the skeleton,
      the rest pose and the face, and prints a summary. The `skellyforge` console script
      now works and doubles as a smoke test.
- [x] **C3 / #8 — `RestPose.from_yaml` fails loud.** It now rejects an entry naming a
      segment that does not exist, a segment with no entry, anything other than exactly
      one root, a segment that is its own parent, and a `connect_at` not owned by the
      parent — **including the defaulted one**, which was previously unchecked (#8).
      Verified: a one-character typo in a segment name used to silently move that limb to
      the world origin; it now raises.
- [x] **C4 / #10 — the swallowed `ValueError` is gone.** `hydrate_segment` no longer
      wraps the Kabsch fit in `try/except ValueError: pass`. Which closed form a segment
      gets is now a static property, `RigidBodySegment.supports_rigid_fit`, so a genuine
      runtime failure raises instead of being mistaken for a collinear segment.
- [x] **H6 — a landmark naming its segment by an alias is no longer silently unowned.**
      The loader now matches a segment's aliases when collecting its landmarks, and
      `SkeletonDefinition` raises if any landmark ends up owned by zero or by more than
      one segment.
- [x] **H7 — the Kabsch collinearity guard left machine epsilon.** It was
      `np.finfo(float64).eps` (~2e-16 relative); it is now the shared
      `MINIMUM_RELATIVE_SINGULAR_VALUE`, the same conditioning question the Gram-Schmidt
      guard already asked. The viewer reads the same classification instead of applying
      its own `1e-6`.
- [x] **M1 / #5 / #6 — tolerances centralized; the SLERP thresholds reconciled.**
      Fourteen magic numbers left `rotation_quaternion.py`. Scalar `slerp` and
      `slerp_batch` now share `MINIMUM_SLERP_SEPARATION_COSINE`, so they agree by
      construction rather than by luck.
- [x] **M4 — `landmark_name_resolver` is built once.** It was a plain property rebuilding
      the whole index on every access (91 µs), while `__post_init__` built one and threw
      it away.
- [x] **M5 — segment aliases are globally unique-checked.** Landmark aliases already were;
      segment aliases were not, so two segments could both answer to one name.
- [x] **M3 — `calculate_bases_for_segments`' docstring corrected.** It claimed "roughly
      twenty times cheaper"; measured is 5948 µs → 752 µs over twenty segments, a factor
      of about 8. The docstring now also states the scope: three of sixty-one segments.
- [x] **L4 / #13 — `to_euler_xyz` → `to_roll_pitch_yaw`** (and `quaternion_to_euler` →
      `quaternions_to_roll_pitch_yaw`). It returns ZYX-intrinsic angles; the old name
      invited the wrong reading.
- [x] **L5 / #7a — the `.L`/`.R` suffix docstring on `RigidBodySegment` is gone.**
- [x] **L8 — the quaternion module is keyword-only throughout**, matching the rest of the
      package, and `from_rotation_matrix(cls, R)` is now `(cls, *, matrix)`.
- [x] **L9 — `RotationQuaternion` is `eq=False`**, like every other geometry value type,
      and gained `angle_to` / `is_same_rotation`, which know about the double cover.
- [x] **L13 — `rotation_between_vectors` uses `atan2(|cross|, dot)`**, not `arccos(dot)`,
      which loses digits exactly where segments usually sit (near-parallel).
- [x] **L14 — dead branches removed from `slerp_resample`.** It guarded against
      non-increasing timestamps thirty lines after rejecting them.
- [x] **L15 — redundant APIs collapsed.** `rotate_vector_batch` was
      `rotate_vectors_batch` with one vector; `compose_with_constant(pre_multiply=bool)`
      is now `pre_multiply_by_constant` / `post_multiply_by_constant`.
- [x] **L16 — `SegmentPose` records how it was solved** (`solved_by: PoseSolution`), so
      no consumer has to re-derive the branch. The viewer now reads it.
- [x] **M6a — the NaN path is named.** A primary landmark sitting on its own segment
      origin used to divide by zero inside `from_prevalidated_array` and surface four
      frames later as "Quaternion components must be finite"; it now raises naming the
      segment and the landmark.
- [x] **M2a / #12 — `test_synthetic_round_trip` stopped parsing `rest_pose.yaml` itself.**
      `RestPose` now exposes `parents`, `connect_ats` and `relative_orientations`, so the
      test exercises the shipping parser. The viewer does the same.
- [x] **M7a — viewer payload and axis labels.** Compact JSON separators (1.01 MB → 892 KB)
      and the y-axis labels now name the drawn (padded) range rather than the unpadded
      data range they were 8% away from.

### Chunk 2 — High ✅ complete (313 tests passing)

- [x] **H1 — the heel is fixed, and so is the test that certified the bug.** The authored
      quaternion had `w` and `x` swapped, putting the calcaneus 19.8 mm *above* the ankle;
      it is now 68.3 mm below it. The calcaneus also grew from 45 mm to 80 mm, which is
      what the anatomy actually is (about 42 mm posterior to the ankle joint centre and
      about 68 mm inferior, and `hypot(42, 68) = 80`) — the old 45 mm was a
      horizontal-only estimate that left the heel unable to reach the floor.
      **Both feet now stand on one flat ground plane: the six contact landmarks span
      0.016 mm.** `test_the_heel_points_back` became
      `test_the_heel_points_back_and_down`, and a new
      `test_both_feet_stand_on_one_flat_ground_plane` ties the heel orientation, the heel
      length and the foot orientation into a single claim, so none of the three can drift
      alone. Every non-identity rest-pose quaternion now carries its derivation in a
      header comment rather than arriving as a bare magic number.
- [x] **H2 — the pelvis solves to its own authoring frame.** Its approximate y axis now
      points at `left_iliac_crest` instead of `sacrum_top`. Both are defensible anatomy
      for "up", but the crest sits directly above the hip-socket midpoint and so
      orthogonalizes to exactly +y, while the sacrum sits up AND back and tilted the whole
      frame 20.22° posteriorly away from the coordinates it was authored in. The invariant
      is now a test over every fully-specified segment — feed a segment its own rest
      positions, get the identity basis — and all three come back at exactly 0.
- [x] **H3 — bilateral frames follow VRM, and stay right-handed.** Mirroring the
      coordinates without mirroring the axis DECLARATIONS left the right side's frame as
      the left's rotated a half turn about y, so local +z was anterior on the left and
      posterior on the right. The loader now negates x-axis declarations on the right side,
      giving both sides local +x toward the subject's left, +y up and +z forward — the VRM
      convention, right-handed on both sides. The named landmark still lies exactly on its
      declared signed axis; the declaration just becomes the negative half. What a
      right-handed triad cannot avoid is that +x is lateral on the left and medial on the
      right, so it is anterior and distal that correspond across the body.
- [x] **H3b — a latent half-turn in hydration, found while fixing H3.** Hydration's
      direction branch compared a world direction that *had* the primary axis sign applied
      against a local direction that did not, so any segment with a `NEGATIVE_*` primary
      axis would have hydrated 180° out, silently and only on that segment. Nothing shipped
      had one until H3 introduced the right clavicle. Both sides now apply the sign, in
      hydration and in the roll resolver alike.
- [x] **H4 / #4 — the vectorized quaternion half is tested.** A new
      `test_rotation_quaternion.py` (48 tests) checks every batched function against its
      scalar twin over 200 random rotations, walks the shared SLERP threshold from a
      tenth of it to a thousand times it, and adds trajectory tests (a constant spin
      reports exactly that spin; resampling at the source timestamps returns the source).
      **Module coverage went from 34% to 96%.**
- [x] **H4b — two numerical bugs the new tests caught immediately.** `to_axis_angle`
      recovered `sin(theta/2)` as `sqrt(1 - w^2)`, which cancels catastrophically as `w`
      approaches 1 — exactly where small rotations live — and `angle_to` used
      `2*arccos(|dot|)`, whose vertical tangent at 1 turned a one-ulp difference into an
      angle of 1e-8. Both now take the vector part's norm directly (it *is* `sin(theta/2)`
      for a unit quaternion) and pair it with `atan2`.
- [x] **H5 / viewer(b) — the roll convention is in the library.**
      `skeleton_parts/roll_resolution.py` holds `ContinuousRollResolver`, the
      parallel-transport convention that resolves the roll two landmarks leave free for
      **56 of the 61 segments**. It is stateful and per-take, leaves rigid-fit poses
      untouched, and reports its work as `PoseSolution.TRANSPORTED_ROLL` — a third
      solution kind, because "measured" and "well-behaved" are not the same claim about
      the world. Tests cover the three things a convention has to do: never disturb the
      measured direction, stay continuous through the pole where the naive per-frame roll
      flips, and reset cleanly between takes. It lives in `skeleton_parts` rather than
      `math/kinematics` because it needs a skeleton and a pose, not just vectors — which
      also broke the circular import the first placement created.
- [x] **#9 — the origin-at-zero invariant is enforced.** When a segment owns its own
      origin landmark, that landmark must sit at `[0, 0, 0]` in the segment's frame, since
      that is what being the origin means and since `length`, the rest pose's forward
      kinematics and hydration all read it that way. `length` lost its special case as a
      result. One test fixture was quietly violating it and has been corrected.

### Chunk 3 — Medium ✅ complete (315 tests passing)

- [x] **M2b — the duplicated perpendicular is gone.** `_default_perpendicular` in
      `coordinate_frame_ops` and `_perpendicular` in the viewer were byte-identical; the
      library now exports `default_perpendicular` and the viewer imports it. The viewer's
      copy of the roll convention went with it (H5).
- [x] **M6b — `estimate_segment_lengths` refuses rather than skips.** It used to `continue`
      past a segment whose origin or primary landmark was missing and return a silently
      short dictionary, which looks exactly like a skeleton with fewer segments - so a
      caller comparing estimated against authored lengths would have compared a subset
      without knowing. It now raises, naming the segments and the missing landmarks, and
      `measurable_segments` is the explicit way to work with partial data. The test that
      asserted the skip was correct now asserts the refusal.
- [x] **M6c / #19a — the left-handed warning fires for underspecified definitions too.**
      `__post_init__` returned early before reaching `warnings.warn`, so a frame declared
      left-handed before it had a secondary axis went through in silence, against a
      docstring promising that "every construction of one announces itself loudly".
      `LeftHandedCoordinateSystemWarning` also has a real docstring now instead of a `#`
      comment for a body.
- [x] **M7b / viewer(a) — the viewer is genuinely self-contained.** three.js r128 and
      OrbitControls are vendored under `scripts/vendor/` and inlined into the HTML, so it
      opens with no network and renders under a strict content security policy. It fails
      loudly, naming the missing file, rather than writing a page that would come up blank.
      `scripts/vendor/README.md` records where they came from and why r128 is pinned -
      `examples/js/` was removed after r128, so moving forward means moving the viewer to
      ES modules, not just bumping a number.
- [x] **M7c — the generated HTML is gitignored.** It is a build artifact, and it was
      landing in git as a one-line megabyte, so every regeneration was a whole-file diff.
      `git rm --cached scripts/skeleton_viewer.html` will stop tracking the copy already
      committed.
- [x] **M8 — `test_rest_pose` uses `tmp_path`.** It wrote `_bad_rest_pose.yaml` into the
      package's own `definitions/` directory and unlinked it in a `finally`, so a crashed
      run left a stray file inside the shipped package. The rewritten file also adds six
      tests for the failures C3 introduced.

### Chunk 4 — Low ✅ complete (317 tests passing)

- [x] **L1 / #3 — the README is this project's.** It described the old postprocessing GUI,
      on Python 3.9/3.10, via a file that no longer exists.
- [x] **#14 — `pyproject.toml` cleaned out.** The description is no longer "Basic template
      of a python repository". `scipy` and `pandas` went with the deleted
      `post_processing`; `toml` and `pyarrow` had no uses anywhere; `pytest` moved to the
      dev extra, which also picked up `coverage`. The runtime dependency list is now numpy,
      pyyaml, beartype and skellylogs. `[tool.coverage.run]` was added, and the
      `ini_options` comment stopped citing pydantic, which is not a dependency at all.
      **`uv.lock` needs regenerating** — `uv lock` — since the dependency set changed.
- [x] **L2 — CLAUDE.md rewritten.** It said `core/math/kinematics/` "was deleted" (it
      exists and hydration depends on it), said Phase 3 was "not started" (it works end to
      end), and cited 222 tests. It now also carries an **invariants** section, because the
      things most likely to be broken by a well-meaning change are exactly the ones no
      signature states.
- [x] **L2b — the work plan updated**, with Phase 3 marked done, the linkage/chain layers
      named as what actually remains, and an audit-sweep changelog.
- [x] **L3 — one extension.** `arm.yml`, `foot.yml`, `hand.yml` and `leg.yml` are now
      `.yaml`, matching the other three components and every other YAML in the repo.
      (The old files are in `_to_delete/old_yml_extensions/`; this session cannot delete on
      your machine, only move.)
- [x] **L6 / #7b — the sided-x contradiction is gone.** The module docstring said a sided
      landmark's x "must be >= 0"; the function docstring forty lines later said it may be
      negative. The module docstring now describes what the stage actually does, mirroring
      included.
- [x] **L7 — `raise_unless_snake_case_segment_name` checks snake_case.** It checked
      "lowercase and no spaces", which let `left-arm.l` through. It is a pattern now.
- [x] **L10 / #11 — `KinematicChain.linkages` is a tuple**, not a mutable list inside a
      frozen dataclass. Both placeholder layers also gained real docstrings saying what
      they are for and validation that refuses an empty one.
- [x] **L11 / #19b — `lowercase_names` uses an allowlist.** It lowercased every string in
      the tree and exempted `definition`, so any future field holding a citation, a units
      string or a path would have been silently mangled and would have had to remember to
      join the blocklist. It now lowercases every KEY, plus the values under the keys that
      actually hold names. An unrecognized value is left alone, which is the safe way round.
- [x] **L12 — `PointRingBuffer.capacity` and `.point_names` are write-once.** Rebinding
      either after the storage was allocated left the cursor, the name index and the array
      describing three different buffers. The buffer is still mutable where it should be.
- [x] **L17 — alias sprawl trimmed.** `SACRUM_TOP` answered to five names; three were
      hand-coined synonyms no tracker emits. The remaining two are the standard vertebral
      names, and the reasoning is written above them, since every alias enlarges the space
      in which landmark names have to stay globally unique.
- [x] **#15 — `default-vrm.gltf.json5` is documented rather than orphaned.** It is the VRM
      humanoid the rest pose's orientations were derived against, so `rest_pose.yaml` now
      says so and points at it. It is reference data, not dead data.
- [x] **#16 — stale `__pycache__` cleared.** Eight directories, holding `.pyc` files for
      modules deleted in the bloodbath, moved to `_to_delete/stale_pycache/`.
- [x] **#17 — the type aliases live together.** `LinkageNameString`, `ChainNameString`,
      `SkeletonNameString` and `BlendshapeName` moved from their own modules into
      `type_overloads.py` alongside the others, each with a line saying which layer of the
      ontology it names.
- [x] **#18 — the orphaned-landmark error says what it means.** It blamed a
      "`reference_frame`", which is the YAML key, while checking `landmark.segment`, which
      is the field. It now names both, and lists segment aliases as well as canonical names.

---

## Anything still open

Nothing on the list above. Three things are worth knowing:

1. **`uv.lock` is stale** — the dependency set changed, so run `uv lock`.
2. **`_to_delete/` needs deleting by you.** This session can move files on your machine
   but not delete them. It holds the old `.yml` component files, eight stale `__pycache__`
   directories, and one scratch file.
3. **`scripts/skeleton_viewer.html` is now gitignored but still tracked.**
   `git rm --cached scripts/skeleton_viewer.html` finishes that.

---

# Original three-pass audit report

**Scope:** the entire `skellyforge` repo (core math, skeleton_parts, definitions YAML,
post_processing, tests, viewer generator + HTML, packaging, docs). Three independent
passes (broad survey → deep math/definitions → tests/viewer/dead-code) were collated
below. All math findings were verified by hand-derivation and, where possible, execution.

**Test run:** `pytest skellyforge/tests/` → **252 passed, 5 errors**. The 5 errors are
NOT code failures — they are `tmp_path` fixture `PermissionError`s raised because the
audit sandbox blocks pytest's temp directory. In a normal environment they pass.

---

## 1. Executive summary

Skelly Forge's *new* core (Phase 1 geometry + Phase 2 definitions, rebuilt after the
"bloodbath") is genuinely excellent: the affine algebra is correct and fail-loud, the
YAML pipeline is clean, the definitions are internally consistent (61 segments / 124
landmarks / 52 blendshapes — verified by hand), and the test suite is strong. I found
**no incorrect math** in the geometry/kinematics layer.

The problems are almost all on the **periphery**: a broken CLI entrypoint, an entire
dead/broken `post_processing` package, stale packaging/docs, ~400 lines of completely
untested vectorized quaternion code (where a real scalar-vs-batch drift already crept in),
and several unvalidated invariants that happen to hold today only because the shipped
data is well-authored. There is also meaningful doc drift: CLAUDE.md and the work plan are
behind the code (they say Phase 3 "not started" and "old kinematics deleted", but
`rest_pose`, `skeleton_hydration`, `segment_length_estimation`, and rebuilt
`core/math/kinematics/` now exist).

---

## 2. What is good

- **The affine algebra is right and well-defended.** `Point` / `Displacement` /
  `UnitVector` are siblings, not substitutes; `Point + Point` is blocked by runtime
  `isinstance` guards (not just type hints), so the algebra holds even where beartype
  is off. The `from_prevalidated_array` split (validate once, trust derived results)
  is a real, well-documented performance win.
- **The math checks out.** I hand-verified: the quaternion→matrix formula,
  `from_rotation_matrix` (Shepperd), `to_axis_angle`, `to_euler` (ZYX-intrinsic),
  scalar and batch SLERP, the Gram–Schmidt sign logic in
  `calculate_orthonormal_basis` (including the tricky negative-axis cases), the Kabsch
  SVD convention in `align_point_sets_kabsch`, the rest-pose forward kinematics, and
  the ring buffer's double-write indexing. All correct.
- **The YAML loader is a clean four-stage pipeline** (includes → lowercase → sided →
  reference-frame), each stage a pure fail-loud function, with cycle-guarded includes and
  global cross-validation at `SkeletonDefinition`.
- **The definitions are consistent.** I independently recounted the shipped YAML: pelvis
  1 / spine 5 / skull 1 / arm 4 / hand 40 / leg 4 / foot 6 = **61 segments**;
  15+9+14+6+66+6+8 = **124 landmarks**; 52 face blendshapes. The shared-joint model
  (elbow owned by upper arm, used as lower arm's origin) and the self-contained-component
  model (foot owns `ankle_origin`, distinct from leg's `ankle`) are coherent and match
  the documented ontology.
- **The test suite is strong where it exists.** 252 passing, including excellent
  property-style tests: all 30 signed-axis pairings × both handednesses for the basis
  solver, a synthetic forward→hydrate round-trip with noise robustness split by bone
  length, Kabsch recovery of known transforms, ring-buffer wrap/aliasing behavior.
- **The viewer is genuinely useful and in sync.** `generate_skeleton_viewer.py`
  synthesizes a looped motion, hydrates it, and emits error time-series; the committed
  `skeleton_viewer.html` matches the current generator (real data present, not stale).

---

## 3. Findings by severity

### CRITICAL — broken as shipped

1. **`skellyforge/__main__.py` crashes on import.** It does
   `from skellyforge.standard_human.standard_human_model import StandardHuman`, a
   pre-bloodbath path that no longer exists. Verified:
   `ModuleNotFoundError: No module named 'skellyforge.standard_human'`. The
   `skellyforge` console entry point in `pyproject.toml` is therefore dead.

2. **The entire `core/post_processing/` package is dead/broken code.** Every module in it
   imports modules that do not exist or use wrong paths:
   - `from skellyforge.data_models.trajectory_3d import Trajectory3d` (module deleted)
   - `from skellyforge.post_processing.filters.filter_config import ...` (missing the
     `core.` segment; real path is `skellyforge.core.post_processing...`)
   - `from skellyforge.post_processing.filters.core.butter import butter_from_config`
     (no `core/` subdir; the function lives in `butterworth_filter.py`)
   - uses `pydantic` and `tqdm`, which are **not** declared in `pyproject.toml`.
   All four `__init__.py` files are empty, so nothing imports this package — it is 100%
   orphaned, and would raise `ModuleNotFoundError` if anything tried. It is pre-bloodbath
   leftover that the "rebuilt from scratch" note never actually deleted.

3. **`README.md` is the wrong project's README.** It describes
   `postprocessing_gui` / `postprocess_GUI.py`, a Python 3.9/3.10 conda install, and
   "FreeMoCap 1.0" — nothing about the current skeleton/geometry work.

### HIGH — correctness/consistency

4. **~400 lines of vectorized quaternion code has zero tests and zero callers.** All 12
   module-level functions in `rotation_quaternion.py` (`hamilton_product`,
   `normalize_quaternion_array`, `quaternion_to_rotation_matrix`,
   `quaternion_to_axis_angle`, `quaternion_to_euler`, `rotate_vector_batch`,
   `rotate_vectors_batch`, `slerp_batch`, `slerp_resample`,
   `compute_angular_velocity`, `compose_with_constant`,
   `conjugate_quaternion_array`) are unimported and untested. The work plan's claim that
   `rotation_quaternion.py` "has no callers outside their own tests" is wrong — it has no
   tests either. Only the scalar `RotationQuaternion` class is exercised. This is the
   exact code meant for the per-frame hot loop.

5. **Scalar vs batch SLERP drift.** `RotationQuaternion.slerp` switches to NLERP at
   `dot > 0.9995` (~1.8°); `slerp_batch` switches at `dot > 1 - 1e-10` (~0.0008°).
   The two implementations of the same operation behave differently on the same input.
   This is precisely the kind of "double definition" divergence that untested parallel
   implementations breed.

6. **Magic numbers bypass the centralized tolerance module.** `numeric_tolerances.py`
   was introduced to kill ad-hoc constants, but `rotation_quaternion.py` reintroduces
   them: `normalize_quaternion_array` and `quaternion_to_axis_angle` use bare
   `1e-10`; `from_rotation_vector` uses `1e-12`. These should use
   `MINIMUM_QUATERNION_NORM` etc.

7. **Contradictory docs (two places).**
   - `RigidBodySegment.name` docstring says names are "optionally suffixed `.L` or
     `.R`" — but that convention was explicitly dropped in favor of the `left_/``right_`
     prefix (as `naming.py` and the work plan state).
   - The top-of-file docstring in `skeleton_yaml_loader.py` claims a sided landmark's x
     "must be >= 0 or the sides would silently swap", while the function docstring and a
     passing test assert the opposite (negative x is legitimate for fan-shaped hands/feet).

### MEDIUM — latent gaps (unvalidated invariants; all hold today by good data)

8. **Rest-pose default `connect_at` is not validated.** `build_rest_pose` falls back to
   `frame_definition.origin_point_name` when a segment has no explicit `connect_at`,
   and rotates that landmark's local position by the *parent's* orientation — but never
   checks the fallback landmark is owned by the parent. `RestPose.from_yaml` validates
   only *explicit* `connect_at`. Every shipped segment happens to satisfy it (lower
   arm/leg, toes, metacarpals, phalanges), so this is latent, not triggered.

9. **"Origin at [0,0,0] in its own frame" is an unvalidated invariant.** `length` and the
   direction path of `hydrate_segment` assume a self-owned origin landmark sits at
   `[0,0,0]`; the loader never enforces it. All shipped self-owned origins
   (`pelvis_origin`, `shoulder`, `hip_joint`, `carpal_origin`, `ankle_origin`,
   `head_center`) are `[0,0,0]`, so it holds — but a future authoring mistake would
   silently produce wrong lengths/orientations.

10. **`hydrate_segment` swallows all `ValueError` from Kabsch.** The broad
    `except ValueError: pass` around `point_set.fit_pose` is meant for the collinear
    case, but it would also mask any other fit error and silently fall through to the
    direction path.

11. **Dead placeholder layers.** `kinematic_chain` has **zero importers and zero tests**;
    `segment_linkage` is imported only by it. They are documented placeholders, but
    `KinematicChain` also holds a *mutable* `list` inside a frozen dataclass.

### LOW — cleanup, consistency, hygiene

12. **Rest-pose YAML parsing is triplicated.** `RestPose.from_yaml`,
    `generate_skeleton_viewer.py::_read_rest_pose_tree`, and
    `test_synthetic_round_trip.py::_read_rest_pose_tree` each parse
    `rest_pose.yaml` independently, because `RestPose` only stores *world*
    orientations/origins and does not expose the relative orientations / parent tree the
    generator and test need. Fix the API, not the callers.

13. **`RotationQuaternion.to_euler_xyz` is misnamed.** It returns ZYX-intrinsic
    `(roll, pitch, yaw)`; the name "xyz" invites the wrong frame/order reading.

14. **Unused / dead dependencies.** `toml` and `pyarrow` have zero uses anywhere;
    `scipy` and `pandas` are used only by the dead `post_processing` code;
    `pydantic` and `tqdm` are used (by dead code) but not declared; `skellylogs` is
    declared but its import is commented out in `__init__.py`.

15. **`definitions/human_skeleton/default-vrm.gltf.json5` (817 lines) is referenced by
    nothing.** Not the loader, not the generator, not the viewer. Dead data.

16. **Stale `__pycache__` from deleted modules** (`critically_damped_orientation`,
    `body_side`, `sided_component_expansion`, `segment_reference_geometry`,
    `underspecified_rigid_body_segment`, `unresolved_reference`, `yaml_include_loader`,
    `skeleton_yaml_config`, `axis_definition`) — repo hygiene only.

17. **Scattered type aliases.** `LinkageNameString`, `ChainNameString`,
    `SkeletonNameString`, `BlendshapeName` are defined inline in their modules rather
    than in `type_overloads.py` with the others.

18. **Terminology drift.** `skeleton_definition.py`'s orphaned-landmark error says a
    landmark names a "`reference_frame`" that isn't a segment, but it is actually
    checking `landmark.segment` (the owning segment). The YAML key
    `reference_frame` was renamed to the model field `segment`, and the messages did
    not fully follow.

19. **`LeftHandedCoordinateSystemWarning` has no message** (its body is a `#` comment),
    and `lowercase_names`'s "prose" exception keys on the literal key `definition`
    anywhere in the tree (fragile if a future field is named `definition` and is not prose).

20. **Git hygiene (note only — user owns git):** the index shows staged "AD" deletes of old
    flat component files (`arm.yaml`, `axial.yaml`, `eye.yaml`, `foot.yaml`,
    `hand.yaml`, `leg.yaml`, `pelvis.yaml`) at the old location — leftovers of the
    move into `components/` and the axial/eye merge into spine/skull. Commit messages in
    recent history are also checkpoint-noise ("chkpt", "fixin stuff", "mathy wathy"),
    which will make future archaeology harder.

---

## 4. Math audit (what I verified, and the caveats)

Verified correct by derivation:
- Quaternion ↔ matrix (both directions), axis-angle, rotation-vector (exp/log maps),
  Euler angles, SLERP (scalar + batch), angular velocity (global + body frame).
- `calculate_orthonormal_basis` sign algebra: `tertiary = handedness × cyclic_sign ×
  (primary × secondary)`, including negative primary/secondary axes and left-handedness.
- Kabsch: `H = ref_centeredᵀ · obs_centered`, `R = V·diag(1,1,sign)·Uᵀ`,
  `t = obs_centroid − R·ref_centroid`, with proper-reflection rejection.
- `rotation_between_vectors` (shortest arc + antiparallel fallback),
  `Transform` apply/inverse, ring-buffer double-write window indexing, rest-pose
  forward kinematics.

The only real math-flavored issues are the SLERP threshold divergence (#5), the
misnamed `to_euler_xyz` (#13), and the unvalidated origin-at-zero invariant (#9) —
none of which bite the shipped data.

---

## 5. Definition-file audit

- Counts, naming, sidedness mirroring, shared-joint and self-contained-component models all
  check out and are loadable (verified by the 61/124/52 tests and by hand).
- The rest pose is self-consistent: every `connect_at` points at a parent-owned landmark;
  coincident cross-component points (`ankle` vs `ankle_origin`, `hip_socket` vs
  `hip_joint`) resolve to the same world position in the T-pose, exactly as the work plan
  says they should.
- Observations, not bugs: the hand is authored flat (z=0) with fingers running +y, so the
  T-pose fingers inherit the arm's lateral orientation rather than a palm-forward pose —
  a deliberate simplification; and the glenohumeral "shoulder" is placed at the clavicle's
  `acromion` (an approximation the comments acknowledge).

---

## 6. Test-coverage audit

- **Strong:** vector algebra, scalar quaternion, transform, basis solver (2400+ combos),
  ring buffer, Kabsch, hydration, loader, rest pose, synthetic round-trip.
- **Gap:** the vectorized quaternion half is entirely untested (#4), and `segment_linkage`
  / `kinematic_chain` have no tests (#11). `skeleton_pose` is covered only indirectly.
- **Tooling:** no coverage tool is configured (`pytest-cov` absent, no CI coverage gate).
  A coverage run would immediately flag the ~400-line dead/untested block.
- **Environment note:** 5 include tests use `tmp_path` and error out under the audit
  sandbox (not a code issue).

---

## 7. Viewer audit

- Good: regenerable, in sync, synthesizes a looped motion + hydrates it + plots error
  time-series; correct classification of rigid-fit vs direction segments.
- Issues: (a) the "self-contained" claim is false — it loads three.js + OrbitControls from
  a CDN and pins the deprecated non-module OrbitControls build (`three@0.128.0`); (b) the
  continuous parallel-transport roll resolution lives only in the generator script, not in
  the library, so the library's 2-landmark orientations keep arbitrary roll — that logic
  belongs in `skeleton_hydration`; (c) it re-parses `rest_pose.yaml` (#12); (d)
  `json.dumps` without separators bloats the emitted HTML.

---

## 8. Architecture & philosophy

- The seven-layer ontology is sound and the boundary rule (SkellyForge never imports
  FreeMoCap; tracker "keypoints" vs model "landmarks") is respected throughout.
- The single-class `RigidBodySegment` that promotes from underspecified (direction) to
  fully-specified (triad) by *adding data, not a new type* is a genuinely good design.
- Two overlapping orientation mechanisms now exist — `calculate_basis` (Gram–Schmidt,
  batched, groups by axis convention) and `hydrate_segment` (Kabsch over all owned
  landmarks, or shortest-arc direction). They are mostly complementary (speed vs
  robustness), but their overlap for exactly-3-landmark segments is worth an explicit
  decision, not an accident.
- The "define every quaternion op twice (scalar + vectorized)" pattern is a defensible
  hot-path choice, but it is exactly where drift (#5) and the coverage gap (#4) live. If it
  stays, the two halves need a parity test.
- The Phase-3 tier (`rest_pose`, `skeleton_hydration`,
  `segment_length_estimation`, `point_ring_buffer`, `face_blendshapes`) is wired to
  tests and the viewer generator but **not to each other or any consumer** — e.g.
  hydration does not yet use rest pose or estimated lengths. That is the documented "next
  integration step", but `kinematic_chain` (#11) is not even that far along.

---

## 9. Prioritized fix list

1. Delete (or rewrite on the new types) `core/post_processing/`. (Critical #2)
2. Fix or remove `__main__.py`; restore a working `skellyforge` entrypoint. (#1)
3. Replace `README.md` with one describing this repo; refresh `pyproject.toml`
   (description, `requires-python`, move `pytest` to dev extras, drop
   `toml`/`pyarrow`/unused deps, declare `pydantic`/`tqdm` only if kept). (#3, #14)
4. Delete `default-vrm.gltf.json5` if truly unused. (#15)
5. Add tests + parity tests for the vectorized quaternion functions, and reconcile the
   SLERP thresholds. (#4, #5)
6. Route the remaining magic numbers through `numeric_tolerances.py`. (#6)
7. Fix the two contradictory docstrings. (#7)
8. Validate the origin-at-zero invariant and the default-`connect_at` parent ownership
   in the loader/rest-pose; narrow `hydrate_segment`'s exception. (#8, #9, #10)
9. Expose relative orientations/parent tree on `RestPose` and delete the two duplicated
   parsers. (#12)
10. Move the continuous-roll resolution from the generator into the library. (Viewer b)
11. Fix `to_euler_xyz` naming; consolidate type aliases; correct "reference_frame" →
    "segment" error text. (#13, #17, #18)

---

## 10. Pass-by-pass notes (the three passes)

**Pass 1 (broad survey):** mapped the tree, read all source/docs/YAML, ran the suite,
identified the broken `__main__`, dead `post_processing`, stale README/pyproject, and
the orphaned `default-vrm.gltf.json5`.

**Pass 2 (step back; deep math + definitions):** hand-verified every math function and the
sign algebra; recounted the definitions (61/124/52 confirmed); found the SLERP threshold
divergence, the magic-number relapse, the two doc contradictions, the unvalidated
origin-at-zero and default-`connect_at` invariants, and the broad `except` in hydration.

**Pass 3 (step back; tests/viewer/dead-code/redundancy):** coverage-mapped every module,
exposing the fully-untested vectorized quaternion half and the untested linkage/chain
placeholders; confirmed the triple-duplicated rest-pose parsing; audited the viewer
generator + committed HTML (in sync, CDN dependency, roll logic in the wrong layer);
reconciled the doc drift between CLAUDE.md / work plan and the actual code.
