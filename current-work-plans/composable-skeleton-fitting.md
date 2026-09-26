# Composable skeleton fitting and non-rigid linkages

Status: implementation design, 2026-09-25. The C++ build scaffold and scalar smoke
solve are implemented; skeleton mathematics and saved recordings are unchanged.
See `../cpp/README.md` for native dependencies and build commands.
This supersedes the fixed-dimension-only next-step
recommendation in `full-skeleton-solve.md`; it does not claim the new human model
has been selected or validated.

## Existing vocabulary and boundaries

Use the seven layers of the FreeMoCap kinematic ontology: keypoint, mapping,
landmark, segment, linkage, chain, skeleton. A COCO-compatible fitting pipeline
is a composition of operations on these entities, not a new SkeletonDefinition
subclass or a tracker-specific copy of the skeleton.

- Tracker defines keypoints and keypoint-to-landmark mappings. FreeMoCap applies
  those mappings. Forge never imports Tracker or FreeMoCap.
- An AnatomicalLandmark owns its reference_frame and local_position. A fitted
  landmark target references that existing definition, not another attachment
  table. Mapped positions and fitted positions remain distinct.
- hydrate_skeleton supplies segment poses and scale estimates. Its established
  calculations remain available for initialization and diagnostics.
- fit_model_scale produces ModelScaleFit. Call this model-scale fitting, not
  subject calibration. Camera calibration is a different operation.
- JointDefinition defines topology; RestPose supplies relative quaternions;
  chains own connected forward/inverse kinematics; the skeleton composes them.
- fit_connected_pose and fit_connected_sequence currently optimize fixed-size
  connected poses. Their numerical convergence does not establish an adequate
  torso/shoulder model.

Relevant code, relative to this repository:

- `skellyforge/core/skeleton/pose/hydration.py`
- `skellyforge/core/skeleton/pose/model_scale_fitting.py`
- `skellyforge/core/skeleton/linkage/joint_definition.py`
- `skellyforge/core/skeleton/pose/rest_pose.py`
- `skellyforge/core/skeleton/chain/synthesis.py`
- `skellyforge/core/skeleton/pose/fit_connected_{pose,sequence}.py`

## Proposed ontology amendment

Keep RigidBodySegment rigid: its local landmark geometry does not change during
a fixed-model solve. Generalize a linkage from "two segments sharing a point"
to "a declared relative transform relationship between parent and child
segments." Sharing a point is the existing special case.

Static linkage definitions specify parent/child attachments, the frame in which
movement is expressed, permitted displacement coordinates, and their reference
values. Hydrated linkages carry relative quaternion rotation and any declared
displacement coordinates. A joint angle alone no longer describes every linkage.
Definitions resolve references to objects at load, as the ontology already
requires. No name parsing to infer spine or shoulder behavior.

Non-rigid does not mean unconstrained XYZ translation. The first candidate is
one scalar extension along a declared parent-local unit axis. Other displacement
models must be explicitly declared; none are inferred from an Euler convention
or a joint's existing `ball` label. This is a kinematic model, not a claim to
simulate forces, tissue stiffness, or spinal mechanics.

Replace the ontology's blanket "closed-form fits, never iterative repair" with:
segment hydration retains its declared direct/closed-form methods; connected
skeleton fitting may optimize declared linkage coordinates against mapped
landmarks and explicit pose/time preferences. Iterative fitting never overwrites
keypoints or mapped landmarks, or silently changes the model.

This amendment is staged here because this work is in Forge. The canonical
ontology under FreeMoCap's `current-work-plans/01-data-model/ontology.md` must be
updated during its own integration stage; it has not been edited in this stage.

### Exact forward-kinematics definition

Notation: q is a unit wxyz quaternion; rotate(q, v) rotates vector v. p and c
denote parent and child. a_p and a_c are attachment positions in their respective
segment-local frames, already converted to world distance units using each
owner's ModelScaleFit. t is a segment origin, q_pc a parent-relative quaternion.
d is a displacement expressed in the parent's frame, in the same distance units.

```text
q_c = q_p * q_pc
t_c = t_p + rotate(q_p, a_p + d) - rotate(q_c, a_c)
x_landmark = t_owner + rotate(q_owner, scaled_local_position)
```

Existing behavior is exactly a_c = 0 and d = 0: the child's origin is the
parent's connect_at landmark. No second landmark needs authoring for that case.
For the candidate axial connection, d = u * e, with declared unit axis u and
scalar extension e. Its reference value is zero; nonzero nominal separation is
in the authored attachments. Reference lengths and axes come from definitions,
not hardcoded human names in the solver.

The exact connection check becomes:

```text
(t_c + rotate(q_c, a_c)) - (t_p + rotate(q_p, a_p)) = rotate(q_p, d)
```

A nonzero d is an explicit connection span, not two allegedly shared points that
have accidentally separated. The viewer must show this span separately from the
rigid segments. Snapshot/replay must retain its definition and per-frame value.

Crucially, this does NOT stretch an existing RigidBodySegment. Before applying it
to the standard human, decide which existing spinal spans belong to linkages
and which structures remain rigid segments. Moving a child while continuing to
draw the old rigid span as if it reaches that child is invalid. A deformable
segment would require a separate explicit ontology change; it is not implemented
by changing segment_scales each frame.

## Parameters supplied to Ceres

| Parameter | Storage and frame | Optimized? |
| --- | --- | --- |
| Root origin per frame | XYZ, recording distance units, world frame | Yes |
| Root quaternion per frame | wxyz, local-to-world | Yes, quaternion manifold |
| Selected linkage quaternion per frame | wxyz, child-to-parent | Yes, quaternion manifold |
| Declared linkage extension per frame | Scalar distance, declared parent-local axis | Only for non-rigid linkages |
| ModelScaleFit | Existing fitted scale and per-segment scale field | Fixed for this stage |
| Landmark local positions and attachments | SkeletonDefinition plus owning segment scale | Fixed |
| Mapped landmark targets, timestamps | Recorded inputs with source relationships | Fixed |

Orientations are RotationQuaternion throughout Forge's public definitions,
poses, serialization, and viewer data. Ceres QuaternionManifold has four stored
components and three internal tangent dimensions. Its local update coordinates
do not become a second pose representation. Current experimental rotation-vector
increments are an implementation detail to replace, not an ontology concept.

## Composable residual blocks

Each block declares the existing landmarks/linkages it reads, the parameter
blocks it depends on, units, scaling, missing-input behavior, and diagnostic name.
All blocks contribute to ONE solve. They are not sequential geometric repairs.
Names below describe operations, not finalized Python classes.

### Landmark agreement

```text
r_landmark = (fitted_world_position - mapped_world_position) / position_scale
```

Three dimensionless residuals per selected landmark per frame. Dependencies are
the root and ancestor linkage coordinates used by FK. position_scale is an
explicit positive distance, not an invented uncertainty estimate. Robust loss
acts on the squared norm of this three-component block, rather than treating
each world axis as a separate outlier.

Select targets through the existing mapping provenance. Duplicate aliases must
not count a keypoint twice; means/offsets do not add independent evidence. Omit
missing targets, reject malformed/infinite input, and report target support.
Do not alter directly_measured_landmark_names: its existing model-scale-fitting
meaning includes affine combinations and is not an independence classification.

### Linkage extension preference

```text
r_extension = (e - e_reference) / extension_scale
```

One dimensionless residual for a declared scalar extension. The reference and
positive scale are explicit modeling choices; they are not measured elasticity.
Bounds e_min <= e <= e_max are separate. Their values and supporting evidence
remain a human-model decision; this document selects no anatomical numbers.
If multiple spinal connections can trade extension without changing fitted
landmarks, the selected result is prior-dependent and must be reported as such.

### Quaternion pose preference

For q_ref and q representing the same parent-relative linkage frame:

```text
q_error = conjugate(q_ref) * q
r_rotation = 2 * sign_near_identity(q_error) * vector_part(q_error) / rotation_scale
```

This is a quaternion chordal residual, with norm 2*sin(theta/2)/rotation_scale.
It approximates angular error near zero; it is NOT the existing geodesic
rotation-vector objective at large angles. q and -q have identical cost. The
shortest-sign choice needs a deterministic convention and explicit tests around
180 degrees, where the representation is ambiguous. Neither quaternion
components nor Euler display conventions define anatomical angular bounds.

The initial Ceres parity check must reproduce the existing objective before
adopting this candidate residual. Any subsequent switch is an explicit change in
the fitting mathematics with a separate comparison, not a silent backend change.
Rest-pose preferences remain independent from hydrated initialization.

### Time-domain residuals

For adjacent timestamps with dt > 0, use the quaternion residual above with
q_error = conjugate(q_i) * q_next and denominator angular_speed_scale*sqrt(dt).
The angular scale is in radians/second (a small-angle interpretation for the
chordal candidate). Translation and extension terms are:

```text
r_translation = (t_next - t_i) / (linear_speed_scale * sqrt(dt))
r_extension_motion = (e_next - e_i) / (extension_speed_scale * sqrt(dt))
```

Multiply per-frame landmark, extension, and pose costs by trapezoidal time
weights. For robust losses, scale the cost by that weight, not the input to the
robust loss: otherwise the outlier threshold changes with sample timing.
These are motion preferences, not hard speed limits. Fast-motion attenuation is
a required diagnostic. Missing frames have no landmark residuals; they may be
inferred from neighbors/preferences and must be marked. Reject an entirely
unobserved solve. Split recording discontinuities; no automatic long-gap bridge.
Use posthoc timestamps; no realtime equivalence or streaming claim.

### Shoulder-specific composition

SC landmarks remain attached to the fitted thoracic structure. Shoulder midpoint
and offset mappings remain visible mapped landmarks but do not hard-position the
fitted thorax. Select the shoulder-girdle linkage model before writing its blocks:
the present clavicle-to-arm attachment is not a full scapular model. Review the
published scapulothoracic formulation for permitted movements and required
attachments. Do not substitute a fixed arm-elevation ratio or a free scapular
pose. Record which movements are inferred from priors with the available points.

## Exact relationships, bounds, and losses

FK enforces attachment equations and rigid segment geometry exactly. Unit
quaternions use manifolds. Scalar displacement bounds use Ceres bounds. Soft
preferences use residuals. A large residual weight is not an exact constraint.
Ceres does not provide arbitrary equality constraints. Do not use an unreviewed
penalty to imitate a connection that must be exact.

## Python / C++ implementation boundary

Prefer a small compiled C++ implementation of repeatedly evaluated FK/residual
calculations, using Ceres automatic differentiation. Python retains definitions,
mapping provenance supplied by the caller, pipeline composition, data access,
and viewer generation. Compile the resolved SkeletonDefinition into indexed
numerical inputs; do not maintain another human skeleton or parse tracker names
in C++. Compare compiled FK with Python FK on identical inputs.

Ceres in C++ is selected; PyCeres is not a candidate backend. The compiled
extension owns the Ceres problem and exposes Python entry points through
pybind11. Establish build, packaging, memory ownership and error handling before
implementing new skeleton mathematics. See `../cpp/README.md` for the initial
CMake/scikit-build-core scaffold and Poe commands. Verify Python 3.11/3.12 and
Windows first, then supported Linux/macOS builds.

The BS reference is `clients/bs/python_code/rigid_body_solver/core/optimization.py`.
Reuse its parameter-block/residual-block composition concept. Do not copy its
fixed-skull assumption, untimed smoothing, finite-difference derivatives, or
reconstructed-keypoint naming into Forge's landmark output.

## Implementation and validation sequence

1. Establish the C++ build and binding infrastructure, reproducible dependencies,
   Poe commands, wheel packaging, native tests and Python-boundary smoke tests.
   These checks initially use a scalar AutoDiff solve, not new skeleton math.
2. Specify and verify the compiled parameter/residual interfaces, quaternion
   manifolds, bounds, input ownership, errors and diagnostics. Establish Ceres
   parity for existing fixed-geometry mathematics before changing that model.
3. Add linkage-definition/pose support and exact FK for declared displacement.
   Verify zero displacement reproduces current FK, attachment equations, rigid
   landmark distances, unit/world-frame invariance, quaternion sign equivalence,
   and snapshot/replay. Resolve spine ownership before altering human YAML.
   Establish a Ceres implementation of the EXISTING fit on fixed geometry and
   identical objectives. Compare objective evaluation, observable fitted
   landmarks, convergence reporting, and derivatives on controlled cases. Do not
   require identical unobservable joint rotations or identical local minima.
4. Introduce reviewed spine/shoulder definitions and residual choices separately.
   Use arm elevation, shrug, bend/twist, static jitter, occlusion and outliers;
   include motions not generated by the candidate model itself.
5. Use the existing Forge recording-fit viewer and prepared recording, including
   frames 200 and 203 and the full motion interval. Show mapped landmarks,
   keypoints, fitted landmarks, segment poses, and non-rigid connection spans.
   Report each residual family, bound saturation, extension trajectories,
   missing support, timings, and termination. Do not repair geometry in rendering.
6. Keep one accepted connected fitting pipeline once the replacement passes;
   parity scaffolding is not a permanent legacy-solver product. No performance
   claim before profiling compiled residual evaluation and representative data.
7. Integrate into FreeMoCap only after Forge's human commit/push handoff. Persist
   definitions, parameter settings, poses and diagnostics with explicit schema
   handling. Old snapshots do not silently acquire new linkage definitions.

Current prepared data remains read-only. Real-data agreement is not anatomical
ground truth. No numeric compliance bounds, shoulder topology, or anatomical
accuracy claim is established by this document.

## Sources

- Ceres modeling: https://ceres-solver.org/nnls_modeling.html
- Ceres manifolds, derivatives and constraint limitations:
  https://ceres-solver.org/modeling_faqs.html
- PyCeres compiled/Python factors: https://github.com/cvg/pyceres
- Shoulder model: Seth et al. (2016),
  https://doi.org/10.1371/journal.pone.0141028


## Torso implementation checkpoint: native tree assembly

The native Ceres lab now accepts general parent-before-child trees. Viewer 10
exercises five segments with dense synthetic observations. This is infrastructure
validation only; the existing human YAML and production fitting remain unchanged.

The torso subset will reuse pelvis, sacrolumbar, thoracic, left_clavicle and
right_clavicle from the existing SkeletonDefinition. JointDefinition.connect_at
provides each parent-owned attachment; RestPose supplies relative quaternions;
ModelScaleFit supplies fixed per-segment scale. Do not duplicate their dimensions
or attachment tables in a separate human definition. Synthetic model scale must
be labeled as an authored input, not an estimate from four observed points.

Only hip_socket and acromion observations belong in the initial four-landmark
experiment. Their model ownership must remain explicit. No chest/neck midpoint,
sternoclavicular offset, or reference quaternion becomes an independent measured
target. Generated reference poses are evaluation data, not initialization inputs.

Before that solve:
1. Separate observed local landmark bindings from the geometry drawn by the viewer.
2. Accept explicit quaternion/root initialization for underobserved segments;
   construct it from available landmarks and authored rest pose, not synthetic truth.
3. Compare explicit relative rest-pose residuals enabled/disabled. Report their
   costs and bias; do not call them anatomical limits or production defaults.
4. Resolve the earlier phrase "two spine length adjustments": the current lab
   changes linkage displacement, not RigidBodySegment local geometry. A displacement
   at the sacrolumbar-to-thoracic connection preserves a rigid thorax and fixed
   sternoclavicular attachments. Changing thoracic extent while keeping those
   attachments fixed in its local frame requires a different, explicitly declared
   model. Do not silently implement both incompatible interpretations.
5. Add synthetic shrug/shoulder elevation/twist, then real recording evaluation.

No downstream FreeMoCap update is needed for these laboratory stages.


### Superseding decision: rigid torso first

The owner selected a fully rigid torso baseline before any deformation work.
Experiment 11 implements four-landmark fitting using the existing human definitions,
explicit observation-only initialization, and relative-rest-quaternion residuals
enabled/disabled. No linkage displacement or variable segment length is introduced.
The deformable spine/shoulder-bearing-frame discussion above is deferred, not an
approved implementation requirement. Synthetic results and residual definitions are
in scripts/solver_lab.md. Real-recording fitting and production integration remain
subsequent stages; do not confuse this lab experiment with the application pipeline.


### Approved next comparison: axial spine deformation

After inspecting real-recording frame 192, the owner approved a variable-length
sacrolumbar/thoracic experiment. The implemented laboratory policy scales owned
local Z by length/reference, preserving X/Y and the sternoclavicular pair's shape.
This supersedes the rigid-only experimental scope, not the production ontology.
RigidBodySegment, saved Parquet and production fitting remain unchanged. The native
sequence solver declares axial extents explicitly; it does not infer deformation
from names. See scripts/solver_lab.md for residuals, bounds and evaluation limits.

### Approved expansion: full connected body in the existing Ceres lab

Include the saved model's head/neck, arms/hands and legs/feet in the same problem
as the torso, with the existing two axial spine lengths. Reuse the saved skeleton,
joint tree, mappings, rest pose and model scale. This is an experiment inside
SkellyForge; it does not change FreeMoCap's production pipeline or source Parquet.

Keypoints remain the measurements; mappings hydrate landmark positions. All
landmark definitions remain present regardless of keypoint availability. The native
adapter accepts explicit local-landmark indices for the supplied per-frame targets.
It does not infer missing measurements or remove model landmarks. Saved segment
poses are an explicitly labeled initialization, not measurement targets.

The viewer must expose mechanical policy separately from measurement support,
retain hover identities, label every displayed axis origin, and organize Ceres
blocks by body region and segment with upstream dependencies retained. Initial
review interval is frames 180–213, including the bend and raised-arm poses.
