# Solver lab

Run from SkellyForge:

```powershell
uv run --no-sync poe native-install
uv run --no-sync poe solver-viewer
uv run --no-sync poe test-solver-viewer
uv run --no-sync poe solver-viewer-serve
```

Open http://127.0.0.1:8773/solver_viewer.html. The generated page embeds its
data and libraries; it also works offline. Generation performs actual native
fits. Browser controls only select and display their saved results.

## Organization

| File | Responsibility |
| --- | --- |
| `generate_solver_viewer.py` | Generate and assemble the offline page; static cube experiment |
| `solver_sequence_experiment.py` | Cube sequence fits and their diagnostics |
| `solver_linkage_experiment.py` | Two-segment example, observations, fits and diagnostics |
| `solver_lab_data.py` | Experiment catalog and common viewer data format |
| `solver_viewer.js` | Experiment/run selection, playback, diagnostics and download |
| `solver_scene.js` | Draw saved segment geometry, landmarks, axes and attachments |
| `solver_time_series.js` | Shared Plotly charts, segment selection and plot resizing |
| `solver_layout.js` | Shared sidebar splitter |
| `solver_viewer.html.template`, `solver_viewer.css` | Common layout and presentation |
| `check_solver_viewer.cjs` | Control-flow smoke check using actual Three geometry and stubbed DOM/Plotly/WebGL |

## Adding an experiment

Provide one catalog entry with a stable id, label, explanation, applicable
controls, fitting methods, segment definitions and runs. A run contains input
parameters, timestamps and method results. Each method supplies frames, settings,
an objective description and summary diagnostics. Each frame supplies segment
poses and landmark arrays, convergence, costs and named numerical diagnostics.
Do not invent a shared iteration history for separate native solves.

Segment definitions supply display names, landmark names, wireframe edges and
an optional local attachment point. The scene draws all segments; the plot selector
chooses one segment without hiding the others. Numerical outputs use millimeters
and wxyz quaternions. Known synthetic poses are evaluation references, never fitting
inputs. Metadata records Ceres version and the native binary hash.

New experiments should not add geometry-specific branches to shared rendering,
playback or plotting. Add numerical calculations to the experiment generator/native
implementation and publish their results as diagnostics. Keep detailed settings
collapsed and only expose controls that the selected experiment supports. Preserve
panel resizing and shared playback rather than creating another viewer page.

This is a small explicit data contract, not an inheritance framework or a second
skeleton ontology. Before large recordings or long optimization histories are added,
split the embedded banks into separately loaded experiment assets; avoid expanding
the current Cartesian product of precomputed options indefinitely.

## Two-segment stage

Each 200 mm rigid segment has eight non-collinear local landmarks. The parent's
attachment is `[0, 0, 100]`; the child's is `[0, 0, -100]`. Ceres fits two unit
quaternions and one world attachment position. Segment translations are derived
from that shared point and their rotated local attachments. The linkage therefore
cannot separate, without a stiffness weight. Both segments can rotate freely.

Independent fits provide initialization; their predicted attachment positions are
averaged for the initial shared point. The gray layer shows that feasible initial
pose, not the disconnected initializer. Independent and connected fits use exactly
the same noisy observations. Per-frame attachment separation and joint error against
known truth are reported alongside landmark residuals. No temporal prior is applied.

Native checks cover noiseless recovery, noisy attachment coincidence, rigid distances,
cost accounting, world-transform invariance and invalid attachments. The viewer smoke
check visits every experiment/method/segment and first/last frames. It is not a browser
rendering test. Review the interactive page as well.

This stage tests a point linkage with well-observed segment rotations. It does not
yet establish joint limits, non-rigid spine connections, sparse human landmark fitting
or adoption into the production skeleton pipeline.


## Partial child observations

The child-observation control selects the first 8, 4, 2 or 1 local landmark IDs;
the parent retains all eight. Withheld observations are omitted from the native
objective, represented as null in the viewer, and never used for initialization.
Full known geometry remains available only for evaluation. The same noise samples
are used across observation-count and fitting-method comparisons.

Two child landmarks plus the attachment form a non-collinear triangle in this
example, so the connected fit can determine the child rotation. Two landmarks
alone leave rotation about their line undetermined. One child landmark plus the
attachment leaves roll about that line undetermined even in the connected fit.
Convergence does not imply a unique pose. The viewer states that limitation;
withheld-point errors evaluate the chosen solution, not a guaranteed recovery.

The rigid-fit API keeps its default non-collinear support requirement. The lab
explicitly opts into underconstrained independent fits, starting at identity. The
linked fitter permits sparse child initialization while requiring full parent
support. There is no temporal initialization or hidden roll prior. Larger purple
wireframe markers identify fitted withheld points; the plots omit observation dots
for those points. Diagnostics report their RMS error against known synthetic truth.


## Ceres problem inspector

The resizable Ceres problem panel displays parameter blocks, their current values,
QuaternionManifold dimensions, grouped landmark residual blocks and their parameter
connections. Sequence experiments show a three-frame excerpt with the corresponding
motion residual blocks; the summary counts the whole sequence problem. Independent
fits show separate problems, without implying cross-frame coupling. A connected
linkage shares a joint-position parameter block; no attachment residual is invented.

`scripts/solver_problem.js` renders this assembly schematic from the saved experiment
and method definitions. It is not runtime Ceres introspection. Keep it synchronized
with the native assembly whenever adding residual families or parameterizations.
The shared joint position is reconstructed from the returned parent pose. Synthetic
truth never appears as a fitting parameter or residual target.

Use Ceres terminology in explanations: parameter blocks contain unknown values;
residual blocks connect those blocks and evaluate errors; manifolds constrain valid
parameter updates; LossFunction controls robustification; ceres::Solve optimizes the
Problem. Distinguish exact shared-parameter relationships from weighted residual
penalties. Name the implemented residual family and parameter blocks when discussing
temporal fitting, rather than describing it only as smoothing.


## Linked sequence with an observation gap

Experiment 05 calls `fit_linked_sequence` in `cpp/src/linked_sequence.cpp`.
One Ceres Problem owns two QuaternionManifold parameter blocks and one shared
Euclidean joint-position parameter block per timestamp. LandmarkResidual blocks
use observed landmarks only, with trapezoidal time weights and a 10 mm position
scale. Each interior timestamp adds a TranslationAccelerationResidual on joint
positions and two QuaternionAccelerationResidual blocks on segment rotations.
The scales are 3000 mm/s^2 and 20 rad/s^2, respectively; these remain illustrative.
Each temporal block connects three timestamps. Ceres AutoDiff provides derivatives;
SPARSE_NORMAL_CHOLESKY solves the coupled sequence. Attachment coincidence remains
exact through shared parameterization, not a residual penalty.

Initialization uses connected fits on observed frames. Inside a child gap, parent
observations initialize the parent quaternion and joint position. Child quaternions
start from Slerp between surrounding observed-frame fits, using timestamps. These
are initial parameter values only: Ceres subsequently optimizes every quaternion
and joint block. No known synthetic pose or withheld target enters initialization
or residual evaluation. Initial blocks are returned for the gray visualization.

The current native boundary requires full parent observations at every frame and
child observations that are either complete or absent. Child gaps must be bounded
by observed frames. Arbitrary landmark masks and missing parents are later work.
The per-frame comparison provides no child estimate during a gap; null poses remain
null in plots and rendering. Amber shading marks the missing interval. The Ceres
inspector shows child quaternion blocks coupled through temporal residuals even
when there are zero child landmark residual blocks. It reports whole-problem counts;
the diagram is a local three-frame excerpt, not every residual touching that frame.

The sequence result reports costs for landmarks, joint acceleration, parent angular
acceleration and child angular acceleration. Tests cover compatible constant-velocity
recovery through gaps, exact attachments, unit quaternions, native block counts,
objective accounting, unit/time scaling, rejected unbounded gaps and exclusion of
withheld observations. Known-motion gap error evaluates this synthetic example;
it does not make occluded motion directly observed.


## Three-segment chain

Experiment 06 uses `cpp/src/chain_sequence.cpp` and `scripts/solver_chain_experiment.py`.
One Ceres Problem owns three WORLD wxyz quaternion parameter blocks and one root
XYZ parameter block per frame (164 blocks over 41 frames). QuaternionManifold keeps
all rotations unit length. A segment's translation is derived recursively:

```text
translation[0] = root
translation[b] = translation[b-1]
               + rotate(q[b-1], parent_attachment[b-1])
               - rotate(q[b], child_attachment[b-1])
```

`ChainLandmarkResidual` predicts a landmark from that derived translation and the
segment's rotated local point. Its DynamicAutoDiffCostFunction receives root XYZ
and every quaternion block from the root through the target segment. Distal residuals
therefore carry derivatives into upstream quaternion blocks. This FK function is
shared by residual evaluation and returned segment translations. Both attachment
relationships are exact; there are no separate joint-position blocks or attachment
penalties. Each interior timestamp adds four acceleration residual blocks: root XYZ
and the three world quaternions. Position and motion scales match experiment 05.

Root and distal segments stay observed. The middle segment has a bounded gap of
0, 5 or 11 frames. Its initial quaternion is Slerp between observed-frame fits; no
withheld observations or known poses enter Ceres. The synthetic motion includes a
bend centered in the gap, so this initialization does not already solve the problem.
The gray layer shows the feasible initialized chain; orange shows the optimized
chain. Both linkage separations and middle initialization/fitted errors are reported.
The middle attachment vector is constrained by distal data, but roll about that
vector remains unobserved in the gap and receives a temporal preference.

The inspector draws all upstream parameter connections, including distal residuals
when the middle landmark residual blocks are absent. Tests verify exact attachment
coincidence, noiseless compatible-motion recovery, unit/time scaling, objective
accounting and an isolated middle bend recovered from distal residuals. This stage
still has rigid segments and no joint-angle bounds or non-rigid displacement blocks.


## Navigating the Ceres map

`solver_problem.js` assembles the diagram data; `solver_map.js` owns layout and
interaction. Each timestamp has a separate column. Parameter blocks and grouped
landmark residual blocks share that column; temporal residual blocks sit below.
Colors and text tags distinguish quaternion/position parameters, chain/ordinary
landmark residuals, and angular/position motion residuals.

Connections are hidden until a block is hovered or selected. Hover traces direct
neighbors; click pins the selection, and the detail area lists the actual connected
block IDs, parameter values or residual formula. Unrelated blocks dim. All connections
can be enabled for an overview. This remains a grouped assembly schematic, not a
runtime Ceres graph dump. Neither the display nor selection changes fitting data.

Drag background to pan; wheel zooms about the pointer. Buttons provide zoom and Fit
all. Double-click a block to center/zoom/pin it. Keyboard users can Tab to nodes and
use Enter/Space to select; Escape clears, +/- zoom and F fits. Expand retains the
resizable panel behavior. View transforms persist during playback and reset to fit
when changing experiment/method. Fit mode adjusts to panel resizing automatically.


## Bounded non-rigid linkage experiment

Experiment 07 compares the same fully observed synthetic recording with a fixed
second linkage and with a scalar displacement at that linkage. All segment-local
landmarks and rigid segment dimensions remain fixed. Only the middle-to-distal
connection changes. For the second linkage:

```text
translation[2] = translation[1]
               + rotate(q[1], parent_attachment[1] + [0, 0, displacement])
               - rotate(q[2], child_attachment[1])
```

This scalar is expressed along the MIDDLE segment's local +Z axis, in millimeters.
`fit_chain_sequence` adds one Euclidean parameter block per frame when
`allow_displacement=True`. Ceres bounds it to [-40, 40] mm in this example.
Distal ChainLandmarkResidual blocks include that scalar block. A
DisplacementPriorResidual favors zero displacement with a 20 mm scale and
trapezoidal time weighting. DisplacementAccelerationResidual connects three scalar
blocks, using a 500 mm/s^2 scale. These residuals each have ONE component, unlike
the three-component landmark and pose-motion residuals. Bounds are hard limits;
the prior and acceleration residuals are weighted preferences.

Displacement initializes to zero; it is not initialized from synthetic truth.
The no-displacement branch retains its existing parameterization and block counts.
The map shows the new scalar blocks, bounds, prior and temporal connections, and
reports 205 parameter blocks / 1220 residual blocks for the 41-frame enabled model.
A separate displacement plot compares the fitted scalar with known synthetic values.
The gold connection line depicts the modeled separation. Diagnostics distinguish
that separation from error in the displacement equation and report bound saturation.

All three segments stay observed here, deliberately isolating the new connection
model from missing-observation ambiguity. These axes, scales and bounds are synthetic
experiment choices, not human spine definitions or clinical/anatomical estimates.
Tests cover rigid geometry, displacement recovery, fixed-linkage agreement at zero,
hard bounds, residual-cost accounting and distance/time scaling. No production
skeleton definition or FreeMoCap pipeline is modified by this experiment.


## 08: displacement with missing middle landmarks

This experiment reuses the existing native chain Problem. It adds no residual
families or fitting rules. Both methods receive identical root/distal observations
and the same omitted middle observations. The bounded model retains one scalar
parameter per frame, its bounds, zero prior and acceleration residuals.

For this particular geometry, let A be the first attachment world position and B
be the distal segment's second-link attachment world position. Then:

    B - A = R_middle * [0, 0, 160 + displacement]

The +/-40 mm bounds keep this length positive. Exact A and B therefore determine
displacement and the middle local Z axis in world coordinates. They do not determine
roll around that axis. A local-Z quaternion rotation changes withheld middle
landmarks without changing either attachment or either observed neighbor. Tests
verify this invariance. This claim is specific to the axial attachment geometry,
not a general identifiability result for arbitrary skeletons.

Noisy observations and temporal/zero-displacement residuals can bias the fit.
Convergence means numerical termination, not recovery of the missing roll or ground
truth. Known synthetic trajectories are evaluation/display data, never residual
targets in the missing interval. The viewer exposes costs, gap errors, displacement
error and saturated-frame counts rather than claiming uncertainty estimates.

Next checkpoint: review these parameter/residual definitions against Forge's
landmarks, segments, linkages, chains and skeletons before adding more synthetic
stages or applying this model to a real torso. Synthetic scales are not anatomical
defaults.


## 09: fixed branching connections

The native sequence experiment now accepts parent_indices=[0,0] for two children
of segment 0, or [0,1] for the existing serial chain. It remains a three-segment
laboratory solver, not a production skeleton adapter. Displacement is rejected for
the branching topology. Both attachment equations are exact:

    t_child = t_parent + R(q_parent) a_parent - R(q_child) a_child

A child landmark residual references root XYZ, q_parent and its own q_child.
Sibling quaternions are excluded from that residual's parameter list. Both branches
therefore influence the same parent blocks. All poses use world wxyz quaternions
with QuaternionManifold. Existing acceleration residual blocks couple timestamps;
no new motion preferences or anatomical assumptions were introduced.

Forge interpretation: segments supply fixed local landmark geometry; linkages
supply attachment positions and parent relationships. Ceres owns the numerical
parameter blocks and weighted residual evaluation. The synthetic experiment is an
explicit example of those relationships, not a new ontology or tracker dependency.

Branch A can have a bounded observation gap. Unlike the serial chain case, Branch B
does not constrain Branch A's quaternion through a downstream attachment. Temporal
residuals bridge that gap; they do not provide measurements of the withheld motion.
Metrics include exact attachment errors, per-segment landmark RMS against known
truth, quaternion geodesic angular errors in degrees, and gap-only errors. Weighted
objective costs remain separate. Swapping fully observed branches must preserve
results after reordering; tests exercise this symmetry and the earlier chain modes.


## 10: general tree assembly before sparse torso fitting

The native path assembly accepts a topologically ordered rooted tree: segment 0
is the root, and parent_indices[b-1] is the parent of segment b. Parent indices
must precede the child. Every landmark residual receives only the root-position
block and quaternions on its ancestor path. Child observation gaps require
observed endpoints for independent-fit/Slerp initialization. The root is fully
observed. Sparse per-segment landmark subsets and explicit initialization are not
yet supported. Legacy displacement remains restricted to the original three-body
axial chain; no displacement is inferred for other topologies.

The five-segment viewer example uses dense synthetic observations on simple rigid
objects. It tests tree assembly, not anatomical geometry or COCO reconstruction.
A complete four-landmark torso solve still needs observation bindings independent
of display geometry, an initialization that does not require three observed
landmarks per segment, and explicit priors on undetermined pose coordinates.


## 11: rigid torso from four landmarks

This case loads the existing SkeletonDefinition and RestPose. The selected segments
are pelvis, sacrolumbar, thoracic and both clavicles. Parent-owned connect_at
landmarks supply attachments. The authored model is scaled to 1700 mm; this is a
controlled synthetic input, not a size estimate from four measured points. All
segments stay rigid and every linkage stays connected, with no displacement.

Native local/observed lists contain only left/right hip_socket and left/right
acromion targets (2, 0, 0, 1, 1 per segment). Display geometry contains other authored
landmarks, but those never enter the landmark residuals. Initialization uses the
observed hip midpoint, the hip axis and shoulder-midpoint-to-hip-midpoint axis to
form an orthonormal basis, then composes authored relative rest quaternions. Those
midpoints are initialization only. Synthetic reference poses are not supplied to
the solve or used to initialize it. The generated reference motions include both
shoulder elevation, left-only shrug and thoracic twist.

The native explicit-initialization path validates quaternion units, dimensions,
finite inputs and matching observation/local arrays. It accepts segments with zero
observed landmarks. The previous independent-fit initialization remains the default
for densely observed experiments.

Both methods have identical observations, initialization and acceleration residuals.
The prior-enabled method adds one RelativePoseResidual per linkage per timestamp:

    error_q = inverse(rest_relative_q) * inverse(parent_world_q) * child_world_q
    residual = sqrt(trapezoid_time_weight) * principal_Log(error_q) / 2 radians

Quaternions are wxyz parameter blocks on QuaternionManifold. The principal log is
only a residual calculation, never the stored pose representation. The 2-radian
scale is an explicit experimental preference, not a measurement uncertainty or
anatomical bound. The prior has its own reported cost and graph/filter type. No root
rest-pose prior is added.

With 21 frames: 126 parameter blocks; 198 residual blocks without the relative pose
prior, 282 with it. Four landmark blocks per frame each have three components.
Four relative-pose blocks per frame each have three components when enabled.
The remaining 114 blocks penalize root/quaternion acceleration. Loss is squared.

Default 1 mm noisy shoulder elevation: no-prior observed RMS is about 0.82 mm,
but the solve reaches its 200-iteration limit and clavicle angular errors exceed
100 degrees. With the prior it converges at about 0.87 mm observed RMS; spine
angular RMS is roughly 2 degrees and clavicle error roughly 11 degrees. These are
synthetic evaluation results, not real-data accuracy claims. The unobserved clavicle
roll and internal spine poses remain model-dependent even when Ceres converges.
The viewer preserves and reports non-converged results; it never labels them solved.


## 12: prepared real recording

The same five-segment native solver and residual scales are applied to saved direct
hip/acromion landmarks. Read-only Parquet loading checks units, reference frame,
source mapping and source hash. The model uses the recording's skeleton snapshot,
rest relative quaternions and fixed per-segment ModelScaleFit. Inputs retain saved
timestamps (the test recording is decimated); no synthetic frame rate is substituted.
Synthetic truth fields are null and the renderer/plots explicitly handle that case.

Initial test-recording result (frames 0-215): 1296 parameter blocks, 3012 residual
blocks, observed RMS 5.30 mm, 95th percentile 11.55 mm, maximum 45.99 mm. Shoulders
have roughly 2.4-2.5 mm RMS; hips roughly 7 mm. Ceres reached 200 iterations with
NO_CONVERGENCE. This is exposed as an intermediate review, not a validated solved
motion. The next numerical investigation should examine termination and internal
pose stability rather than interpreting low observed error as anatomical accuracy.
The last six frames lack targets and are omitted explicitly. No source file changes,
no full-body solve, no deformable segments and no production pipeline changes.


## Axial spine comparison (experiments 12 and 13)

The owner approved trying variable-length sacrolumbar and thoracic spans after the
rigid real-recording baseline visibly buckled during bending. This is an explicit
experimental deformation policy; RigidBodySegment and production hydration remain
unchanged. Each selected segment has a positive reference axial extent, taken from
its authored chest_center or neck_center local Z after saved model scaling.

    deformed_local(x,y,z) = (x, y, z * length / reference_length)

Every owned landmark and parent/child attachment uses that rule inside the native
Ceres residual and forward calculation. Child origins stay exactly attached to the
deformed parent point. The thoracic sternoclavicular pair has equal authored local Z,
so both attachments move equally along Z; their X/Y offsets and pairwise spacing stay
fixed. Pelvis and clavicles retain rigid geometry. There is no linkage gap.

Per frame the flexible model adds two scalar length parameter blocks. Lengths start
at their references. Positive bounds are 0.25 to 2 times reference; these are broad
experimental domain guards, not anatomical limits. LengthPriorResidual has:

    r = sqrt(time_weight) * (length - reference) / (0.25 * reference)

Scalar acceleration uses the existing DisplacementAccelerationResidual with length
values and scale 500 mm/s^2. The name denotes its scalar finite-difference operation;
it does not introduce attachment displacement in this model. Other residual scales,
observations, timestamps, initialization quaternions/root and T-pose are unchanged.
Native results include length values and separate prior/acceleration costs.

The viewer's length plot distinguishes reference lengths (real data) from known
synthetic lengths (experiment 13). Its parameter map shows length bounds, prior and
temporal residual blocks, and each descendant landmark's length dependencies.
Recordings show spine bend angle as a diagnostic, not a prescribed zero-bend target.
Length distribution remains model-dependent with four landmarks; lower residuals do
not establish anatomical segment lengths. Bound saturation is reported explicitly.

Experiment 13 uses dense synthetic landmarks on a tree with two spans contracting
from 80 mm to 56/64 mm. Rigid and flexible methods use identical data. Native tests
recover known lengths with weak priors, verify deformed attachment coincidence,
reference-length equivalence, quaternion/legacy solver regressions, and preservation
of the upper attachment group's shape. Display deformation uses solved lengths;
there is no visual alignment correction or modification of recorded landmarks.


First real-data comparison outcome: flexible observed RMS 5.24 mm versus rigid
5.30 mm, with zero frames at length bounds. At frame 192 the rigid reference lengths
are 272.08 / 269.50 mm; the flexible result is 278.32 / 277.33 mm. The angle between
spine Z axes is 53.59 degrees rigid versus 55.13 degrees flexible. Thus this first
preference configuration does NOT resolve the observed buckling. Both sequence
solves reach the unchanged 200-iteration limit. These facts are retained in the
viewer; neither lower observation RMS nor synthetic recovery is claimed to validate
the real internal spine motion. Exact attachment errors stay below 3e-13 mm.

The synthetic 1 mm noise case estimates contraction lengths 56.97 / 63.79 mm at peak,
versus known 56 / 64 mm. This validates the implemented deformation capability and
makes the real-data modeling limitation distinct from a missing length parameter.

## 14: connected full-body recording review

Shared experiment tuning is named in `scripts/solver_fit_settings.py`: position,
root/angular acceleration, relative rest pose, axial length prior and acceleration,
plus diagnostic tolerances. These values are passed explicitly to Ceres and saved
in each new run's settings. Native defaults, length bounds and termination tolerances
are named in `cpp/include/skellyforge/chain_solver_constants.h`; native length
defaults/bounds are exposed to Python rather than copied into diagnostics. Changing
C++ constants requires rebuilding the extension. Changing experiment settings
requires regenerating the solve. Existing saved runs keep their original settings.
The current length prior is still symmetric; this cleanup changes no values.

The next full-body comparison adds `--variant wider_symmetric` (fractions 0.5/0.5)
and `--variant shortening_biased` (0.5 shortening, 0.25 lengthening). The reference
length remains the zero-cost length. Each branch uses the same LengthPriorResidual
block with its selected inverse scale; no additional parameter or residual blocks
are introduced. The squared cost and its gradient are continuous at the reference;
its curvature differs on either side. The Ceres map reports both scales.

Bounds, temporal residuals, rest-pose residuals, mapping targets and initialization
are unchanged. New methods retain the cached baseline only after checking recording
hash, model, timestamps, targets and initial poses. Each method retains its native
build provenance. These are actual independent solves, not adjustments to the saved
baseline or changes to rendered geometry.

Run `uv run --no-sync poe solver-viewer-body` from SkellyForge. Defaults to recording
frames 180–213, including the bend near 192 and raised arms near 200. Use
`--start N --end M` to select a different consecutive interval. This is a full-body
solve over a declared time window, not a full-recording result. The annotated
previews remain aligned to original recording frame numbers.

The saved SkeletonDefinition supplies all 61 segments and every landmark, including
head, fingers and feet. JointDefinition supplies the tree and connection landmarks;
saved rest quaternions and owner-specific model scales supply the geometry. Only
sacrolumbar and thoracic local-Z lengths vary, using the existing axial experiment.
All other segment geometry is rigid. Every joint remains connected exactly.

Keypoints are measurements. Existing direct mappings identify the measurement
targets for model landmarks. Derived mappings are not counted as additional
independent measurements in this experiment. A keypoint mapped to both a parent
endpoint and its child's coincident origin contributes once; the adapter verifies
that relationship. A source keypoint unavailable in a frame contributes no target
residual. It never removes a landmark from the model. Indexed target lists preserve
the remaining correspondences without requiring a whole segment's targets to drop.

Initialization uses saved segment world quaternions and root position. Where a
quaternion is unavailable, it uses the parent's initial quaternion composed with
the authored relative rest quaternion. These initial values are not fitted targets
or ground truth. The same existing residual scales are used throughout; this is a
baseline experiment, not a claim of an anatomically validated solution.

The viewer provides two separate color modes: rigid/axial geometry and direct
measurement support. Three non-collinear local targets can determine a rigid pose;
fewer cannot determine that pose in isolation. This classification does not prove
global observability, and it does not count priors as measurements. Hover shows the
segment identity, model type and target support. Region/segment focus filters the
Ceres residual groups while retaining their parameter dependencies. Counts remain
whole-problem totals. Every displayed fitted axis has an origin sphere and label.
Saved posthoc context and the fitted model remain separate, explicitly labeled layers.

First full-body result (frames 180–213): CONVERGENCE in 43 iterations, 536 seconds;
2,176 parameter blocks and 6,205 residual blocks. Mapped-target RMS is 24.10 mm,
maximum attachment error 3.46e-13 mm, and no axial lengths reach their bounds.
Right arm and right hand target RMS are 42.13 and 32.66 mm respectively. At frame
192 the axial lengths are 296.17 / 286.36 mm and the angle between their Z axes is
68.22 degrees. These are diagnostics, not ground-truth error or proof that the spine
issue is resolved. The torso-only experiment uses different targets and a different
time interval, so its overall RMS is not a like-for-like performance comparison.

### Full-body length-prior comparison

Same frames 180–213, all 61 segments, the same 2,049 mapped targets, initial poses,
hard bounds and other residual scales. The baseline is cached; the two new cases
are independent Ceres solves. Neither new case converged within the configured
200 updates (Ceres reports 201 iterations including the initial entry).

| Method | Shortening / lengthening fractions | Target RMS, mm | Frame 192 sacrolumbar / thoracic lengths, mm | Frame 192 spine Z-axis angle | Status |
| --- | --- | --- | --- | --- | --- |
| Baseline | 0.25 / 0.25 | 24.097 | 296.17 / 286.36 | 68.22 degrees | Converged |
| Wider symmetric | 0.5 / 0.5 | 23.780 | 310.29 / 289.22 | 72.57 degrees | Iteration limit |
| Shortening biased | 0.5 / 0.25 | 23.785 | 302.78 / 289.46 | 70.66 degrees | Iteration limit |

Neither new case reaches a length bound; attachment errors remain below 5e-13 mm.
The bias reduces lumbar extension compared with the wider symmetric case at this
frame, but neither new result reduces the spine-axis angle below the baseline.
These are provisional optimization results, not evidence that the spine issue is
resolved. Lower target RMS alone does not validate the internal spine geometry.


### Free spine-length diagnostic

`python -m scripts.solver_recording_body --variant free_lengths` solves the same
full-body recording window (180?213) and retains the existing comparisons.
`free_axial_lengths=True` removes LengthPriorResidual, scalar length acceleration,
and the reference-relative bounds for sacrolumbar and thoracic lengths. Only
L >= 0 remains; there is no upper bound. Local X/Y coordinates remain fixed;
local Z coordinates and joint attachment positions follow the existing axial
length model. Joint coincidence remains exact. Root/rotation temporal residuals
and relative-quaternion rest-pose residuals are unchanged. Cervical, pelvis,
clavicles and limbs remain rigid. This is a diagnostic, not a production default.
Chest center remains a defined fitted landmark; its derived mapping is displayed
as context, not counted as another independent keypoint measurement residual.
Freeing these lengths does not guarantee Ceres convergence or a unique pose.


2026-09-26 diagnostic result (recording frames 180?213): free spine lengths
converged in 46 iterations / 10.94 s after restoring compiler Release flags.
6073 residual blocks; both removed length cost families are zero. Maximum joint
attachment discrepancy is 3.44e-13 mm. At frame 192, sacrolumbar/thoracic lengths
are 213.08/281.21 mm and the angle between their +Z axes is 35.41 degrees
(baseline 68.22 degrees). These are fit diagnostics, not anatomical ground truth.

A same-input baseline benchmark isolated the build correction: 536.13 s before,
10.44 s after (51.37x), both 43 iterations, identical final cost and segment
translations. Solver tolerances, thread count, model and residuals were unchanged.
These timings are for this machine/window, not a full-recording throughput claim.
The benchmark artifact is build/full-body-optimized-baseline.json; the free
comparison is saved in the viewer. Ceres tests: 74 Python + 2 native passed;
viewer controls/map checks passed (mocked DOM/WebGL, not screenshot validation).


### Chest-center line preference (2026-09-26)

Run `python -m scripts.solver_recording_body --variant chest_line` for the
free-length comparison with a line preference. Native `LandmarkLinePrior`
specifies the existing chest_center local landmark position and owning segment,
per-frame line origin/lateral/anterior axes, and two residual scales. The new
`ChainLandmarkLineResidual` reuses connected forward geometry and references only
the root, ancestor/world quaternions and applicable axial lengths. No new
parameter blocks or separate fitted landmark positions are introduced.

`scripts/solver_chest_line.py` constructs the line from the existing direct
mappings of left/right hip_socket and acromion to their source keypoints. Up is
shoulder center minus hip center; anterior is normalize(cross(up, right hip minus
left hip)); lateral is cross(anterior, up). This frame follows the measured body,
not a global coordinate direction. A frame with unavailable source keypoints or
a degenerate basis has no line-preference block; landmark definitions remain.

With lateral displacement s and signed anterior displacement a (posterior < 0),
the three residual components are sqrt(time weight) times:
[s / distance_scale, a / distance_scale, max(0,a) / anterior_scale].
There is no along-line/midpoint penalty. The squared anterior-only penalty is
continuous with continuous first derivative at zero. The line preference uses
correlated hip/shoulder evidence as a modeling preference, not an independent
measurement. Initial named scales are 50 mm near-line and 20 mm extra-anterior;
smaller scales strengthen the preference. These are experimental choices.

The viewer adds a toggleable yellow line, labeled endpoints/projection, pink
chest displacement, an anterior reference vector, and a resizable Plotly panel
for lateral/anterior displacement. Cached prior fits receive only this review
geometry; their poses and objectives are unchanged. Only the new method has the
line blocks in its Ceres map. Fit output, overlay and reported line cost agree.

First 34-frame run: 148 iterations, CONVERGENCE, 28.83 s, 2176 parameter blocks,
6107 residual blocks (34 new). Line-distance RMS: 54.56 -> 26.02 mm compared with
free lengths. This measures the preference's effect, not anatomical accuracy.
Native line cost 0.7526182477654242 matches the independently reconstructed cost.
80 Python + 2 native tests passed; viewer control/graph/plot tests passed with
mocked DOM/WebGL (not screenshot validation).
