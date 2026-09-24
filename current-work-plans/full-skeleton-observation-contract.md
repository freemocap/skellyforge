# Connected reconstruction: observation and shoulder contract

Review date: 2026-09-24. Proposal only; no production calculation changes.
This is the next-stage design for `full-skeleton-solve.md`.

## Findings from current code

Both RTMPose and MediaPipe body mapping YAMLs in Tracker currently map:

| Evidence | Existing mapped names | Proposed solver use |
| --- | --- | --- |
| Each shoulder keypoint | acromion, shoulder | One residual per source observation, with an explicitly chosen model attachment |
| Each hip keypoint | hip_joint, hip_socket | One residual per source observation |
| Each wrist keypoint | wrist, carpal_origin | One residual per source observation |
| Left/right ears | ears, head_center, craniocervical_junction | Prefer individual ears; their mean is not additional evidence |
| Shoulder midpoint | neck_center | Initialization/diagnostics; not a measured thoracic endpoint constraint |
| Mean of hips and shoulders | chest_center | Initialization/diagnostics; not an independent torso observation |
| Shoulder-based offsets | SC joints, SC notch, xiphoid | Model-derived estimates; not independent residual targets |
| Elbow, knee, ankle, supported foot/face points | direct aliases | Selected observation targets with documented semantics |

Source files: Tracker's `core/detectors/keypoint_detectors/{rtmpose,mediapipe}/body/`
mapping YAMLs. A named detector keypoint is an estimated image feature, not proof
of the anatomical joint center its alias suggests. Duplicate aliases must not
increase evidence weight. Derived estimates can be used if their dependency and
covariance are handled explicitly; the initial formulation avoids counting them.

Forge `definitions/human_skeleton/components/arm.yaml` attaches the upper arm
directly to the clavicle's acromion landmark. The spine component owns the SC
attachments. Both clavicle and shoulder links are declared ball joints. There is
no scapular segment between them. This is a simplified shoulder girdle, not a
fully represented anatomical shoulder complex.

`core/skeleton/pose/fit_connected_pose.py` currently optimizes three rotation
increments per selected joint, with fixed root, scalar target tolerances and a
penalty toward supplied local rotations. It has no joint-coordinate constraints,
outlier loss or temporal objective. It cannot distinguish observation provenance.
Reuse FK and tested rotation operations; do not assume this solver is sufficient.

FreeMoCap's `core/reconstruction/recording_reconstruction.py` supplies 3D arrays
and names, but no spatial covariance or per-observation quality array. Do not
pretend detector scores are available or calibrated uncertainty at this boundary.

### Reproducibility defect to fix first

`core/skeleton/skeleton_snapshot.py` does not capture or restore a segment's
`observation_frame`. A diagnostic of
`SkeletonSnapshot.capture(SkeletonDefinition.from_default_yaml()).restore()`
loses that field for both pelvis and thoracic. A restored model can therefore use
a different hydration path. FreeMoCap's fit-input model fingerprint also uses
this snapshot, so observation-frame-only edits are absent from that fingerprint.
The saved-output viewer does not hydrate and is unaffected by that particular
round-trip defect. The latest refresh explicitly built the current model, but
future replay/reconstruction must not depend on that workaround.

Add faithful optional frame serialization and round-trip tests in Forge. Old
snapshots with no field must remain identifiable as lacking that definition;
never fill them silently from today's defaults. Then verify fingerprint changes
and producer replay in FreeMoCap after the normal repository handoff.

Forge implementation now adds an optional observation-frame snapshot, captures
and restores it, and tests reconstructed pose equivalence and legacy absence.
FreeMoCap integration is pending. Specifically check old model/fit fingerprints:
serializing a newly available null field may change the canonical representation
even for an old model. Do not bypass a mismatch or label old results current;
define compatibility/rebuild handling and exercise old recording fixtures there.

## Proposed input and output interfaces

Names below describe responsibilities, not committed class names.

- Observation batch: immutable frame numbers, timestamps, source IDs, XYZ values,
  coordinate frame/units, missing mask, and optional explicitly typed quality or
  covariance. Retain whether inputs are raw or already filtered.
- Observation bindings: each unique source ID maps to a predicted model point
  (segment plus local attachment). Record the semantic assumption, source
  dependencies, and residual scale. No tracker-package imports in Forge.
- Subject model: complete skeleton geometry, joint coordinate definitions,
  attachments, fixed dimensions, model version and parameter provenance.
- Solve settings: active coordinates, priors, robust-loss scale, temporal policy,
  convergence thresholds and initialization policy. Persist these with results.
- Result: one connected state per frame, all segment transforms and predicted
  model landmarks derived from it; residuals keyed by observation ID; convergence,
  missing-support and prior-dominated flags. Preserve observations separately.

Never overwrite mapped observation landmarks with model predictions under an
ambiguous channel name. FreeMoCap must publish an explicit distinction in Parquet
and the viewer must use the saved values. Exporters consume the solved state.

## Candidate state and objective

Use root translation and rotation plus local joint coordinates over time.
Root pose is estimated jointly; the hip midpoint initializes it rather than
hard-locking it. Dimensions and attachment offsets remain fixed within a solve.
Every candidate is evaluated by FK. Joint connections are exact by construction;
observations have residuals. Shoulder residuals can change spine coordinates as
well as clavicle and arm coordinates. Do not independently correct child world
rotations afterward.

Proposed objective, in plain text:

    observation loss + declared pose priors + time-scaled motion regularization

Observation loss uses residual vectors between unique source points and predicted
attachments. Begin by comparing squared loss with a documented Huber loss under
controlled outliers; select scales through explicit noise assumptions and
sensitivity tests. Do not derive uncertainty solely from detector confidence.
No scalar weight turns duplicate observations into independent measurements.

Use local SO(3) increments for unrestricted rotational coordinates, with separate
declared hinge/swing/twist parameterizations where the model actually defines
them. Do not turn Euler display conventions or a `ball` label into invented
anatomical limits. Physical coordinate bounds require separately reviewed axes,
references and ranges. Constraints and prior weights are model parameters, not
hidden optimizer damping.

For post hoc, fit a time sequence/window using actual timestamps; permit future
evidence. Define window overlaps, boundary conditions and missing-data behavior.
Compare raw and prefiltered inputs explicitly to avoid unintentional double
smoothing. Realtime will require a separate causal evaluation of the same model;
do not claim equivalence to a future-aware post hoc result.

Sparse points leave spine-bend distribution and axial rotations ambiguous.
Temporal/rest priors select among compatible states; they do not create measured
anatomy. Diagnose support using the observation Jacobian before adding prior rows.
Do not report full statistical covariance without justified noise assumptions.

## Shoulder geometry decision and evaluation gate

For the current topology, a clavicle tip lies on a sphere centered on its SC
attachment, with radius equal to its fixed length. Its endpoint supplies two
rotational constraints; twist about its own axis is not determined by that point
alone. More iterations cannot remove this geometric restriction.

First evaluate that simplified representation with a fixed thorax in controlled
synthetic elevation, shrug and protraction cases, then with free thorax and
independent head/hip/arm observations. Report unreachable residuals rather than
pulling the torso to hide them. Construct out-of-model tests too: tests generated
only with the same FK cannot establish representational adequacy.

Compare a shoulder-girdle model that separates lateral clavicle attachment from
upper-arm center if the existing model fails that gate. Do not add an unconstrained
scapula pose: the current observations cannot uniquely identify it. Any added
scapular motion must use a declared geometric surface/coupling and report its
model-dependent status. No fixed scapulohumeral ratio is adopted by default.

Do not enforce that the thorax never moves when arms rise. It must be allowed to
move when supported by the whole observation set. The requirement is to remove
the hard algebraic dependence on shoulder midpoints, not to freeze the chest.

The first numerical prototype should exercise pelvis, spine, head and both arms
jointly using the generic skeleton API. Its scope must be explicit; unsupported
hands/legs must not be advertised as solved. Whole-body extension then adds their
observations and joint coordinates to the same state and objective.

## Acceptance and implementation order

1. Fix snapshot fidelity; test model and fit fingerprints across the integration
   boundary. No dependency on the new solver is needed for this correctness fix.
2. Implement unique observation bindings and saved provenance. Test duplicate
   aliases, missing points, coordinate transformations and tracker-equivalent
   inputs in the appropriate repositories.
3. Build geometry/observability diagnostics and review shoulder-model alternatives
   before selecting topology and numerical weights.
4. Implement the connected sequence solve with diagnostic outputs. Exact in-model
   cases must recover observable predictions, not an unidentifiable unique joint
   solution. Test fixed dimensions/connections, static jitter, asymmetric and
   overhead arms, shrug, bend/twist, occlusion, outliers, straight limbs and actual
   sample timing. Test arbitrary rigid transforms and dimensionally consistent
   changes of units. A prior must not be mistaken for independent evidence.
5. Integrate one production post hoc stage in FreeMoCap after commit/update;
   publish observations, solved poses and diagnostics together. Replay those rows
   directly in the viewer and check frame 200 plus entire motion sequences.
6. Extend to whole body and exports only after this model/solver contract is
   stable. Scientific users retain observation trajectories and fit diagnostics;
   animation users receive the connected state with disclosed model assumptions.

Model dimensions need their own assessment: spine/clavicle dimensions currently
come partly from constructed points. Do not jointly adjust unconstrained lengths
and joint poses to reduce residuals. A future subject-calibration step must name
the evidence, priors, observability and resulting fixed parameters explicitly.

No production defaults for joint limits, uncertainty, temporal weights or scapular
coupling are selected by this review. Those require the geometry and sensitivity
experiments above. No anatomical-accuracy claim follows from this recording alone.

## References

- [OpenSim IK](https://opensimconfluence.atlassian.net/wiki/spaces/OpenSim/pages/53090032):
  connected model coordinates fitted to weighted marker/coordinate evidence.
- [Seth et al., 2016](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0141028):
  explicit scapulothoracic geometry and validation against independent bone-pin
  data. This supports reviewing representation separately from optimization; it
  does not establish identifiability from our sparse video keypoints.
