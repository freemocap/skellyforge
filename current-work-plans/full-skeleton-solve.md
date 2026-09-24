# Full skeleton solve: proposed next stage

Status: design proposal, not implemented or validated production behavior.

Detailed repository review and proposed interfaces:
[observation and shoulder contract](full-skeleton-observation-contract.md).

The real recording viewer now reads production results without fitting. Keep
that boundary. Do not improve visual agreement by changing lengths, positions or
rotations inside the viewer. Independent segment reconstruction is useful for
initialization and diagnostics; it is not a connected whole-body solution.

## First decision: observations and model quantities

Inventory the producer's keypoint-to-landmark mappings and classify direct
observations, estimates formed from several observations, and model-derived
offsets. Preserve original tracker data and provenance. Do not count a shoulder
observation and several points derived from that shoulder as independent evidence.
Tracker confidence scores are not calibrated spatial uncertainty by default.

The connected thorax determines SC attachment positions in its own frame. The
shoulder midpoint must not hard-position the thorax or cervical base while also
being used to fit shoulder motion. Model landmark predictions and mapped
observation estimates must remain distinguishable in data and visualization.

## Proposed model and solver contract

Candidate state: global root transform and joint coordinates, with recording-wide
dimensions and parent attachments. Forward kinematics generates every segment
pose and predicted observation from that state. The optimizer adjusts the state
jointly; downstream residuals can therefore change upstream joint coordinates.
Do not independently overwrite child world rotations after solving.

Start with pelvis, spine, shoulder girdle, arms and head as one connected problem.
Use hips, shoulders, elbows, wrists and head evidence to constrain it. Design the
API for whole-body use; do not build a second torso-only representation. Evaluate
whether the current clavicle/shoulder structure can represent elevation and shrug
before selecting constraints. Add scapular motion only with an explicit model,
evidence rationale and sensitivity checks. A rigid thorax does not mean a fixed
world pose; clavicle motion can coexist with real torso motion.

Specify observable quantities, remaining ambiguities, joint freedoms and limits,
robust observation residuals, temporal terms, initialization and failure behavior
before implementation. Post hoc can use neighboring frames; define that policy
explicitly rather than inheriting realtime filters accidentally. Regularization
must be disclosed and tested, not introduced as unexplained visual damping.
Retain fixed dimensions initially; any compliant dimensions are a separate model
decision with explicit bounds and regularization, never a drawing adjustment.

## Repository responsibilities and order

1. Review observation semantics and the connected geometry; agree on the first
   solver formulation and acceptance criteria before writing a replacement.
2. Implement and test the generic solver in Forge. Tracker provides detector
   observations; Forge owns geometry and fitting; FreeMoCap owns adapters,
   orchestration and publication. Revisit mapping ownership where model-derived
   anatomy currently lives in Tracker. No cross-subskelly imports.
3. After the human commit/update boundary, run the actual FreeMoCap post hoc
   stage on saved keypoints and publish solved poses and diagnostics in Parquet.
4. Compare saved observations and saved predictions in the viewer. Exporters
   consume the same solved state; they must not solve the skeleton again.

Validation: exact synthetic connected motion; static jitter; bilateral and
unilateral arm elevation; shrug; trunk bend/twist; missing/occluded evidence;
outliers; near-straight limbs; frame-rate and temporal-boundary behavior. Test
connections, fixed dimensions, finite rotations, transform equivariance,
convergence/failure reporting and preservation of observations. Real-data review
tests plausibility and failure modes, not anatomical accuracy without ground truth.

References to review, not mandates to adopt their full anatomical models:
- OpenSim inverse kinematics: weighted observation residuals over connected model
  coordinates: https://opensimconfluence.atlassian.net/wiki/spaces/OpenSim/pages/53090032
- Shoulder-model representational choices and validation against independent data:
  https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0141028

The old experimental two-shoulder fitter is not the accepted full-skeleton
solution. Reuse only independently validated machinery that fits this contract.
