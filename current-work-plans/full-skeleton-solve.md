# Full skeleton solve: proposed next stage

Status: design proposal, not implemented or validated production behavior.

Current design direction (2026-09-25):
[composable skeleton fitting and non-rigid linkages](composable-skeleton-fitting.md).
It specifies the proposed ontology extension, quaternion parameter blocks,
residual equations, Ceres implementation boundary, and validation order. The
fixed-dimension prototype below records previous work, not the selected spine
model. Non-rigid connections are explicitly in scope; no production definition
or calculation has changed yet.

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

## Sequence-fitting implementation, 2026-09-24

The upper-body synthetic viewer now compares 11 target positions with known
connected geometry. Its overhead example fits positions closely while retaining
substantial differences in joint rotations. That is evidence that small point
residuals alone are insufficient; it is not a reason to snap rotations to the
known synthetic answer. Static 1 mm noise also produces several degrees of joint
variation with independent fits.

The next fit uses the existing FK and per-frame observation/pose residuals in one
time-window objective. It does not filter independent fitted quaternions after
the fact. Inputs retain the per-frame initial root and local rotations, selected
landmark targets, fixed dimensions and strictly increasing timestamps in seconds.
The whole supplied window is solved jointly, using future and past observations.
No realtime equivalence or long-recording window stitching is implied.

Explicit objective (all terms are squared norms):

- Observation and pose-prior residuals at frame i are multiplied by sqrt(w_i),
  where w_i is the trapezoidal integration weight in seconds. This keeps the
  relative strength of the priors from depending implicitly on frame rate.
- Joint motion between neighbors uses the shortest rotation vector of
  conjugate(q_i) * q_(i+1), divided by angular-velocity scale * sqrt(dt).
  Root rotation uses the same formula. Root translation uses the displacement
  divided by linear-velocity scale * sqrt(dt). These implement an integrated
  squared-velocity preference, not an anatomical angular limit.
- All positional, rotational and velocity scales are explicit caller settings.
  Synthetic demonstration values are not production defaults or calibrated
  uncertainty. A velocity preference can attenuate real fast motion; validation
  must include moving cases as well as static noise.

Pose priors select among ambiguous configurations and motion priors discourage
frame-to-frame changes. Neither provides new measured information. In particular,
a smooth wrong twist remains possible. Report observation residuals, convergence,
missing target support and the settings used. Do not describe low residuals as a
unique or anatomically verified reconstruction.

Missing observations supply no residual. A fully missing frame may be supported
by neighboring frames and its explicit pose prior; an entirely unobserved window
is rejected. No targets are synthesized to conceal missing observations.

Use an established sparse least-squares implementation for the window solve;
SciPy's trust-region solver supports finite-difference Jacobian sparsity and
rank-deficient problems. SciPy is an owner-approved required runtime dependency.
Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html

Validation gates: unchanged exact static sequences; reduced jitter without broken
connections or changed lengths; actual timestamps (including irregular spacing);
rigid world-transform and consistent unit-change equivariance; missing frames;
explicit non-convergence; and a moving synthetic comparison exposing attenuation.
The synthetic viewer will show independent and sequence fits against the same
targets and known reference. The saved-recording viewer remains a replay of
production results.

## Real-data diagnosis: prepared test recording, frames 196–203

The experimental review uses the saved skeleton/scale fit and unchanged direct
landmark observations, with keypoints checked against the recorded mappings.
The recording remains read-only. The connected result is the sole source of its
segment poses; the orange independent poses are a baseline, not a second stage
to preserve in the eventual production connected pipeline.

At frame 200 the saved left upper-arm length is 286.0 mm, while the direct
shoulder/elbow observations are 244.9 mm apart. Fixed geometry therefore requires
at least one of those endpoint errors to be at least 20.5 mm. No optimizer or
temporal smoothing can remove that discrepancy while preserving both dimensions
and observations. This does not identify whether dimension estimation, tracking,
or landmark correspondence is responsible.

Separating initialization from joint pose priors is implemented: the real review
now uses authored rest rotations from the saved model as joint preferences,
rather than preferring the same independent segment calculation it initializes
from. The fixed dimensions and all weight values were unchanged. This did not
resolve the backward spine bow: the window fit still bends about 37 degrees at
the lumbar/thoracic boundary at frame 200.

A separate single-frame diagnostic (not shipped as a new model/default) compared
free middle-spine rotation with its relative rotation fixed at the authored rest
value. The free result bent 39.1 degrees, with maximum target error 23.6 mm and
total objective 28.19. The straight case had maximum target error 24.4 mm and
objective 39.30. Hip/head residuals increase in the straight case; it is not an
equally good objective value or proof that a straight spine is ground truth.
It shows that endpoint fit quality alone does not select an acceptable internal
spine pose. The general-purpose ball-joint/pose-prior formulation is insufficient.

Next model decision: represent non-rigid spinal connections explicitly in the
linkage/chain ontology and compose their fitting calculations with landmark,
quaternion pose, and time-domain residuals. See the current design linked above.
Do not silently lock the thorax, alter ModelScaleFit per frame, or disguise
connection displacement as rigid segment geometry. Review model-scale fitting
and shoulder attachment geometry alongside the new linkage definitions.
