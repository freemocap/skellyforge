# Accepted fit: integration next, shoulder and foot refinement later

Owner decision, 2026-09-28: the current fit is sufficiently useful to move into
FreeMoCap post-hoc integration. This is an integration baseline, not a claim that
shoulder or foot behavior is finished. Preserve its mathematics during integration.

## Deferred fitting work

### Shoulder / scapula geometry

Observed: excessive clavicle axial roll and implausible clavicle-to-upper-arm
relative rotation, including in the synthetic fixture. Investigate the existing
local XYZ bases, rest-relative quaternions and available geometric information
before choosing new residuals. Do not presume that a scapula model alone fixes
all roll ambiguity. Develop the previously discussed scapula/thorax constraint
model as a separately reviewable extension: define estimated scapular geometry,
clavicle/scapula/humerus relationships, parameters, residual blocks and priors.
SkellyForge owns that math. Tracker keypoints remain observations; inferred
scapular landmarks and poses are model estimates, not new measurements.

Validate on isolated synthetic shoulder motions and both prepared recordings.
Keep actual native Ceres inspection and the simple viewer available. Compare
relative quaternions, landmark deviations, convergence and processing time.
No numerical shoulder change is authorized by this planning note alone.

### Foot segment bases and contact locking

Observed: foot segments slide and rotate on the ground. Review the geometric
basis calculation for the existing foot segments, then add explicit, tunable
contact preferences as a separate fitting extension. Distinguish stance, swing,
and heel/toe pivot behavior; do not freeze the whole foot throughout movement.
Candidate residuals penalize motion of selected contact landmarks in world space
and unwanted ground penetration during supported intervals. Contact strength and
allowed motion must be explicit parameters, not a viewer correction. Reuse the
established filtered-trajectory/ground-alignment evidence where appropriate;
ground alignment itself does not provide time-varying foot locking.

Validate static stance, walking/lift-off and pivot cases; measure drift during
stance, preserved motion during swing, fit deviations and added runtime. Exact
contact model and weights remain to be discussed. This work does not block the
first post-hoc integration.

## Integration order

1. Finish the distribution handoff. Clean-install validation currently fails on
   the published SkellyLogs archive's automatic package discovery (`notes` and
   `skellylogs`). Fix that in its repository with its own human commit/push, then
   repeat installed Forge validation with the dependency retained. Recheck this
   blocker against current code before editing; the human may have fixed it.
2. Confirm the packaged `fit_human` input/output contract and installed native
   extension on prepared test/sample data. Recording preparation still has script
   adapters; reuse their existing contract rather than inventing another solver.
   Keep the supported viewers available for comparing saved and integrated results.
3. In FreeMoCap, extend the existing staged post-hoc machinery. Current entry
   points include `processing_request.py`, `stage_dependencies.py` and
   `stage_execution_plan.py`; stage signatures and checkpoint invalidation already
   exist. Inspect their current state before implementation. Fit after gap filling,
   applicable filtering, person alignment and person-scale/segment preparation.
   Call the packaged Forge solver; never import its development scripts into core.
4. Specify and implement Parquet persistence using existing channel/descriptor
   conventions. Preserve keypoints and mapped landmarks as their existing data.
   Save fitted segment quaternions, translations, variable geometry/linkage state
   needed for faithful reconstruction, model/scale identity and solver provenance.
   Do not mistake a quaternion/translation-only rigid export for the complete fit.
   Decide exact channel identities against the current ontology before coding.
5. Connect the optional fitting stage to the processing UI and existing progress,
   cancellation and failure reporting. Show window progress and usable/nonconverged
   status honestly. Reuse signature invalidation; never treat stale output presence
   as success. Leave the real-time segment path unchanged at this milestone.
6. Integrate the fitted skeleton into FreeMoCap's recording playback viewer.
   Retain the current segment rendering and its existing controls. Add a separate
   fitted stick-skeleton rendering layer using the simplified Forge viewer's
   visual conventions, including the blue/white stick styling and visible local
   XYZ axes. Provide independent toggles for the fitted skeleton and its axes,
   alongside the existing segment layer. Use saved fitted quaternions and the
   full fitted geometry from Parquet; do not refit, reposition or infer roll in
   the renderer. Keep hover identities and distinguish the two representations.
   Both layers must follow the same selected recording frame during playback,
   pause, scrubbing, stepping and looping, including annotated-video alignment
   where available. Recordings without fitted output must retain normal playback.
7. Run both real recordings through the actual application stage. Reload Parquet
   and compare fitted geometry with the Forge baseline in FreeMoCap's playback
   viewer. Verify independent layer/axis visibility, frame alignment and geometry
   after closing and reopening the recording, not just in the standalone Forge
   viewer. Check stage
   reuse/invalidation, disabled fitting, errors/cancellation, and no incomplete run
   being presented as a successful result. Do not rerun upstream image detection
   when a valid reusable checkpoint is available.

Each repository has its own human-owned commit/push handoff. No Git mutations by
agents. The initial integration is complete when the app can run the accepted fit,
report its progress, save it, reload it and display the same geometry correctly
in FreeMoCap's playback viewer alongside the existing segment layer. Working
standalone Forge visualization alone does not satisfy integration acceptance.

## After integration

Proceed to secondary exports from saved Parquet: CSV wide/tall and NumPy, then
animation formats. glTF and BVH require explicit handling of flexible spine
lengths and relaxed linkages; fixed-offset BVH may require an additional fitting
or projection step. Do not change the accepted solve silently to accommodate a
file format. Shoulder/scapula and foot-contact refinements can then evolve as
separate composable improvements with visible before/after comparisons.
