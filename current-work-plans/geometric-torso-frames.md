# Geometric pelvis and thorax frames

The standard human now declares an `observation_frame` for pelvis and thoracic
segments. It uses the existing signed-axis/Gram–Schmidt solver. It is a geometric
construction, not an anatomical fit or an inverse-kinematics optimizer.

- Pelvis: origin at hip midpoint; sideways axis toward the right hip; approximate
  up toward shoulder midpoint, projected perpendicular to the sideways axis.
- Thorax orientation: sideways axis toward the right shoulder from shoulder
  midpoint; approximate up away from hip midpoint, projected perpendicular to
  the sideways axis. Segment position remains chest center.
- Both use the existing segment origin-to-primary distance for scale estimation.
  Neither changes template lengths, joint connections or recording-wide fitting.

`observation_frame` references skeleton landmarks, including landmarks owned by
other segments. Its axes directly express the segment's local-to-world rotation.
The loader validates references and expands sided definitions. Hydration requires
all defining points and a nondegenerate frame; missing, nonfinite or collinear
inputs do not fall back to fitting constructed offset landmarks.

The result is labeled `PoseSolution.OBSERVATION_FRAME`. Roll transport preserves
it, twist backfill cannot overwrite it, and body alignment accepts its full
orientation. Joint provenance still distinguishes this construction from a rigid
marker fit. Other segments retain their existing hydration methods.

Tests verify coordinate transformation and scale behavior, independence from
constructed offsets, explicit axis priority under shoulder elevation, missing
and degenerate inputs, bounded jitter, roll preservation and alignment eligibility.
Older synthetic closure tests now check the explicit frame contract for these
segments: arbitrary authored joint poses need not satisfy the cross-body frame
definition, so their exact inversion is not the promised behavior.

## Integration boundary

Commit/push SkellyForge first. Update its dependency in FreeMoCap, reconstruct the
same saved 3D recording and regenerate the diagnostic viewer. This change does
not yet replace tracker-derived attachment positions throughout core. The next
linked-pose step will place attachments in these frames and solve spine/clavicle
motion together. Flexible lengths, joint limits and constrained IK are not part
of this commit. Existing Parquet results are not rewritten automatically.
