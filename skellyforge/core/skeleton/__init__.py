"""The typed standard-human model: landmarks, segments, the skeleton, and its poses.

Organized semantically:

- the leaf model types live at this level: AnatomicalLandmark, RigidBodySegment,
  SkeletonDefinition, SkeletonPose, FaceBlendShapes, and the naming/resolver helpers;
- loading/ is the YAML loader pipeline (include -> lowercase -> side -> reference frame ->
  objects);
- pose/ is pose and hydration (the rest pose, closed-form hydration, roll resolution, and
  the body-scale fit that gives the dimensionless template a size).
"""
