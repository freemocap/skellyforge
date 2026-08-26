"""Pose and hydration: the rest pose, closed-form hydration, roll, and model scale."""
from skellyforge.core.skeleton.pose.rest_pose import RestPose, build_rest_pose
from skellyforge.core.skeleton.pose.hydration import hydrate_segment, hydrate_skeleton
from skellyforge.core.skeleton.pose.roll_resolution import (
    ContinuousRollResolver,
    SegmentRollReference,
)
from skellyforge.core.skeleton.pose.model_scale_fitting import (
    ModelScaleFit,
    InsufficientScaleEvidence,
    SegmentScaleReading,
    StreamingModelScaleFitter,
    scale_voting_segment_names,
    fit_model_scale,
)

__all__ = [
    "ModelScaleFit",
    "ContinuousRollResolver",
    "InsufficientScaleEvidence",
    "RestPose",
    "SegmentRollReference",
    "SegmentScaleReading",
    "StreamingModelScaleFitter",
    "scale_voting_segment_names",
    "build_rest_pose",
    "fit_model_scale",
    "hydrate_segment",
    "hydrate_skeleton",
]
