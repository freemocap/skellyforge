"""Pose and hydration: the rest pose, closed-form hydration, roll, and body scale."""
from skellyforge.core.skeleton.pose.rest_pose import RestPose, build_rest_pose
from skellyforge.core.skeleton.pose.hydration import hydrate_segment, hydrate_skeleton
from skellyforge.core.skeleton.pose.roll_resolution import (
    ContinuousRollResolver,
    SegmentRollReference,
)
from skellyforge.core.skeleton.pose.body_scale_fitting import (
    BodyScaleFit,
    InsufficientScaleEvidence,
    SegmentScaleReading,
    StreamingBodyScaleFitter,
    body_scale_voting_segment_names,
    fit_body_scale,
)

__all__ = [
    "BodyScaleFit",
    "ContinuousRollResolver",
    "InsufficientScaleEvidence",
    "RestPose",
    "SegmentRollReference",
    "SegmentScaleReading",
    "StreamingBodyScaleFitter",
    "body_scale_voting_segment_names",
    "build_rest_pose",
    "fit_body_scale",
    "hydrate_segment",
    "hydrate_skeleton",
]
