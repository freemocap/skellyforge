"""Pose and hydration: the rest pose, closed-form hydration, roll, and segment lengths."""
from skellyforge.core.skeleton.pose.rest_pose import RestPose, build_rest_pose
from skellyforge.core.skeleton.pose.hydration import hydrate_segment, hydrate_skeleton
from skellyforge.core.skeleton.pose.roll_resolution import (
    ContinuousRollResolver,
    SegmentRollReference,
)
from skellyforge.core.skeleton.pose.segment_length_estimation import (
    estimate_segment_lengths,
)

__all__ = [
    "ContinuousRollResolver",
    "RestPose",
    "SegmentRollReference",
    "build_rest_pose",
    "estimate_segment_lengths",
    "hydrate_segment",
    "hydrate_skeleton",
]
