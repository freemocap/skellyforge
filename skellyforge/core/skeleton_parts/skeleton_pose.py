"""The hydrated pose of a skeleton: each segment's world origin and orientation.

A static skeleton's per-frame face is a pose: where each segment's frame is (origin) and how
it is turned (orientation). The closed-form hydration fills this in from observed landmark
positions; velocities and accelerations come later, when the trajectory is known.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from skellyforge.core.math.geometry.rotation_quaternion import RotationQuaternion
from skellyforge.core.math.geometry.spatial_vectors import Point
from skellyforge.type_overloads import RigidBodySegmentName


@dataclass(frozen=True, slots=True, eq=False)
class SegmentPose:
    """One segment's pose at one instant.

    Attributes:
        segment_name: the segment this pose belongs to.
        origin: the world position of the segment's frame origin.
        orientation: the world orientation (local-to-world rotation). For a two-landmark
            segment the roll about its long axis is arbitrary and resolved downstream.
    """

    segment_name: RigidBodySegmentName
    origin: Point
    orientation: RotationQuaternion


@dataclass(frozen=True, slots=True, eq=False)
class SkeletonPose:
    """The whole skeleton's pose at one instant, keyed by segment name."""

    segment_poses: Mapping[RigidBodySegmentName, SegmentPose]
