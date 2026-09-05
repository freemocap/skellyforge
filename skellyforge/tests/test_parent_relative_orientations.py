"""Local rotations require both segment poses, except at the declared root."""

import math

import pytest

from skellyforge.core.math.geometry.rotation_quaternion import RotationQuaternion
from skellyforge.core.math.geometry.spatial_vectors import Point
from skellyforge.core.skeleton.skeleton_pose import (
    SkeletonPose,
    SegmentPose,
    PoseSolution,
)


def segment(name: str, rotation: RotationQuaternion) -> SegmentPose:
    return SegmentPose(
        segment_name=name,
        origin=Point.from_xyz(x=0.0, y=0.0, z=0.0),
        orientation=rotation,
        scale_estimate=1.0,
        solved_by=PoseSolution.RIGID_FIT,
    )


def test_child_rotation_requires_parent_pose() -> None:
    half = math.sqrt(0.5)
    parent = RotationQuaternion(w=half, x=0.0, y=0.0, z=half)
    local = RotationQuaternion(w=half, x=half, y=0.0, z=0.0)
    child = parent * local
    pose = SkeletonPose(
        segment_poses={
            "root": segment("root", parent),
            "child": segment("child", child),
        }
    )
    values = pose.parent_relative_orientations(parents={"root": None, "child": "root"})
    assert values["root"].is_same_rotation(other=parent)
    assert values["child"].is_same_rotation(other=local)
    partial = SkeletonPose(segment_poses={"child": segment("child", child)})
    assert (
        partial.parent_relative_orientations(parents={"root": None, "child": "root"})
        == {}
    )


def test_unknown_parent_map_entry_fails() -> None:
    pose = SkeletonPose(segment_poses={})
    with pytest.raises(ValueError, match="unknown segment"):
        pose.parent_relative_orientations(parents={"child": "unknown"})
