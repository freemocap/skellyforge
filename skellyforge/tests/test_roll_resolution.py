"""Tests for the parallel-transport roll convention.

Two landmarks do not determine roll, so what these check is not "is the roll correct" -
there is no correct answer - but the three properties that make a CONVENTION usable:
it must not disturb the direction that was measured, it must be continuous frame to frame
including through the pole where the naive shortest-arc roll jumps, and it must leave
measured orientations alone.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from skellyforge.core.math.geometry.rotation_quaternion import RotationQuaternion
from skellyforge.core.math.geometry.spatial_vectors import Point
from skellyforge.core.skeleton_parts.pose.rest_pose import RestPose
from skellyforge.core.skeleton_parts.pose.roll_resolution import (
    ContinuousRollResolver,
    SegmentRollReference,
)
from skellyforge.core.skeleton_parts.skeleton_definition import SkeletonDefinition
from skellyforge.core.skeleton_parts.pose.hydration import hydrate_skeleton
from skellyforge.core.skeleton_parts.skeleton_pose import PoseSolution

SKELETON_YAML_PATH: Path = (
    Path(__file__).resolve().parents[1]
    / "definitions"
    / "human_skeleton"
    / "human_skeleton.yaml"
)
REST_POSE_YAML_PATH: Path = (
    Path(__file__).resolve().parents[1]
    / "definitions"
    / "human_skeleton"
    / "rest_pose.yaml"
)

SWEPT_SEGMENT: str = "left_upper_arm"
FRAME_COUNT: int = 120


def _skeleton() -> SkeletonDefinition:
    return SkeletonDefinition.from_yaml(path=SKELETON_YAML_PATH)


def _swept_poses(*, skeleton: SkeletonDefinition):
    """Hydrated poses for an arm swept a full turn through straight overhead and straight down.

    Straight down is the pole of the shortest arc from the arm's authored distal direction, so
    a stateless per-frame roll flips there. Sweeping through it is the point.
    """
    rest_pose = RestPose.from_yaml(path=REST_POSE_YAML_PATH, skeleton=skeleton)
    shoulder = rest_pose.landmark_positions["left_shoulder"]
    length = skeleton.segments[SWEPT_SEGMENT].length

    for frame in range(FRAME_COUNT):
        angle = 2.0 * np.pi * frame / FRAME_COUNT
        direction = np.array([np.cos(angle), 0.0, np.sin(angle)])
        observed = dict(rest_pose.landmark_positions)
        observed["left_elbow"] = Point.from_prevalidated_array(
            array=shoulder.array + length * direction
        )
        yield hydrate_skeleton(skeleton=skeleton, observed=observed)


def test_the_resolver_leaves_rigid_fit_poses_untouched() -> None:
    skeleton = _skeleton()
    rest_pose = RestPose.from_yaml(path=REST_POSE_YAML_PATH, skeleton=skeleton)
    pose = hydrate_skeleton(skeleton=skeleton, observed=rest_pose.landmark_positions)
    resolved = ContinuousRollResolver.for_skeleton(skeleton=skeleton).resolve_pose(
        pose=pose
    )
    for name, before in pose.segment_poses.items():
        if before.solved_by is not PoseSolution.RIGID_FIT:
            continue
        after = resolved.segment_poses[name]
        assert after.solved_by is PoseSolution.RIGID_FIT
        assert after.orientation.is_same_rotation(other=before.orientation)


def test_resolving_roll_does_not_move_the_measured_direction() -> None:
    """The roll may change; the long axis the data pinned may not."""
    skeleton = _skeleton()
    resolver = ContinuousRollResolver.for_skeleton(skeleton=skeleton)
    reference = SegmentRollReference.for_segment(
        skeleton=skeleton, segment_name=SWEPT_SEGMENT
    )
    for pose in _swept_poses(skeleton=skeleton):
        before = pose.segment_poses[SWEPT_SEGMENT]
        after = resolver.resolve_segment_pose(pose=before)
        assert after.solved_by is PoseSolution.TRANSPORTED_ROLL
        np.testing.assert_allclose(
            after.orientation.rotate_vector(vector=reference.primary_local),
            before.orientation.rotate_vector(vector=reference.primary_local),
            atol=1e-9,
        )


def test_transported_roll_is_continuous_where_the_raw_roll_jumps() -> None:
    """Sweeping an arm through the pole: the raw roll flips, the transported one does not."""
    skeleton = _skeleton()
    resolver = ContinuousRollResolver.for_skeleton(skeleton=skeleton)

    raw_steps: list[float] = []
    resolved_steps: list[float] = []
    previous_raw: RotationQuaternion | None = None
    previous_resolved: RotationQuaternion | None = None

    for pose in _swept_poses(skeleton=skeleton):
        raw = pose.segment_poses[SWEPT_SEGMENT].orientation
        resolved = resolver.resolve_segment_pose(
            pose=pose.segment_poses[SWEPT_SEGMENT]
        ).orientation
        if previous_raw is not None and previous_resolved is not None:
            raw_steps.append(previous_raw.angle_to(other=raw))
            resolved_steps.append(previous_resolved.angle_to(other=resolved))
        previous_raw, previous_resolved = raw, resolved

    # One frame of a 120-frame sweep is 3 degrees of direction change, so a roll that is
    # behaving cannot be stepping much more than that.
    per_frame_limit = np.deg2rad(10.0)
    assert max(resolved_steps) < per_frame_limit, (
        f"transported roll jumped {np.degrees(max(resolved_steps)):.1f} degrees in one frame"
    )
    assert max(raw_steps) > per_frame_limit, (
        "this sweep is supposed to make the raw shortest-arc roll jump; if it no longer "
        "does, the test is not testing anything"
    )


def test_a_reset_resolver_starts_a_fresh_take() -> None:
    """After `reset()`, the same frame resolves exactly as it would on a new resolver.

    Half a sweep is used rather than a whole one: transported around the full circle the
    roll returns to where it started, so a closed loop would compare equal for reasons
    that have nothing to do with the reset.
    """
    skeleton = _skeleton()
    poses = list(_swept_poses(skeleton=skeleton))
    first_pose = poses[0].segment_poses[SWEPT_SEGMENT]

    virgin = ContinuousRollResolver.for_skeleton(skeleton=skeleton).resolve_segment_pose(
        pose=first_pose
    ).orientation

    resolver = ContinuousRollResolver.for_skeleton(skeleton=skeleton)
    for pose in poses[: FRAME_COUNT // 2]:
        resolver.resolve_segment_pose(pose=pose.segment_poses[SWEPT_SEGMENT])
    carried = resolver.resolve_segment_pose(pose=first_pose).orientation
    assert not carried.is_same_rotation(other=virgin, tolerance_radians=1e-3), (
        "half a sweep should have transported the roll somewhere else"
    )

    resolver.reset()
    after_reset = resolver.resolve_segment_pose(pose=first_pose).orientation
    assert after_reset.is_same_rotation(other=virgin, tolerance_radians=1e-12)


def test_resolve_pose_covers_every_direction_only_segment() -> None:
    skeleton = _skeleton()
    rest_pose = RestPose.from_yaml(path=REST_POSE_YAML_PATH, skeleton=skeleton)
    pose = hydrate_skeleton(skeleton=skeleton, observed=rest_pose.landmark_positions)
    resolved = ContinuousRollResolver.for_skeleton(skeleton=skeleton).resolve_pose(
        pose=pose
    )
    assert resolved.segment_names_with_free_roll == ()
    assert set(resolved.segment_poses) == set(pose.segment_poses)


def test_a_segment_whose_primary_sits_on_its_origin_has_no_roll_reference() -> None:
    skeleton = _skeleton()
    with pytest.raises(KeyError):
        SegmentRollReference.for_segment(skeleton=skeleton, segment_name="not_a_segment")
