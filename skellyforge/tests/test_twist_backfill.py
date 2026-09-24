"""Twist backfill: the forearm recovers what its own landmarks cannot see.

The scenario that motivated this layer: rotate the hand (carpals) about the
forearm axis - pure pronation. The elbow/wrist pair is collinear with that
axis, so the forearm's own landmarks carry zero signal and transport freezes
its roll. The measured carpals orientation, however, pins it exactly - and
with rest relative orientations supplied to the resolver, resolve_pose's
terminal-backfill pass transfers that measurement automatically.

The reference is the authored rest relationship. These tests check whether
independent hand evidence supplies twist without confusing wrist flexion with it.
"""

from __future__ import annotations

import numpy as np
import pytest

from skellyforge.core.math.geometry.spatial_vectors import Point
from skellyforge.core.skeleton.chain.twist_backfill import twist_about_local_axis
from skellyforge.core.skeleton.chain.twist_backfill import apply_terminal_twist_backfills
from skellyforge.core.skeleton.skeleton_pose import SkeletonPose, PoseSolution
from skellyforge.core.math.geometry.rotation_quaternion import RotationQuaternion
from skellyforge.core.skeleton.pose.hydration import hydrate_skeleton
from skellyforge.core.skeleton.pose.rest_pose import RestPose
from skellyforge.core.skeleton.pose.roll_resolution import ContinuousRollResolver
from skellyforge.core.skeleton.skeleton_definition import SkeletonDefinition


def _rodrigues(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    unit = axis / np.linalg.norm(axis)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.eye(3) * c + np.cross(np.eye(3), unit * s) + np.outer(unit, unit) * (1 - c)


def test_twist_evidence_is_expressed_in_parent_frame_with_nonidentity_rest():
    skeleton = SkeletonDefinition.from_default_yaml()
    rest = RestPose.from_default_yaml(skeleton=skeleton)
    pose = hydrate_skeleton(skeleton=skeleton, observed=rest.landmark_positions)
    parent, child = "left_lower_arm", "left_carpals"
    parent_rotation = rest.segment_orientations[parent]
    rest_relative = RotationQuaternion.from_rotation_vector(
        rotation_vector=np.array([0.7, -0.4, 0.2]))
    # A known rotation about the parent's local long axis, followed by an
    # unrelated child rest alignment. Neither may be mistaken for the other.
    twist = RotationQuaternion.from_rotation_matrix(
        matrix=_rodrigues(np.array([0., 0., 1.]), np.deg2rad(50)))
    observed = SkeletonPose(segment_poses={
        parent: pose.segment_poses[parent].with_orientation(
            orientation=parent_rotation, solved_by=PoseSolution.TRANSPORTED_ROLL),
        child: pose.segment_poses[child].with_orientation(
            orientation=parent_rotation * twist * rest_relative, solved_by=PoseSolution.RIGID_FIT),
    })
    result = apply_terminal_twist_backfills(
        skeleton=skeleton, pose=observed,
        rest_relative_orientations=dict(rest.relative_orientations) | {child: rest_relative})
    assert result.segment_poses[parent].orientation.is_same_rotation(
        other=parent_rotation * twist, tolerance_radians=1e-9)
    assert result.segment_poses[child] is observed.segment_poses[child]


def _resolved_pose_with_rotated_carpals(
    *, twist_degrees: float, axis_choice: str = "forearm"
):
    """Resolved full-skeleton pose with the left carpals' landmarks rotated
    about an axis through the wrist.

    ``axis_choice="forearm"`` rotates about the elbow->wrist axis (pure
    pronation); ``"perpendicular"`` rotates about a wrist-flexion axis (the
    control case). Resolution runs WITH rest relative orientations supplied,
    i.e. the production configuration including the backfill pass.
    """
    skeleton = SkeletonDefinition.from_default_yaml()
    rest_pose = RestPose.from_default_yaml(skeleton=skeleton)
    landmark_positions = {
        name: point.array.copy() for name, point in rest_pose.landmark_positions.items()
    }

    elbow = landmark_positions["left_elbow"]
    wrist = landmark_positions["left_wrist"]
    forearm_axis = wrist - elbow
    forearm_axis = forearm_axis / np.linalg.norm(forearm_axis)

    if axis_choice == "forearm":
        rotation_axis = forearm_axis
    else:
        perpendicular = np.array([0.0, 0.0, 1.0]) - (
            float(np.array([0.0, 0.0, 1.0]) @ forearm_axis)
        ) * forearm_axis
        rotation_axis = perpendicular / np.linalg.norm(perpendicular)

    rotation = _rodrigues(rotation_axis, np.deg2rad(twist_degrees))
    owned_by_carpals = [
        landmark.name
        for landmark in skeleton.landmarks.values()
        if skeleton.owning_segment_name_of(landmark=landmark) == "left_carpals"
    ]
    for name in owned_by_carpals:
        relative = landmark_positions[name] - wrist
        landmark_positions[name] = wrist + rotation @ relative

    observed = {
        name: Point.from_prevalidated_array(array=position)
        for name, position in landmark_positions.items()
    }
    pose = hydrate_skeleton(skeleton=skeleton, observed=observed, require_all=False)
    resolver = ContinuousRollResolver.for_skeleton(
        skeleton=skeleton,
        rest_relative_orientations=rest_pose.relative_orientations,
    )
    resolved = resolver.resolve_pose(pose=pose)
    return skeleton, resolved


def _signed_twist_between(before: RotationQuaternion, after: RotationQuaternion) -> float:
    delta = before.inverse() * after
    _axis, unsigned = delta.to_axis_angle()
    return float(np.degrees(unsigned)) * (1.0 if delta.w >= 0.0 else -1.0)


def test_pure_pronation_is_recovered_during_resolution() -> None:
    twist_degrees = 60.0
    _, reference_pose = _resolved_pose_with_rotated_carpals(twist_degrees=0.0)
    _, pronated_pose = _resolved_pose_with_rotated_carpals(twist_degrees=twist_degrees)

    applied = _signed_twist_between(
        reference_pose.segment_poses["left_lower_arm"].orientation,
        pronated_pose.segment_poses["left_lower_arm"].orientation,
    )
    assert abs(applied) == pytest.approx(twist_degrees, abs=0.5), (
        f"forearm roll recovered {applied:.3f} deg of the {twist_degrees} deg "
        "pronation the measured hand pinned"
    )

    # And the long axis did not move.
    reference_primary = np.array([0.0, 0.0, 1.0])
    np.testing.assert_allclose(
        pronated_pose.segment_poses["left_lower_arm"].orientation.rotate_vector(
            vector=reference_primary
        ),
        reference_pose.segment_poses["left_lower_arm"].orientation.rotate_vector(
            vector=reference_primary
        ),
        atol=1e-9,
    )


def test_pure_wrist_flexion_does_not_roll_the_forearm() -> None:
    _, reference_pose = _resolved_pose_with_rotated_carpals(twist_degrees=0.0)
    _, flexed_pose = _resolved_pose_with_rotated_carpals(
        twist_degrees=40.0, axis_choice="perpendicular"
    )
    applied = _signed_twist_between(
        reference_pose.segment_poses["left_lower_arm"].orientation,
        flexed_pose.segment_poses["left_lower_arm"].orientation,
    )
    assert abs(applied) < 2.0, (
        "wrist flexion must not be misread as forearm roll"
    )


def test_backfill_is_a_pure_function_of_the_frame() -> None:
    """Resolving the identical frame twice gives bit-identical orientations."""
    _, first = _resolved_pose_with_rotated_carpals(twist_degrees=33.0)
    _, second = _resolved_pose_with_rotated_carpals(twist_degrees=33.0)
    for name, segment_pose in first.segment_poses.items():
        assert segment_pose.orientation.is_same_rotation(
            other=second.segment_poses[name].orientation, tolerance_radians=1e-12
        )


def test_swing_twist_projection_splits_pure_cases_exactly() -> None:
    axis = np.array([0.0, 0.0, 1.0])
    pure_twist = RotationQuaternion.from_components(w=0.866, x=0.0, y=0.0, z=0.5)
    twist_of_pure_twist = twist_about_local_axis(
        relative_rotation=pure_twist, local_axis=axis
    )
    assert twist_of_pure_twist.is_same_rotation(other=pure_twist)

    pure_swing_about_x = RotationQuaternion.from_components(
        w=0.866, x=0.5, y=0.0, z=0.0
    )
    twist_of_pure_swing = twist_about_local_axis(
        relative_rotation=pure_swing_about_x, local_axis=axis
    )
    assert twist_of_pure_swing.is_same_rotation(
        other=RotationQuaternion.identity(), tolerance_radians=1e-12
    )
