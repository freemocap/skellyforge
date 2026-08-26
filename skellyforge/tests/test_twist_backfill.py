"""Twist backfill: the forearm recovers what its own landmarks cannot see.

The scenario that motivated this layer: rotate the hand (carpals) about the
forearm axis - pure pronation. The elbow/wrist pair is collinear with that
axis, so the forearm's own landmarks carry zero signal and transport freezes
its roll. The measured carpals orientation, however, pins it exactly. These
tests prove the backfill transfers that measurement, and refuses to invent
roll when the hand only flexes.
"""

from __future__ import annotations

import numpy as np
import pytest

from skellyforge.core.math.geometry.spatial_vectors import Point
from skellyforge.core.skeleton.chain.twist_backfill import backfill_twist_from_rigid_terminal
from skellyforge.core.skeleton.pose.hydration import hydrate_skeleton
from skellyforge.core.skeleton.pose.rest_pose import RestPose
from skellyforge.core.skeleton.pose.roll_resolution import ContinuousRollResolver
from skellyforge.core.skeleton.skeleton_definition import SkeletonDefinition


def _rodrigues(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    unit = axis / np.linalg.norm(axis)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.eye(3) * c + np.cross(np.eye(3), unit * s) + np.outer(unit, unit) * (1 - c)


def _pose_with_rotated_carpals(*, twist_degrees: float, axis_choice: str = "forearm"):
    """Full-skeleton pose with the left carpals' landmarks rotated about an axis
    through the wrist.

    ``axis_choice="forearm"`` rotates about the elbow->wrist axis (pure
    pronation); ``"perpendicular"`` rotates about a wrist-flexion axis (the
    control case).
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
    resolver = ContinuousRollResolver.for_skeleton(skeleton=skeleton)
    resolved = resolver.resolve_pose(pose=pose)
    return skeleton, resolved, pose


def _twist_of_update(before, after) -> float:
    """Signed angle of the update applied to `before`, about the segment's
    declared primary axis (+z for the lower arm)."""
    delta = before.inverse() * after
    _axis, unsigned = delta.to_axis_angle()
    return float(np.degrees(unsigned)) * (1.0 if delta.w >= 0.0 else -1.0)


def test_pure_pronation_is_recovered_from_the_measured_hand() -> None:
    twist_degrees = 60.0
    skeleton, baseline, _ = _pose_with_rotated_carpals(twist_degrees=0.0)
    _, resolved, _ = _pose_with_rotated_carpals(twist_degrees=twist_degrees)
    chain = skeleton.chains["left_arm"]
    before = resolved.segment_poses["left_lower_arm"].orientation

    updated_pose, report = backfill_twist_from_rigid_terminal(
        skeleton=skeleton,
        chain=chain,
        pose=resolved,
        baseline_pose=baseline,
    )

    assert report.updated_segment == "left_lower_arm"
    assert report.anchor_segment == "left_carpals"

    after = updated_pose.segment_poses["left_lower_arm"].orientation
    applied = _twist_of_update(before=before, after=after)
    # The hand was rotated by twist_degrees about the forearm axis; the
    # forearm must absorb exactly that much roll (sign set by the fixture's
    # rotation direction vs the declared +z primary axis).
    assert abs(applied) == pytest.approx(twist_degrees, abs=0.5), (
        f"applied {applied:.3f} deg, expected ~±{twist_degrees}"
    )

    # And the long axis did not move.
    reference_primary = np.array([0.0, 0.0, 1.0])
    np.testing.assert_allclose(
        after.rotate_vector(vector=reference_primary),
        before.rotate_vector(vector=reference_primary),
        atol=1e-9,
    )


def test_pure_wrist_flexion_does_not_roll_the_forearm() -> None:
    skeleton, baseline, _ = _pose_with_rotated_carpals(twist_degrees=0.0)
    _, resolved, _ = _pose_with_rotated_carpals(
        twist_degrees=40.0, axis_choice="perpendicular"
    )
    chain = skeleton.chains["left_arm"]
    before = resolved.segment_poses["left_lower_arm"].orientation

    _, report = backfill_twist_from_rigid_terminal(
        skeleton=skeleton,
        chain=chain,
        pose=resolved,
        baseline_pose=baseline,
    )

    assert abs(report.twist_applied_degrees) < 2.0, (
        "wrist flexion must not be misread as forearm roll"
    )


def test_backfill_leaves_everything_but_the_updated_segment_alone() -> None:
    skeleton, baseline, _ = _pose_with_rotated_carpals(twist_degrees=0.0)
    _, resolved, _ = _pose_with_rotated_carpals(twist_degrees=35.0)
    chain = skeleton.chains["left_arm"]

    updated_pose, _ = backfill_twist_from_rigid_terminal(
        skeleton=skeleton,
        chain=chain,
        pose=resolved,
        baseline_pose=baseline,
    )
    changed = {
        name
        for name in resolved.segment_poses
        if not updated_pose.segment_poses[name].orientation.is_same_rotation(
            other=resolved.segment_poses[name].orientation, tolerance_radians=1e-12
        )
    }
    assert changed == {"left_lower_arm"}


def test_chain_without_a_rigid_terminal_returns_unchanged() -> None:
    skeleton, baseline, _ = _pose_with_rotated_carpals(twist_degrees=25.0)
    _, resolved, _ = _pose_with_rotated_carpals(twist_degrees=25.0)
    # The leg chains terminate in direction-only segments (toes): no anchor.
    leg_chain = skeleton.chains["left_leg"]

    updated_pose, report = backfill_twist_from_rigid_terminal(
        skeleton=skeleton,
        chain=leg_chain,
        pose=resolved,
        baseline_pose=baseline,
    )
    assert report.updated_segment is None
    assert set(updated_pose.segment_poses) == set(resolved.segment_poses)


def test_report_names_its_anchor_and_magnitude() -> None:
    skeleton, baseline, _ = _pose_with_rotated_carpals(twist_degrees=0.0)
    _, resolved, _ = _pose_with_rotated_carpals(twist_degrees=-45.0)
    _, report = backfill_twist_from_rigid_terminal(
        skeleton=skeleton,
        chain=skeleton.chains["left_arm"],
        pose=resolved,
        baseline_pose=baseline,
    )
    assert report.chain_name == "left_arm"
    assert abs(report.twist_applied_degrees) == pytest.approx(45.0, abs=0.5)
