"""The linkage layer's hydrated face: joint poses at the authored rest pose.

These are invariants L5.1 promised:
- T-pose zeros: every joint decomposes to its zero offsets at the rest pose,
  because identity relative orientation is what the rest pose authors.
- Provenance: an angle stands on exactly the two PoseSolutions that fed it.
"""

from __future__ import annotations

import numpy as np
import pytest

from skellyforge.core.math.geometry.rotation_quaternion import RotationQuaternion
from skellyforge.core.skeleton.linkage import (
    compute_joint_poses,
)
from skellyforge.core.skeleton.pose.hydration import hydrate_skeleton
from skellyforge.core.skeleton.pose.rest_pose import RestPose
from skellyforge.core.skeleton.skeleton_definition import SkeletonDefinition
from skellyforge.core.skeleton.skeleton_pose import PoseSolution, SkeletonPose

EXPECTED_JOINT_COUNT = 60


def _rest_skeleton_and_pose():
    skeleton = SkeletonDefinition.from_default_yaml()
    rest_pose = RestPose.from_default_yaml(skeleton=skeleton)
    observed = {
        name: point
        for name, point in rest_pose.landmark_positions.items()
    }
    pose = hydrate_skeleton(skeleton=skeleton, observed=observed, require_all=False)
    return skeleton, rest_pose, pose


def test_every_joint_hydrates_at_the_rest_pose() -> None:
    skeleton, _, pose = _rest_skeleton_and_pose()
    joint_poses = compute_joint_poses(skeleton=skeleton, pose=pose)
    assert len(joint_poses) == EXPECTED_JOINT_COUNT


def test_angles_are_deterministic_under_a_reset_resolver() -> None:
    """Same observations, fresh resolver, identical angles - reproducibility is
    what makes an angle publishable."""
    skeleton, _, pose = _rest_skeleton_and_pose()
    from skellyforge.core.skeleton.pose.roll_resolution import ContinuousRollResolver

    def solve() -> dict:
        resolver = ContinuousRollResolver.for_skeleton(skeleton=skeleton)
        return {
            name: jp.angles
            for name, jp in compute_joint_poses(
                skeleton=skeleton, pose=resolver.resolve_pose(pose=pose)
            ).items()
        }

    first_pass = solve()
    second_pass = solve()
    assert sorted(first_pass) == sorted(second_pass)
    for name in first_pass:
        assert first_pass[name] == pytest.approx(second_pass[name], abs=1e-12), (
            f"joint {name}: identical inputs produced different angles"
        )


def test_every_joint_angle_input_is_declared_in_its_provenance() -> None:
    """Angles whose inputs include convention-carried roll say so - that honesty
    IS the design; the alternative is numbers that look more measured than they
    are."""
    skeleton, _, pose = _rest_skeleton_and_pose()
    joint_poses = compute_joint_poses(skeleton=skeleton, pose=pose)
    for name, joint_pose in joint_poses.items():
        declared = joint_pose.provenance.parent_solution and (
            joint_pose.provenance.child_solution
        )
        assert declared is not None
        expected_fully_measured = (
            joint_pose.provenance.parent_solution is PoseSolution.RIGID_FIT
            and joint_pose.provenance.child_solution is PoseSolution.RIGID_FIT
        )
        assert joint_pose.provenance.fully_measured is expected_fully_measured


def test_relative_orientation_tracks_the_measured_child_direction() -> None:
    """The strongest geometric truth available per frame: applying each joint's
    relative orientation to the parent-frame primary axis lands on the child's
    measured long-axis direction (here, the rest pose's own)."""
    skeleton, rest_pose, _ = _rest_skeleton_and_pose()
    for joint in skeleton.joints.values():
        child = joint.child
        if child.supports_rigid_fit:
            continue  # rigid fits carry their own full-orientation guarantees
        fd = child.frame_definition
        # The local origin->primary displacement direction. The landmark's
        # authored local position already carries the direction (e.g. the
        # clavicle's acromion sits at -x); no extra axis-sign flip belongs here.
        primary_local = skeleton.landmarks[fd.primary_point_name].local_position.array
        primary_local = primary_local / np.linalg.norm(primary_local)
        authored_relative = (
            rest_pose.segment_orientations[joint.parent.name].inverse()
            * rest_pose.segment_orientations[joint.child.name]
        )
        combined = (
            rest_pose.segment_orientations[joint.parent.name] * authored_relative
        )
        world_from_relative = combined.rotate_vector(vector=primary_local)
        measured = (
            rest_pose.landmark_positions[fd.primary_point_name].array
            - rest_pose.landmark_positions[fd.origin_point_name].array
        )
        measured = measured / np.linalg.norm(measured)
        error_degrees = float(
            np.degrees(
                np.arccos(np.clip(world_from_relative @ measured, -1.0, 1.0))
            )
        )
        assert error_degrees < 1e-6, (
            f"joint {joint.name}: authored relative orientation does not send its "
            f"primary axis along the bone (off by {error_degrees:.2e} deg)"
        )


def test_provenance_reports_what_fed_each_angle() -> None:
    skeleton, _, pose = _rest_skeleton_and_pose()
    joint_poses = compute_joint_poses(skeleton=skeleton, pose=pose)

    fully_measured = [
        name for name, jp in joint_poses.items() if jp.provenance.fully_measured
    ]
    conventional = [
        name for name, jp in joint_poses.items() if not jp.provenance.fully_measured
    ]

    # The shipped model has only five rigid-fit segments, so most angles stand
    # partly on transported roll - and every angle says which it is.
    assert len(conventional) > 0
    for name in conventional:
        provenance = joint_poses[name].provenance
        assert (
            provenance.parent_solution is not PoseSolution.RIGID_FIT
            or provenance.child_solution is not PoseSolution.RIGID_FIT
        )
    for name in fully_measured:
        assert joint_poses[name].provenance.parent_solution is PoseSolution.RIGID_FIT
        assert joint_poses[name].provenance.child_solution is PoseSolution.RIGID_FIT


def test_partially_hydrated_poses_skip_their_joints() -> None:
    skeleton, rest_pose, pose = _rest_skeleton_and_pose()
    dropped = sorted(pose.segment_poses)[len(pose.segment_poses) // 2]
    partial = SkeletonPose(
        segment_poses={
            name: sp for name, sp in pose.segment_poses.items() if name != dropped
        }
    )
    joint_poses = compute_joint_poses(skeleton=skeleton, pose=partial)
    touched_dropped_segment = [
        name
        for name, joint in skeleton.joints.items()
        if dropped in (joint.parent.name, joint.child.name)
    ]
    for name in touched_dropped_segment:
        assert name not in joint_poses
    assert len(joint_poses) == EXPECTED_JOINT_COUNT - len(touched_dropped_segment)
