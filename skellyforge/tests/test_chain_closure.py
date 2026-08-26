"""The FK-closure invariant: angles -> landmarks -> hydration -> angles.

The loop that ties layers 3-6 into one verified system:

1. Take the authored rest pose's per-joint relative orientations.
2. Perturb every joint by a random rotation (the "synthesized motion").
3. Place the root somewhere random in the world (proving no world-frame leak).
4. Synthesize landmark positions forward through the joint tree.
5. Hydrate those landmarks back into segment poses (require_all=False, fresh
   roll resolver) - exactly what the realtime pipeline does with measured data.
6. Decompose every recovered joint.

What must hold, per the design doc's honesty about roll:

- Every joint whose parent AND child hydrate as RIGID_FIT recovers its input
  relative rotation EXACTLY - full closure including roll.
- Every other joint (direction-fit inputs carry transported, convention-carried
  roll) recovers an orientation whose action on the child's primary-axis
  direction matches the synthesized geometry - closure of everything the
  landmarks can observe. Roll is not in the landmarks; nothing here pretends
  otherwise.
"""

from __future__ import annotations

import numpy as np
import pytest

from skellyforge.core.math.geometry.rotation_quaternion import RotationQuaternion
from skellyforge.core.math.geometry.spatial_vectors import Point
from skellyforge.core.skeleton.chain.synthesis import synthesize_pose
from skellyforge.core.skeleton.pose.hydration import hydrate_skeleton
from skellyforge.core.skeleton.pose.rest_pose import RestPose
from skellyforge.core.skeleton.pose.roll_resolution import ContinuousRollResolver
from skellyforge.core.skeleton.skeleton_definition import SkeletonDefinition
from skellyforge.core.skeleton.skeleton_pose import PoseSolution

EXACT_CLOSURE_TOLERANCE_RADIANS = 1e-8
DIRECTION_CLOSURE_TOLERANCE_DEGREES = 1e-6


def _random_rotation(*, rng: np.random.Generator, max_degrees: float) -> RotationQuaternion:
    axis = rng.standard_normal(3)
    axis = axis / np.linalg.norm(axis)
    angle = np.deg2rad(rng.uniform(0.0, max_degrees))
    return RotationQuaternion.from_components(
        w=float(np.cos(angle / 2)),
        x=float(axis[0] * np.sin(angle / 2)),
        y=float(axis[1] * np.sin(angle / 2)),
        z=float(axis[2] * np.sin(angle / 2)),
    )


def _synthesized_observed_landmarks(seed: int):
    skeleton = SkeletonDefinition.from_default_yaml()
    rest_pose = RestPose.from_default_yaml(skeleton=skeleton)

    rng = np.random.default_rng(seed)

    # Authored relative orientations, perturbed per joint: the synthesized motion.
    joint_inputs = {}
    for joint in skeleton.joints.values():
        authored = (
            rest_pose.segment_orientations[joint.parent.name].inverse()
            * rest_pose.segment_orientations[joint.child.name]
        )
        perturbation = _random_rotation(rng=rng, max_degrees=25.0)
        joint_inputs[joint.name] = perturbation * authored

    root_world = _random_rotation(rng=rng, max_degrees=180.0)
    root_origin = Point.from_xyz(
        x=float(rng.uniform(-2000, 2000)),
        y=float(rng.uniform(-2000, 2000)),
        z=float(rng.uniform(500, 1500)),
    )

    world_orientations, world_origins, landmark_positions = synthesize_pose(
        skeleton=skeleton,
        joint_relative_orientations=joint_inputs,
        root_world_orientation=root_world,
        root_origin=root_origin,
    )
    return (
        skeleton,
        joint_inputs,
        {name: point for name, point in landmark_positions.items()},
        world_orientations,
        world_origins,
    )


@pytest.mark.parametrize("seed", [11, 29, 47])
def test_fk_closure(seed: int) -> None:
    skeleton, joint_inputs, landmarks, _, _ = _synthesized_observed_landmarks(seed)

    recovered_pose = hydrate_skeleton(skeleton=skeleton, observed=landmarks, require_all=False)
    resolver = ContinuousRollResolver.for_skeleton(skeleton=skeleton)
    resolved_pose = resolver.resolve_pose(pose=recovered_pose)

    assert len(resolved_pose.segment_poses) == len(skeleton.segments), (
        "exact synthetic landmarks should hydrate every segment"
    )

    from skellyforge.core.skeleton.linkage import compute_joint_poses

    recovered_joints = compute_joint_poses(skeleton=skeleton, pose=resolved_pose)
    assert len(recovered_joints) == len(skeleton.joints)

    # The exact-closure tier applies to exactly the joints whose parent AND
    # child are statically rigid-fit. The shipped model currently has no
    # ADJACENT rigid-rigid pair (the rigid segments are isolated), so this set
    # is computed from the statics and may be empty - asserted as such rather
    # than skipped, so adding a rigid segment forces conscious wiring here.
    expected_exact = {
        joint.name
        for joint in skeleton.joints.values()
        if joint.parent.supports_rigid_fit and joint.child.supports_rigid_fit
    }

    exact_joints: list[str] = []
    for joint_name, joint in skeleton.joints.items():
        recovered = recovered_joints[joint_name]
        input_relative = joint_inputs[joint_name]

        both_rigid = (
            resolved_pose.segment_poses[joint.parent.name].solved_by
            is PoseSolution.RIGID_FIT
            and resolved_pose.segment_poses[joint.child.name].solved_by
            is PoseSolution.RIGID_FIT
        )

        if both_rigid:
            assert joint_name in expected_exact
            error_rad = input_relative.angle_to(other=recovered.relative_orientation)
            assert error_rad < EXACT_CLOSURE_TOLERANCE_RADIANS, (
                f"joint {joint_name}: exact closure lost {error_rad:.3e} rad"
            )
            exact_joints.append(joint_name)
            continue

        # Direction closure: the recovered relative rotation must send the
        # child's local origin->primary displacement onto the same world
        # direction the synthesized geometry has. Roll (unobservable) excluded.
        fd = joint.child.frame_definition
        primary_local = skeleton.landmarks[
            fd.primary_point_name
        ].local_position.array.astype(float)
        step = landmarks[fd.primary_point_name].array - landmarks[fd.origin_point_name].array
        step = step / np.linalg.norm(step)

        from_input = recovered.relative_orientation.rotate_vector(vector=primary_local)
        # NOTE: recovered relative lives in the RECOVERED parent frame; express
        # the synthesized world direction in that same frame before comparing.
        parent_world_recovered = resolved_pose.segment_poses[joint.parent.name].orientation
        target_in_parent = parent_world_recovered.inverse().rotate_vector(
            vector=step.astype(float)
        )
        cosine = float(from_input @ target_in_parent)
        error_deg = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
        assert error_deg < DIRECTION_CLOSURE_TOLERANCE_DEGREES, (
            f"joint {joint_name}: direction closure off by {error_deg:.3e} deg"
        )

    # Whatever the statics promise, the run must deliver - and no more.
    assert set(exact_joints) == expected_exact


def test_synthesis_is_deterministic() -> None:
    first = _synthesized_observed_landmarks(5)[2]
    second = _synthesized_observed_landmarks(5)[2]
    for name, point in first.items():
        np.testing.assert_array_equal(point.array, second[name].array)


def test_root_placement_does_not_leak_into_relative_angles() -> None:
    """Same joint inputs at two different world placements recover identical
    relative orientations up to each side's roll convention."""
    skeleton, joint_inputs, landmarks_a, _, _ = _synthesized_observed_landmarks(13)
    shift = np.array([500.0, -300.0, 900.0])
    shifted_landmarks = {
        name: Point.from_prevalidated_array(array=point.array + shift)
        for name, point in landmarks_a.items()
    }

    def relative_angles(observed):
        pose = hydrate_skeleton(skeleton=skeleton, observed=observed, require_all=False)
        resolver = ContinuousRollResolver.for_skeleton(skeleton=skeleton)
        resolved = resolver.resolve_pose(pose=pose)
        relatives = {}
        for joint in skeleton.joints.values():
            parent_pose = resolved.segment_poses.get(joint.parent.name)
            child_pose = resolved.segment_poses.get(joint.child.name)
            if parent_pose is None or child_pose is None:
                continue
            if (
                parent_pose.solved_by is not PoseSolution.RIGID_FIT
                or child_pose.solved_by is not PoseSolution.RIGID_FIT
            ):
                continue
            relatives[joint.name] = (
                parent_pose.orientation.inverse() * child_pose.orientation
            )
        return relatives

    angles_at_home = relative_angles(landmarks_a)
    angles_shifted = relative_angles(shifted_landmarks)
    if not angles_at_home:
        pytest.skip("no adjacent rigid-fit pairs exist in the shipped model yet")
    assert sorted(angles_at_home) == sorted(angles_shifted)
    for name in angles_at_home:
        error = angles_at_home[name].angle_to(other=angles_shifted[name])
        assert error < 1e-8
