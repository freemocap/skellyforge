"""Forward kinematic synthesis: joint angles -> whole-body poses.

The chain layer's forward math. Given a per-joint relative rotation for every
joint of the skeleton (plus where the root sits), walk the joint tree once and
produce every segment's world orientation and origin, and every landmark's
world position. This is the rest pose's forward pass generalized from one
authored pose to any authored motion - and it is the `F` in the closure test:
angles -> landmarks -> hydration -> recovered angles must round-trip.

The companion inverse direction (landmarks -> poses) is NOT here - that is
hydration (`core/skeleton/pose/hydration.py`), which measures; this module
only synthesizes. Synthesized output is exact by construction: closed-form
composition, no iteration, no repair.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from skellyforge.core.math.geometry.rotation_quaternion import RotationQuaternion
from skellyforge.core.math.geometry.spatial_vectors import Point
from skellyforge.core.skeleton.pose.rest_pose import build_rest_pose
from skellyforge.core.skeleton.skeleton_definition import SkeletonDefinition
from skellyforge.type_overloads import FloatArray, LandmarkNameString, RigidBodySegmentName


def synthesize_pose(
    *,
    skeleton: SkeletonDefinition,
    joint_relative_orientations: Mapping[str, RotationQuaternion],
    root_world_orientation: RotationQuaternion | None = None,
    root_origin: Point | None = None,
) -> tuple[
    dict[RigidBodySegmentName, RotationQuaternion],
    dict[RigidBodySegmentName, Point],
    dict[LandmarkNameString, Point],
]:
    """Walk the joint tree once, producing every segment and landmark in world.

    Args:
        skeleton: the skeleton whose joints define the tree.
        joint_relative_orientations: one rotation per joint NAME, expressed in
            the parent segment's frame - exactly what a joint's hydrated face
            reports, so synthesis and measurement speak the same language. The
            root segment's entry (keyed by its own name) sets its world
            orientation; defaults to identity.
        root_world_orientation: alternative way to set the root's world
            orientation, overriding the map's root entry when given.
        root_origin: where the root segment sits in world. Defaults to the
            world origin.

    Returns:
        ``(world_orientations, world_origins, landmark_positions)`` keyed by
        segment / landmark name - the same shape ``build_rest_pose`` returns,
        of which this is the time-varying generalization.

    Raises:
        KeyError: a joint has no entry in `joint_relative_orientations` -
            synthesis refuses to invent rotations.
    """
    parents: dict[RigidBodySegmentName, RigidBodySegmentName | None] = {
        segment.name: None for segment in skeleton.segments.values()
    }
    connect_ats: dict[RigidBodySegmentName, str] = {}
    orientations_by_segment: dict[RigidBodySegmentName, RotationQuaternion] = {}

    missing = [
        joint.name
        for joint in skeleton.joints.values()
        if joint.name not in joint_relative_orientations
    ]
    if missing:
        raise KeyError(
            f"synthesis needs a relative orientation for every joint - missing "
            f"{sorted(missing)}"
        )

    for joint in skeleton.joints.values():
        parents[joint.child.name] = joint.parent.name
        connect_ats[joint.child.name] = joint.connect_at.name
        orientations_by_segment[joint.child.name] = joint_relative_orientations[
            joint.name
        ]

    root_name = next(name for name, parent in parents.items() if parent is None)
    orientations_by_segment[root_name] = (
        root_world_orientation
        if root_world_orientation is not None
        else joint_relative_orientations.get(root_name)
        or RotationQuaternion.identity()
    )
    connect_ats[root_name] = (
        skeleton.segments[root_name].frame_definition.origin_point_name
    )

    world_orientations, world_origins, landmark_positions = build_rest_pose(
        skeleton=skeleton,
        parents=parents,
        connect_ats=connect_ats,
        orientations=orientations_by_segment,
    )

    if root_origin is not None:
        translation = root_origin.array
        world_origins = {
            name: Point.from_prevalidated_array(array=origin.array + translation)
            for name, origin in world_origins.items()
        }
        landmark_positions = {
            name: Point.from_prevalidated_array(array=point.array + translation)
            for name, point in landmark_positions.items()
        }

    return world_orientations, world_origins, landmark_positions


def synthesize_from_euler(
    *,
    skeleton: SkeletonDefinition,
    joint_euler_angles: Mapping[str, FloatArray],
    root_world_orientation: RotationQuaternion | None = None,
    root_origin: Point | None = None,
) -> tuple[
    dict[RigidBodySegmentName, RotationQuaternion],
    dict[RigidBodySegmentName, Point],
    dict[LandmarkNameString, Point],
]:
    """`synthesize_pose` with per-joint euler triples, composed via each joint's
    own convention.

    This is the authored-input boundary: humans and animation clips speak
    angles, so they are converted to quaternions immediately, here, and every
    computation after this call is quaternion-only.
    """
    joint_relative_orientations = {}
    for joint_name, triple in joint_euler_angles.items():
        from skellyforge.core.math.kinematics.euler_sequence import (
            compose_euler_angles,
        )

        convention = skeleton.joints[joint_name].convention
        joint_relative_orientations[joint_name] = compose_euler_angles(
            sequence=convention.sequence,
            angles=np.asarray(triple, dtype=np.float64) + np.asarray(convention.zero_offsets),
        )
    return synthesize_pose(
        skeleton=skeleton,
        joint_relative_orientations=joint_relative_orientations,
        root_world_orientation=root_world_orientation,
        root_origin=root_origin,
    )
