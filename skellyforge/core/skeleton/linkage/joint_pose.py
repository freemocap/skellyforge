"""The hydrated face of a joint: relative orientation, named angles, provenance."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from skellyforge.core.math.geometry.rotation_quaternion import RotationQuaternion
from skellyforge.core.math.kinematics.euler_sequence import decompose_euler_angles
from skellyforge.core.skeleton.linkage.joint_definition import JointDefinition
from skellyforge.core.skeleton.skeleton_pose import PoseSolution, SegmentPose, SkeletonPose
from skellyforge.type_overloads import FloatArray, LinkageNameString


def relative_orientation(
    *, parent_orientation: RotationQuaternion, child_pose: SegmentPose
) -> RotationQuaternion:
    """The child segment's orientation expressed in its parent's frame.

    This is the linkage layer's primitive: the rotation a joint's angles
    describe. Both inputs are world (local-to-world) orientations.
    """
    return parent_orientation.inverse() * child_pose.orientation


@dataclass(frozen=True, slots=True, eq=False)
class JointInputProvenance:
    """Which closed forms produced the two poses an angle stands on.

    An angle computed through any `TRANSPORTED_ROLL` or `DIRECTION` input is
    partly convention rather than measurement - consumers comparing sessions,
    subjects, or publications deserve to see that without digging.
    """

    parent_solution: PoseSolution
    child_solution: PoseSolution

    @property
    def fully_measured(self) -> bool:
        """Whether both inputs were rigid-body fits (nothing convention-carried)."""
        return self.parent_solution is PoseSolution.RIGID_FIT and (
            self.child_solution is PoseSolution.RIGID_FIT
        )


@dataclass(frozen=True, slots=True, eq=False)
class JointPose:
    """One joint's hydrated state at one instant.

    Attributes:
        name: the joint's name, from its definition.
        relative_orientation: the child's orientation in the parent's frame.
        angles: the relative orientation decomposed under the joint's
            convention, with the convention's zero offsets applied.
        angle_names: the names of `angles`, position-matched.
        provenance: what produced each input pose.
    """

    name: LinkageNameString
    relative_orientation: RotationQuaternion
    angles: tuple[float, float, float]
    angle_names: tuple[str, str, str]
    provenance: JointInputProvenance


def joint_pose_from_segment_poses(
    *,
    joint: JointDefinition,
    parent_pose: SegmentPose,
    child_pose: SegmentPose,
) -> JointPose:
    """Hydrate one joint from its two segments' solved poses."""
    relative = relative_orientation(
        parent_orientation=parent_pose.orientation,
        child_pose=child_pose,
    )
    raw_angles = decompose_euler_angles(
        quaternion=relative, sequence=joint.convention.sequence
    )
    offsets = np.asarray(joint.convention.zero_offsets, dtype=np.float64)
    angles = tuple(float(angle + offset) for angle, offset in zip(raw_angles, offsets))
    return JointPose(
        name=joint.name,
        relative_orientation=relative,
        angles=angles,
        angle_names=joint.convention.angle_names,
        provenance=JointInputProvenance(
            parent_solution=parent_pose.solved_by,
            child_solution=child_pose.solved_by,
        ),
    )


def compute_joint_poses(
    *, skeleton, pose: SkeletonPose  # SkeletonDefinition; untyped to keep layering import-light
) -> Mapping[LinkageNameString, JointPose]:
    """Every joint whose parent AND child poses exist this frame, keyed by name.

    Joints touching a segment the partial hydration dropped are omitted - the
    same rule landmark hydration follows for unobserved segments.
    """
    result: dict[LinkageNameString, JointPose] = {}
    for joint in skeleton.joints.values():
        parent_pose = pose.segment_poses.get(joint.parent.name)
        child_pose = pose.segment_poses.get(joint.child.name)
        if parent_pose is None or child_pose is None:
            continue
        result[joint.name] = joint_pose_from_segment_poses(
            joint=joint, parent_pose=parent_pose, child_pose=child_pose
        )
    return result


def joint_angles_vector(*, joint_poses: Mapping[LinkageNameString, JointPose], joint_name: str) -> FloatArray:
    """One joint's three named angles as an array, in convention order."""
    return np.asarray(joint_poses[joint_name].angles, dtype=np.float64)
