"""Twist backfill: recovering unobservable axial rotation from downstream measurement.

A two-landmark segment cannot see rotation about its own long axis - the elbow
and wrist staying collinear under pronation is the canonical case, and pure
transport freezes the forearm while the hand visibly spins.

But the CHAIN can see it. When a chain contains a MEASURED full orientation (a
rigid-fit segment like the carpals), the relative rotations between it and its
proximal neighbors contain exactly the twist information their own landmark
pairs lack. Backfill decomposes each such pair rotation by swing-twist about
the proximal segment's primary axis and applies the twist component:

- pure pronation (hand spins about the forearm axis): the pair's relative
  rotation is pure twist -> the forearm absorbs all of it. Recovered.
- pure wrist flexion (hand bends perpendicular to the axis): pure swing ->
  the forearm is untouched. Correct.
- combinations split by projection, attributing axial content to the segment
  and swing content to the joint between them.

The reference each pair is measured against is the AUTHORED REST relative
orientation, which makes backfill a pure function of the current frame -
deterministic and history-independent, exactly like the anchored secondary
axes it complements. The constant offset this introduces versus any prior
transport state is a stated convention, not drift.

Honesty boundaries: attributing axial content to the segment (rather than the
joint) is a convention - what makes it scientific is that it is deterministic,
measurement-backed, and stated. Cascading proximal from the measured terminal
is v1 scope; weighting schemes across multiple distal measurements are future
work.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from skellyforge.core.math.geometry.rotation_quaternion import RotationQuaternion
from skellyforge.core.skeleton.pose.roll_resolution import SegmentRollReference
from skellyforge.core.skeleton.skeleton_definition import SkeletonDefinition
from skellyforge.core.skeleton.skeleton_pose import PoseSolution, SkeletonPose
from skellyforge.type_overloads import FloatArray


def twist_about_local_axis(
    *, relative_rotation: RotationQuaternion, local_axis: FloatArray
) -> RotationQuaternion:
    """The twist component of `relative_rotation` about `local_axis`.

    Standard swing-twist projection: the quaternion whose vector part is the
    rotation's vector part projected onto the axis. Identity when the rotation
    carries no component about the axis.
    """
    axis_unit = np.asarray(local_axis, dtype=np.float64)
    axis_unit = axis_unit / np.linalg.norm(axis_unit)
    vector_part = np.array(
        [relative_rotation.x, relative_rotation.y, relative_rotation.z],
        dtype=np.float64,
    )
    projection = float(vector_part @ axis_unit)
    twist_unnormalized = np.concatenate(
        [np.asarray([relative_rotation.w]), axis_unit * projection]
    )
    norm = float(np.linalg.norm(twist_unnormalized))
    if norm < 1e-9:
        return RotationQuaternion.identity()
    return RotationQuaternion.from_components(
        w=float(twist_unnormalized[0]),
        x=float(twist_unnormalized[1]),
        y=float(twist_unnormalized[2]),
        z=float(twist_unnormalized[3]),
    )


def apply_terminal_twist_backfills(
    *,
    skeleton: SkeletonDefinition,
    pose: SkeletonPose,
    rest_relative_orientations: Mapping[str, RotationQuaternion],
) -> SkeletonPose:
    """Backfill rolls along every declared chain that has a measured terminal.

    For each chain, walk distal -> proximal from the DEEPEST rigid-fit segment.
    While the immediately proximal segment exists, hydrates as direction-only
    (convention-carried roll), and has an authored rest relative orientation in
    `rest_relative_orientations`, apply the pair's rest-referenced twist delta
    to it, then continue proximal using the updated orientation.

    Args:
        skeleton: the skeleton whose chains and segments these are.
        pose: the resolver output (anchored + transported) being finalized.
        rest_relative_orientations: authored rest parent-relative rotations
            keyed by CHILD segment name - the same map the rest pose composes
            from, used here as the fixed reference each pair's twist is
            measured against.

    Returns:
        A pose whose convention-carried rolls along measured-terminal chains
        include the axial content their own landmarks could not see.
    """
    updated: dict[str, RotationQuaternion] = {}
    resolved_orientation: dict[str, RotationQuaternion] = {
        name: segment_pose.orientation
        for name, segment_pose in pose.segment_poses.items()
    }

    def current_orientation(segment_name: str) -> RotationQuaternion | None:
        return updated.get(segment_name, resolved_orientation.get(segment_name))

    for chain in skeleton.chains.values():
        anchor_index = None
        for index in range(len(chain.segments) - 1, -1, -1):
            segment_name = chain.segments[index].name
            segment_pose = pose.segment_poses.get(segment_name)
            if segment_pose is None:
                continue
            if segment_pose.solved_by is PoseSolution.RIGID_FIT:
                anchor_index = index
                break

        # Cascade proximal from the measured terminal. Each pair's twist delta
        # is measured against the authored rest relationship, so the result is
        # a pure function of this frame's geometry.
        for index in range(anchor_index, 0, -1) if anchor_index is not None else []:
            parent_name = chain.segments[index - 1].name
            child_name = chain.segments[index].name
            parent_pose = pose.segment_poses.get(parent_name)
            if parent_pose is None:
                break
            if parent_pose.solved_by is PoseSolution.RIGID_FIT:
                break  # a measured parent needs no fill - and stays untouched
            rest_relative = rest_relative_orientations.get(child_name)
            child_orientation = current_orientation(child_name)
            parent_orientation = current_orientation(parent_name)
            if rest_relative is None or child_orientation is None or parent_orientation is None:
                break

            reference = SegmentRollReference.for_segment(
                skeleton=skeleton, segment_name=parent_name
            )
            current_relative = (
                parent_orientation.inverse() * child_orientation
            )
            relative_change = rest_relative.inverse() * current_relative
            twist = twist_about_local_axis(
                relative_rotation=relative_change,
                local_axis=reference.primary_local,
            )

            updated[parent_name] = (
                current_orientation(parent_name) * twist
            )

    if not updated:
        return pose

    new_poses = {}
    for name, segment_pose in pose.segment_poses.items():
        if name in updated:
            new_poses[name] = segment_pose.with_orientation(
                orientation=updated[name],
                solved_by=PoseSolution.TRANSPORTED_ROLL,
            )
        else:
            new_poses[name] = segment_pose
    return SkeletonPose(segment_poses=new_poses)
