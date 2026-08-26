"""Twist backfill: recovering unobservable axial rotation from downstream measurement.

A two-landmark segment cannot see rotation about its own long axis - the elbow
and wrist staying collinear under pronation is the canonical case, and pure
parallel transport freezes the forearm while the hand visibly spins.

But the CHAIN can see it. When a chain's distal end carries a MEASURED full
orientation (a rigid-fit terminal like the carpals), its relative rotation in
the proximal segment's frame contains exactly the twist information the
proximal segment's own landmarks lack. Backfill decomposes that relative
rotation by swing-twist about the proximal segment's primary axis and applies
the twist component to it:

- pure pronation (hand spins about the forearm axis): the relative rotation is
  pure twist -> the forearm absorbs all of it. Recovered.
- pure wrist flexion (hand bends perpendicular to the axis): the relative
  rotation is pure swing -> the forearm is untouched. Correct.
- combinations split by projection, attributing axial content to the segment
  and swing content to the joint between them.

Honesty boundaries: this is still a CONVENTION for how the observed end
rotation distributes between "segment rolled" and "joint deviated" - what
makes it scientific is that the convention is stated, deterministic, and
backed by measurement instead of path history. Multi-hop cascades up a chain
are deliberately deferred until a validated use case exists; v1 fills the
single segment directly proximal to the measured anchor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from skellyforge.core.math.geometry.rotation_quaternion import RotationQuaternion
from skellyforge.core.skeleton.chain.kinematic_chain import KinematicChain
from skellyforge.core.skeleton.pose.roll_resolution import SegmentRollReference
from skellyforge.core.skeleton.skeleton_definition import SkeletonDefinition
from skellyforge.core.skeleton.skeleton_pose import PoseSolution, SkeletonPose
from skellyforge.type_overloads import FloatArray, LinkageNameString


@dataclass(frozen=True, slots=True, eq=False)
class TwistBackfillReport:
    """What one backfill pass did, for observability.

    Attributes:
        chain_name: which chain was processed.
        updated_segment: the direction-only segment whose roll was filled, if any.
        anchor_segment: the measured (rigid-fit) segment the twist came from.
        twist_applied_degrees: signed twist about the updated segment's primary
            axis, in degrees. Zero when nothing was updated.
    """

    chain_name: str
    updated_segment: str | None
    anchor_segment: str | None
    twist_applied_degrees: float


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


def backfill_twist_from_rigid_terminal(
    *,
    skeleton: SkeletonDefinition,
    chain: KinematicChain,
    pose: SkeletonPose,
    baseline_pose: SkeletonPose,
) -> tuple[SkeletonPose, TwistBackfillReport]:
    """Fill the segment directly proximal to the chain's measured terminal.

    Walks the chain distal -> proximal for the deepest segment that hydrated as
    a RIGID_FIT (a measured full orientation). If the segment immediately
    proximal to it carries convention roll, the pair's relative rotation is
    compared against the SAME pair in `baseline_pose`, the change is decomposed
    by swing-twist about that segment's primary axis, and the twist component
    is applied - the axial content that appeared since the baseline.

    The baseline matters because quaternions live on a double cover and the
    resolver's roll is a convention: an absolute twist reading conflates the
    authored rest offsets and that convention with real motion. Only the
    CHANGED axial content since the baseline is measurement. In streaming use
    the baseline is the previous frame's pose; offline, the unrotated
    reference take.

    Returns:
        The (possibly) updated pose and a report naming what happened. A pose
        with no applicable pair is returned unchanged with an empty report.
    """
    anchor_index = None
    for index in range(len(chain.segments) - 1, -1, -1):
        segment_pose = pose.segment_poses.get(chain.segments[index].name)
        if segment_pose is None:
            continue
        if segment_pose.solved_by is PoseSolution.RIGID_FIT:
            anchor_index = index
            break
    empty_report = TwistBackfillReport(
        chain_name=chain.name,
        updated_segment=None,
        anchor_segment=None,
        twist_applied_degrees=0.0,
    )
    if anchor_index is None or anchor_index == 0:
        return pose, empty_report

    anchor_segment = chain.segments[anchor_index]
    parent_segment = chain.segments[anchor_index - 1]
    anchor_pose = pose.segment_poses.get(anchor_segment.name)
    parent_pose = pose.segment_poses.get(parent_segment.name)
    baseline_anchor_pose = baseline_pose.segment_poses.get(anchor_segment.name)
    baseline_parent_pose = baseline_pose.segment_poses.get(parent_segment.name)
    if (
        anchor_pose is None
        or parent_pose is None
        or baseline_anchor_pose is None
        or baseline_parent_pose is None
        or parent_pose.solved_by is PoseSolution.RIGID_FIT
    ):
        # A measured parent needs no backfill - and a convention has no
        # business overwriting a measurement.
        return pose, empty_report

    reference = SegmentRollReference.for_segment(
        skeleton=skeleton, segment_name=parent_segment.name
    )
    current_relative = parent_pose.orientation.inverse() * anchor_pose.orientation
    baseline_relative = (
        baseline_parent_pose.orientation.inverse() * baseline_anchor_pose.orientation
    )
    # Only the CHANGE since the baseline is measurement; the absolute twist
    # would conflate authored rest offsets and resolver convention with motion.
    relative_change = baseline_relative.inverse() * current_relative
    twist = twist_about_local_axis(
        relative_rotation=relative_change, local_axis=reference.primary_local
    )

    # Signed shortest-arc angle: positive = right-handed about the segment's
    # declared +primary axis (to_axis_angle flips the axis for w < 0).
    _axis, unsigned_angle = twist.to_axis_angle()
    twist_degrees = float(np.degrees(unsigned_angle)) * (1.0 if twist.w >= 0.0 else -1.0)

    updated_parent = parent_pose.with_orientation(
        orientation=parent_pose.orientation * twist,
        solved_by=PoseSolution.TRANSPORTED_ROLL,
    )
    updated_poses = dict(pose.segment_poses)
    updated_poses[parent_segment.name] = updated_parent

    return (
        SkeletonPose(segment_poses=updated_poses),
        TwistBackfillReport(
            chain_name=chain.name,
            updated_segment=parent_segment.name,
            anchor_segment=anchor_segment.name,
            twist_applied_degrees=twist_degrees,
        ),
    )
