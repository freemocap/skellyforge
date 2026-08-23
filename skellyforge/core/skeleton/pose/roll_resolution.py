"""Resolving the roll a two-landmark segment leaves free, continuously and without lag.

Two landmarks fix the direction a segment points in and nothing else: the rotation about
that direction is not in the data. On the shipped human skeleton that is fifty-six
segments of sixty-one, so "what roll do those get?" is not an edge case - it is most of
the skeleton, and leaving each frame to pick its own answer makes limbs spin.

The convention here is parallel transport. Each frame, the previous frame's world
secondary axis is carried forward and orthonormalized against the new direction, which
picks the roll closest to the one before it. That is continuous by construction, and
lag-free: it never averages across frames, it only chooses among the rolls that are
equally consistent with THIS frame's measurement.

The alternative - taking the shortest arc from the segment's rest pose every frame -
is stateless but jumps when the direction crosses the pole of that arc, which is exactly
where an arm passes overhead.

A resolver is stateful and therefore per-take. Call `reset()` between recordings, or
build a new one; feeding two takes through one resolver would transport roll across the
cut.

This lives in `skeleton` rather than in `math/kinematics` because it needs a
skeleton and a pose, not just vectors - the math packages stay free of model types so the
dependency only ever points one way.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from skellyforge.core.math.geometry.numeric_tolerances import MINIMUM_VECTOR_NORM
from skellyforge.core.math.geometry.rotation_quaternion import RotationQuaternion
from skellyforge.core.math.kinematics.coordinate_frame_ops import default_perpendicular
from skellyforge.core.skeleton.skeleton_definition import SkeletonDefinition
from skellyforge.core.skeleton.skeleton_pose import (
    PoseSolution,
    SegmentPose,
    SkeletonPose,
)
from skellyforge.type_overloads import FloatArray, RigidBodySegmentName


@dataclass(frozen=True, slots=True, eq=False)
class SegmentRollReference:
    """One segment's local frame, precomputed once so the per-frame path allocates less.

    Attributes:
        primary_local: the segment's origin-to-primary direction in its own local frame.
        secondary_local: a fixed local direction perpendicular to `primary_local`. Which
            one it is does not matter - it only has to be the SAME one every frame, so
            that "the roll that keeps the secondary axis where it was" means one thing.
        local_basis: the `(3, 3)` matrix whose columns are primary, secondary and their
            cross product. Orthonormal, so its transpose is its inverse.
    """

    primary_local: FloatArray
    secondary_local: FloatArray
    local_basis: FloatArray

    @classmethod
    def for_segment(
        cls, *, skeleton: SkeletonDefinition, segment_name: RigidBodySegmentName
    ) -> SegmentRollReference:
        """Build a segment's local roll reference from its authored rest positions."""
        segment = skeleton.segments[segment_name]
        primary_position = skeleton.landmarks[
            segment.frame_definition.primary_point_name
        ].local_position.array
        norm = float(np.linalg.norm(primary_position))
        if norm < MINIMUM_VECTOR_NORM:
            raise ValueError(
                f"segment {segment_name!r}: its primary landmark sits on its own origin, "
                "so it has no local direction to resolve roll about"
            )
        # Signed by the declared primary axis, matching `calculate_direction` on the world
        # side and `_local_primary_direction` in hydration - all three name the same thing.
        primary_local = (
            float(segment.frame_definition.primary_axis.sign) * primary_position / norm
        )
        secondary_local = default_perpendicular(direction=primary_local)
        return cls(
            primary_local=primary_local,
            secondary_local=secondary_local,
            local_basis=np.column_stack(
                [
                    primary_local,
                    secondary_local,
                    np.cross(primary_local, secondary_local),
                ]
            ),
        )


@dataclass(slots=True, eq=False)
class ContinuousRollResolver:
    """Gives every direction-only segment a roll that is continuous across frames.

    Build one per take, feed it each frame's `SkeletonPose` in order, and use what it
    returns. Rigid-fit poses pass through untouched - their roll is measured, and a
    convention has no business overwriting a measurement.

    This is mutable by necessity: carrying the previous frame's roll forward is the whole
    mechanism. The poses it hands back are frozen as ever.
    """

    references_by_segment_name: dict[RigidBodySegmentName, SegmentRollReference]
    _carried_secondary_by_segment_name: dict[RigidBodySegmentName, FloatArray] = field(
        init=False, repr=False, default_factory=dict
    )

    @classmethod
    def for_skeleton(cls, *, skeleton: SkeletonDefinition) -> ContinuousRollResolver:
        """Precompute a roll reference for every segment of a skeleton."""
        return cls(
            references_by_segment_name={
                name: SegmentRollReference.for_segment(
                    skeleton=skeleton, segment_name=name
                )
                for name in skeleton.segments
            }
        )

    def reset(self) -> None:
        """Forget every carried roll, so the next frame starts a fresh take."""
        self._carried_secondary_by_segment_name.clear()

    def resolve_pose(self, *, pose: SkeletonPose) -> SkeletonPose:
        """One frame's poses with every direction-only roll made continuous."""
        return SkeletonPose(
            segment_poses={
                name: self.resolve_segment_pose(pose=segment_pose)
                for name, segment_pose in pose.segment_poses.items()
            }
        )

    def resolve_segment_pose(self, *, pose: SegmentPose) -> SegmentPose:
        """One segment's pose with its free roll resolved, if it has one."""
        if pose.solved_by is not PoseSolution.DIRECTION:
            return pose
        reference = self.references_by_segment_name[pose.segment_name]

        world_primary = pose.orientation.rotate_vector(vector=reference.primary_local)
        first_axis = world_primary / np.linalg.norm(world_primary)

        carried = self._carried_secondary_by_segment_name.get(pose.segment_name)
        roll_reference = (
            pose.orientation.rotate_vector(vector=reference.secondary_local)
            if carried is None
            else carried
        )

        second_axis = roll_reference - float(np.dot(roll_reference, first_axis)) * first_axis
        residual_norm = float(np.linalg.norm(second_axis))
        if residual_norm < MINIMUM_VECTOR_NORM:
            # The carried axis has swung onto the new direction, so there is no roll left
            # in it to preserve. Restart from the deterministic perpendicular rather than
            # normalizing rounding error.
            second_axis = default_perpendicular(direction=first_axis)
        else:
            second_axis = second_axis / residual_norm
        third_axis = np.cross(first_axis, second_axis)

        world_basis = np.column_stack([first_axis, second_axis, third_axis])
        self._carried_secondary_by_segment_name[pose.segment_name] = second_axis

        return pose.with_orientation(
            orientation=RotationQuaternion.from_rotation_matrix(
                matrix=world_basis @ reference.local_basis.T
            ),
            solved_by=PoseSolution.TRANSPORTED_ROLL,
        )
