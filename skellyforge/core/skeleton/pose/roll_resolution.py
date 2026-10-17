"""Resolve unobserved axial rotation by minimum-swing transport.

A direction measurement constrains two rotational degrees of freedom. The
remaining twist starts from the model's rest orientation and is carried in its
parent's rotating frame; independent terminal orientation evidence is applied
by the declared chain model. No bend-plane normal is inferred from adjacent
origins. This is an explicit motion convention, not full anatomical IK.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Mapping

import numpy as np

from skellyforge.core.math.geometry.numeric_tolerances import (
    MINIMUM_VECTOR_NORM,
)
from skellyforge.core.math.geometry.rotation_quaternion import RotationQuaternion
from skellyforge.core.math.kinematics.coordinate_frame_ops import (
    default_perpendicular,
    rotation_between_vectors,
)
from skellyforge.core.math.geometry.spatial_vectors import UnitVector
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
    """

    primary_local: FloatArray
    secondary_local: FloatArray

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
        )


@dataclass(slots=True, eq=False)
class ContinuousRollResolver:
    """Minimum-swing orientation transport in the moving parent's frame.

    Initialize from authored rest rotations. Subsequently carry the previous
    orientation with the parent's rotation, then swing its primary axis onto
    the observed direction. No parent-origin vectors or bend thresholds enter
    this calculation. Rigid-fit orientations pass through unchanged.

    The undetermined twist is a convention, not a measurement. Transport is
    path dependent (including geometric phase); it is not a temporal low-pass
    filter. A missing parent uses world-frame transport until that parent has
    been observed in consecutive frames. Missing segments are not fabricated.
    Use a new resolver or reset between takes. Feed frames in chronological order.
    """

    references_by_segment_name: dict[RigidBodySegmentName, SegmentRollReference]
    skeleton: SkeletonDefinition
    rest_relative_orientations: Mapping[RigidBodySegmentName, RotationQuaternion]
    _previous: dict[RigidBodySegmentName, RotationQuaternion] = field(
        init=False, repr=False, default_factory=dict
    )
    _parent_of: dict[RigidBodySegmentName, RigidBodySegmentName] = field(
        init=False, repr=False, default_factory=dict
    )
    _rest_world: dict[RigidBodySegmentName, RotationQuaternion] = field(
        init=False, repr=False, default_factory=dict
    )

    def __post_init__(self) -> None:
        if set(self.rest_relative_orientations) != set(self.skeleton.segments):
            raise ValueError(
                "Rest rotations must cover the skeleton's exact segment set"
            )
        self.rest_relative_orientations = dict(self.rest_relative_orientations)
        self._parent_of = {
            joint.child.name: joint.parent.name
            for joint in self.skeleton.joints.values()
        }
        visiting = set()

        def resolve_rest(name):
            if name in self._rest_world:
                return self._rest_world[name]
            if name in visiting:
                raise ValueError("Skeleton parent cycle")
            visiting.add(name)
            parent = self._parent_of.get(name)
            rotation = self.rest_relative_orientations[name]
            if parent is not None:
                rotation = resolve_rest(parent) * rotation
            self._rest_world[name] = rotation
            visiting.remove(name)
            return rotation

        for name in self.skeleton.segments:
            resolve_rest(name)

    @classmethod
    def for_skeleton(
        cls,
        *,
        skeleton: SkeletonDefinition,
        rest_relative_orientations: Mapping[RigidBodySegmentName, RotationQuaternion],
    ) -> ContinuousRollResolver:
        """Create a per-take resolver with explicit authored rest rotations."""
        return cls(
            references_by_segment_name={
                name: SegmentRollReference.for_segment(
                    skeleton=skeleton, segment_name=name
                )
                for name in skeleton.segments
            },
            skeleton=skeleton,
            rest_relative_orientations=rest_relative_orientations,
        )

    def reset(self) -> None:
        """Forget the previous take's orientations."""
        self._previous.clear()

    def resolve_pose(self, *, pose: SkeletonPose) -> SkeletonPose:
        """Resolve parents before children, then apply declared terminal twist evidence."""
        if not set(pose.segment_poses).issubset(self.skeleton.segments):
            raise ValueError("Pose contains unknown segments")
        resolved: dict[RigidBodySegmentName, SegmentPose] = {}

        def resolve(name):
            if name in resolved:
                return resolved[name]
            source = pose.segment_poses[name]
            parent = self._parent_of.get(name)
            reference = self._previous.get(name, self._rest_world[name])
            if parent in pose.segment_poses:
                parent_rotation = resolve(parent).orientation
                if name not in self._previous:
                    reference = parent_rotation * self.rest_relative_orientations[name]
                elif parent in self._previous:
                    reference = (
                        parent_rotation * self._previous[parent].inverse() * reference
                    )
            resolved[name] = self._aim(pose=source, reference=reference)
            return resolved[name]

        for name in pose.segment_poses:
            resolve(name)

        from skellyforge.core.skeleton.chain.twist_backfill import (
            apply_terminal_twist_backfills,
        )

        result = apply_terminal_twist_backfills(
            skeleton=self.skeleton,
            pose=SkeletonPose(segment_poses=resolved),
            rest_relative_orientations=self.rest_relative_orientations,
        )
        # Carry the final orientations, including any distal evidence correction.
        self._previous = {
            name: segment.orientation for name, segment in result.segment_poses.items()
        }
        return result

    def resolve_segment_pose(self, *, pose: SegmentPose) -> SegmentPose:
        """Apply the same minimum swing without a parent observation (world frame)."""
        result = self._aim(
            pose=pose,
            reference=self._previous.get(
                pose.segment_name, self._rest_world[pose.segment_name]
            ),
        )
        self._previous[pose.segment_name] = result.orientation
        return result

    def _aim(self, *, pose: SegmentPose, reference: RotationQuaternion) -> SegmentPose:
        if pose.solved_by is not PoseSolution.DIRECTION:
            return pose
        axes = self.references_by_segment_name[pose.segment_name]
        before = reference.rotate_vector(vector=axes.primary_local)
        after = pose.orientation.rotate_vector(vector=axes.primary_local)
        if (
            np.linalg.norm(np.cross(before, after)) < MINIMUM_VECTOR_NORM
            and np.dot(before, after) < 0
        ):
            # Exact half-turn: infinitely many swings exist. Use the reference's
            # transverse axis, so the choice rotates with the reference frame.
            swing = RotationQuaternion.from_rotation_vector(
                rotation_vector=reference.rotate_vector(vector=axes.secondary_local)
                * np.pi
            )
        else:
            swing = rotation_between_vectors(
                from_direction=UnitVector.from_array(values=before),
                to_direction=UnitVector.from_array(values=after),
            )
        return pose.with_orientation(
            orientation=swing * reference, solved_by=PoseSolution.TRANSPORTED_ROLL
        )
