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
)
from skellyforge.core.skeleton.pose.pose_tensor import PoseTensor
from skellyforge.core.skeleton.skeleton_definition import SkeletonDefinition
from skellyforge.core.skeleton.skeleton_pose import (
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
    _tensor_state: object = field(init=False, repr=False)
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
        from skellyforge.core.skeleton.pose.roll_tensor import RollTensorState
        self._tensor_state = RollTensorState(skeleton=self.skeleton,
            references=self.references_by_segment_name,
            rest_relative=self.rest_relative_orientations, rest_world=self._rest_world)

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
        """Forget the previous take's numerical orientation history."""
        self._tensor_state.reset()

    def resolve_tensor(self, *, poses: PoseTensor, reset_before: np.ndarray | None = None) -> PoseTensor:
        """Resolve chronological samples with numeric provenance and explicit resets.

        Absent samples do not imply a new take. Set reset_before on the source
        events that reset the existing pipeline; splitting a batch retains state.
        Inputs are borrowed and never mutated; resolved rotations/codes are new.
        """
        return self._tensor_state.resolve(poses=poses, reset_before=reset_before)

    def resolve_pose(self, *, pose: SkeletonPose) -> SkeletonPose:
        """Adapt existing object callers to the same numerical pose operation."""
        from skellyforge.core.skeleton.pose.pose_tensor import PoseTensor
        if not pose.segment_poses:
            self.reset()
            return pose
        tensor = PoseTensor.from_pose(pose=pose, segment_names=tuple(self.skeleton.segments))
        return self.resolve_tensor(poses=tensor).to_pose()

    def resolve_segment_pose(self, *, pose: SegmentPose) -> SegmentPose:
        """Adapt an independent segment without replacing other segment histories."""
        from skellyforge.core.skeleton.pose.pose_tensor import PoseTensor
        tensor = PoseTensor.from_pose(pose=SkeletonPose(segment_poses={pose.segment_name: pose}),
                                     segment_names=tuple(self.skeleton.segments))
        return self._tensor_state.resolve_independent(poses=tensor).to_pose().segment_poses[pose.segment_name]
