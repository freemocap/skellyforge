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
from collections.abc import Mapping

import numpy as np

from skellyforge.core.math.geometry.numeric_tolerances import (
    MINIMUM_SINE_BETWEEN_DEFINING_VECTORS,
    MINIMUM_VECTOR_NORM,
)
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
    """Gives every direction-only segment a roll that is deterministic per frame.

    Two resolution tiers, by call level:

    - ``resolve_pose`` (skeleton level, what production uses): whenever a
      direction-only segment's PARENT pose exists this frame, the roll is
      ANCHORED - the secondary axis is projected from the direction pointing
      back up the chain toward the parent segment's origin (its own proximal
      joint), which is roll-free measured geometry. Same motion therefore
      yields the same roll regardless of history. When no anchor is usable
      (parent unhydrated, or hint collinear with the segment's long axis -
      straight chains carry no roll reference at all), it falls back to
      parallel transport against the carried roll.

    - ``resolve_segment_pose`` (segment level): pure parallel transport, the
      historical primitive. Available without context; nothing here pretends
      an anchor exists when the caller could not supply one.

    Rigid-fit poses pass through untouched in both tiers - their roll is
    measured, and a convention has no business overwriting a measurement.

    This is mutable by necessity: carrying the previous frame's roll forward is
    the fallback's mechanism (and keeps anchored frames' neighbors continuous).
    The poses it hands back are frozen as ever.
    """

    references_by_segment_name: dict[RigidBodySegmentName, SegmentRollReference]
    skeleton: SkeletonDefinition
    rest_relative_orientations: Mapping[RigidBodySegmentName, RotationQuaternion] | None
    _carried_secondary_by_segment_name: dict[RigidBodySegmentName, FloatArray] = field(
        init=False, repr=False, default_factory=dict
    )
    _parent_of: dict[RigidBodySegmentName, RigidBodySegmentName] = field(
        init=False, repr=False, default_factory=dict
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_parent_of",
            {
                joint.child.name: joint.parent.name
                for joint in self.skeleton.joints.values()
            },
        )

    @classmethod
    def for_skeleton(
        cls,
        *,
        skeleton: SkeletonDefinition,
        rest_relative_orientations: Mapping[
            RigidBodySegmentName, RotationQuaternion
        ]
        | None = None,
    ) -> ContinuousRollResolver:
        """Precompute a roll reference for every segment of a skeleton.

        Pass `rest_relative_orientations` (the rest pose's authored
        parent-relative rotations, keyed by child segment) to enable the
        terminal twist-backfill pass in `resolve_pose`.
        """
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
        """Forget every carried roll, so the next frame starts a fresh take."""
        self._carried_secondary_by_segment_name.clear()

    def resolve_pose(self, *, pose: SkeletonPose) -> SkeletonPose:
        """One frame's poses with every direction-only roll resolved.

        Anchored where the parent's origin provides a reference this frame;
        transported otherwise; then - when rest relative orientations were
        supplied at construction - twist-backfilled along every declared chain
        from its measured rigid-fit terminal.
        """
        resolved: dict[RigidBodySegmentName, SegmentPose] = {}
        for name, segment_pose in pose.segment_poses.items():
            anchor_hint: FloatArray | None = None
            if segment_pose.solved_by is PoseSolution.DIRECTION:
                parent_name = self._parent_of.get(name)
                parent_pose = (
                    pose.segment_poses.get(parent_name) if parent_name else None
                )
                if parent_pose is not None:
                    anchor_hint = (
                        parent_pose.origin.array - segment_pose.origin.array
                    )
            resolved[name] = self._resolve_segment_pose_with_optional_anchor(
                pose=segment_pose, anchor_hint=anchor_hint
            )
        resolved_skeleton = SkeletonPose(segment_poses=resolved)

        if self.rest_relative_orientations is not None:
            from skellyforge.core.skeleton.chain.twist_backfill import (
                apply_terminal_twist_backfills,
            )

            # Local import: the backfill pass is a chain-layer strategy that
            # reads roll resolution's output; an eager import would close the
            # skeleton_definition -> chain -> pose -> skeleton_definition cycle.
            resolved_skeleton = apply_terminal_twist_backfills(
                skeleton=self.skeleton,
                pose=resolved_skeleton,
                rest_relative_orientations=self.rest_relative_orientations,
            )
        return resolved_skeleton

    def resolve_segment_pose(self, *, pose: SegmentPose) -> SegmentPose:
        """One segment's pose with its free roll resolved by parallel transport."""
        return self._resolve_segment_pose_with_optional_anchor(
            pose=pose, anchor_hint=None
        )


    def _resolve_segment_pose_with_optional_anchor(
        self, *, pose: SegmentPose, anchor_hint: FloatArray | None
    ) -> SegmentPose:
        """Resolve one direction-only segment's roll.

        With a usable `anchor_hint` (parent origin minus this origin), the
        secondary axis is projected from it - deterministic per frame. Without
        one, the previous frame's roll is transported forward. Either way the
        measured long axis is untouched and the carry is updated.
        """
        if pose.solved_by is not PoseSolution.DIRECTION:
            return pose
        reference = self.references_by_segment_name[pose.segment_name]

        world_primary = pose.orientation.rotate_vector(vector=reference.primary_local)
        first_axis = world_primary / np.linalg.norm(world_primary)

        anchored_second: FloatArray | None = None
        if anchor_hint is not None:
            hint_norm = float(np.linalg.norm(anchor_hint))
            if hint_norm >= MINIMUM_VECTOR_NORM:
                hint = anchor_hint / hint_norm
                projected = hint - float(np.dot(hint, first_axis)) * first_axis
                projected_norm = float(np.linalg.norm(projected))
                if projected_norm >= MINIMUM_SINE_BETWEEN_DEFINING_VECTORS:
                    # A live anchor: the direction back up the chain, kept
                    # perpendicular to the long axis. Deterministic - no history.
                    anchored_second = projected / projected_norm

        carried = self._carried_secondary_by_segment_name.get(pose.segment_name)
        if anchored_second is not None:
            second_axis = anchored_second
        else:
            roll_reference = (
                pose.orientation.rotate_vector(vector=reference.secondary_local)
                if carried is None
                else carried
            )
            second_axis = (
                roll_reference
                - float(np.dot(roll_reference, first_axis)) * first_axis
            )
            residual_norm = float(np.linalg.norm(second_axis))
            if residual_norm < MINIMUM_VECTOR_NORM:
                # The carried axis has swung onto the new direction, so there is no
                # roll left in it to preserve. Restart from the deterministic
                # perpendicular rather than normalizing rounding error.
                second_axis = default_perpendicular(direction=first_axis)
            else:
                second_axis = second_axis / residual_norm

        third_axis = np.cross(first_axis, second_axis)

        world_basis = np.column_stack([first_axis, second_axis, third_axis])
        # The carry updates on BOTH paths, so a frame that loses its anchor
        # falls back from the last anchored state rather than an ancient one.
        self._carried_secondary_by_segment_name[pose.segment_name] = second_axis

        return pose.with_orientation(
            orientation=RotationQuaternion.from_rotation_matrix(
                matrix=world_basis @ reference.local_basis.T
            ),
            solved_by=PoseSolution.TRANSPORTED_ROLL,
        )
