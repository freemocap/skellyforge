"""The hydrated pose of a skeleton: each segment's world origin, orientation and scale.

A static skeleton's per-frame face is a pose: where each segment's frame is (origin), how
it is turned (orientation) and how big it is (scale). The closed-form hydration fills this
in from observed landmark positions; velocities and accelerations come later, when the
trajectory is known.

Scale is a component of the pose rather than a property of the definition because the
authored template is dimensionless - local positions are fractions of the model's reference unit - so the
map from a segment's frame into the world is a similarity, not a rigid motion. Every
hydrated segment therefore reports its own reading of how big the model is, and
`pose.model_scale_fitting` pools those readings into one.

Every pose records HOW it was solved, because the two solutions carry different amounts of
information: a rigid fit pins the full orientation, while a direction fit pins only the
long axis and leaves roll free. A consumer that has to guess which it got will guess wrong.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

from skellyforge.core.math.geometry.rotation_quaternion import RotationQuaternion
from skellyforge.core.math.geometry.spatial_vectors import Point
from skellyforge.type_overloads import RigidBodySegmentName


class PoseSolution(Enum):
    """Which closed form produced a pose, and therefore how much of it is determined."""

    RIGID_FIT = "rigid_fit"
    """Kabsch over three or more non-collinear landmarks. The full orientation is pinned."""

    DIRECTION = "direction"
    """Shortest-arc rotation from two landmarks. Roll about the long axis is arbitrary."""

    TRANSPORTED_ROLL = "transported_roll"
    """Direction from the data, roll carried from the previous frame by parallel transport.

    The long axis is measured; the roll is a CONVENTION, chosen to be continuous rather
    than recovered from anything. Distinguished from `RIGID_FIT` because the two are not
    the same claim about the world - one is measured, one is merely well-behaved.
    """


@dataclass(frozen=True, slots=True, eq=False)
class SegmentPose:
    """One segment's pose at one instant.

    Attributes:
        segment_name: the segment this pose belongs to.
        origin: the world position of the segment's frame origin.
        orientation: the world orientation (local-to-world rotation).
        scale_estimate: this segment's own reading of the model's size, in world
            units per unit of the skeleton's reference unit - millimetres per unit `H` for the
            standard human, when
            the observations were in millimetres. A rigid fit measures it over every
            observed landmark at once; a direction fit measures it as the observed
            origin-to-primary distance over the authored proportion. It is ONE segment's
            noisy reading, not the model's size: pooling those readings is
            `pose.model_scale_fitting`'s job.
        solved_by: which closed form produced it. `PoseSolution.DIRECTION` means the roll
            about the segment's long axis is arbitrary and must be resolved downstream.
    """

    segment_name: RigidBodySegmentName
    origin: Point
    orientation: RotationQuaternion
    scale_estimate: float
    solved_by: PoseSolution

    def __post_init__(self) -> None:
        if not self.scale_estimate > 0.0:
            raise ValueError(
                f"segment {self.segment_name!r}: scale_estimate must be positive - "
                f"got {self.scale_estimate!r}. A segment with no measurable size has "
                "no pose; it should have been left out of the hydration, not hydrated "
                "with a degenerate scale."
            )

    @property
    def has_resolved_roll(self) -> bool:
        """Whether this pose carries a usable roll about the segment's long axis."""
        return self.solved_by in (
            PoseSolution.RIGID_FIT,
            PoseSolution.TRANSPORTED_ROLL,
        )

    def with_orientation(
        self, *, orientation: RotationQuaternion, solved_by: PoseSolution
    ) -> SegmentPose:
        """This pose with a different orientation, keeping its segment, origin and scale.

        Roll resolution turns a segment about its own long axis, which changes neither where
        it sits nor how big it is - so both ride through untouched.
        """
        return SegmentPose(
            segment_name=self.segment_name,
            origin=self.origin,
            orientation=orientation,
            scale_estimate=self.scale_estimate,
            solved_by=solved_by,
        )


@dataclass(frozen=True, slots=True, eq=False)
class SkeletonPose:
    """The whole skeleton's pose at one instant, keyed by segment name."""

    segment_poses: Mapping[RigidBodySegmentName, SegmentPose]

    @property
    def segment_names_with_free_roll(self) -> tuple[RigidBodySegmentName, ...]:
        """Segments solved by direction only, whose roll a downstream pass must supply."""
        return tuple(
            sorted(
                name
                for name, pose in self.segment_poses.items()
                if not pose.has_resolved_roll
            )
        )
