"""A segment defined by only two landmarks, whose roll is therefore unresolved.

Two points fix a direction but not a frame: rotation about the line joining them is
unconstrained. This type says so structurally - it has no `calculate_basis`, because it
genuinely cannot produce one, and callers needing a full frame must get their roll from
somewhere else (the damped minimal-roll tier).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from skellyforge.core.math.geometry.orthonormal_basis.calculate_orthonormal_basis import (
    direction_along,
)
from skellyforge.core.math.geometry.orthonormal_basis.spatial_axis import SpatialAxis
from skellyforge.core.math.geometry.spatial_vectors import Point, UnitVector
from skellyforge.core.skeleton_parts.anatomical_landmark import AnatomicalLandmark
from skellyforge.core.skeleton_parts.naming import (
    raise_unless_aliases_are_valid,
    raise_unless_snake_case_segment_name,
)
from skellyforge.type_overloads import LandmarkNameString, RigidBodySegmentName


@dataclass(frozen=True, slots=True, eq=False)
class UnderspecifiedRigidBodySegment:
    """A two-landmark segment: an origin, a distal point, and the axis between them.

    Attributes:
        name: snake_case segment name, optionally suffixed `.L` or `.R`.
        origin_landmark: the landmark at the segment's origin.
        distal_landmark: the landmark the primary axis points at.
        primary_axis: the signed local axis running from origin to distal. A negative
            axis means the distal landmark lies on that axis's negative half.
        aliases: other names this segment answers to.
    """

    name: RigidBodySegmentName
    origin_landmark: AnatomicalLandmark
    distal_landmark: AnatomicalLandmark
    primary_axis: SpatialAxis
    aliases: tuple[RigidBodySegmentName, ...] = ()

    def __post_init__(self) -> None:
        raise_unless_snake_case_segment_name(name=self.name)
        raise_unless_aliases_are_valid(name=self.name, aliases=self.aliases)
        if self.origin_landmark.name == self.distal_landmark.name:
            raise ValueError(
                f"segment {self.name!r}: origin and distal landmarks must differ - both "
                f"are {self.origin_landmark.name!r}"
            )

    @property
    def all_names(self) -> tuple[RigidBodySegmentName, ...]:
        """Every name this segment answers to, canonical name first."""
        return (self.name, *self.aliases)

    @property
    def landmark_names(self) -> tuple[LandmarkNameString, LandmarkNameString]:
        """The landmark names this segment needs observed positions for."""
        return (self.origin_landmark.name, self.distal_landmark.name)

    @property
    def length(self) -> float:
        """Origin-to-distal distance, from the landmarks' rest positions."""
        return float(
            (self.distal_landmark.local_position - self.origin_landmark.local_position).norm()
        )

    def calculate_direction(self, *, points: Mapping[str, Point]) -> UnitVector:
        """The observed origin-to-distal direction, signed to match `primary_axis`.

        Vectorized over the leading dimensions of `points`, so one frame, a rolling
        window, and a whole take all go through this unchanged.

        Args:
            points: observed world locations keyed by landmark name.

        Returns:
            The unit direction the segment currently points in.
        """
        origin, distal = (points[name] for name in self.landmark_names)
        return direction_along(
            axis=self.primary_axis,
            displacement=distal - origin,
            description=(
                f"segment {self.name!r}'s primary axis (`{self.origin_landmark.name}` -> "
                f"`{self.distal_landmark.name}`)"
            ),
        )

    def __str__(self) -> str:
        return (
            f"{self.name} (underspecified, roll unresolved): "
            f"{self.origin_landmark.name} -> {self.distal_landmark.name} "
            f"along {self.primary_axis}, length {self.length:.4g}"
        )

