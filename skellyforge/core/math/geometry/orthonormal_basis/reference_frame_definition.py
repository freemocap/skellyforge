"""The declarative spec of a reference frame: which points, which signed axes, which chirality.

A definition is FULLY SPECIFIED when it names a secondary axis and point, and
UNDERSPECIFIED when it does not. An underspecified definition still pins a direction -
the signed primary axis running from the origin point to the primary point - but leaves
roll about that direction free, because two points cannot constrain it. Callers needing a
full triad from an underspecified definition must supply the roll from somewhere else.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

from skellyforge.core.math.geometry.orthonormal_basis.handedness import (
    Handedness,
    LeftHandedCoordinateSystemWarning,
)
from skellyforge.core.math.geometry.orthonormal_basis.spatial_axis import SpatialAxis


@dataclass(frozen=True, slots=True)
class ReferenceFrameDefinition:
    """Names the points, signed axes, and handedness that define a reference frame.

    Attributes:
        origin_point_name: point placed at the origin of the frame.
        primary_axis: signed axis that points EXACTLY at `primary_point_name`.
        primary_point_name: point lying exactly along `primary_axis`.
        secondary_axis: signed axis that points APPROXIMATELY at `secondary_point_name`;
            it is orthogonalized against the primary axis by Gram-Schmidt. Its axis must
            differ from the primary axis, but either sign is allowed. `None` together with
            `secondary_point_name` means the definition is underspecified.
        secondary_point_name: point lying approximately along `secondary_axis`. `None`
            together with `secondary_axis` means the definition is underspecified.
        handedness: chirality of the resulting triad, right-handed by default. Left-handed
            frames emit a `LeftHandedCoordinateSystemWarning` because they flip the sign of
            every cross product, rotation, torque, and angular velocity computed in them.
            The warning fires whether or not the definition is fully specified: an
            underspecified one has no triad yet, but it carries the handedness it will be
            completed with.
    """

    origin_point_name: str
    primary_axis: SpatialAxis
    primary_point_name: str
    secondary_axis: SpatialAxis | None = None
    secondary_point_name: str | None = None
    handedness: Handedness = Handedness.RIGHT_HANDED

    def __post_init__(self) -> None:
        if (self.secondary_axis is None) != (self.secondary_point_name is None):
            raise ValueError(
                "`secondary_axis` and `secondary_point_name` describe one thing and must "
                "be given together or not at all - got secondary_axis="
                f"{self.secondary_axis}, secondary_point_name={self.secondary_point_name!r}"
            )
        if self.origin_point_name == self.primary_point_name:
            raise ValueError(
                "Origin and primary points must differ - both are "
                f"`{self.origin_point_name}`"
            )
        # Warned about before the underspecified early return below, because "every
        # construction of a left-handed frame announces itself" has to mean every one.
        # An underspecified definition has no triad yet, but it carries the handedness it
        # will be completed with, and that is the thing worth hearing about early.
        if self.handedness is Handedness.LEFT_HANDED:
            warnings.warn(
                message=(
                    "Building a LEFT_HANDED coordinate system: every cross product, "
                    "rotation direction, torque, and angular velocity expressed in this "
                    "frame will have the opposite sign from the right-handed convention. "
                    "Use Handedness.RIGHT_HANDED unless you are deliberately matching a "
                    "left-handed external convention."
                ),
                category=LeftHandedCoordinateSystemWarning,
                stacklevel=2,
            )
        if not self.is_fully_specified:
            return

        if self.primary_axis.index == self.secondary_axis.index:
            raise ValueError(
                "Primary and secondary axes must lie along different cartesian axes - "
                f"got {self.primary_axis.name} and {self.secondary_axis.name}"
            )
        if self.primary_point_name == self.secondary_point_name:
            raise ValueError(
                "Primary and secondary points must differ - both are "
                f"`{self.primary_point_name}`"
            )
        if self.origin_point_name == self.secondary_point_name:
            raise ValueError(
                f"Origin point `{self.origin_point_name}` must differ from the "
                "primary and secondary points"
            )

    @property
    def is_fully_specified(self) -> bool:
        """Whether this definition names enough to build a full orthonormal triad."""
        return self.secondary_axis is not None

    @property
    def point_names(self) -> tuple[str, ...]:
        """Every point name this definition refers to: origin, primary, then secondary."""
        if self.secondary_point_name is None:
            return (self.origin_point_name, self.primary_point_name)
        return (self.origin_point_name, self.primary_point_name, self.secondary_point_name)

    @property
    def tertiary_axis(self) -> SpatialAxis:
        """The axis produced purely by the cross product of the other two."""
        if self.secondary_axis is None:
            raise ValueError(
                "An underspecified reference frame definition has no tertiary axis - "
                f"`{self.primary_point_name}` fixes only the {self.primary_axis.name} "
                "direction, leaving roll about it free"
            )
        return self.primary_axis.remaining_axis(self.secondary_axis)

    def __str__(self) -> str:
        if not self.is_fully_specified:
            return (
                f"origin {self.origin_point_name}, "
                f"{self.primary_axis} -> {self.primary_point_name}, roll unresolved"
            )
        return (
            f"origin {self.origin_point_name}, "
            f"{self.primary_axis} -> {self.primary_point_name}, "
            f"{self.secondary_axis} ~ {self.secondary_point_name}, "
            f"{self.handedness.name}"
        )
