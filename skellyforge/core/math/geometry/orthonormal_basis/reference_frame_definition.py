"""The declarative spec of a reference frame: which points, which signed axes, which chirality."""

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
            differ from the primary axis, but either sign is allowed.
        secondary_point_name: point lying approximately along `secondary_axis`.
        handedness: chirality of the resulting triad, right-handed by default. Left-handed
            frames emit a `LeftHandedCoordinateSystemWarning` because they flip the sign of
            every cross product, rotation, torque, and angular velocity computed in them.
    """

    origin_point_name: str
    primary_axis: SpatialAxis
    primary_point_name: str
    secondary_axis: SpatialAxis
    secondary_point_name: str
    handedness: Handedness = Handedness.RIGHT_HANDED

    def __post_init__(self) -> None:
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
        if self.origin_point_name in (self.primary_point_name, self.secondary_point_name):
            raise ValueError(
                f"Origin point `{self.origin_point_name}` must differ from the "
                "primary and secondary points"
            )
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

    @property
    def tertiary_axis(self) -> SpatialAxis:
        """The axis produced purely by the cross product of the other two."""
        return self.primary_axis.remaining_axis(self.secondary_axis)
