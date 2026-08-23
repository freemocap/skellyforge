"""A landmark: a named point defined in the local frame of a segment.

Each landmark is a static DEFINITION - an anatomical description (medical language naming
one individual point in space) plus its rest position in the local frame of the segment
that owns it. Observed positions do not live here; they stream through a
`PointRingBuffer` keyed by landmark name, which is the `Mapping[str, Point]` that the
geometry solvers take.
"""

from __future__ import annotations

from dataclasses import dataclass

from skellyforge.core.math.geometry.spatial_vectors import Point
from skellyforge.core.skeleton_parts.naming import raise_unless_aliases_are_valid
from skellyforge.type_overloads import (
    LandmarkDefinitionString,
    LandmarkNameString,
    RigidBodySegmentName,
)


@dataclass(frozen=True, slots=True, eq=False)
class AnatomicalLandmark:
    """One named point defined in its owning segment's local frame.

    Attributes:
        name: unique identifier.
        anatomical_definition: medical description naming exactly one point on the body.
        local_position: rest position in the owning segment's local frame. A single
            location, so its batch shape is `()`.
        segment: name of the segment whose local frame `local_position` is expressed in.
        aliases: other names this same point answers to, for motion capture sources that
            name it differently. Must be globally unique across the whole landmark set,
            which `LandmarkNameResolver.from_landmarks` is what actually enforces - a
            landmark on its own can only check that its aliases are non-empty, distinct,
            and different from its own name.
    """

    name: LandmarkNameString
    anatomical_definition: LandmarkDefinitionString
    local_position: Point
    segment: RigidBodySegmentName
    aliases: tuple[LandmarkNameString, ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("landmark name must be non-empty")
        if not self.anatomical_definition:
            raise ValueError(f"landmark {self.name!r} needs an anatomical definition")
        if not self.segment:
            raise ValueError(f"landmark {self.name!r} needs an owning segment")
        raise_unless_aliases_are_valid(name=self.name, aliases=self.aliases)
        if self.local_position.batch_shape != ():
            raise ValueError(
                f"landmark {self.name!r} rest position must be a single point of shape "
                f"(3,) - got shape {self.local_position.array.shape}"
            )

    @property
    def all_names(self) -> tuple[LandmarkNameString, ...]:
        """Every name this landmark answers to, canonical name first."""
        return (self.name, *self.aliases)

    def __str__(self) -> str:
        x, y, z = self.local_position.array
        return (
            f"{self.name} (in {self.segment}) at local ({x:.4g}, {y:.4g}, {z:.4g}): "
            f"{self.anatomical_definition}"
            + (f" [aka {', '.join(self.aliases)}]" if self.aliases else "")
        )
