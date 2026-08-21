"""A landmark: a named point defined in the local frame of a segment.

Each landmark is defined by a precise
anatomical description (medical language that names one individual point in
space) plus a rest position in the local frame of its owning segment
"""

from __future__ import annotations

from dataclasses import dataclass

from skellyforge.math_stuff.transform_math import Position, Transform
from skellyforge.skeleton_parts.segments.rigid_body_segment import RigidBodySegment

LandmarkNameString = str
LandmarkDefinitionString = str


@dataclass(frozen=True, slots=True)
class AnatomicalLandmark:
    name: LandmarkNameString
    anatomical_definition: LandmarkDefinitionString
    local_position: Position
    segment: RigidBodySegment  # the owning segment's name (its local frame)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("landmark name must be non-empty")
        if not self.anatomical_definition:
            raise ValueError(f"landmark {self.name!r} needs an anatomical definition")

        if not self.segment:
            raise ValueError(f"landmark {self.name!r} needs a reference_frame")

    def world_position(self, transform: Transform) -> Position:
        raise NotImplementedError()
