"""A landmark: a named point defined in the local frame of a segment.

Each landmark is defined by a precise
anatomical description (medical language that names one individual point in
space) plus a rest position in the local frame of its owning segment
"""

from __future__ import annotations

from dataclasses import dataclass

from skellyforge.core.math.geometry.transform_math import Translation, Transform
from skellyforge.type_overloads import LandmarkNameString, LandmarkDefinitionString, RigidBodySegmentName


@dataclass(frozen=True, slots=True)
class AnatomicalLandmark:
    name: LandmarkNameString
    anatomical_definition: LandmarkDefinitionString
    local_position: Translation
    segment: RigidBodySegmentName  # the owning segment's name (its local frame)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("landmark name must be non-empty")
        if not self.anatomical_definition:
            raise ValueError(f"landmark {self.name!r} needs an anatomical definition")

        if not self.segment:
            raise ValueError(f"landmark {self.name!r} needs a reference_frame")

    def world_position(self, transform: Transform) -> Translation:
        # TODO - use transform to convert local_position into a world_position
        raise NotImplementedError()

    def __str__(self) -> str:
        # todo - nicely formatted string
        pass


@dataclass(frozen=True, slots=True)
class HydratedLandmark:
    world_position: Translation
    landmark: AnatomicalLandmark

    @classmethod
    def hydrate(
        cls, landmark: AnatomicalLandmark, transform: Transform
    ) -> HydratedLandmark:
        return HydratedLandmark(
            landmark=landmark, world_position=landmark.world_position(transform)
        )

    # TODO - find a slicker way to just pass through unhydrated properties
    @property
    def name(self) -> LandmarkNameString:
        return self.landmark.name

    @property
    def anatomical_definition(self) -> LandmarkDefinitionString:
        return self.landmark.anatomical_definition

    @property
    def local_position(self) -> Translation:
        return self.landmark.local_position

    def __str__(self) -> str:
        # todo - nicely formatted string
        pass
