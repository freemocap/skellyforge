"""A landmark: a named point defined in the local frame of a segment.

The static face of the landmark layer. Each landmark is defined by a precise
anatomical description (medical language that names one individual point in
space) plus a rest position in the local frame of its owning segment
('reference_frame' - explicit ownership, never "whoever declares it first").

Hydrated, a landmark becomes a trajectory (a world position per frame).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from skellyforge.skellymodels.standard_human.config_types import LandmarkConfig


@dataclass(frozen=True, slots=True)
class AnatomicalLandmark:
    name: str
    anatomical_definition: str
    rest_position: tuple[float, float, float]
    reference_frame: str  # the owning segment's name (its local frame)

    @classmethod
    def from_config(cls, name: str, config: LandmarkConfig) -> "AnatomicalLandmark":
        return cls(
            name=name,
            anatomical_definition=config.definition,
            rest_position=config.rest_position,
            reference_frame=config.reference_frame,
        )

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("landmark name must be non-empty")
        if not self.anatomical_definition:
            raise ValueError(f"landmark {self.name!r} needs an anatomical definition")
        if len(self.rest_position) != 3 or not all(
            math.isfinite(v) for v in self.rest_position
        ):
            raise ValueError(
                f"landmark {self.name!r} rest_position must be 3 finite numbers"
            )
        if not self.reference_frame:
            raise ValueError(f"landmark {self.name!r} needs a reference_frame")
