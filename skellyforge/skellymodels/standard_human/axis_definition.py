from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from skellyforge.skellymodels.standard_human.config_types import AxisConfig

@dataclass(frozen=True, slots=True)
class AxisConfig:
    axis: Literal["x", "y", "z", "-x", "-y", "-z"]
    target_landmark: str
    rest_direction: tuple[float, float, float] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "AxisConfig":
        return cls(**data)

    def __post_init__(self) -> None:
        if self.rest_direction is not None:
            object.__setattr__(
                self, "rest_direction", tuple(float(v) for v in self.rest_direction)
            )

@dataclass(frozen=True, slots=True)
class AxisDefinition:
    """One direction source for the segment's local frame.

    'axis' names WHICH basis row this direction defines, signed
    ("x"/"y"/"z"/"-x"/"-y"/"-z"). 'target_landmark' names the point the
    direction points toward (origin -> target); it must be rigid on the
    segment. 'rest_direction' is the authored unit direction at the T-pose
    (defaults to the axis row's positive unit vector).

    The axes tuple is the construction recipe, in order: the FIRST axis is the
    primary direction (the frame's hard seed and the length's distal); the
    SECOND (if present) is the twist direction (the roll reference,
    Gram-Schmidt'd against the primary).
    """

    axis: Literal["x", "y", "z", "-x", "-y", "-z"]
    target_landmark: str
    rest_direction: tuple[float, float, float] | None = None

    @classmethod
    def from_config(cls, config: AxisConfig) -> "AxisDefinition":
        return cls(
            axis=config.axis,
            target_landmark=config.target_landmark,
            rest_direction=config.rest_direction,
        )
