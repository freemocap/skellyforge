"""A rigid-body segment: origin + orientation + length, solved from its landmarks.

Fully specified with 3+ non-collinear landmarks; partially specified with only
2 (roll is then carried by the damped minimal-roll tier). Its length is derived
from its landmarks' rest positions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from skellyforge.skellymodels.standard_human.anatomical_landmark import (
    AnatomicalLandmark,
)
from skellyforge.skellymodels.standard_human.config_types import AxisConfig


@dataclass(frozen=True, slots=True)
class AxisDefinition:
    """One declared axis of the segment's local frame.

    'axis' names WHICH basis row this declaration defines, signed
    ("x"/"y"/"z"/"-x"/"-y"/"-z"). 'kind' is EXACT (the defining direction) or
    APPROXIMATE (a soft direction reference, Gram-Schmidt'd). 'target_landmark'
    names the point this axis points toward; it must be rigid on the segment.
    """

    axis: Literal["x", "y", "z", "-x", "-y", "-z"]
    kind: Literal["exact", "approximate"]
    target_landmark: str
    rest_direction: tuple[float, float, float] | None = None

    @classmethod
    def from_config(cls, config: AxisConfig) -> "AxisDefinition":
        return cls(
            axis=config.axis,
            kind=config.kind,
            target_landmark=config.target_landmark,
            rest_direction=config.rest_direction,
        )


@dataclass(frozen=True, slots=True)
class RigidBodySegment:
    """One rigid body. Parent + landmarks + origin are OBJECT references (resolved
    at load by HumanSkeleton.from_yaml - never raw strings after load)."""

    name: str
    parent: "RigidBodySegment | None"
    landmarks: tuple[AnatomicalLandmark, ...]
    origin_landmark: AnatomicalLandmark
    axes: tuple[AxisDefinition, ...]
    rigid_with_parent: bool = False

    def __post_init__(self) -> None:
        if not self.name or self.name != self.name.lower() or " " in self.name:
            raise ValueError(f"segment name must be snake_case, got {self.name!r}")
        if len(self.landmarks) < 2:
            raise ValueError(f"segment {self.name!r}: needs at least 2 landmarks")
        names = [l.name for l in self.landmarks]
        if len(set(names)) != len(names):
            raise ValueError(f"segment {self.name!r}: landmarks must be distinct")
        if self.origin_landmark.name not in names:
            raise ValueError(
                f"segment {self.name!r}: origin_landmark {self.origin_landmark.name!r} "
                f"is not in landmarks"
            )
        if not self.axes:
            raise ValueError(f"segment {self.name!r}: needs at least one axis")

    @property
    def exact_axis(self) -> AxisDefinition:
        for a in self.axes:
            if a.kind == "exact":
                return a
        raise ValueError(f"segment {self.name!r} has no EXACT axis")

    @property
    def length(self) -> float:
        """Origin->distal distance, derived from the exact axis target's rest position
        (authored in THIS segment's local frame)."""
        target_name = self.exact_axis.target_landmark
        distal = next(l for l in self.landmarks if l.name == target_name)
        x, y, z = distal.rest_position
        return math.sqrt(x * x + y * y + z * z)
