"""A rigid-body segment: origin + orientation + length, solved from its landmarks.

Fully specified with 3+ non-collinear landmarks; partially specified with only
2 (roll is then carried by the damped minimal-roll tier). Its length is derived
from its landmarks' rest positions.
"""

from __future__ import annotations


import math
from dataclasses import dataclass

from skellyforge.skellymodels.standard_human import AxisDefinition
from skellyforge.skellymodels.standard_human.anatomical_landmark import (
    AnatomicalLandmark,
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
        base_name = self.name[:-2] if self.name.endswith((".L", ".R")) else self.name
        if not base_name or base_name != base_name.lower() or " " in base_name:
            raise ValueError(f"segment name must be snake_case (optionally .L/.R), got {self.name!r}")
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
    def primary_axis(self) -> AxisDefinition:
        """The primary direction source (the frame's seed + the length's distal)."""
        return self.axes[0]

    @property
    def length(self) -> float:
        """Origin->distal distance, derived from the primary direction's target
        rest position (authored in THIS segment's local frame)."""
        target_name = self.primary_axis.target_landmark
        distal = next(l for l in self.landmarks if l.name == target_name)
        x, y, z = distal.local_position
        return math.sqrt(x * x + y * y + z * z)
