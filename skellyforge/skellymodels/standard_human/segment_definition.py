"""The authored unit of the standard human: one rigid-body segment.

A segment is an **origin**, an **orientation**, and a **length**. All three are
declared here as named keypoints plus a rest pose, and this one
declaration is evaluated twice — against the T-pose to build the reference
geometry, and against each frame to read the live pose.

Keypoint names are side-agnostic within a part; composition prefixes them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class ParentAttachment(str, Enum):
    """Which end of the parent this segment's origin sits at."""

    ORIGIN = "origin"
    """Branch from the parent's origin — e.g. both upper legs and the spine
    branch from the hips' origin."""

    DISTAL = "distal"
    """Continue from the parent's distal end — the common case."""


@dataclass(frozen=True)
class RotationLimits:
    """Joint range-of-motion limits in degrees, in the segment's local frame.

    **Declared, not yet enforced** — the solver ignores these in SF-SM. They are
    specified now so the model is complete and the values are reviewed alongside
    the anatomy they constrain, rather than bolted on later.
    """

    x: tuple[float, float]
    y: tuple[float, float]
    z: tuple[float, float]

    def __post_init__(self) -> None:
        for axis in ("x", "y", "z"):
            low, high = getattr(self, axis)
            if not math.isfinite(low) or not math.isfinite(high):
                raise ValueError(
                    f"rotation_limits.{axis} bounds must be finite, got ({low}, {high})"
                )
            if low > high:
                raise ValueError(
                    f"rotation_limits.{axis} lower bound {low} exceeds upper bound {high}"
                )


@dataclass(frozen=True)
class SegmentDefinition:
    """One VRM-1.0-aligned rigid body, authored side-agnostically."""

    name: str
    parent: str | None
    parent_attachment: ParentAttachment
    origin_keypoint: str
    long_axis_keypoint: str
    twist_keypoint: str | None
    rest_rotation: tuple[float, float, float]
    rest_roll: float
    length_ratio: float
    rotation_limits: RotationLimits | None = None

    def __post_init__(self) -> None:
        if not self.name or self.name != self.name.lower() or " " in self.name:
            raise ValueError(f"segment name must be snake_case, got {self.name!r}")

        if self.origin_keypoint == self.long_axis_keypoint:
            raise ValueError(
                f"segment {self.name!r}: origin_keypoint and long_axis_keypoint are both "
                f"{self.origin_keypoint!r}. The segment vector would be zero-length and no "
                f"orientation could be resolved from it."
            )

        if self.twist_keypoint is not None and self.twist_keypoint == self.long_axis_keypoint:
            raise ValueError(
                f"segment {self.name!r}: twist_keypoint is {self.twist_keypoint!r}, the same as "
                f"long_axis_keypoint. A twist reference collinear with the long axis resolves "
                f"nothing."
            )

        if not math.isfinite(self.length_ratio) or self.length_ratio <= 0.0:
            raise ValueError(
                f"segment {self.name!r}: length_ratio must be finite and > 0, got {self.length_ratio}"
            )

    @property
    def resolves_twist(self) -> bool:
        """Whether this segment's roll is determined by its own keypoints.

        ``False`` means the twist falls back to the critically damped minimal-twist
        solve — the tier is a *consequence* of the declaration, not a separate policy.
        """
        return self.twist_keypoint is not None

    def required_keypoints(self) -> set[str]:
        """Every keypoint this segment needs to be solvable."""
        names = {self.origin_keypoint, self.long_axis_keypoint}
        if self.twist_keypoint is not None:
            names.add(self.twist_keypoint)
        return names
